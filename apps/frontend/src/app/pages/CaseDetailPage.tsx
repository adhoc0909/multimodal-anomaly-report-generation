import React, { useMemo, useState, useRef } from "react";
import { Badge } from "../components/Badge";
import {
  ChevronRight,
  Download,
  Activity,
  ShieldCheck,
  Clock,
  Loader2
} from "lucide-react";
import { decisionLabel, defectTypeLabel, locationLabel } from "../utils/labels";
import { getCaseImageUrl, type ImageVariant } from "../services/media";
import jsPDF from "jspdf";
import html2canvas from "html2canvas";
import { ReportTemplate, type DBReport } from "../components/ReportTemplate";

// 데이터 타입 정의
interface AnomalyCase {
  id: string | number;
  image_id?: string;
  timestamp: string | Date;
  line_id?: string;
  product_group?: string;
  defect_type: string;
  location: string;
  affected_area_pct?: number;
  anomaly_score?: number;
  defect_confidence?: number;
  decision: "ok" | "ng" | "pending" | "anomaly" | "normal";
  llm_summary?: string;
  llm_structured_json?: any;
  pipeline_status?: string;
  pipeline_stage?: string;
  pipeline_error?: string;
  image_path?: string;
  heatmap_path?: string;
  overlay_path?: string;
}

interface CaseDetailPageProps {
  caseData: AnomalyCase;
  onBackToQueue: () => void;
  onBackToOverview: () => void;
}

// PDF 출력을 위한 판정 값 정규화
function normalizeDecisionForPdf(decision?: string): DBReport["decision"] {
  const d = String(decision ?? "").trim().toLowerCase();
  if (d === "ng" || d === "anomaly") return "ng";
  if (d === "ok" || d === "normal") return "ok";
  return "pending";
}

// 이미지 표시 컴포넌트
function ImagePanel({ caseData, active }: { caseData: AnomalyCase; active: ImageVariant }) {
  const url = useMemo(() => getCaseImageUrl(caseData as any, active), [caseData, active]);

  return (
    <div className="bg-gray-100 rounded-2xl aspect-[4/3] overflow-hidden mb-4 flex items-center justify-center border border-gray-200 shadow-inner">
      {url ? (
        <img
          src={url}
          alt={`${active}`}
          className="w-full h-full object-contain"
          crossOrigin="anonymous" 
          loading="lazy"
        />
      ) : (
        <div className="text-gray-400 flex flex-col items-center">
          <Loader2 className="w-8 h-8 animate-spin mb-3 text-gray-300" />
          <p className="text-sm font-medium">이미지 데이터를 불러오는 중...</p>
        </div>
      )}
    </div>
  );
}

export function CaseDetailPage({ caseData, onBackToQueue, onBackToOverview }: CaseDetailPageProps) {
  const [activeTab, setActiveTab] = useState<ImageVariant>("original");
  const reportRef = useRef<HTMLDivElement>(null);
  const [isDownloading, setIsDownloading] = useState(false);

  // 날짜 포맷팅
  const formattedDate = useMemo(() => {
    const d = new Date(caseData.timestamp);
    return isNaN(d.getTime()) ? "N/A" : d.toLocaleString("ko-KR");
  }, [caseData.timestamp]);

  // LLM 데이터 추출 로직
  const llmView = useMemo(() => {
    const source = (caseData.llm_structured_json as any)?.source ?? {};
    const asObj = (v: any) => {
      if (!v) return {};
      if (typeof v === "object") return v;
      if (typeof v === "string") {
        try { return JSON.parse(v.trim()); } catch { return { _text: v }; }
      }
      return {};
    };
    const pick = (...values: any[]) => {
      for (const v of values) { if (typeof v === "string" && v.trim()) return v.trim(); }
      return "";
    };

    const llmReportRoot = asObj(source.llm_report);
    const llmReport = asObj(llmReportRoot.report ?? llmReportRoot);
    const llmSummaryObj = asObj(source.llm_summary);

    return {
      summary: pick(caseData.llm_summary, llmSummaryObj.summary, llmSummaryObj._text, llmReport.summary, source.summary),
      description: pick(llmReport.description, source.defect_description, source.description),
      cause: pick(llmReport.possible_cause, llmReport.cause, llmReport.likely_cause, source.possible_cause),
      recommendation: pick(llmReport.recommendation, source.recommendation)
    };
  }, [caseData]);

  const isLlmComplete = llmView.summary.length > 0;

  // PDF용 데이터 정제 (빈 줄 제거 및 Trim)
  const cleanSummary = useMemo(() => {
    if (!isLlmComplete) return "분석 기록 대기 중";
    return llmView.summary
      .split('\n')
      .map(line => line.trim())
      .filter(line => line.length > 0)
      .join('\n');
  }, [llmView.summary, isLlmComplete]);

  // PDF 전달용 데이터 객체
  const pdfReportData = useMemo<DBReport>(() => ({
    id: caseData.id,
    image_id: caseData.image_id ?? "",
    category: caseData.product_group ?? "-",
    timestamp: String(caseData.timestamp),
    defect_type: caseData.defect_type,
    location: caseData.location,
    affected_area_pct: caseData.affected_area_pct ?? 0,
    ad_score: caseData.anomaly_score ?? 0,
    confidence: caseData.defect_confidence ?? 0,
    decision: normalizeDecisionForPdf(caseData.decision),
    llm_analysis_summary: cleanSummary, // 정제된 텍스트 전달
    image_path: caseData.image_path ?? "",
    heatmap_path: caseData.heatmap_path ?? "",
    overlay_path: caseData.overlay_path ?? "",
  }), [caseData, cleanSummary]);

  // PDF 생성 및 다운로드 함수
  const handleDownloadPdf = async () => {
    if (!reportRef.current) return;
    setIsDownloading(true);

    try {
      const element = reportRef.current;
      const imgs = Array.from(element.querySelectorAll("img"));
      await Promise.all(imgs.map(img => {
        if (img.complete && img.naturalWidth !== 0) return Promise.resolve();
        return new Promise(res => { img.onload = res; img.onerror = res; });
      }));

      const canvas = await html2canvas(element, {
        scale: 2,
        useCORS: true,
        backgroundColor: "#ffffff",
        width: 794,
        height: element.scrollHeight,
        windowWidth: 794,
        onclone: (clonedDoc) => {
            // oklch 컬러 호환성 해결을 위한 강제 변환
            const allElements = clonedDoc.getElementsByTagName("*");
            for (let i = 0; i < allElements.length; i++) {
                const el = allElements[i] as HTMLElement;
                const style = window.getComputedStyle(el);
                if (style.color.includes("oklch")) el.style.color = "#111827";
                if (style.backgroundColor.includes("oklch")) el.style.backgroundColor = "#ffffff";
                if (style.borderColor.includes("oklch")) el.style.borderColor = "#e5e7eb";
            }
        }
      });

      const imgData = canvas.toDataURL("image/png");
      const pdf = new jsPDF("p", "mm", "a4");
      const pdfWidth = pdf.internal.pageSize.getWidth();
      const pdfHeight = (canvas.height * pdfWidth) / canvas.width;

      pdf.addImage(imgData, "PNG", 0, 0, pdfWidth, pdfHeight);
      pdf.save(`Inspection_Report_${caseData.id}.pdf`);
    } catch (e) {
      console.error("PDF 생성 에러:", e);
      alert("PDF 생성 중 오류가 발생했습니다.");
    } finally {
      setIsDownloading(false);
    }
  };

  return (
    <div className="p-8 bg-gray-50 min-h-screen font-sans">
      {/* 상단 네비게이션 */}
      <div className="flex items-center gap-2 text-sm text-gray-500 mb-6">
        <button onClick={onBackToOverview} className="hover:text-blue-600 transition-colors">개요</button>
        <ChevronRight className="w-4 h-4" />
        <button onClick={onBackToQueue} className="hover:text-blue-600 transition-colors">이상 큐</button>
        <ChevronRight className="w-4 h-4" />
        <span className="text-gray-900 font-semibold">{caseData.id}</span>
      </div>

      {/* 헤더 섹션 */}
      <div className="flex justify-between items-start mb-8">
        <div>
          <h1 className="text-3xl font-black text-gray-900 mb-2">{caseData.id}</h1>
          <div className="flex items-center gap-3 text-sm text-gray-500">
            <span className="bg-gray-200 px-2 py-0.5 rounded text-gray-700 font-bold">{caseData.line_id || "Line-A"}</span>
            <span className="flex items-center gap-1"><Clock className="w-3.5 h-3.5" /> {formattedDate}</span>
          </div>
        </div>
        <button
          onClick={handleDownloadPdf}
          disabled={isDownloading}
          className="flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-xl hover:bg-blue-700 transition-all shadow-lg active:scale-95 disabled:bg-blue-300"
        >
          {isDownloading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Download className="w-5 h-5" />}
          <span className="font-bold">{isDownloading ? "PDF 생성 중..." : "리포트 다운로드"}</span>
        </button>
      </div>

      <div className="grid grid-cols-12 gap-8">
        {/* 메인 분석 콘텐츠 (좌측 8컬럼) */}
        <div className="col-span-8 space-y-6">
          {/* 이미지 카드 */}
          <div className="bg-white border border-gray-200 rounded-2xl p-6 shadow-sm">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-xl font-bold text-gray-900">검사 이미지</h2>
              <div className="flex gap-1 bg-gray-100 p-1 rounded-xl">
                {(["original", "heatmap", "overlay"] as const).map((k) => (
                  <button
                    key={k}
                    onClick={() => setActiveTab(k)}
                    className={`px-4 py-1.5 text-xs font-black rounded-lg transition-all ${
                      activeTab === k ? "bg-white text-blue-600 shadow-sm" : "text-gray-500 hover:text-gray-700"
                    }`}
                  >
                    {k.toUpperCase()}
                  </button>
                ))}
              </div>
            </div>
            <ImagePanel caseData={caseData} active={activeTab} />
          </div>

          {/* AI 분석 리포트 카드 */}
          <div className="bg-white border border-gray-200 rounded-2xl p-6 shadow-sm">
            <h2 className="text-xl font-bold text-gray-900 mb-6">AI 심층 분석 리포트</h2>
            <div className="grid grid-cols-3 gap-4 mb-8">
              <div className="bg-gray-50 p-4 rounded-2xl border border-gray-100">
                <span className="text-[10px] font-black text-gray-400 uppercase tracking-widest">Defect Type</span>
                <p className="text-lg font-bold text-gray-900 mt-1">{defectTypeLabel(caseData.defect_type)}</p>
              </div>
              <div className="bg-gray-50 p-4 rounded-2xl border border-gray-100">
                <span className="text-[10px] font-black text-gray-400 uppercase tracking-widest">Location</span>
                <p className="text-lg font-bold text-gray-900 mt-1">{locationLabel(caseData.location)}</p>
              </div>
              <div className="bg-gray-50 p-4 rounded-2xl border border-gray-100">
                <span className="text-[10px] font-black text-gray-400 uppercase tracking-widest">Affected Area</span>
                <p className="text-lg font-bold text-gray-900 mt-1">{caseData.affected_area_pct?.toFixed(2)}%</p>
              </div>
            </div>

            <div className="bg-blue-50/50 border border-blue-100 rounded-2xl p-6">
              <div className="flex items-center gap-2 mb-4">
                <Activity className={`w-5 h-5 text-blue-600 ${!isLlmComplete ? 'animate-pulse' : ''}`} />
                <h3 className="font-bold text-blue-900 text-lg">AI 상세 분석 요약</h3>
              </div>
              <p className="text-sm text-blue-800 leading-relaxed whitespace-pre-wrap font-medium">
                {isLlmComplete ? cleanSummary : "AI가 이미지를 심층 분석하여 결과를 생성하고 있습니다..."}
              </p>
            </div>
          </div>
        </div>

        {/* 사이드바 정보 (우측 4컬럼) */}
        <div className="col-span-4 space-y-6">
          <div className="bg-white border border-gray-200 rounded-2xl shadow-sm overflow-hidden">
            {/* 판정 결과 헤더 */}
            <div className={`p-6 border-b border-gray-100 ${
              caseData.decision === 'ng' || caseData.decision === 'anomaly' ? 'bg-red-50/50' : 'bg-green-50/50'
            }`}>
              <div className="flex items-center justify-between">
                <h2 className="text-lg font-bold text-gray-900 flex items-center gap-2">
                  <ShieldCheck className={`w-5 h-5 ${
                    caseData.decision === 'ng' || caseData.decision === 'anomaly' ? 'text-red-600' : 'text-green-600'
                  }`} /> 자율 판정 결과
                </h2>
                <Badge variant={caseData.decision}>{decisionLabel(caseData.decision)}</Badge>
              </div>
            </div>

            <div className="p-6">
              {/* 스코어 정보 */}
              <div className="grid grid-cols-2 gap-4 mb-8">
                <div className="space-y-1">
                  <span className="text-[10px] font-black text-gray-400 uppercase tracking-widest">Anomaly Score</span>
                  <p className="text-2xl font-mono font-black text-gray-900 tracking-tighter">
                    {caseData.anomaly_score?.toFixed(4)}
                  </p>
                </div>
                <div className="space-y-1 border-l border-gray-100 pl-4">
                  <span className="text-[10px] font-black text-gray-400 uppercase tracking-widest">Confidence</span>
                  <p className="text-2xl font-mono font-black text-gray-900 tracking-tighter">
                    {(caseData.defect_confidence ? caseData.defect_confidence * 100 : 0).toFixed(1)}%
                  </p>
                </div>
              </div>

              {/* 히스토리 타임라인 */}
              <div className="pt-6 border-t border-gray-100">
                <h3 className="text-[10px] font-black text-gray-400 uppercase tracking-widest mb-6 flex items-center gap-2">
                  <Clock className="w-3.5 h-3.5" /> Processing History
                </h3>
                <div className="relative space-y-8 ml-1">
                  {/* 타임라인 실선 */}
                  <div className="absolute left-[3px] top-2 bottom-2 w-[1.5px] bg-gray-100" />

                  {/* 단계 1 */}
                  <div className="relative flex gap-4 items-start">
                    <div className="relative z-10 w-2 h-2 rounded-full bg-blue-500 ring-4 ring-blue-50 mt-1" />
                    <div className="space-y-1">
                      <p className="text-xs font-bold text-gray-800">이상 탐지 분석 완료</p>
                      <p className="text-[10px] text-gray-400 font-medium">{formattedDate}</p>
                    </div>
                  </div>

                  {/* 단계 2 (동적 상태) */}
                  <div className="relative flex gap-4 items-start">
                    <div className={`relative z-10 w-2 h-2 rounded-full mt-1 transition-all duration-500 ${
                      isLlmComplete 
                        ? 'bg-green-500 ring-4 ring-green-50' 
                        : 'bg-blue-300 animate-pulse ring-4 ring-blue-50'
                    }`} />
                    <div className="space-y-1">
                      <p className={`text-xs font-bold ${isLlmComplete ? 'text-gray-900' : 'text-blue-600'}`}>
                        {isLlmComplete ? "최종 AI 리포트 생성 완료" : "심층 분석 리포트 생성 중"}
                      </p>
                      <p className="text-[10px] text-gray-400 font-medium leading-tight">
                        {isLlmComplete ? "데이터베이스 동기화 완료" : "AI Pipeline이 이미지를 분석하고 있습니다."}
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* PDF용 숨겨진 렌더링 영역 */}
      <div style={{ position: 'absolute', top: '-10000px', left: 0, opacity: 0 }}>
        <div ref={reportRef} style={{ width: '794px', padding: '0', margin: '0' }}>
          <ReportTemplate reportData={pdfReportData} />
        </div>
      </div>
    </div>
  );
}
