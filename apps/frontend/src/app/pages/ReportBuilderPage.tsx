import React, { useMemo, useRef, useState } from "react";
import { Download, Loader2 } from "lucide-react";
import { domToPng } from 'modern-screenshot';
import jsPDF from 'jspdf';
import type { AnomalyCase } from "../data/mockData";
import {
  LineChart, Line, BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, LabelList
} from "recharts";
import {
  buildAggregates, toDefectRateTrend, toDefectTypePie, toProductDefectsFixed10,
} from "../selectors/caseSelectors";

interface ReportBuilderPageProps {
  cases: AnomalyCase[];
}

export function ReportBuilderPage({ cases }: ReportBuilderPageProps) {
  const pdfRef = useRef<HTMLDivElement>(null);
  const [isDownloading, setIsDownloading] = useState(false);

  const agg = useMemo(() => buildAggregates(cases), [cases]);
  const reportMetrics = useMemo(() => {
    const total = agg.total;
    return {
      total,
      ngCount: agg.ng,
      reviewCount: agg.review,
      okCount: agg.ok,
      ngRate: total ? ((agg.ng / total) * 100).toFixed(2) : "0.00",
      avgScore: total ? agg.avgScore.toFixed(3) : "0.000",
    };
  }, [agg]);

  const defectRateTrend = useMemo(() => toDefectRateTrend(agg), [agg]);
  const defectTypeData = useMemo(() => toDefectTypePie(agg), [agg]);
  const productDefectsFixed10 = useMemo(() => toProductDefectsFixed10(agg), [agg]);
  const COLORS = ["#ef4444", "#f97316", "#eab308", "#84cc16", "#06b6d4", "#8b5cf6"];

  const handleExportPdf = async () => {
    if (!pdfRef.current) return;
    setIsDownloading(true);
    try {
      const dataUrl = await domToPng(pdfRef.current, { scale: 2.5, backgroundColor: '#ffffff' });
      const pdf = new jsPDF("p", "mm", "a4");
      const pdfWidth = pdf.internal.pageSize.getWidth();
      const imgProps = pdf.getImageProperties(dataUrl);
      const imgHeight = (imgProps.height * pdfWidth) / imgProps.width;
      pdf.addImage(dataUrl, 'PNG', 0, 0, pdfWidth, imgHeight);
      pdf.save(`Report_by period_${new Date().toISOString().split('T')[0]}.pdf`);
    } catch (error) {
      console.error(error);
    } finally {
      setIsDownloading(false);
    }
  };

  return (
    <>
      {/* 1. 웹 브라우저 UI */}
      <div className="p-8 bg-gray-50 min-h-screen">
        <h1 className="text-2xl font-bold text-gray-900 mb-8">운영 리포트</h1>

        {/* 요약 지표 */}
        <div className="bg-white border border-gray-200 rounded-lg p-6 mb-8">
          <div className="grid grid-cols-4 gap-6">
            <div className="bg-gray-50 p-4 rounded-lg text-center">
              <p className="text-sm text-gray-500 mb-1">총 검사 수</p>
              <p className="text-3xl font-bold">{reportMetrics.total}</p>
            </div>
            <div className="bg-red-50 p-4 rounded-lg text-center">
              <p className="text-sm text-red-600 mb-1 font-bold">불량 (NG)</p>
              <p className="text-3xl font-bold text-red-600">{reportMetrics.ngCount} ({reportMetrics.ngRate}%)</p>
            </div>
            <div className="bg-green-50 p-4 rounded-lg text-center">
              <p className="text-sm text-green-600 mb-1 font-bold">정상 (OK)</p>
              <p className="text-3xl font-bold text-green-600">{reportMetrics.okCount}</p>
            </div>
            <div className="bg-blue-50 p-4 rounded-lg text-center">
              <p className="text-sm text-blue-600 mb-1 font-bold">평균 Score</p>
              <p className="text-3xl font-bold text-blue-600">{reportMetrics.avgScore}</p>
            </div>
          </div>
        </div>

        <div className="space-y-8 mb-12">
          <div className="grid grid-cols-2 gap-8">
            <div className="bg-white p-6 rounded-xl border border-gray-200 shadow-sm h-[380px]">
              <h3 className="font-bold mb-4">불량률 추이</h3>
              <ResponsiveContainer width="100%" height="90%">
                <LineChart data={defectRateTrend}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="time" />
                  <YAxis />
                  <Tooltip />
                  <Line type="monotone" dataKey="불량률" stroke="#ef4444" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <div className="bg-white p-6 rounded-xl border border-gray-200 shadow-sm h-[380px]">
              <h3 className="font-bold mb-4">결함 타입 분포</h3>
              <ResponsiveContainer width="100%" height="90%">
                <PieChart>
                  <Pie data={defectTypeData} cx="50%" cy="50%" outerRadius={80} dataKey="value" label={({name, percent}) => `${name} ${(percent * 100).toFixed(0)}%`}>
                    {defectTypeData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="bg-white p-6 rounded-xl border border-gray-200 shadow-sm h-[400px]">
            <h3 className="font-bold mb-4">제품별 주요 불량 비교</h3>
            <ResponsiveContainer width="100%" height="90%">
              <BarChart data={productDefectsFixed10} margin={{ bottom: 30 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="name" tick={{ fontSize: 11 }} interval={0} angle={-15} textAnchor="end" height={60} />
                <YAxis />
                <Tooltip />
                <Legend verticalAlign="bottom" />
                <Bar dataKey="count" name="불량 건수(NG)" fill="#3b82f6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* ✅ 버튼 위치 하단 중앙 고정 */}
        <div className="flex justify-center pb-12">
          <button
            onClick={handleExportPdf}
            disabled={isDownloading}
            className="flex items-center gap-2 px-12 py-4 bg-blue-600 text-white rounded-xl font-bold shadow-lg hover:bg-blue-700 active:scale-95 transition-all"
          >
            {isDownloading ? <Loader2 className="w-6 h-6 animate-spin" /> : <Download className="w-6 h-6" />}
            <span>PDF 리포트 다운로드</span>
          </button>
        </div>
      </div>

      {/* 2. PDF 출력용 숨김 레이아웃 (잘림 방지 초집중 버전) */}
      <div style={{ position: 'fixed', top: 0, left: '-9999px' }}>
        <div ref={pdfRef} className="w-[820px] bg-white p-12 flex flex-col gap-10">
          <div className="border-b-4 border-blue-600 pb-6">
            <h1 className="text-3xl font-black text-gray-900 tracking-tight">운영 분석 리포트</h1>
            <p className="text-gray-400 text-sm mt-2">{new Date().toLocaleString()}</p>
          </div>

          {/* PDF 요약 카드 */}
          <div className="grid grid-cols-4 gap-4">
            <div className="bg-slate-50 p-5 rounded-2xl border border-slate-100 flex flex-col items-center">
              <span className="text-xs font-bold text-slate-500 mb-2">총 검사</span>
              <span className="text-3xl font-black text-slate-800">{reportMetrics.total}</span>
            </div>
            <div className="bg-red-50 p-5 rounded-2xl border border-red-100 flex flex-col items-center">
              <span className="text-xs font-bold text-red-500 mb-2">불량(NG)</span>
              <span className="text-3xl font-black text-red-700">{reportMetrics.ngCount}</span>
              <span className="text-[10px] font-bold text-red-400">{reportMetrics.ngRate}%</span>
            </div>
            <div className="bg-emerald-50 p-5 rounded-2xl border border-emerald-100 flex flex-col items-center">
              <span className="text-xs font-bold text-emerald-500 mb-2">정상(OK)</span>
              <span className="text-3xl font-black text-emerald-700">{reportMetrics.okCount}</span>
            </div>
            <div className="bg-blue-50 p-5 rounded-2xl border border-blue-100 flex flex-col items-center">
              <span className="text-xs font-bold text-blue-500 mb-2">평균 Score</span>
              <span className="text-3xl font-black text-blue-700">{reportMetrics.avgScore}</span>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-8">
            <div className="bg-white rounded-2xl p-6 border border-gray-100 h-[380px]">
              <h3 className="text-sm font-bold text-gray-800 mb-4 flex items-center gap-2"><div className="w-1 h-4 bg-red-500 rounded-full" /> 불량률 변화 추이</h3>
              <ResponsiveContainer width="100%" height="80%">
                <LineChart data={defectRateTrend} margin={{ right: 20 }}>
                  <CartesianGrid vertical={false} stroke="#f0f0f0" />
                  <XAxis dataKey="time" tick={{fontSize: 10}} />
                  <YAxis tick={{fontSize: 10}} />
                  <Line isAnimationActive={false} type="monotone" dataKey="불량률" stroke="#ef4444" strokeWidth={4} />
                </LineChart>
              </ResponsiveContainer>
            </div>
            
            {/* ✅ 파이차트 글씨 잘림 방지: margin 대폭 추가 및 Pie 크기 조정 */}
            <div className="bg-white rounded-2xl p-6 border border-gray-100 h-[380px]">
              <h3 className="text-sm font-bold text-gray-800 mb-4 flex items-center gap-2"><div className="w-1 h-4 bg-orange-500 rounded-full" /> 결함 유형 분포</h3>
              <ResponsiveContainer width="100%" height="80%">
                <PieChart margin={{ left: 40, right: 40, bottom: 20 }}>
                  <Pie 
                    isAnimationActive={false} 
                    data={defectTypeData} 
                    cx="50%" cy="50%" 
                    innerRadius={40} 
                    outerRadius={60} // 크기를 살짝 줄여서 공간 확보
                    dataKey="value" 
                    labelLine={true} 
                    label={({name, percent}) => `${name} ${(percent * 100).toFixed(0)}%`}
                    style={{ fontSize: '10px', fontWeight: 'bold' }}
                  >
                    {defectTypeData.map((_, index) => <Cell key={index} fill={COLORS[index % COLORS.length]} />)}
                  </Pie>
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* 막대 차트 하단 최적화 */}
          <div className="bg-white rounded-2xl border border-gray-100 p-8 flex flex-col h-[520px]">
            <h3 className="text-sm font-bold text-gray-800 mb-6 flex items-center gap-2">
              <div className="w-1 h-4 bg-blue-500 rounded-full" /> 제품별 주요 불량 건수 (Top 10)
            </h3>
            <div className="flex-grow">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={productDefectsFixed10} margin={{ top: 20, right: 30, left: 0, bottom: 90 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f0f0f0" />
                  <XAxis dataKey="name" tick={{fontSize: 10}} interval={0} angle={-35} textAnchor="end" height={90} />
                  <YAxis tick={{fontSize: 10}} />
                  <Legend verticalAlign="bottom" align="center" height={40} iconType="circle" />
                  <Bar isAnimationActive={false} dataKey="count" name="불량 건수(NG)" fill="#3b82f6" radius={[6, 6, 0, 0]} barSize={35}>
                    <LabelList dataKey="count" position="top" style={{ fontSize: '11px', fontWeight: 'bold' }} offset={8} />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="h-4"></div>
          </div>
        </div>
      </div>
    </>
  );
}