import React, { forwardRef } from "react";
import { defectTypeLabel, locationLabel } from "../utils/labels";
import { getCaseImageUrl } from "../services/media";

export interface DBReport {
  id: string | number;
  image_id: string;
  category: string;
  timestamp: string;
  defect_type: string;
  location: string;
  affected_area_pct: number;
  ad_score: number;
  confidence: number;
  decision: "ok" | "ng" | "pending" | "anomaly" | "normal";
  llm_analysis_summary: string;
  image_path: string;
  heatmap_path: string;
  overlay_path: string;
}

interface ReportTemplateProps {
  reportData: DBReport;
}

// 1. 변수명을 CaseDetailPage에서 불러오는 이름인 'ReportTemplate'으로 직접 정의합니다.
export const ReportTemplate = forwardRef<HTMLDivElement, ReportTemplateProps>(
  ({ reportData }, ref) => {
    const rawDate = new Date(reportData.timestamp);
    const formattedDate = isNaN(rawDate.getTime()) ? "-" : rawDate.toLocaleString("ko-KR");
    const originalUrl = getCaseImageUrl(reportData as any, "original");
    const heatmapUrl = getCaseImageUrl(reportData as any, "heatmap");

    const isAnomaly = reportData.decision === "ng" || reportData.decision === "anomaly";

    return (
      <div
        ref={ref}
        style={{
          width: '794px',
          minHeight: '1123px',
          padding: '40px',
          backgroundColor: '#ffffff',
          color: '#111827',
          fontFamily: 'sans-serif',
          boxSizing: 'border-box',
          display: 'flex',
          flexDirection: 'column',
          position: 'relative'
        }}
      >
        {/* 헤더 */}
        <div style={{ borderBottom: '2px solid #111827', paddingBottom: '16px', marginBottom: '24px' }}>
          <h1 style={{ fontSize: '24px', fontWeight: 'bold', margin: 0 }}>품질 검사 결과 보고서</h1>
          <p style={{ fontSize: '12px', color: '#6b7280', marginTop: '8px' }}>
            발행 일시: {formattedDate} | Case ID: #{reportData.id}
          </p>
        </div>

        {/* 이미지 섹션 */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '24px' }}>
          <div>
            <p style={{ fontSize: '12px', fontWeight: 'bold', marginBottom: '8px' }}>원본 이미지</p>
            <div style={{ border: '1px solid #e5e7eb', borderRadius: '4px', overflow: 'hidden', height: '260px', backgroundColor: '#f9fafb' }}>
              <img 
                src={originalUrl} 
                alt="Original" 
                crossOrigin="anonymous" 
                style={{ width: '100%', height: '100%', objectFit: 'contain' }} 
              />
            </div>
          </div>
          <div>
            <p style={{ fontSize: '12px', fontWeight: 'bold', marginBottom: '8px' }}>이상 탐지 (Heatmap)</p>
            <div style={{ border: '1px solid #e5e7eb', borderRadius: '4px', overflow: 'hidden', height: '260px', backgroundColor: '#f9fafb' }}>
              <img 
                src={heatmapUrl} 
                alt="Heatmap" 
                crossOrigin="anonymous" 
                style={{ width: '100%', height: '100%', objectFit: 'contain' }} 
              />
            </div>
          </div>
        </div>

        {/* 상세 정보 테이블 */}
        <section>
          <h3 style={{ 
            fontSize: '16px', 
            borderLeft: '4px solid #1d4ed8', 
            paddingLeft: '8px', 
            marginBottom: '12px' 
          }}>
            검사 상세 정보
          </h3>
          <table style={{ 
            width: '100%', 
            borderCollapse: 'collapse', 
            border: '1px solid #e5e7eb', 
            fontSize: '13px', 
            tableLayout: 'fixed' 
          }}>
            <tbody>
              {[
                { label: "결함 유형", value: defectTypeLabel(reportData.defect_type) },
                { label: "결함 위치", value: locationLabel(reportData.location) },
                { 
                  label: "AI 분석 요약", 
                  value: reportData.llm_analysis_summary, 
                  isLong: true 
                },
                { 
                  label: "신뢰도 / 점수", 
                  value: `정확도: ${(reportData.confidence * 100).toFixed(1)}% | 이상 점수: ${reportData.ad_score.toFixed(2)}` 
                }
              ].map((row, i) => (
                <tr key={i}>
                  <td style={{ 
                    width: '120px', 
                    backgroundColor: '#f9fafb', 
                    border: '1px solid #e5e7eb', 
                    fontWeight: 'bold',
                    /* 위쪽(6px)보다 아래쪽(10px) 패딩을 더 주어 시각적 중앙 정렬 */
                    padding: '6px 12px 10px 12px', 
                    verticalAlign: 'middle'
                  }}>
                    {row.label}
                  </td>
                  <td style={{ 
                    padding: '6px 12px 10px 12px', 
                    border: '1px solid #e5e7eb', 
                    verticalAlign: 'middle', 
                    lineHeight: '1.4',
                    whiteSpace: 'pre-wrap',
                    wordBreak: 'break-all'
                  }}>
                    {/* 데이터 앞뒤의 숨겨진 줄바꿈(\n)이나 공백을 강제로 제거 */}
                    {row.value?.toString().trim().replace(/^\s+|\s+$/g, '') || "내용 없음"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>

        {/* 조치 권고 사항 */}
        {isAnomaly && (
          <div style={{ marginTop: '24px', padding: '16px', backgroundColor: '#fef2f2', border: '1px solid #fecaca', borderRadius: '4px' }}>
            <h4 style={{ color: '#991b1b', fontSize: '14px', margin: '0 0 8px 0' }}>⚠️ 조치 권고 사항</h4>
            <p style={{ fontSize: '13px', color: '#b91c1c', margin: 0 }}>
              해당 부품에서 이상 징후가 발견되었습니다. 즉시 라인을 점검하고 해당 배치의 제품을 전수 조사할 것을 권장합니다.
            </p>
          </div>
        )}
      </div>
    );
  }
);

ReportTemplate.displayName = "ReportTemplate";