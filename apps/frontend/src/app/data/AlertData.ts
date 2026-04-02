// src/app/data/AlertData.ts
export type AlertType = "critical" | "review" | "report" | "consecutive" | "stable" | "system";

export type NotificationSettings = {
  highSeverity: boolean;
  reviewRequest: boolean;
  dailyReport: boolean;
  systemError: boolean;
  consecutiveDefects: boolean;
};

export interface Alert {
  id: string;
  type: AlertType;
  severity: "low" | "med" | "high";
  title: string;
  timestamp: Date;

  // 필드 기반 설명 생성을 위한 메타
  line_id?: string;
  product_group?: string; // review 알림에서 이 값을 우선 사용
  defect_type?: string;
  location?: string;
  confidence?: number; // 0~1
  score?: number; // 0~1
  count?: number;

  // fallback
  description?: string;
}

// 목업 알림
export const mockAlerts: Alert[] = [
  {
    id: "ALERT-1",
    type: "critical",
    severity: "high",
    title: "심각한 결함 감지 (High Severity)",
    timestamp: new Date(),
    line_id: "LINE-A-01",
    defect_type: "seal_issue",
    confidence: 0.98,
  },
  {
    id: "ALERT-2",
    type: "review",
    severity: "med",
    title: "재검토 요청 (AI 판정 보류)",
    timestamp: new Date(Date.now() - 60_000),
    product_group: "cigarette box",
    defect_type: "scratch",
    score: 0.66,
  },
  {
    id: "ALERT-3",
    type: "report",
    severity: "low",
    title: "일일 품질 리포트 생성 완료",
    timestamp: new Date(Date.now() - 120_000),
    description: "금일 발생한 불량 유형 및 원인 분석이 포함된 PDF 리포트가 생성되었습니다.",
  },
  {
    id: "ALERT-4",
    type: "consecutive",
    severity: "high",
    title: "연속 불량 감지",
    timestamp: new Date(Date.now() - 180_000),
    line_id: "LINE-C-03",
    location: "Top-Left",
    count: 3,
  },
  {
    id: "ALERT-5",
    type: "stable",
    severity: "low",
    title: "시스템 안정화 알림",
    timestamp: new Date(Date.now() - 240_000),
    description: "지난 1시간 동안 시스템 오류 없이 모든 이미지가 정상적으로 처리되었습니다.",
  },
];
