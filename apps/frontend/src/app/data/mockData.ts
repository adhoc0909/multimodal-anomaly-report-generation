// src/app/data/mockData.ts

export interface ActionLog {
  who: string;
  when: Date;
  what: string;
}

export interface AnomalyCase {
  id: string;
  timestamp: Date;
  line_id: string;
  shift: string;

  product_group: string;
  image_id: string;
  
  decision: "OK" | "NG" | "REVIEW";
  anomaly_score: number;
  threshold: number;
  
  defect_type: string;
  defect_confidence: number;
  location: string;
  affected_area_pct: number;
  severity: "low" | "med" | "high";
  
  model_name: string;
  model_version: string;
  inference_time_ms: number;
  
  llm_summary: string;
  llm_structured_json: any;
  pipeline_status?: "processing" | "completed" | "failed" | string;
  pipeline_stage?: string;
  pipeline_error?: string;
  operator_note?: string;
  action_log: ActionLog[];
  image_path?: string;
  heatmap_path?: string;
  overlay_path?: string;
}

// 상수 데이터
export const packagingClasses = [
  "cigarette box",
  "drink bottle",
  "drink can",
  "food_bottle",
  "food box",
  "food package",
  "breakfast box",
  "juice bottle",
  "pushpins",
  "screw bag",
] as const;

export const defectTypes = [
  "scratch",
  "contamination",
  "crack",
  "missing_component",
  "seal_issue",
];

export const lines = ["LINE-A-01", "LINE-B-02", "LINE-C-03"];

// Mock 데이터
export const mockCases: AnomalyCase[] = [
  {
    id: "CASE-2026-0208-001",
    timestamp: new Date("2026-02-08T14:32:15"),
    line_id: "LINE-A-01",
    shift: "주간",
    product_group: "food box",
    image_id: "IMG-A01-20260208-143215",
    decision: "NG",
    anomaly_score: 0.87,
    threshold: 0.65,
    defect_type: "seal_issue",
    defect_confidence: 0.91,
    location: "top-right",
    affected_area_pct: 12.3,
    severity: "high",
    model_name: "PatchCore",
    model_version: "v2.3.1",
    inference_time_ms: 142,
    llm_summary: "포장 상단 우측에서 실링 불량이 감지되었습니다.",
    llm_structured_json: {
      defect_location: { x: 0.78, y: 0.15 },
      recommendation: "실링 헤드 온도 점검 필요",
    },
    operator_note: "라인 정지 후 재검사 완료",
    action_log: [
      { who: "김민수", when: new Date("2026-02-08T14:33:00"), what: "불량 확정" },
    ],
  },
  {
    id: "CASE-2026-0208-002",
    timestamp: new Date("2026-02-08T13:45:22"),
    line_id: "LINE-B-02",
    shift: "주간",
    product_group: "drink bottle", 
    image_id: "IMG-B02-20260208-134522",
    decision: "REVIEW",
    anomaly_score: 0.68,
    threshold: 0.65,
    defect_type: "contamination",
    defect_confidence: 0.73,
    location: "center",
    affected_area_pct: 3.2,
    severity: "med",
    model_name: "PatchCore", 
    model_version: "v1.8.2",
    inference_time_ms: 218,
    llm_summary: "병 중앙부에 미세한 이물질이 의심됩니다. 육안 재확인이 필요합니다.",
    llm_structured_json: {
      defect_location: { x: 0.51, y: 0.48 },
      recommendation: "육안 검사 재확인",
    },
    action_log: [
      { who: "박철수", when: new Date("2026-02-08T13:46:00"), what: "재검 요청" },
    ],
  },
  {
    id: "CASE-2026-0208-003",
    timestamp: new Date("2026-02-08T13:21:08"),
    line_id: "LINE-A-01",
    shift: "주간",
    product_group: "food package", 
    image_id: "IMG-A01-20260208-132108",
    decision: "NG",
    anomaly_score: 0.92,
    threshold: 0.65,
    defect_type: "crack",
    defect_confidence: 0.95,
    location: "bottom-left",
    affected_area_pct: 18.7,
    severity: "high",
    model_name: "EfficientAD",
    model_version: "v3.1.0",
    inference_time_ms: 95,
    llm_summary: "포장재 하단 좌측에 명확한 파손이 확인되었습니다.",
    llm_structured_json: {
      defect_location: { x: 0.18, y: 0.82 },
      recommendation: "즉시 폐기 필요",
    },
    action_log: [
      { who: "김민수", when: new Date("2026-02-08T13:22:00"), what: "불량 확정" },
    ],
  },
];
