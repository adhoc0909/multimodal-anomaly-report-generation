import type { AnomalyCase, ActionLog } from "./mockData";

/**
 * 백엔드 실데이터(DB) -> 프론트 AnomalyCase 매핑
 */

const PACKAGING_CLASS_LABEL: Record<string, string> = {
  cigarette_box: "cigarette box",
  drink_bottle: "drink bottle",
  drink_can: "drink can",
  food_bottle: "food_bottle",
  food_box: "food box",
  food_package: "food package",
  breakfast_box: "breakfast box",
  juice_bottle: "juice bottle",
  pushpins: "pushpins",
  screw_bag: "screw bag",
};

const LINES = ["LINE-A-01", "LINE-B-02", "LINE-C-03"];

function hash01(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return (h >>> 0) / 4294967295;
}

function parseDatetime(dt?: string | null): Date | null {
  if (!dt) return null;
  const cleaned = String(dt).replace(/(\.\d{3})\d+/, "$1");
  const d = new Date(cleaned);
  return isNaN(d.getTime()) ? null : d;
}

function parseJsonLike(input: unknown): any {
  if (!input) return {};
  if (typeof input === "object") return input;
  if (typeof input === "string") {
    const trimmed = input.trim();
    if (!trimmed) return {};
    try {
      return JSON.parse(trimmed);
    } catch {
      return { _text: trimmed };
    }
  }
  return {};
}

function pickString(values: unknown[]): string | undefined {
  for (const v of values) {
    if (typeof v === "string" && v.trim()) return v.trim();
  }
  return undefined;
}

function toNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && value.trim()) {
    const n = Number(value);
    if (Number.isFinite(n)) return n;
  }
  return null;
}

function toProductGroup(category: string): string {
  // 'GoodsAD/cigarette_box' 같은 경우 처리
  const pureCat = category.includes('/') ? category.split('/').pop()! : category;
  return PACKAGING_CLASS_LABEL[pureCat] ?? pureCat.replace(/_/g, " ");
}

function toLineId(seed: string): string {
  const v = hash01(seed);
  return LINES[Math.floor(v * LINES.length) % LINES.length];
}

function toShift(d: Date): string {
  const hour = d.getHours();
  return hour >= 7 && hour < 19 ? "주간" : "야간";
}

function normalizeDefectType(
  raw?: string,
  decision?: "OK" | "NG" | "REVIEW",
  hintText?: string
): string {
  if (decision === "OK") return "none";
  const v = (raw ?? "").trim().toLowerCase().replace(/\s+/g, "_").replace(/-/g, "_");

  const map: Record<string, string> = {
    seal_issue: "seal_issue",
    sealing_issue: "seal_issue",
    seal: "seal_issue",
    sealing: "seal_issue",
    contamination: "contamination",
    contaminant: "contamination",
    foreign_object: "contamination",
    foreign_material: "contamination",
    crack: "crack",
    fracture: "crack",
    broken: "crack",
    damage: "crack",
    missing_component: "missing_component",
    missing_part: "missing_component",
    missing: "missing_component",
    scratch: "scratch",
    scuff: "scratch",
    abrasion: "scratch",
    other: "other",
    기타: "other",
    unknown: "other",
    기타결함: "other",
    실링: "seal_issue",
    밀봉: "seal_issue",
    오염: "contamination",
    이물: "contamination",
    파손: "crack",
    균열: "crack",
    누락: "missing_component",
    결손: "missing_component",
    스크래치: "scratch",
    긁힘: "scratch",
  };

  if (v && map[v]) return map[v];

  const text = `${v} ${(hintText ?? "").toLowerCase()}`;
  if (/(seal|sealing|unseal|unsealed|open|opened|opening|flap|lid|cap|실링|밀봉|개봉|열림|벌어짐|뜯김)/.test(text)) {
    return "seal_issue";
  }
  if (/(contamin|foreign|dust|stain|smudge|오염|이물|얼룩)/.test(text)) return "contamination";
  if (/(crack|fracture|broken|damage|tear|파손|균열|찢)/.test(text)) return "crack";
  if (/(missing|absence|lost|누락|결손|빠짐|없)/.test(text)) return "missing_component";
  if (/(scratch|scuff|abrasion|스크래치|긁힘)/.test(text)) return "scratch";

  if (["none", "normal", "ok", "good", "no_defect", "no-defect", ""].includes(v)) {
    return decision === "OK" ? "none" : "other";
  }
  // 비표준/불명확 타입은 프론트에서 "기타"로 묶어 과도한 오분류를 줄인다.
  return decision === "OK" ? "none" : "other";
}

function normalizeLocation(raw?: string, decision?: "OK" | "NG" | "REVIEW"): string {
  if (decision === "OK") return "none";
  const base = (raw ?? "").trim().toLowerCase();
  if (!base) return "center";

  const compact = base.replace(/_/g, "-").replace(/\s+/g, "-");
  const alias: Record<string, string> = {
    centre: "center",
    middle: "center",
    "middle-center": "center",
    "top-center": "top",
    "bottom-center": "bottom",
    "mid-left": "middle-left",
    "mid-right": "middle-right",
  };
  const normalized = alias[compact] ?? compact;
  if (["none", "unknown", "na", "n/a", "-"].includes(normalized)) {
    return decision === "NG" ? "center" : "none";
  }
  return normalized;
}

/**
 * 백엔드의 'ad_decision' (normal, anomaly, review_needed)을
 * 프론트엔드 UI용 (OK, NG, REVIEW)으로 변환
 */
function normalizeDecision(decisionRaw?: string): "OK" | "NG" | "REVIEW" {
  const d = (decisionRaw ?? "").trim().toLowerCase();
  if (d === "normal" || d === "ok") return "OK";
  if (d === "anomaly" || d === "ng") return "NG";
  return "REVIEW"; // review_needed 포함
}

function normalizeDecisionFromLlm(
  row: any,
  llmReport: any
): "OK" | "NG" | "REVIEW" {
  const llmFlagRaw =
    row?.is_anomaly_llm ??
    row?.is_anomaly_LLM ??
    llmReport?.is_anomaly;

  if (typeof llmFlagRaw === "boolean") {
    return llmFlagRaw ? "NG" : "OK";
  }
  if (typeof llmFlagRaw === "string") {
    const v = llmFlagRaw.trim().toLowerCase();
    if (["true", "1", "yes", "anomaly", "ng", "defect", "bad", "불량", "이상"].includes(v)) return "NG";
    if (["false", "0", "no", "normal", "ok", "good", "정상"].includes(v)) return "OK";
  }

  const st = String(row?.pipeline_status ?? "").toLowerCase();
  if (st === "processing" || st === "pending") return "REVIEW";
  if (st === "failed") return "REVIEW";
  // 최종 판정 기준은 LLM only.
  return "REVIEW";
}

function normalizeSeverity(raw?: string, decision?: "OK" | "NG" | "REVIEW"): "low" | "med" | "high" {
  if (decision && decision !== "NG") return "low";
  const s = (raw ?? "").trim().toLowerCase();
  if (s === "high") return "high";
  if (s === "med" || s === "medium") return "med";
  return "low";
}

function toActionLog(decision: "OK" | "NG" | "REVIEW", ts: Date): ActionLog[] {
  const base = ts.getTime();
  if (decision === "OK") return [{ who: "System", when: new Date(base + 1000), what: "자동 승인" }];
  if (decision === "REVIEW") return [{ who: "박철수", when: new Date(base + 60_000), what: "재검 요청" }];
  return [
    { who: "System", when: new Date(base + 1000), what: "AI 불량 감지" },
    { who: "Operator", when: new Date(base + 60_000), what: "불량 확정" }
  ];
}

export function mapReportsToAnomalyCases(raw: any[]): AnomalyCase[] {
  return raw.map((r, idx) => {
    const ts = parseDatetime(r.created_at) ?? parseDatetime(r.datetime) ?? new Date();

    const dataset = r.dataset ?? "UNKNOWN";
    const category = r.category ?? "unknown";

    const llmReportRoot: any = parseJsonLike(r.llm_report);
    const llmReport: any = parseJsonLike(llmReportRoot.report ?? llmReportRoot);
    const llmSummaryObj: any = parseJsonLike(r.llm_summary);
    const decision = normalizeDecisionFromLlm(r, llmReport);

    const defectType = normalizeDefectType(
      pickString([
        llmReport.anomaly_type,
        llmReport.defect_type,
        llmSummaryObj.anomaly_type,
        llmSummaryObj.defect_type,
        r.defect_type,
      ]) ?? (r.has_defect ? "anomaly" : "none"),
      decision,
      pickString([
        llmReport.description,
        llmReport.recommendation,
        llmReport.raw_response,
        llmSummaryObj.summary,
        llmSummaryObj._text,
        r.defect_description,
      ])
    );
    const loc = normalizeLocation(
      pickString([
        llmReport.location,
        llmReport.defect_location?.region,
        llmSummaryObj.location,
        r.location,
        r.region,
      ]) ?? "center",
      decision
    );
    const severity = normalizeSeverity(llmReport.severity, decision);
    const adScore = toNumber(r.ad_score) ?? toNumber(r.confidence) ?? 0;
    const areaRatio = toNumber(r.area_ratio) ?? 0;
    const pipelineStatus = String(r.pipeline_status ?? "").toLowerCase();
    const pipelineStage = String(r.pipeline_stage ?? "");
    const pipelineError = typeof r.pipeline_error === "string" ? r.pipeline_error : "";

    const llmSummary = pickString([
      llmSummaryObj.summary,
      llmSummaryObj._text,
      llmReport.summary,
      r.summary,
    ]) ?? "";
    const llmSummaryDisplay =
      llmSummary ||
      (pipelineStatus === "processing"
        ? "LLM 분석 진행 중입니다."
        : pipelineStatus === "failed"
          ? `LLM 분석 실패: ${pipelineError || "원인 미상"}`
          : "");

    return {
      id: `CASE-${r.id || idx}`,
      timestamp: ts,
      line_id: r.line ?? toLineId(`${dataset}-${category}`),
      shift: toShift(ts),

      product_group: toProductGroup(category),
      image_id: r.image_path ? r.image_path.split('/').pop() : `img_${idx}.jpg`,

      // 이미지 경로 매핑
      image_path: r.image_path ?? undefined,
      heatmap_path: r.heatmap_path ?? undefined,
      overlay_path: r.overlay_path ?? undefined,

      decision,
      anomaly_score: adScore,
      threshold: (r.applied_policy?.t_high) ?? 0.7,

      defect_type: defectType,
      defect_confidence: areaRatio || (toNumber(r.confidence) ?? 0),
      location: loc,
      affected_area_pct: areaRatio * 100,
      severity,

      model_name: "EfficientAD",
      model_version: "v1.0.0",
      inference_time_ms:
        typeof r.ad_inference_duration === "number"
          ? Math.round(r.ad_inference_duration * 1000)
          : (typeof r.inference_time === "number" ? Math.round(r.inference_time * 1000) : 0),

      // LLM 요약 매핑
      llm_summary: llmSummaryDisplay,

      pipeline_status: pipelineStatus || undefined,
      pipeline_stage: pipelineStage || undefined,
      pipeline_error: pipelineError || undefined,
      llm_structured_json: { source: r },
      operator_note: typeof r.llm_report === "string" ? r.llm_report.trim() : "",
      action_log: toActionLog(decision, ts),
    };
  });
}
