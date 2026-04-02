// src/app/api/reportsApi.ts
import { apiRequest, QueryParams } from "./http";

const REPORTS_PATH =
  (import.meta.env.VITE_REPORTS_PATH as string | undefined) ?? "/reports";

export type ReportDTO = {
  id: number;
  filename: string;
  image_path: string;
  dataset: string;
  category: string;
  ground_truth: string | null;
  decision: string;
  confidence: number | null;

  has_defect: number;
  defect_type: string;
  location: string;
  severity: string;

  defect_description: string;
  possible_cause: string;
  product_description: string;

  summary: string;
  impact: string;
  recommendation: string;

  inference_time: number | null; // seconds
  datetime: string;

  heatmap_path?: string | null;
  overlay_path?: string | null;
  pipeline_status?: string | null;
  pipeline_stage?: string | null;
  pipeline_error?: string | null;

};

export type ReportListDTO = {
  items: ReportDTO[];
  total: number;
};

export type ReportListQuery = {
  limit?: number;
  offset?: number;
  dataset?: string;
  category?: string;
  decision?: string;
  date_from?: string;
  date_to?: string;
};

async function fetchReportsRaw(query: ReportListQuery, opts?: { signal?: AbortSignal }) {
  return apiRequest<unknown>(REPORTS_PATH, {
    query: query as QueryParams,
    signal: opts?.signal,
    timeoutMs: 45000,
  });
}


export async function fetchReports(
  query: ReportListQuery,
  opts?: { signal?: AbortSignal }
): Promise<ReportListDTO> {
  const data = await fetchReportsRaw(query, opts);

  // 1) 서버가 배열로 주든, {items,total}로 주든 통일해서 rawItems로 뽑기
  const rawItems = Array.isArray(data)
    ? data
    : ((data as any)?.items ?? []);

  // 2) rawItems를 "항상 ReportDTO"로 정규화
  const items = (rawItems as any[]).map((x, i) => normalizeRemoteToReportDTO(x, i));

  // 3) total 계산(없으면 items.length)
  const total = Array.isArray(data)
    ? items.length
    : Number((data as any)?.total ?? items.length);

  return { items, total };
}

export async function fetchReportsAll(
  baseQuery?: Omit<ReportListQuery, "limit" | "offset">,
  opts?: { signal?: AbortSignal; pageSize?: number; maxItems?: number }
): Promise<ReportDTO[]> {
  const pageSize = opts?.pageSize ?? 500;
  const maxItems = opts?.maxItems ?? 5000;

  let offset = 0;
  let out: ReportDTO[] = [];
  let total = Infinity;

  while (offset < total && out.length < maxItems) {
    const { items, total: t } = await fetchReports(
      { ...(baseQuery ?? {}), limit: pageSize, offset },
      { signal: opts?.signal }
    );

    total = Number.isFinite(t) ? t : Infinity;
    out = out.concat(items);

    if (items.length === 0) break;
    offset += items.length;
  }

  return out.slice(0, maxItems);
}

function normLocation(raw?: string) {
  const s = (raw ?? "").trim().toLowerCase();
  if (!s) return "none";
  // "top right" -> "top-right"
  return s.replace(/\s+/g, "-");
}

function normSeverity(raw?: string) {
  const s = (raw ?? "").trim().toLowerCase();
  if (s === "high") return "high";
  if (s === "medium" || s === "med") return "med";
  return "low";
}

function basename(p?: string) {
  const s = (p ?? "").trim();
  if (!s) return "";
  return s.split("/").pop() ?? s;
}


// 팀원 서버 응답 -> ReportDTO로 변환 (추가)
function normalizeRemoteToReportDTO(x: any, idx: number): ReportDTO {
  const llm = x?.llm_report ?? {};
  const sum = x?.llm_summary ?? {};

  // decision 우선순위: is_anomaly_llm -> (processing이면 review) -> ad_decision
  const pipelineStatus = String(x?.pipeline_status ?? "").toLowerCase();
  const llmFlag =
    typeof x?.is_anomaly_llm === "boolean" ? x.is_anomaly_llm : null;
  const decision =
    llmFlag === true
      ? "ng"
      : llmFlag === false
        ? "ok"
        : pipelineStatus === "processing"
          ? "review_needed"
          : typeof x?.ad_decision === "string"
            ? x.ad_decision
            : typeof x?.decision === "string"
              ? x.decision
              : "review_needed";

  const hasDefect =
    typeof x?.has_defect === "number"
      ? x.has_defect
      : typeof x?.has_defect === "boolean"
        ? (x.has_defect ? 1 : 0)
        : (decision.toLowerCase() === "ng" || decision.toLowerCase() === "anomaly" ? 1 : 0);

  const adScore =
    typeof x?.ad_score === "number"
      ? x.ad_score
      : (typeof llm?.confidence === "number" ? llm.confidence : null);

  // 서버 절대경로(/home/...)는 media.normalizeUrl에서 /outputs/{filename}로 정규화한다.
  // 여기서 비워버리면 상세 원본 이미지가 표시되지 않으므로 원본 값을 유지한다.
  const rawImagePath = typeof x?.image_path === "string" ? x.image_path : "";
  const image_path = rawImagePath;
  const overlayPath = x?.overlay_path ?? x?.mask_path ?? null;

  const item = {
    id: Number(x?.id ?? idx + 1),

    filename: String(x?.filename ?? basename(rawImagePath) ?? `remote_${idx}.png`),
    image_path,

    heatmap_path: x?.heatmap_path ?? null,
    overlay_path: overlayPath,

    dataset: String(x?.dataset ?? "remote"),
    category: String(x?.category ?? llm?.anomaly_type ?? "unknown"),
    ground_truth: x?.ground_truth ?? null,

    decision: String(decision),
    confidence: typeof adScore === "number" ? adScore : null,

    has_defect: hasDefect,

    defect_type: String(x?.defect_type ?? llm?.anomaly_type ?? "none"),
    location: normLocation(llm?.location ?? x?.location ?? x?.region),
    severity: normSeverity(llm?.severity ?? sum?.risk_level ?? x?.severity),

    defect_description: String(x?.defect_description ?? llm?.description ?? ""),
    possible_cause: String(x?.possible_cause ?? ""),
    product_description: String(x?.product_description ?? ""),

    summary: String(x?.summary ?? sum?.summary ?? ""),
    impact: String(x?.impact ?? ""),
    recommendation: String(x?.recommendation ?? llm?.recommendation ?? ""),

    inference_time:
      typeof x?.inference_time === "number"
        ? x.inference_time
        : (typeof x?.llm_inference_duration === "number" ? x.llm_inference_duration : null),

    datetime: String(x?.datetime ?? x?.created_at ?? x?.llm_start_time ?? new Date().toISOString()),

    // Mapper 호환용 원본 필드 패스스루
    created_at: x?.created_at ?? null,
    ad_decision: x?.ad_decision ?? null,
    ad_score: x?.ad_score ?? null,
    area_ratio: x?.area_ratio ?? null,
    region: x?.region ?? null,
    applied_policy: x?.applied_policy ?? null,
    ad_inference_duration: x?.ad_inference_duration ?? null,
    llm_report: x?.llm_report ?? null,
    llm_summary: x?.llm_summary ?? null,
    is_anomaly_llm: x?.is_anomaly_llm ?? null,
    pipeline_status: pipelineStatus || null,
    pipeline_stage: x?.pipeline_stage ?? null,
    pipeline_error: x?.pipeline_error ?? null,
  } as ReportDTO & Record<string, unknown>;

  return item;
}
