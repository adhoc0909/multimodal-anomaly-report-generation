// src/app/selectors/caseSelectors.ts
import type { AnomalyCase } from "../data/mockData";
import { packagingClasses } from "../data/mockData";
import { defectTypeLabel } from "../utils/labels";

export type HourlyDecisionRow = {
  hour: string; 
  불량: number;
  재검토: number;
  정상: number;
};

export type DefectPieRow = { name: string; value: number; type: string };

export type ProductDefectRow = { key: string; name: string; count: number };

function prettyLabel(s: string) {
  return (s ?? "").replace(/_/g, " ");
}

export function buildAggregates(cases: AnomalyCase[]) {
  const total = cases.length;

  let ng = 0;
  let review = 0;
  let ok = 0;

  let sumScore = 0;
  let sumInference = 0;

  const hourly = Array.from({ length: 24 }, () => ({
    total: 0,
    ng: 0,
    review: 0,
    ok: 0,
  }));

  const defectTypeCounts: Record<string, number> = {};
  const productNgCounts: Record<string, number> = {};

  // NG 상위 5개
  const topNg: AnomalyCase[] = [];

  for (const c of cases) {
    sumScore += c.anomaly_score;
    sumInference += Number.isFinite(c.inference_time_ms) ? c.inference_time_ms : 0;

    const h = c.timestamp.getHours();
    hourly[h].total += 1;

    if (c.decision === "NG") {
      ng += 1;
      hourly[h].ng += 1;

      defectTypeCounts[c.defect_type] = (defectTypeCounts[c.defect_type] || 0) + 1;
      productNgCounts[c.product_group] = (productNgCounts[c.product_group] || 0) + 1;

      // topNg 유지(길이 5)
      topNg.push(c);
      topNg.sort((a, b) => b.anomaly_score - a.anomaly_score);
      if (topNg.length > 5) topNg.pop();
    } else if (c.decision === "REVIEW") {
      review += 1;
      hourly[h].review += 1;
    } else {
      ok += 1;
      hourly[h].ok += 1;
    }
  }

  return {
    total,
    ng,
    review,
    ok,
    avgScore: total ? sumScore / total : 0,
    avgInference: total ? sumInference / total : 0,
    hourly,
    defectTypeCounts,
    productNgCounts,
    topNg,
  };
}

export function toHourlyDecisionTrend(agg: ReturnType<typeof buildAggregates>): HourlyDecisionRow[] {
  const out: HourlyDecisionRow[] = [];
  for (let h = 0; h < 24; h++) {
    const x = agg.hourly[h];
    out.push({
      hour: `${h}시`,
      불량: x.ng,
      재검토: x.review,
      정상: x.ok,
    });
  }
  return out;
}

export function toDefectTypePie(agg: ReturnType<typeof buildAggregates>): DefectPieRow[] {
  return Object.entries(agg.defectTypeCounts).map(([type, value]) => ({
    type,
    value,
    name: defectTypeLabel(type),
  }));
}

export function toDefectRateTrend(agg: ReturnType<typeof buildAggregates>) {
  const out: Array<{ time: string; 불량률: number }> = [];
  for (let h = 0; h < 24; h++) {
    const x = agg.hourly[h];
    if (x.total === 0) continue;
    out.push({
      time: `${h}:00`,
      불량률: Number(((x.ng / x.total) * 100).toFixed(1)),
    });
  }
  return out;
}

export function toProductDefectsFixed10(agg: ReturnType<typeof buildAggregates>): ProductDefectRow[] {
  return packagingClasses.map((cls) => ({
    key: cls,
    name: prettyLabel(cls),
    count: agg.productNgCounts[cls] ?? 0,
  }));
}
