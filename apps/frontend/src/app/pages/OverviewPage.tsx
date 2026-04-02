// src/app/pages/OverviewPage.tsx
import React, { useMemo } from "react";
// import type { AnomalyCase } from "../data/mockData";
// import type { Alert } from "../data/AlertData";
import {
  Activity,
  AlertTriangle,
  CheckCircle,
  AlertOctagon,
  FileText,
  Search,
  TrendingUp,
  Clock,
} from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { buildAggregates, toHourlyDecisionTrend } from "../selectors/caseSelectors";
import { defectTypeLabel } from "../utils/labels";

function KPICard({
  title,
  value,
  subtext,
  icon: Icon,
}: {
  title: string;
  value: string;
  subtext: string;
  icon: React.ElementType;
}) {
  return (
    <div className="bg-white p-6 rounded-xl border border-gray-200 shadow-sm flex items-start justify-between">
      <div>
        <p className="text-sm font-medium text-gray-500 mb-1">{title}</p>
        <h3 className="text-3xl font-bold text-gray-900 mb-2">{value}</h3>
        <div className="flex items-center text-sm text-gray-400">{subtext}</div>
      </div>
      <div className="p-3 bg-gray-50 rounded-lg">
        <Icon className="w-6 h-6 text-gray-600" />
      </div>
    </div>
  );
}

function alertDescription(a: Alert) {
  if (a.type === "review") {
    const pg = a.product_group ?? "제품";
    const s = typeof a.score === "number" ? ` (Score: ${a.score.toFixed(2)})` : "";
    return `${pg} 제품에 ${defectTypeLabel(a.defect_type)}가 의심됩니다.${s} 육안 검사가 필요합니다.`;
  }

  if (a.type === "critical") {
    const line = a.line_id ?? "라인";
    const conf = typeof a.confidence === "number" ? `${Math.round(a.confidence * 100)}%` : "";
    return `${line}에서 ${defectTypeLabel(a.defect_type)}이(가) 감지되었습니다. 즉시 확인 바랍니다.`;
  }

  if (a.type === "consecutive") {
    const line = a.line_id ?? "라인";
    const n = a.count ?? 3;
    const loc = a.location ?? "동일 위치";
    return `${line}에서 동일한 위치(${loc})의 결함이 ${n}건 연속 발생했습니다. 공정 점검을 권장합니다.`;
  }

  return a.description ?? "";
}

interface OverviewPageProps {
  cases: AnomalyCase[];
  alerts: Alert[];
  activeModel: string;
}

export function OverviewPage({ cases, alerts, activeModel }: OverviewPageProps) {
  const agg = useMemo(() => buildAggregates(cases), [cases]);

  const defectRate = agg.total ? ((agg.ng / agg.total) * 100).toFixed(1) : "0.0";
  const reviewRate = agg.total ? ((agg.review / agg.total) * 100).toFixed(1) : "0.0";
  const processingCount = useMemo(
    () => cases.filter((c) => c.pipeline_status === "processing").length,
    [cases]
  );
  const hourlyTrend = useMemo(() => toHourlyDecisionTrend(agg), [agg]);

  const getAlertIcon = (type: Alert["type"]) => {
    switch (type) {
      case "critical":
        return AlertOctagon;
      case "review":
        return Search;
      case "report":
        return FileText;
      case "consecutive":
        return TrendingUp;
      default:
        return Activity;
    }
  };

  return (
    <div className="p-6 space-y-6 bg-gray-50 min-h-full">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">실시간 품질 검사 현황 및 이상 탐지 결과</h1>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <KPICard title="검사 수" value={agg.total.toLocaleString()} subtext="오늘 누적 검사량" icon={Activity} />
        <KPICard title="불량률" value={`${defectRate}%`} subtext="오늘 검사 대비 불량" icon={AlertTriangle} />
        <KPICard title="재검률" value={`${reviewRate}%`} subtext="AI 판정 보류 건" icon={CheckCircle} />
        <KPICard title="AI 분석 진행중" value={processingCount.toString()} subtext="LLM 처리 대기/진행 건" icon={Clock} />
      </div>

      <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-6">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-2">
            <Activity className="w-5 h-5 text-gray-700" />
            <h2 className="text-lg font-bold text-gray-900">실시간 탐지 알림</h2>
          </div>
        </div>

        <div className="space-y-4">
          {alerts.length === 0 ? (
            <div className="p-4 rounded-lg border border-gray-200 bg-gray-50 text-sm text-gray-500">
              실시간 알림이 없습니다.
            </div>
          ) : (
            alerts.map((alert) => {
              const Icon = getAlertIcon(alert.type);
              const isCritical = alert.severity === "high";
              const isMedium = alert.severity === "med";

              return (
                <div
                  key={alert.id}
                  className={`p-4 rounded-lg border flex gap-4 items-start ${
                    isCritical
                      ? "bg-red-50 border-red-100"
                      : isMedium
                        ? "bg-orange-50 border-orange-100"
                        : "bg-blue-50 border-blue-100"
                  }`}
                >
                  <div className="mt-1">
                    <Icon
                      className={`w-5 h-5 ${
                        isCritical ? "text-red-600" : isMedium ? "text-orange-600" : "text-blue-600"
                      }`}
                    />
                  </div>

                  <div className="flex-1">
                    <div className="flex justify-between items-start">
                      <h3
                        className={`font-semibold ${
                          isCritical ? "text-red-900" : isMedium ? "text-orange-900" : "text-blue-900"
                        }`}
                      >
                        {alert.title}
                      </h3>
                      <span className="text-xs text-gray-500">
                        {alert.timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                      </span>
                    </div>

                    <p
                      className={`text-sm mt-1 ${
                        isCritical ? "text-red-700" : isMedium ? "text-orange-700" : "text-blue-700"
                      }`}
                    >
                      {alertDescription(alert)}
                    </p>
                  </div>
                </div>
              );
            })
          )}
        </div>
      </div>

      <div className="bg-white border border-gray-200 rounded-lg p-6">
        <h3 className="text-lg font-bold text-gray-900 mb-4">시간대별 검사 결과</h3>
        <ResponsiveContainer width="100%" height={320}>
          <BarChart data={hourlyTrend}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis dataKey="hour" tick={{ fontSize: 12 }} stroke="#6b7280" interval={0} />
            <YAxis tick={{ fontSize: 12 }} stroke="#6b7280" />
            <Tooltip />
            <Legend />
            <Bar dataKey="불량" fill="#ef4444" stackId="a" />
            <Bar dataKey="재검토" fill="#f59e0b" stackId="a" />
            <Bar dataKey="정상" fill="#10b981" stackId="a" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
