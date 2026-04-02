// src/app/pages/AnomalyQueuePage.tsx
import React, { useEffect, useMemo, useState } from "react";
import { AnomalyCase } from "../data/mockData";
import { Search } from "lucide-react";
import { defectTypeLabel, locationLabel } from "../utils/labels";

interface AnomalyQueuePageProps {
  cases: AnomalyCase[];
  onCaseClick: (caseId: string) => void;
}

export function AnomalyQueuePage({ cases, onCaseClick }: AnomalyQueuePageProps) {
  const PAGE_SIZE = 10;
  const [page, setPage] = useState(1);
  const totalPages = Math.max(1, Math.ceil(cases.length / PAGE_SIZE));

  useEffect(() => {
    setPage(1);
  }, [cases.length]);

  const visibleCases = useMemo(() => {
    const start = (page - 1) * PAGE_SIZE;
    return cases.slice(start, start + PAGE_SIZE);
  }, [cases, page]);

  const goPrev = () => setPage((p) => Math.max(1, p - 1));
  const goNext = () => setPage((p) => Math.min(totalPages, p + 1));

  return (
    <div className="p-6 space-y-6 bg-gray-50 min-h-full">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">이상 큐</h1>
        <p className="text-gray-500 mt-1">
          총 {cases.length}개 케이스 · {page}/{totalPages} 페이지
        </p>
      </div>

      <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
        <table className="w-full text-sm text-left">
          <thead className="bg-gray-50 text-gray-500 font-medium border-b border-gray-200">
            <tr>
              <th className="px-6 py-3">시간</th>
              <th className="px-6 py-3">케이스 ID</th>
              <th className="px-6 py-3">라인</th>
              <th className="px-6 py-3">교대</th>
              <th className="px-6 py-3">제품</th>
              <th className="px-6 py-3">결함 타입</th>
              <th className="px-6 py-3">위치</th>
              <th className="px-6 py-3">판정</th>
              <th className="px-6 py-3"></th>
            </tr>
          </thead>

          <tbody className="divide-y divide-gray-100">
            {visibleCases.map((c) => (
              <tr key={c.id} className="hover:bg-gray-50 transition-colors">
                <td className="px-6 py-4 text-gray-500 whitespace-nowrap">
                  {c.timestamp.toLocaleDateString()} <br />
                  {c.timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                </td>

                <td
                  className="px-6 py-4 font-medium text-blue-600 cursor-pointer hover:underline"
                  onClick={() => onCaseClick(c.id)}
                >
                  {c.id}
                </td>

                <td className="px-6 py-4 text-gray-700">{c.line_id}</td>
                <td className="px-6 py-4 text-gray-500">{c.shift}</td>

                <td className="px-6 py-4">
                  <div className="flex flex-col">
                    <span className="text-sm text-gray-700">{c.product_group}</span>
                  </div>
                </td>

                <td className="px-6 py-4 text-gray-700">{defectTypeLabel(c.defect_type)}</td>
                <td className="px-6 py-4 text-gray-500">{locationLabel(c.location)}</td>

                <td className="px-6 py-4">
                  {c.pipeline_status === "processing" ? (
                    <div className="flex flex-col gap-1">
                      <span className="px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-700 w-fit">
                        PROCESSING
                      </span>
                      <span className="text-[11px] text-blue-600">{c.pipeline_stage || "llm_running"}</span>
                    </div>
                  ) : c.pipeline_status === "failed" ? (
                    <div className="flex flex-col gap-1">
                      <span className="px-2 py-1 rounded-full text-xs font-medium bg-red-100 text-red-700 w-fit">
                        LLM FAILED
                      </span>
                    </div>
                  ) : (
                    <span
                      className={`px-2 py-1 rounded-full text-xs font-medium ${
                        c.decision === "NG"
                          ? "bg-red-100 text-red-700"
                          : c.decision === "REVIEW"
                            ? "bg-orange-100 text-orange-700"
                            : "bg-green-100 text-green-700"
                      }`}
                    >
                      {c.decision}
                    </span>
                  )}
                </td>
              
                <td className="px-6 py-4 text-right">
                  <button
                    onClick={() => onCaseClick(c.id)}
                    className="p-1 hover:bg-gray-100 rounded text-gray-400 hover:text-blue-600"
                  >
                    <Search className="w-4 h-4" />
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>

        <div className="flex items-center justify-end gap-2 px-6 py-4 border-t border-gray-100 bg-gray-50">
          <button
            onClick={goPrev}
            disabled={page <= 1}
            className="px-3 py-1.5 text-sm border border-gray-300 rounded disabled:opacity-40 disabled:cursor-not-allowed bg-white"
          >
            이전
          </button>
          <span className="text-sm text-gray-600 min-w-20 text-center">
            {page} / {totalPages}
          </span>
          <button
            onClick={goNext}
            disabled={page >= totalPages}
            className="px-3 py-1.5 text-sm border border-gray-300 rounded disabled:opacity-40 disabled:cursor-not-allowed bg-white"
          >
            다음
          </button>
        </div>
      </div>
    </div>
  );
}
