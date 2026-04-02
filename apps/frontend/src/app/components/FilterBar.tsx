// src/app/components/FilterBar.tsx
import React from "react";
import { Filter, X } from "lucide-react";
import {
  defectTypeLabel as defectTypeLabelRaw,
  decisionLabel as decisionLabelRaw,
} from "../utils/labels";
import { packagingClasses } from "../data/mockData";

export interface FilterState {
  dateRange: string;
  line: string;
  productGroup: string;
  defectType: string;
  decision: string;
  scoreRange: [number, number];
}

interface FilterBarProps {
  filters: FilterState;
  onFilterChange: (filters: FilterState) => void;
}

function prettyLabel(s: string) {
  return (s ?? "").replace(/_/g, " ");
}

const PRODUCT_GROUP_OPTIONS = packagingClasses.map((v) => ({
  value: v,
  label: prettyLabel(v),
}));

function labelOfProductGroup(value: string) {
  const hit = PRODUCT_GROUP_OPTIONS.find((o) => o.value === value);
  return hit ? hit.label : value;
}

function labelOf(mapper: unknown, key: string) {
  if (!key) return key;
  if (typeof mapper === "function") return (mapper as (k: string) => string)(key);
  if (mapper && typeof mapper === "object") return (mapper as Record<string, string>)[key] ?? key;
  return key;
}

export function FilterBar({ filters, onFilterChange }: FilterBarProps) {
  const hasActiveFilters =
    filters.line !== "all" ||
    filters.productGroup !== "all" ||
    filters.defectType !== "all" ||
    filters.decision !== "all" ||
    filters.scoreRange[0] > 0 ||
    filters.scoreRange[1] < 1;

  const clearFilters = () => {
    onFilterChange({
      dateRange: "today",
      line: "all",
      productGroup: "all",
      defectType: "all",
      decision: "all",
      scoreRange: [0, 1],
    });
  };

  return (
    <div className="bg-white border-b border-gray-200 sticky top-0 z-10">
      <div className="px-8 py-4">
        <div className="flex items-center gap-3 flex-wrap">
          <div className="flex items-center gap-2 text-sm text-gray-700">
            <Filter className="w-4 h-4" />
            <span className="font-medium">필터</span>
          </div>

          <select
            value={filters.dateRange}
            onChange={(e) => onFilterChange({ ...filters, dateRange: e.target.value })}
            className="px-3 py-2 border border-gray-300 rounded-lg text-sm bg-white hover:border-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="today">오늘</option>
            <option value="yesterday">어제</option>
            <option value="week">최근 7일</option>
            <option value="month">최근 30일</option>
          </select>

          <select
            value={filters.line}
            onChange={(e) => onFilterChange({ ...filters, line: e.target.value })}
            className="px-3 py-2 border border-gray-300 rounded-lg text-sm bg-white hover:border-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="all">전체 라인</option>
            <option value="LINE-A-01">LINE-A-01</option>
            <option value="LINE-B-02">LINE-B-02</option>
            <option value="LINE-C-03">LINE-C-03</option>
          </select>

          <select
            value={filters.productGroup}
            onChange={(e) => onFilterChange({ ...filters, productGroup: e.target.value })}
            className="px-3 py-2 border border-gray-300 rounded-lg text-sm bg-white hover:border-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="all">전체 제품군</option>
            {PRODUCT_GROUP_OPTIONS.map((o) => (
              <option key={o.value} value={o.value}>
                {o.label}
              </option>
            ))}
          </select>

          <select
            value={filters.defectType}
            onChange={(e) => onFilterChange({ ...filters, defectType: e.target.value })}
            className="px-3 py-2 border border-gray-300 rounded-lg text-sm bg-white hover:border-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="all">전체 결함 타입</option>
            <option value="seal_issue">{labelOf(defectTypeLabelRaw, "seal_issue")}</option>
            <option value="contamination">{labelOf(defectTypeLabelRaw, "contamination")}</option>
            <option value="crack">{labelOf(defectTypeLabelRaw, "crack")}</option>
            <option value="missing_component">{labelOf(defectTypeLabelRaw, "missing_component")}</option>
            <option value="scratch">{labelOf(defectTypeLabelRaw, "scratch")}</option>
            <option value="other">{labelOf(defectTypeLabelRaw, "other")}</option>
          </select>

          <select
            value={filters.decision}
            onChange={(e) => onFilterChange({ ...filters, decision: e.target.value })}
            className="px-3 py-2 border border-gray-300 rounded-lg text-sm bg-white hover:border-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="all">전체 판정</option>
            <option value="OK">{labelOf(decisionLabelRaw, "OK")}</option>
            <option value="NG">{labelOf(decisionLabelRaw, "NG")}</option>
            <option value="REVIEW">{labelOf(decisionLabelRaw, "REVIEW")}</option>
          </select>

          {hasActiveFilters && (
            <button
              onClick={clearFilters}
              className="flex items-center gap-1 px-3 py-2 text-sm text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors"
            >
              <X className="w-4 h-4" />
              <span>초기화</span>
            </button>
          )}
        </div>

        {hasActiveFilters && (
          <div className="flex items-center gap-2 mt-3 flex-wrap">
            <span className="text-xs text-gray-500">활성 필터:</span>

            {filters.line !== "all" && (
              <span className="px-2 py-1 bg-blue-50 text-blue-700 text-xs rounded">
                라인: {filters.line}
              </span>
            )}

            {filters.productGroup !== "all" && (
              <span className="px-2 py-1 bg-blue-50 text-blue-700 text-xs rounded">
                제품군: {labelOfProductGroup(filters.productGroup)}
              </span>
            )}

            {filters.defectType !== "all" && (
              <span className="px-2 py-1 bg-blue-50 text-blue-700 text-xs rounded">
                결함: {labelOf(defectTypeLabelRaw, filters.defectType)}
              </span>
            )}

            {filters.decision !== "all" && (
              <span className="px-2 py-1 bg-blue-50 text-blue-700 text-xs rounded">
                판정: {labelOf(decisionLabelRaw, filters.decision)}
              </span>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
