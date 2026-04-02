// src/app/App.tsx
import React, { useMemo, useState, useEffect } from "react";
import { AlertOctagon } from "lucide-react";

import { Sidebar } from "./components/Sidebar";
import { FilterBar, FilterState } from "./components/FilterBar";
import { LoadingState } from "./components/LoadingState";
import { EmptyState } from "./components/EmptyState";

import { OverviewPage } from "./pages/OverviewPage";
import { AnomalyQueuePage } from "./pages/AnomalyQueuePage";
import { CaseDetailPage } from "./pages/CaseDetailPage";
import { ReportBuilderPage } from "./pages/ReportBuilderPage";
import { SettingsPage } from "./pages/SettingsPage";
import { fetchLlmModelSettings, updateLlmModelSettings } from "./api/settingsApi";

import { AnomalyCase } from "./data/mockData";
import { Alert, NotificationSettings } from "./data/AlertData";
import { getDateRangeWindow } from "./utils/dateUtils";

import { useReportCases } from "./hooks/useReportCases";
import { useLocalStorageState } from "./hooks/useLocalStorageState";
import { useUrlPageState } from "./hooks/useUrlPageState";

const MODEL_VERSION: Record<string, string> = {
  PatchCore: "v2.3.1",
  EfficientAD: "v3.1.0",
};

const DEFAULT_NOTI: NotificationSettings = {
  highSeverity: true,
  reviewRequest: true,
  dailyReport: false,
  systemError: true,
  consecutiveDefects: true,
};

const REPORT_CASES_OPTIONS = {
  query: {},
  pageSize: 250,
  maxItems: 1000,
};

const MODEL_STORAGE_OPTIONS = {
  serialize: (v: string) => v,
  deserialize: (raw: string) => raw || "internvl",
};

const NOTIFICATION_STORAGE_OPTIONS = {
  normalize: (v: NotificationSettings) => ({ ...DEFAULT_NOTI, ...(v ?? {}) }),
};

const API_KEY_STORAGE_OPTIONS = {
  serialize: (v: string) => v || "",
  deserialize: (raw: string) => raw || "",
};

export default function App() {
  const { currentPage, selectedCaseId, navigate, openCase } = useUrlPageState();

  const [showToast, setShowToast] = useState(false);
  const [isExiting, setIsExiting] = useState(false);
  const [lastNgCount, setLastNgCount] = useState(0);

  const [activeModel, setActiveModel] = useLocalStorageState<string>(
    "activeModel",
    "internvl",
    MODEL_STORAGE_OPTIONS,
  );
  const [llmModels, setLlmModels] = useState<string[]>(["internvl"]);
  const [modelSyncing, setModelSyncing] = useState(false);

  const [notificationSettings, setNotificationSettings] =
    useLocalStorageState<NotificationSettings>(
      "notificationSettings",
      DEFAULT_NOTI,
      NOTIFICATION_STORAGE_OPTIONS,
    );

  const [apiKey, setApiKey] = useLocalStorageState<string>(
    "apiKey",
    "",
    API_KEY_STORAGE_OPTIONS
  );

  const { cases: backendCases, loading, error, refetch } =
    useReportCases(REPORT_CASES_OPTIONS);

  useEffect(() => {
    const interval = setInterval(() => {
      refetch();
    }, 5000);
    return () => clearInterval(interval);
  }, [refetch]);

  useEffect(() => {
    const ac = new AbortController();
    void (async () => {
      try {
        const cfg = await fetchLlmModelSettings({ signal: ac.signal });
        const models = Array.isArray(cfg.available_models)
          ? cfg.available_models
          : [];
        if (models.length > 0) setLlmModels(models);
        if (cfg.active_model) setActiveModel(cfg.active_model);
      } catch (err) {
        console.warn("Failed to sync llm model settings", err);
      }
    })();
    return () => ac.abort();
  }, [setActiveModel]);

  const cases: AnomalyCase[] = backendCases;

  useEffect(() => {
    const currentNgCount = cases.filter((c) => c.decision === "NG").length;

    if (!notificationSettings.highSeverity) {
      setShowToast(false);
      setLastNgCount(currentNgCount);
      return;
    }

    if (lastNgCount > 0 && currentNgCount > lastNgCount) {
      setShowToast(true);
      setIsExiting(false);

      const timer = setTimeout(() => {
        setIsExiting(true);
        setTimeout(() => setShowToast(false), 400);
      }, 5000);

      return () => clearTimeout(timer);
    }

    setLastNgCount(currentNgCount);
  }, [cases, lastNgCount, notificationSettings.highSeverity]);

  const alerts: Alert[] = useMemo(() => {
    const out: Alert[] = [];
    const ngCases = [...cases]
      .filter((c) => c.decision === "NG")
      .sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());

    if (notificationSettings.highSeverity) {
      out.push(
        ...ngCases.map((c) => ({
          id: c.id,
          type: "critical" as const,
          severity: "high" as const,
          title: "결함 감지",
          description: `${c.product_group} 제품에서 결함 발생`,
          timestamp: c.timestamp,
          line_id: c.line_id,
          defect_type: c.defect_type,
        })),
      );
    }

    if (notificationSettings.consecutiveDefects) {
      const counts = new Map<
        string,
        { lineId: string; location: string; count: number; timestamp: Date }
      >();
      for (const c of ngCases.slice(0, 200)) {
        const key = `${c.line_id}|${c.location}`;
        const prev = counts.get(key);
        if (!prev) {
          counts.set(key, {
            lineId: c.line_id,
            location: c.location,
            count: 1,
            timestamp: c.timestamp,
          });
        } else {
          prev.count += 1;
          if (c.timestamp.getTime() > prev.timestamp.getTime())
            prev.timestamp = c.timestamp;
        }
      }

      for (const [key, value] of counts.entries()) {
        if (value.count < 3) continue;
        out.push({
          id: `consecutive-${key}`,
          type: "consecutive",
          severity: "high",
          title: "연속 불량 감지",
          timestamp: value.timestamp,
          line_id: value.lineId,
          location: value.location,
          count: value.count,
        });
      }
    }

    if (notificationSettings.systemError && error) {
      out.push({
        id: "system-error-alert",
        type: "system",
        severity: "high",
        title: "시스템 오류",
        timestamp: new Date(),
        description: "백엔드 통신 오류가 발생했습니다.",
      });
    }

    return out
      .sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime())
      .slice(0, 5);
  }, [
    cases,
    error,
    notificationSettings.consecutiveDefects,
    notificationSettings.highSeverity,
    notificationSettings.systemError,
  ]);

  const [filters, setFilters] = useState<FilterState>({
    dateRange: "today",
    line: "all",
    productGroup: "all",
    defectType: "all",
    decision: "all",
    scoreRange: [0, 1],
  });

  const casesWithSettings = useMemo(() => {
    const version = MODEL_VERSION[activeModel] ?? "v1.0.0";
    return cases.map((c) => ({
      ...c,
      model_name: activeModel,
      model_version: version,
    }));
  }, [cases, activeModel]);

  const filteredCases = useMemo(() => {
    const window = getDateRangeWindow(filters.dateRange);

    return casesWithSettings.filter((c) => {
      if (window) {
        const t = c.timestamp.getTime();
        if (t < window.from.getTime() || t > window.to.getTime()) return false;
      }
      if (filters.line !== "all" && c.line_id !== filters.line) return false;
      if (filters.productGroup !== "all" && c.product_group !== filters.productGroup)
        return false;
      if (filters.defectType !== "all" && c.defect_type !== filters.defectType)
        return false;
      if (filters.decision !== "all" && c.decision !== filters.decision)
        return false;
      if (
        c.anomaly_score < filters.scoreRange[0] ||
        c.anomaly_score > filters.scoreRange[1]
      )
        return false;
      return true;
    });
  }, [casesWithSettings, filters]);

  const handleModelChange = (nextModel: string) => {
    if (!nextModel) return;
    setModelSyncing(true);
    void (async () => {
      try {
        const updated = await updateLlmModelSettings(nextModel);
        setActiveModel(updated.active_model || nextModel);
        if (
          Array.isArray(updated.available_models) &&
          updated.available_models.length > 0
        ) {
          setLlmModels(updated.available_models);
        }
      } catch (err) {
        console.error("Failed to update llm model", err);
      } finally {
        setModelSyncing(false);
      }
    })();
  };

  const currentCase = selectedCaseId
    ? casesWithSettings.find((c) => c.id === selectedCaseId) ?? null
    : null;

  const renderPage = () => {
    if (loading && !error && cases.length === 0) {
      return (
        <LoadingState
          title="데이터 로드 중"
          message="백엔드 서버에서 검사 이력을 가져오고 있습니다."
        />
      );
    }

    if (error && currentPage !== "settings" && cases.length === 0) {
      return (
        <div className="p-6">
          <EmptyState
            type="error"
            title="연동 실패"
            description="백엔드 서버와 통신할 수 없습니다. 서버 실행 상태를 확인하세요."
          />
          <div className="flex justify-center mt-4">
            <button
              onClick={refetch}
              className="px-4 py-2 bg-gray-900 text-white rounded-lg"
            >
              다시 시도
            </button>
          </div>
        </div>
      );
    }

    if (currentPage === "detail" && currentCase) {
      return (
        <CaseDetailPage
          caseData={currentCase}
          onBackToQueue={() => navigate("queue")}
          onBackToOverview={() => navigate("overview")}
        />
      );
    }

    switch (currentPage) {
      case "overview":
        return (
          <OverviewPage
            cases={filteredCases}
            alerts={alerts}
            activeModel={activeModel}
          />
        );
      case "queue":
        return <AnomalyQueuePage cases={filteredCases} onCaseClick={openCase} />;
      case "report":
        return <ReportBuilderPage cases={filteredCases} />;
      case "settings":
        return (
          <SettingsPage
            activeModel={activeModel}
            onModelChange={handleModelChange}
            llmModels={llmModels}
            modelSyncing={modelSyncing}
            notifications={notificationSettings}
            onNotificationsChange={setNotificationSettings}
            apiKey={apiKey}
            onApiKeyChange={setApiKey}
          />
        );
      default:
        return (
          <OverviewPage
            cases={filteredCases}
            alerts={alerts}
            activeModel={activeModel}
          />
        );
    }
  };

  return (
    <div className="flex h-screen bg-gray-50 relative">
      {showToast && (
        <div
          className={`fixed top-6 right-6 z-[9999] ${
            isExiting ? "animate-toast-out" : "animate-toast-in"
          }`}
        >
          <div className="bg-red-600 text-white px-6 py-4 rounded-lg shadow-2xl flex items-center space-x-3 border-2 border-red-400">
            <div className="bg-white p-1 rounded-full">
              <AlertOctagon className="w-6 h-6 text-red-600" />
            </div>
            <div>
              <p className="font-bold">신규 결함 감지!</p>
              <p className="text-sm opacity-90 font-medium">
                최신 검사에서 불량이 발견되었습니다.
              </p>
            </div>
            <button
              onClick={() => {
                setIsExiting(true);
                setTimeout(() => setShowToast(false), 400);
              }}
            >
              ✕
            </button>
          </div>
        </div>
      )}

      <Sidebar currentPage={currentPage} onNavigate={(page) => navigate(page as any)} />

      <div className="flex-1 flex flex-col overflow-hidden">
        {(currentPage === "overview" ||
          currentPage === "queue" ||
          currentPage === "report") && (
          <FilterBar filters={filters} onFilterChange={setFilters} />
        )}
        <div className="flex-1 overflow-y-auto">{renderPage()}</div>
      </div>
    </div>
  );
}