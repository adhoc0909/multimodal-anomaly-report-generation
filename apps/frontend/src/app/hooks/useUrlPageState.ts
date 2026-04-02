// src/app/hooks/useUrlPageState.ts
import { useCallback, useEffect, useMemo, useState } from "react";

export type Page = "overview" | "queue" | "report" | "settings" | "detail";

function readRoute(): { page: Page; caseId: string | null } {
  const params = new URLSearchParams(window.location.search);
  const page = (params.get("page") ?? "overview") as Page;
  const caseId = params.get("case");
  if (page === "detail" && !caseId) return { page: "overview", caseId: null };
  return { page, caseId };
}

function writeRoute(
  page: Page,
  caseId: string | null,
  mode: "push" | "replace" = "push",
) {
  const url = new URL(window.location.href);
  url.searchParams.set("page", page);
  if (caseId) url.searchParams.set("case", caseId);
  else url.searchParams.delete("case");

  const state = { page, caseId };
  if (mode === "replace") window.history.replaceState(state, "", url);
  else window.history.pushState(state, "", url);
}

export function useUrlPageState() {
  const initial = useMemo(() => readRoute(), []);
  const [currentPage, setCurrentPage] = useState<Page>(initial.page);
  const [selectedCaseId, setSelectedCaseId] = useState<string | null>(initial.caseId);

  useEffect(() => {
    const onPop = () => {
      const { page, caseId } = readRoute();
      setCurrentPage(page);
      setSelectedCaseId(caseId);
    };
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);

  const navigate = useCallback((page: Page) => {
    setCurrentPage(page);
    setSelectedCaseId(null);
    writeRoute(page, null, "push");
  }, []);

  const openCase = useCallback((caseId: string) => {
    setSelectedCaseId(caseId);
    setCurrentPage("detail");
    writeRoute("detail", caseId, "push");
  }, []);

  const setDetail = useCallback((caseId: string | null) => {
    setSelectedCaseId(caseId);
    if (caseId) {
      setCurrentPage("detail");
      writeRoute("detail", caseId, "push");
    } else {
      setCurrentPage("overview");
      writeRoute("overview", null, "push");
    }
  }, []);

  return {
    currentPage,
    selectedCaseId,
    navigate,
    openCase,
    setDetail,
  };
}