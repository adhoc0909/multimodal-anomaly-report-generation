// src/app/api/http.ts
export const API_BASE =
  (import.meta.env.VITE_API_BASE_URL as string | undefined) ??
  "http://127.0.0.1:8000";


export type QueryParams = Record<
  string,
  string | number | boolean | null | undefined
>;

export class ApiError extends Error {
  status?: number; // 0이면 네트워크/abort/timeout
  url?: string;
  body?: unknown;

  constructor(
    message: string,
    opts?: { status?: number; url?: string; body?: unknown }
  ) {
    super(message);
    this.name = "ApiError";
    this.status = opts?.status;
    this.url = opts?.url;
    this.body = opts?.body;
  }
}

export function isApiError(err: unknown): err is ApiError {
  return (
    err instanceof ApiError ||
    (typeof err === "object" && !!err && (err as any).name === "ApiError")
  );
}

function buildQueryString(query?: QueryParams) {
  if (!query) return "";
  const sp = new URLSearchParams();
  for (const [k, v] of Object.entries(query)) {
    if (v === undefined || v === null || v === "") continue;
    sp.set(k, String(v));
  }
  const s = sp.toString();
  return s ? `?${s}` : "";
}

export function apiUrl(path: string, query?: QueryParams) {
  const p = path.startsWith("/") ? path : `/${path}`;
  return `${API_BASE}${p}${buildQueryString(query)}`;
}

function fastapiDetailMessage(payload: unknown): string | null {
  if (!payload || typeof payload !== "object") return null;
  const detail = (payload as any).detail;
  if (!detail) return null;

  if (typeof detail === "string") return detail;

  if (Array.isArray(detail)) {
    const first = detail[0];
    if (typeof first === "string") return first;
    if (first && typeof first === "object") {
      return first.msg ?? first.message ?? JSON.stringify(first);
    }
    return JSON.stringify(detail);
  }

  if (typeof detail === "object") return JSON.stringify(detail);
  return null;
}

function mergeSignals(a: AbortSignal, b: AbortSignal): AbortSignal {
  const Any = (AbortSignal as any).any as
    | undefined
    | ((signals: AbortSignal[]) => AbortSignal);
  if (Any) return Any([a, b]);

  const controller = new AbortController();
  const onAbort = () => controller.abort();
  if (a.aborted || b.aborted) controller.abort();
  a.addEventListener("abort", onAbort, { once: true });
  b.addEventListener("abort", onAbort, { once: true });
  return controller.signal;
}

export async function apiRequest<T>(
  path: string,
  opts?: {
    method?: "GET" | "POST" | "PUT" | "PATCH" | "DELETE";
    query?: QueryParams;
    body?: unknown;
    headers?: Record<string, string>;
    signal?: AbortSignal;
    timeoutMs?: number;
    credentials?: RequestCredentials;
  }
): Promise<T> {
  const url = apiUrl(path, opts?.query);

  const hasBody = opts?.body !== undefined;
  const headers: Record<string, string> = {
    Accept: "application/json",
    ...(opts?.headers ?? {}),
  };
  if (hasBody) headers["Content-Type"] = "application/json";

  const timeoutMs = opts?.timeoutMs ?? 15000;
  const timeoutController = new AbortController();
  const timer = window.setTimeout(() => timeoutController.abort(), timeoutMs);

  const signal = opts?.signal
    ? mergeSignals(opts.signal, timeoutController.signal)
    : timeoutController.signal;

  try {
    const res = await fetch(url, {
      method: opts?.method ?? "GET",
      headers,
      body: hasBody ? JSON.stringify(opts!.body) : undefined,
      signal,
      credentials: opts?.credentials,
    });

    if (res.status === 204) return undefined as T;

    const contentType = res.headers.get("content-type") ?? "";
    const isJson = contentType.includes("application/json");

    let payload: unknown = null;
    try {
      if (isJson) {
        const text = await res.text();
        payload = text ? JSON.parse(text) : null;
      } else {
        payload = await res.text();
      }
    } catch {
      payload = null;
    }

    if (!res.ok) {
      const detail = fastapiDetailMessage(payload);
      const msg = detail ? `${res.status} ${detail}` : `HTTP ${res.status}`;
      throw new ApiError(msg, { status: res.status, url, body: payload });
    }

    return payload as T;
  } catch (err: any) {
    if (err?.name === "AbortError") {
      throw new ApiError("Request aborted/timeout", { status: 0, url });
    }
    if (isApiError(err)) throw err;
    throw new ApiError(err?.message ?? "Network error", { status: 0, url });
  } finally {
    window.clearTimeout(timer);
  }
}

if (import.meta.env.DEV) {
  console.log("VITE_API_BASE_URL =", import.meta.env.VITE_API_BASE_URL);
  console.log("API_BASE =", API_BASE);
}
