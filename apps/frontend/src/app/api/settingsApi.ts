import { apiRequest } from "./http";

export type LlmModelSettingsDTO = {
  active_model: string;
  available_models: string[];
};

const LLM_MODEL_SETTINGS_PATH =
  (import.meta.env.VITE_LLM_MODEL_SETTINGS_PATH as string | undefined) ?? "/settings/llm-model";

export async function fetchLlmModelSettings(opts?: { signal?: AbortSignal }) {
  return apiRequest<LlmModelSettingsDTO>(LLM_MODEL_SETTINGS_PATH, {
    method: "GET",
    signal: opts?.signal,
  });
}

export async function updateLlmModelSettings(model: string, opts?: { signal?: AbortSignal }) {
  return apiRequest<LlmModelSettingsDTO>(LLM_MODEL_SETTINGS_PATH, {
    method: "PUT",
    body: { model },
    signal: opts?.signal,
  });
}
