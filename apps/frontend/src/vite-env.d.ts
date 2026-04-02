/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE_URL?: string;
  readonly VITE_LLM_MODEL_SETTINGS_PATH?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
