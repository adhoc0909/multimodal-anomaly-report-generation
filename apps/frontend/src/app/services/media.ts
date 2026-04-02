// src/app/services/media.ts
import { API_BASE } from "../api/http";

export type ImageVariant = "original" | "heatmap" | "overlay";

function toOutputStaticUrl(path: string): string | null {
  const parts = path.split("/");
  const filename = parts[parts.length - 1];
  if (!filename) return null;
  return `${API_BASE}/outputs/${encodeURIComponent(filename)}`;
}

function normalizeUrl(path?: string | null): string | null {
  if (!path) return null;
  const p = String(path).trim();
  if (!p) return null;

  // 이미 절대 URL이면 그대로 사용
  if (p.startsWith("http://") || p.startsWith("https://")) return p;

  // FastAPI static mounts
  if (p.startsWith("/outputs/") || p.startsWith("/home/ubuntu/dataset/")) {
    return `${API_BASE}${p}`;
  }

  if (p.startsWith("outputs/") || p.startsWith("home/ubuntu/dataset/")) {
    return `${API_BASE}/${p}`;
  }

  // 서버 파일시스템 절대경로(/home/.../orig_xxx.png 등)는 /outputs/{filename}로 매핑
  if (p.startsWith("/")) return toOutputStaticUrl(p);

  // 그 외(로컬 파일 경로 등)는 프론트에서 안전하게 처리 불가 → null
  return null;
}

export function getCaseImageUrl(c: AnomalyCase, variant: ImageVariant): string | null {
  if (variant === "original") return normalizeUrl(c.image_path);
  if (variant === "heatmap") return normalizeUrl(c.heatmap_path);
  return normalizeUrl(c.overlay_path);
}
