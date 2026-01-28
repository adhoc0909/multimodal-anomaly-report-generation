from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np

def save_heatmap_and_overlay(image_bgr: np.ndarray, heatmap: np.ndarray, out_dir: str | Path, stem: str) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hm_u8 = (np.clip(heatmap,0,1) * 255).astype(np.uint8)
    heatmap_path = out_dir / f"{stem}_heatmap.png"
    cv2.imwrite(str(heatmap_path), hm_u8)

    heat_color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image_bgr, 0.65, heat_color, 0.35, 0)
    overlay_path = out_dir / f"{stem}_overlay.png"
    cv2.imwrite(str(overlay_path), overlay)

    return {"heatmap_path": str(heatmap_path), "overlay_path": str(overlay_path)}
