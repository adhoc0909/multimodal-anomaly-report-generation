from __future__ import annotations
import numpy as np
import cv2

def nine_grid_location(cx: float, cy: float) -> str:
    xs = "left" if cx < 1/3 else ("center" if cx < 2/3 else "right")
    ys = "top" if cy < 1/3 else ("middle" if cy < 2/3 else "bottom")
    return f"{ys}-{xs}"

def structure_from_heatmap(heatmap: np.ndarray, *, thr: float = 0.35) -> dict:
    hm = np.asarray(heatmap)
    if hm.ndim != 2:
        raise ValueError("heatmap must be HxW")
    H, W = hm.shape
    binm = (hm >= thr).astype(np.uint8)
    area_ratio = float(binm.mean())
    has_defect = area_ratio > 0.001

    if not has_defect:
        return {
            "has_defect": False,
            "location": "none",
            "area_ratio": area_ratio,
            "shape": "none",
            "severity": "low",
            "thr": thr,
        }

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(binm, connectivity=8)
    if num <= 1:
        cx, cy = 0.5, 0.5
        shape = "blob"
    else:
        areas = stats[1:, cv2.CC_STAT_AREA]
        idx = int(np.argmax(areas)) + 1
        cx_px, cy_px = centroids[idx]
        cx, cy = float(cx_px / max(W,1)), float(cy_px / max(H,1))
        w = stats[idx, cv2.CC_STAT_WIDTH]
        h = stats[idx, cv2.CC_STAT_HEIGHT]
        shape = "elongated" if max(w,h) / max(1, min(w,h)) >= 2.5 else "blob"

    location = nine_grid_location(cx, cy)
    if area_ratio < 0.01:
        severity = "low"
    elif area_ratio < 0.05:
        severity = "medium"
    else:
        severity = "high"

    return {
        "has_defect": True,
        "location": location,
        "area_ratio": area_ratio,
        "shape": shape,
        "severity": severity,
        "thr": thr,
    }
