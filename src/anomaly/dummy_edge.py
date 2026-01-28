from __future__ import annotations
import cv2
import numpy as np
from ..common.types import AnomalyResult

class DummyEdgeAnomaly:
    """CPU-friendly placeholder.

    Heatmap: blurred canny edges.
    Score: mean heatmap.

    Step 2: replace this with PatchCore / AnomalyCLIP adapter.
    """

    def __init__(self, blur_ksize: int = 5, edge_low: int = 50, edge_high: int = 150):
        self.blur_ksize = blur_ksize
        self.edge_low = edge_low
        self.edge_high = edge_high

    def infer(self, image_bgr: np.ndarray, *, templates_bgr: list[np.ndarray] | None = None) -> AnomalyResult:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        if self.blur_ksize and self.blur_ksize > 1:
            k = int(self.blur_ksize) | 1
            gray = cv2.GaussianBlur(gray, (k, k), 0)
        edges = cv2.Canny(gray, self.edge_low, self.edge_high).astype(np.float32) / 255.0
        heatmap = cv2.GaussianBlur(edges, (15, 15), 0)
        heatmap = np.clip(heatmap, 0.0, 1.0).astype(np.float32)
        score = float(heatmap.mean())
        return AnomalyResult(score=score, heatmap=heatmap)
