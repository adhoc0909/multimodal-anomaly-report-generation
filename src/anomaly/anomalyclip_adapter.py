"""AnomalyCLIP adapter (Step 2)

This file is intentionally a stub in Step 1.

In Step 2 you will:
- load AnomalyCLIP checkpoints
- implement infer() to return anomaly score + heatmap
"""
from __future__ import annotations
import numpy as np
from ..common.types import AnomalyResult

class AnomalyCLIPAdapter:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Step 2: implement AnomalyCLIPAdapter")

    def infer(self, image_bgr: np.ndarray, *, templates_bgr: list[np.ndarray] | None = None) -> AnomalyResult:
        raise NotImplementedError
