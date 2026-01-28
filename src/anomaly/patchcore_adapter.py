"""PatchCore adapter (Step 2)

This file is intentionally a stub in Step 1.

In Step 2 you will:
- import PatchCore implementation (e.g., anomalib or official PatchCore repo)
- load pretrained weights / memory bank
- implement infer() to return:
    - anomaly score (float)
    - pixel-level heatmap (HxW float32 in [0,1]) if available
"""
from __future__ import annotations
import numpy as np
from ..common.types import AnomalyResult

class PatchCoreAdapter:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Step 2: implement PatchCoreAdapter")

    def infer(self, image_bgr: np.ndarray, *, templates_bgr: list[np.ndarray] | None = None) -> AnomalyResult:
        raise NotImplementedError
