from __future__ import annotations
from typing import Protocol
import numpy as np
from ..common.types import AnomalyResult

class AnomalyModel(Protocol):
    def infer(self, image_bgr: np.ndarray, *, templates_bgr: list[np.ndarray] | None = None) -> AnomalyResult:
        ...
