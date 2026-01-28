from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np

@dataclass
class MMADSample:
    image_rel: str
    meta: Dict[str, Any]

@dataclass
class AnomalyResult:
    score: float
    heatmap: np.ndarray  # HxW float32 in [0,1]
    heatmap_path: Optional[str] = None
    overlay_path: Optional[str] = None
