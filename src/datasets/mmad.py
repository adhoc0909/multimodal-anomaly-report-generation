from __future__ import annotations
from pathlib import Path
from typing import List
from ..common.io import read_json
from ..common.types import MMADSample

def load_mmad_samples(mmad_json_path: str | Path) -> List[MMADSample]:
    d = read_json(mmad_json_path)
    return [MMADSample(image_rel=k, meta=v) for k, v in d.items()]

def get_templates(meta: dict, *, k: int, use_similar: bool) -> List[str]:
    if k <= 0:
        return []
    if use_similar and "similar_templates" in meta:
        return list(meta.get("similar_templates", []))[:k]
    return list(meta.get("random_templates", []))[:k]
