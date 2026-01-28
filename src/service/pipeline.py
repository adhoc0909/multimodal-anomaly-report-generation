from __future__ import annotations
from pathlib import Path
from datetime import datetime
import hashlib

from ..common.io import imread_bgr
from ..structure.defect import structure_from_heatmap
from ..structure.render import save_heatmap_and_overlay
from ..report.schema import load_schema, validate_report

def _stem(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:12]

class InspectionPipeline:
    def __init__(self, *, anomaly_model, mllm_client, runtime_cfg):
        self.anomaly_model = anomaly_model
        self.mllm = mllm_client
        self.cfg = runtime_cfg
        self.schema = load_schema(self.cfg.report["schema_path"])

    def inspect(self, image_abs: str, *, templates_abs: list[str] | None = None) -> dict:
        img = imread_bgr(image_abs)
        templates_bgr = [imread_bgr(p) for p in (templates_abs or [])]
        ar = self.anomaly_model.infer(img, templates_bgr=templates_bgr)

        art_dir = Path(self.cfg.paths.artifact_root) / "artifacts"
        artifacts = save_heatmap_and_overlay(img, ar.heatmap, art_dir, _stem(image_abs))

        structured = structure_from_heatmap(ar.heatmap)
        structured["confidence"] = max(0.5, min(0.99, float(ar.score) * 3.0))

        base = self.mllm.generate_report(structured=structured)
        report = {
            "timestamp": base.get("timestamp") or datetime.utcnow().isoformat(),
            "image": image_abs,
            "decision": base["decision"],
            "confidence": float(base["confidence"]),
            "defect": base["defect"],
            "summary": base["summary"],
            "impact": base["impact"],
            "recommendation": base["recommendation"],
            "artifacts": artifacts,
        }
        validate_report(report, self.schema)
        return report
