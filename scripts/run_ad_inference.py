"""Unified PatchCore inference runner (checkpoint backend).

This script generates AD prediction JSON for downstream LLM evaluation.

Examples:
    # Checkpoint backend
    python scripts/run_ad_inference.py \
        --checkpoint-dir /path/to/checkpoints \
        --data-root /path/to/MMAD \
        --mmad-json /path/to/mmad.json \
        --output output/ad_predictions.json \
        --device cuda
"""
from __future__ import annotations

import argparse
import gc
import inspect
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.nn.functional import interpolate
from tqdm import tqdm

SCRIPT_PATH = Path(__file__).resolve()
PROJ_ROOT = SCRIPT_PATH.parents[1]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from src.utils.loaders import load_config

DEFAULT_AD_POLICY: Dict[str, Any] = {
    "schema_version": "ad_context_rules_v2",
    "default": {
        "review_band": 0.08,
        "location_mode": "normal",
        "reliability": "medium",
        "ad_weight": 0.6,
        "min_location_confidence": 0.25,
        # Legacy compatibility keys
        "t_low": 0.35,
        "t_high": 0.65,
        "use_bbox": True,
    },
    "classes": {
        "GoodsAD/food_box": {
            "review_band": 0.30,
            "reliability": "low",
            "ad_weight": 0.2,
            "location_mode": "off",
            "min_location_confidence": 0.35,
            # Legacy compatibility keys
            "t_low": 0.35,
            "t_high": 0.95,
            "use_bbox": False,
        },
        "GoodsAD/food_package": {
            "review_band": 0.275,
            "reliability": "low",
            "ad_weight": 0.2,
            "location_mode": "off",
            "min_location_confidence": 0.35,
            # Legacy compatibility keys
            "t_low": 0.38,
            "t_high": 0.93,
            "use_bbox": False,
        },
        "MVTec-LOCO/pushpins": {
            "review_band": 0.12,
            "reliability": "low",
            "ad_weight": 0.2,
            "location_mode": "off",
            "min_location_confidence": 0.35,
            # Legacy compatibility keys
            "t_low": 0.43,
            "t_high": 0.67,
            "use_bbox": False,
        },
        "MVTec-LOCO/screw_bag": {
            "review_band": 0.115,
            "reliability": "low",
            "ad_weight": 0.25,
            "location_mode": "strict",
            "min_location_confidence": 0.4,
            # Legacy compatibility keys
            "t_low": 0.49,
            "t_high": 0.72,
            "use_bbox": True,
        },
        "MVTec-LOCO/breakfast_box": {
            "review_band": 0.09,
            "reliability": "medium",
            "ad_weight": 0.45,
            "location_mode": "normal",
            "min_location_confidence": 0.3,
            # Legacy compatibility keys
            "t_low": 0.42,
            "t_high": 0.60,
            "use_bbox": True,
        },
    },
}


def parse_image_path(image_path: str) -> Tuple[str, str]:
    parts = image_path.split("/")
    if len(parts) >= 2:
        return parts[0], parts[1]
    return "", ""


def _is_hub_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    keys = [
        "locate the file on the hub",
        "hf_hub_offline",
        "huggingface.co",
        "local cache",
    ]
    return any(k in msg for k in keys)


def _is_oom_error(exc: Exception) -> bool:
    return "out of memory" in str(exc).lower()


def chunked(items: List[str], chunk_size: int) -> Iterator[List[str]]:
    if chunk_size <= 0:
        chunk_size = 1
    for i in range(0, len(items), chunk_size):
        yield items[i : i + chunk_size]


def load_image_for_inference(image_path: Path, decode_reduced: int) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int]]]:
    original_size: Optional[Tuple[int, int]] = None
    try:
        with Image.open(image_path) as img:
            original_size = (int(img.height), int(img.width))
    except Exception:
        original_size = None

    flags = {
        1: cv2.IMREAD_COLOR,
        2: cv2.IMREAD_REDUCED_COLOR_2,
        4: cv2.IMREAD_REDUCED_COLOR_4,
        8: cv2.IMREAD_REDUCED_COLOR_8,
    }
    image = cv2.imread(str(image_path), flags.get(int(decode_reduced), cv2.IMREAD_COLOR))
    if image is None:
        return None, None

    if original_size is None:
        h, w = image.shape[:2]
        original_size = (int(h), int(w))
    return image, original_size


def _load_image_job(job: Tuple[Path, int]) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int]]]:
    image_path, decode_reduced = job
    return load_image_for_inference(image_path, decode_reduced)


def compute_confidence_level(anomaly_score: float) -> Dict[str, Any]:
    if anomaly_score > 0.8:
        level = "high"
        reason = "Strong anomaly signal (score > 0.8)"
    elif anomaly_score < 0.2:
        level = "high"
        reason = "Strong normal signal (score < 0.2)"
    elif anomaly_score > 0.6 or anomaly_score < 0.4:
        level = "medium"
        reason = "Moderate confidence (score away from threshold neighborhood)"
    else:
        level = "low"
        reason = "Score near threshold neighborhood"

    return {
        "level": level,
        "reliability": "medium",
        "reason": reason,
        "reliability_reason": "Default reliability (no class calibration applied).",
    }


def compute_defect_location(anomaly_map: np.ndarray, threshold: float) -> Dict[str, Any]:
    h, w = anomaly_map.shape
    defect_mask = anomaly_map > float(threshold)

    if not defect_mask.any():
        return {
            "has_defect": False,
            "region": "none",
            "bbox": None,
            "center": None,
            "area_ratio": 0.0,
        }

    coords = np.where(defect_mask)
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()

    center_y = (y_min + y_max) / 2 / h
    center_x = (x_min + x_max) / 2 / w

    region_y = "top" if center_y < 1 / 3 else ("bottom" if center_y > 2 / 3 else "middle")
    region_x = "left" if center_x < 1 / 3 else ("right" if center_x > 2 / 3 else "center")
    region = "center" if (region_y == "middle" and region_x == "center") else f"{region_y}-{region_x}"

    area_ratio = float(defect_mask.sum()) / (h * w)
    confidence = float(anomaly_map[defect_mask].max())

    return {
        "has_defect": True,
        "region": region,
        "bbox": [int(x_min), int(y_min), int(x_max), int(y_max)],
        "center": [round(center_x, 3), round(center_y, 3)],
        "area_ratio": round(area_ratio, 4),
        "confidence": round(confidence, 4),
    }


def scale_defect_location(
    defect_location: Dict[str, Any],
    *,
    from_size: Tuple[int, int],
    to_size: Tuple[int, int],
) -> Dict[str, Any]:
    """Scale bbox coordinates from map size to original image size."""
    if not defect_location.get("has_defect", False):
        return defect_location

    bbox = defect_location.get("bbox")
    if not bbox or len(bbox) != 4:
        return defect_location

    from_h, from_w = int(from_size[0]), int(from_size[1])
    to_h, to_w = int(to_size[0]), int(to_size[1])
    if from_h <= 0 or from_w <= 0:
        return defect_location

    sx = float(to_w) / float(from_w)
    sy = float(to_h) / float(from_h)

    x0, y0, x1, y1 = bbox
    scaled = dict(defect_location)
    scaled["bbox"] = [
        int(round(x0 * sx)),
        int(round(y0 * sy)),
        int(round(x1 * sx)),
        int(round(y1 * sy)),
    ]
    return scaled


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _deep_copy_json_obj(obj: Dict[str, Any]) -> Dict[str, Any]:
    return json.loads(json.dumps(obj))


def load_ad_policy(policy_path: Optional[Path]) -> Dict[str, Any]:
    policy = _deep_copy_json_obj(DEFAULT_AD_POLICY)
    if policy_path is None:
        return policy
    if not policy_path.exists():
        return policy

    try:
        with open(policy_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
    except Exception:
        return policy

    if not isinstance(loaded, dict):
        return policy

    default_loaded = loaded.get("default", {})
    if isinstance(default_loaded, dict):
        policy["default"].update(default_loaded)

    classes_loaded = loaded.get("classes", {})
    if isinstance(classes_loaded, dict):
        for key, value in classes_loaded.items():
            if not isinstance(value, dict):
                continue
            existing = policy["classes"].get(key, {})
            merged = dict(existing)
            merged.update(value)
            policy["classes"][key] = merged

    if isinstance(loaded.get("schema_version"), str):
        policy["schema_version"] = loaded["schema_version"]
    return policy


def resolve_class_policy(policy: Dict[str, Any], dataset: str, category: str) -> Dict[str, Any]:
    class_key = f"{dataset}/{category}"
    merged = dict(policy.get("default", {}))
    class_policy = policy.get("classes", {}).get(class_key, {})
    if isinstance(class_policy, dict):
        merged.update(class_policy)

    reliability = str(merged.get("reliability", merged.get("reliability_prior", "medium"))).lower()
    if reliability not in {"high", "medium", "low"}:
        reliability = "medium"

    # Prefer v2 key first; fallback to legacy use_bbox only when location_mode is absent.
    if "location_mode" in merged:
        location_mode = str(merged.get("location_mode", "normal")).lower()
    elif "use_bbox" in merged:
        location_mode = "normal" if bool(merged.get("use_bbox", True)) else "off"
    else:
        location_mode = "normal"
    if location_mode not in {"normal", "strict", "off"}:
        location_mode = "normal"

    review_band = _safe_float(merged.get("review_band"))
    legacy_center: Optional[float] = _safe_float(merged.get("decision_center"))
    t_low = _safe_float(merged.get("t_low"))
    t_high = _safe_float(merged.get("t_high"))
    if t_low is not None and t_high is not None and t_low > t_high:
        t_low, t_high = t_high, t_low
    if legacy_center is None and t_low is not None and t_high is not None:
        legacy_center = (t_low + t_high) / 2.0
    if review_band is None:
        if t_low is not None and t_high is not None:
            review_band = (t_high - t_low) / 2.0
        else:
            review_band = 0.08
    review_band = max(0.02, min(0.30, float(review_band)))

    ad_weight = _safe_float(merged.get("ad_weight"))
    if ad_weight is None:
        ad_weight = 0.6
    ad_weight = _clip01(ad_weight)

    min_loc_conf = _safe_float(merged.get("min_location_confidence"))
    if min_loc_conf is None:
        min_loc_conf = 0.25
    min_loc_conf = _clip01(min_loc_conf)

    return {
        "class_key": class_key,
        "reliability": reliability,
        "ad_weight": ad_weight,
        "location_mode": location_mode,
        "min_location_confidence": min_loc_conf,
        "review_band": review_band,
        "legacy_center": legacy_center,
    }


def load_ad_calibration(calibration_path: Optional[Path]) -> Dict[str, Any]:
    if calibration_path is None:
        return {}
    if not calibration_path.exists():
        return {}
    try:
        with open(calibration_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
    except Exception:
        return {}
    if not isinstance(loaded, dict):
        return {}
    classes = loaded.get("classes")
    if not isinstance(classes, dict):
        loaded["classes"] = {}
    return loaded


def resolve_class_calibration(calibration: Dict[str, Any], class_key: str) -> Dict[str, Any]:
    if not calibration:
        return {}
    class_map = calibration.get("classes")
    if not isinstance(class_map, dict):
        return {}
    raw = class_map.get(class_key)
    if not isinstance(raw, dict):
        return {}

    out: Dict[str, Any] = {}
    center = (
        _safe_float(raw.get("center_threshold"))
        or _safe_float(raw.get("decision_threshold"))
        or _safe_float(raw.get("threshold_opt"))
    )
    review_band = (
        _safe_float(raw.get("review_band"))
        or _safe_float(raw.get("uncertainty_band"))
        or _safe_float(raw.get("decision_band"))
    )
    if center is not None:
        out["center_threshold"] = center
    if review_band is not None:
        out["review_band"] = max(0.02, min(0.30, float(review_band)))

    for key in ("auroc", "fpr", "fnr", "n_samples"):
        val = _safe_float(raw.get(key))
        if val is not None:
            out[key] = val
    return out


def decide_with_policy(
    anomaly_score: float,
    class_policy: Dict[str, Any],
    *,
    model_threshold: float,
    class_calibration: Optional[Dict[str, Any]] = None,
) -> Tuple[str, bool, Dict[str, Any]]:
    class_calibration = class_calibration or {}
    center = _safe_float(class_calibration.get("center_threshold"))
    if center is None:
        center = _safe_float(class_policy.get("legacy_center"))
    if center is None:
        center = float(model_threshold)

    base_band = float(class_policy.get("review_band", 0.08))
    calib_band = _safe_float(class_calibration.get("review_band"))
    if calib_band is not None:
        base_band = max(base_band, calib_band)

    reliability_scale = {
        "high": 0.9,
        "medium": 1.0,
        "low": 1.3,
    }.get(str(class_policy.get("reliability", "medium")), 1.0)
    uncertainty_band = max(0.02, min(0.35, base_band * reliability_scale))

    score_delta = float(anomaly_score) - float(center)
    if score_delta > uncertainty_band:
        decision = "anomaly"
        review_needed = False
    elif score_delta < -uncertainty_band:
        decision = "normal"
        review_needed = False
    else:
        decision = "review_needed"
        review_needed = True

    decision_confidence = round(
        _clip01(abs(score_delta) / max(1e-6, uncertainty_band * 2.5)),
        4,
    )
    return decision, review_needed, {
        "decision_confidence": decision_confidence,
        "basis": {
            "center_threshold": round(float(center), 4),
            "uncertainty_band": round(float(uncertainty_band), 4),
            "score_delta": round(float(score_delta), 4),
            "used_calibration": bool(class_calibration),
        },
    }


def compute_location_confidence(
    *,
    anomaly_score: float,
    model_threshold: float,
    class_policy: Dict[str, Any],
    decision_basis: Dict[str, Any],
    map_stats: Optional[Dict[str, float]],
    defect_location: Dict[str, Any],
    review_needed: bool,
) -> float:
    if not defect_location.get("has_defect", False):
        return 0.0

    center = _safe_float(decision_basis.get("center_threshold"))
    if center is None:
        center = float(model_threshold)
    band = _safe_float(decision_basis.get("uncertainty_band"))
    if band is None:
        band = float(class_policy.get("review_band", 0.08))
    band = max(1e-6, float(band))
    score_margin = abs(float(anomaly_score) - float(center))
    score_factor = _clip01(score_margin / (2.0 * band))

    area_ratio = float(defect_location.get("area_ratio", 0.0))
    if area_ratio <= 0.2:
        area_factor = 1.0
    elif area_ratio <= 0.5:
        area_factor = 0.7
    else:
        area_factor = 0.4

    peak_factor = 0.5
    if map_stats is not None:
        max_v = float(map_stats.get("max", 0.0))
        mean_v = float(map_stats.get("mean", 0.0))
        std_v = float(map_stats.get("std", 0.0))
        sharpness = (max_v - mean_v) / max(1e-6, std_v)
        peak_factor = _clip01(sharpness / 6.0)

    reliability_factor = {
        "high": 1.0,
        "medium": 0.8,
        "low": 0.6,
    }.get(str(class_policy.get("reliability", "medium")), 0.8)

    location_mode = str(class_policy.get("location_mode", "normal"))
    location_mode_factor = {
        "normal": 1.0,
        "strict": 0.8,
        "off": 0.1,
    }.get(location_mode, 1.0)

    review_factor = 0.65 if review_needed else 1.0

    score = (0.40 * score_factor + 0.40 * peak_factor + 0.20 * area_factor)
    score = score * reliability_factor * location_mode_factor * review_factor
    return round(_clip01(score), 4)


def build_reason_codes(
    *,
    anomaly_score: float,
    class_policy: Dict[str, Any],
    class_calibration: Dict[str, Any],
    decision: str,
    decision_confidence: float,
    decision_basis: Dict[str, Any],
    defect_location: Dict[str, Any],
    location_confidence: float,
) -> List[str]:
    codes: List[str] = []
    reliability = str(class_policy.get("reliability", "medium"))
    location_mode = str(class_policy.get("location_mode", "normal"))
    min_loc_conf = float(class_policy.get("min_location_confidence", 0.25))

    if decision == "review_needed":
        codes.append("NEAR_DECISION_BOUNDARY")
    if decision_confidence < 0.35:
        codes.append("LOW_DECISION_CONFIDENCE")
    if reliability == "low":
        codes.append("LOW_RELIABILITY_CLASS")
    elif reliability == "medium":
        codes.append("MEDIUM_RELIABILITY_CLASS")
    if not class_calibration:
        codes.append("NO_CLASS_CALIBRATION")
    if location_mode == "off":
        codes.append("LOCATION_HINT_DISABLED")
    elif location_mode == "strict":
        codes.append("STRICT_LOCATION_POLICY")
    if not defect_location.get("has_defect", False):
        codes.append("NO_DEFECT_REGION_FROM_MAP")
    if location_confidence < min_loc_conf:
        codes.append("LOW_LOCATION_CONFIDENCE")

    threshold = _safe_float(decision_basis.get("center_threshold"))
    band = _safe_float(decision_basis.get("uncertainty_band"))
    if threshold is not None and band is not None:
        if anomaly_score > threshold + (2.0 * band):
            codes.append("STRONG_ANOMALY_MARGIN")
        elif anomaly_score < threshold - (2.0 * band):
            codes.append("STRONG_NORMAL_MARGIN")

    if defect_location.get("has_defect", False):
        area_ratio = float(defect_location.get("area_ratio", 0.0))
        if area_ratio > 0.5:
            codes.append("DIFFUSE_ANOMALY_REGION")
    return codes


def build_llm_guidance(
    *,
    decision: str,
    class_policy: Dict[str, Any],
    decision_confidence: float,
    reason_codes: List[str],
    location_confidence: float,
) -> Dict[str, Any]:
    reliability = str(class_policy.get("reliability", "medium"))
    ad_weight = float(class_policy.get("ad_weight", 0.7))
    min_loc_conf = float(class_policy.get("min_location_confidence", 0.25))
    location_mode = str(class_policy.get("location_mode", "normal"))

    if decision == "review_needed" or decision_confidence < 0.35 or reliability == "low":
        anomaly_use = "low"
    elif reliability == "medium":
        anomaly_use = "medium"
    else:
        anomaly_use = "high"

    if location_mode == "off":
        location_use = "none"
    elif location_mode in {"normal", "strict"} and location_confidence >= min_loc_conf:
        if reliability == "high":
            location_use = "high"
        elif reliability == "medium":
            location_use = "medium"
        else:
            location_use = "low"
    else:
        location_use = "low"

    if decision == "review_needed":
        instruction = (
            "AD decision is near the uncertainty boundary. Use AD as weak evidence and rely on visual verification."
        )
    elif anomaly_use == "low":
        instruction = (
            "AD reliability is limited for this class. Treat anomaly score as weak evidence."
        )
    elif anomaly_use == "medium":
        instruction = (
            "AD evidence is moderately useful. Cross-check with visual cues before final judgement."
        )
    else:
        instruction = (
            "AD evidence is strong for anomaly existence. Use location hints only if they are consistent with the image."
        )

    return {
        "use_ad_for_anomaly_judgement": anomaly_use,
        "use_ad_for_location": location_use,
        "ad_weight": round(_clip01(ad_weight), 4),
        "location_confidence": location_confidence,
        "decision_confidence": round(_clip01(decision_confidence), 4),
        "instruction": instruction,
        "reason_codes": reason_codes,
    }


class PatchCoreCheckpointRunner:
    def __init__(
        self,
        checkpoint_dir: Path,
        *,
        version: Optional[int],
        threshold: float,
        device: str,
        input_size: Tuple[int, int],
        allow_online_backbone: bool,
        sync_timing: bool,
        postprocess_map: str,
        use_amp: bool,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.version = version
        self.threshold = threshold
        self.device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
        self.input_size = input_size
        self.allow_online_backbone = allow_online_backbone
        self.sync_timing = sync_timing
        self.postprocess_map = postprocess_map
        self.use_amp = bool(use_amp and self.device.type == "cuda")
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        self._models: Dict[str, Any] = {}
        self._warmup_done: set[str] = set()

    def _find_checkpoint(self, dataset: str, category: str) -> Optional[Path]:
        patchcore_dir = self.checkpoint_dir / "Patchcore"
        if not patchcore_dir.exists():
            patchcore_dir = self.checkpoint_dir

        category_dir = patchcore_dir / dataset / category
        if not category_dir.exists():
            return None

        if self.version is not None:
            ckpt = category_dir / f"v{self.version}" / "model.ckpt"
            return ckpt if ckpt.exists() else None

        versions = []
        for v_dir in category_dir.iterdir():
            if v_dir.is_dir() and v_dir.name.startswith("v"):
                try:
                    versions.append((int(v_dir.name[1:]), v_dir))
                except ValueError:
                    continue
        if not versions:
            return None

        latest = max(versions, key=lambda x: x[0])[1]
        ckpt = latest / "model.ckpt"
        return ckpt if ckpt.exists() else None

    def list_available_models(self) -> List[Tuple[str, str]]:
        available = []
        patchcore_dir = self.checkpoint_dir / "Patchcore"
        if not patchcore_dir.exists():
            patchcore_dir = self.checkpoint_dir

        if not patchcore_dir.exists():
            return available

        for dataset_dir in patchcore_dir.iterdir():
            if not dataset_dir.is_dir() or dataset_dir.name.startswith("."):
                continue
            if dataset_dir.name in ["eval", "predictions"]:
                continue

            for category_dir in dataset_dir.iterdir():
                if not category_dir.is_dir() or category_dir.name.startswith("."):
                    continue

                if self._find_checkpoint(dataset_dir.name, category_dir.name) is not None:
                    available.append((dataset_dir.name, category_dir.name))

        return sorted(available)

    @staticmethod
    def _load_from_checkpoint(ckpt_path: Path, *, pre_trained_override: Optional[bool]):
        from anomalib.models import Patchcore

        load_kwargs: Dict[str, Any] = {
            "map_location": "cpu",
            "weights_only": False,
        }
        if pre_trained_override is not None:
            load_kwargs["pre_trained"] = pre_trained_override

        try:
            return Patchcore.load_from_checkpoint(str(ckpt_path), **load_kwargs)
        except TypeError as exc:
            if "pre_trained" in str(exc):
                load_kwargs.pop("pre_trained", None)
                return Patchcore.load_from_checkpoint(str(ckpt_path), **load_kwargs)
            raise

    @staticmethod
    def _load_manual_no_hub(ckpt_path: Path):
        from anomalib.models import Patchcore

        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        hyper = ckpt.get("hyper_parameters", {})

        sig = inspect.signature(Patchcore.__init__)
        allowed = set(sig.parameters.keys()) - {"self"}
        init_kwargs = {k: v for k, v in hyper.items() if k in allowed}
        init_kwargs["pre_trained"] = False

        model = Patchcore(**init_kwargs)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Warning: missing keys during manual checkpoint load ({len(missing)})")
        if unexpected:
            print(f"  Warning: unexpected keys during manual checkpoint load ({len(unexpected)})")
        return model

    def get_model(self, dataset: str, category: str):
        key = f"{dataset}/{category}"
        if key in self._models:
            return self._models[key]

        ckpt_path = self._find_checkpoint(dataset, category)
        if ckpt_path is None:
            raise FileNotFoundError(f"Checkpoint not found for {key}")

        try:
            model = self._load_from_checkpoint(ckpt_path, pre_trained_override=False)
        except Exception as exc:
            if not _is_hub_error(exc):
                raise

            print("  Warning: Hub lookup error during checkpoint load; retrying with no-download loader")
            try:
                model = self._load_manual_no_hub(ckpt_path)
            except Exception as manual_exc:
                if self.allow_online_backbone:
                    print("  Warning: no-download loader failed; retrying with online backbone resolution")
                    model = self._load_from_checkpoint(ckpt_path, pre_trained_override=None)
                else:
                    raise RuntimeError(
                        "Checkpoint load failed due to Hub resolution. "
                        "Retry with --allow-online-backbone to permit online download."
                    ) from manual_exc

        model.eval()
        model.to(self.device)
        self._models[key] = model
        return model

    def _preprocess_tensor(self, image: np.ndarray) -> torch.Tensor:
        h, w = self.input_size
        img = cv2.resize(image, (w, h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        return torch.from_numpy(img)

    def _preprocess_batch(self, images: List[np.ndarray]) -> torch.Tensor:
        batch = torch.stack([self._preprocess_tensor(img) for img in images], dim=0).float()
        return batch.to(self.device, non_blocking=True)

    def warmup_model(self, dataset: str, category: str) -> None:
        key = f"{dataset}/{category}"
        if key in self._warmup_done:
            return

        model = self.get_model(dataset, category)
        dummy = torch.randn(1, 3, self.input_size[0], self.input_size[1], device=self.device)
        with torch.inference_mode():
            _ = model(dummy)
        if self.sync_timing and self.device.type == "cuda":
            torch.cuda.synchronize()
        self._warmup_done.add(key)

    def predict(
        self,
        dataset: str,
        category: str,
        image: np.ndarray,
        original_size: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Any]:
        original_sizes = [original_size] if original_size is not None else None
        return self.predict_batch(dataset, category, [image], original_sizes=original_sizes)[0]

    def predict_batch(
        self,
        dataset: str,
        category: str,
        images: List[np.ndarray],
        *,
        original_sizes: Optional[List[Tuple[int, int]]] = None,
    ) -> List[Dict[str, Any]]:
        if not images:
            return []

        model = self.get_model(dataset, category)
        input_tensor = self._preprocess_batch(images)
        if original_sizes is None or len(original_sizes) != len(images):
            original_sizes = [img.shape[:2] for img in images]

        with torch.inference_mode():
            if self.use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(input_tensor)
            else:
                outputs = model(input_tensor)
            if self.sync_timing and self.device.type == "cuda":
                torch.cuda.synchronize()

        anomaly_map = getattr(outputs, "anomaly_map", None)
        pred_score = getattr(outputs, "pred_score", None)
        pred_label = getattr(outputs, "pred_label", None)

        if anomaly_map is not None:
            if self.postprocess_map == "input":
                target_size = (self.input_size[0], self.input_size[1])
                if anomaly_map.shape[-2:] != target_size:
                    anomaly_map = interpolate(anomaly_map, size=target_size, mode="bilinear", align_corners=False)

        threshold_used = self.threshold
        post = getattr(model, "post_processor", None)
        if post is not None and hasattr(post, "normalized_image_threshold"):
            thr = post.normalized_image_threshold
            if isinstance(thr, torch.Tensor):
                threshold_used = float(thr.detach().cpu())
            else:
                threshold_used = float(thr)

        scores = pred_score.detach().flatten().cpu().tolist() if pred_score is not None else None
        labels = pred_label.detach().flatten().cpu().tolist() if pred_label is not None else None

        results: List[Dict[str, Any]] = []
        for idx, original_size in enumerate(original_sizes):
            anomaly_map_np: Optional[np.ndarray] = None
            if anomaly_map is not None:
                map_i = anomaly_map[idx : idx + 1]
                if self.postprocess_map == "original":
                    if map_i.shape[-2:] != (original_size[0], original_size[1]):
                        map_i = interpolate(map_i, size=original_size, mode="bilinear", align_corners=False)
                # AMP(fp16) path can overflow in numpy reductions; keep map in fp32 for stable stats/location.
                anomaly_map_np = map_i[0, 0].detach().float().cpu().numpy()

            if scores is not None:
                anomaly_score = float(scores[idx])
            elif anomaly_map_np is not None:
                anomaly_score = float(np.max(anomaly_map_np))
            else:
                anomaly_score = 0.0

            if labels is not None:
                is_anomaly = bool(labels[idx])
            else:
                is_anomaly = anomaly_score > self.threshold

            results.append(
                {
                    "anomaly_score": anomaly_score,
                    "anomaly_map": anomaly_map_np,
                    "is_anomaly": is_anomaly,
                    "threshold": threshold_used,
                    "backend": "ckpt",
                    "original_size": [int(original_size[0]), int(original_size[1])],
                    "map_size": [int(anomaly_map_np.shape[0]), int(anomaly_map_np.shape[1])]
                    if anomaly_map_np is not None
                    else None,
                }
            )

        del input_tensor, outputs
        if pred_score is not None:
            del pred_score
        if pred_label is not None:
            del pred_label

        return results

    def clear_cache(self) -> None:
        self._models.clear()
        self._warmup_done.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def load_images(mmad_json: Path, *, datasets: Optional[List[str]], categories: Optional[List[str]]) -> List[str]:
    with open(mmad_json, "r", encoding="utf-8") as f:
        mmad_data = json.load(f)

    image_paths = list(mmad_data.keys())

    if datasets or categories:
        filtered = []
        datasets_set = set(datasets or [])
        categories_set = set(categories or [])
        for path in image_paths:
            dataset, category = parse_image_path(path)
            if datasets_set and dataset not in datasets_set:
                continue
            if categories_set and category not in categories_set:
                continue
            filtered.append(path)
        image_paths = filtered

    return image_paths


def extract_prediction_list(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        predictions = payload.get("predictions")
        if isinstance(predictions, list):
            return predictions
    return []


def build_output_payload(
    *,
    results: List[Dict[str, Any]],
    output_format: str,
    policy: Dict[str, Any],
    calibration: Dict[str, Any],
    context_mode: str,
    backend: str,
    model_threshold: float,
) -> Any:
    if output_format == "list":
        return results

    return {
        "schema_version": "ad_report_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "backend": backend,
        "context_mode": context_mode,
        "model_threshold": float(model_threshold),
        "decision_rules_version": str(policy.get("schema_version", "ad_policy_v1")),
        "context_calibration_version": str(calibration.get("schema_version", "none")) if calibration else "none",
        "bbox_spec": {
            "format": "xyxy",
            "order": ["x_min", "y_min", "x_max", "y_max"],
            "reference": "original_image_pixels",
        },
        "predictions": results,
    }


def save_results(
    path: Path,
    results: List[Dict[str, Any]],
    *,
    output_format: str,
    policy: Dict[str, Any],
    calibration: Dict[str, Any],
    context_mode: str,
    backend: str,
    model_threshold: float,
) -> None:
    payload = build_output_payload(
        results=results,
        output_format=output_format,
        policy=policy,
        calibration=calibration,
        context_mode=context_mode,
        backend=backend,
        model_threshold=model_threshold,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def build_result_dict(
    image_path: str,
    dataset: str,
    category: str,
    pred: Dict[str, Any],
    *,
    policy: Dict[str, Any],
    calibration: Dict[str, Any],
    context_mode: str,
    include_map_stats: bool,
) -> Dict[str, Any]:
    if context_mode == "legacy":
        return build_result_dict_legacy(
            image_path=image_path,
            dataset=dataset,
            category=category,
            pred=pred,
            include_map_stats=include_map_stats,
        )

    anomaly_score = round(float(pred["anomaly_score"]), 4)
    threshold = float(pred.get("threshold", 0.5))
    anomaly_map = pred.get("anomaly_map")
    model_is_anomaly = bool(pred["is_anomaly"])
    original_size_list = pred.get("original_size")
    map_size_list = pred.get("map_size")
    original_size = tuple(original_size_list) if isinstance(original_size_list, list) and len(original_size_list) == 2 else None
    map_size = tuple(map_size_list) if isinstance(map_size_list, list) and len(map_size_list) == 2 else None

    class_policy = resolve_class_policy(policy, dataset, category)
    class_calibration = resolve_class_calibration(calibration, class_policy["class_key"])
    decision, review_needed, decision_meta = decide_with_policy(
        anomaly_score,
        class_policy,
        model_threshold=threshold,
        class_calibration=class_calibration,
    )
    confidence = compute_confidence_level(anomaly_score)
    confidence["reliability"] = class_policy["reliability"]
    if class_policy["reliability"] == "low":
        confidence["reliability_reason"] = "Policy marks this class as low reliability for AD-only judgement."
    elif class_policy["reliability"] == "medium":
        confidence["reliability_reason"] = "Policy marks this class as medium reliability; cross-check with visual cues."
    else:
        confidence["reliability_reason"] = "Policy marks this class as high reliability."
    if class_calibration:
        auroc = _safe_float(class_calibration.get("auroc"))
        if auroc is not None:
            confidence["reliability_reason"] += f" Calibrated AUROC={auroc:.3f}."
    else:
        confidence["reliability_reason"] += " No class-level calibration found."

    default_location = {
        "has_defect": False,
        "region": "none",
        "bbox": None,
        "center": None,
        "area_ratio": 0.0,
    }
    map_stats: Optional[Dict[str, float]] = None
    if anomaly_map is not None:
        map_stats = {
            "max": round(float(np.max(anomaly_map)), 4),
            "mean": round(float(np.mean(anomaly_map)), 4),
            "std": round(float(np.std(anomaly_map)), 4),
        }
        defect_location_raw = compute_defect_location(anomaly_map, threshold)
    else:
        defect_location_raw = default_location

    if (
        defect_location_raw.get("has_defect", False)
        and original_size is not None
        and map_size is not None
        and tuple(original_size) != tuple(map_size)
    ):
        defect_location_raw = scale_defect_location(
            defect_location_raw,
            from_size=(int(map_size[0]), int(map_size[1])),
            to_size=(int(original_size[0]), int(original_size[1])),
        )

    location_confidence = compute_location_confidence(
        anomaly_score=anomaly_score,
        model_threshold=threshold,
        class_policy=class_policy,
        decision_basis=decision_meta["basis"],
        map_stats=map_stats,
        defect_location=defect_location_raw,
        review_needed=review_needed,
    )
    reason_codes = build_reason_codes(
        anomaly_score=anomaly_score,
        class_policy=class_policy,
        class_calibration=class_calibration,
        decision=decision,
        decision_confidence=decision_meta["decision_confidence"],
        decision_basis=decision_meta["basis"],
        defect_location=defect_location_raw,
        location_confidence=location_confidence,
    )
    report_guidance = build_llm_guidance(
        decision=decision,
        class_policy=class_policy,
        decision_confidence=decision_meta["decision_confidence"],
        reason_codes=reason_codes,
        location_confidence=location_confidence,
    )

    min_location_conf = float(class_policy["min_location_confidence"])
    if class_policy["location_mode"] == "strict":
        min_location_conf = max(min_location_conf, 0.5)

    use_location_for_report = (
        class_policy["location_mode"] != "off"
        and bool(defect_location_raw.get("has_defect", False))
        and decision != "normal"
        and location_confidence >= min_location_conf
    )

    report_location = defect_location_raw if use_location_for_report else default_location
    if use_location_for_report:
        report_location = dict(defect_location_raw)
        report_location["location_confidence"] = location_confidence
    else:
        report_location = dict(default_location)
        report_location["location_confidence"] = location_confidence

    out: Dict[str, Any] = {
        "context_version": "ad_context_v2",
        "image_path": image_path,
        "dataset": dataset,
        "category": category,
        "backend": pred.get("backend"),
        "anomaly_score": anomaly_score,
        "model_is_anomaly": model_is_anomaly,
        "model_threshold": threshold,
        "decision": decision,
        "review_needed": review_needed,
        "decision_confidence": decision_meta["decision_confidence"],
        "decision_basis": decision_meta["basis"],
        "class_key": class_policy["class_key"],
        "confidence": confidence,
        "defect_location_raw": defect_location_raw,
        "defect_location": report_location,
        "use_location_for_report": use_location_for_report,
        "report_guidance": report_guidance,
        "reason_codes": reason_codes,
    }

    if include_map_stats and map_stats is not None:
        out["map_stats"] = map_stats
    return out


def build_result_dict_legacy(
    *,
    image_path: str,
    dataset: str,
    category: str,
    pred: Dict[str, Any],
    include_map_stats: bool,
) -> Dict[str, Any]:
    anomaly_score = round(float(pred["anomaly_score"]), 4)
    threshold = float(pred.get("threshold", 0.5))
    anomaly_map = pred.get("anomaly_map")
    model_is_anomaly = bool(pred["is_anomaly"])

    default_location = {
        "has_defect": False,
        "region": "none",
        "bbox": None,
        "center": None,
        "area_ratio": 0.0,
    }
    map_stats: Optional[Dict[str, float]] = None
    if anomaly_map is not None:
        map_stats = {
            "max": round(float(np.max(anomaly_map)), 4),
            "mean": round(float(np.mean(anomaly_map)), 4),
            "std": round(float(np.std(anomaly_map)), 4),
        }
        defect_location_raw = compute_defect_location(anomaly_map, threshold)
    else:
        defect_location_raw = default_location

    decision = "anomaly" if model_is_anomaly else "normal"
    review_needed = False
    decision_confidence = round(_clip01(abs(anomaly_score - threshold) / 0.25), 4)
    use_location_for_report = bool(model_is_anomaly and defect_location_raw.get("has_defect", False))
    defect_location = defect_location_raw if use_location_for_report else dict(default_location)
    if use_location_for_report:
        defect_location = dict(defect_location_raw)
        defect_location["location_confidence"] = round(_clip01(float(defect_location_raw.get("confidence", 0.0))), 4)
    else:
        defect_location["location_confidence"] = 0.0

    confidence = compute_confidence_level(anomaly_score)
    confidence["reliability"] = "medium"
    confidence["reliability_reason"] = "Legacy AD context mode (no class-wise trust calibration)."

    report_guidance = {
        "use_ad_for_anomaly_judgement": "medium",
        "use_ad_for_location": "medium" if use_location_for_report else "low",
        "ad_weight": 0.6,
        "location_confidence": defect_location.get("location_confidence", 0.0),
        "decision_confidence": decision_confidence,
        "instruction": "Legacy AD context: use AD as supporting evidence and validate with image.",
        "reason_codes": [],
    }

    out: Dict[str, Any] = {
        "context_version": "ad_context_legacy_v1",
        "image_path": image_path,
        "dataset": dataset,
        "category": category,
        "backend": pred.get("backend"),
        "anomaly_score": anomaly_score,
        "model_is_anomaly": model_is_anomaly,
        "model_threshold": threshold,
        "decision": decision,
        "review_needed": review_needed,
        "decision_confidence": decision_confidence,
        "decision_basis": {
            "center_threshold": round(threshold, 4),
            "uncertainty_band": 0.0,
            "score_delta": round(anomaly_score - threshold, 4),
            "used_calibration": False,
        },
        "class_key": f"{dataset}/{category}",
        "confidence": confidence,
        "defect_location_raw": defect_location_raw,
        "defect_location": defect_location,
        "use_location_for_report": use_location_for_report,
        "report_guidance": report_guidance,
        "reason_codes": [],
    }

    if include_map_stats and map_stats is not None:
        out["map_stats"] = map_stats
    return out


def resolve_config(args: argparse.Namespace) -> Tuple[Optional[int], Optional[List[str]], Optional[List[str]], Tuple[int, int]]:
    config_version = args.version
    config_datasets = args.datasets
    config_categories = args.categories
    input_size = tuple(args.input_size) if args.input_size is not None else None

    if args.config and Path(args.config).exists():
        config = load_config(args.config)
        if config_version is None:
            config_version = config.get("predict", {}).get("version")
        if config_datasets is None:
            config_datasets = config.get("data", {}).get("datasets")
        if config_categories is None:
            config_categories = config.get("data", {}).get("categories")
        if input_size is None:
            input_size = tuple(config.get("data", {}).get("image_size", [700, 700]))

    if input_size is None:
        input_size = (700, 700)

    return config_version, config_datasets, config_categories, input_size


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified PatchCore AD inference")

    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Checkpoint root")

    # Common runtime
    parser.add_argument("--config", type=str, default="configs/anomaly.yaml", help="Config file for filters/version")
    parser.add_argument("--version", type=int, default=None, help="Checkpoint version override")
    parser.add_argument("--datasets", type=str, nargs="*", default=None, help="Dataset filter override")
    parser.add_argument("--categories", type=str, nargs="*", default=None, help="Category filter override")
    parser.add_argument("--threshold", type=float, default=0.5, help="Default anomaly threshold")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help="Inference device")
    parser.add_argument("--input-size", type=int, nargs=2, default=None, help="Input size override [H W] for ckpt")

    # Data IO
    parser.add_argument("--data-root", type=str, required=True, help="Root directory containing images")
    parser.add_argument("--mmad-json", type=str, required=True, help="Path to mmad.json")
    parser.add_argument("--max-images", type=int, default=None, help="Maximum images to process")
    parser.add_argument("--batch-size", type=int, default=8, help="Inference batch size per category")
    parser.add_argument("--io-workers", type=int, default=8, help="Parallel workers for image loading")
    parser.add_argument("--decode-reduced", type=int, default=1, choices=[1, 2, 4, 8], help="OpenCV reduced decode factor for faster loading (1=full)")
    parser.add_argument("--output", type=str, default="output/ad_predictions.json", help="Output JSON path")
    parser.add_argument(
        "--output-format",
        type=str,
        default="report",
        choices=["report", "mllm", "list"],
        help="Output JSON format: report(recommended), mllm(alias), or list(legacy)",
    )
    parser.add_argument(
        "--policy-json",
        type=str,
        default="configs/ad_policy.json",
        help="Decision-rules JSON path (internal only; not exported in output JSON)",
    )
    parser.add_argument(
        "--calibration-json",
        type=str,
        default="configs/ad_calibration.json",
        help="Optional calibration JSON with class-wise threshold bands for AD context.",
    )
    parser.add_argument(
        "--context-mode",
        type=str,
        default="v2",
        choices=["v2", "legacy"],
        help="AD context generation mode. v2 uses reliability-aware guidance; legacy keeps old-style context.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from existing output")
    parser.add_argument("--save-interval", type=int, default=0, help="Periodic save interval (0=save only at end)")
    parser.add_argument("--no-map-stats", action="store_true", help="Disable anomaly map summary stats")
    parser.add_argument("--profile-interval", type=int, default=128, help="Print stage timing every N processed images (0=disabled)")

    # Stability/perf knobs
    parser.add_argument("--allow-online-backbone", action="store_true", help="Allow online backbone resolution when checkpoint loading fails")
    parser.add_argument("--sync-timing", action="store_true", help="Synchronize CUDA per image (accurate timing, slower)")
    parser.add_argument("--gc-interval", type=int, default=0, help="Run gc.collect every N images (0=disabled)")
    parser.add_argument("--keep-model-cache", action="store_true", dest="keep_model_cache", help="Keep loaded model cache across categories")
    parser.add_argument("--no-keep-model-cache", action="store_false", dest="keep_model_cache", help="Clear model cache after each category")
    parser.add_argument(
        "--postprocess-map",
        type=str,
        default="input",
        choices=["original", "input"],
        help="Anomaly-map postprocess scale: original(slower) or input(faster, bbox scaled to original)",
    )
    parser.add_argument("--amp", action="store_true", dest="amp", help="Enable CUDA mixed precision")
    parser.add_argument("--no-amp", action="store_false", dest="amp", help="Disable CUDA mixed precision")
    parser.set_defaults(amp=True, keep_model_cache=False)

    args = parser.parse_args()

    data_root = Path(args.data_root)
    mmad_json = Path(args.mmad_json)
    output_path = Path(args.output)
    include_map_stats = not args.no_map_stats
    policy_path = Path(args.policy_json) if args.policy_json else None
    policy = load_ad_policy(policy_path)
    calibration_path = Path(args.calibration_json) if args.calibration_json else None
    calibration = load_ad_calibration(calibration_path)

    if not data_root.exists():
        print(f"Error: Data root not found: {data_root}")
        sys.exit(1)
    if not mmad_json.exists():
        print(f"Error: MMAD JSON not found: {mmad_json}")
        sys.exit(1)

    config_version, config_datasets, config_categories, input_size = resolve_config(args)

    print("Backend: ckpt")
    print(f"Config: {args.config}")
    print(f"Output format: {args.output_format}")
    print(f"Context mode: {args.context_mode}")
    print(f"Decision rules: {policy_path if policy_path is not None else 'default (built-in)'}")
    print(f"Calibration: {calibration_path if calibration_path is not None else 'none'}")
    print(f"Batch size: {args.batch_size}")
    print(f"IO workers: {args.io_workers}")
    print(f"Decode reduced: x{args.decode_reduced}")
    print(f"Save interval: {args.save_interval}")
    print(f"Profile interval: {args.profile_interval}")
    print(f"Postprocess map: {args.postprocess_map}")
    print(f"AMP: {args.amp}")
    print(f"Keep model cache: {args.keep_model_cache}")
    print(f"Version: v{config_version}" if config_version is not None else "Version: latest")
    print(f"Filters: datasets={config_datasets} categories={config_categories}")
    print(f"Input size: {input_size}")
    if not args.checkpoint_dir:
        print("Error: --checkpoint-dir is required")
        sys.exit(1)
    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        print(f"Error: Checkpoint directory not found: {checkpoint_dir}")
        sys.exit(1)
    runner = PatchCoreCheckpointRunner(
        checkpoint_dir=checkpoint_dir,
        version=config_version,
        threshold=args.threshold,
        device=args.device,
        input_size=input_size,
        allow_online_backbone=args.allow_online_backbone,
        sync_timing=args.sync_timing,
        postprocess_map=args.postprocess_map,
        use_amp=args.amp,
    )

    runtime_device = str(runner.device) if hasattr(runner, "device") else args.device
    print(f"Device: {runtime_device}")
    if args.device == "cuda" and runtime_device == "cpu":
        print("Warning: CUDA requested but not available; running on CPU (much slower).")
    print()

    print(f"Loading MMAD data from: {mmad_json}")
    image_paths = load_images(mmad_json, datasets=config_datasets, categories=config_categories)
    if args.max_images:
        image_paths = image_paths[: args.max_images]
    print(f"Total images to process: {len(image_paths)}")

    existing_results: Dict[str, Dict[str, Any]] = {}
    if args.resume and output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            existing_payload = json.load(f)
        existing_list = extract_prediction_list(existing_payload)
        existing_results = {r["image_path"]: r for r in existing_list}
        print(f"Loaded {len(existing_results)} existing results")

    available_models = set(runner.list_available_models())
    print(f"Available models: {len(available_models)}")
    for dataset, category in sorted(available_models):
        print(f"  - {dataset}/{category}")

    images_by_category: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    skipped_no_model = 0
    for image_path in image_paths:
        if image_path in existing_results:
            continue
        dataset, category = parse_image_path(image_path)
        if (dataset, category) not in available_models:
            skipped_no_model += 1
            continue
        images_by_category[(dataset, category)].append(image_path)

    for key in images_by_category:
        images_by_category[key].sort()

    total_to_process = sum(len(v) for v in images_by_category.values())
    print()
    print("=" * 60)
    print("Running inference")
    print("=" * 60)
    print(f"Images to process: {total_to_process} across {len(images_by_category)} categories")

    results: List[Dict[str, Any]] = list(existing_results.values())
    processed = len(existing_results)
    errors = 0
    total_inference_time = 0.0
    window_read_time = 0.0
    window_infer_time = 0.0
    window_build_time = 0.0
    window_save_time = 0.0
    window_images = 0
    io_workers = max(1, int(args.io_workers))
    io_pool = ThreadPoolExecutor(max_workers=io_workers) if io_workers > 1 else None

    try:
        for (dataset, category), cat_images in images_by_category.items():
            cat_key = f"{dataset}/{category}"
            print(f"\n[{cat_key}] Loading model and processing {len(cat_images)} images...")

            try:
                runner.warmup_model(dataset, category)
            except Exception as exc:
                print(f"  Failed to load model: {exc}")
                errors += len(cat_images)
                continue

            cat_start = time.perf_counter()
            pbar = tqdm(total=len(cat_images), desc=f"  {cat_key}", ncols=100, leave=False)

            for batch_paths in chunked(cat_images, max(1, int(args.batch_size))):
                read_t0 = time.perf_counter()
                batch_images: List[np.ndarray] = []
                batch_original_sizes: List[Tuple[int, int]] = []
                valid_paths: List[str] = []
                read_inputs: List[Tuple[str, Path]] = []
                for image_path in batch_paths:
                    image_full_path = data_root / image_path
                    if not image_full_path.exists():
                        errors += 1
                        continue
                    read_inputs.append((image_path, image_full_path))

                read_jobs = [(path, int(args.decode_reduced)) for _, path in read_inputs]
                if io_pool is not None:
                    decoded = list(io_pool.map(_load_image_job, read_jobs))
                else:
                    decoded = [_load_image_job(job) for job in read_jobs]

                for (image_path, _), (image, original_size) in zip(read_inputs, decoded):
                    if image is None or original_size is None:
                        errors += 1
                        continue
                    batch_images.append(image)
                    batch_original_sizes.append(original_size)
                    valid_paths.append(image_path)
                window_read_time += time.perf_counter() - read_t0

                if not batch_images:
                    pbar.update(len(batch_paths))
                    pbar.set_postfix({"done": processed, "err": errors})
                    continue

                try:
                    t0 = time.perf_counter()
                    preds = runner.predict_batch(
                        dataset,
                        category,
                        batch_images,
                        original_sizes=batch_original_sizes,
                    )
                    infer_time = time.perf_counter() - t0
                    total_inference_time += infer_time
                    window_infer_time += infer_time
                except RuntimeError as exc:
                    if _is_oom_error(exc) and len(batch_images) > 1:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        preds: List[Optional[Dict[str, Any]]] = []
                        for single_img, single_original_size in zip(batch_images, batch_original_sizes):
                            try:
                                t0 = time.perf_counter()
                                pred = runner.predict(
                                    dataset,
                                    category,
                                    single_img,
                                    original_size=single_original_size,
                                )
                                infer_time = time.perf_counter() - t0
                                total_inference_time += infer_time
                                window_infer_time += infer_time
                                preds.append(pred)
                            except Exception:
                                preds.append(None)
                    else:
                        errors += len(valid_paths)
                        pbar.update(len(batch_paths))
                        pbar.set_postfix({"done": processed, "err": errors})
                        continue
                except Exception:
                    errors += len(valid_paths)
                    pbar.update(len(batch_paths))
                    pbar.set_postfix({"done": processed, "err": errors})
                    continue

                for image_path, pred in zip(valid_paths, preds):
                    if pred is None:
                        errors += 1
                        continue
                    try:
                        build_t0 = time.perf_counter()
                        result_dict = build_result_dict(
                            image_path,
                            dataset,
                            category,
                            pred,
                            policy=policy,
                            calibration=calibration,
                            context_mode=args.context_mode,
                            include_map_stats=include_map_stats,
                        )
                        window_build_time += time.perf_counter() - build_t0
                        results.append(result_dict)
                        processed += 1
                        window_images += 1

                        if args.gc_interval > 0 and processed % args.gc_interval == 0:
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                        if args.save_interval > 0 and processed % args.save_interval == 0:
                            save_t0 = time.perf_counter()
                            save_results(
                                output_path,
                                results,
                                output_format=args.output_format,
                                policy=policy,
                                calibration=calibration,
                                context_mode=args.context_mode,
                                backend="ckpt",
                                model_threshold=args.threshold,
                            )
                            window_save_time += time.perf_counter() - save_t0
                    except Exception:
                        errors += 1

                if args.profile_interval > 0 and window_images >= args.profile_interval:
                    read_ms = (window_read_time / max(1, window_images)) * 1000.0
                    infer_ms = (window_infer_time / max(1, window_images)) * 1000.0
                    build_ms = (window_build_time / max(1, window_images)) * 1000.0
                    save_ms = (window_save_time / max(1, window_images)) * 1000.0
                    if torch.cuda.is_available():
                        mem_alloc = torch.cuda.memory_allocated() / (1024 ** 3)
                        mem_rsrv = torch.cuda.memory_reserved() / (1024 ** 3)
                        print(
                            f"  [profile] {window_images} imgs | read {read_ms:.1f} ms/img | "
                            f"infer {infer_ms:.1f} ms/img | build {build_ms:.1f} ms/img | "
                            f"save {save_ms:.1f} ms/img | gpu alloc/reserved {mem_alloc:.2f}/{mem_rsrv:.2f} GB"
                        )
                    else:
                        print(
                            f"  [profile] {window_images} imgs | read {read_ms:.1f} ms/img | "
                            f"infer {infer_ms:.1f} ms/img | build {build_ms:.1f} ms/img | save {save_ms:.1f} ms/img"
                        )
                    window_read_time = 0.0
                    window_infer_time = 0.0
                    window_build_time = 0.0
                    window_save_time = 0.0
                    window_images = 0

                pbar.update(len(batch_paths))
                pbar.set_postfix({"done": processed, "err": errors})

            pbar.close()

            if args.profile_interval > 0 and window_images > 0:
                read_ms = (window_read_time / max(1, window_images)) * 1000.0
                infer_ms = (window_infer_time / max(1, window_images)) * 1000.0
                build_ms = (window_build_time / max(1, window_images)) * 1000.0
                save_ms = (window_save_time / max(1, window_images)) * 1000.0
                if torch.cuda.is_available():
                    mem_alloc = torch.cuda.memory_allocated() / (1024 ** 3)
                    mem_rsrv = torch.cuda.memory_reserved() / (1024 ** 3)
                    print(
                        f"  [profile-tail] {window_images} imgs | read {read_ms:.1f} ms/img | "
                        f"infer {infer_ms:.1f} ms/img | build {build_ms:.1f} ms/img | "
                        f"save {save_ms:.1f} ms/img | gpu alloc/reserved {mem_alloc:.2f}/{mem_rsrv:.2f} GB"
                    )
                else:
                    print(
                        f"  [profile-tail] {window_images} imgs | read {read_ms:.1f} ms/img | "
                        f"infer {infer_ms:.1f} ms/img | build {build_ms:.1f} ms/img | save {save_ms:.1f} ms/img"
                    )
                window_read_time = 0.0
                window_infer_time = 0.0
                window_build_time = 0.0
                window_save_time = 0.0
                window_images = 0

            cat_elapsed = time.perf_counter() - cat_start
            cat_ms_per_img = (cat_elapsed / len(cat_images) * 1000.0) if cat_images else 0.0
            print(f"  Done: {len(cat_images)} images in {cat_elapsed:.1f}s ({cat_ms_per_img:.0f}ms/img)")

            if not args.keep_model_cache:
                runner.clear_cache()
    finally:
        if io_pool is not None:
            io_pool.shutdown(wait=True)

    save_results(
        output_path,
        results,
        output_format=args.output_format,
        policy=policy,
        calibration=calibration,
        context_mode=args.context_mode,
        backend="ckpt",
        model_threshold=args.threshold,
    )

    processed_from_this_run = max(0, processed - len(existing_results))
    ms_per_img = (total_inference_time / processed_from_this_run * 1000.0) if processed_from_this_run > 0 else 0.0

    print()
    print("=" * 60)
    print("Inference complete")
    print("=" * 60)
    print(f"Processed (this run): {processed_from_this_run}")
    print(f"Skipped (no model): {skipped_no_model}")
    print(f"Errors: {errors}")
    print(f"Total inference time: {total_inference_time:.2f}s")
    print(f"Average: {ms_per_img:.1f}ms/img")
    print(f"Output saved to: {output_path}")

    if results:
        print()
        print("Sample output:")
        print(json.dumps(results[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
