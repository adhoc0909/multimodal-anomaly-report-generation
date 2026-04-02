"""AD 정책 해석 및 결정 로직."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import src.storage.pg as pg
from src.utils.loaders import load_json

logger = logging.getLogger(__name__)


# ── JSON 로더 ──────────────────────────────────────────────────────────────────

def _load_json_doc(path: Path, *, label: str) -> dict[str, Any]:
    if not path.exists():
        logger.warning("%s file not found: %s", label, path)
        return {}
    try:
        loaded = load_json(str(path))
        if isinstance(loaded, dict):
            return loaded
    except Exception:
        logger.exception("Failed to load %s file: %s", label, path)
    return {}


def _load_ad_policy_doc(path: Path) -> dict[str, Any]:
    return _load_json_doc(path, label="AD policy")


def _load_ad_calibration_doc(path: Path) -> dict[str, Any]:
    return _load_json_doc(path, label="AD calibration")


# ── 카테고리 메타데이터 동기화 ───────────────────────────────────────────────────

def _sync_category_metadata_from_policy_doc(conn, doc: dict[str, Any], *, default_line: str | None = None) -> int:
    """Seed/update category_metadata from ad_policy.json classes."""
    classes = doc.get("classes") if isinstance(doc, dict) else None
    if not isinstance(classes, dict):
        logger.warning("Skip category_metadata sync: AD policy has no valid classes object")
        return 0

    rows: list[dict[str, Any]] = []
    for class_key in classes.keys():
        if not isinstance(class_key, str):
            continue
        key = class_key.strip().strip("/")
        if not key:
            continue

        policy = _resolve_class_policy_from_json(doc, key)
        if not policy:
            continue

        dataset: str | None = None
        if "/" in key:
            dataset = key.split("/", 1)[0].strip() or None

        row: dict[str, Any] = {
            "category": key,
            "dataset": dataset,
            "line": default_line or None,
            "t_low": policy.get("t_low"),
            "t_high": policy.get("t_high"),
            "review_band": policy.get("review_band"),
            "reliability": policy.get("reliability"),
            "ad_weight": policy.get("ad_weight"),
            "location_mode": policy.get("location_mode"),
            "min_location_confidence": policy.get("min_location_confidence"),
            "use_bbox": policy.get("use_bbox"),
        }
        rows.append(row)

    if not rows:
        logger.warning("Skip category_metadata sync: no valid class policy rows")
        return 0

    try:
        return pg.upsert_category_metadata(conn, rows)
    except Exception:
        logger.exception("Failed to sync category_metadata from AD policy doc")
        return 0


# ── 수치 유틸 ─────────────────────────────────────────────────────────────────

def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


# ── 정책 키 생성 ──────────────────────────────────────────────────────────────

def _candidate_class_keys(dataset: str, model_category: str) -> list[str]:
    key = model_category.strip().strip("/")
    keys: list[str] = []
    if key:
        keys.append(key)

    ds = dataset.strip().strip("/")
    if key and ds and "/" not in key:
        keys.append(f"{ds}/{key}")

    out: list[str] = []
    seen: set[str] = set()
    for k in keys:
        if k and k not in seen:
            out.append(k)
            seen.add(k)
    return out


# ── 정책 해석 ─────────────────────────────────────────────────────────────────

def _resolve_class_policy_from_json(doc: dict[str, Any], class_key: str) -> dict[str, Any] | None:
    if not isinstance(doc, dict):
        return None

    merged: dict[str, Any] = {}
    source = "ad_policy_json"
    default_raw = doc.get("default")
    if isinstance(default_raw, dict):
        merged.update(default_raw)
    classes_raw = doc.get("classes")
    if isinstance(classes_raw, dict):
        class_raw = classes_raw.get(class_key)
        if isinstance(class_raw, dict):
            merged.update(class_raw)
            source = "ad_policy_json_class"

    if not merged:
        return None

    reliability = str(merged.get("reliability", merged.get("reliability_prior", "medium"))).lower()
    if reliability not in {"high", "medium", "low"}:
        reliability = "medium"

    ad_weight = _safe_float(merged.get("ad_weight"))
    if ad_weight is None:
        ad_weight = {"high": 0.70, "medium": 0.45, "low": 0.20}.get(reliability, 0.45)
    ad_weight = _clip(float(ad_weight), 0.0, 1.0)

    location_mode = str(merged.get("location_mode", "normal")).lower()
    if location_mode not in {"off", "normal", "strict"}:
        location_mode = "normal"

    min_location_confidence = _safe_float(merged.get("min_location_confidence"))
    if min_location_confidence is None:
        min_location_confidence = {"high": 0.25, "medium": 0.35, "low": 0.45}.get(reliability, 0.35)
    min_location_confidence = _clip(float(min_location_confidence), 0.0, 1.0)

    use_bbox_raw = merged.get("use_bbox")
    if isinstance(use_bbox_raw, bool):
        use_bbox = use_bbox_raw
    else:
        use_bbox = location_mode != "off"

    review_band = _safe_float(merged.get("review_band"))
    legacy_center = _safe_float(merged.get("decision_center"))
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
    review_band = _clip(float(review_band), 0.02, 0.30)

    out: dict[str, Any] = {
        "class_key": class_key,
        "source": source,
        "reliability": reliability,
        "ad_weight": ad_weight,
        "location_mode": location_mode,
        "min_location_confidence": min_location_confidence,
        "use_bbox": use_bbox,
        "review_band": review_band,
        "legacy_center": legacy_center,
    }
    if t_low is not None:
        out["t_low"] = float(t_low)
    if t_high is not None:
        out["t_high"] = float(t_high)
    return out


def _resolve_class_policy_from_bounds(raw: dict[str, Any], class_key: str) -> dict[str, Any]:
    reliability = str(raw.get("reliability", "medium")).lower()
    if reliability not in {"high", "medium", "low"}:
        reliability = "medium"

    ad_weight = _safe_float(raw.get("ad_weight"))
    if ad_weight is None:
        ad_weight = {"high": 0.70, "medium": 0.45, "low": 0.20}.get(reliability, 0.45)
    ad_weight = _clip(float(ad_weight), 0.0, 1.0)

    location_mode = str(raw.get("location_mode", "normal")).lower()
    if location_mode not in {"off", "normal", "strict"}:
        location_mode = "normal"

    min_location_confidence = _safe_float(raw.get("min_location_confidence"))
    if min_location_confidence is None:
        min_location_confidence = {"high": 0.25, "medium": 0.35, "low": 0.45}.get(reliability, 0.35)
    min_location_confidence = _clip(float(min_location_confidence), 0.0, 1.0)

    use_bbox_raw = raw.get("use_bbox")
    if isinstance(use_bbox_raw, bool):
        use_bbox = use_bbox_raw
    else:
        use_bbox = location_mode != "off"

    t_low = _safe_float(raw.get("t_low"))
    t_high = _safe_float(raw.get("t_high"))
    if t_low is not None and t_high is not None and t_low > t_high:
        t_low, t_high = t_high, t_low

    if t_low is None:
        t_low = 0.5
    if t_high is None:
        t_high = 0.8
    if t_high <= t_low:
        center = _clip((t_low + t_high) / 2.0, 0.0, 1.0)
        t_low = _clip(center - 0.05, 0.0, 1.0)
        t_high = _clip(center + 0.05, 0.0, 1.0)

    return {
        "class_key": class_key,
        "source": str(raw.get("source", "default")),
        "reliability": reliability,
        "ad_weight": ad_weight,
        "location_mode": location_mode,
        "min_location_confidence": min_location_confidence,
        "use_bbox": use_bbox,
        "review_band": _clip((t_high - t_low) / 2.0, 0.02, 0.30),
        "legacy_center": (t_low + t_high) / 2.0,
        "t_low": float(t_low),
        "t_high": float(t_high),
    }


def _resolve_class_calibration(doc: dict[str, Any], class_key: str) -> dict[str, Any]:
    if not isinstance(doc, dict):
        return {}
    classes = doc.get("classes")
    if not isinstance(classes, dict):
        return {}
    raw = classes.get(class_key)
    if not isinstance(raw, dict):
        return {}

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

    out: dict[str, Any] = {}
    if center is not None:
        out["center_threshold"] = float(center)
    if review_band is not None:
        out["review_band"] = _clip(float(review_band), 0.02, 0.30)
    return out


# ── 의사결정 ──────────────────────────────────────────────────────────────────

def _decide_with_policy(
    anomaly_score: float,
    class_policy: dict[str, Any],
    *,
    model_threshold: float,
    class_calibration: dict[str, Any] | None = None,
) -> tuple[str, bool, dict[str, Any]]:
    _ = class_calibration  # Kept for API compatibility; decision now uses explicit t_low/t_high only.

    t_low = _safe_float(class_policy.get("t_low"))
    t_high = _safe_float(class_policy.get("t_high"))
    if t_low is None or t_high is None:
        center = _safe_float(class_policy.get("legacy_center"))
        if center is None:
            center = float(model_threshold)
        center = _clip(float(center), 0.0, 1.0)
        review_band = _safe_float(class_policy.get("review_band"))
        if review_band is None:
            review_band = 0.08
        band = _clip(float(review_band), 0.02, 0.35)
        t_low = _clip(center - band, 0.0, 1.0)
        t_high = _clip(center + band, 0.0, 1.0)
    else:
        t_low = _clip(float(t_low), 0.0, 1.0)
        t_high = _clip(float(t_high), 0.0, 1.0)
        if t_low > t_high:
            t_low, t_high = t_high, t_low
        center = _clip((t_low + t_high) / 2.0, 0.0, 1.0)
        band = _clip((t_high - t_low) / 2.0, 0.02, 0.35)

    score = float(anomaly_score)
    score_delta = score - center
    if score >= t_high:
        decision = "anomaly"
        review_needed = False
    elif score <= t_low:
        decision = "normal"
        review_needed = False
    else:
        decision = "review_needed"
        review_needed = True

    if review_needed:
        half_band = max(1e-6, (t_high - t_low) / 2.0)
        near_boundary = min(score - t_low, t_high - score)
        decision_confidence = round(_clip(near_boundary / half_band, 0.0, 1.0), 4)
    elif decision == "anomaly":
        denom = max(1e-6, 1.0 - t_high)
        decision_confidence = round(_clip((score - t_high) / denom, 0.0, 1.0), 4)
    else:
        denom = max(1e-6, t_low)
        decision_confidence = round(_clip((t_low - score) / denom, 0.0, 1.0), 4)

    return decision, review_needed, {
        "decision_confidence": decision_confidence,
        "basis": {
            "center_threshold": round(center, 4),
            "uncertainty_band": round(float(band), 4),
            "score_delta": round(float(score_delta), 4),
            "used_calibration": False,
            "t_low": round(float(t_low), 4),
            "t_high": round(float(t_high), 4),
        },
    }


def _to_bool_like(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    s = str(value).strip().lower()
    if not s:
        return None
    if s in {"true", "1", "yes", "y", "anomaly", "ng", "defect", "bad"}:
        return True
    if s in {"false", "0", "no", "n", "normal", "ok", "good"}:
        return False
    return None


def _final_decision_from_llm(llm_response: dict[str, Any]) -> str:
    llm_flag = _to_bool_like(llm_response.get("is_anomaly_llm"))
    if llm_flag is True:
        return "anomaly"
    if llm_flag is False:
        return "normal"
    return "review_needed"


def _final_decision_with_guardrail(
    *,
    llm_response: dict[str, Any],
    ad_decision: str,
    ad_data: dict[str, Any],
    policy: dict[str, Any],
) -> str:
    _ = ad_data
    _ = policy
    ad_decision_norm = str(ad_decision or "").strip().lower()
    if ad_decision_norm in {"anomaly", "normal"}:
        return ad_decision_norm
    return _final_decision_from_llm(llm_response)


def _has_llm_content(llm_response: dict[str, Any]) -> bool:
    summary = llm_response.get("llm_summary")
    report = llm_response.get("llm_report")
    if isinstance(summary, dict) and summary:
        return True
    if isinstance(summary, str) and summary.strip():
        return True
    if isinstance(report, dict):
        for key in ("summary", "description", "possible_cause", "recommendation", "raw_response"):
            value = report.get(key)
            if isinstance(value, str) and value.strip():
                return True
        return bool(report)
    return False
