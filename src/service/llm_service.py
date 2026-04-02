from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict

from src.mllm.base import REPORT_PROMPT_WITH_AD, format_ad_info

logger = logging.getLogger(__name__)


def _resolve_few_shot_path(ref_path: str) -> str | None:
    candidate = Path(ref_path)
    if candidate.exists():
        return str(candidate)

    data_dir = os.getenv("DATA_DIR")
    if not data_dir:
        return None

    data_root = Path(data_dir)
    prefixes = (
        "/home/ubuntu/dataset/",
    )
    for prefix in prefixes:
        if not ref_path.startswith(prefix):
            continue
        suffix = ref_path[len(prefix) :]
        mapped = data_root / suffix
        if mapped.exists():
            return str(mapped)
    return None


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _norm_decision(ad_decision: str) -> str:
    d = str(ad_decision or "").strip().lower()
    if d in {"anomaly", "normal", "review_needed"}:
        return d
    return "review_needed"


def _decision_mode_instruction(ad_decision: str) -> str:
    d = _norm_decision(ad_decision)
    if d == "anomaly":
        return (
            "- AD 판정이 확정 이상(ANOMALY)입니다.\n"
            "- is_anomaly를 true로 고정하고, 결함 원인 분석과 조치 계획을 구체적으로 작성하세요."
        )
    if d == "normal":
        return (
            "- AD 판정이 확정 정상(NORMAL)입니다.\n"
            "- is_anomaly를 false로 고정하고, 정상 근거를 간결하게 작성하세요."
        )
    return (
        "- AD 판정이 REVIEW_NEEDED 구간입니다.\n"
        "- 이 경우에만 이미지 근거로 is_anomaly(true/false)를 최종 판단하세요."
    )


def _is_none_like(value: Any) -> bool:
    s = str(value or "").strip().lower()
    return s in {"", "-", "none", "없음", "해당 없음", "n/a"}


def _extract_defect_type_hint(ad_data: Dict[str, Any]) -> str | None:
    for key in ("defect_type", "predicted_defect_type", "gt_defect_type"):
        value = ad_data.get(key)
        if isinstance(value, str):
            v = value.strip().lower().replace(" ", "_")
            if v and v not in {"none", "normal", "good", "review_needed", "anomaly"}:
                return v

    reason_codes = ad_data.get("reason_codes")
    if isinstance(reason_codes, str):
        v = reason_codes.strip().lower().replace(" ", "_")
        if v and v not in {"none", "normal", "good"}:
            return v
    if isinstance(reason_codes, (list, tuple)):
        for code in reason_codes:
            if isinstance(code, str):
                v = code.strip().lower().replace(" ", "_")
                if v and v not in {"none", "normal", "good"}:
                    return v

    source_path = str(ad_data.get("ingest_source_path") or "").strip()
    if not source_path:
        return None
    stem = Path(source_path).stem.lower()
    tokens = [t for t in re.split(r"[^a-z0-9]+", stem) if t]
    filtered = [t for t in tokens if t not in {"good", "normal", "defect", "anomaly", "image", "img", "line", "incoming", "surface"} and not t.isdigit()]
    if not filtered:
        return None
    if len(filtered) >= 2:
        return f"{filtered[0]}_{filtered[1]}"
    return filtered[0]


def _enforce_report_by_ad_decision(
    result: Dict[str, Any],
    *,
    ad_decision: str,
    region: Any,
) -> Dict[str, Any]:
    """Outside [t_low, t_high], force LLM output to follow AD decision."""
    d = _norm_decision(ad_decision)
    if d not in {"anomaly", "normal"}:
        return result

    out = dict(result)
    forced_is_anomaly = d == "anomaly"
    out["is_anomaly_LLM"] = forced_is_anomaly

    llm_report = out.get("llm_report")
    if isinstance(llm_report, dict):
        report = dict(llm_report)
        if forced_is_anomaly:
            if _is_none_like(report.get("anomaly_type")):
                report["anomaly_type"] = "other"
            if _is_none_like(report.get("severity")):
                report["severity"] = "medium"
            if _is_none_like(report.get("location")):
                report["location"] = str(region or "unknown")
            if _is_none_like(report.get("possible_cause")):
                report["possible_cause"] = "공정 편차 또는 취급 과정 이상 가능성"
            if _is_none_like(report.get("recommendation")):
                report["recommendation"] = "동일 라인 샘플 재검사 및 원인 공정 점검 후 재발방지 조치를 수행하세요."
            conf = _safe_float(report.get("confidence"))
            if conf is None or conf < 0.55:
                report["confidence"] = 0.55
        else:
            report["anomaly_type"] = "none"
            report["severity"] = "none"
            report["location"] = "none"
            report["possible_cause"] = "none"
            conf = _safe_float(report.get("confidence"))
            if conf is None or conf > 0.75:
                report["confidence"] = 0.75
        out["llm_report"] = report

    llm_summary = out.get("llm_summary")
    if isinstance(llm_summary, dict):
        summary = dict(llm_summary)
        if forced_is_anomaly:
            if _is_none_like(summary.get("risk_level")):
                summary["risk_level"] = "medium"
        else:
            summary["risk_level"] = "none"
        out["llm_summary"] = summary
    return out


class LlmService:
    """LLM report adapter returning DB-compatible keys."""

    def __init__(self, client, domain_rag_service=None) -> None:
        self.client = client
        self.domain_rag_service = domain_rag_service

    def generate_dynamic_report(
        self,
        image_path: str,
        ref_path: str | None,
        category: str,
        ad_data: Dict[str, Any],
        policy: Dict[str, Any],
        domain_rag_enabled: bool = True,
    ) -> Dict[str, Any]:
        ad_decision_raw = _norm_decision(str(ad_data.get("ad_decision", "review_needed")))
        defect_type_hint = _extract_defect_type_hint(ad_data)

        few_shot_paths: list[str] = []
        if ref_path:
            resolved = _resolve_few_shot_path(ref_path)
            if resolved:
                few_shot_paths = [resolved]
            else:
                logger.warning("Skipping unavailable RAG reference image for LLM: %s", ref_path)

        decision_confidence = ad_data.get("decision_confidence")
        if decision_confidence is None:
            fallback_confidence = ad_data.get("confidence")
            if isinstance(fallback_confidence, (int, float, str)):
                decision_confidence = fallback_confidence

        ad_info = {
            "anomaly_score": ad_data.get("ad_score", 0.0),
            "decision": ad_decision_raw,
            "is_anomaly": ad_decision_raw == "anomaly",
            "decision_confidence": decision_confidence,
            "reason_codes": ad_data.get("reason_codes"),
            "defect_type_hint": defect_type_hint,
            "defect_location": {
                "has_defect": ad_data.get("has_defect"),
                "region": ad_data.get("region"),
                "area_ratio": ad_data.get("area_ratio"),
            },
        }
        confidence_raw = ad_data.get("confidence")
        if isinstance(confidence_raw, dict):
            ad_info["confidence"] = confidence_raw
        else:
            try:
                if confidence_raw is not None:
                    ad_info["location_confidence"] = float(confidence_raw)
            except (TypeError, ValueError):
                pass

        if isinstance(policy, dict):
            policy_info: Dict[str, Any] = {}
            for key in (
                "reliability",
                "ad_weight",
                "review_band",
                "t_low",
                "t_high",
                "location_mode",
                "min_location_confidence",
                "use_bbox",
            ):
                value = policy.get(key)
                if value is not None:
                    policy_info[key] = value
            if policy_info:
                ad_info["policy"] = policy_info
                if "confidence" not in ad_info and isinstance(policy_info.get("reliability"), str):
                    ad_info["confidence"] = {"reliability": str(policy_info["reliability"]).strip().lower()}

        if isinstance(ad_data.get("decision_basis"), dict):
            ad_info["decision_basis"] = ad_data["decision_basis"]
        if "review_needed" in ad_data:
            ad_info["review_needed"] = bool(ad_data.get("review_needed"))

        ad_info_text = format_ad_info(ad_info)

        report_instruction: str | None = None

        # 요청 단위로 domain RAG 사용 여부를 명시적으로 비활성화한 경우
        if not domain_rag_enabled:
            prompt_mode = "ad_only_disabled"

        # normal 판정: 결함 도메인 지식을 주입하면 LLM이 결함으로 오해할 소지가 있어 skip
        # review_needed 판정: 경계 케이스로 도메인 지식 주입이 판정을 왜곡할 수 있어 skip
        elif ad_decision_raw in ("normal", "review_needed"):
            prompt_mode = "ad_only_skipped"

        elif self.domain_rag_service is not None:
            try:
                report_instruction = self.domain_rag_service.build_report_instruction(
                    model_category=category,
                    ad_data=ad_data,
                    ad_info_text=ad_info_text,
                )
            except Exception:
                logger.exception("Domain knowledge RAG prompt build failed for category=%s", category)
            # RAG 호출은 시도했으나 결과가 없는 경우 (서비스 미준비, 검색 실패 등)
            prompt_mode = "domain_rag" if report_instruction else "ad_only_fallback"

        else:
            # domain_rag_service 자체가 초기화되지 않은 경우
            prompt_mode = "ad_only_fallback"

        if not report_instruction:
            report_instruction = REPORT_PROMPT_WITH_AD.format(
                category=category,
                ad_info=ad_info_text,
            )

        report_instruction = (
            f"{report_instruction}\n\n"
            f"판정 적용 규칙:\n{_decision_mode_instruction(ad_decision_raw)}"
        )

        result = self.client.generate_report(
            image_path=image_path,
            category=category,
            ad_info=ad_info,
            few_shot_paths=few_shot_paths,
            instruction=report_instruction,
        )

        result = _enforce_report_by_ad_decision(
            result,
            ad_decision=ad_decision_raw,
            region=ad_data.get("region"),
        )

        llm_report = result.get("llm_report")
        if isinstance(llm_report, dict):
            meta = dict(llm_report.get("_metadata", {})) if isinstance(llm_report.get("_metadata"), dict) else {}
            meta.update(
                {
                    "prompt_mode": prompt_mode,
                    "decision_mode": ad_decision_raw,
                    "ad_info_used": ad_info,
                    "reference_image_path": few_shot_paths[0] if few_shot_paths else None,
                }
            )
            llm_report = dict(llm_report)
            llm_report["_metadata"] = meta

        return {
            "is_anomaly_llm": result.get("is_anomaly_LLM"),
            "llm_report": llm_report,
            "llm_summary": result.get("llm_summary"),
            "llm_inference_duration": result.get("llm_inference_duration"),
            "ad_decision": ad_decision_raw,
        }
