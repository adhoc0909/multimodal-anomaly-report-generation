"""검사 파이프라인 오케스트레이션."""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI

import src.storage.pg as pg
from src.mllm.factory import get_llm_client
from src.service.llm_service import LlmService

from .config import (
    CHECKPOINT_DIR,
    DATABASE_URL,
    DOMAIN_RAG_ENABLED,
    INCOMING_DEFAULT_CATEGORY,
    INCOMING_DEFAULT_DATASET,
    INCOMING_DEFAULT_LINE,
    INCOMING_IMAGE_EXTS,
    INCOMING_ROOT,
    INCOMING_STABLE_SECONDS,
    LINE_PATTERN,
    LLM_LOCK_TIMEOUT_SEC,
    LLM_MODEL_ALIASES,
)
from .policy import (
    _candidate_class_keys,
    _clip,
    _decide_with_policy,
    _final_decision_with_guardrail,
    _has_llm_content,
    _resolve_class_calibration,
    _resolve_class_policy_from_bounds,
    _resolve_class_policy_from_json,
    _safe_float,
)

logger = logging.getLogger(__name__)


# ── 파일 래퍼 ─────────────────────────────────────────────────────────────────

class LocalImageUpload:
    """UploadFile-like wrapper for filesystem image ingestion."""

    def __init__(self, path: Path) -> None:
        self.filename = path.name
        self.file = path.open("rb")

    def close(self) -> None:
        try:
            self.file.close()
        except Exception:
            pass


# ── DB 헬퍼 ──────────────────────────────────────────────────────────────────

def _safe_update_report(conn, report_id: int, payload: dict[str, Any], *, context: str) -> bool:
    """Best-effort DB update with one rollback+retry for aborted transactions."""
    try:
        pg.update_report(conn, report_id, payload)
        return True
    except Exception:
        logger.exception("Report update failed (%s) | report_id=%s", context, report_id)
        try:
            conn.rollback()
        except Exception:
            pass
        try:
            pg.update_report(conn, report_id, payload)
            return True
        except Exception:
            logger.exception("Report update retry failed (%s) | report_id=%s", context, report_id)
            return False


def _mark_pipeline_failed(
    conn,
    *,
    report_id: int,
    stage: str,
    error: str,
    extra: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = dict(extra or {})
    payload.update({
        "ad_decision": "review_needed",
        "pipeline_status": "failed",
        "pipeline_stage": stage,
        "pipeline_error": str(error)[:500],
    })
    _safe_update_report(conn, report_id, payload, context=f"mark_failed:{stage}")


# ── LLM 모델 전환 ─────────────────────────────────────────────────────────────

def _set_llm_model(app: FastAPI, model_name: str) -> None:
    selected = model_name.strip().lower()
    if not selected:
        raise ValueError("Model name must not be empty")
    normalized = LLM_MODEL_ALIASES.get(selected, selected)
    if normalized != selected:
        logger.info("Normalized LLM model alias: %s -> %s", selected, normalized)
    selected = normalized
    app.state.llm_client = get_llm_client(selected)
    app.state.llm_service = LlmService(
        client=app.state.llm_client,
        domain_rag_service=getattr(app.state, "domain_rag_service", None),
    )
    app.state.llm_model = selected


# ── 체크포인트/카테고리 탐색 ───────────────────────────────────────────────────

def _checkpoint_exists_for_category_key(category_key: str) -> bool:
    key = category_key.strip().strip("/")
    if not key:
        return False

    roots = [CHECKPOINT_DIR / "Patchcore", CHECKPOINT_DIR]
    seen: set[Path] = set()
    for root in roots:
        if root in seen or not root.exists():
            continue
        seen.add(root)

        category_dir = root / key
        if category_dir.exists() and category_dir.is_dir():
            for v_dir in category_dir.iterdir():
                if not v_dir.is_dir() or not v_dir.name.startswith("v"):
                    continue
                if (v_dir / "model.ckpt").exists():
                    return True
    return False


def _discover_model_category_key(category: str) -> str | None:
    selected = category.strip().strip("/")
    if not selected or "/" in selected:
        return None

    roots = [CHECKPOINT_DIR / "Patchcore", CHECKPOINT_DIR]
    seen: set[Path] = set()
    for root in roots:
        if root in seen or not root.exists():
            continue
        seen.add(root)

        dataset_dirs = sorted([d for d in root.iterdir() if d.is_dir()], key=lambda p: p.name)
        for ds_dir in dataset_dirs:
            if ds_dir.name.startswith("."):
                continue
            candidate = f"{ds_dir.name}/{selected}"
            if _checkpoint_exists_for_category_key(candidate):
                return candidate
    return None


def _resolve_model_category(dataset: str, category: str) -> str:
    selected = category.strip()
    if not selected:
        raise ValueError("Category must not be empty")
    if "/" in selected:
        return selected

    ds = dataset.strip().strip("/")
    if ds:
        candidate = f"{ds}/{selected}"
        if _checkpoint_exists_for_category_key(candidate):
            return candidate
    discovered = _discover_model_category_key(selected)
    if discovered:
        return discovered
    return selected


# ── incoming 경로 파싱 ────────────────────────────────────────────────────────

def _normalize_line(raw: str | None) -> str | None:
    if not raw:
        return None
    s = raw.strip()
    if not s:
        return None

    up = s.upper()
    if up.startswith("LINE-"):
        return up

    m = LINE_PATTERN.search(s)
    if not m:
        if len(s) == 1 and s.isalpha():
            return f"LINE-{s.upper()}-01"
        return None

    token = m.group(1).upper()
    if len(token) == 1 and token.isalpha():
        return f"LINE-{token}-01"
    return f"LINE-{token}"


def _resolve_incoming_context(image_path: Path) -> tuple[str, str, str] | None:
    try:
        rel_parts = image_path.relative_to(INCOMING_ROOT).parts
    except ValueError:
        rel_parts = image_path.parts

    dir_parts = rel_parts[:-1] if len(rel_parts) >= 1 else ()

    dataset = INCOMING_DEFAULT_DATASET
    category = INCOMING_DEFAULT_CATEGORY

    if len(dir_parts) >= 2:
        dataset = dir_parts[0] or INCOMING_DEFAULT_DATASET
        category = dir_parts[1] or INCOMING_DEFAULT_CATEGORY
    elif len(dir_parts) == 1:
        dataset = INCOMING_DEFAULT_DATASET or dir_parts[0]
        category = INCOMING_DEFAULT_CATEGORY

    line: str | None = None
    if len(dir_parts) >= 3:
        line = _normalize_line(dir_parts[2])

    if line is None:
        if len(dir_parts) >= 3:
            tokens = list(dir_parts[2:])
        else:
            tokens = list(dir_parts)
        for token in reversed(tokens):
            line = _normalize_line(token)
            if line is not None:
                break

    line = line or INCOMING_DEFAULT_LINE

    if not category:
        return None
    return dataset, category, line


def _collect_incoming_images(root: Path) -> list[Path]:
    items: list[tuple[float, Path]] = []
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in INCOMING_IMAGE_EXTS:
            continue
        try:
            items.append((p.stat().st_mtime, p))
        except OSError:
            continue
    items.sort(key=lambda x: x[0])
    return [p for _, p in items]


def _is_stable_file(path: Path) -> bool:
    try:
        age = datetime.now().timestamp() - path.stat().st_mtime
        return age >= INCOMING_STABLE_SECONDS
    except OSError:
        return False


# ── 파이프라인 핵심 ───────────────────────────────────────────────────────────

def _finalize_rag_llm_pipeline(
    app: FastAPI,
    *,
    report_id: int,
    model_category: str,
    ad_result: dict[str, Any],
    ad_decision: str,
    policy: dict[str, Any],
    domain_rag_enabled: bool | None = None,
) -> None:
    effective_domain_rag = domain_rag_enabled if domain_rag_enabled is not None else DOMAIN_RAG_ENABLED
    conn = pg.connect_fast(DATABASE_URL)
    ref_path: str | None = None
    try:
        _safe_update_report(
            conn,
            report_id,
            {"pipeline_status": "processing", "pipeline_stage": "rag_running"},
            context="rag_running",
        )

        try:
            rag_results = app.state.rag_service.search_closest_normal(ad_result["original_path"], model_category)
            ref_path = rag_results[0]["path"] if rag_results else None
            _safe_update_report(
                conn,
                report_id,
                {
                    "similar_image_path": ref_path,
                    "pipeline_status": "processing",
                    "pipeline_stage": "rag_done",
                },
                context="rag_done",
            )
        except Exception as exc:
            logger.exception("RAG stage failed for report_id=%s", report_id)
            _safe_update_report(
                conn,
                report_id,
                {
                    "pipeline_status": "processing",
                    "pipeline_stage": "rag_failed",
                    "pipeline_error": f"RAG: {str(exc)[:500]}",
                },
                context="rag_failed",
            )

        _safe_update_report(
            conn,
            report_id,
            {"pipeline_status": "processing", "pipeline_stage": "llm_waiting_lock"},
            context="llm_waiting_lock",
        )
        llm_started = datetime.now()
        acquired = app.state.llm_lock.acquire(timeout=LLM_LOCK_TIMEOUT_SEC)
        if not acquired:
            _mark_pipeline_failed(
                conn,
                report_id=report_id,
                stage="llm_lock_timeout",
                error=f"LLM lock wait timeout ({LLM_LOCK_TIMEOUT_SEC:.1f}s)",
            )
            return

        try:
            _safe_update_report(
                conn,
                report_id,
                {"pipeline_status": "processing", "pipeline_stage": "llm_running"},
                context="llm_running",
            )
            llm_response = app.state.llm_service.generate_dynamic_report(
                ad_result["original_path"],
                ref_path,
                model_category,
                ad_result,
                policy,
                domain_rag_enabled=effective_domain_rag,
            )
        finally:
            app.state.llm_lock.release()

        if not _has_llm_content(llm_response):
            _mark_pipeline_failed(
                conn,
                report_id=report_id,
                stage="llm_empty",
                error="LLM returned empty/invalid content",
                extra=llm_response,
            )
            return

        llm_decision = _final_decision_with_guardrail(
            llm_response=llm_response,
            ad_decision=ad_decision,
            ad_data=ad_result,
            policy=policy,
        )
        _safe_update_report(
            conn,
            report_id,
            {
                **llm_response,
                "ad_decision": llm_decision,
                "llm_start_time": llm_started,
                "pipeline_status": "completed",
                "pipeline_stage": "llm_done",
                "pipeline_error": None,
            },
            context="llm_done",
        )
    except Exception as exc:
        logger.exception("LLM stage failed for report_id=%s", report_id)
        _mark_pipeline_failed(
            conn,
            report_id=report_id,
            stage="llm_failed",
            error=f"LLM: {str(exc)[:500]}",
        )
    finally:
        conn.close()


def _run_inspection_pipeline(
    app: FastAPI,
    *,
    conn,
    file_obj,
    category: str,
    dataset: str,
    line: str,
    ingest_source_path: str | None,
    domain_rag_enabled: bool | None = None,
) -> dict[str, Any]:
    model_category = _resolve_model_category(dataset=dataset, category=category)
    class_keys = _candidate_class_keys(dataset=dataset, model_category=model_category)

    policy: dict[str, Any] | None = None
    for key in class_keys:
        policy = _resolve_class_policy_from_json(getattr(app.state, "ad_policy_doc", {}), key)
        if policy is not None:
            break
    if policy is None:
        db_policy = pg.get_category_policy(conn, model_category)
        policy = _resolve_class_policy_from_bounds(db_policy, model_category)
    policy_source = str(policy.get("source", "default"))

    if hasattr(file_obj, "file"):
        file_obj.file.seek(0)

    ad_start_time = datetime.now()
    with app.state.inspect_lock:
        ad_results = app.state.ad_service.predict_batch(
            [file_obj],
            category=model_category,
            dataset=dataset,
            threshold=None,
        )
    ad_result = ad_results[0]
    score = float(ad_result["ad_score"])
    model_thr = float(ad_result.get("model_threshold", 0.5))
    model_is_anomaly = bool(ad_result.get("model_is_anomaly", score > model_thr))

    calibration: dict[str, Any] = {}
    for key in class_keys:
        calibration = _resolve_class_calibration(getattr(app.state, "ad_calibration_doc", {}), key)
        if calibration:
            break

    ad_decision, review_needed, decision_meta = _decide_with_policy(
        score,
        policy,
        model_threshold=model_thr,
        class_calibration=calibration,
    )
    basis = decision_meta.get("basis", {})
    basis_t_low = _safe_float(basis.get("t_low"))
    basis_t_high = _safe_float(basis.get("t_high"))
    if basis_t_low is not None and basis_t_high is not None:
        t_low = _clip(float(basis_t_low), 0.0, 1.0)
        t_high = _clip(float(basis_t_high), 0.0, 1.0)
        if t_low > t_high:
            t_low, t_high = t_high, t_low
        center = _clip((t_low + t_high) / 2.0, 0.0, 1.0)
        band = max(1e-6, (t_high - t_low) / 2.0)
    else:
        center = _clip(float(_safe_float(basis.get("center_threshold")) or model_thr), 0.0, 1.0)
        band = _clip(float(_safe_float(basis.get("uncertainty_band")) or policy.get("review_band", 0.08)), 0.0, 0.35)
        t_low = _clip(center - band, 0.0, 1.0)
        t_high = _clip(center + band, 0.0, 1.0)
    applied_policy = {
        "class_key": str(policy.get("class_key", model_category)),
        "source": policy_source,
        "reliability": str(policy.get("reliability", "medium")),
        "review_band": round(float(policy.get("review_band", band)), 4),
        "t_low": round(float(t_low), 4),
        "t_high": round(float(t_high), 4),
        "decision_basis": basis,
        "used_calibration": bool(calibration),
    }

    logger.info(
        "AD decision | category=%s score=%.4f center=%.4f band=%.4f range=[%.4f, %.4f] "
        "source=%s model_thr=%.4f model_is_anomaly=%s review_needed=%s decision=%s",
        model_category,
        score,
        center,
        band,
        t_low,
        t_high,
        policy_source,
        model_thr,
        model_is_anomaly,
        review_needed,
        ad_decision,
    )
    ad_result["ad_decision"] = ad_decision
    ad_result["review_needed"] = review_needed
    ad_result["decision_confidence"] = decision_meta.get("decision_confidence")
    ad_result["decision_basis"] = basis
    ad_result["ingest_source_path"] = ingest_source_path

    initial_data = {
        "dataset": dataset,
        "category": model_category,
        "line": line,
        "ad_score": score,
        "ad_decision": "review_needed",
        "is_anomaly_ad": ad_decision == "anomaly",
        "has_defect": ad_result.get("has_defect", False),
        "region": ad_result.get("region"),
        "area_ratio": ad_result.get("area_ratio"),
        "bbox": ad_result.get("bbox"),
        "center": ad_result.get("center"),
        "confidence": ad_result.get("confidence"),
        "ingest_source_path": ingest_source_path,
        "image_path": ad_result["original_path"],
        "heatmap_path": ad_result["heatmap_path"],
        "mask_path": ad_result["mask_path"],
        "overlay_path": ad_result["overlay_path"],
        "created_at": ad_start_time,
        "ad_inference_duration": (datetime.now() - ad_start_time).total_seconds(),
        "pipeline_status": "processing",
        "pipeline_stage": "ad_done",
        "applied_policy": applied_policy,
    }
    report_id = pg.insert_report(conn, initial_data)

    try:
        app.state.pipeline_executor.submit(
            _finalize_rag_llm_pipeline,
            app,
            report_id=report_id,
            model_category=model_category,
            ad_result=ad_result,
            ad_decision=ad_decision,
            policy=policy,
            domain_rag_enabled=domain_rag_enabled,
        )
    except Exception as exc:
        logger.exception("Failed to enqueue RAG/LLM pipeline for report_id=%s", report_id)
        pg.update_report(
            conn,
            report_id,
            {
                "pipeline_status": "failed",
                "pipeline_stage": "enqueue_failed",
                "pipeline_error": f"QUEUE: {str(exc)[:500]}",
            },
        )

    return {
        "status": "accepted",
        "report_id": report_id,
        "ad_decision": "review_needed",
        "category": model_category,
        "source_image": ingest_source_path,
        "pipeline_status": "processing",
        "pipeline_stage": "ad_done",
    }


def _run_single_inspection(
    app: FastAPI,
    file_obj,
    category: str,
    dataset: str,
    line: str,
    ingest_source_path: str | None,
    domain_rag_enabled: bool | None = None,
) -> dict[str, Any]:
    conn = pg.connect_fast(DATABASE_URL)
    try:
        return _run_inspection_pipeline(
            app,
            conn=conn,
            file_obj=file_obj,
            category=category,
            dataset=dataset,
            line=line,
            ingest_source_path=ingest_source_path,
            domain_rag_enabled=domain_rag_enabled,
        )
    finally:
        conn.close()
