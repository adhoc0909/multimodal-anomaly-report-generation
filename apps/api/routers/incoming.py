"""GET /incoming/status 엔드포인트."""
from __future__ import annotations

from fastapi import APIRouter

from ..config import (
    DOMAIN_KNOWLEDGE_JSON_PATH,
    DOMAIN_RAG_ENABLED,
    DOMAIN_RAG_PERSIST_DIR,
    DOMAIN_RAG_TOP_K,
    INCOMING_DEFAULT_CATEGORY,
    INCOMING_DEFAULT_DATASET,
    INCOMING_DEFAULT_LINE,
    INCOMING_ROOT,
    INCOMING_SCAN_INTERVAL_SEC,
    INCOMING_STABLE_SECONDS,
    INCOMING_WATCH_ENABLED,
    LLM_LOCK_TIMEOUT_SEC,
    PIPELINE_STALE_SECONDS,
    PIPELINE_WATCHDOG_INTERVAL_SEC,
    PIPELINE_WORKERS,
)

router = APIRouter()


@router.get("/incoming/status")
async def get_incoming_status():
    return {
        "enabled": INCOMING_WATCH_ENABLED,
        "root": str(INCOMING_ROOT),
        "scan_interval_sec": INCOMING_SCAN_INTERVAL_SEC,
        "stable_seconds": INCOMING_STABLE_SECONDS,
        "pipeline_workers": PIPELINE_WORKERS,
        "llm_lock_timeout_sec": LLM_LOCK_TIMEOUT_SEC,
        "pipeline_watchdog_interval_sec": PIPELINE_WATCHDOG_INTERVAL_SEC,
        "pipeline_stale_seconds": PIPELINE_STALE_SECONDS,
        "domain_rag_enabled": DOMAIN_RAG_ENABLED,
        "domain_rag_json_path": str(DOMAIN_KNOWLEDGE_JSON_PATH),
        "domain_rag_persist_dir": str(DOMAIN_RAG_PERSIST_DIR),
        "domain_rag_top_k": DOMAIN_RAG_TOP_K,
        "default_dataset": INCOMING_DEFAULT_DATASET,
        "default_category": INCOMING_DEFAULT_CATEGORY,
        "default_line": INCOMING_DEFAULT_LINE,
    }
