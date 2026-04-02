"""FastAPI 애플리케이션 진입점 - 앱 초기화 및 라우터 등록."""
from __future__ import annotations

import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import src.storage.pg as pg
from src.service.ad_service import AdService
from src.service.domain_rag_service import DomainKnowledgeRagService
from src.service.visual_rag_service import RagService

from .background import incoming_watch_loop, pipeline_watchdog_loop
from .config import (
    AD_CALIBRATION_PATH,
    AD_POLICY_PATH,
    CHECKPOINT_DIR,
    DATA_DIR,
    DATABASE_URL,
    DOMAIN_KNOWLEDGE_JSON_PATH,
    DOMAIN_RAG_ENABLED,
    DOMAIN_RAG_PERSIST_DIR,
    DOMAIN_RAG_TOP_K,
    INCOMING_DEFAULT_LINE,
    INCOMING_ROOT,
    INCOMING_WATCH_ENABLED,
    MODEL_NAME,
    OUTPUT_DIR,
    PIPELINE_WORKERS,
    RAG_INDEX_DIR,
)
from .pipeline import _set_llm_model
from .policy import (
    _load_ad_calibration_doc,
    _load_ad_policy_doc,
    _sync_category_metadata_from_policy_doc,
)
from .routers import incoming, inspect, reports, settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── 정적 파일 서빙 유틸 ───────────────────────────────────────────────────────

class CorsStaticFiles(StaticFiles):
    """Static file mount that always exposes CORS headers for browser-side rendering."""

    def file_response(self, full_path, stat_result, scope, status_code=200):  # type: ignore[override]
        response = super().file_response(full_path, stat_result, scope, status_code=status_code)
        if getattr(response, "status_code", 0) == 200:
            response.headers.setdefault("Access-Control-Allow-Origin", "*")
            response.headers.setdefault("Access-Control-Allow-Methods", "GET,HEAD,OPTIONS")
            response.headers.setdefault("Access-Control-Allow-Headers", "*")
            response.headers.setdefault("Cross-Origin-Resource-Policy", "cross-origin")
        return response


class CachedStaticFiles(CorsStaticFiles):
    """Static file mount with cache headers for immutable output filenames."""

    def file_response(self, full_path, stat_result, scope, status_code=200):  # type: ignore[override]
        response = super().file_response(full_path, stat_result, scope, status_code=status_code)
        if getattr(response, "status_code", 0) == 200:
            response.headers.setdefault("Cache-Control", "public, max-age=604800, immutable")
        return response


# ── 앱 생명주기 ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Boot-time DB connectivity and additive migrations
    bootstrap_conn = pg.connect(DATABASE_URL, ensure_schema=True)
    ad_policy_doc = _load_ad_policy_doc(AD_POLICY_PATH)
    class_count = len(ad_policy_doc.get("classes", {})) if isinstance(ad_policy_doc, dict) else 0
    logger.info(
        "Boot policy sync start | ad_policy_path=%s classes=%d",
        AD_POLICY_PATH,
        class_count,
    )
    synced = _sync_category_metadata_from_policy_doc(
        bootstrap_conn,
        ad_policy_doc,
        default_line=INCOMING_DEFAULT_LINE,
    )
    if synced > 0:
        logger.info("Synced %d category_metadata row(s) from AD policy doc", synced)
    else:
        logger.warning("No category_metadata rows synced from AD policy doc")
    bootstrap_conn.close()

    app.state.ad_service = AdService(
        checkpoint_dir=str(CHECKPOINT_DIR),
        output_root=str(OUTPUT_DIR),
    )
    app.state.ad_policy_doc = ad_policy_doc
    app.state.ad_calibration_doc = _load_ad_calibration_doc(AD_CALIBRATION_PATH)
    app.state.rag_service = RagService(index_dir=str(RAG_INDEX_DIR))
    app.state.domain_rag_service = DomainKnowledgeRagService(
        json_path=str(DOMAIN_KNOWLEDGE_JSON_PATH),
        persist_dir=str(DOMAIN_RAG_PERSIST_DIR),
        enabled=DOMAIN_RAG_ENABLED,
        top_k=DOMAIN_RAG_TOP_K,
    )
    _set_llm_model(app, MODEL_NAME)
    app.state.inspect_lock = threading.Lock()
    app.state.llm_lock = threading.Lock()
    app.state.pipeline_executor = ThreadPoolExecutor(max_workers=PIPELINE_WORKERS)
    app.state.incoming_unresolved = set()

    incoming_task: asyncio.Task | None = None
    watchdog_task: asyncio.Task | None = asyncio.create_task(pipeline_watchdog_loop())
    if INCOMING_WATCH_ENABLED:
        INCOMING_ROOT.mkdir(parents=True, exist_ok=True)
        incoming_task = asyncio.create_task(incoming_watch_loop(app))

    try:
        yield
    finally:
        if incoming_task is not None:
            incoming_task.cancel()
            try:
                await incoming_task
            except asyncio.CancelledError:
                pass
        if watchdog_task is not None:
            watchdog_task.cancel()
            try:
                await watchdog_task
            except asyncio.CancelledError:
                pass
        try:
            app.state.pipeline_executor.shutdown(wait=False, cancel_futures=False)
        except Exception:
            logger.exception("Failed to shutdown pipeline executor")


# ── FastAPI 앱 ────────────────────────────────────────────────────────────────

app = FastAPI(title="Industrial AI Inspection System", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", CachedStaticFiles(directory=str(OUTPUT_DIR), check_dir=False), name="outputs")
app.mount("/home/ubuntu/dataset", CorsStaticFiles(directory=str(DATA_DIR), check_dir=False), name="datasets")

app.include_router(inspect.router)
app.include_router(reports.router)
app.include_router(settings.router)
app.include_router(incoming.router)
