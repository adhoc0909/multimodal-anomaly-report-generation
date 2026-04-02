"""백그라운드 비동기 루프 (incoming 감시, pipeline watchdog)."""
from __future__ import annotations

import asyncio
import logging

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool

import src.storage.pg as pg

from .config import (
    DATABASE_URL,
    INCOMING_ROOT,
    INCOMING_SCAN_INTERVAL_SEC,
    INCOMING_STABLE_SECONDS,
    PIPELINE_STALE_SECONDS,
    PIPELINE_WATCHDOG_INTERVAL_SEC,
)
from .pipeline import (
    LocalImageUpload,
    _collect_incoming_images,
    _is_stable_file,
    _resolve_incoming_context,
    _run_inspection_pipeline,
)

logger = logging.getLogger(__name__)


def _scan_incoming_once(app: FastAPI) -> int:
    if not INCOMING_ROOT.exists():
        return 0

    conn = pg.connect_fast(DATABASE_URL)
    processed = 0
    skipped_existing = 0
    skipped_unresolved = 0
    failed = 0
    try:
        for image_path in _collect_incoming_images(INCOMING_ROOT):
            if not _is_stable_file(image_path):
                continue

            source_path = str(image_path.resolve())
            if pg.has_ingest_source_path(conn, source_path):
                skipped_existing += 1
                continue

            resolved = _resolve_incoming_context(image_path)
            if resolved is None:
                skipped_unresolved += 1
                unresolved: set[str] = app.state.incoming_unresolved
                if source_path not in unresolved:
                    logger.warning(
                        "Skipping incoming image without category mapping: %s "
                        "(use incoming/{dataset}/{category}/... structure or set INCOMING_DEFAULT_CATEGORY)",
                        source_path,
                    )
                    unresolved.add(source_path)
                continue
            dataset, category, line = resolved

            local_file = LocalImageUpload(image_path)
            try:
                _run_inspection_pipeline(
                    app,
                    conn=conn,
                    file_obj=local_file,
                    category=category,
                    dataset=dataset,
                    line=line,
                    ingest_source_path=source_path,
                )
                processed += 1
                app.state.incoming_unresolved.discard(source_path)
                logger.info("Incoming image processed: %s", source_path)
            except Exception:
                try:
                    conn.rollback()
                except Exception:
                    pass
                failed += 1
                logger.exception("Incoming image processing failed: %s", source_path)
            finally:
                local_file.close()

        if processed > 0 or failed > 0 or skipped_unresolved > 0:
            logger.info(
                "Incoming scan summary | processed=%d skipped_existing=%d unresolved=%d failed=%d",
                processed,
                skipped_existing,
                skipped_unresolved,
                failed,
            )
        return processed
    finally:
        conn.close()


def _mark_stale_pipelines_once() -> int:
    conn = pg.connect_fast(DATABASE_URL)
    try:
        updated = pg.mark_stale_processing_reports(
            conn,
            stale_seconds=PIPELINE_STALE_SECONDS,
            limit=1000,
        )
        return int(updated)
    finally:
        conn.close()


async def incoming_watch_loop(app: FastAPI) -> None:
    logger.info(
        "Incoming watcher enabled | root=%s | interval=%.1fs | stable=%.1fs",
        INCOMING_ROOT,
        INCOMING_SCAN_INTERVAL_SEC,
        INCOMING_STABLE_SECONDS,
    )
    while True:
        try:
            processed = await run_in_threadpool(_scan_incoming_once, app)
            if processed > 0:
                logger.info("Incoming watcher processed %d image(s)", processed)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Incoming watcher loop failed")

        await asyncio.sleep(max(1.0, INCOMING_SCAN_INTERVAL_SEC))


async def pipeline_watchdog_loop() -> None:
    logger.info(
        "Pipeline watchdog enabled | interval=%.1fs | stale_after=%ds",
        PIPELINE_WATCHDOG_INTERVAL_SEC,
        PIPELINE_STALE_SECONDS,
    )
    while True:
        try:
            updated = await run_in_threadpool(_mark_stale_pipelines_once)
            if updated > 0:
                logger.warning(
                    "Pipeline watchdog marked %d stale report(s) as failed",
                    updated,
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Pipeline watchdog loop failed")
        await asyncio.sleep(PIPELINE_WATCHDOG_INTERVAL_SEC)
