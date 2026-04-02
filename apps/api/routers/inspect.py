"""POST /inspect 엔드포인트."""
from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool

from ..pipeline import _run_single_inspection

router = APIRouter()


@router.post("/inspect")
async def run_inspection(
    request: Request,
    file: UploadFile = File(...),
    category: str = Form(...),
    dataset: str = Form("default"),
    line: str = Form("line_1"),
    domain_rag_enabled: bool | None = Form(None),
):
    """Run AD immediately and enqueue RAG/LLM stages for asynchronous completion."""
    try:
        file.file.seek(0)
        return await run_in_threadpool(
            _run_single_inspection,
            request.app,
            file,
            category,
            dataset,
            line,
            None,
            domain_rag_enabled,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
