"""GET/PUT /settings/llm-model 엔드포인트."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from src.mllm.factory import list_llm_models

from ..config import MODEL_NAME
from ..pipeline import _set_llm_model

router = APIRouter()


class LlmModelUpdateRequest(BaseModel):
    model: str


@router.get("/settings/llm-model")
async def get_llm_model_settings(request: Request):
    return {
        "active_model": str(getattr(request.app.state, "llm_model", MODEL_NAME)),
        "available_models": list_llm_models(),
    }


@router.put("/settings/llm-model")
async def update_llm_model_settings(payload: LlmModelUpdateRequest, request: Request):
    model_name = payload.model.strip()
    try:
        _set_llm_model(request.app, model_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "active_model": str(request.app.state.llm_model),
        "available_models": list_llm_models(),
    }
