from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from src.rag.visual_rag import VisualRAG


class RagService:
    """Visual RAG wrapper with frontend-serving path normalization."""

    def __init__(self, index_dir: str, model_type: str = "dinov2_vits14") -> None:
        self.rag = VisualRAG(index_dir=index_dir, model_name=model_type)

    def search_closest_normal(self, query_image_path: str, category: str, top_k: int = 1) -> List[Dict[str, Any]]:
        results = self.rag.search(query_image_path, category=category, k=top_k)
        normalized: List[Dict[str, Any]] = []
        for item in results:
            raw_path = str(item.get("path", ""))
            filename = Path(raw_path).name if raw_path else ""
            normalized.append(
                {
                    # Public URL path served by FastAPI static mount.
                    "path": f"/home/ubuntu/dataset/{category}/train/good/{filename}" if filename else "",
                    "score": float(item.get("score", 0.0)),
                }
            )
        return normalized
