from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class DomainKnowledgeRagService:
    """Retrieve domain knowledge context and build report prompt for API LLM flow."""

    def __init__(
        self,
        *,
        json_path: str,
        persist_dir: str,
        enabled: bool = True,
        top_k: int = 3,
    ) -> None:
        self.enabled = bool(enabled)
        self.top_k = max(1, int(top_k))
        self.json_path = Path(json_path)
        self.persist_dir = Path(persist_dir)

        self._retriever = None
        self._report_prompt_rag = None
        self._ready = False

        self._initialize()

    @property
    def ready(self) -> bool:
        return self.enabled and self._ready and self._retriever is not None and self._report_prompt_rag is not None

    def _initialize(self) -> None:
        if not self.enabled:
            logger.info("Domain knowledge RAG disabled by config")
            return

        if not self.json_path.exists():
            logger.warning("Domain knowledge JSON not found: %s", self.json_path)
            return

        try:
            from src.rag import Indexer, Retrievers
            from src.rag.prompt import report_prompt_rag

            indexer = Indexer(json_path=str(self.json_path), persist_dir=str(self.persist_dir))
            vectorstore = indexer.get_or_create()
            self._retriever = Retrievers(vectorstore)
            self._report_prompt_rag = report_prompt_rag
            self._ready = True

            count = "unknown"
            try:
                count = str(vectorstore._collection.count())  # type: ignore[attr-defined]
            except Exception:
                pass
            logger.info(
                "Domain knowledge RAG ready | docs=%s | json=%s | persist=%s",
                count,
                self.json_path,
                self.persist_dir,
            )
        except Exception:
            logger.exception(
                "Failed to initialize domain knowledge RAG | json=%s persist=%s",
                self.json_path,
                self.persist_dir,
            )

    @staticmethod
    def _split_model_category(model_category: str) -> tuple[str | None, str]:
        raw = str(model_category or "").strip().strip("/")
        if not raw:
            return None, ""
        if "/" not in raw:
            return None, raw
        dataset, category = raw.split("/", 1)
        return (dataset or None), (category or raw)

    @staticmethod
    def _extract_defect_type_hint(category: str, ad_data: dict[str, Any] | None) -> str | None:
        data = ad_data or {}

        for key in ("defect_type", "predicted_defect_type", "gt_defect_type"):
            value = data.get(key)
            if isinstance(value, str):
                v = value.strip().lower().replace(" ", "_")
                if v and v not in {"none", "normal", "good", "review_needed", "anomaly"}:
                    return v

        reason_codes = data.get("reason_codes")
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

        source_path = str(data.get("ingest_source_path") or "").strip()
        if not source_path:
            return None

        stem = Path(source_path).stem.lower()
        tokens = [t for t in re.split(r"[^a-z0-9]+", stem) if t]
        if not tokens:
            return None

        stop = {
            "good",
            "normal",
            "defect",
            "anomaly",
            "image",
            "img",
            "line",
            "incoming",
            "surface",
        }
        category_tokens = {t for t in re.split(r"[^a-z0-9]+", category.lower()) if t}
        filtered = [t for t in tokens if t not in stop and t not in category_tokens and not t.isdigit()]
        if not filtered:
            return None

        if len(filtered) >= 2:
            return f"{filtered[0]}_{filtered[1]}"
        return filtered[0]

    @staticmethod
    def _build_query(category: str, ad_data: dict[str, Any] | None) -> str:
        if not category:
            return "defect anomaly inspection"

        ad_decision = str((ad_data or {}).get("ad_decision", "")).lower()
        if ad_decision == "normal":
            return f"{category} normal product appearance"
        if ad_decision == "anomaly":
            decision_conf = 0.0
            try:
                decision_conf = float((ad_data or {}).get("decision_confidence") or 0.0)
            except (TypeError, ValueError):
                decision_conf = 0.0

        region = (ad_data or {}).get("region")
        area_ratio = (ad_data or {}).get("area_ratio")
        extra: list[str] = []
        if isinstance(region, str) and region.strip():
            extra.append(f"region {region.strip()}")
        if area_ratio is not None:
            extra.append(f"area_ratio {area_ratio}")

        suffix = f" {' '.join(extra)}" if extra else ""
        if ad_decision == "anomaly" and decision_conf >= 0.8:
            return f"{category} defect failure case root cause inspection{suffix}"
        return f"{category} defect anomaly inspection{suffix}"

    def build_report_instruction(
        self,
        *,
        model_category: str,
        ad_data: dict[str, Any] | None = None,
        ad_info_text: str = "",
    ) -> str | None:
        """Return report prompt with retrieved domain knowledge, or None when unavailable."""
        if not self.ready:
            return None

        ad_decision = str((ad_data or {}).get("ad_decision", "")).strip().lower()
        # normal/review_needed 판정 시 결함 도메인 지식 주입을 방지하기 위한 안전망
        # 1차 skip 판단은 llm_service에서 수행하며, 여기서는 이중 안전망으로 동작
        if ad_decision in ("normal", "review_needed"):
            logger.info(
                "Skip domain knowledge RAG for %s | model_category=%s",
                ad_decision,
                model_category,
            )
            return None

        dataset, category = self._split_model_category(model_category)
        if not category:
            return None

        query = self._build_query(category, ad_data)
        defect_type_hint = self._extract_defect_type_hint(category, ad_data)

        try:
            docs = self._retriever.retrieve(  # type: ignore[union-attr]
                query,
                dataset=dataset,
                category=category,
                defect_type=defect_type_hint,
                k=self.top_k,
            )
            if not docs and defect_type_hint:
                docs = self._retriever.retrieve(  # type: ignore[union-attr]
                    query,
                    dataset=dataset,
                    category=category,
                    defect_type=None,
                    k=self.top_k,
                )
            if not docs and dataset:
                docs = self._retriever.retrieve(  # type: ignore[union-attr]
                    query,
                    dataset=None,
                    category=category,
                    defect_type=None,
                    k=self.top_k,
                )
            context = self._retriever.format_context(docs)  # type: ignore[union-attr]
            logger.info(
                "Domain knowledge RAG retrieved %d docs | dataset=%s | category=%s | defect_type_hint=%s",
                len(docs),
                dataset or "*",
                category,
                defect_type_hint or "-",
            )
            return self._report_prompt_rag(  # type: ignore[misc]
                category=category,
                domain_knowledge=context,
                ad_info=ad_info_text,
            )
        except Exception:
            logger.exception("Domain knowledge retrieval failed | model_category=%s", model_category)
            return None
