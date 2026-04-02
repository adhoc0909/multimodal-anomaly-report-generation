from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.loaders import load_json

logger = logging.getLogger(__name__)

# Cognex PDF: 1-indexed page range → (category, defect_type)
# 선택한 11 pages만 매핑 (나머지는 스킵)
_PDF_PAGE_MAP: Dict[Tuple[int, int], Tuple[str, str]] = {
    (5, 6):   ("food_package", "seal_void_fold"),
    (6, 7):   ("food_bottle",  "tamper_safety_seal"),
    (7, 8):   ("food_package", "contamination_foreign"),
    (8, 9):   ("drink_bottle", "bottle_cap"),
    (9, 10):  ("food_box",     "cosmetic_scratch_crack"),
    (11, 12): ("breakfast_box","assembly_kitting"),
    (12, 13): ("breakfast_box","missing_item"),
    (14, 15): ("juice_bottle", "label_quality"),
    (15, 16): ("food_bottle",  "allergen_label"),
    (16, 17): ("juice_bottle", "skewed_label"),
    (20, 21): ("food_box",     "mislabeled_packaging"),
}


def _get_pdf_category(page_num: int) -> Tuple[str, str] | None:
    """1-indexed 페이지 번호로 (category, defect_type) 반환. 선택 범위 밖이면 None."""
    for (start, end), mapping in _PDF_PAGE_MAP.items():
        if start <= page_num < end:
            return mapping
    return None


class PDFKnowledgeLoader:
    """Cognex 패키징 가이드 PDF에서 선택된 pages만 로드해 Document 리스트로 변환.

    선택 기준: GoodsAD / MVTec-LOCO 카테고리와 직접 매핑되는 11 pages.
    각 Document의 metadata는 기존 domain_knowledge.json 스키마와 호환:
        dataset, category, defect_type, source_type="pdf", page
    """

    def __init__(
        self,
        pdf_path: str | Path,
        dataset_name: str = "PackagingGuide",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        min_length: int = 200,
    ):
        self.pdf_path = Path(pdf_path)
        self.dataset_name = dataset_name
        self.min_length = min_length
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def load(self) -> List[Document]:
        """PDF 로드 → 선택된 pages 필터링 → 청크 → Documents 반환."""
        loader = PyPDFLoader(str(self.pdf_path))
        pages = loader.load()  # 각 page가 하나의 Document (metadata["page"] 0-indexed)

        docs: List[Document] = []
        for page_doc in pages:
            page_num = page_doc.metadata.get("page", 0) + 1  # 0-indexed → 1-indexed
            mapping = _get_pdf_category(page_num)
            if mapping is None:
                continue  # 선택된 11 pages 밖이면 스킵

            category, defect_type = mapping
            chunks = self.splitter.split_documents([page_doc])
            for chunk in chunks:
                text = chunk.page_content.strip()
                if len(text) < self.min_length:
                    continue
                chunk.metadata.update({
                    "dataset": self.dataset_name,
                    "category": category,
                    "defect_type": defect_type,
                    "source_type": "pdf",
                    "page": page_num,
                })
                docs.append(chunk)

        logger.info(
            "PDFKnowledgeLoader: %d chunks from %d selected pages (%s)",
            len(docs), len(_PDF_PAGE_MAP), self.pdf_path.name,
        )
        return docs


class JSONKnowledgeLoader:
    """domain_knowledge.json 포맷의 JSON 파일을 Document 리스트로 변환.

    포맷: {dataset: {category: {defect_type: description}}}
    Indexer.load_documents()와 동일한 파싱 로직이나,
    vectorstore 빌드 없이 Document만 반환해 add_documents()로 주입 가능.
    각 Document의 metadata:
        dataset, category, defect_type, source_type="json"
    """

    def __init__(self, json_path: str | Path):
        self.json_path = Path(json_path)

    def load(self) -> List[Document]:
        """JSON 로드 → Documents 반환."""
        data = load_json(str(self.json_path))
        docs: List[Document] = []

        for dataset, categories in data.items():
            for category, defect_types in categories.items():
                for defect_type, description in defect_types.items():
                    description = description.strip()
                    if not description:
                        continue
                    docs.append(Document(
                        page_content=description,
                        metadata={
                            "dataset": dataset,
                            "category": category,
                            "defect_type": defect_type,
                            "source_type": "json",
                        },
                    ))

        logger.info(
            "JSONKnowledgeLoader: %d documents from %s",
            len(docs), self.json_path.name,
        )
        return docs
