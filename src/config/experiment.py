"""Experiment configuration loader."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

from src.service.settings import load_yaml


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""

    ad_model: Optional[str] = None       # patchcore | winclip | efficientad | null
    llm: str = "qwen"                    # model name from MODEL_REGISTRY
    few_shot: int = 1
    similar_template: bool = True
    max_images: Optional[int] = None
    sample_per_folder: Optional[int] = None  # 폴더(dataset/category/split)별 N장 샘플링
    sample_seed: int = 42
    data_root: Optional[str] = None
    mmad_json: Optional[str] = None
    max_image_size: Tuple[int, int] = (512, 512)  # LLM에 전달할 이미지 최대 크기
    output_dir: str = "outputs/eval"
    batch_mode: bool = False
    resume: bool = False

    # AD model settings
    ad_config: Optional[str] = None          # anomaly.yaml 경로 (run_ad_inference.py용)
    ad_output: Optional[str] = None          # AD 예측 JSON 경로 (있으면 inference 스킵)
    ad_threshold: Optional[float] = None     # anomaly threshold override
    ad_checkpoint_dir: Optional[str] = None  # 체크포인트 루트 경로
    ad_version: Optional[int] = None         # 체크포인트 버전 (null = 최신)

    # RAG settings
    rag: bool = False                        # RAG 도메인 지식 주입 여부
    rag_json_path: Optional[str] = None      # domain_knowledge.json 경로
    rag_persist_dir: Optional[str] = None    # Chroma vectorstore 경로 (Config A/B/C 비교용)
    rag_k: int = 3                           # 검색할 문서 수

    @property
    def experiment_name(self) -> str:
        """Auto-generate experiment name: {ad_model}_{llm}_{few_shot}shot[_rag]"""
        ad = self.ad_model or "no_ad"
        rag_suffix = "_rag" if self.rag else ""
        return f"{ad}_{self.llm}_{self.few_shot}shot{rag_suffix}"


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load ExperimentConfig from a YAML file.

    Supports ``${ENV_VAR:-default}`` expansion via ``load_yaml()``.
    """
    d = load_yaml(path)

    eval_section = d.get("eval", {})
    ad_section = d.get("ad", {})
    rag_section = d.get("rag", {})

    return ExperimentConfig(
        ad_model=d.get("ad_model"),
        llm=d.get("llm", "qwen"),
        few_shot=eval_section.get("few_shot", 1),
        similar_template=eval_section.get("similar_template", True),
        max_images=eval_section.get("max_images"),
        sample_per_folder=eval_section.get("sample_per_folder"),
        sample_seed=eval_section.get("sample_seed", 42),
        max_image_size=tuple(eval_section.get("max_image_size", [512, 512])),
        data_root=d.get("data_root"),
        mmad_json=d.get("mmad_json"),
        output_dir=d.get("output_dir", "outputs/eval"),
        batch_mode=eval_section.get("batch_mode", False),
        resume=eval_section.get("resume", False),
        ad_config=ad_section.get("config"),
        ad_output=ad_section.get("output"),
        ad_threshold=ad_section.get("threshold"),
        ad_checkpoint_dir=ad_section.get("checkpoint_dir"),
        ad_version=ad_section.get("version"),
        rag=rag_section.get("enabled", False),
        rag_json_path=rag_section.get("json_path"),
        rag_k=rag_section.get("top_k", 3),
    )
