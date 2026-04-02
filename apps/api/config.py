from __future__ import annotations

import os
import re
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATABASE_URL = os.getenv("DATABASE_URL", os.getenv("PG_DSN", "postgresql://son:1234@localhost/inspection"))
MODEL_NAME = os.getenv("LLM_MODEL", "internvl")
CHECKPOINT_DIR = Path(os.getenv("AD_CHECKPOINT_DIR", str(PROJECT_ROOT / "checkpoints")))
RAG_INDEX_DIR = Path(os.getenv("RAG_INDEX_DIR", str(PROJECT_ROOT / "rag")))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", str(PROJECT_ROOT / "outputs")))
DATA_DIR = Path(os.getenv("DATA_DIR", str(PROJECT_ROOT / "datasets")))
AD_POLICY_PATH = Path(os.getenv("AD_POLICY_PATH", str(PROJECT_ROOT / "configs" / "ad_policy.json")))
AD_CALIBRATION_PATH = Path(os.getenv("AD_CALIBRATION_PATH", str(PROJECT_ROOT / "configs" / "ad_calibration.json")))
DOMAIN_KNOWLEDGE_JSON_PATH = Path(
    os.getenv("DOMAIN_KNOWLEDGE_JSON_PATH", str(DATA_DIR / "domain_knowledge.json"))
)
DOMAIN_RAG_PERSIST_DIR = Path(
    os.getenv("DOMAIN_RAG_PERSIST_DIR", str(PROJECT_ROOT / "vectorstore" / "domain_knowledge"))
)
DOMAIN_RAG_TOP_K = max(1, int(os.getenv("DOMAIN_RAG_TOP_K", "3")))

INCOMING_ROOT = Path(os.getenv("INCOMING_ROOT", "/home/ubuntu/incoming"))
INCOMING_SCAN_INTERVAL_SEC = float(os.getenv("INCOMING_SCAN_INTERVAL_SEC", "5"))
INCOMING_STABLE_SECONDS = float(os.getenv("INCOMING_STABLE_SECONDS", "2"))
INCOMING_DEFAULT_DATASET = os.getenv("INCOMING_DEFAULT_DATASET", "incoming")
INCOMING_DEFAULT_CATEGORY = os.getenv("INCOMING_DEFAULT_CATEGORY", "")
INCOMING_DEFAULT_LINE = os.getenv("INCOMING_DEFAULT_LINE", "LINE-A-01")
INCOMING_IMAGE_EXTS: frozenset[str] = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".webp"})

PIPELINE_WORKERS = max(1, int(os.getenv("PIPELINE_WORKERS", "2")))
LLM_LOCK_TIMEOUT_SEC = max(1.0, float(os.getenv("LLM_LOCK_TIMEOUT_SEC", "30")))
PIPELINE_WATCHDOG_INTERVAL_SEC = max(2.0, float(os.getenv("PIPELINE_WATCHDOG_INTERVAL_SEC", "10")))
PIPELINE_STALE_SECONDS = max(30, int(float(os.getenv("PIPELINE_STALE_SECONDS", "120"))))

INCOMING_WATCH_ENABLED = _env_bool("INCOMING_WATCH_ENABLED", True)
DOMAIN_RAG_ENABLED = _env_bool("DOMAIN_RAG_ENABLED", True)

LINE_PATTERN = re.compile(r"(?i)line[_-]?([a-z0-9]+)")

LLM_MODEL_ALIASES: dict[str, str] = {
    "internv1": "internvl",
    "internv1-8b": "internvl-8b",
    "internv1-4b": "internvl-4b",
    "internv1-2b": "internvl-2b",
    "internv1-1b": "internvl-1b",
}
