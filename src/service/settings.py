from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os
import re
import yaml

_env_pattern = re.compile(r"\$\{([^:}]+)(?::-(.*?))?\}")

def _expand_env(text: str) -> str:
    def repl(m):
        key = m.group(1)
        default = m.group(2) if m.group(2) is not None else ""
        return os.environ.get(key, default)
    return _env_pattern.sub(repl, text)

def load_yaml(path: str | Path) -> dict:
    raw = Path(path).read_text(encoding="utf-8")
    raw = _expand_env(raw)
    return yaml.safe_load(raw)

@dataclass
class RuntimePaths:
    data_root: str
    mmad_json: str
    artifact_root: str
    db_path: str

@dataclass
class RuntimeConfig:
    paths: RuntimePaths
    anomaly: dict
    mllm: dict
    report: dict

def load_runtime_config(path: str | Path) -> RuntimeConfig:
    d = load_yaml(path)
    p = d["paths"]
    return RuntimeConfig(
        paths=RuntimePaths(
            data_root=str(p["data_root"]),
            mmad_json=str(p["mmad_json"]),
            artifact_root=str(p["artifact_root"]),
            db_path=str(p["db_path"]),
        ),
        anomaly=d["anomaly"],
        mllm=d["mllm"],
        report=d["report"],
    )
