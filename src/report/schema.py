from __future__ import annotations
from pathlib import Path
import json
from jsonschema import validate

def load_schema(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))

def validate_report(report: dict, schema: dict) -> None:
    validate(instance=report, schema=schema)
