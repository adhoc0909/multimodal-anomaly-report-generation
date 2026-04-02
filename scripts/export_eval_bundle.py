#!/usr/bin/env python3
"""Export inspection_reports into evaluation artifacts (CSV/JSONL).

Standalone helper for server-side verification runs.
Does not modify any existing application code.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from psycopg2.extras import RealDictCursor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.storage import pg  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database-url",
        type=str,
        default=os.getenv("DATABASE_URL", os.getenv("PG_DSN", "postgresql://son:1234@localhost/inspection")),
        help="PostgreSQL DSN. Defaults to DATABASE_URL/PG_DSN env.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Output directory. Default: output/eval_bundle_<timestamp>",
    )
    parser.add_argument("--dataset", type=str, default="", help="Filter by dataset.")
    parser.add_argument("--category", type=str, default="", help="Filter by category.")
    parser.add_argument("--line", type=str, default="", help="Filter by line.")
    parser.add_argument(
        "--status",
        type=str,
        default="completed",
        help="Filter by pipeline_status (default: completed). Use empty string for all.",
    )
    parser.add_argument(
        "--since-id",
        type=int,
        default=0,
        help="Export rows with id > since_id.",
    )
    parser.add_argument(
        "--min-created-at",
        type=str,
        default="",
        help="ISO datetime filter (inclusive), e.g. 2026-02-24T00:00:00+00:00",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max rows per fetch (0 means no limit).",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Continuously poll DB and append new rows.",
    )
    parser.add_argument(
        "--interval-sec",
        type=float,
        default=5.0,
        help="Polling interval when --watch is enabled.",
    )
    parser.add_argument(
        "--max-empty-polls",
        type=int,
        default=0,
        help="Stop watch mode after this many empty polls (0 = never stop).",
    )
    return parser.parse_args()


def _parse_iso_datetime(raw: str) -> datetime | None:
    s = (raw or "").strip()
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except ValueError as exc:
        raise ValueError(f"Invalid --min-created-at: {raw}") from exc


def _build_query(
    *,
    dataset: str,
    category: str,
    line: str,
    status: str,
    since_id: int,
    min_created_at: datetime | None,
    limit: int,
) -> Tuple[str, List[Any]]:
    clauses = ["1=1"]
    params: List[Any] = []

    if status.strip():
        clauses.append("pipeline_status = %s")
        params.append(status.strip())
    if dataset.strip():
        clauses.append("dataset = %s")
        params.append(dataset.strip())
    if category.strip():
        clauses.append("category = %s")
        params.append(category.strip())
    if line.strip():
        clauses.append("line = %s")
        params.append(line.strip())
    if since_id > 0:
        clauses.append("id > %s")
        params.append(int(since_id))
    if min_created_at is not None:
        clauses.append("created_at >= %s")
        params.append(min_created_at)

    sql = f"""
    SELECT
      id,
      dataset,
      category,
      line,
      ad_score,
      ad_decision,
      is_anomaly_ad,
      has_defect,
      region,
      area_ratio,
      bbox,
      center,
      confidence,
      ingest_source_path,
      image_path,
      heatmap_path,
      mask_path,
      overlay_path,
      similar_image_path,
      is_anomaly_llm,
      llm_report,
      llm_summary,
      applied_policy,
      ad_inference_duration,
      llm_inference_duration,
      pipeline_status,
      pipeline_stage,
      pipeline_error,
      created_at,
      updated_at
    FROM inspection_reports
    WHERE {' AND '.join(clauses)}
    ORDER BY id ASC
    """
    if limit > 0:
        sql += " LIMIT %s"
        params.append(int(limit))

    return sql, params


def _safe_json_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except TypeError:
        return str(value)


def _llm_final_decision(is_anomaly_llm: Any) -> str:
    if is_anomaly_llm is True:
        return "anomaly"
    if is_anomaly_llm is False:
        return "normal"
    return "review_needed"


def _extract_ad_meta(llm_report: Any) -> Dict[str, Any]:
    if not isinstance(llm_report, dict):
        return {}
    metadata = llm_report.get("_metadata")
    if not isinstance(metadata, dict):
        return {}
    ad_info_used = metadata.get("ad_info_used")
    if not isinstance(ad_info_used, dict):
        return {}
    return ad_info_used


def _flatten_row(row: Dict[str, Any]) -> Dict[str, Any]:
    ad_meta = _extract_ad_meta(row.get("llm_report"))
    meta_policy = ad_meta.get("policy") if isinstance(ad_meta.get("policy"), dict) else {}
    meta_basis = ad_meta.get("decision_basis") if isinstance(ad_meta.get("decision_basis"), dict) else {}

    meta_ad_decision = str(ad_meta.get("decision", "")).strip().lower()
    llm_decision = _llm_final_decision(row.get("is_anomaly_llm"))
    hard_conflict = (
        meta_ad_decision in {"anomaly", "normal"}
        and llm_decision in {"anomaly", "normal"}
        and meta_ad_decision != llm_decision
    )

    flat: Dict[str, Any] = {
        "id": row.get("id"),
        "dataset": row.get("dataset"),
        "category": row.get("category"),
        "line": row.get("line"),
        "image_path": row.get("image_path"),
        "ingest_source_path": row.get("ingest_source_path"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
        "pipeline_status": row.get("pipeline_status"),
        "pipeline_stage": row.get("pipeline_stage"),
        "pipeline_error": row.get("pipeline_error"),
        "ad_score": row.get("ad_score"),
        "ad_decision_column": row.get("ad_decision"),
        "is_anomaly_ad_column": row.get("is_anomaly_ad"),
        "has_defect": row.get("has_defect"),
        "region": row.get("region"),
        "area_ratio": row.get("area_ratio"),
        "location_confidence": row.get("confidence"),
        "is_anomaly_llm": row.get("is_anomaly_llm"),
        "llm_final_decision": llm_decision,
        "meta_ad_decision": meta_ad_decision or "",
        "meta_decision_confidence": ad_meta.get("decision_confidence"),
        "meta_review_needed": ad_meta.get("review_needed"),
        "meta_center_threshold": meta_basis.get("center_threshold"),
        "meta_uncertainty_band": meta_basis.get("uncertainty_band"),
        "meta_score_delta": meta_basis.get("score_delta"),
        "policy_reliability": meta_policy.get("reliability"),
        "policy_ad_weight": meta_policy.get("ad_weight"),
        "policy_review_band": meta_policy.get("review_band"),
        "policy_t_low": meta_policy.get("t_low"),
        "policy_t_high": meta_policy.get("t_high"),
        "hard_conflict": hard_conflict,
        "ad_inference_duration": row.get("ad_inference_duration"),
        "llm_inference_duration": row.get("llm_inference_duration"),
        "llm_report_json": _safe_json_text(row.get("llm_report")),
        "llm_summary_json": _safe_json_text(row.get("llm_summary")),
        "applied_policy_json": _safe_json_text(row.get("applied_policy")),
    }
    return flat


def _fetch_rows(conn, sql: str, params: List[Any]) -> List[Dict[str, Any]]:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, tuple(params))
        rows = cur.fetchall()
    return [dict(r) for r in rows]


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]], *, append: bool) -> int:
    mode = "a" if append else "w"
    count = 0
    with path.open(mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            count += 1
    return count


def _write_csv(path: Path, rows: List[Dict[str, Any]], *, append: bool, fieldnames: List[str]) -> int:
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not append:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})
    return len(rows)


def _update_summary(summary_path: Path, flat_rows: List[Dict[str, Any]], total_rows: int) -> None:
    by_category = Counter()
    by_llm = Counter()
    by_meta_ad = Counter()
    by_conflict = Counter()
    for r in flat_rows:
        by_category[str(r.get("category") or "")] += 1
        by_llm[str(r.get("llm_final_decision") or "")] += 1
        by_meta_ad[str(r.get("meta_ad_decision") or "")] += 1
        by_conflict[str(r.get("hard_conflict"))] += 1

    payload = {
        "exported_at": datetime.now().isoformat(),
        "total_rows": total_rows,
        "counts": {
            "by_category": dict(by_category),
            "by_llm_final_decision": dict(by_llm),
            "by_meta_ad_decision": dict(by_meta_ad),
            "by_hard_conflict": dict(by_conflict),
        },
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)


def _default_output_dir() -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "output" / f"eval_bundle_{ts}"


def main() -> None:
    args = parse_args()
    min_created_at = _parse_iso_datetime(args.min_created_at)

    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else _default_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_jsonl_path = out_dir / "reports_raw.jsonl"
    pred_csv_path = out_dir / "predictions.csv"
    conflict_csv_path = out_dir / "hard_conflicts.csv"
    summary_path = out_dir / "summary.json"

    fieldnames = [
        "id",
        "dataset",
        "category",
        "line",
        "image_path",
        "ingest_source_path",
        "created_at",
        "updated_at",
        "pipeline_status",
        "pipeline_stage",
        "pipeline_error",
        "ad_score",
        "ad_decision_column",
        "is_anomaly_ad_column",
        "has_defect",
        "region",
        "area_ratio",
        "location_confidence",
        "is_anomaly_llm",
        "llm_final_decision",
        "meta_ad_decision",
        "meta_decision_confidence",
        "meta_review_needed",
        "meta_center_threshold",
        "meta_uncertainty_band",
        "meta_score_delta",
        "policy_reliability",
        "policy_ad_weight",
        "policy_review_band",
        "policy_t_low",
        "policy_t_high",
        "hard_conflict",
        "ad_inference_duration",
        "llm_inference_duration",
        "llm_report_json",
        "llm_summary_json",
        "applied_policy_json",
    ]

    print(f"[export] output_dir={out_dir}")

    conn = pg.connect_fast(args.database_url)
    total_rows = 0
    all_flat_rows: List[Dict[str, Any]] = []
    last_id = int(args.since_id)
    empty_polls = 0
    first_batch = True

    try:
        while True:
            sql, params = _build_query(
                dataset=args.dataset,
                category=args.category,
                line=args.line,
                status=args.status,
                since_id=last_id,
                min_created_at=min_created_at,
                limit=int(args.limit),
            )
            rows = _fetch_rows(conn, sql, params)

            if rows:
                flat_rows = [_flatten_row(r) for r in rows]
                _write_jsonl(raw_jsonl_path, rows, append=not first_batch)
                _write_csv(pred_csv_path, flat_rows, append=not first_batch, fieldnames=fieldnames)
                conflict_rows = [r for r in flat_rows if bool(r.get("hard_conflict"))]
                if conflict_rows:
                    _write_csv(
                        conflict_csv_path,
                        conflict_rows,
                        append=(not first_batch and conflict_csv_path.exists()),
                        fieldnames=fieldnames,
                    )

                batch_ids = [int(r["id"]) for r in rows if r.get("id") is not None]
                if batch_ids:
                    last_id = max(batch_ids)
                total_rows += len(rows)
                all_flat_rows.extend(flat_rows)
                _update_summary(summary_path, all_flat_rows, total_rows)

                print(
                    f"[export] +{len(rows)} rows (total={total_rows}) "
                    f"last_id={last_id} conflicts={len(conflict_rows)}"
                )
                first_batch = False
                empty_polls = 0
            else:
                empty_polls += 1
                if not args.watch:
                    break
                if args.max_empty_polls > 0 and empty_polls >= args.max_empty_polls:
                    print(f"[export] stop: max empty polls reached ({args.max_empty_polls})")
                    break
                time.sleep(max(0.5, float(args.interval_sec)))

            if not args.watch:
                break
    finally:
        conn.close()

    print(f"[done] raw={raw_jsonl_path}")
    print(f"[done] pred={pred_csv_path}")
    print(f"[done] conflicts={conflict_csv_path if conflict_csv_path.exists() else '(none)'}")
    print(f"[done] summary={summary_path}")


if __name__ == "__main__":
    main()
