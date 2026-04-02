"""PostgreSQL CRUD for inspection reports with legacy compatibility."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import psycopg2
from psycopg2 import errorcodes
from psycopg2.extras import Json, RealDictCursor

logger = logging.getLogger(__name__)

REPORT_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS inspection_reports (
  id                     SERIAL PRIMARY KEY,
  dataset                VARCHAR(50),
  category               VARCHAR(100),
  line                   VARCHAR(50),
  ad_score               FLOAT,
  ad_decision            VARCHAR(20),
  is_anomaly_ad          BOOLEAN,
  has_defect             BOOLEAN,
  region                 TEXT,
  bbox                   JSONB,
  center                 JSONB,
  area_ratio             FLOAT,
  confidence             FLOAT,
  ingest_source_path     TEXT,
  image_path             TEXT,
  heatmap_path           TEXT,
  mask_path              TEXT,
  overlay_path           TEXT,
  similar_image_path     TEXT,
  ad_start_time          TIMESTAMP,
  ad_inference_duration  FLOAT,
  is_anomaly_llm         BOOLEAN,
  llm_report             JSONB,
  llm_summary            JSONB,
  llm_start_time         TIMESTAMP,
  llm_inference_duration FLOAT,
  pipeline_status        VARCHAR(30) DEFAULT 'processing',
  pipeline_stage         VARCHAR(30) DEFAULT 'ad_done',
  pipeline_error         TEXT,
  applied_policy         JSONB DEFAULT '{}'::jsonb,
  created_at             TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at             TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

CATEGORY_METADATA_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS category_metadata (
  category                VARCHAR(100) PRIMARY KEY,
  dataset                 VARCHAR(50),
  line                    VARCHAR(50) DEFAULT 'line_1',
  t_low                   FLOAT DEFAULT 0.5,
  t_high                  FLOAT DEFAULT 0.8,
  review_band             FLOAT,
  reliability             VARCHAR(20),
  ad_weight               FLOAT,
  location_mode           VARCHAR(20),
  min_location_confidence FLOAT,
  use_bbox                BOOLEAN,
  created_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

REPORT_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_reports_category ON inspection_reports(category);
CREATE INDEX IF NOT EXISTS idx_reports_decision ON inspection_reports(ad_decision);
CREATE INDEX IF NOT EXISTS idx_reports_created ON inspection_reports(created_at);
CREATE INDEX IF NOT EXISTS idx_reports_ingest_source_path ON inspection_reports(ingest_source_path);
CREATE INDEX IF NOT EXISTS idx_reports_pipeline_status ON inspection_reports(pipeline_status);
"""

ALTER_REPORT_COLUMNS_SQL = [
    "ALTER TABLE inspection_reports ADD COLUMN IF NOT EXISTS dataset VARCHAR(50);",
    "ALTER TABLE inspection_reports ADD COLUMN IF NOT EXISTS category VARCHAR(100);",
    "ALTER TABLE inspection_reports ADD COLUMN IF NOT EXISTS line VARCHAR(50);",
    "ALTER TABLE inspection_reports ADD COLUMN IF NOT EXISTS ad_score FLOAT;",
    "ALTER TABLE inspection_reports ADD COLUMN IF NOT EXISTS ad_decision VARCHAR(20);",
    "ALTER TABLE inspection_reports ADD COLUMN IF NOT EXISTS is_anomaly_ad BOOLEAN;",
    "ALTER TABLE inspection_reports ADD COLUMN IF NOT EXISTS has_defect BOOLEAN;",
    "ALTER TABLE inspection_reports ADD COLUMN IF NOT EXISTS region TEXT;",
    "ALTER TABLE inspection_reports ADD COLUMN IF NOT EXISTS bbox JSONB;",
    "ALTER TABLE inspection_reports ADD COLUMN IF NOT EXISTS center JSONB;",
    "ALTER TABLE inspection_reports ADD COLUMN IF NOT EXISTS area_ratio FLOAT;",
    "ALTER TABLE inspection_reports ADD COLUMN IF NOT EXISTS confidence FLOAT;",
    "ALTER TABLE inspection_reports ADD COLUMN IF NOT EXISTS ingest_source_path TEXT;",
    "ALTER TABLE inspection_reports ADD COLUMN IF NOT EXISTS image_path TEXT;",
    "ALTER TABLE inspection_reports ADD COLUMN IF NOT EXISTS heatmap_path TEXT;",
    "ALTER TABLE inspection_reports ADD COLUMN IF NOT EXISTS mask_path TEXT;",
    "ALTER TABLE inspection_reports ADD COLUMN IF NOT EXISTS overlay_path TEXT;",
    "ALTER TABLE inspection_reports ADD COLUMN IF NOT EXISTS similar_image_path TEXT;",
    "ALTER TABLE inspection_reports ADD COLUMN IF NOT EXISTS ad_start_time TIMESTAMP;",
    "ALTER TABLE inspection_reports ADD COLUMN IF NOT EXISTS ad_inference_duration FLOAT;",
    "ALTER TABLE inspection_reports ADD COLUMN IF NOT EXISTS is_anomaly_llm BOOLEAN;",
    "ALTER TABLE inspection_reports ADD COLUMN IF NOT EXISTS llm_report JSONB;",
    "ALTER TABLE inspection_reports ADD COLUMN IF NOT EXISTS llm_summary JSONB;",
    "ALTER TABLE inspection_reports ADD COLUMN IF NOT EXISTS llm_start_time TIMESTAMP;",
    "ALTER TABLE inspection_reports ADD COLUMN IF NOT EXISTS llm_inference_duration FLOAT;",
    "ALTER TABLE inspection_reports ADD COLUMN IF NOT EXISTS applied_policy JSONB DEFAULT '{}'::jsonb;",
    "ALTER TABLE inspection_reports ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;",
    "ALTER TABLE inspection_reports ADD COLUMN IF NOT EXISTS pipeline_status VARCHAR(30) DEFAULT 'processing';",
    "ALTER TABLE inspection_reports ADD COLUMN IF NOT EXISTS pipeline_stage VARCHAR(30) DEFAULT 'ad_done';",
    "ALTER TABLE inspection_reports ADD COLUMN IF NOT EXISTS pipeline_error TEXT;",
    "ALTER TABLE inspection_reports ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;",
]

DEFAULT_POLICY = {"t_low": 0.5, "t_high": 0.8}

ALLOWED_COLUMNS = {
    "dataset",
    "category",
    "line",
    "ad_score",
    "ad_decision",
    "is_anomaly_ad",
    "has_defect",
    "region",
    "bbox",
    "center",
    "area_ratio",
    "confidence",
    "ingest_source_path",
    "image_path",
    "heatmap_path",
    "mask_path",
    "overlay_path",
    "similar_image_path",
    "ad_start_time",
    "ad_inference_duration",
    "is_anomaly_llm",
    "llm_report",
    "llm_summary",
    "llm_start_time",
    "llm_inference_duration",
    "pipeline_status",
    "pipeline_stage",
    "pipeline_error",
    "applied_policy",
    "created_at",
    "updated_at",
}

JSON_COLUMNS = {"bbox", "center", "llm_report", "llm_summary", "applied_policy"}

COLUMN_ALIASES = {
    "is_anomaly_ad": "is_anomaly_ad",
    "is_anomaly_AD": "is_anomaly_ad",
    "is_anomaly_llm": "is_anomaly_llm",
    "is_anomaly_LLM": "is_anomaly_llm",
    "ad_start_time": "ad_start_time",
    "AD_start_time": "ad_start_time",
    "ad_inference_duration": "ad_inference_duration",
    "AD_inference_duration": "ad_inference_duration",
}


def connect(dsn: str, *, ensure_schema: bool = True) -> psycopg2.extensions.connection:
    """Connect to PostgreSQL. Schema sync is optional for hot paths."""
    conn = psycopg2.connect(dsn)
    if ensure_schema:
        create_tables(conn)
    return conn


def connect_fast(dsn: str) -> psycopg2.extensions.connection:
    """Connect without running schema DDL on every request."""
    return connect(dsn, ensure_schema=False)


def create_tables(conn: psycopg2.extensions.connection) -> None:
    """Create required tables and ensure additive schema compatibility."""
    with conn.cursor() as cur:
        cur.execute(REPORT_SCHEMA_SQL)
        cur.execute(CATEGORY_METADATA_SCHEMA_SQL)
        for ddl in ALTER_REPORT_COLUMNS_SQL:
            cur.execute(ddl)
        cur.execute(REPORT_INDEX_SQL)
    conn.commit()


def _normalize_column_name(raw_key: str) -> Optional[str]:
    col = COLUMN_ALIASES.get(raw_key, raw_key)
    if col in ALLOWED_COLUMNS:
        return col
    col_lower = COLUMN_ALIASES.get(raw_key.lower(), raw_key.lower())
    if col_lower in ALLOWED_COLUMNS:
        return col_lower
    return None


def _adapt_json_value(column: str, value: Any) -> Any:
    if column not in JSON_COLUMNS or value is None:
        return value
    if isinstance(value, (dict, list)):
        return Json(value)
    if isinstance(value, str):
        try:
            return Json(json.loads(value))
        except json.JSONDecodeError:
            return Json({"raw": value})
    return Json(value)


def _normalized_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for raw_key, raw_value in data.items():
        col = _normalize_column_name(str(raw_key))
        if col is None:
            continue
        out[col] = _adapt_json_value(col, raw_value)
    return out


def _existing_report_columns(conn: psycopg2.extensions.connection) -> set[str]:
    sql = """
    SELECT column_name
    FROM information_schema.columns
    WHERE table_schema = 'public' AND table_name = 'inspection_reports'
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
    return {str(r[0]) for r in rows}


def insert_report(conn: psycopg2.extensions.connection, data: Dict[str, Any]) -> int:
    """Insert a report row and return its id."""
    payload = _normalized_payload(data)
    if not payload:
        raise ValueError("No valid columns to insert into inspection_reports")

    columns = list(payload.keys())
    placeholders = ", ".join(["%s"] * len(columns))
    col_names = ", ".join(columns)
    sql = f"INSERT INTO inspection_reports ({col_names}) VALUES ({placeholders}) RETURNING id"

    with conn.cursor() as cur:
        cur.execute(sql, [payload[c] for c in columns])
        report_id = cur.fetchone()[0]
    conn.commit()
    return int(report_id)


def update_report(conn: psycopg2.extensions.connection, report_id: int, data: Dict[str, Any]) -> None:
    """Update report row by id with partial data."""
    payload = _normalized_payload(data)
    if not payload:
        return

    attempted_drop_missing = False
    while True:
        columns = list(payload.keys())
        if not columns:
            return

        set_clause = ", ".join([f"{col} = %s" for col in columns] + ["updated_at = CURRENT_TIMESTAMP"])
        sql = f"UPDATE inspection_reports SET {set_clause} WHERE id = %s"
        try:
            with conn.cursor() as cur:
                cur.execute(sql, [payload[c] for c in columns] + [report_id])
            conn.commit()
            return
        except psycopg2.Error as exc:
            conn.rollback()
            if exc.pgcode != errorcodes.UNDEFINED_COLUMN or attempted_drop_missing:
                raise
            existing = _existing_report_columns(conn)
            dropped = [c for c in columns if c not in existing]
            if not dropped:
                raise
            attempted_drop_missing = True
            payload = {k: v for k, v in payload.items() if k in existing}
            logger.warning(
                "Dropped unknown DB columns during update_report(id=%s): %s",
                report_id,
                ", ".join(sorted(dropped)),
            )


def list_reports(conn: psycopg2.extensions.connection, limit: int = 50) -> List[Dict[str, Any]]:
    """Return the most recent N reports."""
    sql = """
    SELECT * FROM inspection_reports
    ORDER BY created_at DESC NULLS LAST, id DESC
    LIMIT %s
    """
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, (limit,))
        rows = cur.fetchall()
    return [dict(r) for r in rows]


def get_report(conn: psycopg2.extensions.connection, report_id: int) -> Optional[Dict[str, Any]]:
    """Return a single report by id, or None."""
    sql = "SELECT * FROM inspection_reports WHERE id = %s"
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, (report_id,))
        row = cur.fetchone()
    return dict(row) if row else None


def get_filtered_reports(
    conn: psycopg2.extensions.connection,
    *,
    category: str | None = None,
    decision: str | None = None,
    limit: int = 5000,
    offset: int = 0,
    include_full: bool = False,
) -> List[Dict[str, Any]]:
    """Fetch reports with optional category/decision filters."""
    clauses = ["1=1"]
    params: List[Any] = []

    if category:
        clauses.append("category = %s")
        params.append(category)
    if decision:
        clauses.append("ad_decision = %s")
        params.append(decision)

    params.extend([limit, offset])
    select_sql = "*"
    if not include_full:
        select_sql = """
            id,
            dataset,
            category,
            line,
            ad_score,
            ad_decision,
            has_defect,
            region,
            area_ratio,
            confidence,
            image_path,
            heatmap_path,
            mask_path,
            overlay_path,
            ad_inference_duration,
            is_anomaly_llm,
            llm_report,
            llm_summary,
            llm_inference_duration,
            pipeline_status,
            pipeline_stage,
            pipeline_error,
            applied_policy,
            created_at
        """

    sql = f"""
    SELECT {select_sql} FROM inspection_reports
    WHERE {' AND '.join(clauses)}
    ORDER BY created_at DESC NULLS LAST, id DESC
    LIMIT %s OFFSET %s
    """
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, tuple(params))
        rows = cur.fetchall()
    return [dict(r) for r in rows]


def count_filtered_reports(
    conn: psycopg2.extensions.connection,
    *,
    category: str | None = None,
    decision: str | None = None,
) -> int:
    """Count reports with the same filters used by get_filtered_reports."""
    clauses = ["1=1"]
    params: List[Any] = []

    if category:
        clauses.append("category = %s")
        params.append(category)
    if decision:
        clauses.append("ad_decision = %s")
        params.append(decision)

    sql = f"""
    SELECT COUNT(*) AS n
    FROM inspection_reports
    WHERE {' AND '.join(clauses)}
    """
    with conn.cursor() as cur:
        cur.execute(sql, tuple(params))
        row = cur.fetchone()
    return int(row[0]) if row else 0


def has_ingest_source_path(conn: psycopg2.extensions.connection, source_path: str) -> bool:
    """Return True when a report with the same incoming source path already exists."""
    if not source_path:
        return False
    sql = "SELECT 1 FROM inspection_reports WHERE ingest_source_path = %s LIMIT 1"
    with conn.cursor() as cur:
        cur.execute(sql, (source_path,))
        row = cur.fetchone()
    return bool(row)


def get_category_policy(conn: psycopg2.extensions.connection, category: str) -> Dict[str, Any]:
    """Load per-category threshold policy with defaults."""
    sql = """
    SELECT
      t_low, t_high, review_band, reliability,
      ad_weight, location_mode, min_location_confidence, use_bbox
    FROM category_metadata
    WHERE category = %s
    """
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (category,))
            row = cur.fetchone()
        if row:
            policy = dict(row)
            out: Dict[str, Any] = {
                "t_low": float(policy.get("t_low", DEFAULT_POLICY["t_low"])),
                "t_high": float(policy.get("t_high", DEFAULT_POLICY["t_high"])),
                "source": "category_metadata",
            }
            review_band = policy.get("review_band")
            if review_band is not None:
                try:
                    out["review_band"] = float(review_band)
                except (TypeError, ValueError):
                    pass
            reliability = policy.get("reliability")
            if isinstance(reliability, str) and reliability.strip():
                out["reliability"] = reliability.strip().lower()
            ad_weight = policy.get("ad_weight")
            if ad_weight is not None:
                try:
                    out["ad_weight"] = float(ad_weight)
                except (TypeError, ValueError):
                    pass
            location_mode = policy.get("location_mode")
            if isinstance(location_mode, str) and location_mode.strip():
                out["location_mode"] = location_mode.strip().lower()
            min_location_conf = policy.get("min_location_confidence")
            if min_location_conf is not None:
                try:
                    out["min_location_confidence"] = float(min_location_conf)
                except (TypeError, ValueError):
                    pass
            use_bbox = policy.get("use_bbox")
            if isinstance(use_bbox, bool):
                out["use_bbox"] = use_bbox
            return out
    except Exception as exc:
        logger.warning("Failed to load category policy for %s: %s", category, exc)
    return {
        **dict(DEFAULT_POLICY),
        "source": "default",
    }


def upsert_category_metadata(
    conn: psycopg2.extensions.connection,
    rows: List[Dict[str, Any]],
) -> int:
    """Upsert category threshold metadata rows.

    Args:
        rows: List of dicts with at least ``category`` key.
              Supported keys: dataset, line, t_low, t_high, review_band, reliability,
              ad_weight, location_mode, min_location_confidence, use_bbox.

    Returns:
        Number of rows attempted for upsert.
    """
    if not rows:
        return 0

    sql = """
    INSERT INTO category_metadata (
      category, dataset, line, t_low, t_high, review_band, reliability,
      ad_weight, location_mode, min_location_confidence, use_bbox
    ) VALUES (
      %(category)s, %(dataset)s, %(line)s, %(t_low)s, %(t_high)s, %(review_band)s, %(reliability)s,
      %(ad_weight)s, %(location_mode)s, %(min_location_confidence)s, %(use_bbox)s
    )
    ON CONFLICT (category) DO UPDATE SET
      dataset = COALESCE(EXCLUDED.dataset, category_metadata.dataset),
      line = COALESCE(EXCLUDED.line, category_metadata.line),
      t_low = COALESCE(EXCLUDED.t_low, category_metadata.t_low),
      t_high = COALESCE(EXCLUDED.t_high, category_metadata.t_high),
      review_band = COALESCE(EXCLUDED.review_band, category_metadata.review_band),
      reliability = COALESCE(EXCLUDED.reliability, category_metadata.reliability),
      ad_weight = COALESCE(EXCLUDED.ad_weight, category_metadata.ad_weight),
      location_mode = COALESCE(EXCLUDED.location_mode, category_metadata.location_mode),
      min_location_confidence = COALESCE(EXCLUDED.min_location_confidence, category_metadata.min_location_confidence),
      use_bbox = COALESCE(EXCLUDED.use_bbox, category_metadata.use_bbox)
    """

    payload_rows: List[Dict[str, Any]] = []
    for row in rows:
        category = str(row.get("category", "")).strip()
        if not category:
            continue
        payload_rows.append(
            {
                "category": category,
                "dataset": row.get("dataset"),
                "line": row.get("line"),
                "t_low": row.get("t_low"),
                "t_high": row.get("t_high"),
                "review_band": row.get("review_band"),
                "reliability": row.get("reliability"),
                "ad_weight": row.get("ad_weight"),
                "location_mode": row.get("location_mode"),
                "min_location_confidence": row.get("min_location_confidence"),
                "use_bbox": row.get("use_bbox"),
            }
        )

    if not payload_rows:
        return 0

    with conn.cursor() as cur:
        cur.executemany(sql, payload_rows)
    conn.commit()
    return len(payload_rows)


def mark_stale_processing_reports(
    conn: psycopg2.extensions.connection,
    *,
    stale_seconds: int,
    limit: int = 1000,
) -> int:
    """Mark long-running processing rows as failed to avoid infinite UI pending state."""
    stale_seconds = int(stale_seconds)
    if stale_seconds <= 0:
        return 0

    sql = """
    WITH stale AS (
      SELECT id
      FROM inspection_reports
      WHERE pipeline_status = 'processing'
        AND COALESCE(updated_at, created_at) < (CURRENT_TIMESTAMP - (%s * INTERVAL '1 second'))
      ORDER BY COALESCE(updated_at, created_at) ASC
      LIMIT %s
    )
    UPDATE inspection_reports r
       SET pipeline_status = 'failed',
           pipeline_stage = 'timeout',
           pipeline_error = COALESCE(NULLIF(r.pipeline_error, ''), %s),
           ad_decision = COALESCE(NULLIF(r.ad_decision, ''), 'review_needed'),
           updated_at = CURRENT_TIMESTAMP
      FROM stale
     WHERE r.id = stale.id
    RETURNING r.id
    """
    timeout_msg = f"PIPELINE TIMEOUT: processing exceeded {stale_seconds}s"
    with conn.cursor() as cur:
        cur.execute(sql, (stale_seconds, int(limit), timeout_msg))
        rows = cur.fetchall()
    conn.commit()
    return len(rows)
