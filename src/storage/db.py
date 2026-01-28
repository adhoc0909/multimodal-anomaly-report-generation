from __future__ import annotations
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional

SCHEMA_SQL = '''
CREATE TABLE IF NOT EXISTS reports (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp TEXT NOT NULL,
  image TEXT NOT NULL,
  decision TEXT NOT NULL,
  confidence REAL NOT NULL,
  location TEXT,
  area_ratio REAL,
  shape TEXT,
  severity TEXT,
  summary TEXT,
  impact TEXT,
  recommendation TEXT,
  heatmap_path TEXT,
  overlay_path TEXT
);
'''

def connect(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute(SCHEMA_SQL)
    conn.commit()
    return conn

def insert_report(conn: sqlite3.Connection, report: Dict[str, Any]) -> int:
    defect = report.get("defect", {}) or {}
    artifacts = report.get("artifacts", {}) or {}
    cur = conn.cursor()
    cur.execute(
        '''INSERT INTO reports
           (timestamp,image,decision,confidence,location,area_ratio,shape,severity,summary,impact,recommendation,heatmap_path,overlay_path)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)''',
        (
            report["timestamp"], report["image"], report["decision"], float(report["confidence"]),
            defect.get("location"), float(defect.get("area_ratio", 0.0)), defect.get("shape"), defect.get("severity"),
            report.get("summary"), report.get("impact"), report.get("recommendation"),
            artifacts.get("heatmap_path"), artifacts.get("overlay_path")
        )
    )
    conn.commit()
    return int(cur.lastrowid)

def list_reports(conn: sqlite3.Connection, limit: int = 50) -> List[Dict[str, Any]]:
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT id,timestamp,image,decision,confidence,location,area_ratio,shape,severity,summary,impact,recommendation,heatmap_path,overlay_path "
        "FROM reports ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    cols = ["id","timestamp","image","decision","confidence","location","area_ratio","shape","severity","summary","impact","recommendation","heatmap_path","overlay_path"]
    return [dict(zip(cols, r)) for r in rows]

def get_report(conn: sqlite3.Connection, report_id: int) -> Optional[Dict[str, Any]]:
    cur = conn.cursor()
    r = cur.execute(
        "SELECT id,timestamp,image,decision,confidence,location,area_ratio,shape,severity,summary,impact,recommendation,heatmap_path,overlay_path "
        "FROM reports WHERE id=?", (report_id,)
    ).fetchone()
    if not r:
        return None
    cols = ["id","timestamp","image","decision","confidence","location","area_ratio","shape","severity","summary","impact","recommendation","heatmap_path","overlay_path"]
    d = dict(zip(cols, r))
    d["defect"] = {"location": d.pop("location"), "area_ratio": d.pop("area_ratio"), "shape": d.pop("shape"), "severity": d.pop("severity")}
    d["artifacts"] = {"heatmap_path": d.pop("heatmap_path"), "overlay_path": d.pop("overlay_path")}
    return d
