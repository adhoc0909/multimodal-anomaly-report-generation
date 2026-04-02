"""GET /reports 엔드포인트."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

import src.storage.pg as pg

from ..config import DATABASE_URL

router = APIRouter()


@router.get("/reports")
async def get_reports(
    category: Optional[str] = Query(None),
    decision: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(500, ge=1, le=5000),
    limit: Optional[int] = Query(None, ge=1, le=5000),
    offset: Optional[int] = Query(None, ge=0),
    include_full: bool = Query(False),
):
    """Fetch reports in the same shape expected by current frontend."""
    if limit is not None or offset is not None:
        effective_limit = int(limit or page_size)
        effective_offset = int(offset or 0)
        effective_page = (effective_offset // max(1, effective_limit)) + 1
    else:
        effective_limit = int(page_size)
        effective_offset = (page - 1) * page_size
        effective_page = int(page)

    conn = pg.connect_fast(DATABASE_URL)
    try:
        total = pg.count_filtered_reports(
            conn,
            category=category,
            decision=decision,
        )
        reports = pg.get_filtered_reports(
            conn,
            category=category,
            decision=decision,
            limit=effective_limit,
            offset=effective_offset,
            include_full=include_full,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail="데이터 조회 중 오류가 발생했습니다.") from exc
    finally:
        conn.close()

    return {
        "page": effective_page,
        "page_size": effective_limit,
        "offset": effective_offset,
        "total_count": total,
        "total": total,
        "items": reports,
    }
