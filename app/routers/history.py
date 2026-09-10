"""历史统计 API。"""
from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter(prefix="/api/history", tags=["history"])


@router.get("/summary")
def summary(request: Request):
    """历史统计：样本总数、故障类型分布、场景分布、告警数、最近样本。"""
    return request.app.state.db.stats()
