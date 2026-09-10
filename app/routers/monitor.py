"""监测相关 API：启停引擎、实时状态、样本、频谱、告警。"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.schemas import MonitorStart

router = APIRouter(prefix="/api/monitor", tags=["monitor"])


def _engine(request: Request):
    return request.app.state.engine


def _db(request: Request):
    return request.app.state.db


@router.post("/start")
def start(body: MonitorStart, request: Request):
    engine = _engine(request)
    try:
        engine.start(body.scenario)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"running": engine.running, "scenario": engine.scenario}


@router.post("/stop")
def stop(request: Request):
    engine = _engine(request)
    engine.stop()
    return {"running": engine.running}


@router.get("/status")
def status(request: Request):
    return _engine(request).status()


@router.get("/samples")
def samples(request: Request, limit: int = 100, offset: int = 0):
    return {"items": _db(request).recent_samples(limit, offset)}


@router.get("/spectrum")
def spectrum(request: Request):
    spec = _engine(request).spectrum()
    if spec is None:
        raise HTTPException(status_code=404, detail="尚无频谱数据，请等待首个采样周期")
    return spec


@router.get("/alerts")
def alerts(request: Request, limit: int = 50, active_only: bool = False):
    items = (_db(request).active_alerts()
             if active_only else _db(request).recent_alerts(limit))
    return {"items": items}


@router.post("/alerts/{alert_id}/ack")
def ack_alert(alert_id: int, request: Request):
    if not _db(request).ack_alert(alert_id):
        raise HTTPException(status_code=404, detail="告警不存在或已确认")
    return {"acked": True, "id": alert_id}
