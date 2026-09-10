"""诊断与模型信息 API：上传振动信号做离线诊断，查看模型元信息与指标。"""
from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException, Request

from app.config import N_POINTS, SAMPLE_RATE
from app.schemas import DiagnoseRequest
from app.services import preprocess
from app.services.fault_detector import ModelUnavailable, aggregate_predictions
from app.services.signal import FAULT_NAMES_ZH

router = APIRouter(prefix="/api", tags=["diagnose"])


def _models(request: Request):
    return request.app.state.models


@router.post("/diagnose")
def diagnose(body: DiagnoseRequest, request: Request):
    """离线诊断：上传振动信号（一维浮点数组，任意长度/采样率）。

    内部流程：重采样到 12kHz → 去直流 → 0.5s 窗口/50% 重叠滑窗 →
    逐窗口模型诊断 → 概率均值投票聚合。

    返回：整体诊断（故障类型/置信度/严重度/RUL）+ 各窗口明细。
    """
    x = np.asarray(body.samples, dtype=np.float64)
    if len(x) < 512:
        raise HTTPException(status_code=400, detail="信号过短（至少 512 点）")
    try:
        xr = preprocess.resample_to(x, body.sample_rate)
        windows, padded = preprocess.prepare_windows(xr)
        # 转频：用户给定则用给定值，否则从信号频谱自动估计
        shaft_freq = (body.shaft_freq if body.shaft_freq is not None
                      else preprocess.estimate_shaft_freq(xr, SAMPLE_RATE))
        results = [
            _models(request).predict(w, SAMPLE_RATE, shaft_freq)
            for w in windows
        ]
    except ModelUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    agg = aggregate_predictions(results)
    agg["fault_name_zh"] = FAULT_NAMES_ZH.get(agg["fault_type"],
                                              agg["fault_type"])
    agg["n_windows"] = len(results)
    agg["padded"] = padded
    if body.shaft_freq is None:
        agg["estimated_shaft_freq"] = round(shaft_freq, 2)
    agg["windows"] = [
        {
            "index": i,
            "fault_type": r["fault_type"],
            "confidence": r["confidence"],
            "severity": r["severity"],
            "rul_hours": r["rul_hours"],
        }
        for i, r in enumerate(results)
    ]
    return agg


@router.get("/models/info")
def models_info(request: Request):
    models = _models(request)
    if not models.ready:
        return {"ready": False, "hint": "模型未训练，请运行 "
                "python -m training.generate_dataset && "
                "python -m training.train_models"}
    meta = models.meta
    return {"ready": True, **meta}


@router.get("/diagnose/demo")
def demo_signal(request: Request, fault_type: str = "inner"):
    """获取一段演示信号（来自 data/raw/demo_signals.npz）。"""
    from app.config import RAW_DIR

    path = RAW_DIR / "demo_signals.npz"
    if not path.exists():
        raise HTTPException(status_code=404, detail="演示信号尚未生成")
    data = np.load(path)
    if fault_type not in data.files:
        raise HTTPException(status_code=400,
                            detail=f"未知故障类型，可选 {list(data.files)}")
    x = data[fault_type][:N_POINTS]
    res = _models(request).predict(x)
    return {"fault_type_truth": fault_type,
            "fault_name_zh": FAULT_NAMES_ZH[fault_type],
            "samples": x.tolist(), **res}
