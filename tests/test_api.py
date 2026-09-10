"""API 端到端测试（需先完成模型训练）。"""
from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from app.config import MODEL_DIR
from app.main import app
from app.services import signal as sig

_MODEL_FILES = ("fault_classifier", "severity_regressor",
                "rul_regressor", "label_encoder")
pytestmark = pytest.mark.skipif(
    not all((MODEL_DIR / f"{n}.joblib").exists() for n in _MODEL_FILES),
    reason="模型未训练，先运行 python -m training.train_models")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["models_ready"] is True


def test_models_info(client):
    r = client.get("/api/models/info")
    assert r.status_code == 200
    body = r.json()
    assert body["ready"] is True
    assert body["metrics"]["classification_accuracy"] > 0.95
    assert body["metrics"]["severity_mae"] < 0.1
    assert len(body["feature_names"]) == 20


def test_diagnose_normal(client):
    x = sig.simulate_signal("normal", 0.0, seed=31)
    r = client.post("/api/diagnose", json={
        "samples": x.tolist(), "sample_rate": 12000, "shaft_freq": 25})
    assert r.status_code == 200
    body = r.json()
    assert body["fault_type"] == "normal"
    assert body["fault_name_zh"] == "正常"
    assert body["severity"] < 15


def test_diagnose_inner_fault(client):
    x = sig.simulate_signal("inner", 0.8, seed=37)
    r = client.post("/api/diagnose", json={
        "samples": x.tolist(), "sample_rate": 12000, "shaft_freq": 25})
    assert r.status_code == 200
    body = r.json()
    assert body["fault_type"] == "inner"
    assert body["fault_name_zh"] == "内圈故障"
    assert body["severity"] > 40
    assert body["confidence"] > 0.8


def test_diagnose_rejects_short_signal(client):
    r = client.post("/api/diagnose", json={"samples": [0.0] * 100})
    assert r.status_code == 400


def test_diagnose_long_signal_multi_window(client):
    """长信号自动滑窗聚合：4 段外圈故障拼接（24000 点）→ 7 个窗口 → 外圈故障。"""
    import numpy as np

    parts = [sig.simulate_signal("outer", 0.8, seed=i) for i in range(4)]
    long_sig = np.concatenate(parts)
    r = client.post("/api/diagnose", json={
        "samples": long_sig.tolist(), "sample_rate": 12000, "shaft_freq": 25})
    assert r.status_code == 200
    body = r.json()
    assert body["fault_type"] == "outer"
    assert body["n_windows"] == 7
    assert len(body["windows"]) == 7
    assert body["confidence"] > 0.9


def test_diagnose_resamples_48k_signal(client):
    """48kHz 采集信号应被重采样到 12kHz 后正确诊断。"""
    x = sig.simulate_signal("inner", 0.8, sample_rate=48000, n_points=24000,
                            seed=3)
    r = client.post("/api/diagnose", json={
        "samples": x.tolist(), "sample_rate": 48000, "shaft_freq": 25})
    assert r.status_code == 200
    body = r.json()
    assert body["fault_type"] == "inner"
    assert body["n_windows"] == 1


def test_monitor_lifecycle_and_alerts(client):
    # 停止默认场景 → 以内圈故障场景重启
    client.post("/api/monitor/stop")
    r = client.post("/api/monitor/start", json={"scenario": "inner"})
    assert r.status_code == 200 and r.json()["scenario"] == "inner"

    # 轮询等待场景切换后的新样本（引擎先采样后等待，几秒内必有数据）
    base_count = client.get("/api/monitor/status").json()["sample_count"]
    deadline = time.time() + 10.0
    status = None
    while time.time() < deadline:
        status = client.get("/api/monitor/status").json()
        if status["sample_count"] > base_count and status["last"]["scenario"] == "inner":
            break
        time.sleep(0.5)
    assert status is not None
    assert status["running"] is True
    assert status["sample_count"] >= 1
    assert status["last"] is not None
    assert status["last"]["scenario"] == "inner"

    # 轮询等待故障告警出现（告警去重：内圈告警应恰好 1 条）
    deadline = time.time() + 10.0
    while time.time() < deadline:
        status = client.get("/api/monitor/status").json()
        fault_alerts = [a for a in status["alerts"] if "内圈" in a["message"]]
        if len(fault_alerts) >= 1:
            break
        time.sleep(0.5)
    assert len(fault_alerts) == 1

    # 频谱接口
    spec = client.get("/api/monitor/spectrum").json()
    assert "wave" in spec and "freq" in spec and "env_amp" in spec
    assert spec["markers"]["bpfo"] > 0

    # 样本与历史接口
    samples = client.get("/api/monitor/samples?limit=5").json()["items"]
    assert len(samples) >= 1 and "probs" in samples[0]
    summary = client.get("/api/history/summary").json()
    assert summary["total_samples"] >= 1

    # 告警确认
    ack_id = fault_alerts[0]["id"]
    assert client.post(f"/api/monitor/alerts/{ack_id}/ack").status_code == 200

    client.post("/api/monitor/stop")
    assert client.get("/api/monitor/status").json()["running"] is False


def test_degradation_scenario(client):
    client.post("/api/monitor/start", json={"scenario": "degradation"})
    # 轮询等待退化场景样本
    deadline = time.time() + 10.0
    status = None
    while time.time() < deadline:
        status = client.get("/api/monitor/status").json()
        if status.get("last") is not None:
            break
        time.sleep(0.5)
    assert status is not None and status["scenario"] == "degradation"
    assert status["last"] is not None
    # 退化初期应接近健康状态（健康指数高）
    assert status["last"]["health_index"] > 0.5
    client.post("/api/monitor/stop")


def test_invalid_scenario_rejected(client):
    r = client.post("/api/monitor/start", json={"scenario": "bogus"})
    assert r.status_code == 400
