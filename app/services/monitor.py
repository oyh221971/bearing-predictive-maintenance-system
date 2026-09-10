"""实时监测引擎：后台线程周期性生成振动样本 → 特征提取 → 模型诊断 → 落库与告警。

场景说明
--------
normal      正常运转：无缺陷，仅转频谐波与噪声
inner       内圈故障：恒定中等严重度 + 缓慢波动
outer       外圈故障：恒定中等严重度 + 缓慢波动
ball        滚动体故障：恒定中等严重度 + 缓慢波动
degradation 全寿命退化：严重度按指数退化模型逐步增长，
            走到寿命终点后自动开启新一轮全寿命模拟
"""
from __future__ import annotations

import threading
import time

import numpy as np

from app.config import (
    ALERT_RUL_HOURS,
    DEGRADE_LIFE_HOURS_PER_TICK,
    FAULT_PROB_THRESHOLD,
    MONITOR_INTERVAL_SECONDS,
    NOMINAL_LIFE_HOURS,
    SAMPLE_RATE,
    SHAFT_FREQ,
)
from app.database import Database
from app.services import signal as sig
from app.services.fault_detector import DiagnosticModels
from app.services.features import envelope_spectrum

SCENARIOS = ("normal", "inner", "outer", "ball", "degradation")

# 恒定故障场景的基础严重度（退化场景由轨迹决定）
_BASE_SEVERITY = {"normal": 0.0, "inner": 0.50, "outer": 0.45, "ball": 0.40}


class MonitorEngine:
    """周期采样 + 推理 + 告警的监测引擎（后台守护线程）。"""

    def __init__(self, db: Database, models: DiagnosticModels):
        self._db = db
        self._models = models
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._rng = np.random.default_rng(1234)

        self.scenario = "normal"
        self.sample_count = 0
        self._ticks = 0
        self._last_status: dict | None = None
        self._last_spectrum: dict | None = None
        self._last_signal: np.ndarray | None = None
        self._shaft_freq = SHAFT_FREQ
        self._ema = {"rul": None, "severity": None, "hi": None}

        # 退化场景状态
        self._deg = {"t_frac": 0.0, "p": 1.8, "life": NOMINAL_LIFE_HOURS}

    # ---------- 生命周期 ----------
    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, scenario: str) -> None:
        if scenario not in SCENARIOS:
            raise ValueError(f"未知场景: {scenario}，可选 {SCENARIOS}")
        if scenario == "degradation":
            self._reset_degradation()
        self.scenario = scenario
        if not self.running:
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._worker, name="monitor-engine", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._thread = None

    def _reset_degradation(self) -> None:
        self._deg = {
            "t_frac": 0.0,
            "p": float(self._rng.uniform(1.5, 2.2)),
            "life": NOMINAL_LIFE_HOURS,
        }
        self._ema = {"rul": None, "severity": None, "hi": None}

    # ---------- 主循环 ----------
    def _worker(self) -> None:
        """先采样、后等待：线程一启动立即产出首个样本，缩短看板等待。"""
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception as exc:  # 单次失败不影响后续采样
                print(f"[monitor] 采样异常: {exc}")
            self._stop.wait(MONITOR_INTERVAL_SECONDS)

    def _tick(self) -> None:
        self._ticks += 1
        scenario = self.scenario

        # 1) 决定当前严重度
        if scenario == "degradation":
            fault_type = ("inner", "outer", "ball")[self._ticks % 3]
            step = DEGRADE_LIFE_HOURS_PER_TICK / self._deg["life"]
            self._deg["t_frac"] = min(self._deg["t_frac"] + step, 1.0)
            severity = float(self._deg["t_frac"] ** self._deg["p"])
        else:
            fault_type = scenario
            base = _BASE_SEVERITY[scenario]
            # 缓慢波动 + 小幅噪声，模拟负载变化
            severity = float(np.clip(
                base + 0.06 * np.sin(self._ticks * 0.25) +
                self._rng.normal(0.0, 0.015), 0.0, 0.97))

        # 2) 生成信号（转频缓慢波动）
        self._shaft_freq = SHAFT_FREQ * (1.0 + 0.04 * np.sin(self._ticks * 0.13))
        x = sig.simulate_signal(
            fault_type, severity, self._shaft_freq, rng=self._rng)
        self._last_signal = x

        # 3) 模型诊断
        res = self._models.predict(x, SAMPLE_RATE, self._shaft_freq)
        rul_raw = float(res["rul_hours"])
        sev_raw = float(res["severity"])
        hi_raw = float(np.clip(rul_raw / NOMINAL_LIFE_HOURS, 0.0, 1.0))

        # 指数滑动平均平滑，避免逐样本抖动
        rul = self._smooth("rul", rul_raw)
        severity = self._smooth("severity", sev_raw)
        health_index = self._smooth("hi", hi_raw)
        temperature = sig.bearing_temperature(severity / 100.0, rng=self._rng)

        # 4) 频谱（供前端图表）
        self._last_spectrum = self._compute_spectrum(x)

        # 5) 落库
        sample_id = self._db.insert_sample(
            scenario=scenario,
            fault_type=str(res["fault_type"]),
            confidence=float(res["confidence"]),
            severity=severity,
            rul_hours=rul,
            health_index=health_index,
            rms=float(np.sqrt(np.mean(x ** 2))),
            kurtosis=float(res["features"][4]),
            crest=float(res["features"][3]),
            temperature=temperature,
            probs=res["probs"],
        )
        self.sample_count += 1

        # 6) 告警判定与同步
        alerts = self._evaluate_alerts(res, rul)
        self._db.sync_alerts(alerts)

        # 7) 退化到寿命终点 → 重置并提示
        if scenario == "degradation" and self._deg["t_frac"] >= 1.0:
            self._db.sync_alerts([])  # 解除旧告警
            self._reset_degradation()
            self._db.sync_alerts([
                ("上一轮全寿命模拟结束，已开启新一轮全寿命退化模拟", "info")])

        # 8) 更新状态快照
        with self._lock:
            self._last_status = {
                "id": sample_id,
                "ts": time.time(),
                "scenario": scenario,
                "fault_type": res["fault_type"],
                "confidence": res["confidence"],
                "probs": res["probs"],
                "severity": severity,
                "rul_hours": rul,
                "health_index": health_index,
                "rms": float(np.sqrt(np.mean(x ** 2))),
                "kurtosis": float(res["features"][4]),
                "crest": float(res["features"][3]),
                "temperature": temperature,
            }

    # ---------- 内部工具 ----------
    def _smooth(self, key: str, value: float) -> float:
        alpha = 0.35
        prev = self._ema.get(key)
        if prev is None:
            self._ema[key] = value
            return value
        smoothed = alpha * value + (1 - alpha) * prev
        self._ema[key] = smoothed
        return smoothed

    def _evaluate_alerts(self, res: dict, rul: float) -> list[tuple[str, str]]:
        """生成当前应处于活跃状态的告警 (message, level) 列表。

        注意：message 必须是稳定文本（不含逐采样波动的置信度/数值），
        否则 sync_alerts 按消息去重会失效，产生重复告警。
        """
        alerts: list[tuple[str, str]] = []
        fault = res["fault_type"]
        conf = res["confidence"]
        if fault != "normal" and conf >= FAULT_PROB_THRESHOLD:
            zh = sig.FAULT_NAMES_ZH[fault]
            sev = res["severity"]
            if sev >= 70:
                level = "critical"
            elif sev >= 40:
                level = "warning"
            else:
                level = "info"
            alerts.append((f"检测到{zh}，请关注故障发展趋势", level))
        if rul < ALERT_RUL_HOURS:
            level = "critical" if rul < 50 else "warning"
            alerts.append(("剩余寿命不足，建议尽快安排检修维护", level))
        return alerts

    def _compute_spectrum(self, x: np.ndarray) -> dict:
        """计算波形（下采样）、频谱与包络谱（≤800 点供前端绘图）。"""
        n = len(x)

        def _downsample(f: np.ndarray, a: np.ndarray, max_pts: int = 800):
            stride = max(1, int(np.ceil(len(f) / max_pts)))
            return f[::stride].tolist(), a[::stride].tolist()

        wave_t = np.arange(n) / SAMPLE_RATE * 1000.0  # ms
        freq = np.fft.rfftfreq(n, d=1.0 / SAMPLE_RATE)
        amp = np.abs(np.fft.rfft(x)) ** 2
        amp = amp / (amp.max() + 1e-30)
        env_freq, env_amp = envelope_spectrum(x, SAMPLE_RATE)
        env_amp = env_amp / (env_amp.max() + 1e-30)

        freqs = sig.fault_frequencies(self._shaft_freq)
        return {
            "wave_t": _downsample(wave_t, x, 600)[0],
            "wave": _downsample(wave_t, x, 600)[1],
            "freq": _downsample(freq, amp)[0],
            "amp": _downsample(freq, amp)[1],
            "env_freq": _downsample(env_freq, env_amp)[0],
            "env_amp": _downsample(env_freq, env_amp)[1],
            "markers": {
                "bpfo": freqs["bpfo"], "bpfi": freqs["bpfi"],
                "bsf": 2 * freqs["bsf"], "ftf": freqs["ftf"],
                "shaft": [self._shaft_freq, 2 * self._shaft_freq,
                          3 * self._shaft_freq],
            },
            "shaft_freq": self._shaft_freq,
        }

    # ---------- 状态查询 ----------
    def status(self) -> dict:
        with self._lock:
            last = dict(self._last_status) if self._last_status else None
        return {
            "running": self.running,
            "scenario": self.scenario,
            "sample_count": self.sample_count,
            "interval_seconds": MONITOR_INTERVAL_SECONDS,
            "models_ready": self._models.ready,
            "last": last,
            "alerts": self._db.active_alerts(),
        }

    def spectrum(self) -> dict | None:
        return self._last_spectrum

    def current_signal(self) -> np.ndarray | None:
        return self._last_signal
