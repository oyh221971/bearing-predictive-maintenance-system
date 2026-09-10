"""轴承振动信号物理仿真模块。

基于滚动轴承运动学与冲击-共振响应模型，合成四种状态的振动加速度信号：
  - normal：正常（轴转频谐波 + 宽带噪声）
  - inner ：内圈故障（冲击以 BPFI 重复，幅值随轴转频调制）
  - outer ：外圈故障（冲击以 BPFO 重复）
  - ball  ：滚动体故障（冲击以 2×BSF 重复，幅值随保持架转频调制）

并提供全寿命退化轨迹模拟，供剩余寿命(RUL)模型训练使用。

故障特征频率公式（滚动轴承经典运动学）：
    BPFO = (n/2)·fr·(1 - d/D·cosθ)   外圈
    BPFI = (n/2)·fr·(1 + d/D·cosθ)   内圈
    BSF  = (D/2d)·fr·(1 - (d/D·cosθ)²)  滚动体自转
    FTF  = (fr/2)·(1 - d/D·cosθ)     保持架
其中 n=滚动体数, fr=转频, d=滚动体直径, D=节圆直径, θ=接触角。
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from app.config import (
    BEARING,
    N_POINTS,
    RESONANCE_DECAY,
    RESONANCE_FREQ,
    SAMPLE_RATE,
    SHAFT_FREQ,
)

FAULT_TYPES = ("normal", "inner", "outer", "ball")

FAULT_NAMES_ZH = {
    "normal": "正常",
    "inner": "内圈故障",
    "outer": "外圈故障",
    "ball": "滚动体故障",
}


def fault_frequencies(shaft_freq: float = SHAFT_FREQ) -> dict[str, float]:
    """按轴承几何参数计算各部件故障特征频率（Hz）。

    返回 dict：bpfo / bpfi / bsf / ftf。
    """
    n = BEARING["n_balls"]
    d = BEARING["ball_diameter_mm"]
    D = BEARING["pitch_diameter_mm"]
    theta = math.radians(BEARING["contact_angle_deg"])
    ratio = (d / D) * math.cos(theta)
    bpfo = (n / 2.0) * shaft_freq * (1.0 - ratio)
    bpfi = (n / 2.0) * shaft_freq * (1.0 + ratio)
    bsf = (D / (2.0 * d)) * shaft_freq * (1.0 - ratio ** 2)
    ftf = (shaft_freq / 2.0) * (1.0 - ratio)
    return {"bpfo": bpfo, "bpfi": bpfi, "bsf": bsf, "ftf": ftf}


def _fault_impulse_rate(fault_type: str, shaft_freq: float) -> float:
    """故障冲击的重复频率：内圈=BPFI，外圈=BPFO，滚动体=2×BSF。"""
    freqs = fault_frequencies(shaft_freq)
    return {
        "inner": freqs["bpfi"],
        "outer": freqs["bpfo"],
        "ball": 2.0 * freqs["bsf"],
    }[fault_type]


def simulate_signal(
    fault_type: str = "normal",
    severity: float = 0.0,
    shaft_freq: float = SHAFT_FREQ,
    sample_rate: float = SAMPLE_RATE,
    n_points: int = N_POINTS,
    resonance_freq: float = RESONANCE_FREQ,
    decay: float = RESONANCE_DECAY,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """合成一段轴承振动加速度信号。

    Parameters
    ----------
    fault_type : 故障类型 normal/inner/outer/ball
    severity   : 故障严重度 0.0~1.0（0 表示无缺陷）
    shaft_freq : 转频 Hz（默认 25 Hz，即 1500 rpm）
    其余为采样与冲击响应参数。

    模型：正常成分 = 转频谐波 + 白噪声；
          故障成分 = 以故障冲击频率重复的"冲击-共振衰减振荡"脉冲串，
          内圈/滚动体故障的冲击幅值还受到轴转频/保持架转频调制。
    """
    fault_type = fault_type.lower()
    if fault_type not in FAULT_TYPES:
        raise ValueError(f"未知故障类型: {fault_type}")
    if rng is None:
        rng = np.random.default_rng(seed)

    t = np.arange(n_points) / sample_rate
    x = np.zeros(n_points, dtype=np.float64)

    # 1) 轴转频及其谐波（设备正常运转的基础振动；劣化加剧时轻微增大，模拟松动）
    for k in range(1, 4):
        x += ((0.30 / k) * (1.0 + 0.4 * severity)
              * np.sin(2 * np.pi * k * shaft_freq * t
                       + rng.uniform(0.0, 2 * np.pi)))

    # 2) 宽带背景噪声（随劣化增大，模拟润滑不良/磨损引起的本底振动抬升）
    x += rng.normal(0.0, 0.08 * (1.0 + 0.6 * severity), n_points)

    # 3) 故障冲击串
    if fault_type != "normal" and severity > 0.0:
        f_fault = _fault_impulse_rate(fault_type, shaft_freq)
        period = 1.0 / f_fault
        amp = 1.1 * severity + 0.4 * severity ** 2    # 冲击幅值随严重度增长
        freqs = fault_frequencies(shaft_freq)
        mod_freq = {"inner": shaft_freq, "ball": freqs["ftf"]}.get(fault_type)
        ring_len = int(0.004 * sample_rate)           # 单个冲击响应的持续点数

        k_start = 0
        k_end = int(math.ceil(t[-1] / period))
        for k in range(k_start, k_end + 1):
            # 冲击到达时刻（1.5% 随机抖动，模拟滚动体滑移）
            tk = k * period + rng.normal(0.0, 0.015 * period)
            idx = int(round(tk * sample_rate))
            if idx < 0 or idx >= n_points:   # 冲击落在窗口之外
                continue
            # 幅值调制：内圈缺陷进出载荷区 / 滚动体随保持架公转
            if mod_freq is not None:
                a = amp * (1.0 + 0.8 * math.cos(2 * np.pi * mod_freq * tk))
                a = max(a, 0.1 * amp)
            else:
                a = amp
            # 冲击激起的共振衰减振荡（单自由度欠阻尼响应）
            end = min(idx + ring_len, n_points)
            tau = np.arange(end - idx) / sample_rate
            x[idx:end] += a * np.exp(-decay * tau) * np.sin(
                2 * np.pi * resonance_freq * tau)

    return x


def bearing_temperature(severity: float, seed: int | None = None,
                        rng: np.random.Generator | None = None) -> float:
    """由故障严重度估计轴承温度（℃）：正常约 40℃，严重故障可达 85℃ 以上。"""
    if rng is None:
        rng = np.random.default_rng(seed)
    return float(40.0 + 45.0 * severity ** 1.5 + rng.normal(0.0, 0.6))


def degradation_trajectory(
    total_life_hours: float,
    n_steps: int,
    exponent_range: tuple[float, float] = (1.5, 2.2),
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """模拟一条全寿命退化轨迹。

    采用经典的指数退化模型：严重度 severity(t) = t^p，t∈[0,1] 为寿命比，
    p 随机取 1.5~2.2（初期缓慢劣化、后期加速）。健康指数 HI = 1 - severity。

    Returns
    -------
    (t_frac, severity, hi, rul_hours)：每个寿命步的四个数组。
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    t_frac = np.linspace(0.0, 1.0, n_steps)
    p = rng.uniform(*exponent_range)
    severity = t_frac ** p
    # 健康指数 = 1 - 严重度，叠加小幅观测噪声
    hi = np.clip(1.0 - severity + rng.normal(0.0, 0.01, n_steps), 0.0, 1.0)
    rul_hours = (1.0 - t_frac) * total_life_hours
    return t_frac, severity, hi, rul_hours


def simulate_degradation_run(
    fault_type: str,
    total_life_hours: float,
    n_steps: int,
    sample_rate: float = SAMPLE_RATE,
    n_points: int = N_POINTS,
    seed: int | None = None,
) -> dict:
    """模拟一次完整的退化运行：返回每个寿命步的信号与标签。"""
    rng = np.random.default_rng(seed)
    shaft_freq = SHAFT_FREQ * float(rng.uniform(0.9, 1.1))
    t_frac, severity, hi, rul = degradation_trajectory(
        total_life_hours, n_steps, rng=rng)
    signals = [
        simulate_signal(fault_type, float(s), shaft_freq, sample_rate,
                        n_points, rng=rng)
        for s in severity
    ]
    return {
        "fault_type": fault_type,
        "total_life_hours": total_life_hours,
        "shaft_freq": shaft_freq,
        "t_frac": t_frac,
        "severity": severity,
        "health_index": hi,
        "rul_hours": rul,
        "signals": np.asarray(signals),
    }
