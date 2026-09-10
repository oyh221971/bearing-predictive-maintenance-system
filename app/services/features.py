"""特征提取模块：时域统计特征 + 频域特征 + 包络谱故障特征。

包络谱（Envelope Spectrum）是轴承诊断的经典方法：
    振动信号 → 共振频带带通滤波 → Hilbert 变换取包络 → FFT
故障冲击在包络谱的故障特征频率处呈现明显谱峰，可用于故障定位。

提取 20 维特征，FEATURE_NAMES 的顺序即特征向量顺序，
训练与推理共用，保证一致。
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal
from scipy import stats

from app.config import BAND_HIGH, BAND_LOW, SAMPLE_RATE
from app.services.signal import fault_frequencies

FEATURE_NAMES = [
    # 时域（9 维）
    "rms", "peak", "p2p", "crest_factor", "kurtosis", "skewness",
    "shape_factor", "impulse_factor", "margin_factor",
    # 频域（4 维）
    "spec_centroid", "spec_spread", "dominant_freq", "band_power_ratio",
    # 包络谱故障特征（7 维）
    "env_bpfo", "env_bpfo2", "env_bpfi", "env_bpfi2",
    "env_bsf2", "env_bsf4", "env_max",
]


def _time_domain_features(x: np.ndarray) -> dict[str, float]:
    """时域统计特征：反映振动幅值、冲击性与波形形状。"""
    abs_x = np.abs(x)
    rms = float(np.sqrt(np.mean(x ** 2)))
    peak = float(np.max(abs_x))
    p2p = float(np.ptp(x))
    abs_mean = float(np.mean(abs_x))
    sqrt_mean = float(np.mean(np.sqrt(abs_x)))
    return {
        "rms": rms,
        "peak": peak,
        "p2p": p2p,
        "crest_factor": peak / rms if rms > 1e-12 else 0.0,
        "kurtosis": float(stats.kurtosis(x, fisher=True)),   # 超额峰度，冲击信号显著增大
        "skewness": float(stats.skew(x)),
        "shape_factor": rms / abs_mean if abs_mean > 1e-12 else 0.0,
        "impulse_factor": peak / abs_mean if abs_mean > 1e-12 else 0.0,
        "margin_factor": peak / (sqrt_mean ** 2) if sqrt_mean > 1e-12 else 0.0,
    }


def _frequency_domain_features(x: np.ndarray, sample_rate: float) -> dict[str, float]:
    """频域统计特征：频谱重心、散布、主频、共振频带能量占比。"""
    n = len(x)
    spec = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    total = float(spec.sum())
    if total <= 1e-30:
        return {"spec_centroid": 0.0, "spec_spread": 0.0,
                "dominant_freq": 0.0, "band_power_ratio": 0.0}
    centroid = float(np.sum(freqs * spec) / total)
    spread = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * spec) / total))
    dominant = float(freqs[int(np.argmax(spec))])
    band_mask = (freqs >= BAND_LOW) & (freqs <= BAND_HIGH)
    band_ratio = float(spec[band_mask].sum() / total)
    return {"spec_centroid": centroid, "spec_spread": spread,
            "dominant_freq": dominant, "band_power_ratio": band_ratio}


def envelope_spectrum(
    x: np.ndarray, sample_rate: float = SAMPLE_RATE
) -> tuple[np.ndarray, np.ndarray]:
    """计算归一化包络谱。

    流程：共振频带带通滤波 → Hilbert 包络 → 去直流 → FFT → 按总和归一化。

    Returns
    -------
    (freqs, env_spec_norm)：包络谱频率轴与归一化幅值。
    """
    x = np.asarray(x, dtype=np.float64)
    sos = sp_signal.butter(4, [BAND_LOW, BAND_HIGH], btype="bandpass",
                           fs=sample_rate, output="sos")
    banded = sp_signal.sosfiltfilt(sos, x)
    envelope = np.abs(sp_signal.hilbert(banded))
    envelope = envelope - envelope.mean()
    env_spec = np.abs(np.fft.rfft(envelope))
    env_freqs = np.fft.rfftfreq(len(x), d=1.0 / sample_rate)
    total = float(env_spec.sum())
    if total <= 1e-30:
        return env_freqs, np.zeros_like(env_spec)
    return env_freqs, env_spec / total


def _envelope_features(x: np.ndarray, sample_rate: float,
                       shaft_freq: float) -> dict[str, float]:
    """包络谱故障特征：故障特征频率及其谐波处的归一化谱峰幅值。"""
    env_freqs, env_spec = envelope_spectrum(x, sample_rate)
    if float(env_spec.sum()) <= 1e-30:
        return {k: 0.0 for k in (
            "env_bpfo", "env_bpfo2", "env_bpfi", "env_bpfi2",
            "env_bsf2", "env_bsf4", "env_max")}

    freqs = fault_frequencies(shaft_freq)
    targets = {
        "env_bpfo": freqs["bpfo"],
        "env_bpfo2": 2 * freqs["bpfo"],
        "env_bpfi": freqs["bpfi"],
        "env_bpfi2": 2 * freqs["bpfi"],
        "env_bsf2": 2 * freqs["bsf"],   # 滚动体每转碰两次内外圈，冲击频率为 2×BSF
        "env_bsf4": 4 * freqs["bsf"],
    }
    out: dict[str, float] = {}
    for name, f_target in targets.items():
        # 取目标频率最近谱线及左右各 2 条中的最大值（容忍频率分辨率偏差）
        bin_center = int(round(f_target * len(x) / sample_rate))
        lo, hi = max(bin_center - 2, 1), min(bin_center + 3, len(env_spec))
        out[name] = float(env_spec[lo:hi].max())
    out["env_max"] = float(env_spec[1:].max())
    return out


def extract_features(
    x: np.ndarray,
    sample_rate: float = SAMPLE_RATE,
    shaft_freq: float = 25.0,
) -> np.ndarray:
    """提取 20 维特征向量（顺序与 FEATURE_NAMES 一致）。

    Parameters
    ----------
    x           : 单通道振动加速度信号（一维数组）
    sample_rate : 采样频率 Hz
    shaft_freq  : 转频 Hz（用于计算故障特征频率位置）
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("输入必须是一维振动信号")
    if len(x) < 256:
        raise ValueError(f"信号过短: {len(x)} 点")
    feats: dict[str, float] = {}
    feats.update(_time_domain_features(x))
    feats.update(_frequency_domain_features(x, sample_rate))
    feats.update(_envelope_features(x, sample_rate, shaft_freq))
    return np.asarray([feats[name] for name in FEATURE_NAMES], dtype=np.float64)
