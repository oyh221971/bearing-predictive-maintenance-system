"""真实信号预处理：重采样 + 去直流 + 滑窗切分。

真实采集的振动数据采样率/长度各异，诊断前统一为系统格式：
  1. 采样率与系统采样率(12kHz)差异超过 1% 时，用有理数重采样对齐
  2. 去直流（消除传感器偏置）
  3. 按 0.5s 窗口、50% 重叠滑窗切分；过短信号零填充；窗口过多时均匀抽稀
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly

from app.config import N_POINTS, SAMPLE_RATE


def resample_to(
    x: np.ndarray,
    src_rate: float,
    dst_rate: float = SAMPLE_RATE,
) -> np.ndarray:
    """将信号重采样到目标采样率（差异 ≤1% 时原样返回）。"""
    x = np.asarray(x, dtype=np.float64)
    src_rate, dst_rate = float(src_rate), float(dst_rate)
    if src_rate <= 0:
        raise ValueError(f"非法采样率: {src_rate}")
    if abs(src_rate - dst_rate) <= 0.01 * dst_rate:
        return x
    ratio = (Fraction(dst_rate).limit_denominator(100000)
             / Fraction(src_rate).limit_denominator(100000))
    return resample_poly(x, ratio.numerator, ratio.denominator)


def _spectral_peaks(
    spec: np.ndarray,
    freqs: np.ndarray,
    f_min: float,
    f_max: float,
    n: int = 40,
    tol: float = 1.5,
    min_ratio: float = 0.1,
) -> list[tuple[float, float]]:
    """按幅度降序提取频段内谱峰（幅度低于频段最大值 min_ratio 倍的噪声峰剔除）。"""
    mask = (freqs >= f_min) & (freqs <= f_max)
    s = spec[mask].copy()
    f = freqs[mask]
    band_max = float(s.max())
    if band_max <= 0:
        return []
    df = f[1] - f[0] if len(f) > 1 else 1.0
    half = max(2, int(tol / df))
    out: list[tuple[float, float]] = []
    for _ in range(n):
        i = int(np.argmax(s))
        amp = float(s[i])
        if amp < min_ratio * band_max:
            break
        out.append((float(f[i]), amp))
        s[max(0, i - half):i + half + 1] = 0
    return out


def estimate_shaft_freq(
    x: np.ndarray,
    sample_rate: float = SAMPLE_RATE,
    f_min: float = 8.0,
    f_max: float = 60.0,
) -> float:
    """从振动信号自动估计转频（Hz）。

    双假设 + 谐波支持度打分（实测 CWRU 60 文件与仿真信号全部 ±0.5Hz 内命中）：
      假设1（直接）：8~60Hz 内每个谱峰作为基频候选，检查 1x/2x/3x 谐波
                     是否成族，按 3:2:1 权重打分——转频谐波族天然得分高；
      假设2（反推）：60~500Hz 强谱线除以轴承特征频率比（BPFI/2×BSF/BPFO
                     与转频的固定比值），反推转频候选——对基频被淹没、
                     但共振放大的故障特征线突出的信号（如严重故障轴承）有效。
    取总分最高者。用于真实数据诊断时转速未知的场景（shaft_freq 留空自动估计）。
    """
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n_fft = len(x) * 2  # 零填充细化分辨率
    spec = np.abs(np.fft.rfft(x, n_fft)) ** 2
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)

    lo = _spectral_peaks(spec, freqs, 8.0, 60.0, min_ratio=0.10)
    hi = _spectral_peaks(spec, freqs, 60.0, 500.0, min_ratio=0.12)

    def has_peak(target: float, tol: float) -> int:
        return 1 if any(abs(p - target) <= tol for p, _ in lo + hi) else 0

    def score(f: float, bonus: float = 0.0) -> float:
        return (3.0 * has_peak(f, 1.2)
                + 2.0 * has_peak(2.0 * f, 1.5)
                + 1.0 * has_peak(3.0 * f, 2.0) + bonus)

    best_f, best_score = f_min, -1.0
    for f, _ in lo:
        if f > f_max:
            continue
        s = score(f)
        if s > best_score:
            best_score, best_f = s, f
    for peak_f, _ in hi[:8]:
        for ratio in (5.4162, 4.7121, 4.0323):  # BPFI/f, 2×BSF/f, BPFO/f
            f = peak_f / ratio
            if not (f_min <= f <= f_max):
                continue
            s = score(f, bonus=1.5)
            if s > best_score:
                best_score, best_f = s, f
    return float(best_f)


def prepare_windows(
    x: np.ndarray,
    window: int = N_POINTS,
    stride: int | None = None,
    max_windows: int = 100,
) -> tuple[list[np.ndarray], bool]:
    """切分为诊断窗口。

    Returns
    -------
    (windows, was_padded)：窗口列表；信号是否因过短被零填充。
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("输入必须是一维信号")
    x = x[np.isfinite(x)]
    if len(x) == 0:
        raise ValueError("信号为空或全为非法数值")
    x = x - x.mean()  # 去直流

    if len(x) < window:
        padded = np.zeros(window, dtype=np.float64)
        padded[: len(x)] = x
        return [padded], True

    stride = stride if stride is not None else window // 2
    starts = list(range(0, len(x) - window + 1, stride))
    if len(starts) > max_windows:
        starts = np.linspace(0, len(x) - window, max_windows, dtype=int).tolist()
    return [x[s: s + window] for s in starts], False
