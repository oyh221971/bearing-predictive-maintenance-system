"""信号预处理（重采样/滑窗）单元测试。"""
from __future__ import annotations

import numpy as np
import pytest

from app.config import N_POINTS, SAMPLE_RATE
from app.services import preprocess
from app.services import signal as sig


def test_resample_48k_to_12k_quarters_length():
    x = sig.simulate_signal("inner", 0.7, sample_rate=48000, n_points=24000,
                            seed=1)
    out = preprocess.resample_to(x, 48000.0, SAMPLE_RATE)
    assert len(out) == pytest.approx(6000, abs=2)


def test_resample_same_rate_is_identity():
    x = sig.simulate_signal("normal", 0.0, seed=2)
    out = preprocess.resample_to(x, 12000.0, SAMPLE_RATE)
    assert out is x  # 采样率一致时原样返回，避免不必要拷贝


def test_resample_rejects_bad_rate():
    with pytest.raises(ValueError):
        preprocess.resample_to(np.zeros(100), 0.0)


def test_prepare_windows_overlap():
    """10000 点、6000 点窗口、50% 重叠 → 2 个窗口（注意输入已做去直流）。"""
    x = np.arange(10000, dtype=np.float64)
    windows, padded = preprocess.prepare_windows(x)
    assert not padded
    assert len(windows) == 2
    assert len(windows[0]) == N_POINTS
    baseline = x - x.mean()
    assert windows[0][0] == pytest.approx(baseline[0])
    assert windows[1][0] == pytest.approx(baseline[N_POINTS // 2])


def test_prepare_windows_pads_short_signal():
    x = np.random.default_rng(0).normal(size=1000)
    windows, padded = preprocess.prepare_windows(x)
    assert padded is True
    assert len(windows) == 1 and len(windows[0]) == N_POINTS


def test_prepare_windows_removes_dc_and_nan():
    rng = np.random.default_rng(0)
    x = rng.normal(size=8000)
    x[10] = np.nan
    x[11] = np.inf
    x += 100.0  # 直流偏置
    windows, _ = preprocess.prepare_windows(x)
    assert all(np.isfinite(w).all() for w in windows)
    # 100 的直流偏置应被去除，窗口均值应接近 0
    assert abs(windows[0].mean()) < 0.1


def test_estimate_shaft_freq_synthetic():
    """转频估计应能从谐波族定位到真实转频（±0.5Hz）。"""
    for freq in (22.0, 25.0, 28.0):
        x = sig.simulate_signal("normal", 0.0, shaft_freq=freq, seed=1)
        est = preprocess.estimate_shaft_freq(x)
        assert abs(est - freq) < 0.5, f"真实 {freq}Hz 估计为 {est}Hz"


def test_estimate_shaft_freq_real_cwru():
    """CWRU 1797rpm 文件应估计到 ~29.95Hz（无数据则跳过）。"""
    from pathlib import Path
    path = Path("data/cwru/105_0.mat")
    if not path.exists():
        pytest.skip("CWRU 数据未下载")
    from scipy.io import loadmat
    mat = loadmat(str(path))
    x = np.asarray(mat["X105_DE_time"]).ravel()
    est = preprocess.estimate_shaft_freq(x[6000:12000])
    assert abs(est - 29.95) < 0.5, f"估计为 {est}Hz"
