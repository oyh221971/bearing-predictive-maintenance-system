"""信号仿真模块单元测试。"""
from __future__ import annotations

import numpy as np
import pytest

from app.config import N_POINTS, SAMPLE_RATE
from app.services import signal as sig


def test_fault_frequencies_match_theory():
    """SKF 6205 @25Hz 转频的特征频率应与经典公式一致。"""
    f = sig.fault_frequencies(25.0)
    assert f["bpfo"] == pytest.approx(89.62, rel=0.01)
    assert f["bpfi"] == pytest.approx(135.38, rel=0.01)
    assert f["bsf"] == pytest.approx(58.92, rel=0.01)
    assert f["ftf"] == pytest.approx(9.96, rel=0.01)


def test_simulate_signal_shape_and_dtype():
    for ft in sig.FAULT_TYPES:
        x = sig.simulate_signal(ft, 0.5 if ft != "normal" else 0.0, seed=1)
        assert x.shape == (N_POINTS,)
        assert np.isfinite(x).all()


def test_fault_raises_energy_and_grows_with_severity():
    rms = lambda x: float(np.sqrt(np.mean(x ** 2)))
    normal = sig.simulate_signal("normal", 0.0, seed=7)
    mild = sig.simulate_signal("inner", 0.3, seed=7)
    heavy = sig.simulate_signal("inner", 0.8, seed=7)
    assert rms(normal) < rms(mild) < rms(heavy)


def test_outer_fault_increases_impulse_indicators():
    """外圈故障冲击密集且幅值恒定，典型指标是波峰因子/脉冲因子显著增大、
    峰度小幅上升（与真实轴承外圈故障特性一致）。"""
    from app.services.features import FEATURE_NAMES, extract_features
    normal = sig.simulate_signal("normal", 0.0, seed=3)
    outer = sig.simulate_signal("outer", 0.7, seed=3)
    fn = extract_features(normal)
    fo = extract_features(outer)
    i_crest = FEATURE_NAMES.index("crest_factor")
    i_impulse = FEATURE_NAMES.index("impulse_factor")
    i_kurt = FEATURE_NAMES.index("kurtosis")
    assert fo[i_crest] > fn[i_crest] + 1.0
    assert fo[i_impulse] > fn[i_impulse] + 1.2
    assert fo[i_kurt] > fn[i_kurt] + 0.3


def test_degradation_trajectory_monotonic():
    t_frac, severity, hi, rul = sig.degradation_trajectory(2000.0, 30, seed=9)
    assert np.all(np.diff(severity) >= 0)            # 严重度单调递增
    assert np.all(np.diff(rul) < 0)                   # 剩余寿命单调递减
    assert np.all((severity >= 0) & (severity <= 1))
    assert rul[0] == pytest.approx(2000.0) and rul[-1] == pytest.approx(0.0, abs=1e-9)


def test_temperature_grows_with_severity():
    low = sig.bearing_temperature(0.1, seed=5)
    high = sig.bearing_temperature(0.9, seed=5)
    assert high > low + 20


def test_unknown_fault_type_raises():
    with pytest.raises(ValueError):
        sig.simulate_signal("bogus", 0.5)
