"""特征提取模块单元测试。"""
from __future__ import annotations

import numpy as np
import pytest

from app.services import signal as sig
from app.services.features import (FEATURE_NAMES, envelope_spectrum,
                                   extract_features)


def test_feature_vector_shape_and_finiteness():
    for ft in sig.FAULT_TYPES:
        x = sig.simulate_signal(ft, 0.5 if ft != "normal" else 0.0, seed=11)
        feats = extract_features(x, 12000.0, 25.0)
        assert feats.shape == (20,)
        assert len(FEATURE_NAMES) == 20
        assert np.isfinite(feats).all()


def test_inner_fault_shows_bpfi_peak_in_envelope():
    """内圈故障信号的包络谱在 BPFI 处应有远高于正常信号的谱峰。"""
    normal = sig.simulate_signal("normal", 0.0, seed=13)
    inner = sig.simulate_signal("inner", 0.8, seed=13)
    fn = extract_features(normal, 12000.0, 25.0)
    fi = extract_features(inner, 12000.0, 25.0)
    i_bpfi = FEATURE_NAMES.index("env_bpfi")
    assert fi[i_bpfi] > fn[i_bpfi] * 10


def test_outer_fault_shows_bpfo_peak():
    normal = sig.simulate_signal("normal", 0.0, seed=17)
    outer = sig.simulate_signal("outer", 0.8, seed=17)
    i_bpfo = FEATURE_NAMES.index("env_bpfo")
    assert (extract_features(outer, 12000.0, 25.0)[i_bpfo]
            > extract_features(normal, 12000.0, 25.0)[i_bpfo] * 10)


def test_inner_fault_signal_is_impulsive():
    """内圈故障冲击幅值受转频调制，稀疏性更强，峰度与波峰因子应显著升高。"""
    normal = sig.simulate_signal("normal", 0.0, seed=19)
    inner = sig.simulate_signal("inner", 0.8, seed=19)
    i_kurt = FEATURE_NAMES.index("kurtosis")
    i_crest = FEATURE_NAMES.index("crest_factor")
    fn, fi = extract_features(normal), extract_features(inner)
    assert fi[i_kurt] > fn[i_kurt] + 2
    assert fi[i_crest] > fn[i_crest] + 1


def test_envelope_spectrum_length():
    x = sig.simulate_signal("outer", 0.7, seed=23)
    freqs, amp = envelope_spectrum(x, 12000.0)
    assert len(freqs) == len(amp) == len(x) // 2 + 1
    assert np.isfinite(amp).all() and amp.sum() > 0


def test_short_signal_raises():
    with pytest.raises(ValueError):
        extract_features(np.zeros(100))
