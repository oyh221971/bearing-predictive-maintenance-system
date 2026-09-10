"""pytest 共享 fixture。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# 确保项目根目录在 sys.path（`python -m pytest` 时已含，直接 `pytest` 也兼容）
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def models_ready() -> bool:
    from app.config import MODEL_DIR
    return all((MODEL_DIR / f"{n}.joblib").exists()
               for n in ("fault_classifier", "severity_regressor",
                         "rul_regressor", "label_encoder"))
