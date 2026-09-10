"""模型推理封装：加载训练好的 joblib 模型，提供线程安全的统一预测接口。"""
from __future__ import annotations

import json
import threading

import joblib
import numpy as np

from app.config import MODEL_DIR, SAMPLE_RATE, SHAFT_FREQ
from app.services.features import extract_features


class ModelUnavailable(RuntimeError):
    """模型文件缺失时抛出。"""


class DiagnosticModels:
    """诊断模型单例：故障分类器 + 严重度回归器 + RUL 回归器。

    首次预测时惰性加载模型文件，之后常驻内存；线程安全。
    """

    _instance: "DiagnosticModels | None" = None
    _lock = threading.Lock()

    def __new__(cls) -> "DiagnosticModels":
        with cls._lock:
            if cls._instance is None:
                obj = super().__new__(cls)
                obj._loaded = False
                obj._meta: dict | None = None
                cls._instance = obj
        return cls._instance

    # ---------- 加载 ----------
    @property
    def ready(self) -> bool:
        """模型文件是否齐全（不触发加载）。"""
        names = ("fault_classifier", "severity_regressor",
                 "rul_regressor", "label_encoder")
        return all((MODEL_DIR / f"{n}.joblib").exists() for n in names)

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            if not self.ready:
                raise ModelUnavailable(
                    "模型未训练或文件缺失，请先运行: "
                    "python -m training.generate_dataset && "
                    "python -m training.train_models")
            self.classifier = joblib.load(MODEL_DIR / "fault_classifier.joblib")
            self.severity_reg = joblib.load(MODEL_DIR / "severity_regressor.joblib")
            self.rul_reg = joblib.load(MODEL_DIR / "rul_regressor.joblib")
            self.encoder = joblib.load(MODEL_DIR / "label_encoder.joblib")
            self._meta = json.loads((MODEL_DIR / "meta.json").read_text("utf-8"))
            self._loaded = True

    @property
    def meta(self) -> dict:
        self._ensure_loaded()
        return self._meta

    # ---------- 推理 ----------
    def predict(
        self,
        x: np.ndarray,
        sample_rate: float = SAMPLE_RATE,
        shaft_freq: float = SHAFT_FREQ,
    ) -> dict:
        """对一段振动信号做完整诊断。

        Returns
        -------
        dict: fault_type / confidence / probs / severity(%) / rul_hours / features
        """
        self._ensure_loaded()
        feats = extract_features(x, sample_rate, shaft_freq)
        probs_raw = self.classifier.predict_proba(feats.reshape(1, -1))[0]
        probs = {str(cls_): float(p)
                 for cls_, p in zip(self.encoder.classes_, probs_raw)}
        idx = int(np.argmax(probs_raw))
        fault_type = str(self.encoder.classes_[idx])
        severity = float(np.clip(
            self.severity_reg.predict(feats.reshape(1, -1))[0], 0.0, 1.0)) * 100.0
        rul = float(np.clip(
            self.rul_reg.predict(feats.reshape(1, -1))[0], 0.0, None))
        return {
            "fault_type": fault_type,
            "confidence": probs[fault_type],
            "probs": probs,
            "severity": severity,
            "rul_hours": rul,
            "features": feats.tolist(),
        }


def aggregate_predictions(results: list[dict]) -> dict:
    """汇总多窗口预测结果：概率均值投票 + 严重度/RUL 取中位数。

    Parameters
    ----------
    results : DiagnosticModels.predict() 返回值的列表（每个窗口一个）。

    Returns
    -------
    dict: fault_type / confidence / probs / severity / rul_hours
    """
    if not results:
        raise ValueError("预测结果为空")
    classes = list(results[0]["probs"].keys())
    mean_probs = {c: float(np.mean([r["probs"][c] for r in results]))
                  for c in classes}
    fault_type = max(mean_probs, key=mean_probs.get)
    return {
        "fault_type": fault_type,
        "confidence": mean_probs[fault_type],
        "probs": mean_probs,
        "severity": float(np.median([r["severity"] for r in results])),
        "rul_hours": float(np.median([r["rul_hours"] for r in results])),
    }
