"""迁移评估：用磁盘上现有模型直接测试 CWRU 真实数据。

在"用真实数据重训之前"运行，衡量仿真训练的模型对真实数据的泛化能力（迁移学习效果）：
    python -m training.import_cwru          # 先生成 real_classify_features.csv
    python -m training.evaluate_real        # 本脚本（重训前运行）

重训之后磁盘模型已包含真实数据，此评估不再代表迁移能力。
"""
from __future__ import annotations

import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, mean_absolute_error)

from app.config import MODEL_DIR, RAW_DIR
from app.services.features import FEATURE_NAMES
from app.services.signal import FAULT_NAMES_ZH


def _console_utf8() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def main() -> None:
    _console_utf8()
    real_csv = RAW_DIR / "real_classify_features.csv"
    if not real_csv.exists():
        raise SystemExit("缺少真实特征数据，请先运行: python -m training.import_cwru")

    clf_path = MODEL_DIR / "fault_classifier.joblib"
    enc_path = MODEL_DIR / "label_encoder.joblib"
    sev_path = MODEL_DIR / "severity_regressor.joblib"
    if not clf_path.exists():
        raise SystemExit("模型不存在，请先运行: python -m training.train_models")

    df = pd.read_csv(real_csv)
    X = df[FEATURE_NAMES].to_numpy()
    y = df["fault_type"].to_numpy()
    sev_true = df["severity"].to_numpy()

    clf = joblib.load(clf_path)
    enc = joblib.load(enc_path)
    sev_reg = joblib.load(sev_path)

    y_pred = enc.inverse_transform(clf.predict(X))
    sev_pred = np.clip(sev_reg.predict(X), 0.0, 1.0)

    acc = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y, y_pred, labels=enc.classes_)

    print("=" * 62)
    print("  现有模型在 CWRU 真实数据上的迁移评估")
    print("=" * 62)
    print(f"  真实样本数: {len(df)}（来自 {df['source'].nunique()} 个文件）")
    print(f"  故障分类准确率: {acc:.2%}")
    print(f"  严重度 MAE: {mean_absolute_error(sev_true, sev_pred):.3f}"
          f"（真值 0.30/0.60/0.90 三档）")
    print("-" * 62)
    print("  各类别指标:")
    for k in enc.classes_:
        r = report[k]
        print(f"    {FAULT_NAMES_ZH[k]:<6s} P={r['precision']:.2%}  "
              f"R={r['recall']:.2%}  F1={r['f1-score']:.2%}")
    print("-" * 62)
    print("  混淆矩阵 (行=真实, 列=预测):")
    print("         " + "".join(f"{FAULT_NAMES_ZH[c]:>8}" for c in enc.classes_))
    for i, row in enumerate(cm):
        print(f"    {FAULT_NAMES_ZH[enc.classes_[i]]:<6}"
              + "".join(f"{v:>8}" for v in row))
    print("=" * 62)
    print("  下一步: python -m training.train_models  # 合并真实数据重训")
    print("  提示  : 若迁移准确率已很高，说明仿真模型特征体系与真实数据同源；")
    print("          重训后 train_models 会输出'按文件留出'的真实数据验证指标。")


if __name__ == "__main__":
    main()
