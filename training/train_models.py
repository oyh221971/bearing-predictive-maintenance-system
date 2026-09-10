"""模型训练脚本：训练故障诊断分类器、严重度回归器与 RUL 预测器。

输入：data/raw/*.csv（由 generate_dataset.py 生成）
输出：training/models/ 下的 joblib 模型 + meta.json 元信息（含评估指标）

用法：
    python -m training.train_models
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, classification_report,
                             mean_absolute_error, r2_score)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder

from app.config import (
    ALERT_RUL_HOURS,
    BEARING,
    FAULT_PROB_THRESHOLD,
    MODEL_DIR,
    N_POINTS,
    RAW_DIR,
    SAMPLE_RATE,
)
from app.services.features import FEATURE_NAMES

# 是否将 FE（风扇端）通道纳入分类器训练（环境变量 TRAIN_USE_FE=1 开启）。
# FE 通道故障特征天然更弱，默认为 False 以保证 DE 通道判别精度。
USE_FE_CHANNEL = os.environ.get("TRAIN_USE_FE", "0") == "1"


def _console_utf8() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def _build_classifier() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=400, min_samples_leaf=2, n_jobs=-1, random_state=42,
        class_weight="balanced")  # 真实数据类别不均衡（外圈样本多），平衡权重


def train_classifier(df_cls: pd.DataFrame) -> dict:
    """故障诊断（4 分类）+ 严重度回归。"""
    X = df_cls[FEATURE_NAMES].to_numpy()
    y = df_cls["fault_type"].to_numpy()
    sev = df_cls["severity"].to_numpy()
    # 严重度回归仅用 DE 通道 + 仿真数据训练：FE 通道故障特征偏弱，
    # 若参与训练会拉低严重度标定（分类器不受影响，用全部通道增强鲁棒性）
    sev_mask = df_cls["channel"].fillna("DE").to_numpy() != "FE"

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42)

    encoder = LabelEncoder().fit(y_tr)
    clf = _build_classifier()
    clf.fit(X_tr, encoder.transform(y_tr))

    y_pred = encoder.inverse_transform(clf.predict(X_te))
    acc = float(accuracy_score(y_te, y_pred))
    report = classification_report(y_te, y_pred, output_dict=True,
                                   zero_division=0)

    X_s_tr, X_s_te, s_tr, s_te = train_test_split(
        X[sev_mask], sev[sev_mask], test_size=0.25, random_state=42)
    reg = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=42)
    reg.fit(X_s_tr, s_tr)
    s_pred = np.clip(reg.predict(X_s_te), 0.0, 1.0)
    sev_mae = float(mean_absolute_error(s_te, s_pred))
    sev_r2 = float(r2_score(s_te, s_pred))

    metrics = {
        "classification_accuracy": acc,
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "per_class": {k: {"precision": v["precision"], "recall": v["recall"],
                          "f1": v["f1-score"]}
                      for k, v in report.items() if k not in ("accuracy",
                                                              "macro avg",
                                                              "weighted avg")},
        "severity_mae": sev_mae,
        "severity_r2": sev_r2,
        "n_train": len(X_tr), "n_test": len(X_te),
    }
    joblib.dump(clf, MODEL_DIR / "fault_classifier.joblib")
    joblib.dump(reg, MODEL_DIR / "severity_regressor.joblib")
    joblib.dump(encoder, MODEL_DIR / "label_encoder.joblib")
    return metrics


def train_rul(df_deg: pd.DataFrame) -> dict:
    """剩余寿命 RUL 回归（按退化轨迹分组切分，防止同轨迹数据泄漏）。"""
    X = df_deg[FEATURE_NAMES].to_numpy()
    y = df_deg["rul_hours"].to_numpy()
    groups = df_deg["run_id"].to_numpy()

    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    tr_idx, te_idx = next(gss.split(X, y, groups))

    model = GradientBoostingRegressor(
        n_estimators=400, learning_rate=0.05, max_depth=4,
        subsample=0.9, random_state=42)
    model.fit(X[tr_idx], y[tr_idx])

    pred = np.clip(model.predict(X[te_idx]), 0.0, None)
    mae = float(mean_absolute_error(y[te_idx], pred))
    r2 = float(r2_score(y[te_idx], pred))
    metrics = {"rul_mae_hours": mae, "rul_r2": r2,
               "n_train": int(len(tr_idx)), "n_test": int(len(te_idx))}
    joblib.dump(model, MODEL_DIR / "rul_regressor.joblib")
    return metrics


def evaluate_real_holdout(df_synth: pd.DataFrame, df_real: pd.DataFrame) -> dict:
    """真实数据独立验证：按源文件分组留出 25%，训练集 = 仿真 + 其余 75% 真实文件。

    按文件分组防止同一文件（同一工况连续段）的窗口同时进入训练/测试造成泄漏。
    """
    sources = np.asarray(df_real["source"].unique(), dtype=object)
    rng = np.random.default_rng(42)
    rng.shuffle(sources)
    n_test_files = max(1, int(round(len(sources) * 0.25)))
    test_sources = set(sources[:n_test_files])
    real_test = df_real[df_real["source"].isin(test_sources)]
    real_train = df_real[~df_real["source"].isin(test_sources)]
    if not USE_FE_CHANNEL:  # 与部署模型保持一致：分类器训练不使用 FE 通道
        real_train = real_train[real_train["channel"] != "FE"]
    train_df = pd.concat([df_synth, real_train], ignore_index=True)

    X_tr = train_df[FEATURE_NAMES].to_numpy()
    y_tr = train_df["fault_type"].to_numpy()
    X_te = real_test[FEATURE_NAMES].to_numpy()
    y_te = real_test["fault_type"].to_numpy()

    enc = LabelEncoder().fit(y_tr)
    clf = _build_classifier()
    clf.fit(X_tr, enc.transform(y_tr))
    y_pred = enc.inverse_transform(clf.predict(X_te))
    report = classification_report(y_te, y_pred, output_dict=True,
                                   zero_division=0)
    # DE 通道单独的报告（与部署模型训练配置一致）
    de_mask_arr = np.asarray(real_test["channel"].fillna("DE") == "DE")
    report_de = classification_report(y_te[de_mask_arr],
                                      y_pred[de_mask_arr],
                                      output_dict=True, zero_division=0)

    sev_reg = RandomForestRegressor(n_estimators=300, n_jobs=-1,
                                    random_state=42)
    sev_train = train_df[train_df["channel"].fillna("DE") != "FE"]
    sev_test = real_test[real_test["channel"].fillna("DE") != "FE"]
    sev_reg.fit(sev_train[FEATURE_NAMES].to_numpy(),
                sev_train["severity"].to_numpy())
    sev_pred = np.clip(
        sev_reg.predict(sev_test[FEATURE_NAMES].to_numpy()), 0.0, 1.0)

    # 仅 DE 通道的独立指标（与历史版本公平对比；FE 通道故障特征天然更弱）
    de_mask = np.asarray(real_test["channel"].fillna("DE") == "DE")
    de_acc = float(accuracy_score(y_te[de_mask], y_pred[de_mask]))
    # 仅"标准工况"DE 指标：原 39 文件（@6 时钟位、007~021 缺陷），
    # 用于与早期 90.5% 的基准公平对比（不含新增的 @3/@12/028 更难数据）
    std_prefixes = tuple(f"{n}" for n in (
        "97_Normal", "98_Normal", "99_Normal",
        "105_", "106_", "107_", "108_", "118_", "119_", "120_", "121_",
        "130@6", "131@6", "132@6", "133@6",
        "169_", "170_", "171_", "172_", "185_", "186_", "187_", "188_",
        "197@6", "198@6", "199@6", "200@6",
        "209_", "210_", "211_", "212_", "222_", "223_", "224_", "225_",
        "234_", "235_", "236_", "237_"))
    std_mask = np.asarray(
        [str(s).startswith(std_prefixes) for s in real_test["source"]])
    de_std_mask = de_mask & std_mask
    de_std_acc = float(accuracy_score(y_te[de_std_mask], y_pred[de_std_mask])) \
        if de_std_mask.sum() else float("nan")

    return {
        "accuracy": float(accuracy_score(y_te, y_pred)),
        "accuracy_de_only": de_acc,
        "accuracy_de_standard": de_std_acc,
        "macro_f1": float(report_de["macro avg"]["f1-score"]),
        "per_class": {k: {"precision": v["precision"], "recall": v["recall"],
                          "f1": v["f1-score"]}
                      for k, v in report_de.items() if k not in ("accuracy",
                                                                 "macro avg",
                                                                 "weighted avg")},
        "severity_mae": float(mean_absolute_error(
            sev_test["severity"].to_numpy(), sev_pred)),
        "n_train_files": int(len(sources) - n_test_files),
        "n_test_files": int(n_test_files),
        "n_test": int(len(real_test)),
    }


def main() -> None:
    _console_utf8()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    cls_csv = RAW_DIR / "classify_features.csv"
    deg_csv = RAW_DIR / "degradation_features.csv"
    for p in (cls_csv, deg_csv):
        if not p.exists():
            raise SystemExit(f"缺少数据集 {p}，请先运行: python -m training.generate_dataset")

    df_cls = pd.read_csv(cls_csv)
    df_deg = pd.read_csv(deg_csv)

    # 若存在 CWRU 真实数据特征（import_cwru.py 产出），自动合并参与训练
    df_real = None
    df_real_all = None
    real_csv = RAW_DIR / "real_classify_features.csv"
    if real_csv.exists():
        df_real_all = pd.read_csv(real_csv)
        df_real = df_real_all
        if not USE_FE_CHANNEL:
            df_real = df_real[df_real["channel"] != "FE"].reset_index(drop=True)
        n_fe = int(df_real_all["channel"].isin(["FE"]).sum())
        print(f"  [真实数据] 合并 {len(df_real)} 条 CWRU 特征"
              f"（{df_real['source'].nunique()} 个文件）参与训练"
              + (f"，已排除 {n_fe} 条 FE 通道" if not USE_FE_CHANNEL and n_fe else ""))
        df_cls = pd.concat([df_cls, df_real], ignore_index=True)

    print("=" * 60)
    print("训练模型 ...")
    m_cls = train_classifier(df_cls)
    print(f"  [故障诊断] 测试准确率 {m_cls['classification_accuracy']:.2%}"
          f"  macro-F1 {m_cls['macro_f1']:.2%}（仿真+真实混合切分）")
    for k, v in m_cls["per_class"].items():
        print(f"      {k:<7s}  P={v['precision']:.2%}  R={v['recall']:.2%}  F1={v['f1']:.2%}")
    print(f"  [严重度]   MAE {m_cls['severity_mae']:.3f}  R² {m_cls['severity_r2']:.3f}")

    m_real = None
    if df_real_all is not None:
        m_real = evaluate_real_holdout(pd.read_csv(cls_csv), df_real_all)
        print("-" * 60)
        print(f"  [真实数据独立验证] 按文件留出 {m_real['n_test_files']} 个文件"
              f"（{m_real['n_test']} 个窗口）")
        print(f"      全部通道准确率 {m_real['accuracy']:.2%}  "
              f"仅DE通道 {m_real['accuracy_de_only']:.2%}  "
              f"标准工况DE {m_real['accuracy_de_standard']:.2%}  "
              f"严重度MAE {m_real['severity_mae']:.3f}")
        for k, v in m_real["per_class"].items():
            print(f"      {k:<7s}  P={v['precision']:.2%}  R={v['recall']:.2%}  F1={v['f1']:.2%}")

    m_rul = train_rul(df_deg)
    print(f"  [剩余寿命] MAE {m_rul['rul_mae_hours']:.1f} h  R² {m_rul['rul_r2']:.3f}")

    metrics_all = {**m_cls, **m_rul}
    if m_real is not None:
        metrics_all["real_data"] = m_real
    meta = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "feature_names": FEATURE_NAMES,
        "n_features": len(FEATURE_NAMES),
        "fault_classes": list(df_cls["fault_type"].value_counts().index),
        "sample_rate": SAMPLE_RATE,
        "n_points": N_POINTS,
        "bearing": BEARING,
        "thresholds": {
            "fault_prob": FAULT_PROB_THRESHOLD,
            "alert_rul_hours": ALERT_RUL_HOURS,
        },
        "data": {
            "n_samples": int(len(df_cls)),
            "real_samples": int(len(df_real)) if df_real is not None else 0,
        },
        "metrics": metrics_all,
    }
    with open(MODEL_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"模型与元信息已保存到 {MODEL_DIR}")
    print(f"总耗时 {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
