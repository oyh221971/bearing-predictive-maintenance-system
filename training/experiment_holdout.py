"""消融实验：定位不同训练配置对真实数据留出验证的影响。

对比配置：
  1. 仿真 + 全部真实DE（含 @3/@12/028），class_weight=balanced  ← 当前默认
  2. 同 1，但不做类别平衡
  3. 同 1，但训练剔除 @3/@12 时钟位文件
  4. 同 1，但训练剔除 028 大缺陷文件
  5. 仅仿真数据（迁移参考）

用法：python -m training.experiment_holdout
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from app.config import RAW_DIR
from app.services.features import FEATURE_NAMES
from app.services.signal import FAULT_NAMES_ZH

STD_PREFIXES = tuple(
    "97_Normal 98_Normal 99_Normal "
    "105_ 106_ 107_ 108_ 118_ 119_ 120_ 121_ "
    "130@6 131@6 132@6 133@6 "
    "169_ 170_ 171_ 172_ 185_ 186_ 187_ 188_ "
    "197@6 198@6 199@6 200@6 "
    "209_ 210_ 211_ 212_ 222_ 223_ 224_ 225_ "
    "234_ 235_ 236_ 237_".split())


def _console_utf8():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def is_standard(src: str) -> bool:
    return str(src).startswith(STD_PREFIXES)


def is_clock_shift(src: str) -> bool:
    s = str(src)
    return s.startswith(("144_", "145_", "146_", "147_", "156_",
                         "246_", "247_", "248_", "249_",
                         "258_", "259_", "260_", "261_"))


def is_028(src: str) -> bool:
    return str(src).startswith(("3001", "3002", "3003", "3004",
                                "3005", "3006", "3007", "3008"))


def run_config(name: str, train_df: pd.DataFrame, real_test: pd.DataFrame,
               balanced: bool = True):
    enc = LabelEncoder().fit(train_df["fault_type"])
    cw = "balanced" if balanced else None
    clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                 n_jobs=-1, random_state=42, class_weight=cw)
    clf.fit(train_df[FEATURE_NAMES].to_numpy(),
            enc.transform(train_df["fault_type"]))
    te = real_test[real_test["channel"] == "DE"]
    y_true = te["fault_type"].to_numpy()
    y_pred = enc.inverse_transform(clf.predict(te[FEATURE_NAMES].to_numpy()))

    acc = accuracy_score(y_true, y_pred)
    std = te[te["source"].apply(is_standard)]
    acc_std = accuracy_score(std["fault_type"].to_numpy(),
                             enc.inverse_transform(clf.predict(
                                 std[FEATURE_NAMES].to_numpy())))
    normal_recall = (y_pred[y_true == "normal"] == "normal").mean()
    print(f"  {name:<34s} DE准确率 {acc:6.2%} | 标准工况DE {acc_std:6.2%} "
          f"| 正常召回 {normal_recall:6.2%}")
    return y_true, y_pred


def main():
    _console_utf8()
    synth = pd.read_csv(RAW_DIR / "classify_features.csv")
    real = pd.read_csv(RAW_DIR / "real_classify_features.csv")
    real_de = real[real["channel"] == "DE"].reset_index(drop=True)

    # 固定留出划分（与训练脚本同种子）
    sources = np.asarray(real["source"].unique(), dtype=object)
    rng = np.random.default_rng(42)
    rng.shuffle(sources)
    n_test_files = max(1, int(round(len(sources) * 0.25)))
    test_sources = set(sources[:n_test_files])
    real_test = real[real["source"].isin(test_sources)]
    real_train_all = real[~real["source"].isin(test_sources)]

    print("=" * 76)
    print(f"消融实验（留出 {n_test_files} 个文件，测试集 = 全部通道）")
    print("-" * 76)

    y_true, y_pred = run_config(
        "1. 仿真+全部DE(balanced)[当前]",
        pd.concat([synth, real_train_all[real_train_all["channel"] == "DE"]],
                  ignore_index=True), real_test)

    run_config(
        "2. 同1但无类别平衡",
        pd.concat([synth, real_train_all[real_train_all["channel"] == "DE"]],
                  ignore_index=True), real_test, balanced=False)

    no_shift = real_train_all[~real_train_all["source"].apply(is_clock_shift)]
    run_config("3. 训练剔除@3/@12",
               pd.concat([synth, no_shift[no_shift["channel"] == "DE"]],
                         ignore_index=True), real_test)

    no_028 = real_train_all[~real_train_all["source"].apply(is_028)]
    run_config("4. 训练剔除028",
               pd.concat([synth, no_028[no_028["channel"] == "DE"]],
                         ignore_index=True), real_test)

    run_config("5. 仅仿真(迁移参考)", synth, real_test)

    print("-" * 76)
    print("配置1 在 DE 标准工况上的混淆矩阵 (行=真实, 列=预测):")
    std_mask = te["source"].apply(is_standard).to_numpy()
    y_true_s = te["fault_type"].to_numpy()[std_mask]
    y_pred_s = y_pred[std_mask]
    cm = confusion_matrix(y_true_s, y_pred_s, labels=["normal", "inner",
                                                      "outer", "ball"])
    print("         " + "".join(f"{FAULT_NAMES_ZH[c]:>8}"
                               for c in ("normal", "inner", "outer", "ball")))
    for i, row in enumerate(cm):
        print(f"    {FAULT_NAMES_ZH[('normal','inner','outer','ball')[i]]:<6}"
              + "".join(f"{v:>8}" for v in row))


if __name__ == "__main__":
    main()
