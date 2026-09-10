"""数据集生成脚本：物理仿真 → 特征提取 → 落盘 CSV。

生成两份训练数据：
  1. classify_features.csv   —— 故障诊断/严重度数据集（4 类故障状态 × 多严重度 × 多随机种子）
  2. degradation_features.csv —— 全寿命退化数据集（多条退化轨迹，标签为剩余寿命 RUL）

以及一份演示原始信号 data/raw/demo_signals.npz。

用法：
    python -m training.generate_dataset
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from app.config import RAW_DIR, SAMPLE_RATE
from app.services.features import FEATURE_NAMES, extract_features
from app.services import signal as sig


def _console_utf8() -> None:
    """Windows 控制台默认 GBK，重配为 UTF-8 以免中文打印乱码。"""
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def build_classify_dataset() -> pd.DataFrame:
    """故障分类数据集：4 类 × 多严重度 × 多随机种子。"""
    rows: list[dict] = []
    plan = [
        ("normal", [0.0], 30),                       # 正常轴承
        ("inner", [0.15, 0.35, 0.55, 0.75, 0.95], 25),
        ("outer", [0.15, 0.35, 0.55, 0.75, 0.95], 25),
        ("ball", [0.15, 0.35, 0.55, 0.75, 0.95], 25),
    ]
    rng = np.random.default_rng(2024)
    for fault_type, severities, n_seeds in plan:
        for severity in severities:
            for seed in range(n_seeds):
                # 转频在额定值 ±15% 内随机波动，增强模型对工况变化的鲁棒性
                shaft_freq = sig.SHAFT_FREQ * float(rng.uniform(0.85, 1.15))
                x = sig.simulate_signal(fault_type, severity, shaft_freq,
                                        rng=rng)
                feats = extract_features(x, SAMPLE_RATE, shaft_freq)
                row = {"fault_type": fault_type, "severity": severity,
                       "shaft_freq": shaft_freq}
                row.update({n: v for n, v in zip(FEATURE_NAMES, feats)})
                rows.append(row)
    return pd.DataFrame(rows)


def build_degradation_dataset(n_runs: int = 60, n_steps: int = 30) -> pd.DataFrame:
    """全寿命退化数据集：每条轨迹一个 run_id，标签为剩余寿命（小时）。"""
    rows: list[dict] = []
    rng = np.random.default_rng(7)
    fault_pool = ("inner", "outer", "ball")
    for run_id in range(n_runs):
        fault_type = fault_pool[run_id % len(fault_pool)]
        total_life = float(rng.uniform(1200.0, 3000.0))
        run = sig.simulate_degradation_run(
            fault_type, total_life, n_steps, seed=1000 + run_id)
        for i in range(n_steps):
            feats = extract_features(run["signals"][i], SAMPLE_RATE,
                                     run["shaft_freq"])
            row = {
                "run_id": run_id,
                "fault_type": fault_type,
                "t_frac": run["t_frac"][i],
                "severity": run["severity"][i],
                "health_index": run["health_index"][i],
                "rul_hours": run["rul_hours"][i],
                "shaft_freq": run["shaft_freq"],
            }
            row.update({n: v for n, v in zip(FEATURE_NAMES, feats)})
            rows.append(row)
    return pd.DataFrame(rows)


def save_demo_signals() -> None:
    """保存每类状态的演示原始信号（severity=0.6），供前端/测试展示。"""
    demo: dict[str, np.ndarray] = {}
    for fault_type in sig.FAULT_TYPES:
        sev = 0.0 if fault_type == "normal" else 0.6
        demo[fault_type] = sig.simulate_signal(fault_type, sev, seed=42)
    np.savez(RAW_DIR / "demo_signals.npz", **demo)


def main() -> None:
    _console_utf8()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("=" * 60)
    print("生成轴承故障仿真数据集 ...")
    df_cls = build_classify_dataset()
    cls_path = RAW_DIR / "classify_features.csv"
    df_cls.to_csv(cls_path, index=False)
    print(f"  故障分类数据集: {len(df_cls)} 条 -> {cls_path.name}")
    print(f"  类别分布: {dict(df_cls['fault_type'].value_counts())}")

    df_deg = build_degradation_dataset()
    deg_path = RAW_DIR / "degradation_features.csv"
    df_deg.to_csv(deg_path, index=False)
    print(f"  退化数据集: {len(df_deg)} 条 (来自 {df_deg['run_id'].nunique()} 条全寿命轨迹)"
          f" -> {deg_path.name}")

    save_demo_signals()
    print(f"  演示信号已保存 -> demo_signals.npz")
    print(f"总耗时 {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
