"""离线批量诊断 CLI：对真实采集的振动数据文件（CSV/TXT）做故障诊断。

用法：
    python -m tools.diagnose_file --file 振动数据.csv --fs 48000 --rpm 1797
    python -m tools.diagnose_file --file 振动数据.txt --fs 12000 --freq 29.95 --json report.json

参数说明：
    --file   数据文件路径（单列数值；逗号/空格/制表符分隔均可，自动跳过表头等非数值行）
    --fs     采集采样频率 Hz（默认 12000）
    --rpm    主轴转速 rpm（与 --freq 二选一，用于计算故障特征频率位置）
    --freq   转频 Hz（默认 25，即 1500 rpm）
    --json   可选，将完整报告保存为 JSON 文件
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from app.config import SAMPLE_RATE
from app.services import preprocess
from app.services.fault_detector import DiagnosticModels, aggregate_predictions
from app.services.signal import FAULT_NAMES_ZH

MAX_POINTS = 2_000_000  # 超过则抽稀，控制内存与耗时


def _console_utf8() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def load_signal(path: Path) -> np.ndarray:
    """宽松解析文本振动数据：按行读取，跳过非数值 token。"""
    raw = path.read_text(encoding="utf-8-sig", errors="ignore")
    vals: list[float] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line[0] in "#%//":
            continue
        for tok in (line.replace(",", " ").replace(";", " ")
                    .replace("\t", " ").split()):
            try:
                vals.append(float(tok))
            except ValueError:
                continue  # 表头/单位等非数值内容
    arr = np.asarray(vals, dtype=np.float64)
    if len(arr) == 0:
        raise SystemExit(f"[错误] 未能从 {path} 解析出数值数据")
    return arr


def main() -> None:
    _console_utf8()
    ap = argparse.ArgumentParser(description="轴承振动数据离线批量诊断")
    ap.add_argument("--file", required=True, help="振动数据文件 (CSV/TXT)")
    ap.add_argument("--fs", type=float, default=12000.0, help="采样频率 Hz")
    ap.add_argument("--rpm", type=float, default=None,
                    help="主轴转速 rpm（默认自动识别）")
    ap.add_argument("--freq", type=float, default=None,
                    help="转频 Hz（与 --rpm 二选一，默认自动识别）")
    ap.add_argument("--json", default=None, help="报告输出路径 (可选)")
    args = ap.parse_args()

    if args.rpm is not None:
        shaft_freq = args.rpm / 60.0
    elif args.freq is not None:
        shaft_freq = args.freq
    else:
        shaft_freq = None  # 自动估计

    path = Path(args.file)
    if not path.exists():
        raise SystemExit(f"[错误] 文件不存在: {path}")
    x = load_signal(path)
    if len(x) > MAX_POINTS:
        step = int(np.ceil(len(x) / MAX_POINTS))
        x = x[::step]
        print(f"[提示] 数据量 {len(x) * step} 点过大，已按 {step} 倍抽稀")

    xr = preprocess.resample_to(x, args.fs, SAMPLE_RATE)
    windows, padded = preprocess.prepare_windows(xr)
    if shaft_freq is None:
        shaft_freq = preprocess.estimate_shaft_freq(xr, SAMPLE_RATE)

    models = DiagnosticModels()
    results = [models.predict(w, SAMPLE_RATE, shaft_freq) for w in windows]
    agg = aggregate_predictions(results)

    # ---------- 控制台报告 ----------
    print("=" * 62)
    print("  轴承振动离线诊断报告")
    print("=" * 62)
    print(f"  文件        : {path.name}")
    print(f"  原始点数    : {len(x)}  (采样率 {args.fs:g} Hz)")
    print(f"  有效窗口数  : {len(windows)}  (0.5s/窗口, 50% 重叠"
          + ("，数据过短已补零)" if padded else ")"))
    print(f"  转频        : {shaft_freq:.2f} Hz"
          + ("（自动识别）" if args.rpm is None and args.freq is None else ""))
    print("-" * 62)
    print(f"  诊断结论    : {FAULT_NAMES_ZH[agg['fault_type']]}"
          f"  (置信度 {agg['confidence']:.1%})")
    print(f"  故障严重度  : {agg['severity']:.1f}%")
    print(f"  剩余寿命    : {agg['rul_hours']:.0f} h")
    print("  各类别概率  :")
    for k, v in agg["probs"].items():
        bar = "█" * int(round(v * 30))
        print(f"    {FAULT_NAMES_ZH[k]:<6s} {v:6.1%}  {bar}")
    print("-" * 62)
    print("  窗口明细 (前 10 个):")
    print(f"    {'窗口':>4} {'诊断':<8} {'置信度':>8} {'严重度':>8} {'RUL/h':>8}")
    for i, w in enumerate(results[:10]):
        print(f"    {i:>4} {FAULT_NAMES_ZH[w['fault_type']]:<8} "
              f"{w['confidence']:>7.1%} {w['severity']:>7.1f}% {w['rul_hours']:>8.0f}")
    print("=" * 62)

    if args.json:
        report = {
            "file": str(path),
            "sample_rate": args.fs,
            "shaft_freq": shaft_freq,
            "n_points": int(len(x)),
            "n_windows": len(windows),
            "padded": padded,
            "diagnosis": agg,
            "windows": [
                {"index": i, "fault_type": r["fault_type"],
                 "confidence": r["confidence"], "severity": r["severity"],
                 "rul_hours": r["rul_hours"]}
                for i, r in enumerate(results)
            ],
        }
        Path(args.json).write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"报告已保存: {args.json}")


if __name__ == "__main__":
    main()
