"""将 CWRU 凯斯西储大学轴承数据集 (.mat) 转换为系统训练特征 CSV。

CWRU 是滚动轴承故障诊断领域最权威的公开数据集：
  - 试验台：2hp 电机 + SKF 6205 轴承（与本系统几何参数一致）
  - 故障类型：内圈(IR)/外圈(OR)/滚动体(B)，单点电火花加工缺陷
  - 缺陷直径：0.007" / 0.014" / 0.021" 三档严重度
  - 采样：驱动端(DE) 12kHz 与 48kHz，转速 1797/1772/1750/1730 rpm

数据获取：官方站点不稳定，推荐 GitHub 镜像。本项目自带下载脚本：
    pwsh -File tools/download_cwru.ps1        # 自动下载 39 个 6205 DE 文件到 data/cwru/

用法：
  1. 把 .mat 文件放到 data/cwru/（或用上面的下载脚本）
  2. python -m training.import_cwru           # 生成 real_classify_features.csv
  3. python -m training.train_models          # 自动合并真实数据重新训练

文件名识别（兼容官方命名与常见镜像命名）：
  描述式：IR007_0 / OR021@6_0 / B014_0 / Normal_0 / 97_Normal_0
  数字式：105_0 / 130@6_0 / 197@6_0 ...（CWRU 官方文件编号）
  直径代码 007/014/021 映射严重度 0.30 / 0.60 / 0.90
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import resample_poly

from app.config import CWru_DIR, N_POINTS, RAW_DIR, SAMPLE_RATE
from app.services.features import FEATURE_NAMES, extract_features

_SEVERITY_BY_DIAMETER = {"007": 0.30, "014": 0.60, "021": 0.90}

_PATTERNS = [
    (re.compile(r"IR(\d{3})_"), "inner"),
    (re.compile(r"OR(\d{3})@\d+_"), "outer"),
    (re.compile(r"B(\d{3})_"), "ball"),
    (re.compile(r"normal", re.I), "normal"),
]

# CWRU 官方文件编号 → (故障类型, 严重度)；覆盖 6205 DE @12k 常用文件
_NUMERIC_MAP = {
    "97": ("normal", 0.0), "98": ("normal", 0.0),
    "99": ("normal", 0.0), "100": ("normal", 0.0),
    "105": ("inner", 0.30), "106": ("inner", 0.30),
    "107": ("inner", 0.30), "108": ("inner", 0.30),
    "118": ("ball", 0.30), "119": ("ball", 0.30),
    "120": ("ball", 0.30), "121": ("ball", 0.30),
    "130": ("outer", 0.30), "131": ("outer", 0.30),
    "132": ("outer", 0.30), "133": ("outer", 0.30),
    "144": ("outer", 0.30), "145": ("outer", 0.30),
    "146": ("outer", 0.30), "147": ("outer", 0.30),
    "156": ("outer", 0.30), "157": ("outer", 0.30),
    "158": ("outer", 0.30), "159": ("outer", 0.30),
    "169": ("inner", 0.60), "170": ("inner", 0.60),
    "171": ("inner", 0.60), "172": ("inner", 0.60),
    "185": ("ball", 0.60), "186": ("ball", 0.60),
    "187": ("ball", 0.60), "188": ("ball", 0.60),
    "197": ("outer", 0.60), "198": ("outer", 0.60),
    "199": ("outer", 0.60), "200": ("outer", 0.60),
    "209": ("inner", 0.90), "210": ("inner", 0.90),
    "211": ("inner", 0.90), "212": ("inner", 0.90),
    "222": ("ball", 0.90), "223": ("ball", 0.90),
    "224": ("ball", 0.90), "225": ("ball", 0.90),
    "234": ("outer", 0.90), "235": ("outer", 0.90),
    "236": ("outer", 0.90), "237": ("outer", 0.90),
    "246": ("outer", 0.90), "247": ("outer", 0.90),
    "248": ("outer", 0.90), "249": ("outer", 0.90),
    "258": ("outer", 0.90), "259": ("outer", 0.90),
    "260": ("outer", 0.90), "261": ("outer", 0.90),
    "3001": ("inner", 1.00), "3002": ("inner", 1.00),
    "3003": ("inner", 1.00), "3004": ("inner", 1.00),
    "3005": ("ball", 1.00), "3006": ("ball", 1.00),
    "3007": ("ball", 1.00), "3008": ("ball", 1.00),
}


def _console_utf8() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def identify_file(name: str) -> tuple[str, float] | None:
    """由文件名识别 (故障类型, 严重度)。无法识别返回 None。"""
    stem = Path(name).stem
    # 1) 描述式命名（含目录前缀的镜像命名也能命中）
    for pattern, fault in _PATTERNS[:3]:
        m = pattern.search(stem)
        if m and m.group(1) in _SEVERITY_BY_DIAMETER:
            return fault, _SEVERITY_BY_DIAMETER[m.group(1)]
    # 2) Normal 关键字
    if _PATTERNS[3][0].search(stem):
        return "normal", 0.0
    # 3) 数字编号命名: 105_0 / 130@6_0 / 3001_0 / 97_Normal_0
    m = re.match(r"(\d{3,4})", stem)
    if m:
        return _NUMERIC_MAP.get(m.group(1))
    return None


def _find_channel_keys(mat: dict) -> tuple[str | None, str | None]:
    """定位驱动端(DE)与风扇端(FE)振动时间序列的键，兼容多种命名。"""
    de = fe = None
    for key in mat:
        if key.endswith("_DE_time") or key.endswith("_DE_TIME"):
            de = key
        elif key.endswith("_FE_time") or key.endswith("_FE_TIME"):
            fe = key
    if de is None:
        for key in mat:
            kl = key.lower()
            if "de" in kl and "time" in kl and not kl.startswith("__"):
                de = key
                break
    return de, fe


def _infer_sample_rate(key: str, path: Path, n_points: int) -> float:
    """推断采样率：键前缀 → 目录名 → 官方编号 → 数据长度启发式。"""
    if key.upper().startswith("X012"):
        return 12000.0
    if key.upper().startswith("X048"):
        return 48000.0
    folder = str(path.parent).lower()
    if "48k" in folder or "48_khz" in folder:
        return 48000.0
    if "12k" in folder or "12_khz" in folder:
        return 12000.0
    m = re.match(r"X?(\d{3})", key)
    if m and m.group(1) in ("97", "98", "99", "100"):
        return 48000.0  # 官方正常基线 97~100 为 48kHz
    # 长度启发式：48kHz 的 10s 记录约 48 万点；12kHz 通常 ≤ 25 万点
    return 48000.0 if n_points > 400_000 else 12000.0


def load_rpm(mat: dict, fallback: float = 1797.0) -> float:
    for key in mat:
        if "RPM" in key.upper():
            arr = np.asarray(mat[key], dtype=np.float64).ravel()
            if len(arr):
                return float(np.median(arr))
    return fallback


def process_file(path: Path) -> pd.DataFrame:
    """处理单个 .mat：同时提取 DE 与 FE 两通道，逐窗口提特征。"""
    info = identify_file(path.name)
    fault_type, severity = info
    mat = loadmat(str(path))
    de_key, fe_key = _find_channel_keys(mat)
    if de_key is None and fe_key is None:
        raise ValueError(f"未找到 DE/FE 振动数据，实际键: {list(mat)[:8]}")
    shaft_freq = load_rpm(mat) / 60.0

    rows = []
    stride = N_POINTS  # 窗口不重叠
    for channel, key in (("DE", de_key), ("FE", fe_key)):
        if key is None:
            continue
        x = np.asarray(mat[key], dtype=np.float64).ravel()
        fs = _infer_sample_rate(str(key), path, len(x))
        # 对齐系统采样率
        if abs(fs - SAMPLE_RATE) > 0.01 * SAMPLE_RATE:
            from math import gcd
            up, down = int(SAMPLE_RATE), int(fs)
            g = gcd(up, down)
            x = resample_poly(x, up // g, down // g)
        for start in range(0, len(x) - N_POINTS + 1, stride):
            feats = extract_features(x[start:start + N_POINTS], SAMPLE_RATE,
                                     shaft_freq)
            row = {"fault_type": fault_type, "severity": severity,
                   "shaft_freq": shaft_freq, "channel": channel,
                   "source": path.name}
            row.update({n: v for n, v in zip(FEATURE_NAMES, feats)})
            rows.append(row)
    if not rows:
        print(f"  [跳过] {path.name}: 信号长度不足一个窗口")
    return pd.DataFrame(rows)


def main() -> None:
    _console_utf8()
    CWru_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(CWru_DIR.glob("*.mat"))
    if not files:
        raise SystemExit(
            f"[错误] {CWru_DIR} 下没有 .mat 文件。\n"
            "请运行: pwsh -File tools/download_cwru.ps1 （自动下载）\n"
            "或手动从 CWRU 镜像下载 6205 驱动端数据放入该目录。")

    t0 = time.time()
    print("=" * 62)
    print(f"导入 CWRU 真实数据 ({len(files)} 个文件) ...")
    frames, skipped, errors = [], [], []
    for path in files:
        try:
            info = identify_file(path.name)
            if info is None:
                skipped.append(path.name)
                continue
            frames.append(process_file(path))
            print(f"  [OK] {path.name:<20s} -> {info[0]:<6s} sev={info[1]:.2f}")
        except Exception as exc:
            errors.append((path.name, str(exc)))
            print(f"  [错误] {path.name}: {exc}")
    if skipped:
        print(f"  [跳过] {len(skipped)} 个无法识别的文件: {skipped[:5]}")
    if errors:
        print(f"  [失败] {len(errors)} 个文件处理出错")

    if not frames:
        raise SystemExit("没有成功处理任何文件，请检查文件内容。")
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna()
    out = RAW_DIR / "real_classify_features.csv"
    df.to_csv(out, index=False)
    print("-" * 62)
    print(f"真实特征数据: {len(df)} 条 -> {out.name}")
    print(f"类别分布: {dict(df['fault_type'].value_counts())}")
    print(f"严重度分布: {dict(df['severity'].value_counts())}")
    print(f"通道分布: {dict(df['channel'].value_counts())}")
    print(f"总耗时 {time.time() - t0:.1f}s")
    print("下一步: python -m training.train_models  （将自动合并真实数据重训）")


if __name__ == "__main__":
    main()
