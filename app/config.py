"""全局配置：路径、轴承几何参数、采样参数、监测参数。

所有模块共享此处的配置，保证训练与推理的采样/几何参数一致。
"""
from __future__ import annotations

from pathlib import Path

# ---------- 路径 ----------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CWru_DIR = DATA_DIR / "cwru"          # CWRU 真实数据集 .mat 文件目录
MODEL_DIR = BASE_DIR / "training" / "models"
STATIC_DIR = BASE_DIR / "app" / "static"
DB_PATH = DATA_DIR / "bearing_monitor.db"

# ---------- 轴承几何参数（SKF 6205 深沟球轴承，CWRU 数据集同款） ----------
BEARING = {
    "name": "SKF 6205",
    "n_balls": 9,                # 滚动体数量
    "ball_diameter_mm": 7.94,    # 滚动体直径
    "pitch_diameter_mm": 39.04,  # 节圆直径
    "contact_angle_deg": 0.0,    # 接触角
}

# ---------- 采样与工况 ----------
SAMPLE_RATE = 12000              # 采样频率 Hz
WINDOW_SECONDS = 0.5             # 每个样本窗口时长 s
N_POINTS = int(SAMPLE_RATE * WINDOW_SECONDS)   # 每样本点数 = 6000
SHAFT_RPM = 1500                 # 主轴转速 rpm
SHAFT_FREQ = SHAFT_RPM / 60.0    # 转频 Hz（25 Hz）
RESONANCE_FREQ = 3000.0          # 结构共振频率 Hz（冲击激发的载波）
RESONANCE_DECAY = 800.0          # 冲击响应的指数衰减系数 1/s（τ=1.25ms，冲击清晰分离）

# ---------- 监测引擎 ----------
MONITOR_INTERVAL_SECONDS = 2.0   # 每多少秒产生一个监测样本
NOMINAL_LIFE_HOURS = 2000.0      # 额定寿命（用于健康指数换算）
DEGRADE_LIFE_HOURS_PER_TICK = 40.0   # 退化场景中每个样本推进的"寿命小时数"
FAULT_PROB_THRESHOLD = 0.6       # 故障概率告警阈值
ALERT_RUL_HOURS = 200.0          # 剩余寿命告警阈值 h

# ---------- 包络谱带通滤波频带（围绕共振频率） ----------
BAND_LOW = RESONANCE_FREQ - 1500.0
BAND_HIGH = RESONANCE_FREQ + 1500.0

# ---------- 服务 ----------
HOST = "127.0.0.1"
PORT = 8000
