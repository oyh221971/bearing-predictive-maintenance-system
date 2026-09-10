# 轴承故障预测性维护系统

基于 **物理仿真 + 机器学习 + 真实数据（CWRU）验证** 的滚动轴承故障诊断与剩余寿命预测系统。
上传一段振动信号，自动完成故障类型识别（内圈/外圈/滚动体）、严重度评估与剩余寿命（RUL）预测，
配套实时监控看板、告警管理与 REST API。

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5%2B-orange)
![Tests](https://img.shields.io/badge/Tests-31%2F31%20passed-brightgreen)
![Acceptance](https://img.shields.io/badge/验收-44%2F44%20passed-brightgreen)

> 技术栈：Python · FastAPI · scikit-learn · NumPy/SciPy · SQLite · ECharts（本地化，无外网依赖，无任何大模型 API）

---

## ✨ 功能特性

| 模块 | 说明 |
| --- | --- |
| 🎲 物理仿真信号引擎 | 基于 SKF 6205 几何参数的冲击-共振模型，合成 4 类状态、任意严重度、全寿命退化轨迹的振动信号 |
| 🧮 特征工程 | 20 维特征：时域统计（9）+ 频域统计（4）+ Hilbert 包络谱故障特征（7） |
| 🤖 三个机器学习模型 | 随机森林故障诊断（4 分类）+ 随机森林严重度回归 + 梯度提升剩余寿命回归 |
| 📡 实时监测引擎 | 后台线程周期性采样→诊断→落库→告警，支持 5 种场景一键切换 |
| 🚨 告警管理 | 故障概率/剩余寿命双阈值告警，自动解除、自动去重、可确认 |
| 🖥 中文监控看板 | 健康指数/剩余寿命/温度 KPI、实时波形、频谱（特征频率标注）、包络谱、健康趋势、告警与历史 |
| 📤 离线诊断 | 看板文件上传 / CLI / REST API 三种方式，**任意采样率自动重采样、转频自动识别** |
| 🔄 真实数据流水线 | CWRU 数据自动下载 → 特征导入 → 迁移评估 → 合并重训 → 独立验证，全脚本化 |
| ✅ 一键验收 | 44 项系统验收 + 31 项单元/API 测试，改动后随时复验 |

## 🎯 能力范围

### ✅ 支持

- **轴承型号**：SKF 6205 深沟球轴承（与 CWRU 数据集同款），采样率 12 kHz、0.5 s 窗口（其他采样率自动重采样对齐）；
- **故障类型**：正常 / 内圈故障 / 外圈故障 / 滚动体故障（4 分类）；
- **严重度**：0~100%，已用真实缺陷尺寸校准（0.007"→30%、0.014"→60%、0.021"→90%、0.028"→100%）；
- **剩余寿命**：相对寿命趋势估计（供维护排程参考，见下方局限）；
- **工况**：恒定转速（转速未知时留空自动识别）；外圈故障覆盖 3/6/12 点钟三个载荷位置。

### ❌ 不支持 / 已知局限

1. **不是安全认证系统**：诊断结果仅供维护决策参考，不可用于安全关键场景（见 SECURITY.md 免责声明）；
2. **换轴承型号需重训**：模型特征与 6205 几何参数绑定，其他型号请按 §7.2 流程用自有数据重训；
3. **RUL 早期固有不确定**：轴承退化初期信号与正常状态几乎无差异，单快照 RUL 估计误差大（这是物理极限，系统用指数平滑缓解）；
4. **变速工况**：请填写平均转速；快速大幅变速会影响包络谱特征定位；
5. **风扇端（FE）传感器**：默认模型按驱动端（DE）校准，覆盖 FE 需 `TRAIN_USE_FE=1` 重训；
6. **数据域**：准确率基于 CWRU 实验室数据验证，现场工况（不同安装、传递路径、负载）请先按 §7.2 用自有数据校准。

## 📊 准确度（真实数据实测，可复现）

| 评估 | 数据集 | 指标 |
| --- | --- | --- |
| 混合测试集 | 仿真 405 条 + CWRU 真实 1205 条 | 准确率 100% / 严重度 MAE 0.011 |
| **迁移能力**（仿真模型直接考真实数据） | CWRU 39 文件 800 窗口 | 准确率 **66.9%** |
| **重训后留出验证**（按文件分组留出 25%，从未参与训练） | CWRU 60 文件 2265 窗口 | **DE 通道 90.3%** · 标准工况 88.1% · 严重度 MAE **0.059** · 正常召回 98.3% |
| 全量复测（已知数据回归） | CWRU 60 文件 | DE 99.8%；三类时钟位 @6/@3/@12 均 ≥99% |
| 端到端抽查 | 7 个文件（含模型未见过工况） | 7/7 命中 |
| 剩余寿命（仿真退化轨迹） | 60 条轨迹 1800 窗口 | MAE 250 h（额定寿命 2000 h），R² 0.68 —— **趋势参考** |
| 系统验收 | `tools/acceptance_check.py` | 44/44 通过；单窗口诊断延迟均值 ~84 ms |

> ⚠️ **重要**：以上准确率基于 CWRU 凯斯西储实验室数据。迁移实验表明仿真模型直接用于真实数据只有 66.9%，
> 重训后达到 90.3%——**现场部署务必先按 §7.2 用自有数据校准验证**，不要直接拿默认模型做生产判断。

## 🚀 快速开始

### 环境要求

- Python 3.10+（开发环境 3.12）
- Windows / macOS / Linux

### 安装与启动（仓库自带训练好的模型，克隆即用）

```bash
git clone https://github.com/oyh221971/-.git
cd -
python -m pip install -r requirements.txt
python run.py          # Windows 也可双击 start.bat
```

浏览器打开：

- **监控看板**：<http://127.0.0.1:8000>
- **API 文档（Swagger）**：<http://127.0.0.1:8000/docs>

> 模型缺失时 `run.py` 会自动完成「数据生成 + 训练」（约 1 分钟）再启动。

## 🖥 如何正确使用

### 1️⃣ 看板实时监测（新手从这里开始）

1. 打开看板，顶部「监测场景」选择：
   - `正常运转` —— 观察健康基线；
   - `内圈故障` / `外圈故障` / `滚动体故障` —— 看 AI 秒级改判 + 频谱图上特征频率竖线对位 + 告警触发；
   - `全寿命退化模拟` —— 最完整的演示：健康指数逐步下滑、剩余寿命递减、触发生命告警、失效后自动开启新一轮。
2. 观察 KPI 卡片（健康指数/剩余寿命/RMS/温度）、AI 诊断面板（四类概率条）、频谱/包络谱/健康趋势图。
3. 告警会在条件满足时自动产生与解除；「活动告警」面板可查看。

### 2️⃣ 离线诊断你的真实数据（真正的用法）

**方式 A · 看板**：页面「离线诊断」面板 → 选择 CSV/TXT 文件（单列数值）→ 填采样频率 → 转频留空（自动识别）→ 开始诊断。

**方式 B · 命令行**：

```bash
python -m tools.diagnose_file --file 振动数据.csv --fs 48000          # 转频自动识别
python -m tools.diagnose_file --file 振动数据.txt --fs 12000 --rpm 1797 --json report.json
```

**方式 C · REST API**：

```bash
curl -X POST http://127.0.0.1:8000/api/diagnose \
  -H "Content-Type: application/json" \
  -d '{"samples": [0.1, -0.2, ...], "sample_rate": 48000, "shaft_freq": null}'
```

**要点**：① 用**加速度通道**原始数据（速度/位移信号会降低准确率）；② 系统自动去直流、重采样、滑窗
（0.5 s 窗口 / 50% 重叠 / 概率均值投票聚合），任意长度均可；③ 转速未知时 `shaft_freq` 留空，
系统用「谐波族匹配 + 故障特征频率反推」双假设自动估计（CWRU 60 文件实测误差 <0.15 Hz）。

### 3️⃣ 接入真实采集设备（生产方向）

任何能发 HTTP 的采集端（工控机 + IEPE 加速度传感器、树莓派 + ADXL345、PLC 网关）每 0.5 s
向 `/api/diagnose` POST 一帧即可获得诊断结果；传感器建议装在轴承座径向、采样率 ≥10 kHz。

### 4️⃣ 用自有数据重训校准（强烈建议）

见 §7.2，全程三条命令：下载数据 → `import_cwru` → `train_models`。

### 5️⃣ 日常验收

```bash
python -m pytest tests -q              # 31 项单元/API 测试
python -m tools.acceptance_check       # 44 项系统验收（需服务已启动）
```

## 📁 目录结构

```
bearing-pdm/
├── run.py                      # 一键启动（自动训练 + 启动服务）
├── start.bat                   # Windows 双击启动
├── requirements.txt
├── LICENSE                     # MIT
├── SECURITY.md                 # 安全策略与部署加固清单
├── app/
│   ├── main.py                 # FastAPI 应用入口
│   ├── config.py               # 全局配置（轴承几何/采样/监测参数）
│   ├── database.py             # SQLite 存储（样本/告警，线程安全）
│   ├── schemas.py              # Pydantic 请求/响应模型
│   ├── services/
│   │   ├── signal.py           # 轴承故障信号物理仿真 + 退化轨迹
│   │   ├── features.py         # 时域/频域/包络谱特征提取（20 维）
│   │   ├── preprocess.py       # 重采样/滑窗/转频自动估计
│   │   ├── fault_detector.py   # 模型加载与推理封装
│   │   └── monitor.py          # 实时监测引擎（后台线程 + 告警）
│   ├── routers/                # monitor / diagnose / history 路由
│   └── static/                 # 前端看板（原生 HTML/CSS/JS + 本地 ECharts）
├── training/
│   ├── generate_dataset.py     # 仿真数据 → 特征 CSV
│   ├── import_cwru.py          # CWRU 真实数据导入
│   ├── train_models.py         # 训练三模型（自动合并真实数据 + 留出验证）
│   ├── evaluate_real.py        # 迁移评估（重训前）
│   ├── experiment_holdout.py   # 消融实验
│   └── models/                 # 已训练模型（随仓库发布，克隆即用）
├── tools/
│   ├── diagnose_file.py        # 离线批量诊断 CLI
│   ├── acceptance_check.py     # 一键系统验收（44 项）
│   └── download_cwru*.ps1      # CWRU 数据自动下载（数据本身不入库）
├── data/
│   ├── raw/demo_signals.npz    # 演示信号（随仓库发布）
│   └── bearing_monitor.db      # 运行期生成（不入库）
└── tests/                      # pytest 单元 + API 端到端测试
```

## 🔬 技术原理

### 轴承故障特征频率（运动学模型）

任一部件出现局部缺陷时，缺陷每次通过接触区产生一次冲击，重复频率由几何唯一确定：

| 缺陷位置 | 特征频率 | SKF 6205 @1500rpm 参考值 |
| --- | --- | --- |
| 外圈 BPFO | (n/2)·fr·(1 − d/D·cosθ) | 89.6 Hz |
| 内圈 BPFI | (n/2)·fr·(1 + d/D·cosθ) | 135.4 Hz |
| 滚动体 2×BSF | (D/d)·fr·(1 − (d/D·cosθ)²) | 117.8 Hz |
| 保持架 FTF | (fr/2)·(1 − d/D·cosθ) | 10.0 Hz |

### 信号模型与特征

正常成分 = 转频谐波 + 宽带噪声；故障成分 = 以故障冲击频率重复的「冲击-共振衰减振荡」脉冲串，
内圈/滚动体冲击幅值受转频/保持架转频调制。诊断特征：时域（RMS/峰度/波峰因子等）+
频域（谱重心/主频/共振带能量）+ **包络谱**（共振带带通滤波 → Hilbert 包络 → FFT，
取 BPFO/BPFI/2×BSF 及谐波处归一化谱峰——故障定位的关键）。

### 模型

| 任务 | 算法 | 训练数据 |
| --- | --- | --- |
| 故障诊断（4 分类） | RandomForest ×400 树，类别平衡 | 仿真 405 + CWRU 真实 1205（DE 通道） |
| 严重度回归 | RandomForest ×300 树 | 同上（按真实缺陷直径 4 档校准） |
| 剩余寿命 RUL | GradientBoosting ×400 树 | 仿真退化轨迹 1800 条（按轨迹分组切分防泄漏） |

## 🧪 测试与验收

```bash
python -m pytest tests -q             # 31 项：信号/特征/预处理/API 端到端
python -m tools.acceptance_check      # 44 项：模型指标/60文件复测/API/监测闭环/告警去重/前端/延迟
```

## 📦 数据说明

- 仓库**不包含** CWRU 原始 .mat 数据（约 170 MB），通过
  `powershell -NoProfile -ExecutionPolicy Bypass -File tools\download_cwru.ps1`
  （及 `download_cwru_extra.ps1`）从公开镜像自动下载；
- CWRU 数据集：Case Western Reserve University Bearing Data Center，
  引用请标注 "K.A. Loparo, Case Western Reserve University Bearing Data Center"；
- 前端图表库 ECharts（Apache-2.0）已本地化，运行不依赖外部 CDN。

## 🔒 安全说明

- 本仓库**不含任何 API 密钥、令牌、密码或个人数据**；系统不依赖任何大模型/云服务，全部本地运行；
- 服务默认仅监听 `127.0.0.1`，未加鉴权，**请勿直接暴露公网**；生产部署加固清单见 [SECURITY.md](SECURITY.md)；
- 发现安全问题请通过 GitHub Issues 报告（标注 `[安全]`）。

## 📄 许可证

[MIT](LICENSE) © 2026 oyh221971

## 🙏 致谢

CWRU 轴承数据中心的公开数据集、ECharts（Apache-2.0）、FastAPI、scikit-learn 社区。
