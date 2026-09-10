# Bearing Fault Predictive Maintenance System

A rolling-element bearing fault diagnosis and remaining useful life (RUL) prediction system built on
**physics-based simulation + machine learning + real-world data (CWRU) validation**.
Feed it a vibration signal and it automatically identifies the fault type (inner race / outer race /
rolling element), estimates severity and remaining life, with a real-time monitoring dashboard,
alert management and a REST API.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5%2B-orange)
![Tests](https://img.shields.io/badge/Tests-31%2F31%20passed-brightgreen)
![Acceptance](https://img.shields.io/badge/Acceptance-44%2F44%20passed-brightgreen)

> Tech stack: Python · FastAPI · scikit-learn · NumPy/SciPy · SQLite · ECharts (vendored locally —
> no internet or LLM API required at runtime)

**Read this in [中文](README.md).**

---

## ✨ Features

| Module | Description |
| --- | --- |
| 🎲 Physics-based signal engine | Impulse-resonance model based on SKF 6205 geometry; synthesizes 4 bearing states, arbitrary severity and full run-to-failure trajectories |
| 🧮 Feature engineering | 20 features: time-domain statistics (9) + frequency-domain (4) + Hilbert envelope-spectrum fault features (7) |
| 🤖 Three ML models | Random Forest fault classifier (4 classes) + Random Forest severity regressor + Gradient Boosting RUL regressor |
| 📡 Real-time monitor | Background thread: sample → diagnose → persist → alert; 5 scenarios switchable on the fly |
| 🚨 Alert management | Dual thresholds (fault probability / RUL), auto-resolve, auto-deduplicate, ack-able |
| 🖥 Dashboard (Chinese UI) | Health index / RUL / RMS / temperature KPIs, live waveform, spectrum with fault-frequency markers, envelope spectrum, health trend, alerts and history |
| 📤 Offline diagnosis | Dashboard file upload / CLI / REST API — arbitrary sample rates auto-resampled, **shaft speed auto-estimated** |
| 🔄 Real-data pipeline | CWRU auto-download → feature import → transfer evaluation → merged retraining → held-out validation, fully scripted |
| ✅ One-command acceptance | 44 acceptance checks + 31 unit/API tests, re-runnable after any change |

## 🎯 Scope

### ✅ Supported

- **Bearing**: SKF 6205 deep-groove ball bearing (same as the CWRU dataset), 12 kHz sampling,
  0.5 s windows (other sample rates are resampled automatically);
- **Fault types**: normal / inner-race / outer-race / rolling-element (4 classes);
- **Severity**: 0–100%, calibrated against real defect sizes
  (0.007" → 30%, 0.014" → 60%, 0.021" → 90%, 0.028" → 100%);
- **RUL**: relative lifetime trend for maintenance scheduling (see limitations below);
- **Operating conditions**: constant speed (auto-estimated when unknown); outer-race faults at
  3/6/12 o'clock load positions.

### ❌ Not supported / known limitations

1. **Not a safety-certified system** — outputs are for maintenance decision support only
   (see the disclaimer in [SECURITY.md](SECURITY.md));
2. **Other bearing models require retraining** — features are tied to the 6205 geometry; retrain
   with your own data following §7.2;
3. **RUL is inherently uncertain early in life** — a bearing near the start of its life sounds
   identical to a healthy one; single-snapshot RUL error is large (a physical limit, mitigated by
   exponential smoothing in the dashboard);
4. **Variable speed** — provide the average shaft speed; large rapid speed changes degrade
   envelope-spectrum feature localization;
5. **Fan-end (FE) sensors** — the default model is calibrated on drive-end (DE) data; FE coverage
   requires retraining with `TRAIN_USE_FE=1`;
6. **Data domain** — accuracy figures below are validated on CWRU laboratory data; always calibrate
   with your own field data (§7.2) before production use.

## 📊 Accuracy (measured on real data, reproducible)

| Evaluation | Dataset | Result |
| --- | --- | --- |
| Mixed train/test split | 405 synthetic + 1,205 real CWRU samples | 100% accuracy / severity MAE 0.011 |
| **Transfer** (synthetic-only model on real data) | CWRU 39 files, 800 windows | **66.9%** accuracy |
| **Retrained, file-grouped hold-out** (25% of files never trained on) | CWRU 60 files, 2,265 windows | **DE 90.3%** · standard-condition 88.1% · severity MAE **0.059** · normal recall 98.3% |
| Full re-test (known-data regression) | CWRU 60 files | DE 99.8%; all three clock positions @6/@3/@12 ≥ 99% |
| End-to-end spot check | 7 files (incl. conditions never seen by the model) | 7/7 correct |
| RUL (synthetic degradation runs) | 60 runs, 1,800 windows | MAE 250 h on a 2,000 h rated life, R² 0.68 — **use as a trend only** |
| System acceptance | `tools/acceptance_check.py` | 44/44 passed; per-window diagnosis latency ~84 ms mean |

> ⚠️ **Important**: these numbers are measured on CWRU laboratory data. The transfer experiment
> shows a synthetic-only model reaches just 66.9% on real data, rising to 90.3% after retraining —
> **always calibrate and validate with your own field data (§7.2) before production judgment**.

## 🚀 Quick Start

### Requirements

- Python 3.10+ (developed on 3.12)
- Windows / macOS / Linux

### Install & run (trained models are included in the repo — clone and run)

```bash
git clone https://github.com/oyh221971/bearing-predictive-maintenance-system.git
cd bearing-predictive-maintenance-system
python -m pip install -r requirements.txt
python run.py          # on Windows you can also double-click start.bat
```

Open in your browser:

- **Dashboard**: <http://127.0.0.1:8000>
- **API docs (Swagger)**: <http://127.0.0.1:8000/docs>

> If the models are missing, `run.py` automatically generates data and trains (~1 minute) before
> starting the server.

## 🖥 How to Use It Correctly

### 1️⃣ Live monitoring dashboard (start here)

1. Open the dashboard and pick a scenario from the top bar:
   - `normal` — healthy baseline;
   - `inner` / `outer` / `ball` — watch the AI switch diagnosis within seconds, the fault-frequency
     markers align in the spectrum, and alerts fire;
   - `degradation` — the full story: health index decays, RUL decreases, life alert triggers, the
     bearing "fails" and a new run starts automatically.
2. Watch the KPI cards, the AI diagnosis panel (four-class probability bars), spectrum / envelope
   spectrum / health trend charts.
3. Alerts appear and auto-resolve as conditions change; the "active alerts" panel lists them.

### 2️⃣ Offline diagnosis of your own data (the real use case)

**A · Dashboard**: use the "离线诊断 (Offline diagnosis)" panel → choose a CSV/TXT file
(single numeric column) → enter the sample rate → leave shaft frequency empty for auto-estimation
→ diagnose.

**B · CLI**:

```bash
python -m tools.diagnose_file --file vibration.csv --fs 48000        # auto shaft speed
python -m tools.diagnose_file --file vibration.txt --fs 12000 --rpm 1797 --json report.json
```

**C · REST API**:

```bash
curl -X POST http://127.0.0.1:8000/api/diagnose \
  -H "Content-Type: application/json" \
  -d '{"samples": [0.1, -0.2, ...], "sample_rate": 48000, "shaft_freq": null}'
```

**Key points**: ① use the raw **acceleration** channel (velocity/displacement signals reduce
accuracy); ② the system auto-removes DC, resamples, and windows (0.5 s windows, 50% overlap,
probability-vote aggregation) — any length works; ③ leave `shaft_freq` empty when the speed is
unknown — the system estimates it with a dual-hypothesis algorithm (harmonic-series matching +
fault-frequency ratio inference; measured error < 0.15 Hz across all 60 CWRU files).

### 3️⃣ Connecting real acquisition hardware (production direction)

Any HTTP-capable edge device (industrial PC + IEPE accelerometer, Raspberry Pi + ADXL345, PLC
gateway) can POST a frame to `/api/diagnose` every 0.5 s. Mount the sensor radially on the bearing
housing, sample at ≥ 10 kHz.

### 4️⃣ Retraining with your own data (strongly recommended)

See §7.2 — three commands: download data → `import_cwru` → `train_models`.

### 5️⃣ Routine acceptance

```bash
python -m pytest tests -q              # 31 unit/API tests
python -m tools.acceptance_check       # 44 acceptance checks (server must be running)
```

## 📁 Project Structure

```
bearing-pdm/
├── run.py                      # one-click entry (auto-train if needed, then serve)
├── start.bat                   # double-click launcher for Windows
├── requirements.txt
├── LICENSE                     # MIT
├── SECURITY.md                 # security policy & hardening checklist
├── app/
│   ├── main.py                 # FastAPI entry point
│   ├── config.py               # global config (bearing geometry / sampling / monitoring)
│   ├── database.py             # SQLite storage (samples/alerts, thread-safe)
│   ├── schemas.py              # Pydantic request/response models
│   ├── services/
│   │   ├── signal.py           # physics-based fault signal simulation + degradation
│   │   ├── features.py         # time/frequency/envelope feature extraction (20 dims)
│   │   ├── preprocess.py       # resampling / windowing / shaft-frequency auto-estimation
│   │   ├── fault_detector.py   # model loading & inference wrapper
│   │   └── monitor.py          # real-time monitoring engine (background thread + alerts)
│   ├── routers/                # monitor / diagnose / history API routes
│   └── static/                 # dashboard (vanilla HTML/CSS/JS + vendored ECharts)
├── training/
│   ├── generate_dataset.py     # simulation → feature CSVs
│   ├── import_cwru.py          # CWRU real-data import
│   ├── train_models.py         # trains the 3 models (merges real data + held-out validation)
│   ├── evaluate_real.py        # transfer evaluation (run before retraining)
│   ├── experiment_holdout.py   # ablation experiments
│   └── models/                 # trained models (shipped in the repo, ready to run)
├── tools/
│   ├── diagnose_file.py        # offline batch diagnosis CLI
│   ├── acceptance_check.py     # one-command acceptance (44 checks)
│   └── download_cwru*.ps1      # CWRU data auto-download (data itself is not in the repo)
├── data/
│   ├── raw/demo_signals.npz    # demo signals (shipped)
│   └── bearing_monitor.db      # runtime database (not shipped)
└── tests/                      # pytest unit + API end-to-end tests
```

## 🔬 How It Works

### Bearing fault characteristic frequencies (kinematic model)

When a local defect passes through the contact zone it produces one impact; the repetition rate is
uniquely determined by the bearing geometry:

| Defect location | Characteristic frequency | SKF 6205 @ 1500 rpm |
| --- | --- | --- |
| Outer race (BPFO) | (n/2)·fr·(1 − d/D·cosθ) | 89.6 Hz |
| Inner race (BPFI) | (n/2)·fr·(1 + d/D·cosθ) | 135.4 Hz |
| Rolling element (2×BSF) | (D/d)·fr·(1 − (d/D·cosθ)²) | 117.8 Hz |
| Cage (FTF) | (fr/2)·(1 − d/D·cosθ) | 10.0 Hz |

### Signal model & features

Healthy components = shaft harmonics + broadband noise; fault components = a train of
"impact-resonance" damped oscillations repeating at the fault frequency, with amplitude modulation
at shaft/cage frequency for inner-race/ball faults. Diagnostic features: time domain (RMS,
kurtosis, crest factor, …) + frequency domain (spectral centroid, dominant frequency, resonance
band energy) + **envelope spectrum** (band-pass around the resonance → Hilbert envelope → FFT,
reading the normalized peaks at BPFO/BPFI/2×BSF and their harmonics — the key to fault
localization).

### Models

| Task | Algorithm | Training data |
| --- | --- | --- |
| Fault classification (4 classes) | RandomForest ×400 trees, class-balanced | 405 synthetic + 1,205 real CWRU (DE channel) |
| Severity regression | RandomForest ×300 trees | same, calibrated to 4 real defect diameters |
| RUL regression | GradientBoosting ×400 trees | 1,800 synthetic degradation windows (group-split by run to prevent leakage) |

## 🧪 Tests & Acceptance

```bash
python -m pytest tests -q             # 31 tests: signal / features / preprocessing / API e2e
python -m tools.acceptance_check      # 44 checks: model metrics / 60-file re-test / API / monitoring loop / alert dedup / frontend / latency
```

## 📦 Data Notes

- The repo does **not** contain the raw CWRU .mat files (~170 MB); download them with
  `powershell -NoProfile -ExecutionPolicy Bypass -File tools\download_cwru.ps1`
  (plus `download_cwru_extra.ps1`) from a public mirror;
- CWRU dataset: Case Western Reserve University Bearing Data Center. If you publish work based on
  it, cite "K.A. Loparo, Case Western Reserve University Bearing Data Center";
- The chart library ECharts (Apache-2.0) is vendored — the dashboard runs fully offline.

## 🔒 Security

- This repository contains **no API keys, tokens, passwords or personal data**; the system depends
  on no external LLM/cloud service and runs fully locally;
- The server listens on `127.0.0.1` only and has no authentication — **do not expose it directly to
  the public internet**; see the hardening checklist in [SECURITY.md](SECURITY.md);
- Report security issues via GitHub Issues (tagged `[security]`).

## 📄 License

[MIT](LICENSE) © 2026 oyh221971

## 🙏 Acknowledgements

The CWRU Bearing Data Center public dataset, ECharts (Apache-2.0), and the FastAPI / scikit-learn
communities.
