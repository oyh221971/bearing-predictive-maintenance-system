"""系统验收检查：一键对运行中的系统做完整验收，输出结构化报告。

覆盖 8 大类：
  A. 模型文件与训练指标（阈值达标）
  B. 全部 60 个 CWRU 真实文件全量复测（按类别/严重度/时钟位/通道细分）
  C. API 离线诊断（仿真 4 类 + 真实 4 类 + 48kHz 重采样 + 长信号多窗口）
  D. API 边界（过短信号拒绝、非法场景拒绝）
  E. 实时监测闭环（采样→诊断→告警去重→确认→停止）
  F. 频谱/历史/模型信息/健康检查接口
  G. 前端页面与静态资源
  H. 诊断延迟

用法：
    python run.py                    # 先启动服务
    python -m tools.acceptance_check # 再运行验收
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly

BASE = "http://127.0.0.1:8000"
CWru_DIR = Path("data/cwru")

RESULTS: list[tuple[str, bool, str]] = []


def _console_utf8():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"  [{'✅ PASS' if ok else '❌ FAIL'}] {name}" + (f"  — {detail}" if detail else ""))


def http_json(path: str, method: str = "GET", body: dict | None = None,
             timeout: float = 60) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_status(path: str, method: str = "GET", body: dict | None = None,
                timeout: float = 60) -> int:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code


# ============================================================ A. 模型与指标
def section_a() -> None:
    print("A. 模型文件与训练指标")
    from app.config import MODEL_DIR
    files = ("fault_classifier", "severity_regressor", "rul_regressor",
             "label_encoder")
    missing = [f for f in files if not (MODEL_DIR / f"{f}.joblib").exists()]
    check("A1 四个模型文件齐全", not missing, f"缺失: {missing}" if missing else "")

    meta = json.loads((MODEL_DIR / "meta.json").read_text("utf-8"))
    m = meta["metrics"]
    check("A2 混合测试准确率 ≥95%", m["classification_accuracy"] >= 0.95,
          f"{m['classification_accuracy']:.1%}")
    rd = m.get("real_data", {})
    check("A3 真实数据留出验证(DE) ≥85%", rd.get("accuracy_de_only", 0) >= 0.85,
          f"{rd.get('accuracy_de_only', 0):.1%}")
    check("A4 真实数据严重度MAE ≤0.10", rd.get("severity_mae", 9) <= 0.10,
          f"{rd.get('severity_mae', 0):.3f}")
    check("A5 真实数据参与训练样本 ≥1200(DE校准)",
          meta["data"]["real_samples"] >= 1200,
          f"{meta['data']['real_samples']} 条")
    check("A6 特征维度=20", meta["n_features"] == 20, f"{meta['n_features']}")


# ============================================ B. CWRU 全量复测
def _clock_of(name: str) -> str:
    if name.startswith(("144_", "145_", "146_", "147_",
                        "246_", "247_", "248_", "249_")):
        return "@3"
    if name.startswith(("156_", "258_", "259_", "260_", "261_")):
        return "@12"
    if "@6" in name or name.startswith(("234_", "235_", "236_", "237_")):
        return "@6"
    return "-"


def section_b() -> None:
    print("B. CWRU 真实数据全量复测（60 文件）")
    from training.import_cwru import (_find_channel_keys, _infer_sample_rate,
                                      identify_file, load_rpm)
    from app.services.fault_detector import DiagnosticModels

    models = DiagnosticModels()
    recs = []  # (name, channel, truth, severity, n_windows, n_correct)
    for path in sorted(CWru_DIR.glob("*.mat")):
        info = identify_file(path.name)
        if info is None:
            continue
        truth, sev = info
        mat = loadmat(str(path))
        de_key, fe_key = _find_channel_keys(mat)
        shaft = load_rpm(mat) / 60.0
        for ch, key in (("DE", de_key), ("FE", fe_key)):
            if key is None:
                continue
            x = np.asarray(mat[key], dtype=np.float64).ravel()
            fs = _infer_sample_rate(str(key), path, len(x))
            if abs(fs - 12000) > 120:
                g = np.gcd(12000, int(fs))
                x = resample_poly(x, 12000 // g, int(fs) // g)
            windows = [x[i:i + 6000] for i in range(0, len(x) - 6000 + 1, 6000)]
            correct = sum(
                models.predict(w, 12000.0, shaft)["fault_type"] == truth
                for w in windows)
            recs.append((path.name, ch, truth, sev, len(windows), correct))

    n_files = len({r[0] for r in recs})
    total, correct = sum(r[4] for r in recs), sum(r[5] for r in recs)
    check("B1 复测文件数=60", n_files == 60, f"{n_files} 个")

    de = [r for r in recs if r[1] == "DE"]
    acc_de = sum(r[5] for r in de) / sum(r[4] for r in de)
    check("B2 DE通道准确率 ≥85%", acc_de >= 0.85, f"{acc_de:.1%}")

    fe = [r for r in recs if r[1] == "FE"]
    acc_fe = sum(r[5] for r in fe) / max(1, sum(r[4] for r in fe))
    check("B2b FE通道为可选覆盖(默认DE校准模型，仅信息项)", acc_fe >= 0.25,
          f"FE准确率 {acc_fe:.1%}（需 TRAIN_USE_FE=1 重训才覆盖）")

    # 按类别
    parts = []
    for cls in ("normal", "inner", "outer", "ball"):
        sub = [r for r in de if r[2] == cls]
        a = sum(r[5] for r in sub) / max(1, sum(r[4] for r in sub))
        parts.append(f"{cls}={a:.0%}")
    check("B4 DE各类别准确率均 ≥80%",
          all(sum(r[5] for r in de if r[2] == c)
              / max(1, sum(r[4] for r in de if r[2] == c)) >= 0.80
              for c in ("normal", "inner", "outer", "ball")),
          " ".join(parts))

    # 按严重度
    parts = []
    for sev in (0.0, 0.30, 0.60, 0.90, 1.00):
        sub = [r for r in de if abs(r[3] - sev) < 1e-9]
        a = sum(r[5] for r in sub) / max(1, sum(r[4] for r in sub))
        parts.append(f"{sev:.2f}={a:.0%}")
    check("B5 DE各严重度档准确率均 ≥75%",
          all(sum(r[5] for r in de if abs(r[3] - s) < 1e-9)
              / max(1, sum(r[4] for r in de if abs(r[3] - s) < 1e-9)) >= 0.75
              for s in (0.0, 0.30, 0.60, 0.90, 1.00)),
          " ".join(parts))

    # 外圈按时钟位
    parts = []
    for clk in ("@6", "@3", "@12"):
        sub = [r for r in de if r[2] == "outer" and _clock_of(r[0]) == clk]
        a = sum(r[5] for r in sub) / max(1, sum(r[4] for r in sub))
        parts.append(f"{clk}={a:.0%}")
    check("B6 外圈三种时钟位均 ≥70%",
          all(sum(r[5] for r in de if r[2] == "outer" and _clock_of(r[0]) == c)
              / max(1, sum(r[4] for r in de
                           if r[2] == "outer" and _clock_of(r[0]) == c)) >= 0.70
              for c in ("@6", "@3", "@12")),
          " ".join(parts))

    # 最差文件 Top3
    file_acc = {}
    for name, ch, truth, sev, n, c in de:
        file_acc.setdefault(name, [0, 0])
        file_acc[name][0] += n
        file_acc[name][1] += c
    worst = sorted(((a[1] / a[0], n) for n, a in file_acc.items()))[:3]
    check("B7 最差文件准确率 >50%",
          all(a > 0.5 for a, _ in worst),
          "；".join(f"{n}={a:.0%}" for a, n in worst))


# ==================================================== C. API 离线诊断
def section_c() -> None:
    print("C. API 离线诊断")
    from app.services import signal as sig

    def diagnose(x, fs=12000, freq=25.0):
        return http_json("/api/diagnose", "POST",
                         {"samples": x.tolist(), "sample_rate": fs,
                          "shaft_freq": freq})

    cases = [("normal", 0.0, 12000, 25.0), ("inner", 0.7, 12000, 25.0),
             ("outer", 0.7, 12000, 25.0), ("ball", 0.7, 12000, 25.0),
             ("inner", 0.7, 48000, 25.0), ("ball", 0.7, 24000, 25.0)]
    ok_all = True
    for truth, sev, fs, freq in cases:
        x = sig.simulate_signal(truth, sev, sample_rate=fs, seed=8,
                                n_points=6000 if fs == 12000 else 12000)
        r = diagnose(x, fs, freq)
        ok = r["fault_type"] == truth
        ok_all &= ok
        check(f"C 仿真{truth}@{fs // 1000}k", ok,
              f"预测={r['fault_type']} 置信度={r['confidence']:.0%}")

    # 真实文件：转频留空 → 自动识别（同时验证自动转速估计）
    real_cases = [("105_0.mat", "inner", 12000), ("234_0.mat", "outer", 12000),
                  ("185_0.mat", "ball", 12000), ("97_Normal_0.mat", "normal", 48000)]
    for name, truth, fs in real_cases:
        mat = loadmat(str(CWru_DIR / name))
        key = [k for k in mat if k.endswith("_DE_time")][0]
        x = np.asarray(mat[key]).ravel()
        if abs(fs - 12000) > 120:
            g = np.gcd(12000, fs)
            x = resample_poly(x, 12000 // g, fs // g)
        r = http_json("/api/diagnose", "POST",
                      {"samples": x[:24000].tolist(), "sample_rate": 12000,
                       "shaft_freq": None})
        est = r.get("estimated_shaft_freq", 0)
        ok = r["fault_type"] == truth and abs(est - 29.95) < 0.5
        ok_all &= ok
        check(f"C 真实{name}(自动转频)", ok,
              f"预测={r['fault_type']} 置信度={r['confidence']:.0%}"
              f" 转频估计={est}Hz")

    # 长信号多窗口
    parts = [sig.simulate_signal("outer", 0.8, seed=i) for i in range(4)]
    r = diagnose(np.concatenate(parts))
    ok = r["fault_type"] == "outer" and r["n_windows"] == 7
    ok_all &= ok
    check("C 长信号滑窗聚合(7窗口)", ok, f"n_windows={r['n_windows']}")
    check("C API诊断用例汇总", ok_all)


# ==================================================== D. 边界
def section_d() -> None:
    print("D. API 边界处理")
    s1 = http_status("/api/diagnose", "POST", {"samples": [0.0] * 100})
    check("D1 过短信号返回400", s1 == 400, f"HTTP {s1}")
    s2 = http_status("/api/monitor/start", "POST", {"scenario": "bogus"})
    check("D2 非法场景返回400", s2 == 400, f"HTTP {s2}")
    s3 = http_status("/api/diagnose/demo?fault_type=bogus")
    check("D3 非法演示类型返回400", s3 == 400, f"HTTP {s3}")


# ============================================== E. 监测闭环
def section_e() -> None:
    print("E. 实时监测闭环")
    http_json("/api/monitor/stop", "POST")
    r = http_json("/api/monitor/start", "POST", {"scenario": "inner"})
    check("E1 场景切换生效", r["scenario"] == "inner", f"scenario={r['scenario']}")

    deadline = time.time() + 12
    status = None
    while time.time() < deadline:
        status = http_json("/api/monitor/status")
        if (status["sample_count"] >= 1
                and status["last"]["scenario"] == "inner"):
            break
        time.sleep(0.5)
    check("E2 采样与实时诊断产出", status["last"]["scenario"] == "inner",
          f"诊断={status['last']['fault_type']} 置信度={status['last']['confidence']:.0%}")

    deadline = time.time() + 12
    while time.time() < deadline:
        status = http_json("/api/monitor/status")
        if len(status["alerts"]) >= 1:
            break
        time.sleep(0.5)
    inner_alerts = [a for a in status["alerts"] if "内圈" in a["message"]]
    check("E3 故障告警触发且去重(恰好1条)", len(inner_alerts) == 1,
          f"告警数={len(inner_alerts)} [{inner_alerts[0]['level']}]")

    ack = http_json(f"/api/monitor/alerts/{inner_alerts[0]['id']}/ack", "POST")
    check("E4 告警确认", ack.get("acked") is True, f"id={ack.get('id')}")

    spec = http_json("/api/monitor/spectrum")
    ok = all(k in spec for k in ("wave", "freq", "env_amp", "markers"))
    check("E5 频谱接口(波形/频谱/包络谱/标记)", ok,
          f"markers={ {k: round(v, 1) for k, v in spec['markers'].items() if k != 'shaft'} }")

    http_json("/api/monitor/stop", "POST")
    check("E6 监测停止", not http_json("/api/monitor/status")["running"])


# ============================================== F. 其它接口
def section_f() -> None:
    print("F. 其它接口")
    h = http_json("/api/health")
    check("F1 健康检查", h["status"] == "ok" and h["models_ready"],
          f"status={h['status']} models_ready={h['models_ready']}")
    s = http_json("/api/monitor/samples?limit=5")
    check("F2 样本历史", len(s["items"]) >= 1, f"{len(s['items'])} 条")
    hs = http_json("/api/history/summary")
    check("F3 历史统计", hs["total_samples"] >= 1,
          f"总样本={hs['total_samples']}")
    mi = http_json("/api/models/info")
    check("F4 模型信息含真实数据指标", "real_data" in mi["metrics"],
          f"DE={mi['metrics']['real_data'].get('accuracy_de_only', 0):.1%}")


# ============================================== G. 前端
def section_g() -> None:
    print("G. 前端页面与静态资源")
    import urllib.request as ur
    html = ur.urlopen(BASE + "/", timeout=30).read().decode("utf-8")
    ids = ("kpiHi", "chartWave", "chartSpec", "chartEnv", "chartTrend",
           "fileInput", "alertList", "historyTable", "scenario")
    check("G1 看板页面核心元素齐全",
          all(f'id="{i}"' in html for i in ids),
          f"缺失: {[i for i in ids if f'id=\"{i}\"' not in html]}")
    for path in ("/static/vendor/echarts.min.js", "/static/css/style.css",
                 "/static/js/app.js"):
        s = http_status(path)
        check(f"G2 静态资源 {path}", s == 200, f"HTTP {s}")
    check("G3 Swagger 文档", http_status("/docs") == 200)


# ============================================== H. 延迟
def section_h() -> None:
    print("H. 诊断延迟")
    from app.services import signal as sig
    x = sig.simulate_signal("inner", 0.7, seed=9)
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        http_json("/api/diagnose", "POST",
                  {"samples": x.tolist(), "sample_rate": 12000,
                   "shaft_freq": 25.0})
        times.append(time.perf_counter() - t0)
    avg = sum(times) / len(times)
    check("H1 单窗口诊断平均延迟 <2s", avg < 2.0,
          f"avg={avg * 1000:.0f}ms  max={max(times) * 1000:.0f}ms")


def main() -> None:
    _console_utf8()
    print("=" * 68)
    print("  轴承故障预测性维护系统 · 验收报告")
    print(f"  目标服务: {BASE}    时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 68)

    try:
        http_json("/api/health", timeout=10)
    except Exception:
        print("  [❌] 服务未启动！请先运行: python run.py")
        sys.exit(2)

    section_a()
    section_b()
    section_c()
    section_d()
    section_e()
    section_f()
    section_g()
    section_h()

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    failed = [n for n, ok, _ in RESULTS if not ok]
    print("=" * 68)
    print(f"  验收结果: {passed}/{len(RESULTS)} 项通过")
    if failed:
        print(f"  未通过项: {failed}")
    verdict = "✅ 验收通过" if not failed else "❌ 存在未通过项，请修复后重跑"
    print(f"  结论: {verdict}")
    print("=" * 68)
    # 验收结束恢复演示场景
    try:
        http_json("/api/monitor/start", "POST", {"scenario": "degradation"})
        print("  (已恢复'全寿命退化'演示场景)")
    except Exception:
        pass
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
