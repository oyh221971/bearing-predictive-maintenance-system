/* ===== 轴承故障预测性维护系统 · 前端看板逻辑 ===== */
(function () {
  "use strict";

  // ---------- 常量 ----------
  const FAULT_ZH = { normal: "正常", inner: "内圈故障", outer: "外圈故障", ball: "滚动体故障" };
  const SCENARIO_ZH = {
    normal: "正常运转", inner: "内圈故障", outer: "外圈故障",
    ball: "滚动体故障", degradation: "全寿命退化",
  };
  const COLORS = {
    green: "#22c55e", orange: "#f59e0b", red: "#ef4444", blue: "#3d8bfd",
    purple: "#a78bfa", cyan: "#22d3ee",
  };
  const POLL_MS = 2000;

  // ---------- DOM ----------
  const $ = (id) => document.getElementById(id);
  const el = {
    statusDot: $("statusDot"), statusText: $("statusText"),
    scenario: $("scenario"), btnStart: $("btnStart"), btnStop: $("btnStop"),
    kpiHi: $("kpiHi"), kpiHiBar: $("kpiHiBar"),
    kpiRul: $("kpiRul"), kpiRulSub: $("kpiRulSub"),
    kpiRms: $("kpiRms"), kpiTemp: $("kpiTemp"),
    kpiState: $("kpiState"), kpiStateSub: $("kpiStateSub"),
    diagType: $("diagType"), diagConf: $("diagConf"),
    probBars: $("probBars"), severityValue: $("severityValue"), severityBar: $("severityBar"),
    alertList: $("alertList"), historyBody: document.querySelector("#historyTable tbody"),
    fileInput: $("fileInput"), fsInput: $("fsInput"), freqInput: $("freqInput"),
    btnDiagnose: $("btnDiagnose"), offlineResult: $("offlineResult"),
  };

  // ---------- ECharts 图表 ----------
  const charts = {};
  const AXIS_STYLE = {
    axisLine: { lineStyle: { color: "#2c3e50" } },
    axisLabel: { color: "#7d93ab" },
    splitLine: { lineStyle: { color: "#1d2b3a" } },
  };

  function initChart(id) {
    const dom = $(id);
    if (!dom) return null;
    const c = echarts.init(dom, null, { renderer: "canvas" });
    charts[id] = c;
    return c;
  }

  function baseOption(series, extra) {
    return Object.assign({
      backgroundColor: "transparent",
      grid: { left: 55, right: 20, top: 25, bottom: 30 },
      tooltip: { trigger: "axis", backgroundColor: "#1a2634", borderColor: "#243447", textStyle: { color: "#dbe6f1" } },
      series: series,
    }, extra || {});
  }

  function initWaveChart() {
    return initChart("chartWave").setOption(baseOption([{
      type: "line", name: "振动加速度", showSymbol: false, data: [],
      lineStyle: { color: COLORS.cyan, width: 1.2 },
      areaStyle: { color: "rgba(34,211,238,0.08)" },
    }], {
      xAxis: Object.assign({ type: "value", name: "ms", min: 0, max: 500, nameTextStyle: { color: "#7d93ab" } }, AXIS_STYLE),
      yAxis: Object.assign({ type: "value", name: "m/s²", nameTextStyle: { color: "#7d93ab" } }, AXIS_STYLE),
    }));
  }

  function markLines(spec) {
    const m = spec.markers || {};
    const lines = [];
    const push = (name, value, color) => value != null && lines.push({
      name, xAxis: value, label: { formatter: name, color, fontSize: 10 },
      lineStyle: { color, type: "dashed", width: 1 },
    });
    push("BPFO", m.bpfo, COLORS.orange);
    push("BPFI", m.bpfi, COLORS.red);
    push("2×BSF", m.bsf, COLORS.purple);
    (m.shaft || []).forEach((f, i) => push(`f${i + 1}`, f, "#4b5c6e"));
    return lines;
  }

  function initSpecChart() {
    return initChart("chartSpec").setOption(baseOption([{
      type: "line", name: "功率谱", showSymbol: false, data: [],
      lineStyle: { color: COLORS.blue, width: 1.2 }, markLine: { silent: true, symbol: "none", data: [] },
    }], {
      xAxis: Object.assign({ type: "value", name: "Hz", min: 0, max: 6000, nameTextStyle: { color: "#7d93ab" } }, AXIS_STYLE),
      yAxis: Object.assign({ type: "value", name: "归一化功率", nameTextStyle: { color: "#7d93ab" } }, AXIS_STYLE),
    }));
  }

  function initEnvChart() {
    return initChart("chartEnv").setOption(baseOption([{
      type: "line", name: "包络谱", showSymbol: false, data: [],
      lineStyle: { color: COLORS.purple, width: 1.2 }, markLine: { silent: true, symbol: "none", data: [] },
    }], {
      xAxis: Object.assign({ type: "value", name: "Hz", min: 0, max: 1000, nameTextStyle: { color: "#7d93ab" } }, AXIS_STYLE),
      yAxis: Object.assign({ type: "value", name: "归一化幅值", nameTextStyle: { color: "#7d93ab" } }, AXIS_STYLE),
    }));
  }

  function initTrendChart() {
    return initChart("chartTrend").setOption(baseOption([
      {
        type: "line", name: "剩余寿命 (h)", showSymbol: false, data: [],
        lineStyle: { color: COLORS.blue, width: 1.6 },
      },
      {
        type: "line", name: "健康指数", showSymbol: false, data: [], yAxisIndex: 1,
        lineStyle: { color: COLORS.green, width: 1.6 },
        areaStyle: { color: "rgba(34,197,94,0.08)" },
      },
    ], {
      xAxis: Object.assign({ type: "category", data: [], axisLabel: { color: "#7d93ab", fontSize: 10 } }, AXIS_STYLE),
      yAxis: [
        Object.assign({ type: "value", name: "小时", min: 0, nameTextStyle: { color: "#7d93ab" } }, AXIS_STYLE),
        Object.assign({ type: "value", name: "健康指数", min: 0, max: 1, splitLine: { show: false } }, AXIS_STYLE),
      ],
      legend: { textStyle: { color: "#7d93ab" }, top: 0 },
    }));
  }

  function fmtTime(ts) {
    const d = new Date(ts * 1000);
    const p = (v) => String(v).padStart(2, "0");
    return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
  }

  // ---------- 状态更新 ----------
  let lastSampleId = null;

  function setConnection(ok, text) {
    el.statusDot.className = "status-dot " + (ok ? "on" : "err");
    el.statusText.textContent = text;
  }

  function updateKpi(last) {
    const hi = last ? last.health_index : null;
    el.kpiHi.textContent = hi == null ? "--" : hi.toFixed(2);
    el.kpiHiBar.style.width = (hi == null ? 0 : Math.round(hi * 100)) + "%";
    el.kpiHiBar.style.background = hi == null ? COLORS.blue : (hi > 0.6 ? COLORS.green : hi > 0.3 ? COLORS.orange : COLORS.red);

    if (last) {
      const rul = last.rul_hours;
      el.kpiRul.textContent = rul > 999 ? Math.round(rul) + " h" : rul.toFixed(0) + " h";
      const pct = Math.round(last.health_index * 100);
      el.kpiRulSub.textContent = `额定寿命 2000 h · 已消耗约 ${100 - pct}%`;
      el.kpiRms.textContent = last.rms.toFixed(3);
      el.kpiTemp.textContent = last.temperature.toFixed(1);
    } else {
      el.kpiRul.textContent = "--"; el.kpiRulSub.textContent = "小时（模型估计）";
      el.kpiRms.textContent = "--"; el.kpiTemp.textContent = "--";
    }
  }

  function updateStateCard(last, status) {
    const card = el.kpiState, sub = el.kpiStateSub;
    card.className = "card-value state-value";
    if (!last) { card.textContent = "等待数据"; sub.textContent = "正在采集首个样本…"; return; }
    const { fault_type: ft, severity, rul_hours: rul, confidence } = last;
    if (rul < 50 || severity >= 70) {
      card.classList.add("crit"); card.textContent = "严重告警"; sub.textContent = "建议立即停机检修";
    } else if (ft !== "normal" && confidence >= 0.6) {
      card.classList.add("warn"); card.textContent = "故障预警"; sub.textContent = "建议安排检修计划";
    } else if (rul < 200) {
      card.classList.add("warn"); card.textContent = "寿命预警"; sub.textContent = "剩余寿命不足 200 h";
    } else {
      card.classList.add("ok"); card.textContent = "运行正常"; sub.textContent = status.scenario === "degradation" ? "全寿命退化模拟中" : "未发现明显异常";
    }
  }

  function updateDiagnosis(last) {
    if (!last) {
      el.diagType.textContent = "等待数据…"; el.diagType.className = "diag-type";
      el.diagConf.textContent = ""; el.probBars.innerHTML = "";
      el.severityValue.textContent = "--"; el.severityBar.style.width = "0%";
      return;
    }
    const zh = FAULT_ZH[last.fault_type] || last.fault_type;
    el.diagType.textContent = zh;
    const sev = last.severity;
    el.diagType.className = "diag-type " +
      (last.fault_type === "normal" ? "normal" : sev >= 70 ? "crit" : "fault");
    el.diagConf.textContent = `置信度 ${(last.confidence * 100).toFixed(1)}%`;

    const order = ["normal", "inner", "outer", "ball"];
    el.probBars.innerHTML = "";
    order.forEach((k) => {
      const p = (last.probs && last.probs[k]) || 0;
      const row = document.createElement("div");
      row.className = "prob-row";
      row.innerHTML = `
        <span class="prob-name">${FAULT_ZH[k]}</span>
        <div class="prob-track"><div class="prob-fill" style="width:${(p * 100).toFixed(1)}%"></div></div>
        <span class="prob-pct">${(p * 100).toFixed(1)}%</span>`;
      el.probBars.appendChild(row);
    });

    el.severityValue.textContent = sev.toFixed(0) + "%";
    el.severityBar.style.width = sev.toFixed(0) + "%";
    el.severityBar.className = "bar-fill sev" + (sev >= 70 ? " high" : "");
  }

  function updateAlerts(alerts) {
    if (!alerts || alerts.length === 0) {
      el.alertList.innerHTML = '<li class="alert-empty">暂无告警</li>';
      return;
    }
    el.alertList.innerHTML = "";
    alerts.forEach((a) => {
      const li = document.createElement("li");
      li.className = a.level || "info";
      li.innerHTML = `${a.message}<span class="alert-time">${fmtTime(a.ts)}</span>`;
      el.alertList.appendChild(li);
    });
  }

  function updateHistory(items) {
    const body = el.historyBody;
    body.innerHTML = "";
    items.slice(0, 20).forEach((s) => {
      const tr = document.createElement("tr");
      const zh = FAULT_ZH[s.fault_type] || s.fault_type;
      const tagCls = s.fault_type === "normal" ? "normal" : "inner";
      tr.innerHTML = `
        <td>${fmtTime(s.ts)}</td>
        <td>${SCENARIO_ZH[s.scenario] || s.scenario}</td>
        <td><span class="tag ${tagCls}">${zh}</span></td>
        <td>${(s.confidence * 100).toFixed(1)}%</td>
        <td>${s.severity.toFixed(0)}%</td>
        <td>${s.rul_hours.toFixed(0)} h</td>
        <td>${s.health_index.toFixed(2)}</td>
        <td>${s.rms.toFixed(3)}</td>
        <td>${s.temperature.toFixed(1)} °C</td>`;
      body.appendChild(tr);
    });
  }

  // ---------- 数据拉取 ----------
  async function fetchJson(url, options) {
    const res = await fetch(url, options);
    if (!res.ok) throw new Error(`${url} -> HTTP ${res.status}`);
    return res.json();
  }

  async function pollStatus() {
    const status = await fetchJson("/api/monitor/status");
    setConnection(true, status.running ? `监测运行中 · 场景：${SCENARIO_ZH[status.scenario] || status.scenario} · 样本 ${status.sample_count}` : "监测已停止");
    el.scenario.value = status.scenario;

    const last = status.last;
    updateKpi(last);
    updateDiagnosis(last);
    updateStateCard(last, status);
    updateAlerts(status.alerts);

    // 新样本到达 → 刷新波形/频谱/包络谱
    if (last && last.id !== lastSampleId) {
      lastSampleId = last.id;
      try {
        const spec = await fetchJson("/api/monitor/spectrum");
        charts.chartWave.setOption({
          series: [{ data: spec.wave_t.map((t, i) => [t, spec.wave[i]]) }],
        });
        charts.chartSpec.setOption({
          series: [{
            data: spec.freq.map((f, i) => [f, spec.amp[i]]),
            markLine: { data: markLines(spec) },
          }],
        });
        charts.chartEnv.setOption({
          series: [{
            data: spec.env_freq.map((f, i) => [f, spec.env_amp[i]]),
            markLine: { data: markLines(spec) },
          }],
        });
      } catch (e) { /* 首个样本尚在计算频谱时忽略 */ }
    }
  }

  async function pollHistory() {
    const data = await fetchJson("/api/monitor/samples?limit=120");
    const items = data.items.slice().reverse(); // 时间正序
    updateHistory(data.items);

    const labels = items.map((s) => fmtTime(s.ts));
    charts.chartTrend.setOption({
      xAxis: { data: labels },
      series: [
        { data: items.map((s) => s.rul_hours.toFixed(0)) },
        { data: items.map((s) => s.health_index.toFixed(3)) },
      ],
    });
  }

  // ---------- 轮询主循环 ----------
  async function loop() {
    try {
      await pollStatus();
      await pollHistory();
    } catch (e) {
      setConnection(false, "后端连接断开，正在重试…");
    }
  }

  // ---------- 事件 ----------
  el.btnStart.addEventListener("click", async () => {
    try {
      await fetchJson("/api/monitor/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ scenario: el.scenario.value }),
      });
    } catch (e) { setConnection(false, "启动失败"); }
  });
  el.btnStop.addEventListener("click", async () => {
    try { await fetchJson("/api/monitor/stop", { method: "POST" }); }
    catch (e) { setConnection(false, "停止失败"); }
  });

  // ---------- 离线诊断（上传真实数据文件） ----------
  el.btnDiagnose.addEventListener("click", runOfflineDiagnose);

  function parseSignalText(text) {
    const vals = [];
    text.split(/\r?\n/).forEach((line) => {
      const t = line.trim();
      if (!t || t[0] === "#" || t[0] === "%") return;
      t.replace(/,/g, " ").replace(/;/g, " ").split(/\s+/).forEach((tok) => {
        const v = parseFloat(tok);
        if (Number.isFinite(v)) vals.push(v);
      });
    });
    return vals;
  }

  async function runOfflineDiagnose() {
    const result = el.offlineResult;
    const file = el.fileInput.files[0];
    if (!file) { result.textContent = "请先选择数据文件"; return; }
    result.textContent = `正在解析 ${file.name} …`;
    try {
      const text = await file.text();
      let vals = parseSignalText(text);
      if (vals.length < 512) {
        result.textContent = "解析出的数据点不足 512 个，请检查文件格式（应为单列数值）";
        return;
      }
      if (vals.length > 600000) {   // 控制请求体积，超长信号抽稀
        const step = Math.ceil(vals.length / 600000);
        vals = vals.filter((_, i) => i % step === 0);
      }
      result.textContent = `已解析 ${vals.length} 个数据点，正在逐窗口诊断…`;
      const freqVal = parseFloat(el.freqInput.value);
      const body = await fetchJson("/api/diagnose", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          samples: vals,
          sample_rate: parseFloat(el.fsInput.value) || 12000,
          shaft_freq: Number.isFinite(freqVal) && freqVal > 0 ? freqVal : null,
        }),
      });
      renderOfflineResult(body, file.name, vals.length);
    } catch (e) {
      result.textContent = "诊断失败：" + (e && e.message ? e.message : e);
    }
  }

  function renderOfflineResult(r, fileName, nPoints) {
    const result = el.offlineResult;
    const zh = FAULT_ZH[r.fault_type] || r.fault_type;
    const color = r.fault_type === "normal" ? COLORS.green
      : r.severity >= 70 ? COLORS.red : COLORS.orange;
    const autoFreq = r.estimated_shaft_freq != null
      ? ` · 自动识别转频 ${r.estimated_shaft_freq} Hz` : "";
    let html = `<div class="offline-summary">${fileName}（${nPoints} 点 → ${r.n_windows} 个诊断窗口`
      + `${r.padded ? "，数据不足已补零" : ""}）→ 诊断结论：`
      + `<b style="color:${color}">${zh}</b> · 置信度 ${(r.confidence * 100).toFixed(1)}%`
      + ` · 严重度 ${r.severity.toFixed(0)}% · 剩余寿命 ${r.rul_hours.toFixed(0)} h`
      + autoFreq + `</div>`;
    html += "<table><thead><tr><th>窗口</th><th>诊断</th><th>置信度</th>"
      + "<th>严重度</th><th>RUL (h)</th></tr></thead><tbody>";
    (r.windows || []).slice(0, 20).forEach((w) => {
      html += `<tr><td>${w.index}</td><td>${FAULT_ZH[w.fault_type] || w.fault_type}</td>`
        + `<td>${(w.confidence * 100).toFixed(1)}%</td><td>${w.severity.toFixed(0)}%</td>`
        + `<td>${w.rul_hours.toFixed(0)}</td></tr>`;
    });
    html += "</tbody></table>";
    result.innerHTML = html;
  }

  window.addEventListener("resize", () => {
    Object.values(charts).forEach((c) => c && c.resize());
  });

  // ---------- 启动 ----------
  initWaveChart(); initSpecChart(); initEnvChart(); initTrendChart();
  loop();
  setInterval(loop, POLL_MS);
})();
