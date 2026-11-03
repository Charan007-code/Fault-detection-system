// script.js - dashboard logic. Talks only to the Flask REST API.
const $ = (id) => document.getElementById(id);
let modelInfo = null;
let lastSample = null;
let autoTimer = null;

async function api(url, options = {}) {
  const res = await fetch(url, options);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.error || `HTTP ${res.status}`);
  return body;
}
const fmt = (v, d = 2) => (v === null || v === undefined ? "n/a" : Number(v).toFixed(d));
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

// ---------------- Charts ----------------
const MAX_POINTS = 30;
function lineChart(id, datasets, yTitle) {
  return new Chart($(id), {
    type: "line",
    data: { labels: [], datasets: datasets.map((d) => ({ ...d, data: [], tension: 0.3, spanGaps: true, pointRadius: 2 })) },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      scales: { y: { beginAtZero: true, title: { display: true, text: yTitle } } },
    },
  });
}
const latencyChart = lineChart("latencyChart", [
  { label: "Latency (ms)", borderColor: "#2a7de1" },
  { label: "Jitter (ms)", borderColor: "#d98b0b" },
], "milliseconds");
const trafficChart = lineChart("trafficChart", [
  { label: "Throughput (Mbps)", borderColor: "#1a9c5b" },
], "Mbps");
const classChart = new Chart($("classChart"), {
  type: "bar",
  data: { labels: [], datasets: [{ label: "F1-score per class (test set)", data: [], backgroundColor: "#2a7de1" }] },
  options: { responsive: true, maintainAspectRatio: false, indexAxis: "y", scales: { x: { min: 0, max: 1 } } },
});

function pushPoint(chart, label, values) {
  chart.data.labels.push(label);
  values.forEach((v, i) => chart.data.datasets[i].data.push(v));
  if (chart.data.labels.length > MAX_POINTS) {
    chart.data.labels.shift();
    chart.data.datasets.forEach((d) => d.data.shift());
  }
  chart.update();
}

// ---------------- Model info ----------------
async function loadModelInfo() {
  const info = await api("/api/model-info");
  const badge = $("model-badge");
  if (!info.loaded) {
    badge.textContent = "no model trained";
    $("model-info").textContent = info.message;
    $("predict-form").innerHTML = '<p class="muted">Train a model first.</p>';
    return;
  }
  modelInfo = info;
  badge.textContent = `model: ${info.data_source.toUpperCase()} data`;
  badge.className = `badge ${info.data_source}`;
  $("synthetic-banner").classList.toggle("hidden", info.data_source !== "synthetic");

  const m = info.metrics;
  $("model-info").innerHTML = `
    <dl class="metrics">
      <dt>Accuracy</dt><dd>${fmt(m.accuracy, 4)}</dd>
      <dt>Majority baseline</dt><dd>${fmt(m.baseline_accuracy, 4)}</dd>
      <dt>Macro precision / recall / F1</dt><dd>${fmt(m.precision_macro, 3)} / ${fmt(m.recall_macro, 3)} / ${fmt(m.f1_macro, 3)}</dd>
      <dt>Split</dt><dd>${esc(m.split_strategy)} (train ${m.n_train}, test ${m.n_test})</dd>
      <dt>Detectable classes</dt><dd>${info.classes.map(esc).join(", ")}</dd>
    </dl>
    ${(m.split_notes || []).map((n) => `<p class="warn">⚠ ${esc(n)}</p>`).join("")}`;
  classChart.data.labels = Object.keys(info.per_class);
  classChart.data.datasets[0].data = Object.values(info.per_class).map((c) => c["f1-score"]);
  classChart.update();
  $("cm-img").src = info.report_images.confusion_matrix + "?t=" + Date.now();
  $("fi-img").src = info.report_images.feature_importance + "?t=" + Date.now();
  buildForm(info);
}

function buildForm(info) {
  const form = $("predict-form");
  form.innerHTML = "";
  info.numeric_features.forEach((f) => {
    const r = info.numeric_ranges[f] || [];
    form.insertAdjacentHTML("beforeend",
      `<label>${esc(f)}<input type="number" step="any" name="${esc(f)}" placeholder="${fmt(r[0])} … ${fmt(r[1])}"></label>`);
  });
  info.categorical_features.forEach((f) => {
    const opts = (info.categorical_values[f] || []).map((v) => `<option>${esc(v)}</option>`).join("");
    form.insertAdjacentHTML("beforeend",
      `<label>${esc(f)}<select name="${esc(f)}"><option value="">(missing)</option>${opts}</select></label>`);
  });
}

// ---------------- Monitoring ----------------
const TILES = [
  ["latency_ms", "Latency (ms)"], ["jitter_ms", "Jitter (ms)"], ["packet_loss_pct", "Packet loss (%)"],
  ["throughput_mbps", "Throughput (Mbps)"], ["interface_errors_per_s", "If errors /s"],
  ["interface_drops_per_s", "If drops /s"], ["cpu_percent", "CPU (%)"], ["memory_percent", "Memory (%)"],
];
let lastPlotted = null;
function renderSample(s) {
  if (s.timestamp === lastPlotted) return;
  lastPlotted = s.timestamp;
  lastSample = s;
  $("tiles").innerHTML = TILES.map(([k, label]) =>
    `<div class="tile"><div class="k">${label}</div><div class="v">${fmt(s[k])}</div></div>`).join("") +
    `<div class="tile"><div class="k">Interfaces up</div><div class="v">${s.interfaces_up}/${s.interfaces_total}</div></div>`;
  $("monitor-method").textContent = `Method: ${s.method}` +
    (s.packet_loss_pct === null ? ` — packet loss not measurable with this method (probe failures: ${fmt(s.probe_failure_pct)}%)` : "") +
    (s.notes && s.notes.length ? ` — ${s.notes.join("; ")}` : "");
  const cov = s.model_feature_coverage;
  $("coverage").textContent = cov
    ? `Model features available from this monitor: ${cov.available.length}/${cov.available.length + cov.missing.length}` +
      (cov.can_predict ? "" : " — not enough to predict from live data (model was trained on different measurements).")
    : "";
  const t = new Date(s.timestamp).toLocaleTimeString();
  pushPoint(latencyChart, t, [s.latency_ms, s.jitter_ms]);
  pushPoint(trafficChart, t, [s.throughput_mbps]);
}

async function measure() {
  const btn = $("btn-monitor");
  btn.disabled = true;
  try {
    const host = encodeURIComponent($("host").value.trim());
    renderSample(await api(`/api/monitor?host=${host}`));
  } catch (e) {
    $("monitor-method").innerHTML = `<span class="error">${esc(e.message)}</span>`;
  } finally {
    btn.disabled = false;
  }
}

// ---------------- Prediction ----------------
function fillFromMonitor() {
  if (!lastSample) { alert("Run a measurement first."); return; }
  let filled = 0;
  for (const el of $("predict-form").elements) {
    const v = lastSample[el.name];
    if (v !== undefined && v !== null) { el.value = v; filled++; }
  }
  if (!filled) alert("None of the model's features are measured by the live monitor.");
}

async function predict() {
  const features = {};
  for (const el of $("predict-form").elements) if (el.name && el.value !== "") features[el.name] = el.value;
  const box = $("result");
  box.classList.remove("hidden");
  try {
    const r = await api("/api/predict", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ features }),
    });
    const probs = Object.entries(r.probabilities).sort((a, b) => b[1] - a[1])
      .map(([k, v]) => `${esc(k)}: ${(v * 100).toFixed(1)}%`).join(" · ");
    box.innerHTML = `
      <div class="label">${esc(r.prediction)}
        <span class="sev sev-${r.severity ?? "null"}">${esc(r.severity ?? "severity n/a")}</span></div>
      <p>Confidence: <strong>${(r.confidence * 100).toFixed(1)}%</strong> <span class="muted">(${esc(r.confidence_note)})</span></p>
      <p class="muted">${probs}</p>
      ${r.warnings.map((w) => `<p class="warn">⚠ ${esc(w)}</p>`).join("")}`;
    loadHistory();
  } catch (e) {
    box.innerHTML = `<p class="error">${esc(e.message)}</p>`;
  }
}

// ---------------- History ----------------
async function loadHistory() {
  const type = $("history-type").value;
  const data = await api(`/api/history?type=${type}&limit=50`);
  const table = $("history");
  if (!data.records.length) { table.innerHTML = '<tr><td class="muted">No records yet.</td></tr>'; return; }
  const cols = type === "predictions"
    ? ["id", "timestamp", "predicted_label", "confidence", "severity", "model_source"]
    : ["id", "timestamp", "host", "method", "latency_ms", "jitter_ms", "packet_loss_pct", "throughput_mbps", "cpu_percent"];
  table.innerHTML = `<tr>${cols.map((c) => `<th>${c}</th>`).join("")}</tr>` +
    data.records.map((r) => `<tr>${cols.map((c) => `<td>${esc(r[c] ?? "")}</td>`).join("")}</tr>`).join("");
}

// ---------------- Automatic monitoring status ----------------
async function loadStatus() {
  const s = await api("/api/status");
  const el = $("auto-status");
  if (!s.enabled) {
    el.textContent = "Off. Start with:  python automate.py serve --auto-monitor 30   (or python app.py --auto-monitor 30)";
  } else {
    const last = s.last_prediction
      ? `last auto-prediction: ${s.last_prediction.prediction} (${(s.last_prediction.confidence * 100).toFixed(0)}% votes, ${s.last_prediction.model_source} model)`
      : (s.last_skip_reason || "waiting for first run");
    el.innerHTML = `On: ${esc(s.host)} every ${s.interval_s}s · runs: ${s.runs} · last: ${esc(s.last_run || "-")}<br>${esc(last)}` +
      (s.last_error ? `<br><span class="error">${esc(s.last_error)}</span>` : "");
  }
  $("alerts").innerHTML = s.alerts.length
    ? s.alerts.map((a) => `<li><span class="sev sev-${a.severity}">${esc(a.severity)}</span> ${esc(a.timestamp)} — <strong>${esc(a.prediction)}</strong> (${(a.confidence * 100).toFixed(0)}% votes, ${esc(a.model_source)} model)</li>`).join("")
    : '<li class="muted">No high/critical alerts.</li>';
  if (s.last_sample) renderSample(s.last_sample);
  if (s.enabled) loadHistory().catch(() => {});
}

// ---------------- Wire up ----------------
$("btn-monitor").addEventListener("click", measure);
$("btn-fill").addEventListener("click", fillFromMonitor);
$("btn-predict").addEventListener("click", predict);
$("btn-history").addEventListener("click", loadHistory);
$("history-type").addEventListener("change", loadHistory);
$("auto").addEventListener("change", (e) => {
  clearInterval(autoTimer);
  if (e.target.checked) autoTimer = setInterval(measure, 10000);
});
loadModelInfo().catch((e) => ($("model-info").textContent = e.message));
loadHistory().catch(() => {});
loadStatus().catch(() => {});
setInterval(() => loadStatus().catch(() => {}), 10000);
