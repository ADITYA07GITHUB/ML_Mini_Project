/**
 * NetGuard AI - Application Controller & User Interface Logic
 */

import { NetGuardAPI } from "./api.js";
import { NetGuardCharts } from "./charts.js";

const state = {
  activeView: "overview",
  currentMode: "binary",
  currentModel: "Random Forest",
  metadata: null,
  models: [],
  metrics: null,
  featureImportance: null,
  uploadedDataset: null,
  lastAnalysis: null,
  selectedRecordForExplain: null,
  filterType: "all",
};

// DOM Content Loaded Initializer
document.addEventListener("DOMContentLoaded", async () => {
  setupNavigation();
  setupEventHandlers();
  await initializeApp();
});

async function initializeApp() {
  try {
    // 1. Fetch System Health & Metadata
    const health = await NetGuardAPI.getHealth();
    document.getElementById("healthStatus").textContent = health.models_ready ? "Models Online" : "Needs Training";

    state.metadata = await NetGuardAPI.getMetadata();
    state.models = await NetGuardAPI.getModels();

    // 2. Populate Model Selectors
    populateModelSelectors();

    // 3. Load Metrics & Feature Importance for default mode
    await loadModeMetrics(state.currentMode);

    // 4. Automatically load sample test data for instant demonstration
    await autoLoadSample();

    // 5. Update Overview Dashboard
    renderOverview();

  } catch (err) {
    console.error("Initialization error:", err);
    showNotification("Failed to initialize dashboard data: " + err.message, "danger");
  }
}

function setupNavigation() {
  const navBtns = document.querySelectorAll(".nav-btn");
  navBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      const targetView = btn.dataset.view;
      switchView(targetView);
    });
  });
}

function switchView(viewName) {
  state.activeView = viewName;
  document.querySelectorAll(".nav-btn").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.view === viewName);
  });
  document.querySelectorAll(".page-view").forEach(view => {
    view.classList.toggle("active", view.id === `view-${viewName}`);
  });

  // Trigger view-specific re-renders
  if (viewName === "performance") {
    renderPerformanceView();
  } else if (viewName === "explainability") {
    renderExplainabilityView();
  } else if (viewName === "overview") {
    renderOverview();
  }
}

function populateModelSelectors() {
  const selector = document.getElementById("modelSelector");
  if (!selector) return;
  selector.innerHTML = "";
  state.models.forEach(m => {
    const opt = document.createElement("option");
    opt.value = m.name;
    opt.textContent = `${m.name} (${m.category})`;
    if (m.name === state.currentModel) opt.selected = true;
    selector.appendChild(opt);
  });
}

async function loadModeMetrics(mode) {
  try {
    state.metrics = await NetGuardAPI.getMetrics(mode);
    state.featureImportance = await NetGuardAPI.getFeatureImportance(mode);
  } catch (e) {
    console.warn(`Could not load metrics for ${mode}:`, e);
  }
}

async function autoLoadSample() {
  try {
    const sample = await NetGuardAPI.loadSampleData();
    state.uploadedDataset = sample;
    updateDatasetInfoBadge(sample.filename, sample.total_records);
    renderTrafficPreview(sample);
  } catch (e) {
    console.warn("Could not load sample data:", e);
  }
}

function updateDatasetInfoBadge(filename, count) {
  const badge = document.getElementById("activeDatasetBadge");
  if (badge) badge.textContent = `${filename} (${count} flows)`;
}

/* -------------------------------------------------------------
 * 1. Overview Page Logic
 * ------------------------------------------------------------- */
function renderOverview() {
  if (state.lastAnalysis) {
    const sum = state.lastAnalysis.summary;
    document.getElementById("statTotalFlows").textContent = sum.total_analyzed.toLocaleString();
    document.getElementById("statSuspiciousCount").textContent = sum.suspicious_count.toLocaleString();
    document.getElementById("statSuspiciousPct").textContent = `${sum.suspicious_percent}%`;
    document.getElementById("statAnomalyCount").textContent = sum.anomaly_count.toLocaleString();
    document.getElementById("statAnomalyPct").textContent = `${sum.anomaly_percent}%`;
    document.getElementById("statActiveModel").textContent = sum.model_used;

    NetGuardCharts.renderTrafficDistribution("overviewDistributionChart", sum.class_distribution);
  } else if (state.uploadedDataset && state.uploadedDataset.label_distribution) {
    document.getElementById("statTotalFlows").textContent = state.uploadedDataset.total_records.toLocaleString();
    NetGuardCharts.renderTrafficDistribution("overviewDistributionChart", state.uploadedDataset.label_distribution);
  }

  // Top features preview in overview
  if (state.featureImportance && state.featureImportance.global_importance) {
    const topFeats = state.featureImportance.global_importance.top_features.slice(0, 5);
    const container = document.getElementById("overviewTopFeaturesList");
    if (container) {
      container.innerHTML = topFeats.map((f, i) => `
        <div style="display:flex; justify-content:space-between; padding:0.4rem 0; border-bottom:1px solid rgba(255,255,255,0.05); font-size:0.8rem;">
          <span style="font-family:var(--font-mono); color:var(--text-primary);">${i+1}. ${f.feature}</span>
          <span style="color:var(--accent-cyan); font-weight:600;">${(f.importance * 100).toFixed(1)}%</span>
        </div>
      `).join("");
    }
  }
}

/* -------------------------------------------------------------
 * 2. Traffic Analysis Page Logic
 * ------------------------------------------------------------- */
function setupEventHandlers() {
  // Mode toggle on Detection & Performance
  document.querySelectorAll(".mode-toggle-btn").forEach(btn => {
    btn.addEventListener("click", async (e) => {
      const newMode = e.target.dataset.mode;
      state.currentMode = newMode;
      document.querySelectorAll(".mode-toggle-btn").forEach(b => b.classList.toggle("btn-primary", b.dataset.mode === newMode));
      await loadModeMetrics(newMode);
      if (state.activeView === "performance") renderPerformanceView();
    });
  });

  // Model Selector change
  const modelSelect = document.getElementById("modelSelector");
  if (modelSelect) {
    modelSelect.addEventListener("change", (e) => {
      state.currentModel = e.target.value;
    });
  }

  // File Upload Drag & Drop
  const dropzone = document.getElementById("csvDropzone");
  const fileInput = document.getElementById("csvFileInput");
  if (dropzone && fileInput) {
    dropzone.addEventListener("click", () => fileInput.click());
    dropzone.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.classList.add("dragover"); });
    dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
    dropzone.addEventListener("drop", async (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
      if (e.dataTransfer.files.length > 0) {
        await handleFileUpload(e.dataTransfer.files[0]);
      }
    });
    fileInput.addEventListener("change", async (e) => {
      if (e.target.files.length > 0) {
        await handleFileUpload(e.target.files[0]);
      }
    });
  }

  // Button: Load Demo Sample
  const btnLoadSample = document.getElementById("btnLoadSample");
  if (btnLoadSample) {
    btnLoadSample.addEventListener("click", async () => {
      await autoLoadSample();
      showNotification("Demo dataset successfully loaded!", "cyan");
    });
  }

  // Button: Run Detection
  const btnRunAnalyze = document.getElementById("btnRunAnalyze");
  if (btnRunAnalyze) {
    btnRunAnalyze.addEventListener("click", async () => {
      await executeAnalysis();
    });
  }

  // Table Filter Buttons in Detection Lab
  document.querySelectorAll(".table-filter-btn").forEach(btn => {
    btn.addEventListener("click", (e) => {
      document.querySelectorAll(".table-filter-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      state.filterType = btn.dataset.filter;
      renderPredictionsTable();
    });
  });

  // Modal Close
  const modalClose = document.getElementById("modalClose");
  if (modalClose) {
    modalClose.addEventListener("click", () => {
      document.getElementById("explainModal").classList.remove("active");
    });
  }
}

async function handleFileUpload(file) {
  try {
    showNotification(`Uploading and validating ${file.name}...`, "cyan");
    const data = await NetGuardAPI.uploadCSV(file);
    state.uploadedDataset = data;
    updateDatasetInfoBadge(data.filename, data.total_records);
    renderTrafficPreview(data);
    showNotification(`Successfully validated ${data.filename} (${data.total_records} records).`, "emerald");
  } catch (err) {
    console.error("Upload error:", err);
    showNotification(err.message, "danger");
  }
}

function renderTrafficPreview(data) {
  document.getElementById("previewFilename").textContent = data.filename;
  document.getElementById("previewRowCount").textContent = data.total_records.toLocaleString();
  document.getElementById("previewColCount").textContent = data.total_features;
  document.getElementById("previewMissingCount").textContent = data.missing_values_count ?? 0;
  document.getElementById("previewLabelCol").textContent = data.detected_label_column ?? "None (Unlabeled)";

  // Render Table
  const thead = document.getElementById("trafficPreviewThead");
  const tbody = document.getElementById("trafficPreviewTbody");
  if (!thead || !tbody || !data.preview_rows || data.preview_rows.length === 0) return;

  const cols = data.preview_columns || Object.keys(data.preview_rows[0]);
  thead.innerHTML = `<tr>${cols.map(c => `<th>${c}</th>`).join("")}</tr>`;
  tbody.innerHTML = data.preview_rows.map(row => `
    <tr>${cols.map(c => `<td>${row[c] !== null && row[c] !== undefined ? row[c] : "-"}</td>`).join("")}</tr>
  `).join("");
}

/* -------------------------------------------------------------
 * 3. Detection Lab Logic
 * ------------------------------------------------------------- */
async function executeAnalysis() {
  const btn = document.getElementById("btnRunAnalyze");
  if (btn) {
    btn.disabled = true;
    btn.innerHTML = `<span class="status-dot" style="background-color:var(--accent-cyan);"></span> Analyzing Network Traffic...`;
  }

  try {
    const res = await NetGuardAPI.analyzeTraffic(
      state.currentModel,
      state.currentMode,
      null, // uses datastore
      false,
      200
    );
    state.lastAnalysis = res;

    // Render stats
    renderDetectionSummary(res.summary);
    renderPredictionsTable();

    // Enable Download Button
    const downloadBtn = document.getElementById("btnDownloadResults");
    if (downloadBtn) {
      downloadBtn.href = NetGuardAPI.getDownloadUrl(res.analysis_id);
      downloadBtn.style.display = "inline-flex";
    }

    showNotification(`Analysis complete: ${res.summary.suspicious_count} suspicious flows detected.`, "emerald");
  } catch (err) {
    console.error("Analysis error:", err);
    showNotification(err.message, "danger");
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = `⚡ Run Intrusion Analysis`;
    }
  }
}

function renderDetectionSummary(summary) {
  document.getElementById("detAnalyzedCount").textContent = summary.total_analyzed;
  document.getElementById("detSuspiciousCount").textContent = `${summary.suspicious_count} (${summary.suspicious_percent}%)`;
  document.getElementById("detAnomalyCount").textContent = `${summary.anomaly_count} (${summary.anomaly_percent}%)`;
  document.getElementById("detModelBadge").textContent = `${summary.model_used} [${summary.mode}]`;
}

function renderPredictionsTable() {
  if (!state.lastAnalysis) return;
  const tbody = document.getElementById("predictionsTbody");
  if (!tbody) return;

  let records = state.lastAnalysis.predictions;

  // Apply filters
  if (state.filterType === "suspicious") {
    records = records.filter(r => r.is_suspicious);
  } else if (state.filterType === "normal") {
    records = records.filter(r => !r.is_suspicious);
  } else if (state.filterType === "anomaly") {
    records = records.filter(r => r.is_anomaly);
  }

  // Display first 100 matching rows
  const displaySlice = records.slice(0, 100);

  tbody.innerHTML = displaySlice.map(r => {
    const isSusp = r.is_suspicious;
    const badgeCls = isSusp ? "badge-suspicious" : "badge-normal";
    const anomBadge = r.is_anomaly
      ? `<span class="badge badge-anomaly">ANOMALY (${r.anomaly_score})</span>`
      : `<span class="badge badge-clean">INLIER (${r.anomaly_score})</span>`;

    const confPct = (r.confidence * 100).toFixed(1);

    return `
      <tr>
        <td>#${r.record_index + 1}</td>
        <td><span class="badge ${badgeCls}">${r.predicted_label}</span></td>
        <td>
          <div style="display:flex; align-items:center; gap:0.4rem;">
            <div style="flex:1; height:6px; background:#0d1527; border-radius:3px; overflow:hidden;">
              <div style="width:${confPct}%; height:100%; background:${isSusp ? 'var(--accent-rose)' : 'var(--accent-emerald)'};"></div>
            </div>
            <span style="font-size:0.75rem;">${confPct}%</span>
          </div>
        </td>
        <td>${anomBadge}</td>
        <td>Cluster ${r.traffic_cluster}</td>
        <td>
          <button class="btn btn-sm btn-secondary btn-explain-row" data-index="${r.record_index}">
            🔍 Explain
          </button>
        </td>
      </tr>
    `;
  }).join("");

  // Attach click events to "Explain" buttons
  document.querySelectorAll(".btn-explain-row").forEach(btn => {
    btn.addEventListener("click", async () => {
      const idx = parseInt(btn.dataset.index, 10);
      await openExplainModal(idx);
    });
  });
}

async function openExplainModal(recordIndex) {
  const modal = document.getElementById("explainModal");
  if (!modal) return;
  modal.classList.add("active");

  const modalBody = document.getElementById("explainModalBody");
  modalBody.innerHTML = `<p style="color:var(--text-secondary);">Computing SHAP feature contributions for record #${recordIndex + 1}...</p>`;

  try {
    let rawRecord = null;
    if (state.lastAnalysis?.predictions) {
      const found = state.lastAnalysis.predictions.find(p => p.record_index === recordIndex);
      if (found && found.full_record) {
        rawRecord = found.full_record;
      }
    }

    const exp = await NetGuardAPI.explainRecord(rawRecord, state.currentModel, state.currentMode, recordIndex);

    modalBody.innerHTML = `
      <div style="margin-bottom:1rem; display:flex; justify-content:space-between; align-items:center;">
        <div>
          <span style="font-size:0.85rem; color:var(--text-muted);">Record #${recordIndex + 1} Predicted As:</span>
          <h3 style="color:${exp.is_suspicious ? 'var(--accent-rose)' : 'var(--accent-emerald)'}; font-size:1.2rem;">
            ${exp.predicted_label} (${(exp.confidence * 100).toFixed(1)}% Confidence)
          </h3>
        </div>
        <span class="badge badge-clean">${exp.method}</span>
      </div>

      <div style="height:220px; margin-bottom:1.25rem;">
        <canvas id="modalShapCanvas"></canvas>
      </div>

      <div class="alert-box warning" style="font-size:0.75rem; margin-bottom:1rem;">
        ⚠️ ${exp.disclaimer}
      </div>

      <h4 style="font-size:0.9rem; margin-bottom:0.5rem; color:var(--text-secondary);">Top Contributing Network Features:</h4>
      <div style="display:flex; flex-direction:column; gap:0.5rem;">
        ${exp.top_contributions.map(c => `
          <div style="background:var(--bg-card-hover); padding:0.6rem 0.85rem; border-radius:var(--radius-sm); border:1px solid var(--border-color);">
            <div style="display:flex; justify-content:space-between; margin-bottom:0.2rem;">
              <strong style="font-family:var(--font-mono); color:var(--text-primary); font-size:0.825rem;">${c.feature}</strong>
              <span style="font-size:0.8rem; font-weight:600; color:${c.shap_value > 0 ? 'var(--accent-rose)' : 'var(--accent-emerald)'};">
                ${c.shap_value > 0 ? '+' : ''}${c.shap_value} (${c.direction})
              </span>
            </div>
            <p style="font-size:0.75rem; color:var(--text-secondary); line-height:1.4;">${c.description}</p>
          </div>
        `).join("")}
      </div>
    `;

    NetGuardCharts.renderLocalAttribution("modalShapCanvas", exp.top_contributions);
  } catch (err) {
    modalBody.innerHTML = `<p style="color:var(--accent-rose);">Failed to explain record: ${err.message}</p>`;
  }
}

/* -------------------------------------------------------------
 * 4. Model Performance Page Logic
 * ------------------------------------------------------------- */
function renderPerformanceView() {
  if (!state.metrics) return;
  const metrics = state.metrics;

  // Comparison Table
  const tableBody = document.getElementById("comparisonTableTbody");
  if (tableBody && metrics.comparison_table) {
    tableBody.innerHTML = metrics.comparison_table.map((m, idx) => `
      <tr style="${idx === 0 ? 'background:rgba(6,182,212,0.06);' : ''}">
        <td><strong>${m.model}</strong> ${idx === 0 ? '<span class="badge badge-clean" style="font-size:0.6rem;">BEST</span>' : ''}</td>
        <td>${(m.accuracy * 100).toFixed(2)}%</td>
        <td>${(m.precision * 100).toFixed(2)}%</td>
        <td>${(m.recall * 100).toFixed(2)}%</td>
        <td><strong style="color:var(--accent-cyan);">${(m.f1_score * 100).toFixed(2)}%</strong></td>
        <td>${m.roc_auc ? (m.roc_auc).toFixed(3) : "N/A"}</td>
        <td>${m.training_time_sec}s</td>
        <td>${m.latency_ms} ms</td>
      </tr>
    `).join("");
  }

  // Model comparison chart
  NetGuardCharts.renderModelComparison("modelComparisonChart", metrics.comparison_table);

  // Render Confusion Matrix for Top Model
  const topModel = metrics.comparison_table[0].model;
  const topModelDetails = metrics.detailed_results[topModel];
  if (topModelDetails) {
    renderConfusionMatrixDisplay(topModelDetails);
    if (topModelDetails.roc_curve) {
      NetGuardCharts.renderRocCurve("rocCurveChart", topModelDetails.roc_curve, topModel);
    }
  }
}

function renderConfusionMatrixDisplay(modelDetails) {
  const container = document.getElementById("confusionMatrixContainer");
  if (!container || !modelDetails.confusion_matrix) return;

  const cm = modelDetails.confusion_matrix;
  const classes = modelDetails.class_names;

  let html = `<table style="text-align:center;"><thead><tr><th>Actual \\ Predicted</th>`;
  classes.forEach(c => html += `<th>Pred: ${c}</th>`);
  html += `</tr></thead><tbody>`;

  for (let i = 0; i < classes.length; i++) {
    html += `<tr><td><strong>True: ${classes[i]}</strong></td>`;
    for (let j = 0; j < classes.length; j++) {
      const isDiag = (i === j);
      const bg = isDiag ? "rgba(16, 185, 129, 0.15)" : (cm[i][j] > 0 ? "rgba(244, 63, 94, 0.15)" : "transparent");
      const color = isDiag ? "var(--accent-emerald)" : (cm[i][j] > 0 ? "var(--accent-rose)" : "var(--text-muted)");
      html += `<td style="background:${bg}; color:${color}; font-weight:700; font-size:0.9rem;">${cm[i][j]}</td>`;
    }
    html += `</tr>`;
  }
  html += `</tbody></table>`;
  container.innerHTML = html;
}

/* -------------------------------------------------------------
 * 5. Explainability Page Logic
 * ------------------------------------------------------------- */
function renderExplainabilityView() {
  if (!state.featureImportance) return;
  const featData = state.featureImportance;

  if (featData.global_importance) {
    NetGuardCharts.renderFeatureImportance("globalImportanceChart", featData.global_importance.top_features);

    // Feature glossary
    const glossaryContainer = document.getElementById("featureGlossaryList");
    if (glossaryContainer) {
      glossaryContainer.innerHTML = featData.global_importance.top_features.slice(0, 8).map(f => `
        <div style="background:var(--bg-card); border:1px solid var(--border-color); padding:0.85rem; border-radius:var(--radius-sm);">
          <div style="display:flex; justify-content:space-between; margin-bottom:0.25rem;">
            <strong style="color:var(--accent-cyan); font-family:var(--font-mono); font-size:0.85rem;">${f.feature}</strong>
            <span style="font-size:0.75rem; color:var(--text-muted);">Gini: ${(f.importance * 100).toFixed(1)}%</span>
          </div>
          <p style="font-size:0.8rem; color:var(--text-secondary);">${f.description}</p>
        </div>
      `).join("");
    }
  }

  // Permutation Importance Table
  if (featData.permutation_importance && featData.permutation_importance.features) {
    const permTbody = document.getElementById("permImportanceTbody");
    if (permTbody) {
      permTbody.innerHTML = featData.permutation_importance.features.slice(0, 7).map((p, idx) => `
        <tr>
          <td>#${idx + 1}</td>
          <td><code>${p.feature}</code></td>
          <td style="color:var(--accent-emerald); font-weight:600;">+${p.importance_mean}</td>
          <td style="color:var(--text-muted);">&plusmn;${p.importance_std}</td>
        </tr>
      `).join("");
    }
  }
}

/* -------------------------------------------------------------
 * Utility: Notification Toast
 * ------------------------------------------------------------- */
function showNotification(msg, type = "cyan") {
  const toast = document.getElementById("notificationToast");
  if (!toast) return;
  toast.textContent = msg;
  toast.className = `alert-box ${type === 'danger' ? 'danger' : ''}`;
  toast.style.display = "block";
  toast.style.position = "fixed";
  toast.style.bottom = "20px";
  toast.style.right = "20px";
  toast.style.zIndex = "9999";
  toast.style.maxWidth = "400px";

  setTimeout(() => {
    toast.style.display = "none";
  }, 4500);
}
