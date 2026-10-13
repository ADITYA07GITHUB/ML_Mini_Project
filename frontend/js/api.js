/**
 * NetGuard AI - Frontend API Service
 * Encapsulates asynchronous HTTP requests to the FastAPI backend.
 */

const API_BASE = "/api";

export const NetGuardAPI = {
  /**
   * Fetch health and system readiness
   */
  async getHealth() {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
    return await res.json();
  },

  /**
   * Fetch system metadata and active feature set
   */
  async getMetadata() {
    const res = await fetch(`${API_BASE}/metadata`);
    if (!res.ok) throw new Error(`Metadata fetch failed: ${res.statusText}`);
    return await res.json();
  },

  /**
   * List available ML models
   */
  async getModels() {
    const res = await fetch(`${API_BASE}/models`);
    if (!res.ok) throw new Error(`Models fetch failed: ${res.statusText}`);
    return await res.json();
  },

  /**
   * Upload CSV network flow dataset
   */
  async uploadCSV(file) {
    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch(`${API_BASE}/upload`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || "CSV upload failed");
    }
    return await res.json();
  },

  /**
   * Load sample test flow data for instant testing
   */
  async loadSampleData() {
    const res = await fetch(`${API_BASE}/sample-data`);
    if (!res.ok) throw new Error(`Failed to load sample dataset: ${res.statusText}`);
    return await res.json();
  },

  /**
   * Run supervised classification and anomaly detection
   */
  async analyzeTraffic(modelName = "Random Forest", mode = "binary", records = null, useSample = false, sampleSize = 150) {
    const payload = {
      model_name: modelName,
      mode: mode,
      records: records,
      use_sample: useSample,
      sample_size: sampleSize,
    };

    const res = await fetch(`${API_BASE}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || "Analysis failed");
    }
    return await res.json();
  },

  /**
   * Fetch evaluation metrics and comparison table
   */
  async getMetrics(mode = "binary") {
    const res = await fetch(`${API_BASE}/metrics?mode=${encodeURIComponent(mode)}`);
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || "Failed to load metrics");
    }
    return await res.json();
  },

  /**
   * Fetch global feature importance & permutation importance
   */
  async getFeatureImportance(mode = "binary") {
    const res = await fetch(`${API_BASE}/feature-importance?mode=${encodeURIComponent(mode)}`);
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || "Failed to load feature importance");
    }
    return await res.json();
  },

  /**
   * Generate local instance explanation for single network flow
   */
  async explainRecord(record = null, modelName = "Random Forest", mode = "binary", recordIndex = null) {
    const payload = {
      record: record,
      record_index: recordIndex,
      model_name: modelName,
      mode: mode,
    };

    const res = await fetch(`${API_BASE}/explain`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || "Local explanation failed");
    }
    return await res.json();
  },

  /**
   * Get download URL for export CSV
   */
  getDownloadUrl(analysisId) {
    return `${API_BASE}/download/${analysisId}`;
  }
};
