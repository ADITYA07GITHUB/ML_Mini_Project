/**
 * NetGuard AI - Interactive Chart.js Visualizations
 */

export const NetGuardCharts = {
  instances: {},

  destroy(id) {
    if (this.instances[id]) {
      this.instances[id].destroy();
      delete this.instances[id];
    }
  },

  /**
   * Traffic Label Distribution (Donut Chart)
   */
  renderTrafficDistribution(canvasId, distributionObj) {
    this.destroy(canvasId);
    const ctx = document.getElementById(canvasId)?.getContext("2d");
    if (!ctx) return;

    const labels = Object.keys(distributionObj || {});
    const data = Object.values(distributionObj || {});

    const backgroundColors = [
      "#10b981", // Normal / Benign (Emerald)
      "#ef4444", // DDoS (Rose)
      "#f59e0b", // PortScan (Amber)
      "#8b5cf6", // DoS Hulk (Violet)
      "#06b6d4", // Bot (Cyan)
      "#ec4899", // Other
    ];

    this.instances[canvasId] = new Chart(ctx, {
      type: "doughnut",
      data: {
        labels: labels,
        datasets: [{
          data: data,
          backgroundColor: backgroundColors.slice(0, labels.length),
          borderWidth: 2,
          borderColor: "#162032",
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: "bottom",
            labels: { color: "#94a3b8", boxWidth: 12, font: { family: "Inter", size: 11 } }
          },
          tooltip: {
            callbacks: {
              label: (context) => {
                const total = context.dataset.data.reduce((a, b) => a + b, 0);
                const val = context.raw;
                const pct = ((val / total) * 100).toFixed(1);
                return ` ${context.label}: ${val} (${pct}%)`;
              }
            }
          }
        },
        cutout: "68%"
      }
    });
  },

  /**
   * Model Performance Comparison Bar Chart
   */
  renderModelComparison(canvasId, comparisonTable) {
    this.destroy(canvasId);
    const ctx = document.getElementById(canvasId)?.getContext("2d");
    if (!ctx) return;

    const labels = comparisonTable.map(m => m.model);
    const accData = comparisonTable.map(m => (m.accuracy * 100).toFixed(2));
    const f1Data = comparisonTable.map(m => (m.f1_score * 100).toFixed(2));

    this.instances[canvasId] = new Chart(ctx, {
      type: "bar",
      data: {
        labels: labels,
        datasets: [
          {
            label: "Accuracy (%)",
            data: accData,
            backgroundColor: "rgba(6, 182, 212, 0.75)",
            borderColor: "#06b6d4",
            borderWidth: 1,
            borderRadius: 4
          },
          {
            label: "F1-Score (%)",
            data: f1Data,
            backgroundColor: "rgba(139, 92, 246, 0.75)",
            borderColor: "#8b5cf6",
            borderWidth: 1,
            borderRadius: 4
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: {
            min: Math.max(0, Math.min(...f1Data, ...accData) - 5),
            max: 100,
            grid: { color: "rgba(255, 255, 255, 0.05)" },
            ticks: { color: "#94a3b8", font: { size: 10 } }
          },
          x: {
            grid: { display: false },
            ticks: { color: "#94a3b8", font: { size: 10 } }
          }
        },
        plugins: {
          legend: {
            position: "top",
            labels: { color: "#94a3b8", boxWidth: 12, font: { size: 11 } }
          }
        }
      }
    });
  },

  /**
   * Interactive ROC Curve Line Chart
   */
  renderRocCurve(canvasId, rocData, modelName = "Model") {
    this.destroy(canvasId);
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!rocData) return;

    const points = rocData.fpr.map((fpr, i) => ({ x: fpr, y: rocData.tpr[i] }));
    const baseline = [{ x: 0, y: 0 }, { x: 1, y: 1 }];

    this.instances[canvasId] = new Chart(ctx, {
      type: "line",
      data: {
        datasets: [
          {
            label: `${modelName} (AUC = ${rocData.auc})`,
            data: points,
            borderColor: "#06b6d4",
            backgroundColor: "rgba(6, 182, 212, 0.15)",
            fill: true,
            tension: 0.2,
            borderWidth: 2,
            pointRadius: 2
          },
          {
            label: "Random Baseline (AUC = 0.50)",
            data: baseline,
            borderColor: "rgba(148, 163, 184, 0.5)",
            borderDash: [5, 5],
            borderWidth: 1.5,
            pointRadius: 0,
            fill: false
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            type: "linear",
            min: 0,
            max: 1,
            title: { display: true, text: "False Positive Rate (FPR)", color: "#94a3b8", font: { size: 11 } },
            grid: { color: "rgba(255, 255, 255, 0.05)" },
            ticks: { color: "#94a3b8" }
          },
          y: {
            type: "linear",
            min: 0,
            max: 1,
            title: { display: true, text: "True Positive Rate (TPR / Recall)", color: "#94a3b8", font: { size: 11 } },
            grid: { color: "rgba(255, 255, 255, 0.05)" },
            ticks: { color: "#94a3b8" }
          }
        },
        plugins: {
          legend: {
            position: "bottom",
            labels: { color: "#94a3b8", boxWidth: 12, font: { size: 11 } }
          }
        }
      }
    });
  },

  /**
   * Global Feature Importance Horizontal Bar Chart
   */
  renderFeatureImportance(canvasId, topFeatures) {
    this.destroy(canvasId);
    const ctx = document.getElementById(canvasId)?.getContext("2d");
    if (!ctx || !topFeatures || topFeatures.length === 0) return;

    // Show top 12 features
    const slice = topFeatures.slice(0, 12).reverse();
    const labels = slice.map(f => f.feature);
    const values = slice.map(f => f.importance);

    this.instances[canvasId] = new Chart(ctx, {
      type: "bar",
      data: {
        labels: labels,
        datasets: [{
          label: "Importance (Gini MDI)",
          data: values,
          backgroundColor: "rgba(56, 189, 248, 0.8)",
          borderColor: "#38bdf8",
          borderWidth: 1,
          borderRadius: 4
        }]
      },
      options: {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            grid: { color: "rgba(255, 255, 255, 0.05)" },
            ticks: { color: "#94a3b8" }
          },
          y: {
            grid: { display: false },
            ticks: { color: "#cbd5e1", font: { family: "JetBrains Mono", size: 10 } }
          }
        },
        plugins: {
          legend: { display: false }
        }
      }
    });
  },

  /**
   * Local Feature Attribution Diverging Bar Chart (SHAP values)
   */
  renderLocalAttribution(canvasId, contributions) {
    this.destroy(canvasId);
    const ctx = document.getElementById(canvasId)?.getContext("2d");
    if (!ctx || !contributions || contributions.length === 0) return;

    const slice = contributions.slice(0, 8).reverse();
    const labels = slice.map(c => c.feature);
    const values = slice.map(c => c.shap_value);
    const colors = values.map(v => v > 0 ? "rgba(244, 63, 94, 0.8)" : "rgba(16, 185, 129, 0.8)");
    const borderColors = values.map(v => v > 0 ? "#f43f5e" : "#10b981");

    this.instances[canvasId] = new Chart(ctx, {
      type: "bar",
      data: {
        labels: labels,
        datasets: [{
          label: "SHAP Impact",
          data: values,
          backgroundColor: colors,
          borderColor: borderColors,
          borderWidth: 1,
          borderRadius: 4
        }]
      },
      options: {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            grid: { color: "rgba(255, 255, 255, 0.08)" },
            ticks: { color: "#94a3b8" },
            title: { display: true, text: "Normal (-)  <--- Impact on Prediction --->  Suspicious (+)", color: "#94a3b8", font: { size: 10 } }
          },
          y: {
            grid: { display: false },
            ticks: { color: "#cbd5e1", font: { family: "JetBrains Mono", size: 10 } }
          }
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (context) => ` Contribution: ${context.raw > 0 ? '+' : ''}${context.raw}`
            }
          }
        }
      }
    });
  }
};
