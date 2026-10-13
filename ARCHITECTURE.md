# NetGuard AI - System Architecture & Component Design
## Technical Specifications & Data Flow

---

## 1. High-Level Architectural Diagram

```
+─────────────────────────────────────────────────────────────────────────────+
|                              CLIENT BROWSER                                 |
|                                                                             |
|  [ Modern Cybersecurity Dark Theme UI: Deep Navy, Cyan & Violet Accents ]  |
|  • Page 1: Overview Dashboard (Telemetry, Threat Distribution, Quick Stats) |
|  • Page 2: Traffic Analysis (CSV Upload, Schema Validator, Preview Table)   |
|  • Page 3: Detection Lab (Model Selector, Mode Toggle, Anomaly Badges)      |
|  • Page 4: Model Performance (Algorithm Matrix, Confusion Matrix, ROC-AUC)  |
|  • Page 5: Explainability (Gini MDI Bar Chart, Permutation Table, SHAP)     |
|  • Page 6: System Information (Dataset Card, Pipeline Specifications)       |
+──────────────────────────────────────┬──────────────────────────────────────+
                                       │ HTTP / JSON (Fetch API)
                                       ▼
+─────────────────────────────────────────────────────────────────────────────+
|                         FASTAPI BACKEND SUBSYSTEM                           |
|                                                                             |
|  [ API Router Layer ]                                                       |
|  • GET  /api/health            ──> Health check & Model readiness           |
|  • GET  /api/metadata          ──> Active mode, classes, feature metadata   |
|  • GET  /api/models            ──> Model registry specifications            |
|  • POST /api/upload            ──> CSV parser & schema validation           |
|  • POST /api/analyze           ──> Batch classification & anomaly inference |
|  • GET  /api/metrics           ──> Measured evaluation benchmark reports    |
|  • GET  /api/feature-importance──> Global MDI & Permutation statistics      |
|  • POST /api/explain           ──> Local SHAP instance attribution          |
|  • GET  /api/download/{id}     ──> CSV export of classified flows           |
|                                                                             |
|  [ Core Application Services ]                                              |
|  • PredictionService (Model Cache, Inference Pipeline)                      |
|  • AnomalyService (Isolation Forest, Outlier Scoring)                       |
|  • ExplanationService (SHAP TreeExplainer, Domain Glossary)                 |
+──────────────────┬───────────────────────────────────┬──────────────────────+
                   │                                   │
                   ▼                                   ▼
+──────────────────────────────────────+  +───────────────────────────────────+
|     DATA PREPROCESSING PIPELINE      |  |     MODEL INFERENCE ENGINE        |
|                                      |  |                                   |
| • Header Whitespace Sanitizer        |  | • Random Forest (100 Trees)       |
| • Infinite & NaN Median Imputer      |  | • SVM (RBF Kernel)                |
| • RobustScaler (IQR Scaling)         |  | • K-Nearest Neighbors (k=5)       |
| • SelectKBest (ANOVA F-statistic)    |  | • Logistic Regression (Baseline)  |
| • LabelEncoder (Binary/Multiclass)   |  | • Multilayer Perceptron (MLP)     |
| • Zero-Variance Feature Filter       |  | • Isolation Forest (Anomalies)    |
| • Non-Informative Column Removal     |  | • KMeans (Traffic Clusters)       |
+──────────────────┬───────────────────+  +───────────────────┬───────────────+
                   │                                          │
                   ▼                                          ▼
+─────────────────────────────────────────────────────────────────────────────+
|                        STORAGE & PERSISTENCE LAYER                          |
|                                                                             |
| • models/*.joblib  (Serialized fitted preprocessors & ML estimators)       |
| • reports/*.json   (Metrics summaries, feature importances, ROC points)     |
| • reports/*.png    (Confusion matrix heatmaps & ROC curve diagnostic plots) |
| • data/demo/*.csv  (CIC-IDS2017 schema demo & sample validation files)      |
+─────────────────────────────────────────────────────────────────────────────+
```

---

## 2. Subsystem Descriptions

### 2.1 Ingestion & Sanitization Subsystem (`ml/data_loader.py`)
- **Encoding Robustness:** Tries `utf-8`, `latin-1`, and `cp1252` to prevent encoding crashes on heterogeneous PCAP-converted CSVs.
- **Header Trimming:** Strips leading and trailing whitespaces from headers (a widespread quirk in raw CIC-IDS2017 files where columns are saved with spaces like `' Destination Port'`).
- **Infinite Value Conversion:** Detects string `'Infinity'`, `'-Infinity'`, float `np.inf`, and string `'NaN'`, converting them to standard IEEE `np.nan`.
- **Target Segregation:** Excludes target columns (`Label`, `Attack`, `Class`) and non-informative identifiers (`Flow ID`, `Source IP`, `Destination IP`, `Timestamp`) from model feature matrices.

### 2.2 Leakage-Free Preprocessing Subsystem (`ml/preprocessing.py`)
- **Partitioning Order:** Data is partitioned into train (75%) and test (25%) splits before fitting any data transformation.
- **Stateful Imputation:** Missing values are imputed using the median computed exclusively on training rows:
  $$\hat{\mu}_{1/2} = \text{median}(X_{\text{train}})$$
- **RobustScaler Transformation:** Scales numerical features using the median and Interquartile Range ($IQR$), preventing extreme volumetric attack outliers from distorting feature variances.
- **Dimensionality Reduction & Selection:** Uses ANOVA F-statistic (`SelectKBest(score_func=f_classif, k=20)`) to select the top 20 most discriminatory flow features.

### 2.3 Supervised Classification Engine (`ml/train.py`, `ml/predict.py`)
- **Random Forest:** 100-tree ensemble with max depth 16. Provides low-latency predictions ($0.02\text{ ms}$) and native tree explainability.
- **Support Vector Machine (SVM):** Implements an RBF kernel for non-linear boundaries.
- **K-Nearest Neighbors (KNN):** Distance-based instance learner with $k=5$.
- **Logistic Regression:** Linear decision boundary baseline modeling log-odds.
- **Multilayer Perceptron (MLP):** Feedforward artificial neural network with hidden layers $(64, 32)$ and ReLU activations.

### 2.4 Unsupervised Anomaly Detection Engine (`ml/anomaly_detection.py`)
- **Isolation Forest:** Builds 100 isolation trees with contamination rate $\alpha = 0.10$.
- **Calibrated Scoring:** Maps raw tree path depths to a normalized range $[0.0, 1.0]$. Records with score $\ge 0.65$ are flagged as anomalous.
- **K-Means Clustering:** Partitions flows into $K=4$ clusters to group interactive browsing, volumetric transfers, scanning probes, and bulk data.

### 2.5 Explainable AI (XAI) Attribution Subsystem (`ml/explainability.py`)
- **Global Importance (MDI):** Calculates Mean Decrease in Impurity across all Random Forest trees.
- **Permutation Importance:** Shuffles each feature on the held-out test split 5 times and computes the score decrease, verifying generalization relevance.
- **SHAP TreeExplainer:** Computes game-theoretic Shapley additive contributions for any selected flow record.
- **Domain Translation:** Maps technical flow metrics to plain-English operational cybersecurity explanations.

### 2.6 FastAPI REST API Subsystem (`backend/app/`)
- Built with Python 3.12+ and FastAPI.
- Validates all requests and responses using Pydantic schemas.
- Implements an in-memory `PredictionService` cache so loaded model weights remain in memory for instant millisecond inference.
- Mounts and serves frontend static files from `/static` and single-page routing at `/`.

### 2.7 Responsive Cyber-Defense Frontend (`frontend/`)
- Pure HTML5, modern CSS3 design system, and Vanilla JavaScript (ES6 modules).
- No external heavy framework overhead (React, Angular, Vue) required.
- Incorporates Chart.js via CDN for interactive donut distribution charts, model comparison bar graphs, ROC curves, and SHAP attribution bars.

---

## 3. Data Flow Execution Sequence

```
1. User uploads CSV / Clicks "Use Demo Dataset" in UI
       │
       ▼
2. POST /api/upload receives file buffer
       │
       ▼
3. load_dataset() strips headers, replaces Inf with NaN
       │
       ▼
4. validate_schema() checks column count & label candidates
       │
       ▼
5. UploadResponse returns preview rows & summary statistics to UI
       │
       ▼
6. User selects model (e.g. Random Forest) & Clicks "Run Intrusion Analysis"
       │
       ▼
7. POST /api/analyze sends record batch to PredictionService
       │
       ▼
8. NetGuardPreprocessor transforms features using training-fitted scaler & mask
       │
       ▼
9. Classifier predicts class labels & class probabilities
       │
       ▼
10. IsolationForest calculates calibrated anomaly scores & KMeans clusters
       │
       ▼
11. Results returned as JSON; Predictions table rendered with badges & gauges
       │
       ▼
12. User clicks "Explain" on record #42
       │
       ▼
13. POST /api/explain calculates SHAP feature contributions for record #42
       │
       ▼
14. Modal displays diverging bar chart showing features driving "Suspicious" alert
```

---

## 4. Defensive & Ethical Design Principles

1. **Passive Flow Analysis Only:** The system analyzes offline flow statistics generated by flow meters. It does not perform active sniffing, promiscuous mode packet capture, or raw socket injection.
2. **No Automated Blocking:** Predictions are diagnostic alerts for security analysts; the platform does not execute automated firewall drop rules.
3. **Privacy Preservation:** Payload contents are never stored or displayed. Only statistical traffic headers (packet counts, durations, byte rates) are processed.
4. **Local Execution:** Operates 100% locally on standard student workstations without transmitting telemetry to third-party cloud services.
