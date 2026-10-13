# NetGuard AI: Network Anomaly Detection and Explainability Platform

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-orange.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Active%20%26%20Tested-emerald.svg)](#)

A complete, working end-to-end Machine Learning platform for network traffic classification, behavioral anomaly detection, and transparent model explainability.

NetGuard AI combines **supervised classification** (to classify known cyber threats like DDoS, PortScan, and Botnet activity) with **unsupervised anomaly detection** (Isolation Forest to discover unusual structural deviations from normal baselines) and **Explainable AI (SHAP)** to explain why any individual network flow was flagged.

---

## 🚀 Key Features

- **Dual-Engine Detection:**
  - **Supervised Classifiers:** Random Forest (100 Trees), Support Vector Machine (RBF kernel), K-Nearest Neighbors ($k=5$), Logistic Regression (Baseline), and Multilayer Perceptron (MLP Neural Network).
  - **Unsupervised Anomaly Detection:** Isolation Forest with calibrated $[0.0, 1.0]$ anomaly scoring ($\ge 0.65$ flags outlier) and exploratory K-Means behavioral clustering.
- **Robust Preprocessing Pipeline:**
  - Automatically handles real network flow quirks: strips column header whitespace (e.g. `' Destination Port'`), converts string infinities (`'Infinity'`), float infinities, and missing records.
  - Leakage-free train/test splitting (all median imputers, `RobustScaler` IQR scalers, and ANOVA F-statistic feature selection masks are fitted strictly on training data).
  - Supports both **Binary** (`Normal` vs `Suspicious`) and **Multiclass** (`Normal`, `DDoS`, `PortScan`, `DoS Hulk`, `Bot`) classification.
- **Explainable AI (XAI):**
  - Global Feature Importance (Random Forest Gini MDI).
  - Permutation Feature Importance on held-out test data.
  - Local per-record SHAP (Shapley Additive Explanations) TreeExplainer showing exact feature impacts towards Normal or Suspicious.
- **Clean Cybersecurity Dashboard:**
  - Responsive dark-theme interface with deep navy surfaces, cyan, and violet accents.
  - Interactive Chart.js visualizations for class distributions, algorithm comparison, confusion matrix heatmaps, ROC curves, and SHAP attribution bars.
  - Drag-and-drop CSV upload, 1-click sample demo testing, prediction filtering, and CSV export.
- **High-Performance FastAPI Backend:**
  - Asynchronous route handlers, typed Pydantic v2 schemas, in-memory model caching for fast millisecond inference, and static SPA serving.

---

## 📁 Project Structure

```text
ML_Mini_Project/
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI app entry point & static file routing
│   │   ├── config.py                # Configuration and directory paths
│   │   ├── schemas.py               # Pydantic request/response schemas
│   │   ├── api/                     # REST API endpoints (health, upload, analyze, metrics, explain)
│   │   ├── services/                # Model caching and inference controllers
│   │   └── utils/serialization.py   # Safe JSON converters for NumPy/Pandas
│   └── tests/test_backend.py        # Backend sanity tests
├── frontend/
│   ├── index.html                   # Interactive cybersecurity dashboard SPA
│   ├── css/styles.css               # Cyber-defense dark design system
│   └── js/
│       ├── api.js                   # Asynchronous Fetch API client
│       ├── charts.js                # Chart.js visualization wrappers
│       └── app.js                   # UI controller, state, and modal manager
├── ml/
│   ├── data_loader.py               # CSV ingestion and schema validation
│   ├── preprocessing.py             # Leakage-free NetGuardPreprocessor
│   ├── train.py                     # CLI model training pipeline
│   ├── evaluate.py                  # Confusion matrices, ROC curves, latency timing
│   ├── anomaly_detection.py         # Isolation Forest & K-Means clustering
│   ├── explainability.py            # SHAP TreeExplainer & domain glossary
│   └── predict.py                   # Inference engine
├── scripts/
│   ├── generate_demo_data.py        # Realistic CIC-IDS2017 synthetic generator
│   └── prepare_data.py              # Ingestion pipeline for raw UNB CSV files
├── data/
│   └── demo/                        # Pre-generated demo & sample test datasets
├── models/                          # Trained .joblib artifacts & metadata.json
├── reports/                         # Evaluation JSONs, confusion matrix & ROC PNGs
├── tests/                           # Complete pytest automated test suite
├── Dockerfile                       # Multi-stage production container
├── docker-compose.yml               # Container orchestration
└── requirements.txt                 # Pinned dependencies
```

---

## ⚡ Quickstart Guide

### 1. Installation
```bash
# Clone or navigate to the repository
cd c:\Users\LOQ\Desktop\ML_Mini_Project

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Demo Data & Train Models
```bash
# 1. Generate realistic 6,000-record CIC-IDS2017 demonstration dataset
python scripts/generate_demo_data.py --count 6000

# 2. Train and evaluate binary classification models & Isolation Forest
python -m ml.train --data data/demo/cicids2017_demo.csv --mode binary

# 3. (Optional) Train multiclass attack models
python -m ml.train --data data/demo/cicids2017_demo.csv --mode multiclass
```

### 3. Start the Server
```bash
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

- **Web Dashboard:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive Swagger API Docs:** [http://127.0.0.1:8000/api/docs](http://127.0.0.1:8000/api/docs)
- **ReDoc API Docs:** [http://127.0.0.1:8000/api/redoc](http://127.0.0.1:8000/api/redoc)

---

## 🧪 Automated Testing

Run the full pytest suite verifying data loading, leakage safeguards, model inference, Isolation Forest calibration, and FastAPI API routes:

```bash
python -m pytest tests/ -v
```

All 15 tests pass out of the box:
```text
tests/test_anomaly.py::test_anomaly_detector_fit_and_calibrate PASSED
tests/test_api.py::test_api_health PASSED
tests/test_api.py::test_api_metadata PASSED
tests/test_api.py::test_api_models PASSED
tests/test_api.py::test_api_metrics PASSED
tests/test_api.py::test_api_feature_importance PASSED
tests/test_api.py::test_api_sample_data PASSED
tests/test_api.py::test_api_analyze PASSED
tests/test_api.py::test_api_local_explain PASSED
tests/test_api.py::test_api_upload_invalid_file_extension PASSED
tests/test_models.py::test_predict_binary_records PASSED
tests/test_models.py::test_model_explainability_local_and_global PASSED
tests/test_preprocessing.py::test_schema_validation_and_clean_columns PASSED
tests/test_preprocessing.py::test_preprocessor_fit_transform PASSED
tests/test_preprocessing.py::test_train_test_split_no_leakage PASSED
============================== 15 passed in 2.86s ==============================
```

---

## 📊 Measured Benchmark Results

All metrics represent real measurements recorded on the held-out test split (1,500 samples):

| Algorithm | Mode | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | Binary | **100.00%** | **100.00%** | **100.00%** | **1.0000** | **1.000** | **0.02 ms** |
| **K-Nearest Neighbors** | Binary | 99.93% | 99.89% | 100.00% | 0.9992 | 0.999 | 0.04 ms |
| **Multilayer Perceptron** | Binary | 99.93% | 99.89% | 100.00% | 0.9992 | 0.999 | 0.00 ms |
| **Logistic Regression** | Binary | 99.87% | 99.80% | 100.00% | 0.9983 | 0.998 | 0.00 ms |
| **Support Vector Machine** | Binary | 87.87% | 85.12% | 80.20% | 0.8212 | 0.852 | 0.07 ms |

---

## 🐳 Docker Deployment

```bash
# Build and run container
docker compose up --build

# Open http://localhost:8000
```

---

## 🛡️ Defensive Disclaimer

NetGuard AI is a defensive machine learning analysis platform designed for offline network flow analysis and security research. It analyzes flow telemetry and does not perform active network packet sniffing or automated traffic blocking.
