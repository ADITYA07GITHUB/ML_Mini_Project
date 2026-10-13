"""
NetGuard AI - FastAPI Route & Integration Tests
"""

import io
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_api_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "models_ready" in data


def test_api_metadata():
    response = client.get("/api/metadata")
    assert response.status_code == 200
    data = response.json()
    assert "models_available" in data
    assert "classes" in data
    assert "disclaimer" in data


def test_api_models():
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 5
    model_names = [m["name"] for m in data]
    assert "Random Forest" in model_names
    assert "Isolation Forest" in model_names


def test_api_metrics():
    response = client.get("/api/metrics?mode=binary")
    assert response.status_code == 200
    data = response.json()
    assert "comparison_table" in data
    assert len(data["comparison_table"]) > 0


def test_api_feature_importance():
    response = client.get("/api/feature-importance?mode=binary")
    assert response.status_code == 200
    data = response.json()
    assert "global_importance" in data


def test_api_sample_data():
    response = client.get("/api/sample-data")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["total_records"] > 0


def test_api_analyze():
    # Analyze using sample demo data in datastore
    payload = {
        "model_name": "Random Forest",
        "mode": "binary",
        "use_sample": True,
        "sample_size": 25
    }
    response = client.post("/api/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "summary" in data
    assert "predictions" in data
    assert "analysis_id" in data
    assert data["summary"]["total_analyzed"] > 0

    analysis_id = data["analysis_id"]

    # Test CSV Download with analysis_id
    dl_response = client.get(f"/api/download/{analysis_id}")
    assert dl_response.status_code == 200
    assert "text/csv" in dl_response.headers.get("content-type", "")
    assert "Record_ID" in dl_response.text


def test_api_local_explain():
    payload = {
        "record": {
            "Destination Port": 80,
            "Flow Duration": 12000.0,
            "Total Fwd Packets": 5,
            "Flow Packets/s": 250.0
        },
        "model_name": "Random Forest",
        "mode": "binary"
    }
    response = client.post("/api/explain", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_label" in data
    assert "top_contributions" in data
    assert len(data["top_contributions"]) > 0


def test_api_upload_invalid_file_extension():
    # Attempt uploading .txt file should return 400 Bad Request
    file_bytes = b"Hello world this is not a CSV"
    files = {"file": ("malicious.txt", io.BytesIO(file_bytes), "text/plain")}
    response = client.post("/api/upload", files=files)
    assert response.status_code == 400
    assert "Only CSV files" in response.json()["detail"]


def test_api_predict_alias():
    payload = {
        "model_name": "Random Forest",
        "mode": "binary",
        "use_sample": True,
        "sample_size": 15
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    assert "summary" in response.json()


def test_api_analyze_isolation_forest():
    payload = {
        "model_name": "Isolation Forest",
        "mode": "binary",
        "use_sample": True,
        "sample_size": 15
    }
    response = client.post("/api/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["summary"]["model_used"] == "Isolation Forest"


def test_api_model_specific_metrics():
    response = client.get("/api/metrics/Random Forest?mode=binary")
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "Random Forest"
    assert "summary" in data


def test_api_report_artifact():
    response = client.get("/api/reports/roc_random_forest.png")
    assert response.status_code == 200
    assert response.headers.get("content-type") == "image/png"


def test_docs_and_root_aliases():
    res_docs = client.get("/docs", follow_redirects=True)
    assert res_docs.status_code == 200

    res_health = client.get("/health")
    assert res_health.status_code == 200

