"""
NetGuard AI - Backend Package Sanity Tests
"""

from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_backend_health_and_docs():
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"

    docs_res = client.get("/api/docs")
    assert docs_res.status_code == 200
