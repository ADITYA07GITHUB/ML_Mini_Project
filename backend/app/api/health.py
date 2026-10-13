"""
NetGuard AI - Health, System Metadata & Model Registry Endpoints
"""

import os
import sys
import json
from fastapi import APIRouter
from backend.app.config import settings
from backend.app.schemas import HealthResponse, MetadataResponse, ModelInfo

router = APIRouter(tags=["Health & System"])


@router.get("/health", response_model=HealthResponse)
def get_health():
    """Verify backend system status and model readiness."""
    meta_path = os.path.join(settings.MODELS_DIR, "metadata.json")
    models_ready = os.path.exists(meta_path)
    
    return HealthResponse(
        status="healthy",
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        models_ready=models_ready,
        python_version=sys.version.split()[0]
    )


@router.get("/metadata", response_model=MetadataResponse)
def get_metadata():
    """Returns training run metadata, active feature set, and educational disclaimers."""
    meta_path = os.path.join(settings.MODELS_DIR, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
    else:
        meta = {
            "active_mode": "binary",
            "models_available": [],
            "features_selected": [],
            "classes": ["Normal", "Suspicious"],
            "best_model": None,
            "last_trained": None
        }

    return MetadataResponse(
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        active_mode=meta.get("active_mode", "binary"),
        models_available=meta.get("models_available", ["Random Forest", "Support Vector Machine", "K-Nearest Neighbors", "Logistic Regression", "Multilayer Perceptron"]),
        features_selected=meta.get("features_selected", []),
        classes=meta.get("classes", ["Normal", "Suspicious"]),
        best_model=meta.get("best_model"),
        last_trained=meta.get("last_trained"),
        disclaimer="NetGuard AI is a defensive machine learning analysis platform. It analyzes network flow records and does not constitute a certified production Intrusion Prevention System (IPS)."
    )


@router.get("/models")
def list_models():
    """List available supervised classification and unsupervised anomaly models."""
    return [
        {
            "name": "Random Forest",
            "category": "Supervised Ensemble (Bagging)",
            "supports_probabilities": True,
            "description": "Ensemble of 100 decision trees. Highly robust against feature multicollinearity and non-linear boundaries. Provides native MDI and SHAP tree explainability."
        },
        {
            "name": "Support Vector Machine",
            "category": "Supervised Kernel Method",
            "supports_probabilities": True,
            "description": "Radial Basis Function (RBF) kernel separating high-dimensional hyperplanes. Effective for complex margins, scaled with RobustScaler."
        },
        {
            "name": "K-Nearest Neighbors",
            "category": "Supervised Instance-Based (Lazy Learner)",
            "supports_probabilities": True,
            "description": "Non-parametric distance-based classifier (k=5). Classifies traffic flows based on Euclidean proximity to known behavioral clusters."
        },
        {
            "name": "Logistic Regression",
            "category": "Supervised Linear Baseline",
            "supports_probabilities": True,
            "description": "Standard linear decision baseline. Demonstrates linear separability benchmarks for network flow features."
        },
        {
            "name": "Multilayer Perceptron",
            "category": "Deep Learning / Neural Network",
            "supports_probabilities": True,
            "description": "Feedforward artificial neural network with hidden layers (64, 32) and ReLU activations."
        },
        {
            "name": "Isolation Forest",
            "category": "Unsupervised Anomaly Detector",
            "supports_probabilities": False,
            "description": "Isolates anomalies by randomly selecting features and splitting values. Normal points require many splits, whereas anomalous outliers are isolated near tree roots."
        }
    ]
