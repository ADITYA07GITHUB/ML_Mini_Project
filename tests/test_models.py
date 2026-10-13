"""
NetGuard AI - Supervised Model & Inference Tests
"""

import os
import pytest
import numpy as np
import pandas as pd
from ml.predict import NetGuardPredictor
from ml.explainability import NetGuardExplainer


def test_predict_binary_records():
    # Test inference using saved binary models
    predictor = NetGuardPredictor(model_name="Random Forest", mode="binary", models_dir="models")
    
    test_df = pd.DataFrame([{
        "Destination Port": 80,
        "Flow Duration": 15000.0,
        "Total Fwd Packets": 10,
        "Total Backward Packets": 12,
        "Flow Packets/s": 1500.0,
        "Flow Bytes/s": 250000.0,
        "Flow IAT Mean": 120.0,
        "SYN Flag Count": 1,
        "ACK Flag Count": 1,
    }])

    results = predictor.predict_records(test_df)
    assert len(results) == 1
    record = results[0]

    assert "predicted_label" in record
    assert record["predicted_label"] in ["Normal", "Suspicious"]
    assert "confidence" in record
    assert 0.0 <= record["confidence"] <= 1.0
    assert "anomaly_score" in record
    assert 0.0 <= record["anomaly_score"] <= 1.0
    assert "is_anomaly" in record


def test_model_explainability_local_and_global():
    predictor = NetGuardPredictor(model_name="Random Forest", mode="binary", models_dir="models")
    explainer = NetGuardExplainer(predictor.classifier, predictor.preprocessor.selected_feature_names)

    # Global importance
    global_res = explainer.get_global_feature_importance(top_n=5)
    assert "top_features" in global_res
    assert len(global_res["top_features"]) <= 5
    assert "disclaimer" in global_res

    # Local instance explanation
    dummy_vector = np.zeros(len(predictor.preprocessor.selected_feature_names))
    local_res = explainer.explain_instance(dummy_vector, {"Destination Port": 80})
    assert "top_contributions" in local_res
    assert len(local_res["top_contributions"]) > 0
