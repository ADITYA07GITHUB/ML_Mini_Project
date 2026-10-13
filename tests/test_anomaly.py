"""
NetGuard AI - Unsupervised Anomaly Detection Tests
"""

import os
import pytest
import numpy as np
from ml.anomaly_detection import NetGuardAnomalyDetector


def test_anomaly_detector_fit_and_calibrate():
    # Synthetic normal network baseline
    rng = np.random.default_rng(42)
    X_normal = rng.normal(loc=0.0, scale=1.0, size=(200, 10))

    detector = NetGuardAnomalyDetector(contamination=0.10, random_state=42)
    detector.fit(X_normal)

    assert detector.is_fitted
    assert detector.model is not None

    # Compute scores for inliers and extreme outliers
    X_outliers = rng.uniform(low=20.0, high=50.0, size=(10, 10))
    scores_normal = detector.compute_anomaly_scores(X_normal)
    scores_outlier = detector.compute_anomaly_scores(X_outliers)

    # Scores must be bounded within [0.0, 1.0]
    assert np.all(scores_normal >= 0.0) and np.all(scores_normal <= 1.0)
    assert np.all(scores_outlier >= 0.0) and np.all(scores_outlier <= 1.0)

    # Outliers should have higher anomaly scores than average inliers
    assert np.mean(scores_outlier) > np.mean(scores_normal)

    # Predict test
    is_anomaly, anom_scores, clusters = detector.predict(X_outliers)
    assert len(is_anomaly) == 10
    assert len(clusters) == 10
