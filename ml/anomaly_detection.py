"""
NetGuard AI - Unsupervised Anomaly Detection Module
===================================================
Implements unsupervised anomaly detection using Isolation Forest, Local Outlier Factor (LOF),
and K-Means traffic clustering.

Concept:
- Supervised classification learns explicit boundaries from labeled attack instances.
- Unsupervised anomaly detection learns the baseline distribution of normal/benign network
  traffic and flags records that deviate statistically from that distribution, without
  requiring prior knowledge of specific attack signatures.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans
import joblib

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


class NetGuardAnomalyDetector:
    """
    Unsupervised network anomaly detector utilizing Isolation Forest
    with calibrated anomaly scores between 0.0 (completely normal) and 1.0 (severe outlier).
    """
    def __init__(self, contamination: float = 0.10, random_state: int = 42, anomaly_threshold: float = 0.65):
        self.contamination = contamination
        self.random_state = random_state
        self.anomaly_threshold = anomaly_threshold
        
        self.model: Optional[IsolationForest] = None
        self.kmeans: Optional[KMeans] = None
        self.is_fitted: bool = False
        self.feature_names: List[str] = []
        
        # Calibration bounds calculated on training data
        self.score_min: float = -0.5
        self.score_max: float = 0.5

    @property
    def isolation_forest(self) -> Optional[IsolationForest]:
        """Convenience alias for the underlying IsolationForest model."""
        return self.model

    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None):
        """
        Fit IsolationForest and exploratory KMeans clustering on network flows.
        Best practice: Fit on benign/normal flows or general network baseline.
        """
        logger.info(f"Fitting IsolationForest (contamination={self.contamination})...")
        self.feature_names = feature_names or [f"f_{i}" for i in range(X.shape[1])]
        
        self.model = IsolationForest(
            n_estimators=100,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.model.fit(X)
        
        # Fit exploratory KMeans for traffic clustering (k=4 clusters: Web, High-rate, Probe, Bulk)
        n_clusters = min(4, max(2, len(X) // 50))
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init="auto")
        self.kmeans.fit(X)

        # Calibrate decision function bounds on training data
        raw_scores = self.model.decision_function(X)
        self.score_min = float(np.percentile(raw_scores, 1))
        self.score_max = float(np.percentile(raw_scores, 99))
        if self.score_max <= self.score_min:
            self.score_max = self.score_min + 1.0

        self.is_fitted = True
        logger.info(f"Anomaly detector successfully fitted. Calibration range: [{self.score_min:.3f}, {self.score_max:.3f}]")
        return self

    def compute_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Compute calibrated anomaly scores in [0.0, 1.0].
        Lower decision_function value = more anomalous -> mapped to higher anomaly score.
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Anomaly detector is not fitted yet.")

        # decision_function: large positive = inlier, negative = outlier
        raw = self.model.decision_function(X)
        
        # Invert and scale to [0, 1]: (score_max - raw) / (score_max - score_min)
        scaled = (self.score_max - raw) / (self.score_max - self.score_min)
        scores = np.clip(scaled, 0.0, 1.0)
        return scores

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict anomaly flags, calibrated scores, and cluster labels.
        Returns:
            is_anomaly: boolean array (True if anomaly score >= threshold)
            anomaly_scores: float array in [0.0, 1.0]
            cluster_ids: int array of KMeans cluster assignment
        """
        scores = self.compute_anomaly_scores(X)
        is_anomaly = scores >= self.anomaly_threshold
        clusters = self.kmeans.predict(X) if self.kmeans is not None else np.zeros(len(X), dtype=int)
        return is_anomaly, scores, clusters

    def save(self, filepath: str):
        """Serialize detector to disk using joblib."""
        joblib.dump(self, filepath)
        logger.info(f"Saved AnomalyDetector to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "NetGuardAnomalyDetector":
        """Load fitted detector from disk."""
        obj = joblib.load(filepath)
        if not isinstance(obj, NetGuardAnomalyDetector):
            raise TypeError("Loaded artifact is not a NetGuardAnomalyDetector instance.")
        return obj
