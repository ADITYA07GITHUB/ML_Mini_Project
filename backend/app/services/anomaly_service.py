"""
NetGuard AI - Anomaly Detection Service
"""

import os
import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from backend.app.config import settings
from ml.anomaly_detection import NetGuardAnomalyDetector
from ml.preprocessing import NetGuardPreprocessor
from ml.data_loader import split_features_and_label

logger = logging.getLogger("backend.services.anomaly")


class AnomalyService:
    def __init__(self):
        self._detector: Optional[NetGuardAnomalyDetector] = None
        self._preprocessor: Optional[NetGuardPreprocessor] = None

    def _ensure_loaded(self):
        if self._detector is None:
            path = os.path.join(settings.MODELS_DIR, "anomaly_detector.joblib")
            if os.path.exists(path):
                self._detector = NetGuardAnomalyDetector.load(path)
            else:
                raise FileNotFoundError("Anomaly detector artifact not found. Please train models first.")

        if self._preprocessor is None:
            prep_path = os.path.join(settings.MODELS_DIR, "preprocessor_binary.joblib")
            if os.path.exists(prep_path):
                self._preprocessor = NetGuardPreprocessor.load(prep_path)
            else:
                raise FileNotFoundError("Preprocessor artifact not found. Please train models first.")

    def detect_anomalies(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        self._ensure_loaded()
        X, _ = split_features_and_label(df)
        X_trans = self._preprocessor.transform_features(X)
        is_anomaly, scores, clusters = self._detector.predict(X_trans)

        results = []
        for i in range(len(df)):
            results.append({
                "record_index": i,
                "is_anomaly": bool(is_anomaly[i]),
                "anomaly_score": round(float(scores[i]), 4),
                "traffic_cluster": int(clusters[i])
            })
        return results


anomaly_service = AnomalyService()
