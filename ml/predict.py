"""
NetGuard AI - Model Inference Engine
====================================
Loads saved model artifacts, preprocessors, and anomaly detectors to run
deterministic, reproducible inference on network flow records.
"""

import os
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import joblib

from ml.preprocessing import NetGuardPreprocessor
from ml.anomaly_detection import NetGuardAnomalyDetector

logger = logging.getLogger("ml.predict")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

MODEL_ALIASES = {
    "random forest": "Random Forest",
    "rf": "Random Forest",
    "random_forest": "Random Forest",
    "support vector machine": "Support Vector Machine",
    "svm": "Support Vector Machine",
    "support_vector_machine": "Support Vector Machine",
    "k-nearest neighbors": "K-Nearest Neighbors",
    "k-nearest_neighbors": "K-Nearest Neighbors",
    "knn": "K-Nearest Neighbors",
    "logistic regression": "Logistic Regression",
    "logistic_regression": "Logistic Regression",
    "lr": "Logistic Regression",
    "multilayer perceptron": "Multilayer Perceptron",
    "multilayer_perceptron": "Multilayer Perceptron",
    "mlp": "Multilayer Perceptron",
    "neural network": "Multilayer Perceptron",
    "neural_network": "Multilayer Perceptron",
    "isolation forest": "Isolation Forest",
    "isolation_forest": "Isolation Forest",
    "iforest": "Isolation Forest",
    "anomaly detector": "Isolation Forest",
    "anomaly_detector": "Isolation Forest"
}


def normalize_model_name(name: str) -> str:
    """Normalize user input or API model string into canonical model name."""
    clean = str(name).strip().lower()
    return MODEL_ALIASES.get(clean, name)


class NetGuardPredictor:
    """
    Encapsulates loaded preprocessors and models for production inference.
    """
    def __init__(
        self,
        model_name: str = "Random Forest",
        mode: str = "binary",
        models_dir: str = "models"
    ):
        self.model_name = normalize_model_name(model_name)
        self.mode = mode
        self.models_dir = models_dir
        
        self.preprocessor: Optional[NetGuardPreprocessor] = None
        self.classifier = None
        self.anomaly_detector: Optional[NetGuardAnomalyDetector] = None
        
        self.load_artifacts()

    def load_artifacts(self):
        """Load trained preprocessor, classifier, and anomaly detector."""
        prep_path = os.path.join(self.models_dir, f"preprocessor_{self.mode}.joblib")
        if not os.path.exists(prep_path):
            raise FileNotFoundError(f"Preprocessor artifact not found at {prep_path}. Train models first using 'python -m ml.train'.")
            
        self.preprocessor = NetGuardPreprocessor.load(prep_path)

        # Anomaly detector artifact
        anom_path = os.path.join(self.models_dir, "anomaly_detector.joblib")
        if os.path.exists(anom_path):
            self.anomaly_detector = NetGuardAnomalyDetector.load(anom_path)
        else:
            self.anomaly_detector = None
            logger.warning("Anomaly detector artifact not found. Anomaly scores will not be available.")

        # If model is Isolation Forest, use the anomaly detector's underlying isolation forest
        if self.model_name == "Isolation Forest":
            if self.anomaly_detector is None:
                raise FileNotFoundError(f"Isolation Forest artifact not found at {anom_path}.")
            self.classifier = self.anomaly_detector.model
            return

        clean_name = self.model_name.lower().replace(" ", "_")
        clf_path = os.path.join(self.models_dir, f"{clean_name}_{self.mode}.joblib")
        if not os.path.exists(clf_path):
            raise FileNotFoundError(f"Classifier artifact not found at {clf_path}.")
            
        self.classifier = joblib.load(clf_path)

    def predict_records(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Process a DataFrame of network flows, returning structured classification and anomaly outputs.
        """
        if self.preprocessor is None or self.classifier is None:
            raise RuntimeError("Inference artifacts are not loaded.")

        if len(df) == 0:
            return []

        # Transform inputs using fitted preprocessor
        X_trans = self.preprocessor.transform_features(df)
        
        # Anomaly detection (computed for all runs)
        if self.anomaly_detector is not None:
            is_anomaly, anom_scores, clusters = self.anomaly_detector.predict(X_trans)
        else:
            is_anomaly = np.zeros(len(df), dtype=bool)
            anom_scores = np.zeros(len(df), dtype=float)
            clusters = np.zeros(len(df), dtype=int)

        # Handle Isolation Forest as the primary classification engine
        if self.model_name == "Isolation Forest":
            results = []
            for i in range(len(df)):
                anom_flag = bool(is_anomaly[i])
                score = round(float(anom_scores[i]), 4)
                label = "Suspicious" if anom_flag else "Normal"
                conf = score if anom_flag else round(1.0 - score, 4)
                results.append({
                    "record_index": i,
                    "predicted_label": label,
                    "is_suspicious": anom_flag,
                    "confidence": conf,
                    "probabilities": {
                        "Normal": round(1.0 - score, 4),
                        "Suspicious": score
                    },
                    "is_anomaly": anom_flag,
                    "anomaly_score": score,
                    "traffic_cluster": int(clusters[i])
                })
            return results

        # Supervised predictions
        y_pred_num = self.classifier.predict(X_trans)
        labels = self.preprocessor.inverse_transform_labels(y_pred_num)

        # Probabilities if model supports it
        probabilities = None
        clf_classes = getattr(self.classifier, "classes_", None)
        if hasattr(self.classifier, "predict_proba"):
            try:
                probabilities = self.classifier.predict_proba(X_trans)
            except Exception:
                probabilities = None

        results = []
        for i in range(len(df)):
            label = labels[i]
            
            # Confidence score and mapped probability distribution
            if probabilities is not None and clf_classes is not None and len(clf_classes) > 0:
                prob_dict = {}
                for c_col, c_val in enumerate(clf_classes):
                    c_name = self.preprocessor.inverse_transform_labels([c_val])[0]
                    prob_dict[c_name] = round(float(probabilities[i][c_col]), 4)
                confidence = float(np.max(probabilities[i]))
            else:
                prob_dict = {label: 1.0}
                confidence = 1.0

            results.append({
                "record_index": i,
                "predicted_label": label,
                "is_suspicious": label != "Normal",
                "confidence": round(confidence, 4),
                "probabilities": prob_dict,
                "is_anomaly": bool(is_anomaly[i]),
                "anomaly_score": round(float(anom_scores[i]), 4),
                "traffic_cluster": int(clusters[i]),
            })

        return results
