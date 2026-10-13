"""
NetGuard AI - Explanation Service
=================================
Manages global and local model explainability requests.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from backend.app.config import settings
from backend.app.services.prediction_service import prediction_service
from ml.explainability import NetGuardExplainer, FEATURE_DESCRIPTIONS
from ml.preprocessing import NetGuardPreprocessor
from ml.predict import normalize_model_name

logger = logging.getLogger("backend.services.explanation")


class ExplanationService:
    def __init__(self):
        self._explainers: Dict[str, NetGuardExplainer] = {}

    def get_global_importance(self, mode: str = "binary") -> Dict[str, Any]:
        """Fetch precomputed global importance report or compute live."""
        report_path = os.path.join(settings.REPORTS_DIR, f"feature_importance_{mode}.json")
        if os.path.exists(report_path):
            with open(report_path, "r") as f:
                return json.load(f)

        # Fallback compute on the fly
        predictor = prediction_service.get_predictor("Random Forest", mode)
        explainer = NetGuardExplainer(predictor.classifier, predictor.preprocessor.selected_feature_names)
        return {
            "global_importance": explainer.get_global_feature_importance(),
            "selected_features": predictor.preprocessor.selected_feature_names
        }

    def explain_record(self, record_dict: Dict[str, Any], model_name: str = "Random Forest", mode: str = "binary") -> Dict[str, Any]:
        """Generate local instance attribution (SHAP or feature contribution) for a single record."""
        canonical_name = normalize_model_name(model_name)
        predictor = prediction_service.get_predictor(canonical_name, mode)
        df = pd.DataFrame([record_dict])
        
        # Preprocess single record
        X_trans = predictor.preprocessor.transform_features(df)
        
        if canonical_name == "Isolation Forest" and predictor.anomaly_detector is not None:
            is_anomaly, anom_scores, _ = predictor.anomaly_detector.predict(X_trans)
            anom_flag = bool(is_anomaly[0])
            pred_label = "Suspicious" if anom_flag else "Normal"
            confidence = float(anom_scores[0]) if anom_flag else float(1.0 - anom_scores[0])
        else:
            # Supervised prediction
            pred_label = predictor.preprocessor.inverse_transform_labels(predictor.classifier.predict(X_trans))[0]
            
            # Probabilities
            confidence = 1.0
            if hasattr(predictor.classifier, "predict_proba"):
                probs = predictor.classifier.predict_proba(X_trans)[0]
                confidence = float(np.max(probs))

        cache_key = f"{canonical_name.lower()}_{mode.lower()}"
        if cache_key not in self._explainers:
            self._explainers[cache_key] = NetGuardExplainer(
                predictor.classifier, predictor.preprocessor.selected_feature_names
            )
        explainer = self._explainers[cache_key]

        explanation = explainer.explain_instance(X_trans[0], raw_features_dict=record_dict)
        
        return {
            "method": explanation["method"],
            "predicted_label": pred_label,
            "confidence": round(confidence, 4),
            "is_suspicious": pred_label != "Normal",
            "top_contributions": explanation["top_contributions"],
            "disclaimer": explanation["disclaimer"]
        }


explanation_service = ExplanationService()
