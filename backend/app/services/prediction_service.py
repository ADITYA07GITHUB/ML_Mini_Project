"""
NetGuard AI - Prediction Service
=================================
Manages model loading, caching, and batch prediction execution.
"""

import os
import logging
from typing import Dict, Any, List, Optional
import pandas as pd
from backend.app.config import settings
from ml.predict import NetGuardPredictor, normalize_model_name
from ml.data_loader import split_features_and_label

logger = logging.getLogger("backend.services.prediction")


class PredictionService:
    def __init__(self):
        self._predictor_cache: Dict[str, NetGuardPredictor] = {}

    def get_predictor(self, model_name: str, mode: str) -> NetGuardPredictor:
        """Get or initialize cached predictor instance."""
        canonical_name = normalize_model_name(model_name)
        cache_key = f"{canonical_name.lower()}_{mode.lower()}"
        if cache_key not in self._predictor_cache:
            logger.info(f"Loading model into memory: {canonical_name} ({mode})")
            predictor = NetGuardPredictor(
                model_name=canonical_name,
                mode=mode,
                models_dir=settings.MODELS_DIR
            )
            self._predictor_cache[cache_key] = predictor
        return self._predictor_cache[cache_key]

    def analyze_dataframe(
        self,
        df: pd.DataFrame,
        model_name: str = "Random Forest",
        mode: str = "binary"
    ) -> List[Dict[str, Any]]:
        """Run classification and anomaly detection on a DataFrame."""
        predictor = self.get_predictor(model_name, mode)
        X, _ = split_features_and_label(df)
        predictions = predictor.predict_records(X)
        
        # Attach preview snippet and full record dictionary
        from backend.app.utils.serialization import sanitize_value
        preview_fields = ["Destination Port", "Flow Duration", "Total Fwd Packets", "Flow Packets/s"]
        for i, row_pred in enumerate(predictions):
            preview = {}
            for col in preview_fields:
                if col in df.columns:
                    val = df.iloc[i][col]
                    preview[col] = float(val) if pd.notnull(val) else None
            row_pred["raw_preview"] = preview
            if i < len(df):
                row_pred["full_record"] = sanitize_value(df.iloc[i].to_dict())

        return predictions


prediction_service = PredictionService()
