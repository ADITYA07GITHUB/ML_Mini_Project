"""
NetGuard AI - Model Explainability & Feature Attribution Endpoints
"""

import os
import json
import logging
import pandas as pd
from fastapi import APIRouter, Query, HTTPException
from backend.app.config import settings
from backend.app.schemas import LocalExplainRequest, LocalExplainResponse
from backend.app.services.explanation_service import explanation_service

logger = logging.getLogger("backend.api.explain")
router = APIRouter(tags=["Explainability"])


@router.get("/feature-importance")
def get_global_feature_importance(mode: str = Query(default="binary", pattern="^(binary|multiclass)$")):
    """
    Returns global feature importance (Tree MDI & Permutation Importance)
    along with plain-English cybersecurity definitions for top features.
    """
    try:
        return explanation_service.get_global_importance(mode=mode)
    except Exception as e:
        logger.error(f"Error fetching feature importance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/explain", response_model=LocalExplainResponse)
def explain_single_record(req: LocalExplainRequest):
    """
    Generates local feature attribution for a single network flow record using
    SHAP TreeExplainer or calibrated feature weights.
    """
    record = req.record
    if not record and req.record_index is not None:
        from backend.app.api.analysis import SESSION_DATASTORE
        curr_df = SESSION_DATASTORE.get("current_df")
        if curr_df is None:
            sample_path = os.path.join(settings.DATA_DIR, "demo", "cicids2017_sample_test.csv")
            if os.path.exists(sample_path):
                curr_df = pd.read_csv(sample_path)
                curr_df.columns = [c.strip() for c in curr_df.columns]
                SESSION_DATASTORE["current_df"] = curr_df

        if curr_df is not None and 0 <= req.record_index < len(curr_df):
            record = curr_df.iloc[req.record_index].to_dict()

    if not record:
        raise HTTPException(status_code=400, detail="Network flow record or valid record_index is required.")

    try:
        res = explanation_service.explain_record(
            record_dict=record,
            model_name=req.model_name,
            mode=req.mode
        )
        return LocalExplainResponse(**res)
    except FileNotFoundError as e:
        logger.warning(f"Artifact not found for explanation: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating local explanation: {e}")
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")
