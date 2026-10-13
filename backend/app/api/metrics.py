"""
NetGuard AI - Model Performance & Evaluation Metrics Endpoints
"""

import os
import json
import logging
from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import FileResponse
from backend.app.config import settings
from ml.predict import normalize_model_name

logger = logging.getLogger("backend.api.metrics")
router = APIRouter(tags=["Performance & Metrics"])


@router.get("/metrics")
def get_metrics(mode: str = Query(default="binary", pattern="^(binary|multiclass)$")):
    """
    Returns actual, measured model evaluation metrics from saved reports.
    Never hardcoded or fabricated.
    """
    summary_path = os.path.join(settings.REPORTS_DIR, f"metrics_summary_{mode}.json")
    if not os.path.exists(summary_path):
        raise HTTPException(
            status_code=404,
            detail=f"Evaluation metrics for mode '{mode}' not found. Please train models first using 'python -m ml.train --mode {mode}'."
        )

    try:
        with open(summary_path, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Error loading metrics summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to load evaluation metrics.")


@router.get("/metrics/{model_name}")
def get_model_metrics(
    model_name: str,
    mode: str = Query(default="binary", pattern="^(binary|multiclass)$")
):
    """
    Returns evaluation metrics, confusion matrix, and latency for a specific model.
    """
    summary_path = os.path.join(settings.REPORTS_DIR, f"metrics_summary_{mode}.json")
    if not os.path.exists(summary_path):
        raise HTTPException(status_code=404, detail=f"Metrics report for mode '{mode}' not found.")

    with open(summary_path, "r") as f:
        data = json.load(f)

    canonical = normalize_model_name(model_name)
    detailed = data.get("detailed_results", {}).get(canonical)
    comparison = next((m for m in data.get("comparison_table", []) if m.get("model") == canonical), None)

    if not detailed and not comparison:
        available = list(data.get("detailed_results", {}).keys())
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found in {mode} metrics report. Available: {available}"
        )

    return {
        "model": canonical,
        "mode": mode,
        "summary": comparison,
        "detailed": detailed
    }


@router.get("/reports/{filename}")
def get_report_artifact(filename: str):
    """
    Safely download or view generated evaluation plot images (.png) or reports (.json, .md).
    """
    # Prevent path traversal
    safe_filename = os.path.basename(filename)
    file_path = os.path.abspath(os.path.join(settings.REPORTS_DIR, safe_filename))
    
    if not file_path.startswith(os.path.abspath(settings.REPORTS_DIR)) or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Report artifact '{filename}' not found.")

    media_type = "image/png" if filename.endswith(".png") else "application/json" if filename.endswith(".json") else "text/plain"
    return FileResponse(file_path, media_type=media_type)
