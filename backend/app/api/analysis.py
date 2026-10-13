"""
NetGuard AI - Traffic Analysis, Upload & Inference Endpoints
"""

import io
import os
import uuid
import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Response
from fastapi.responses import StreamingResponse

from backend.app.config import settings
from backend.app.schemas import (
    UploadResponse, AnalyzeRequest, AnalyzeResponse,
    AnalyzeSummary, RecordPrediction
)
from backend.app.services.prediction_service import prediction_service
from backend.app.services.anomaly_service import anomaly_service
from backend.app.utils.serialization import sanitize_value
from ml.data_loader import load_dataset, validate_schema

logger = logging.getLogger("backend.api.analysis")
router = APIRouter(tags=["Analysis & Detection"])

# In-memory storage for active session dataset and recent analyses
SESSION_DATASTORE = {
    "current_df": None,
    "filename": None,
    "last_analysis": {}
}


@router.post("/upload", response_model=UploadResponse)
async def upload_csv(file: UploadFile = File(...)):
    """
    Safely ingest a network flow CSV, validate column schemas,
    detect labels, and extract statistical distributions.
    """
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files (.csv) are supported.")

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="The uploaded CSV file is empty.")

    if len(content) > settings.MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Uploaded file exceeds {settings.MAX_UPLOAD_SIZE_BYTES // (1024*1024)}MB size limit."
        )

    try:
        # Read CSV from memory buffer
        df = pd.read_csv(io.BytesIO(content), low_memory=False)
        if len(df) == 0:
            raise HTTPException(status_code=400, detail="The uploaded CSV file contains no data rows.")
        # Strip whitespace from column headers
        df.columns = [c.strip() for c in df.columns]
        # Replace string and float Infinities
        df = df.replace(["Infinity", "infinity", "inf", "-inf", "-Infinity"], np.nan)
        df = df.replace([np.inf, -np.inf, float("inf"), float("-inf")], np.nan)
        
        # Save to session datastore
        SESSION_DATASTORE["current_df"] = df
        SESSION_DATASTORE["filename"] = file.filename

        schema = validate_schema(df)

        # Label distribution if label column detected
        label_dist = None
        if schema["label_column"] and schema["label_column"] in df.columns:
            raw_dist = df[schema["label_column"]].astype(str).str.strip().value_counts().to_dict()
            label_dist = {str(k): int(v) for k, v in raw_dist.items()}

        # Compute summary stats for top numeric features
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:8].tolist()
        numeric_stats = {}
        for c in numeric_cols:
            series = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(series) > 0:
                numeric_stats[c] = {
                    "mean": round(float(series.mean()), 2),
                    "min": round(float(series.min()), 2),
                    "max": round(float(series.max()), 2),
                    "std": round(float(series.std()), 2)
                }

        # Preview rows (first 10 records)
        preview_cols = list(df.columns[:12])
        preview_df = df[preview_cols].head(10).copy()
        # Sanitize for JSON
        preview_rows = [sanitize_value(row.to_dict()) for _, row in preview_df.iterrows()]

        return UploadResponse(
            status="success",
            filename=file.filename,
            total_records=len(df),
            total_features=len(df.columns),
            detected_label_column=schema["label_column"],
            label_distribution=label_dist,
            missing_values_count=schema["total_missing_values"],
            preview_columns=preview_cols,
            preview_rows=preview_rows,
            numeric_feature_stats=numeric_stats
        )

    except Exception as e:
        logger.error(f"Error parsing uploaded CSV: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to process CSV: {str(e)}")


@router.get("/sample-data")
def get_sample_data():
    """
    Returns pre-loaded sample network traffic records from cicids2017_sample_test.csv
    for quick demonstration and testing in the UI.
    """
    sample_path = os.path.join(settings.DATA_DIR, "demo", "cicids2017_sample_test.csv")
    if not os.path.exists(sample_path):
        raise HTTPException(status_code=404, detail="Sample dataset file not found.")

    df = pd.read_csv(sample_path)
    df.columns = [c.strip() for c in df.columns]
    SESSION_DATASTORE["current_df"] = df
    SESSION_DATASTORE["filename"] = "cicids2017_sample_test.csv"

    preview_cols = list(df.columns[:10])
    preview_rows = [sanitize_value(row.to_dict()) for _, row in df.head(15).iterrows()]

    label_dist = {}
    if "Label" in df.columns:
        label_dist = {str(k): int(v) for k, v in df["Label"].value_counts().items()}

    return {
        "status": "success",
        "filename": "cicids2017_sample_test.csv",
        "total_records": len(df),
        "total_features": len(df.columns),
        "label_distribution": label_dist,
        "preview_columns": preview_cols,
        "preview_rows": preview_rows
    }


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze_traffic(req: AnalyzeRequest):
    """
    Executes supervised classification and unsupervised anomaly detection
    over network traffic records using the specified ML model.
    """
    # Determine input records
    if req.records and len(req.records) > 0:
        df = pd.DataFrame(req.records)
    elif SESSION_DATASTORE["current_df"] is not None:
        df = SESSION_DATASTORE["current_df"].copy()
        if req.use_sample and len(df) > req.sample_size:
            df = df.head(req.sample_size)
    else:
        # Load sample demo data automatically
        sample_path = os.path.join(settings.DATA_DIR, "demo", "cicids2017_sample_test.csv")
        if os.path.exists(sample_path):
            df = pd.read_csv(sample_path).head(150)
            df.columns = [c.strip() for c in df.columns]
            SESSION_DATASTORE["current_df"] = df
        else:
            raise HTTPException(status_code=400, detail="No dataset loaded. Upload a CSV or load sample data first.")

    # Limit max records
    if len(df) > settings.MAX_ANALYSIS_RECORDS:
        df = df.head(settings.MAX_ANALYSIS_RECORDS)

    try:
        predictions = prediction_service.analyze_dataframe(
            df=df,
            model_name=req.model_name,
            mode=req.mode
        )
    except FileNotFoundError as e:
        logger.warning(f"Requested model not found: {e}")
        raise HTTPException(status_code=404, detail=f"Model artifact not found: {str(e)}")
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    # Aggregate metrics
    total = len(predictions)
    suspicious_count = sum(1 for p in predictions if p["is_suspicious"])
    normal_count = total - suspicious_count
    anomaly_count = sum(1 for p in predictions if p["is_anomaly"])

    class_dist = {}
    for p in predictions:
        lbl = p["predicted_label"]
        class_dist[lbl] = class_dist.get(lbl, 0) + 1

    summary = AnalyzeSummary(
        total_analyzed=total,
        suspicious_count=suspicious_count,
        suspicious_percent=round((suspicious_count / max(total, 1)) * 100.0, 2),
        normal_count=normal_count,
        normal_percent=round((normal_count / max(total, 1)) * 100.0, 2),
        anomaly_count=anomaly_count,
        anomaly_percent=round((anomaly_count / max(total, 1)) * 100.0, 2),
        class_distribution=class_dist,
        model_used=req.model_name,
        mode=req.mode
    )

    analysis_id = str(uuid.uuid4())[:8]
    SESSION_DATASTORE["last_analysis"][analysis_id] = {
        "summary": summary.model_dump(),
        "predictions": predictions,
        "input_df": df
    }

    return AnalyzeResponse(
        summary=summary,
        predictions=[RecordPrediction(**p) for p in predictions],
        analysis_id=analysis_id
    )


@router.post("/predict", response_model=AnalyzeResponse)
def predict_traffic(req: AnalyzeRequest):
    """
    Alias endpoint for /analyze to support standard ML prediction API conventions.
    """
    return analyze_traffic(req)


@router.get("/download/{analysis_id}")
def export_predictions_csv(analysis_id: str):
    """
    Export classification and anomaly analysis records to a downloadable CSV file.
    """
    analysis = SESSION_DATASTORE["last_analysis"].get(analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail=f"Analysis ID '{analysis_id}' not found or session expired.")

    predictions = analysis["predictions"]
    input_df = analysis["input_df"]

    # Combine predictions with features
    export_rows = []
    for i, p in enumerate(predictions):
        row = {
            "Record_ID": p["record_index"],
            "Predicted_Label": p["predicted_label"],
            "Is_Suspicious": "Yes" if p["is_suspicious"] else "No",
            "Confidence": p["confidence"],
            "Anomaly_Score": p["anomaly_score"],
            "Is_Anomaly": "Yes" if p["is_anomaly"] else "No",
            "Traffic_Cluster": p["traffic_cluster"]
        }
        # Add key input features
        for col in input_df.columns:
            if col not in ["Label", "label", "Unnamed: 0"]:
                row[col] = input_df.iloc[i][col] if i < len(input_df) else None
        export_rows.append(row)

    out_df = pd.DataFrame(export_rows)
    stream = io.StringIO()
    out_df.to_csv(stream, index=False)
    
    response = StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv"
    )
    response.headers["Content-Disposition"] = f"attachment; filename=network_analysis_{analysis_id}.csv"
    return response
