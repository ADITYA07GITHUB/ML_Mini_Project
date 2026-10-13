"""
NetGuard AI - Pydantic Request & Response Schemas
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "healthy"
    app_name: str
    version: str
    models_ready: bool
    python_version: str


class MetadataResponse(BaseModel):
    app_name: str
    version: str
    active_mode: str
    models_available: List[str]
    features_selected: List[str]
    classes: List[str]
    best_model: Optional[str] = None
    last_trained: Optional[str] = None
    disclaimer: str


class ModelInfo(BaseModel):
    name: str
    category: str
    supports_probabilities: bool
    description: str


class UploadResponse(BaseModel):
    status: str
    filename: str
    total_records: int
    total_features: int
    detected_label_column: Optional[str] = None
    label_distribution: Optional[Dict[str, int]] = None
    missing_values_count: int
    preview_columns: List[str]
    preview_rows: List[Dict[str, Any]]
    numeric_feature_stats: Dict[str, Dict[str, float]]


class AnalyzeRequest(BaseModel):
    model_name: str = Field(default="Random Forest")
    mode: str = Field(default="binary")
    records: Optional[List[Dict[str, Any]]] = None
    use_sample: bool = False
    sample_size: int = 100


class RecordPrediction(BaseModel):
    record_index: int
    predicted_label: str
    is_suspicious: bool
    confidence: float
    probabilities: Dict[str, float]
    is_anomaly: bool
    anomaly_score: float
    traffic_cluster: int
    raw_preview: Optional[Dict[str, Any]] = None
    full_record: Optional[Dict[str, Any]] = None


class AnalyzeSummary(BaseModel):
    total_analyzed: int
    suspicious_count: int
    suspicious_percent: float
    normal_count: int
    normal_percent: float
    anomaly_count: int
    anomaly_percent: float
    class_distribution: Dict[str, int]
    model_used: str
    mode: str


class AnalyzeResponse(BaseModel):
    summary: AnalyzeSummary
    predictions: List[RecordPrediction]
    analysis_id: str


class LocalExplainRequest(BaseModel):
    record: Optional[Dict[str, Any]] = None
    record_index: Optional[int] = None
    model_name: str = "Random Forest"
    mode: str = "binary"


class FeatureAttribution(BaseModel):
    feature: str
    shap_value: float
    direction: str
    observed_value: Optional[Any] = None
    description: str


class LocalExplainResponse(BaseModel):
    method: str
    predicted_label: str
    confidence: float
    is_suspicious: bool
    top_contributions: List[FeatureAttribution]
    disclaimer: str


class ErrorResponse(BaseModel):
    detail: str
    error_code: str
