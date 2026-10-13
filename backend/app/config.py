"""
NetGuard AI - Application Configuration
"""

import os
from pydantic import BaseModel

class Settings(BaseModel):
    APP_NAME: str = "Network Analysis Dashboard"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    BASE_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    MODELS_DIR: str = os.path.join(BASE_DIR, "models")
    REPORTS_DIR: str = os.path.join(BASE_DIR, "reports")
    DATA_DIR: str = os.path.join(BASE_DIR, "data")
    FRONTEND_DIR: str = os.path.join(BASE_DIR, "frontend")
    
    MAX_UPLOAD_SIZE_BYTES: int = 25 * 1024 * 1024  # 25 MB limit
    MAX_ANALYSIS_RECORDS: int = 10000              # Max records processed in single request
    DEFAULT_MODEL: str = "Random Forest"
    DEFAULT_MODE: str = "binary"

settings = Settings()
