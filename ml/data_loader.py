"""
NetGuard AI - Network Traffic Data Loader
==========================================
Robust dataset loader designed specifically for CIC-IDS2017 and network flow CSVs.
Handles leading/trailing whitespaces in column names, varied encodings, infinite values,
and provides schema validation.
"""

import os
import logging
from typing import Tuple, Dict, Any, List, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def load_dataset(file_path: str, max_rows: Optional[int] = None) -> pd.DataFrame:
    """
    Safely load a network flow CSV dataset into a Pandas DataFrame.
    Tries utf-8 first, then latin-1 / ISO-8859-1.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")

    logger.info(f"Loading dataset from: {file_path}")
    encodings = ["utf-8", "latin-1", "cp1252"]
    df = None
    
    for enc in encodings:
        try:
            df = pd.read_csv(file_path, encoding=enc, nrows=max_rows, low_memory=False)
            logger.info(f"Successfully read CSV with encoding '{enc}'. Shape: {df.shape}")
            break
        except (UnicodeDecodeError, Exception) as e:
            logger.warning(f"Failed to read with encoding {enc}: {e}")
            continue

    if df is None:
        raise ValueError(f"Unable to read dataset at {file_path} with supported encodings.")

    # 1. Clean column names: strip whitespace (crucial for CIC-IDS2017 raw files)
    df.columns = [c.strip() for c in df.columns]

    # 2. Convert string and float 'Infinity', 'inf', 'NaN' to numpy nan
    df = df.replace(["Infinity", "infinity", "inf", "INF", "Inf"], np.nan)
    df = df.replace(["-Infinity", "-infinity", "-inf", "-INF", "-Inf"], np.nan)
    df = df.replace(["NaN", "nan", "NA", "null", ""], np.nan)
    df = df.replace([np.inf, -np.inf, float("inf"), float("-inf")], np.nan)

    return df


def validate_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate the structure and contents of a network flow dataset.
    Returns metadata including column count, row count, detected label column, and missing values.
    """
    label_candidates = ["Label", "label", "Attack", "attack", "Class", "class"]
    detected_label = None
    for cand in label_candidates:
        if cand in df.columns:
            detected_label = cand
            break

    # Count infs / NaNs
    null_counts = df.isnull().sum()
    total_nulls = int(null_counts.sum())
    cols_with_nulls = {k: int(v) for k, v in null_counts.items() if v > 0}

    # Identify numeric vs object columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = [c for c in df.columns if c not in numeric_cols]

    return {
        "num_rows": int(len(df)),
        "num_columns": int(len(df.columns)),
        "columns": df.columns.tolist(),
        "numeric_columns_count": len(numeric_cols),
        "non_numeric_columns": non_numeric_cols,
        "label_column": detected_label,
        "total_missing_values": total_nulls,
        "columns_with_missing": cols_with_nulls,
    }


def split_features_and_label(df: pd.DataFrame, label_col: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Split the dataset into feature matrix X and ground truth label y.
    Excludes identifiers, timestamps, and target columns from model features.
    """
    if label_col is None:
        # Attempt auto-detection
        schema = validate_schema(df)
        label_col = schema["label_column"]

    # Non-informative columns to drop if present (IP addresses, flow ID, timestamps cause leakage)
    drop_candidates = [
        "Flow ID", "Source IP", "Destination IP", "Timestamp",
        "Source Port", "Unnamed: 0", "id", "Index"
    ]
    drop_cols = [c for c in drop_candidates if c in df.columns]

    if label_col and label_col in df.columns:
        y = df[label_col].copy()
        X = df.drop(columns=[label_col] + drop_cols, errors="ignore")
    else:
        y = None
        X = df.drop(columns=drop_cols, errors="ignore")

    return X, y
