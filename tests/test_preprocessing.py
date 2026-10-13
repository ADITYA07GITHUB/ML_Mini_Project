"""
NetGuard AI - Preprocessing & Data Loading Tests
"""

import os
import pytest
import numpy as np
import pandas as pd
from ml.data_loader import load_dataset, validate_schema, split_features_and_label
from ml.preprocessing import NetGuardPreprocessor, prepare_train_test_split


def test_schema_validation_and_clean_columns(tmp_path):
    # Create temporary CSV with raw CIC-IDS2017 quirks (spaces in headers, Infinity, NaN)
    csv_file = tmp_path / "raw_quirks.csv"
    data = {
        " Destination Port": [80, 443, 8080],
        " Flow Duration ": [1000, 2000, 3000],
        " Flow Packets/s": ["Infinity", "150.5", "NaN"],
        "Label ": ["BENIGN", "DDoS", "BENIGN"]
    }
    pd.DataFrame(data).to_csv(csv_file, index=False)

    df = load_dataset(str(csv_file))
    
    # Headers must be stripped
    assert "Destination Port" in df.columns
    assert "Flow Duration" in df.columns
    assert "Flow Packets/s" in df.columns
    assert "Label" in df.columns

    # Infinities and NaNs must be converted to float nan
    schema = validate_schema(df)
    assert schema["num_rows"] == 3
    assert schema["num_columns"] == 4
    assert schema["label_column"] == "Label"
    assert schema["total_missing_values"] >= 2


def test_preprocessor_fit_transform():
    # Synthetic small dataset
    df = pd.DataFrame({
        "Destination Port": [80, 443, 22, 8080, 53, 1234],
        "Flow Duration": [100.0, 500.0, 120.0, 450.0, 60.0, 900.0],
        "Flow Packets/s": [10.0, np.nan, 50.0, np.inf, 2.0, 100.0],
        "Total Fwd Packets": [2, 5, 1, 4, 1, 10],
        "Label": ["BENIGN", "BENIGN", "PortScan", "DDoS", "BENIGN", "DDoS"]
    })

    X, y = split_features_and_label(df, label_col="Label")
    
    prep = NetGuardPreprocessor(k_features=3)
    prep.fit(X, y, multiclass=False)

    assert prep.is_fitted
    assert len(prep.selected_feature_names) <= 3
    assert prep.classes_ == ["Normal", "Suspicious"]

    # Transform
    X_trans = prep.transform(X)
    assert X_trans.shape[0] == 6
    assert X_trans.shape[1] == len(prep.selected_feature_names)
    assert not np.isnan(X_trans).any()  # No NaNs remaining!

    y_num = prep.transform_labels(y)
    assert set(np.unique(y_num)).issubset({0, 1})


def test_train_test_split_no_leakage():
    # Verify split splits before fitting
    df = pd.DataFrame({
        "Destination Port": [80, 443, 22, 8080, 53, 1234, 80, 443],
        "Flow Duration": [100.0, 500.0, 120.0, 450.0, 60.0, 900.0, 200.0, 300.0],
        "Flow Packets/s": [10.0, 20.0, 50.0, 100.0, 2.0, 100.0, 15.0, 25.0],
        "Label": ["BENIGN", "BENIGN", "PortScan", "DDoS", "BENIGN", "DDoS", "BENIGN", "DDoS"]
    })

    X_train, X_test, y_train, y_test, prep, report = prepare_train_test_split(
        df=df, test_size=0.25, random_state=42, multiclass=False, k_features=2
    )

    assert len(X_train) == 6
    assert len(X_test) == 2
    assert prep.is_fitted
    assert not np.isnan(X_train).any()
    assert not np.isnan(X_test).any()
