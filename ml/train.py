"""
NetGuard AI - Model Training & Experiment Pipeline
===================================================
Reproducible training workflow implementing:
- Dataset validation & preprocessing
- Train/test splitting without leakage
- Training multiple algorithms:
    * Random Forest Classifier
    * Support Vector Machine (RBF kernel)
    * K-Nearest Neighbors (KNN)
    * Baseline Logistic Regression
    * Multilayer Perceptron (MLP Neural Network)
- Isolation Forest unsupervised anomaly detector
- Model evaluation & comparison table generation
- Serialization to models/ and reports/

Usage:
    python -m ml.train --data data/demo/cicids2017_demo.csv --mode binary
"""

import os
import sys
import time
import json
import argparse
import logging
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import joblib

from ml.data_loader import load_dataset, validate_schema
from ml.preprocessing import prepare_train_test_split, NetGuardPreprocessor
from ml.anomaly_detection import NetGuardAnomalyDetector
from ml.evaluate import evaluate_classifier
from ml.explainability import NetGuardExplainer

logger = logging.getLogger("ml.train")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def train_pipeline(
    data_path: str,
    mode: str = "binary",
    k_features: int = 20,
    out_dir: str = "models",
    reports_dir: str = "reports",
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Executes the complete reproducible training workflow.
    """
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    is_multiclass = (mode.lower() == "multiclass")
    logger.info(f"Starting NetGuard AI training workflow. Mode: {mode.upper()} | Data: {data_path}")

    # 1. Load dataset
    df = load_dataset(data_path)
    schema = validate_schema(df)
    logger.info(f"Loaded {schema['num_rows']} records with {schema['num_columns']} columns.")

    # 2. Preprocess and split without data leakage
    X_train, X_test, y_train, y_test, preprocessor, prep_report = prepare_train_test_split(
        df=df,
        test_size=0.25,
        random_state=random_seed,
        multiclass=is_multiclass,
        k_features=k_features
    )
    logger.info(f"Train split: {len(X_train)} samples | Test split: {len(X_test)} samples")
    
    # Save fitted preprocessor
    preprocessor_path = os.path.join(out_dir, f"preprocessor_{mode}.joblib")
    preprocessor.save(preprocessor_path)

    # 3. Define candidate classifiers
    candidate_models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=16, random_state=random_seed, n_jobs=-1
        ),
        "Support Vector Machine": SVC(
            kernel="rbf", C=1.0, probability=True, max_iter=2500, random_state=random_seed
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=5, n_jobs=-1
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=random_seed
        ),
        "Multilayer Perceptron": MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=350, random_state=random_seed
        )
    }

    results = {}
    saved_model_paths = {}

    # 4. Train and evaluate each classifier
    for name, model in candidate_models.items():
        logger.info(f"\n---> Training: {name}...")
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_duration = time.perf_counter() - t0
        logger.info(f"Fitted {name} in {train_duration:.2f} seconds.")

        # Save model
        clean_name = name.lower().replace(" ", "_")
        model_filename = f"{clean_name}_{mode}.joblib"
        model_path = os.path.join(out_dir, model_filename)
        joblib.dump(model, model_path)
        saved_model_paths[name] = model_filename

        # Evaluate on held-out test split
        eval_metrics = evaluate_classifier(
            model=model,
            X_test=X_test,
            y_test=y_test,
            class_names=preprocessor.classes_,
            model_name=name,
            train_duration_sec=train_duration,
            reports_dir=reports_dir
        )
        results[name] = eval_metrics

    # 5. Train Anomaly Detector (Isolation Forest)
    logger.info("\n---> Training Unsupervised Anomaly Detector (Isolation Forest)...")
    anomaly_detector = NetGuardAnomalyDetector(contamination=0.10, random_state=random_seed)
    
    # If binary, fit on normal class samples where available for true semi-supervised baseline
    if not is_multiclass:
        normal_idx = np.where(y_train == 0)[0]
        X_anomaly_train = X_train[normal_idx] if len(normal_idx) > 100 else X_train
    else:
        # Multiclass: fit on class 0 (Benign/Normal)
        normal_idx = np.where(y_train == 0)[0]
        X_anomaly_train = X_train[normal_idx] if len(normal_idx) > 100 else X_train
        
    anomaly_detector.fit(X_anomaly_train, feature_names=preprocessor.selected_feature_names)
    anomaly_model_path = os.path.join(out_dir, "anomaly_detector.joblib")
    anomaly_detector.save(anomaly_model_path)

    # 6. Compute Feature Importance & Explainability (Random Forest)
    logger.info("\n---> Generating Feature Importance & Explainability Data...")
    rf_model = candidate_models["Random Forest"]
    explainer = NetGuardExplainer(rf_model, preprocessor.selected_feature_names)
    global_importance = explainer.get_global_feature_importance(top_n=15)
    perm_importance = explainer.compute_permutation_importance(X_test, y_test, top_n=10)

    explainability_payload = {
        "global_importance": global_importance,
        "permutation_importance": perm_importance,
        "selected_features": preprocessor.selected_feature_names
    }
    
    expl_path = os.path.join(reports_dir, f"feature_importance_{mode}.json")
    with open(expl_path, "w") as f:
        json.dump(explainability_payload, f, indent=2)

    # 7. Generate Comparison Summary Table
    comparison_table = []
    for m_name, res in results.items():
        comparison_table.append({
            "model": m_name,
            "accuracy": res["accuracy"],
            "precision": res["precision"],
            "recall": res["recall"],
            "f1_score": res["f1_score"],
            "macro_f1": res["macro_f1"],
            "roc_auc": res.get("roc_auc"),
            "training_time_sec": res["training_time_sec"],
            "latency_ms": res["inference_latency_ms"]
        })

    # Sort comparison table by F1-score descending
    comparison_table.sort(key=lambda x: x["f1_score"], reverse=True)

    summary = {
        "dataset_path": data_path,
        "mode": mode,
        "classes": preprocessor.classes_,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "preprocessing_report": prep_report,
        "comparison_table": comparison_table,
        "models_evaluated": list(results.keys()),
        "best_model": comparison_table[0]["model"],
        "saved_model_paths": saved_model_paths,
        "detailed_results": results
    }

    # Save summary report
    summary_path = os.path.join(reports_dir, f"metrics_summary_{mode}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Also update general metadata.json in models/
    meta_path = os.path.join(out_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump({
            "active_mode": mode,
            "last_trained": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_rows": schema["num_rows"],
            "features_selected": preprocessor.selected_feature_names,
            "classes": preprocessor.classes_,
            "best_model": comparison_table[0]["model"],
            "models_available": list(saved_model_paths.keys())
        }, f, indent=2)

    logger.info("\n=======================================================")
    logger.info("   TRAINING & EVALUATION PIPELINE COMPLETED")
    logger.info("=======================================================")
    for row in comparison_table:
        logger.info(f"{row['model']:<25} | Acc: {row['accuracy']:.4f} | F1: {row['f1_score']:.4f} | Latency: {row['latency_ms']:.2f}ms")
    logger.info("=======================================================\n")

    return summary


def main():
    parser = argparse.ArgumentParser(description="NetGuard AI - Model Training Pipeline")
    parser.add_argument("--data", type=str, default="data/demo/cicids2017_demo.csv", help="Path to CSV dataset")
    parser.add_argument("--mode", type=str, default="binary", choices=["binary", "multiclass"], help="Classification mode")
    parser.add_argument("--k-features", type=int, default=20, help="Number of features to select")
    parser.add_argument("--out-dir", type=str, default="models", help="Model artifact directory")
    parser.add_argument("--reports-dir", type=str, default="reports", help="Reports and plots directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    train_pipeline(
        data_path=args.data,
        mode=args.mode,
        k_features=args.k_features,
        out_dir=args.out_dir,
        reports_dir=args.reports_dir,
        random_seed=args.seed
    )

if __name__ == "__main__":
    main()
