"""
NetGuard AI - Comprehensive Model Evaluation Engine
====================================================
Computes real, mathematically rigorous evaluation metrics for network intrusion
detection pipelines, handling class imbalance, per-class breakdowns, confusion matrices,
ROC analysis, and execution latency.

All metrics are computed on held-out test splits. NO FABRICATED METRICS.
"""

import time
import logging
import os
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend for automated report generation
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def evaluate_classifier(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str],
    model_name: str = "Classifier",
    train_duration_sec: float = 0.0,
    reports_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluates a trained classifier on the test split.
    Computes accuracy, precision, recall, f1, per-class breakdown,
    confusion matrix, ROC-AUC (if supported), and prediction latency.
    """
    logger.info(f"Evaluating model: {model_name} on {len(X_test)} test samples...")
    
    # Measure prediction latency
    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    t1 = time.perf_counter()
    total_eval_time = t1 - t0
    latency_ms_per_sample = (total_eval_time / max(len(X_test), 1)) * 1000.0

    is_binary = len(class_names) == 2
    
    # Calculate global metrics
    acc = float(accuracy_score(y_test, y_pred))
    
    if is_binary:
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        prec_macro = prec
        rec_macro = rec
        f1_macro = f1
    else:
        prec_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
        rec_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
        f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        prec = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
        rec = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
        f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    # Per-class metrics
    clf_dict = classification_report(
        y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    per_class_metrics = {}
    for c_name in class_names:
        if c_name in clf_dict:
            per_class_metrics[c_name] = {
                "precision": round(float(clf_dict[c_name]["precision"]), 4),
                "recall": round(float(clf_dict[c_name]["recall"]), 4),
                "f1_score": round(float(clf_dict[c_name]["f1-score"]), 4),
                "support": int(clf_dict[c_name]["support"]),
            }

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = confusion_matrix(y_test, y_pred, normalize="true")
    
    # ROC-AUC if predict_proba or decision_function available
    roc_auc = None
    roc_data = None
    y_probs = None
    
    if hasattr(model, "predict_proba"):
        try:
            y_probs = model.predict_proba(X_test)
            if is_binary:
                # Probability of positive class (Suspicious = 1)
                roc_auc = float(roc_auc_score(y_test, y_probs[:, 1]))
                fpr, tpr, _ = roc_curve(y_test, y_probs[:, 1])
                # Sample down ROC points for clean JSON payload
                indices = np.linspace(0, len(fpr) - 1, min(50, len(fpr))).astype(int)
                roc_data = {
                    "fpr": [round(float(x), 4) for x in fpr[indices]],
                    "tpr": [round(float(x), 4) for x in tpr[indices]],
                    "auc": round(roc_auc, 4)
                }
            else:
                roc_auc = float(roc_auc_score(y_test, y_probs, multi_class="ovr", average="macro"))
        except Exception as e:
            logger.warning(f"Could not compute ROC-AUC for {model_name}: {e}")

    # Render and save plots if reports_dir is provided
    plot_paths = {}
    if reports_dir:
        os.makedirs(reports_dir, exist_ok=True)
        # 1. Confusion Matrix plot
        cm_path = os.path.join(reports_dir, f"cm_{model_name.lower().replace(' ', '_')}.png")
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f"Confusion Matrix: {model_name}")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        plt.savefig(cm_path, dpi=150)
        plt.close()
        plot_paths["confusion_matrix"] = cm_path

        # 2. ROC curve plot (binary)
        if roc_data:
            roc_path = os.path.join(reports_dir, f"roc_{model_name.lower().replace(' ', '_')}.png")
            plt.figure(figsize=(6, 5))
            plt.plot(roc_data["fpr"], roc_data["tpr"], label=f"{model_name} (AUC = {roc_auc:.3f})", color="#06B6D4", lw=2)
            plt.plot([0, 1], [0, 1], "k--", alpha=0.6)
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve: {model_name}")
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(roc_path, dpi=150)
            plt.close()
            plot_paths["roc_curve"] = roc_path

    result = {
        "model_name": model_name,
        "is_binary": is_binary,
        "class_names": class_names,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "macro_precision": round(prec_macro, 4),
        "macro_recall": round(rec_macro, 4),
        "macro_f1": round(f1_macro, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "roc_curve": roc_data,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": [[round(float(v), 3) for v in row] for row in cm_norm],
        "per_class": per_class_metrics,
        "training_time_sec": round(train_duration_sec, 3),
        "inference_latency_ms": round(latency_ms_per_sample, 4),
        "test_samples": len(X_test),
        "plots": plot_paths,
    }
    
    logger.info(f"{model_name} Results -> Accuracy: {acc:.4f}, F1: {f1:.4f}, Latency: {latency_ms_per_sample:.3f}ms")
    return result
