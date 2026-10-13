# NetGuard AI - Reports & Evaluation Directory

This directory stores reproducible evaluation metrics, comparison tables, and visual diagnostic plots automatically generated during pipeline execution.

## Generated Files
- `metrics_summary_binary.json` / `metrics_summary_multiclass.json`: Comprehensive JSON report containing accuracy, precision, recall, F1-scores, per-class metrics, confusion matrix values, ROC curve points, training duration, and prediction latency.
- `feature_importance_binary.json` / `feature_importance_multiclass.json`: Global Tree MDI feature importance scores and model-agnostic Permutation Importance on held-out test splits.
- `cm_<model>.png`: Confusion matrix heatmap images generated with Matplotlib and Seaborn.
- `roc_<model>.png`: Receiver Operating Characteristic (ROC) curve plots displaying empirical AUC scores.
