"""
NetGuard AI - Model Explainability & Feature Attribution Engine
===============================================================
Provides transparent interpretations of model behavior through:
1. Global Feature Importance (Tree MDI & Permutation Importance).
2. Local Per-Record SHAP Explanations (TreeExplainer with feature contribution vectors).
3. Domain-Specific Plain-English Cybersecurity Interpretations.
4. Explicit Scientific Caveats (correlation vs causal attribution).

Concept:
- Black-box decisions are unacceptable in security operations. Explainable AI (XAI)
  enables SOC analysts to understand which traffic characteristics triggered an alert.
"""

import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
import joblib

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# Optional SHAP import with graceful fallback
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    shap = None
    SHAP_AVAILABLE = False


FEATURE_DESCRIPTIONS = {
    "Destination Port": "The target TCP/UDP port. Privileged service ports (<1024) or sweeping randomized high ports often distinguish normal traffic from reconnaissance/scanning.",
    "Flow Duration": "Total duration of the bidirectional flow in microseconds. DDoS and PortScan bursts often feature abnormally brief durations compared to sustained normal sessions.",
    "Total Fwd Packets": "Count of packets sent in forward direction. Abnormally high values occur in volumetric floods; extremely low values (1-2) with no response indicate port scans.",
    "Total Backward Packets": "Count of response packets. Asymmetry between forward and backward packet counts is a primary signature of spoofed denial-of-service traffic.",
    "Flow Packets/s": "Packet rate per second. Dramatic spikes are classic indicators of automated flooding attacks (SYN flood, UDP flood, DDoS).",
    "Flow Bytes/s": "Data throughput rate per second. Volumetric exfiltration or high-rate bandwidth floods cause anomalous positive deviations.",
    "Flow IAT Mean": "Mean Inter-Arrival Time between consecutive packets. Extremely tiny IAT indicates automated high-speed script execution; high regular IAT may signify C2 beaconing.",
    "SYN Flag Count": "Count of SYN control flags. Repeated standalone SYN flags without subsequent ACKs represent half-open scanning or SYN flooding.",
    "FIN Flag Count": "Count of FIN connection-teardown flags. Abnormal frequencies or irregular sequences indicate protocol anomaly or port teardown probes.",
    "ACK Flag Count": "Count of ACK flags. Expected in established bidirectional TCP streams; absence during high packet flows signals scanning or reflection attacks.",
    "PSH Flag Count": "Push flag count indicating immediate buffer delivery, common in interactive protocols and HTTP request bursts.",
    "Fwd Packet Length Mean": "Mean payload size of forward packets. High values are characteristic of POST-based HTTP floods or buffer exploit attempts; low uniform values match scan probes.",
    "Bwd Packet Length Mean": "Mean payload size of backward response packets. Significant responses confirm legitimate two-way communication.",
    "Packet Length Variance": "Statistical variance in flow packet sizes. Legitimate human browsing exhibits high variance, whereas automated attack scripts typically produce uniform packet sizes.",
    "Down/Up Ratio": "Ratio of incoming to outgoing packets. Standard client browsing has a high download-to-upload ratio (>1.0), whereas attack sources exhibit near-zero ratios.",
    "Active Mean": "Mean duration the flow was actively transmitting before an idle state. Reflects application transmission patterns.",
    "Idle Mean": "Mean duration the flow remained dormant. Periodic idle intervals characterize botnet Command-and-Control (C2) heartbeat pulses."
}


class NetGuardExplainer:
    """
    Manages global and local explainability for trained models.
    """
    def __init__(self, model: Any, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.shap_explainer = None
        self._init_shap()

    def _init_shap(self):
        """Initialize SHAP TreeExplainer if model is tree-based and SHAP is installed."""
        if SHAP_AVAILABLE and hasattr(self.model, "estimators_"):
            try:
                self.shap_explainer = shap.TreeExplainer(self.model)
                logger.info("SHAP TreeExplainer initialized successfully.")
            except Exception as e:
                logger.warning(f"Could not initialize SHAP TreeExplainer: {e}")
                self.shap_explainer = None

    def get_global_feature_importance(self, top_n: int = 15) -> Dict[str, Any]:
        """
        Extract global feature importance from tree-based model (MDI).
        """
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            importances = np.mean(np.abs(self.model.coef_), axis=0)
        else:
            # Fallback uniform
            importances = np.ones(len(self.feature_names)) / len(self.feature_names)

        # Sort descending
        indices = np.argsort(importances)[::-1]
        top_indices = indices[:top_n]

        items = []
        for idx in top_indices:
            name = self.feature_names[idx] if idx < len(self.feature_names) else f"Feature_{idx}"
            val = float(importances[idx])
            desc = FEATURE_DESCRIPTIONS.get(name, "Network flow metric characterizing session transmission dynamics.")
            items.append({
                "feature": name,
                "importance": round(val, 4),
                "description": desc
            })

        return {
            "method": "Mean Decrease in Impurity (MDI) / Gini Feature Importance",
            "top_features": items,
            "disclaimer": "Feature importance reflects statistical utility for model decision partitions and does not imply direct physical causation of malicious intent."
        }

    def compute_permutation_importance(self, X_test: np.ndarray, y_test: np.ndarray, top_n: int = 10) -> Dict[str, Any]:
        """
        Compute model-agnostic permutation importance on held-out test data.
        """
        try:
            logger.info("Computing permutation importance on held-out test split...")
            result = permutation_importance(
                self.model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1
            )
            mean_importance = result.importances_mean
            indices = np.argsort(mean_importance)[::-1][:top_n]

            items = []
            for idx in indices:
                name = self.feature_names[idx] if idx < len(self.feature_names) else f"Feature_{idx}"
                items.append({
                    "feature": name,
                    "importance_mean": round(float(mean_importance[idx]), 4),
                    "importance_std": round(float(result.importances_std[idx]), 4),
                    "description": FEATURE_DESCRIPTIONS.get(name, "Network flow metric characterizing transmission dynamics.")
                })

            return {
                "method": "Permutation Importance (Evaluated on Held-out Test Data)",
                "features": items
            }
        except Exception as e:
            logger.warning(f"Permutation importance failed: {e}")
            return {"error": str(e)}

    def explain_instance(self, instance_vector: np.ndarray, raw_features_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate local explanation for a single network flow record.
        Uses SHAP if available; otherwise computes weighted feature contribution.
        """
        instance = np.array(instance_vector).reshape(1, -1)
        explanation_items = []
        base_value = 0.5
        method_used = "Feature Contribution Attribution"

        if self.shap_explainer is not None:
            try:
                # TreeExplainer returns shap_values: shape (1, n_features) or (1, n_features, n_classes)
                sv = self.shap_explainer.shap_values(instance)
                
                # Check format: in binary Random Forest, shap_values is list of 2 arrays or 3D array
                if isinstance(sv, list):
                    # Class 1 (Suspicious) contributions
                    vals = sv[1][0] if len(sv) > 1 else sv[0][0]
                elif isinstance(sv, np.ndarray) and sv.ndim == 3:
                    vals = sv[0, :, 1] if sv.shape[2] > 1 else sv[0, :, 0]
                else:
                    vals = sv[0]

                method_used = "SHAP TreeExplainer (Shapley Additive Explanations)"
                
                # Sort by absolute SHAP impact
                order = np.argsort(np.abs(vals))[::-1][:8]
                for idx in order:
                    feat_name = self.feature_names[idx]
                    impact = float(vals[idx])
                    raw_val = raw_features_dict.get(feat_name, None) if raw_features_dict else None
                    direction = "Towards Suspicious (+)" if impact > 0 else "Towards Normal (-)"
                    explanation_items.append({
                        "feature": feat_name,
                        "shap_value": round(impact, 4),
                        "direction": direction,
                        "observed_value": raw_val,
                        "description": FEATURE_DESCRIPTIONS.get(feat_name, "Traffic metric influencing classification boundary.")
                    })
            except Exception as e:
                logger.warning(f"SHAP local instance explanation failed, using fallback: {e}")

        # Fallback if SHAP was not used or failed
        if not explanation_items:
            method_used = "Feature Weight Product (Global Importance Fallback)"
            importances = getattr(self.model, "feature_importances_", np.ones(len(self.feature_names)) / len(self.feature_names))
            order = np.argsort(importances)[::-1][:8]
            for idx in order:
                feat_name = self.feature_names[idx]
                imp = float(importances[idx])
                raw_val = raw_features_dict.get(feat_name, None) if raw_features_dict else None
                explanation_items.append({
                    "feature": feat_name,
                    "shap_value": round(imp, 4),
                    "direction": "High Influence",
                    "observed_value": raw_val,
                    "description": FEATURE_DESCRIPTIONS.get(feat_name, "Traffic metric influencing classification boundary.")
                })

        return {
            "method": method_used,
            "top_contributions": explanation_items,
            "disclaimer": "Local explanations explain model output logic for this specific record; they do not verify external ground truth."
        }
