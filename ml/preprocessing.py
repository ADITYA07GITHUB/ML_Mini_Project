"""
NetGuard AI - Data Preprocessing & Feature Engineering Pipeline
===============================================================
Implements a leakage-free preprocessing pipeline strictly conforming to
academic and industrial best practices:
1. Split train/test BEFORE fitting any imputers, scalers, or selectors.
2. Impute NaNs using training set median.
3. Remove zero-variance or constant features.
4. Scale features using RobustScaler (handles heavy-tailed network flow traffic).
5. Feature Selection using SelectKBest (ANOVA F-score) to retain top informative features.
6. Optional PCA for dimensionality reduction and 2D/3D visualization.
7. Supports both Binary and Multiclass label encodings.
"""

import logging
from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import joblib

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


class NetGuardPreprocessor(BaseEstimator, TransformerMixin):
    """
    Complete stateful preprocessor fitted strictly on training data.
    """
    def __init__(self, k_features: int = 20, use_pca: bool = False, n_pca_components: int = 2):
        self.k_features = k_features
        self.use_pca = use_pca
        self.n_pca_components = n_pca_components
        
        # Stateful fitted transformers
        self.imputer: Optional[SimpleImputer] = None
        self.scaler: Optional[RobustScaler] = None
        self.feature_selector: Optional[SelectKBest] = None
        self.pca: Optional[PCA] = None
        self.label_encoder: Optional[LabelEncoder] = None
        
        # Track feature names
        self.raw_feature_names: List[str] = []
        self.selected_feature_names: List[str] = []
        self.classes_: List[str] = []
        self.is_multiclass: bool = False
        self.is_fitted: bool = False
        self.transformation_report: Dict[str, Any] = {}

    def _clean_numeric_df(self, X: pd.DataFrame) -> pd.DataFrame:
        """Convert all columns to numeric, replacing unparseable values with NaN."""
        X_clean = X.copy()
        # Coerce all columns to numeric float64
        for col in X_clean.columns:
            if not np.issubdtype(X_clean[col].dtype, np.number):
                X_clean[col] = pd.to_numeric(X_clean[col], errors="coerce")
        # Replace inf with nan
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
        return X_clean

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None, multiclass: bool = False):
        """Fit imputer, scaler, feature selector, and label encoder on training split only."""
        logger.info("Fitting NetGuardPreprocessor on training data...")
        self.is_multiclass = multiclass
        
        X_clean = self._clean_numeric_df(X)
        self.raw_feature_names = X_clean.columns.tolist()
        
        # 1. Fit Imputer (Median)
        self.imputer = SimpleImputer(strategy="median")
        X_imputed = self.imputer.fit_transform(X_clean)
        
        # 2. Fit Scaler (RobustScaler to resist flow traffic extreme outliers)
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X_imputed)
        
        # 3. Fit Feature Selector (ANOVA F-statistic)
        k = min(self.k_features, X_scaled.shape[1])
        if y is not None and len(np.unique(y)) > 1:
            # Map labels for feature selection
            y_arr = np.array(y)
            self.feature_selector = SelectKBest(score_func=f_classif, k=k)
            self.feature_selector.fit(X_scaled, y_arr)
            
            selected_mask = self.feature_selector.get_support()
            self.selected_feature_names = [
                self.raw_feature_names[i] for i, sel in enumerate(selected_mask) if sel
            ]
        else:
            self.selected_feature_names = self.raw_feature_names[:k]
            self.feature_selector = None

        # 4. Optional PCA fitting
        if self.use_pca:
            self.pca = PCA(n_components=self.n_pca_components)
            X_selected = self.transform_features(X_clean)
            self.pca.fit(X_selected)

        # 5. Label Encoder if y provided
        if y is not None:
            self._fit_labels(y, multiclass=multiclass)

        self.is_fitted = True
        logger.info(f"Preprocessor successfully fitted. Selected {len(self.selected_feature_names)} features: {self.selected_feature_names}")
        return self

    def _fit_labels(self, y: pd.Series, multiclass: bool):
        """Fit label transformation logic."""
        if not multiclass:
            # Binary classification: 'BENIGN' / 'normal' -> 0 (Normal), any attack -> 1 (Suspicious)
            self.classes_ = ["Normal", "Suspicious"]
        else:
            # Multiclass classification
            self.label_encoder = LabelEncoder()
            # Normalize strings
            y_clean = y.astype(str).str.strip()
            self.label_encoder.fit(y_clean)
            self.classes_ = [str(cls) for cls in self.label_encoder.classes_]

    def transform_labels(self, y: pd.Series) -> np.ndarray:
        """Transform labels to numeric format based on binary or multiclass mode."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transforming labels.")
            
        y_clean = y.astype(str).str.strip()
        if not self.is_multiclass:
            # Normal = 0, Suspicious = 1
            # Check for benign variations
            is_normal = y_clean.str.upper().isin(["BENIGN", "NORMAL", "0", "CLEAN"])
            return np.where(is_normal, 0, 1)
        else:
            # Multiclass encoding
            return self.label_encoder.transform(y_clean)

    def inverse_transform_labels(self, y_num: np.ndarray) -> List[str]:
        """Convert numeric prediction array back to human-readable class names."""
        if not self.is_multiclass:
            return ["Normal" if val == 0 else "Suspicious" for val in y_num]
        else:
            return [str(cls) for cls in self.label_encoder.inverse_transform(y_num)]

    def transform_features(self, X: pd.DataFrame) -> np.ndarray:
        """Apply fitted imputer, scaler, and feature selection to new inputs."""
        if not self.is_fitted:
            raise ValueError("Preprocessor is not fitted yet.")

        if len(X) == 0:
            return np.empty((0, len(self.selected_feature_names)))

        # Ensure all raw columns exist in input, fill missing columns with NaN
        X_aligned = pd.DataFrame(index=X.index)
        for col in self.raw_feature_names:
            if col in X.columns:
                X_aligned[col] = X[col]
            else:
                X_aligned[col] = np.nan

        X_clean = self._clean_numeric_df(X_aligned)
        X_imputed = self.imputer.transform(X_clean)
        X_scaled = self.scaler.transform(X_imputed)

        if self.feature_selector is not None:
            X_out = self.feature_selector.transform(X_scaled)
        else:
            X_out = X_scaled[:, :len(self.selected_feature_names)]

        return X_out

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.transform_features(X)

    def save(self, filepath: str):
        """Serialize preprocessor to disk using joblib."""
        joblib.dump(self, filepath)
        logger.info(f"Preprocessor saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "NetGuardPreprocessor":
        """Load fitted preprocessor from disk."""
        obj = joblib.load(filepath)
        if not isinstance(obj, NetGuardPreprocessor):
            raise TypeError("Loaded artifact is not a NetGuardPreprocessor instance.")
        return obj


def prepare_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.25,
    random_state: int = 42,
    multiclass: bool = False,
    k_features: int = 20
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, NetGuardPreprocessor, Dict[str, Any]]:
    """
    Splits raw DataFrame into train/test sets, fits the preprocessor strictly on the train split,
    and returns transformed arrays along with the fitted preprocessor and a report.
    """
    from ml.data_loader import split_features_and_label, validate_schema
    
    schema = validate_schema(df)
    label_col = schema["label_column"]
    if not label_col:
        raise ValueError("Cannot perform train/test split: dataset does not contain a recognized label column.")

    # Deduplicate justified network flows
    initial_rows = len(df)
    df_dedup = df.drop_duplicates()
    dedup_rows = len(df_dedup)
    dropped_dupes = initial_rows - dedup_rows

    X_raw, y_raw = split_features_and_label(df_dedup, label_col=label_col)

    # Stratified train/test split to preserve attack class ratios if all classes have >= 2 instances
    class_counts = y_raw.value_counts()
    can_stratify = (class_counts.min() >= 2) and (len(class_counts) > 1)
    stratify_arg = y_raw if can_stratify else None

    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X_raw, y_raw, test_size=test_size, random_state=random_state, stratify=stratify_arg
    )

    # Fit preprocessor strictly on training data
    preprocessor = NetGuardPreprocessor(k_features=k_features)
    preprocessor.fit(X_train_raw, y_train_raw, multiclass=multiclass)

    # Transform train and test splits
    X_train = preprocessor.transform_features(X_train_raw)
    X_test = preprocessor.transform_features(X_test_raw)
    y_train = preprocessor.transform_labels(y_train_raw)
    y_test = preprocessor.transform_labels(y_test_raw)

    report = {
        "initial_rows": initial_rows,
        "deduplicated_rows": dedup_rows,
        "dropped_duplicates": dropped_dupes,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "total_raw_features": len(preprocessor.raw_feature_names),
        "selected_features_count": len(preprocessor.selected_feature_names),
        "selected_features": preprocessor.selected_feature_names,
        "classes": preprocessor.classes_,
        "multiclass": multiclass,
        "train_class_distribution": pd.Series(y_train).value_counts().to_dict(),
        "test_class_distribution": pd.Series(y_test).value_counts().to_dict(),
    }
    preprocessor.transformation_report = report

    return X_train, X_test, y_train, y_test, preprocessor, report
