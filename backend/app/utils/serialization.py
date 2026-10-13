"""
NetGuard AI - Safe Serialization Utilities
"""

import math
from typing import Any
import numpy as np
import pandas as pd


def sanitize_value(val: Any) -> Any:
    """Recursively converts NumPy, Pandas, NaN, and Inf values into standard JSON-compliant Python types."""
    if val is None:
        return None
    try:
        if pd.isna(val) and not isinstance(val, (list, tuple, dict, np.ndarray)):
            return None
    except Exception:
        pass
    if isinstance(val, (np.integer, int)):
        return int(val)
    if isinstance(val, (np.floating, float)):
        if math.isnan(val) or np.isnan(val):
            return None
        if math.isinf(val) or np.isinf(val):
            return 999999999.0 if val > 0 else -999999999.0
        return float(val)
    if isinstance(val, (np.bool_, bool)):
        return bool(val)
    if isinstance(val, np.ndarray):
        return [sanitize_value(x) for x in val]
    if isinstance(val, pd.Series):
        return [sanitize_value(x) for x in val]
    if isinstance(val, dict):
        return {str(k): sanitize_value(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [sanitize_value(x) for x in val]
    return str(val)
