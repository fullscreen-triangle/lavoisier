import numpy as np
import pandas as pd
from typing import Optional

def normalize(df: pd.DataFrame, method: str, feature: Optional[str] = None) -> pd.DataFrame:
    """Normalize samples using different methods."""
    if method == "sum":
        normalized = df.divide(df.sum(axis=1), axis=0)
    elif method == "max":
        normalized = df.divide(df.max(axis=1), axis=0)
    elif method == "euclidean":
        normalized = df.apply(lambda x: x / np.linalg.norm(x), axis=1)
    elif method == "feature":
        normalized = df.divide(df[feature], axis=0)
    else:
        msg = "method must be `sum`, `max`, `euclidean` or `feature`."
        raise ValueError(msg)
    normalized[normalized.isna()] = 0
    return normalized

def scale(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """scales features using different methods."""
    if method == "autoscaling":
        scaled = (df - df.mean()) / df.std()
    elif method == "rescaling":
        scaled = (df - df.min()) / (df.max() - df.min())
    elif method == "pareto":
        scaled = (df - df.mean()) / df.std().apply(np.sqrt)
    else:
        msg = "Available methods are `autoscaling`, `rescaling` and `pareto`."
        raise ValueError(msg)
    scaled[scaled.isna()] = 0
    return scaled

def transform(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """perform common data transformations."""
    if method == "log":
        transformed = df.apply(np.log10)
    elif method == "power":
        transformed = df.apply(np.sqrt)
    else:
        msg = "Available methods are `log` and `power`"
        raise ValueError(msg)
    return transformed
