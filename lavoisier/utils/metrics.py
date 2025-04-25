import os
import numpy as np
import pandas as pd
from typing import Optional, Union
from scipy.stats import spearmanr
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.stattools import jarque_bera, durbin_watson
from statsmodels.tools.tools import add_constant

def cv(df: pd.DataFrame, fill_value: Optional[float] = None) -> pd.Series:
    """Computes the Coefficient of Variation for each column."""
    res = df.std() / df.mean()
    if fill_value is not None:
        res = res.fillna(fill_value)
    return res

def robust_cv(df: pd.DataFrame, fill_value: Optional[float] = None) -> pd.Series:
    """Estimation of the coefficient of variation using the MAD and median."""
    res = mad(df) / df.median()
    if fill_value is not None:
        res = res.fillna(fill_value)
    return res

def mad(df: pd.DataFrame) -> pd.Series:
    """Computes the median absolute deviation for each column."""
    if df.shape[0] == 1:
        return pd.Series(data=np.nan, index=df.columns)
    else:
        return df.apply(lambda x: np.median(np.abs(x - np.median(x))))

def mean_absolute_deviation(df: pd.DataFrame) -> pd.Series:
    """Computes the mean absolute deviation for each column."""
    return df.apply(lambda x: np.mean(np.abs(x - np.mean(x))))

def sd_ratio(df1: pd.DataFrame, df2: pd.DataFrame, robust: bool = False,
             fill_value: Optional[float] = None) -> pd.Series:
    """Computes the ratio between standard deviations."""
    if robust:
        ratio = mad(df1) / mad(df2)
    else:
        ratio = df1.std() / df2.std()
    if fill_value is not None:
        ratio = ratio.fillna(fill_value)
    return ratio

def detection_rate(df: pd.DataFrame, threshold: float = 0.0) -> pd.Series:
    """Computes fraction of values above threshold."""
    dr = (df > threshold).sum().astype(int) / df.shape[0]
    return dr

def metadata_correlation(y: pd.Series, x: pd.Series, mode: str = "ols") -> dict:
    """Computes correlation metrics between two variables."""
    if mode == "ols":
        ols = OLS(y, add_constant(x)).fit()
        r2 = ols.rsquared
        jb = jarque_bera(ols.resid)[1]
        dw = durbin_watson(ols.resid)
        res = {"r2": r2, "DW": dw, "JB": jb}
    else:
        res = spearmanr(y, x)[0]
    return res

def sample_to_path(samples, path):
    """Map sample names to raw path if available."""
    available_files = os.listdir(path)
    filenames = [os.path.splitext(x)[0] for x in available_files]
    full_path = [os.path.join(path, x) for x in available_files]
    d = dict()
    for k, name in enumerate(filenames):
        if name in samples:
            d[name] = full_path[k]
    return d

def _find_closest_sorted(x: np.ndarray, xq: Union[np.ndarray, float, int]) -> np.ndarray:
    """Find the index in x closest to each xq element."""
    if isinstance(xq, (float, int)):
        xq = np.array(xq)
    if (x.size == 0) or (xq.size == 0):
        msg = "`x` and `xq` must be non-empty arrays"
        raise ValueError(msg)
    ind = np.searchsorted(x, xq)
    if ind.size == 1:
        if ind == 0:
            return ind
        elif ind == x.size:
            return ind - 1
        else:
            return ind - ((xq - x[ind - 1]) < (x[ind] - xq))
    else:
        mask = (ind > 0) & (ind < x.size)
        ind[mask] -= (xq[mask] - x[ind[mask] - 1]) < (x[ind[mask]] - xq[mask])
        ind[ind == x.size] = x.size - 1
        return ind

def find_closest(x: np.ndarray, xq: Union[np.ndarray, float, int],
                 is_sorted: bool = True) -> np.ndarray:
    """Find closest values in array."""
    if is_sorted:
        return _find_closest_sorted(x, xq)
    else:
        sorted_index = np.argsort(x)
        closest_index = _find_closest_sorted(x[sorted_index], xq)
        return sorted_index[closest_index]
