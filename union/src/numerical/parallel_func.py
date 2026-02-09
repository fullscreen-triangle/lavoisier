"""
Parallel Functions for Mass Spectrometry Data Processing
========================================================

Utility functions for parallel processing of mass spectrometry data.
Uses numba vectorization when available, falls back to numpy otherwise.
"""

import numpy as np
from typing import Tuple, List, Union

# Try to import numba for vectorization
try:
    from numba import vectorize, float64, int64
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("[INFO] Numba not available, using numpy fallback")


if NUMBA_AVAILABLE:
    @vectorize([float64(float64, int64), float64(float64, float64)], target='cpu')
    def ppm_window_para(mz, ppm):
        """
        Calculate m/z with ppm offset (vectorized with numba).

        Args:
            mz: m/z value(s)
            ppm: Parts per million offset (positive or negative)

        Returns:
            mz * (1 + ppm/1e6)
        """
        return mz * (1 + 0.000001 * ppm)

    @vectorize([float64(float64, float64)], target='cpu')
    def pr_window_calc_para(mz, delta):
        """Calculate m/z + delta (vectorized)."""
        return mz + delta

    @vectorize([float64(float64, float64)], target='cpu')
    def ppm_calc_para(mz_obs, mz_lib):
        """Calculate ppm difference between observed and library m/z."""
        return 1e6 * (mz_obs - mz_lib) / mz_lib

else:
    # Numpy fallback implementations
    def ppm_window_para(mz, ppm):
        """
        Calculate m/z with ppm offset (numpy fallback).

        Args:
            mz: m/z value(s) - can be float or array
            ppm: Parts per million offset (positive or negative)

        Returns:
            mz * (1 + ppm/1e6)
        """
        mz_arr = np.asarray(mz)
        return mz_arr * (1 + 0.000001 * ppm)

    def pr_window_calc_para(mz, delta):
        """Calculate m/z + delta (numpy fallback)."""
        return np.asarray(mz) + delta

    def ppm_calc_para(mz_obs, mz_lib):
        """Calculate ppm difference between observed and library m/z."""
        mz_obs_arr = np.asarray(mz_obs)
        mz_lib_arr = np.asarray(mz_lib)
        return 1e6 * (mz_obs_arr - mz_lib_arr) / mz_lib_arr


def ppm_window_bounds(
    mz_value: Union[float, np.ndarray],
    ppm: float = 20.0
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Calculate m/z window bounds based on ppm tolerance.

    Args:
        mz_value: The m/z value(s)
        ppm: Parts per million tolerance (default: 20 ppm)

    Returns:
        Tuple of (lower_bound, upper_bound)

    Example:
        >>> ppm_window_bounds(100.0, 20.0)
        (99.998, 100.002)
    """
    lower = ppm_window_para(mz_value, -ppm)
    upper = ppm_window_para(mz_value, ppm)
    return (lower, upper)


def find_peaks_in_window(
    mz_array: np.ndarray,
    intensity_array: np.ndarray,
    target_mz: float,
    ppm: float = 20.0
) -> List[Tuple[float, float]]:
    """
    Find peaks within a ppm window around target m/z.

    Args:
        mz_array: Array of m/z values
        intensity_array: Array of intensity values
        target_mz: Target m/z value
        ppm: PPM tolerance

    Returns:
        List of (mz, intensity) tuples for peaks in window
    """
    lower, upper = ppm_window_bounds(target_mz, ppm)
    mask = (mz_array >= lower) & (mz_array <= upper)

    matching_mz = mz_array[mask]
    matching_intensity = intensity_array[mask]

    return list(zip(matching_mz, matching_intensity))


def mass_difference_ppm(
    mz1: float,
    mz2: float
) -> float:
    """
    Calculate mass difference in ppm.

    Args:
        mz1: First m/z value
        mz2: Second m/z value

    Returns:
        Mass difference in ppm
    """
    return abs(mz1 - mz2) / mz1 * 1e6
