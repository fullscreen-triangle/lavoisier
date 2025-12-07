"""
Parallel Functions for Mass Spectrometry Data Processing
========================================================

Utility functions for parallel processing of mass spectrometry data.
"""

import numpy as np
from typing import Tuple, List


def ppm_window_para(
    mz_value: float,
    ppm: float = 20.0
) -> Tuple[float, float]:
    """
    Calculate m/z window based on ppm tolerance.

    Args:
        mz_value: The m/z value
        ppm: Parts per million tolerance (default: 20 ppm)

    Returns:
        Tuple of (lower_bound, upper_bound)

    Example:
        >>> ppm_window_para(100.0, 20.0)
        (99.998, 100.002)
    """
    delta = mz_value * ppm / 1e6
    return (mz_value - delta, mz_value + delta)


def ppm_window_array(
    mz_array: np.ndarray,
    ppm: float = 20.0
) -> np.ndarray:
    """
    Calculate m/z windows for an array of m/z values.

    Args:
        mz_array: Array of m/z values
        ppm: Parts per million tolerance

    Returns:
        Array of shape (N, 2) with [lower_bound, upper_bound] for each m/z
    """
    delta = mz_array * ppm / 1e6
    lower = mz_array - delta
    upper = mz_array + delta
    return np.column_stack([lower, upper])


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
    lower, upper = ppm_window_para(target_mz, ppm)
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
