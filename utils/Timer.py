import logging
import time
from pathlib import Path
from typing import Optional, Union, Dict, Any
from contextlib import contextmanager
import numpy as np
import cv2
from datetime import datetime


class Timer:
    """Simple timer class for performance monitoring"""

    def __init__(self):
        self.start_time = None
        self.history: Dict[str, float] = {}

    @contextmanager
    def measure(self, name: str):
        """Context manager for measuring execution time of a code block"""
        start = time.perf_counter()
        yield
        elapsed = time.perf_counter() - start
        self.history[name] = elapsed

    def summary(self) -> Dict[str, float]:
        """Return summary of all timed operations"""
        return self.history

    def reset(self):
        """Reset timer history"""
        self.history.clear()


def setup_logging(
        name: str,
        level: int = logging.INFO,
        log_file: Optional[Union[str, Path]] = None
) -> logging.Logger:
    """Setup logging configuration"""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if log_file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if it doesn't"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_spectrum(
        mz_array: np.ndarray,
        intensity_array: np.ndarray,
        mode: str = 'max'
) -> np.ndarray:
    """Normalize spectrum intensities"""
    if mode == 'max':
        return intensity_array / np.max(intensity_array)
    elif mode == 'sum':
        return intensity_array / np.sum(intensity_array)
    elif mode == 'tic':
        return intensity_array / np.trapz(intensity_array, mz_array)
    else:
        raise ValueError(f"Unknown normalization mode: {mode}")


def align_spectra(
        reference: np.ndarray,
        target: np.ndarray,
        max_shift: float = 0.5
) -> np.ndarray:
    """Align target spectrum to reference using cross-correlation"""
    corr = np.correlate(reference, target, mode='full')
    shift = np.argmax(corr) - len(reference) + 1

    if abs(shift) > max_shift * len(reference):
        return target

    return np.roll(target, -shift)


def calculate_snr(
        intensity_array: np.ndarray,
        window_size: int = 50
) -> float:
    """Calculate signal-to-noise ratio"""
    # Estimate noise as median of moving standard deviation
    noise = np.median([
        np.std(intensity_array[i:i + window_size])
        for i in range(0, len(intensity_array) - window_size, window_size // 2)
    ])

    signal = np.max(intensity_array)
    return signal / noise if noise > 0 else float('inf')


def generate_timestamp() -> str:
    """Generate formatted timestamp"""
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def hash_spectrum(
        mz_array: np.ndarray,
        intensity_array: np.ndarray,
        bins: int = 1024
) -> np.ndarray:
    """Generate a hash representation of a spectrum"""
    # Create histogram representation
    hist, _ = np.histogram(mz_array, bins=bins, weights=intensity_array)
    # Normalize
    hist = hist / np.max(hist)
    # Convert to binary hash
    return (hist > np.median(hist)).astype(np.uint8)


def find_peaks(
        mz_array: np.ndarray,
        intensity_array: np.ndarray,
        height_threshold: float = 0.1,
        distance: int = 10
) -> Dict[str, np.ndarray]:
    """Find peaks in spectrum"""
    # Normalize intensities
    normalized = normalize_spectrum(mz_array, intensity_array)

    # Find peaks
    peaks = []
    for i in range(1, len(normalized) - 1):
        if (normalized[i] > height_threshold and
                normalized[i] > normalized[i - 1] and
                normalized[i] > normalized[i + 1]):

            # Check distance from previous peak
            if not peaks or (i - peaks[-1]) >= distance:
                peaks.append(i)

    peaks = np.array(peaks)

    return {
        'indices': peaks,
        'mz_values': mz_array[peaks],
        'intensities': intensity_array[peaks]
    }


def estimate_resolution(
        mz_array: np.ndarray,
        intensity_array: np.ndarray,
        peak_indices: np.ndarray
) -> np.ndarray:
    """Estimate resolution at each peak (FWHM)"""
    resolutions = []

    for peak_idx in peak_indices:
        peak_intensity = intensity_array[peak_idx]
        half_max = peak_intensity / 2

        # Find left intersection
        left_idx = peak_idx
        while left_idx > 0 and intensity_array[left_idx] > half_max:
            left_idx -= 1

        # Find right intersection
        right_idx = peak_idx
        while right_idx < len(intensity_array) - 1 and intensity_array[right_idx] > half_max:
            right_idx += 1

        # Calculate FWHM
        fwhm = mz_array[right_idx] - mz_array[left_idx]
        resolution = mz_array[peak_idx] / fwhm if fwhm > 0 else float('inf')
        resolutions.append(resolution)

    return np.array(resolutions)


def smooth_spectrum(
        intensity_array: np.ndarray,
        window: int = 5,
        polyorder: int = 2
) -> np.ndarray:
    """Apply Savitzky-Golay smoothing to spectrum"""
    from scipy.signal import savgol_filter
    return savgol_filter(intensity_array, window, polyorder)


def interpolate_spectrum(
        mz_array: np.ndarray,
        intensity_array: np.ndarray,
        new_mz_points: np.ndarray
) -> np.ndarray:
    """Interpolate spectrum to new m/z points"""
    from scipy.interpolate import interp1d
    f = interp1d(mz_array, intensity_array, kind='cubic', bounds_error=False, fill_value=0)
    return f(new_mz_points)


class SpectrumCache:
    """Simple cache for processed spectra"""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: Dict[str, Any] = {}
        self._access_times: Dict[str, float] = {}

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        if key in self._cache:
            self._access_times[key] = time.time()
            return self._cache[key]
        return None

    def put(self, key: str, value: Any):
        """Put item in cache"""
        if len(self._cache) >= self.max_size:
            # Remove least recently used item
            oldest_key = min(self._access_times.items(), key=lambda x: x[1])[0]
            del self._cache[oldest_key]
            del self._access_times[oldest_key]

        self._cache[key] = value
        self._access_times[key] = time.time()

    def clear(self):
        """Clear cache"""
        self._cache.clear()
        self._access_times.clear()
