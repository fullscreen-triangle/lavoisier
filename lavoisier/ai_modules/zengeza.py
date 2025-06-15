"""
Zengeza: Intelligent Noise Reduction Module

This module implements sophisticated noise detection and removal algorithms
for mass spectrometry data using statistical analysis, spectral entropy,
and machine learning techniques.
"""

import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.optimize import minimize_scalar
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class NoiseProfile:
    """Represents a comprehensive noise profile for MS data"""
    baseline_noise: float
    peak_noise_ratio: float
    systematic_noise: np.ndarray
    entropy_threshold: float
    isolation_threshold: float
    confidence_score: float
    noise_pattern: str  # 'gaussian', 'poisson', 'uniform', 'mixed'

class ZengezaNoiseReducer:
    """
    Advanced noise reduction system for mass spectrometry data.
    
    Combines multiple techniques:
    - Statistical noise modeling
    - Spectral entropy analysis
    - Isolation forest for outlier detection
    - Wavelet denoising
    - Adaptive filtering
    """
    
    def __init__(self, 
                 entropy_window: int = 50,
                 isolation_contamination: float = 0.1,
                 wavelet: str = 'db8',
                 adaptive_threshold: float = 0.95):
        self.entropy_window = entropy_window
        self.isolation_contamination = isolation_contamination
        self.wavelet = wavelet
        self.adaptive_threshold = adaptive_threshold
        self.noise_profiles: Dict[str, NoiseProfile] = {}
        self.scaler = StandardScaler()
        
    def analyze_noise_characteristics(self, mz_array: np.ndarray, 
                                    intensity_array: np.ndarray,
                                    spectrum_id: str = "default") -> NoiseProfile:
        """
        Comprehensive noise analysis of MS spectrum data.
        """
        logger.info(f"Analyzing noise characteristics for spectrum {spectrum_id}")
        
        # 1. Baseline noise estimation using robust statistics
        baseline_noise = self._estimate_baseline_noise(intensity_array)
        
        # 2. Peak-to-noise ratio analysis
        peak_noise_ratio = self._calculate_peak_noise_ratio(intensity_array, baseline_noise)
        
        # 3. Systematic noise pattern detection
        systematic_noise = self._detect_systematic_noise(mz_array, intensity_array)
        
        # 4. Spectral entropy analysis
        entropy_threshold = self._calculate_spectral_entropy_threshold(intensity_array)
        
        # 5. Isolation forest analysis
        isolation_threshold = self._isolation_forest_analysis(mz_array, intensity_array)
        
        # 6. Noise pattern classification
        noise_pattern = self._classify_noise_pattern(intensity_array)
        
        # 7. Confidence scoring
        confidence_score = self._calculate_confidence_score(
            baseline_noise, peak_noise_ratio, entropy_threshold, isolation_threshold
        )
        
        profile = NoiseProfile(
            baseline_noise=baseline_noise,
            peak_noise_ratio=peak_noise_ratio,
            systematic_noise=systematic_noise,
            entropy_threshold=entropy_threshold,
            isolation_threshold=isolation_threshold,
            confidence_score=confidence_score,
            noise_pattern=noise_pattern
        )
        
        self.noise_profiles[spectrum_id] = profile
        return profile
    
    def _estimate_baseline_noise(self, intensity_array: np.ndarray) -> float:
        """
        Estimate baseline noise using robust statistical methods.
        """
        # Use median absolute deviation for robust noise estimation
        median_intensity = np.median(intensity_array)
        mad = np.median(np.abs(intensity_array - median_intensity))
        
        # Convert MAD to standard deviation equivalent
        baseline_noise = 1.4826 * mad
        
        # Validate using percentile-based approach
        percentile_noise = np.percentile(intensity_array, 25)
        
        # Take weighted average based on data distribution
        skewness = stats.skew(intensity_array)
        if abs(skewness) < 1:  # Approximately symmetric
            return baseline_noise
        else:  # Highly skewed data
            return 0.7 * baseline_noise + 0.3 * percentile_noise
    
    def _calculate_peak_noise_ratio(self, intensity_array: np.ndarray, 
                                  baseline_noise: float) -> float:
        """
        Calculate signal-to-noise ratio for peak detection.
        """
        # Find peaks using scipy
        peaks, properties = signal.find_peaks(intensity_array, 
                                            height=baseline_noise * 3,
                                            prominence=baseline_noise * 2)
        
        if len(peaks) == 0:
            return 1.0
        
        # Calculate average peak intensity
        peak_intensities = intensity_array[peaks]
        avg_peak_intensity = np.mean(peak_intensities)
        
        return avg_peak_intensity / baseline_noise if baseline_noise > 0 else 1.0
    
    def _detect_systematic_noise(self, mz_array: np.ndarray, 
                               intensity_array: np.ndarray) -> np.ndarray:
        """
        Detect systematic noise patterns using frequency analysis.
        """
        # Apply FFT to detect periodic noise
        fft_intensity = np.fft.fft(intensity_array)
        freqs = np.fft.fftfreq(len(intensity_array))
        
        # Find dominant frequencies (potential systematic noise)
        power_spectrum = np.abs(fft_intensity) ** 2
        dominant_freq_indices = np.argsort(power_spectrum)[-10:]  # Top 10 frequencies
        
        # Reconstruct systematic noise component
        systematic_fft = np.zeros_like(fft_intensity)
        systematic_fft[dominant_freq_indices] = fft_intensity[dominant_freq_indices]
        
        systematic_noise = np.real(np.fft.ifft(systematic_fft))
        
        return systematic_noise
    
    def _calculate_spectral_entropy_threshold(self, intensity_array: np.ndarray) -> float:
        """
        Calculate spectral entropy threshold for noise identification.
        """
        # Normalize intensities to probabilities
        normalized_intensities = intensity_array / np.sum(intensity_array)
        
        # Calculate entropy in sliding windows
        entropies = []
        for i in range(len(intensity_array) - self.entropy_window + 1):
            window = normalized_intensities[i:i + self.entropy_window]
            # Add small constant to avoid log(0)
            window = window + 1e-10
            entropy = -np.sum(window * np.log2(window))
            entropies.append(entropy)
        
        # Threshold based on entropy distribution
        entropy_threshold = np.percentile(entropies, 75)  # Upper quartile
        
        return entropy_threshold
    
    def _isolation_forest_analysis(self, mz_array: np.ndarray, 
                                 intensity_array: np.ndarray) -> float:
        """
        Use Isolation Forest to detect anomalous (noisy) data points.
        """
        # Prepare features for isolation forest
        features = np.column_stack([mz_array, intensity_array])
        
        # Standardize features
        features_scaled = self.scaler.fit_transform(features)
        
        # Apply Isolation Forest
        iso_forest = IsolationForest(
            contamination=self.isolation_contamination,
            random_state=42,
            n_estimators=100
        )
        
        outlier_labels = iso_forest.fit_predict(features_scaled)
        outlier_scores = iso_forest.score_samples(features_scaled)
        
        # Calculate threshold based on outlier scores
        isolation_threshold = np.percentile(outlier_scores, 10)  # Bottom 10%
        
        return isolation_threshold
    
    def _classify_noise_pattern(self, intensity_array: np.ndarray) -> str:
        """
        Classify the type of noise pattern in the data.
        """
        # Test for different noise distributions
        
        # 1. Normality test (Gaussian noise)
        _, p_normal = stats.normaltest(intensity_array)
        
        # 2. Test for Poisson-like characteristics
        mean_intensity = np.mean(intensity_array)
        var_intensity = np.var(intensity_array)
        poisson_ratio = var_intensity / mean_intensity if mean_intensity > 0 else float('inf')
        
        # 3. Test for uniformity
        _, p_uniform = stats.kstest(intensity_array, 'uniform')
        
        # Classification logic
        if p_normal > 0.05:
            return 'gaussian'
        elif abs(poisson_ratio - 1.0) < 0.5:
            return 'poisson'
        elif p_uniform > 0.05:
            return 'uniform'
        else:
            return 'mixed'
    
    def _calculate_confidence_score(self, baseline_noise: float, 
                                  peak_noise_ratio: float,
                                  entropy_threshold: float, 
                                  isolation_threshold: float) -> float:
        """
        Calculate confidence score for noise analysis quality.
        """
        # Normalize individual scores
        baseline_score = min(1.0, baseline_noise / np.mean([baseline_noise, peak_noise_ratio]))
        ratio_score = min(1.0, peak_noise_ratio / 10.0)  # Assume good SNR is around 10
        entropy_score = min(1.0, entropy_threshold / 10.0)  # Normalize entropy
        isolation_score = min(1.0, abs(isolation_threshold) / 0.5)  # Normalize isolation score
        
        # Weighted combination
        confidence = (0.3 * baseline_score + 0.3 * ratio_score + 
                     0.2 * entropy_score + 0.2 * isolation_score)
        
        return confidence
    
    def remove_noise(self, mz_array: np.ndarray, intensity_array: np.ndarray,
                    spectrum_id: str = "default", 
                    noise_profile: Optional[NoiseProfile] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove noise from MS spectrum using the analyzed noise profile.
        """
        if noise_profile is None:
            noise_profile = self.analyze_noise_characteristics(
                mz_array, intensity_array, spectrum_id
            )
        
        logger.info(f"Removing noise from spectrum {spectrum_id} using {noise_profile.noise_pattern} noise model")
        
        cleaned_intensity = intensity_array.copy()
        
        # 1. Baseline noise removal
        cleaned_intensity = self._remove_baseline_noise(
            cleaned_intensity, noise_profile.baseline_noise
        )
        
        # 2. Systematic noise removal
        cleaned_intensity = self._remove_systematic_noise(
            cleaned_intensity, noise_profile.systematic_noise
        )
        
        # 3. Wavelet denoising
        cleaned_intensity = self._wavelet_denoise(cleaned_intensity)
        
        # 4. Adaptive filtering based on spectral entropy
        cleaned_mz, cleaned_intensity = self._entropy_based_filtering(
            mz_array, cleaned_intensity, noise_profile.entropy_threshold
        )
        
        # 5. Isolation forest-based outlier removal
        cleaned_mz, cleaned_intensity = self._isolation_based_filtering(
            cleaned_mz, cleaned_intensity, noise_profile.isolation_threshold
        )
        
        logger.info(f"Noise removal complete. Reduced {len(intensity_array)} to {len(cleaned_intensity)} points")
        
        return cleaned_mz, cleaned_intensity
    
    def _remove_baseline_noise(self, intensity_array: np.ndarray, 
                             baseline_noise: float) -> np.ndarray:
        """Remove baseline noise component."""
        # Subtract baseline noise, ensuring no negative values
        cleaned = intensity_array - baseline_noise
        cleaned[cleaned < 0] = 0
        return cleaned
    
    def _remove_systematic_noise(self, intensity_array: np.ndarray, 
                               systematic_noise: np.ndarray) -> np.ndarray:
        """Remove systematic noise component."""
        if len(systematic_noise) == len(intensity_array):
            cleaned = intensity_array - systematic_noise
            cleaned[cleaned < 0] = 0
            return cleaned
        return intensity_array
    
    def _wavelet_denoise(self, intensity_array: np.ndarray) -> np.ndarray:
        """Apply wavelet denoising (simplified implementation)."""
        # For now, use a simple moving average as wavelet approximation
        # In production, you would use pywt for proper wavelet denoising
        window_size = min(5, len(intensity_array) // 10)
        if window_size >= 3:
            cleaned = signal.savgol_filter(intensity_array, window_size, 2)
            return np.maximum(cleaned, 0)  # Ensure non-negative
        return intensity_array
    
    def _entropy_based_filtering(self, mz_array: np.ndarray, 
                               intensity_array: np.ndarray,
                               entropy_threshold: float) -> Tuple[np.ndarray, np.ndarray]:
        """Filter data points based on local spectral entropy."""
        if len(intensity_array) < self.entropy_window:
            return mz_array, intensity_array
        
        keep_indices = []
        
        for i in range(len(intensity_array)):
            start_idx = max(0, i - self.entropy_window // 2)
            end_idx = min(len(intensity_array), i + self.entropy_window // 2)
            
            window_intensities = intensity_array[start_idx:end_idx]
            normalized = window_intensities / np.sum(window_intensities)
            normalized = normalized + 1e-10  # Avoid log(0)
            
            local_entropy = -np.sum(normalized * np.log2(normalized))
            
            if local_entropy <= entropy_threshold:
                keep_indices.append(i)
        
        if len(keep_indices) > 0:
            return mz_array[keep_indices], intensity_array[keep_indices]
        else:
            # If all points filtered out, return original (conservative approach)
            return mz_array, intensity_array
    
    def _isolation_based_filtering(self, mz_array: np.ndarray, 
                                 intensity_array: np.ndarray,
                                 isolation_threshold: float) -> Tuple[np.ndarray, np.ndarray]:
        """Filter outliers using isolation forest threshold."""
        if len(mz_array) < 10:  # Need minimum points for isolation forest
            return mz_array, intensity_array
        
        features = np.column_stack([mz_array, intensity_array])
        features_scaled = StandardScaler().fit_transform(features)
        
        iso_forest = IsolationForest(
            contamination=self.isolation_contamination,
            random_state=42,
            n_estimators=100
        )
        
        outlier_scores = iso_forest.fit(features_scaled).score_samples(features_scaled)
        
        # Keep points above the isolation threshold
        keep_indices = outlier_scores >= isolation_threshold
        
        if np.any(keep_indices):
            return mz_array[keep_indices], intensity_array[keep_indices]
        else:
            # Conservative: return original if all filtered out
            return mz_array, intensity_array
    
    def get_noise_report(self, spectrum_id: str = "default") -> Dict[str, Any]:
        """
        Generate a comprehensive noise analysis report.
        """
        if spectrum_id not in self.noise_profiles:
            raise ValueError(f"No noise profile found for spectrum {spectrum_id}")
        
        profile = self.noise_profiles[spectrum_id]
        
        report = {
            "spectrum_id": spectrum_id,
            "baseline_noise_level": profile.baseline_noise,
            "signal_to_noise_ratio": profile.peak_noise_ratio,
            "noise_pattern_type": profile.noise_pattern,
            "entropy_threshold": profile.entropy_threshold,
            "isolation_threshold": profile.isolation_threshold,
            "analysis_confidence": profile.confidence_score,
            "systematic_noise_detected": len(profile.systematic_noise) > 0,
            "recommendations": self._generate_recommendations(profile)
        }
        
        return report
    
    def _generate_recommendations(self, profile: NoiseProfile) -> List[str]:
        """Generate recommendations based on noise analysis."""
        recommendations = []
        
        if profile.confidence_score < 0.5:
            recommendations.append("Low confidence in noise analysis - consider manual review")
        
        if profile.peak_noise_ratio < 3:
            recommendations.append("Poor signal-to-noise ratio - consider increasing acquisition time")
        
        if profile.noise_pattern == 'mixed':
            recommendations.append("Complex noise pattern detected - may need specialized preprocessing")
        
        if len(profile.systematic_noise) > 0:
            recommendations.append("Systematic noise detected - check instrument calibration")
        
        if profile.baseline_noise > np.mean([profile.baseline_noise, profile.peak_noise_ratio]):
            recommendations.append("High baseline noise - consider baseline correction")
        
        return recommendations 