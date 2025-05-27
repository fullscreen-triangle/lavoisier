"""
Fidelity Analyzer Module

Comprehensive assessment of data fidelity including signal preservation,
information retention, and reconstruction accuracy between pipelines.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error


@dataclass
class FidelityResult:
    """Container for fidelity assessment results"""
    metric_name: str
    fidelity_score: float  # 0-1 score
    interpretation: str
    metadata: Dict[str, Any] = None


class FidelityAnalyzer:
    """
    Comprehensive fidelity assessment for pipeline data preservation
    """
    
    def __init__(self):
        """Initialize fidelity analyzer"""
        self.results = []
        
    def assess_signal_fidelity(
        self,
        original_data: np.ndarray,
        processed_data: np.ndarray,
        method: str = "correlation"
    ) -> FidelityResult:
        """
        Assess signal fidelity between original and processed data
        
        Args:
            original_data: Original signal data
            processed_data: Processed signal data
            method: Fidelity assessment method ('correlation', 'mse', 'mae')
            
        Returns:
            FidelityResult object
        """
        # Ensure same shape
        min_samples = min(len(original_data), len(processed_data))
        original_data = original_data[:min_samples]
        processed_data = processed_data[:min_samples]
        
        if method == "correlation":
            # Calculate correlation-based fidelity
            if original_data.ndim > 1:
                correlations = []
                for i in range(original_data.shape[1]):
                    corr, _ = pearsonr(original_data[:, i], processed_data[:, i])
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
                fidelity_score = np.mean(correlations) if correlations else 0.0
            else:
                corr, _ = pearsonr(original_data, processed_data)
                fidelity_score = abs(corr) if not np.isnan(corr) else 0.0
                
        elif method == "mse":
            # MSE-based fidelity (inverted and normalized)
            mse = mean_squared_error(original_data.flatten(), processed_data.flatten())
            max_possible_mse = np.var(original_data)
            fidelity_score = 1.0 - min(mse / max_possible_mse, 1.0)
            
        elif method == "mae":
            # MAE-based fidelity (inverted and normalized)
            mae = mean_absolute_error(original_data.flatten(), processed_data.flatten())
            max_possible_mae = np.mean(np.abs(original_data - np.mean(original_data)))
            fidelity_score = 1.0 - min(mae / max_possible_mae, 1.0)
        
        # Interpretation
        if fidelity_score > 0.9:
            interpretation = f"Excellent signal fidelity ({fidelity_score:.3f}) using {method}"
        elif fidelity_score > 0.8:
            interpretation = f"Good signal fidelity ({fidelity_score:.3f}) using {method}"
        elif fidelity_score > 0.6:
            interpretation = f"Moderate signal fidelity ({fidelity_score:.3f}) using {method}"
        else:
            interpretation = f"Poor signal fidelity ({fidelity_score:.3f}) using {method}"
        
        result = FidelityResult(
            metric_name=f"Signal Fidelity ({method.upper()})",
            fidelity_score=fidelity_score,
            interpretation=interpretation,
            metadata={
                'method': method,
                'original_shape': original_data.shape,
                'processed_shape': processed_data.shape
            }
        )
        
        self.results.append(result)
        return result
    
    def assess_spectral_fidelity(
        self,
        original_spectra: np.ndarray,
        processed_spectra: np.ndarray
    ) -> FidelityResult:
        """
        Assess spectral fidelity for mass spectrometry data
        
        Args:
            original_spectra: Original spectral data
            processed_spectra: Processed spectral data
            
        Returns:
            FidelityResult object
        """
        # Peak preservation analysis
        peak_fidelities = []
        
        for i in range(min(len(original_spectra), len(processed_spectra))):
            orig_spectrum = original_spectra[i]
            proc_spectrum = processed_spectra[i]
            
            # Find peaks in original spectrum
            from scipy.signal import find_peaks
            orig_peaks, _ = find_peaks(orig_spectrum, height=np.mean(orig_spectrum))
            
            if len(orig_peaks) > 0:
                # Calculate peak preservation
                orig_peak_intensities = orig_spectrum[orig_peaks]
                proc_peak_intensities = proc_spectrum[orig_peaks]
                
                # Correlation of peak intensities
                if len(orig_peak_intensities) > 1:
                    corr, _ = pearsonr(orig_peak_intensities, proc_peak_intensities)
                    peak_fidelity = abs(corr) if not np.isnan(corr) else 0.0
                else:
                    peak_fidelity = 1.0 if orig_peak_intensities[0] > 0 and proc_peak_intensities[0] > 0 else 0.0
                
                peak_fidelities.append(peak_fidelity)
        
        avg_peak_fidelity = np.mean(peak_fidelities) if peak_fidelities else 0.0
        
        # Overall spectral correlation
        overall_corr, _ = pearsonr(original_spectra.flatten(), processed_spectra.flatten())
        overall_fidelity = abs(overall_corr) if not np.isnan(overall_corr) else 0.0
        
        # Combined fidelity score
        fidelity_score = 0.6 * overall_fidelity + 0.4 * avg_peak_fidelity
        
        interpretation = f"Spectral fidelity: {fidelity_score:.3f} (Overall: {overall_fidelity:.3f}, Peaks: {avg_peak_fidelity:.3f})"
        
        result = FidelityResult(
            metric_name="Spectral Fidelity",
            fidelity_score=fidelity_score,
            interpretation=interpretation,
            metadata={
                'overall_correlation': overall_fidelity,
                'peak_fidelity': avg_peak_fidelity,
                'num_spectra': min(len(original_spectra), len(processed_spectra))
            }
        )
        
        self.results.append(result)
        return result
    
    def assess_information_preservation(
        self,
        original_data: np.ndarray,
        processed_data: np.ndarray
    ) -> FidelityResult:
        """
        Assess information preservation using entropy measures
        
        Args:
            original_data: Original data
            processed_data: Processed data
            
        Returns:
            FidelityResult object
        """
        # Calculate entropy for both datasets
        def calculate_entropy(data):
            # Discretize data for entropy calculation
            hist, _ = np.histogram(data.flatten(), bins=50, density=True)
            hist = hist[hist > 0]  # Remove zero bins
            return -np.sum(hist * np.log2(hist))
        
        original_entropy = calculate_entropy(original_data)
        processed_entropy = calculate_entropy(processed_data)
        
        # Information preservation score
        if original_entropy > 0:
            preservation_score = min(processed_entropy / original_entropy, 1.0)
        else:
            preservation_score = 1.0 if processed_entropy == 0 else 0.0
        
        # Mutual information approximation
        # Flatten and discretize both datasets
        orig_flat = original_data.flatten()
        proc_flat = processed_data.flatten()
        
        # Create joint histogram
        hist_2d, _, _ = np.histogram2d(orig_flat, proc_flat, bins=20)
        hist_2d = hist_2d / np.sum(hist_2d)  # Normalize
        
        # Calculate mutual information
        mutual_info = 0.0
        for i in range(hist_2d.shape[0]):
            for j in range(hist_2d.shape[1]):
                if hist_2d[i, j] > 0:
                    marginal_i = np.sum(hist_2d[i, :])
                    marginal_j = np.sum(hist_2d[:, j])
                    if marginal_i > 0 and marginal_j > 0:
                        mutual_info += hist_2d[i, j] * np.log2(hist_2d[i, j] / (marginal_i * marginal_j))
        
        # Normalize mutual information
        max_entropy = max(original_entropy, processed_entropy)
        normalized_mi = mutual_info / max_entropy if max_entropy > 0 else 0.0
        
        # Combined information fidelity
        fidelity_score = 0.5 * preservation_score + 0.5 * normalized_mi
        
        interpretation = f"Information preservation: {fidelity_score:.3f} (Entropy ratio: {preservation_score:.3f}, MI: {normalized_mi:.3f})"
        
        result = FidelityResult(
            metric_name="Information Preservation",
            fidelity_score=fidelity_score,
            interpretation=interpretation,
            metadata={
                'original_entropy': original_entropy,
                'processed_entropy': processed_entropy,
                'mutual_information': mutual_info,
                'preservation_ratio': preservation_score
            }
        )
        
        self.results.append(result)
        return result
    
    def assess_reconstruction_fidelity(
        self,
        original_data: np.ndarray,
        reconstructed_data: np.ndarray
    ) -> FidelityResult:
        """
        Assess reconstruction fidelity for compressed/reconstructed data
        
        Args:
            original_data: Original data
            reconstructed_data: Reconstructed data
            
        Returns:
            FidelityResult object
        """
        # Multiple reconstruction quality metrics
        
        # 1. Structural Similarity Index (simplified)
        def ssim_1d(x, y):
            mu_x = np.mean(x)
            mu_y = np.mean(y)
            sigma_x = np.var(x)
            sigma_y = np.var(y)
            sigma_xy = np.mean((x - mu_x) * (y - mu_y))
            
            c1 = 0.01 ** 2
            c2 = 0.03 ** 2
            
            numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
            denominator = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
            
            return numerator / denominator if denominator > 0 else 0.0
        
        # Calculate SSIM for each feature/channel
        if original_data.ndim > 1:
            ssim_scores = []
            for i in range(original_data.shape[1]):
                ssim = ssim_1d(original_data[:, i], reconstructed_data[:, i])
                ssim_scores.append(ssim)
            avg_ssim = np.mean(ssim_scores)
        else:
            avg_ssim = ssim_1d(original_data, reconstructed_data)
        
        # 2. Peak Signal-to-Noise Ratio
        mse = mean_squared_error(original_data.flatten(), reconstructed_data.flatten())
        max_val = np.max(original_data)
        psnr = 20 * np.log10(max_val / np.sqrt(mse)) if mse > 0 else float('inf')
        psnr_normalized = min(psnr / 40.0, 1.0)  # Normalize assuming 40dB is excellent
        
        # 3. Relative error
        relative_error = np.mean(np.abs(original_data - reconstructed_data)) / np.mean(np.abs(original_data))
        relative_fidelity = max(0.0, 1.0 - relative_error)
        
        # Combined reconstruction fidelity
        fidelity_score = 0.4 * avg_ssim + 0.3 * psnr_normalized + 0.3 * relative_fidelity
        
        interpretation = f"Reconstruction fidelity: {fidelity_score:.3f} (SSIM: {avg_ssim:.3f}, PSNR: {psnr:.1f}dB, RelErr: {relative_error:.3f})"
        
        result = FidelityResult(
            metric_name="Reconstruction Fidelity",
            fidelity_score=fidelity_score,
            interpretation=interpretation,
            metadata={
                'ssim': avg_ssim,
                'psnr': psnr,
                'relative_error': relative_error,
                'mse': mse
            }
        )
        
        self.results.append(result)
        return result
    
    def compare_pipeline_fidelity(
        self,
        reference_data: np.ndarray,
        numerical_output: np.ndarray,
        visual_output: np.ndarray
    ) -> Dict[str, FidelityResult]:
        """
        Compare fidelity between numerical and visual pipelines
        
        Args:
            reference_data: Reference/ground truth data
            numerical_output: Numerical pipeline output
            visual_output: Visual pipeline output
            
        Returns:
            Dictionary with fidelity results for each pipeline
        """
        # Assess fidelity for numerical pipeline
        num_fidelity = self.assess_signal_fidelity(reference_data, numerical_output, "correlation")
        
        # Assess fidelity for visual pipeline
        vis_fidelity = self.assess_signal_fidelity(reference_data, visual_output, "correlation")
        
        return {
            'numerical': num_fidelity,
            'visual': vis_fidelity
        }
    
    def generate_fidelity_report(self) -> pd.DataFrame:
        """Generate comprehensive fidelity assessment report"""
        if not self.results:
            return pd.DataFrame()
        
        data = []
        for result in self.results:
            data.append({
                'Metric': result.metric_name,
                'Fidelity Score': result.fidelity_score,
                'Interpretation': result.interpretation
            })
        
        return pd.DataFrame(data)
    
    def get_overall_fidelity_score(self) -> float:
        """Calculate overall fidelity score across all metrics"""
        if not self.results:
            return 0.0
        
        return np.mean([r.fidelity_score for r in self.results])
    
    def clear_results(self) -> None:
        """Clear all stored results"""
        self.results = [] 