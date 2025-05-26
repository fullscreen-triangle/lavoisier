"""
Bias Detection Module for Pipeline Comparison

Identifies systematic biases and errors in pipeline performance comparisons.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class BiasResult:
    """Container for bias detection results"""
    bias_type: str
    test_statistic: float
    p_value: float
    bias_magnitude: float
    confidence_interval: Tuple[float, float]
    interpretation: str
    significant: bool


class BiasDetector:
    """
    Comprehensive bias detection for pipeline comparisons
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize bias detector
        
        Args:
            alpha: Significance level for bias tests
        """
        self.alpha = alpha
        self.results = []
    
    def detect_systematic_bias(
        self,
        differences: np.ndarray,
        reference_value: float = 0.0
    ) -> BiasResult:
        """
        Detect systematic bias using one-sample t-test
        
        Args:
            differences: Differences between pipelines
            reference_value: Expected value under no bias
            
        Returns:
            BiasResult object
        """
        # One-sample t-test against reference value
        t_stat, p_value = stats.ttest_1samp(differences, reference_value)
        
        # Calculate bias magnitude
        bias_magnitude = np.mean(differences) - reference_value
        
        # Confidence interval for bias
        se = np.std(differences, ddof=1) / np.sqrt(len(differences))
        t_crit = stats.t.ppf(1 - self.alpha/2, len(differences) - 1)
        ci_lower = bias_magnitude - t_crit * se
        ci_upper = bias_magnitude + t_crit * se
        
        significant = p_value < self.alpha
        
        if significant:
            direction = "positive" if bias_magnitude > 0 else "negative"
            interpretation = f"Significant {direction} systematic bias detected (magnitude: {bias_magnitude:.3f})"
        else:
            interpretation = f"No significant systematic bias detected (magnitude: {bias_magnitude:.3f})"
        
        result = BiasResult(
            bias_type="Systematic Bias",
            test_statistic=t_stat,
            p_value=p_value,
            bias_magnitude=bias_magnitude,
            confidence_interval=(ci_lower, ci_upper),
            interpretation=interpretation,
            significant=significant
        )
        
        self.results.append(result)
        return result
    
    def detect_proportional_bias(
        self,
        method1_values: np.ndarray,
        method2_values: np.ndarray
    ) -> BiasResult:
        """
        Detect proportional bias using regression analysis
        
        Args:
            method1_values: Values from first method (e.g., numerical)
            method2_values: Values from second method (e.g., visual)
            
        Returns:
            BiasResult object
        """
        # Calculate differences and averages
        differences = method2_values - method1_values
        averages = (method1_values + method2_values) / 2
        
        # Linear regression: differences ~ averages
        slope, intercept, r_value, p_value, std_err = stats.linregress(averages, differences)
        
        # Test if slope is significantly different from 0
        t_stat = slope / std_err
        
        significant = p_value < self.alpha
        
        if significant:
            interpretation = f"Significant proportional bias detected (slope: {slope:.3f}, p={p_value:.3f})"
        else:
            interpretation = f"No significant proportional bias detected (slope: {slope:.3f}, p={p_value:.3f})"
        
        # Confidence interval for slope
        df = len(averages) - 2
        t_crit = stats.t.ppf(1 - self.alpha/2, df)
        ci_lower = slope - t_crit * std_err
        ci_upper = slope + t_crit * std_err
        
        result = BiasResult(
            bias_type="Proportional Bias",
            test_statistic=t_stat,
            p_value=p_value,
            bias_magnitude=slope,
            confidence_interval=(ci_lower, ci_upper),
            interpretation=interpretation,
            significant=significant
        )
        
        self.results.append(result)
        return result
    
    def detect_mass_dependent_bias(
        self,
        mz_values: np.ndarray,
        differences: np.ndarray
    ) -> BiasResult:
        """
        Detect mass-dependent bias in mass spectrometry data
        
        Args:
            mz_values: m/z values
            differences: Differences between methods
            
        Returns:
            BiasResult object
        """
        # Test for correlation between m/z and differences
        correlation, p_value = stats.pearsonr(mz_values, differences)
        
        # Convert correlation to t-statistic
        n = len(mz_values)
        t_stat = correlation * np.sqrt((n - 2) / (1 - correlation**2))
        
        significant = p_value < self.alpha
        
        if significant:
            direction = "positive" if correlation > 0 else "negative"
            interpretation = f"Significant {direction} mass-dependent bias (r={correlation:.3f}, p={p_value:.3f})"
        else:
            interpretation = f"No significant mass-dependent bias (r={correlation:.3f}, p={p_value:.3f})"
        
        # Confidence interval for correlation
        z_r = 0.5 * np.log((1 + correlation) / (1 - correlation))
        se_z = 1 / np.sqrt(n - 3)
        z_crit = stats.norm.ppf(1 - self.alpha/2)
        
        z_lower = z_r - z_crit * se_z
        z_upper = z_r + z_crit * se_z
        
        r_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
        r_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
        
        result = BiasResult(
            bias_type="Mass-Dependent Bias",
            test_statistic=t_stat,
            p_value=p_value,
            bias_magnitude=correlation,
            confidence_interval=(r_lower, r_upper),
            interpretation=interpretation,
            significant=significant
        )
        
        self.results.append(result)
        return result
    
    def detect_intensity_dependent_bias(
        self,
        intensity_values: np.ndarray,
        differences: np.ndarray
    ) -> BiasResult:
        """
        Detect intensity-dependent bias
        
        Args:
            intensity_values: Intensity values
            differences: Differences between methods
            
        Returns:
            BiasResult object
        """
        # Log-transform intensities to handle wide dynamic range
        log_intensities = np.log10(intensity_values + 1)  # +1 to handle zeros
        
        # Test for correlation
        correlation, p_value = stats.pearsonr(log_intensities, differences)
        
        # Convert to t-statistic
        n = len(intensity_values)
        t_stat = correlation * np.sqrt((n - 2) / (1 - correlation**2))
        
        significant = p_value < self.alpha
        
        if significant:
            direction = "positive" if correlation > 0 else "negative"
            interpretation = f"Significant {direction} intensity-dependent bias (r={correlation:.3f}, p={p_value:.3f})"
        else:
            interpretation = f"No significant intensity-dependent bias (r={correlation:.3f}, p={p_value:.3f})"
        
        # Confidence interval for correlation
        z_r = 0.5 * np.log((1 + correlation) / (1 - correlation))
        se_z = 1 / np.sqrt(n - 3)
        z_crit = stats.norm.ppf(1 - self.alpha/2)
        
        z_lower = z_r - z_crit * se_z
        z_upper = z_r + z_crit * se_z
        
        r_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
        r_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
        
        result = BiasResult(
            bias_type="Intensity-Dependent Bias",
            test_statistic=t_stat,
            p_value=p_value,
            bias_magnitude=correlation,
            confidence_interval=(r_lower, r_upper),
            interpretation=interpretation,
            significant=significant
        )
        
        self.results.append(result)
        return result
    
    def generate_bias_report(self) -> pd.DataFrame:
        """Generate comprehensive bias detection report"""
        if not self.results:
            return pd.DataFrame()
        
        data = []
        for result in self.results:
            data.append({
                'Bias Type': result.bias_type,
                'Test Statistic': result.test_statistic,
                'P-value': result.p_value,
                'Bias Magnitude': result.bias_magnitude,
                'CI Lower': result.confidence_interval[0],
                'CI Upper': result.confidence_interval[1],
                'Significant': result.significant,
                'Interpretation': result.interpretation
            })
        
        return pd.DataFrame(data)
    
    def plot_bias_analysis(self, save_path: Optional[str] = None) -> plt.Figure:
        """Create comprehensive bias analysis plots"""
        if not self.results:
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Bias Detection Analysis', fontsize=16, fontweight='bold')
        
        # Extract data
        bias_types = [result.bias_type for result in self.results]
        p_values = [result.p_value for result in self.results]
        magnitudes = [result.bias_magnitude for result in self.results]
        significant = [result.significant for result in self.results]
        
        # Plot 1: P-values
        colors = ['red' if sig else 'blue' for sig in significant]
        axes[0, 0].bar(bias_types, p_values, color=colors, alpha=0.7)
        axes[0, 0].axhline(y=self.alpha, color='red', linestyle='--', label=f'α = {self.alpha}')
        axes[0, 0].set_ylabel('P-value')
        axes[0, 0].set_title('Bias Test P-values')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].legend()
        
        # Plot 2: Bias magnitudes
        axes[0, 1].bar(bias_types, magnitudes, color=colors, alpha=0.7)
        axes[0, 1].set_ylabel('Bias Magnitude')
        axes[0, 1].set_title('Bias Magnitudes')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Significance summary
        sig_count = sum(significant)
        total_count = len(significant)
        
        labels = ['Significant Bias', 'No Significant Bias']
        sizes = [sig_count, total_count - sig_count]
        colors_pie = ['lightcoral', 'lightblue']
        
        axes[1, 0].pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
        axes[1, 0].set_title('Bias Detection Summary')
        
        # Plot 4: Confidence intervals
        for i, result in enumerate(self.results):
            ci_lower, ci_upper = result.confidence_interval
            axes[1, 1].errorbar(i, result.bias_magnitude,
                              yerr=[[result.bias_magnitude - ci_lower], [ci_upper - result.bias_magnitude]],
                              fmt='o', capsize=5, capthick=2,
                              color='red' if result.significant else 'blue')
        
        axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1, 1].set_xticks(range(len(bias_types)))
        axes[1, 1].set_xticklabels(bias_types, rotation=45)
        axes[1, 1].set_ylabel('Bias Magnitude')
        axes[1, 1].set_title('Bias Magnitudes with Confidence Intervals')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def clear_results(self) -> None:
        """Clear all stored results"""
        self.results = [] 