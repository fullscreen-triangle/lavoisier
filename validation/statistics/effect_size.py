"""
Effect Size Calculator for Pipeline Comparison

Provides comprehensive effect size calculations to quantify the practical
significance of differences between numerical and visual pipelines.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class EffectSizeResult:
    """Container for effect size calculation results"""
    metric_name: str
    effect_size_type: str
    value: float
    confidence_interval: Tuple[float, float]
    interpretation: str
    magnitude: str  # small, medium, large


class EffectSizeCalculator:
    """
    Comprehensive effect size calculator for comparing pipeline performance
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize the effect size calculator
        
        Args:
            confidence_level: Confidence level for confidence intervals
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.results = []
    
    def cohens_d(
        self, 
        group1: np.ndarray, 
        group2: np.ndarray,
        pooled: bool = True
    ) -> EffectSizeResult:
        """
        Calculate Cohen's d for mean differences
        
        Args:
            group1: First group (e.g., numerical pipeline results)
            group2: Second group (e.g., visual pipeline results)
            pooled: Whether to use pooled standard deviation
            
        Returns:
            EffectSizeResult object
        """
        mean1, mean2 = np.mean(group1), np.mean(group2)
        
        if pooled:
            # Pooled standard deviation
            n1, n2 = len(group1), len(group2)
            pooled_std = np.sqrt(((n1 - 1) * np.var(group1, ddof=1) + 
                                 (n2 - 1) * np.var(group2, ddof=1)) / (n1 + n2 - 2))
            d = (mean1 - mean2) / pooled_std
        else:
            # Use standard deviation of control group (group1)
            d = (mean1 - mean2) / np.std(group1, ddof=1)
        
        # Confidence interval for Cohen's d
        n1, n2 = len(group1), len(group2)
        se_d = np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2)))
        t_crit = stats.t.ppf(1 - self.alpha/2, n1 + n2 - 2)
        
        ci_lower = d - t_crit * se_d
        ci_upper = d + t_crit * se_d
        
        # Interpret magnitude (Cohen's conventions)
        abs_d = abs(d)
        if abs_d < 0.2:
            magnitude = "negligible"
        elif abs_d < 0.5:
            magnitude = "small"
        elif abs_d < 0.8:
            magnitude = "medium"
        else:
            magnitude = "large"
        
        interpretation = f"Cohen's d = {d:.3f} ({magnitude} effect)"
        
        result = EffectSizeResult(
            metric_name="Mean Difference",
            effect_size_type="Cohen's d",
            value=d,
            confidence_interval=(ci_lower, ci_upper),
            interpretation=interpretation,
            magnitude=magnitude
        )
        
        self.results.append(result)
        return result
    
    def glass_delta(
        self, 
        experimental: np.ndarray, 
        control: np.ndarray
    ) -> EffectSizeResult:
        """
        Calculate Glass's Δ (delta) using control group standard deviation
        
        Args:
            experimental: Experimental group (e.g., visual pipeline)
            control: Control group (e.g., numerical pipeline)
            
        Returns:
            EffectSizeResult object
        """
        mean_exp = np.mean(experimental)
        mean_ctrl = np.mean(control)
        std_ctrl = np.std(control, ddof=1)
        
        delta = (mean_exp - mean_ctrl) / std_ctrl
        
        # Confidence interval (approximate)
        n_exp, n_ctrl = len(experimental), len(control)
        se_delta = np.sqrt((n_exp + n_ctrl) / (n_exp * n_ctrl) + delta**2 / (2 * n_ctrl))
        t_crit = stats.t.ppf(1 - self.alpha/2, n_ctrl - 1)
        
        ci_lower = delta - t_crit * se_delta
        ci_upper = delta + t_crit * se_delta
        
        # Interpret magnitude
        abs_delta = abs(delta)
        if abs_delta < 0.2:
            magnitude = "negligible"
        elif abs_delta < 0.5:
            magnitude = "small"
        elif abs_delta < 0.8:
            magnitude = "medium"
        else:
            magnitude = "large"
        
        interpretation = f"Glass's Δ = {delta:.3f} ({magnitude} effect)"
        
        result = EffectSizeResult(
            metric_name="Experimental vs Control",
            effect_size_type="Glass's Δ",
            value=delta,
            confidence_interval=(ci_lower, ci_upper),
            interpretation=interpretation,
            magnitude=magnitude
        )
        
        self.results.append(result)
        return result
    
    def hedges_g(
        self, 
        group1: np.ndarray, 
        group2: np.ndarray
    ) -> EffectSizeResult:
        """
        Calculate Hedges' g (bias-corrected Cohen's d)
        
        Args:
            group1: First group
            group2: Second group
            
        Returns:
            EffectSizeResult object
        """
        # First calculate Cohen's d
        cohens_d_result = self.cohens_d(group1, group2, pooled=True)
        d = cohens_d_result.value
        
        # Bias correction factor
        n1, n2 = len(group1), len(group2)
        df = n1 + n2 - 2
        correction_factor = 1 - (3 / (4 * df - 1))
        
        g = d * correction_factor
        
        # Confidence interval
        se_g = np.sqrt((n1 + n2) / (n1 * n2) + g**2 / (2 * df)) * correction_factor
        t_crit = stats.t.ppf(1 - self.alpha/2, df)
        
        ci_lower = g - t_crit * se_g
        ci_upper = g + t_crit * se_g
        
        # Interpret magnitude
        abs_g = abs(g)
        if abs_g < 0.2:
            magnitude = "negligible"
        elif abs_g < 0.5:
            magnitude = "small"
        elif abs_g < 0.8:
            magnitude = "medium"
        else:
            magnitude = "large"
        
        interpretation = f"Hedges' g = {g:.3f} ({magnitude} effect, bias-corrected)"
        
        result = EffectSizeResult(
            metric_name="Mean Difference (Bias-Corrected)",
            effect_size_type="Hedges' g",
            value=g,
            confidence_interval=(ci_lower, ci_upper),
            interpretation=interpretation,
            magnitude=magnitude
        )
        
        self.results.append(result)
        return result
    
    def cliffs_delta(
        self, 
        group1: np.ndarray, 
        group2: np.ndarray
    ) -> EffectSizeResult:
        """
        Calculate Cliff's delta (non-parametric effect size)
        
        Args:
            group1: First group
            group2: Second group
            
        Returns:
            EffectSizeResult object
        """
        n1, n2 = len(group1), len(group2)
        
        # Count dominance
        dominance_count = 0
        for x1 in group1:
            for x2 in group2:
                if x1 > x2:
                    dominance_count += 1
                elif x1 < x2:
                    dominance_count -= 1
        
        delta = dominance_count / (n1 * n2)
        
        # Confidence interval (approximate using normal approximation)
        se_delta = np.sqrt((n1 + n2 + 1) / (3 * n1 * n2))
        z_crit = stats.norm.ppf(1 - self.alpha/2)
        
        ci_lower = delta - z_crit * se_delta
        ci_upper = delta + z_crit * se_delta
        
        # Interpret magnitude (Cliff's conventions)
        abs_delta = abs(delta)
        if abs_delta < 0.147:
            magnitude = "negligible"
        elif abs_delta < 0.33:
            magnitude = "small"
        elif abs_delta < 0.474:
            magnitude = "medium"
        else:
            magnitude = "large"
        
        interpretation = f"Cliff's δ = {delta:.3f} ({magnitude} effect, non-parametric)"
        
        result = EffectSizeResult(
            metric_name="Ordinal Dominance",
            effect_size_type="Cliff's δ",
            value=delta,
            confidence_interval=(ci_lower, ci_upper),
            interpretation=interpretation,
            magnitude=magnitude
        )
        
        self.results.append(result)
        return result
    
    def eta_squared(
        self, 
        groups: List[np.ndarray]
    ) -> EffectSizeResult:
        """
        Calculate eta-squared for ANOVA-type comparisons
        
        Args:
            groups: List of groups to compare
            
        Returns:
            EffectSizeResult object
        """
        # Combine all data
        all_data = np.concatenate(groups)
        grand_mean = np.mean(all_data)
        
        # Calculate sum of squares
        ss_total = np.sum((all_data - grand_mean)**2)
        
        ss_between = 0
        for group in groups:
            group_mean = np.mean(group)
            ss_between += len(group) * (group_mean - grand_mean)**2
        
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        
        # Confidence interval (approximate)
        # Using non-central F distribution approximation
        k = len(groups)  # number of groups
        n = len(all_data)  # total sample size
        
        # Approximate confidence interval
        se_eta = np.sqrt(2 * eta_squared * (1 - eta_squared) / (n - k))
        z_crit = stats.norm.ppf(1 - self.alpha/2)
        
        ci_lower = max(0, eta_squared - z_crit * se_eta)
        ci_upper = min(1, eta_squared + z_crit * se_eta)
        
        # Interpret magnitude (Cohen's conventions for eta-squared)
        if eta_squared < 0.01:
            magnitude = "negligible"
        elif eta_squared < 0.06:
            magnitude = "small"
        elif eta_squared < 0.14:
            magnitude = "medium"
        else:
            magnitude = "large"
        
        interpretation = f"η² = {eta_squared:.3f} ({magnitude} effect, {eta_squared*100:.1f}% variance explained)"
        
        result = EffectSizeResult(
            metric_name="Variance Explained",
            effect_size_type="η² (Eta-squared)",
            value=eta_squared,
            confidence_interval=(ci_lower, ci_upper),
            interpretation=interpretation,
            magnitude=magnitude
        )
        
        self.results.append(result)
        return result
    
    def omega_squared(
        self, 
        groups: List[np.ndarray]
    ) -> EffectSizeResult:
        """
        Calculate omega-squared (less biased than eta-squared)
        
        Args:
            groups: List of groups to compare
            
        Returns:
            EffectSizeResult object
        """
        # First calculate eta-squared components
        all_data = np.concatenate(groups)
        grand_mean = np.mean(all_data)
        
        ss_total = np.sum((all_data - grand_mean)**2)
        
        ss_between = 0
        ss_within = 0
        k = len(groups)
        
        for group in groups:
            group_mean = np.mean(group)
            ss_between += len(group) * (group_mean - grand_mean)**2
            ss_within += np.sum((group - group_mean)**2)
        
        n = len(all_data)
        ms_within = ss_within / (n - k)
        
        # Omega-squared calculation
        omega_squared = (ss_between - (k - 1) * ms_within) / (ss_total + ms_within)
        omega_squared = max(0, omega_squared)  # Cannot be negative
        
        # Confidence interval (approximate)
        se_omega = np.sqrt(2 * omega_squared * (1 - omega_squared) / (n - k))
        z_crit = stats.norm.ppf(1 - self.alpha/2)
        
        ci_lower = max(0, omega_squared - z_crit * se_omega)
        ci_upper = min(1, omega_squared + z_crit * se_omega)
        
        # Interpret magnitude
        if omega_squared < 0.01:
            magnitude = "negligible"
        elif omega_squared < 0.06:
            magnitude = "small"
        elif omega_squared < 0.14:
            magnitude = "medium"
        else:
            magnitude = "large"
        
        interpretation = f"ω² = {omega_squared:.3f} ({magnitude} effect, {omega_squared*100:.1f}% variance explained, bias-corrected)"
        
        result = EffectSizeResult(
            metric_name="Variance Explained (Bias-Corrected)",
            effect_size_type="ω² (Omega-squared)",
            value=omega_squared,
            confidence_interval=(ci_lower, ci_upper),
            interpretation=interpretation,
            magnitude=magnitude
        )
        
        self.results.append(result)
        return result
    
    def r_squared(
        self, 
        observed: np.ndarray, 
        predicted: np.ndarray
    ) -> EffectSizeResult:
        """
        Calculate R-squared for regression-type relationships
        
        Args:
            observed: Observed values
            predicted: Predicted values
            
        Returns:
            EffectSizeResult object
        """
        # Calculate R-squared
        ss_res = np.sum((observed - predicted)**2)
        ss_tot = np.sum((observed - np.mean(observed))**2)
        
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        r_squared = max(0, r_squared)  # Cannot be negative
        
        # Confidence interval using Fisher transformation
        r = np.sqrt(r_squared)
        n = len(observed)
        
        # Fisher z-transformation
        z_r = 0.5 * np.log((1 + r) / (1 - r))
        se_z = 1 / np.sqrt(n - 3)
        z_crit = stats.norm.ppf(1 - self.alpha/2)
        
        z_lower = z_r - z_crit * se_z
        z_upper = z_r + z_crit * se_z
        
        # Transform back to r, then to r-squared
        r_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
        r_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
        
        ci_lower = max(0, r_lower**2)
        ci_upper = min(1, r_upper**2)
        
        # Interpret magnitude (Cohen's conventions for R-squared)
        if r_squared < 0.02:
            magnitude = "negligible"
        elif r_squared < 0.13:
            magnitude = "small"
        elif r_squared < 0.26:
            magnitude = "medium"
        else:
            magnitude = "large"
        
        interpretation = f"R² = {r_squared:.3f} ({magnitude} effect, {r_squared*100:.1f}% variance explained)"
        
        result = EffectSizeResult(
            metric_name="Prediction Accuracy",
            effect_size_type="R² (R-squared)",
            value=r_squared,
            confidence_interval=(ci_lower, ci_upper),
            interpretation=interpretation,
            magnitude=magnitude
        )
        
        self.results.append(result)
        return result
    
    def generate_summary_report(self) -> pd.DataFrame:
        """Generate a summary report of all effect size calculations"""
        if not self.results:
            return pd.DataFrame()
        
        data = []
        for result in self.results:
            data.append({
                'Metric': result.metric_name,
                'Effect Size Type': result.effect_size_type,
                'Value': result.value,
                'CI Lower': result.confidence_interval[0],
                'CI Upper': result.confidence_interval[1],
                'Magnitude': result.magnitude,
                'Interpretation': result.interpretation
            })
        
        return pd.DataFrame(data)
    
    def plot_effect_sizes(self, save_path: Optional[str] = None) -> plt.Figure:
        """Create visualization of effect sizes"""
        if not self.results:
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Effect Size Analysis', fontsize=16, fontweight='bold')
        
        # Extract data
        effect_types = [result.effect_size_type for result in self.results]
        values = [result.value for result in self.results]
        magnitudes = [result.magnitude for result in self.results]
        ci_lower = [result.confidence_interval[0] for result in self.results]
        ci_upper = [result.confidence_interval[1] for result in self.results]
        
        # Color mapping for magnitudes
        magnitude_colors = {
            'negligible': 'lightgray',
            'small': 'lightblue', 
            'medium': 'orange',
            'large': 'red'
        }
        colors = [magnitude_colors.get(mag, 'gray') for mag in magnitudes]
        
        # Plot 1: Effect sizes with confidence intervals
        y_pos = np.arange(len(effect_types))
        axes[0, 0].barh(y_pos, values, color=colors, alpha=0.7)
        axes[0, 0].errorbar(values, y_pos, 
                           xerr=[np.array(values) - np.array(ci_lower), 
                                np.array(ci_upper) - np.array(values)],
                           fmt='none', color='black', capsize=3)
        axes[0, 0].set_yticks(y_pos)
        axes[0, 0].set_yticklabels(effect_types)
        axes[0, 0].set_xlabel('Effect Size')
        axes[0, 0].set_title('Effect Sizes with Confidence Intervals')
        
        # Plot 2: Magnitude distribution
        magnitude_counts = pd.Series(magnitudes).value_counts()
        axes[0, 1].pie(magnitude_counts.values, labels=magnitude_counts.index, 
                      colors=[magnitude_colors[mag] for mag in magnitude_counts.index],
                      autopct='%1.1f%%', startangle=90)
        axes[0, 1].set_title('Effect Size Magnitude Distribution')
        
        # Plot 3: Effect sizes by type
        df = pd.DataFrame({
            'Effect_Type': effect_types,
            'Value': values,
            'Magnitude': magnitudes
        })
        
        sns.boxplot(data=df, x='Magnitude', y='Value', ax=axes[1, 0])
        axes[1, 0].set_title('Effect Size Values by Magnitude Category')
        
        # Plot 4: Confidence interval widths
        ci_widths = np.array(ci_upper) - np.array(ci_lower)
        axes[1, 1].bar(range(len(effect_types)), ci_widths, color=colors, alpha=0.7)
        axes[1, 1].set_xticks(range(len(effect_types)))
        axes[1, 1].set_xticklabels(effect_types, rotation=45, ha='right')
        axes[1, 1].set_ylabel('Confidence Interval Width')
        axes[1, 1].set_title('Precision of Effect Size Estimates')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def clear_results(self) -> None:
        """Clear all stored results"""
        self.results = [] 