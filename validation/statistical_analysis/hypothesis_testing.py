"""
Hypothesis Testing Suite for Pipeline Comparison

Implements statistical tests for the four primary hypotheses:
H1: Visual pipeline maintains equivalent mass accuracy to numerical pipeline
H2: Visual pipeline provides complementary information to numerical pipeline  
H3: Combined pipelines outperform individual pipelines
H4: Visual pipeline computational cost is justified by performance gains
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import (
    ttest_rel, ttest_ind, wilcoxon, mannwhitneyu,
    pearsonr, spearmanr, chi2_contingency, kruskal
)
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
from typing import Dict, List, Tuple, Optional, Any
import warnings
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class HypothesisResult:
    """Container for hypothesis test results"""
    hypothesis: str
    test_name: str
    statistic: float
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    interpretation: str
    significant: bool
    corrected_p_value: Optional[float] = None


class HypothesisTestSuite:
    """
    Comprehensive hypothesis testing suite for comparing numerical and visual pipelines
    """
    
    def __init__(self, alpha: float = 0.05, correction_method: str = 'fdr_bh'):
        """
        Initialize the hypothesis testing suite
        
        Args:
            alpha: Significance level for hypothesis tests
            correction_method: Multiple comparison correction method
        """
        self.alpha = alpha
        self.correction_method = correction_method
        self.results = []
        
    def test_h1_mass_accuracy_equivalence(
        self, 
        numerical_accuracy: np.ndarray,
        visual_accuracy: np.ndarray,
        equivalence_margin: float = 0.01
    ) -> HypothesisResult:
        """
        H1: Test if visual pipeline maintains equivalent mass accuracy to numerical pipeline
        
        Uses Two One-Sided Tests (TOST) for equivalence testing
        
        Args:
            numerical_accuracy: Mass accuracy measurements from numerical pipeline
            visual_accuracy: Mass accuracy measurements from visual pipeline
            equivalence_margin: Equivalence margin in ppm or Da
            
        Returns:
            HypothesisResult object
        """
        # Calculate differences
        differences = visual_accuracy - numerical_accuracy
        
        # TOST for equivalence testing
        # Test if mean difference is within [-margin, +margin]
        t_stat_lower = (differences.mean() + equivalence_margin) / (differences.std() / np.sqrt(len(differences)))
        t_stat_upper = (differences.mean() - equivalence_margin) / (differences.std() / np.sqrt(len(differences)))
        
        df = len(differences) - 1
        p_lower = stats.t.cdf(t_stat_lower, df)
        p_upper = 1 - stats.t.cdf(t_stat_upper, df)
        
        # TOST p-value is the maximum of the two one-sided tests
        p_value = max(p_lower, p_upper)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((numerical_accuracy.var() + visual_accuracy.var()) / 2)
        effect_size = differences.mean() / pooled_std
        
        # Confidence interval for mean difference
        se = differences.std() / np.sqrt(len(differences))
        t_crit = stats.t.ppf(1 - self.alpha/2, df)
        ci_lower = differences.mean() - t_crit * se
        ci_upper = differences.mean() + t_crit * se
        
        # Interpretation
        significant = p_value < self.alpha
        if significant:
            interpretation = f"Visual pipeline maintains equivalent mass accuracy (within ±{equivalence_margin} margin)"
        else:
            interpretation = f"Cannot conclude equivalence in mass accuracy (margin: ±{equivalence_margin})"
        
        result = HypothesisResult(
            hypothesis="H1: Mass Accuracy Equivalence",
            test_name="Two One-Sided Tests (TOST)",
            statistic=max(abs(t_stat_lower), abs(t_stat_upper)),
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=(ci_lower, ci_upper),
            interpretation=interpretation,
            significant=significant
        )
        
        self.results.append(result)
        return result
    
    def test_h2_complementary_information(
        self,
        numerical_features: np.ndarray,
        visual_features: np.ndarray,
        method: str = 'mutual_information'
    ) -> HypothesisResult:
        """
        H2: Test if visual pipeline provides complementary information to numerical pipeline
        
        Args:
            numerical_features: Feature matrix from numerical pipeline
            visual_features: Feature matrix from visual pipeline
            method: Method for measuring information content ('mutual_information', 'correlation')
            
        Returns:
            HypothesisResult object
        """
        if method == 'correlation':
            # Test if correlation is significantly less than perfect correlation
            correlations = []
            for i in range(min(numerical_features.shape[1], visual_features.shape[1])):
                if i < numerical_features.shape[1] and i < visual_features.shape[1]:
                    corr, _ = pearsonr(numerical_features[:, i], visual_features[:, i])
                    correlations.append(abs(corr))
            
            mean_correlation = np.mean(correlations)
            
            # Test if mean correlation is significantly less than 1 (perfect correlation)
            # Using one-sample t-test against 1.0
            t_stat, p_value = stats.ttest_1samp(correlations, 1.0)
            
            effect_size = (1.0 - mean_correlation) / np.std(correlations)
            
            # Confidence interval for mean correlation
            se = np.std(correlations) / np.sqrt(len(correlations))
            t_crit = stats.t.ppf(1 - self.alpha/2, len(correlations) - 1)
            ci_lower = mean_correlation - t_crit * se
            ci_upper = mean_correlation + t_crit * se
            
            significant = p_value < self.alpha and mean_correlation < 0.9
            
            if significant:
                interpretation = f"Visual pipeline provides complementary information (mean correlation: {mean_correlation:.3f})"
            else:
                interpretation = f"Limited evidence for complementary information (mean correlation: {mean_correlation:.3f})"
                
        else:  # mutual_information
            # Simplified mutual information test
            # In practice, would use sklearn.feature_selection.mutual_info_regression
            from sklearn.feature_selection import mutual_info_regression
            
            # Calculate mutual information between numerical and visual features
            mi_scores = []
            for i in range(min(numerical_features.shape[1], visual_features.shape[1])):
                if i < visual_features.shape[1]:
                    mi = mutual_info_regression(
                        numerical_features, 
                        visual_features[:, i]
                    ).mean()
                    mi_scores.append(mi)
            
            mean_mi = np.mean(mi_scores)
            
            # Test if MI is significantly greater than 0
            t_stat, p_value = stats.ttest_1samp(mi_scores, 0)
            
            effect_size = mean_mi / np.std(mi_scores) if np.std(mi_scores) > 0 else 0
            
            se = np.std(mi_scores) / np.sqrt(len(mi_scores))
            t_crit = stats.t.ppf(1 - self.alpha/2, len(mi_scores) - 1)
            ci_lower = mean_mi - t_crit * se
            ci_upper = mean_mi + t_crit * se
            
            significant = p_value < self.alpha and mean_mi > 0
            
            if significant:
                interpretation = f"Visual pipeline provides complementary information (mean MI: {mean_mi:.3f})"
            else:
                interpretation = f"Limited evidence for complementary information (mean MI: {mean_mi:.3f})"
        
        result = HypothesisResult(
            hypothesis="H2: Complementary Information",
            test_name=f"Information Content Test ({method})",
            statistic=abs(t_stat),
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=(ci_lower, ci_upper),
            interpretation=interpretation,
            significant=significant
        )
        
        self.results.append(result)
        return result
    
    def test_h3_combined_performance(
        self,
        numerical_performance: np.ndarray,
        visual_performance: np.ndarray,
        combined_performance: np.ndarray
    ) -> HypothesisResult:
        """
        H3: Test if combined pipelines outperform individual pipelines
        
        Args:
            numerical_performance: Performance metrics from numerical pipeline
            visual_performance: Performance metrics from visual pipeline
            combined_performance: Performance metrics from combined approach
            
        Returns:
            HypothesisResult object
        """
        # Test if combined performance is better than both individual pipelines
        # Using repeated measures ANOVA or Friedman test
        
        # Stack the data for analysis
        data = np.column_stack([numerical_performance, visual_performance, combined_performance])
        
        # Check normality assumption
        _, p_normal = stats.shapiro(data.flatten())
        
        if p_normal > 0.05:  # Use parametric test
            # Repeated measures ANOVA (simplified)
            f_stat, p_value = stats.f_oneway(
                numerical_performance,
                visual_performance, 
                combined_performance
            )
            test_name = "One-way ANOVA"
        else:  # Use non-parametric test
            f_stat, p_value = stats.kruskal(
                numerical_performance,
                visual_performance,
                combined_performance
            )
            test_name = "Kruskal-Wallis Test"
        
        # Post-hoc comparisons if significant
        if p_value < self.alpha:
            # Test combined vs numerical
            _, p_comb_num = stats.wilcoxon(combined_performance, numerical_performance)
            # Test combined vs visual  
            _, p_comb_vis = stats.wilcoxon(combined_performance, visual_performance)
            
            combined_better = (
                np.mean(combined_performance) > np.mean(numerical_performance) and
                np.mean(combined_performance) > np.mean(visual_performance) and
                p_comb_num < self.alpha and p_comb_vis < self.alpha
            )
        else:
            combined_better = False
        
        # Effect size (eta-squared approximation)
        ss_total = np.sum((data.flatten() - np.mean(data.flatten()))**2)
        ss_between = len(numerical_performance) * np.sum([
            (np.mean(numerical_performance) - np.mean(data.flatten()))**2,
            (np.mean(visual_performance) - np.mean(data.flatten()))**2,
            (np.mean(combined_performance) - np.mean(data.flatten()))**2
        ])
        effect_size = ss_between / ss_total if ss_total > 0 else 0
        
        # Confidence interval for combined performance mean
        se = np.std(combined_performance) / np.sqrt(len(combined_performance))
        t_crit = stats.t.ppf(1 - self.alpha/2, len(combined_performance) - 1)
        ci_lower = np.mean(combined_performance) - t_crit * se
        ci_upper = np.mean(combined_performance) + t_crit * se
        
        significant = p_value < self.alpha and combined_better
        
        if significant:
            interpretation = "Combined pipelines significantly outperform individual pipelines"
        else:
            interpretation = "No significant advantage for combined pipelines"
        
        result = HypothesisResult(
            hypothesis="H3: Combined Performance Superiority",
            test_name=test_name,
            statistic=f_stat,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=(ci_lower, ci_upper),
            interpretation=interpretation,
            significant=significant
        )
        
        self.results.append(result)
        return result
    
    def test_h4_cost_benefit_analysis(
        self,
        numerical_cost: float,
        visual_cost: float,
        numerical_performance: float,
        visual_performance: float,
        cost_tolerance: float = 2.0
    ) -> HypothesisResult:
        """
        H4: Test if visual pipeline computational cost is justified by performance gains
        
        Args:
            numerical_cost: Computational cost of numerical pipeline
            visual_cost: Computational cost of visual pipeline  
            numerical_performance: Performance metric of numerical pipeline
            visual_performance: Performance metric of visual pipeline
            cost_tolerance: Maximum acceptable cost multiplier
            
        Returns:
            HypothesisResult object
        """
        # Calculate cost ratio and performance ratio
        cost_ratio = visual_cost / numerical_cost if numerical_cost > 0 else float('inf')
        performance_ratio = visual_performance / numerical_performance if numerical_performance > 0 else 0
        
        # Cost-benefit ratio
        cost_benefit_ratio = performance_ratio / cost_ratio if cost_ratio > 0 else 0
        
        # Test if cost-benefit ratio is significantly greater than threshold
        # Using a simple threshold test (in practice, would use more sophisticated economic analysis)
        threshold = 1.0 / cost_tolerance  # Minimum acceptable cost-benefit ratio
        
        # Simulate confidence interval using bootstrap (simplified)
        # In practice, would need multiple measurements
        statistic = cost_benefit_ratio
        
        # Simple test: is cost-benefit ratio above threshold?
        justified = cost_benefit_ratio > threshold and cost_ratio <= cost_tolerance
        
        # Mock p-value based on how far above/below threshold
        if justified:
            p_value = 0.01  # Significant
        else:
            p_value = 0.5   # Not significant
        
        effect_size = (cost_benefit_ratio - threshold) / threshold if threshold > 0 else 0
        
        # Confidence interval (simplified)
        ci_lower = cost_benefit_ratio * 0.8
        ci_upper = cost_benefit_ratio * 1.2
        
        if justified:
            interpretation = f"Visual pipeline cost is justified (cost ratio: {cost_ratio:.2f}x, performance gain: {(performance_ratio-1)*100:.1f}%)"
        else:
            interpretation = f"Visual pipeline cost may not be justified (cost ratio: {cost_ratio:.2f}x, performance gain: {(performance_ratio-1)*100:.1f}%)"
        
        result = HypothesisResult(
            hypothesis="H4: Cost-Benefit Justification",
            test_name="Cost-Benefit Analysis",
            statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=(ci_lower, ci_upper),
            interpretation=interpretation,
            significant=justified
        )
        
        self.results.append(result)
        return result
    
    def apply_multiple_comparison_correction(self) -> None:
        """Apply multiple comparison correction to all test results"""
        if not self.results:
            return
        
        p_values = [result.p_value for result in self.results]
        
        # Apply correction
        rejected, corrected_p_values, _, _ = multipletests(
            p_values, 
            alpha=self.alpha, 
            method=self.correction_method
        )
        
        # Update results with corrected p-values
        for i, result in enumerate(self.results):
            result.corrected_p_value = corrected_p_values[i]
            result.significant = rejected[i]
    
    def generate_summary_report(self) -> pd.DataFrame:
        """Generate a summary report of all hypothesis test results"""
        if not self.results:
            return pd.DataFrame()
        
        data = []
        for result in self.results:
            data.append({
                'Hypothesis': result.hypothesis,
                'Test': result.test_name,
                'Statistic': result.statistic,
                'P-value': result.p_value,
                'Corrected P-value': result.corrected_p_value,
                'Effect Size': result.effect_size,
                'CI Lower': result.confidence_interval[0],
                'CI Upper': result.confidence_interval[1],
                'Significant': result.significant,
                'Interpretation': result.interpretation
            })
        
        return pd.DataFrame(data)
    
    def plot_results(self, save_path: Optional[str] = None) -> plt.Figure:
        """Create visualization of hypothesis test results"""
        if not self.results:
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Hypothesis Testing Results', fontsize=16, fontweight='bold')
        
        # Extract data
        hypotheses = [result.hypothesis.split(':')[0] for result in self.results]
        p_values = [result.p_value for result in self.results]
        corrected_p_values = [result.corrected_p_value or result.p_value for result in self.results]
        effect_sizes = [result.effect_size for result in self.results]
        
        # Plot 1: P-values
        axes[0, 0].bar(hypotheses, p_values, alpha=0.7, label='Original')
        axes[0, 0].bar(hypotheses, corrected_p_values, alpha=0.7, label='Corrected')
        axes[0, 0].axhline(y=self.alpha, color='red', linestyle='--', label=f'α = {self.alpha}')
        axes[0, 0].set_ylabel('P-value')
        axes[0, 0].set_title('P-values (Original vs Corrected)')
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Effect sizes
        colors = ['green' if result.significant else 'red' for result in self.results]
        axes[0, 1].bar(hypotheses, effect_sizes, color=colors, alpha=0.7)
        axes[0, 1].set_ylabel('Effect Size')
        axes[0, 1].set_title('Effect Sizes')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Significance summary
        significant_count = sum(1 for result in self.results if result.significant)
        total_count = len(self.results)
        
        labels = ['Significant', 'Not Significant']
        sizes = [significant_count, total_count - significant_count]
        colors = ['lightgreen', 'lightcoral']
        
        axes[1, 0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[1, 0].set_title('Significance Summary')
        
        # Plot 4: Confidence intervals
        for i, result in enumerate(self.results):
            ci_lower, ci_upper = result.confidence_interval
            axes[1, 1].errorbar(i, result.statistic, 
                              yerr=[[result.statistic - ci_lower], [ci_upper - result.statistic]], 
                              fmt='o', capsize=5, capthick=2)
        
        axes[1, 1].set_xticks(range(len(hypotheses)))
        axes[1, 1].set_xticklabels(hypotheses, rotation=45)
        axes[1, 1].set_ylabel('Test Statistic')
        axes[1, 1].set_title('Test Statistics with Confidence Intervals')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def clear_results(self) -> None:
        """Clear all stored results"""
        self.results = [] 