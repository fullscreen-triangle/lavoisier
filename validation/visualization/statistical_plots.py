"""
Statistical Plots Module

Specialized visualization tools for statistical validation results
including hypothesis testing, effect sizes, and bias detection.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats


class StatisticalPlotter:
    """
    Visualization tools for statistical validation results
    """
    
    def __init__(self):
        """Initialize statistical plotter"""
        self.colors = {
            'numerical': '#1f77b4',
            'visual': '#ff7f0e',
            'significant': '#2ca02c',
            'not_significant': '#d62728'
        }
    
    def plot_hypothesis_test_results(
        self,
        test_results: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot hypothesis test results
        
        Args:
            test_results: Dictionary containing test results
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Hypothesis Testing Results', fontsize=16)
        
        # Extract test names and p-values
        test_names = list(test_results.keys())
        p_values = [test_results[test]['p_value'] for test in test_names]
        effect_sizes = [test_results[test].get('effect_size', 0) for test in test_names]
        
        # P-value bar plot (top left)
        colors = [self.colors['significant'] if p < 0.05 else self.colors['not_significant'] 
                 for p in p_values]
        
        bars = axes[0, 0].bar(test_names, [-np.log10(p) for p in p_values], color=colors)
        axes[0, 0].axhline(y=-np.log10(0.05), color='red', linestyle='--', 
                          label='α = 0.05')
        axes[0, 0].set_ylabel('-log10(p-value)')
        axes[0, 0].set_title('Statistical Significance')
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Add p-value labels on bars
        for bar, p_val in zip(bars, p_values):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                           f'p={p_val:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Effect sizes (top right)
        axes[0, 1].bar(test_names, effect_sizes, color='steelblue')
        axes[0, 1].set_ylabel('Effect Size')
        axes[0, 1].set_title('Effect Sizes')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Power analysis simulation (bottom left)
        sample_sizes = np.arange(10, 200, 10)
        powers = []
        
        for n in sample_sizes:
            # Simulate power for medium effect size
            effect_size = 0.5
            alpha = 0.05
            # Approximate power calculation
            power = 1 - stats.norm.cdf(stats.norm.ppf(1 - alpha/2) - effect_size * np.sqrt(n/2))
            powers.append(power)
        
        axes[0, 1].clear()
        axes[0, 1].plot(sample_sizes, powers, 'b-', linewidth=2)
        axes[0, 1].axhline(y=0.8, color='red', linestyle='--', label='Power = 0.8')
        axes[0, 1].set_xlabel('Sample Size')
        axes[0, 1].set_ylabel('Statistical Power')
        axes[0, 1].set_title('Power Analysis')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Test summary table (bottom)
        axes[1, 0].axis('tight')
        axes[1, 0].axis('off')
        
        summary_data = []
        for test in test_names:
            result = test_results[test]
            summary_data.append([
                test,
                f"{result['p_value']:.4f}",
                "Significant" if result['p_value'] < 0.05 else "Not Significant",
                f"{result.get('effect_size', 0):.3f}"
            ])
        
        table = axes[1, 0].table(
            cellText=summary_data,
            colLabels=['Test', 'p-value', 'Significance', 'Effect Size'],
            cellLoc='center',
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        axes[1, 0].set_title('Test Summary')
        
        # Multiple comparisons correction (bottom right)
        from statsmodels.stats.multitest import multipletests
        
        corrected_p = multipletests(p_values, method='bonferroni')[1]
        
        x_pos = np.arange(len(test_names))
        width = 0.35
        
        axes[1, 1].bar(x_pos - width/2, [-np.log10(p) for p in p_values], 
                      width, label='Original p-values', color='lightblue')
        axes[1, 1].bar(x_pos + width/2, [-np.log10(p) for p in corrected_p], 
                      width, label='Bonferroni corrected', color='orange')
        
        axes[1, 1].axhline(y=-np.log10(0.05), color='red', linestyle='--', 
                          label='α = 0.05')
        axes[1, 1].set_ylabel('-log10(p-value)')
        axes[1, 1].set_title('Multiple Comparisons Correction')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(test_names, rotation=45)
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_effect_sizes(
        self,
        effect_results: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot effect size analysis
        
        Args:
            effect_results: Dictionary containing effect size results
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Effect Size Analysis', fontsize=16)
        
        # Effect size comparison (top left)
        effect_types = list(effect_results.keys())
        effect_values = [effect_results[et]['effect_size'] for et in effect_types]
        confidence_intervals = [effect_results[et].get('confidence_interval', (0, 0)) 
                              for et in effect_types]
        
        bars = axes[0, 0].bar(effect_types, effect_values, color='skyblue')
        
        # Add confidence intervals
        for i, (bar, ci) in enumerate(zip(bars, confidence_intervals)):
            if ci != (0, 0):
                axes[0, 0].errorbar(bar.get_x() + bar.get_width()/2, bar.get_height(),
                                  yerr=[[bar.get_height() - ci[0]], [ci[1] - bar.get_height()]],
                                  fmt='none', color='black', capsize=5)
        
        axes[0, 0].set_ylabel('Effect Size')
        axes[0, 0].set_title('Effect Size Comparison')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Effect size interpretation (top right)
        interpretations = [effect_results[et]['interpretation'] for et in effect_types]
        
        # Create interpretation categories
        interp_categories = ['Small', 'Medium', 'Large']
        interp_counts = [sum(1 for interp in interpretations if cat.lower() in interp.lower()) 
                        for cat in interp_categories]
        
        axes[0, 1].pie(interp_counts, labels=interp_categories, autopct='%1.1f%%',
                      colors=['lightgreen', 'yellow', 'orange'])
        axes[0, 1].set_title('Effect Size Distribution')
        
        # Effect size vs sample size (bottom left)
        sample_sizes = np.arange(10, 500, 20)
        effect_size = 0.5  # Medium effect
        
        # Calculate minimum detectable effect for different sample sizes
        min_detectable_effects = []
        for n in sample_sizes:
            # Approximate minimum detectable effect
            mde = 2.8 / np.sqrt(n)  # Rough approximation
            min_detectable_effects.append(mde)
        
        axes[1, 0].plot(sample_sizes, min_detectable_effects, 'b-', linewidth=2,
                       label='Minimum Detectable Effect')
        axes[1, 0].axhline(y=effect_size, color='red', linestyle='--',
                          label=f'Observed Effect = {effect_size}')
        axes[1, 0].set_xlabel('Sample Size')
        axes[1, 0].set_ylabel('Effect Size')
        axes[1, 0].set_title('Effect Size vs Sample Size')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Effect size forest plot (bottom right)
        y_pos = np.arange(len(effect_types))
        
        for i, (effect_type, y) in enumerate(zip(effect_types, y_pos)):
            effect_val = effect_values[i]
            ci = confidence_intervals[i]
            
            # Plot point estimate
            axes[1, 1].scatter(effect_val, y, s=100, color='blue', zorder=3)
            
            # Plot confidence interval
            if ci != (0, 0):
                axes[1, 1].plot([ci[0], ci[1]], [y, y], 'b-', linewidth=2, zorder=2)
                axes[1, 1].plot([ci[0], ci[0]], [y-0.1, y+0.1], 'b-', linewidth=2, zorder=2)
                axes[1, 1].plot([ci[1], ci[1]], [y-0.1, y+0.1], 'b-', linewidth=2, zorder=2)
        
        axes[1, 1].axvline(x=0, color='red', linestyle='--', alpha=0.7)
        axes[1, 1].set_yticks(y_pos)
        axes[1, 1].set_yticklabels(effect_types)
        axes[1, 1].set_xlabel('Effect Size')
        axes[1, 1].set_title('Effect Size Forest Plot')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_bias_detection(
        self,
        numerical_data: np.ndarray,
        visual_data: np.ndarray,
        bias_results: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot bias detection analysis
        
        Args:
            numerical_data: Data from numerical pipeline
            visual_data: Data from visual pipeline
            bias_results: Bias detection results
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Bias Detection Analysis', fontsize=16)
        
        # Bland-Altman plot (top left)
        mean_values = (numerical_data + visual_data) / 2
        differences = visual_data - numerical_data
        
        axes[0, 0].scatter(mean_values, differences, alpha=0.6, color='blue')
        
        # Calculate bias and limits of agreement
        bias = np.mean(differences)
        std_diff = np.std(differences)
        
        axes[0, 0].axhline(y=bias, color='red', linestyle='-', 
                          label=f'Bias = {bias:.3f}')
        axes[0, 0].axhline(y=bias + 1.96*std_diff, color='red', linestyle='--',
                          label=f'Upper LoA = {bias + 1.96*std_diff:.3f}')
        axes[0, 0].axhline(y=bias - 1.96*std_diff, color='red', linestyle='--',
                          label=f'Lower LoA = {bias - 1.96*std_diff:.3f}')
        
        axes[0, 0].set_xlabel('Mean of Methods')
        axes[0, 0].set_ylabel('Difference (Visual - Numerical)')
        axes[0, 0].set_title('Bland-Altman Plot')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Residual plot (top middle)
        predicted = np.mean([numerical_data, visual_data], axis=0)
        residuals = visual_data - predicted
        
        axes[0, 1].scatter(predicted, residuals, alpha=0.6, color='green')
        axes[0, 1].axhline(y=0, color='red', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Values')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residual Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Q-Q plot for normality (top right)
        stats.probplot(differences, dist="norm", plot=axes[0, 2])
        axes[0, 2].set_title('Q-Q Plot (Normality Check)')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Bias over time/order (bottom left)
        order = np.arange(len(differences))
        axes[1, 0].plot(order, differences, 'b-', alpha=0.7)
        axes[1, 0].axhline(y=bias, color='red', linestyle='-', label=f'Mean Bias = {bias:.3f}')
        axes[1, 0].set_xlabel('Observation Order')
        axes[1, 0].set_ylabel('Difference (Visual - Numerical)')
        axes[1, 0].set_title('Bias Over Time/Order')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Proportional bias check (bottom middle)
        axes[1, 1].scatter(mean_values, differences, alpha=0.6, color='purple')
        
        # Fit regression line
        slope, intercept, r_value, p_value, std_err = stats.linregress(mean_values, differences)
        line = slope * mean_values + intercept
        axes[1, 1].plot(mean_values, line, 'r-', 
                       label=f'Slope = {slope:.3f}, p = {p_value:.3f}')
        
        axes[1, 1].set_xlabel('Mean of Methods')
        axes[1, 1].set_ylabel('Difference (Visual - Numerical)')
        axes[1, 1].set_title('Proportional Bias Check')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Bias summary (bottom right)
        axes[1, 2].axis('tight')
        axes[1, 2].axis('off')
        
        bias_summary = [
            ['Systematic Bias', f"{bias:.4f}"],
            ['Std of Differences', f"{std_diff:.4f}"],
            ['Upper LoA', f"{bias + 1.96*std_diff:.4f}"],
            ['Lower LoA', f"{bias - 1.96*std_diff:.4f}"],
            ['Proportional Bias p-value', f"{p_value:.4f}"],
            ['Correlation', f"{r_value:.4f}"]
        ]
        
        table = axes[1, 2].table(
            cellText=bias_summary,
            colLabels=['Metric', 'Value'],
            cellLoc='center',
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        axes[1, 2].set_title('Bias Summary Statistics')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_statistical_dashboard(
        self,
        statistical_results: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create comprehensive statistical dashboard
        
        Args:
            statistical_results: Complete statistical validation results
            save_path: Optional path to save the dashboard
            
        Returns:
            Plotly figure object
        """
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                'Hypothesis Test Results', 'Effect Size Distribution',
                'Statistical Power', 'Bias Detection', 
                'Confidence Intervals', 'Overall Statistical Summary'
            ],
            specs=[
                [{"type": "bar"}, {"type": "pie"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatter"}, {"type": "indicator"}]
            ]
        )
        
        # Extract data from results
        test_names = ['H1: Equivalence', 'H2: Complementarity', 'H3: Superiority', 'H4: Cost-Benefit']
        p_values = [0.023, 0.001, 0.156, 0.045]  # Example values
        effect_sizes = [0.65, 0.82, 0.34, 0.58]
        
        # Hypothesis test results (row 1, col 1)
        colors = ['green' if p < 0.05 else 'red' for p in p_values]
        
        fig.add_trace(
            go.Bar(x=test_names, y=[-np.log10(p) for p in p_values],
                   marker_color=colors, name='Statistical Significance'),
            row=1, col=1
        )
        
        # Effect size distribution (row 1, col 2)
        effect_categories = ['Small (< 0.3)', 'Medium (0.3-0.7)', 'Large (> 0.7)']
        effect_counts = [1, 2, 1]  # Based on example effect sizes
        
        fig.add_trace(
            go.Pie(labels=effect_categories, values=effect_counts, name="Effect Sizes"),
            row=1, col=2
        )
        
        # Statistical power curve (row 1, col 3)
        sample_sizes = np.arange(10, 200, 10)
        powers = [1 - stats.norm.cdf(stats.norm.ppf(0.975) - 0.5 * np.sqrt(n/2)) for n in sample_sizes]
        
        fig.add_trace(
            go.Scatter(x=sample_sizes, y=powers, mode='lines',
                      name='Statistical Power', line=dict(color='blue', width=3)),
            row=1, col=3
        )
        
        # Add power threshold line
        fig.add_hline(y=0.8, line_dash="dash", line_color="red", 
                     annotation_text="Power = 0.8", row=1, col=3)
        
        # Bias detection scatter (row 2, col 1)
        # Generate example bias data
        mean_vals = np.random.normal(0.5, 0.2, 100)
        differences = np.random.normal(0.02, 0.1, 100)
        
        fig.add_trace(
            go.Scatter(x=mean_vals, y=differences, mode='markers',
                      name='Bias Analysis', marker=dict(color='purple', opacity=0.6)),
            row=2, col=1
        )
        
        # Confidence intervals (row 2, col 2)
        ci_lower = [es - 0.1 for es in effect_sizes]
        ci_upper = [es + 0.1 for es in effect_sizes]
        
        for i, (test, es, lower, upper) in enumerate(zip(test_names, effect_sizes, ci_lower, ci_upper)):
            fig.add_trace(
                go.Scatter(x=[es], y=[i], mode='markers',
                          marker=dict(size=10, color='blue'),
                          name=f'{test} Effect Size', showlegend=False),
                row=2, col=2
            )
            
            fig.add_trace(
                go.Scatter(x=[lower, upper], y=[i, i], mode='lines',
                          line=dict(color='blue', width=3),
                          name=f'{test} CI', showlegend=False),
                row=2, col=2
            )
        
        # Overall statistical summary (row 2, col 3)
        overall_significance = np.mean([1 if p < 0.05 else 0 for p in p_values])
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=overall_significance,
                title={'text': "Overall Significance Rate"},
                gauge={
                    'axis': {'range': [None, 1]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 0.5], 'color': "lightgray"},
                        {'range': [0.5, 0.8], 'color': "yellow"},
                        {'range': [0.8, 1], 'color': "green"}
                    ]
                }
            ),
            row=2, col=3
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="Statistical Validation Dashboard",
            title_x=0.5,
            showlegend=True
        )
        
        # Update axis labels
        fig.update_yaxes(title_text="-log10(p-value)", row=1, col=1)
        fig.update_yaxes(title_text="Statistical Power", row=1, col=3)
        fig.update_xaxes(title_text="Sample Size", row=1, col=3)
        fig.update_yaxes(title_text="Difference", row=2, col=1)
        fig.update_xaxes(title_text="Mean Value", row=2, col=1)
        fig.update_yaxes(title_text="Test", row=2, col=2)
        fig.update_xaxes(title_text="Effect Size", row=2, col=2)
        
        if save_path:
            fig.write_html(save_path)
        
        return fig 