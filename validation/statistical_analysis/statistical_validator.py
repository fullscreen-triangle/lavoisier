"""
Statistical Validator - Comprehensive Statistical Analysis Framework

Combines hypothesis testing, effect size calculation, and bias detection
for complete statistical validation of pipeline comparisons.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from .hypothesis_testing import HypothesisTestSuite, HypothesisResult
from .effect_size import EffectSizeCalculator, EffectSizeResult
from .bias_detection import BiasDetector, BiasResult


@dataclass
class ValidationReport:
    """Container for complete validation report"""
    hypothesis_results: List[HypothesisResult]
    effect_size_results: List[EffectSizeResult]
    bias_results: List[BiasResult]
    summary_statistics: Dict[str, Any]
    recommendations: List[str]
    overall_conclusion: str


class StatisticalValidator:
    """
    Comprehensive statistical validation framework for pipeline comparison
    """
    
    def __init__(self, alpha: float = 0.05, confidence_level: float = 0.95):
        """
        Initialize statistical validator
        
        Args:
            alpha: Significance level for statistical tests
            confidence_level: Confidence level for effect sizes
        """
        self.alpha = alpha
        self.confidence_level = confidence_level
        
        # Initialize component validators
        self.hypothesis_tester = HypothesisTestSuite(alpha=alpha)
        self.effect_calculator = EffectSizeCalculator(confidence_level=confidence_level)
        self.bias_detector = BiasDetector(alpha=alpha)
        
        self.validation_report = None
    
    def validate_pipelines(
        self,
        numerical_data: Dict[str, np.ndarray],
        visual_data: Dict[str, np.ndarray],
        combined_data: Optional[Dict[str, np.ndarray]] = None,
        performance_metrics: Optional[Dict[str, Dict[str, float]]] = None
    ) -> ValidationReport:
        """
        Perform comprehensive statistical validation of pipelines
        
        Args:
            numerical_data: Dictionary of numerical pipeline data
            visual_data: Dictionary of visual pipeline data
            combined_data: Optional dictionary of combined pipeline data
            performance_metrics: Optional performance metrics for cost-benefit analysis
            
        Returns:
            ValidationReport object
        """
        # Clear previous results
        self.hypothesis_tester.clear_results()
        self.effect_calculator.clear_results()
        self.bias_detector.clear_results()
        
        # 1. Hypothesis Testing
        hypothesis_results = self._run_hypothesis_tests(
            numerical_data, visual_data, combined_data, performance_metrics
        )
        
        # 2. Effect Size Calculations
        effect_size_results = self._calculate_effect_sizes(numerical_data, visual_data)
        
        # 3. Bias Detection
        bias_results = self._detect_biases(numerical_data, visual_data)
        
        # 4. Generate summary statistical_analysis
        summary_stats = self._generate_summary_statistics(
            numerical_data, visual_data, combined_data
        )
        
        # 5. Generate recommendations and conclusions
        recommendations = self._generate_recommendations(
            hypothesis_results, effect_size_results, bias_results
        )
        
        overall_conclusion = self._generate_overall_conclusion(
            hypothesis_results, effect_size_results, bias_results
        )
        
        # Create validation report
        self.validation_report = ValidationReport(
            hypothesis_results=hypothesis_results,
            effect_size_results=effect_size_results,
            bias_results=bias_results,
            summary_statistics=summary_stats,
            recommendations=recommendations,
            overall_conclusion=overall_conclusion
        )
        
        return self.validation_report
    
    def _run_hypothesis_tests(
        self,
        numerical_data: Dict[str, np.ndarray],
        visual_data: Dict[str, np.ndarray],
        combined_data: Optional[Dict[str, np.ndarray]],
        performance_metrics: Optional[Dict[str, Dict[str, float]]]
    ) -> List[HypothesisResult]:
        """Run all hypothesis tests"""
        results = []
        
        # H1: Mass accuracy equivalence
        if 'mass_accuracy' in numerical_data and 'mass_accuracy' in visual_data:
            h1_result = self.hypothesis_tester.test_h1_mass_accuracy_equivalence(
                numerical_data['mass_accuracy'],
                visual_data['mass_accuracy']
            )
            results.append(h1_result)
        
        # H2: Complementary information
        if 'features' in numerical_data and 'features' in visual_data:
            h2_result = self.hypothesis_tester.test_h2_complementary_information(
                numerical_data['features'],
                visual_data['features']
            )
            results.append(h2_result)
        
        # H3: Combined performance superiority
        if combined_data and 'performance' in numerical_data and 'performance' in visual_data and 'performance' in combined_data:
            h3_result = self.hypothesis_tester.test_h3_combined_performance(
                numerical_data['performance'],
                visual_data['performance'],
                combined_data['performance']
            )
            results.append(h3_result)
        
        # H4: Cost-benefit justification
        if performance_metrics:
            h4_result = self.hypothesis_tester.test_h4_cost_benefit_analysis(
                performance_metrics['numerical']['cost'],
                performance_metrics['visual']['cost'],
                performance_metrics['numerical']['performance'],
                performance_metrics['visual']['performance']
            )
            results.append(h4_result)
        
        # Apply multiple comparison correction
        self.hypothesis_tester.apply_multiple_comparison_correction()
        
        return results
    
    def _calculate_effect_sizes(
        self,
        numerical_data: Dict[str, np.ndarray],
        visual_data: Dict[str, np.ndarray]
    ) -> List[EffectSizeResult]:
        """Calculate effect sizes for all comparisons"""
        results = []
        
        # Compare all matching data types
        for key in numerical_data.keys():
            if key in visual_data:
                # Cohen's d
                cohens_d = self.effect_calculator.cohens_d(
                    numerical_data[key], visual_data[key]
                )
                results.append(cohens_d)
                
                # Hedges' g (bias-corrected)
                hedges_g = self.effect_calculator.hedges_g(
                    numerical_data[key], visual_data[key]
                )
                results.append(hedges_g)
                
                # Cliff's delta (non-parametric)
                cliffs_delta = self.effect_calculator.cliffs_delta(
                    numerical_data[key], visual_data[key]
                )
                results.append(cliffs_delta)
        
        return results
    
    def _detect_biases(
        self,
        numerical_data: Dict[str, np.ndarray],
        visual_data: Dict[str, np.ndarray]
    ) -> List[BiasResult]:
        """Detect various types of bias"""
        results = []
        
        # Calculate differences for bias detection
        for key in numerical_data.keys():
            if key in visual_data:
                differences = visual_data[key] - numerical_data[key]
                
                # Systematic bias
                systematic_bias = self.bias_detector.detect_systematic_bias(differences)
                results.append(systematic_bias)
                
                # Proportional bias
                proportional_bias = self.bias_detector.detect_proportional_bias(
                    numerical_data[key], visual_data[key]
                )
                results.append(proportional_bias)
        
        # Mass-dependent bias (if m/z data available)
        if 'mz_values' in numerical_data and 'mass_accuracy' in numerical_data and 'mass_accuracy' in visual_data:
            mass_bias = self.bias_detector.detect_mass_dependent_bias(
                numerical_data['mz_values'],
                visual_data['mass_accuracy'] - numerical_data['mass_accuracy']
            )
            results.append(mass_bias)
        
        # Intensity-dependent bias (if intensity data available)
        if 'intensity' in numerical_data and 'mass_accuracy' in numerical_data and 'mass_accuracy' in visual_data:
            intensity_bias = self.bias_detector.detect_intensity_dependent_bias(
                numerical_data['intensity'],
                visual_data['mass_accuracy'] - numerical_data['mass_accuracy']
            )
            results.append(intensity_bias)
        
        return results
    
    def _generate_summary_statistics(
        self,
        numerical_data: Dict[str, np.ndarray],
        visual_data: Dict[str, np.ndarray],
        combined_data: Optional[Dict[str, np.ndarray]]
    ) -> Dict[str, Any]:
        """Generate summary statistical_analysis for all data"""
        summary = {}
        
        # Numerical pipeline statistical_analysis
        summary['numerical'] = {}
        for key, data in numerical_data.items():
            summary['numerical'][key] = {
                'mean': np.mean(data),
                'std': np.std(data),
                'median': np.median(data),
                'min': np.min(data),
                'max': np.max(data),
                'n_samples': len(data)
            }
        
        # Visual pipeline statistical_analysis
        summary['visual'] = {}
        for key, data in visual_data.items():
            summary['visual'][key] = {
                'mean': np.mean(data),
                'std': np.std(data),
                'median': np.median(data),
                'min': np.min(data),
                'max': np.max(data),
                'n_samples': len(data)
            }
        
        # Combined pipeline statistical_analysis (if available)
        if combined_data:
            summary['combined'] = {}
            for key, data in combined_data.items():
                summary['combined'][key] = {
                    'mean': np.mean(data),
                    'std': np.std(data),
                    'median': np.median(data),
                    'min': np.min(data),
                    'max': np.max(data),
                    'n_samples': len(data)
                }
        
        return summary
    
    def _generate_recommendations(
        self,
        hypothesis_results: List[HypothesisResult],
        effect_size_results: List[EffectSizeResult],
        bias_results: List[BiasResult]
    ) -> List[str]:
        """Generate actionable recommendations based on results"""
        recommendations = []
        
        # Analyze hypothesis test results
        significant_hypotheses = [r for r in hypothesis_results if r.significant]
        if len(significant_hypotheses) == 0:
            recommendations.append(
                "No hypotheses were statistically significant. Consider increasing sample size or refining methods."
            )
        
        # Check for mass accuracy equivalence
        h1_results = [r for r in hypothesis_results if "H1" in r.hypothesis]
        if h1_results and h1_results[0].significant:
            recommendations.append(
                "Visual pipeline maintains equivalent mass accuracy to numerical pipeline. This is a positive finding."
            )
        elif h1_results:
            recommendations.append(
                "Visual pipeline does not maintain equivalent mass accuracy. Consider calibration improvements."
            )
        
        # Check for complementary information
        h2_results = [r for r in hypothesis_results if "H2" in r.hypothesis]
        if h2_results and h2_results[0].significant:
            recommendations.append(
                "Visual pipeline provides complementary information. Consider developing hybrid approaches."
            )
        
        # Analyze effect sizes
        large_effects = [r for r in effect_size_results if r.magnitude == "large"]
        if large_effects:
            recommendations.append(
                f"Found {len(large_effects)} large effect sizes, indicating substantial practical differences."
            )
        
        # Analyze bias results
        significant_biases = [r for r in bias_results if r.significant]
        if significant_biases:
            bias_types = [r.bias_type for r in significant_biases]
            recommendations.append(
                f"Detected significant biases: {', '.join(bias_types)}. Address these before deployment."
            )
        
        # Cost-benefit analysis
        h4_results = [r for r in hypothesis_results if "H4" in r.hypothesis]
        if h4_results and not h4_results[0].significant:
            recommendations.append(
                "Visual pipeline cost may not be justified by performance gains. Consider optimization."
            )
        
        return recommendations
    
    def _generate_overall_conclusion(
        self,
        hypothesis_results: List[HypothesisResult],
        effect_size_results: List[EffectSizeResult],
        bias_results: List[BiasResult]
    ) -> str:
        """Generate overall conclusion about pipeline comparison"""
        
        # Count significant results
        sig_hypotheses = len([r for r in hypothesis_results if r.significant])
        total_hypotheses = len(hypothesis_results)
        
        large_effects = len([r for r in effect_size_results if r.magnitude == "large"])
        medium_effects = len([r for r in effect_size_results if r.magnitude == "medium"])
        
        sig_biases = len([r for r in bias_results if r.significant])
        
        # Generate conclusion based on evidence
        if sig_hypotheses >= total_hypotheses * 0.75:  # Most hypotheses supported
            if sig_biases == 0:
                conclusion = "Strong evidence supports the visual pipeline with no significant biases detected."
            else:
                conclusion = "Visual pipeline shows promise but significant biases need to be addressed."
        elif sig_hypotheses >= total_hypotheses * 0.5:  # Some hypotheses supported
            conclusion = "Mixed evidence for visual pipeline effectiveness. Further investigation recommended."
        else:  # Few hypotheses supported
            if large_effects > 0:
                conclusion = "Limited statistical significance but large effect sizes suggest practical importance."
            else:
                conclusion = "Insufficient evidence to support visual pipeline superiority over numerical methods."
        
        # Add effect size context
        if large_effects + medium_effects > 0:
            conclusion += f" Effect size analysis reveals {large_effects} large and {medium_effects} medium effects."
        
        # Add bias warning if needed
        if sig_biases > 0:
            conclusion += f" Warning: {sig_biases} significant bias(es) detected."
        
        return conclusion
    
    def generate_comprehensive_report(self) -> pd.DataFrame:
        """Generate comprehensive statistical report"""
        if not self.validation_report:
            return pd.DataFrame()
        
        # Combine all results into a single report
        report_data = []
        
        # Add hypothesis test results
        for result in self.validation_report.hypothesis_results:
            report_data.append({
                'Analysis Type': 'Hypothesis Test',
                'Test/Metric': result.hypothesis,
                'Method': result.test_name,
                'Statistic': result.statistic,
                'P-value': result.p_value,
                'Effect Size': result.effect_size,
                'Significant': result.significant,
                'Interpretation': result.interpretation
            })
        
        # Add effect size results
        for result in self.validation_report.effect_size_results:
            report_data.append({
                'Analysis Type': 'Effect Size',
                'Test/Metric': result.metric_name,
                'Method': result.effect_size_type,
                'Statistic': result.value,
                'P-value': None,
                'Effect Size': result.value,
                'Significant': result.magnitude in ['medium', 'large'],
                'Interpretation': result.interpretation
            })
        
        # Add bias detection results
        for result in self.validation_report.bias_results:
            report_data.append({
                'Analysis Type': 'Bias Detection',
                'Test/Metric': result.bias_type,
                'Method': 'Statistical Test',
                'Statistic': result.test_statistic,
                'P-value': result.p_value,
                'Effect Size': result.bias_magnitude,
                'Significant': result.significant,
                'Interpretation': result.interpretation
            })
        
        return pd.DataFrame(report_data)
    
    def plot_validation_summary(self, save_path: Optional[str] = None) -> plt.Figure:
        """Create comprehensive validation summary plot"""
        if not self.validation_report:
            return None
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Comprehensive Statistical Validation Summary', fontsize=16, fontweight='bold')
        
        # Plot 1: Hypothesis test results
        h_results = self.validation_report.hypothesis_results
        if h_results:
            h_names = [r.hypothesis.split(':')[0] for r in h_results]
            h_significant = [r.significant for r in h_results]
            h_colors = ['green' if sig else 'red' for sig in h_significant]
            
            axes[0, 0].bar(h_names, [1]*len(h_names), color=h_colors, alpha=0.7)
            axes[0, 0].set_title('Hypothesis Test Results')
            axes[0, 0].set_ylabel('Significance')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Effect size distribution
        e_results = self.validation_report.effect_size_results
        if e_results:
            magnitudes = [r.magnitude for r in e_results]
            magnitude_counts = pd.Series(magnitudes).value_counts()
            
            colors = {'negligible': 'gray', 'small': 'lightblue', 'medium': 'orange', 'large': 'red'}
            plot_colors = [colors.get(mag, 'gray') for mag in magnitude_counts.index]
            
            axes[0, 1].pie(magnitude_counts.values, labels=magnitude_counts.index, 
                          colors=plot_colors, autopct='%1.1f%%', startangle=90)
            axes[0, 1].set_title('Effect Size Distribution')
        
        # Plot 3: Bias detection summary
        b_results = self.validation_report.bias_results
        if b_results:
            bias_types = [r.bias_type for r in b_results]
            bias_significant = [r.significant for r in b_results]
            bias_colors = ['red' if sig else 'blue' for sig in bias_significant]
            
            axes[0, 2].bar(bias_types, [1]*len(bias_types), color=bias_colors, alpha=0.7)
            axes[0, 2].set_title('Bias Detection Results')
            axes[0, 2].set_ylabel('Detection')
            axes[0, 2].tick_params(axis='x', rotation=45)
        
        # Plot 4: P-value distribution
        all_p_values = []
        if h_results:
            all_p_values.extend([r.p_value for r in h_results])
        if b_results:
            all_p_values.extend([r.p_value for r in b_results])
        
        if all_p_values:
            axes[1, 0].hist(all_p_values, bins=20, alpha=0.7, edgecolor='black')
            axes[1, 0].axvline(x=self.alpha, color='red', linestyle='--', label=f'α = {self.alpha}')
            axes[1, 0].set_xlabel('P-value')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('P-value Distribution')
            axes[1, 0].legend()
        
        # Plot 5: Effect size values
        if e_results:
            effect_values = [r.value for r in e_results]
            effect_types = [r.effect_size_type for r in e_results]
            
            axes[1, 1].scatter(range(len(effect_values)), effect_values, 
                              c=[colors.get(r.magnitude, 'gray') for r in e_results], 
                              s=100, alpha=0.7)
            axes[1, 1].set_xticks(range(len(effect_types)))
            axes[1, 1].set_xticklabels(effect_types, rotation=45, ha='right')
            axes[1, 1].set_ylabel('Effect Size Value')
            axes[1, 1].set_title('Effect Size Values')
        
        # Plot 6: Summary statistical_analysis
        summary_text = f"""
Overall Conclusion:
{self.validation_report.overall_conclusion}

Key Findings:
• {len([r for r in h_results if r.significant])}/{len(h_results)} hypotheses significant
• {len([r for r in e_results if r.magnitude in ['medium', 'large']])}/{len(e_results)} medium/large effects
• {len([r for r in b_results if r.significant])}/{len(b_results)} significant biases

Recommendations:
{chr(10).join(['• ' + rec for rec in self.validation_report.recommendations[:3]])}
        """
        
        axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[1, 2].set_xlim(0, 1)
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].axis('off')
        axes[1, 2].set_title('Summary & Recommendations')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def save_validation_report(self, output_dir: str = "validation_results") -> None:
        """Save complete validation report to files"""
        if not self.validation_report:
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save comprehensive report as CSV
        report_df = self.generate_comprehensive_report()
        report_df.to_csv(output_path / "statistical_validation_report.csv", index=False)
        
        # Save individual component reports
        if self.validation_report.hypothesis_results:
            hyp_df = self.hypothesis_tester.generate_summary_report()
            hyp_df.to_csv(output_path / "hypothesis_test_results.csv", index=False)
        
        if self.validation_report.effect_size_results:
            effect_df = self.effect_calculator.generate_summary_report()
            effect_df.to_csv(output_path / "effect_size_results.csv", index=False)
        
        if self.validation_report.bias_results:
            bias_df = self.bias_detector.generate_bias_report()
            bias_df.to_csv(output_path / "bias_detection_results.csv", index=False)
        
        # Save plots
        self.plot_validation_summary(str(output_path / "validation_summary.png"))
        
        if self.validation_report.hypothesis_results:
            self.hypothesis_tester.plot_results(str(output_path / "hypothesis_tests.png"))
        
        if self.validation_report.effect_size_results:
            self.effect_calculator.plot_effect_sizes(str(output_path / "effect_sizes.png"))
        
        if self.validation_report.bias_results:
            self.bias_detector.plot_bias_analysis(str(output_path / "bias_analysis.png")) 