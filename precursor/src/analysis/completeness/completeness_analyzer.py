"""
Completeness Analyzer Module

Comprehensive assessment of data completeness including spectrum coverage,
processing success rates, missing data patterns, and failure mode analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


@dataclass
class CompletenessResult:
    """Container for completeness assessment results"""
    metric_name: str
    completeness_score: float  # 0-1 score
    total_expected: int
    total_processed: int
    missing_count: int
    failure_count: int
    interpretation: str
    metadata: Dict[str, Any] = None


class CompletenessAnalyzer:
    """
    Comprehensive completeness assessment for pipeline outputs
    """
    
    def __init__(self):
        """Initialize completeness analyzer"""
        self.results = []
        
    def assess_spectrum_coverage(
        self,
        expected_spectra: List[str],
        processed_spectra: List[str],
        pipeline_name: str = "Unknown"
    ) -> CompletenessResult:
        """
        Assess spectrum coverage completeness
        
        Args:
            expected_spectra: List of expected spectrum identifiers
            processed_spectra: List of successfully processed spectrum identifiers
            pipeline_name: Name of the pipeline being assessed
            
        Returns:
            CompletenessResult object
        """
        total_expected = len(expected_spectra)
        total_processed = len(processed_spectra)
        
        # Find missing spectra
        expected_set = set(expected_spectra)
        processed_set = set(processed_spectra)
        missing_spectra = expected_set - processed_set
        missing_count = len(missing_spectra)
        
        # Calculate completeness score
        completeness_score = total_processed / total_expected if total_expected > 0 else 0.0
        
        # Interpretation
        if completeness_score >= 0.95:
            interpretation = f"Excellent spectrum coverage ({completeness_score:.1%})"
        elif completeness_score >= 0.90:
            interpretation = f"Good spectrum coverage ({completeness_score:.1%})"
        elif completeness_score >= 0.80:
            interpretation = f"Moderate spectrum coverage ({completeness_score:.1%})"
        else:
            interpretation = f"Poor spectrum coverage ({completeness_score:.1%})"
        
        result = CompletenessResult(
            metric_name=f"Spectrum Coverage ({pipeline_name})",
            completeness_score=completeness_score,
            total_expected=total_expected,
            total_processed=total_processed,
            missing_count=missing_count,
            failure_count=missing_count,  # Assuming missing = failed
            interpretation=interpretation,
            metadata={
                'missing_spectra': list(missing_spectra),
                'pipeline_name': pipeline_name
            }
        )
        
        self.results.append(result)
        return result
    
    def assess_feature_extraction_success(
        self,
        input_data_count: int,
        successful_extractions: int,
        failed_extractions: int,
        pipeline_name: str = "Unknown"
    ) -> CompletenessResult:
        """
        Assess feature extraction success rate
        
        Args:
            input_data_count: Total number of input data points
            successful_extractions: Number of successful feature extractions
            failed_extractions: Number of failed feature extractions
            pipeline_name: Name of the pipeline being assessed
            
        Returns:
            CompletenessResult object
        """
        total_expected = input_data_count
        total_processed = successful_extractions
        failure_count = failed_extractions
        missing_count = total_expected - total_processed
        
        # Calculate success rate
        completeness_score = total_processed / total_expected if total_expected > 0 else 0.0
        
        # Interpretation
        if completeness_score >= 0.95:
            interpretation = f"Excellent feature extraction success ({completeness_score:.1%})"
        elif completeness_score >= 0.90:
            interpretation = f"Good feature extraction success ({completeness_score:.1%})"
        elif completeness_score >= 0.80:
            interpretation = f"Moderate feature extraction success ({completeness_score:.1%})"
        else:
            interpretation = f"Poor feature extraction success ({completeness_score:.1%})"
        
        result = CompletenessResult(
            metric_name=f"Feature Extraction Success ({pipeline_name})",
            completeness_score=completeness_score,
            total_expected=total_expected,
            total_processed=total_processed,
            missing_count=missing_count,
            failure_count=failure_count,
            interpretation=interpretation,
            metadata={
                'success_rate': completeness_score,
                'failure_rate': failure_count / total_expected if total_expected > 0 else 0,
                'pipeline_name': pipeline_name
            }
        )
        
        self.results.append(result)
        return result
    
    def assess_data_completeness_by_category(
        self,
        data_categories: Dict[str, Dict[str, int]],
        pipeline_name: str = "Unknown"
    ) -> List[CompletenessResult]:
        """
        Assess completeness by data category (e.g., by m/z range, intensity range)
        
        Args:
            data_categories: Dictionary with category names and their expected/processed counts
                           Format: {'category_name': {'expected': int, 'processed': int}}
            pipeline_name: Name of the pipeline being assessed
            
        Returns:
            List of CompletenessResult objects
        """
        results = []
        
        for category_name, counts in data_categories.items():
            expected = counts.get('expected', 0)
            processed = counts.get('processed', 0)
            missing = expected - processed
            
            completeness_score = processed / expected if expected > 0 else 0.0
            
            # Interpretation
            if completeness_score >= 0.95:
                interpretation = f"Excellent completeness in {category_name} ({completeness_score:.1%})"
            elif completeness_score >= 0.90:
                interpretation = f"Good completeness in {category_name} ({completeness_score:.1%})"
            elif completeness_score >= 0.80:
                interpretation = f"Moderate completeness in {category_name} ({completeness_score:.1%})"
            else:
                interpretation = f"Poor completeness in {category_name} ({completeness_score:.1%})"
            
            result = CompletenessResult(
                metric_name=f"{category_name} Completeness ({pipeline_name})",
                completeness_score=completeness_score,
                total_expected=expected,
                total_processed=processed,
                missing_count=missing,
                failure_count=missing,
                interpretation=interpretation,
                metadata={
                    'category': category_name,
                    'pipeline_name': pipeline_name
                }
            )
            
            results.append(result)
        
        self.results.extend(results)
        return results
    
    def compare_pipeline_completeness(
        self,
        numerical_results: Dict[str, int],
        visual_results: Dict[str, int],
        expected_total: int
    ) -> List[CompletenessResult]:
        """
        Compare completeness between numerical and visual pipelines
        
        Args:
            numerical_results: Dictionary with numerical pipeline results
            visual_results: Dictionary with visual pipeline results
            expected_total: Total expected data points
            
        Returns:
            List of CompletenessResult objects for comparison
        """
        results = []
        
        # Assess numerical pipeline
        num_processed = numerical_results.get('processed', 0)
        num_failed = numerical_results.get('failed', 0)
        
        num_result = self.assess_feature_extraction_success(
            expected_total, num_processed, num_failed, "Numerical"
        )
        results.append(num_result)
        
        # Assess visual pipeline
        vis_processed = visual_results.get('processed', 0)
        vis_failed = visual_results.get('failed', 0)
        
        vis_result = self.assess_feature_extraction_success(
            expected_total, vis_processed, vis_failed, "Visual"
        )
        results.append(vis_result)
        
        # Comparative analysis
        completeness_diff = abs(num_result.completeness_score - vis_result.completeness_score)
        
        if completeness_diff < 0.05:
            comparison_interpretation = "Similar completeness between pipelines"
        elif num_result.completeness_score > vis_result.completeness_score:
            comparison_interpretation = f"Numerical pipeline more complete (+{completeness_diff:.1%})"
        else:
            comparison_interpretation = f"Visual pipeline more complete (+{completeness_diff:.1%})"
        
        # Add comparison result
        comparison_result = CompletenessResult(
            metric_name="Pipeline Completeness Comparison",
            completeness_score=min(num_result.completeness_score, vis_result.completeness_score),
            total_expected=expected_total,
            total_processed=min(num_processed, vis_processed),
            missing_count=max(expected_total - num_processed, expected_total - vis_processed),
            failure_count=max(num_failed, vis_failed),
            interpretation=comparison_interpretation,
            metadata={
                'numerical_completeness': num_result.completeness_score,
                'visual_completeness': vis_result.completeness_score,
                'completeness_difference': completeness_diff
            }
        )
        
        results.append(comparison_result)
        self.results.append(comparison_result)
        
        return results
    
    def analyze_missing_data_patterns(
        self,
        missing_data_info: Dict[str, List[Any]],
        pipeline_name: str = "Unknown"
    ) -> CompletenessResult:
        """
        Analyze patterns in missing data
        
        Args:
            missing_data_info: Dictionary with information about missing data
                              Format: {'mz_values': [...], 'intensities': [...], 'scan_times': [...]}
            pipeline_name: Name of the pipeline being assessed
            
        Returns:
            CompletenessResult object
        """
        total_missing = 0
        pattern_analysis = {}
        
        for data_type, missing_values in missing_data_info.items():
            missing_count = len(missing_values)
            total_missing += missing_count
            
            if missing_count > 0:
                # Analyze patterns
                if data_type == 'mz_values' and missing_values:
                    # Analyze m/z range patterns
                    mz_array = np.array(missing_values)
                    pattern_analysis[f'{data_type}_range'] = {
                        'min': np.min(mz_array),
                        'max': np.max(mz_array),
                        'mean': np.mean(mz_array),
                        'std': np.std(mz_array)
                    }
                elif data_type == 'intensities' and missing_values:
                    # Analyze intensity patterns
                    int_array = np.array(missing_values)
                    pattern_analysis[f'{data_type}_stats'] = {
                        'min': np.min(int_array),
                        'max': np.max(int_array),
                        'mean': np.mean(int_array),
                        'std': np.std(int_array)
                    }
        
        # Calculate overall missing data score (inverse of missing percentage)
        total_data_points = sum(len(values) for values in missing_data_info.values())
        missing_percentage = total_missing / total_data_points if total_data_points > 0 else 0
        completeness_score = 1.0 - missing_percentage
        
        # Interpretation
        if missing_percentage < 0.05:
            interpretation = f"Minimal missing data patterns ({missing_percentage:.1%} missing)"
        elif missing_percentage < 0.10:
            interpretation = f"Low missing data patterns ({missing_percentage:.1%} missing)"
        elif missing_percentage < 0.20:
            interpretation = f"Moderate missing data patterns ({missing_percentage:.1%} missing)"
        else:
            interpretation = f"High missing data patterns ({missing_percentage:.1%} missing)"
        
        result = CompletenessResult(
            metric_name=f"Missing Data Patterns ({pipeline_name})",
            completeness_score=completeness_score,
            total_expected=total_data_points,
            total_processed=total_data_points - total_missing,
            missing_count=total_missing,
            failure_count=total_missing,
            interpretation=interpretation,
            metadata={
                'pattern_analysis': pattern_analysis,
                'missing_percentage': missing_percentage,
                'pipeline_name': pipeline_name
            }
        )
        
        self.results.append(result)
        return result
    
    def assess_processing_failure_modes(
        self,
        failure_data: Dict[str, List[str]],
        pipeline_name: str = "Unknown"
    ) -> CompletenessResult:
        """
        Assess processing failure modes and their frequencies
        
        Args:
            failure_data: Dictionary with failure types and their occurrences
                         Format: {'error_type': ['error1', 'error2', ...]}
            pipeline_name: Name of the pipeline being assessed
            
        Returns:
            CompletenessResult object
        """
        total_failures = sum(len(errors) for errors in failure_data.values())
        failure_analysis = {}
        
        # Analyze failure patterns
        for failure_type, errors in failure_data.items():
            failure_count = len(errors)
            failure_percentage = failure_count / total_failures if total_failures > 0 else 0
            
            failure_analysis[failure_type] = {
                'count': failure_count,
                'percentage': failure_percentage,
                'examples': errors[:5]  # Store first 5 examples
            }
        
        # Find most common failure mode
        most_common_failure = max(failure_analysis.keys(), 
                                key=lambda x: failure_analysis[x]['count']) if failure_analysis else "None"
        
        # Calculate failure diversity (how many different types of failures)
        failure_diversity = len(failure_data)
        
        # Completeness score based on failure rate (assuming some baseline)
        # This is inverse - fewer failures = higher completeness
        max_expected_failures = 100  # Baseline assumption
        failure_rate = min(1.0, total_failures / max_expected_failures)
        completeness_score = 1.0 - failure_rate
        
        # Interpretation
        if total_failures == 0:
            interpretation = "No processing failures detected"
        elif total_failures < 10:
            interpretation = f"Low failure rate ({total_failures} failures, most common: {most_common_failure})"
        elif total_failures < 50:
            interpretation = f"Moderate failure rate ({total_failures} failures, most common: {most_common_failure})"
        else:
            interpretation = f"High failure rate ({total_failures} failures, most common: {most_common_failure})"
        
        result = CompletenessResult(
            metric_name=f"Processing Failure Analysis ({pipeline_name})",
            completeness_score=completeness_score,
            total_expected=max_expected_failures,
            total_processed=max_expected_failures - total_failures,
            missing_count=0,
            failure_count=total_failures,
            interpretation=interpretation,
            metadata={
                'failure_analysis': failure_analysis,
                'most_common_failure': most_common_failure,
                'failure_diversity': failure_diversity,
                'pipeline_name': pipeline_name
            }
        )
        
        self.results.append(result)
        return result
    
    def plot_completeness_analysis(self, save_path: Optional[str] = None) -> plt.Figure:
        """Create comprehensive completeness analysis visualization"""
        if not self.results:
            return None
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Completeness Analysis Results', fontsize=16, fontweight='bold')
        
        # Extract data
        metrics = [r.metric_name for r in self.results]
        scores = [r.completeness_score for r in self.results]
        total_expected = [r.total_expected for r in self.results]
        total_processed = [r.total_processed for r in self.results]
        missing_counts = [r.missing_count for r in self.results]
        
        # Plot 1: Completeness scores
        colors = ['green' if score > 0.9 else 'orange' if score > 0.8 else 'red' for score in scores]
        axes[0, 0].bar(range(len(metrics)), scores, color=colors, alpha=0.7)
        axes[0, 0].set_xticks(range(len(metrics)))
        axes[0, 0].set_xticklabels([m.split('(')[0].strip() for m in metrics], rotation=45, ha='right')
        axes[0, 0].set_ylabel('Completeness Score')
        axes[0, 0].set_title('Completeness Scores')
        axes[0, 0].set_ylim(0, 1)
        
        # Plot 2: Expected vs Processed
        x_pos = np.arange(len(metrics))
        width = 0.35
        
        axes[0, 1].bar(x_pos - width/2, total_expected, width, label='Expected', alpha=0.7)
        axes[0, 1].bar(x_pos + width/2, total_processed, width, label='Processed', alpha=0.7)
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels([m.split('(')[0].strip() for m in metrics], rotation=45, ha='right')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Expected vs Processed')
        axes[0, 1].legend()
        
        # Plot 3: Missing data distribution
        if missing_counts:
            axes[0, 2].pie(missing_counts, labels=[m.split('(')[0].strip() for m in metrics], 
                          autopct='%1.1f%%', startangle=90)
            axes[0, 2].set_title('Missing Data Distribution')
        
        # Plot 4: Completeness score distribution
        axes[1, 0].hist(scores, bins=10, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(x=np.mean(scores), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(scores):.3f}')
        axes[1, 0].set_xlabel('Completeness Score')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Score Distribution')
        axes[1, 0].legend()
        
        # Plot 5: Pipeline comparison (if available)
        pipeline_results = {}
        for result in self.results:
            pipeline_name = result.metadata.get('pipeline_name', 'Unknown') if result.metadata else 'Unknown'
            if pipeline_name not in pipeline_results:
                pipeline_results[pipeline_name] = []
            pipeline_results[pipeline_name].append(result.completeness_score)
        
        if len(pipeline_results) > 1:
            pipeline_names = list(pipeline_results.keys())
            pipeline_scores = [np.mean(scores) for scores in pipeline_results.values()]
            
            axes[1, 1].bar(pipeline_names, pipeline_scores, alpha=0.7)
            axes[1, 1].set_ylabel('Average Completeness Score')
            axes[1, 1].set_title('Pipeline Comparison')
            axes[1, 1].set_ylim(0, 1)
        
        # Plot 6: Summary statistical_analysis
        summary_text = f"""
Completeness Analysis Summary:

Total Metrics Assessed: {len(self.results)}
Average Completeness: {np.mean(scores):.1%}
Best Performance: {np.max(scores):.1%}
Worst Performance: {np.min(scores):.1%}

Total Expected: {sum(total_expected):,}
Total Processed: {sum(total_processed):,}
Total Missing: {sum(missing_counts):,}

Overall Success Rate: {sum(total_processed)/sum(total_expected):.1%}
        """
        
        axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[1, 2].set_xlim(0, 1)
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].axis('off')
        axes[1, 2].set_title('Summary Statistics')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_completeness_report(self) -> pd.DataFrame:
        """Generate comprehensive completeness assessment report"""
        if not self.results:
            return pd.DataFrame()
        
        data = []
        for result in self.results:
            data.append({
                'Metric': result.metric_name,
                'Completeness Score': result.completeness_score,
                'Total Expected': result.total_expected,
                'Total Processed': result.total_processed,
                'Missing Count': result.missing_count,
                'Failure Count': result.failure_count,
                'Interpretation': result.interpretation
            })
        
        return pd.DataFrame(data)
    
    def get_overall_completeness_score(self) -> float:
        """Calculate overall completeness score across all metrics"""
        if not self.results:
            return 0.0
        
        # Weight by total expected (larger datasets have more influence)
        total_weight = sum(r.total_expected for r in self.results)
        if total_weight == 0:
            return np.mean([r.completeness_score for r in self.results])
        
        weighted_sum = sum(r.completeness_score * r.total_expected for r in self.results)
        return weighted_sum / total_weight
    
    def clear_results(self) -> None:
        """Clear all stored results"""
        self.results = [] 