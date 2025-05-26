"""
Data Quality Assessment Module

Comprehensive assessment of data quality for pipeline outputs including
completeness, consistency, accuracy, and validity checks.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings


@dataclass
class QualityAssessmentResult:
    """Container for quality assessment results"""
    metric_name: str
    score: float  # 0-1 quality score
    threshold: float
    passed: bool
    details: Dict[str, Any]
    interpretation: str


class DataQualityAssessor:
    """
    Comprehensive data quality assessment for pipeline outputs
    """
    
    def __init__(self, quality_thresholds: Optional[Dict[str, float]] = None):
        """
        Initialize data quality assessor
        
        Args:
            quality_thresholds: Custom quality thresholds for different metrics
        """
        self.quality_thresholds = quality_thresholds or {
            'completeness': 0.95,
            'consistency': 0.90,
            'accuracy': 0.85,
            'validity': 0.90,
            'uniqueness': 0.95,
            'timeliness': 0.90
        }
        self.results = []
    
    def assess_completeness(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        expected_size: Optional[int] = None
    ) -> QualityAssessmentResult:
        """
        Assess data completeness (missing values, expected size)
        
        Args:
            data: Input data to assess
            expected_size: Expected number of data points
            
        Returns:
            QualityAssessmentResult object
        """
        if isinstance(data, pd.DataFrame):
            total_cells = data.size
            missing_cells = data.isnull().sum().sum()
            completeness_score = 1.0 - (missing_cells / total_cells) if total_cells > 0 else 0.0
            
            details = {
                'total_cells': total_cells,
                'missing_cells': missing_cells,
                'missing_percentage': (missing_cells / total_cells) * 100 if total_cells > 0 else 0,
                'columns_with_missing': data.columns[data.isnull().any()].tolist()
            }
        else:
            # Handle numpy arrays
            data_flat = data.flatten()
            total_values = len(data_flat)
            missing_values = np.sum(np.isnan(data_flat))
            completeness_score = 1.0 - (missing_values / total_values) if total_values > 0 else 0.0
            
            details = {
                'total_values': total_values,
                'missing_values': missing_values,
                'missing_percentage': (missing_values / total_values) * 100 if total_values > 0 else 0
            }
        
        # Check expected size if provided
        if expected_size is not None:
            actual_size = len(data)
            size_completeness = min(1.0, actual_size / expected_size)
            completeness_score = min(completeness_score, size_completeness)
            details['expected_size'] = expected_size
            details['actual_size'] = actual_size
            details['size_completeness'] = size_completeness
        
        threshold = self.quality_thresholds['completeness']
        passed = completeness_score >= threshold
        
        if passed:
            interpretation = f"Excellent data completeness ({completeness_score:.1%})"
        else:
            interpretation = f"Poor data completeness ({completeness_score:.1%}), below threshold ({threshold:.1%})"
        
        result = QualityAssessmentResult(
            metric_name="Data Completeness",
            score=completeness_score,
            threshold=threshold,
            passed=passed,
            details=details,
            interpretation=interpretation
        )
        
        self.results.append(result)
        return result
    
    def assess_consistency(
        self,
        numerical_data: np.ndarray,
        visual_data: np.ndarray
    ) -> QualityAssessmentResult:
        """
        Assess consistency between numerical and visual pipeline outputs
        
        Args:
            numerical_data: Output from numerical pipeline
            visual_data: Output from visual pipeline
            
        Returns:
            QualityAssessmentResult object
        """
        # Ensure arrays are the same shape for comparison
        min_length = min(len(numerical_data), len(visual_data))
        numerical_data = numerical_data[:min_length]
        visual_data = visual_data[:min_length]
        
        # Calculate correlation as consistency measure
        correlation, p_value = stats.pearsonr(numerical_data.flatten(), visual_data.flatten())
        
        # Handle NaN correlation
        if np.isnan(correlation):
            correlation = 0.0
            p_value = 1.0
        
        consistency_score = max(0.0, correlation)
        
        # Calculate additional consistency metrics
        mse = np.mean((numerical_data - visual_data)**2)
        mae = np.mean(np.abs(numerical_data - visual_data))
        
        # Relative error
        numerical_mean = np.mean(np.abs(numerical_data))
        relative_error = mae / numerical_mean if numerical_mean > 0 else float('inf')
        
        details = {
            'correlation': correlation,
            'p_value': p_value,
            'mse': mse,
            'mae': mae,
            'relative_error': relative_error,
            'data_points_compared': min_length
        }
        
        threshold = self.quality_thresholds['consistency']
        passed = consistency_score >= threshold
        
        if passed:
            interpretation = f"Good consistency between pipelines (r={correlation:.3f})"
        else:
            interpretation = f"Poor consistency between pipelines (r={correlation:.3f}), below threshold ({threshold:.3f})"
        
        result = QualityAssessmentResult(
            metric_name="Pipeline Consistency",
            score=consistency_score,
            threshold=threshold,
            passed=passed,
            details=details,
            interpretation=interpretation
        )
        
        self.results.append(result)
        return result
    
    def assess_accuracy(
        self,
        predicted_values: np.ndarray,
        true_values: np.ndarray
    ) -> QualityAssessmentResult:
        """
        Assess accuracy of predictions against ground truth
        
        Args:
            predicted_values: Predicted values from pipeline
            true_values: Ground truth values
            
        Returns:
            QualityAssessmentResult object
        """
        # Ensure arrays are the same length
        min_length = min(len(predicted_values), len(true_values))
        predicted_values = predicted_values[:min_length]
        true_values = true_values[:min_length]
        
        # Calculate accuracy metrics
        mse = np.mean((predicted_values - true_values)**2)
        mae = np.mean(np.abs(predicted_values - true_values))
        rmse = np.sqrt(mse)
        
        # R-squared
        ss_res = np.sum((true_values - predicted_values)**2)
        ss_tot = np.sum((true_values - np.mean(true_values))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Accuracy score (1 - normalized RMSE)
        true_range = np.max(true_values) - np.min(true_values)
        normalized_rmse = rmse / true_range if true_range > 0 else 0
        accuracy_score = max(0.0, 1.0 - normalized_rmse)
        
        details = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r_squared': r_squared,
            'normalized_rmse': normalized_rmse,
            'data_points': min_length
        }
        
        threshold = self.quality_thresholds['accuracy']
        passed = accuracy_score >= threshold
        
        if passed:
            interpretation = f"Good prediction accuracy (score={accuracy_score:.3f}, R²={r_squared:.3f})"
        else:
            interpretation = f"Poor prediction accuracy (score={accuracy_score:.3f}, R²={r_squared:.3f})"
        
        result = QualityAssessmentResult(
            metric_name="Prediction Accuracy",
            score=accuracy_score,
            threshold=threshold,
            passed=passed,
            details=details,
            interpretation=interpretation
        )
        
        self.results.append(result)
        return result
    
    def assess_validity(
        self,
        data: np.ndarray,
        valid_range: Optional[Tuple[float, float]] = None,
        expected_distribution: Optional[str] = None
    ) -> QualityAssessmentResult:
        """
        Assess data validity (range checks, distribution checks)
        
        Args:
            data: Data to validate
            valid_range: Expected valid range (min, max)
            expected_distribution: Expected distribution type ('normal', 'uniform', etc.)
            
        Returns:
            QualityAssessmentResult object
        """
        data_flat = data.flatten()
        total_points = len(data_flat)
        
        validity_score = 1.0
        details = {'total_points': total_points}
        
        # Range validation
        if valid_range is not None:
            min_val, max_val = valid_range
            valid_points = np.sum((data_flat >= min_val) & (data_flat <= max_val))
            range_validity = valid_points / total_points if total_points > 0 else 0
            validity_score = min(validity_score, range_validity)
            
            details.update({
                'valid_range': valid_range,
                'points_in_range': valid_points,
                'range_validity': range_validity,
                'data_min': np.min(data_flat),
                'data_max': np.max(data_flat)
            })
        
        # Distribution validation
        if expected_distribution is not None:
            if expected_distribution.lower() == 'normal':
                # Shapiro-Wilk test for normality
                if len(data_flat) <= 5000:  # Shapiro-Wilk limitation
                    _, p_value = stats.shapiro(data_flat)
                    distribution_validity = 1.0 if p_value > 0.05 else 0.5
                else:
                    # Use Kolmogorov-Smirnov test for larger samples
                    _, p_value = stats.kstest(data_flat, 'norm')
                    distribution_validity = 1.0 if p_value > 0.05 else 0.5
                
                validity_score = min(validity_score, distribution_validity)
                details.update({
                    'expected_distribution': expected_distribution,
                    'normality_p_value': p_value,
                    'distribution_validity': distribution_validity
                })
        
        # Check for infinite or NaN values
        finite_points = np.sum(np.isfinite(data_flat))
        finite_validity = finite_points / total_points if total_points > 0 else 0
        validity_score = min(validity_score, finite_validity)
        
        details.update({
            'finite_points': finite_points,
            'infinite_points': total_points - finite_points,
            'finite_validity': finite_validity
        })
        
        threshold = self.quality_thresholds['validity']
        passed = validity_score >= threshold
        
        if passed:
            interpretation = f"Data meets validity requirements (score={validity_score:.3f})"
        else:
            interpretation = f"Data fails validity checks (score={validity_score:.3f})"
        
        result = QualityAssessmentResult(
            metric_name="Data Validity",
            score=validity_score,
            threshold=threshold,
            passed=passed,
            details=details,
            interpretation=interpretation
        )
        
        self.results.append(result)
        return result
    
    def assess_uniqueness(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        identifier_columns: Optional[List[str]] = None
    ) -> QualityAssessmentResult:
        """
        Assess data uniqueness (duplicate detection)
        
        Args:
            data: Data to assess
            identifier_columns: Columns that should be unique (for DataFrames)
            
        Returns:
            QualityAssessmentResult object
        """
        if isinstance(data, pd.DataFrame):
            if identifier_columns:
                # Check uniqueness of specified columns
                total_rows = len(data)
                unique_rows = len(data.drop_duplicates(subset=identifier_columns))
                uniqueness_score = unique_rows / total_rows if total_rows > 0 else 0
                
                details = {
                    'total_rows': total_rows,
                    'unique_rows': unique_rows,
                    'duplicate_rows': total_rows - unique_rows,
                    'identifier_columns': identifier_columns
                }
            else:
                # Check overall row uniqueness
                total_rows = len(data)
                unique_rows = len(data.drop_duplicates())
                uniqueness_score = unique_rows / total_rows if total_rows > 0 else 0
                
                details = {
                    'total_rows': total_rows,
                    'unique_rows': unique_rows,
                    'duplicate_rows': total_rows - unique_rows
                }
        else:
            # Handle numpy arrays
            data_flat = data.flatten()
            total_values = len(data_flat)
            unique_values = len(np.unique(data_flat))
            uniqueness_score = unique_values / total_values if total_values > 0 else 0
            
            details = {
                'total_values': total_values,
                'unique_values': unique_values,
                'duplicate_values': total_values - unique_values
            }
        
        threshold = self.quality_thresholds['uniqueness']
        passed = uniqueness_score >= threshold
        
        if passed:
            interpretation = f"Good data uniqueness ({uniqueness_score:.1%})"
        else:
            interpretation = f"Poor data uniqueness ({uniqueness_score:.1%}), many duplicates detected"
        
        result = QualityAssessmentResult(
            metric_name="Data Uniqueness",
            score=uniqueness_score,
            threshold=threshold,
            passed=passed,
            details=details,
            interpretation=interpretation
        )
        
        self.results.append(result)
        return result
    
    def assess_timeliness(
        self,
        data_timestamps: np.ndarray,
        expected_frequency: Optional[str] = None,
        max_delay: Optional[float] = None
    ) -> QualityAssessmentResult:
        """
        Assess data timeliness (for time-series data)
        
        Args:
            data_timestamps: Array of timestamps
            expected_frequency: Expected data frequency ('1min', '1hour', etc.)
            max_delay: Maximum acceptable delay in seconds
            
        Returns:
            QualityAssessmentResult object
        """
        if len(data_timestamps) < 2:
            # Not enough data for timeliness assessment
            result = QualityAssessmentResult(
                metric_name="Data Timeliness",
                score=1.0,
                threshold=self.quality_thresholds['timeliness'],
                passed=True,
                details={'insufficient_data': True},
                interpretation="Insufficient data for timeliness assessment"
            )
            self.results.append(result)
            return result
        
        # Calculate time intervals
        time_diffs = np.diff(data_timestamps)
        mean_interval = np.mean(time_diffs)
        std_interval = np.std(time_diffs)
        
        # Assess regularity (coefficient of variation)
        cv = std_interval / mean_interval if mean_interval > 0 else float('inf')
        regularity_score = max(0.0, 1.0 - cv)  # Lower CV is better
        
        timeliness_score = regularity_score
        
        details = {
            'mean_interval': mean_interval,
            'std_interval': std_interval,
            'coefficient_of_variation': cv,
            'min_interval': np.min(time_diffs),
            'max_interval': np.max(time_diffs),
            'total_timepoints': len(data_timestamps)
        }
        
        # Check against expected frequency if provided
        if expected_frequency is not None:
            # Convert frequency to seconds (simplified)
            freq_map = {
                '1min': 60, '5min': 300, '15min': 900, '30min': 1800,
                '1hour': 3600, '1day': 86400
            }
            expected_interval = freq_map.get(expected_frequency, mean_interval)
            
            interval_deviation = abs(mean_interval - expected_interval) / expected_interval
            frequency_score = max(0.0, 1.0 - interval_deviation)
            timeliness_score = min(timeliness_score, frequency_score)
            
            details.update({
                'expected_frequency': expected_frequency,
                'expected_interval': expected_interval,
                'frequency_score': frequency_score
            })
        
        # Check maximum delay if provided
        if max_delay is not None:
            delay_violations = np.sum(time_diffs > max_delay)
            delay_score = 1.0 - (delay_violations / len(time_diffs))
            timeliness_score = min(timeliness_score, delay_score)
            
            details.update({
                'max_delay': max_delay,
                'delay_violations': delay_violations,
                'delay_score': delay_score
            })
        
        threshold = self.quality_thresholds['timeliness']
        passed = timeliness_score >= threshold
        
        if passed:
            interpretation = f"Good data timeliness (score={timeliness_score:.3f})"
        else:
            interpretation = f"Poor data timeliness (score={timeliness_score:.3f}), irregular intervals"
        
        result = QualityAssessmentResult(
            metric_name="Data Timeliness",
            score=timeliness_score,
            threshold=threshold,
            passed=passed,
            details=details,
            interpretation=interpretation
        )
        
        self.results.append(result)
        return result
    
    def comprehensive_assessment(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        reference_data: Optional[np.ndarray] = None,
        ground_truth: Optional[np.ndarray] = None,
        **kwargs
    ) -> List[QualityAssessmentResult]:
        """
        Perform comprehensive quality assessment
        
        Args:
            data: Primary data to assess
            reference_data: Reference data for consistency checks
            ground_truth: Ground truth for accuracy assessment
            **kwargs: Additional parameters for specific assessments
            
        Returns:
            List of QualityAssessmentResult objects
        """
        results = []
        
        # Completeness assessment
        results.append(self.assess_completeness(data, kwargs.get('expected_size')))
        
        # Consistency assessment (if reference data provided)
        if reference_data is not None:
            data_array = data if isinstance(data, np.ndarray) else data.values
            results.append(self.assess_consistency(data_array, reference_data))
        
        # Accuracy assessment (if ground truth provided)
        if ground_truth is not None:
            data_array = data if isinstance(data, np.ndarray) else data.values
            results.append(self.assess_accuracy(data_array, ground_truth))
        
        # Validity assessment
        data_array = data if isinstance(data, np.ndarray) else data.values
        results.append(self.assess_validity(
            data_array,
            kwargs.get('valid_range'),
            kwargs.get('expected_distribution')
        ))
        
        # Uniqueness assessment
        results.append(self.assess_uniqueness(data, kwargs.get('identifier_columns')))
        
        # Timeliness assessment (if timestamps provided)
        if 'timestamps' in kwargs:
            results.append(self.assess_timeliness(
                kwargs['timestamps'],
                kwargs.get('expected_frequency'),
                kwargs.get('max_delay')
            ))
        
        return results
    
    def generate_quality_report(self) -> pd.DataFrame:
        """Generate comprehensive quality assessment report"""
        if not self.results:
            return pd.DataFrame()
        
        data = []
        for result in self.results:
            data.append({
                'Quality Metric': result.metric_name,
                'Score': result.score,
                'Threshold': result.threshold,
                'Passed': result.passed,
                'Interpretation': result.interpretation
            })
        
        return pd.DataFrame(data)
    
    def plot_quality_assessment(self, save_path: Optional[str] = None) -> plt.Figure:
        """Create visualization of quality assessment results"""
        if not self.results:
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Data Quality Assessment Results', fontsize=16, fontweight='bold')
        
        # Extract data
        metrics = [result.metric_name for result in self.results]
        scores = [result.score for result in self.results]
        thresholds = [result.threshold for result in self.results]
        passed = [result.passed for result in self.results]
        
        # Plot 1: Quality scores vs thresholds
        colors = ['green' if p else 'red' for p in passed]
        x_pos = np.arange(len(metrics))
        
        axes[0, 0].bar(x_pos, scores, color=colors, alpha=0.7, label='Actual Score')
        axes[0, 0].bar(x_pos, thresholds, color='gray', alpha=0.3, label='Threshold')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels([m.replace(' ', '\n') for m in metrics], rotation=0, ha='center')
        axes[0, 0].set_ylabel('Quality Score')
        axes[0, 0].set_title('Quality Scores vs Thresholds')
        axes[0, 0].legend()
        axes[0, 0].set_ylim(0, 1.1)
        
        # Plot 2: Pass/Fail summary
        pass_count = sum(passed)
        fail_count = len(passed) - pass_count
        
        labels = ['Passed', 'Failed']
        sizes = [pass_count, fail_count]
        colors_pie = ['lightgreen', 'lightcoral']
        
        axes[0, 1].pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
        axes[0, 1].set_title('Quality Assessment Summary')
        
        # Plot 3: Quality score distribution
        axes[1, 0].hist(scores, bins=10, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(x=np.mean(scores), color='red', linestyle='--', label=f'Mean: {np.mean(scores):.3f}')
        axes[1, 0].set_xlabel('Quality Score')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Quality Score Distribution')
        axes[1, 0].legend()
        
        # Plot 4: Detailed scores
        axes[1, 1].scatter(range(len(scores)), scores, c=colors, s=100, alpha=0.7)
        for i, (score, threshold) in enumerate(zip(scores, thresholds)):
            axes[1, 1].plot([i, i], [threshold, score], 'k-', alpha=0.3)
        
        axes[1, 1].set_xticks(range(len(metrics)))
        axes[1, 1].set_xticklabels([m.split()[0] for m in metrics], rotation=45, ha='right')
        axes[1, 1].set_ylabel('Quality Score')
        axes[1, 1].set_title('Individual Quality Scores')
        axes[1, 1].set_ylim(0, 1.1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def get_overall_quality_score(self) -> float:
        """Calculate overall quality score as weighted average"""
        if not self.results:
            return 0.0
        
        # Weight by importance (can be customized)
        weights = {
            'Data Completeness': 0.25,
            'Pipeline Consistency': 0.20,
            'Prediction Accuracy': 0.25,
            'Data Validity': 0.15,
            'Data Uniqueness': 0.10,
            'Data Timeliness': 0.05
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for result in self.results:
            weight = weights.get(result.metric_name, 0.1)  # Default weight
            weighted_sum += result.score * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def clear_results(self) -> None:
        """Clear all stored results"""
        self.results = [] 