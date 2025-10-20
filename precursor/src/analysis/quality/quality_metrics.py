"""
Quality Metrics Module

Comprehensive quality metrics for data assessment including
accuracy, precision, reliability, and overall quality scoring.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import pearsonr, spearmanr


@dataclass
class QualityMetric:
    """Container for quality metric results"""
    metric_name: str
    value: float
    interpretation: str
    category: str  # 'accuracy', 'precision', 'reliability', 'completeness'
    metadata: Dict[str, Any] = None


class QualityMetrics:
    """
    Comprehensive quality metrics calculator
    """
    
    def __init__(self):
        """Initialize quality metrics calculator"""
        self.metrics = []
        
    def calculate_accuracy_metrics(
        self,
        true_values: np.ndarray,
        predicted_values: np.ndarray,
        task_type: str = "regression"
    ) -> List[QualityMetric]:
        """
        Calculate accuracy-related quality metrics
        
        Args:
            true_values: Ground truth values
            predicted_values: Predicted values
            task_type: Type of task ('regression' or 'classification')
            
        Returns:
            List of QualityMetric objects
        """
        metrics = []
        
        if task_type == "regression":
            # Mean Absolute Error
            mae = np.mean(np.abs(true_values - predicted_values))
            mae_metric = QualityMetric(
                metric_name="Mean Absolute Error",
                value=mae,
                interpretation=f"Average absolute error: {mae:.4f}",
                category="accuracy",
                metadata={'lower_is_better': True}
            )
            metrics.append(mae_metric)
            
            # Root Mean Square Error
            rmse = np.sqrt(np.mean((true_values - predicted_values) ** 2))
            rmse_metric = QualityMetric(
                metric_name="Root Mean Square Error",
                value=rmse,
                interpretation=f"RMS error: {rmse:.4f}",
                category="accuracy",
                metadata={'lower_is_better': True}
            )
            metrics.append(rmse_metric)
            
            # R-squared
            ss_res = np.sum((true_values - predicted_values) ** 2)
            ss_tot = np.sum((true_values - np.mean(true_values)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            r2_metric = QualityMetric(
                metric_name="R-squared",
                value=r2,
                interpretation=f"Variance explained: {r2:.3f} ({r2*100:.1f}%)",
                category="accuracy",
                metadata={'higher_is_better': True}
            )
            metrics.append(r2_metric)
            
            # Correlation coefficient
            corr, _ = pearsonr(true_values, predicted_values)
            corr = abs(corr) if not np.isnan(corr) else 0.0
            corr_metric = QualityMetric(
                metric_name="Correlation",
                value=corr,
                interpretation=f"Linear correlation: {corr:.3f}",
                category="accuracy",
                metadata={'higher_is_better': True}
            )
            metrics.append(corr_metric)
            
        elif task_type == "classification":
            # Convert to integer labels if needed
            if true_values.dtype != int:
                true_labels = np.round(true_values).astype(int)
                pred_labels = np.round(predicted_values).astype(int)
            else:
                true_labels = true_values
                pred_labels = predicted_values
            
            # Accuracy
            acc = accuracy_score(true_labels, pred_labels)
            acc_metric = QualityMetric(
                metric_name="Classification Accuracy",
                value=acc,
                interpretation=f"Correct predictions: {acc:.3f} ({acc*100:.1f}%)",
                category="accuracy",
                metadata={'higher_is_better': True}
            )
            metrics.append(acc_metric)
            
            # Precision (macro average)
            prec = precision_score(true_labels, pred_labels, average='macro', zero_division=0)
            prec_metric = QualityMetric(
                metric_name="Precision",
                value=prec,
                interpretation=f"Precision (macro): {prec:.3f}",
                category="precision",
                metadata={'higher_is_better': True}
            )
            metrics.append(prec_metric)
            
            # Recall (macro average)
            rec = recall_score(true_labels, pred_labels, average='macro', zero_division=0)
            rec_metric = QualityMetric(
                metric_name="Recall",
                value=rec,
                interpretation=f"Recall (macro): {rec:.3f}",
                category="precision",
                metadata={'higher_is_better': True}
            )
            metrics.append(rec_metric)
            
            # F1 Score (macro average)
            f1 = f1_score(true_labels, pred_labels, average='macro', zero_division=0)
            f1_metric = QualityMetric(
                metric_name="F1 Score",
                value=f1,
                interpretation=f"F1 score (macro): {f1:.3f}",
                category="precision",
                metadata={'higher_is_better': True}
            )
            metrics.append(f1_metric)
        
        self.metrics.extend(metrics)
        return metrics
    
    def calculate_precision_metrics(
        self,
        data: np.ndarray,
        reference_data: Optional[np.ndarray] = None
    ) -> List[QualityMetric]:
        """
        Calculate precision-related quality metrics
        
        Args:
            data: Data to assess
            reference_data: Reference data for comparison
            
        Returns:
            List of QualityMetric objects
        """
        metrics = []
        
        # Signal-to-Noise Ratio
        signal_power = np.mean(data ** 2)
        noise_power = np.var(data)
        snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
        snr_metric = QualityMetric(
            metric_name="Signal-to-Noise Ratio",
            value=snr,
            interpretation=f"SNR: {snr:.2f} dB",
            category="precision",
            metadata={'higher_is_better': True, 'unit': 'dB'}
        )
        metrics.append(snr_metric)
        
        # Coefficient of Variation
        cv = np.std(data) / np.mean(data) if np.mean(data) != 0 else float('inf')
        cv_metric = QualityMetric(
            metric_name="Coefficient of Variation",
            value=cv,
            interpretation=f"Relative variability: {cv:.3f}",
            category="precision",
            metadata={'lower_is_better': True}
        )
        metrics.append(cv_metric)
        
        # Dynamic Range
        dynamic_range = np.max(data) - np.min(data)
        dr_metric = QualityMetric(
            metric_name="Dynamic Range",
            value=dynamic_range,
            interpretation=f"Value range: {dynamic_range:.3f}",
            category="precision",
            metadata={'context_dependent': True}
        )
        metrics.append(dr_metric)
        
        # Precision relative to reference
        if reference_data is not None:
            relative_error = np.mean(np.abs(data - reference_data)) / np.mean(np.abs(reference_data))
            precision_score = max(0.0, 1.0 - relative_error)
            prec_metric = QualityMetric(
                metric_name="Relative Precision",
                value=precision_score,
                interpretation=f"Precision vs reference: {precision_score:.3f}",
                category="precision",
                metadata={'higher_is_better': True}
            )
            metrics.append(prec_metric)
        
        self.metrics.extend(metrics)
        return metrics
    
    def calculate_reliability_metrics(
        self,
        data: np.ndarray,
        repeated_measurements: Optional[List[np.ndarray]] = None
    ) -> List[QualityMetric]:
        """
        Calculate reliability-related quality metrics
        
        Args:
            data: Primary data
            repeated_measurements: List of repeated measurements for reliability assessment
            
        Returns:
            List of QualityMetric objects
        """
        metrics = []
        
        # Internal consistency (Cronbach's alpha approximation)
        if data.ndim > 1 and data.shape[1] > 1:
            # Split data into halves
            n_items = data.shape[1]
            half1 = data[:, :n_items//2]
            half2 = data[:, n_items//2:]
            
            # Calculate correlation between halves
            sum1 = np.sum(half1, axis=1)
            sum2 = np.sum(half2, axis=1)
            
            if len(sum1) > 1:
                corr, _ = pearsonr(sum1, sum2)
                alpha = 2 * corr / (1 + corr) if not np.isnan(corr) and corr > 0 else 0.0
                
                alpha_metric = QualityMetric(
                    metric_name="Internal Consistency",
                    value=alpha,
                    interpretation=f"Cronbach's alpha: {alpha:.3f}",
                    category="reliability",
                    metadata={'higher_is_better': True}
                )
                metrics.append(alpha_metric)
        
        # Test-retest reliability
        if repeated_measurements is not None and len(repeated_measurements) > 1:
            correlations = []
            for i in range(len(repeated_measurements) - 1):
                for j in range(i + 1, len(repeated_measurements)):
                    corr, _ = pearsonr(repeated_measurements[i].flatten(), 
                                     repeated_measurements[j].flatten())
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
            
            if correlations:
                avg_reliability = np.mean(correlations)
                rel_metric = QualityMetric(
                    metric_name="Test-Retest Reliability",
                    value=avg_reliability,
                    interpretation=f"Average correlation: {avg_reliability:.3f}",
                    category="reliability",
                    metadata={'higher_is_better': True}
                )
                metrics.append(rel_metric)
        
        # Stability metric (based on variance)
        stability = 1.0 / (1.0 + np.var(data))
        stab_metric = QualityMetric(
            metric_name="Stability",
            value=stability,
            interpretation=f"Data stability: {stability:.3f}",
            category="reliability",
            metadata={'higher_is_better': True}
        )
        metrics.append(stab_metric)
        
        self.metrics.extend(metrics)
        return metrics
    
    def calculate_completeness_metrics(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        expected_size: Optional[int] = None
    ) -> List[QualityMetric]:
        """
        Calculate completeness-related quality metrics
        
        Args:
            data: Data to assess
            expected_size: Expected data size
            
        Returns:
            List of QualityMetric objects
        """
        metrics = []
        
        if isinstance(data, np.ndarray):
            # Missing value rate
            total_elements = data.size
            missing_elements = np.sum(np.isnan(data))
            completeness_rate = 1.0 - (missing_elements / total_elements)
            
            comp_metric = QualityMetric(
                metric_name="Data Completeness",
                value=completeness_rate,
                interpretation=f"Complete data: {completeness_rate:.3f} ({completeness_rate*100:.1f}%)",
                category="completeness",
                metadata={'higher_is_better': True}
            )
            metrics.append(comp_metric)
            
        elif isinstance(data, pd.DataFrame):
            # Column completeness
            total_cells = data.size
            missing_cells = data.isnull().sum().sum()
            completeness_rate = 1.0 - (missing_cells / total_cells)
            
            comp_metric = QualityMetric(
                metric_name="Data Completeness",
                value=completeness_rate,
                interpretation=f"Complete data: {completeness_rate:.3f} ({completeness_rate*100:.1f}%)",
                category="completeness",
                metadata={'higher_is_better': True}
            )
            metrics.append(comp_metric)
            
            # Column-wise completeness
            col_completeness = 1.0 - (data.isnull().sum() / len(data))
            min_col_completeness = col_completeness.min()
            
            min_comp_metric = QualityMetric(
                metric_name="Minimum Column Completeness",
                value=min_col_completeness,
                interpretation=f"Worst column: {min_col_completeness:.3f}",
                category="completeness",
                metadata={'higher_is_better': True}
            )
            metrics.append(min_comp_metric)
        
        # Size completeness
        if expected_size is not None:
            actual_size = len(data) if hasattr(data, '__len__') else data.size
            size_completeness = min(1.0, actual_size / expected_size)
            
            size_metric = QualityMetric(
                metric_name="Size Completeness",
                value=size_completeness,
                interpretation=f"Size ratio: {size_completeness:.3f}",
                category="completeness",
                metadata={'higher_is_better': True}
            )
            metrics.append(size_metric)
        
        self.metrics.extend(metrics)
        return metrics
    
    def calculate_overall_quality_score(
        self,
        weights: Optional[Dict[str, float]] = None
    ) -> QualityMetric:
        """
        Calculate overall quality score from all metrics
        
        Args:
            weights: Weights for different metric categories
            
        Returns:
            QualityMetric object with overall score
        """
        if not self.metrics:
            return QualityMetric(
                metric_name="Overall Quality",
                value=0.0,
                interpretation="No metrics available",
                category="overall"
            )
        
        # Default weights
        if weights is None:
            weights = {
                'accuracy': 0.3,
                'precision': 0.25,
                'reliability': 0.25,
                'completeness': 0.2
            }
        
        # Group metrics by category
        category_scores = {}
        for metric in self.metrics:
            category = metric.category
            if category not in category_scores:
                category_scores[category] = []
            
            # Normalize score based on whether higher or lower is better
            score = metric.value
            if metric.metadata and metric.metadata.get('lower_is_better', False):
                # For metrics where lower is better, invert the score
                # Assume reasonable upper bound for normalization
                if metric.metric_name in ["Mean Absolute Error", "Root Mean Square Error"]:
                    score = max(0.0, 1.0 - min(1.0, score))
                elif metric.metric_name == "Coefficient of Variation":
                    score = max(0.0, 1.0 - min(1.0, score))
            
            category_scores[category].append(score)
        
        # Calculate weighted average
        total_score = 0.0
        total_weight = 0.0
        
        for category, scores in category_scores.items():
            if category in weights:
                avg_score = np.mean(scores)
                weight = weights[category]
                total_score += avg_score * weight
                total_weight += weight
        
        overall_score = total_score / total_weight if total_weight > 0 else 0.0
        
        # Interpretation
        if overall_score > 0.9:
            interpretation = f"Excellent quality ({overall_score:.3f})"
        elif overall_score > 0.8:
            interpretation = f"Good quality ({overall_score:.3f})"
        elif overall_score > 0.6:
            interpretation = f"Moderate quality ({overall_score:.3f})"
        elif overall_score > 0.4:
            interpretation = f"Poor quality ({overall_score:.3f})"
        else:
            interpretation = f"Very poor quality ({overall_score:.3f})"
        
        overall_metric = QualityMetric(
            metric_name="Overall Quality Score",
            value=overall_score,
            interpretation=interpretation,
            category="overall",
            metadata={
                'weights': weights,
                'category_scores': {cat: np.mean(scores) for cat, scores in category_scores.items()}
            }
        )
        
        self.metrics.append(overall_metric)
        return overall_metric
    
    def compare_quality_metrics(
        self,
        other_metrics: 'QualityMetrics'
    ) -> pd.DataFrame:
        """
        Compare quality metrics with another QualityMetrics instance
        
        Args:
            other_metrics: Another QualityMetrics instance to compare with
            
        Returns:
            DataFrame with comparison results
        """
        comparison_data = []
        
        # Create dictionaries for easy lookup
        self_dict = {m.metric_name: m for m in self.metrics}
        other_dict = {m.metric_name: m for m in other_metrics.metrics}
        
        # Find common metrics
        common_metrics = set(self_dict.keys()) & set(other_dict.keys())
        
        for metric_name in common_metrics:
            self_metric = self_dict[metric_name]
            other_metric = other_dict[metric_name]
            
            difference = self_metric.value - other_metric.value
            
            # Determine which is better
            higher_is_better = self_metric.metadata and self_metric.metadata.get('higher_is_better', True)
            if higher_is_better:
                better = "Self" if difference > 0 else "Other" if difference < 0 else "Equal"
            else:
                better = "Other" if difference > 0 else "Self" if difference < 0 else "Equal"
            
            comparison_data.append({
                'Metric': metric_name,
                'Self': self_metric.value,
                'Other': other_metric.value,
                'Difference': difference,
                'Better': better,
                'Category': self_metric.category
            })
        
        return pd.DataFrame(comparison_data)
    
    def generate_quality_report(self) -> pd.DataFrame:
        """Generate comprehensive quality report"""
        if not self.metrics:
            return pd.DataFrame()
        
        data = []
        for metric in self.metrics:
            data.append({
                'Metric': metric.metric_name,
                'Value': metric.value,
                'Category': metric.category,
                'Interpretation': metric.interpretation
            })
        
        return pd.DataFrame(data)
    
    def get_metrics_by_category(self, category: str) -> List[QualityMetric]:
        """Get all metrics for a specific category"""
        return [m for m in self.metrics if m.category == category]
    
    def clear_metrics(self) -> None:
        """Clear all stored metrics"""
        self.metrics = [] 