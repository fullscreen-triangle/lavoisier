"""
Compound Identification Validation Module

Evaluates compound identification accuracy using confusion matrices,
ROC curves, precision-recall analysis, and sensitivity assessment.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    classification_report, accuracy_score, f1_score
)


@dataclass
class IdentificationResult:
    """Container for compound identification results"""
    metric_name: str
    value: float
    confidence_interval: Optional[Tuple[float, float]]
    interpretation: str
    metadata: Dict = None


class CompoundIdentificationValidator:
    """
    Comprehensive compound identification accuracy assessment
    """
    
    def __init__(self):
        """Initialize compound identification validator"""
        self.results = []
        
    def evaluate_identification_accuracy(
        self,
        true_labels: np.ndarray,
        predicted_labels: np.ndarray,
        prediction_scores: Optional[np.ndarray] = None
    ) -> List[IdentificationResult]:
        """
        Evaluate compound identification accuracy
        
        Args:
            true_labels: Ground truth compound labels
            predicted_labels: Predicted compound labels
            prediction_scores: Optional prediction confidence scores
            
        Returns:
            List of IdentificationResult objects
        """
        results = []
        
        # Basic accuracy metrics
        accuracy = accuracy_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels, average='weighted')
        
        results.append(IdentificationResult(
            metric_name="Overall Accuracy",
            value=accuracy,
            confidence_interval=self._calculate_accuracy_ci(accuracy, len(true_labels)),
            interpretation=f"{'Excellent' if accuracy > 0.9 else 'Good' if accuracy > 0.8 else 'Moderate' if accuracy > 0.7 else 'Poor'} identification accuracy"
        ))
        
        results.append(IdentificationResult(
            metric_name="F1 Score",
            value=f1,
            confidence_interval=None,
            interpretation=f"{'Excellent' if f1 > 0.9 else 'Good' if f1 > 0.8 else 'Moderate' if f1 > 0.7 else 'Poor'} F1 performance"
        ))
        
        # Confusion matrix analysis
        cm = confusion_matrix(true_labels, predicted_labels)
        
        # Calculate per-class metrics
        if len(np.unique(true_labels)) > 2:  # Multi-class
            report = classification_report(true_labels, predicted_labels, output_dict=True)
            
            # Average precision and recall
            avg_precision = report['weighted avg']['precision']
            avg_recall = report['weighted avg']['recall']
            
            results.append(IdentificationResult(
                metric_name="Average Precision",
                value=avg_precision,
                confidence_interval=None,
                interpretation=f"{'High' if avg_precision > 0.8 else 'Moderate' if avg_precision > 0.6 else 'Low'} precision across compound classes"
            ))
            
            results.append(IdentificationResult(
                metric_name="Average Recall",
                value=avg_recall,
                confidence_interval=None,
                interpretation=f"{'High' if avg_recall > 0.8 else 'Moderate' if avg_recall > 0.6 else 'Low'} recall across compound classes"
            ))
        
        # ROC analysis (if scores provided and binary classification)
        if prediction_scores is not None and len(np.unique(true_labels)) == 2:
            fpr, tpr, _ = roc_curve(true_labels, prediction_scores)
            roc_auc = auc(fpr, tpr)
            
            results.append(IdentificationResult(
                metric_name="ROC AUC",
                value=roc_auc,
                confidence_interval=None,
                interpretation=f"{'Excellent' if roc_auc > 0.9 else 'Good' if roc_auc > 0.8 else 'Fair' if roc_auc > 0.7 else 'Poor'} discriminative ability",
                metadata={'fpr': fpr, 'tpr': tpr}
            ))
        
        # Precision-Recall analysis (if scores provided)
        if prediction_scores is not None:
            if len(np.unique(true_labels)) == 2:
                precision, recall, _ = precision_recall_curve(true_labels, prediction_scores)
                pr_auc = auc(recall, precision)
                
                results.append(IdentificationResult(
                    metric_name="PR AUC",
                    value=pr_auc,
                    confidence_interval=None,
                    interpretation=f"{'Excellent' if pr_auc > 0.9 else 'Good' if pr_auc > 0.8 else 'Fair' if pr_auc > 0.7 else 'Poor'} precision-recall performance",
                    metadata={'precision': precision, 'recall': recall}
                ))
        
        self.results.extend(results)
        return results
    
    def compare_pipeline_identification(
        self,
        true_labels: np.ndarray,
        numerical_predictions: np.ndarray,
        visual_predictions: np.ndarray,
        numerical_scores: Optional[np.ndarray] = None,
        visual_scores: Optional[np.ndarray] = None
    ) -> Dict[str, List[IdentificationResult]]:
        """
        Compare identification performance between pipelines
        
        Args:
            true_labels: Ground truth labels
            numerical_predictions: Numerical pipeline predictions
            visual_predictions: Visual pipeline predictions
            numerical_scores: Optional numerical pipeline scores
            visual_scores: Optional visual pipeline scores
            
        Returns:
            Dictionary with results for each pipeline
        """
        # Evaluate numerical pipeline
        numerical_results = self.evaluate_identification_accuracy(
            true_labels, numerical_predictions, numerical_scores
        )
        
        # Evaluate visual pipeline
        visual_results = self.evaluate_identification_accuracy(
            true_labels, visual_predictions, visual_scores
        )
        
        return {
            'numerical': numerical_results,
            'visual': visual_results
        }
    
    def analyze_false_discovery_rate(
        self,
        true_labels: np.ndarray,
        predicted_labels: np.ndarray,
        confidence_threshold: float = 0.5
    ) -> IdentificationResult:
        """
        Analyze false discovery rate
        
        Args:
            true_labels: Ground truth labels
            predicted_labels: Predicted labels
            confidence_threshold: Threshold for positive predictions
            
        Returns:
            IdentificationResult object
        """
        # Calculate confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels)
        
        if cm.shape == (2, 2):  # Binary classification
            tn, fp, fn, tp = cm.ravel()
            fdr = fp / (fp + tp) if (fp + tp) > 0 else 0.0
        else:  # Multi-class - calculate average FDR
            fdr_per_class = []
            for i in range(cm.shape[0]):
                tp = cm[i, i]
                fp = np.sum(cm[:, i]) - tp
                class_fdr = fp / (fp + tp) if (fp + tp) > 0 else 0.0
                fdr_per_class.append(class_fdr)
            fdr = np.mean(fdr_per_class)
        
        result = IdentificationResult(
            metric_name="False Discovery Rate",
            value=fdr,
            confidence_interval=None,
            interpretation=f"{'Low' if fdr < 0.1 else 'Moderate' if fdr < 0.2 else 'High'} false discovery rate ({fdr:.1%})"
        )
        
        self.results.append(result)
        return result
    
    def sensitivity_analysis(
        self,
        true_labels: np.ndarray,
        prediction_scores: np.ndarray,
        thresholds: Optional[np.ndarray] = None
    ) -> IdentificationResult:
        """
        Perform sensitivity analysis across different thresholds
        
        Args:
            true_labels: Ground truth labels
            prediction_scores: Prediction confidence scores
            thresholds: Optional custom thresholds to test
            
        Returns:
            IdentificationResult object
        """
        if thresholds is None:
            thresholds = np.linspace(0.1, 0.9, 9)
        
        sensitivities = []
        specificities = []
        
        for threshold in thresholds:
            predicted_labels = (prediction_scores >= threshold).astype(int)
            
            if len(np.unique(true_labels)) == 2:  # Binary classification
                cm = confusion_matrix(true_labels, predicted_labels)
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                    
                    sensitivities.append(sensitivity)
                    specificities.append(specificity)
        
        # Find optimal threshold (Youden's index)
        if sensitivities and specificities:
            youden_scores = np.array(sensitivities) + np.array(specificities) - 1
            optimal_idx = np.argmax(youden_scores)
            optimal_threshold = thresholds[optimal_idx]
            optimal_sensitivity = sensitivities[optimal_idx]
            optimal_specificity = specificities[optimal_idx]
        else:
            optimal_threshold = 0.5
            optimal_sensitivity = 0.0
            optimal_specificity = 0.0
        
        result = IdentificationResult(
            metric_name="Optimal Sensitivity",
            value=optimal_sensitivity,
            confidence_interval=None,
            interpretation=f"Optimal threshold: {optimal_threshold:.3f}, Sensitivity: {optimal_sensitivity:.3f}, Specificity: {optimal_specificity:.3f}",
            metadata={
                'optimal_threshold': optimal_threshold,
                'optimal_specificity': optimal_specificity,
                'thresholds': thresholds,
                'sensitivities': sensitivities,
                'specificities': specificities
            }
        )
        
        self.results.append(result)
        return result
    
    def _calculate_accuracy_ci(self, accuracy: float, n: int, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for accuracy"""
        from scipy import stats
        
        z = stats.norm.ppf(1 - (1 - confidence) / 2)
        se = np.sqrt(accuracy * (1 - accuracy) / n)
        margin = z * se
        
        return (max(0, accuracy - margin), min(1, accuracy + margin))
    
    def plot_identification_results(self, save_path: Optional[str] = None) -> plt.Figure:
        """Create comprehensive identification results visualization"""
        if not self.results:
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Compound Identification Performance', fontsize=16, fontweight='bold')
        
        # Extract metrics
        metrics = [r.metric_name for r in self.results]
        values = [r.value for r in self.results]
        
        # Plot 1: Performance metrics
        performance_metrics = ['Overall Accuracy', 'F1 Score', 'Average Precision', 'Average Recall']
        perf_values = [r.value for r in self.results if r.metric_name in performance_metrics]
        perf_names = [r.metric_name for r in self.results if r.metric_name in performance_metrics]
        
        if perf_values:
            colors = ['green' if v > 0.8 else 'orange' if v > 0.6 else 'red' for v in perf_values]
            axes[0, 0].bar(range(len(perf_names)), perf_values, color=colors, alpha=0.7)
            axes[0, 0].set_xticks(range(len(perf_names)))
            axes[0, 0].set_xticklabels(perf_names, rotation=45, ha='right')
            axes[0, 0].set_ylabel('Score')
            axes[0, 0].set_title('Performance Metrics')
            axes[0, 0].set_ylim(0, 1)
        
        # Plot 2: ROC Curve (if available)
        roc_result = next((r for r in self.results if r.metric_name == 'ROC AUC'), None)
        if roc_result and roc_result.metadata:
            fpr = roc_result.metadata['fpr']
            tpr = roc_result.metadata['tpr']
            
            axes[0, 1].plot(fpr, tpr, 'b-', label=f'ROC (AUC = {roc_result.value:.3f})')
            axes[0, 1].plot([0, 1], [0, 1], 'r--', label='Random')
            axes[0, 1].set_xlabel('False Positive Rate')
            axes[0, 1].set_ylabel('True Positive Rate')
            axes[0, 1].set_title('ROC Curve')
            axes[0, 1].legend()
        else:
            axes[0, 1].text(0.5, 0.5, 'ROC data not available', ha='center', va='center')
            axes[0, 1].set_title('ROC Curve')
        
        # Plot 3: Precision-Recall Curve (if available)
        pr_result = next((r for r in self.results if r.metric_name == 'PR AUC'), None)
        if pr_result and pr_result.metadata:
            precision = pr_result.metadata['precision']
            recall = pr_result.metadata['recall']
            
            axes[1, 0].plot(recall, precision, 'g-', label=f'PR (AUC = {pr_result.value:.3f})')
            axes[1, 0].set_xlabel('Recall')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].set_title('Precision-Recall Curve')
            axes[1, 0].legend()
        else:
            axes[1, 0].text(0.5, 0.5, 'PR data not available', ha='center', va='center')
            axes[1, 0].set_title('Precision-Recall Curve')
        
        # Plot 4: Sensitivity Analysis (if available)
        sens_result = next((r for r in self.results if r.metric_name == 'Optimal Sensitivity'), None)
        if sens_result and sens_result.metadata:
            thresholds = sens_result.metadata['thresholds']
            sensitivities = sens_result.metadata['sensitivities']
            specificities = sens_result.metadata['specificities']
            
            axes[1, 1].plot(thresholds, sensitivities, 'b-', label='Sensitivity')
            axes[1, 1].plot(thresholds, specificities, 'r-', label='Specificity')
            axes[1, 1].axvline(x=sens_result.metadata['optimal_threshold'], 
                              color='g', linestyle='--', label='Optimal')
            axes[1, 1].set_xlabel('Threshold')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].set_title('Sensitivity Analysis')
            axes[1, 1].legend()
        else:
            axes[1, 1].text(0.5, 0.5, 'Sensitivity data not available', ha='center', va='center')
            axes[1, 1].set_title('Sensitivity Analysis')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_identification_report(self) -> pd.DataFrame:
        """Generate comprehensive identification performance report"""
        if not self.results:
            return pd.DataFrame()
        
        data = []
        for result in self.results:
            data.append({
                'Metric': result.metric_name,
                'Value': result.value,
                'Confidence Interval': str(result.confidence_interval) if result.confidence_interval else 'N/A',
                'Interpretation': result.interpretation
            })
        
        return pd.DataFrame(data)
    
    def clear_results(self) -> None:
        """Clear all stored results"""
        self.results = [] 