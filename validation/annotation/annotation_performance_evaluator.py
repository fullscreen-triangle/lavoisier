"""
Annotation Performance Evaluator Module

Comprehensive evaluation of annotation performance across different metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


@dataclass
class AnnotationPerformanceResult:
    """Container for annotation performance results"""
    metric_name: str
    value: float
    interpretation: str
    metadata: Dict[str, Any] = None


class AnnotationPerformanceEvaluator:
    """
    Comprehensive annotation performance evaluation
    """
    
    def __init__(self):
        """Initialize annotation performance evaluator"""
        self.results = []
        
    def evaluate_classification_performance(
        self,
        true_labels: np.ndarray,
        predicted_labels: np.ndarray
    ) -> List[AnnotationPerformanceResult]:
        """
        Evaluate classification performance
        
        Args:
            true_labels: True labels
            predicted_labels: Predicted labels
            
        Returns:
            List of AnnotationPerformanceResult objects
        """
        results = []
        
        # Accuracy
        acc = accuracy_score(true_labels, predicted_labels)
        results.append(AnnotationPerformanceResult(
            metric_name="Accuracy",
            value=acc,
            interpretation=f"Classification accuracy: {acc:.3f} ({acc*100:.1f}%)"
        ))
        
        # Precision
        prec = precision_score(true_labels, predicted_labels, average='macro', zero_division=0)
        results.append(AnnotationPerformanceResult(
            metric_name="Precision",
            value=prec,
            interpretation=f"Precision (macro): {prec:.3f}"
        ))
        
        # Recall
        rec = recall_score(true_labels, predicted_labels, average='macro', zero_division=0)
        results.append(AnnotationPerformanceResult(
            metric_name="Recall",
            value=rec,
            interpretation=f"Recall (macro): {rec:.3f}"
        ))
        
        # F1 Score
        f1 = f1_score(true_labels, predicted_labels, average='macro', zero_division=0)
        results.append(AnnotationPerformanceResult(
            metric_name="F1 Score",
            value=f1,
            interpretation=f"F1 score (macro): {f1:.3f}"
        ))
        
        self.results.extend(results)
        return results
    
    def evaluate_ranking_performance(
        self,
        true_relevance: np.ndarray,
        predicted_scores: np.ndarray,
        k: int = 10
    ) -> AnnotationPerformanceResult:
        """
        Evaluate ranking performance
        
        Args:
            true_relevance: True relevance scores
            predicted_scores: Predicted scores
            k: Number of top results to consider
            
        Returns:
            AnnotationPerformanceResult object
        """
        # Sort by predicted scores
        sorted_indices = np.argsort(predicted_scores)[::-1]
        
        # Calculate precision at k
        top_k_indices = sorted_indices[:k]
        top_k_relevance = true_relevance[top_k_indices]
        
        precision_at_k = np.mean(top_k_relevance)
        
        interpretation = f"Precision@{k}: {precision_at_k:.3f}"
        
        result = AnnotationPerformanceResult(
            metric_name=f"Precision@{k}",
            value=precision_at_k,
            interpretation=interpretation,
            metadata={'k': k}
        )
        
        self.results.append(result)
        return result
    
    def generate_performance_report(self) -> pd.DataFrame:
        """Generate annotation performance report"""
        if not self.results:
            return pd.DataFrame()
        
        data = []
        for result in self.results:
            data.append({
                'Metric': result.metric_name,
                'Value': result.value,
                'Interpretation': result.interpretation
            })
        
        return pd.DataFrame(data)
    
    def clear_results(self) -> None:
        """Clear all stored results"""
        self.results = [] 