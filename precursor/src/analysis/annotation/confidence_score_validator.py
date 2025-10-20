"""
Confidence Score Validator Module

Comprehensive validation of confidence scores and calibration analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


@dataclass
class ConfidenceResult:
    """Container for confidence score validation results"""
    metric_name: str
    value: float
    interpretation: str
    metadata: Dict[str, Any] = None


class ConfidenceScoreValidator:
    """
    Comprehensive confidence score validation
    """
    
    def __init__(self):
        """Initialize confidence score validator"""
        self.results = []
        
    def validate_calibration(
        self,
        confidence_scores: np.ndarray,
        true_labels: np.ndarray,
        n_bins: int = 10
    ) -> ConfidenceResult:
        """
        Validate confidence score calibration
        
        Args:
            confidence_scores: Confidence scores (0-1)
            true_labels: True binary labels
            n_bins: Number of bins for calibration
            
        Returns:
            ConfidenceResult object
        """
        # Calculate calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            true_labels, confidence_scores, n_bins=n_bins
        )
        
        # Calculate calibration error (reliability)
        calibration_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
        
        # Calculate Brier score
        brier_score = brier_score_loss(true_labels, confidence_scores)
        
        interpretation = f"Calibration error: {calibration_error:.3f}, Brier score: {brier_score:.3f}"
        
        result = ConfidenceResult(
            metric_name="Confidence Calibration",
            value=1.0 - calibration_error,  # Higher is better
            interpretation=interpretation,
            metadata={
                'calibration_error': calibration_error,
                'brier_score': brier_score,
                'fraction_of_positives': fraction_of_positives.tolist(),
                'mean_predicted_value': mean_predicted_value.tolist()
            }
        )
        
        self.results.append(result)
        return result
    
    def analyze_confidence_distribution(
        self,
        confidence_scores: np.ndarray
    ) -> ConfidenceResult:
        """
        Analyze confidence score distribution
        
        Args:
            confidence_scores: Confidence scores
            
        Returns:
            ConfidenceResult object
        """
        mean_confidence = np.mean(confidence_scores)
        std_confidence = np.std(confidence_scores)
        
        # Check for overconfidence (scores concentrated near 1)
        high_confidence_rate = np.mean(confidence_scores > 0.9)
        low_confidence_rate = np.mean(confidence_scores < 0.1)
        
        interpretation = f"Mean confidence: {mean_confidence:.3f}, Std: {std_confidence:.3f}"
        
        if high_confidence_rate > 0.5:
            interpretation += " (potentially overconfident)"
        elif low_confidence_rate > 0.5:
            interpretation += " (potentially underconfident)"
        
        result = ConfidenceResult(
            metric_name="Confidence Distribution",
            value=mean_confidence,
            interpretation=interpretation,
            metadata={
                'std_confidence': std_confidence,
                'high_confidence_rate': high_confidence_rate,
                'low_confidence_rate': low_confidence_rate
            }
        )
        
        self.results.append(result)
        return result
    
    def generate_confidence_report(self) -> pd.DataFrame:
        """Generate confidence score validation report"""
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