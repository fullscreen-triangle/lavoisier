"""
Information Content Analyzer Module

Comprehensive analysis of information content in features including
entropy analysis, mutual information, and information-theoretic metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import KBinsDiscretizer
from scipy.stats import entropy


@dataclass
class InformationResult:
    """Container for information content analysis results"""
    metric_name: str
    value: float
    interpretation: str
    feature_rankings: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = None


class InformationContentAnalyzer:
    """
    Comprehensive information content analysis for features
    """
    
    def __init__(self):
        """Initialize information content analyzer"""
        self.results = []
        
    def calculate_entropy(
        self,
        data: np.ndarray,
        bins: int = 50,
        normalize: bool = True
    ) -> InformationResult:
        """
        Calculate entropy of data
        
        Args:
            data: Data to analyze
            bins: Number of bins for discretization
            normalize: Whether to normalize entropy
            
        Returns:
            InformationResult object
        """
        # Flatten data if multidimensional
        if data.ndim > 1:
            entropies = []
            for i in range(data.shape[1]):
                feature_data = data[:, i]
                feature_entropy = self._calculate_single_entropy(feature_data, bins, normalize)
                entropies.append(feature_entropy)
            
            avg_entropy = np.mean(entropies)
            feature_rankings = {f"Feature_{i}": ent for i, ent in enumerate(entropies)}
            
            interpretation = f"Average entropy: {avg_entropy:.3f}"
            if normalize:
                interpretation += " (normalized)"
        else:
            avg_entropy = self._calculate_single_entropy(data, bins, normalize)
            feature_rankings = None
            interpretation = f"Data entropy: {avg_entropy:.3f}"
            if normalize:
                interpretation += " (normalized)"
        
        result = InformationResult(
            metric_name="Entropy",
            value=avg_entropy,
            interpretation=interpretation,
            feature_rankings=feature_rankings,
            metadata={
                'bins': bins,
                'normalized': normalize,
                'data_shape': data.shape
            }
        )
        
        self.results.append(result)
        return result
    
    def _calculate_single_entropy(
        self,
        data: np.ndarray,
        bins: int,
        normalize: bool
    ) -> float:
        """Calculate entropy for single feature"""
        # Remove NaN values
        clean_data = data[~np.isnan(data)]
        
        if len(clean_data) == 0:
            return 0.0
        
        # Discretize continuous data
        hist, _ = np.histogram(clean_data, bins=bins, density=True)
        
        # Remove zero bins
        hist = hist[hist > 0]
        
        if len(hist) == 0:
            return 0.0
        
        # Calculate entropy
        ent = entropy(hist, base=2)
        
        # Normalize if requested
        if normalize:
            max_entropy = np.log2(len(hist))
            ent = ent / max_entropy if max_entropy > 0 else 0.0
        
        return ent
    
    def calculate_mutual_information(
        self,
        features: np.ndarray,
        target: np.ndarray,
        task_type: str = "regression"
    ) -> InformationResult:
        """
        Calculate mutual information between features and target
        
        Args:
            features: Feature matrix
            target: Target variable
            task_type: Type of task ('regression' or 'classification')
            
        Returns:
            InformationResult object
        """
        # Calculate mutual information
        if task_type == "regression":
            mi_scores = mutual_info_regression(features, target, random_state=42)
        else:
            mi_scores = mutual_info_classif(features, target, random_state=42)
        
        # Create feature rankings
        feature_rankings = {f"Feature_{i}": score for i, score in enumerate(mi_scores)}
        
        # Calculate average mutual information
        avg_mi = np.mean(mi_scores)
        
        # Interpretation
        if avg_mi > 0.5:
            interpretation = f"High mutual information ({avg_mi:.3f}) - features are highly informative"
        elif avg_mi > 0.2:
            interpretation = f"Moderate mutual information ({avg_mi:.3f}) - features contain useful information"
        elif avg_mi > 0.05:
            interpretation = f"Low mutual information ({avg_mi:.3f}) - features have limited information"
        else:
            interpretation = f"Very low mutual information ({avg_mi:.3f}) - features may not be informative"
        
        result = InformationResult(
            metric_name="Mutual Information",
            value=avg_mi,
            interpretation=interpretation,
            feature_rankings=feature_rankings,
            metadata={
                'task_type': task_type,
                'max_mi': np.max(mi_scores),
                'min_mi': np.min(mi_scores),
                'std_mi': np.std(mi_scores)
            }
        )
        
        self.results.append(result)
        return result
    
    def comprehensive_information_analysis(
        self,
        features: np.ndarray,
        target: Optional[np.ndarray] = None,
        **kwargs
    ) -> List[InformationResult]:
        """
        Run comprehensive information content analysis
        
        Args:
            features: Feature matrix
            target: Target variable (optional)
            **kwargs: Additional parameters
            
        Returns:
            List of InformationResult objects
        """
        results = []
        
        # Entropy analysis
        bins = kwargs.get('bins', 50)
        results.append(self.calculate_entropy(features, bins=bins))
        
        # Mutual information with target (if provided)
        if target is not None:
            task_type = kwargs.get('task_type', 'regression')
            results.append(self.calculate_mutual_information(features, target, task_type))
        
        return results
    
    def generate_information_report(self) -> pd.DataFrame:
        """Generate comprehensive information content report"""
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