"""
Database Search Analyzer Module

Comprehensive analysis of database search results and performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass


@dataclass
class DatabaseSearchResult:
    """Container for database search analysis results"""
    metric_name: str
    value: float
    interpretation: str
    metadata: Dict[str, Any] = None


class DatabaseSearchAnalyzer:
    """
    Comprehensive database search analysis
    """
    
    def __init__(self):
        """Initialize database search analyzer"""
        self.results = []
        
    def analyze_search_performance(
        self,
        search_scores: np.ndarray,
        true_matches: np.ndarray,
        score_threshold: float = 0.8
    ) -> DatabaseSearchResult:
        """
        Analyze database search performance
        
        Args:
            search_scores: Search confidence scores
            true_matches: True match indicators
            score_threshold: Threshold for positive matches
            
        Returns:
            DatabaseSearchResult object
        """
        # Calculate metrics
        predicted_matches = search_scores >= score_threshold
        
        tp = np.sum(predicted_matches & true_matches)
        fp = np.sum(predicted_matches & ~true_matches)
        tn = np.sum(~predicted_matches & ~true_matches)
        fn = np.sum(~predicted_matches & true_matches)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        interpretation = f"Search performance: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1_score:.3f}"
        
        result = DatabaseSearchResult(
            metric_name="Search Performance",
            value=f1_score,
            interpretation=interpretation,
            metadata={
                'precision': precision,
                'recall': recall,
                'threshold': score_threshold,
                'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
            }
        )
        
        self.results.append(result)
        return result
    
    def generate_search_report(self) -> pd.DataFrame:
        """Generate database search analysis report"""
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