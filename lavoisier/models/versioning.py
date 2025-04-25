"""
Model versioning and metadata for model repository
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import datetime


@dataclass
class ModelMetadata:
    """Metadata for a model version"""
    created_at: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    metrics: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    tags: List[str] = field(default_factory=list)


@dataclass
class ModelVersion:
    """Information about a model version"""
    version: int
    metadata: ModelMetadata
    path: str


class ModelPerformance:
    """Performance metrics for a model"""
    
    def __init__(self, metrics: Dict[str, Any]):
        """
        Initialize performance metrics
        
        Args:
            metrics: Dictionary of metrics
        """
        self.metrics = metrics
    
    def get_metric(self, name: str) -> Any:
        """
        Get a specific metric
        
        Args:
            name: Name of the metric
            
        Returns:
            Metric value or None if not found
        """
        return self.metrics.get(name)
    
    def set_metric(self, name: str, value: Any) -> None:
        """
        Set a metric value
        
        Args:
            name: Name of the metric
            value: Metric value
        """
        self.metrics[name] = value
    
    def compare(self, other: 'ModelPerformance') -> Dict[str, Dict[str, Any]]:
        """
        Compare this performance with another
        
        Args:
            other: Another ModelPerformance instance
            
        Returns:
            Dictionary of metric comparisons
        """
        result = {}
        
        # Find common metrics
        common_metrics = set(self.metrics.keys()).intersection(other.metrics.keys())
        
        for metric in common_metrics:
            this_value = self.metrics[metric]
            other_value = other.metrics[metric]
            
            # Calculate difference if numeric
            if isinstance(this_value, (int, float)) and isinstance(other_value, (int, float)):
                diff = this_value - other_value
                percent = (diff / other_value * 100) if other_value != 0 else float('inf')
                
                result[metric] = {
                    "this": this_value,
                    "other": other_value,
                    "diff": diff,
                    "percent_change": percent
                }
            else:
                result[metric] = {
                    "this": this_value,
                    "other": other_value,
                    "diff": None,
                    "percent_change": None
                }
        
        # Add metrics unique to this model
        only_this = set(self.metrics.keys()) - set(other.metrics.keys())
        for metric in only_this:
            result[metric] = {
                "this": self.metrics[metric],
                "other": None,
                "diff": None,
                "percent_change": None
            }
        
        # Add metrics unique to other model
        only_other = set(other.metrics.keys()) - set(self.metrics.keys())
        for metric in only_other:
            result[metric] = {
                "this": None,
                "other": other.metrics[metric],
                "diff": None,
                "percent_change": None
            }
        
        return result 