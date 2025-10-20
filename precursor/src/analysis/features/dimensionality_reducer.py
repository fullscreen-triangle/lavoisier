"""
Dimensionality Reducer Module

Comprehensive dimensionality reduction analysis and comparison.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score


@dataclass
class DimensionalityResult:
    """Container for dimensionality reduction results"""
    method_name: str
    explained_variance_ratio: float
    silhouette_score: float
    interpretation: str
    metadata: Dict[str, Any] = None


class DimensionalityReducer:
    """
    Comprehensive dimensionality reduction analysis
    """
    
    def __init__(self):
        """Initialize dimensionality reducer"""
        self.results = []
        
    def analyze_pca(
        self,
        features: np.ndarray,
        n_components: int = 2
    ) -> DimensionalityResult:
        """
        Analyze PCA dimensionality reduction
        
        Args:
            features: Feature matrix
            n_components: Number of components
            
        Returns:
            DimensionalityResult object
        """
        pca = PCA(n_components=n_components)
        transformed = pca.fit_transform(features)
        
        explained_variance = np.sum(pca.explained_variance_ratio_)
        
        # Calculate silhouette score if we have enough samples
        if len(features) > n_components * 2:
            try:
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=min(8, len(features)//10), random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(transformed)
                sil_score = silhouette_score(transformed, cluster_labels)
            except:
                sil_score = 0.0
        else:
            sil_score = 0.0
        
        interpretation = f"PCA explains {explained_variance:.1%} of variance with {n_components} components"
        
        result = DimensionalityResult(
            method_name="PCA",
            explained_variance_ratio=explained_variance,
            silhouette_score=sil_score,
            interpretation=interpretation,
            metadata={
                'n_components': n_components,
                'individual_variance_ratios': pca.explained_variance_ratio_.tolist()
            }
        )
        
        self.results.append(result)
        return result
    
    def comprehensive_dimensionality_analysis(
        self,
        features: np.ndarray,
        **kwargs
    ) -> List[DimensionalityResult]:
        """
        Run comprehensive dimensionality analysis
        
        Args:
            features: Feature matrix
            **kwargs: Additional parameters
            
        Returns:
            List of DimensionalityResult objects
        """
        results = []
        
        # PCA analysis
        n_components = kwargs.get('n_components', min(10, features.shape[1]))
        results.append(self.analyze_pca(features, n_components))
        
        return results
    
    def generate_dimensionality_report(self) -> pd.DataFrame:
        """Generate dimensionality reduction report"""
        if not self.results:
            return pd.DataFrame()
        
        data = []
        for result in self.results:
            data.append({
                'Method': result.method_name,
                'Explained Variance': result.explained_variance_ratio,
                'Silhouette Score': result.silhouette_score,
                'Interpretation': result.interpretation
            })
        
        return pd.DataFrame(data)
    
    def clear_results(self) -> None:
        """Clear all stored results"""
        self.results = [] 