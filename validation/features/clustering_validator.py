"""
Clustering Validator Module

Comprehensive clustering validation and analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score


@dataclass
class ClusteringResult:
    """Container for clustering validation results"""
    method_name: str
    n_clusters: int
    silhouette_score: float
    inertia: float
    interpretation: str
    metadata: Dict[str, Any] = None


class ClusteringValidator:
    """
    Comprehensive clustering validation
    """
    
    def __init__(self):
        """Initialize clustering validator"""
        self.results = []
        
    def validate_kmeans_clustering(
        self,
        features: np.ndarray,
        n_clusters_range: Tuple[int, int] = (2, 10)
    ) -> List[ClusteringResult]:
        """
        Validate K-means clustering across different cluster numbers
        
        Args:
            features: Feature matrix
            n_clusters_range: Range of cluster numbers to test
            
        Returns:
            List of ClusteringResult objects
        """
        results = []
        min_clusters, max_clusters = n_clusters_range
        
        for n_clusters in range(min_clusters, min(max_clusters + 1, len(features))):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features)
            
            # Calculate metrics
            sil_score = silhouette_score(features, cluster_labels)
            inertia = kmeans.inertia_
            
            interpretation = f"K-means with {n_clusters} clusters: silhouette={sil_score:.3f}"
            
            result = ClusteringResult(
                method_name="K-means",
                n_clusters=n_clusters,
                silhouette_score=sil_score,
                inertia=inertia,
                interpretation=interpretation,
                metadata={'cluster_centers': kmeans.cluster_centers_.tolist()}
            )
            
            results.append(result)
        
        self.results.extend(results)
        return results
    
    def find_optimal_clusters(
        self,
        features: np.ndarray
    ) -> ClusteringResult:
        """
        Find optimal number of clusters
        
        Args:
            features: Feature matrix
            
        Returns:
            ClusteringResult object for optimal clustering
        """
        clustering_results = self.validate_kmeans_clustering(features)
        
        # Find best based on silhouette score
        best_result = max(clustering_results, key=lambda x: x.silhouette_score)
        
        return best_result
    
    def generate_clustering_report(self) -> pd.DataFrame:
        """Generate clustering validation report"""
        if not self.results:
            return pd.DataFrame()
        
        data = []
        for result in self.results:
            data.append({
                'Method': result.method_name,
                'Clusters': result.n_clusters,
                'Silhouette Score': result.silhouette_score,
                'Inertia': result.inertia,
                'Interpretation': result.interpretation
            })
        
        return pd.DataFrame(data)
    
    def clear_results(self) -> None:
        """Clear all stored results"""
        self.results = [] 