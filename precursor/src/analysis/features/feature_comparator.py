"""
Feature Extraction Comparator Module

Compares feature extraction performance between numerical and visual pipelines
using PCA, t-SNE, mutual information, and clustering analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.feature_selection import mutual_info_regression
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
import warnings


@dataclass
class FeatureComparisonResult:
    """Container for feature comparison results"""
    metric_name: str
    numerical_score: float
    visual_score: float
    comparison_score: float  # How similar/different the features are
    interpretation: str
    metadata: Dict = None


class FeatureExtractorComparator:
    """
    Comprehensive feature extraction comparison between pipelines
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize feature comparator
        
        Args:
            random_state: Random state for reproducible results
        """
        self.random_state = random_state
        self.results = []
        
    def compare_feature_spaces(
        self,
        numerical_features: np.ndarray,
        visual_features: np.ndarray,
        labels: Optional[np.ndarray] = None
    ) -> List[FeatureComparisonResult]:
        """
        Compare feature spaces between pipelines
        
        Args:
            numerical_features: Features from numerical pipeline
            visual_features: Features from visual pipeline
            labels: Optional ground truth labels for supervised analysis
            
        Returns:
            List of FeatureComparisonResult objects
        """
        results = []
        
        # Ensure features have the same number of samples
        min_samples = min(len(numerical_features), len(visual_features))
        numerical_features = numerical_features[:min_samples]
        visual_features = visual_features[:min_samples]
        if labels is not None:
            labels = labels[:min_samples]
        
        # 1. Dimensionality comparison
        results.append(self._compare_dimensionality(numerical_features, visual_features))
        
        # 2. Variance explained comparison (PCA)
        results.append(self._compare_variance_explained(numerical_features, visual_features))
        
        # 3. Clustering quality comparison
        results.append(self._compare_clustering_quality(numerical_features, visual_features))
        
        # 4. Feature correlation analysis
        results.append(self._compare_feature_correlations(numerical_features, visual_features))
        
        # 5. Mutual information analysis
        if labels is not None:
            results.append(self._compare_mutual_information(numerical_features, visual_features, labels))
        
        # 6. Discriminative power comparison
        if labels is not None:
            results.append(self._compare_discriminative_power(numerical_features, visual_features, labels))
        
        self.results.extend(results)
        return results
    
    def _compare_dimensionality(
        self,
        numerical_features: np.ndarray,
        visual_features: np.ndarray
    ) -> FeatureComparisonResult:
        """Compare effective dimensionality of feature spaces"""
        
        # Calculate effective dimensionality using PCA
        def effective_dimensionality(features, variance_threshold=0.95):
            pca = PCA()
            pca.fit(features)
            cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
            return np.argmax(cumsum_variance >= variance_threshold) + 1
        
        num_eff_dim = effective_dimensionality(numerical_features)
        vis_eff_dim = effective_dimensionality(visual_features)
        
        # Comparison score: how similar the effective dimensionalities are
        max_dim = max(num_eff_dim, vis_eff_dim)
        comparison_score = 1.0 - abs(num_eff_dim - vis_eff_dim) / max_dim
        
        interpretation = f"Numerical: {num_eff_dim}D, Visual: {vis_eff_dim}D effective dimensions"
        if comparison_score > 0.8:
            interpretation += " (very similar complexity)"
        elif comparison_score > 0.6:
            interpretation += " (moderately similar complexity)"
        else:
            interpretation += " (different complexity)"
        
        return FeatureComparisonResult(
            metric_name="Effective Dimensionality",
            numerical_score=num_eff_dim,
            visual_score=vis_eff_dim,
            comparison_score=comparison_score,
            interpretation=interpretation,
            metadata={
                'numerical_dimensions': numerical_features.shape[1],
                'visual_dimensions': visual_features.shape[1]
            }
        )
    
    def _compare_variance_explained(
        self,
        numerical_features: np.ndarray,
        visual_features: np.ndarray
    ) -> FeatureComparisonResult:
        """Compare variance explained by top principal components"""
        
        # PCA analysis for both feature sets
        pca_num = PCA(n_components=min(10, numerical_features.shape[1]))
        pca_vis = PCA(n_components=min(10, visual_features.shape[1]))
        
        pca_num.fit(numerical_features)
        pca_vis.fit(visual_features)
        
        # Compare variance explained by first 3 components
        num_var_3 = np.sum(pca_num.explained_variance_ratio_[:3])
        vis_var_3 = np.sum(pca_vis.explained_variance_ratio_[:3])
        
        # Comparison score: similarity in variance concentration
        comparison_score = 1.0 - abs(num_var_3 - vis_var_3)
        
        interpretation = f"Top 3 PCs explain {num_var_3:.1%} (numerical) vs {vis_var_3:.1%} (visual) of variance"
        
        return FeatureComparisonResult(
            metric_name="Variance Concentration",
            numerical_score=num_var_3,
            visual_score=vis_var_3,
            comparison_score=comparison_score,
            interpretation=interpretation,
            metadata={
                'numerical_explained_variance': pca_num.explained_variance_ratio_,
                'visual_explained_variance': pca_vis.explained_variance_ratio_
            }
        )
    
    def _compare_clustering_quality(
        self,
        numerical_features: np.ndarray,
        visual_features: np.ndarray
    ) -> FeatureComparisonResult:
        """Compare clustering quality in feature spaces"""
        
        # Try different numbers of clusters
        k_values = [2, 3, 4, 5]
        num_silhouettes = []
        vis_silhouettes = []
        
        for k in k_values:
            if len(numerical_features) > k:
                # Numerical features clustering
                kmeans_num = KMeans(n_clusters=k, random_state=self.random_state)
                num_labels = kmeans_num.fit_predict(numerical_features)
                num_sil = silhouette_score(numerical_features, num_labels)
                num_silhouettes.append(num_sil)
                
                # Visual features clustering
                kmeans_vis = KMeans(n_clusters=k, random_state=self.random_state)
                vis_labels = kmeans_vis.fit_predict(visual_features)
                vis_sil = silhouette_score(visual_features, vis_labels)
                vis_silhouettes.append(vis_sil)
        
        # Average silhouette scores
        avg_num_sil = np.mean(num_silhouettes) if num_silhouettes else 0
        avg_vis_sil = np.mean(vis_silhouettes) if vis_silhouettes else 0
        
        # Comparison score
        comparison_score = 1.0 - abs(avg_num_sil - avg_vis_sil)
        
        interpretation = f"Average silhouette: {avg_num_sil:.3f} (numerical) vs {avg_vis_sil:.3f} (visual)"
        
        return FeatureComparisonResult(
            metric_name="Clustering Quality",
            numerical_score=avg_num_sil,
            visual_score=avg_vis_sil,
            comparison_score=comparison_score,
            interpretation=interpretation,
            metadata={
                'k_values': k_values,
                'numerical_silhouettes': num_silhouettes,
                'visual_silhouettes': vis_silhouettes
            }
        )
    
    def _compare_feature_correlations(
        self,
        numerical_features: np.ndarray,
        visual_features: np.ndarray
    ) -> FeatureComparisonResult:
        """Compare internal feature correlations"""
        
        # Calculate correlation matrices
        num_corr = np.corrcoef(numerical_features.T)
        vis_corr = np.corrcoef(visual_features.T)
        
        # Remove diagonal and get upper triangular
        num_corr_vals = num_corr[np.triu_indices_from(num_corr, k=1)]
        vis_corr_vals = vis_corr[np.triu_indices_from(vis_corr, k=1)]
        
        # Calculate average absolute correlation
        avg_num_corr = np.mean(np.abs(num_corr_vals))
        avg_vis_corr = np.mean(np.abs(vis_corr_vals))
        
        # Comparison score: similarity in correlation patterns
        comparison_score = 1.0 - abs(avg_num_corr - avg_vis_corr)
        
        interpretation = f"Average |correlation|: {avg_num_corr:.3f} (numerical) vs {avg_vis_corr:.3f} (visual)"
        
        return FeatureComparisonResult(
            metric_name="Feature Correlations",
            numerical_score=avg_num_corr,
            visual_score=avg_vis_corr,
            comparison_score=comparison_score,
            interpretation=interpretation,
            metadata={
                'numerical_correlation_matrix': num_corr,
                'visual_correlation_matrix': vis_corr
            }
        )
    
    def _compare_mutual_information(
        self,
        numerical_features: np.ndarray,
        visual_features: np.ndarray,
        labels: np.ndarray
    ) -> FeatureComparisonResult:
        """Compare mutual information with labels"""
        
        # Calculate mutual information for each feature
        num_mi = []
        vis_mi = []
        
        for i in range(numerical_features.shape[1]):
            mi = mutual_info_regression(numerical_features[:, i:i+1], labels, random_state=self.random_state)
            num_mi.append(mi[0])
        
        for i in range(visual_features.shape[1]):
            mi = mutual_info_regression(visual_features[:, i:i+1], labels, random_state=self.random_state)
            vis_mi.append(mi[0])
        
        # Average mutual information
        avg_num_mi = np.mean(num_mi)
        avg_vis_mi = np.mean(vis_mi)
        
        # Comparison score
        max_mi = max(avg_num_mi, avg_vis_mi)
        comparison_score = 1.0 - abs(avg_num_mi - avg_vis_mi) / max_mi if max_mi > 0 else 1.0
        
        interpretation = f"Average MI with labels: {avg_num_mi:.3f} (numerical) vs {avg_vis_mi:.3f} (visual)"
        
        return FeatureComparisonResult(
            metric_name="Mutual Information",
            numerical_score=avg_num_mi,
            visual_score=avg_vis_mi,
            comparison_score=comparison_score,
            interpretation=interpretation,
            metadata={
                'numerical_mi_per_feature': num_mi,
                'visual_mi_per_feature': vis_mi
            }
        )
    
    def _compare_discriminative_power(
        self,
        numerical_features: np.ndarray,
        visual_features: np.ndarray,
        labels: np.ndarray
    ) -> FeatureComparisonResult:
        """Compare discriminative power using simple classification"""
        
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        
        # Simple logistic regression with cross-validation
        lr = LogisticRegression(random_state=self.random_state, max_iter=1000)
        
        # Numerical features
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            num_scores = cross_val_score(lr, numerical_features, labels, cv=3, scoring='accuracy')
            avg_num_acc = np.mean(num_scores)
        
        # Visual features
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vis_scores = cross_val_score(lr, visual_features, labels, cv=3, scoring='accuracy')
            avg_vis_acc = np.mean(vis_scores)
        
        # Comparison score
        comparison_score = 1.0 - abs(avg_num_acc - avg_vis_acc)
        
        interpretation = f"Classification accuracy: {avg_num_acc:.3f} (numerical) vs {avg_vis_acc:.3f} (visual)"
        
        return FeatureComparisonResult(
            metric_name="Discriminative Power",
            numerical_score=avg_num_acc,
            visual_score=avg_vis_acc,
            comparison_score=comparison_score,
            interpretation=interpretation,
            metadata={
                'numerical_cv_scores': num_scores,
                'visual_cv_scores': vis_scores
            }
        )
    
    def analyze_feature_complementarity(
        self,
        numerical_features: np.ndarray,
        visual_features: np.ndarray
    ) -> FeatureComparisonResult:
        """Analyze how complementary the feature sets are"""
        
        # Calculate cross-correlation between feature sets
        cross_correlations = []
        
        min_features = min(numerical_features.shape[1], visual_features.shape[1])
        
        for i in range(min_features):
            for j in range(min_features):
                corr, _ = pearsonr(numerical_features[:, i], visual_features[:, j])
                if not np.isnan(corr):
                    cross_correlations.append(abs(corr))
        
        avg_cross_corr = np.mean(cross_correlations) if cross_correlations else 0
        
        # Complementarity score (lower correlation = more complementary)
        complementarity_score = 1.0 - avg_cross_corr
        
        interpretation = f"Average cross-correlation: {avg_cross_corr:.3f}"
        if complementarity_score > 0.7:
            interpretation += " (highly complementary features)"
        elif complementarity_score > 0.5:
            interpretation += " (moderately complementary features)"
        else:
            interpretation += " (similar/redundant features)"
        
        result = FeatureComparisonResult(
            metric_name="Feature Complementarity",
            numerical_score=avg_cross_corr,
            visual_score=avg_cross_corr,
            comparison_score=complementarity_score,
            interpretation=interpretation,
            metadata={'cross_correlations': cross_correlations}
        )
        
        self.results.append(result)
        return result
    
    def plot_feature_comparison(self, save_path: Optional[str] = None) -> plt.Figure:
        """Create comprehensive feature comparison visualization"""
        if not self.results:
            return None
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Feature Extraction Comparison', fontsize=16, fontweight='bold')
        
        # Extract data
        metrics = [r.metric_name for r in self.results]
        num_scores = [r.numerical_score for r in self.results]
        vis_scores = [r.visual_score for r in self.results]
        comp_scores = [r.comparison_score for r in self.results]
        
        # Plot 1: Score comparison
        x_pos = np.arange(len(metrics))
        width = 0.35
        
        axes[0, 0].bar(x_pos - width/2, num_scores, width, label='Numerical', alpha=0.7)
        axes[0, 0].bar(x_pos + width/2, vis_scores, width, label='Visual', alpha=0.7)
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels([m.split()[0] for m in metrics], rotation=45, ha='right')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Pipeline Comparison')
        axes[0, 0].legend()
        
        # Plot 2: Similarity scores
        colors = ['green' if score > 0.8 else 'orange' if score > 0.6 else 'red' for score in comp_scores]
        axes[0, 1].bar(x_pos, comp_scores, color=colors, alpha=0.7)
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels([m.split()[0] for m in metrics], rotation=45, ha='right')
        axes[0, 1].set_ylabel('Similarity Score')
        axes[0, 1].set_title('Feature Space Similarity')
        axes[0, 1].set_ylim(0, 1)
        
        # Plot 3: PCA comparison (if available)
        variance_result = next((r for r in self.results if 'Variance' in r.metric_name), None)
        if variance_result and variance_result.metadata:
            num_var = variance_result.metadata.get('numerical_explained_variance', [])
            vis_var = variance_result.metadata.get('visual_explained_variance', [])
            
            if len(num_var) > 0 and len(vis_var) > 0:
                max_components = min(10, len(num_var), len(vis_var))
                components = range(1, max_components + 1)
                
                axes[0, 2].plot(components, num_var[:max_components], 'b-o', label='Numerical')
                axes[0, 2].plot(components, vis_var[:max_components], 'r-o', label='Visual')
                axes[0, 2].set_xlabel('Principal Component')
                axes[0, 2].set_ylabel('Explained Variance Ratio')
                axes[0, 2].set_title('PCA Comparison')
                axes[0, 2].legend()
        
        # Plot 4: Clustering quality (if available)
        cluster_result = next((r for r in self.results if 'Clustering' in r.metric_name), None)
        if cluster_result and cluster_result.metadata:
            k_values = cluster_result.metadata.get('k_values', [])
            num_sil = cluster_result.metadata.get('numerical_silhouettes', [])
            vis_sil = cluster_result.metadata.get('visual_silhouettes', [])
            
            if k_values and num_sil and vis_sil:
                axes[1, 0].plot(k_values, num_sil, 'b-o', label='Numerical')
                axes[1, 0].plot(k_values, vis_sil, 'r-o', label='Visual')
                axes[1, 0].set_xlabel('Number of Clusters')
                axes[1, 0].set_ylabel('Silhouette Score')
                axes[1, 0].set_title('Clustering Quality')
                axes[1, 0].legend()
        
        # Plot 5: Correlation heatmap comparison
        corr_result = next((r for r in self.results if 'Correlation' in r.metric_name), None)
        if corr_result and corr_result.metadata:
            num_corr = corr_result.metadata.get('numerical_correlation_matrix')
            if num_corr is not None and num_corr.shape[0] <= 20:  # Only plot if not too large
                im = axes[1, 1].imshow(num_corr, cmap='coolwarm', vmin=-1, vmax=1)
                axes[1, 1].set_title('Numerical Feature Correlations')
                plt.colorbar(im, ax=axes[1, 1])
        
        # Plot 6: Summary radar chart
        if len(comp_scores) >= 3:
            angles = np.linspace(0, 2*np.pi, len(comp_scores), endpoint=False)
            angles = np.concatenate((angles, [angles[0]]))  # Complete the circle
            comp_scores_plot = comp_scores + [comp_scores[0]]
            
            axes[1, 2] = plt.subplot(2, 3, 6, projection='polar')
            axes[1, 2].plot(angles, comp_scores_plot, 'o-', linewidth=2)
            axes[1, 2].fill(angles, comp_scores_plot, alpha=0.25)
            axes[1, 2].set_xticks(angles[:-1])
            axes[1, 2].set_xticklabels([m.split()[0] for m in metrics])
            axes[1, 2].set_ylim(0, 1)
            axes[1, 2].set_title('Similarity Radar')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_comparison_report(self) -> pd.DataFrame:
        """Generate comprehensive feature comparison report"""
        if not self.results:
            return pd.DataFrame()
        
        data = []
        for result in self.results:
            data.append({
                'Metric': result.metric_name,
                'Numerical Score': result.numerical_score,
                'Visual Score': result.visual_score,
                'Similarity Score': result.comparison_score,
                'Interpretation': result.interpretation
            })
        
        return pd.DataFrame(data)
    
    def clear_results(self) -> None:
        """Clear all stored results"""
        self.results = [] 