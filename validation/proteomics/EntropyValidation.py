"""
S-Entropy Validation Pipeline for Proteomics
Implements clustering, benchmarking, and comprehensive validation
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import time
from pathlib import Path
import json

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform

import matplotlib.pyplot as plt
import seaborn as sns

from SEntropy_Proteomics_Core import SEntropyProteomicsEngine, SEntropySpectrum


class SEntropyValidationPipeline:
    """
    Complete validation pipeline for S-Entropy proteomics
    """

    def __init__(self, engine: SEntropyProteomicsEngine, output_dir: str):
        self.engine = engine
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results = {
            'clustering': {},
            'benchmarks': {},
            'validation': {},
            'figures': []
        }

    # ========================================================================
    # CLUSTERING ANALYSIS
    # ========================================================================

    def run_clustering_analysis(
            self,
            spectra: List[SEntropySpectrum],
            ground_truth_labels: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Run comprehensive clustering analysis on S-Entropy features

        Args:
            spectra: List of processed spectra
            ground_truth_labels: Optional ground truth labels for validation

        Returns:
            Dictionary with clustering results
        """
        print("\n" + "=" * 80)
        print("CLUSTERING ANALYSIS")
        print("=" * 80)

        # Extract feature matrix
        feature_matrix = np.vstack([s.sentropy_features_14d for s in spectra])
        spectrum_ids = [s.spectrum_id for s in spectra]

        print(f"Feature matrix shape: {feature_matrix.shape}")
        print(f"Number of spectra: {len(spectra)}")

        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(feature_matrix)

        clustering_results = {}

        # Test different numbers of clusters
        n_clusters_range = self.engine.config['clustering']['n_clusters_range']

        for n_clusters in n_clusters_range:
            print(f"\n--- Testing n_clusters = {n_clusters} ---")

            cluster_result = {
                'n_clusters': n_clusters,
                'methods': {}
            }

            # K-Means
            if 'kmeans' in self.engine.config['clustering']['methods']:
                kmeans_result = self._run_kmeans(
                    features_scaled, n_clusters, spectrum_ids
                )
                cluster_result['methods']['kmeans'] = kmeans_result
                print(f"K-Means - Silhouette: {kmeans_result['silhouette']:.4f}")

            # Hierarchical
            if 'hierarchical' in self.engine.config['clustering']['methods']:
                hierarchical_result = self._run_hierarchical(
                    features_scaled, n_clusters, spectrum_ids
                )
                cluster_result['methods']['hierarchical'] = hierarchical_result
                print(f"Hierarchical - Silhouette: {hierarchical_result['silhouette']:.4f}")

            # DBSCAN (only for first iteration)
            if 'dbscan' in self.engine.config['clustering']['methods'] and n_clusters == n_clusters_range[0]:
                dbscan_result = self._run_dbscan(
                    features_scaled, spectrum_ids
                )
                cluster_result['methods']['dbscan'] = dbscan_result
                print(f"DBSCAN - Silhouette: {dbscan_result['silhouette']:.4f}")

            # Compare with ground truth if available
            if ground_truth_labels is not None:
                cluster_result['ground_truth_comparison'] = self._compare_with_ground_truth(
                    cluster_result, ground_truth_labels
                )

            clustering_results[f'n_clusters_{n_clusters}'] = cluster_result

        # Find best clustering
        best_result = self._find_best_clustering(clustering_results)
        clustering_results['best'] = best_result

        print(f"\n{'=' * 80}")
        print(f"BEST CLUSTERING: {best_result['method']} with n_clusters={best_result['n_clusters']}")
        print(f"Silhouette Score: {best_result['silhouette']:.4f}")
        print(f"{'=' * 80}\n")

        self.results['clustering'] = clustering_results

        return clustering_results

    def _run_kmeans(
            self,
            features: np.ndarray,
            n_clusters: int,
            spectrum_ids: List[str]
    ) -> Dict:
        """Run K-Means clustering"""
        start_time = time.time()

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)

        # Compute metrics
        silhouette = silhouette_score(features, labels)
        davies_bouldin = davies_bouldin_score(features, labels)
        calinski_harabasz = calinski_harabasz_score(features, labels)

        processing_time = time.time() - start_time

        return {
            'labels': labels.tolist(),
            'spectrum_ids': spectrum_ids,
            'cluster_centers': kmeans.cluster_centers_.tolist(),
            'silhouette': float(silhouette),
            'davies_bouldin': float(davies_bouldin),
            'calinski_harabasz': float(calinski_harabasz),
            'processing_time': processing_time,
            'inertia': float(kmeans.inertia_)
        }

    def _run_hierarchical(
            self,
            features: np.ndarray,
            n_clusters: int,
            spectrum_ids: List[str]
    ) -> Dict:
        """Run Hierarchical clustering"""
        start_time = time.time()

        hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        labels = hierarchical.fit_predict(features)

        # Compute metrics
        silhouette = silhouette_score(features, labels)
        davies_bouldin = davies_bouldin_score(features, labels)
        calinski_harabasz = calinski_harabasz_score(features, labels)

        processing_time = time.time() - start_time

        return {
            'labels': labels.tolist(),
            'spectrum_ids': spectrum_ids,
            'silhouette': float(silhouette),
            'davies_bouldin': float(davies_bouldin),
            'calinski_harabasz': float(calinski_harabasz),
            'processing_time': processing_time
        }

    def _run_dbscan(
            self,
            features: np.ndarray,
            spectrum_ids: List[str]
    ) -> Dict:
        """Run DBSCAN clustering"""
        start_time = time.time()

        # Estimate eps using k-distance graph
        distances = pdist(features, metric='euclidean')
        eps = np.percentile(distances, 10)

        dbscan = DBSCAN(eps=eps, min_samples=5)
        labels = dbscan.fit_predict(features)

        # Compute metrics (excluding noise points)
        mask = labels != -1
        if np.sum(mask) > 1 and len(np.unique(labels[mask])) > 1:
            silhouette = silhouette_score(features[mask], labels[mask])
            davies_bouldin = davies_bouldin_score(features[mask], labels[mask])
            calinski_harabasz = calinski_harabasz_score(features[mask], labels[mask])
        else:
            silhouette = -1.0
            davies_bouldin = 999.0
            calinski_harabasz = 0.0

        processing_time = time.time() - start_time

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        return {
            'labels': labels.tolist(),
            'spectrum_ids': spectrum_ids,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'silhouette': float(silhouette),
            'davies_bouldin': float(davies_bouldin),
            'calinski_harabasz': float(calinski_harabasz),
            'processing_time': processing_time,
            'eps': float(eps)
        }

    def _compare_with_ground_truth(
            self,
            cluster_result: Dict,
            ground_truth: np.ndarray
    ) -> Dict:
        """Compare clustering results with ground truth labels"""
        comparison = {}

        for method_name, method_result in cluster_result['methods'].items():
            predicted_labels = np.array(method_result['labels'])

            # Remove noise points for DBSCAN
            if method_name == 'dbscan':
                mask = predicted_labels != -1
                predicted_labels = predicted_labels[mask]
                gt_labels = ground_truth[mask]
            else:
                gt_labels = ground_truth

            if len(predicted_labels) > 0:
                ari = adjusted_rand_score(gt_labels, predicted_labels)
                nmi = normalized_mutual_info_score(gt_labels, predicted_labels)

                comparison[method_name] = {
                    'adjusted_rand_index': float(ari),
                    'normalized_mutual_info': float(nmi)
                }

        return comparison

    def _find_best_clustering(self, clustering_results: Dict) -> Dict:
        """Find best clustering result based on silhouette score"""
        best_score = -1
        best_result = None

        for n_cluster_key, cluster_result in clustering_results.items():
            if n_cluster_key == 'best':
                continue

            for method_name, method_result in cluster_result['methods'].items():
                if method_result['silhouette'] > best_score:
                    best_score = method_result['silhouette']
                    best_result = {
                        'method': method_name,
                        'n_clusters': cluster_result['n_clusters'],
                        'silhouette': method_result['silhouette'],
                        'davies_bouldin': method_result['davies_bouldin'],
                        'calinski_harabasz': method_result['calinski_harabasz'],
                        'labels': method_result['labels']
                    }

        return best_result

    # ========================================================================
    # BENCHMARKING
    # ========================================================================

    def run_benchmarking(
            self,
            spectra: List[SEntropySpectrum],
            traditional_features: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Benchmark S-Entropy features against traditional features

        Args:
            spectra: List of S-Entropy processed spectra
            traditional_features: Optional traditional feature matrix for comparison

        Returns:
            Benchmarking results
        """
        print("\n" + "=" * 80)
        print("BENCHMARKING ANALYSIS")
        print("=" * 80)

        benchmarks = {}

        # Extract S-Entropy features
        sentropy_features = np.vstack([s.sentropy_features_14d for s in spectra])

        # Benchmark 1: Processing time
        benchmarks['processing_time'] = {
            'mean_time_per_spectrum': float(np.mean([s.processing_time for s in spectra])),
            'total_time': float(np.sum([s.processing_time for s in spectra])),
            'std_time': float(np.std([s.processing_time for s in spectra]))
        }

        print(f"\nProcessing Time:")
        print(f"  Mean: {benchmarks['processing_time']['mean_time_per_spectrum']:.6f} s/spectrum")
        print(f"  Total: {benchmarks['processing_time']['total_time']:.4f} s")

        # Benchmark 2: Feature quality
        benchmarks['feature_quality'] = self._evaluate_feature_quality(sentropy_features)

        print(f"\nFeature Quality:")
        print(f"  Mean variance: {benchmarks['feature_quality']['mean_variance']:.4f}")
        print(f"  Mean correlation: {benchmarks['feature_quality']['mean_correlation']:.4f}")

        # Benchmark 3: Comparison with traditional features (if provided)
        if traditional_features is not None:
            benchmarks['comparison'] = self._compare_with_traditional(
                sentropy_features, traditional_features
            )

            print(f"\nComparison with Traditional Features:")
            print(f"  S-Entropy silhouette: {benchmarks['comparison']['sentropy_silhouette']:.4f}")
            print(f"  Traditional silhouette: {benchmarks['comparison']['traditional_silhouette']:.4f}")
            print(f"  Improvement: {benchmarks['comparison']['improvement_percent']:.2f}%")

        # Benchmark 4: Dimensionality analysis
        benchmarks['dimensionality'] = self._analyze_dimensionality(sentropy_features)

        print(f"\nDimensionality Analysis:")
        print(f"  Explained variance (3 PCs): {benchmarks['dimensionality']['explained_variance_3pc']:.4f}")
        print(f"  Intrinsic dimensionality: {benchmarks['dimensionality']['intrinsic_dim']:.2f}")

        self.results['benchmarks'] = benchmarks

        print(f"\n{'=' * 80}\n")

        return benchmarks

    def _evaluate_feature_quality(self, features: np.ndarray) -> Dict:
        """Evaluate quality of feature representation"""
        # Variance per feature
        variances = np.var(features, axis=0)

        # Correlation matrix
        corr_matrix = np.corrcoef(features.T)

        # Mean absolute correlation (excluding diagonal)
        mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
        mean_corr = np.mean(np.abs(corr_matrix[mask]))

        return {
            'mean_variance': float(np.mean(variances)),
            'std_variance': float(np.std(variances)),
            'min_variance': float(np.min(variances)),
            'max_variance': float(np.max(variances)),
            'mean_correlation': float(mean_corr),
            'max_correlation': float(np.max(np.abs(corr_matrix[mask])))
        }

    def _compare_with_traditional(
            self,
            sentropy_features: np.ndarray,
            traditional_features: np.ndarray
    ) -> Dict:
        """Compare S-Entropy features with traditional features"""
        # Standardize both
        scaler = StandardScaler()
        sentropy_scaled = scaler.fit_transform(sentropy_features)
        traditional_scaled = scaler.fit_transform(traditional_features)

        # Cluster both with same parameters
        n_clusters = 5

        kmeans_sentropy = KMeans(n_clusters=n_clusters, random_state=42)
        labels_sentropy = kmeans_sentropy.fit_predict(sentropy_scaled)

        kmeans_traditional = KMeans(n_clusters=n_clusters, random_state=42)
        labels_traditional = kmeans_traditional.fit_predict(traditional_scaled)

        # Compute metrics
        sil_sentropy = silhouette_score(sentropy_scaled, labels_sentropy)
        sil_traditional = silhouette_score(traditional_scaled, labels_traditional)

        db_sentropy = davies_bouldin_score(sentropy_scaled, labels_sentropy)
        db_traditional = davies_bouldin_score(traditional_scaled, labels_traditional)

        ch_sentropy = calinski_harabasz_score(sentropy_scaled, labels_sentropy)
        ch_traditional = calinski_harabasz_score(traditional_scaled, labels_traditional)

        # Compute improvement
        improvement = ((sil_sentropy - sil_traditional) / sil_traditional) * 100

        return {
            'sentropy_silhouette': float(sil_sentropy),
            'traditional_silhouette': float(sil_traditional),
            'sentropy_davies_bouldin': float(db_sentropy),
            'traditional_davies_bouldin': float(db_traditional),
            'sentropy_calinski_harabasz': float(ch_sentropy),
            'traditional_calinski_harabasz': float(ch_traditional),
            'improvement_percent': float(improvement)
        }

    def _analyze_dimensionality(self, features: np.ndarray) -> Dict:
        """Analyze intrinsic dimensionality of features"""
        # PCA analysis
        pca = PCA()
        pca.fit(features)

        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)

        # Find number of components for 95% variance
        n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1

        # Estimate intrinsic dimensionality using correlation dimension
        intrinsic_dim = self._estimate_intrinsic_dimension(features)

        return {
            'explained_variance_1pc': float(explained_variance[0]),
            'explained_variance_3pc': float(cumulative_variance[2]),
            'explained_variance_5pc': float(cumulative_variance[4]),
            'n_components_95': int(n_components_95),
            'intrinsic_dim': float(intrinsic_dim),
            'all_explained_variance': explained_variance.tolist()
        }

    def _estimate_intrinsic_dimension(self, features: np.ndarray) -> float:
        """Estimate intrinsic dimensionality using MLE method"""
        # Sample subset for efficiency
        n_samples = min(1000, len(features))
        indices = np.random.choice(len(features), n_samples, replace=False)
        sample = features[indices]

        # Compute pairwise distances
        distances = pdist(sample, metric='euclidean')

        # Use MLE estimator
        k = 10  # Number of nearest neighbors
        distances_sorted = np.sort(distances)

        if len(distances_sorted) < k:
            return float(features.shape[1])

        r_k = distances_sorted[k - 1]
        r_1 = distances_sorted[0]

        if r_k > 0 and r_1 > 0:
            intrinsic_dim = k / np.log(r_k / r_1)
        else:
            intrinsic_dim = float(features.shape[1])

        return min(intrinsic_dim, features.shape[1])

    # ========================================================================
    # VALIDATION METRICS
    # ========================================================================

    def run_validation(self, spectra: List[SEntropySpectrum]) -> Dict:
        """
        Run comprehensive validation tests
        """
        print("\n" + "=" * 80)
        print("VALIDATION ANALYSIS")
        print("=" * 80)

        validation = {}

        # Validation 1: Coordinate consistency
        validation['coordinate_consistency'] = self._validate_coordinate_consistency(spectra)

        print(f"\nCoordinate Consistency:")
        print(f"  Mean coordinate magnitude: {validation['coordinate_consistency']['mean_magnitude']:.4f}")
        print(f"  Coordinate range: [{validation['coordinate_consistency']['min_coord']:.4f}, "
              f"{validation['coordinate_consistency']['max_coord']:.4f}]")

        # Validation 2: S-value distribution
        validation['s_value_distribution'] = self._validate_s_value_distribution(spectra)

        print(f"\nS-Value Distribution:")
        print(f"  Mean: {validation['s_value_distribution']['mean']:.4f}")
        print(f"  Std: {validation['s_value_distribution']['std']:.4f}")

        # Validation 3: Feature stability
        validation['feature_stability'] = self._validate_feature_stability(spectra)

        print(f"\nFeature Stability:")
        print(f"  Mean CV: {validation['feature_stability']['mean_cv']:.4f}")

        # Validation 4: Sequence encoding validation (if sequences available)
        sequences_available = sum(1 for s in spectra if s.peptide_sequence is not None)
        if sequences_available > 0:
            validation['sequence_encoding'] = self._validate_sequence_encoding(spectra)

            print(f"\nSequence Encoding:")
            print(f"  Sequences encoded: {sequences_available}")
            print(f"  Mean path length: {validation['sequence_encoding']['mean_path_length']:.4f}")

        self.results['validation'] = validation

        print(f"\n{'=' * 80}\n")

        return validation

    def _validate_coordinate_consistency(self, spectra: List[SEntropySpectrum]) -> Dict:
        """Validate that coordinates are consistent and well-formed"""
        all_coords = []

        for spectrum in spectra:
            if len(spectrum.sentropy_coords_3d) > 0:
                all_coords.append(spectrum.sentropy_coords_3d)

        if not all_coords:
            return {}

        all_coords = np.vstack(all_coords)

        # Compute statistics
        magnitudes = np.linalg.norm(all_coords, axis=1)

        return {
            'mean_magnitude': float(np.mean(magnitudes)),
            'std_magnitude': float(np.std(magnitudes)),
            'min_coord': float(np.min(all_coords)),
            'max_coord': float(np.max(all_coords)),
            'n_coordinates': len(all_coords)
        }

    def _validate_s_value_distribution(self, spectra: List[SEntropySpectrum]) -> Dict:
        """Validate S-value distribution"""
        s_values = [s.s_value for s in spectra]

        return {
            'mean': float(np.mean(s_values)),
            'std': float(np.std(s_values)),
            'min': float(np.min(s_values)),
            'max': float(np.max(s_values)),
            'median': float(np.median(s_values)),
            'q25': float(np.percentile(s_values, 25)),
            'q75': float(np.percentile(s_values, 75))
        }

    def _validate_feature_stability(self, spectra: List[SEntropySpectrum]) -> Dict:
        """Validate feature stability across spectra"""
        features = np.vstack([s.sentropy_features_14d for s in spectra])

        # Coefficient of variation for each feature
        means = np.mean(features, axis=0)
        stds = np.std(features, axis=0)
        cvs = stds / (means + 1e-10)

        return {
            'mean_cv': float(np.mean(cvs)),
            'max_cv': float(np.max(cvs)),
            'min_cv': float(np.min(cvs)),
            'all_cvs': cvs.tolist()
        }

    def _validate_sequence_encoding(self, spectra: List[SEntropySpectrum]) -> Dict:
        """Validate peptide sequence encoding"""
        path_lengths = []
        endpoint_distances = []

        for spectrum in spectra:
            if spectrum.sequence_sentropy_coords is not None:
                coords = spectrum.sequence_sentropy_coords

                # Path length
                if len(coords) > 1:
                    path_length = np.sum(np.linalg.norm(np.diff(coords, axis=0), axis=1))
                    path_lengths.append(path_length)

                    # Endpoint distance
                    endpoint_dist = np.linalg.norm(coords[-1] - coords[0])
                    endpoint_distances.append(endpoint_dist)

        if not path_lengths:
            return {}

        return {
            'mean_path_length': float(np.mean(path_lengths)),
            'std_path_length': float(np.std(path_lengths)),
            'mean_endpoint_distance': float(np.mean(endpoint_distances)),
            'mean_tortuosity': float(np.mean(np.array(path_lengths) / (np.array(endpoint_distances) + 1e-10)))
        }

    # ========================================================================
    # SAVE RESULTS
    # ========================================================================

    def save_results(self):
        """Save all validation results"""
        # Save JSON
        output_file = self.output_dir / 'validation_results.json'
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"Saved validation results: {output_file}")

        # Save summary CSV
        self._save_summary_csv()

    def _save_summary_csv(self):
        """Save summary statistics as CSV"""
        summary_data = []

        # Clustering summary
        if 'clustering' in self.results and 'best' in self.results['clustering']:
            best = self.results['clustering']['best']
            summary_data.append({
                'metric': 'best_clustering_method',
                'value': best['method']
            })
            summary_data.append({
                'metric': 'best_clustering_n_clusters',
                'value': best['n_clusters']
            })
            summary_data.append({
                'metric': 'best_clustering_silhouette',
                'value': best['silhouette']
            })

        # Benchmarking summary
        if 'benchmarks' in self.results:
            if 'processing_time' in self.results['benchmarks']:
                summary_data.append({
                    'metric': 'mean_processing_time',
                    'value': self.results['benchmarks']['processing_time']['mean_time_per_spectrum']
                })

            if 'comparison' in self.results['benchmarks']:
                summary_data.append({
                    'metric': 'improvement_vs_traditional',
                    'value': self.results['benchmarks']['comparison']['improvement_percent']
                })

        # Validation summary
        if 'validation' in self.results:
            if 's_value_distribution' in self.results['validation']:
                summary_data.append({
                    'metric': 'mean_s_value',
                    'value': self.results['validation']['s_value_distribution']['mean']
                })

        df = pd.DataFrame(summary_data)
        output_file = self.output_dir / 'validation_summary.csv'
        df.to_csv(output_file, index=False)

        print(f"Saved summary CSV: {output_file}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Example usage of validation pipeline
    """
    from SEntropy_Proteomics_Core import SEntropyProteomicsEngine

    # Initialize engine
    engine = SEntropyProteomicsEngine()

    # Initialize validation pipeline
    validation_pipeline = SEntropyValidationPipeline(
        engine=engine,
        output_dir='results/validation'
    )

    # TODO: Load your spectra here
    # spectra = engine.process_mzml_file('path/to/data.mzml')

    # For now, create dummy data
    print("Creating dummy data for demonstration...")
    spectra = []
    for i in range(100):
        spectrum = engine.process_spectrum(
            mz_array=np.random.rand(50) * 1000,
            intensity_array=np.random.rand(50) * 1e6,
            precursor_mz=500.0 + np.random.rand() * 500,
            precursor_charge=2,
            spectrum_id=f"spectrum_{i:04d}",
            peptide_sequence="PEPTIDE" if i % 2 == 0 else None,
            retention_time=10.0 + i * 0.5
        )
        spectra.append(spectrum)

    print(f"Generated {len(spectra)} dummy spectra")

    # Run validation pipeline
    clustering_results = validation_pipeline.run_clustering_analysis(spectra)
    benchmark_results = validation_pipeline.run_benchmarking(spectra)
    validation_results = validation_pipeline.run_validation(spectra)

    # Save results
    validation_pipeline.save_results()

    print("\n" + "=" * 80)
    print("VALIDATION PIPELINE COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
