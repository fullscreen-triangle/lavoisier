"""
Coverage Assessment Module

Comprehensive assessment of data coverage including spectral coverage,
feature space coverage, and temporal coverage analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans


@dataclass
class CoverageResult:
    """Container for coverage assessment results"""
    coverage_type: str
    coverage_score: float  # 0-1 score
    coverage_percentage: float
    gaps_detected: int
    interpretation: str
    metadata: Dict[str, Any] = None


class CoverageAssessment:
    """
    Comprehensive coverage assessment for various data types
    """
    
    def __init__(self):
        """Initialize coverage assessment"""
        self.results = []
        
    def assess_spectral_coverage(
        self,
        mz_values: np.ndarray,
        expected_range: Tuple[float, float],
        resolution: float = 0.1
    ) -> CoverageResult:
        """
        Assess spectral coverage for mass spectrometry data
        
        Args:
            mz_values: m/z values from spectra
            expected_range: Expected (min, max) m/z range
            resolution: Expected resolution for coverage assessment
            
        Returns:
            CoverageResult object
        """
        min_mz, max_mz = expected_range
        
        # Create expected m/z grid
        expected_mz = np.arange(min_mz, max_mz + resolution, resolution)
        
        # Find coverage
        covered_points = 0
        gaps = []
        
        for expected_point in expected_mz:
            # Check if any actual m/z value is within resolution of expected point
            distances = np.abs(mz_values - expected_point)
            if np.min(distances) <= resolution:
                covered_points += 1
            else:
                gaps.append(expected_point)
        
        coverage_percentage = (covered_points / len(expected_mz)) * 100
        coverage_score = covered_points / len(expected_mz)
        
        # Identify significant gaps (consecutive missing points)
        significant_gaps = 0
        if gaps:
            gap_array = np.array(gaps)
            gap_diffs = np.diff(gap_array)
            consecutive_gaps = np.where(gap_diffs <= resolution * 1.1)[0]
            
            # Count groups of consecutive gaps
            if len(consecutive_gaps) > 0:
                gap_groups = []
                current_group = [consecutive_gaps[0]]
                
                for i in range(1, len(consecutive_gaps)):
                    if consecutive_gaps[i] == consecutive_gaps[i-1] + 1:
                        current_group.append(consecutive_gaps[i])
                    else:
                        if len(current_group) >= 3:  # Significant if 3+ consecutive missing
                            gap_groups.append(current_group)
                        current_group = [consecutive_gaps[i]]
                
                if len(current_group) >= 3:
                    gap_groups.append(current_group)
                
                significant_gaps = len(gap_groups)
        
        # Interpretation
        if coverage_score > 0.95:
            interpretation = f"Excellent spectral coverage ({coverage_percentage:.1f}%)"
        elif coverage_score > 0.9:
            interpretation = f"Good spectral coverage ({coverage_percentage:.1f}%)"
        elif coverage_score > 0.8:
            interpretation = f"Moderate spectral coverage ({coverage_percentage:.1f}%)"
        else:
            interpretation = f"Poor spectral coverage ({coverage_percentage:.1f}%)"
        
        if significant_gaps > 0:
            interpretation += f" with {significant_gaps} significant gaps"
        
        result = CoverageResult(
            coverage_type="Spectral Coverage",
            coverage_score=coverage_score,
            coverage_percentage=coverage_percentage,
            gaps_detected=len(gaps),
            interpretation=interpretation,
            metadata={
                'expected_range': expected_range,
                'resolution': resolution,
                'total_expected_points': len(expected_mz),
                'covered_points': covered_points,
                'significant_gaps': significant_gaps
            }
        )
        
        self.results.append(result)
        return result
    
    def assess_feature_space_coverage(
        self,
        features: np.ndarray,
        n_clusters: int = 10
    ) -> CoverageResult:
        """
        Assess coverage of feature space using clustering analysis
        
        Args:
            features: Feature matrix (samples x features)
            n_clusters: Number of clusters for coverage assessment
            
        Returns:
            CoverageResult object
        """
        # Perform clustering to identify feature space regions
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features)
        
        # Calculate coverage metrics
        unique_clusters = len(np.unique(cluster_labels))
        cluster_coverage = unique_clusters / n_clusters
        
        # Calculate cluster sizes
        cluster_sizes = []
        for i in range(n_clusters):
            cluster_size = np.sum(cluster_labels == i)
            cluster_sizes.append(cluster_size)
        
        # Identify empty or sparse clusters
        min_expected_size = len(features) / (n_clusters * 2)  # At least half the average
        sparse_clusters = sum(1 for size in cluster_sizes if size < min_expected_size)
        empty_clusters = sum(1 for size in cluster_sizes if size == 0)
        
        # Calculate uniformity of coverage
        cluster_sizes_array = np.array(cluster_sizes)
        if len(cluster_sizes_array) > 0:
            cv_clusters = np.std(cluster_sizes_array) / np.mean(cluster_sizes_array) if np.mean(cluster_sizes_array) > 0 else float('inf')
            uniformity_score = max(0.0, 1.0 - cv_clusters)
        else:
            uniformity_score = 0.0
        
        # Overall coverage score combines cluster coverage and uniformity
        coverage_score = 0.7 * cluster_coverage + 0.3 * uniformity_score
        coverage_percentage = coverage_score * 100
        
        # Interpretation
        if coverage_score > 0.9:
            interpretation = f"Excellent feature space coverage ({coverage_percentage:.1f}%)"
        elif coverage_score > 0.8:
            interpretation = f"Good feature space coverage ({coverage_percentage:.1f}%)"
        elif coverage_score > 0.6:
            interpretation = f"Moderate feature space coverage ({coverage_percentage:.1f}%)"
        else:
            interpretation = f"Poor feature space coverage ({coverage_percentage:.1f}%)"
        
        if empty_clusters > 0:
            interpretation += f" with {empty_clusters} empty regions"
        elif sparse_clusters > 0:
            interpretation += f" with {sparse_clusters} sparse regions"
        
        result = CoverageResult(
            coverage_type="Feature Space Coverage",
            coverage_score=coverage_score,
            coverage_percentage=coverage_percentage,
            gaps_detected=empty_clusters + sparse_clusters,
            interpretation=interpretation,
            metadata={
                'n_clusters': n_clusters,
                'unique_clusters': unique_clusters,
                'cluster_sizes': cluster_sizes,
                'empty_clusters': empty_clusters,
                'sparse_clusters': sparse_clusters,
                'uniformity_score': uniformity_score
            }
        )
        
        self.results.append(result)
        return result
    
    def assess_temporal_coverage(
        self,
        timestamps: np.ndarray,
        expected_interval: Optional[float] = None,
        tolerance: float = 0.1
    ) -> CoverageResult:
        """
        Assess temporal coverage for time-series data
        
        Args:
            timestamps: Array of timestamps
            expected_interval: Expected time interval between samples
            tolerance: Tolerance for interval variations (fraction)
            
        Returns:
            CoverageResult object
        """
        if len(timestamps) < 2:
            return CoverageResult(
                coverage_type="Temporal Coverage",
                coverage_score=0.0,
                coverage_percentage=0.0,
                gaps_detected=0,
                interpretation="Insufficient data for temporal coverage assessment"
            )
        
        # Sort timestamps
        sorted_timestamps = np.sort(timestamps)
        intervals = np.diff(sorted_timestamps)
        
        # Estimate expected interval if not provided
        if expected_interval is None:
            expected_interval = np.median(intervals)
        
        # Find gaps and irregular intervals
        gap_threshold = expected_interval * (1 + tolerance)
        gaps = intervals > gap_threshold
        num_gaps = np.sum(gaps)
        
        # Calculate coverage
        total_time = sorted_timestamps[-1] - sorted_timestamps[0]
        expected_samples = int(total_time / expected_interval) + 1
        actual_samples = len(timestamps)
        
        coverage_score = min(1.0, actual_samples / expected_samples)
        coverage_percentage = coverage_score * 100
        
        # Identify significant gaps
        significant_gaps = []
        for i, is_gap in enumerate(gaps):
            if is_gap:
                gap_size = intervals[i] / expected_interval
                if gap_size > 2.0:  # More than 2x expected interval
                    significant_gaps.append({
                        'start_time': sorted_timestamps[i],
                        'end_time': sorted_timestamps[i+1],
                        'duration': intervals[i],
                        'missing_samples': int(gap_size) - 1
                    })
        
        # Calculate regularity
        interval_cv = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else float('inf')
        regularity_score = max(0.0, 1.0 - interval_cv)
        
        # Adjust coverage score for regularity
        adjusted_coverage_score = 0.8 * coverage_score + 0.2 * regularity_score
        
        # Interpretation
        if adjusted_coverage_score > 0.95:
            interpretation = f"Excellent temporal coverage ({coverage_percentage:.1f}%)"
        elif adjusted_coverage_score > 0.9:
            interpretation = f"Good temporal coverage ({coverage_percentage:.1f}%)"
        elif adjusted_coverage_score > 0.8:
            interpretation = f"Moderate temporal coverage ({coverage_percentage:.1f}%)"
        else:
            interpretation = f"Poor temporal coverage ({coverage_percentage:.1f}%)"
        
        if len(significant_gaps) > 0:
            total_missing = sum(gap['missing_samples'] for gap in significant_gaps)
            interpretation += f" with {len(significant_gaps)} significant gaps ({total_missing} missing samples)"
        
        result = CoverageResult(
            coverage_type="Temporal Coverage",
            coverage_score=adjusted_coverage_score,
            coverage_percentage=coverage_percentage,
            gaps_detected=num_gaps,
            interpretation=interpretation,
            metadata={
                'expected_interval': expected_interval,
                'actual_samples': actual_samples,
                'expected_samples': expected_samples,
                'significant_gaps': significant_gaps,
                'regularity_score': regularity_score,
                'interval_cv': interval_cv
            }
        )
        
        self.results.append(result)
        return result
    
    def assess_intensity_coverage(
        self,
        intensities: np.ndarray,
        dynamic_range_db: float = 60.0
    ) -> CoverageResult:
        """
        Assess intensity coverage across dynamic range
        
        Args:
            intensities: Intensity values
            dynamic_range_db: Expected dynamic range in dB
            
        Returns:
            CoverageResult object
        """
        # Remove zero and negative intensities
        valid_intensities = intensities[intensities > 0]
        
        if len(valid_intensities) == 0:
            return CoverageResult(
                coverage_type="Intensity Coverage",
                coverage_score=0.0,
                coverage_percentage=0.0,
                gaps_detected=0,
                interpretation="No valid intensity values found"
            )
        
        # Convert to log scale
        log_intensities = np.log10(valid_intensities)
        
        # Define intensity bins across dynamic range
        max_intensity = np.max(log_intensities)
        min_intensity = max_intensity - (dynamic_range_db / 10.0)  # Convert dB to log10
        
        # Create bins
        n_bins = 20
        bin_edges = np.linspace(min_intensity, max_intensity, n_bins + 1)
        bin_counts, _ = np.histogram(log_intensities, bins=bin_edges)
        
        # Calculate coverage
        occupied_bins = np.sum(bin_counts > 0)
        coverage_score = occupied_bins / n_bins
        coverage_percentage = coverage_score * 100
        
        # Identify gaps
        empty_bins = n_bins - occupied_bins
        
        # Calculate distribution uniformity
        if occupied_bins > 0:
            occupied_counts = bin_counts[bin_counts > 0]
            cv_bins = np.std(occupied_counts) / np.mean(occupied_counts)
            uniformity_score = max(0.0, 1.0 - cv_bins)
        else:
            uniformity_score = 0.0
        
        # Adjust coverage for uniformity
        adjusted_coverage_score = 0.8 * coverage_score + 0.2 * uniformity_score
        
        # Interpretation
        if adjusted_coverage_score > 0.9:
            interpretation = f"Excellent intensity coverage ({coverage_percentage:.1f}%)"
        elif adjusted_coverage_score > 0.8:
            interpretation = f"Good intensity coverage ({coverage_percentage:.1f}%)"
        elif adjusted_coverage_score > 0.6:
            interpretation = f"Moderate intensity coverage ({coverage_percentage:.1f}%)"
        else:
            interpretation = f"Poor intensity coverage ({coverage_percentage:.1f}%)"
        
        if empty_bins > 0:
            interpretation += f" with {empty_bins} intensity gaps"
        
        result = CoverageResult(
            coverage_type="Intensity Coverage",
            coverage_score=adjusted_coverage_score,
            coverage_percentage=coverage_percentage,
            gaps_detected=empty_bins,
            interpretation=interpretation,
            metadata={
                'dynamic_range_db': dynamic_range_db,
                'n_bins': n_bins,
                'occupied_bins': occupied_bins,
                'bin_counts': bin_counts.tolist(),
                'uniformity_score': uniformity_score,
                'intensity_range': (np.min(valid_intensities), np.max(valid_intensities))
            }
        )
        
        self.results.append(result)
        return result
    
    def comprehensive_coverage_assessment(
        self,
        data: Dict[str, np.ndarray],
        **kwargs
    ) -> List[CoverageResult]:
        """
        Run comprehensive coverage assessment
        
        Args:
            data: Dictionary containing different data types
            **kwargs: Additional parameters for specific assessments
            
        Returns:
            List of CoverageResult objects
        """
        results = []
        
        # Spectral coverage
        if 'mz_values' in data:
            expected_range = kwargs.get('mz_range', (50, 1000))
            resolution = kwargs.get('mz_resolution', 0.1)
            results.append(self.assess_spectral_coverage(
                data['mz_values'], expected_range, resolution
            ))
        
        # Feature space coverage
        if 'features' in data:
            n_clusters = kwargs.get('n_clusters', 10)
            results.append(self.assess_feature_space_coverage(
                data['features'], n_clusters
            ))
        
        # Temporal coverage
        if 'timestamps' in data:
            expected_interval = kwargs.get('expected_interval')
            results.append(self.assess_temporal_coverage(
                data['timestamps'], expected_interval
            ))
        
        # Intensity coverage
        if 'intensities' in data:
            dynamic_range = kwargs.get('dynamic_range_db', 60.0)
            results.append(self.assess_intensity_coverage(
                data['intensities'], dynamic_range
            ))
        
        return results
    
    def generate_coverage_report(self) -> pd.DataFrame:
        """Generate comprehensive coverage report"""
        if not self.results:
            return pd.DataFrame()
        
        data = []
        for result in self.results:
            data.append({
                'Coverage Type': result.coverage_type,
                'Score': result.coverage_score,
                'Percentage': result.coverage_percentage,
                'Gaps': result.gaps_detected,
                'Interpretation': result.interpretation
            })
        
        return pd.DataFrame(data)
    
    def get_overall_coverage_score(self) -> float:
        """Calculate overall coverage score across all assessments"""
        if not self.results:
            return 0.0
        
        return np.mean([r.coverage_score for r in self.results])
    
    def identify_critical_gaps(self) -> List[Dict[str, Any]]:
        """Identify critical coverage gaps across all assessments"""
        critical_gaps = []
        
        for result in self.results:
            if result.coverage_score < 0.8:  # Below 80% coverage
                critical_gaps.append({
                    'coverage_type': result.coverage_type,
                    'score': result.coverage_score,
                    'gaps': result.gaps_detected,
                    'severity': 'Critical' if result.coverage_score < 0.6 else 'Moderate'
                })
        
        return critical_gaps
    
    def clear_results(self) -> None:
        """Clear all stored results"""
        self.results = [] 