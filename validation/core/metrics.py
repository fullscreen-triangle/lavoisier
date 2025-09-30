"""
Performance Metrics for Validation Framework

Comprehensive metrics for evaluating method performance across different aspects.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from scipy import stats
import logging

@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    # Identification metrics
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    
    # Confidence metrics
    mean_confidence: float
    std_confidence: float
    confidence_reliability: float
    
    # Processing metrics
    processing_time: float
    memory_usage: float
    throughput: float  # spectra per second
    
    # Coverage metrics
    molecular_coverage: float
    database_hit_rate: float
    novel_detection_rate: float
    
    # Quality metrics
    signal_to_noise: float
    spectral_quality: float
    identification_quality: float
    
    # Method-specific metrics
    custom_metrics: Dict[str, float]

@dataclass
class ValidationMetrics:
    """Container for validation-specific metrics"""
    # Statistical validation
    statistical_significance: bool
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    
    # Robustness metrics
    cross_validation_score: float
    generalization_error: float
    stability_score: float
    
    # Comparison metrics
    improvement_over_baseline: float
    relative_performance: float
    ranking_score: float

class MetricsCalculator:
    """Calculator for comprehensive performance metrics"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_identification_metrics(self, 
                                       predictions: List[str],
                                       ground_truth: List[str],
                                       confidence_scores: List[float]) -> Dict[str, float]:
        """
        Calculate identification performance metrics
        
        Args:
            predictions: Predicted molecular identifications
            ground_truth: True molecular identifications  
            confidence_scores: Confidence scores for predictions
            
        Returns:
            Dictionary of identification metrics
        """
        if len(predictions) != len(ground_truth):
            self.logger.warning("Length mismatch between predictions and ground truth")
            return {}
        
        # Basic accuracy
        accuracy = accuracy_score(ground_truth, predictions)
        
        # For precision/recall, we need to handle multi-class case
        # Create binary classification for each unique molecule
        unique_molecules = list(set(ground_truth + predictions))
        
        if len(unique_molecules) > 1:
            # Multi-class precision/recall
            precision, recall, f1, _ = precision_recall_fscore_support(
                ground_truth, predictions, average='weighted', zero_division=0
            )
        else:
            precision, recall, f1 = 1.0, 1.0, 1.0
        
        # Confidence metrics
        mean_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        std_confidence = np.std(confidence_scores) if confidence_scores else 0.0
        
        # Confidence reliability (correlation between confidence and correctness)
        if confidence_scores:
            correctness = [1.0 if pred == truth else 0.0 for pred, truth in zip(predictions, ground_truth)]
            confidence_reliability = np.corrcoef(confidence_scores, correctness)[0, 1]
            if np.isnan(confidence_reliability):
                confidence_reliability = 0.0
        else:
            confidence_reliability = 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'mean_confidence': mean_confidence,
            'std_confidence': std_confidence,
            'confidence_reliability': confidence_reliability
        }
    
    def calculate_coverage_metrics(self,
                                 identifications: List[str],
                                 database_size: int,
                                 total_detectable: int) -> Dict[str, float]:
        """
        Calculate coverage metrics
        
        Args:
            identifications: List of identified molecules
            database_size: Size of reference database
            total_detectable: Total number of detectable compounds
            
        Returns:
            Dictionary of coverage metrics
        """
        unique_identifications = set(identifications)
        n_identified = len(unique_identifications)
        
        # Molecular coverage
        molecular_coverage = n_identified / total_detectable if total_detectable > 0 else 0.0
        
        # Database hit rate
        database_hit_rate = n_identified / database_size if database_size > 0 else 0.0
        
        # Novel detection rate (assume some identifications are novel)
        # This would need ground truth about what's novel
        novel_detection_rate = 0.1  # Placeholder
        
        return {
            'molecular_coverage': molecular_coverage,
            'database_hit_rate': database_hit_rate,
            'novel_detection_rate': novel_detection_rate,
            'unique_identifications': n_identified,
            'total_identifications': len(identifications)
        }
    
    def calculate_processing_metrics(self,
                                   processing_times: List[float],
                                   memory_usage: List[float],
                                   n_spectra_processed: int) -> Dict[str, float]:
        """
        Calculate processing performance metrics
        
        Args:
            processing_times: List of processing times
            memory_usage: List of memory usage measurements
            n_spectra_processed: Number of spectra processed
            
        Returns:
            Dictionary of processing metrics
        """
        mean_processing_time = np.mean(processing_times) if processing_times else 0.0
        std_processing_time = np.std(processing_times) if processing_times else 0.0
        
        mean_memory = np.mean(memory_usage) if memory_usage else 0.0
        
        # Throughput (spectra per second)
        throughput = n_spectra_processed / mean_processing_time if mean_processing_time > 0 else 0.0
        
        return {
            'mean_processing_time': mean_processing_time,
            'std_processing_time': std_processing_time,
            'mean_memory_usage': mean_memory,
            'throughput': throughput,
            'total_spectra': n_spectra_processed
        }
    
    def calculate_quality_metrics(self,
                                spectra_data: List[Any],
                                identifications: List[str],
                                confidence_scores: List[float]) -> Dict[str, float]:
        """
        Calculate data and identification quality metrics
        
        Args:
            spectra_data: Raw spectral data
            identifications: Molecular identifications
            confidence_scores: Identification confidence scores
            
        Returns:
            Dictionary of quality metrics
        """
        # Signal-to-noise ratio (simplified calculation)
        signal_to_noise = 10.0  # Placeholder - would calculate from spectral data
        
        # Spectral quality (based on peak count, intensity distribution, etc.)
        spectral_quality = 0.85  # Placeholder
        
        # Identification quality (based on confidence score distribution)
        if confidence_scores:
            high_confidence_rate = sum(1 for score in confidence_scores if score > 0.8) / len(confidence_scores)
            identification_quality = high_confidence_rate
        else:
            identification_quality = 0.0
        
        return {
            'signal_to_noise': signal_to_noise,
            'spectral_quality': spectral_quality,
            'identification_quality': identification_quality
        }
    
    def calculate_stellas_specific_metrics(self,
                                         original_results: Dict[str, Any],
                                         stellas_results: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate S-Stellas framework specific metrics
        
        Args:
            original_results: Results without S-Stellas transformation
            stellas_results: Results with S-Stellas transformation
            
        Returns:
            Dictionary of S-Stellas specific metrics
        """
        # Information access improvement
        info_access_improvement = stellas_results.get('information_access', 5.0) - original_results.get('information_access', 5.0)
        
        # Accuracy improvement
        accuracy_improvement = stellas_results.get('accuracy', 0.0) - original_results.get('accuracy', 0.0)
        
        # S-Entropy coordinate transformation quality
        coordinate_fidelity = stellas_results.get('coordinate_fidelity', 0.95)
        
        # Network convergence (for SENN)
        network_convergence = stellas_results.get('network_convergence', 0.9)
        
        # Cross-modal validation success
        cross_modal_validation = stellas_results.get('bmd_validation_success', 0.8)
        
        return {
            'information_access_improvement': info_access_improvement,
            'accuracy_improvement': accuracy_improvement,
            'coordinate_transformation_fidelity': coordinate_fidelity,
            'senn_network_convergence': network_convergence,
            'cross_modal_validation_success': cross_modal_validation,
            'overall_stellas_benefit': (info_access_improvement + accuracy_improvement + coordinate_fidelity) / 3
        }
    
    def calculate_statistical_validation(self,
                                       method_a_scores: List[float],
                                       method_b_scores: List[float],
                                       alpha: float = 0.05) -> ValidationMetrics:
        """
        Calculate statistical validation metrics for method comparison
        
        Args:
            method_a_scores: Performance scores for method A
            method_b_scores: Performance scores for method B
            alpha: Significance level
            
        Returns:
            ValidationMetrics object
        """
        # Paired t-test
        if len(method_a_scores) == len(method_b_scores):
            t_stat, p_value = stats.ttest_rel(method_b_scores, method_a_scores)
        else:
            t_stat, p_value = stats.ttest_ind(method_b_scores, method_a_scores)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(method_a_scores) + np.var(method_b_scores)) / 2)
        effect_size = (np.mean(method_b_scores) - np.mean(method_a_scores)) / pooled_std if pooled_std > 0 else 0.0
        
        # Confidence interval for difference
        diff_scores = np.array(method_b_scores) - np.array(method_a_scores[:len(method_b_scores)])
        confidence_interval = stats.t.interval(
            1 - alpha, len(diff_scores) - 1,
            loc=np.mean(diff_scores), scale=stats.sem(diff_scores)
        )
        
        # Statistical significance
        statistical_significance = p_value < alpha
        
        # Improvement over baseline
        improvement = (np.mean(method_b_scores) - np.mean(method_a_scores)) / np.mean(method_a_scores) * 100 if np.mean(method_a_scores) > 0 else 0.0
        
        return ValidationMetrics(
            statistical_significance=statistical_significance,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=confidence_interval,
            cross_validation_score=0.8,  # Placeholder
            generalization_error=0.15,   # Placeholder
            stability_score=0.9,         # Placeholder
            improvement_over_baseline=improvement,
            relative_performance=np.mean(method_b_scores) / np.mean(method_a_scores) if np.mean(method_a_scores) > 0 else 1.0,
            ranking_score=0.85           # Placeholder
        )
    
    def create_performance_report(self,
                                results: Dict[str, Any],
                                method_name: str,
                                dataset_name: str) -> PerformanceMetrics:
        """
        Create comprehensive performance metrics report
        
        Args:
            results: Method results dictionary
            method_name: Name of the method
            dataset_name: Name of the dataset
            
        Returns:
            PerformanceMetrics object
        """
        # Extract data from results
        predictions = results.get('predictions', [])
        ground_truth = results.get('ground_truth', [])
        confidence_scores = results.get('confidence_scores', [])
        processing_times = results.get('processing_times', [])
        memory_usage = results.get('memory_usage', [])
        
        # Calculate different metric categories
        id_metrics = self.calculate_identification_metrics(predictions, ground_truth, confidence_scores)
        coverage_metrics = self.calculate_coverage_metrics(predictions, 1000, 800)  # Placeholder values
        processing_metrics = self.calculate_processing_metrics(processing_times, memory_usage, len(predictions))
        quality_metrics = self.calculate_quality_metrics([], predictions, confidence_scores)
        
        # Custom metrics from method
        custom_metrics = results.get('custom_metrics', {})
        
        return PerformanceMetrics(
            accuracy=id_metrics.get('accuracy', 0.0),
            precision=id_metrics.get('precision', 0.0),
            recall=id_metrics.get('recall', 0.0),
            f1_score=id_metrics.get('f1_score', 0.0),
            mean_confidence=id_metrics.get('mean_confidence', 0.0),
            std_confidence=id_metrics.get('std_confidence', 0.0),
            confidence_reliability=id_metrics.get('confidence_reliability', 0.0),
            processing_time=processing_metrics.get('mean_processing_time', 0.0),
            memory_usage=processing_metrics.get('mean_memory_usage', 0.0),
            throughput=processing_metrics.get('throughput', 0.0),
            molecular_coverage=coverage_metrics.get('molecular_coverage', 0.0),
            database_hit_rate=coverage_metrics.get('database_hit_rate', 0.0),
            novel_detection_rate=coverage_metrics.get('novel_detection_rate', 0.0),
            signal_to_noise=quality_metrics.get('signal_to_noise', 0.0),
            spectral_quality=quality_metrics.get('spectral_quality', 0.0),
            identification_quality=quality_metrics.get('identification_quality', 0.0),
            custom_metrics=custom_metrics
        )
    
    def compare_methods(self,
                       baseline_metrics: PerformanceMetrics,
                       comparison_metrics: PerformanceMetrics) -> Dict[str, float]:
        """
        Compare two sets of performance metrics
        
        Args:
            baseline_metrics: Baseline method performance
            comparison_metrics: Comparison method performance
            
        Returns:
            Dictionary of comparison results
        """
        comparisons = {}
        
        # Calculate improvements
        metrics_to_compare = [
            'accuracy', 'precision', 'recall', 'f1_score',
            'mean_confidence', 'throughput', 'molecular_coverage'
        ]
        
        for metric in metrics_to_compare:
            baseline_value = getattr(baseline_metrics, metric)
            comparison_value = getattr(comparison_metrics, metric)
            
            if baseline_value > 0:
                improvement = (comparison_value - baseline_value) / baseline_value * 100
                comparisons[f'{metric}_improvement_percent'] = improvement
                comparisons[f'{metric}_ratio'] = comparison_value / baseline_value
            else:
                comparisons[f'{metric}_improvement_percent'] = 0.0
                comparisons[f'{metric}_ratio'] = 1.0
        
        # Overall performance score
        performance_weights = {
            'accuracy': 0.3,
            'f1_score': 0.25,
            'molecular_coverage': 0.2,
            'throughput': 0.1,
            'mean_confidence': 0.15
        }
        
        baseline_score = sum(
            getattr(baseline_metrics, metric) * weight 
            for metric, weight in performance_weights.items()
        )
        
        comparison_score = sum(
            getattr(comparison_metrics, metric) * weight 
            for metric, weight in performance_weights.items()
        )
        
        comparisons['overall_performance_improvement'] = (
            (comparison_score - baseline_score) / baseline_score * 100 if baseline_score > 0 else 0.0
        )
        
        return comparisons
