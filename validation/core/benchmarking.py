"""
Benchmarking Framework for Method Comparison

Provides systematic comparison of numerical, vision, and S-Stellas methods.
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Type
from dataclasses import dataclass
from pathlib import Path
import logging
import json
from datetime import datetime

from .base_validator import BaseValidator, ValidationResult
from .metrics import PerformanceMetrics, ValidationMetrics, MetricsCalculator
from .data_loader import MZMLDataLoader, SpectrumInfo, DatasetInfo

@dataclass
class BenchmarkResult:
    """Container for benchmark results"""
    method_name: str
    dataset_name: str
    with_stellas: bool
    performance_metrics: PerformanceMetrics
    validation_metrics: Optional[ValidationMetrics]
    processing_stats: Dict[str, float]
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'method_name': self.method_name,
            'dataset_name': self.dataset_name,
            'with_stellas': self.with_stellas,
            'performance_metrics': self.performance_metrics.__dict__,
            'validation_metrics': self.validation_metrics.__dict__ if self.validation_metrics else None,
            'processing_stats': self.processing_stats,
            'timestamp': self.timestamp
        }

class BenchmarkRunner:
    """Runs comprehensive benchmarks across methods and datasets"""
    
    def __init__(self, output_directory: str = "validation/results"):
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(exist_ok=True)
        
        self.data_loader = MZMLDataLoader()
        self.metrics_calculator = MetricsCalculator()
        
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging for benchmark runs"""
        log_file = self.output_directory / "benchmark.log"
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(file_handler)
    
    def run_method_benchmark(self,
                           validator: BaseValidator,
                           dataset_name: str,
                           with_stellas: bool = False,
                           iterations: int = 3) -> BenchmarkResult:
        """
        Run benchmark for a single method on a dataset
        
        Args:
            validator: Method validator instance
            dataset_name: Name of dataset to test
            with_stellas: Whether to use S-Stellas transformation
            iterations: Number of iterations for timing
            
        Returns:
            BenchmarkResult with performance metrics
        """
        self.logger.info(f"Running benchmark: {validator.method_name} on {dataset_name} "
                        f"(S-Stellas: {with_stellas})")
        
        # Load dataset
        try:
            spectra, dataset_info = self.data_loader.load_dataset(dataset_name)
            if not spectra:
                raise ValueError(f"No spectra loaded from {dataset_name}")
                
            self.logger.info(f"Loaded {len(spectra)} spectra from {dataset_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to load dataset {dataset_name}: {e}")
            return self._create_error_result(validator.method_name, dataset_name, with_stellas)
        
        # Run multiple iterations for timing
        processing_times = []
        memory_usage = []
        results = []
        
        for iteration in range(iterations):
            self.logger.info(f"Iteration {iteration + 1}/{iterations}")
            
            try:
                # Measure processing time
                start_time = time.time()
                start_memory = self._get_memory_usage()  # Placeholder
                
                # Process dataset
                result = validator.process_dataset(spectra, stellas_transform=with_stellas)
                
                end_time = time.time()
                end_memory = self._get_memory_usage()  # Placeholder
                
                processing_time = end_time - start_time
                memory_delta = end_memory - start_memory
                
                processing_times.append(processing_time)
                memory_usage.append(memory_delta)
                results.append(result)
                
                self.logger.info(f"Iteration {iteration + 1} completed in {processing_time:.2f}s")
                
            except Exception as e:
                self.logger.error(f"Error in iteration {iteration + 1}: {e}")
                continue
        
        if not results:
            self.logger.error("No successful iterations")
            return self._create_error_result(validator.method_name, dataset_name, with_stellas)
        
        # Use the last result for metrics calculation
        final_result = results[-1]
        
        # Calculate processing statistics
        processing_stats = {
            'mean_processing_time': np.mean(processing_times),
            'std_processing_time': np.std(processing_times),
            'min_processing_time': np.min(processing_times),
            'max_processing_time': np.max(processing_times),
            'mean_memory_usage': np.mean(memory_usage),
            'iterations': len(processing_times),
            'throughput': len(spectra) / np.mean(processing_times) if processing_times else 0
        }
        
        # Create performance metrics
        performance_metrics = PerformanceMetrics(
            accuracy=final_result.accuracy,
            precision=final_result.precision,
            recall=final_result.recall,
            f1_score=final_result.f1_score,
            mean_confidence=np.mean(final_result.confidence_scores) if final_result.confidence_scores else 0.0,
            std_confidence=np.std(final_result.confidence_scores) if final_result.confidence_scores else 0.0,
            confidence_reliability=0.8,  # Placeholder
            processing_time=processing_stats['mean_processing_time'],
            memory_usage=processing_stats['mean_memory_usage'],
            throughput=processing_stats['throughput'],
            molecular_coverage=len(set(final_result.identifications)) / 100.0,  # Placeholder denominator
            database_hit_rate=0.75,  # Placeholder
            novel_detection_rate=0.1,  # Placeholder
            signal_to_noise=10.0,  # Placeholder
            spectral_quality=0.85,  # Placeholder
            identification_quality=final_result.accuracy,
            custom_metrics=final_result.custom_metrics
        )
        
        return BenchmarkResult(
            method_name=validator.method_name,
            dataset_name=dataset_name,
            with_stellas=with_stellas,
            performance_metrics=performance_metrics,
            validation_metrics=None,  # Will be calculated in comparison
            processing_stats=processing_stats,
            timestamp=datetime.now().isoformat()
        )
    
    def run_comprehensive_benchmark(self,
                                  validators: List[BaseValidator],
                                  dataset_names: List[str]) -> Dict[str, List[BenchmarkResult]]:
        """
        Run comprehensive benchmark across all methods and datasets
        
        Args:
            validators: List of validator instances
            dataset_names: List of dataset names to test
            
        Returns:
            Dictionary mapping method names to benchmark results
        """
        self.logger.info("Starting comprehensive benchmark")
        
        all_results = {}
        
        for validator in validators:
            method_results = []
            
            for dataset_name in dataset_names:
                # Test without S-Stellas
                result_without = self.run_method_benchmark(
                    validator, dataset_name, with_stellas=False
                )
                method_results.append(result_without)
                
                # Test with S-Stellas (if supported)
                if hasattr(validator, 'apply_stellas_transform'):
                    result_with = self.run_method_benchmark(
                        validator, dataset_name, with_stellas=True
                    )
                    method_results.append(result_with)
            
            all_results[validator.method_name] = method_results
        
        # Save results
        self._save_benchmark_results(all_results)
        
        self.logger.info("Comprehensive benchmark completed")
        return all_results
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage (placeholder implementation)"""
        # This would use psutil or similar to get actual memory usage
        return 0.0
    
    def _create_error_result(self, method_name: str, dataset_name: str, with_stellas: bool) -> BenchmarkResult:
        """Create error result when benchmark fails"""
        return BenchmarkResult(
            method_name=method_name,
            dataset_name=dataset_name,
            with_stellas=with_stellas,
            performance_metrics=PerformanceMetrics(
                accuracy=0.0, precision=0.0, recall=0.0, f1_score=0.0,
                mean_confidence=0.0, std_confidence=0.0, confidence_reliability=0.0,
                processing_time=0.0, memory_usage=0.0, throughput=0.0,
                molecular_coverage=0.0, database_hit_rate=0.0, novel_detection_rate=0.0,
                signal_to_noise=0.0, spectral_quality=0.0, identification_quality=0.0,
                custom_metrics={}
            ),
            validation_metrics=None,
            processing_stats={},
            timestamp=datetime.now().isoformat()
        )
    
    def _save_benchmark_results(self, results: Dict[str, List[BenchmarkResult]]):
        """Save benchmark results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as JSON
        json_file = self.output_directory / f"benchmark_results_{timestamp}.json"
        json_data = {
            method: [result.to_dict() for result in method_results]
            for method, method_results in results.items()
        }
        
        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        self.logger.info(f"Results saved to {json_file}")
        
        # Save as CSV for easy analysis
        csv_file = self.output_directory / f"benchmark_summary_{timestamp}.csv"
        self._create_summary_csv(results, csv_file)
    
    def _create_summary_csv(self, results: Dict[str, List[BenchmarkResult]], output_file: Path):
        """Create CSV summary of benchmark results"""
        rows = []
        
        for method_name, method_results in results.items():
            for result in method_results:
                row = {
                    'method': result.method_name,
                    'dataset': result.dataset_name,
                    'stellas_transform': result.with_stellas,
                    'accuracy': result.performance_metrics.accuracy,
                    'precision': result.performance_metrics.precision,
                    'recall': result.performance_metrics.recall,
                    'f1_score': result.performance_metrics.f1_score,
                    'processing_time': result.performance_metrics.processing_time,
                    'throughput': result.performance_metrics.throughput,
                    'molecular_coverage': result.performance_metrics.molecular_coverage,
                    'mean_confidence': result.performance_metrics.mean_confidence,
                    'timestamp': result.timestamp
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False)
        
        self.logger.info(f"Summary CSV saved to {output_file}")

class ComparisonStudy:
    """Specialized comparison studies for method evaluation"""
    
    def __init__(self, benchmark_runner: BenchmarkRunner):
        self.benchmark_runner = benchmark_runner
        self.metrics_calculator = MetricsCalculator()
        self.logger = logging.getLogger(__name__)
    
    def stellas_enhancement_study(self,
                                validators: List[BaseValidator],
                                dataset_names: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Study the enhancement provided by S-Stellas transformation
        
        Args:
            validators: List of validators to test
            dataset_names: List of datasets to test on
            
        Returns:
            Dictionary of enhancement results per method
        """
        self.logger.info("Running S-Stellas enhancement study")
        
        enhancement_results = {}
        
        for validator in validators:
            if not hasattr(validator, 'apply_stellas_transform'):
                self.logger.info(f"Skipping {validator.method_name} - no S-Stellas support")
                continue
            
            method_enhancements = {}
            
            for dataset_name in dataset_names:
                # Baseline performance
                baseline_result = self.benchmark_runner.run_method_benchmark(
                    validator, dataset_name, with_stellas=False
                )
                
                # Enhanced performance
                enhanced_result = self.benchmark_runner.run_method_benchmark(
                    validator, dataset_name, with_stellas=True
                )
                
                # Calculate improvements
                improvements = self.metrics_calculator.compare_methods(
                    baseline_result.performance_metrics,
                    enhanced_result.performance_metrics
                )
                
                method_enhancements[dataset_name] = improvements
            
            enhancement_results[validator.method_name] = method_enhancements
        
        # Save enhancement study results
        self._save_enhancement_results(enhancement_results)
        
        return enhancement_results
    
    def cross_dataset_validation(self,
                                validators: List[BaseValidator],
                                dataset_names: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Test method generalization across datasets
        
        Args:
            validators: List of validators to test
            dataset_names: List of datasets
            
        Returns:
            Cross-validation results
        """
        self.logger.info("Running cross-dataset validation")
        
        cross_validation_results = {}
        
        for validator in validators:
            method_results = {}
            
            # Test each dataset pair
            for train_dataset in dataset_names:
                for test_dataset in dataset_names:
                    if train_dataset == test_dataset:
                        continue
                    
                    pair_key = f"{train_dataset}_to_{test_dataset}"
                    
                    try:
                        # Load training data and train
                        train_spectra, _ = self.benchmark_runner.data_loader.load_dataset(train_dataset)
                        validator.train_model(train_spectra)
                        
                        # Test on different dataset
                        test_result = self.benchmark_runner.run_method_benchmark(
                            validator, test_dataset, with_stellas=False
                        )
                        
                        method_results[pair_key] = {
                            'accuracy': test_result.performance_metrics.accuracy,
                            'f1_score': test_result.performance_metrics.f1_score,
                            'processing_time': test_result.performance_metrics.processing_time
                        }
                        
                    except Exception as e:
                        self.logger.error(f"Cross-validation failed for {pair_key}: {e}")
                        method_results[pair_key] = {
                            'accuracy': 0.0,
                            'f1_score': 0.0,
                            'processing_time': float('inf')
                        }
            
            cross_validation_results[validator.method_name] = method_results
        
        return cross_validation_results
    
    def method_ranking_study(self,
                           benchmark_results: Dict[str, List[BenchmarkResult]]) -> pd.DataFrame:
        """
        Create comprehensive ranking of methods
        
        Args:
            benchmark_results: Results from comprehensive benchmark
            
        Returns:
            DataFrame with method rankings
        """
        self.logger.info("Creating method ranking study")
        
        ranking_data = []
        
        # Define weights for different metrics
        metric_weights = {
            'accuracy': 0.25,
            'f1_score': 0.20,
            'processing_time': 0.15,  # Lower is better, so we'll invert
            'throughput': 0.15,
            'molecular_coverage': 0.15,
            'mean_confidence': 0.10
        }
        
        for method_name, results in benchmark_results.items():
            for result in results:
                metrics = result.performance_metrics
                
                # Calculate weighted score
                score = 0.0
                score += metrics.accuracy * metric_weights['accuracy']
                score += metrics.f1_score * metric_weights['f1_score']
                score += (1.0 / (1.0 + metrics.processing_time)) * metric_weights['processing_time']  # Invert time
                score += min(1.0, metrics.throughput / 100.0) * metric_weights['throughput']  # Normalize throughput
                score += metrics.molecular_coverage * metric_weights['molecular_coverage']
                score += metrics.mean_confidence * metric_weights['mean_confidence']
                
                ranking_data.append({
                    'method': method_name,
                    'dataset': result.dataset_name,
                    'stellas_transform': result.with_stellas,
                    'overall_score': score,
                    'accuracy': metrics.accuracy,
                    'f1_score': metrics.f1_score,
                    'processing_time': metrics.processing_time,
                    'throughput': metrics.throughput,
                    'molecular_coverage': metrics.molecular_coverage,
                    'mean_confidence': metrics.mean_confidence
                })
        
        df = pd.DataFrame(ranking_data)
        df = df.sort_values('overall_score', ascending=False)
        
        # Save ranking results
        output_file = self.benchmark_runner.output_directory / "method_rankings.csv"
        df.to_csv(output_file, index=False)
        
        self.logger.info(f"Method rankings saved to {output_file}")
        
        return df
    
    def _save_enhancement_results(self, results: Dict[str, Dict[str, Dict[str, float]]]):
        """Save enhancement study results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.benchmark_runner.output_directory / f"stellas_enhancement_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Enhancement study results saved to {output_file}")
