"""
Scalability Tester Module

Comprehensive scalability testing for pipeline performance analysis
including data size scaling, computational complexity assessment,
and resource utilization scaling.
"""

import numpy as np
import pandas as pd
import time
import psutil
import threading
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class ScalabilityResult:
    """Container for scalability test results"""
    test_name: str
    data_sizes: List[int]
    execution_times: List[float]
    memory_usage: List[float]
    cpu_usage: List[float]
    scaling_factor: float  # How performance scales with data size
    interpretation: str
    metadata: Dict[str, Any] = None


class ScalabilityTester:
    """
    Comprehensive scalability testing for pipeline performance
    """
    
    def __init__(self):
        """Initialize scalability tester"""
        self.results = []
        self.monitoring_active = False
        
    def test_data_size_scaling(
        self,
        pipeline_function: Callable,
        base_data: np.ndarray,
        size_multipliers: List[float] = [0.5, 1.0, 2.0, 4.0, 8.0],
        test_name: str = "Data Size Scaling"
    ) -> ScalabilityResult:
        """
        Test how pipeline performance scales with data size
        
        Args:
            pipeline_function: Function to test (should accept data as input)
            base_data: Base dataset to scale
            size_multipliers: Multipliers for data size
            test_name: Name of the test
            
        Returns:
            ScalabilityResult object
        """
        data_sizes = []
        execution_times = []
        memory_usage = []
        cpu_usage = []
        
        for multiplier in size_multipliers:
            # Create scaled dataset
            if multiplier <= 1.0:
                # Subsample for smaller sizes
                n_samples = int(len(base_data) * multiplier)
                scaled_data = base_data[:n_samples]
            else:
                # Replicate for larger sizes
                n_repeats = int(multiplier)
                scaled_data = np.tile(base_data, (n_repeats, 1))
            
            data_size = scaled_data.size
            data_sizes.append(data_size)
            
            # Monitor performance
            performance_metrics = self._monitor_function_performance(
                pipeline_function, scaled_data
            )
            
            execution_times.append(performance_metrics['execution_time'])
            memory_usage.append(performance_metrics['peak_memory_mb'])
            cpu_usage.append(performance_metrics['avg_cpu_percent'])
        
        # Calculate scaling factor (slope of log-log plot)
        scaling_factor = self._calculate_scaling_factor(data_sizes, execution_times)
        
        # Interpretation
        if scaling_factor < 1.2:
            complexity = "Linear"
            interpretation = f"Excellent scalability - {complexity} complexity (factor: {scaling_factor:.2f})"
        elif scaling_factor < 1.8:
            complexity = "Log-linear"
            interpretation = f"Good scalability - {complexity} complexity (factor: {scaling_factor:.2f})"
        elif scaling_factor < 2.5:
            complexity = "Quadratic"
            interpretation = f"Moderate scalability - {complexity} complexity (factor: {scaling_factor:.2f})"
        else:
            complexity = "Polynomial/Exponential"
            interpretation = f"Poor scalability - {complexity} complexity (factor: {scaling_factor:.2f})"
        
        result = ScalabilityResult(
            test_name=test_name,
            data_sizes=data_sizes,
            execution_times=execution_times,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            scaling_factor=scaling_factor,
            interpretation=interpretation,
            metadata={
                'size_multipliers': size_multipliers,
                'complexity_class': complexity,
                'base_data_size': base_data.size
            }
        )
        
        self.results.append(result)
        return result
    
    def test_computational_complexity(
        self,
        pipeline_function: Callable,
        data_generator: Callable,
        complexity_parameters: List[int],
        parameter_name: str = "complexity_param",
        test_name: str = "Computational Complexity"
    ) -> ScalabilityResult:
        """
        Test computational complexity with varying algorithm parameters
        
        Args:
            pipeline_function: Function to test
            data_generator: Function that generates data given complexity parameter
            complexity_parameters: List of complexity parameters to test
            parameter_name: Name of the complexity parameter
            test_name: Name of the test
            
        Returns:
            ScalabilityResult object
        """
        data_sizes = []
        execution_times = []
        memory_usage = []
        cpu_usage = []
        
        for param in complexity_parameters:
            # Generate data with specified complexity
            test_data = data_generator(param)
            data_size = test_data.size if hasattr(test_data, 'size') else len(test_data)
            data_sizes.append(data_size)
            
            # Monitor performance
            performance_metrics = self._monitor_function_performance(
                pipeline_function, test_data
            )
            
            execution_times.append(performance_metrics['execution_time'])
            memory_usage.append(performance_metrics['peak_memory_mb'])
            cpu_usage.append(performance_metrics['avg_cpu_percent'])
        
        # Calculate scaling factor
        scaling_factor = self._calculate_scaling_factor(complexity_parameters, execution_times)
        
        interpretation = f"Complexity scaling with {parameter_name}: factor {scaling_factor:.2f}"
        
        result = ScalabilityResult(
            test_name=test_name,
            data_sizes=data_sizes,
            execution_times=execution_times,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            scaling_factor=scaling_factor,
            interpretation=interpretation,
            metadata={
                'complexity_parameters': complexity_parameters,
                'parameter_name': parameter_name
            }
        )
        
        self.results.append(result)
        return result
    
    def test_concurrent_scaling(
        self,
        pipeline_function: Callable,
        test_data: np.ndarray,
        thread_counts: List[int] = [1, 2, 4, 8],
        test_name: str = "Concurrent Scaling"
    ) -> ScalabilityResult:
        """
        Test how pipeline scales with concurrent execution
        
        Args:
            pipeline_function: Function to test
            test_data: Data to process
            thread_counts: Number of threads to test
            test_name: Name of the test
            
        Returns:
            ScalabilityResult object
        """
        data_sizes = []
        execution_times = []
        memory_usage = []
        cpu_usage = []
        
        for n_threads in thread_counts:
            data_sizes.append(test_data.size)
            
            # Create thread-safe wrapper
            def thread_worker():
                return pipeline_function(test_data)
            
            # Monitor concurrent execution
            start_time = time.time()
            
            # Start monitoring
            monitor_thread = threading.Thread(target=self._start_resource_monitoring)
            self.monitoring_active = True
            monitor_thread.start()
            
            # Run concurrent tasks
            threads = []
            for _ in range(n_threads):
                thread = threading.Thread(target=thread_worker)
                threads.append(thread)
                thread.start()
            
            # Wait for completion
            for thread in threads:
                thread.join()
            
            execution_time = time.time() - start_time
            
            # Stop monitoring
            self.monitoring_active = False
            monitor_thread.join()
            
            execution_times.append(execution_time)
            
            # Get resource usage (simplified for concurrent case)
            process = psutil.Process()
            memory_usage.append(process.memory_info().rss / 1024 / 1024)  # MB
            cpu_usage.append(process.cpu_percent())
        
        # Calculate scaling efficiency
        baseline_time = execution_times[0]
        scaling_factor = self._calculate_concurrent_scaling_factor(thread_counts, execution_times, baseline_time)
        
        interpretation = f"Concurrent scaling efficiency: {scaling_factor:.2f}"
        if scaling_factor > 0.8:
            interpretation += " (Excellent parallelization)"
        elif scaling_factor > 0.6:
            interpretation += " (Good parallelization)"
        elif scaling_factor > 0.4:
            interpretation += " (Moderate parallelization)"
        else:
            interpretation += " (Poor parallelization)"
        
        result = ScalabilityResult(
            test_name=test_name,
            data_sizes=data_sizes,
            execution_times=execution_times,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            scaling_factor=scaling_factor,
            interpretation=interpretation,
            metadata={
                'thread_counts': thread_counts,
                'baseline_time': baseline_time
            }
        )
        
        self.results.append(result)
        return result
    
    def compare_pipeline_scalability(
        self,
        numerical_pipeline: Callable,
        visual_pipeline: Callable,
        test_data: np.ndarray,
        size_multipliers: List[float] = [0.5, 1.0, 2.0, 4.0]
    ) -> Dict[str, ScalabilityResult]:
        """
        Compare scalability between numerical and visual pipelines
        
        Args:
            numerical_pipeline: Numerical pipeline function
            visual_pipeline: Visual pipeline function
            test_data: Test data
            size_multipliers: Data size multipliers
            
        Returns:
            Dictionary with scalability results for each pipeline
        """
        # Test numerical pipeline
        numerical_result = self.test_data_size_scaling(
            numerical_pipeline, test_data, size_multipliers, "Numerical Pipeline Scaling"
        )
        
        # Test visual pipeline
        visual_result = self.test_data_size_scaling(
            visual_pipeline, test_data, size_multipliers, "Visual Pipeline Scaling"
        )
        
        return {
            'numerical': numerical_result,
            'visual': visual_result
        }
    
    def _monitor_function_performance(
        self,
        func: Callable,
        data: np.ndarray
    ) -> Dict[str, float]:
        """
        Monitor performance metrics during function execution
        
        Args:
            func: Function to monitor
            data: Data to pass to function
            
        Returns:
            Dictionary with performance metrics
        """
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Start CPU monitoring
        cpu_percent_start = process.cpu_percent()
        
        # Execute function and measure time
        start_time = time.time()
        try:
            result = func(data)
        except Exception as e:
            print(f"Function execution failed: {e}")
            result = None
        execution_time = time.time() - start_time
        
        # Get final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        peak_memory = max(initial_memory, final_memory)
        
        # Get CPU usage
        cpu_percent_end = process.cpu_percent()
        avg_cpu_percent = (cpu_percent_start + cpu_percent_end) / 2
        
        return {
            'execution_time': execution_time,
            'peak_memory_mb': peak_memory,
            'avg_cpu_percent': avg_cpu_percent,
            'memory_delta_mb': final_memory - initial_memory
        }
    
    def _calculate_scaling_factor(
        self,
        sizes: List[Union[int, float]],
        times: List[float]
    ) -> float:
        """
        Calculate scaling factor from size and time data
        
        Args:
            sizes: Data sizes or complexity parameters
            times: Execution times
            
        Returns:
            Scaling factor (slope in log-log space)
        """
        if len(sizes) < 2 or len(times) < 2:
            return 1.0
        
        # Convert to log space
        log_sizes = np.log(np.array(sizes))
        log_times = np.log(np.array(times))
        
        # Calculate slope (scaling factor)
        try:
            slope = np.polyfit(log_sizes, log_times, 1)[0]
            return abs(slope)
        except:
            return 1.0
    
    def _calculate_concurrent_scaling_factor(
        self,
        thread_counts: List[int],
        execution_times: List[float],
        baseline_time: float
    ) -> float:
        """
        Calculate concurrent scaling efficiency
        
        Args:
            thread_counts: Number of threads
            execution_times: Execution times
            baseline_time: Single-thread baseline time
            
        Returns:
            Scaling efficiency (0-1, where 1 is perfect scaling)
        """
        if len(thread_counts) < 2:
            return 1.0
        
        # Calculate theoretical speedup vs actual speedup
        theoretical_speedups = thread_counts
        actual_speedups = [baseline_time / time for time in execution_times]
        
        # Calculate efficiency as ratio of actual to theoretical
        efficiencies = [actual / theoretical for actual, theoretical in zip(actual_speedups, theoretical_speedups)]
        
        # Return average efficiency
        return np.mean(efficiencies)
    
    def _start_resource_monitoring(self):
        """Start background resource monitoring"""
        while self.monitoring_active:
            time.sleep(0.1)  # Monitor every 100ms
    
    def generate_scalability_report(self) -> pd.DataFrame:
        """Generate comprehensive scalability report"""
        if not self.results:
            return pd.DataFrame()
        
        data = []
        for result in self.results:
            data.append({
                'Test': result.test_name,
                'Scaling Factor': result.scaling_factor,
                'Max Data Size': max(result.data_sizes),
                'Max Execution Time': max(result.execution_times),
                'Max Memory (MB)': max(result.memory_usage),
                'Interpretation': result.interpretation
            })
        
        return pd.DataFrame(data)
    
    def plot_scalability_results(
        self,
        result: ScalabilityResult,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot scalability test results
        
        Args:
            result: ScalabilityResult to plot
            save_path: Optional path to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Scalability Analysis: {result.test_name}', fontsize=14)
        
        # Execution time vs data size
        axes[0, 0].loglog(result.data_sizes, result.execution_times, 'bo-')
        axes[0, 0].set_xlabel('Data Size')
        axes[0, 0].set_ylabel('Execution Time (s)')
        axes[0, 0].set_title('Execution Time Scaling')
        axes[0, 0].grid(True)
        
        # Memory usage vs data size
        axes[0, 1].loglog(result.data_sizes, result.memory_usage, 'ro-')
        axes[0, 1].set_xlabel('Data Size')
        axes[0, 1].set_ylabel('Memory Usage (MB)')
        axes[0, 1].set_title('Memory Usage Scaling')
        axes[0, 1].grid(True)
        
        # CPU usage vs data size
        axes[1, 0].semilogx(result.data_sizes, result.cpu_usage, 'go-')
        axes[1, 0].set_xlabel('Data Size')
        axes[1, 0].set_ylabel('CPU Usage (%)')
        axes[1, 0].set_title('CPU Usage vs Data Size')
        axes[1, 0].grid(True)
        
        # Scaling factor visualization
        axes[1, 1].text(0.1, 0.5, f'Scaling Factor: {result.scaling_factor:.2f}\n\n{result.interpretation}',
                       transform=axes[1, 1].transAxes, fontsize=12,
                       verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightblue'))
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].set_title('Scalability Summary')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def get_best_scaling_result(self) -> Optional[ScalabilityResult]:
        """Get the result with the best scaling factor"""
        if not self.results:
            return None
        
        return min(self.results, key=lambda x: x.scaling_factor)
    
    def get_worst_scaling_result(self) -> Optional[ScalabilityResult]:
        """Get the result with the worst scaling factor"""
        if not self.results:
            return None
        
        return max(self.results, key=lambda x: x.scaling_factor)
    
    def clear_results(self) -> None:
        """Clear all stored results"""
        self.results = [] 