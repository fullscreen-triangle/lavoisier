"""
Performance Benchmarking Module

Comprehensive benchmarking suite for comparing numerical and visual pipeline performance.
"""

import time
import psutil
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json


@dataclass
class BenchmarkResult:
    """Container for benchmark results"""
    pipeline_name: str
    task_name: str
    execution_time: float
    memory_usage: float
    cpu_usage: float
    throughput: float
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceBenchmark:
    """
    Comprehensive performance benchmarking for pipeline comparison
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize performance benchmark
        
        Args:
            output_dir: Directory to save benchmark results
        """
        self.output_dir = Path(output_dir) if output_dir else Path("benchmark_results")
        self.output_dir.mkdir(exist_ok=True)
        self.results = []
        
    def benchmark_function(
        self,
        func: Callable,
        args: Tuple = (),
        kwargs: Dict = None,
        pipeline_name: str = "Unknown",
        task_name: str = "Unknown",
        n_runs: int = 3,
        warmup_runs: int = 1
    ) -> List[BenchmarkResult]:
        """
        Benchmark a function with multiple runs
        
        Args:
            func: Function to benchmark
            args: Function arguments
            kwargs: Function keyword arguments
            pipeline_name: Name of the pipeline being tested
            task_name: Name of the task being performed
            n_runs: Number of benchmark runs
            warmup_runs: Number of warmup runs (not counted)
            
        Returns:
            List of BenchmarkResult objects
        """
        if kwargs is None:
            kwargs = {}
        
        results = []
        
        # Warmup runs
        for _ in range(warmup_runs):
            try:
                func(*args, **kwargs)
            except Exception:
                pass  # Ignore warmup errors
        
        # Actual benchmark runs
        for run_idx in range(n_runs):
            # Get initial system state
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Monitor CPU usage during execution
            cpu_percent_start = psutil.cpu_percent(interval=None)
            
            start_time = time.perf_counter()
            success = True
            error_message = None
            result_data = None
            
            try:
                result_data = func(*args, **kwargs)
            except Exception as e:
                success = False
                error_message = str(e)
            
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            
            # Get final system state
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_usage = final_memory - initial_memory
            cpu_usage = psutil.cpu_percent(interval=0.1)
            
            # Calculate throughput (items per second)
            # This is task-specific and may need customization
            throughput = 1.0 / execution_time if execution_time > 0 else 0
            
            result = BenchmarkResult(
                pipeline_name=pipeline_name,
                task_name=task_name,
                execution_time=execution_time,
                memory_usage=memory_usage,
                cpu_usage=cpu_usage,
                throughput=throughput,
                success=success,
                error_message=error_message,
                metadata={
                    'run_index': run_idx,
                    'result_size': len(result_data) if hasattr(result_data, '__len__') else None
                }
            )
            
            results.append(result)
            self.results.append(result)
        
        return results
    
    def benchmark_data_processing(
        self,
        numerical_func: Callable,
        visual_func: Callable,
        test_data: Any,
        task_name: str = "Data Processing"
    ) -> Dict[str, List[BenchmarkResult]]:
        """
        Benchmark data processing functions from both pipelines
        
        Args:
            numerical_func: Numerical pipeline function
            visual_func: Visual pipeline function
            test_data: Test data to process
            task_name: Name of the processing task
            
        Returns:
            Dictionary with results for each pipeline
        """
        results = {}
        
        # Benchmark numerical pipeline
        results['numerical'] = self.benchmark_function(
            numerical_func,
            args=(test_data,),
            pipeline_name="Numerical",
            task_name=task_name
        )
        
        # Benchmark visual pipeline
        results['visual'] = self.benchmark_function(
            visual_func,
            args=(test_data,),
            pipeline_name="Visual",
            task_name=task_name
        )
        
        return results
    
    def benchmark_scalability(
        self,
        func: Callable,
        data_sizes: List[int],
        pipeline_name: str,
        task_name: str = "Scalability Test"
    ) -> List[BenchmarkResult]:
        """
        Benchmark function scalability with different data sizes
        
        Args:
            func: Function to benchmark
            data_sizes: List of data sizes to test
            pipeline_name: Name of the pipeline
            task_name: Name of the task
            
        Returns:
            List of BenchmarkResult objects
        """
        results = []
        
        for size in data_sizes:
            # Generate test data of specified size
            test_data = np.random.randn(size, 100)  # Adjust as needed
            
            size_results = self.benchmark_function(
                func,
                args=(test_data,),
                pipeline_name=pipeline_name,
                task_name=f"{task_name} (size={size})",
                n_runs=3
            )
            
            # Add data size to metadata
            for result in size_results:
                result.metadata['data_size'] = size
            
            results.extend(size_results)
        
        return results
    
    def compare_pipelines(
        self,
        pipeline_results: Dict[str, List[BenchmarkResult]]
    ) -> pd.DataFrame:
        """
        Compare performance between pipelines
        
        Args:
            pipeline_results: Dictionary of pipeline results
            
        Returns:
            DataFrame with comparison statistical_analysis
        """
        comparison_data = []
        
        for pipeline_name, results in pipeline_results.items():
            successful_results = [r for r in results if r.success]
            
            if successful_results:
                execution_times = [r.execution_time for r in successful_results]
                memory_usages = [r.memory_usage for r in successful_results]
                cpu_usages = [r.cpu_usage for r in successful_results]
                throughputs = [r.throughput for r in successful_results]
                
                comparison_data.append({
                    'Pipeline': pipeline_name,
                    'Mean Execution Time (s)': np.mean(execution_times),
                    'Std Execution Time (s)': np.std(execution_times),
                    'Mean Memory Usage (MB)': np.mean(memory_usages),
                    'Std Memory Usage (MB)': np.std(memory_usages),
                    'Mean CPU Usage (%)': np.mean(cpu_usages),
                    'Mean Throughput (items/s)': np.mean(throughputs),
                    'Success Rate (%)': len(successful_results) / len(results) * 100,
                    'Total Runs': len(results)
                })
        
        return pd.DataFrame(comparison_data)
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance report
        
        Returns:
            Dictionary containing performance analysis
        """
        if not self.results:
            return {}
        
        # Group results by pipeline and task
        grouped_results = {}
        for result in self.results:
            key = (result.pipeline_name, result.task_name)
            if key not in grouped_results:
                grouped_results[key] = []
            grouped_results[key].append(result)
        
        report = {
            'summary': {},
            'detailed_results': {},
            'comparisons': {}
        }
        
        # Generate summary statistical_analysis
        for (pipeline, task), results in grouped_results.items():
            successful_results = [r for r in results if r.success]
            
            if successful_results:
                execution_times = [r.execution_time for r in successful_results]
                memory_usages = [r.memory_usage for r in successful_results]
                
                report['detailed_results'][f"{pipeline}_{task}"] = {
                    'mean_execution_time': np.mean(execution_times),
                    'median_execution_time': np.median(execution_times),
                    'std_execution_time': np.std(execution_times),
                    'min_execution_time': np.min(execution_times),
                    'max_execution_time': np.max(execution_times),
                    'mean_memory_usage': np.mean(memory_usages),
                    'success_rate': len(successful_results) / len(results),
                    'total_runs': len(results)
                }
        
        return report
    
    def plot_performance_comparison(
        self,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create comprehensive performance comparison plots
        
        Args:
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        if not self.results:
            return None
        
        # Convert results to DataFrame
        df_data = []
        for result in self.results:
            if result.success:
                df_data.append({
                    'Pipeline': result.pipeline_name,
                    'Task': result.task_name,
                    'Execution Time (s)': result.execution_time,
                    'Memory Usage (MB)': result.memory_usage,
                    'CPU Usage (%)': result.cpu_usage,
                    'Throughput (items/s)': result.throughput
                })
        
        df = pd.DataFrame(df_data)
        
        if df.empty:
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Performance Comparison Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Execution Time Comparison
        sns.boxplot(data=df, x='Pipeline', y='Execution Time (s)', ax=axes[0, 0])
        axes[0, 0].set_title('Execution Time Distribution')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Memory Usage Comparison
        sns.boxplot(data=df, x='Pipeline', y='Memory Usage (MB)', ax=axes[0, 1])
        axes[0, 1].set_title('Memory Usage Distribution')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Throughput Comparison
        sns.boxplot(data=df, x='Pipeline', y='Throughput (items/s)', ax=axes[1, 0])
        axes[1, 0].set_title('Throughput Distribution')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 4: CPU Usage Comparison
        sns.boxplot(data=df, x='Pipeline', y='CPU Usage (%)', ax=axes[1, 1])
        axes[1, 1].set_title('CPU Usage Distribution')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_scalability_analysis(
        self,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create scalability analysis plots
        
        Args:
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        # Filter results with data_size metadata
        scalability_results = [r for r in self.results if 'data_size' in r.metadata and r.success]
        
        if not scalability_results:
            return None
        
        # Convert to DataFrame
        df_data = []
        for result in scalability_results:
            df_data.append({
                'Pipeline': result.pipeline_name,
                'Data Size': result.metadata['data_size'],
                'Execution Time (s)': result.execution_time,
                'Memory Usage (MB)': result.memory_usage,
                'Throughput (items/s)': result.throughput
            })
        
        df = pd.DataFrame(df_data)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Scalability Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Execution Time vs Data Size
        for pipeline in df['Pipeline'].unique():
            pipeline_data = df[df['Pipeline'] == pipeline]
            axes[0, 0].plot(pipeline_data['Data Size'], pipeline_data['Execution Time (s)'], 
                           marker='o', label=pipeline)
        axes[0, 0].set_xlabel('Data Size')
        axes[0, 0].set_ylabel('Execution Time (s)')
        axes[0, 0].set_title('Execution Time vs Data Size')
        axes[0, 0].legend()
        axes[0, 0].set_xscale('log')
        axes[0, 0].set_yscale('log')
        
        # Plot 2: Memory Usage vs Data Size
        for pipeline in df['Pipeline'].unique():
            pipeline_data = df[df['Pipeline'] == pipeline]
            axes[0, 1].plot(pipeline_data['Data Size'], pipeline_data['Memory Usage (MB)'], 
                           marker='o', label=pipeline)
        axes[0, 1].set_xlabel('Data Size')
        axes[0, 1].set_ylabel('Memory Usage (MB)')
        axes[0, 1].set_title('Memory Usage vs Data Size')
        axes[0, 1].legend()
        axes[0, 1].set_xscale('log')
        
        # Plot 3: Throughput vs Data Size
        for pipeline in df['Pipeline'].unique():
            pipeline_data = df[df['Pipeline'] == pipeline]
            axes[1, 0].plot(pipeline_data['Data Size'], pipeline_data['Throughput (items/s)'], 
                           marker='o', label=pipeline)
        axes[1, 0].set_xlabel('Data Size')
        axes[1, 0].set_ylabel('Throughput (items/s)')
        axes[1, 0].set_title('Throughput vs Data Size')
        axes[1, 0].legend()
        axes[1, 0].set_xscale('log')
        
        # Plot 4: Efficiency (Throughput/Memory) vs Data Size
        df['Efficiency'] = df['Throughput (items/s)'] / (df['Memory Usage (MB)'] + 1)
        for pipeline in df['Pipeline'].unique():
            pipeline_data = df[df['Pipeline'] == pipeline]
            axes[1, 1].plot(pipeline_data['Data Size'], pipeline_data['Efficiency'], 
                           marker='o', label=pipeline)
        axes[1, 1].set_xlabel('Data Size')
        axes[1, 1].set_ylabel('Efficiency (Throughput/Memory)')
        axes[1, 1].set_title('Efficiency vs Data Size')
        axes[1, 1].legend()
        axes[1, 1].set_xscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def save_results(self, filename: str = "benchmark_results.json") -> None:
        """
        Save benchmark results to file
        
        Args:
            filename: Name of the output file
        """
        output_path = self.output_dir / filename
        
        # Convert results to serializable format
        serializable_results = []
        for result in self.results:
            serializable_results.append({
                'pipeline_name': result.pipeline_name,
                'task_name': result.task_name,
                'execution_time': result.execution_time,
                'memory_usage': result.memory_usage,
                'cpu_usage': result.cpu_usage,
                'throughput': result.throughput,
                'success': result.success,
                'error_message': result.error_message,
                'metadata': result.metadata
            })
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    
    def load_results(self, filename: str = "benchmark_results.json") -> None:
        """
        Load benchmark results from file
        
        Args:
            filename: Name of the input file
        """
        input_path = self.output_dir / filename
        
        if not input_path.exists():
            raise FileNotFoundError(f"Benchmark results file not found: {input_path}")
        
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        self.results = []
        for item in data:
            result = BenchmarkResult(
                pipeline_name=item['pipeline_name'],
                task_name=item['task_name'],
                execution_time=item['execution_time'],
                memory_usage=item['memory_usage'],
                cpu_usage=item['cpu_usage'],
                throughput=item['throughput'],
                success=item['success'],
                error_message=item.get('error_message'),
                metadata=item.get('metadata', {})
            )
            self.results.append(result)
    
    def clear_results(self) -> None:
        """Clear all stored results"""
        self.results = [] 