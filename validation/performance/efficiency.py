"""
Efficiency Analysis Module

Analyzes computational efficiency and resource utilization
for comparing numerical and visual pipelines.
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
import threading
import queue


@dataclass
class EfficiencyMetrics:
    """Container for efficiency metrics"""
    pipeline_name: str
    task_name: str
    cpu_efficiency: float  # Operations per CPU second
    memory_efficiency: float  # Operations per MB
    energy_efficiency: float  # Operations per unit energy
    throughput_efficiency: float  # Items processed per second
    resource_utilization: Dict[str, float]
    cost_per_operation: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class EfficiencyAnalyzer:
    """
    Comprehensive efficiency analysis for pipeline comparison
    """
    
    def __init__(self, monitoring_interval: float = 0.1):
        """
        Initialize efficiency analyzer
        
        Args:
            monitoring_interval: Interval for resource monitoring in seconds
        """
        self.monitoring_interval = monitoring_interval
        self.results = []
        self._monitoring_active = False
        self._resource_data = []
        
    def analyze_computational_efficiency(
        self,
        func: Callable,
        args: Tuple = (),
        kwargs: Dict = None,
        pipeline_name: str = "Unknown",
        task_name: str = "Unknown",
        data_size: int = 1000
    ) -> EfficiencyMetrics:
        """
        Analyze computational efficiency of a function
        
        Args:
            func: Function to analyze
            args: Function arguments
            kwargs: Function keyword arguments
            pipeline_name: Name of the pipeline
            task_name: Name of the task
            data_size: Size of data being processed
            
        Returns:
            EfficiencyMetrics object
        """
        if kwargs is None:
            kwargs = {}
        
        # Start resource monitoring
        self._start_monitoring()
        
        # Execute function and measure time
        start_time = time.perf_counter()
        start_cpu_time = time.process_time()
        
        try:
            result = func(*args, **kwargs)
            success = True
        except Exception as e:
            result = None
            success = False
        
        end_time = time.perf_counter()
        end_cpu_time = time.process_time()
        
        # Stop monitoring and get resource data
        resource_stats = self._stop_monitoring()
        
        # Calculate efficiency metrics
        wall_time = end_time - start_time
        cpu_time = end_cpu_time - start_cpu_time
        
        # CPU efficiency (operations per CPU second)
        cpu_efficiency = data_size / cpu_time if cpu_time > 0 else 0
        
        # Memory efficiency (operations per MB)
        avg_memory = np.mean([stat['memory_mb'] for stat in resource_stats]) if resource_stats else 0
        memory_efficiency = data_size / avg_memory if avg_memory > 0 else 0
        
        # Throughput efficiency (items per wall-clock second)
        throughput_efficiency = data_size / wall_time if wall_time > 0 else 0
        
        # Energy efficiency (approximate using CPU usage)
        avg_cpu_percent = np.mean([stat['cpu_percent'] for stat in resource_stats]) if resource_stats else 0
        energy_proxy = avg_cpu_percent * wall_time  # Simplified energy proxy
        energy_efficiency = data_size / energy_proxy if energy_proxy > 0 else 0
        
        # Resource utilization summary
        resource_utilization = {
            'avg_cpu_percent': avg_cpu_percent,
            'max_memory_mb': max([stat['memory_mb'] for stat in resource_stats]) if resource_stats else 0,
            'avg_memory_mb': avg_memory,
            'wall_time_seconds': wall_time,
            'cpu_time_seconds': cpu_time
        }
        
        # Cost per operation (simplified using time as proxy)
        cost_per_operation = wall_time / data_size if data_size > 0 else float('inf')
        
        metrics = EfficiencyMetrics(
            pipeline_name=pipeline_name,
            task_name=task_name,
            cpu_efficiency=cpu_efficiency,
            memory_efficiency=memory_efficiency,
            energy_efficiency=energy_efficiency,
            throughput_efficiency=throughput_efficiency,
            resource_utilization=resource_utilization,
            cost_per_operation=cost_per_operation,
            metadata={
                'data_size': data_size,
                'success': success,
                'result_size': len(result) if hasattr(result, '__len__') else None
            }
        )
        
        self.results.append(metrics)
        return metrics
    
    def compare_pipeline_efficiency(
        self,
        numerical_func: Callable,
        visual_func: Callable,
        test_data: Any,
        task_name: str = "Pipeline Comparison"
    ) -> Dict[str, EfficiencyMetrics]:
        """
        Compare efficiency between numerical and visual pipelines
        
        Args:
            numerical_func: Numerical pipeline function
            visual_func: Visual pipeline function
            test_data: Test data to process
            task_name: Name of the task
            
        Returns:
            Dictionary with efficiency metrics for each pipeline
        """
        data_size = len(test_data) if hasattr(test_data, '__len__') else 1000
        
        # Analyze numerical pipeline
        numerical_metrics = self.analyze_computational_efficiency(
            numerical_func,
            args=(test_data,),
            pipeline_name="Numerical",
            task_name=task_name,
            data_size=data_size
        )
        
        # Analyze visual pipeline
        visual_metrics = self.analyze_computational_efficiency(
            visual_func,
            args=(test_data,),
            pipeline_name="Visual",
            task_name=task_name,
            data_size=data_size
        )
        
        return {
            'numerical': numerical_metrics,
            'visual': visual_metrics
        }
    
    def analyze_scalability_efficiency(
        self,
        func: Callable,
        data_sizes: List[int],
        pipeline_name: str,
        task_name: str = "Scalability Analysis"
    ) -> List[EfficiencyMetrics]:
        """
        Analyze how efficiency scales with data size
        
        Args:
            func: Function to analyze
            data_sizes: List of data sizes to test
            pipeline_name: Name of the pipeline
            task_name: Name of the task
            
        Returns:
            List of EfficiencyMetrics for different data sizes
        """
        results = []
        
        for size in data_sizes:
            # Generate test data of specified size
            test_data = np.random.randn(size, 10)  # Adjust dimensions as needed
            
            metrics = self.analyze_computational_efficiency(
                func,
                args=(test_data,),
                pipeline_name=pipeline_name,
                task_name=f"{task_name} (size={size})",
                data_size=size
            )
            
            results.append(metrics)
        
        return results
    
    def calculate_efficiency_ratios(
        self,
        baseline_metrics: EfficiencyMetrics,
        comparison_metrics: EfficiencyMetrics
    ) -> Dict[str, float]:
        """
        Calculate efficiency ratios between two sets of metrics
        
        Args:
            baseline_metrics: Baseline efficiency metrics
            comparison_metrics: Comparison efficiency metrics
            
        Returns:
            Dictionary of efficiency ratios
        """
        ratios = {}
        
        # CPU efficiency ratio
        if baseline_metrics.cpu_efficiency > 0:
            ratios['cpu_efficiency_ratio'] = comparison_metrics.cpu_efficiency / baseline_metrics.cpu_efficiency
        else:
            ratios['cpu_efficiency_ratio'] = float('inf') if comparison_metrics.cpu_efficiency > 0 else 1.0
        
        # Memory efficiency ratio
        if baseline_metrics.memory_efficiency > 0:
            ratios['memory_efficiency_ratio'] = comparison_metrics.memory_efficiency / baseline_metrics.memory_efficiency
        else:
            ratios['memory_efficiency_ratio'] = float('inf') if comparison_metrics.memory_efficiency > 0 else 1.0
        
        # Throughput efficiency ratio
        if baseline_metrics.throughput_efficiency > 0:
            ratios['throughput_efficiency_ratio'] = comparison_metrics.throughput_efficiency / baseline_metrics.throughput_efficiency
        else:
            ratios['throughput_efficiency_ratio'] = float('inf') if comparison_metrics.throughput_efficiency > 0 else 1.0
        
        # Energy efficiency ratio
        if baseline_metrics.energy_efficiency > 0:
            ratios['energy_efficiency_ratio'] = comparison_metrics.energy_efficiency / baseline_metrics.energy_efficiency
        else:
            ratios['energy_efficiency_ratio'] = float('inf') if comparison_metrics.energy_efficiency > 0 else 1.0
        
        # Cost ratio (inverse - lower is better)
        if baseline_metrics.cost_per_operation > 0:
            ratios['cost_ratio'] = comparison_metrics.cost_per_operation / baseline_metrics.cost_per_operation
        else:
            ratios['cost_ratio'] = float('inf') if comparison_metrics.cost_per_operation > 0 else 1.0
        
        return ratios
    
    def _start_monitoring(self) -> None:
        """Start resource monitoring in background thread"""
        self._monitoring_active = True
        self._resource_data = []
        self._monitor_queue = queue.Queue()
        
        def monitor_resources():
            process = psutil.Process()
            while self._monitoring_active:
                try:
                    cpu_percent = process.cpu_percent()
                    memory_info = process.memory_info()
                    memory_mb = memory_info.rss / 1024 / 1024
                    
                    self._resource_data.append({
                        'timestamp': time.time(),
                        'cpu_percent': cpu_percent,
                        'memory_mb': memory_mb
                    })
                    
                    time.sleep(self.monitoring_interval)
                except:
                    break
        
        self._monitor_thread = threading.Thread(target=monitor_resources)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
    
    def _stop_monitoring(self) -> List[Dict]:
        """Stop resource monitoring and return collected data"""
        self._monitoring_active = False
        if hasattr(self, '_monitor_thread'):
            self._monitor_thread.join(timeout=1.0)
        
        return self._resource_data.copy()
    
    def generate_efficiency_report(self) -> pd.DataFrame:
        """Generate comprehensive efficiency report"""
        if not self.results:
            return pd.DataFrame()
        
        data = []
        for metrics in self.results:
            data.append({
                'Pipeline': metrics.pipeline_name,
                'Task': metrics.task_name,
                'CPU Efficiency (ops/cpu_sec)': metrics.cpu_efficiency,
                'Memory Efficiency (ops/MB)': metrics.memory_efficiency,
                'Throughput Efficiency (ops/sec)': metrics.throughput_efficiency,
                'Energy Efficiency (ops/energy_unit)': metrics.energy_efficiency,
                'Cost per Operation (sec/op)': metrics.cost_per_operation,
                'Avg CPU %': metrics.resource_utilization.get('avg_cpu_percent', 0),
                'Max Memory (MB)': metrics.resource_utilization.get('max_memory_mb', 0),
                'Wall Time (sec)': metrics.resource_utilization.get('wall_time_seconds', 0),
                'Data Size': metrics.metadata.get('data_size', 0)
            })
        
        return pd.DataFrame(data)
    
    def plot_efficiency_comparison(self, save_path: Optional[str] = None) -> plt.Figure:
        """Create comprehensive efficiency comparison plots"""
        if not self.results:
            return None
        
        df = self.generate_efficiency_report()
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Efficiency Analysis Comparison', fontsize=16, fontweight='bold')
        
        # Plot 1: CPU Efficiency
        sns.boxplot(data=df, x='Pipeline', y='CPU Efficiency (ops/cpu_sec)', ax=axes[0, 0])
        axes[0, 0].set_title('CPU Efficiency')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Memory Efficiency
        sns.boxplot(data=df, x='Pipeline', y='Memory Efficiency (ops/MB)', ax=axes[0, 1])
        axes[0, 1].set_title('Memory Efficiency')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Throughput Efficiency
        sns.boxplot(data=df, x='Pipeline', y='Throughput Efficiency (ops/sec)', ax=axes[0, 2])
        axes[0, 2].set_title('Throughput Efficiency')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # Plot 4: Energy Efficiency
        sns.boxplot(data=df, x='Pipeline', y='Energy Efficiency (ops/energy_unit)', ax=axes[1, 0])
        axes[1, 0].set_title('Energy Efficiency')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 5: Cost per Operation
        sns.boxplot(data=df, x='Pipeline', y='Cost per Operation (sec/op)', ax=axes[1, 1])
        axes[1, 1].set_title('Cost per Operation')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Plot 6: Resource Utilization
        resource_data = []
        for metrics in self.results:
            resource_data.append({
                'Pipeline': metrics.pipeline_name,
                'CPU %': metrics.resource_utilization.get('avg_cpu_percent', 0),
                'Memory (MB)': metrics.resource_utilization.get('max_memory_mb', 0)
            })
        
        resource_df = pd.DataFrame(resource_data)
        
        # Scatter plot of CPU vs Memory usage
        for pipeline in resource_df['Pipeline'].unique():
            pipeline_data = resource_df[resource_df['Pipeline'] == pipeline]
            axes[1, 2].scatter(pipeline_data['CPU %'], pipeline_data['Memory (MB)'], 
                             label=pipeline, alpha=0.7, s=100)
        
        axes[1, 2].set_xlabel('CPU Usage (%)')
        axes[1, 2].set_ylabel('Memory Usage (MB)')
        axes[1, 2].set_title('Resource Utilization')
        axes[1, 2].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_scalability_efficiency(self, save_path: Optional[str] = None) -> plt.Figure:
        """Create scalability efficiency plots"""
        # Filter results with data_size metadata
        scalability_results = [r for r in self.results if 'data_size' in r.metadata]
        
        if not scalability_results:
            return None
        
        # Convert to DataFrame
        df_data = []
        for metrics in scalability_results:
            df_data.append({
                'Pipeline': metrics.pipeline_name,
                'Data Size': metrics.metadata['data_size'],
                'CPU Efficiency': metrics.cpu_efficiency,
                'Memory Efficiency': metrics.memory_efficiency,
                'Throughput Efficiency': metrics.throughput_efficiency,
                'Cost per Operation': metrics.cost_per_operation
            })
        
        df = pd.DataFrame(df_data)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Scalability Efficiency Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: CPU Efficiency vs Data Size
        for pipeline in df['Pipeline'].unique():
            pipeline_data = df[df['Pipeline'] == pipeline]
            axes[0, 0].plot(pipeline_data['Data Size'], pipeline_data['CPU Efficiency'], 
                           marker='o', label=pipeline)
        axes[0, 0].set_xlabel('Data Size')
        axes[0, 0].set_ylabel('CPU Efficiency (ops/cpu_sec)')
        axes[0, 0].set_title('CPU Efficiency vs Data Size')
        axes[0, 0].legend()
        axes[0, 0].set_xscale('log')
        
        # Plot 2: Memory Efficiency vs Data Size
        for pipeline in df['Pipeline'].unique():
            pipeline_data = df[df['Pipeline'] == pipeline]
            axes[0, 1].plot(pipeline_data['Data Size'], pipeline_data['Memory Efficiency'], 
                           marker='o', label=pipeline)
        axes[0, 1].set_xlabel('Data Size')
        axes[0, 1].set_ylabel('Memory Efficiency (ops/MB)')
        axes[0, 1].set_title('Memory Efficiency vs Data Size')
        axes[0, 1].legend()
        axes[0, 1].set_xscale('log')
        
        # Plot 3: Throughput Efficiency vs Data Size
        for pipeline in df['Pipeline'].unique():
            pipeline_data = df[df['Pipeline'] == pipeline]
            axes[1, 0].plot(pipeline_data['Data Size'], pipeline_data['Throughput Efficiency'], 
                           marker='o', label=pipeline)
        axes[1, 0].set_xlabel('Data Size')
        axes[1, 0].set_ylabel('Throughput Efficiency (ops/sec)')
        axes[1, 0].set_title('Throughput Efficiency vs Data Size')
        axes[1, 0].legend()
        axes[1, 0].set_xscale('log')
        
        # Plot 4: Cost per Operation vs Data Size
        for pipeline in df['Pipeline'].unique():
            pipeline_data = df[df['Pipeline'] == pipeline]
            axes[1, 1].plot(pipeline_data['Data Size'], pipeline_data['Cost per Operation'], 
                           marker='o', label=pipeline)
        axes[1, 1].set_xlabel('Data Size')
        axes[1, 1].set_ylabel('Cost per Operation (sec/op)')
        axes[1, 1].set_title('Cost per Operation vs Data Size')
        axes[1, 1].legend()
        axes[1, 1].set_xscale('log')
        axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def get_efficiency_summary(self) -> Dict[str, Any]:
        """Get summary of efficiency analysis"""
        if not self.results:
            return {}
        
        # Group by pipeline
        pipeline_groups = {}
        for metrics in self.results:
            if metrics.pipeline_name not in pipeline_groups:
                pipeline_groups[metrics.pipeline_name] = []
            pipeline_groups[metrics.pipeline_name].append(metrics)
        
        summary = {}
        for pipeline_name, metrics_list in pipeline_groups.items():
            cpu_efficiencies = [m.cpu_efficiency for m in metrics_list]
            memory_efficiencies = [m.memory_efficiency for m in metrics_list]
            throughput_efficiencies = [m.throughput_efficiency for m in metrics_list]
            costs = [m.cost_per_operation for m in metrics_list]
            
            summary[pipeline_name] = {
                'mean_cpu_efficiency': np.mean(cpu_efficiencies),
                'mean_memory_efficiency': np.mean(memory_efficiencies),
                'mean_throughput_efficiency': np.mean(throughput_efficiencies),
                'mean_cost_per_operation': np.mean(costs),
                'total_runs': len(metrics_list)
            }
        
        return summary
    
    def clear_results(self) -> None:
        """Clear all stored results"""
        self.results = [] 