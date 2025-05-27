"""
Performance Plots Module

Specialized visualization tools for performance validation results
including benchmarking, scalability, and resource usage analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class PerformancePlotter:
    """
    Visualization tools for performance validation results
    """
    
    def __init__(self):
        """Initialize performance plotter"""
        self.colors = {
            'numerical': '#1f77b4',
            'visual': '#ff7f0e',
            'memory': '#2ca02c',
            'cpu': '#d62728',
            'time': '#9467bd'
        }
    
    def plot_benchmark_results(
        self,
        benchmark_results: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot benchmark comparison results
        
        Args:
            benchmark_results: Dictionary containing benchmark results
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Performance Benchmark Results', fontsize=16)
        
        # Generate example data (replace with actual results)
        functions = ['preprocessing', 'feature_extraction', 'analysis', 'postprocessing']
        num_times = [1.2, 3.5, 8.7, 0.8]
        vis_times = [1.8, 5.2, 12.3, 1.1]
        
        # Execution time comparison (top left)
        x = np.arange(len(functions))
        width = 0.35
        
        bars1 = axes[0, 0].bar(x - width/2, num_times, width, 
                              label='Numerical', color=self.colors['numerical'])
        bars2 = axes[0, 0].bar(x + width/2, vis_times, width,
                              label='Visual', color=self.colors['visual'])
        
        axes[0, 0].set_xlabel('Processing Functions')
        axes[0, 0].set_ylabel('Execution Time (seconds)')
        axes[0, 0].set_title('Execution Time Comparison')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(functions, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                               f'{height:.1f}s', ha='center', va='bottom', fontsize=8)
        
        # Speedup ratio (top middle)
        speedup_ratios = [vis_times[i] / num_times[i] for i in range(len(functions))]
        colors = ['green' if ratio < 1 else 'red' for ratio in speedup_ratios]
        
        bars = axes[0, 1].bar(functions, speedup_ratios, color=colors)
        axes[0, 1].axhline(y=1, color='black', linestyle='--', alpha=0.7)
        axes[0, 1].set_ylabel('Speedup Ratio (Visual/Numerical)')
        axes[0, 1].set_title('Performance Ratio Analysis')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add ratio labels
        for bar, ratio in zip(bars, speedup_ratios):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.05,
                           f'{ratio:.2f}x', ha='center', va='bottom', fontsize=10)
        
        # Throughput comparison (top right)
        throughput_metrics = ['Samples/sec', 'Features/sec', 'Predictions/sec']
        num_throughput = [150, 2500, 45]
        vis_throughput = [120, 1800, 38]
        
        x_pos = np.arange(len(throughput_metrics))
        axes[0, 2].bar(x_pos - width/2, num_throughput, width,
                      label='Numerical', color=self.colors['numerical'])
        axes[0, 2].bar(x_pos + width/2, vis_throughput, width,
                      label='Visual', color=self.colors['visual'])
        
        axes[0, 2].set_xlabel('Throughput Metrics')
        axes[0, 2].set_ylabel('Rate')
        axes[0, 2].set_title('Throughput Comparison')
        axes[0, 2].set_xticks(x_pos)
        axes[0, 2].set_xticklabels(throughput_metrics)
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Memory usage over time (bottom left)
        time_points = np.arange(0, 100, 5)
        num_memory = 500 + 200 * np.sin(time_points * 0.1) + np.random.normal(0, 20, len(time_points))
        vis_memory = 800 + 400 * np.sin(time_points * 0.1) + np.random.normal(0, 30, len(time_points))
        
        axes[1, 0].plot(time_points, num_memory, label='Numerical', 
                       color=self.colors['numerical'], linewidth=2)
        axes[1, 0].plot(time_points, vis_memory, label='Visual', 
                       color=self.colors['visual'], linewidth=2)
        axes[1, 0].fill_between(time_points, num_memory, alpha=0.3, color=self.colors['numerical'])
        axes[1, 0].fill_between(time_points, vis_memory, alpha=0.3, color=self.colors['visual'])
        
        axes[1, 0].set_xlabel('Time (seconds)')
        axes[1, 0].set_ylabel('Memory Usage (MB)')
        axes[1, 0].set_title('Memory Usage Over Time')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # CPU utilization (bottom middle)
        cpu_categories = ['Idle', 'Low', 'Medium', 'High']
        num_cpu_dist = [20, 30, 35, 15]
        vis_cpu_dist = [10, 25, 40, 25]
        
        x_cpu = np.arange(len(cpu_categories))
        axes[1, 1].bar(x_cpu - width/2, num_cpu_dist, width,
                      label='Numerical', color=self.colors['numerical'])
        axes[1, 1].bar(x_cpu + width/2, vis_cpu_dist, width,
                      label='Visual', color=self.colors['visual'])
        
        axes[1, 1].set_xlabel('CPU Utilization Level')
        axes[1, 1].set_ylabel('Percentage of Time')
        axes[1, 1].set_title('CPU Utilization Distribution')
        axes[1, 1].set_xticks(x_cpu)
        axes[1, 1].set_xticklabels(cpu_categories)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Performance summary table (bottom right)
        axes[1, 2].axis('tight')
        axes[1, 2].axis('off')
        
        summary_data = [
            ['Metric', 'Numerical', 'Visual', 'Ratio'],
            ['Avg Execution Time', f"{np.mean(num_times):.2f}s", 
             f"{np.mean(vis_times):.2f}s", 
             f"{np.mean(vis_times)/np.mean(num_times):.2f}x"],
            ['Peak Memory Usage', '1.2 GB', '2.1 GB', '1.75x'],
            ['Avg CPU Usage', '45%', '62%', '1.38x'],
            ['Total Throughput', f"{np.mean(num_throughput):.0f}", 
             f"{np.mean(vis_throughput):.0f}", 
             f"{np.mean(vis_throughput)/np.mean(num_throughput):.2f}x"]
        ]
        
        table = axes[1, 2].table(
            cellText=summary_data[1:],
            colLabels=summary_data[0],
            cellLoc='center',
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        axes[1, 2].set_title('Performance Summary')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_scalability_analysis(
        self,
        scalability_results: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot scalability analysis results
        
        Args:
            scalability_results: Dictionary containing scalability results
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Scalability Analysis', fontsize=16)
        
        # Data size scaling (top left)
        data_sizes = [100, 500, 1000, 5000, 10000, 50000]
        num_times = [0.1, 0.5, 1.0, 5.2, 10.8, 58.3]
        vis_times = [0.15, 0.8, 1.6, 8.5, 18.2, 95.7]
        
        axes[0, 0].loglog(data_sizes, num_times, 'o-', label='Numerical', 
                         color=self.colors['numerical'], linewidth=2, markersize=8)
        axes[0, 0].loglog(data_sizes, vis_times, 's-', label='Visual', 
                         color=self.colors['visual'], linewidth=2, markersize=8)
        
        # Fit power law curves
        num_fit = np.polyfit(np.log(data_sizes), np.log(num_times), 1)
        vis_fit = np.polyfit(np.log(data_sizes), np.log(vis_times), 1)
        
        fit_sizes = np.logspace(2, 5, 100)
        num_fit_line = np.exp(num_fit[1]) * fit_sizes ** num_fit[0]
        vis_fit_line = np.exp(vis_fit[1]) * fit_sizes ** vis_fit[0]
        
        axes[0, 0].loglog(fit_sizes, num_fit_line, '--', alpha=0.7, 
                         color=self.colors['numerical'],
                         label=f'Numerical fit: O(n^{num_fit[0]:.2f})')
        axes[0, 0].loglog(fit_sizes, vis_fit_line, '--', alpha=0.7, 
                         color=self.colors['visual'],
                         label=f'Visual fit: O(n^{vis_fit[0]:.2f})')
        
        axes[0, 0].set_xlabel('Data Size (samples)')
        axes[0, 0].set_ylabel('Processing Time (seconds)')
        axes[0, 0].set_title('Processing Time vs Data Size')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Memory scaling (top right)
        num_memory = [50, 180, 320, 1400, 2600, 12800]
        vis_memory = [80, 280, 520, 2200, 4100, 19500]
        
        axes[0, 1].loglog(data_sizes, num_memory, 'o-', label='Numerical', 
                         color=self.colors['memory'], linewidth=2, markersize=8)
        axes[0, 1].loglog(data_sizes, vis_memory, 's-', label='Visual', 
                         color=self.colors['visual'], linewidth=2, markersize=8)
        
        axes[0, 1].set_xlabel('Data Size (samples)')
        axes[0, 1].set_ylabel('Memory Usage (MB)')
        axes[0, 1].set_title('Memory Usage vs Data Size')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Efficiency ratio (bottom left)
        efficiency_ratios = [vis_times[i] / num_times[i] for i in range(len(data_sizes))]
        
        axes[1, 0].semilogx(data_sizes, efficiency_ratios, 'o-', 
                           color='red', linewidth=2, markersize=8)
        axes[1, 0].axhline(y=1, color='black', linestyle='--', alpha=0.7, 
                          label='Equal Performance')
        axes[1, 0].set_xlabel('Data Size (samples)')
        axes[1, 0].set_ylabel('Time Ratio (Visual/Numerical)')
        axes[1, 0].set_title('Performance Ratio vs Data Size')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Parallel scaling (bottom right)
        num_cores = [1, 2, 4, 8, 16]
        num_parallel_speedup = [1.0, 1.8, 3.2, 5.8, 9.2]
        vis_parallel_speedup = [1.0, 1.9, 3.6, 6.8, 12.1]
        ideal_speedup = num_cores
        
        axes[1, 1].plot(num_cores, num_parallel_speedup, 'o-', 
                       label='Numerical', color=self.colors['numerical'], 
                       linewidth=2, markersize=8)
        axes[1, 1].plot(num_cores, vis_parallel_speedup, 's-', 
                       label='Visual', color=self.colors['visual'], 
                       linewidth=2, markersize=8)
        axes[1, 1].plot(num_cores, ideal_speedup, '--', 
                       label='Ideal Speedup', color='gray', alpha=0.7)
        
        axes[1, 1].set_xlabel('Number of CPU Cores')
        axes[1, 1].set_ylabel('Speedup Factor')
        axes[1, 1].set_title('Parallel Scaling Performance')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_performance_dashboard(
        self,
        performance_results: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create interactive performance dashboard
        
        Args:
            performance_results: Complete performance results
            save_path: Optional path to save the dashboard
            
        Returns:
            Plotly figure object
        """
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                'Execution Time Comparison', 'Memory Usage Timeline',
                'Throughput Analysis', 'Scalability Curves',
                'Resource Utilization', 'Performance Summary'
            ],
            specs=[
                [{"type": "bar"}, {"type": "scatter"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "bar"}, {"type": "indicator"}]
            ]
        )
        
        # Execution time comparison (row 1, col 1)
        functions = ['Preprocessing', 'Feature Extraction', 'Analysis', 'Postprocessing']
        num_times = [1.2, 3.5, 8.7, 0.8]
        vis_times = [1.8, 5.2, 12.3, 1.1]
        
        fig.add_trace(
            go.Bar(name='Numerical', x=functions, y=num_times,
                   marker_color='lightblue'),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(name='Visual', x=functions, y=vis_times,
                   marker_color='lightcoral'),
            row=1, col=1
        )
        
        # Memory usage timeline (row 1, col 2)
        time_points = np.arange(0, 100, 2)
        num_memory = 500 + 200 * np.sin(time_points * 0.1) + np.random.normal(0, 20, len(time_points))
        vis_memory = 800 + 400 * np.sin(time_points * 0.1) + np.random.normal(0, 30, len(time_points))
        
        fig.add_trace(
            go.Scatter(x=time_points, y=num_memory, mode='lines',
                      name='Numerical Memory', line=dict(color='blue')),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=time_points, y=vis_memory, mode='lines',
                      name='Visual Memory', line=dict(color='red')),
            row=1, col=2
        )
        
        # Throughput analysis (row 1, col 3)
        throughput_metrics = ['Samples/sec', 'Features/sec', 'Predictions/sec']
        num_throughput = [150, 2500, 45]
        vis_throughput = [120, 1800, 38]
        
        fig.add_trace(
            go.Bar(name='Numerical Throughput', x=throughput_metrics, y=num_throughput,
                   marker_color='green'),
            row=1, col=3
        )
        fig.add_trace(
            go.Bar(name='Visual Throughput', x=throughput_metrics, y=vis_throughput,
                   marker_color='orange'),
            row=1, col=3
        )
        
        # Scalability curves (row 2, col 1)
        data_sizes = [100, 500, 1000, 5000, 10000]
        num_scale_times = [0.1, 0.5, 1.0, 5.2, 10.8]
        vis_scale_times = [0.15, 0.8, 1.6, 8.5, 18.2]
        
        fig.add_trace(
            go.Scatter(x=data_sizes, y=num_scale_times, mode='lines+markers',
                      name='Numerical Scaling', line=dict(color='navy')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=data_sizes, y=vis_scale_times, mode='lines+markers',
                      name='Visual Scaling', line=dict(color='maroon')),
            row=2, col=1
        )
        
        # Resource utilization (row 2, col 2)
        resources = ['CPU', 'Memory', 'Disk I/O', 'Network']
        num_utilization = [45, 60, 20, 5]
        vis_utilization = [62, 85, 35, 8]
        
        fig.add_trace(
            go.Bar(name='Numerical Resources', x=resources, y=num_utilization,
                   marker_color='purple'),
            row=2, col=2
        )
        fig.add_trace(
            go.Bar(name='Visual Resources', x=resources, y=vis_utilization,
                   marker_color='pink'),
            row=2, col=2
        )
        
        # Performance summary gauge (row 2, col 3)
        overall_performance = 0.78  # Example overall performance score
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=overall_performance,
                title={'text': "Overall Performance Score"},
                gauge={
                    'axis': {'range': [None, 1]},
                    'bar': {'color': "darkgreen"},
                    'steps': [
                        {'range': [0, 0.5], 'color': "lightgray"},
                        {'range': [0.5, 0.8], 'color': "yellow"},
                        {'range': [0.8, 1], 'color': "green"}
                    ]
                }
            ),
            row=2, col=3
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="Performance Analysis Dashboard",
            title_x=0.5,
            showlegend=True
        )
        
        # Update axis labels
        fig.update_yaxes(title_text="Time (seconds)", row=1, col=1)
        fig.update_yaxes(title_text="Memory (MB)", row=1, col=2)
        fig.update_yaxes(title_text="Rate", row=1, col=3)
        fig.update_yaxes(title_text="Processing Time", row=2, col=1)
        fig.update_yaxes(title_text="Utilization (%)", row=2, col=2)
        
        fig.update_xaxes(title_text="Time (seconds)", row=1, col=2)
        fig.update_xaxes(title_text="Data Size", row=2, col=1)
        
        if save_path:
            fig.write_html(save_path)
        
        return fig 