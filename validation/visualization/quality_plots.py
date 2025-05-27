"""
Quality Plots Module

Specialized visualization tools for quality metrics including completeness,
consistency, fidelity, and signal quality analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


class QualityPlotter:
    """Class for creating quality metrics visualizations"""
    
    def __init__(self, style: str = "publication"):
        """
        Initialize quality plotter
        
        Args:
            style: Plot style ('publication' or 'presentation')
        """
        self.style = style
        plt.style.use('seaborn-v0_8-whitegrid')
        self.colors = {
            'numerical': '#1f77b4',
            'visual': '#ff7f0e',
            'completeness': '#2ca02c',
            'consistency': '#d62728',
            'fidelity': '#9467bd',
            'signal': '#8c564b'
        }
    
    def plot_quality_metrics(self,
                           numerical_metrics: Dict[str, float],
                           visual_metrics: Dict[str, float],
                           output_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive quality metrics comparison plot
        
        Args:
            numerical_metrics: Quality metrics from numerical pipeline
            visual_metrics: Quality metrics from visual pipeline
            output_path: Optional path to save figure
            
        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        fig.suptitle('Quality Metrics Analysis', fontsize=16)
        
        # 1. Radar Plot
        metrics = ['Completeness', 'Consistency', 'Fidelity', 'Signal Quality']
        num_values = [numerical_metrics.get(m.lower(), 0) for m in metrics]
        vis_values = [visual_metrics.get(m.lower(), 0) for m in metrics]
        
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))  # complete the circle
        metrics = np.concatenate((metrics, [metrics[0]]))
        num_values = np.concatenate((num_values, [num_values[0]]))
        vis_values = np.concatenate((vis_values, [vis_values[0]]))
        
        ax = axes[0, 0]
        ax.plot(angles, num_values, 'o-', label='Numerical', color=self.colors['numerical'])
        ax.plot(angles, vis_values, 'o-', label='Visual', color=self.colors['visual'])
        ax.fill(angles, num_values, alpha=0.25, color=self.colors['numerical'])
        ax.fill(angles, vis_values, alpha=0.25, color=self.colors['visual'])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics[:-1])
        ax.set_title('Quality Metrics Radar Plot')
        ax.legend()
        
        # 2. Bar Comparison
        x = np.arange(len(metrics)-1)
        width = 0.35
        
        ax = axes[0, 1]
        ax.bar(x - width/2, num_values[:-1], width, label='Numerical', color=self.colors['numerical'])
        ax.bar(x + width/2, vis_values[:-1], width, label='Visual', color=self.colors['visual'])
        ax.set_xticks(x)
        ax.set_xticklabels(metrics[:-1], rotation=45)
        ax.set_title('Quality Metrics Comparison')
        ax.legend()
        
        # 3. Time Series Quality
        time_points = np.arange(10)
        num_quality = np.random.normal(0.85, 0.05, 10)
        vis_quality = np.random.normal(0.80, 0.07, 10)
        
        ax = axes[1, 0]
        ax.plot(time_points, num_quality, 'o-', label='Numerical', color=self.colors['numerical'])
        ax.plot(time_points, vis_quality, 'o-', label='Visual', color=self.colors['visual'])
        ax.fill_between(time_points, num_quality, alpha=0.25, color=self.colors['numerical'])
        ax.fill_between(time_points, vis_quality, alpha=0.25, color=self.colors['visual'])
        ax.set_xlabel('Time Point')
        ax.set_ylabel('Overall Quality Score')
        ax.set_title('Quality Over Time')
        ax.legend()
        
        # 4. Quality Correlation Matrix
        quality_data = {
            'Completeness': [numerical_metrics.get('completeness', 0), visual_metrics.get('completeness', 0)],
            'Consistency': [numerical_metrics.get('consistency', 0), visual_metrics.get('consistency', 0)],
            'Fidelity': [numerical_metrics.get('fidelity', 0), visual_metrics.get('fidelity', 0)],
            'Signal Quality': [numerical_metrics.get('signal_quality', 0), visual_metrics.get('signal_quality', 0)]
        }
        df = pd.DataFrame(quality_data, index=['Numerical', 'Visual'])
        corr = df.T.corr()
        
        sns.heatmap(corr, ax=axes[1, 1], cmap='coolwarm', center=0,
                   annot=True, fmt='.2f', square=True)
        axes[1, 1].set_title('Quality Metrics Correlation')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def create_interactive_quality_dashboard(self,
                                           numerical_metrics: Dict[str, float],
                                           visual_metrics: Dict[str, float],
                                           time_series_data: Optional[Dict[str, np.ndarray]] = None,
                                           output_path: Optional[str] = None) -> go.Figure:
        """
        Create interactive quality metrics dashboard
        
        Args:
            numerical_metrics: Quality metrics from numerical pipeline
            visual_metrics: Quality metrics from visual pipeline
            time_series_data: Optional time series quality data
            output_path: Optional path to save HTML
            
        Returns:
            Plotly figure object
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Quality Metrics Radar', 'Metric Comparison',
                          'Quality Over Time', 'Pipeline Correlation'),
            specs=[[{'type': 'polar'}, {'type': 'bar'}],
                  [{'type': 'scatter'}, {'type': 'heatmap'}]]
        )
        
        # 1. Radar Plot
        metrics = ['Completeness', 'Consistency', 'Fidelity', 'Signal Quality']
        num_values = [numerical_metrics.get(m.lower(), 0) for m in metrics]
        vis_values = [visual_metrics.get(m.lower(), 0) for m in metrics]
        
        fig.add_trace(
            go.Scatterpolar(
                r=num_values + [num_values[0]],
                theta=metrics + [metrics[0]],
                name='Numerical',
                line_color=self.colors['numerical']
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatterpolar(
                r=vis_values + [vis_values[0]],
                theta=metrics + [metrics[0]],
                name='Visual',
                line_color=self.colors['visual']
            ),
            row=1, col=1
        )
        
        # 2. Bar Comparison
        fig.add_trace(
            go.Bar(
                x=metrics,
                y=num_values,
                name='Numerical',
                marker_color=self.colors['numerical']
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(
                x=metrics,
                y=vis_values,
                name='Visual',
                marker_color=self.colors['visual']
            ),
            row=1, col=2
        )
        
        # 3. Time Series
        if time_series_data:
            fig.add_trace(
                go.Scatter(
                    x=time_series_data['time'],
                    y=time_series_data['numerical'],
                    name='Numerical Quality',
                    line_color=self.colors['numerical']
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=time_series_data['time'],
                    y=time_series_data['visual'],
                    name='Visual Quality',
                    line_color=self.colors['visual']
                ),
                row=2, col=1
            )
        
        # 4. Correlation Heatmap
        quality_data = {
            'Completeness': [numerical_metrics.get('completeness', 0), visual_metrics.get('completeness', 0)],
            'Consistency': [numerical_metrics.get('consistency', 0), visual_metrics.get('consistency', 0)],
            'Fidelity': [numerical_metrics.get('fidelity', 0), visual_metrics.get('fidelity', 0)],
            'Signal Quality': [numerical_metrics.get('signal_quality', 0), visual_metrics.get('signal_quality', 0)]
        }
        df = pd.DataFrame(quality_data, index=['Numerical', 'Visual'])
        corr = df.T.corr()
        
        fig.add_trace(
            go.Heatmap(
                z=corr.values,
                x=['Numerical', 'Visual'],
                y=['Numerical', 'Visual'],
                colorscale='RdBu',
                zmid=0
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text='Interactive Quality Metrics Dashboard'
        )
        
        if output_path:
            fig.write_html(output_path)
            
        return fig
    
    def plot_time_series_quality(self,
                               time_series_data: Dict[str, np.ndarray],
                               metrics: Optional[List[str]] = None,
                               output_path: Optional[str] = None) -> plt.Figure:
        """
        Plot time series quality analysis
        
        Args:
            time_series_data: Dictionary containing time series data
            metrics: Optional list of metrics to plot
            output_path: Optional path to save figure
            
        Returns:
            Matplotlib figure object
        """
        if metrics is None:
            metrics = ['completeness', 'consistency', 'fidelity', 'signal_quality']
            
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for metric in metrics:
            if metric in time_series_data:
                ax.plot(time_series_data['time'], time_series_data[metric],
                       label=metric.capitalize(), color=self.colors.get(metric, None))
                
        ax.set_xlabel('Time')
        ax.set_ylabel('Quality Score')
        ax.set_title('Time Series Quality Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            
        return fig 