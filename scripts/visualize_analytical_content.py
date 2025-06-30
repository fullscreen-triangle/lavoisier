#!/usr/bin/env python3
"""
Comprehensive Analytical Content Visualization

This script creates detailed visualizations of the actual analytical content
from both numerical and visual pipelines, including XIC plots, mass spectra,
metabolite heatmaps, and all pipeline outputs.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from pathlib import Path
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy import signal
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Import our visualization modules
from validation.visualization.feature_plots import FeaturePlotter
from validation.visualization.quality_plots import QualityPlotter
from validation.visualization.performance_plots import PerformancePlotter
from validation.visualization.report_generator import ReportGenerator


class AnalyticalContentVisualizer:
    """
    Comprehensive visualization of analytical content from both pipelines
    """
    
    def __init__(self, output_dir: str = 'analytical_visualizations'):
        """
        Initialize the visualizer
        
        Args:
            output_dir: Base directory for output visualizations
        """
        self.base_dir = Path(output_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Create timestamped run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.base_dir / timestamp
        
        # Create subdirectories
        for subdir in [
            'xic_plots',
            'mass_spectra',
            'heatmaps',
            'feature_analysis',
            'interactive_dashboards',
            'publication_figures',
            'time_series',
            'reports'
        ]:
            (self.run_dir / subdir).mkdir(parents=True, exist_ok=True)
        
        # Initialize visualization tools with consistent style
        style = "publication"
        self.feature_plotter = FeaturePlotter(style=style)
        self.quality_plotter = QualityPlotter(style=style)
        self.performance_plotter = PerformancePlotter()
        self.report_generator = ReportGenerator()
        
        # Set color scheme
        self.colors = {
            'numerical': '#1f77b4',
            'visual': '#ff7f0e',
            'detected': '#2ca02c',
            'missed': '#d62728',
            'uncertain': '#9467bd'
        }
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
    
    def generate_synthetic_metabolomics_data(self) -> Dict[str, Any]:
        """
        Generate realistic synthetic metabolomics data for both pipelines
        
        Returns:
            Dictionary containing synthetic data for visualization
        """
        print("Generating synthetic metabolomics data...")
        
        # Time points for chromatography (30 minutes)
        time_points = np.linspace(0, 30, 1500)
        
        # Mass range for mass spectrometry
        mz_range = np.linspace(50, 500, 2000)
        
        # Define metabolites with their properties
        metabolites = {
            'Glucose': {'mz': 180.063, 'rt': 15.2, 'intensity': 1e6, 'width': 0.8},
            'Fructose': {'mz': 180.063, 'rt': 16.8, 'intensity': 8e5, 'width': 0.7},
            'Sucrose': {'mz': 342.116, 'rt': 18.5, 'intensity': 6e5, 'width': 0.9},
            'Lactate': {'mz': 90.032, 'rt': 6.3, 'intensity': 1.2e6, 'width': 0.6},
            'Pyruvate': {'mz': 88.016, 'rt': 7.1, 'intensity': 4e5, 'width': 0.5},
            'Alanine': {'mz': 89.047, 'rt': 12.1, 'intensity': 7e5, 'width': 0.7},
            'Glycine': {'mz': 75.032, 'rt': 8.9, 'intensity': 5e5, 'width': 0.6},
            'Serine': {'mz': 105.043, 'rt': 11.2, 'intensity': 6e5, 'width': 0.8},
            'Leucine': {'mz': 131.095, 'rt': 8.7, 'intensity': 8e5, 'width': 0.6},
            'Valine': {'mz': 117.079, 'rt': 14.8, 'intensity': 7e5, 'width': 0.7},
            'Citrate': {'mz': 192.027, 'rt': 18.5, 'intensity': 3e5, 'width': 1.0},
            'Malate': {'mz': 134.022, 'rt': 16.2, 'intensity': 4e5, 'width': 0.8},
            'Succinate': {'mz': 118.027, 'rt': 14.1, 'intensity': 5e5, 'width': 0.7},
            'Fumarate': {'mz': 116.011, 'rt': 13.8, 'intensity': 3e5, 'width': 0.6},
            'ATP': {'mz': 507.181, 'rt': 22.3, 'intensity': 2e5, 'width': 1.2},
            'ADP': {'mz': 427.201, 'rt': 21.1, 'intensity': 3e5, 'width': 1.1},
            'AMP': {'mz': 347.221, 'rt': 19.8, 'intensity': 4e5, 'width': 1.0},
            'NAD+': {'mz': 663.425, 'rt': 25.2, 'intensity': 1e5, 'width': 1.3},
            'NADH': {'mz': 665.441, 'rt': 24.8, 'intensity': 8e4, 'width': 1.3},
            'Glutamate': {'mz': 147.053, 'rt': 13.5, 'intensity': 9e5, 'width': 0.8}
        }
        
        # Generate XIC data for each metabolite
        xic_data = {}
        for name, props in metabolites.items():
            # Numerical pipeline XIC
            num_xic = props['intensity'] * np.exp(-0.5 * ((time_points - props['rt']) / props['width']) ** 2)
            num_xic += np.random.normal(0, props['intensity'] * 0.05, len(time_points))
            num_xic = np.maximum(num_xic, 0)  # Ensure non-negative
            
            # Visual pipeline XIC (with slight differences)
            vis_xic = props['intensity'] * 0.95 * np.exp(-0.5 * ((time_points - props['rt']) / props['width']) ** 2)
            vis_xic += np.random.normal(0, props['intensity'] * 0.06, len(time_points))
            vis_xic = np.maximum(vis_xic, 0)
            
            xic_data[name] = {
                'time': time_points,
                'numerical': num_xic,
                'visual': vis_xic,
                'mz': props['mz'],
                'rt': props['rt']
            }
        
        # Generate mass spectra
        spectra_data = {}
        
        # Full scan spectrum
        full_scan_num = np.random.exponential(1000, len(mz_range))
        full_scan_vis = np.random.exponential(1100, len(mz_range))
        
        # Add metabolite peaks to full scan
        for name, props in metabolites.items():
            peak_idx = np.argmin(np.abs(mz_range - props['mz']))
            full_scan_num[peak_idx] += props['intensity'] / 1000
            full_scan_vis[peak_idx] += props['intensity'] / 1000 * 0.95
        
        spectra_data['full_scan'] = {
            'mz': mz_range,
            'numerical': full_scan_num,
            'visual': full_scan_vis
        }
        
        # MS/MS spectra for selected compounds
        # Glucose fragmentation pattern
        glucose_fragments = {
            180.063: 100,  # Molecular ion
            162.053: 45,   # Loss of H2O
            144.042: 30,   # Loss of 2 H2O
            126.032: 25,   # Loss of 3 H2O
            108.021: 15,   # Loss of 4 H2O
            90.011: 10,    # Loss of 5 H2O
            72.001: 8,     # Further fragmentation
            60.021: 12     # C2H4O2
        }
        
        glucose_mz = list(glucose_fragments.keys())
        glucose_int_num = list(glucose_fragments.values())
        glucose_int_vis = [i * 0.95 + np.random.normal(0, 2) for i in glucose_int_num]
        
        spectra_data['glucose_msms'] = {
            'mz': glucose_mz,
            'numerical': glucose_int_num,
            'visual': glucose_int_vis
        }
        
        # Generate sample matrix (metabolite abundances across samples)
        n_samples = 24
        sample_names = [f'Sample_{i+1:02d}' for i in range(n_samples)]
        
        # Create abundance matrix
        np.random.seed(42)
        abundance_matrix_num = np.random.lognormal(mean=10, sigma=1.5, size=(len(metabolites), n_samples))
        
        # Add some biological variation patterns
        # Group 1: Control samples (samples 1-8)
        # Group 2: Treatment A samples (samples 9-16)
        # Group 3: Treatment B samples (samples 17-24)
        
        for i, (name, props) in enumerate(metabolites.items()):
            # Control group baseline
            abundance_matrix_num[i, 0:8] *= 1.0
            
            # Treatment A effects
            if name in ['Glucose', 'Lactate', 'Pyruvate']:
                abundance_matrix_num[i, 8:16] *= 1.5  # Upregulated
            elif name in ['ATP', 'ADP', 'Citrate']:
                abundance_matrix_num[i, 8:16] *= 0.7  # Downregulated
            
            # Treatment B effects
            if name in ['Alanine', 'Glycine', 'Serine']:
                abundance_matrix_num[i, 16:24] *= 2.0  # Highly upregulated
            elif name in ['Succinate', 'Fumarate', 'Malate']:
                abundance_matrix_num[i, 16:24] *= 0.5  # Downregulated
        
        # Visual pipeline matrix (with slight differences)
        abundance_matrix_vis = abundance_matrix_num * np.random.normal(0.95, 0.1, abundance_matrix_num.shape)
        abundance_matrix_vis = np.abs(abundance_matrix_vis)  # Ensure positive
        
        # Create time series data
        time_series = self._generate_time_series_data(metabolites)
        
        return {
            'xic_data': xic_data,
            'spectra_data': spectra_data,
            'abundance_data': {
                'numerical': abundance_matrix_num,
                'visual': abundance_matrix_vis,
                'metabolites': list(metabolites.keys()),
                'samples': sample_names
            },
            'time_series': time_series
        }
    
    def _generate_time_series_data(self, metabolites: Dict[str, Dict[str, float]]) -> Dict[str, np.ndarray]:
        """Generate time series data for metabolites"""
        n_timepoints = 100
        time = np.linspace(0, 24, n_timepoints)  # 24 hours
        
        base_signal = np.sin(time * np.pi / 12)  # 12-hour cycle
        noise = lambda: np.random.normal(0, 0.05, n_timepoints)
        
        time_series = {'time': time}
        
        for name in metabolites:
            time_series[f'{name}_numerical'] = 0.8 + 0.2 * base_signal + noise()
            time_series[f'{name}_visual'] = 0.75 + 0.18 * base_signal + noise()
        
        return time_series
    
    def create_comprehensive_xic_plots(self, data: Dict[str, Any]) -> None:
        """
        Create comprehensive XIC visualizations
        
        Args:
            data: Dictionary containing XIC data
        """
        print("Creating XIC visualizations...")
        
        xic_data = data['xic_data']
        
        # Create static XIC plots
        for name, compound_data in xic_data.items():
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.plot(compound_data['time'], compound_data['numerical'],
                   label='Numerical', color=self.colors['numerical'])
            ax.plot(compound_data['time'], compound_data['visual'],
                   label='Visual', color=self.colors['visual'])
            
            ax.set_xlabel('Retention Time (min)')
            ax.set_ylabel('Intensity')
            ax.set_title(f'{name} XIC (m/z {compound_data["mz"]:.3f})')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.savefig(self.run_dir / 'xic_plots' / f'{name}_xic.png',
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        # Create interactive XIC dashboard
        fig = make_subplots(rows=4, cols=5, subplot_titles=list(xic_data.keys()))
        
        row, col = 1, 1
        for name, compound_data in xic_data.items():
            fig.add_trace(
                go.Scatter(x=compound_data['time'],
                          y=compound_data['numerical'],
                          name=f'{name} (Numerical)',
                          line=dict(color=self.colors['numerical'])),
                row=row, col=col
            )
            
            fig.add_trace(
                go.Scatter(x=compound_data['time'],
                          y=compound_data['visual'],
                          name=f'{name} (Visual)',
                          line=dict(color=self.colors['visual'])),
                row=row, col=col
            )
            
            col += 1
            if col > 5:
                col = 1
                row += 1
        
        fig.update_layout(height=1200, showlegend=False,
                         title_text='Interactive XIC Comparison')
        
        fig.write_html(self.run_dir / 'interactive_dashboards' / 'xic_dashboard.html')
        
        print("✓ XIC visualizations complete")
    
    def create_mass_spectra_visualizations(self, data: Dict[str, Any]) -> None:
        """
        Create mass spectra visualizations
        
        Args:
            data: Dictionary containing mass spectra data
        """
        print("Creating mass spectra visualizations...")
        
        spectra_data = data['spectra_data']
        
        # Full scan spectrum
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(spectra_data['full_scan']['mz'],
                spectra_data['full_scan']['numerical'],
                label='Numerical', color=self.colors['numerical'])
        ax.plot(spectra_data['full_scan']['mz'],
                spectra_data['full_scan']['visual'],
                label='Visual', color=self.colors['visual'])
        
        ax.set_xlabel('m/z')
        ax.set_ylabel('Intensity')
        ax.set_title('Full Scan Mass Spectrum')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.savefig(self.run_dir / 'mass_spectra' / 'full_scan.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # MS/MS spectrum
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.bar(spectra_data['glucose_msms']['mz'],
               spectra_data['glucose_msms']['numerical'],
               alpha=0.6, label='Numerical', color=self.colors['numerical'])
        ax.bar(spectra_data['glucose_msms']['mz'],
               spectra_data['glucose_msms']['visual'],
               alpha=0.6, label='Visual', color=self.colors['visual'])
        
        ax.set_xlabel('m/z')
        ax.set_ylabel('Relative Intensity (%)')
        ax.set_title('Glucose MS/MS Spectrum')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.savefig(self.run_dir / 'mass_spectra' / 'glucose_msms.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create interactive mass spectra dashboard
        fig = make_subplots(rows=2, cols=1,
                          subplot_titles=['Full Scan Mass Spectrum',
                                        'Glucose MS/MS Spectrum'])
        
        # Full scan
        fig.add_trace(
            go.Scatter(x=spectra_data['full_scan']['mz'],
                      y=spectra_data['full_scan']['numerical'],
                      name='Numerical',
                      line=dict(color=self.colors['numerical'])),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=spectra_data['full_scan']['mz'],
                      y=spectra_data['full_scan']['visual'],
                      name='Visual',
                      line=dict(color=self.colors['visual'])),
            row=1, col=1
        )
        
        # MS/MS
        fig.add_trace(
            go.Bar(x=spectra_data['glucose_msms']['mz'],
                  y=spectra_data['glucose_msms']['numerical'],
                  name='Numerical MS/MS',
                  marker_color=self.colors['numerical']),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(x=spectra_data['glucose_msms']['mz'],
                  y=spectra_data['glucose_msms']['visual'],
                  name='Visual MS/MS',
                  marker_color=self.colors['visual']),
            row=2, col=1
        )
        
        fig.update_layout(height=800, showlegend=True,
                         title_text='Interactive Mass Spectra Comparison')
        
        fig.write_html(self.run_dir / 'interactive_dashboards' / 'mass_spectra_dashboard.html')
        
        print("✓ Mass spectra visualizations complete")
    
    def create_metabolite_heatmaps(self, data: Dict[str, Any]) -> None:
        """
        Create metabolite abundance heatmaps
        
        Args:
            data: Dictionary containing abundance data
        """
        print("Creating metabolite heatmaps...")
        
        abundance_data = data['abundance_data']
        
        # Create static heatmaps
        for pipeline in ['numerical', 'visual']:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            sns.heatmap(abundance_data[pipeline],
                       xticklabels=abundance_data['samples'],
                       yticklabels=abundance_data['metabolites'],
                       cmap='viridis',
                       ax=ax)
            
            ax.set_title(f'{pipeline.capitalize()} Pipeline Metabolite Abundances')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            plt.savefig(self.run_dir / 'heatmaps' / f'{pipeline}_heatmap.png',
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        # Create interactive heatmap
        fig = make_subplots(rows=1, cols=2,
                          subplot_titles=['Numerical Pipeline',
                                        'Visual Pipeline'])
        
        for i, pipeline in enumerate(['numerical', 'visual']):
            fig.add_trace(
                go.Heatmap(z=abundance_data[pipeline],
                          x=abundance_data['samples'],
                          y=abundance_data['metabolites'],
                          colorscale='Viridis',
                          name=pipeline.capitalize()),
                row=1, col=i+1
            )
        
        fig.update_layout(height=800,
                         title_text='Interactive Metabolite Abundance Comparison')
        
        fig.write_html(self.run_dir / 'interactive_dashboards' / 'metabolite_heatmap.html')
        
        print("✓ Metabolite heatmaps complete")
    
    def create_feature_analysis_plots(self, data: Dict[str, Any]) -> None:
        """
        Create feature analysis plots using FeaturePlotter
        
        Args:
            data: Dictionary containing abundance data
        """
        print("Creating feature analysis plots...")
        
        abundance_data = data['abundance_data']
        
        # Extract features
        numerical_features = abundance_data['numerical'].T
        visual_features = abundance_data['visual'].T
        feature_names = abundance_data['metabolites']
        
        # Create static feature plots
        feature_fig = self.feature_plotter.plot_feature_comparison(
            numerical_features=numerical_features,
            visual_features=visual_features,
            feature_names=feature_names,
            output_path=str(self.run_dir / 'feature_analysis' / 'feature_comparison.png')
        )
        
        # Create interactive feature dashboard
        feature_dashboard = self.feature_plotter.create_interactive_feature_dashboard(
            numerical_features=numerical_features,
            visual_features=visual_features,
            feature_names=feature_names,
            output_path=str(self.run_dir / 'interactive_dashboards' / 'feature_dashboard.html')
        )
        
        print("✓ Feature analysis plots complete")
    
    def create_time_series_plots(self, data: Dict[str, Any]) -> None:
        """
        Create time series analysis plots
        
        Args:
            data: Dictionary containing time series data
        """
        print("Creating time series plots...")
        
        time_series = data['time_series']
        metabolites = data['abundance_data']['metabolites']
        
        # Create static time series plots
        for metabolite in metabolites:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.plot(time_series['time'],
                   time_series[f'{metabolite}_numerical'],
                   label='Numerical', color=self.colors['numerical'])
            ax.plot(time_series['time'],
                   time_series[f'{metabolite}_visual'],
                   label='Visual', color=self.colors['visual'])
            
            ax.set_xlabel('Time (hours)')
            ax.set_ylabel('Normalized Intensity')
            ax.set_title(f'{metabolite} Time Series')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.savefig(self.run_dir / 'time_series' / f'{metabolite}_time_series.png',
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        # Create interactive time series dashboard
        fig = make_subplots(rows=4, cols=5,
                          subplot_titles=metabolites)
        
        row, col = 1, 1
        for metabolite in metabolites:
            fig.add_trace(
                go.Scatter(x=time_series['time'],
                          y=time_series[f'{metabolite}_numerical'],
                          name=f'{metabolite} (Numerical)',
                          line=dict(color=self.colors['numerical'])),
                row=row, col=col
            )
            
            fig.add_trace(
                go.Scatter(x=time_series['time'],
                          y=time_series[f'{metabolite}_visual'],
                          name=f'{metabolite} (Visual)',
                          line=dict(color=self.colors['visual'])),
                row=row, col=col
            )
            
            col += 1
            if col > 5:
                col = 1
                row += 1
        
        fig.update_layout(height=1200, showlegend=False,
                         title_text='Interactive Time Series Comparison')
        
        fig.write_html(self.run_dir / 'interactive_dashboards' / 'time_series_dashboard.html')
        
        print("✓ Time series plots complete")
    
    def generate_summary_report(self, data: Dict[str, Any]) -> None:
        """
        Generate comprehensive summary report
        
        Args:
            data: Complete analytical data
        """
        print("Generating summary report...")
        
        report_path = self.run_dir / 'reports' / 'analytical_summary.html'
        
        self.report_generator.create_report(
            validation_results={
                'analytical_data': data,
                'static_plots_dir': str(self.run_dir / 'static_plots'),
                'interactive_plots_dir': str(self.run_dir / 'interactive_dashboards'),
                'time_series_dir': str(self.run_dir / 'time_series')
            },
            output_path=str(report_path)
        )
        
        print(f"✓ Summary report generated at {report_path}")
    
    def run_complete_analysis(self) -> None:
        """Run complete analytical content visualization pipeline"""
        print("\n=== Starting Analytical Content Visualization Pipeline ===\n")
        
        # Generate synthetic data
        data = self.generate_synthetic_metabolomics_data()
        
        # Create all visualizations
        self.create_comprehensive_xic_plots(data)
        self.create_mass_spectra_visualizations(data)
        self.create_metabolite_heatmaps(data)
        self.create_feature_analysis_plots(data)
        self.create_time_series_plots(data)
        
        # Generate summary report
        self.generate_summary_report(data)
        
        print("\n=== Visualization Pipeline Complete ===")
        print(f"Results saved in: {self.run_dir}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analytical Content Visualization")
    parser.add_argument("--output", type=str, default="analytical_visualizations",
                      help="Output directory for visualizations")
    args = parser.parse_args()
    
    visualizer = AnalyticalContentVisualizer(output_dir=args.output)
    visualizer.run_complete_analysis()


if __name__ == "__main__":
    main() 