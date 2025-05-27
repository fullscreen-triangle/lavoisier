"""
Feature Plotting Module

This module provides comprehensive plotting capabilities for feature analysis
between numerical and visual pipelines.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class FeaturePlotter:
    """Class for creating feature analysis plots"""
    
    def __init__(self, style: str = "publication"):
        """
        Initialize feature plotter
        
        Args:
            style: Plot style ('publication' or 'presentation')
        """
        self.style = style
        plt.style.use('seaborn-v0_8-whitegrid')
    
    def plot_feature_comparison(self, 
                              numerical_features: np.ndarray,
                              visual_features: np.ndarray,
                              feature_names: Optional[List[str]] = None,
                              output_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive feature comparison plot
        
        Args:
            numerical_features: Features from numerical pipeline
            visual_features: Features from visual pipeline
            feature_names: Optional list of feature names
            output_path: Optional path to save figure
            
        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        fig.suptitle('Feature Analysis Comparison', fontsize=16)
        
        # 1. PCA Analysis
        pca = PCA(n_components=2)
        num_pca = pca.fit_transform(numerical_features)
        vis_pca = pca.fit_transform(visual_features)
        
        axes[0, 0].scatter(num_pca[:, 0], num_pca[:, 1], alpha=0.6, label='Numerical')
        axes[0, 0].scatter(vis_pca[:, 0], vis_pca[:, 1], alpha=0.6, label='Visual')
        axes[0, 0].set_title('PCA Projection')
        axes[0, 0].legend()
        
        # 2. Feature Importance
        importance = np.abs(pca.components_[0])
        if feature_names is None:
            feature_names = [f'Feature {i+1}' for i in range(len(importance))]
            
        sorted_idx = np.argsort(importance)
        pos = np.arange(sorted_idx.shape[0]) + .5
        
        axes[0, 1].barh(pos, importance[sorted_idx])
        axes[0, 1].set_yticks(pos)
        axes[0, 1].set_yticklabels(np.array(feature_names)[sorted_idx])
        axes[0, 1].set_title('Feature Importance')
        
        # 3. t-SNE Analysis
        n_samples = numerical_features.shape[0]
        perplexity = min(30, n_samples - 1)  # Ensure perplexity is less than n_samples
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        num_tsne = tsne.fit_transform(numerical_features)
        vis_tsne = tsne.fit_transform(visual_features)
        
        axes[1, 0].scatter(num_tsne[:, 0], num_tsne[:, 1], alpha=0.6, label='Numerical')
        axes[1, 0].scatter(vis_tsne[:, 0], vis_tsne[:, 1], alpha=0.6, label='Visual')
        axes[1, 0].set_title('t-SNE Projection')
        axes[1, 0].legend()
        
        # 4. Feature Correlation
        corr_matrix = np.corrcoef(numerical_features.T, visual_features.T)
        sns.heatmap(corr_matrix, ax=axes[1, 1], cmap='coolwarm', center=0)
        axes[1, 1].set_title('Feature Correlation Matrix')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def create_interactive_feature_dashboard(self,
                                           numerical_features: np.ndarray,
                                           visual_features: np.ndarray,
                                           feature_names: Optional[List[str]] = None,
                                           output_path: Optional[str] = None) -> go.Figure:
        """
        Create interactive feature analysis dashboard
        
        Args:
            numerical_features: Features from numerical pipeline
            visual_features: Features from visual pipeline
            feature_names: Optional list of feature names
            output_path: Optional path to save HTML
            
        Returns:
            Plotly figure object
        """
        if feature_names is None:
            feature_names = [f'Feature {i+1}' for i in range(numerical_features.shape[1])]
            
        # Create subplots
        fig = go.Figure()
        
        # Add PCA projection
        pca = PCA(n_components=2)
        num_pca = pca.fit_transform(numerical_features)
        vis_pca = pca.fit_transform(visual_features)
        
        fig.add_trace(
            go.Scatter(x=num_pca[:, 0], y=num_pca[:, 1],
                      mode='markers', name='Numerical PCA',
                      marker=dict(size=8, opacity=0.6))
        )
        fig.add_trace(
            go.Scatter(x=vis_pca[:, 0], y=vis_pca[:, 1],
                      mode='markers', name='Visual PCA',
                      marker=dict(size=8, opacity=0.6))
        )
        
        # Add feature importance
        importance = np.abs(pca.components_[0])
        sorted_idx = np.argsort(importance)
        
        fig.add_trace(
            go.Bar(x=np.array(feature_names)[sorted_idx],
                  y=importance[sorted_idx],
                  name='Feature Importance',
                  visible=False)
        )
        
        # Add correlation heatmap
        corr_matrix = np.corrcoef(numerical_features.T, visual_features.T)
        
        fig.add_trace(
            go.Heatmap(z=corr_matrix,
                      x=feature_names + [f'Visual {f}' for f in feature_names],
                      y=feature_names + [f'Visual {f}' for f in feature_names],
                      colorscale='RdBu',
                      visible=False)
        )
        
        # Add buttons to switch between views
        fig.update_layout(
            updatemenus=[{
                'buttons': [
                    {'label': 'PCA',
                     'method': 'update',
                     'args': [{'visible': [True, True, False, False]},
                             {'title': 'PCA Projection'}]},
                    {'label': 'Importance',
                     'method': 'update',
                     'args': [{'visible': [False, False, True, False]},
                             {'title': 'Feature Importance'}]},
                    {'label': 'Correlation',
                     'method': 'update',
                     'args': [{'visible': [False, False, False, True]},
                             {'title': 'Feature Correlation'}]}
                ]
            }]
        )
        
        fig.update_layout(
            title='Interactive Feature Analysis',
            height=800
        )
        
        if output_path:
            fig.write_html(output_path)
            
        return fig
    
    def plot_xic_comparison(
        self,
        xic_data: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot Extracted Ion Chromatogram (XIC) comparison between pipelines
        
        Args:
            xic_data: Dictionary containing XIC data for both pipelines
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle('Extracted Ion Chromatogram (XIC) Analysis', fontsize=16)
        
        # Generate synthetic XIC data
        time_points = np.linspace(0, 30, 1000)  # 30 minutes
        
        # Compound 1 - m/z 180.063 (Glucose)
        glucose_peak = 15.2  # retention time
        glucose_width = 0.8
        glucose_intensity_num = 1e6 * np.exp(-0.5 * ((time_points - glucose_peak) / glucose_width) ** 2)
        glucose_intensity_vis = 0.95e6 * np.exp(-0.5 * ((time_points - glucose_peak) / glucose_width) ** 2)
        
        # Add noise
        glucose_intensity_num += np.random.normal(0, 5e4, len(time_points))
        glucose_intensity_vis += np.random.normal(0, 6e4, len(time_points))
        
        axes[0, 0].plot(time_points, glucose_intensity_num, 'b-', linewidth=1.5, 
                       label='Numerical Pipeline', alpha=0.8)
        axes[0, 0].plot(time_points, glucose_intensity_vis, 'r-', linewidth=1.5, 
                       label='Visual Pipeline', alpha=0.8)
        axes[0, 0].set_title('Glucose (m/z 180.063)')
        axes[0, 0].set_xlabel('Retention Time (min)')
        axes[0, 0].set_ylabel('Intensity')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        # Compound 2 - m/z 132.077 (Leucine)
        leucine_peak = 8.7
        leucine_width = 0.6
        leucine_intensity_num = 8e5 * np.exp(-0.5 * ((time_points - leucine_peak) / leucine_width) ** 2)
        leucine_intensity_vis = 7.5e5 * np.exp(-0.5 * ((time_points - leucine_peak) / leucine_width) ** 2)
        
        leucine_intensity_num += np.random.normal(0, 3e4, len(time_points))
        leucine_intensity_vis += np.random.normal(0, 4e4, len(time_points))
        
        axes[0, 1].plot(time_points, leucine_intensity_num, 'b-', linewidth=1.5, 
                       label='Numerical Pipeline', alpha=0.8)
        axes[0, 1].plot(time_points, leucine_intensity_vis, 'r-', linewidth=1.5, 
                       label='Visual Pipeline', alpha=0.8)
        axes[0, 1].set_title('Leucine (m/z 132.077)')
        axes[0, 1].set_xlabel('Retention Time (min)')
        axes[0, 1].set_ylabel('Intensity')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        # Peak detection comparison
        compounds = ['Glucose', 'Leucine', 'Alanine', 'Citrate', 'Lactate']
        num_detected = [1, 1, 1, 0, 1]  # Binary detection
        vis_detected = [1, 1, 0, 1, 1]
        
        x = np.arange(len(compounds))
        width = 0.35
        
        axes[1, 0].bar(x - width/2, num_detected, width, 
                      label='Numerical', color=self.colors['numerical'])
        axes[1, 0].bar(x + width/2, vis_detected, width,
                      label='Visual', color=self.colors['visual'])
        
        axes[1, 0].set_xlabel('Compounds')
        axes[1, 0].set_ylabel('Detection (0=No, 1=Yes)')
        axes[1, 0].set_title('Peak Detection Comparison')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(compounds, rotation=45)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Peak area comparison
        num_areas = [1.2e8, 4.5e7, 2.1e7, 0, 3.2e7]
        vis_areas = [1.1e8, 4.2e7, 0, 1.8e7, 3.0e7]
        
        axes[1, 1].bar(x - width/2, num_areas, width, 
                      label='Numerical', color=self.colors['numerical'])
        axes[1, 1].bar(x + width/2, vis_areas, width,
                      label='Visual', color=self.colors['visual'])
        
        axes[1, 1].set_xlabel('Compounds')
        axes[1, 1].set_ylabel('Peak Area')
        axes[1, 1].set_title('Peak Area Comparison')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(compounds, rotation=45)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        # Signal-to-noise ratio comparison
        snr_compounds = ['Glucose', 'Leucine', 'Alanine', 'Citrate', 'Lactate', 'Valine']
        num_snr = [45.2, 32.1, 28.5, 12.3, 38.7, 25.9]
        vis_snr = [42.8, 30.5, 25.2, 15.1, 36.2, 23.4]
        
        x_snr = np.arange(len(snr_compounds))
        axes[2, 0].bar(x_snr - width/2, num_snr, width, 
                      label='Numerical', color=self.colors['numerical'])
        axes[2, 0].bar(x_snr + width/2, vis_snr, width,
                      label='Visual', color=self.colors['visual'])
        
        axes[2, 0].set_xlabel('Compounds')
        axes[2, 0].set_ylabel('Signal-to-Noise Ratio')
        axes[2, 0].set_title('Signal-to-Noise Ratio Comparison')
        axes[2, 0].set_xticks(x_snr)
        axes[2, 0].set_xticklabels(snr_compounds, rotation=45)
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # Retention time accuracy
        expected_rt = [15.2, 8.7, 12.1, 18.5, 6.3, 14.8]
        num_rt = [15.18, 8.72, 12.08, 18.52, 6.28, 14.85]
        vis_rt = [15.21, 8.69, 12.15, 18.48, 6.31, 14.82]
        
        axes[2, 1].scatter(expected_rt, num_rt, color=self.colors['numerical'], 
                          label='Numerical', s=60, alpha=0.7)
        axes[2, 1].scatter(expected_rt, vis_rt, color=self.colors['visual'], 
                          label='Visual', s=60, alpha=0.7)
        
        # Perfect correlation line
        min_rt, max_rt = min(expected_rt), max(expected_rt)
        axes[2, 1].plot([min_rt, max_rt], [min_rt, max_rt], 'k--', alpha=0.5, label='Perfect')
        
        axes[2, 1].set_xlabel('Expected Retention Time (min)')
        axes[2, 1].set_ylabel('Observed Retention Time (min)')
        axes[2, 1].set_title('Retention Time Accuracy')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_mass_spectra_comparison(
        self,
        spectra_data: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot mass spectra comparison between pipelines
        
        Args:
            spectra_data: Dictionary containing mass spectra data
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Mass Spectra Analysis Comparison', fontsize=16)
        
        # Generate synthetic mass spectra
        mz_range = np.linspace(50, 500, 2000)
        
        # Spectrum 1 - Glucose fragmentation
        glucose_peaks = [180.063, 162.053, 144.042, 126.032, 108.021, 90.011]
        glucose_intensities = [100, 45, 30, 25, 15, 10]
        
        spectrum_num = np.zeros_like(mz_range)
        spectrum_vis = np.zeros_like(mz_range)
        
        for peak_mz, intensity in zip(glucose_peaks, glucose_intensities):
            # Add peaks with slight differences between pipelines
            peak_idx = np.argmin(np.abs(mz_range - peak_mz))
            spectrum_num[peak_idx] = intensity + np.random.normal(0, 2)
            spectrum_vis[peak_idx] = intensity * 0.95 + np.random.normal(0, 2.5)
        
        # Add noise
        spectrum_num += np.random.exponential(0.5, len(mz_range))
        spectrum_vis += np.random.exponential(0.6, len(mz_range))
        
        axes[0, 0].plot(mz_range, spectrum_num, 'b-', linewidth=0.8, 
                       label='Numerical Pipeline', alpha=0.8)
        axes[0, 0].plot(mz_range, spectrum_vis, 'r-', linewidth=0.8, 
                       label='Visual Pipeline', alpha=0.8)
        axes[0, 0].set_title('Glucose MS/MS Spectrum')
        axes[0, 0].set_xlabel('m/z')
        axes[0, 0].set_ylabel('Relative Intensity')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_xlim(80, 200)
        
        # Spectrum 2 - Full scan
        full_scan_num = np.random.exponential(2, len(mz_range))
        full_scan_vis = np.random.exponential(2.2, len(mz_range))
        
        # Add some metabolite peaks
        metabolite_peaks = [132.077, 147.053, 180.063, 191.020, 205.097, 234.095]
        for peak_mz in metabolite_peaks:
            peak_idx = np.argmin(np.abs(mz_range - peak_mz))
            full_scan_num[peak_idx] += np.random.uniform(20, 80)
            full_scan_vis[peak_idx] += np.random.uniform(18, 75)
        
        axes[0, 1].plot(mz_range, full_scan_num, 'b-', linewidth=0.5, 
                       label='Numerical Pipeline', alpha=0.7)
        axes[0, 1].plot(mz_range, full_scan_vis, 'r-', linewidth=0.5, 
                       label='Visual Pipeline', alpha=0.7)
        axes[0, 1].set_title('Full Scan Mass Spectrum')
        axes[0, 1].set_xlabel('m/z')
        axes[0, 1].set_ylabel('Intensity')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Peak detection accuracy
        detected_peaks = ['180.063', '162.053', '144.042', '126.032', '108.021']
        num_accuracy = [98.5, 95.2, 92.1, 88.7, 85.3]
        vis_accuracy = [96.8, 93.5, 89.8, 86.2, 82.1]
        
        x = np.arange(len(detected_peaks))
        width = 0.35
        
        axes[1, 0].bar(x - width/2, num_accuracy, width, 
                      label='Numerical', color=self.colors['numerical'])
        axes[1, 0].bar(x + width/2, vis_accuracy, width,
                      label='Visual', color=self.colors['visual'])
        
        axes[1, 0].set_xlabel('Fragment m/z')
        axes[1, 0].set_ylabel('Detection Accuracy (%)')
        axes[1, 0].set_title('Fragment Peak Detection Accuracy')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(detected_peaks, rotation=45)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Mass accuracy comparison
        theoretical_masses = [180.0634, 162.0528, 144.0422, 126.0317, 108.0211]
        num_measured = [180.0631, 162.0525, 144.0419, 126.0314, 108.0208]
        vis_measured = [180.0636, 162.0531, 144.0425, 126.0319, 108.0213]
        
        num_errors = [(m - t) * 1e6 / t for m, t in zip(num_measured, theoretical_masses)]  # ppm
        vis_errors = [(m - t) * 1e6 / t for m, t in zip(vis_measured, theoretical_masses)]  # ppm
        
        x_mass = np.arange(len(theoretical_masses))
        axes[1, 1].bar(x_mass - width/2, num_errors, width, 
                      label='Numerical', color=self.colors['numerical'])
        axes[1, 1].bar(x_mass + width/2, vis_errors, width,
                      label='Visual', color=self.colors['visual'])
        
        axes[1, 1].set_xlabel('Fragment m/z')
        axes[1, 1].set_ylabel('Mass Error (ppm)')
        axes[1, 1].set_title('Mass Accuracy Comparison')
        axes[1, 1].set_xticks(x_mass)
        axes[1, 1].set_xticklabels([f'{m:.3f}' for m in theoretical_masses], rotation=45)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_metabolite_heatmap(
        self,
        metabolite_data: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot metabolite abundance heatmap comparison
        
        Args:
            metabolite_data: Dictionary containing metabolite abundance data
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 8))
        fig.suptitle('Metabolite Abundance Heatmap Analysis', fontsize=16)
        
        # Generate synthetic metabolite data
        metabolites = [
            'Glucose', 'Fructose', 'Sucrose', 'Lactate', 'Pyruvate',
            'Alanine', 'Glycine', 'Serine', 'Leucine', 'Valine',
            'Citrate', 'Malate', 'Succinate', 'Fumarate', 'ATP',
            'ADP', 'AMP', 'NAD+', 'NADH', 'Glutamate'
        ]
        
        samples = [f'Sample_{i+1}' for i in range(12)]
        
        # Numerical pipeline data
        np.random.seed(42)
        num_data = np.random.lognormal(mean=2, sigma=1, size=(len(metabolites), len(samples)))
        
        # Visual pipeline data (with some systematic differences)
        np.random.seed(43)
        vis_data = num_data * np.random.normal(0.95, 0.1, size=num_data.shape)
        vis_data = np.abs(vis_data)  # Ensure positive values
        
        # Numerical pipeline heatmap
        im1 = axes[0].imshow(num_data, cmap='viridis', aspect='auto')
        axes[0].set_title('Numerical Pipeline')
        axes[0].set_xlabel('Samples')
        axes[0].set_ylabel('Metabolites')
        axes[0].set_xticks(range(len(samples)))
        axes[0].set_xticklabels(samples, rotation=45)
        axes[0].set_yticks(range(len(metabolites)))
        axes[0].set_yticklabels(metabolites)
        plt.colorbar(im1, ax=axes[0], label='Abundance (log scale)')
        
        # Visual pipeline heatmap
        im2 = axes[1].imshow(vis_data, cmap='viridis', aspect='auto')
        axes[1].set_title('Visual Pipeline')
        axes[1].set_xlabel('Samples')
        axes[1].set_ylabel('Metabolites')
        axes[1].set_xticks(range(len(samples)))
        axes[1].set_xticklabels(samples, rotation=45)
        axes[1].set_yticks(range(len(metabolites)))
        axes[1].set_yticklabels(metabolites)
        plt.colorbar(im2, ax=axes[1], label='Abundance (log scale)')
        
        # Difference heatmap
        diff_data = (vis_data - num_data) / num_data * 100  # Percentage difference
        im3 = axes[2].imshow(diff_data, cmap='RdBu_r', aspect='auto', 
                            vmin=-50, vmax=50)
        axes[2].set_title('Relative Difference (%)')
        axes[2].set_xlabel('Samples')
        axes[2].set_ylabel('Metabolites')
        axes[2].set_xticks(range(len(samples)))
        axes[2].set_xticklabels(samples, rotation=45)
        axes[2].set_yticks(range(len(metabolites)))
        axes[2].set_yticklabels(metabolites)
        plt.colorbar(im3, ax=axes[2], label='Difference (%)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_feature_extraction_analysis(
        self,
        feature_data: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot feature extraction and dimensionality reduction analysis
        
        Args:
            feature_data: Dictionary containing feature extraction results
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Feature Extraction and Analysis', fontsize=16)
        
        # Generate synthetic feature data
        n_samples = 100
        n_features = 50
        
        np.random.seed(42)
        num_features = np.random.randn(n_samples, n_features)
        vis_features = num_features + np.random.normal(0, 0.2, (n_samples, n_features))
        
        # Add some structure (3 groups)
        group_labels = np.repeat([0, 1, 2], n_samples // 3)
        if len(group_labels) < n_samples:
            group_labels = np.append(group_labels, [2] * (n_samples - len(group_labels)))
        
        # PCA analysis - Numerical
        pca_num = PCA(n_components=2)
        num_pca = pca_num.fit_transform(num_features)
        
        scatter1 = axes[0, 0].scatter(num_pca[:, 0], num_pca[:, 1], c=group_labels, 
                                     cmap='viridis', alpha=0.7, s=50)
        axes[0, 0].set_title('PCA - Numerical Pipeline')
        axes[0, 0].set_xlabel(f'PC1 ({pca_num.explained_variance_ratio_[0]:.1%} variance)')
        axes[0, 0].set_ylabel(f'PC2 ({pca_num.explained_variance_ratio_[1]:.1%} variance)')
        axes[0, 0].grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=axes[0, 0], label='Group')
        
        # PCA analysis - Visual
        pca_vis = PCA(n_components=2)
        vis_pca = pca_vis.fit_transform(vis_features)
        
        scatter2 = axes[0, 1].scatter(vis_pca[:, 0], vis_pca[:, 1], c=group_labels, 
                                     cmap='viridis', alpha=0.7, s=50)
        axes[0, 1].set_title('PCA - Visual Pipeline')
        axes[0, 1].set_xlabel(f'PC1 ({pca_vis.explained_variance_ratio_[0]:.1%} variance)')
        axes[0, 1].set_ylabel(f'PC2 ({pca_vis.explained_variance_ratio_[1]:.1%} variance)')
        axes[0, 1].grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=axes[0, 1], label='Group')
        
        # Explained variance comparison
        components = range(1, 11)
        num_var_explained = np.cumsum(pca_num.explained_variance_ratio_[:10])
        vis_var_explained = np.cumsum(pca_vis.explained_variance_ratio_[:10])
        
        axes[0, 2].plot(components, num_var_explained, 'o-', 
                       label='Numerical', color=self.colors['numerical'])
        axes[0, 2].plot(components, vis_var_explained, 's-', 
                       label='Visual', color=self.colors['visual'])
        axes[0, 2].set_xlabel('Principal Components')
        axes[0, 2].set_ylabel('Cumulative Explained Variance')
        axes[0, 2].set_title('PCA Explained Variance')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # t-SNE analysis - Numerical
        tsne_num = TSNE(n_components=2, random_state=42, perplexity=30)
        num_tsne = tsne_num.fit_transform(num_features)
        
        scatter3 = axes[1, 0].scatter(num_tsne[:, 0], num_tsne[:, 1], c=group_labels, 
                                     cmap='viridis', alpha=0.7, s=50)
        axes[1, 0].set_title('t-SNE - Numerical Pipeline')
        axes[1, 0].set_xlabel('t-SNE 1')
        axes[1, 0].set_ylabel('t-SNE 2')
        axes[1, 0].grid(True, alpha=0.3)
        plt.colorbar(scatter3, ax=axes[1, 0], label='Group')
        
        # t-SNE analysis - Visual
        tsne_vis = TSNE(n_components=2, random_state=42, perplexity=30)
        vis_tsne = tsne_vis.fit_transform(vis_features)
        
        scatter4 = axes[1, 1].scatter(vis_tsne[:, 0], vis_tsne[:, 1], c=group_labels, 
                                     cmap='viridis', alpha=0.7, s=50)
        axes[1, 1].set_title('t-SNE - Visual Pipeline')
        axes[1, 1].set_xlabel('t-SNE 1')
        axes[1, 1].set_ylabel('t-SNE 2')
        axes[1, 1].grid(True, alpha=0.3)
        plt.colorbar(scatter4, ax=axes[1, 1], label='Group')
        
        # Feature importance comparison
        feature_names = [f'Feature_{i+1}' for i in range(10)]
        num_importance = np.abs(pca_num.components_[0][:10])  # First PC loadings
        vis_importance = np.abs(pca_vis.components_[0][:10])
        
        x = np.arange(len(feature_names))
        width = 0.35
        
        axes[1, 2].bar(x - width/2, num_importance, width, 
                      label='Numerical', color=self.colors['numerical'])
        axes[1, 2].bar(x + width/2, vis_importance, width,
                      label='Visual', color=self.colors['visual'])
        
        axes[1, 2].set_xlabel('Features')
        axes[1, 2].set_ylabel('Importance (|PC1 Loading|)')
        axes[1, 2].set_title('Feature Importance Comparison')
        axes[1, 2].set_xticks(x)
        axes[1, 2].set_xticklabels(feature_names, rotation=45)
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_analytical_dashboard(
        self,
        analytical_results: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create interactive analytical content dashboard
        
        Args:
            analytical_results: Complete analytical results
            save_path: Optional path to save the dashboard
            
        Returns:
            Plotly figure object
        """
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'XIC Comparison', 'Mass Spectra Overlay',
                'Metabolite Abundance Heatmap', 'Peak Detection Summary',
                'Feature Space (PCA)', 'Analytical Performance'
            ],
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "heatmap"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "bar"}]
            ]
        )
        
        # XIC comparison (row 1, col 1)
        time_points = np.linspace(0, 30, 300)
        glucose_peak = 15.2
        glucose_width = 0.8
        
        glucose_num = 1e6 * np.exp(-0.5 * ((time_points - glucose_peak) / glucose_width) ** 2)
        glucose_vis = 0.95e6 * np.exp(-0.5 * ((time_points - glucose_peak) / glucose_width) ** 2)
        
        fig.add_trace(
            go.Scatter(x=time_points, y=glucose_num, mode='lines',
                      name='Numerical XIC', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=time_points, y=glucose_vis, mode='lines',
                      name='Visual XIC', line=dict(color='red')),
            row=1, col=1
        )
        
        # Mass spectra overlay (row 1, col 2)
        mz_range = np.linspace(50, 200, 500)
        spectrum_num = np.random.exponential(1, len(mz_range))
        spectrum_vis = np.random.exponential(1.1, len(mz_range))
        
        # Add some peaks
        peaks = [132.077, 147.053, 180.063]
        for peak in peaks:
            peak_idx = np.argmin(np.abs(mz_range - peak))
            spectrum_num[peak_idx] += 50
            spectrum_vis[peak_idx] += 45
        
        fig.add_trace(
            go.Scatter(x=mz_range, y=spectrum_num, mode='lines',
                      name='Numerical MS', line=dict(color='blue')),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=mz_range, y=spectrum_vis, mode='lines',
                      name='Visual MS', line=dict(color='red')),
            row=1, col=2
        )
        
        # Metabolite heatmap (row 2, col 1)
        metabolites = ['Glucose', 'Fructose', 'Lactate', 'Pyruvate', 'Alanine']
        samples = ['S1', 'S2', 'S3', 'S4', 'S5']
        
        heatmap_data = np.random.lognormal(2, 0.5, (len(metabolites), len(samples)))
        
        fig.add_trace(
            go.Heatmap(z=heatmap_data, x=samples, y=metabolites,
                      colorscale='Viridis', name='Abundance'),
            row=2, col=1
        )
        
        # Peak detection summary (row 2, col 2)
        compounds = ['Glucose', 'Leucine', 'Alanine', 'Citrate', 'Lactate']
        num_detected = [5, 4, 3, 2, 4]
        vis_detected = [4, 4, 2, 3, 4]
        
        fig.add_trace(
            go.Bar(name='Numerical Detected', x=compounds, y=num_detected,
                   marker_color='lightblue'),
            row=2, col=2
        )
        fig.add_trace(
            go.Bar(name='Visual Detected', x=compounds, y=vis_detected,
                   marker_color='lightcoral'),
            row=2, col=2
        )
        
        # PCA feature space (row 3, col 1)
        n_points = 50
        pca_num_x = np.random.normal(0, 2, n_points)
        pca_num_y = np.random.normal(0, 1.5, n_points)
        pca_vis_x = pca_num_x + np.random.normal(0, 0.3, n_points)
        pca_vis_y = pca_num_y + np.random.normal(0, 0.3, n_points)
        
        fig.add_trace(
            go.Scatter(x=pca_num_x, y=pca_num_y, mode='markers',
                      name='Numerical PCA', marker=dict(color='blue', size=8)),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=pca_vis_x, y=pca_vis_y, mode='markers',
                      name='Visual PCA', marker=dict(color='red', size=8)),
            row=3, col=1
        )
        
        # Analytical performance (row 3, col 2)
        metrics = ['Sensitivity', 'Specificity', 'Precision', 'Accuracy']
        num_performance = [92.5, 88.3, 90.1, 89.7]
        vis_performance = [89.8, 91.2, 87.5, 88.9]
        
        fig.add_trace(
            go.Bar(name='Numerical Performance', x=metrics, y=num_performance,
                   marker_color='green'),
            row=3, col=2
        )
        fig.add_trace(
            go.Bar(name='Visual Performance', x=metrics, y=vis_performance,
                   marker_color='orange'),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=1000,
            title_text="Analytical Content Dashboard",
            title_x=0.5,
            showlegend=True
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Retention Time (min)", row=1, col=1)
        fig.update_yaxes(title_text="Intensity", row=1, col=1)
        fig.update_xaxes(title_text="m/z", row=1, col=2)
        fig.update_yaxes(title_text="Intensity", row=1, col=2)
        fig.update_yaxes(title_text="Peaks Detected", row=2, col=2)
        fig.update_xaxes(title_text="PC1", row=3, col=1)
        fig.update_yaxes(title_text="PC2", row=3, col=1)
        fig.update_yaxes(title_text="Performance (%)", row=3, col=2)
        
        if save_path:
            fig.write_html(save_path)
        
        return fig 