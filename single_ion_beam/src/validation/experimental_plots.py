"""
Experimental visualization framework for categorical thermodynamics and S-entropy coordinates.

This module generates publication-quality plots for validation results including:
- 3D S-space scatter plots with convex hulls
- Ionization mode comparison
- Sample classification confusion matrices
- Network analysis
- MS2 coverage heatmaps
- Categorical temperature surfaces
- Maxwell-Boltzmann distributions
- Entropy production curves
- Ideal gas law validation
- Performance profiling
- PCA analysis
- Metabolite overlap Venn diagrams
- Correlation heatmaps
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
from scipy import stats
import seaborn as sns
from matplotlib.patches import Ellipse
from matplotlib_venn import venn3
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class ExperimentalPlotter:
    """Generate all experimental validation plots."""
    
    def __init__(self, output_dir: str = './figures'):
        """
        Initialize plotter.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save figures
        """
        self.output_dir = output_dir
        self.colors = {
            'M3': '#1f77b4',  # Blue
            'M4': '#ff7f0e',  # Orange
            'M5': '#2ca02c',  # Green
            'positive': '#d62728',  # Red
            'negative': '#9467bd'   # Purple
        }
        
    def plot_3d_s_space_with_hulls(self,
                                   s_coordinates: np.ndarray,
                                   sample_labels: np.ndarray,
                                   save_path: Optional[str] = None):
        """
        Plot 3D S-space scatter with convex hulls for each sample.
        
        Parameters:
        -----------
        s_coordinates : np.ndarray
            (N, 3) S-entropy coordinates
        sample_labels : np.ndarray
            Sample identifiers
        save_path : str, optional
            Path to save figure
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        unique_samples = np.unique(sample_labels)
        
        for sample in unique_samples:
            mask = sample_labels == sample
            s_sample = s_coordinates[mask]
            
            # Scatter plot
            ax.scatter(s_sample[:, 0], s_sample[:, 1], s_sample[:, 2],
                      c=self.colors.get(sample, '#888888'),
                      label=sample, alpha=0.3, s=1)
            
            # Convex hull
            if len(s_sample) >= 4:  # Need at least 4 points for 3D hull
                try:
                    hull = ConvexHull(s_sample)
                    # Plot hull faces
                    for simplex in hull.simplices:
                        triangle = s_sample[simplex]
                        poly = Poly3DCollection([triangle], alpha=0.1,
                                              facecolor=self.colors.get(sample, '#888888'),
                                              edgecolor='none')
                        ax.add_collection3d(poly)
                except:
                    pass  # Skip if hull fails
        
        ax.set_xlabel('$S_k$ (Knowledge)', fontsize=12)
        ax.set_ylabel('$S_t$ (Transformation)', fontsize=12)
        ax.set_zlabel('$S_e$ (Entropy)', fontsize=12)
        ax.set_title('3D S-Space Visualization with Convex Hulls\n(46,458 spectra)', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()
    
    def plot_ionization_mode_comparison(self,
                                       s_coordinates: np.ndarray,
                                       mode_labels: np.ndarray,
                                       save_path: Optional[str] = None):
        """
        Plot 2D ionization mode comparison (S_k vs S_e).
        
        Parameters:
        -----------
        s_coordinates : np.ndarray
            (N, 3) S-entropy coordinates
        mode_labels : np.ndarray
            Ionization mode labels
        save_path : str, optional
            Path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        unique_modes = np.unique(mode_labels)
        
        for mode in unique_modes:
            mask = mode_labels == mode
            s_mode = s_coordinates[mask]
            
            ax.scatter(s_mode[:, 0], s_mode[:, 2],
                      c=self.colors.get(mode, '#888888'),
                      label=f'{mode.capitalize()} ESI',
                      alpha=0.3, s=5)
        
        ax.set_xlabel('$S_k$ (Knowledge)', fontsize=12)
        ax.set_ylabel('$S_e$ (Entropy)', fontsize=12)
        ax.set_title('Ionization Mode Comparison in S-Space', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()
    
    def plot_confusion_matrix(self,
                             confusion_matrix: np.ndarray,
                             class_labels: List[str],
                             classifier_type: str = 'RF',
                             save_path: Optional[str] = None):
        """
        Plot confusion matrix for sample classification.
        
        Parameters:
        -----------
        confusion_matrix : np.ndarray
            Confusion matrix
        class_labels : List[str]
            Class labels
        classifier_type : str
            Classifier type (SVM, RF, NN)
        save_path : str, optional
            Path to save figure
        """
        fig, ax = plt.subplots(figsize=(8, 7))
        
        # Normalize to percentages
        cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis] * 100
        
        sns.heatmap(cm_normalized, annot=True, fmt='.1f', cmap='Blues',
                   xticklabels=class_labels, yticklabels=class_labels,
                   cbar_kws={'label': 'Percentage (%)'}, ax=ax)
        
        ax.set_xlabel('Predicted Sample', fontsize=12)
        ax.set_ylabel('True Sample', fontsize=12)
        ax.set_title(f'Sample Classification Confusion Matrix ({classifier_type})', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()
    
    def plot_ms2_coverage_heatmap(self,
                                 ms2_coverage: Dict[str, Dict[str, int]],
                                 save_path: Optional[str] = None):
        """
        Plot MS2 coverage heatmap (samples x ionization modes).
        
        Parameters:
        -----------
        ms2_coverage : Dict
            MS2 coverage data {sample: {mode: count}}
        save_path : str, optional
            Path to save figure
        """
        # Convert to matrix
        samples = sorted(ms2_coverage.keys())
        modes = ['negative', 'positive']
        
        data = np.array([[ms2_coverage[s][m] for m in modes] for s in samples])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(data, annot=True, fmt='d', cmap='YlOrRd',
                   xticklabels=[m.capitalize() for m in modes],
                   yticklabels=samples,
                   cbar_kws={'label': 'MS2 Spectra Count'}, ax=ax)
        
        ax.set_xlabel('Ionization Mode', fontsize=12)
        ax.set_ylabel('Sample', fontsize=12)
        ax.set_title('MS2 Coverage Across Samples and Ionization Modes', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()
    
    def plot_maxwell_boltzmann_distribution(self,
                                           intensities: np.ndarray,
                                           scale_parameter: float,
                                           save_path: Optional[str] = None):
        """
        Plot Maxwell-Boltzmann intensity distribution.
        
        Parameters:
        -----------
        intensities : np.ndarray
            Intensity values
        scale_parameter : float
            Fitted scale parameter
        save_path : str, optional
            Path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Histogram of observed data
        counts, bins, _ = ax.hist(intensities, bins=100, density=True,
                                  alpha=0.6, color='skyblue', label='Observed')
        
        # Theoretical Maxwell-Boltzmann
        x = np.linspace(0, np.max(intensities), 1000)
        # Chi distribution with 3 degrees of freedom (Maxwell-Boltzmann)
        theoretical = stats.chi.pdf(x / scale_parameter, df=3) / scale_parameter
        ax.plot(x, theoretical, 'r-', linewidth=2, label='Maxwell-Boltzmann (theoretical)')
        
        ax.set_xlabel('Intensity', fontsize=12)
        ax.set_ylabel('Probability Density', fontsize=12)
        ax.set_title(f'Maxwell-Boltzmann Distribution Validation\n({len(intensities):,} peaks, scale={scale_parameter:.3f})',
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, np.percentile(intensities, 99))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()
    
    def plot_ideal_gas_law_validation(self,
                                     pv_values: np.ndarray,
                                     t_cat_values: np.ndarray,
                                     slope: float,
                                     r_squared: float,
                                     save_path: Optional[str] = None):
        """
        Plot ideal gas law validation (PV vs T_cat).
        
        Parameters:
        -----------
        pv_values : np.ndarray
            PV values
        t_cat_values : np.ndarray
            Categorical temperature values
        slope : float
            Fitted slope
        r_squared : float
            R-squared value
        save_path : str, optional
            Path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Scatter plot
        ax.scatter(t_cat_values, pv_values, alpha=0.3, s=5, color='steelblue')
        
        # Fit line
        fit_line = slope * t_cat_values + np.mean(pv_values) - slope * np.mean(t_cat_values)
        ax.plot(t_cat_values, fit_line, 'r-', linewidth=2, label=f'Fit: slope={slope:.3f}')
        
        # Ideal line (slope=1)
        ideal_line = t_cat_values
        ax.plot(t_cat_values, ideal_line, 'k--', linewidth=2, alpha=0.5, label='Ideal: slope=1.0')
        
        ax.set_xlabel('$T_{cat}$ (Categorical Temperature)', fontsize=12)
        ax.set_ylabel('$PV$ (Pressure × Volume)', fontsize=12)
        ax.set_title(f'Ideal Gas Law Validation: $PV = k_B T_{{cat}}$\n$R^2$={r_squared:.4f}',
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()
    
    def plot_pca_with_ellipses(self,
                              pca_coordinates: np.ndarray,
                              sample_labels: np.ndarray,
                              variance_explained: np.ndarray,
                              save_path: Optional[str] = None):
        """
        Plot PCA of S-entropy coordinates with confidence ellipses.
        
        Parameters:
        -----------
        pca_coordinates : np.ndarray
            (N, 2) PCA coordinates
        sample_labels : np.ndarray
            Sample identifiers
        variance_explained : np.ndarray
            Variance explained by each PC
        save_path : str, optional
            Path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        unique_samples = np.unique(sample_labels)
        
        for sample in unique_samples:
            mask = sample_labels == sample
            pca_sample = pca_coordinates[mask]
            
            # Scatter plot
            ax.scatter(pca_sample[:, 0], pca_sample[:, 1],
                      c=self.colors.get(sample, '#888888'),
                      label=sample, alpha=0.4, s=10)
            
            # Confidence ellipse (95%)
            if len(pca_sample) > 2:
                mean = np.mean(pca_sample, axis=0)
                cov = np.cov(pca_sample.T)
                
                # Eigenvalues and eigenvectors
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                order = eigenvalues.argsort()[::-1]
                eigenvalues = eigenvalues[order]
                eigenvectors = eigenvectors[:, order]
                
                # 95% confidence (chi-square with 2 df)
                chi2_val = 5.991  # 95% for 2 df
                width, height = 2 * np.sqrt(chi2_val * eigenvalues)
                angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
                
                ellipse = Ellipse(mean, width, height, angle=angle,
                                facecolor='none',
                                edgecolor=self.colors.get(sample, '#888888'),
                                linewidth=2, linestyle='--')
                ax.add_patch(ellipse)
        
        ax.set_xlabel(f'PC1 ({variance_explained[0]*100:.1f}% variance)', fontsize=12)
        ax.set_ylabel(f'PC2 ({variance_explained[1]*100:.1f}% variance)', fontsize=12)
        ax.set_title('PCA of S-Entropy Coordinates with 95% Confidence Ellipses', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()
    
    def plot_metabolite_overlap_venn(self,
                                    metabolite_counts: Dict[str, int],
                                    save_path: Optional[str] = None):
        """
        Plot metabolite overlap Venn diagram (M3/M4/M5).
        
        Parameters:
        -----------
        metabolite_counts : Dict
            Metabolite counts for each sample and overlaps
        save_path : str, optional
            Path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Venn diagram
        venn = venn3(subsets=(
            metabolite_counts.get('M3_only', 100),
            metabolite_counts.get('M4_only', 100),
            metabolite_counts.get('M3_M4', 50),
            metabolite_counts.get('M5_only', 100),
            metabolite_counts.get('M3_M5', 50),
            metabolite_counts.get('M4_M5', 50),
            metabolite_counts.get('M3_M4_M5', 30)
        ), set_labels=('M3', 'M4', 'M5'), ax=ax)
        
        # Color the patches
        if venn.get_patch_by_id('100'):
            venn.get_patch_by_id('100').set_color(self.colors['M3'])
            venn.get_patch_by_id('100').set_alpha(0.5)
        if venn.get_patch_by_id('010'):
            venn.get_patch_by_id('010').set_color(self.colors['M4'])
            venn.get_patch_by_id('010').set_alpha(0.5)
        if venn.get_patch_by_id('001'):
            venn.get_patch_by_id('001').set_color(self.colors['M5'])
            venn.get_patch_by_id('001').set_alpha(0.5)
        
        ax.set_title('Metabolite Overlap Across Samples', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()
    
    def plot_network_properties(self,
                               network_props: Dict[str, float],
                               save_path: Optional[str] = None):
        """
        Plot network properties bar chart.
        
        Parameters:
        -----------
        network_props : Dict
            Network properties {property: value}
        save_path : str, optional
            Path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        properties = list(network_props.keys())
        values = list(network_props.values())
        
        bars = ax.bar(properties, values, color='steelblue', alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}' if val < 100 else f'{int(val)}',
                   ha='center', va='bottom', fontsize=10)
        
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title('Network Properties', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()
    
    def plot_entropy_production(self,
                               retention_times: np.ndarray,
                               entropy_rates: np.ndarray,
                               sample_labels: np.ndarray,
                               save_path: Optional[str] = None):
        """
        Plot entropy production over retention time (dS/dt curves).
        
        Parameters:
        -----------
        retention_times : np.ndarray
            Retention times
        entropy_rates : np.ndarray
            Entropy production rates
        sample_labels : np.ndarray
            Sample identifiers
        save_path : str, optional
            Path to save figure
        """
        fig, ax = plt.subplots(figsize=(12, 7))
        
        unique_samples = np.unique(sample_labels)
        
        for sample in unique_samples:
            mask = sample_labels == sample
            rt_sample = retention_times[mask]
            ds_sample = entropy_rates[mask]
            
            # Sort by retention time
            sort_idx = np.argsort(rt_sample)
            rt_sorted = rt_sample[sort_idx]
            ds_sorted = ds_sample[sort_idx]
            
            # Bin and average
            bins = np.linspace(0, 60, 50)
            bin_indices = np.digitize(rt_sorted, bins)
            bin_means = [ds_sorted[bin_indices == i].mean() if np.any(bin_indices == i) else 0
                        for i in range(1, len(bins))]
            bin_centers = (bins[:-1] + bins[1:]) / 2
            
            ax.plot(bin_centers, bin_means, linewidth=2,
                   color=self.colors.get(sample, '#888888'),
                   label=sample, marker='o', markersize=4)
        
        ax.set_xlabel('Retention Time (min)', fontsize=12)
        ax.set_ylabel('$dS/dt$ (Entropy Production Rate)', fontsize=12)
        ax.set_title('Entropy Production Over Retention Time', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()
    
    def plot_performance_profiling(self,
                                  performance_data: Dict[str, Dict[str, float]],
                                  save_path: Optional[str] = None):
        """
        Plot performance profiling (processing time, memory, accuracy vs time).
        
        Parameters:
        -----------
        performance_data : Dict
            Performance metrics
        save_path : str, optional
            Path to save figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Processing time breakdown (stacked bar)
        if 'processing_time' in performance_data:
            times = performance_data['processing_time']
            stages = list(times.keys())
            values = list(times.values())
            
            axes[0].bar(stages, values, color='steelblue', alpha=0.7, edgecolor='black')
            axes[0].set_ylabel('Time (s)', fontsize=11)
            axes[0].set_title('Processing Time Breakdown', fontsize=12, fontweight='bold')
            axes[0].grid(True, alpha=0.3, axis='y')
            axes[0].tick_params(axis='x', rotation=45)
        
        # Memory usage
        if 'memory_usage' in performance_data:
            memory = performance_data['memory_usage']
            time_points = list(range(len(memory)))
            
            axes[1].plot(time_points, memory, linewidth=2, color='orangered', marker='o')
            axes[1].set_xlabel('Processing Stage', fontsize=11)
            axes[1].set_ylabel('Memory (MB)', fontsize=11)
            axes[1].set_title('Memory Usage Profile', fontsize=12, fontweight='bold')
            axes[1].grid(True, alpha=0.3)
        
        # Accuracy vs time Pareto front
        if 'pareto_front' in performance_data:
            pareto = performance_data['pareto_front']
            times = [p['time'] for p in pareto]
            accuracies = [p['accuracy'] for p in pareto]
            
            axes[2].scatter(times, accuracies, s=100, c='green', alpha=0.6, edgecolor='black')
            axes[2].plot(times, accuracies, 'k--', alpha=0.3)
            axes[2].set_xlabel('Processing Time (s)', fontsize=11)
            axes[2].set_ylabel('Accuracy (%)', fontsize=11)
            axes[2].set_title('Accuracy vs Time Pareto Front', fontsize=12, fontweight='bold')
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()
    
    def plot_correlation_heatmap(self,
                                correlation_matrix: np.ndarray,
                                file_labels: List[str],
                                save_path: Optional[str] = None):
        """
        Plot pairwise correlation heatmap.
        
        Parameters:
        -----------
        correlation_matrix : np.ndarray
            Correlation matrix
        file_labels : List[str]
            File labels
        save_path : str, optional
            Path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 9))
        
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   xticklabels=file_labels, yticklabels=file_labels,
                   vmin=-1, vmax=1, center=0,
                   cbar_kws={'label': 'Correlation Coefficient'}, ax=ax)
        
        ax.set_title('Pairwise Correlation Heatmap', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()


if __name__ == "__main__":
    print("Experimental plotting module loaded successfully.")
    print("Use ExperimentalPlotter class to generate validation plots.")
