"""
Panel chart generators for validation results.

Creates multi-panel figures for:
- Five modality validation
- Chromatographic separation validation
- Temporal resolution validation
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from typing import Dict, List, Optional
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9


class ValidationPanelChart:
    """Generator for validation panel charts."""
    
    def __init__(self, output_dir: str = "./validation_figures"):
        self.output_dir = output_dir
        import os
        os.makedirs(output_dir, exist_ok=True)
        
    def plot_five_modalities(self,
                            results: Dict,
                            save_path: Optional[str] = None):
        """
        Create 5-panel chart for all modalities.
        
        Parameters:
        -----------
        results : Dict
            Dictionary mapping modality names to ValidationResult objects
        save_path : str, optional
            Path to save figure
        """
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        modality_names = [
            'Optical Spectroscopy',
            'Refractive Index',
            'Vibrational Spectroscopy',
            'Metabolic GPS',
            'Temporal-Causal Dynamics'
        ]
        
        positions = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)]
        
        for idx, (name, pos) in enumerate(zip(modality_names, positions)):
            if name not in results:
                continue
                
            result = results[name]
            ax = fig.add_subplot(gs[pos[0], pos[1]])
            
            # Predicted vs Measured scatter plot
            ax.scatter(result.measured_values, result.predicted_values,
                      alpha=0.6, s=50, label='Data')
            
            # Perfect correlation line
            min_val = min(result.measured_values.min(), result.predicted_values.min())
            max_val = max(result.measured_values.max(), result.predicted_values.max())
            ax.plot([min_val, max_val], [min_val, max_val],
                   'r--', linewidth=2, label='Perfect correlation')
            
            ax.set_xlabel('Measured')
            ax.set_ylabel('Predicted')
            ax.set_title(f'{name}\nError: {result.error_percent:.2f}%')
            ax.legend(frameon=True, loc='upper left')
            ax.grid(True, alpha=0.3)
            
            # Add info text
            info_text = (f'ε = {result.exclusion_factor:.1e}\n'
                        f'I = {result.information_bits:.1f} bits\n'
                        f'R = {result.resolution:.2e}')
            ax.text(0.95, 0.05, info_text,
                   transform=ax.transAxes,
                   verticalalignment='bottom',
                   horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   fontsize=8)
        
        # Summary panel
        ax_summary = fig.add_subplot(gs[2, 1])
        ax_summary.axis('off')
        
        # Calculate combined metrics
        epsilon_product = np.prod([r.exclusion_factor for r in results.values()])
        total_bits = np.sum([r.information_bits for r in results.values()])
        N_0 = 1e60
        N_5 = N_0 * epsilon_product
        
        summary_text = (
            'MULTIMODAL UNIQUENESS\n\n'
            f'N₀ (initial ambiguity): {N_0:.1e}\n'
            f'Combined exclusion: {epsilon_product:.1e}\n'
            f'N₅ (final ambiguity): {N_5:.1e}\n'
            f'Total information: {total_bits:.1f} bits\n\n'
            f'Unique identification: {"✓ YES" if N_5 < 1 else "✗ NO"}'
        )
        
        ax_summary.text(0.1, 0.9, summary_text,
                       transform=ax_summary.transAxes,
                       verticalalignment='top',
                       fontsize=11,
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
                       family='monospace')
        
        plt.suptitle('Five Modality Validation: Complete Molecular Characterization',
                    fontsize=14, fontweight='bold', y=0.995)
        
        if save_path is None:
            save_path = f'{self.output_dir}/five_modalities_validation.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved five modalities validation to: {save_path}")
    
    def plot_chromatography(self,
                           result,
                           retention_data: Optional[Dict] = None,
                           save_path: Optional[str] = None):
        """
        Create 4-panel chromatography validation chart.
        
        Parameters:
        -----------
        result : ChromatographyResult
            Chromatography validation results
        retention_data : Dict, optional
            Retention time validation data
        save_path : str, optional
            Path to save figure
        """
        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Panel 1: Van Deemter curve
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(result.flow_rates, result.predicted_H, 'b-',
                linewidth=2, label='Predicted')
        ax1.scatter(result.flow_rates, result.measured_H,
                   c='red', s=50, alpha=0.6, label='Measured', zorder=10)
        
        # Mark optimal point
        ax1.axvline(result.optimal_flow_rate, color='green',
                   linestyle='--', linewidth=2, alpha=0.7,
                   label=f'Optimal: {result.optimal_flow_rate:.2f} cm/s')
        ax1.axhline(result.minimum_H, color='green',
                   linestyle='--', linewidth=1, alpha=0.5)
        
        ax1.set_xlabel('Flow Rate u (cm/s)')
        ax1.set_ylabel('HETP H (cm)')
        ax1.set_title(f'Van Deemter Equation: H = A + B/u + Cu\nError: {result.error_percent:.2f}%')
        ax1.legend(frameon=True)
        ax1.grid(True, alpha=0.3)
        
        # Add coefficient text
        coeff_text = (f'A = {result.A_coefficient:.3f} cm\n'
                     f'B = {result.B_coefficient:.3f} cm²/s\n'
                     f'C = {result.C_coefficient:.3f} s')
        ax1.text(0.95, 0.95, coeff_text,
                transform=ax1.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
                fontsize=9, family='monospace')
        
        # Panel 2: Resolution vs Peak Pair
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Generate resolution data
        n_peaks = 10
        peak_positions = np.linspace(0, 60, n_peaks)
        peak_widths = 0.5 * np.ones(n_peaks)
        
        resolutions = []
        for i in range(n_peaks - 1):
            delta_S = peak_positions[i+1] - peak_positions[i]
            R_s = delta_S / (4 * peak_widths[i])
            resolutions.append(R_s)
        
        peak_pairs = [f'{i+1}-{i+2}' for i in range(len(resolutions))]
        colors = ['green' if r > 1.5 else 'orange' for r in resolutions]
        
        ax2.bar(range(len(resolutions)), resolutions, color=colors, alpha=0.7)
        ax2.axhline(1.5, color='red', linestyle='--', linewidth=2,
                   label='Baseline separation (R_s = 1.5)')
        ax2.set_xlabel('Peak Pair')
        ax2.set_ylabel('Resolution R_s')
        ax2.set_title(f'Categorical Resolution\nMean R_s = {result.resolution:.1f}')
        ax2.set_xticks(range(len(resolutions)))
        ax2.set_xticklabels(peak_pairs, rotation=45, ha='right')
        ax2.legend(frameon=True)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Panel 3: Peak Capacity
        ax3 = fig.add_subplot(gs[1, 0])
        
        # Visualize peak capacity
        delta_S_max = 60
        sigma_S = 0.5
        n_c = int(1 + delta_S_max / (4 * sigma_S))
        
        # Draw peaks
        x = np.linspace(0, delta_S_max, 1000)
        peak_positions_viz = np.linspace(0, delta_S_max, n_c)
        
        y_total = np.zeros_like(x)
        for pos in peak_positions_viz:
            y = np.exp(-(x - pos)**2 / (2 * sigma_S**2))
            y_total += y
            ax3.fill_between(x, 0, y, alpha=0.3)
        
        ax3.plot(x, y_total, 'k-', linewidth=2, label='Total signal')
        ax3.set_xlabel('Categorical Position S')
        ax3.set_ylabel('Intensity (a.u.)')
        ax3.set_title(f'Peak Capacity: n_c = {n_c}')
        ax3.legend(frameon=True)
        ax3.grid(True, alpha=0.3)
        
        # Panel 4: Retention Time validation
        ax4 = fig.add_subplot(gs[1, 1])
        
        if retention_data is not None:
            ax4.scatter(retention_data['measured'], retention_data['predicted'],
                       alpha=0.6, s=70, c='purple')
            
            min_val = min(retention_data['measured'].min(), retention_data['predicted'].min())
            max_val = max(retention_data['measured'].max(), retention_data['predicted'].max())
            ax4.plot([min_val, max_val], [min_val, max_val],
                    'r--', linewidth=2, label='Perfect correlation')
            
            ax4.set_xlabel('Measured Retention Time (min)')
            ax4.set_ylabel('Predicted Retention Time (min)')
            ax4.set_title(f'Retention Time Prediction\nError: {retention_data["error_percent"]:.2f}%')
            ax4.legend(frameon=True)
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Retention time\ndata not provided',
                    transform=ax4.transAxes,
                    ha='center', va='center',
                    fontsize=12, style='italic')
            ax4.axis('off')
        
        plt.suptitle('Chromatographic Separation Validation',
                    fontsize=14, fontweight='bold', y=0.995)
        
        if save_path is None:
            save_path = f'{self.output_dir}/chromatography_validation.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved chromatography validation to: {save_path}")
    
    def plot_temporal_resolution(self,
                                result,
                                hardware_data: Optional[Dict] = None,
                                save_path: Optional[str] = None):
        """
        Create 4-panel temporal resolution validation chart.
        
        Parameters:
        -----------
        result : TemporalResult
            Temporal resolution validation results
        hardware_data : Dict, optional
            Hardware oscillator data
        save_path : str, optional
            Path to save figure
        """
        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Panel 1: Enhancement factors
        ax1 = fig.add_subplot(gs[0, 0])
        
        factors = result.enhancement_factors
        names = ['Oscillators\n(K)', 'Demons\n(M)', 'Cascade\n(2^R)', 'Total']
        values = [factors['oscillators_K'], factors['demons_M'],
                 factors['cascade_2R'], factors['total_enhancement']]
        
        # Log scale bar chart
        ax1.bar(names, np.log10(values), color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
               alpha=0.7, edgecolor='black', linewidth=1.5)
        ax1.set_ylabel('log₁₀(Enhancement Factor)')
        ax1.set_title('Enhancement Factor Breakdown')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (name, val) in enumerate(zip(names, values)):
            ax1.text(i, np.log10(val) + 0.5, f'{val:.1e}',
                    ha='center', va='bottom', fontsize=8, rotation=0)
        
        # Panel 2: Trans-Planckian comparison
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Time scale comparison (log scale)
        time_scales = {
            'Planck time': 5.39e-44,
            'Achieved Δt': result.temporal_precision,
            'Attosecond': 1e-18,
            'Femtosecond': 1e-15,
            'Picosecond': 1e-12,
            'Nanosecond': 1e-9
        }
        
        names_time = list(time_scales.keys())
        values_time = list(time_scales.values())
        colors_time = ['red', 'green', 'blue', 'blue', 'blue', 'blue']
        
        y_pos = np.arange(len(names_time))
        ax2.barh(y_pos, np.log10(values_time), color=colors_time, alpha=0.7,
                edgecolor='black', linewidth=1.5)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(names_time)
        ax2.set_xlabel('log₁₀(Time / seconds)')
        ax2.set_title(f'Trans-Planckian Precision\n{-np.log10(result.planck_time_ratio):.1f} orders below Planck')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, val in enumerate(values_time):
            ax2.text(np.log10(val) - 2, i, f'{val:.1e} s',
                    ha='right', va='center', fontsize=8)
        
        # Panel 3: Hardware oscillator frequencies
        ax3 = fig.add_subplot(gs[1, 0])
        
        if hardware_data is not None:
            all_freqs = []
            all_names = []
            all_colors = []
            color_map = {'CPU': '#1f77b4', 'GPU': '#ff7f0e', 'RAM': '#2ca02c',
                        'LED': '#d62728', 'Network': '#9467bd', 'USB': '#8c564b'}
            
            for hw_type, freqs in hardware_data.items():
                for i, f in enumerate(freqs):
                    all_freqs.append(f)
                    all_names.append(f'{hw_type}_{i+1}')
                    all_colors.append(color_map.get(hw_type, 'gray'))
            
            y_pos = np.arange(len(all_freqs))
            ax3.barh(y_pos, np.log10(all_freqs), color=all_colors, alpha=0.7,
                    edgecolor='black', linewidth=0.5)
            ax3.set_yticks(y_pos)
            ax3.set_yticklabels(all_names, fontsize=7)
            ax3.set_xlabel('log₁₀(Frequency / Hz)')
            ax3.set_title(f'Hardware Oscillator Network\n{len(all_freqs)} oscillators')
            ax3.grid(True, alpha=0.3, axis='x')
        else:
            # Default visualization
            ax3.text(0.5, 0.5, 'Hardware data\nnot provided',
                    transform=ax3.transAxes,
                    ha='center', va='center',
                    fontsize=12, style='italic')
            ax3.axis('off')
        
        # Panel 4: Precision summary
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')
        
        summary_text = (
            'TEMPORAL RESOLUTION SUMMARY\n\n'
            f'Temporal precision Δt:\n'
            f'  {result.temporal_precision:.2e} seconds\n\n'
            f'Phase precision δφ:\n'
            f'  {result.phase_precision:.1e} radians\n\n'
            f'Oscillators (K): {result.oscillator_count}\n'
            f'Demon channels (M): {result.demon_channels}\n'
            f'Cascade depth (R): {result.cascade_depth}\n\n'
            f'Orders below Planck time:\n'
            f'  {-np.log10(result.planck_time_ratio):.2f}\n\n'
            f'Validation error: {result.error_percent:.2f}%'
        )
        
        ax4.text(0.1, 0.9, summary_text,
                transform=ax4.transAxes,
                verticalalignment='top',
                fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
                family='monospace')
        
        plt.suptitle('Trans-Planckian Temporal Resolution Validation',
                    fontsize=14, fontweight='bold', y=0.995)
        
        if save_path is None:
            save_path = f'{self.output_dir}/temporal_resolution_validation.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved temporal resolution validation to: {save_path}")
    
    def plot_all_validations(self,
                            modality_results: Dict,
                            chromatography_result,
                            temporal_result,
                            retention_data: Optional[Dict] = None,
                            hardware_data: Optional[Dict] = None):
        """
        Generate all validation charts.
        
        Parameters:
        -----------
        modality_results : Dict
            Results from five modalities
        chromatography_result : ChromatographyResult
            Chromatography validation results
        temporal_result : TemporalResult
            Temporal resolution results
        retention_data : Dict, optional
            Retention time data
        hardware_data : Dict, optional
            Hardware oscillator data
        """
        print("\nGenerating validation charts...")
        
        self.plot_five_modalities(modality_results)
        self.plot_chromatography(chromatography_result, retention_data)
        self.plot_temporal_resolution(temporal_result, hardware_data)
        
        print("\n✓ All validation charts generated successfully!")
        print(f"  Output directory: {self.output_dir}")
