"""
Comprehensive Complexity Scaling Analysis
Compares CV (thermodynamic droplet) and Numerical (S-Entropy) methods
across spectral complexity levels

Author: [Kundai Sachikonye]
Date: 2025-10-27
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality style
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.2)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5

# Color palette (colorblind-friendly)
COLORS = {
    'simple': '#0173B2',    # Blue
    'medium': '#DE8F05',    # Orange
    'complex': '#029E73',   # Green
    'cv': '#CC78BC',        # Purple
    'numerical': '#CA9161', # Brown
}

class ComplexityScalingAnalyzer:
    """Comprehensive analysis of complexity scaling in both methods"""

    def __init__(self):
        # Set data directory
        self.data_dir = Path(__file__).parent.parent / 'data'

        # Auto-detect available spectra
        self.spectra_ids = self._detect_spectra()
        self.cv_data = {}
        self.numerical_data = {}
        self.complexity_levels = {}

        print(f"Data directory: {self.data_dir}")
        print(f"Detected {len(self.spectra_ids)} spectra: {self.spectra_ids}")

    def _detect_spectra(self):
        """Auto-detect available spectrum IDs from data directory"""
        spectra_ids = set()
        droplets_dir = self.data_dir / 'vision' / 'droplets'
        if droplets_dir.exists():
            for file in droplets_dir.glob('spectrum_*_droplets.tsv'):
                # Extract spectrum ID from filename
                spec_id = int(file.stem.split('_')[1])
                spectra_ids.add(spec_id)
        return sorted(list(spectra_ids))

    def load_data(self):
        """Load all spectra and droplet data"""
        print("Loading data...")

        for spec_id in self.spectra_ids:
            # Load CV method data (from vision/droplets/)
            try:
                cv_file = self.data_dir / 'vision' / 'droplets' / f'spectrum_{spec_id}_droplets.tsv'
                self.cv_data[spec_id] = pd.read_csv(cv_file, sep='\t')
                print(f"  Loaded CV data for spectrum {spec_id}: {len(self.cv_data[spec_id])} droplets")
            except FileNotFoundError:
                print(f"  Warning: spectrum_{spec_id}_droplets.tsv not found")

            # Load numerical method data (from numerical/)
            try:
                num_file = self.data_dir / 'numerical' / f'spectrum_{spec_id}.tsv'
                self.numerical_data[spec_id] = pd.read_csv(num_file, sep='\t')
                print(f"  Loaded numerical data for spectrum {spec_id}: {len(self.numerical_data[spec_id])} peaks")
            except FileNotFoundError:
                print(f"  Warning: spectrum_{spec_id}.tsv not found")

        # Classify complexity levels
        self._classify_complexity()

    def _classify_complexity(self):
        """Classify spectra into complexity levels based on peak count"""
        peak_counts = {sid: len(self.numerical_data[sid])
                      for sid in self.numerical_data.keys()}

        # Sort by peak count
        sorted_ids = sorted(peak_counts.items(), key=lambda x: x[1])

        # Divide into tertiles
        n = len(sorted_ids)
        self.complexity_levels = {
            'simple': [sid for sid, _ in sorted_ids[:n//3]],
            'medium': [sid for sid, _ in sorted_ids[n//3:2*n//3]],
            'complex': [sid for sid, _ in sorted_ids[2*n//3:]]
        }

        print("\nComplexity classification:")
        for level, ids in self.complexity_levels.items():
            counts = [peak_counts[sid] for sid in ids]
            print(f"  {level.upper()}: {ids} ({min(counts)}-{max(counts)} peaks)")

    def create_comprehensive_figure(self):
        """Create comprehensive 6×3 panel figure"""

        fig = plt.figure(figsize=(24, 36))
        gs = GridSpec(6, 3, figure=fig, hspace=0.3, wspace=0.3)

        # Row 1: Peak/Droplet Count Distributions
        self._plot_count_distributions(fig, gs[0, :])

        # Row 2: Intensity Distributions
        self._plot_intensity_distributions(fig, gs[1, :])

        # Row 3: S-Entropy Coordinate Distributions (CV method)
        self._plot_cv_sentropy_distributions(fig, gs[2, :])

        # Row 4: Physical Parameter Distributions (CV method)
        self._plot_cv_physical_distributions(fig, gs[3, :])

        # Row 5: Complexity Metrics Comparison
        self._plot_complexity_metrics(fig, gs[4, :])

        # Row 6: Scaling Relationships
        self._plot_scaling_relationships(fig, gs[5, :])

        # Add panel labels
        self._add_panel_labels(fig)

        # Save figure
        plt.savefig('complexity_scaling_comprehensive.png',
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig('complexity_scaling_comprehensive.pdf',
                   bbox_inches='tight', facecolor='white')
        print("\nFigure saved: complexity_scaling_comprehensive.png/pdf")

        return fig

    def _plot_count_distributions(self, fig, gs):
        """Row 1: Peak/Droplet count distributions"""

        # Create a 1x3 sub-grid within the passed SubplotSpec
        gs_sub = gs.subgridspec(1, 3, wspace=0.3)

        # Panel A: Peak counts (numerical method)
        ax1 = fig.add_subplot(gs_sub[0])
        peak_counts = []
        labels = []
        colors = []

        for level in ['simple', 'medium', 'complex']:
            for sid in self.complexity_levels[level]:
                if sid in self.numerical_data:
                    peak_counts.append(len(self.numerical_data[sid]))
                    labels.append(f"S{sid}")
                    colors.append(COLORS[level])

        x_pos = np.arange(len(peak_counts))
        bars = ax1.bar(x_pos, peak_counts, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax1.set_xlabel('Spectrum ID', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Peak Count', fontsize=14, fontweight='bold')
        ax1.set_title('A. Spectral Complexity (Numerical Method)',
                     fontsize=16, fontweight='bold', pad=20)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(labels, rotation=45, ha='right')
        ax1.grid(axis='y', alpha=0.3, linestyle='--')

        # Add value labels on bars
        for i, (bar, count) in enumerate(zip(bars, peak_counts)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(count)}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Panel B: Droplet counts (CV method)
        ax2 = fig.add_subplot(gs_sub[1])
        droplet_counts = []
        labels2 = []
        colors2 = []

        for level in ['simple', 'medium', 'complex']:
            for sid in self.complexity_levels[level]:
                if sid in self.cv_data:
                    droplet_counts.append(len(self.cv_data[sid]))
                    labels2.append(f"S{sid}")
                    colors2.append(COLORS[level])

        x_pos2 = np.arange(len(droplet_counts))
        bars2 = ax2.bar(x_pos2, droplet_counts, color=colors2, alpha=0.7,
                       edgecolor='black', linewidth=1.5)
        ax2.set_xlabel('Spectrum ID', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Droplet Count', fontsize=14, fontweight='bold')
        ax2.set_title('B. Spectral Complexity (CV Method)',
                     fontsize=16, fontweight='bold', pad=20)
        ax2.set_xticks(x_pos2)
        ax2.set_xticklabels(labels2, rotation=45, ha='right')
        ax2.grid(axis='y', alpha=0.3, linestyle='--')

        for i, (bar, count) in enumerate(zip(bars2, droplet_counts)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(count)}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Panel C: Correlation between methods
        ax3 = fig.add_subplot(gs_sub[2])

        # Match spectra IDs
        matched_peaks = []
        matched_droplets = []
        matched_colors = []
        matched_labels = []

        for level in ['simple', 'medium', 'complex']:
            for sid in self.complexity_levels[level]:
                if sid in self.numerical_data and sid in self.cv_data:
                    matched_peaks.append(len(self.numerical_data[sid]))
                    matched_droplets.append(len(self.cv_data[sid]))
                    matched_colors.append(COLORS[level])
                    matched_labels.append(f"S{sid}")

        # Scatter plot
        ax3.scatter(matched_peaks, matched_droplets,
                   c=matched_colors, s=200, alpha=0.7,
                   edgecolors='black', linewidth=2)

        # Add labels
        for i, label in enumerate(matched_labels):
            ax3.annotate(label, (matched_peaks[i], matched_droplets[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10, fontweight='bold')

        # Fit line
        if len(matched_peaks) > 1:
            z = np.polyfit(matched_peaks, matched_droplets, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(matched_peaks), max(matched_peaks), 100)
            ax3.plot(x_line, p(x_line), 'k--', linewidth=2, alpha=0.5,
                    label=f'y = {z[0]:.3f}x + {z[1]:.1f}')

            # Calculate R²
            r2 = np.corrcoef(matched_peaks, matched_droplets)[0,1]**2
            ax3.text(0.05, 0.95, f'R² = {r2:.4f}',
                    transform=ax3.transAxes, fontsize=12, fontweight='bold',
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax3.set_xlabel('Peak Count (Numerical)', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Droplet Count (CV)', fontsize=14, fontweight='bold')
        ax3.set_title('C. Method Correlation',
                     fontsize=16, fontweight='bold', pad=20)
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.legend(fontsize=11, loc='lower right')

        # Add legend for complexity levels
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=COLORS['simple'], label='Simple'),
                          Patch(facecolor=COLORS['medium'], label='Medium'),
                          Patch(facecolor=COLORS['complex'], label='Complex')]
        ax3.legend(handles=legend_elements, loc='upper left', fontsize=11)

    def _plot_intensity_distributions(self, fig, gs):
        """Row 2: Intensity distributions across complexity levels"""

        # Create subgrid for this row
        gs_sub = gs.subgridspec(1, 3, wspace=0.3)

        for idx, level in enumerate(['simple', 'medium', 'complex']):
            ax = fig.add_subplot(gs_sub[idx])

            # Collect intensity data for this complexity level
            all_intensities = []

            for sid in self.complexity_levels[level]:
                if sid in self.numerical_data:
                    intensities = self.numerical_data[sid]['i'].values
                    # Normalize to [0, 1]
                    intensities = intensities / intensities.max()
                    all_intensities.append(intensities)

            if all_intensities:
                # Plot distributions
                for i, intensities in enumerate(all_intensities):
                    ax.hist(intensities, bins=50, alpha=0.5,
                           label=f"S{self.complexity_levels[level][i]}",
                           color=COLORS[level], edgecolor='black', linewidth=0.5)

                ax.set_xlabel('Normalized Intensity', fontsize=14, fontweight='bold')
                ax.set_ylabel('Frequency', fontsize=14, fontweight='bold')
                ax.set_title(f'{chr(68+idx)}. {level.upper()} Complexity\nIntensity Distribution',
                           fontsize=16, fontweight='bold', pad=20)
                ax.set_yscale('log')
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.legend(fontsize=10, loc='upper right')

                # Add statistics
                all_combined = np.concatenate(all_intensities)
                median_val = np.median(all_combined)
                mean_val = np.mean(all_combined)
                ax.axvline(median_val, color='red', linestyle='--', linewidth=2,
                          label=f'Median: {median_val:.3f}')
                ax.axvline(mean_val, color='blue', linestyle='--', linewidth=2,
                          label=f'Mean: {mean_val:.3f}')

    def _plot_cv_sentropy_distributions(self, fig, gs):
        """Row 3: S-Entropy coordinate distributions from CV method"""

        # Create subgrid for this row
        gs_sub = gs.subgridspec(1, 3, wspace=0.3)

        features = ['s_knowledge', 's_time', 's_entropy']
        titles = ['G. S_knowledge Distribution', 'H. S_time Distribution', 'I. S_entropy Distribution']

        for idx, (feature, title) in enumerate(zip(features, titles)):
            ax = fig.add_subplot(gs_sub[idx])

            # Collect data for each complexity level
            data_by_level = {level: [] for level in ['simple', 'medium', 'complex']}

            for level in ['simple', 'medium', 'complex']:
                for sid in self.complexity_levels[level]:
                    if sid in self.cv_data and feature in self.cv_data[sid].columns:
                        data_by_level[level].extend(self.cv_data[sid][feature].values)

            # Create violin plots
            positions = [1, 2, 3]
            parts = ax.violinplot([data_by_level['simple'],
                                   data_by_level['medium'],
                                   data_by_level['complex']],
                                  positions=positions,
                                  showmeans=True, showmedians=True,
                                  widths=0.7)

            # Color the violins
            for i, (pc, level) in enumerate(zip(parts['bodies'], ['simple', 'medium', 'complex'])):
                pc.set_facecolor(COLORS[level])
                pc.set_alpha(0.7)
                pc.set_edgecolor('black')
                pc.set_linewidth(1.5)

            # Style the other elements
            for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans'):
                if partname in parts:
                    vp = parts[partname]
                    vp.set_edgecolor('black')
                    vp.set_linewidth(2)

            ax.set_xticks(positions)
            ax.set_xticklabels(['Simple', 'Medium', 'Complex'], fontsize=12, fontweight='bold')
            ax.set_ylabel(feature.replace('_', ' ').title(), fontsize=14, fontweight='bold')
            ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
            ax.grid(axis='y', alpha=0.3, linestyle='--')

            # Add statistics
            for i, level in enumerate(['simple', 'medium', 'complex']):
                data = data_by_level[level]
                if data:
                    mean_val = np.mean(data)
                    std_val = np.std(data)
                    ax.text(positions[i], ax.get_ylim()[1]*0.95,
                           f'μ={mean_val:.3f}\nσ={std_val:.3f}',
                           ha='center', va='top', fontsize=9,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    def _plot_cv_physical_distributions(self, fig, gs):
        """Row 4: Physical parameter distributions from CV method"""

        # Create subgrid for this row
        gs_sub = gs.subgridspec(1, 3, wspace=0.3)

        features = ['velocity', 'radius', 'phase_coherence']
        titles = ['J. Velocity Distribution', 'K. Radius Distribution', 'L. Phase Coherence Distribution']

        for idx, (feature, title) in enumerate(zip(features, titles)):
            ax = fig.add_subplot(gs_sub[idx])

            # Collect data for each complexity level
            data_by_level = {level: [] for level in ['simple', 'medium', 'complex']}

            for level in ['simple', 'medium', 'complex']:
                for sid in self.complexity_levels[level]:
                    if sid in self.cv_data and feature in self.cv_data[sid].columns:
                        data_by_level[level].extend(self.cv_data[sid][feature].values)

            # Create box plots with swarm overlay
            positions = [1, 2, 3]
            bp = ax.boxplot([data_by_level['simple'],
                            data_by_level['medium'],
                            data_by_level['complex']],
                           positions=positions,
                           widths=0.5,
                           patch_artist=True,
                           showfliers=False)

            # Color the boxes
            for patch, level in zip(bp['boxes'], ['simple', 'medium', 'complex']):
                patch.set_facecolor(COLORS[level])
                patch.set_alpha(0.7)
                patch.set_edgecolor('black')
                patch.set_linewidth(1.5)

            # Style whiskers, caps, medians
            for element in ['whiskers', 'caps', 'medians']:
                for item in bp[element]:
                    item.set_color('black')
                    item.set_linewidth(2)

            ax.set_xticks(positions)
            ax.set_xticklabels(['Simple', 'Medium', 'Complex'], fontsize=12, fontweight='bold')
            ax.set_ylabel(feature.replace('_', ' ').title(), fontsize=14, fontweight='bold')
            ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
            ax.grid(axis='y', alpha=0.3, linestyle='--')

            # Add coefficient of variation
            for i, level in enumerate(['simple', 'medium', 'complex']):
                data = data_by_level[level]
                if data:
                    cv = (np.std(data) / np.mean(data)) * 100
                    ax.text(positions[i], ax.get_ylim()[0],
                           f'CV={cv:.2f}%',
                           ha='center', va='bottom', fontsize=9, fontweight='bold',
                           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    def _plot_complexity_metrics(self, fig, gs):
        """Row 5: Comprehensive complexity metrics"""

        # Create a 1x3 sub-grid within the passed SubplotSpec
        gs_sub = gs.subgridspec(1, 3, wspace=0.3)

        # Panel M: Shannon entropy vs peak count
        ax1 = fig.add_subplot(gs_sub[0])

        peak_counts = []
        shannon_entropies = []
        colors_list = []
        labels_list = []

        for level in ['simple', 'medium', 'complex']:
            for sid in self.complexity_levels[level]:
                if sid in self.numerical_data:
                    intensities = self.numerical_data[sid]['i'].values
                    # Normalize
                    probs = intensities / intensities.sum()
                    # Calculate Shannon entropy
                    entropy = -np.sum(probs * np.log2(probs + 1e-10))

                    peak_counts.append(len(self.numerical_data[sid]))
                    shannon_entropies.append(entropy)
                    colors_list.append(COLORS[level])
                    labels_list.append(f"S{sid}")

        ax1.scatter(peak_counts, shannon_entropies, c=colors_list, s=200,
                   alpha=0.7, edgecolors='black', linewidth=2)

        for i, label in enumerate(labels_list):
            ax1.annotate(label, (peak_counts[i], shannon_entropies[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10, fontweight='bold')

        ax1.set_xlabel('Peak Count', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Shannon Entropy (bits)', fontsize=14, fontweight='bold')
        ax1.set_title('M. Spectral Complexity:\nShannon Entropy',
                     fontsize=16, fontweight='bold', pad=20)
        ax1.grid(True, alpha=0.3, linestyle='--')

        # Panel N: Gini coefficient (intensity inequality)
        ax2 = fig.add_subplot(gs_sub[1])

        peak_counts2 = []
        gini_coeffs = []
        colors_list2 = []
        labels_list2 = []

        for level in ['simple', 'medium', 'complex']:
            for sid in self.complexity_levels[level]:
                if sid in self.numerical_data:
                    intensities = self.numerical_data[sid]['i'].values
                    # Calculate Gini coefficient
                    sorted_int = np.sort(intensities)
                    n = len(sorted_int)
                    index = np.arange(1, n + 1)
                    gini = (2 * np.sum(index * sorted_int)) / (n * np.sum(sorted_int)) - (n + 1) / n

                    peak_counts2.append(len(self.numerical_data[sid]))
                    gini_coeffs.append(gini)
                    colors_list2.append(COLORS[level])
                    labels_list2.append(f"S{sid}")

        ax2.scatter(peak_counts2, gini_coeffs, c=colors_list2, s=200,
                   alpha=0.7, edgecolors='black', linewidth=2)

        for i, label in enumerate(labels_list2):
            ax2.annotate(label, (peak_counts2[i], gini_coeffs[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10, fontweight='bold')

        ax2.set_xlabel('Peak Count', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Gini Coefficient', fontsize=14, fontweight='bold')
        ax2.set_title('N. Intensity Inequality:\nGini Coefficient',
                     fontsize=16, fontweight='bold', pad=20)
        ax2.grid(True, alpha=0.3, linestyle='--')

        # Panel O: Mean S_entropy from CV method
        ax3 = fig.add_subplot(gs_sub[2])

        droplet_counts = []
        mean_sentropies = []
        colors_list3 = []
        labels_list3 = []

        for level in ['simple', 'medium', 'complex']:
            for sid in self.complexity_levels[level]:
                if sid in self.cv_data and 's_entropy' in self.cv_data[sid].columns:
                    mean_s = self.cv_data[sid]['s_entropy'].mean()

                    droplet_counts.append(len(self.cv_data[sid]))
                    mean_sentropies.append(mean_s)
                    colors_list3.append(COLORS[level])
                    labels_list3.append(f"S{sid}")

        ax3.scatter(droplet_counts, mean_sentropies, c=colors_list3, s=200,
                   alpha=0.7, edgecolors='black', linewidth=2)

        for i, label in enumerate(labels_list3):
            ax3.annotate(label, (droplet_counts[i], mean_sentropies[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10, fontweight='bold')

        ax3.set_xlabel('Droplet Count', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Mean S_entropy', fontsize=14, fontweight='bold')
        ax3.set_title('O. CV Method Complexity:\nMean S_entropy',
                     fontsize=16, fontweight='bold', pad=20)
        ax3.grid(True, alpha=0.3, linestyle='--')

    def _plot_scaling_relationships(self, fig, gs):
        """Row 6: Scaling relationships and power laws"""

        # Create a 1x3 sub-grid within the passed SubplotSpec
        gs_sub = gs.subgridspec(1, 3, wspace=0.3)

        # Panel P: m/z range vs complexity
        ax1 = fig.add_subplot(gs_sub[0])

        peak_counts = []
        mz_ranges = []
        colors_list = []
        labels_list = []

        for level in ['simple', 'medium', 'complex']:
            for sid in self.complexity_levels[level]:
                if sid in self.numerical_data:
                    mz_vals = self.numerical_data[sid]['mz'].values
                    mz_range = mz_vals.max() - mz_vals.min()

                    peak_counts.append(len(self.numerical_data[sid]))
                    mz_ranges.append(mz_range)
                    colors_list.append(COLORS[level])
                    labels_list.append(f"S{sid}")

        ax1.scatter(peak_counts, mz_ranges, c=colors_list, s=200,
                   alpha=0.7, edgecolors='black', linewidth=2)

        for i, label in enumerate(labels_list):
            ax1.annotate(label, (peak_counts[i], mz_ranges[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10, fontweight='bold')

        ax1.set_xlabel('Peak Count', fontsize=14, fontweight='bold')
        ax1.set_ylabel('m/z Range (Da)', fontsize=14, fontweight='bold')
        ax1.set_title('P. Spectral Coverage:\nm/z Range',
                     fontsize=16, fontweight='bold', pad=20)
        ax1.grid(True, alpha=0.3, linestyle='--')

        # Panel Q: Physics quality vs complexity (CV method)
        ax2 = fig.add_subplot(gs_sub[1])

        droplet_counts = []
        mean_physics_quality = []
        std_physics_quality = []
        colors_list2 = []
        labels_list2 = []

        for level in ['simple', 'medium', 'complex']:
            for sid in self.complexity_levels[level]:
                if sid in self.cv_data and 'physics_quality' in self.cv_data[sid].columns:
                    mean_pq = self.cv_data[sid]['physics_quality'].mean()
                    std_pq = self.cv_data[sid]['physics_quality'].std()

                    droplet_counts.append(len(self.cv_data[sid]))
                    mean_physics_quality.append(mean_pq)
                    std_physics_quality.append(std_pq)
                    colors_list2.append(COLORS[level])
                    labels_list2.append(f"S{sid}")

        ax2.errorbar(droplet_counts, mean_physics_quality,
                    yerr=std_physics_quality,
                    fmt='o', markersize=12, capsize=5, capthick=2,
                    elinewidth=2, alpha=0.7)

        for i in range(len(droplet_counts)):
            ax2.scatter(droplet_counts[i], mean_physics_quality[i],
                       c=colors_list2[i], s=200, edgecolors='black',
                       linewidth=2, zorder=10)
            ax2.annotate(labels_list2[i],
                        (droplet_counts[i], mean_physics_quality[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10, fontweight='bold')

        ax2.set_xlabel('Droplet Count', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Physics Quality Score', fontsize=14, fontweight='bold')
        ax2.set_title('Q. Physical Validation:\nQuality vs Complexity',
                     fontsize=16, fontweight='bold', pad=20)
        ax2.grid(True, alpha=0.3, linestyle='--')

        # Panel R: Complexity scaling comparison
        ax3 = fig.add_subplot(gs_sub[2])

        # Get matched data
        matched_peaks = []
        matched_droplets = []
        matched_shannon = []
        matched_sentropy = []
        matched_colors = []
        matched_labels = []

        for level in ['simple', 'medium', 'complex']:
            for sid in self.complexity_levels[level]:
                if (sid in self.numerical_data and sid in self.cv_data and
                    's_entropy' in self.cv_data[sid].columns):

                    # Numerical complexity (Shannon entropy)
                    intensities = self.numerical_data[sid]['i'].values
                    probs = intensities / intensities.sum()
                    shannon = -np.sum(probs * np.log2(probs + 1e-10))

                    # CV complexity (mean S_entropy)
                    sentropy = self.cv_data[sid]['s_entropy'].mean()

                    matched_peaks.append(len(self.numerical_data[sid]))
                    matched_droplets.append(len(self.cv_data[sid]))
                    matched_shannon.append(shannon)
                    matched_sentropy.append(sentropy)
                    matched_colors.append(COLORS[level])
                    matched_labels.append(f"S{sid}")

        # Normalize both metrics to [0, 1]
        shannon_norm = (np.array(matched_shannon) - min(matched_shannon)) / \
                      (max(matched_shannon) - min(matched_shannon))
        sentropy_norm = (np.array(matched_sentropy) - min(matched_sentropy)) / \
                       (max(matched_sentropy) - min(matched_sentropy))

        ax3.scatter(shannon_norm, sentropy_norm, c=matched_colors, s=200,
                   alpha=0.7, edgecolors='black', linewidth=2)

        for i, label in enumerate(matched_labels):
            ax3.annotate(label, (shannon_norm[i], sentropy_norm[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10, fontweight='bold')

        # Diagonal line (perfect correlation)
        ax3.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5,
                label='Perfect correlation')

        # Calculate correlation
        if len(shannon_norm) > 1:
            r = np.corrcoef(shannon_norm, sentropy_norm)[0, 1]
            ax3.text(0.05, 0.95, f'r = {r:.4f}',
                    transform=ax3.transAxes, fontsize=12, fontweight='bold',
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax3.set_xlabel('Normalized Shannon Entropy\n(Numerical Method)',
                      fontsize=14, fontweight='bold')
        ax3.set_ylabel('Normalized Mean S_entropy\n(CV Method)',
                      fontsize=14, fontweight='bold')
        ax3.set_title('R. Method Correlation:\nComplexity Metrics',
                     fontsize=16, fontweight='bold', pad=20)
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.legend(fontsize=11, loc='lower right')
        ax3.set_xlim(-0.05, 1.05)
        ax3.set_ylim(-0.05, 1.05)

    def _add_panel_labels(self, fig):
        """Add panel labels A-R"""
        # This is handled in individual plot titles
        pass

    def generate_statistics_report(self):
        """Generate comprehensive statistics report"""

        print("\n" + "="*80)
        print("COMPLEXITY SCALING ANALYSIS - STATISTICAL REPORT")
        print("="*80)

        for level in ['simple', 'medium', 'complex']:
            print(f"\n{level.upper()} COMPLEXITY SPECTRA:")
            print("-" * 60)

            # Numerical method statistics
            peak_counts = []
            shannon_entropies = []
            gini_coeffs = []
            mz_ranges = []

            for sid in self.complexity_levels[level]:
                if sid in self.numerical_data:
                    data = self.numerical_data[sid]
                    peak_counts.append(len(data))

                    # Shannon entropy
                    intensities = data['i'].values
                    probs = intensities / intensities.sum()
                    shannon = -np.sum(probs * np.log2(probs + 1e-10))
                    shannon_entropies.append(shannon)

                    # Gini coefficient
                    sorted_int = np.sort(intensities)
                    n = len(sorted_int)
                    index = np.arange(1, n + 1)
                    gini = (2 * np.sum(index * sorted_int)) / (n * np.sum(sorted_int)) - (n + 1) / n
                    gini_coeffs.append(gini)

                    # m/z range
                    mz_range = data['mz'].max() - data['mz'].min()
                    mz_ranges.append(mz_range)

            if peak_counts:
                print(f"  Numerical Method:")
                print(f"    Peak count: {np.mean(peak_counts):.1f} ± {np.std(peak_counts):.1f}")
                print(f"    Shannon entropy: {np.mean(shannon_entropies):.3f} ± {np.std(shannon_entropies):.3f} bits")
                print(f"    Gini coefficient: {np.mean(gini_coeffs):.3f} ± {np.std(gini_coeffs):.3f}")
                print(f"    m/z range: {np.mean(mz_ranges):.1f} ± {np.std(mz_ranges):.1f} Da")

            # CV method statistics
            droplet_counts = []
            mean_sentropies = []
            mean_velocities = []
            mean_radii = []
            mean_physics_quality = []

            for sid in self.complexity_levels[level]:
                if sid in self.cv_data:
                    data = self.cv_data[sid]
                    droplet_counts.append(len(data))

                    if 's_entropy' in data.columns:
                        mean_sentropies.append(data['s_entropy'].mean())
                    if 'velocity' in data.columns:
                        mean_velocities.append(data['velocity'].mean())
                    if 'radius' in data.columns:
                        mean_radii.append(data['radius'].mean())
                    if 'physics_quality' in data.columns:
                        mean_physics_quality.append(data['physics_quality'].mean())

            if droplet_counts:
                print(f"  CV Method:")
                print(f"    Droplet count: {np.mean(droplet_counts):.1f} ± {np.std(droplet_counts):.1f}")
                if mean_sentropies:
                    print(f"    Mean S_entropy: {np.mean(mean_sentropies):.4f} ± {np.std(mean_sentropies):.4f}")
                if mean_velocities:
                    print(f"    Mean velocity: {np.mean(mean_velocities):.4f} ± {np.std(mean_velocities):.4f} m/s")
                if mean_radii:
                    print(f"    Mean radius: {np.mean(mean_radii):.4f} ± {np.std(mean_radii):.4f} mm")
                if mean_physics_quality:
                    print(f"    Mean physics quality: {np.mean(mean_physics_quality):.4f} ± {np.std(mean_physics_quality):.4f}")

        # Cross-method correlations
        print(f"\nCROSS-METHOD CORRELATIONS:")
        print("-" * 60)

        matched_peaks = []
        matched_droplets = []
        matched_shannon = []
        matched_sentropy = []

        for level in ['simple', 'medium', 'complex']:
            for sid in self.complexity_levels[level]:
                if (sid in self.numerical_data and sid in self.cv_data and
                    's_entropy' in self.cv_data[sid].columns):

                    matched_peaks.append(len(self.numerical_data[sid]))
                    matched_droplets.append(len(self.cv_data[sid]))

                    intensities = self.numerical_data[sid]['i'].values
                    probs = intensities / intensities.sum()
                    shannon = -np.sum(probs * np.log2(probs + 1e-10))
                    matched_shannon.append(shannon)

                    matched_sentropy.append(self.cv_data[sid]['s_entropy'].mean())

        if len(matched_peaks) > 1:
            r_counts = np.corrcoef(matched_peaks, matched_droplets)[0, 1]
            r_complexity = np.corrcoef(matched_shannon, matched_sentropy)[0, 1]

            print(f"  Peak count vs Droplet count: r = {r_counts:.4f}")
            print(f"  Shannon entropy vs Mean S_entropy: r = {r_complexity:.4f}")

        print("\n" + "="*80)

def main():
    """Main execution function"""

    print("="*80)
    print("COMPREHENSIVE COMPLEXITY SCALING ANALYSIS")
    print("Comparing CV (thermodynamic droplet) and Numerical (S-Entropy) methods")
    print("="*80)

    # Initialize analyzer
    analyzer = ComplexityScalingAnalyzer()

    # Load data
    analyzer.load_data()

    # Generate comprehensive figure
    print("\nGenerating comprehensive figure...")
    fig = analyzer.create_comprehensive_figure()

    # Generate statistics report
    analyzer.generate_statistics_report()

    print("\nAnalysis complete!")
    print("Output files:")
    print("  - complexity_scaling_comprehensive.png")
    print("  - complexity_scaling_comprehensive.pdf")

if __name__ == "__main__":
    main()
