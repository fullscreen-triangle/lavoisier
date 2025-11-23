"""
Individual Spectrum Deep Dive Analysis
Detailed characterization of each spectrum using both methods

Author: Kundai Sachikonye
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
from scipy.signal import find_peaks
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality style
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.0)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

class SpectrumDeepDive:
    """Detailed analysis of individual spectra"""

    def __init__(self, spectrum_id):
        self.spectrum_id = spectrum_id
        self.data_dir = Path(__file__).parent.parent / 'data'
        self.numerical_data = None
        self.cv_data = None

    def load_data(self):
        """Load data for this spectrum"""
        try:
            num_file = self.data_dir / 'numerical' / f'spectrum_{self.spectrum_id}.tsv'
            self.numerical_data = pd.read_csv(num_file, sep='\t')
            print(f"Loaded numerical data: {len(self.numerical_data)} peaks")
        except FileNotFoundError:
            print(f"Warning: spectrum_{self.spectrum_id}.tsv not found in {self.data_dir / 'numerical'}")

        try:
            cv_file = self.data_dir / 'vision' / 'droplets' / f'spectrum_{self.spectrum_id}_droplets.tsv'
            self.cv_data = pd.read_csv(cv_file, sep='\t')
            print(f"Loaded CV data: {len(self.cv_data)} droplets")
        except FileNotFoundError:
            print(f"Warning: spectrum_{self.spectrum_id}_droplets.tsv not found in {self.data_dir / 'vision' / 'droplets'}")

    def create_deep_dive_figure(self):
        """Create 4-panel deep dive figure"""

        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

        # Panel A: Stick spectrum with annotations
        self._plot_annotated_spectrum(fig, gs[0, 0])

        # Panel B: S-Entropy 3D scatter (CV method)
        self._plot_sentropy_3d(fig, gs[0, 1])

        # Panel C: Physical parameters (CV method)
        self._plot_physical_parameters(fig, gs[1, 0])

        # Panel D: Detailed statistics table
        self._plot_statistics_table(fig, gs[1, 1])

        plt.suptitle(f'Spectrum {self.spectrum_id} - Deep Dive Analysis',
                    fontsize=20, fontweight='bold', y=0.995)

        # Save figure
        plt.savefig(f'spectrum_{self.spectrum_id}_deepdive.png',
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(f'spectrum_{self.spectrum_id}_deepdive.pdf',
                   bbox_inches='tight', facecolor='white')
        print(f"Figure saved: spectrum_{self.spectrum_id}_deepdive.png/pdf")

        return fig

    def _plot_annotated_spectrum(self, fig, gs):
        """Panel A: Annotated stick spectrum"""
        ax = fig.add_subplot(gs)

        if self.numerical_data is None:
            ax.text(0.5, 0.5, 'No data available',
                   ha='center', va='center', fontsize=16)
            return

        mz = self.numerical_data['mz'].values
        intensity = self.numerical_data['i'].values

        # Normalize intensity
        intensity_norm = intensity / intensity.max()

        # Plot stick spectrum
        ax.vlines(mz, 0, intensity_norm, colors='black', linewidth=1.5, alpha=0.7)

        # Find and annotate top 10 peaks
        top_indices = np.argsort(intensity)[-10:]
        for idx in top_indices:
            ax.annotate(f'{mz[idx]:.2f}',
                       xy=(mz[idx], intensity_norm[idx]),
                       xytext=(0, 10), textcoords='offset points',
                       ha='center', fontsize=8, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

        ax.set_xlabel('m/z', fontsize=14, fontweight='bold')
        ax.set_ylabel('Normalized Intensity', fontsize=14, fontweight='bold')
        ax.set_title('A. Stick Spectrum (Top 10 Peaks Annotated)',
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, linestyle='--')

        # Add statistics
        stats_text = f'Peaks: {len(mz)}\n'
        stats_text += f'm/z range: {mz.min():.2f} - {mz.max():.2f}\n'
        stats_text += f'Base peak: {mz[intensity.argmax()]:.2f} m/z'

        ax.text(0.02, 0.98, stats_text,
               transform=ax.transAxes, fontsize=10, fontweight='bold',
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    def _plot_sentropy_3d(self, fig, gs):
        """Panel B: 3D S-Entropy scatter plot"""
        ax = fig.add_subplot(gs, projection='3d')

        if self.cv_data is None or 's_knowledge' not in self.cv_data.columns:
            ax.text(0.5, 0.5, 0.5, 'No CV data available',
                   ha='center', va='center', fontsize=16)
            return

        s_knowledge = self.cv_data['s_knowledge'].values
        s_time = self.cv_data['s_time'].values
        s_entropy = self.cv_data['s_entropy'].values
        intensity = self.cv_data['intensity'].values

        # Color by intensity
        scatter = ax.scatter(s_knowledge, s_time, s_entropy,
                           c=intensity, cmap='viridis', s=50, alpha=0.6,
                           edgecolors='black', linewidth=0.5)

        ax.set_xlabel('S_knowledge', fontsize=12, fontweight='bold')
        ax.set_ylabel('S_time', fontsize=12, fontweight='bold')
        ax.set_zlabel('S_entropy', fontsize=12, fontweight='bold')
        ax.set_title('B. S-Entropy 3D Space (CV Method)',
                    fontsize=16, fontweight='bold', pad=20)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
        cbar.set_label('Intensity', fontsize=12, fontweight='bold')

    def _plot_physical_parameters(self, fig, gs):
        """Panel C: Physical parameters from CV method"""
        ax = fig.add_subplot(gs)

        if self.cv_data is None:
            ax.text(0.5, 0.5, 'No CV data available',
                   ha='center', va='center', fontsize=16)
            return

        # Create 2x2 subgrid
        gs_sub = gs.subgridspec(2, 2, hspace=0.4, wspace=0.4)

        # Velocity distribution
        ax1 = fig.add_subplot(gs_sub[0, 0])
        if 'velocity' in self.cv_data.columns:
            ax1.hist(self.cv_data['velocity'], bins=30, color='skyblue',
                    edgecolor='black', alpha=0.7)
            ax1.set_xlabel('Velocity (m/s)', fontsize=10, fontweight='bold')
            ax1.set_ylabel('Count', fontsize=10, fontweight='bold')
            ax1.set_title('Velocity Distribution', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)

            mean_v = self.cv_data['velocity'].mean()
            std_v = self.cv_data['velocity'].std()
            ax1.axvline(mean_v, color='red', linestyle='--', linewidth=2,
                       label=f'μ={mean_v:.3f}')
            ax1.legend(fontsize=9)

        # Radius distribution
        ax2 = fig.add_subplot(gs_sub[0, 1])
        if 'radius' in self.cv_data.columns:
            ax2.hist(self.cv_data['radius'], bins=30, color='lightcoral',
                    edgecolor='black', alpha=0.7)
            ax2.set_xlabel('Radius (mm)', fontsize=10, fontweight='bold')
            ax2.set_ylabel('Count', fontsize=10, fontweight='bold')
            ax2.set_title('Radius Distribution', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)

            mean_r = self.cv_data['radius'].mean()
            std_r = self.cv_data['radius'].std()
            ax2.axvline(mean_r, color='red', linestyle='--', linewidth=2,
                       label=f'μ={mean_r:.3f}')
            ax2.legend(fontsize=9)

        # Phase coherence distribution
        ax3 = fig.add_subplot(gs_sub[1, 0])
        if 'phase_coherence' in self.cv_data.columns:
            ax3.hist(self.cv_data['phase_coherence'], bins=30, color='lightgreen',
                    edgecolor='black', alpha=0.7)
            ax3.set_xlabel('Phase Coherence', fontsize=10, fontweight='bold')
            ax3.set_ylabel('Count', fontsize=10, fontweight='bold')
            ax3.set_title('Phase Coherence Distribution', fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3)

            mean_pc = self.cv_data['phase_coherence'].mean()
            ax3.axvline(mean_pc, color='red', linestyle='--', linewidth=2,
                       label=f'μ={mean_pc:.3f}')
            ax3.legend(fontsize=9)

        # Physics quality distribution
        ax4 = fig.add_subplot(gs_sub[1, 1])
        if 'physics_quality' in self.cv_data.columns:
            ax4.hist(self.cv_data['physics_quality'], bins=30, color='plum',
                    edgecolor='black', alpha=0.7)
            ax4.set_xlabel('Physics Quality', fontsize=10, fontweight='bold')
            ax4.set_ylabel('Count', fontsize=10, fontweight='bold')
            ax4.set_title('Physics Quality Distribution', fontsize=12, fontweight='bold')
            ax4.grid(True, alpha=0.3)

            mean_pq = self.cv_data['physics_quality'].mean()
            ax4.axvline(mean_pq, color='red', linestyle='--', linewidth=2,
                       label=f'μ={mean_pq:.3f}')
            ax4.legend(fontsize=9)

    def _plot_statistics_table(self, fig, gs):
        """Panel D: Comprehensive statistics table"""
        ax = fig.add_subplot(gs)
        ax.axis('off')

        # Collect statistics
        stats_data = []

        # Numerical method statistics
        if self.numerical_data is not None:
            mz = self.numerical_data['mz'].values
            intensity = self.numerical_data['i'].values

            stats_data.append(['NUMERICAL METHOD', '', ''])
            stats_data.append(['Peak count', len(mz), ''])
            stats_data.append(['m/z range', f'{mz.min():.2f} - {mz.max():.2f}', 'Da'])
            stats_data.append(['Base peak m/z', f'{mz[intensity.argmax()]:.2f}', 'Da'])
            stats_data.append(['Base peak intensity', f'{intensity.max():.0f}', 'counts'])

            # Shannon entropy
            probs = intensity / intensity.sum()
            shannon = -np.sum(probs * np.log2(probs + 1e-10))
            stats_data.append(['Shannon entropy', f'{shannon:.3f}', 'bits'])

            # Gini coefficient
            sorted_int = np.sort(intensity)
            n = len(sorted_int)
            index = np.arange(1, n + 1)
            gini = (2 * np.sum(index * sorted_int)) / (n * np.sum(sorted_int)) - (n + 1) / n
            stats_data.append(['Gini coefficient', f'{gini:.3f}', ''])

            stats_data.append(['', '', ''])

        # CV method statistics
        if self.cv_data is not None:
            stats_data.append(['CV METHOD', '', ''])
            stats_data.append(['Droplet count', len(self.cv_data), ''])

            if 's_knowledge' in self.cv_data.columns:
                stats_data.append(['S_knowledge mean',
                                 f"{self.cv_data['s_knowledge'].mean():.4f}", ''])
                stats_data.append(['S_knowledge std',
                                 f"{self.cv_data['s_knowledge'].std():.4f}", ''])

            if 's_time' in self.cv_data.columns:
                stats_data.append(['S_time mean',
                                 f"{self.cv_data['s_time'].mean():.4f}", ''])
                stats_data.append(['S_time std',
                                 f"{self.cv_data['s_time'].std():.4f}", ''])

            if 's_entropy' in self.cv_data.columns:
                stats_data.append(['S_entropy mean',
                                 f"{self.cv_data['s_entropy'].mean():.4f}", ''])
                stats_data.append(['S_entropy std',
                                 f"{self.cv_data['s_entropy'].std():.4f}", ''])

            if 'velocity' in self.cv_data.columns:
                stats_data.append(['Velocity mean',
                                 f"{self.cv_data['velocity'].mean():.4f}", 'm/s'])
                stats_data.append(['Velocity CV',
                                 f"{(self.cv_data['velocity'].std()/self.cv_data['velocity'].mean()*100):.2f}", '%'])

            if 'radius' in self.cv_data.columns:
                stats_data.append(['Radius mean',
                                 f"{self.cv_data['radius'].mean():.4f}", 'mm'])
                stats_data.append(['Radius CV',
                                 f"{(self.cv_data['radius'].std()/self.cv_data['radius'].mean()*100):.2f}", '%'])

            if 'phase_coherence' in self.cv_data.columns:
                stats_data.append(['Phase coherence mean',
                                 f"{self.cv_data['phase_coherence'].mean():.4f}", ''])

            if 'physics_quality' in self.cv_data.columns:
                stats_data.append(['Physics quality mean',
                                 f"{self.cv_data['physics_quality'].mean():.4f}", ''])
                stats_data.append(['Physics quality std',
                                 f"{self.cv_data['physics_quality'].std():.4f}", ''])

        # Create table
        table = ax.table(cellText=stats_data,
                        colLabels=['Parameter', 'Value', 'Unit'],
                        cellLoc='left',
                        loc='center',
                        colWidths=[0.5, 0.3, 0.2])

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Style header
        for i in range(3):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Style section headers
        for i, row in enumerate(stats_data):
            if row[0] in ['NUMERICAL METHOD', 'CV METHOD']:
                for j in range(3):
                    table[(i+1, j)].set_facecolor('#E0E0E0')
                    table[(i+1, j)].set_text_props(weight='bold')

        ax.set_title('D. Comprehensive Statistics',
                    fontsize=16, fontweight='bold', pad=20)

def main():
    """Analyze all spectra"""

    # Auto-detect available spectra
    data_dir = Path(__file__).parent.parent / 'data'
    droplets_dir = data_dir / 'vision' / 'droplets'
    spectra_ids = set()

    if droplets_dir.exists():
        for file in droplets_dir.glob('spectrum_*_droplets.tsv'):
            # Extract spectrum ID from filename
            spec_id = int(file.stem.split('_')[1])
            spectra_ids.add(spec_id)

    spectra_ids = sorted(list(spectra_ids))

    print("="*80)
    print("INDIVIDUAL SPECTRUM DEEP DIVE ANALYSIS")
    print(f"Data directory: {data_dir}")
    print(f"Found {len(spectra_ids)} spectra: {spectra_ids}")
    print("="*80)

    for spec_id in spectra_ids:
        print(f"\nAnalyzing Spectrum {spec_id}...")
        print("-" * 60)

        analyzer = SpectrumDeepDive(spec_id)
        analyzer.load_data()

        if analyzer.numerical_data is not None or analyzer.cv_data is not None:
            fig = analyzer.create_deep_dive_figure()
            plt.close(fig)
        else:
            print(f"  Skipping spectrum {spec_id} (no data available)")

    print("\n" + "="*80)
    print("Analysis complete!")
    print("Output files:")
    for spec_id in spectra_ids:
        print(f"  - spectrum_{spec_id}_deepdive.png/pdf")

if __name__ == "__main__":
    main()
