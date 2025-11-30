"""
Fragmentation Spectra Visualization - REAL DATA

Shows selected spectra through pipeline stages using ACTUAL data.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path
script_dir = Path(__file__).parent
src_dir = script_dir.parent
sys.path.insert(0, str(src_dir))

from virtual.load_real_data import load_comparison_data

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300


def visualize_spectrum_sentropy(spectrum_coords, platform_name, spectrum_idx, output_dir):
    """
    Visualize a single spectrum's S-Entropy coordinates in 3D

    Args:
        spectrum_coords: Nx3 array of S-Entropy coordinates for one spectrum
        platform_name: Platform name
        spectrum_idx: Spectrum index
        output_dir: Output directory
    """
    if spectrum_coords is None or len(spectrum_coords) == 0:
        return None

    fig = plt.figure(figsize=(14, 5))
    fig.suptitle(f'Spectrum {spectrum_idx} - S-Entropy Transformation\n{platform_name}',
                 fontsize=13, fontweight='bold')

    s_k = spectrum_coords[:, 0]
    s_t = spectrum_coords[:, 1]
    s_e = spectrum_coords[:, 2]

    # 3D scatter
    ax1 = fig.add_subplot(131, projection='3d')
    scatter = ax1.scatter(s_k, s_t, s_e, c=s_e, cmap='viridis', s=20, alpha=0.6)
    ax1.set_xlabel('S_k')
    ax1.set_ylabel('S_t')
    ax1.set_zlabel('S_e')
    ax1.set_title(f'{len(s_e)} droplets')
    plt.colorbar(scatter, ax=ax1, shrink=0.6)

    # S_k vs S_t
    ax2 = fig.add_subplot(132)
    ax2.scatter(s_k, s_t, c=s_e, cmap='viridis', s=30, alpha=0.6)
    ax2.set_xlabel('S-Knowledge')
    ax2.set_ylabel('S-Time')
    ax2.set_title('S_k vs S_t')
    ax2.grid(alpha=0.3)

    # S-Entropy histogram
    ax3 = fig.add_subplot(133)
    ax3.hist(s_e, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax3.set_xlabel('S-Entropy')
    ax3.set_ylabel('Frequency')
    ax3.set_title(f'S_e Distribution\nMean={s_e.mean():.4f}')
    ax3.grid(alpha=0.3)

    plt.tight_layout()

    output_file = output_dir / f"spectrum_{spectrum_idx}_{platform_name}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    return output_file


def main():
    """Main visualization workflow"""
    print("="*80)
    print("FRAGMENTATION SPECTRA VISUALIZATION - REAL DATA")
    print("="*80)

    # Determine paths
    script_dir = Path(__file__).parent
    precursor_root = script_dir.parent.parent
    results_dir = precursor_root / "results" / "fragmentation_comparison"
    output_dir = precursor_root / "visualizations"
    output_dir.mkdir(exist_ok=True)

    # Load REAL data
    print("\nLoading REAL spectra data...")
    data = load_comparison_data(str(results_dir))

    if not data:
        print("ERROR: No data loaded!")
        return

    # Visualize 10 random spectra from each platform
    n_spectra_to_plot = min(10, min([d['n_spectra'] for d in data.values()]))

    for platform_name, platform_data in data.items():
        print(f"\n{platform_name}: Visualizing {n_spectra_to_plot} spectra...")

        coords_by_spectrum = platform_data['coords_by_spectrum']

        # Select random spectra
        n_available = len(coords_by_spectrum)
        indices = np.random.choice(n_available, min(n_spectra_to_plot, n_available), replace=False)

        for idx in indices:
            spectrum_coords = coords_by_spectrum[idx]
            output_file = visualize_spectrum_sentropy(spectrum_coords, platform_name, idx, output_dir)
            if output_file:
                print(f"  ✓ Spectrum {idx}: {len(spectrum_coords)} droplets")

    print("\n" + "="*80)
    print("✓ FRAGMENTATION SPECTRA VISUALIZATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
