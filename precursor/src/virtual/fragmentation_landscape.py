"""
Fragmentation Landscape Visualization - REAL DATA

Creates 3D landscapes of ACTUAL fragmentation events in S-Entropy space.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
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


def create_fragmentation_landscape_3d(data, platform_name, output_dir):
    """
    Create 3D landscape of fragmentation in S-Entropy space
    """
    s_k = data['s_knowledge']
    s_t = data['s_time']
    s_e = data['s_entropy']

    fig = plt.figure(figsize=(16, 6))

    # 3D scatter with S_e as color
    ax1 = fig.add_subplot(131, projection='3d')
    scatter = ax1.scatter(s_k, s_t, s_e, c=s_e, cmap='viridis', s=2, alpha=0.5)
    ax1.set_xlabel('S-Knowledge')
    ax1.set_ylabel('S-Time')
    ax1.set_zlabel('S-Entropy')
    ax1.set_title(f'Fragmentation Landscape\n{platform_name}')
    plt.colorbar(scatter, ax=ax1, label='S-Entropy', shrink=0.7)

    # View from top (S_k vs S_t)
    ax2 = fig.add_subplot(132)
    ax2.hexbin(s_k, s_t, gridsize=50, cmap='plasma', mincnt=1)
    ax2.set_xlabel('S-Knowledge')
    ax2.set_ylabel('S-Time')
    ax2.set_title('Top View (S_k vs S_t)')

    # Density heatmap
    ax3 = fig.add_subplot(133)
    h, xedges, yedges = np.histogram2d(s_k, s_e, bins=50)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax3.imshow(h.T, extent=extent, origin='lower', cmap='hot', aspect='auto')
    ax3.set_xlabel('S-Knowledge')
    ax3.set_ylabel('S-Entropy')
    ax3.set_title('Density Heatmap (S_k vs S_e)')
    plt.colorbar(im, ax=ax3, label='Density')

    plt.tight_layout()

    output_file = output_dir / f"fragmentation_landscape_{platform_name}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_file.name}")
    return output_file


def main():
    """Main visualization workflow"""
    print("="*80)
    print("FRAGMENTATION LANDSCAPE VISUALIZATION - REAL DATA")
    print("="*80)

    # Determine paths
    script_dir = Path(__file__).parent
    precursor_root = script_dir.parent.parent
    results_dir = precursor_root / "results" / "fragmentation_comparison"
    output_dir = precursor_root / "visualizations"
    output_dir.mkdir(exist_ok=True)

    # Load REAL data
    print("\nLoading REAL fragmentation data...")
    data = load_comparison_data(str(results_dir))

    if not data:
        print("ERROR: No data loaded!")
        return

    # Create visualizations
    for platform_name, platform_data in data.items():
        print(f"\n{platform_name}: {platform_data['n_droplets']} droplets")
        create_fragmentation_landscape_3d(platform_data, platform_name, output_dir)

    print("\n" + "="*80)
    print("✓ FRAGMENTATION LANDSCAPE COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
