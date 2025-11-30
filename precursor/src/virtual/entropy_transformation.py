"""
S-Entropy Transformation Visualization - REAL DATA

Loads ACTUAL S-Entropy coordinates from stage_02_sentropy_data.tab
and visualizes the 3D transformation space.
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


def create_sentropy_3d_plot(data, platform_name, output_dir):
    """
    Create 3D scatter plot of REAL S-Entropy coordinates
    """
    s_k = data['s_knowledge']
    s_t = data['s_time']
    s_e = data['s_entropy']

    fig = plt.figure(figsize=(15, 12))

    # Main 3D plot
    ax = fig.add_subplot(221, projection='3d')
    scatter = ax.scatter(s_k, s_t, s_e, c=s_e, cmap='viridis', s=1, alpha=0.6)
    ax.set_xlabel('S-Knowledge', fontsize=11, fontweight='bold')
    ax.set_ylabel('S-Time', fontsize=11, fontweight='bold')
    ax.set_zlabel('S-Entropy', fontsize=11, fontweight='bold')
    ax.set_title(f'S-Entropy 3D Space\n{platform_name}\n{data["n_droplets"]} droplets from {data["n_spectra"]} spectra',
                 fontsize=12, fontweight='bold')
    plt.colorbar(scatter, ax=ax, label='S-Entropy', shrink=0.5)

    # 2D projections
    # S_k vs S_t
    ax2 = fig.add_subplot(222)
    ax2.hexbin(s_k, s_t, gridsize=50, cmap='viridis', mincnt=1)
    ax2.set_xlabel('S-Knowledge', fontsize=10, fontweight='bold')
    ax2.set_ylabel('S-Time', fontsize=10, fontweight='bold')
    ax2.set_title('S_k vs S_t Projection', fontsize=11, fontweight='bold')

    # S_k vs S_e
    ax3 = fig.add_subplot(223)
    ax3.hexbin(s_k, s_e, gridsize=50, cmap='plasma', mincnt=1)
    ax3.set_xlabel('S-Knowledge', fontsize=10, fontweight='bold')
    ax3.set_ylabel('S-Entropy', fontsize=10, fontweight='bold')
    ax3.set_title('S_k vs S_e Projection', fontsize=11, fontweight='bold')

    # S_t vs S_e
    ax4 = fig.add_subplot(224)
    ax4.hexbin(s_t, s_e, gridsize=50, cmap='inferno', mincnt=1)
    ax4.set_xlabel('S-Time', fontsize=10, fontweight='bold')
    ax4.set_ylabel('S-Entropy', fontsize=10, fontweight='bold')
    ax4.set_title('S_t vs S_e Projection', fontsize=11, fontweight='bold')

    plt.tight_layout()

    output_file = output_dir / f"sentropy_3d_{platform_name}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_file.name}")
    return output_file


def create_sentropy_distributions(data, platform_name, output_dir):
    """
    Create distribution plots for each S-Entropy dimension
    """
    s_k = data['s_knowledge']
    s_t = data['s_time']
    s_e = data['s_entropy']

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(f'S-Entropy Coordinate Distributions - {platform_name}',
                 fontsize=14, fontweight='bold')

    # S-Knowledge
    axes[0, 0].hist(s_k, bins=100, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('S-Knowledge', fontsize=10, fontweight='bold')
    axes[0, 0].set_ylabel('Frequency', fontsize=10, fontweight='bold')
    axes[0, 0].set_title(f'S_k Distribution\nMean={s_k.mean():.2f}, Std={s_k.std():.2f}', fontsize=11)
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].boxplot(s_k, vert=True)
    axes[0, 1].set_ylabel('S-Knowledge', fontsize=10, fontweight='bold')
    axes[0, 1].set_title('S_k Boxplot', fontsize=11)
    axes[0, 1].grid(alpha=0.3)

    # S-Time
    axes[1, 0].hist(s_t, bins=100, color='lightgreen', edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('S-Time', fontsize=10, fontweight='bold')
    axes[1, 0].set_ylabel('Frequency', fontsize=10, fontweight='bold')
    axes[1, 0].set_title(f'S_t Distribution\nMean={s_t.mean():.2f}, Std={s_t.std():.2f}', fontsize=11)
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].boxplot(s_t, vert=True)
    axes[1, 1].set_ylabel('S-Time', fontsize=10, fontweight='bold')
    axes[1, 1].set_title('S_t Boxplot', fontsize=11)
    axes[1, 1].grid(alpha=0.3)

    # S-Entropy
    axes[2, 0].hist(s_e, bins=100, color='salmon', edgecolor='black', alpha=0.7)
    axes[2, 0].set_xlabel('S-Entropy', fontsize=10, fontweight='bold')
    axes[2, 0].set_ylabel('Frequency', fontsize=10, fontweight='bold')
    axes[2, 0].set_title(f'S_e Distribution\nMean={s_e.mean():.4f}, Std={s_e.std():.4f}', fontsize=11)
    axes[2, 0].grid(alpha=0.3)

    axes[2, 1].boxplot(s_e, vert=True)
    axes[2, 1].set_ylabel('S-Entropy', fontsize=10, fontweight='bold')
    axes[2, 1].set_title('S_e Boxplot', fontsize=11)
    axes[2, 1].grid(alpha=0.3)

    plt.tight_layout()

    output_file = output_dir / f"sentropy_distributions_{platform_name}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_file.name}")
    return output_file


def main():
    """Main visualization workflow"""
    print("="*80)
    print("S-ENTROPY TRANSFORMATION VISUALIZATION - REAL DATA")
    print("="*80)

    # Determine paths
    script_dir = Path(__file__).parent
    precursor_root = script_dir.parent.parent
    results_dir = precursor_root / "results" / "fragmentation_comparison"
    output_dir = precursor_root / "visualizations"
    output_dir.mkdir(exist_ok=True)

    print(f"\nResults directory: {results_dir}")
    print(f"Output directory: {output_dir}\n")

    # Load REAL data
    print("Loading REAL S-Entropy data...")
    data = load_comparison_data(str(results_dir))

    if not data:
        print("ERROR: No data loaded!")
        return

    # Create visualizations for each platform
    for platform_name, platform_data in data.items():
        print(f"\n{platform_name}:")
        print(f"  Spectra: {platform_data['n_spectra']}")
        print(f"  Droplets: {platform_data['n_droplets']}")

        # 3D plots
        create_sentropy_3d_plot(platform_data, platform_name, output_dir)

        # Distribution plots
        create_sentropy_distributions(platform_data, platform_name, output_dir)

    print("\n" + "="*80)
    print("✓ S-ENTROPY VISUALIZATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
