"""
Validation Charts - REAL DATA

Creates validation plots for theoretical predictions using ACTUAL data.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
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


def create_entropy_intensity_validation(data, platform_name, output_dir):
    """
    Validate intensity-entropy relationship using REAL data

    Note: Since intensity data is not directly in the stage_02 file,
    we'll show S-Entropy distributions and theoretical predictions.
    """
    s_e = data['s_entropy']

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(f'S-Entropy Validation - {platform_name}',
                 fontsize=14, fontweight='bold')

    # S-Entropy distribution
    ax1 = axes[0, 0]
    ax1.hist(s_e, bins=100, color='skyblue', edgecolor='black', alpha=0.7, density=True)
    ax1.set_xlabel('S-Entropy', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Probability Density', fontsize=11, fontweight='bold')
    ax1.set_title(f'S-Entropy Distribution\n{len(s_e)} droplets', fontsize=12)
    ax1.grid(alpha=0.3)
    ax1.axvline(s_e.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean={s_e.mean():.4f}')
    ax1.axvline(np.median(s_e), color='orange', linestyle='--', linewidth=2, label=f'Median={np.median(s_e):.4f}')
    ax1.legend()

    # Log-scale S-Entropy
    ax2 = axes[0, 1]
    s_e_positive = s_e[s_e > 0]
    if len(s_e_positive) > 0:
        ax2.hist(np.log10(s_e_positive), bins=100, color='lightgreen', edgecolor='black', alpha=0.7)
        ax2.set_xlabel('log10(S-Entropy)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax2.set_title(f'Log-Scale S-Entropy\n{len(s_e_positive)} positive values', fontsize=12)
        ax2.grid(alpha=0.3)

    # Theoretical termination probability
    ax3 = axes[1, 0]
    s_e_mean = s_e.mean() if s_e.mean() > 0 else 1e-6
    termination_prob = np.exp(-s_e / s_e_mean)
    ax3.scatter(s_e, termination_prob, s=1, alpha=0.3, c=s_e, cmap='viridis')
    ax3.set_xlabel('S-Entropy', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Termination Probability α', fontsize=11, fontweight='bold')
    ax3.set_title('Theoretical α = exp(-S_e/<S_e>)', fontsize=12)
    ax3.grid(alpha=0.3)

    # S-Entropy quartiles
    ax4 = axes[1, 1]
    quartiles = np.percentile(s_e, [0, 25, 50, 75, 100])
    ax4.bar(['Min', 'Q1', 'Median', 'Q3', 'Max'], quartiles,
            color=['red', 'orange', 'yellow', 'lightgreen', 'green'],
            edgecolor='black', alpha=0.7)
    ax4.set_ylabel('S-Entropy', fontsize=11, fontweight='bold')
    ax4.set_title('S-Entropy Quartiles', fontsize=12)
    ax4.grid(alpha=0.3, axis='y')
    for i, val in enumerate(quartiles):
        ax4.text(i, val, f'{val:.4f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()

    output_file = output_dir / f"validation_entropy_{platform_name}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_file.name}")
    return output_file


def create_platform_comparison(data_dict, output_dir):
    """
    Compare S-Entropy distributions across platforms
    """
    if len(data_dict) < 2:
        print("  ! Skipping comparison (need 2+ platforms)")
        return None

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Platform Comparison - REAL DATA', fontsize=14, fontweight='bold')

    platforms = list(data_dict.keys())
    colors = ['skyblue', 'lightcoral']

    # S-Knowledge comparison
    ax1 = axes[0]
    for i, (platform, data) in enumerate(data_dict.items()):
        ax1.hist(data['s_knowledge'], bins=50, alpha=0.6, color=colors[i],
                 label=platform, edgecolor='black')
    ax1.set_xlabel('S-Knowledge', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax1.set_title('S_k Distribution', fontsize=12)
    ax1.legend()
    ax1.grid(alpha=0.3)

    # S-Time comparison
    ax2 = axes[1]
    for i, (platform, data) in enumerate(data_dict.items()):
        ax2.hist(data['s_time'], bins=50, alpha=0.6, color=colors[i],
                 label=platform, edgecolor='black')
    ax2.set_xlabel('S-Time', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax2.set_title('S_t Distribution', fontsize=12)
    ax2.legend()
    ax2.grid(alpha=0.3)

    # S-Entropy comparison
    ax3 = axes[2]
    for i, (platform, data) in enumerate(data_dict.items()):
        ax3.hist(data['s_entropy'], bins=50, alpha=0.6, color=colors[i],
                 label=platform, edgecolor='black')
    ax3.set_xlabel('S-Entropy', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax3.set_title('S_e Distribution', fontsize=12)
    ax3.legend()
    ax3.grid(alpha=0.3)

    plt.tight_layout()

    output_file = output_dir / "platform_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_file.name}")
    return output_file


def main():
    """Main validation workflow"""
    print("="*80)
    print("VALIDATION CHARTS - REAL DATA")
    print("="*80)

    # Determine paths
    script_dir = Path(__file__).parent
    precursor_root = script_dir.parent.parent
    results_dir = precursor_root / "results" / "fragmentation_comparison"
    output_dir = precursor_root / "visualizations"
    output_dir.mkdir(exist_ok=True)

    # Load REAL data
    print("\nLoading REAL data...")
    data = load_comparison_data(str(results_dir))

    if not data:
        print("ERROR: No data loaded!")
        return

    # Create validation charts for each platform
    for platform_name, platform_data in data.items():
        print(f"\n{platform_name}:")
        create_entropy_intensity_validation(platform_data, platform_name, output_dir)

    # Platform comparison
    if len(data) >= 2:
        print("\nPlatform comparison:")
        create_platform_comparison(data, output_dir)

    print("\n" + "="*80)
    print("✓ VALIDATION CHARTS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
