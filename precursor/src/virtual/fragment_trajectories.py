"""
Fragment Trajectory Analysis - REAL DATA
==========================================

Analyzes fragmentation trajectories in S-Entropy space using ACTUAL experimental data.
Shows how fragments evolve through categorical states during fragmentation.

Based on:
- categorical_analysis.py
- trajectory_analysis.py
- Categorical fragmentation theory
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy.stats import gaussian_kde
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


def plot_3d_trajectory(coords, ax, color='blue', label='Trajectory', alpha=0.7):
    """
    Plot 3D trajectory in S-Entropy space

    Args:
        coords: Nx3 array of [s_k, s_t, s_e]
        ax: 3D matplotlib axis
        color: Line color
        label: Legend label
        alpha: Transparency
    """
    s_k = coords[:, 0]
    s_t = coords[:, 1]
    s_e = coords[:, 2]

    # Plot trajectory line
    ax.plot(s_k, s_t, s_e, color=color, linewidth=2, alpha=alpha, label=label)

    # Mark start point (larger, diamond)
    ax.scatter([s_k[0]], [s_t[0]], [s_e[0]],
               color=color, s=100, marker='D',
               edgecolor='black', linewidth=2, zorder=10)

    # Mark end point (larger, square)
    ax.scatter([s_k[-1]], [s_t[-1]], [s_e[-1]],
               color=color, s=100, marker='s',
               edgecolor='black', linewidth=2, zorder=10)

    # Scatter all points along trajectory
    ax.scatter(s_k, s_t, s_e, c=color, s=10, alpha=0.3)


def plot_2d_projections(coords_by_spectrum, platform_name, output_dir):
    """
    Plot 2D projections of fragment trajectories

    Args:
        coords_by_spectrum: List of Nx3 coordinate arrays
        platform_name: Platform name
        output_dir: Output directory
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Color map
    n_spectra = len(coords_by_spectrum)
    colors = plt.cm.viridis(np.linspace(0, 1, min(n_spectra, 50)))

    # Sample spectra if too many
    sample_indices = np.linspace(0, n_spectra-1, min(n_spectra, 50), dtype=int)

    for idx_pos, spectrum_idx in enumerate(sample_indices):
        coords = coords_by_spectrum[spectrum_idx]

        if len(coords) == 0:
            continue

        s_k = coords[:, 0]
        s_t = coords[:, 1]
        s_e = coords[:, 2]

        color = colors[idx_pos]
        alpha = 0.5

        # Panel 1: S_k vs S_e (Most important for fragmentation)
        axes[0].plot(s_k, s_e, color=color, linewidth=1.5, alpha=alpha)
        axes[0].scatter(s_k[0], s_e[0], color=color, s=50, marker='o',
                       edgecolor='black', linewidth=1, zorder=5, alpha=0.8)
        axes[0].scatter(s_k[-1], s_e[-1], color=color, s=50, marker='s',
                       edgecolor='black', linewidth=1, zorder=5, alpha=0.8)

        # Panel 2: S_t vs S_e
        axes[1].plot(s_t, s_e, color=color, linewidth=1.5, alpha=alpha)
        axes[1].scatter(s_t[0], s_e[0], color=color, s=50, marker='o',
                       edgecolor='black', linewidth=1, zorder=5, alpha=0.8)
        axes[1].scatter(s_t[-1], s_e[-1], color=color, s=50, marker='s',
                       edgecolor='black', linewidth=1, zorder=5, alpha=0.8)

        # Panel 3: S_k vs S_t
        axes[2].plot(s_k, s_t, color=color, linewidth=1.5, alpha=alpha)
        axes[2].scatter(s_k[0], s_t[0], color=color, s=50, marker='o',
                       edgecolor='black', linewidth=1, zorder=5, alpha=0.8)
        axes[2].scatter(s_k[-1], s_t[-1], color=color, s=50, marker='s',
                       edgecolor='black', linewidth=1, zorder=5, alpha=0.8)

    # Labels and styling
    axes[0].set_xlabel('S-Knowledge', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('S-Entropy', fontsize=13, fontweight='bold')
    axes[0].set_title('Knowledge-Entropy Projection\n(Fragmentation Energy)',
                     fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, linestyle='--')

    axes[1].set_xlabel('S-Time', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('S-Entropy', fontsize=13, fontweight='bold')
    axes[1].set_title('Time-Entropy Projection\n(Temporal Evolution)',
                     fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, linestyle='--')

    axes[2].set_xlabel('S-Knowledge', fontsize=13, fontweight='bold')
    axes[2].set_ylabel('S-Time', fontsize=13, fontweight='bold')
    axes[2].set_title('Knowledge-Time Projection\n(Phase Space)',
                     fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3, linestyle='--')

    # Add legend explanation
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='gray', edgecolor='black', label='○ = Precursor (start)'),
        Patch(facecolor='gray', edgecolor='black', label='□ = Fragment (end)'),
        Patch(facecolor='purple', alpha=0.5, label=f'{len(sample_indices)} spectra shown')
    ]
    axes[2].legend(handles=legend_elements, loc='best', fontsize=10)

    fig.suptitle(f'Fragment Trajectories in S-Entropy Space - {platform_name}\n'
                f'{n_spectra} total spectra',
                fontsize=16, fontweight='bold')

    plt.tight_layout()

    output_file = output_dir / f"fragment_trajectories_2d_{platform_name}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_file.name}")
    return output_file


def plot_3d_trajectories(coords_by_spectrum, platform_name, output_dir):
    """
    Plot 3D trajectories in S-Entropy space

    Args:
        coords_by_spectrum: List of Nx3 coordinate arrays
        platform_name: Platform name
        output_dir: Output directory
    """
    fig = plt.figure(figsize=(16, 12))

    # Create 2x2 grid for different views
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Sample spectra
    n_spectra = len(coords_by_spectrum)
    sample_indices = np.linspace(0, n_spectra-1, min(n_spectra, 30), dtype=int)
    colors = plt.cm.plasma(np.linspace(0, 1, len(sample_indices)))

    # View 1: Standard view
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    for idx_pos, spectrum_idx in enumerate(sample_indices):
        coords = coords_by_spectrum[spectrum_idx]
        if len(coords) > 0:
            plot_3d_trajectory(coords, ax1, color=colors[idx_pos],
                             label=None, alpha=0.6)

    ax1.set_xlabel('S-Knowledge', fontsize=11, fontweight='bold')
    ax1.set_ylabel('S-Time', fontsize=11, fontweight='bold')
    ax1.set_zlabel('S-Entropy', fontsize=11, fontweight='bold')
    ax1.set_title('View 1: Standard', fontsize=12, fontweight='bold')
    ax1.view_init(elev=20, azim=45)

    # View 2: Top-down
    ax2 = fig.add_subplot(gs[0, 1], projection='3d')
    for idx_pos, spectrum_idx in enumerate(sample_indices):
        coords = coords_by_spectrum[spectrum_idx]
        if len(coords) > 0:
            plot_3d_trajectory(coords, ax2, color=colors[idx_pos],
                             label=None, alpha=0.6)

    ax2.set_xlabel('S-Knowledge', fontsize=11, fontweight='bold')
    ax2.set_ylabel('S-Time', fontsize=11, fontweight='bold')
    ax2.set_zlabel('S-Entropy', fontsize=11, fontweight='bold')
    ax2.set_title('View 2: Top-Down', fontsize=12, fontweight='bold')
    ax2.view_init(elev=90, azim=0)

    # View 3: Side view
    ax3 = fig.add_subplot(gs[1, 0], projection='3d')
    for idx_pos, spectrum_idx in enumerate(sample_indices):
        coords = coords_by_spectrum[spectrum_idx]
        if len(coords) > 0:
            plot_3d_trajectory(coords, ax3, color=colors[idx_pos],
                             label=None, alpha=0.6)

    ax3.set_xlabel('S-Knowledge', fontsize=11, fontweight='bold')
    ax3.set_ylabel('S-Time', fontsize=11, fontweight='bold')
    ax3.set_zlabel('S-Entropy', fontsize=11, fontweight='bold')
    ax3.set_title('View 3: Side', fontsize=12, fontweight='bold')
    ax3.view_init(elev=0, azim=0)

    # View 4: Front view
    ax4 = fig.add_subplot(gs[1, 1], projection='3d')
    for idx_pos, spectrum_idx in enumerate(sample_indices):
        coords = coords_by_spectrum[spectrum_idx]
        if len(coords) > 0:
            plot_3d_trajectory(coords, ax4, color=colors[idx_pos],
                             label=None, alpha=0.6)

    ax4.set_xlabel('S-Knowledge', fontsize=11, fontweight='bold')
    ax4.set_ylabel('S-Time', fontsize=11, fontweight='bold')
    ax4.set_zlabel('S-Entropy', fontsize=11, fontweight='bold')
    ax4.set_title('View 4: Front', fontsize=12, fontweight='bold')
    ax4.view_init(elev=0, azim=90)

    fig.suptitle(f'3D Fragment Trajectories - {platform_name}\n'
                f'{len(sample_indices)} spectra shown from {n_spectra} total',
                fontsize=16, fontweight='bold')

    output_file = output_dir / f"fragment_trajectories_3d_{platform_name}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_file.name}")
    return output_file


def plot_density_heatmaps(all_coords, platform_name, output_dir):
    """
    Plot density heatmaps of fragment distributions in S-Entropy space

    Args:
        all_coords: Nx3 array of all coordinates
        platform_name: Platform name
        output_dir: Output directory
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    s_k = all_coords[:, 0]
    s_t = all_coords[:, 1]
    s_e = all_coords[:, 2]

    # Panel 1: Knowledge-Entropy density
    try:
        xy = np.vstack([s_k, s_e])
        if len(s_k) > 2:
            z = gaussian_kde(xy)(xy)
            scatter = axes[0, 0].scatter(s_k, s_e, c=z, s=5, cmap='viridis', alpha=0.5)
            plt.colorbar(scatter, ax=axes[0, 0], label='Density')
        else:
            axes[0, 0].scatter(s_k, s_e, s=20, alpha=0.5)
    except:
        axes[0, 0].scatter(s_k, s_e, s=20, alpha=0.5)

    axes[0, 0].set_xlabel('S-Knowledge', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('S-Entropy', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Knowledge-Entropy Density\n(Fragment Energy Distribution)',
                        fontsize=13, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, linestyle='--')

    # Panel 2: Time-Entropy density
    try:
        xy = np.vstack([s_t, s_e])
        if len(s_t) > 2:
            z = gaussian_kde(xy)(xy)
            scatter = axes[0, 1].scatter(s_t, s_e, c=z, s=5, cmap='plasma', alpha=0.5)
            plt.colorbar(scatter, ax=axes[0, 1], label='Density')
        else:
            axes[0, 1].scatter(s_t, s_e, s=20, alpha=0.5)
    except:
        axes[0, 1].scatter(s_t, s_e, s=20, alpha=0.5)

    axes[0, 1].set_xlabel('S-Time', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('S-Entropy', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Time-Entropy Density\n(Temporal Evolution)',
                        fontsize=13, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, linestyle='--')

    # Panel 3: Knowledge-Time density
    try:
        xy = np.vstack([s_k, s_t])
        if len(s_k) > 2:
            z = gaussian_kde(xy)(xy)
            scatter = axes[1, 0].scatter(s_k, s_t, c=z, s=5, cmap='inferno', alpha=0.5)
            plt.colorbar(scatter, ax=axes[1, 0], label='Density')
        else:
            axes[1, 0].scatter(s_k, s_t, s=20, alpha=0.5)
    except:
        axes[1, 0].scatter(s_k, s_t, s=20, alpha=0.5)

    axes[1, 0].set_xlabel('S-Knowledge', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('S-Time', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Knowledge-Time Density\n(Phase Space)',
                        fontsize=13, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, linestyle='--')

    # Panel 4: 2D histogram of S_k vs S_e (most important)
    h = axes[1, 1].hist2d(s_k, s_e, bins=50, cmap='YlOrRd', cmin=1)
    plt.colorbar(h[3], ax=axes[1, 1], label='Count')

    axes[1, 1].set_xlabel('S-Knowledge', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('S-Entropy', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Knowledge-Entropy Histogram\n(Fragmentation Landscape)',
                        fontsize=13, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, linestyle='--')

    fig.suptitle(f'Fragment Density Analysis - {platform_name}\n'
                f'{len(all_coords)} total fragments',
                fontsize=16, fontweight='bold')

    plt.tight_layout()

    output_file = output_dir / f"fragment_density_{platform_name}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_file.name}")
    return output_file


def plot_intensity_entropy_relationship(coords_by_spectrum, platform_name, output_dir):
    """
    Plot intensity as termination probability: I ∝ exp(-|E|/⟨E⟩)

    Args:
        coords_by_spectrum: List of Nx3 coordinate arrays
        platform_name: Platform name
        output_dir: Output directory
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Compute edge density and intensity for each fragment
    edge_densities = []
    entropies = []

    for coords in coords_by_spectrum:
        if len(coords) == 0:
            continue

        # S-entropy is the third column
        s_e = coords[:, 2]

        # Edge density approximation: number of peaks
        edge_density = len(coords)

        # Mean entropy for this spectrum
        mean_entropy = np.mean(s_e)

        edge_densities.append(edge_density)
        entropies.append(mean_entropy)

    edge_densities = np.array(edge_densities)
    entropies = np.array(entropies)

    # Panel 1: Edge density vs entropy
    axes[0].scatter(edge_densities, entropies, s=30, alpha=0.5,
                   c=entropies, cmap='viridis', edgecolor='black', linewidth=0.5)

    # Fit trend
    if len(edge_densities) > 2:
        z = np.polyfit(edge_densities, entropies, 1)
        p = np.poly1d(z)
        x_fit = np.linspace(edge_densities.min(), edge_densities.max(), 100)
        axes[0].plot(x_fit, p(x_fit), 'r--', linewidth=3,
                    label=f'Trend: S_e = {z[0]:.4f}|E| + {z[1]:.3f}')
        axes[0].legend(fontsize=11)

    axes[0].set_xlabel('Fragment Count (|E|)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Mean S-Entropy', fontsize=12, fontweight='bold')
    axes[0].set_title('Entropy vs Fragment Count\n(Network Complexity)',
                     fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3, linestyle='--')

    # Panel 2: Log-linear relationship (intensity as termination probability)
    # Use 1/edge_density as proxy for intensity (fewer fragments = higher intensity)
    pseudo_intensity = 1.0 / (edge_densities + 1)

    axes[1].semilogy(entropies, pseudo_intensity, 'o', markersize=8, alpha=0.6,
                    color='darkblue', markeredgecolor='black', markeredgewidth=0.5)

    # Fit exponential
    if len(entropies) > 2:
        try:
            from scipy.optimize import curve_fit
            def exp_model(x, a, b):
                return a * np.exp(-b * x)

            # Filter out zeros
            mask = pseudo_intensity > 0
            if np.sum(mask) > 2:
                popt, _ = curve_fit(exp_model, entropies[mask], pseudo_intensity[mask])
                x_fit = np.linspace(entropies.min(), entropies.max(), 100)
                y_fit = exp_model(x_fit, *popt)
                axes[1].plot(x_fit, y_fit, 'r-', linewidth=3,
                           label=f'I = {popt[0]:.2e} exp(-{popt[1]:.3f}|E|)')
                axes[1].legend(fontsize=11)
        except:
            pass

    axes[1].set_xlabel('Mean S-Entropy (|E|)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Pseudo-Intensity (1/fragments)', fontsize=12, fontweight='bold')
    axes[1].set_title('Intensity as Termination Probability\nI ∝ exp(-|E|/⟨E⟩)',
                     fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3, linestyle='--')

    fig.suptitle(f'Intensity-Entropy Relationship - {platform_name}\n'
                f'Categorical Fragmentation Theory Validation',
                fontsize=16, fontweight='bold')

    plt.tight_layout()

    output_file = output_dir / f"intensity_entropy_{platform_name}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_file.name}")
    return output_file


def main():
    """Main visualization workflow"""
    print("="*80)
    print("FRAGMENT TRAJECTORY ANALYSIS - REAL DATA")
    print("="*80)

    # Determine paths
    script_dir = Path(__file__).parent
    precursor_root = script_dir.parent.parent
    results_dir = precursor_root / "results" / "fragmentation_comparison"
    output_dir = precursor_root / "visualizations"
    output_dir.mkdir(exist_ok=True)

    # Load REAL data
    print("\nLoading REAL fragment data...")
    data = load_comparison_data(str(results_dir))

    if not data:
        print("ERROR: No data loaded!")
        return

    # Analyze each platform
    for platform_name, platform_data in data.items():
        print(f"\n{platform_name}: Analyzing {platform_data['n_spectra']} spectra...")

        coords_by_spectrum = platform_data['coords_by_spectrum']

        # Flatten all coordinates for density analysis
        all_coords = np.vstack([coords for coords in coords_by_spectrum if len(coords) > 0])

        print(f"  Total fragments: {len(all_coords)}")
        print(f"  Avg fragments/spectrum: {len(all_coords) / platform_data['n_spectra']:.1f}")

        # Generate visualizations
        print(f"  Creating 2D projections...")
        plot_2d_projections(coords_by_spectrum, platform_name, output_dir)

        print(f"  Creating 3D trajectories...")
        plot_3d_trajectories(coords_by_spectrum, platform_name, output_dir)

        print(f"  Creating density heatmaps...")
        plot_density_heatmaps(all_coords, platform_name, output_dir)

        print(f"  Creating intensity-entropy analysis...")
        plot_intensity_entropy_relationship(coords_by_spectrum, platform_name, output_dir)

    print("\n" + "="*80)
    print("✓ FRAGMENT TRAJECTORY ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
