"""
Phase Diagrams - REAL DATA

Creates polar histograms to visualize phase-lock networks and their angular
distributions in S-Entropy space using ACTUAL experimental data.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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


def compute_phase_angles(coords):
    """
    Compute phase angles in S-Entropy space

    Args:
        coords: Nx3 array of [s_k, s_t, s_e]

    Returns:
        dict with angular distributions
    """
    s_k = coords[:, 0]
    s_t = coords[:, 1]
    s_e = coords[:, 2]

    # Compute angles in different planes
    angles = {}

    # S_k - S_t plane (azimuthal angle)
    angles['kt_plane'] = np.arctan2(s_t, s_k)

    # S_k - S_e plane
    angles['ke_plane'] = np.arctan2(s_e, s_k)

    # S_t - S_e plane
    angles['te_plane'] = np.arctan2(s_e, s_t)

    # 3D spherical angles
    r = np.sqrt(s_k**2 + s_t**2 + s_e**2)
    angles['theta'] = np.arccos(s_e / (r + 1e-10))  # Polar angle
    angles['phi'] = np.arctan2(s_t, s_k)  # Azimuthal angle

    # Phase coherence (angular spread)
    angles['radii'] = r

    return angles


def create_polar_histogram(angles, title, output_dir, filename, n_bins=36):
    """
    Create polar histogram showing angular distribution

    Args:
        angles: Array of angles in radians
        title: Plot title
        output_dir: Output directory
        filename: Output filename
        n_bins: Number of angular bins
    """
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    # Create histogram
    counts, bins = np.histogram(angles, bins=n_bins, range=(-np.pi, np.pi))

    # Compute bin centers
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Plot bars
    width = 2 * np.pi / n_bins
    bars = ax.bar(bin_centers, counts, width=width, alpha=0.7,
                   edgecolor='black', linewidth=1)

    # Color by height
    colors = plt.cm.viridis(counts / counts.max())
    for bar, color in zip(bars, colors):
        bar.set_facecolor(color)

    # Add radial grid
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    # Title
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # Add statistics
    mean_angle = np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())
    std_angle = np.sqrt(-2 * np.log(np.sqrt(np.cos(angles).mean()**2 + np.sin(angles).mean()**2)))

    stats_text = f'Mean: {np.degrees(mean_angle):.1f}°\nStd: {np.degrees(std_angle):.1f}°\nN: {len(angles)}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    output_file = output_dir / filename
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_file.name}")
    return output_file


def create_comprehensive_phase_diagram(data, platform_name, output_dir):
    """
    Create comprehensive phase diagram with multiple polar plots
    """
    s_k = data['s_knowledge']
    s_t = data['s_time']
    s_e = data['s_entropy']

    coords = np.column_stack([s_k, s_t, s_e])

    # Compute phase angles
    angles = compute_phase_angles(coords)

    # Create comprehensive figure
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    fig.suptitle(f'Phase Network Diagrams - {platform_name}\n{data["n_droplets"]} droplets from {data["n_spectra"]} spectra',
                 fontsize=16, fontweight='bold')

    n_bins = 36

    # Panel 1: S_k - S_t plane (azimuthal)
    ax1 = fig.add_subplot(gs[0, 0], projection='polar')
    counts, bins = np.histogram(angles['kt_plane'], bins=n_bins, range=(-np.pi, np.pi))
    bin_centers = (bins[:-1] + bins[1:]) / 2
    width = 2 * np.pi / n_bins
    bars = ax1.bar(bin_centers, counts, width=width, alpha=0.7, edgecolor='black', linewidth=1)
    colors = plt.cm.plasma(counts / counts.max())
    for bar, color in zip(bars, colors):
        bar.set_facecolor(color)
    ax1.set_theta_zero_location('N')
    ax1.set_theta_direction(-1)
    ax1.set_title('S_k - S_t Plane\n(Knowledge-Time)', fontsize=12, fontweight='bold', pad=15)

    # Panel 2: S_k - S_e plane
    ax2 = fig.add_subplot(gs[0, 1], projection='polar')
    counts, bins = np.histogram(angles['ke_plane'], bins=n_bins, range=(-np.pi, np.pi))
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bars = ax2.bar(bin_centers, counts, width=width, alpha=0.7, edgecolor='black', linewidth=1)
    colors = plt.cm.viridis(counts / counts.max())
    for bar, color in zip(bars, colors):
        bar.set_facecolor(color)
    ax2.set_theta_zero_location('N')
    ax2.set_theta_direction(-1)
    ax2.set_title('S_k - S_e Plane\n(Knowledge-Entropy)', fontsize=12, fontweight='bold', pad=15)

    # Panel 3: S_t - S_e plane
    ax3 = fig.add_subplot(gs[0, 2], projection='polar')
    counts, bins = np.histogram(angles['te_plane'], bins=n_bins, range=(-np.pi, np.pi))
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bars = ax3.bar(bin_centers, counts, width=width, alpha=0.7, edgecolor='black', linewidth=1)
    colors = plt.cm.inferno(counts / counts.max())
    for bar, color in zip(bars, colors):
        bar.set_facecolor(color)
    ax3.set_theta_zero_location('N')
    ax3.set_theta_direction(-1)
    ax3.set_title('S_t - S_e Plane\n(Time-Entropy)', fontsize=12, fontweight='bold', pad=15)

    # Panel 4: 3D Polar angle (θ)
    ax4 = fig.add_subplot(gs[1, 0], projection='polar')
    counts, bins = np.histogram(angles['theta'], bins=n_bins, range=(0, np.pi))
    bin_centers = (bins[:-1] + bins[1:]) / 2
    width_theta = np.pi / n_bins
    bars = ax4.bar(bin_centers, counts, width=width_theta, alpha=0.7, edgecolor='black', linewidth=1)
    colors = plt.cm.coolwarm(counts / counts.max())
    for bar, color in zip(bars, colors):
        bar.set_facecolor(color)
    ax4.set_theta_zero_location('N')
    ax4.set_title('3D Polar Angle (θ)\n(from S_e axis)', fontsize=12, fontweight='bold', pad=15)

    # Panel 5: 3D Azimuthal angle (φ)
    ax5 = fig.add_subplot(gs[1, 1], projection='polar')
    counts, bins = np.histogram(angles['phi'], bins=n_bins, range=(-np.pi, np.pi))
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bars = ax5.bar(bin_centers, counts, width=width, alpha=0.7, edgecolor='black', linewidth=1)
    colors = plt.cm.twilight(counts / counts.max())
    for bar, color in zip(bars, colors):
        bar.set_facecolor(color)
    ax5.set_theta_zero_location('N')
    ax5.set_theta_direction(-1)
    ax5.set_title('3D Azimuthal Angle (φ)\n(in S_k-S_t plane)', fontsize=12, fontweight='bold', pad=15)

    # Panel 6: Radial distribution (shown as polar)
    ax6 = fig.add_subplot(gs[1, 2], projection='polar')
    theta_uniform = np.linspace(0, 2*np.pi, len(angles['radii']))
    sc = ax6.scatter(theta_uniform, angles['radii'], c=angles['radii'],
                     cmap='YlOrRd', s=1, alpha=0.3)
    ax6.set_title('Radial Distribution\n(Distance from origin)', fontsize=12, fontweight='bold', pad=15)
    plt.colorbar(sc, ax=ax6, label='Radius', shrink=0.8)

    # Panel 7: Phase coherence by angle
    ax7 = fig.add_subplot(gs[2, :])
    ax7.hexbin(angles['phi'], angles['theta'], gridsize=30, cmap='YlGnBu', mincnt=1)
    ax7.set_xlabel('Azimuthal Angle φ (radians)', fontsize=11, fontweight='bold')
    ax7.set_ylabel('Polar Angle θ (radians)', fontsize=11, fontweight='bold')
    ax7.set_title('Phase Coherence Map\nAngular Density in 3D Spherical Coordinates',
                  fontsize=12, fontweight='bold')
    ax7.grid(alpha=0.3)

    plt.tight_layout()

    output_file = output_dir / f"phase_diagram_comprehensive_{platform_name}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_file.name}")
    return output_file


def main():
    """Main visualization workflow"""
    print("="*80)
    print("PHASE DIAGRAMS - POLAR HISTOGRAMS - REAL DATA")
    print("="*80)

    # Determine paths
    script_dir = Path(__file__).parent
    precursor_root = script_dir.parent.parent
    results_dir = precursor_root / "results" / "fragmentation_comparison"
    output_dir = precursor_root / "visualizations"
    output_dir.mkdir(exist_ok=True)

    # Load REAL data
    print("\nLoading REAL S-Entropy data...")
    data = load_comparison_data(str(results_dir))

    if not data:
        print("ERROR: No data loaded!")
        return

    # Create phase diagrams for each platform
    for platform_name, platform_data in data.items():
        print(f"\n{platform_name}: Creating phase diagrams from {platform_data['n_droplets']} droplets...")

        # Stack coordinates
        coords = np.column_stack([
            platform_data['s_knowledge'],
            platform_data['s_time'],
            platform_data['s_entropy']
        ])

        # Compute phase angles
        angles = compute_phase_angles(coords)

        # Create individual polar histograms
        create_polar_histogram(
            angles['kt_plane'],
            f'{platform_name}\nS_k - S_t Plane Angular Distribution',
            output_dir,
            f"phase_polar_kt_{platform_name}.png"
        )

        create_polar_histogram(
            angles['phi'],
            f'{platform_name}\n3D Azimuthal Angle Distribution',
            output_dir,
            f"phase_polar_azimuthal_{platform_name}.png"
        )

        # Create comprehensive diagram
        create_comprehensive_phase_diagram(platform_data, platform_name, output_dir)

    print("\n" + "="*80)
    print("✓ PHASE DIAGRAM VISUALIZATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
