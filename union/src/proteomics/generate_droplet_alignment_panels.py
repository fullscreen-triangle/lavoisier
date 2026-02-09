"""
Droplet Alignment Panel Generator
==================================

Generates publication-quality panels showing:
1. 3D droplet wave patterns for precursor and fragments
2. Alignment/segment matching visualization
3. Matching scores

Creates panels for both proteomics and metabolomics.

Author: Kundai Sachikonye
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize, LightSource
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
from scipy.ndimage import gaussian_filter

# Set publication style
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
})


# ============================================================================
# 3D DROPLET WAVE GENERATION
# ============================================================================

def generate_3d_droplet_wave(
    size: int = 100,
    amplitude: float = 1.0,
    wavelength: float = 15.0,
    decay_rate: float = 0.03,
    center_offset: Tuple[float, float] = (0, 0),
    phase: float = 0.0,
    surface_tension: float = 0.05,
    velocity: float = 3.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a 3D droplet impact wave pattern.

    Returns X, Y, Z meshgrid arrays for 3D plotting.

    Physics-based parameters:
    - amplitude: Related to droplet kinetic energy
    - wavelength: Related to surface tension (higher tension = longer wavelength)
    - decay_rate: Related to viscosity
    - velocity: Impact velocity affects amplitude
    """
    x = np.linspace(-size/2, size/2, size)
    y = np.linspace(-size/2, size/2, size)
    X, Y = np.meshgrid(x, y)

    # Apply center offset
    X_shifted = X - center_offset[0]
    Y_shifted = Y - center_offset[1]

    # Radial distance from impact center
    R = np.sqrt(X_shifted**2 + Y_shifted**2)

    # Wave pattern based on droplet physics
    # Amplitude modulated by velocity and surface tension
    amp_factor = amplitude * (velocity / 3.0) * (0.05 / surface_tension)

    # Capillary wave pattern
    Z = amp_factor * np.sin(2 * np.pi * R / wavelength + phase) * np.exp(-decay_rate * R)

    # Add central splash region
    central_splash = amp_factor * 1.5 * np.exp(-(R**2) / (wavelength**2 / 4))
    Z += central_splash

    # Apply Gaussian smoothing for realism
    Z = gaussian_filter(Z, sigma=1.5)

    return X, Y, Z


def generate_fragment_droplet(
    parent_params: Dict,
    fragment_ratio: float = 0.7,
    position_offset: Tuple[float, float] = (0, 0)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a fragment droplet based on parent parameters.

    Fragments have:
    - Smaller amplitude (less energy)
    - Slightly different wavelength (different mass)
    - Consistent phase relationship (conservation laws)
    """
    # Fragment parameters derived from parent
    frag_amplitude = parent_params['amplitude'] * fragment_ratio
    frag_wavelength = parent_params['wavelength'] * (0.8 + 0.4 * fragment_ratio)
    frag_velocity = parent_params['velocity'] * np.sqrt(fragment_ratio)

    return generate_3d_droplet_wave(
        size=parent_params['size'],
        amplitude=frag_amplitude,
        wavelength=frag_wavelength,
        decay_rate=parent_params['decay_rate'] * 1.2,
        center_offset=position_offset,
        phase=parent_params['phase'] + np.pi * 0.3,  # Phase shift from fragmentation
        surface_tension=parent_params['surface_tension'],
        velocity=frag_velocity
    )


# ============================================================================
# ALIGNMENT VISUALIZATION
# ============================================================================

def compute_alignment_score(Z1: np.ndarray, Z2: np.ndarray) -> Dict:
    """
    Compute alignment metrics between two droplet patterns.

    Returns multiple scores:
    - Spatial overlap: Correlation of wave patterns
    - Wavelength match: Similarity of frequency content
    - Energy ratio: Ratio of wave energies
    - Phase coherence: Consistency of phase relationship
    """
    # Normalize
    Z1_norm = (Z1 - Z1.mean()) / (Z1.std() + 1e-10)
    Z2_norm = (Z2 - Z2.mean()) / (Z2.std() + 1e-10)

    # Spatial overlap (correlation)
    correlation = np.corrcoef(Z1_norm.flatten(), Z2_norm.flatten())[0, 1]
    spatial_overlap = (correlation + 1) / 2  # Map to 0-1

    # Energy ratio
    energy1 = np.sum(Z1**2)
    energy2 = np.sum(Z2**2)
    energy_ratio = min(energy1, energy2) / (max(energy1, energy2) + 1e-10)

    # Wavelength match via FFT
    fft1 = np.abs(np.fft.fft2(Z1_norm))
    fft2 = np.abs(np.fft.fft2(Z2_norm))
    fft_corr = np.corrcoef(fft1.flatten(), fft2.flatten())[0, 1]
    wavelength_match = (fft_corr + 1) / 2

    # Phase coherence
    phase1 = np.angle(np.fft.fft2(Z1_norm))
    phase2 = np.angle(np.fft.fft2(Z2_norm))
    phase_diff = np.abs(phase1 - phase2)
    phase_coherence = 1 - np.mean(phase_diff) / np.pi

    # Combined score
    combined = 0.3 * spatial_overlap + 0.25 * wavelength_match + 0.25 * energy_ratio + 0.2 * phase_coherence

    return {
        'spatial_overlap': float(spatial_overlap),
        'wavelength_match': float(wavelength_match),
        'energy_ratio': float(energy_ratio),
        'phase_coherence': float(phase_coherence),
        'combined_score': float(combined)
    }


def generate_alignment_heatmap(Z1: np.ndarray, Z2: np.ndarray) -> np.ndarray:
    """Generate a heatmap showing local alignment quality."""
    window_size = 20
    stride = 10

    h, w = Z1.shape
    n_rows = (h - window_size) // stride + 1
    n_cols = (w - window_size) // stride + 1

    alignment_map = np.zeros((n_rows, n_cols))

    for i in range(n_rows):
        for j in range(n_cols):
            y_start = i * stride
            x_start = j * stride

            patch1 = Z1[y_start:y_start+window_size, x_start:x_start+window_size]
            patch2 = Z2[y_start:y_start+window_size, x_start:x_start+window_size]

            # Local correlation
            if patch1.std() > 0 and patch2.std() > 0:
                corr = np.corrcoef(patch1.flatten(), patch2.flatten())[0, 1]
                alignment_map[i, j] = (corr + 1) / 2
            else:
                alignment_map[i, j] = 0.5

    return alignment_map


# ============================================================================
# PROTEOMICS PANEL
# ============================================================================

def generate_proteomics_panel(output_path: Path):
    """
    Generate proteomics droplet alignment panel.

    Shows peptide precursor and b/y-ion fragments with alignment scores.
    """
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle('Proteomics: Peptide Fragmentation Droplet Alignment\n(Peptide: EAIPR → Fragments: b₄=EAIP, y₃=IPR)',
                 fontsize=14, fontweight='bold')

    # Parent peptide parameters
    parent_params = {
        'size': 100,
        'amplitude': 1.0,
        'wavelength': 18.0,
        'decay_rate': 0.025,
        'phase': 0.0,
        'surface_tension': 0.05,
        'velocity': 3.5
    }

    # Generate parent droplet
    X_parent, Y_parent, Z_parent = generate_3d_droplet_wave(**parent_params)

    # Generate fragment droplets
    X_b4, Y_b4, Z_b4 = generate_fragment_droplet(parent_params, fragment_ratio=0.75)
    X_y3, Y_y3, Z_y3 = generate_fragment_droplet(parent_params, fragment_ratio=0.55,
                                                   position_offset=(5, 3))

    # Color maps
    ls = LightSource(azdeg=315, altdeg=45)

    # Panel A: Parent Precursor (3D)
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')

    # Create shaded surface
    rgb = ls.shade(Z_parent, cmap=cm.viridis, vert_exag=0.5, blend_mode='soft')
    surf1 = ax1.plot_surface(X_parent, Y_parent, Z_parent, facecolors=rgb,
                             linewidth=0, antialiased=True, alpha=0.9)

    ax1.set_xlabel('X (μm)')
    ax1.set_ylabel('Y (μm)')
    ax1.set_zlabel('Wave Amplitude')
    ax1.set_title('Panel A: Precursor Droplet [M+H]⁺\n(m/z 584.35, EAIPR)', fontweight='bold')
    ax1.view_init(elev=25, azim=45)

    # Add physics annotations
    ax1.text2D(0.02, 0.95,
               f'v = {parent_params["velocity"]:.1f} m/s\n'
               f'σ = {parent_params["surface_tension"]:.3f} N/m\n'
               f'λ = {parent_params["wavelength"]:.1f} μm',
               transform=ax1.transAxes, fontsize=9,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               verticalalignment='top')

    # Panel B: Fragment Droplets (3D overlay)
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')

    # Plot b4 fragment
    rgb_b4 = ls.shade(Z_b4, cmap=cm.Blues, vert_exag=0.5, blend_mode='soft')
    ax2.plot_surface(X_b4, Y_b4, Z_b4, facecolors=rgb_b4,
                    linewidth=0, antialiased=True, alpha=0.7)

    # Plot y3 fragment (offset in z for visibility)
    Z_y3_offset = Z_y3 - 0.3
    rgb_y3 = ls.shade(Z_y3, cmap=cm.Reds, vert_exag=0.5, blend_mode='soft')
    ax2.plot_surface(X_y3, Y_y3, Z_y3_offset, facecolors=rgb_y3,
                    linewidth=0, antialiased=True, alpha=0.7)

    ax2.set_xlabel('X (μm)')
    ax2.set_ylabel('Y (μm)')
    ax2.set_zlabel('Wave Amplitude')
    ax2.set_title('Panel B: Fragment Droplets\n(b₄: EAIP, y₃: IPR)', fontweight='bold')
    ax2.view_init(elev=25, azim=45)

    # Legend
    blue_patch = mpatches.Patch(color='blue', alpha=0.7, label='b₄-ion (m/z 427.23)')
    red_patch = mpatches.Patch(color='red', alpha=0.7, label='y₃-ion (m/z 385.23)')
    ax2.legend(handles=[blue_patch, red_patch], loc='upper right')

    # Panel C: Alignment Heatmap
    ax3 = fig.add_subplot(2, 2, 3)

    # Compute alignment between parent and b4
    alignment_map = generate_alignment_heatmap(Z_parent, Z_b4)

    im = ax3.imshow(alignment_map, cmap='RdYlGn', vmin=0, vmax=1,
                   extent=[-50, 50, -50, 50], origin='lower')

    # Add contours of parent wave
    contours = ax3.contour(X_parent, Y_parent, Z_parent, levels=5,
                          colors='black', alpha=0.5, linewidths=0.5)

    ax3.set_xlabel('X (μm)')
    ax3.set_ylabel('Y (μm)')
    ax3.set_title('Panel C: Spatial Alignment Map\n(Parent-Fragment Correlation)', fontweight='bold')

    cbar = plt.colorbar(im, ax=ax3, shrink=0.8)
    cbar.set_label('Local Alignment Score', fontsize=9)

    # Add segment annotations
    ax3.axhline(y=0, color='white', linestyle='--', alpha=0.5)
    ax3.axvline(x=0, color='white', linestyle='--', alpha=0.5)
    ax3.text(25, 25, 'Q1', fontsize=12, fontweight='bold', color='white', ha='center')
    ax3.text(-25, 25, 'Q2', fontsize=12, fontweight='bold', color='white', ha='center')
    ax3.text(-25, -25, 'Q3', fontsize=12, fontweight='bold', color='white', ha='center')
    ax3.text(25, -25, 'Q4', fontsize=12, fontweight='bold', color='white', ha='center')

    # Panel D: Alignment Scores
    ax4 = fig.add_subplot(2, 2, 4)

    # Compute scores
    scores_b4 = compute_alignment_score(Z_parent, Z_b4)
    scores_y3 = compute_alignment_score(Z_parent, Z_y3)

    categories = ['Spatial\nOverlap', 'Wavelength\nMatch', 'Energy\nRatio',
                 'Phase\nCoherence', 'Combined\nScore']

    b4_values = [scores_b4['spatial_overlap'], scores_b4['wavelength_match'],
                scores_b4['energy_ratio'], scores_b4['phase_coherence'],
                scores_b4['combined_score']]

    y3_values = [scores_y3['spatial_overlap'], scores_y3['wavelength_match'],
                scores_y3['energy_ratio'], scores_y3['phase_coherence'],
                scores_y3['combined_score']]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax4.bar(x - width/2, b4_values, width, label='b₄-ion (EAIP)',
                   color='#3498db', edgecolor='black')
    bars2 = ax4.bar(x + width/2, y3_values, width, label='y₃-ion (IPR)',
                   color='#e74c3c', edgecolor='black')

    ax4.set_ylabel('Score')
    ax4.set_title('Panel D: Fragment-Parent Alignment Scores', fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(categories)
    ax4.legend(loc='upper right')
    ax4.set_ylim(0, 1.1)

    # Add value labels
    for bar, val in zip(bars1, b4_values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', fontsize=8)
    for bar, val in zip(bars2, y3_values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', fontsize=8)

    # Add horizontal threshold line
    ax4.axhline(y=0.6, color='green', linestyle='--', alpha=0.5, label='Threshold')

    # Summary annotation
    ax4.annotate(f'b₄ Combined: {scores_b4["combined_score"]:.3f}\n'
                f'y₃ Combined: {scores_y3["combined_score"]:.3f}\n'
                f'Mean: {(scores_b4["combined_score"]+scores_y3["combined_score"])/2:.3f}',
                xy=(4.5, 0.3), fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  Saved: {output_path}")
    return output_path, {'b4_scores': scores_b4, 'y3_scores': scores_y3}


# ============================================================================
# METABOLOMICS PANEL
# ============================================================================

def generate_metabolomics_panel(output_path: Path):
    """
    Generate metabolomics droplet alignment panel.

    Shows metabolite precursor and fragment ions with alignment scores.
    Example: Phosphatidylcholine fragmentation
    """
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle('Metabolomics: Lipid Fragmentation Droplet Alignment\n'
                 '(PC 34:1 → Fragments: Headgroup [184], Fatty Acid [281])',
                 fontsize=14, fontweight='bold')

    # Parent metabolite parameters (larger molecule = different wave characteristics)
    parent_params = {
        'size': 100,
        'amplitude': 1.2,
        'wavelength': 22.0,  # Longer wavelength for larger molecule
        'decay_rate': 0.020,
        'phase': 0.0,
        'surface_tension': 0.06,
        'velocity': 2.8
    }

    # Generate parent droplet
    X_parent, Y_parent, Z_parent = generate_3d_droplet_wave(**parent_params)

    # Generate fragment droplets
    # Headgroup (smaller, characteristic fragment)
    X_head, Y_head, Z_head = generate_fragment_droplet(parent_params, fragment_ratio=0.25,
                                                        position_offset=(3, -2))
    # Fatty acid (larger fragment)
    X_fa, Y_fa, Z_fa = generate_fragment_droplet(parent_params, fragment_ratio=0.45,
                                                   position_offset=(-2, 4))

    # Color maps
    ls = LightSource(azdeg=315, altdeg=45)

    # Panel A: Parent Precursor (3D)
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')

    rgb = ls.shade(Z_parent, cmap=cm.plasma, vert_exag=0.5, blend_mode='soft')
    ax1.plot_surface(X_parent, Y_parent, Z_parent, facecolors=rgb,
                    linewidth=0, antialiased=True, alpha=0.9)

    ax1.set_xlabel('X (μm)')
    ax1.set_ylabel('Y (μm)')
    ax1.set_zlabel('Wave Amplitude')
    ax1.set_title('Panel A: Precursor Droplet [M+H]⁺\n(m/z 760.58, PC 34:1)', fontweight='bold')
    ax1.view_init(elev=30, azim=60)

    ax1.text2D(0.02, 0.95,
               f'v = {parent_params["velocity"]:.1f} m/s\n'
               f'σ = {parent_params["surface_tension"]:.3f} N/m\n'
               f'λ = {parent_params["wavelength"]:.1f} μm',
               transform=ax1.transAxes, fontsize=9,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               verticalalignment='top')

    # Panel B: Fragment Droplets (3D)
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')

    # Headgroup fragment (phosphocholine)
    rgb_head = ls.shade(Z_head, cmap=cm.Greens, vert_exag=0.5, blend_mode='soft')
    ax2.plot_surface(X_head, Y_head, Z_head, facecolors=rgb_head,
                    linewidth=0, antialiased=True, alpha=0.8)

    # Fatty acid fragment
    Z_fa_offset = Z_fa - 0.25
    rgb_fa = ls.shade(Z_fa, cmap=cm.Oranges, vert_exag=0.5, blend_mode='soft')
    ax2.plot_surface(X_fa, Y_fa, Z_fa_offset, facecolors=rgb_fa,
                    linewidth=0, antialiased=True, alpha=0.8)

    ax2.set_xlabel('X (μm)')
    ax2.set_ylabel('Y (μm)')
    ax2.set_zlabel('Wave Amplitude')
    ax2.set_title('Panel B: Fragment Droplets\n(Headgroup + Fatty Acid)', fontweight='bold')
    ax2.view_init(elev=30, azim=60)

    green_patch = mpatches.Patch(color='green', alpha=0.7, label='Phosphocholine (m/z 184.07)')
    orange_patch = mpatches.Patch(color='orange', alpha=0.7, label='Oleic acid (m/z 281.25)')
    ax2.legend(handles=[green_patch, orange_patch], loc='upper right')

    # Panel C: Segment Matching Visualization
    ax3 = fig.add_subplot(2, 2, 3)

    # Create a composite visualization showing wave profile matching
    # Radial profile comparison
    center = 50
    radii = np.arange(0, 45, 1)

    # Extract radial profiles
    profile_parent = []
    profile_head = []
    profile_fa = []

    for r in radii:
        # Create circular mask
        y, x = np.ogrid[:100, :100]
        mask = ((x - center)**2 + (y - center)**2 >= r**2) & \
               ((x - center)**2 + (y - center)**2 < (r+1)**2)

        if np.any(mask):
            profile_parent.append(np.mean(Z_parent[mask]))
            profile_head.append(np.mean(Z_head[mask]))
            profile_fa.append(np.mean(Z_fa[mask]))
        else:
            profile_parent.append(0)
            profile_head.append(0)
            profile_fa.append(0)

    profile_parent = np.array(profile_parent)
    profile_head = np.array(profile_head)
    profile_fa = np.array(profile_fa)

    # Normalize
    profile_parent = (profile_parent - profile_parent.min()) / (profile_parent.max() - profile_parent.min() + 1e-10)
    profile_head = (profile_head - profile_head.min()) / (profile_head.max() - profile_head.min() + 1e-10)
    profile_fa = (profile_fa - profile_fa.min()) / (profile_fa.max() - profile_fa.min() + 1e-10)

    ax3.plot(radii, profile_parent, 'purple', linewidth=2.5, label='Parent (PC 34:1)')
    ax3.plot(radii, profile_head, 'green', linewidth=2, linestyle='--', label='Headgroup [184]')
    ax3.plot(radii, profile_fa, 'orange', linewidth=2, linestyle='-.', label='Fatty Acid [281]')

    # Highlight matching regions
    ax3.fill_between(radii, profile_parent, profile_head,
                     where=np.abs(profile_parent - profile_head) < 0.3,
                     alpha=0.3, color='green', label='Matching segments')

    ax3.set_xlabel('Radial Distance from Impact Center (μm)')
    ax3.set_ylabel('Normalized Wave Amplitude')
    ax3.set_title('Panel C: Radial Wave Profile Matching', fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.set_xlim(0, 45)
    ax3.set_ylim(-0.1, 1.1)

    # Add segment annotations
    ax3.axvspan(5, 15, alpha=0.1, color='blue', label='Core region')
    ax3.axvspan(15, 30, alpha=0.1, color='yellow', label='Wave region')
    ax3.axvspan(30, 45, alpha=0.1, color='gray', label='Decay region')

    ax3.annotate('Core', xy=(10, 0.95), fontsize=10, ha='center')
    ax3.annotate('Wave Propagation', xy=(22, 0.95), fontsize=10, ha='center')
    ax3.annotate('Decay', xy=(37, 0.95), fontsize=10, ha='center')

    # Panel D: Comprehensive Scores
    ax4 = fig.add_subplot(2, 2, 4)

    # Compute scores
    scores_head = compute_alignment_score(Z_parent, Z_head)
    scores_fa = compute_alignment_score(Z_parent, Z_fa)

    # Create radar-like bar chart
    categories = ['Spatial', 'Wavelength', 'Energy', 'Phase', 'Combined']

    head_values = [scores_head['spatial_overlap'], scores_head['wavelength_match'],
                  scores_head['energy_ratio'], scores_head['phase_coherence'],
                  scores_head['combined_score']]

    fa_values = [scores_fa['spatial_overlap'], scores_fa['wavelength_match'],
                scores_fa['energy_ratio'], scores_fa['phase_coherence'],
                scores_fa['combined_score']]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax4.bar(x - width/2, head_values, width, label='Phosphocholine [184]',
                   color='#27ae60', edgecolor='black')
    bars2 = ax4.bar(x + width/2, fa_values, width, label='Oleic Acid [281]',
                   color='#e67e22', edgecolor='black')

    ax4.set_ylabel('Score')
    ax4.set_title('Panel D: Fragment-Parent Alignment Scores', fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(categories)
    ax4.legend(loc='upper right')
    ax4.set_ylim(0, 1.1)

    # Add value labels
    for bar, val in zip(bars1, head_values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', fontsize=8)
    for bar, val in zip(bars2, fa_values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', fontsize=8)

    # Threshold line
    ax4.axhline(y=0.6, color='green', linestyle='--', alpha=0.5)

    # Summary
    ax4.annotate(f'Headgroup: {scores_head["combined_score"]:.3f}\n'
                f'Fatty Acid: {scores_fa["combined_score"]:.3f}\n'
                f'Overall: {(scores_head["combined_score"]+scores_fa["combined_score"])/2:.3f}',
                xy=(4.5, 0.25), fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  Saved: {output_path}")
    return output_path, {'headgroup_scores': scores_head, 'fatty_acid_scores': scores_fa}


# ============================================================================
# MAIN
# ============================================================================

def generate_all_alignment_panels(output_dir: Path = None) -> Dict:
    """Generate both proteomics and metabolomics alignment panels."""
    print("\n" + "=" * 70)
    print("DROPLET ALIGNMENT PANEL GENERATION")
    print("=" * 70)

    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / 'publication'
    else:
        output_dir = Path(output_dir)

    # Create output directories
    proteomics_dir = output_dir / 'proteomics' / 'figures'
    metabolomics_dir = output_dir / 'mass-computing' / 'figures'

    proteomics_dir.mkdir(parents=True, exist_ok=True)
    metabolomics_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    print("\nGenerating Proteomics Alignment Panel...")
    prot_path, prot_scores = generate_proteomics_panel(
        proteomics_dir / 'droplet_alignment_proteomics.png'
    )
    results['proteomics'] = {'path': str(prot_path), 'scores': prot_scores}

    print("\nGenerating Metabolomics Alignment Panel...")
    metab_path, metab_scores = generate_metabolomics_panel(
        metabolomics_dir / 'droplet_alignment_metabolomics.png'
    )
    results['metabolomics'] = {'path': str(metab_path), 'scores': metab_scores}

    # Save summary
    summary = {
        'proteomics': {
            'peptide': 'EAIPR',
            'fragments': ['b4 (EAIP)', 'y3 (IPR)'],
            'b4_combined_score': prot_scores['b4_scores']['combined_score'],
            'y3_combined_score': prot_scores['y3_scores']['combined_score'],
        },
        'metabolomics': {
            'precursor': 'PC 34:1 (m/z 760.58)',
            'fragments': ['Phosphocholine (m/z 184)', 'Oleic acid (m/z 281)'],
            'headgroup_combined_score': metab_scores['headgroup_scores']['combined_score'],
            'fatty_acid_combined_score': metab_scores['fatty_acid_scores']['combined_score'],
        }
    }

    summary_path = output_dir / 'droplet_alignment_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved summary: {summary_path}")

    print("\n" + "=" * 70)
    print("PANEL GENERATION COMPLETE")
    print("=" * 70)

    return results


if __name__ == "__main__":
    generate_all_alignment_panels()
