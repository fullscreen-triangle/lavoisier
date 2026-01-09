"""
Virtual Gas Ensemble Panel: Demonstrating Molecular Identity Through Measurement
================================================================================

This visualization demonstrates the core insight: by observing one molecule
through sliding windows, it becomes every other molecule in the ensemble.

Each panel shows 3 molecules:
- ONE with full radar chart (the observed molecule)
- TWO showing relational effects (Van der Waals, dipole, etc.)

Four panels show the sliding window evolution: the observed molecule
shifts identity, becoming each of the others.
"""

import sys
import os
import json
import time
import math
from pathlib import Path
from typing import Dict, List, Tuple, Any

sys.path.insert(0, os.path.dirname(__file__))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Wedge, Arc
from matplotlib.collections import LineCollection, PatchCollection
import matplotlib.gridspec as gridspec
import numpy as np

from virtual_molecule import VirtualMolecule, SCoordinate
from virtual_chamber import VirtualChamber


# Color scheme
COLORS = {
    'observed': '#2E86AB',      # Blue - currently observed
    'related1': '#A23B72',      # Magenta - related molecule 1
    'related2': '#F18F01',      # Orange - related molecule 2
    'vdw': '#6C757D',           # Gray - Van der Waals
    'dipole': '#3A7D44',        # Green - dipole
    'vibration': '#C73E1D',     # Red - vibrational coupling
    'bg': '#F8F9FA',
    'dark': '#212529',
    'grid': '#DEE2E6',
}

# Hardware parameters for radar chart
HARDWARE_PARAMS = [
    'CPU\nCycle',
    'Memory\nBus', 
    'Power\nSupply',
    'I/O\nLatency',
    'Cache\nTiming',
    'Network\nJitter'
]


def create_molecule_radar(ax, molecule: VirtualMolecule, color: str, 
                         is_observed: bool = False, alpha: float = 1.0,
                         label: str = None):
    """
    Create a radar chart representation of a molecule.
    
    For the observed molecule: full radar with all hardware parameters
    For related molecules: simplified outline showing relational position
    """
    n_params = len(HARDWARE_PARAMS)
    angles = np.linspace(0, 2 * np.pi, n_params, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon
    
    # Generate values from molecule properties
    # Map S-coordinates and frequency to hardware-like parameters
    base_values = [
        molecule.s_coord.S_k,                    # CPU cycle (knowledge)
        molecule.s_coord.S_t,                    # Memory bus (temporal)
        molecule.s_coord.S_e,                    # Power supply (evolution)
        molecule.phase / (2 * np.pi) if molecule.phase else 0.5,  # I/O latency
        molecule.amplitude if molecule.amplitude else 0.5,         # Cache timing
        (molecule.frequency % 1e9) / 1e9 if molecule.frequency else 0.5,  # Network jitter
    ]
    
    # Ensure values are in [0.1, 1] for visibility
    values = [max(0.1, min(1.0, v)) for v in base_values]
    values += values[:1]  # Close
    
    if is_observed:
        # Full radar chart for observed molecule
        ax.fill(angles, values, color=color, alpha=0.3)
        ax.plot(angles, values, 'o-', color=color, linewidth=2, markersize=6)
        
        # Add parameter labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(HARDWARE_PARAMS, fontsize=7)
        
        # Add value annotations
        for i, (angle, value) in enumerate(zip(angles[:-1], values[:-1])):
            ax.annotate(f'{value:.2f}', xy=(angle, value), 
                       fontsize=6, ha='center', va='bottom',
                       color=color)
    else:
        # Simplified outline for related molecules
        ax.plot(angles, values, '-', color=color, linewidth=1.5, alpha=alpha)
        ax.fill(angles, values, color=color, alpha=0.1 * alpha)
    
    # Set radar properties
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['', '', '', ''], fontsize=6)
    ax.grid(True, color=COLORS['grid'], linestyle='-', linewidth=0.5)
    
    if label:
        ax.set_title(label, fontsize=9, fontweight='bold', color=color, pad=10)


def draw_interaction_line(ax, pos1: Tuple[float, float], pos2: Tuple[float, float],
                         interaction_type: str, strength: float):
    """Draw an interaction line between two molecule positions."""
    
    colors = {
        'vdw': COLORS['vdw'],
        'dipole': COLORS['dipole'],
        'vibration': COLORS['vibration'],
    }
    
    styles = {
        'vdw': '--',      # Dashed for Van der Waals
        'dipole': '-.',   # Dash-dot for dipole
        'vibration': ':',  # Dotted for vibrational
    }
    
    color = colors.get(interaction_type, COLORS['vdw'])
    style = styles.get(interaction_type, '--')
    
    ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 
           linestyle=style, color=color, linewidth=1.5 * strength, alpha=0.7)


def draw_molecule_circle(ax, pos: Tuple[float, float], color: str, 
                        size: float = 0.15, is_observed: bool = False,
                        label: str = None):
    """Draw a molecule as a circle with optional glow for observed."""
    
    if is_observed:
        # Glow effect for observed molecule
        for i in range(3):
            glow = Circle(pos, size * (1.3 - i * 0.1), 
                         facecolor=color, alpha=0.1 * (3 - i), 
                         edgecolor='none')
            ax.add_patch(glow)
    
    # Main circle
    circle = Circle(pos, size, facecolor=color, edgecolor='white', 
                   linewidth=2 if is_observed else 1, alpha=0.9)
    ax.add_patch(circle)
    
    if label:
        ax.text(pos[0], pos[1], label, ha='center', va='center',
               fontsize=10 if is_observed else 8, fontweight='bold', color='white')


def create_ensemble_snapshot(ax, molecules: List[VirtualMolecule], 
                            observed_idx: int, time_label: str):
    """
    Create a snapshot of the gas ensemble at a given time.
    
    Shows 3 molecules with the observed one highlighted and
    interactions between them.
    """
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.5, 2.5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Molecule positions (triangular arrangement)
    positions = [
        (1.0, 1.8),   # Top
        (0.3, 0.5),   # Bottom left
        (1.7, 0.5),   # Bottom right
    ]
    
    colors = [COLORS['observed'], COLORS['related1'], COLORS['related2']]
    labels = ['α', 'β', 'γ']
    
    # Draw interactions first (behind molecules)
    interactions = [
        (0, 1, 'vdw', 0.8),
        (0, 2, 'dipole', 0.6),
        (1, 2, 'vibration', 0.7),
    ]
    
    for i, j, itype, strength in interactions:
        draw_interaction_line(ax, positions[i], positions[j], itype, strength)
    
    # Draw molecules
    for i, (mol, pos, color, label) in enumerate(zip(molecules, positions, colors, labels)):
        is_obs = (i == observed_idx)
        # Swap color to observed color if this is the observed molecule
        draw_color = COLORS['observed'] if is_obs else color
        draw_molecule_circle(ax, pos, draw_color, 
                           size=0.22 if is_obs else 0.15,
                           is_observed=is_obs, label=label)
    
    # Add time label
    ax.text(1.0, 2.4, time_label, ha='center', va='center', 
           fontsize=10, fontweight='bold')
    
    # Add S-coordinate annotation for observed molecule
    obs_mol = molecules[observed_idx]
    obs_pos = positions[observed_idx]
    ax.text(obs_pos[0], obs_pos[1] - 0.4, 
           f'S=({obs_mol.s_coord.S_k:.2f}, {obs_mol.s_coord.S_t:.2f}, {obs_mol.s_coord.S_e:.2f})',
           ha='center', va='top', fontsize=7, color=COLORS['observed'],
           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    # Legend for interactions (only on first panel)
    return ax


def generate_ensemble_panel():
    """
    Generate the complete 4-panel figure showing sliding window molecular identity.
    
    Panel layout:
    [Ensemble t1] [Radar α]
    [Ensemble t2] [Radar β]  
    [Ensemble t3] [Radar γ]
    [Ensemble t4] [Radar α→β→γ]
    """
    
    # Generate three molecules from hardware
    chamber = VirtualChamber()
    molecules = []
    for i in range(3):
        mol = chamber.sample()
        molecules.append(mol)
        time.sleep(0.01)  # Ensure different timestamps
    
    # Create figure
    fig = plt.figure(figsize=(14, 16))
    
    # Use gridspec for complex layout
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.3,
                          width_ratios=[1, 1.2])
    
    # Time labels for sliding window
    time_labels = [
        'Window $t_1$: Observing α',
        'Window $t_2$: α → β',
        'Window $t_3$: β → γ',
        'Window $t_4$: Cycle Complete'
    ]
    
    # Which molecule is "observed" at each time step
    observed_sequence = [0, 1, 2, 0]  # α, β, γ, back to α
    
    # Create each row
    for row, (time_label, obs_idx) in enumerate(zip(time_labels, observed_sequence)):
        
        # Left: Ensemble snapshot
        ax_ensemble = fig.add_subplot(gs[row, 0])
        create_ensemble_snapshot(ax_ensemble, molecules, obs_idx, time_label)
        
        # Right: Radar chart of observed molecule
        ax_radar = fig.add_subplot(gs[row, 1], projection='polar')
        
        # For the final panel, show overlay of all three
        if row == 3:
            # Show all three molecules overlaid to demonstrate equivalence
            for i, mol in enumerate(molecules):
                colors = [COLORS['observed'], COLORS['related1'], COLORS['related2']]
                alphas = [1.0, 0.5, 0.5]
                create_molecule_radar(ax_radar, mol, colors[i], 
                                     is_observed=(i == 0), alpha=alphas[i])
            ax_radar.set_title('All Three: Same S-Window', fontsize=9, 
                              fontweight='bold', color=COLORS['dark'], pad=15)
        else:
            # Show the currently observed molecule
            obs_mol = molecules[obs_idx]
            labels = ['Molecule α (Observed)', 'Molecule β (Observed)', 'Molecule γ (Observed)']
            colors = [COLORS['observed'], COLORS['related1'], COLORS['related2']]
            create_molecule_radar(ax_radar, obs_mol, colors[obs_idx], 
                                 is_observed=True, label=labels[obs_idx])
    
    # Add main title
    fig.suptitle('Virtual Gas Ensemble: One Molecule Becomes All\n'
                'Through S-Entropy Sliding Windows',
                fontsize=14, fontweight='bold', y=0.98)
    
    # Add explanation text at bottom
    explanation = (
        "Each molecule exists only during measurement. As the observation window slides through S-entropy space,\n"
        "the measured molecule (shown with full radar chart) assumes the identity of each molecule in sequence.\n"
        "The 'three molecules' are the same categorical state observed at different S-coordinates."
    )
    fig.text(0.5, 0.01, explanation, ha='center', va='bottom', fontsize=9,
            style='italic', color=COLORS['dark'])
    
    # Add interaction legend
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['vdw'], alpha=0.7, label='Van der Waals'),
        mpatches.Patch(facecolor=COLORS['dipole'], alpha=0.7, label='Dipole-Dipole'),
        mpatches.Patch(facecolor=COLORS['vibration'], alpha=0.7, label='Vibrational'),
    ]
    fig.legend(handles=legend_elements, loc='lower right', 
              bbox_to_anchor=(0.98, 0.02), fontsize=8, title='Interactions')
    
    return fig, molecules


def generate_detailed_ensemble_panel():
    """
    Generate a more detailed panel showing the hardware-to-molecule mapping.
    """
    
    # Generate molecules
    chamber = VirtualChamber()
    molecules = []
    hardware_data = []
    
    for i in range(3):
        t1 = time.perf_counter_ns()
        t2 = time.perf_counter_ns()
        delta = (t2 - t1) * 1e-9
        
        mol = VirtualMolecule.from_hardware_timing(delta)
        molecules.append(mol)
        hardware_data.append({
            'delta_ns': t2 - t1,
            'delta_p': delta,
        })
        time.sleep(0.005)
    
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.3)
    
    # Top row: Three individual molecule radar charts
    for i, (mol, hw) in enumerate(zip(molecules, hardware_data)):
        ax = fig.add_subplot(gs[0, i], projection='polar')
        colors = [COLORS['observed'], COLORS['related1'], COLORS['related2']]
        labels = [f'Molecule α\nΔt = {hw["delta_ns"]} ns',
                 f'Molecule β\nΔt = {hw["delta_ns"]} ns',
                 f'Molecule γ\nΔt = {hw["delta_ns"]} ns']
        create_molecule_radar(ax, mol, colors[i], is_observed=True, label=labels[i])
    
    # Bottom left: Ensemble view with all three
    ax_ensemble = fig.add_subplot(gs[1, 0])
    ax_ensemble.set_xlim(-0.5, 2.5)
    ax_ensemble.set_ylim(-0.3, 2.3)
    ax_ensemble.set_aspect('equal')
    ax_ensemble.axis('off')
    
    positions = [(1.0, 1.6), (0.4, 0.4), (1.6, 0.4)]
    colors = [COLORS['observed'], COLORS['related1'], COLORS['related2']]
    labels = ['α', 'β', 'γ']
    
    # Draw interactions
    for i in range(3):
        for j in range(i+1, 3):
            draw_interaction_line(ax_ensemble, positions[i], positions[j], 
                                 ['vdw', 'dipole', 'vibration'][i], 0.7)
    
    # Draw molecules
    for i, (mol, pos, color, label) in enumerate(zip(molecules, positions, colors, labels)):
        draw_molecule_circle(ax_ensemble, pos, color, size=0.18, 
                           is_observed=(i==0), label=label)
    
    ax_ensemble.set_title('Ensemble Configuration', fontsize=10, fontweight='bold')
    
    # Bottom middle: S-coordinate space
    ax_sspace = fig.add_subplot(gs[1, 1])
    
    for i, mol in enumerate(molecules):
        ax_sspace.scatter(mol.s_coord.S_k, mol.s_coord.S_e, 
                         c=colors[i], s=200, alpha=0.8, edgecolors='white', linewidth=2)
        ax_sspace.annotate(labels[i], (mol.s_coord.S_k, mol.s_coord.S_e),
                          fontsize=12, fontweight='bold', ha='center', va='center',
                          color='white')
    
    ax_sspace.set_xlabel('$S_k$ (Knowledge Entropy)', fontsize=10)
    ax_sspace.set_ylabel('$S_e$ (Evolution Entropy)', fontsize=10)
    ax_sspace.set_title('S-Entropy Space Positions', fontsize=10, fontweight='bold')
    ax_sspace.set_xlim(0, 1.1)
    ax_sspace.set_ylim(0, 1.1)
    ax_sspace.grid(True, alpha=0.3)
    
    # Bottom right: Hardware timing distribution
    ax_timing = fig.add_subplot(gs[1, 2])
    
    deltas = [hw['delta_ns'] for hw in hardware_data]
    bars = ax_timing.bar(['α', 'β', 'γ'], deltas, color=colors, alpha=0.8, 
                        edgecolor='white', linewidth=2)
    ax_timing.set_ylabel('Timing Delta (ns)', fontsize=10)
    ax_timing.set_title('Hardware Timing Source', fontsize=10, fontweight='bold')
    
    for bar, delta in zip(bars, deltas):
        ax_timing.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                      f'{delta}', ha='center', va='bottom', fontsize=9)
    
    # Main title
    fig.suptitle('Virtual Gas Ensemble: Hardware Oscillations → Categorical Molecules\n'
                'Three Timing Samples Create Three Molecules in S-Entropy Space',
                fontsize=13, fontweight='bold', y=0.98)
    
    return fig, molecules, hardware_data


def save_panels(output_dir: Path):
    """Generate and save all ensemble panels."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'panels': []
    }
    
    print("=" * 70)
    print(" Generating Virtual Gas Ensemble Panels")
    print("=" * 70)
    
    # Panel 1: Sliding window demonstration
    print("\n  Generating: Sliding Window Panel...")
    fig1, mols1 = generate_ensemble_panel()
    file1 = output_dir / 'panel_ensemble_sliding_window.png'
    fig1.savefig(file1, bbox_inches='tight', facecolor='white', dpi=150)
    plt.close(fig1)
    print(f"    ✓ Saved to: {file1}")
    
    results['panels'].append({
        'name': 'sliding_window',
        'file': str(file1),
        'molecules': len(mols1)
    })
    
    # Panel 2: Detailed hardware mapping
    print("\n  Generating: Hardware Mapping Panel...")
    fig2, mols2, hw2 = generate_detailed_ensemble_panel()
    file2 = output_dir / 'panel_ensemble_hardware_mapping.png'
    fig2.savefig(file2, bbox_inches='tight', facecolor='white', dpi=150)
    plt.close(fig2)
    print(f"    ✓ Saved to: {file2}")
    
    results['panels'].append({
        'name': 'hardware_mapping',
        'file': str(file2),
        'molecules': len(mols2),
        'hardware_data': hw2
    })
    
    # Save results summary
    results_file = output_dir / 'ensemble_panels_summary.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n{'='*70}")
    print(f" Panels saved to: {output_dir}")
    print(f"{'='*70}")
    
    return results


def main():
    output_dir = Path(__file__).parent.parent / 'results' / 'panels'
    return save_panels(output_dir)


if __name__ == "__main__":
    main()

