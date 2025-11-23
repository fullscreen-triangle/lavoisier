#!/usr/bin/env python3
"""
fig1_panel_c_hardware_sources.py
Visualize hardware oscillation sources and frequency spectrum
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyBboxPatch


if __name__ == "__main__":
    # Configuration
    OUTPUT_PDF = "fig1_panel_c_hardware_sources.pdf"
    OUTPUT_PNG = "fig1_panel_c_hardware_sources.png"

    # Hardware oscillation sources with frequencies
    sources = {
        'Quantum Membrane': {'freq': 1e15, 'color': 'purple'},
        'Intracellular Circuits': {'freq': 1e6, 'color': 'blue'},
        'Display Refresh': {'freq': 240, 'color': 'cyan'},
        'Network Packets': {'freq': 1e3, 'color': 'green'},
        'CPU Clock': {'freq': 3e9, 'color': 'orange'},
        'Thermal Fluctuations': {'freq': 1e-3, 'color': 'red'},
        'Cellular Information': {'freq': 10, 'color': 'magenta'},
        'Organism Allometric': {'freq': 1e-8, 'color': 'brown'}
    }

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), dpi=300)

    # Panel 1: Frequency spectrum
    frequencies = [np.log10(s['freq']) for s in sources.values()]
    labels = list(sources.keys())
    colors = [s['color'] for s in sources.values()]

    bars = ax1.barh(range(len(labels)), frequencies, color=colors,
                    edgecolor='black', linewidth=1.5, alpha=0.7)

    # Add frequency labels
    for i, (bar, freq) in enumerate(zip(bars, frequencies)):
        width = bar.get_width()
        ax1.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                f'{10**freq:.2e} Hz',
                ha='left', va='center', fontsize=9, fontweight='bold')

    # Styling
    ax1.set_yticks(range(len(labels)))
    ax1.set_yticklabels(labels, fontsize=11)
    ax1.set_xlabel('Log₁₀ Frequency (Hz)', fontsize=12, fontweight='bold')
    ax1.set_title('Hardware Oscillation Sources: Frequency Spectrum',
                fontsize=13, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Add scale annotations
    ax1.axvline(0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax1.axvline(9, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax1.text(0, -1, '1 Hz', ha='center', fontsize=9, style='italic')
    ax1.text(9, -1, '1 GHz', ha='center', fontsize=9, style='italic')

    # Panel 2: Coupling diagram
    ax2.axis('off')
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 8)

    # Title
    ax2.text(5, 7.5, 'Hardware-Molecular Oscillatory Coupling',
            fontsize=13, fontweight='bold', ha='center')

    # Hardware sources (left)
    hw_y = 6
    for i, (name, info) in enumerate(sources.items()):
        if 'Display' in name or 'Network' in name or 'CPU' in name or 'Thermal' in name:
            y_pos = hw_y - i*0.6
            box = FancyBboxPatch((0.5, y_pos-0.2), 2, 0.4,
                                boxstyle="round,pad=0.05",
                                edgecolor='black', facecolor=info['color'],
                                linewidth=1.5, alpha=0.6)
            ax2.add_patch(box)
            ax2.text(1.5, y_pos, name.split()[0], fontsize=9,
                    ha='center', va='center', fontweight='bold')

    # Molecular scales (right)
    mol_y = 6
    for i, (name, info) in enumerate(sources.items()):
        if 'Quantum' in name or 'Cellular' in name or 'Organism' in name:
            y_pos = mol_y - i*0.8
            box = FancyBboxPatch((7.5, y_pos-0.2), 2, 0.4,
                                boxstyle="round,pad=0.05",
                                edgecolor='black', facecolor=info['color'],
                                linewidth=1.5, alpha=0.6)
            ax2.add_patch(box)
            ax2.text(8.5, y_pos, name, fontsize=9,
                    ha='center', va='center', fontweight='bold')

    # Central coupling region
    coupling_box = FancyBboxPatch((3.5, 2), 3, 3,
                                boxstyle="round,pad=0.1",
                                edgecolor='red', facecolor='lightyellow',
                                linewidth=3, alpha=0.8)
    ax2.add_patch(coupling_box)
    ax2.text(5, 4, 'Phase-Lock\nCoupling', fontsize=12, ha='center', va='center',
            fontweight='bold', color='darkred')
    ax2.text(5, 3.2, 'BMD Stream', fontsize=10, ha='center', va='center',
            style='italic')

    # Draw coupling arrows
    for y_hw in [5.8, 5.2, 4.6, 4.0]:
        ax2.annotate('', xy=(3.5, 3.5), xytext=(2.5, y_hw),
                    arrowprops=dict(arrowstyle='->', lw=2, color='blue', alpha=0.6))

    for y_mol in [5.8, 5.0, 4.2]:
        ax2.annotate('', xy=(6.5, 3.5), xytext=(7.5, y_mol),
                    arrowprops=dict(arrowstyle='<-', lw=2, color='green', alpha=0.6))

    # Bottom annotation
    ax2.text(5, 1, 'Hardware oscillations provide irreducible phase-lock constraints',
            fontsize=10, ha='center', style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))

    plt.tight_layout()

    # Save
    plt.savefig(OUTPUT_PDF, format='pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_PNG, format='png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_PDF}")
    print(f"✓ Saved: {OUTPUT_PNG}")

    plt.close()
