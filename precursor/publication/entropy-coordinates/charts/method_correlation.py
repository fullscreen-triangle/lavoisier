#!/usr/bin/env python3
"""
fig2_panel_b_correlation_schematic.py
Create schematic diagram showing bijective transformation and method correlation
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


if __name__ == "__main__":
    # Configuration
    OUTPUT_PDF = "fig2_panel_b_correlation_schematic.pdf"
    OUTPUT_PNG = "fig2_panel_b_correlation_schematic.png"

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(5, 9.5, 'Bijective S-Entropy Transformation',
            fontsize=16, fontweight='bold', ha='center')

    # Box 1: Raw Spectrum (Platform-Dependent)
    box1 = FancyBboxPatch((0.5, 6.5), 2, 1.5, boxstyle="round,pad=0.1",
                        edgecolor='red', facecolor='lightcoral', linewidth=2)
    ax.add_patch(box1)
    ax.text(1.5, 7.5, 'Raw Spectrum', fontsize=12, fontweight='bold', ha='center')
    ax.text(1.5, 7.1, 'Platform-Dependent', fontsize=9, ha='center', style='italic')
    ax.text(1.5, 6.8, '(m/z, intensity)', fontsize=9, ha='center')

    # Arrow 1: Forward transformation
    arrow1 = FancyArrowPatch((2.5, 7.25), (4.0, 7.25),
                            arrowstyle='->', mutation_scale=30, linewidth=3,
                            color='blue')
    ax.add_patch(arrow1)
    ax.text(3.25, 7.6, 'S-Entropy\nTransform', fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    # Box 2: S-Entropy Space (Platform-Independent)
    box2 = FancyBboxPatch((4.0, 6.5), 2, 1.5, boxstyle="round,pad=0.1",
                        edgecolor='green', facecolor='lightgreen', linewidth=2)
    ax.add_patch(box2)
    ax.text(5.0, 7.5, 'S-Entropy Space', fontsize=12, fontweight='bold', ha='center')
    ax.text(5.0, 7.1, 'Platform-Independent', fontsize=9, ha='center',
            style='italic', color='darkgreen')
    ax.text(5.0, 6.8, '(S_k, S_t, S_e)', fontsize=9, ha='center')

    # Arrow 2: Bijective (both directions)
    arrow2 = FancyArrowPatch((6.0, 7.25), (7.5, 7.25),
                            arrowstyle='<->', mutation_scale=30, linewidth=3,
                            color='purple')
    ax.add_patch(arrow2)
    ax.text(6.75, 7.6, 'Bijective\nMapping', fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor='plum', alpha=0.7))

    # Box 3: Categorical States
    box3 = FancyBboxPatch((7.5, 6.5), 2, 1.5, boxstyle="round,pad=0.1",
                        edgecolor='orange', facecolor='lightyellow', linewidth=2)
    ax.add_patch(box3)
    ax.text(8.5, 7.5, 'Categorical States', fontsize=12, fontweight='bold', ha='center')
    ax.text(8.5, 7.1, 'Hardware-Grounded', fontsize=9, ha='center', style='italic')
    ax.text(8.5, 6.8, '(Discrete classes)', fontsize=9, ha='center')

    # Method comparison section
    ax.text(5, 5.5, 'Method Correlation Analysis',
            fontsize=14, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, pad=0.5))

    # Box 4: Numerical Method
    box4 = FancyBboxPatch((1.0, 3.5), 3.5, 1.2, boxstyle="round,pad=0.1",
                        edgecolor='steelblue', facecolor='lightblue', linewidth=2)
    ax.add_patch(box4)
    ax.text(2.75, 4.3, 'Numerical Method', fontsize=11, fontweight='bold', ha='center')
    ax.text(2.75, 3.9, '• Spectral entropy calculation', fontsize=9, ha='center')
    ax.text(2.75, 3.7, '• Information-theoretic metrics', fontsize=9, ha='center')

    # Box 5: CV Method
    box5 = FancyBboxPatch((5.5, 3.5), 3.5, 1.2, boxstyle="round,pad=0.1",
                        edgecolor='coral', facecolor='mistyrose', linewidth=2)
    ax.add_patch(box5)
    ax.text(7.25, 4.3, 'Computer Vision Method', fontsize=11, fontweight='bold', ha='center')
    ax.text(7.25, 3.9, '• 65,878-D feature extraction', fontsize=9, ha='center')
    ax.text(7.25, 3.7, '• Hardware oscillation capture', fontsize=9, ha='center')

    # Correlation arrow
    arrow3 = FancyArrowPatch((4.5, 4.1), (5.5, 4.1),
                            arrowstyle='<->', mutation_scale=30, linewidth=4,
                            color='darkgreen')
    ax.add_patch(arrow3)

    # Correlation box
    corr_box = FancyBboxPatch((3.8, 2.5), 2.4, 0.8, boxstyle="round,pad=0.1",
                            edgecolor='darkgreen', facecolor='lightgreen',
                            linewidth=3)
    ax.add_patch(corr_box)
    ax.text(5.0, 3.0, 'r = 0.9508', fontsize=14, fontweight='bold', ha='center')
    ax.text(5.0, 2.7, 'p < 0.0001', fontsize=11, ha='center')

    # Bottom annotation
    ax.text(5, 1.5, 'Both methods converge to same S-entropy representation',
            fontsize=11, ha='center', style='italic',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6))
    ax.text(5, 1.0, 'Platform-independent identification regardless of acquisition hardware',
            fontsize=10, ha='center', fontweight='bold')

    # Add legend
    legend_elements = [
        mpatches.Rectangle((0, 0), 1, 1, fc='lightcoral', ec='red', label='Platform-Dependent'),
        mpatches.Rectangle((0, 0), 1, 1, fc='lightgreen', ec='green', label='Platform-Independent'),
        mpatches.Rectangle((0, 0), 1, 1, fc='lightyellow', ec='orange', label='Hardware-Grounded')
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=10, framealpha=0.9)

    plt.tight_layout()

    # Save
    plt.savefig(OUTPUT_PDF, format='pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_PNG, format='png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_PDF}")
    print(f"✓ Saved: {OUTPUT_PNG}")

    plt.close()
