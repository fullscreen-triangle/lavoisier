#!/usr/bin/env python3
"""
fig1_panel_a_coordinate_system.py
Schematic diagram of 14D coordinate system
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
import numpy as np

# Configuration


if __name__ == "__main__":
    OUTPUT_PDF = "fig1_panel_a_coordinate_system.pdf"
    OUTPUT_PNG = "fig1_panel_a_coordinate_system.png"

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10), dpi=300)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(7, 9.5, '14-Dimensional S-Entropy Coordinate System',
            fontsize=16, fontweight='bold', ha='center')

    # Define feature categories and their positions
    categories = {
        'Spectral Features': {
            'pos': (2, 7),
            'features': ['f1: Base peak m/z', 'f2: Peak count', 'f3: m/z range',
                        'f4: Spacing variance'],
            'color': 'lightblue'
        },
        'Statistical Features': {
            'pos': (7, 7),
            'features': ['f5: Mean intensity', 'f6: Std intensity',
                        'f7: Skewness', 'f8: Kurtosis'],
            'color': 'lightgreen'
        },
        'Information Features': {
            'pos': (12, 7),
            'features': ['f9: Spectral entropy', 'f10: Structural entropy',
                        'f11: Mutual information', 'f12: Complexity'],
            'color': 'lightyellow'
        },
        'Temporal Features': {
            'pos': (5, 3),
            'features': ['f13: Temporal coordinate', 'f14: Phase coherence'],
            'color': 'lightcoral'
        }
    }

    # Draw category boxes
    for cat_name, cat_info in categories.items():
        x, y = cat_info['pos']

        # Main box
        box = FancyBboxPatch((x-1.5, y-1.5), 3, 2.5, boxstyle="round,pad=0.15",
                            edgecolor='black', facecolor=cat_info['color'],
                            linewidth=2)
        ax.add_patch(box)

        # Category title
        ax.text(x, y+0.9, cat_name, fontsize=12, fontweight='bold', ha='center')

        # Features
        for i, feature in enumerate(cat_info['features']):
            ax.text(x, y+0.5-i*0.35, feature, fontsize=9, ha='center')

    # Central S-Entropy Space circle
    circle = Circle((7, 5), 1.2, edgecolor='red', facecolor='mistyrose',
                    linewidth=3)
    ax.add_patch(circle)
    ax.text(7, 5.3, 'S-Entropy', fontsize=13, fontweight='bold', ha='center')
    ax.text(7, 4.9, 'Space', fontsize=13, fontweight='bold', ha='center')
    ax.text(7, 4.5, '(S_k, S_t, S_e)', fontsize=10, ha='center', style='italic')

    # Draw arrows from categories to central space
    for cat_name, cat_info in categories.items():
        x, y = cat_info['pos']
        arrow = FancyArrowPatch((x, y-1.5), (7, 5),
                            arrowstyle='->', mutation_scale=25, linewidth=2,
                            color='darkblue', alpha=0.6)
        ax.add_patch(arrow)

    # Bottom section: Transformation pipeline
    pipeline_y = 1.5
    ax.text(7, pipeline_y+0.5, 'Transformation Pipeline',
            fontsize=13, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Pipeline boxes
    pipeline_steps = [
        ('Raw\nSpectrum', 'lightgray', 1),
        ('14D\nFeatures', 'lightblue', 3.5),
        ('S-Entropy\nCoords', 'lightgreen', 6),
        ('Categorical\nStates', 'lightyellow', 8.5),
        ('Metabolite\nID', 'lightcoral', 11)
    ]

    for i, (label, color, x_pos) in enumerate(pipeline_steps):
        box = FancyBboxPatch((x_pos-0.6, pipeline_y-0.8), 1.2, 0.8,
                            boxstyle="round,pad=0.05",
                            edgecolor='black', facecolor=color, linewidth=1.5)
        ax.add_patch(box)
        ax.text(x_pos, pipeline_y-0.4, label, fontsize=9, ha='center',
                fontweight='bold')

        # Add arrows between steps
        if i < len(pipeline_steps) - 1:
            arrow = FancyArrowPatch((x_pos+0.6, pipeline_y-0.4),
                                (pipeline_steps[i+1][2]-0.6, pipeline_y-0.4),
                                arrowstyle='->', mutation_scale=20, linewidth=2,
                                color='black')
            ax.add_patch(arrow)

    # Add annotations
    ax.text(7, 0.3, 'Bijective mapping preserves information while achieving dimensionality reduction',
            fontsize=10, ha='center', style='italic',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    plt.tight_layout()

    # Save
    plt.savefig(OUTPUT_PDF, format='pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_PNG, format='png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_PDF}")
    print(f"✓ Saved: {OUTPUT_PNG}")

    plt.close()
