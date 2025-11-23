#!/usr/bin/env python3
"""
fig1_panel_d_bijective.py
Diagram showing bijective transformation from raw spectrum to categorical states
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np



if __name__ == "__main__":
    # Configuration
    OUTPUT_PDF = "fig1_panel_d_bijective.pdf"
    OUTPUT_PNG = "fig1_panel_d_bijective.png"

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(7, 9.5, 'Bijective Transformation: Raw Spectrum → Categorical States',
            fontsize=15, fontweight='bold', ha='center')

    # Step 1: Raw Spectrum
    box1 = FancyBboxPatch((0.5, 6), 2.5, 2, boxstyle="round,pad=0.1",
                        edgecolor='red', facecolor='lightcoral', linewidth=2)
    ax.add_patch(box1)
    ax.text(1.75, 7.5, 'Raw Spectrum', fontsize=11, fontweight='bold', ha='center')
    ax.text(1.75, 7.1, 'Platform-Dependent', fontsize=9, ha='center', style='italic')
    ax.text(1.75, 6.7, '(m/z, intensity)', fontsize=9, ha='center')
    ax.text(1.75, 6.3, 'Example: 862 peaks', fontsize=8, ha='center')

    # Arrow 1
    arrow1 = FancyArrowPatch((3, 7), (4.5, 7),
                            arrowstyle='->', mutation_scale=30, linewidth=3,
                            color='blue')
    ax.add_patch(arrow1)
    ax.text(3.75, 7.5, '14D Feature\nExtraction', fontsize=9, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    # Step 2: Feature Space
    box2 = FancyBboxPatch((4.5, 6), 2.5, 2, boxstyle="round,pad=0.1",
                        edgecolor='blue', facecolor='lightblue', linewidth=2)
    ax.add_patch(box2)
    ax.text(5.75, 7.5, 'Feature Space', fontsize=11, fontweight='bold', ha='center')
    ax.text(5.75, 7.1, '14D Coordinates', fontsize=9, ha='center')
    ax.text(5.75, 6.7, 'Structural (4D)', fontsize=8, ha='center')
    ax.text(5.75, 6.5, 'Statistical (4D)', fontsize=8, ha='center')
    ax.text(5.75, 6.3, 'Information (4D)', fontsize=8, ha='center')
    ax.text(5.75, 6.1, 'Temporal (2D)', fontsize=8, ha='center')

    # Arrow 2
    arrow2 = FancyArrowPatch((7, 7), (8.5, 7),
                            arrowstyle='->', mutation_scale=30, linewidth=3,
                            color='green')
    ax.add_patch(arrow2)
    ax.text(7.75, 7.5, 'S-Entropy\nTransform', fontsize=9, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    # Step 3: S-Entropy Space
    box3 = FancyBboxPatch((8.5, 6), 2.5, 2, boxstyle="round,pad=0.1",
                        edgecolor='green', facecolor='lightgreen', linewidth=2)
    ax.add_patch(box3)
    ax.text(9.75, 7.5, 'S-Entropy Space', fontsize=11, fontweight='bold', ha='center')
    ax.text(9.75, 7.1, 'Platform-Independent', fontsize=9, ha='center',
            style='italic', color='darkgreen')
    ax.text(9.75, 6.7, '(S_k, S_t, S_e)', fontsize=9, ha='center')
    ax.text(9.75, 6.3, 'CV < 1%', fontsize=8, ha='center', fontweight='bold')

    # Arrow 3
    arrow3 = FancyArrowPatch((11, 7), (12.5, 7),
                            arrowstyle='->', mutation_scale=30, linewidth=3,
                            color='orange')
    ax.add_patch(arrow3)
    ax.text(11.75, 7.5, 'Categorical\nMapping', fontsize=9, ha='center',
            bbox=dict(boxstyle='round', facecolor='moccasin', alpha=0.7))

    # Step 4: Categorical States
    box4 = FancyBboxPatch((12.5, 6), 1.5, 2, boxstyle="round,pad=0.1",
                        edgecolor='orange', facecolor='moccasin', linewidth=2)
    ax.add_patch(box4)
    ax.text(13.25, 7.5, 'Categorical', fontsize=11, fontweight='bold', ha='center')
    ax.text(13.25, 7.2, 'States', fontsize=11, fontweight='bold', ha='center')
    ax.text(13.25, 6.7, 'Discrete', fontsize=9, ha='center')
    ax.text(13.25, 6.4, 'Classes', fontsize=9, ha='center')

    # Bijective property demonstration
    ax.text(7, 5, 'Bijective Properties:', fontsize=12, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    # Property 1: Information preservation
    ax.text(1, 4.2, '1. Information Preservation', fontsize=10, fontweight='bold')
    ax.text(1, 3.8, '   Reconstruction error ε < 0.01', fontsize=9)
    ax.text(1, 3.5, '   Complete spectral recovery possible', fontsize=9)

    # Property 2: Platform independence
    ax.text(5, 4.2, '2. Platform Independence', fontsize=10, fontweight='bold')
    ax.text(5, 3.8, '   ||f(M_A) - f(M_B)||₂ < 0.01', fontsize=9)
    ax.text(5, 3.5, '   Same metabolite → same coordinates', fontsize=9)

    # Property 3: Categorical consistency
    ax.text(9, 4.2, '3. Categorical Consistency', fontsize=10, fontweight='bold')
    ax.text(9, 3.8, '   Lipid classes occupy distinct regions', fontsize=9)
    ax.text(9, 3.5, '   Phase relationships preserved', fontsize=9)

    # Inverse transformation
    ax.annotate('', xy=(3, 5.5), xytext=(11, 5.5),
                arrowprops=dict(arrowstyle='<-', lw=2, color='purple',
                            linestyle='--', alpha=0.7))
    ax.text(7, 5.2, 'Inverse Transform (Reconstruction)', fontsize=9,
            ha='center', style='italic', color='purple')

    # Mathematical notation
    math_text = r'$\Phi: M \mapsto \mathbf{f}(M)$ is bijective'
    ax.text(7, 2.5, math_text, fontsize=12, ha='center',
            bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.8))

    # Bottom summary
    ax.text(7, 1.5, 'Enables zero-shot transfer learning across platforms',
            fontsize=11, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.6))
    ax.text(7, 0.8, 'Models trained on Platform A work directly on Platform B without retraining',
            fontsize=10, ha='center', style='italic')

    plt.tight_layout()

    # Save
    plt.savefig(OUTPUT_PDF, format='pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_PNG, format='png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_PDF}")
    print(f"✓ Saved: {OUTPUT_PNG}")

    plt.close()
