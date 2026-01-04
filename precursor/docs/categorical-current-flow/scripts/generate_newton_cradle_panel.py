#!/usr/bin/env python3
"""
Generate Panel C-1: Newton's Cradle Model for Current Flow
Shows electron chain and displacement propagation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from pathlib import Path

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10

def main():
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25)
    
    # Panel A: Electron Chain in Wire
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(-0.5, 4)
    ax1.set_ylim(-0.5, 1.5)
    ax1.set_aspect('equal')
    
    # Wire (cylinder cross-section)
    wire = Rectangle((0, 0.3), 3.5, 0.4, facecolor='lightgray', 
                     edgecolor='black', linewidth=2)
    ax1.add_patch(wire)
    
    # Electrons
    n_electrons = 8
    for i in range(n_electrons):
        x = 0.2 + i * 0.4
        circle = Circle((x, 0.5), 0.12, facecolor='royalblue', 
                        edgecolor='darkblue', linewidth=1)
        ax1.add_patch(circle)
        ax1.text(x, 0.5, 'e⁻', ha='center', va='center', fontsize=7, color='white')
    
    # Lattice ions (fixed)
    for i in range(n_electrons + 1):
        x = 0.0 + i * 0.4
        ax1.scatter([x], [0.9], s=100, c='red', marker='+', linewidth=2)
    
    ax1.text(1.75, 1.2, 'Lattice ions (fixed)', ha='center', fontsize=10)
    ax1.text(1.75, 0.0, 'Electron chain (mobile)', ha='center', fontsize=10)
    
    ax1.set_title('(A) Wire Cross-Section: Electron Chain', fontweight='bold')
    ax1.axis('off')
    
    # Panel B: Displacement Propagation (Newton's Cradle)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_xlim(-0.5, 4)
    ax2.set_ylim(-0.5, 2.5)
    ax2.set_aspect('equal')
    
    # Three time snapshots
    times = [0, 1, 2]
    y_offsets = [2.0, 1.0, 0.0]
    labels = ['t = 0: Push', 't = dt: Propagate', 't = 2dt: Exit']
    
    for t, y_off, label in zip(times, y_offsets, labels):
        # Wire
        wire = Rectangle((0.3, y_off + 0.1), 3.0, 0.3, facecolor='lightgray', 
                         edgecolor='black', linewidth=1)
        ax2.add_patch(wire)
        
        # Electrons
        n_e = 6
        for i in range(n_e):
            if t == 0:
                # First electron pushed
                x = 0.4 + i * 0.4 + (0.15 if i == 0 else 0)
            elif t == 1:
                # Middle electron displaced
                x = 0.4 + i * 0.4 + (0.15 if i == 2 else 0)
            else:
                # Last electron exits
                x = 0.4 + i * 0.4 + (0.15 if i == n_e-1 else 0)
            
            circle = Circle((x, y_off + 0.25), 0.08, facecolor='royalblue', 
                            edgecolor='darkblue', linewidth=1)
            ax2.add_patch(circle)
        
        ax2.text(-0.3, y_off + 0.25, label, ha='right', fontsize=9, fontweight='bold')
        
        # Arrows showing push/exit
        if t == 0:
            ax2.annotate('', xy=(0.45, y_off + 0.25), xytext=(0.1, y_off + 0.25),
                        arrowprops=dict(arrowstyle='->', color='red', lw=2))
        elif t == 2:
            ax2.annotate('', xy=(3.6, y_off + 0.25), xytext=(3.3, y_off + 0.25),
                        arrowprops=dict(arrowstyle='->', color='green', lw=2))
    
    ax2.text(2, -0.3, 'Signal propagates at ~c\nElectrons barely move!',
             ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    ax2.set_title("(B) Newton's Cradle: Displacement Propagation", fontweight='bold')
    ax2.axis('off')
    
    # Panel C: Signal Speed vs Drift Velocity
    ax3 = fig.add_subplot(gs[1, 0])
    
    categories = ['Signal\nSpeed', 'Drift\nVelocity']
    values = [3e8, 1e-4]  # m/s
    colors = ['red', 'blue']
    
    bars = ax3.bar(categories, values, color=colors, edgecolor='black', linewidth=2)
    ax3.set_yscale('log')
    ax3.set_ylabel('Velocity (m/s)', fontsize=11)
    ax3.set_title('(C) Speed Comparison: Signal vs Drift', fontweight='bold')
    ax3.set_ylim(1e-5, 1e9)
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax3.text(bar.get_x() + bar.get_width()/2, val * 2, 
                 f'{val:.0e} m/s', ha='center', fontsize=10, fontweight='bold')
    
    # Ratio annotation
    ratio = values[0] / values[1]
    ax3.text(0.5, 1e2, f'Ratio: {ratio:.0e}',
             ha='center', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    ax3.grid(True, alpha=0.3, which='both')
    
    # Panel D: Why Current is Categorical
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_xlim(-0.5, 3.5)
    ax4.set_ylim(-0.5, 2)
    ax4.set_aspect('equal')
    
    # Categorical view
    ax4.text(0.5, 1.7, 'Classical View:', fontsize=11, fontweight='bold')
    ax4.text(0.5, 1.4, 'Electrons flow like water', fontsize=10, style='italic')
    
    ax4.text(2.5, 1.7, 'Categorical View:', fontsize=11, fontweight='bold', color='green')
    ax4.text(2.5, 1.4, 'States propagate', fontsize=10, style='italic', color='green')
    
    # Diagrams
    # Classical (wrong)
    for i in range(3):
        circle = Circle((0.3 + i*0.3, 0.8), 0.08, facecolor='blue', 
                        edgecolor='black', linewidth=1)
        ax4.add_patch(circle)
    ax4.annotate('', xy=(1.2, 0.8), xytext=(0.8, 0.8),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax4.plot([0.1, 0.9], [0.5, 0.5], 'r-', linewidth=3)
    ax4.plot([0.3, 0.7], [0.4, 0.6], 'r-', linewidth=3)
    ax4.text(0.5, 0.25, 'WRONG', fontsize=10, fontweight='bold', color='red', ha='center')
    
    # Categorical (correct)
    # State labels
    for i, label in enumerate(['|0>', '|1>', '|0>']):
        x = 2.0 + i * 0.4
        ax4.text(x, 0.8, label, fontsize=10, ha='center', 
                color='green' if label == '|1>' else 'gray')
        circle = Circle((x, 0.6), 0.08, 
                        facecolor='green' if label == '|1>' else 'lightgray',
                        edgecolor='black', linewidth=1)
        ax4.add_patch(circle)
    
    ax4.annotate('', xy=(2.6, 0.6), xytext=(2.1, 0.6),
                arrowprops=dict(arrowstyle='->', color='green', lw=2, ls='--'))
    ax4.text(2.35, 0.35, 'State propagates', fontsize=9, ha='center', color='green')
    ax4.text(2.35, 0.1, 'CORRECT', fontsize=10, fontweight='bold', color='green', ha='center')
    
    ax4.set_title('(D) Current as Categorical State Propagation', fontweight='bold')
    ax4.axis('off')
    
    plt.suptitle("Panel C-1: Newton's Cradle Model — Current as State Propagation", 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # Save
    output_dir = Path(__file__).parent.parent / 'figures'
    output_dir.mkdir(exist_ok=True)
    
    plt.savefig(output_dir / 'panel_newton_cradle.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(output_dir / 'panel_newton_cradle.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    print(f"Saved: {output_dir / 'panel_newton_cradle.png'}")
    print(f"Saved: {output_dir / 'panel_newton_cradle.pdf'}")
    
    plt.close()

if __name__ == "__main__":
    main()

