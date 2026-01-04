#!/usr/bin/env python3
"""
Generate Panel C-5/C-6: Maxwell's Equations from S-Dynamics
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
    
    # Panel A: E-field from S-Gradient (Gauss's Law)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)
    ax1.set_aspect('equal')
    
    # Point charge
    ax1.add_patch(Circle((0, 0), 0.15, facecolor='red', edgecolor='black', linewidth=2))
    ax1.text(0, 0, '+', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    
    # Electric field lines (radial)
    n_lines = 12
    for i in range(n_lines):
        theta = 2 * np.pi * i / n_lines
        # Arrow from center outward
        dx = 1.5 * np.cos(theta)
        dy = 1.5 * np.sin(theta)
        ax1.annotate('', xy=(dx, dy), xytext=(0.3*np.cos(theta), 0.3*np.sin(theta)),
                    arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
    
    # S-potential contours
    for r in [0.5, 1.0, 1.5]:
        circle = Circle((0, 0), r, facecolor='none', edgecolor='gray', 
                        linewidth=1, linestyle='--')
        ax1.add_patch(circle)
    ax1.text(1.2, 0.3, r'$\Phi_S$ = const', fontsize=9, color='gray')
    
    ax1.text(0, -1.8, r'$\mathbf{E} = -\nabla \Phi_S$' + '\nGauss: Field from S-gradient',
             ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    ax1.set_title("(A) Gauss's Law: E from S-Gradient", fontweight='bold')
    ax1.axis('off')
    
    # Panel B: B-field from S-Curl (Ampère's Law)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-2, 2)
    ax2.set_aspect('equal')
    
    # Current-carrying wire (into page)
    ax2.add_patch(Circle((0, 0), 0.2, facecolor='gray', edgecolor='black', linewidth=2))
    ax2.scatter([0], [0], s=50, c='black', marker='x', linewidth=2)
    ax2.text(0.35, 0.1, 'I (into page)', fontsize=9)
    
    # Magnetic field lines (circular)
    for r in [0.5, 1.0, 1.5]:
        circle = Circle((0, 0), r, facecolor='none', edgecolor='green', 
                        linewidth=1.5)
        ax2.add_patch(circle)
        # Add arrows to show direction
        for theta in [0, np.pi/2, np.pi, 3*np.pi/2]:
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            dx = -0.15 * np.sin(theta)
            dy = 0.15 * np.cos(theta)
            ax2.annotate('', xy=(x+dx, y+dy), xytext=(x, y),
                        arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
    
    ax2.text(0, -1.8, r'$\mathbf{B} = \nabla \times \mathbf{A}_S$' + '\nAmpère: B from S-curl',
             ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))
    
    ax2.set_title("(B) Ampère's Law: B from S-Curl", fontweight='bold')
    ax2.axis('off')
    
    # Panel C: Coupled E-B Oscillation
    ax3 = fig.add_subplot(gs[1, 0])
    
    x = np.linspace(0, 4*np.pi, 200)
    
    E = np.sin(x)
    B = np.cos(x)
    
    ax3.plot(x, E, 'b-', linewidth=2.5, label='E-field')
    ax3.plot(x, B, 'g-', linewidth=2.5, label='B-field')
    
    # Phase relationship
    ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    ax3.set_xlabel('Position (arbitrary units)', fontsize=11)
    ax3.set_ylabel('Field amplitude', fontsize=11)
    ax3.set_title('(C) Coupled E-B Oscillation: Electromagnetic Wave', fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    ax3.text(6, 0.6, 'E and B:\n90° phase shift\nPerpendicular',
             ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    # Panel D: Speed of Light from S-Dynamics
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_xlim(-0.5, 4)
    ax4.set_ylim(-0.5, 2.5)
    ax4.set_aspect('equal')
    
    # Wave equation
    ax4.text(2, 2.1, r'$\nabla^2 \mathbf{E} = \mu_0 \varepsilon_0 \frac{\partial^2 \mathbf{E}}{\partial t^2}$',
             ha='center', fontsize=14,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))
    
    ax4.text(2, 1.5, 'Wave equation from S-dynamics:', ha='center', fontsize=11)
    
    # Speed of light formula
    ax4.text(2, 1.0, r'$c = \frac{1}{\sqrt{\mu_0 \varepsilon_0}}$',
             ha='center', fontsize=16, fontweight='bold', color='darkblue')
    
    ax4.text(2, 0.5, f'= 299,792,458 m/s',
             ha='center', fontsize=12)
    
    # S-dynamics interpretation
    ax4.text(2, 0.0, 'S-transformation rate ↔ Wave velocity',
             ha='center', fontsize=11, style='italic',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    ax4.set_title('(D) Speed of Light from S-Dynamics', fontweight='bold')
    ax4.axis('off')
    
    plt.suptitle("Panel C-5/C-6: Maxwell's Equations from Categorical S-Dynamics", 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # Save
    output_dir = Path(__file__).parent.parent / 'figures'
    output_dir.mkdir(exist_ok=True)
    
    plt.savefig(output_dir / 'panel_maxwell_equations.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(output_dir / 'panel_maxwell_equations.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    print(f"Saved: {output_dir / 'panel_maxwell_equations.png'}")
    print(f"Saved: {output_dir / 'panel_maxwell_equations.pdf'}")
    
    plt.close()

if __name__ == "__main__":
    main()

