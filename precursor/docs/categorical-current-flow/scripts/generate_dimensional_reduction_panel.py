#!/usr/bin/env python3
"""
Generate Panel C-2: Dimensional Reduction for Current Flow
Shows 3D wire → 0D cross-section × 1D S-transformation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch
from pathlib import Path

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10

def main():
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25)
    
    # Panel A: 3D Wire
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    
    # Wire as cylinder
    theta = np.linspace(0, 2*np.pi, 50)
    z = np.linspace(0, 5, 50)
    theta, z = np.meshgrid(theta, z)
    r = 0.5
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    ax1.plot_surface(x, y, z, alpha=0.5, color='lightblue', edgecolor='none')
    
    # Show cross-sections
    for z_val in [0, 2.5, 5]:
        circle_theta = np.linspace(0, 2*np.pi, 50)
        circle_x = r * np.cos(circle_theta)
        circle_y = r * np.sin(circle_theta)
        ax1.plot(circle_x, circle_y, [z_val]*len(circle_theta), 'b-', linewidth=2)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z (length)')
    ax1.set_title('(A) 3D Wire: Infinite Degrees of Freedom', fontweight='bold')
    
    # Panel B: Cross-Section (0D parameterized by radius)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_aspect('equal')
    
    # Wire cross-section
    circle = Circle((0, 0), 1.0, facecolor='lightblue', edgecolor='blue', linewidth=2)
    ax2.add_patch(circle)
    
    # Radial coordinate
    ax2.arrow(0, 0, 0.9, 0, head_width=0.1, head_length=0.08, fc='red', ec='red')
    ax2.text(0.5, 0.15, 'r', fontsize=12, color='red', fontweight='bold')
    
    # Current density (uniform)
    for theta in np.linspace(0, 2*np.pi, 8, endpoint=False):
        for r_val in [0.3, 0.6]:
            x = r_val * np.cos(theta)
            y = r_val * np.sin(theta)
            ax2.scatter([x], [y], s=40, c='green', marker='o')
    
    ax2.text(0, -1.3, 'Cross-section: All paths parallel\n0D (just radius matters)', 
             ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    ax2.set_title('(B) 0D Cross-Section: Radius Only', fontweight='bold')
    ax2.axis('off')
    
    # Panel C: S-Transformation Along Length
    ax3 = fig.add_subplot(gs[1, 0])
    
    z = np.linspace(0, 10, 100)  # Position along wire
    
    # S-potential (voltage drop)
    V_total = 10  # Volts
    V_z = V_total * (1 - z / 10)
    
    # S-coordinates
    S_k = 0.5 * np.ones_like(z)  # Knowledge entropy (constant)
    S_t = z / 10  # Temporal progression
    S_e = 1 - V_z / V_total  # Evolution entropy
    
    ax3.plot(z, V_z, 'b-', linewidth=2.5, label='S-potential V(z)')
    ax3.plot(z, S_t * V_total, 'g--', linewidth=2, label=r'$S_t$ (scaled)')
    ax3.plot(z, S_e * V_total, 'r:', linewidth=2, label=r'$S_e$ (scaled)')
    
    ax3.set_xlabel('Position z along wire', fontsize=11)
    ax3.set_ylabel('Potential / S-coordinate', fontsize=11)
    ax3.set_title('(C) 1D S-Transformation Along Length', fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    ax3.text(5, 8, r'$\mathbf{E} = -\nabla \Phi_S$' + '\nElectric field from S-gradient',
             ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    # Panel D: Complete Dimensional Reduction
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_xlim(-0.5, 4)
    ax4.set_ylim(-0.5, 2.5)
    ax4.set_aspect('equal')
    
    # 3D → 0D × 1D
    # 3D box
    ax4.add_patch(Rectangle((0.2, 1.8), 0.6, 0.5, facecolor='lightblue', 
                             edgecolor='blue', linewidth=2))
    ax4.text(0.5, 2.05, '3D', ha='center', fontsize=12, fontweight='bold')
    ax4.text(0.5, 1.6, 'Wire', ha='center', fontsize=9)
    
    # Arrow
    ax4.annotate('', xy=(1.3, 2.05), xytext=(0.9, 2.05),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax4.text(1.1, 2.25, 'Reduce', ha='center', fontsize=9)
    
    # 0D circle
    ax4.add_patch(Circle((1.7, 2.05), 0.2, facecolor='lightgreen', 
                         edgecolor='green', linewidth=2))
    ax4.text(1.7, 2.05, '0D', ha='center', fontsize=10, fontweight='bold')
    
    # Times symbol
    ax4.text(2.1, 2.05, '×', ha='center', fontsize=16, fontweight='bold')
    
    # 1D line
    ax4.plot([2.4, 3.2], [2.05, 2.05], 'orange', linewidth=4)
    ax4.text(2.8, 2.25, '1D', ha='center', fontsize=12, fontweight='bold')
    ax4.text(2.8, 1.85, r'$\mathcal{S}$-transform', ha='center', fontsize=9)
    
    # Formula
    ax4.text(2.0, 1.2, r'Wire = $\int_0^R 2\pi r \, dr \times \mathcal{S}$',
             ha='center', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    # Resistance formula
    ax4.text(2.0, 0.6, r'$R = \rho \frac{L}{A} = \rho \frac{L}{\pi r^2}$',
             ha='center', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.9))
    
    ax4.text(2.0, 0.1, 'Resistance from 0D (area) × 1D (length/conductivity)',
             ha='center', fontsize=10)
    
    ax4.set_title('(D) Complete Reduction: 3D → 0D × 1D', fontweight='bold')
    ax4.axis('off')
    
    plt.suptitle('Panel C-2: Dimensional Reduction — Wire as Cross-Section × S-Transform', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # Save
    output_dir = Path(__file__).parent.parent / 'figures'
    output_dir.mkdir(exist_ok=True)
    
    plt.savefig(output_dir / 'panel_dimensional_reduction.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(output_dir / 'panel_dimensional_reduction.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    print(f"Saved: {output_dir / 'panel_dimensional_reduction.png'}")
    print(f"Saved: {output_dir / 'panel_dimensional_reduction.pdf'}")
    
    plt.close()

if __name__ == "__main__":
    main()

