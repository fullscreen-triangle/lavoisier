#!/usr/bin/env python3
"""
Generate Panel C-7: Scattering Apertures in Current Flow
Shows lattice scattering as categorical apertures.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch
from pathlib import Path

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10

def main():
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25)
    
    # Panel A: Aperture in k-Space
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-3, 3)
    ax1.set_aspect('equal')
    
    # Fermi surface (circle)
    fermi = Circle((0, 0), 2, facecolor='lightblue', edgecolor='blue', 
                   linewidth=2, alpha=0.5)
    ax1.add_patch(fermi)
    ax1.text(0, 0, 'Fermi\nSurface', ha='center', fontsize=10)
    
    # Scattering apertures (gaps in allowed states)
    # Phonon scattering (small angle)
    theta1 = np.pi/4
    ax1.add_patch(plt.matplotlib.patches.Wedge((0, 0), 2.3, 
                  np.degrees(theta1)-10, np.degrees(theta1)+10,
                  facecolor='red', alpha=0.4))
    ax1.text(2 * np.cos(theta1) * 1.3, 2 * np.sin(theta1) * 1.3, 
             'Phonon\naperture', fontsize=8, ha='center')
    
    # Impurity scattering (larger angle)
    theta2 = 3*np.pi/4
    ax1.add_patch(plt.matplotlib.patches.Wedge((0, 0), 2.3, 
                  np.degrees(theta2)-20, np.degrees(theta2)+20,
                  facecolor='orange', alpha=0.4))
    ax1.text(2 * np.cos(theta2) * 1.3, 2 * np.sin(theta2) * 1.3, 
             'Impurity\naperture', fontsize=8, ha='center')
    
    # Incoming electron
    ax1.annotate('', xy=(1.8, 0), xytext=(3.5, 0),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax1.text(2.8, 0.3, r'$\mathbf{k}$', fontsize=12, color='green')
    
    ax1.set_xlabel(r'$k_x$', fontsize=11)
    ax1.set_ylabel(r'$k_y$', fontsize=11)
    ax1.set_title('(A) Scattering Apertures in k-Space', fontweight='bold')
    
    # Panel B: Scattering Types Table
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')
    
    table_data = [
        ['Scattering Type', 'Selectivity (s)', 'T-Dependence', 'λ (nm)'],
        ['Phonon', '~0.1', r'$\propto T$', '10-100'],
        ['Impurity', '~0.01', 'T-independent', '1-10'],
        ['Electron-electron', '~0.5', r'$\propto T^2$', '100-1000'],
        ['Grain boundary', '~0.001', 'Weak', '0.1-1'],
        ['Surface', '~0.1', 'Complex', '~film thickness']
    ]
    
    colors = [['lightgray']*4] + [
        ['lightgreen', 'lightgreen', 'lightgreen', 'lightgreen'],
        ['lightyellow', 'lightyellow', 'lightyellow', 'lightyellow'],
        ['lightcyan', 'lightcyan', 'lightcyan', 'lightcyan'],
        ['mistyrose', 'mistyrose', 'mistyrose', 'mistyrose'],
        ['lavender', 'lavender', 'lavender', 'lavender']
    ]
    
    table = ax2.table(cellText=table_data, cellColours=colors,
                      loc='center', cellLoc='center',
                      colWidths=[0.3, 0.2, 0.25, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.6)
    
    ax2.set_title('(B) Scattering Types and Selectivities', fontweight='bold', y=0.95)
    
    # Panel C: Mean Free Path from Aperture Density
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Aperture density
    n_a = np.logspace(18, 24, 100)  # per m^3
    
    # Mean free path
    sigma = 1e-19  # Cross-section (m^2)
    lambda_mfp = 1 / (n_a * sigma)  # m
    
    ax3.loglog(n_a, lambda_mfp * 1e9, 'b-', linewidth=2.5)
    
    # Mark different regimes
    ax3.axhline(y=40, color='green', linestyle='--', alpha=0.7, label='λ for Cu (~40 nm)')
    ax3.axhline(y=5, color='orange', linestyle='--', alpha=0.7, label='λ for Fe (~5 nm)')
    
    ax3.set_xlabel('Scatterer density n (m⁻³)', fontsize=11)
    ax3.set_ylabel('Mean free path λ (nm)', fontsize=11)
    ax3.set_title('(C) Mean Free Path from Aperture Density', fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3, which='both')
    
    ax3.text(1e21, 1e2, r'$\lambda = \frac{1}{n_a \sigma}$',
             fontsize=12,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    # Panel D: Resistance as Aperture Barrier Sum
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_xlim(-0.5, 4)
    ax4.set_ylim(-0.5, 2.5)
    ax4.set_aspect('equal')
    
    # Wire with scattering centers
    ax4.add_patch(Rectangle((0.3, 0.8), 3.4, 0.4, facecolor='lightblue', 
                             edgecolor='blue', linewidth=2))
    
    # Scattering centers as apertures
    scatterer_x = [0.8, 1.5, 2.0, 2.8, 3.2]
    for x in scatterer_x:
        ax4.scatter([x], [1.0], s=100, c='red', marker='x', linewidth=2, zorder=5)
    
    # Electron path
    ax4.annotate('', xy=(3.5, 1.0), xytext=(0.5, 1.0),
                arrowprops=dict(arrowstyle='->', color='green', lw=2,
                               connectionstyle='arc3,rad=0.1'))
    
    ax4.text(2.0, 0.5, 'Each scatterer = aperture barrier',
             ha='center', fontsize=10)
    
    # Formula
    ax4.text(2.0, 2.0, r'$R = \sum_a \frac{\Phi_a}{I} = \frac{L}{A} \sum_a \frac{1}{s_a \tau_a}$',
             ha='center', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    ax4.text(2.0, 1.5, 'Resistance = Sum of aperture potentials',
             ha='center', fontsize=10, style='italic')
    
    ax4.set_title('(D) Resistance as Aperture Barrier Sum', fontweight='bold')
    ax4.axis('off')
    
    plt.suptitle('Panel C-7: Lattice Scattering as Categorical Apertures', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # Save
    output_dir = Path(__file__).parent.parent / 'figures'
    output_dir.mkdir(exist_ok=True)
    
    plt.savefig(output_dir / 'panel_scattering_apertures.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(output_dir / 'panel_scattering_apertures.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    print(f"Saved: {output_dir / 'panel_scattering_apertures.png'}")
    print(f"Saved: {output_dir / 'panel_scattering_apertures.pdf'}")
    
    plt.close()

if __name__ == "__main__":
    main()

