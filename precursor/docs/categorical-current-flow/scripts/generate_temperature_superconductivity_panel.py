#!/usr/bin/env python3
"""
Generate Panel C-8/C-9: Temperature Dependence and Superconductivity
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
    
    # Panel A: Resistivity vs Temperature
    ax1 = fig.add_subplot(gs[0, 0])
    
    T = np.linspace(1, 400, 200)  # K
    
    # Normal metal (Matthiessen's rule)
    rho_0 = 0.1  # Residual resistivity (impurities)
    rho_ph = 0.01 * T  # Phonon contribution
    rho_total = rho_0 + rho_ph
    
    ax1.plot(T, rho_total, 'b-', linewidth=2.5, label='Total (Matthiessen)')
    ax1.plot(T, np.ones_like(T) * rho_0, 'g--', linewidth=2, 
             label=r'Impurity $\rho_0$')
    ax1.plot(T, rho_ph, 'r:', linewidth=2, label=r'Phonon $\rho_{ph}(T)$')
    
    ax1.set_xlabel('Temperature T (K)', fontsize=11)
    ax1.set_ylabel(r'Resistivity $\rho$ (normalized)', fontsize=11)
    ax1.set_title("(A) Matthiessen's Rule: Additive Resistivities", fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    ax1.text(200, 2.5, r'$\rho = \rho_0 + \rho_{ph}(T)$',
             fontsize=12,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    # Panel B: Superconducting Transition
    ax2 = fig.add_subplot(gs[0, 1])
    
    T = np.linspace(0, 15, 200)  # K
    T_c = 9.2  # Critical temperature (Nb)
    
    # Resistivity (drops to zero at T_c)
    rho = np.where(T > T_c, 0.5 * (T - T_c) / T_c + 1, 0)
    
    ax2.plot(T, rho, 'b-', linewidth=2.5)
    ax2.axvline(x=T_c, color='red', linestyle='--', linewidth=2, 
                label=f'$T_c$ = {T_c} K')
    
    # Highlight superconducting region
    ax2.fill_between(T[T <= T_c], 0, 0.1, alpha=0.3, color='lightgreen')
    ax2.text(T_c/2, 0.05, 'Superconducting\n(ρ = 0)', ha='center', fontsize=10, color='green')
    
    ax2.set_xlabel('Temperature T (K)', fontsize=11)
    ax2.set_ylabel(r'Resistivity $\rho$ (normalized)', fontsize=11)
    ax2.set_title('(B) Superconducting Transition: Aperture Bypass', fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 15)
    ax2.set_ylim(-0.1, 2)
    
    # Panel C: Cooper Pairing as Aperture Bypass
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_xlim(-0.5, 4)
    ax3.set_ylim(-0.5, 2.5)
    ax3.set_aspect('equal')
    
    # Normal state (with scattering)
    ax3.text(1, 2.2, 'Normal State (T > T_c)', fontsize=11, fontweight='bold')
    
    ax3.add_patch(Rectangle((0.2, 1.5), 2.6, 0.5, facecolor='lightblue', 
                             edgecolor='blue', linewidth=2))
    
    # Electron hitting aperture
    ax3.scatter([1.0], [1.75], s=100, c='blue', edgecolors='black', linewidth=1)
    ax3.scatter([1.5], [1.75], s=100, c='red', marker='x', linewidth=2)  # Scatterer
    ax3.annotate('', xy=(1.3, 1.75), xytext=(1.1, 1.75),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax3.text(1.5, 1.3, 'Blocked by\naperture', ha='center', fontsize=9, color='red')
    
    # Superconducting state (bypass)
    ax3.text(1, 0.8, 'Superconducting (T < T_c)', fontsize=11, fontweight='bold', color='green')
    
    ax3.add_patch(Rectangle((0.2, 0.1), 2.6, 0.5, facecolor='lightgreen', 
                             edgecolor='green', linewidth=2))
    
    # Cooper pair bypassing aperture
    ax3.scatter([0.8, 0.9], [0.35, 0.35], s=80, c='blue', edgecolors='black', linewidth=1)
    ax3.plot([0.8, 0.9], [0.35, 0.35], 'b-', linewidth=3)  # Pairing
    ax3.scatter([1.5], [0.35], s=80, c='red', marker='x', linewidth=2, alpha=0.3)  # Bypassed
    ax3.annotate('', xy=(2.3, 0.35), xytext=(1.0, 0.35),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax3.text(1.7, -0.1, 'Cooper pair\nbypasses aperture', ha='center', fontsize=9, color='green')
    
    ax3.set_title('(C) Cooper Pairs: Aperture Bypass Mechanism', fontweight='bold')
    ax3.axis('off')
    
    # Panel D: Skin Effect (Frequency Dependence)
    ax4 = fig.add_subplot(gs[1, 1])
    
    f = np.logspace(0, 10, 100)  # Hz
    
    # Skin depth
    mu_0 = 4 * np.pi * 1e-7
    sigma = 6e7  # Cu conductivity
    omega = 2 * np.pi * f
    
    delta = np.sqrt(2 / (mu_0 * sigma * omega))  # m
    
    ax4.loglog(f, delta * 1e6, 'b-', linewidth=2.5)
    
    # Mark key frequencies
    freq_markers = {
        '60 Hz\n(power)': 60,
        '1 MHz\n(RF)': 1e6,
        '1 GHz\n(microwave)': 1e9
    }
    
    for label, freq in freq_markers.items():
        delta_val = np.sqrt(2 / (mu_0 * sigma * 2 * np.pi * freq)) * 1e6
        ax4.scatter([freq], [delta_val], s=80, zorder=5, edgecolors='black')
        ax4.annotate(label, xy=(freq, delta_val), xytext=(freq*2, delta_val*1.5),
                    fontsize=8, ha='left')
    
    ax4.set_xlabel('Frequency f (Hz)', fontsize=11)
    ax4.set_ylabel('Skin depth δ (μm)', fontsize=11)
    ax4.set_title('(D) Skin Effect: Frequency-Dependent Aperture Selectivity', fontweight='bold')
    ax4.grid(True, alpha=0.3, which='both')
    
    ax4.text(1e4, 1e3, r'$\delta = \sqrt{\frac{2}{\mu_0 \sigma \omega}}$',
             fontsize=12,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.suptitle('Panel C-8/C-9: Temperature, Superconductivity, and Skin Effect', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # Save
    output_dir = Path(__file__).parent.parent / 'figures'
    output_dir.mkdir(exist_ok=True)
    
    plt.savefig(output_dir / 'panel_temperature_superconductivity.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(output_dir / 'panel_temperature_superconductivity.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    print(f"Saved: {output_dir / 'panel_temperature_superconductivity.png'}")
    print(f"Saved: {output_dir / 'panel_temperature_superconductivity.pdf'}")
    
    plt.close()

if __name__ == "__main__":
    main()

