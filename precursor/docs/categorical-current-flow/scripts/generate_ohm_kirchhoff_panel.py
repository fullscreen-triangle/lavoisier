#!/usr/bin/env python3
"""
Generate Panel C-3/C-4: Ohm's Law and Kirchhoff's Laws from S-Dynamics
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
    
    # Panel A: V = IR from S-Dynamics
    ax1 = fig.add_subplot(gs[0, 0])
    
    I = np.linspace(0, 10, 100)  # Current (A)
    R_values = [1, 2, 5, 10]  # Resistance (Ohm)
    
    for R in R_values:
        V = I * R
        ax1.plot(I, V, linewidth=2.5, label=f'R = {R} Ω')
    
    ax1.set_xlabel('Current I (A)', fontsize=11)
    ax1.set_ylabel('Voltage V (V)', fontsize=11)
    ax1.set_title("(A) Ohm's Law: V = IR", fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    ax1.text(5, 80, r'$V = IR = \frac{\tau_s \cdot g \cdot L}{A} \cdot I$',
             fontsize=11, ha='center',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    # Panel B: Resistivity Formula
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Different materials
    materials = ['Cu', 'Al', 'Fe', 'Graphite', 'Silicon']
    rho = [1.7e-8, 2.8e-8, 1e-7, 1e-5, 6e2]  # Ohm·m
    tau_s = [2.5e-14, 1e-14, 1e-15, 1e-13, 1e-12]  # scattering time (s)
    
    ax2.scatter(tau_s, rho, s=150, c=['orange', 'gray', 'brown', 'black', 'blue'],
                edgecolors='black', linewidth=1.5, zorder=5)
    
    for mat, ts, r in zip(materials, tau_s, rho):
        ax2.annotate(mat, xy=(ts, r), xytext=(ts*1.5, r*0.5), fontsize=10)
    
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel(r'Scattering time $\tau_s$ (s)', fontsize=11)
    ax2.set_ylabel(r'Resistivity $\rho$ (Ω·m)', fontsize=11)
    ax2.set_title('(B) Resistivity from Scattering Partition Lag', fontweight='bold')
    ax2.grid(True, alpha=0.3, which='both')
    
    # Trend line (inverse relationship)
    tau_fit = np.logspace(-15, -11, 100)
    rho_fit = 1e-22 / tau_fit  # Proportionality
    ax2.plot(tau_fit, rho_fit, 'r--', linewidth=1.5, alpha=0.7, label=r'$\rho \propto 1/\tau_s$')
    ax2.legend(loc='upper right', fontsize=9)
    
    # Panel C: Kirchhoff's Current Law (KCL)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_xlim(-1, 3)
    ax3.set_ylim(-1, 2)
    ax3.set_aspect('equal')
    
    # Junction node
    ax3.add_patch(Circle((1, 0.5), 0.15, facecolor='gold', edgecolor='black', linewidth=2))
    ax3.text(1, 0.5, 'N', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Incoming currents
    ax3.annotate('', xy=(0.85, 0.5), xytext=(0, 0.5),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2.5))
    ax3.text(0.4, 0.7, r'$I_1$', fontsize=12, color='blue')
    
    ax3.annotate('', xy=(1, 0.35), xytext=(1, -0.5),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax3.text(1.2, -0.1, r'$I_2$', fontsize=12, color='blue')
    
    # Outgoing currents
    ax3.annotate('', xy=(2, 0.8), xytext=(1.15, 0.55),
                arrowprops=dict(arrowstyle='->', color='red', lw=2.5))
    ax3.text(1.7, 0.85, r'$I_3$', fontsize=12, color='red')
    
    ax3.annotate('', xy=(2, 0.2), xytext=(1.15, 0.45),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax3.text(1.7, 0.1, r'$I_4$', fontsize=12, color='red')
    
    # KCL equation
    ax3.text(1, 1.7, r'$\sum I_{in} = \sum I_{out}$' + '\n(Categorical state conservation)',
             ha='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    ax3.text(1, -0.8, r'$I_1 + I_2 = I_3 + I_4$',
             ha='center', fontsize=12, fontweight='bold')
    
    ax3.set_title("(C) Kirchhoff's Current Law: Conservation at Nodes", fontweight='bold')
    ax3.axis('off')
    
    # Panel D: Kirchhoff's Voltage Law (KVL)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_xlim(-0.5, 3.5)
    ax4.set_ylim(-0.5, 2.5)
    ax4.set_aspect('equal')
    
    # Circuit loop
    loop_x = [0.5, 2.5, 2.5, 0.5, 0.5]
    loop_y = [0.5, 0.5, 1.8, 1.8, 0.5]
    ax4.plot(loop_x, loop_y, 'k-', linewidth=2)
    
    # Voltage source
    ax4.add_patch(Circle((0.5, 1.15), 0.2, facecolor='yellow', edgecolor='black', linewidth=2))
    ax4.text(0.5, 1.15, 'V', ha='center', va='center', fontsize=10, fontweight='bold')
    ax4.text(0.1, 1.15, r'$V_s$', fontsize=10)
    
    # Resistors
    for i, (x, y, label) in enumerate([
        (1.5, 1.95, r'$V_1$'),
        (2.7, 1.15, r'$V_2$'),
        (1.5, 0.35, r'$V_3$')
    ]):
        ax4.add_patch(Rectangle((x-0.3, y-0.1), 0.6, 0.2, 
                                facecolor='lightgray', edgecolor='black', linewidth=1))
        ax4.text(x, y + 0.25, label, ha='center', fontsize=10)
    
    # Loop direction
    ax4.annotate('', xy=(1.5, 1.15), xytext=(1.5, 1.5),
                arrowprops=dict(arrowstyle='->', color='green', lw=2,
                               connectionstyle='arc3,rad=0.5'))
    ax4.text(1.2, 1.3, 'Loop', fontsize=9, color='green')
    
    # KVL equation
    ax4.text(1.5, -0.3, r'$\sum V_{loop} = 0$' + '\n(S-potential single-valued)',
             ha='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    ax4.text(1.5, 2.3, r'$V_s - V_1 - V_2 - V_3 = 0$',
             ha='center', fontsize=12, fontweight='bold')
    
    ax4.set_title("(D) Kirchhoff's Voltage Law: Loop Closure", fontweight='bold')
    ax4.axis('off')
    
    plt.suptitle("Panel C-3/C-4: Ohm's Law and Kirchhoff's Laws from Categorical Dynamics", 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # Save
    output_dir = Path(__file__).parent.parent / 'figures'
    output_dir.mkdir(exist_ok=True)
    
    plt.savefig(output_dir / 'panel_ohm_kirchhoff.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(output_dir / 'panel_ohm_kirchhoff.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    print(f"Saved: {output_dir / 'panel_ohm_kirchhoff.png'}")
    print(f"Saved: {output_dir / 'panel_ohm_kirchhoff.pdf'}")
    
    plt.close()

if __name__ == "__main__":
    main()

