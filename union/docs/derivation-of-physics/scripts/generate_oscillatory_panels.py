"""
Generate visualization panels for Sections 2-3:
- Existence and Constraint Necessity
- Oscillatory Dynamics as Necessary Mode
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Wedge
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from scipy.integrate import odeint
from scipy.signal import find_peaks

# Set style
plt.style.use('default')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f8f8f8'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

def generate_bounded_phase_space_panel():
    """Visualize bounded vs unbounded phase space and Poincaré recurrence."""
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.35)
    
    # Panel 1: Bounded phase space with recurrent trajectory
    ax1 = fig.add_subplot(gs[0, 0])
    theta = np.linspace(0, 20*np.pi, 2000)
    r = 0.8 + 0.15*np.sin(3*theta) + 0.1*np.sin(7*theta)
    x = r * np.cos(theta + 0.5*np.sin(2*theta))
    y = r * np.sin(theta + 0.5*np.sin(2*theta))
    
    # Color by time
    colors = plt.cm.viridis(np.linspace(0, 1, len(x)))
    for i in range(len(x)-1):
        ax1.plot(x[i:i+2], y[i:i+2], color=colors[i], linewidth=0.5, alpha=0.7)
    
    # Draw boundary
    circle = Circle((0, 0), 1.0, fill=False, color='red', linewidth=2, linestyle='--')
    ax1.add_patch(circle)
    ax1.plot(x[0], y[0], 'go', markersize=10, label='Initial', zorder=5)
    ax1.plot(x[-1], y[-1], 'r^', markersize=10, label='Final', zorder=5)
    ax1.set_xlim(-1.3, 1.3)
    ax1.set_ylim(-1.3, 1.3)
    ax1.set_aspect('equal')
    ax1.set_xlabel('q (position)')
    ax1.set_ylabel('p (momentum)')
    ax1.set_title('Bounded Phase Space\n(Poincaré Recurrence)')
    ax1.legend(loc='upper right', fontsize=8)
    
    # Panel 2: Unbounded trajectory escaping
    ax2 = fig.add_subplot(gs[0, 1])
    t = np.linspace(0, 10, 500)
    x_unbound = t * np.cos(t)
    y_unbound = t * np.sin(t)
    
    colors = plt.cm.Reds(np.linspace(0.3, 1, len(t)))
    for i in range(len(t)-1):
        ax2.plot(x_unbound[i:i+2], y_unbound[i:i+2], color=colors[i], linewidth=1.5)
    
    ax2.plot(x_unbound[0], y_unbound[0], 'go', markersize=10, label='Initial')
    ax2.arrow(x_unbound[-2], y_unbound[-2], 
              (x_unbound[-1]-x_unbound[-2])*2, (y_unbound[-1]-y_unbound[-2])*2,
              head_width=0.5, head_length=0.3, fc='red', ec='red')
    ax2.set_xlabel('q (position)')
    ax2.set_ylabel('p (momentum)')
    ax2.set_title('Unbounded Phase Space\n(Trajectory Escapes)')
    ax2.legend(loc='upper left', fontsize=8)
    ax2.set_xlim(-12, 12)
    ax2.set_ylim(-12, 12)
    
    # Panel 3: Stability probability vs phase space volume
    ax3 = fig.add_subplot(gs[0, 2])
    V = np.linspace(0.1, 100, 100)
    P_stable = np.exp(-V/10)
    ax3.semilogy(V, P_stable, 'b-', linewidth=2.5)
    ax3.fill_between(V, P_stable, alpha=0.3)
    ax3.axhline(y=0.01, color='red', linestyle='--', label='Threshold')
    ax3.set_xlabel('Phase Space Volume |C|')
    ax3.set_ylabel('Stability Probability P(E)')
    ax3.set_title('Stability vs Volume\n(Constraint Necessity)')
    ax3.legend()
    ax3.set_ylim(1e-5, 1.5)
    
    # Panel 4: Energy surface visualization
    ax4 = fig.add_subplot(gs[0, 3], projection='3d')
    q = np.linspace(-2, 2, 50)
    p = np.linspace(-2, 2, 50)
    Q, P = np.meshgrid(q, p)
    H = P**2/2 + Q**2/2  # Harmonic oscillator
    
    ax4.plot_surface(Q, P, H, cmap='coolwarm', alpha=0.7, edgecolor='none')
    
    # Energy contour
    theta = np.linspace(0, 2*np.pi, 100)
    E = 1.0
    qc = np.sqrt(2*E) * np.cos(theta)
    pc = np.sqrt(2*E) * np.sin(theta)
    ax4.plot(qc, pc, np.ones_like(qc)*E, 'k-', linewidth=3, label=f'E={E}')
    
    ax4.set_xlabel('q')
    ax4.set_ylabel('p')
    ax4.set_zlabel('H(q,p)')
    ax4.set_title('Energy Surface\n(Bounded Dynamics)')
    
    # Panel 5: Four types of dynamics
    ax5 = fig.add_subplot(gs[1, 0])
    t = np.linspace(0, 10, 500)
    # Static
    ax5.axhline(y=0.5, color='gray', linewidth=2, label='Static')
    ax5.text(0.5, 0.55, 'Static\n(No dynamics)', fontsize=8, color='gray')
    ax5.set_xlim(0, 10)
    ax5.set_ylim(-0.1, 1.1)
    ax5.set_xlabel('Time')
    ax5.set_ylabel('State')
    ax5.set_title('Case (a): Static Equilibrium\n✗ Violates self-reference')
    ax5.legend(loc='upper right', fontsize=8)
    
    # Panel 6: Monotonic dynamics
    ax6 = fig.add_subplot(gs[1, 1])
    y_mono = 1 - np.exp(-t/3)
    ax6.plot(t, y_mono, 'orange', linewidth=2.5, label='Monotonic')
    ax6.arrow(8, y_mono[-50], 1, 0.02, head_width=0.03, head_length=0.2, fc='orange', ec='orange')
    ax6.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Bound')
    ax6.set_xlabel('Time')
    ax6.set_ylabel('State')
    ax6.set_title('Case (b): Monotonic\n✗ Violates boundedness')
    ax6.legend(loc='lower right', fontsize=8)
    ax6.set_xlim(0, 10)
    
    # Panel 7: Chaotic dynamics
    ax7 = fig.add_subplot(gs[1, 2])
    # Lorenz-like chaotic trajectory
    np.random.seed(42)
    y_chaos = np.cumsum(np.random.randn(500)) / 20
    y_chaos = (y_chaos - y_chaos.min()) / (y_chaos.max() - y_chaos.min())
    ax7.plot(t, y_chaos, 'purple', linewidth=1, label='Chaotic')
    ax7.set_xlabel('Time')
    ax7.set_ylabel('State')
    ax7.set_title('Case (c): Chaotic\n✗ Violates consistency')
    ax7.legend(loc='upper right', fontsize=8)
    
    # Panel 8: Oscillatory dynamics
    ax8 = fig.add_subplot(gs[1, 3])
    y_osc = 0.5 + 0.4*np.sin(2*np.pi*t/2.5) + 0.1*np.sin(2*np.pi*t/0.7)
    ax8.plot(t, y_osc, 'green', linewidth=2.5, label='Oscillatory')
    ax8.axhline(y=0.1, color='red', linestyle='--', alpha=0.5)
    ax8.axhline(y=0.9, color='red', linestyle='--', alpha=0.5)
    ax8.set_xlabel('Time')
    ax8.set_ylabel('State')
    ax8.set_title('Case (d): Oscillatory\n✓ Unique valid mode')
    ax8.legend(loc='upper right', fontsize=8)
    ax8.set_xlim(0, 10)
    
    # Panel 9: E = ℏω relationship
    ax9 = fig.add_subplot(gs[2, 0])
    omega = np.linspace(0.1, 10, 100)
    hbar = 1.0  # Natural units
    E = hbar * omega
    n_values = [1, 2, 3, 4]
    for n in n_values:
        ax9.plot(omega, n * E, linewidth=2, label=f'n={n}')
    ax9.set_xlabel('Frequency ω')
    ax9.set_ylabel('Energy E')
    ax9.set_title('Frequency-Energy Identity\nE = nℏω')
    ax9.legend()
    ax9.set_xlim(0, 10)
    
    # Panel 10: Hierarchical timescales
    ax10 = fig.add_subplot(gs[2, 1])
    levels = np.arange(0, 6)
    timescales = 10.0**(-15.0 + 3.0*levels)  # femtoseconds to seconds
    labels = ['Electron\n(10⁻¹⁵ s)', 'Molecular\n(10⁻¹² s)', 'Protein\n(10⁻⁹ s)', 
              'Cell\n(10⁻⁶ s)', 'Organ\n(10⁻³ s)', 'Organism\n(10⁰ s)']
    
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(levels)))
    bars = ax10.barh(levels, np.log10(timescales) + 16, color=colors, edgecolor='black')
    ax10.set_yticks(levels)
    ax10.set_yticklabels(labels, fontsize=8)
    ax10.set_xlabel('log₁₀(timescale) + 16')
    ax10.set_title('Hierarchical Timescale\nSeparation ~10³')
    
    # Add ratio annotations
    for i in range(len(levels)-1):
        ax10.annotate('×10³', xy=(3, levels[i]+0.5), fontsize=8, color='red')
    
    # Panel 11: Recurrence time distribution
    ax11 = fig.add_subplot(gs[2, 2])
    # Simulate recurrence times
    np.random.seed(123)
    recurrence_times = np.random.exponential(scale=50, size=1000)
    ax11.hist(recurrence_times, bins=40, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    
    # Theoretical exponential
    t_theory = np.linspace(0, 300, 100)
    pdf = (1/50) * np.exp(-t_theory/50)
    ax11.plot(t_theory, pdf, 'r-', linewidth=2.5, label='Exponential fit')
    ax11.axvline(x=50, color='orange', linestyle='--', label='Mean recurrence')
    ax11.set_xlabel('Recurrence Time T')
    ax11.set_ylabel('Probability Density')
    ax11.set_title('Recurrence Time Distribution\n(Poincaré Theorem)')
    ax11.legend()
    
    # Panel 12: Action quantization
    ax12 = fig.add_subplot(gs[2, 3])
    # Show quantized orbits
    n_orbits = 5
    for n in range(1, n_orbits + 1):
        theta = np.linspace(0, 2*np.pi, 100)
        r = np.sqrt(n)  # radius proportional to sqrt(n)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        ax12.plot(x, y, linewidth=2, label=f'n={n}, S={n}ℏ')
    
    ax12.set_xlabel('q')
    ax12.set_ylabel('p')
    ax12.set_title('Action Quantization\nS = ∮p dq = nℏ')
    ax12.set_aspect('equal')
    ax12.legend(loc='upper right', fontsize=8)
    ax12.set_xlim(-3, 3)
    ax12.set_ylim(-3, 3)
    
    plt.suptitle('Oscillatory Dynamics in Bounded Phase Space', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('../figures/oscillatory_dynamics_panel.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig('../figures/oscillatory_dynamics_panel.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: oscillatory_dynamics_panel.png/pdf")

if __name__ == "__main__":
    import os
    os.makedirs('../figures', exist_ok=True)
    generate_bounded_phase_space_panel()

