"""
Generate visualization panels for Sections 8-9:
- Forces from Cross-Scale Oscillatory Coupling
- Cosmological Structure from Categorical Exhaustion
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch, Wedge
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

plt.style.use('default')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f8f8f8'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

def generate_forces_cosmology_panel():
    """Visualize forces and cosmological structure."""
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.35)
    
    # Panel 1: Resonance enhancement
    ax1 = fig.add_subplot(gs[0, 0])
    
    omega_ratio = np.linspace(0.5, 2.0, 200)
    
    # Coupling strength near resonance
    gamma = 0.05  # Damping
    coupling = 1 / np.sqrt((1 - omega_ratio**2)**2 + (2*gamma*omega_ratio)**2)
    
    ax1.semilogy(omega_ratio, coupling, 'b-', linewidth=2.5)
    ax1.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='Resonance ω₁=ω₂')
    ax1.fill_between(omega_ratio, 1, coupling, alpha=0.2, where=(coupling>1))
    
    ax1.set_xlabel('Frequency Ratio ω₁/ω₂')
    ax1.set_ylabel('Coupling Enhancement')
    ax1.set_title('Resonance Enhancement\n(Mode Coupling)')
    ax1.legend()
    ax1.set_xlim(0.5, 2.0)
    
    # Panel 2: Force ranges
    ax2 = fig.add_subplot(gs[0, 1])
    
    r = np.logspace(-18, -6, 200)  # meters
    
    # Coulomb (1/r)
    V_em = 1 / r
    V_em = V_em / V_em[100]  # Normalize
    
    # Strong (Yukawa, short range)
    m_pion = 1e14  # 1/range in natural units
    V_strong = np.exp(-m_pion * r) / r
    V_strong = V_strong / np.max(V_strong) * 100
    
    # Weak (very short range)
    m_W = 1e17
    V_weak = np.exp(-m_W * r) / r
    V_weak = V_weak / np.max(V_weak) * 0.01
    
    # Gravity (1/r, very weak)
    V_grav = 1 / r
    V_grav = V_grav / V_grav[100] * 1e-36
    
    ax2.loglog(r, V_em, 'b-', linewidth=2, label='EM (1/r)')
    ax2.loglog(r, V_strong, 'r-', linewidth=2, label='Strong')
    ax2.loglog(r, V_grav, 'purple', linewidth=2, label='Gravity')
    
    # Mark ranges
    ax2.axvline(x=1e-15, color='red', linestyle=':', alpha=0.5)
    ax2.axvline(x=1e-18, color='orange', linestyle=':', alpha=0.5)
    
    ax2.set_xlabel('Distance r (m)')
    ax2.set_ylabel('Potential Strength (normalized)')
    ax2.set_title('Force Ranges\n(Mediator Mass)')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.set_xlim(1e-18, 1e-6)
    ax2.set_ylim(1e-40, 1e3)
    
    # Panel 3: Force hierarchy
    ax3 = fig.add_subplot(gs[0, 2])
    
    forces = ['Strong', 'EM', 'Weak', 'Gravity']
    strengths = [1, 1/137, 1e-6, 1e-39]  # Relative to strong
    colors = ['red', 'blue', 'orange', 'purple']
    
    y_pos = np.arange(len(forces))
    bars = ax3.barh(y_pos, np.log10(strengths) + 40, color=colors, edgecolor='black')
    
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(forces)
    ax3.set_xlabel('log₁₀(Coupling Strength) + 40')
    ax3.set_title('Force Hierarchy\n(40 orders of magnitude)')
    
    # Add labels
    for i, (f, s) in enumerate(zip(forces, strengths)):
        ax3.text(np.log10(s) + 41, i, f'{s:.0e}', va='center', fontsize=9)
    
    # Panel 4: Electromagnetic coupling diagram
    ax4 = fig.add_subplot(gs[0, 3])
    
    # Feynman-like diagram
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 10)
    
    # Electron lines
    ax4.annotate('', xy=(2, 8), xytext=(0, 10),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax4.annotate('', xy=(8, 8), xytext=(10, 10),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax4.annotate('', xy=(0, 0), xytext=(2, 2),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax4.annotate('', xy=(10, 0), xytext=(8, 2),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    
    # Photon (wavy line approximation)
    x_photon = np.linspace(2, 8, 50)
    y_photon = 5 + 0.5*np.sin(10*x_photon)
    ax4.plot(x_photon, y_photon, 'r-', linewidth=2)
    
    # Labels
    ax4.text(1, 9, 'e⁻', fontsize=12, fontweight='bold')
    ax4.text(9, 9, 'e⁻', fontsize=12, fontweight='bold')
    ax4.text(5, 6, 'γ', fontsize=14, color='red', fontweight='bold')
    
    ax4.axis('off')
    ax4.set_title('EM Interaction\n(Charge Coupling)')
    
    # Panel 5: Cyclic cosmology phases
    ax5 = fig.add_subplot(gs[1, 0])
    
    # Draw cycle
    theta = np.linspace(0, 2*np.pi, 100)
    r = 1.0
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    ax5.plot(x, y, 'k-', linewidth=3)
    
    # Mark phases
    phases = [
        (0, 'Expansion', 'green'),
        (np.pi/2, 'Max Extension\n(Heat Death)', 'blue'),
        (np.pi, 'Contraction', 'orange'),
        (3*np.pi/2, 'Max Compression\n(Big Bang)', 'red')
    ]
    
    for angle, label, color in phases:
        px, py = r * np.cos(angle), r * np.sin(angle)
        ax5.plot(px, py, 'o', markersize=15, color=color)
        ax5.text(px*1.4, py*1.4, label, ha='center', va='center', fontsize=9, 
                fontweight='bold', color=color)
    
    # Arrows showing direction
    for i in range(4):
        angle = i * np.pi/2 + np.pi/4
        ax5.annotate('', xy=(0.8*np.cos(angle+0.1), 0.8*np.sin(angle+0.1)),
                    xytext=(0.8*np.cos(angle-0.1), 0.8*np.sin(angle-0.1)),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    ax5.set_xlim(-2, 2)
    ax5.set_ylim(-2, 2)
    ax5.set_aspect('equal')
    ax5.axis('off')
    ax5.set_title('Cyclic Cosmology\n(Categorical Exhaustion)')
    
    # Panel 6: Entropy evolution
    ax6 = fig.add_subplot(gs[1, 1])
    
    t = np.linspace(0, 2, 200)  # One cosmic cycle
    
    # Entropy rises then falls
    S = np.sin(np.pi * t)**2
    
    ax6.plot(t, S, 'b-', linewidth=2.5)
    ax6.fill_between(t, 0, S, alpha=0.2)
    
    # Mark key points
    ax6.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Big Bang')
    ax6.axvline(x=0.5, color='blue', linestyle='--', alpha=0.5, label='Heat Death')
    ax6.axvline(x=1.0, color='orange', linestyle='--', alpha=0.5, label='Big Crunch')
    
    ax6.set_xlabel('Cosmic Time (cycles)')
    ax6.set_ylabel('Entropy S')
    ax6.set_title('Entropy Evolution\n(Cyclic)')
    ax6.legend(loc='upper right', fontsize=8)
    ax6.set_xlim(0, 2)
    
    # Panel 7: Configuration space exploration
    ax7 = fig.add_subplot(gs[1, 2])
    
    # Random walk in 2D as proxy for configuration space
    np.random.seed(42)
    n_steps = 5000
    steps = np.random.randn(n_steps, 2) * 0.1
    trajectory = np.cumsum(steps, axis=0)
    
    # Color by time
    colors = np.linspace(0, 1, n_steps)
    ax7.scatter(trajectory[:, 0], trajectory[:, 1], c=colors, cmap='viridis', 
               s=0.5, alpha=0.5)
    ax7.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=10, label='Start')
    ax7.plot(trajectory[-1, 0], trajectory[-1, 1], 'r^', markersize=10, label='Current')
    
    ax7.set_xlabel('Configuration Coordinate 1')
    ax7.set_ylabel('Configuration Coordinate 2')
    ax7.set_title('Configuration Space\nExploration')
    ax7.legend(loc='upper right')
    
    # Panel 8: Dark energy from vacuum
    ax8 = fig.add_subplot(gs[1, 3])
    
    # Scale factor evolution
    a = np.linspace(0.001, 2, 200)
    
    # Matter dominated (a^-3)
    rho_matter = 1/a**3
    
    # Dark energy (constant)
    rho_de = np.ones_like(a) * 0.7
    
    # Radiation (a^-4) 
    rho_rad = 0.001/a**4
    
    ax8.loglog(a, rho_matter, 'b-', linewidth=2, label='Matter ∝ a⁻³')
    ax8.loglog(a, rho_de, 'purple', linewidth=2, label='Dark Energy (const)')
    ax8.loglog(a, rho_rad, 'r-', linewidth=2, label='Radiation ∝ a⁻⁴')
    
    # Mark transition
    ax8.axvline(x=0.75, color='gray', linestyle='--', alpha=0.5)
    ax8.text(0.8, 0.5, 'Matter-DE\nequality', fontsize=9)
    
    ax8.set_xlabel('Scale Factor a')
    ax8.set_ylabel('Energy Density ρ')
    ax8.set_title('Dark Energy Dominance\n(Unoccupied Modes)')
    ax8.legend(loc='upper right', fontsize=8)
    ax8.set_xlim(0.01, 2)
    ax8.set_ylim(0.01, 1000)
    
    # Panel 9: Structure formation
    ax9 = fig.add_subplot(gs[2, 0])
    
    # Power spectrum
    k = np.logspace(-3, 0, 100)  # Wavenumber
    P_k = k**1 * np.exp(-k/0.3)  # Simplified power spectrum
    
    ax9.loglog(k, P_k, 'b-', linewidth=2.5)
    ax9.fill_between(k, 1e-4, P_k, alpha=0.2)
    
    # Mark scales
    ax9.axvline(x=0.01, color='red', linestyle='--', alpha=0.5)
    ax9.axvline(x=0.1, color='orange', linestyle='--', alpha=0.5)
    
    ax9.text(0.012, 0.3, 'Clusters', fontsize=9, rotation=90)
    ax9.text(0.12, 0.3, 'Galaxies', fontsize=9, rotation=90)
    
    ax9.set_xlabel('Wavenumber k (h/Mpc)')
    ax9.set_ylabel('Power Spectrum P(k)')
    ax9.set_title('Structure Formation\n(Oscillatory Instabilities)')
    
    # Panel 10: Initial conditions necessity
    ax10 = fig.add_subplot(gs[2, 1])
    
    # Phase space: low entropy initial vs random
    np.random.seed(123)
    
    # Low entropy initial
    x_low = np.random.randn(50) * 0.1
    y_low = np.random.randn(50) * 0.1
    
    # High entropy (heat death)
    x_high = np.random.randn(50) * 1
    y_high = np.random.randn(50) * 1
    
    ax10.scatter(x_low, y_low, c='green', s=50, alpha=0.7, label='Low S (Big Bang)')
    ax10.scatter(x_high + 3, y_high, c='red', s=50, alpha=0.7, label='High S (Heat Death)')
    
    # Draw bounding boxes
    rect_low = Rectangle((-0.4, -0.4), 0.8, 0.8, fill=False, edgecolor='green', linewidth=2)
    rect_high = Rectangle((0.5, -2.5), 5, 5, fill=False, edgecolor='red', linewidth=2)
    ax10.add_patch(rect_low)
    ax10.add_patch(rect_high)
    
    ax10.set_xlabel('Configuration 1')
    ax10.set_ylabel('Configuration 2')
    ax10.set_title('Initial Conditions\n(Low S Necessary)')
    ax10.legend(loc='upper right', fontsize=8)
    ax10.set_xlim(-1, 5)
    ax10.set_ylim(-3, 3)
    
    # Panel 11: Categorical exhaustion theorem
    ax11 = fig.add_subplot(gs[2, 2])
    
    # Visited states vs total states
    t = np.linspace(0, 10, 100)
    total_states = 1000
    
    # Monotonic: polynomial growth
    visited_mono = 100 * t**2
    
    # Cyclic: can revisit and explore more
    visited_cyclic = total_states * (1 - np.exp(-t/3))
    
    ax11.plot(t, visited_mono, 'b--', linewidth=2, label='Monotonic (incomplete)')
    ax11.plot(t, visited_cyclic, 'g-', linewidth=2.5, label='Cyclic (exhaustive)')
    ax11.axhline(y=total_states, color='red', linestyle='--', label=f'Total: {total_states}')
    
    ax11.set_xlabel('Time')
    ax11.set_ylabel('States Visited')
    ax11.set_title('Categorical Exhaustion\n(Cyclic Required)')
    ax11.legend(loc='lower right')
    ax11.set_xlim(0, 10)
    ax11.set_ylim(0, 1200)
    
    # Panel 12: Cosmological timeline
    ax12 = fig.add_subplot(gs[2, 3])
    
    # Log scale timeline
    events = [
        (-43, 'Planck\nEpoch'),
        (-36, 'Inflation'),
        (-12, 'Nucleosynthesis'),
        (-6, 'Recombination'),
        (6, 'First Stars'),
        (9, 'Galaxies'),
        (17, 'Now'),
        (20, 'Heat Death?')
    ]
    
    times = [e[0] for e in events]
    labels = [e[1] for e in events]
    
    ax12.scatter(times, np.zeros_like(times), s=100, c=range(len(times)), 
                cmap='plasma', zorder=3, edgecolor='black')
    ax12.plot(times, np.zeros_like(times), 'k-', linewidth=2, zorder=1)
    
    for i, (t, label) in enumerate(zip(times, labels)):
        va = 'bottom' if i % 2 == 0 else 'top'
        y = 0.3 if i % 2 == 0 else -0.3
        ax12.annotate(label, xy=(t, 0), xytext=(t, y),
                     fontsize=8, ha='center', va=va)
    
    ax12.set_xlim(-50, 25)
    ax12.set_ylim(-0.8, 0.8)
    ax12.set_xlabel('log₁₀(time in seconds)')
    ax12.set_title('Cosmological Timeline')
    ax12.axhline(y=0, color='black', linewidth=1)
    ax12.set_yticks([])
    
    plt.suptitle('Forces and Cosmological Structure', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('../figures/forces_cosmology_panel.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('../figures/forces_cosmology_panel.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: forces_cosmology_panel.png/pdf")

if __name__ == "__main__":
    import os
    os.makedirs('../figures', exist_ok=True)
    generate_forces_cosmology_panel()

