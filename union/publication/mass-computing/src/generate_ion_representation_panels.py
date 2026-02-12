#!/usr/bin/env python3
"""
Generate three panels showing equivalent ion representations:
1. Ion as Oscillating Object (harmonic oscillator dynamics, Mathieu stability)
2. Ion as Discrete State (categorical states, transition probabilities)
3. Ion as Partition Coordinates (quantum numbers, phase space partitioning)

Each panel contains 4 charts including 3D visualizations and heatmaps.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import os

# Create custom colormap
colors_entropy = ['#1a1a2e', '#16213e', '#0f3460', '#e94560', '#ff6b6b', '#feca57']
cmap_entropy = LinearSegmentedColormap.from_list('entropy', colors_entropy)

# Scientific color palette
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'tertiary': '#F18F01',
    'quaternary': '#C73E1D',
    'background': '#1a1a2e',
    'grid': '#3a3a5e',
    'text': '#ffffff',
    'accent1': '#00d4aa',
    'accent2': '#ff6b9d',
    'accent3': '#c4b5fd',
    'stable': '#22c55e',
    'unstable': '#ef4444'
}

def set_dark_style(ax, is_3d=False):
    """Apply dark style to axis."""
    ax.set_facecolor(COLORS['background'])
    ax.tick_params(colors=COLORS['text'], labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(COLORS['grid'])
    if is_3d:
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor(COLORS['grid'])
        ax.yaxis.pane.set_edgecolor(COLORS['grid'])
        ax.zaxis.pane.set_edgecolor(COLORS['grid'])
        ax.xaxis.label.set_color(COLORS['text'])
        ax.yaxis.label.set_color(COLORS['text'])
        ax.zaxis.label.set_color(COLORS['text'])

def generate_panel1_oscillatory(save_path):
    """
    Panel 1: Ion as Oscillating Object
    - 3D RF trajectory in quadrupole field
    - Mathieu stability diagram (heatmap)
    - Potential landscape (3D surface)
    - Secular motion time series
    """
    fig = plt.figure(figsize=(16, 14))
    fig.patch.set_facecolor(COLORS['background'])

    # Title
    fig.suptitle('Ion as Oscillating Object\nHarmonic Oscillator Dynamics in RF Fields',
                 fontsize=16, color=COLORS['text'], fontweight='bold', y=0.98)

    # === Chart 1: 3D RF Trajectory ===
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    set_dark_style(ax1, is_3d=True)

    # Generate Mathieu-type trajectory
    t = np.linspace(0, 50, 2000)
    omega = 2 * np.pi * 1.0  # RF frequency
    omega_sec = 0.1 * omega  # Secular frequency
    q_param = 0.4  # Mathieu q parameter

    # Ion trajectory with micromotion
    x = np.cos(omega_sec * t) * (1 + 0.3 * np.cos(omega * t)) * np.exp(-0.02 * t)
    y = np.sin(omega_sec * t) * (1 + 0.3 * np.sin(omega * t + np.pi/4)) * np.exp(-0.02 * t)
    z = 0.5 * np.cos(0.5 * omega_sec * t) * np.exp(-0.02 * t)

    # Color by time
    points = np.array([x, y, z]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    colors = plt.cm.plasma(np.linspace(0, 1, len(t)-1))
    for i in range(len(segments)):
        ax1.plot([segments[i,0,0], segments[i,1,0]],
                [segments[i,0,1], segments[i,1,1]],
                [segments[i,0,2], segments[i,1,2]],
                color=colors[i], linewidth=0.8, alpha=0.8)

    # Add electrode representation
    theta_elec = np.linspace(0, 2*np.pi, 100)
    r_elec = 1.5
    for z_pos in [-0.8, 0.8]:
        ax1.plot(r_elec * np.cos(theta_elec), r_elec * np.sin(theta_elec),
                np.full_like(theta_elec, z_pos), color=COLORS['grid'], alpha=0.3, linewidth=2)

    ax1.set_xlabel('x (r₀ units)', fontsize=9, color=COLORS['text'])
    ax1.set_ylabel('y (r₀ units)', fontsize=9, color=COLORS['text'])
    ax1.set_zlabel('z (r₀ units)', fontsize=9, color=COLORS['text'])
    ax1.set_title('3D RF Trajectory: Secular + Micromotion', fontsize=11,
                  color=COLORS['accent1'], pad=10)

    # Add equation
    ax1.text2D(0.02, 0.95, r'$\frac{d^2u}{d\xi^2} + [a - 2q\cos(2\xi)]u = 0$',
              transform=ax1.transAxes, fontsize=10, color=COLORS['accent2'],
              bbox=dict(boxstyle='round', facecolor=COLORS['background'], alpha=0.8))

    # === Chart 2: Mathieu Stability Diagram (Heatmap) ===
    ax2 = fig.add_subplot(2, 2, 2)
    set_dark_style(ax2)

    # Create stability parameter grid
    a_vals = np.linspace(-1.5, 1.5, 200)
    q_vals = np.linspace(0, 1.2, 200)
    A, Q = np.meshgrid(a_vals, q_vals)

    # Approximate stability regions (first stability region)
    # Mathieu stability condition (simplified)
    beta_sq = A + Q**2 / 2
    stability = np.zeros_like(A)

    # First stability region approximation
    stability_condition = (A > -Q**2/2 - 0.5*Q) & (A < 1 - Q - Q**2/2) & (Q < 0.908)
    stability[stability_condition] = 1

    # Add higher-order corrections for realism
    for i in range(len(q_vals)):
        for j in range(len(a_vals)):
            q, a = q_vals[i], a_vals[j]
            # Approximate stability boundaries
            beta_u = a + 0.5 * q**2
            beta_v = -a + 0.5 * q**2
            if 0 < beta_u < 1 and 0 < beta_v < 1:
                stability[i, j] = 1 + 0.5 * np.exp(-((a-0.2)**2 + (q-0.7)**2)/0.1)

    im2 = ax2.imshow(stability, extent=[a_vals.min(), a_vals.max(), q_vals.min(), q_vals.max()],
                     origin='lower', aspect='auto', cmap='RdYlGn')

    # Mark operating points for different ions
    ions = [
        {'name': 'H⁺', 'a': 0.02, 'q': 0.7, 'color': COLORS['accent1']},
        {'name': 'CH₄⁺', 'a': 0.05, 'q': 0.5, 'color': COLORS['accent2']},
        {'name': 'C₆₀⁺', 'a': 0.01, 'q': 0.2, 'color': COLORS['tertiary']}
    ]
    for ion in ions:
        ax2.scatter(ion['a'], ion['q'], s=100, c=ion['color'], edgecolor='white',
                   linewidth=2, zorder=5, label=ion['name'])

    ax2.set_xlabel('a = 8eU/(mΩ²r₀²)', fontsize=10, color=COLORS['text'])
    ax2.set_ylabel('q = 4eV/(mΩ²r₀²)', fontsize=10, color=COLORS['text'])
    ax2.set_title('Mathieu Stability Diagram', fontsize=11, color=COLORS['accent1'], pad=10)
    ax2.legend(loc='upper right', facecolor=COLORS['background'],
               labelcolor=COLORS['text'], fontsize=9)

    cbar2 = plt.colorbar(im2, ax=ax2, label='Stability Index')
    cbar2.ax.yaxis.label.set_color(COLORS['text'])
    cbar2.ax.tick_params(colors=COLORS['text'])

    # === Chart 3: 3D Potential Landscape ===
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    set_dark_style(ax3, is_3d=True)

    # Create pseudopotential surface
    x_pot = np.linspace(-1.5, 1.5, 100)
    y_pot = np.linspace(-1.5, 1.5, 100)
    X, Y = np.meshgrid(x_pot, y_pot)

    # Quadrupole pseudopotential: Φ* ∝ (x² + y²) for 2D trap
    # With RF modulation effect
    q_eff = 0.4
    Phi_pseudo = 0.5 * q_eff**2 * (X**2 + Y**2)

    # Add saddle point nature of true potential
    Phi_true = 0.3 * (X**2 - Y**2)

    # Combined effective potential
    Phi_eff = Phi_pseudo + 0.2 * Phi_true

    surf = ax3.plot_surface(X, Y, Phi_eff, cmap='coolwarm', alpha=0.85,
                            linewidth=0.1, antialiased=True)

    # Add contour projection
    ax3.contour(X, Y, Phi_eff, zdir='z', offset=Phi_eff.min()-0.2,
                cmap='coolwarm', alpha=0.5, levels=15)

    # Mark trap center
    ax3.scatter([0], [0], [0], color=COLORS['accent1'], s=100, marker='*',
                label='Trap Center')

    ax3.set_xlabel('x/r₀', fontsize=9, color=COLORS['text'])
    ax3.set_ylabel('y/r₀', fontsize=9, color=COLORS['text'])
    ax3.set_zlabel('Φ* (arb. units)', fontsize=9, color=COLORS['text'])
    ax3.set_title('Pseudopotential Landscape', fontsize=11,
                  color=COLORS['accent1'], pad=10)

    ax3.text2D(0.02, 0.95, r'$\Phi^* = \frac{q^2V^2}{4m\Omega^2 r_0^2}(x^2 + y^2)$',
              transform=ax3.transAxes, fontsize=10, color=COLORS['accent2'],
              bbox=dict(boxstyle='round', facecolor=COLORS['background'], alpha=0.8))

    # === Chart 4: Secular Motion Time Series ===
    ax4 = fig.add_subplot(2, 2, 4)
    set_dark_style(ax4)

    # Time axis
    t_sec = np.linspace(0, 20, 1000)

    # Secular frequencies for different directions
    omega_x = 0.3
    omega_y = 0.35
    omega_z = 0.15

    # Secular motion with damping
    u_x = np.exp(-0.05 * t_sec) * np.cos(omega_x * t_sec)
    u_y = 0.8 * np.exp(-0.05 * t_sec) * np.cos(omega_y * t_sec + np.pi/3)
    u_z = 0.6 * np.exp(-0.03 * t_sec) * np.cos(omega_z * t_sec + np.pi/6)

    # Add micromotion superimposed
    omega_rf = 5.0
    u_x_micro = u_x * (1 + 0.15 * np.cos(omega_rf * t_sec))

    ax4.plot(t_sec, u_x_micro, color=COLORS['primary'], linewidth=1.5,
             label='x (with micromotion)', alpha=0.9)
    ax4.plot(t_sec, u_x, color=COLORS['accent1'], linewidth=2,
             label='x (secular envelope)', linestyle='--')
    ax4.plot(t_sec, u_y, color=COLORS['secondary'], linewidth=1.5,
             label='y (secular)', alpha=0.8)
    ax4.plot(t_sec, u_z, color=COLORS['tertiary'], linewidth=1.5,
             label='z (secular)', alpha=0.8)

    ax4.axhline(y=0, color=COLORS['grid'], linestyle='-', alpha=0.3)
    ax4.fill_between(t_sec, -np.exp(-0.05 * t_sec), np.exp(-0.05 * t_sec),
                     alpha=0.1, color=COLORS['accent1'])

    ax4.set_xlabel('Time (Ω⁻¹ units)', fontsize=10, color=COLORS['text'])
    ax4.set_ylabel('Displacement (r₀ units)', fontsize=10, color=COLORS['text'])
    ax4.set_title('Secular Motion: ω = βΩ/2', fontsize=11,
                  color=COLORS['accent1'], pad=10)
    ax4.legend(loc='upper right', facecolor=COLORS['background'],
               labelcolor=COLORS['text'], fontsize=8)
    ax4.set_xlim(0, 20)
    ax4.set_ylim(-1.3, 1.3)

    # Add annotations
    ax4.annotate('Damping\nenvelope', xy=(15, np.exp(-0.05*15)),
                xytext=(17, 0.8), fontsize=9, color=COLORS['accent2'],
                arrowprops=dict(arrowstyle='->', color=COLORS['accent2']))

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.savefig(save_path, dpi=300, facecolor=COLORS['background'],
                edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def generate_panel2_categorical(save_path):
    """
    Panel 2: Ion as Discrete State
    - State space diagram (network visualization)
    - Transition probability matrix (heatmap)
    - Fragmentation tree (3D)
    - Energy level diagram
    """
    fig = plt.figure(figsize=(16, 14))
    fig.patch.set_facecolor(COLORS['background'])

    # Title
    fig.suptitle('Ion as Discrete State\nCategorical Chemical Description',
                 fontsize=16, color=COLORS['text'], fontweight='bold', y=0.98)

    # === Chart 1: State Space Network ===
    ax1 = fig.add_subplot(2, 2, 1)
    set_dark_style(ax1)

    # Define chemical states
    states = {
        'M⁺': (0.5, 0.8),
        '[M-H₂O]⁺': (0.2, 0.5),
        '[M-NH₃]⁺': (0.8, 0.5),
        '[M-CO]⁺': (0.35, 0.2),
        'a₁': (0.65, 0.2),
        'b₁': (0.1, 0.1),
        'y₁': (0.9, 0.1)
    }

    # State populations (categorical probabilities)
    populations = {
        'M⁺': 0.25,
        '[M-H₂O]⁺': 0.20,
        '[M-NH₃]⁺': 0.18,
        '[M-CO]⁺': 0.12,
        'a₁': 0.10,
        'b₁': 0.08,
        'y₁': 0.07
    }

    # Transitions
    transitions = [
        ('M⁺', '[M-H₂O]⁺', 0.35),
        ('M⁺', '[M-NH₃]⁺', 0.30),
        ('M⁺', '[M-CO]⁺', 0.15),
        ('[M-H₂O]⁺', 'a₁', 0.25),
        ('[M-NH₃]⁺', 'a₁', 0.20),
        ('[M-H₂O]⁺', 'b₁', 0.15),
        ('[M-CO]⁺', 'y₁', 0.20),
        ('[M-NH₃]⁺', 'y₁', 0.15)
    ]

    # Draw transitions
    for start, end, prob in transitions:
        x1, y1 = states[start]
        x2, y2 = states[end]
        ax1.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=COLORS['accent2'],
                                   alpha=prob*2, lw=prob*5,
                                   connectionstyle='arc3,rad=0.1'))

    # Draw states
    for state, (x, y) in states.items():
        pop = populations[state]
        size = 800 + pop * 2000
        ax1.scatter(x, y, s=size, c=[COLORS['primary']], alpha=0.8,
                   edgecolor=COLORS['accent1'], linewidth=2)
        ax1.text(x, y, state, ha='center', va='center', fontsize=9,
                color=COLORS['text'], fontweight='bold')
        ax1.text(x, y-0.08, f'P={pop:.2f}', ha='center', va='top',
                fontsize=7, color=COLORS['accent3'])

    ax1.set_xlim(-0.1, 1.1)
    ax1.set_ylim(-0.05, 1.0)
    ax1.axis('off')
    ax1.set_title('State Space: Discrete Chemical Categories', fontsize=11,
                  color=COLORS['accent1'], pad=10)

    # Add equation box
    ax1.text(0.02, 0.98, r'$P(S_j|S_i) = T_{ij}$' + '\nTransition Probabilities',
            transform=ax1.transAxes, fontsize=9, color=COLORS['accent2'],
            bbox=dict(boxstyle='round', facecolor=COLORS['background'], alpha=0.8),
            verticalalignment='top')

    # === Chart 2: Transition Probability Matrix (Heatmap) ===
    ax2 = fig.add_subplot(2, 2, 2)
    set_dark_style(ax2)

    # Create transition matrix
    state_labels = list(states.keys())
    n_states = len(state_labels)
    T = np.zeros((n_states, n_states))

    # Fill diagonal (self-transitions / stability)
    np.fill_diagonal(T, 0.3)

    # Fill off-diagonal from transitions
    state_idx = {s: i for i, s in enumerate(state_labels)}
    for start, end, prob in transitions:
        T[state_idx[start], state_idx[end]] = prob

    # Normalize rows
    T = T / T.sum(axis=1, keepdims=True)

    im2 = ax2.imshow(T, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.5)

    ax2.set_xticks(range(n_states))
    ax2.set_yticks(range(n_states))
    ax2.set_xticklabels(state_labels, rotation=45, ha='right', fontsize=8, color=COLORS['text'])
    ax2.set_yticklabels(state_labels, fontsize=8, color=COLORS['text'])

    # Add values
    for i in range(n_states):
        for j in range(n_states):
            if T[i, j] > 0.05:
                ax2.text(j, i, f'{T[i,j]:.2f}', ha='center', va='center',
                        fontsize=7, color='white' if T[i,j] > 0.25 else 'black')

    ax2.set_xlabel('To State', fontsize=10, color=COLORS['text'])
    ax2.set_ylabel('From State', fontsize=10, color=COLORS['text'])
    ax2.set_title('Transition Matrix T = {Tᵢⱼ}', fontsize=11,
                  color=COLORS['accent1'], pad=10)

    cbar2 = plt.colorbar(im2, ax=ax2, label='Probability')
    cbar2.ax.yaxis.label.set_color(COLORS['text'])
    cbar2.ax.tick_params(colors=COLORS['text'])

    # === Chart 3: 3D Fragmentation Tree ===
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    set_dark_style(ax3, is_3d=True)

    # Fragmentation tree coordinates (x=generation, y=branch, z=m/z)
    nodes = {
        'M⁺': (0, 0, 500),
        '[M-H₂O]⁺': (1, -0.5, 482),
        '[M-NH₃]⁺': (1, 0.5, 483),
        '[M-CO]⁺': (1, 0, 472),
        'a₁': (2, -0.7, 350),
        'b₁': (2, -0.3, 250),
        'y₁': (2, 0.3, 180),
        'imm': (2, 0.7, 130),
        'a₂': (3, -0.5, 220),
        'y₂': (3, 0.5, 100)
    }

    # Node intensities
    intensities = {
        'M⁺': 1.0, '[M-H₂O]⁺': 0.6, '[M-NH₃]⁺': 0.5, '[M-CO]⁺': 0.3,
        'a₁': 0.4, 'b₁': 0.3, 'y₁': 0.35, 'imm': 0.2, 'a₂': 0.2, 'y₂': 0.25
    }

    # Draw edges (fragmentation paths)
    edges = [
        ('M⁺', '[M-H₂O]⁺'), ('M⁺', '[M-NH₃]⁺'), ('M⁺', '[M-CO]⁺'),
        ('[M-H₂O]⁺', 'a₁'), ('[M-H₂O]⁺', 'b₁'),
        ('[M-NH₃]⁺', 'y₁'), ('[M-CO]⁺', 'imm'),
        ('a₁', 'a₂'), ('y₁', 'y₂')
    ]

    for start, end in edges:
        x1, y1, z1 = nodes[start]
        x2, y2, z2 = nodes[end]
        ax3.plot([x1, x2], [y1, y2], [z1, z2],
                color=COLORS['accent2'], alpha=0.6, linewidth=2)

    # Draw nodes
    for name, (x, y, z) in nodes.items():
        intensity = intensities[name]
        ax3.scatter(x, y, z, s=intensity*400, c=[COLORS['primary']],
                   alpha=0.8, edgecolor=COLORS['accent1'], linewidth=2)
        ax3.text(x, y, z+20, name, fontsize=8, color=COLORS['text'], ha='center')

    ax3.set_xlabel('Generation', fontsize=9, color=COLORS['text'])
    ax3.set_ylabel('Branch', fontsize=9, color=COLORS['text'])
    ax3.set_zlabel('m/z', fontsize=9, color=COLORS['text'])
    ax3.set_title('3D Fragmentation Tree', fontsize=11,
                  color=COLORS['accent1'], pad=10)

    # === Chart 4: Energy Level Diagram ===
    ax4 = fig.add_subplot(2, 2, 4)
    set_dark_style(ax4)

    # Energy levels (internal energy)
    levels = {
        'M⁺': 0,
        'TS₁': 1.2,
        '[M-H₂O]⁺': 0.8,
        '[M-NH₃]⁺': 0.9,
        'TS₂': 2.0,
        'a₁ + y₁': 1.5,
        'b₁ + y₂': 1.7
    }

    # Draw levels
    x_positions = {
        'M⁺': 0.5, 'TS₁': 1.5, '[M-H₂O]⁺': 2.5, '[M-NH₃]⁺': 3.0,
        'TS₂': 4.0, 'a₁ + y₁': 5.0, 'b₁ + y₂': 5.5
    }

    for name, energy in levels.items():
        x = x_positions[name]
        is_ts = 'TS' in name
        color = COLORS['quaternary'] if is_ts else COLORS['primary']
        linestyle = '--' if is_ts else '-'
        ax4.hlines(energy, x-0.3, x+0.3, color=color, linewidth=3, linestyle=linestyle)
        ax4.text(x, energy+0.1, name, ha='center', va='bottom', fontsize=8,
                color=COLORS['text'], rotation=45 if len(name) > 5 else 0)

    # Draw arrows for transitions
    arrows = [
        (0.5, 0, 1.5, 1.2),
        (1.5, 1.2, 2.5, 0.8),
        (1.5, 1.2, 3.0, 0.9),
        (2.5, 0.8, 4.0, 2.0),
        (4.0, 2.0, 5.0, 1.5),
        (4.0, 2.0, 5.5, 1.7)
    ]

    for x1, y1, x2, y2 in arrows:
        ax4.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=COLORS['accent2'],
                                   alpha=0.7, lw=1.5))

    ax4.set_xlim(-0.2, 6.2)
    ax4.set_ylim(-0.3, 2.5)
    ax4.set_xlabel('Reaction Coordinate', fontsize=10, color=COLORS['text'])
    ax4.set_ylabel('Internal Energy (eV)', fontsize=10, color=COLORS['text'])
    ax4.set_title('Energy Level Diagram: State Transitions', fontsize=11,
                  color=COLORS['accent1'], pad=10)

    # Add legend
    ax4.plot([], [], color=COLORS['primary'], linewidth=3, label='Stable State')
    ax4.plot([], [], color=COLORS['quaternary'], linewidth=3, linestyle='--', label='Transition State')
    ax4.legend(loc='upper left', facecolor=COLORS['background'],
               labelcolor=COLORS['text'], fontsize=9)

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.savefig(save_path, dpi=300, facecolor=COLORS['background'],
                edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def generate_panel3_partition(save_path):
    """
    Panel 3: Ion as Partition Coordinates
    - 3D quantum number space (n, l, m)
    - Capacity formula visualization
    - Phase space partitioning (heatmap)
    - Selection rules matrix
    """
    fig = plt.figure(figsize=(16, 14))
    fig.patch.set_facecolor(COLORS['background'])

    # Title
    fig.suptitle('Ion as Partition Coordinates\nQuantum-Mechanical State Description',
                 fontsize=16, color=COLORS['text'], fontweight='bold', y=0.98)

    # === Chart 1: 3D Quantum Number Space ===
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    set_dark_style(ax1, is_3d=True)

    # Generate quantum states (n, l, m)
    states_3d = []
    colors_3d = []
    sizes_3d = []

    for n in range(1, 5):  # Principal quantum number
        for l in range(n):  # Angular momentum
            for m in range(-l, l+1):  # Magnetic
                states_3d.append((n, l, m))
                # Color by energy (n) and size by degeneracy
                colors_3d.append(n)
                sizes_3d.append(50 + (2*l + 1) * 30)

    states_3d = np.array(states_3d)

    scatter = ax1.scatter(states_3d[:, 0], states_3d[:, 1], states_3d[:, 2],
                         c=colors_3d, cmap='viridis', s=sizes_3d, alpha=0.8,
                         edgecolor='white', linewidth=0.5)

    # Add shell boundaries
    for n in range(1, 5):
        theta = np.linspace(0, 2*np.pi, 50)
        r = n - 0.5
        x_shell = np.full_like(theta, n)
        y_shell = r * np.cos(theta) / 2
        z_shell = r * np.sin(theta)
        ax1.plot(x_shell, np.abs(y_shell), z_shell, color=COLORS['grid'],
                alpha=0.3, linestyle='--')

    ax1.set_xlabel('n (Principal)', fontsize=9, color=COLORS['text'])
    ax1.set_ylabel('ℓ (Angular)', fontsize=9, color=COLORS['text'])
    ax1.set_zlabel('m (Magnetic)', fontsize=9, color=COLORS['text'])
    ax1.set_title('Quantum Number Space (n, ℓ, m)', fontsize=11,
                  color=COLORS['accent1'], pad=10)

    # Add colorbar
    cbar1 = plt.colorbar(scatter, ax=ax1, shrink=0.6, label='n (shell)')
    cbar1.ax.yaxis.label.set_color(COLORS['text'])
    cbar1.ax.tick_params(colors=COLORS['text'])

    ax1.text2D(0.02, 0.95, r'$(n, \ell, m, s)$: Partition Address',
              transform=ax1.transAxes, fontsize=10, color=COLORS['accent2'],
              bbox=dict(boxstyle='round', facecolor=COLORS['background'], alpha=0.8))

    # === Chart 2: Capacity Formula Visualization ===
    ax2 = fig.add_subplot(2, 2, 2)
    set_dark_style(ax2)

    # C(n) = 2n² capacity formula
    n_vals = np.arange(1, 8)
    capacity = 2 * n_vals**2
    cumulative = np.cumsum(capacity)

    # Bar chart with gradient
    bars = ax2.bar(n_vals, capacity, color=plt.cm.viridis(n_vals/7),
                   edgecolor='white', linewidth=1.5, alpha=0.8)

    # Add cumulative line
    ax2_twin = ax2.twinx()
    ax2_twin.plot(n_vals, cumulative, 'o-', color=COLORS['accent2'],
                  linewidth=2, markersize=8, label='Cumulative')
    ax2_twin.set_ylabel('Cumulative States', fontsize=10, color=COLORS['accent2'])
    ax2_twin.tick_params(axis='y', colors=COLORS['accent2'])
    ax2_twin.spines['right'].set_color(COLORS['accent2'])

    # Add capacity values on bars
    for i, (n, c) in enumerate(zip(n_vals, capacity)):
        ax2.text(n, c + 1, f'{c}', ha='center', va='bottom',
                fontsize=9, color=COLORS['text'], fontweight='bold')

    ax2.set_xlabel('Principal Quantum Number (n)', fontsize=10, color=COLORS['text'])
    ax2.set_ylabel('Capacity C(n) = 2n²', fontsize=10, color=COLORS['text'])
    ax2.set_title('Partition Capacity Formula', fontsize=11,
                  color=COLORS['accent1'], pad=10)
    ax2.set_xticks(n_vals)

    # Add equation
    ax2.text(0.98, 0.95, r'$C(n) = 2n^2$' + '\nShell Capacity',
            transform=ax2.transAxes, fontsize=10, color=COLORS['accent1'],
            ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor=COLORS['background'], alpha=0.8))

    # Add shell names
    shell_names = ['K', 'L', 'M', 'N', 'O', 'P', 'Q']
    for n, name in zip(n_vals, shell_names):
        ax2.text(n, -3, name, ha='center', va='top', fontsize=9,
                color=COLORS['accent3'], fontweight='bold')

    # === Chart 3: Phase Space Partitioning (Heatmap) ===
    ax3 = fig.add_subplot(2, 2, 3)
    set_dark_style(ax3)

    # Create phase space grid
    q = np.linspace(-3, 3, 100)  # Position
    p = np.linspace(-3, 3, 100)  # Momentum
    Q, P = np.meshgrid(q, p)

    # Action variable J = (q² + p²)/2
    J = (Q**2 + P**2) / 2

    # Quantized regions: J = (n + 1/2)ℏ
    n_quantum = np.floor(J)
    n_quantum = np.clip(n_quantum, 0, 6)

    # Add discrete boundaries
    im3 = ax3.imshow(n_quantum, extent=[q.min(), q.max(), p.min(), p.max()],
                     origin='lower', cmap='tab10', aspect='equal')

    # Draw quantization boundaries (circles)
    for n in range(1, 7):
        circle = plt.Circle((0, 0), np.sqrt(2*n), fill=False,
                           color='white', linewidth=1.5, linestyle='--', alpha=0.7)
        ax3.add_patch(circle)
        ax3.text(np.sqrt(n), np.sqrt(n), f'n={n}', fontsize=8,
                color='white', fontweight='bold')

    ax3.set_xlabel('Position q (√ℏ/mω)', fontsize=10, color=COLORS['text'])
    ax3.set_ylabel('Momentum p (√ℏmω)', fontsize=10, color=COLORS['text'])
    ax3.set_title('Phase Space Partitioning', fontsize=11,
                  color=COLORS['accent1'], pad=10)

    cbar3 = plt.colorbar(im3, ax=ax3, label='Quantum Number n')
    cbar3.ax.yaxis.label.set_color(COLORS['text'])
    cbar3.ax.tick_params(colors=COLORS['text'])

    # Add equation
    ax3.text(0.02, 0.98, r'$J = \oint p \, dq = 2\pi\hbar(n + \frac{1}{2})$',
            transform=ax3.transAxes, fontsize=9, color=COLORS['accent2'],
            va='top',
            bbox=dict(boxstyle='round', facecolor=COLORS['background'], alpha=0.8))

    # === Chart 4: Selection Rules Matrix ===
    ax4 = fig.add_subplot(2, 2, 4)
    set_dark_style(ax4)

    # Selection rules: allowed transitions between states
    # Using (n, l) as composite index
    states = [(1,0), (2,0), (2,1), (3,0), (3,1), (3,2), (4,0), (4,1)]
    state_labels = [f'({n},{l})' for n, l in states]
    n_states = len(states)

    # Create selection rule matrix
    # Δl = ±1 for electric dipole transitions
    selection = np.zeros((n_states, n_states))

    for i, (n1, l1) in enumerate(states):
        for j, (n2, l2) in enumerate(states):
            if n1 != n2:  # Different principal quantum numbers
                if abs(l1 - l2) == 1:  # Δl = ±1
                    # Transition strength (simplified)
                    selection[i, j] = 1.0 / abs(n1 - n2)
                elif abs(l1 - l2) == 0:  # Δl = 0 (forbidden for electric dipole)
                    selection[i, j] = 0.1  # Very weak

    im4 = ax4.imshow(selection, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    ax4.set_xticks(range(n_states))
    ax4.set_yticks(range(n_states))
    ax4.set_xticklabels(state_labels, rotation=45, ha='right', fontsize=8, color=COLORS['text'])
    ax4.set_yticklabels(state_labels, fontsize=8, color=COLORS['text'])

    # Add values for allowed transitions
    for i in range(n_states):
        for j in range(n_states):
            if selection[i, j] > 0.15:
                ax4.text(j, i, '✓', ha='center', va='center',
                        fontsize=12, color='white', fontweight='bold')
            elif 0 < selection[i, j] <= 0.15:
                ax4.text(j, i, '×', ha='center', va='center',
                        fontsize=10, color='gray')

    ax4.set_xlabel('Final State (n, ℓ)', fontsize=10, color=COLORS['text'])
    ax4.set_ylabel('Initial State (n, ℓ)', fontsize=10, color=COLORS['text'])
    ax4.set_title('Selection Rules: Δℓ = ±1', fontsize=11,
                  color=COLORS['accent1'], pad=10)

    cbar4 = plt.colorbar(im4, ax=ax4, label='Transition Strength')
    cbar4.ax.yaxis.label.set_color(COLORS['text'])
    cbar4.ax.tick_params(colors=COLORS['text'])

    # Add rule annotation
    ax4.text(0.02, 0.98, 'Electric Dipole:\nΔℓ = ±1\nΔm = 0, ±1',
            transform=ax4.transAxes, fontsize=9, color=COLORS['accent2'],
            va='top',
            bbox=dict(boxstyle='round', facecolor=COLORS['background'], alpha=0.8))

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.savefig(save_path, dpi=300, facecolor=COLORS['background'],
                edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def main():
    """Generate all three ion representation panels."""
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'figures')
    os.makedirs(output_dir, exist_ok=True)

    print("Generating Ion Representation Panels...")
    print("=" * 50)

    # Generate Panel 1: Ion as Oscillating Object
    panel1_path = os.path.join(output_dir, 'ion_representation_oscillatory.png')
    generate_panel1_oscillatory(panel1_path)

    # Generate Panel 2: Ion as Discrete State
    panel2_path = os.path.join(output_dir, 'ion_representation_categorical.png')
    generate_panel2_categorical(panel2_path)

    # Generate Panel 3: Ion as Partition Coordinates
    panel3_path = os.path.join(output_dir, 'ion_representation_partition.png')
    generate_panel3_partition(panel3_path)

    print("=" * 50)
    print("All panels generated successfully!")

    # Create summary JSON
    summary = {
        "panels": [
            {
                "name": "Ion as Oscillating Object",
                "file": "ion_representation_oscillatory.png",
                "charts": [
                    "3D RF Trajectory (secular + micromotion)",
                    "Mathieu Stability Diagram (heatmap)",
                    "3D Pseudopotential Landscape",
                    "Secular Motion Time Series"
                ],
                "key_equations": [
                    "d²u/dξ² + [a - 2q cos(2ξ)]u = 0",
                    "ω_secular = βΩ/2",
                    "a = 8eU/(mΩ²r₀²), q = 4eV/(mΩ²r₀²)"
                ]
            },
            {
                "name": "Ion as Discrete State",
                "file": "ion_representation_categorical.png",
                "charts": [
                    "State Space Network (chemical categories)",
                    "Transition Probability Matrix (heatmap)",
                    "3D Fragmentation Tree",
                    "Energy Level Diagram"
                ],
                "key_concepts": [
                    "Categorical probabilities P(Sj|Si)",
                    "Transition matrix T = {Tij}",
                    "Fragmentation pathways",
                    "Activation energies"
                ]
            },
            {
                "name": "Ion as Partition Coordinates",
                "file": "ion_representation_partition.png",
                "charts": [
                    "3D Quantum Number Space (n, ℓ, m)",
                    "Capacity Formula C(n) = 2n²",
                    "Phase Space Partitioning (heatmap)",
                    "Selection Rules Matrix"
                ],
                "key_equations": [
                    "(n, ℓ, m, s) partition address",
                    "C(n) = 2n² capacity formula",
                    "J = ∮p dq = 2πℏ(n + ½)",
                    "Δℓ = ±1 selection rule"
                ]
            }
        ],
        "equivalence_principle": "All three descriptions are mathematically equivalent representations of the same physical ion"
    }

    import json
    summary_path = os.path.join(output_dir, 'ion_representation_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {summary_path}")

if __name__ == '__main__':
    main()
