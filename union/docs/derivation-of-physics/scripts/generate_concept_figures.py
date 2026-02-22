"""
Generate focused concept-to-consequence figures for the Derivation of Physics paper.

Figure 1: Poincaré recurrence → oscillation necessity
Figure 2: Categorical structure → temporal emergence
Figure 3: Partition geometry → spatial coordinates
Figure 4: Mode occupation → matter/dark matter split
Figure 5: Cross-scale coupling → force hierarchy
Figure 6: Categorical exhaustion → cyclic cosmology
Figure 7: Complete derivation flowchart (axioms → reality)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import (Rectangle, Circle, FancyBboxPatch, FancyArrowPatch,
                                 Wedge, Polygon, Arc, PathPatch)
from matplotlib.path import Path
from matplotlib.colors import LinearSegmentedColormap, Normalize
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
try:
    from scipy.special import sph_harm_y as sph_harm
except ImportError:
    from scipy.special import sph_harm
import os

# Style settings
plt.style.use('default')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['figure.facecolor'] = 'white'

# Color palette
COLORS = {
    'primary': '#1976D2',
    'secondary': '#FF5722',
    'accent': '#4CAF50',
    'dark': '#263238',
    'light': '#ECEFF1',
    'warning': '#FFC107',
    'success': '#4CAF50',
    'danger': '#F44336'
}


def figure_1_poincare_to_oscillation():
    """Figure 1: Poincaré recurrence → oscillation necessity"""
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Panel A: Bounded phase space with trajectories
    ax1 = fig.add_subplot(gs[0, 0])

    # Draw bounded region
    theta = np.linspace(0, 2*np.pi, 100)
    ax1.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=3, label='Boundary')
    ax1.fill(np.cos(theta), np.sin(theta), alpha=0.1, color='blue')

    # Draw recurrent trajectory
    t = np.linspace(0, 50, 2000)
    x = 0.7 * np.cos(t + 0.3*np.sin(2.7*t))
    y = 0.7 * np.sin(t + 0.3*np.cos(1.3*t))

    colors = plt.cm.viridis(np.linspace(0, 1, len(t)))
    for i in range(0, len(t)-1, 10):
        ax1.plot(x[i:i+12], y[i:i+12], color=colors[i], linewidth=0.8, alpha=0.7)

    ax1.plot(x[0], y[0], 'go', markersize=12, zorder=5, label='Initial')
    ax1.plot(x[-1], y[-1], 'r^', markersize=10, zorder=5, label='Returns')

    ax1.set_xlim(-1.3, 1.3)
    ax1.set_ylim(-1.3, 1.3)
    ax1.set_aspect('equal')
    ax1.set_xlabel('Position q')
    ax1.set_ylabel('Momentum p')
    ax1.set_title('A. Bounded Phase Space\n(Finite Volume)', fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)

    # Panel B: Poincaré theorem statement
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')

    # Theorem box
    theorem_text = (
        "POINCARÉ RECURRENCE THEOREM\n\n"
        "Given:\n"
        "• Bounded phase space M\n"
        "• Measure-preserving dynamics φₜ\n"
        "• Finite measure μ(M) < ∞\n\n"
        "Then:\n"
        "Almost every trajectory returns\n"
        "arbitrarily close to its origin:\n\n"
        r"$\liminf_{t→∞} d(φₜ(x), x) = 0$"
    )

    box = FancyBboxPatch((0.05, 0.1), 0.9, 0.8,
                         boxstyle="round,pad=0.02",
                         facecolor='#E3F2FD', edgecolor=COLORS['primary'], linewidth=2)
    ax2.add_patch(box)
    ax2.text(0.5, 0.55, theorem_text, ha='center', va='center', fontsize=10,
            transform=ax2.transAxes, family='serif')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_title('B. The Recurrence Theorem', fontweight='bold')

    # Panel C: Why only oscillatory works
    ax3 = fig.add_subplot(gs[0, 2])

    dynamics = ['Static', 'Monotonic', 'Chaotic', 'Oscillatory']
    issues = ['No dynamics\n(no recurrence)', 'Escapes\nboundary',
              'Destroys\nconsistency', 'VALID\n(returns)']
    colors = [COLORS['danger'], COLORS['danger'], COLORS['danger'], COLORS['success']]
    symbols = ['X', 'X', 'X', 'OK']

    y_pos = np.arange(len(dynamics))
    bars = ax3.barh(y_pos, [0.8]*4, color=colors, edgecolor='black', linewidth=2, height=0.6)

    for i, (dyn, issue, sym) in enumerate(zip(dynamics, issues, symbols)):
        ax3.text(-0.05, i, dyn, ha='right', va='center', fontsize=11, fontweight='bold')
        ax3.text(0.4, i, issue, ha='center', va='center', fontsize=9)
        ax3.text(0.9, i, sym, ha='center', va='center', fontsize=16,
                color='white' if sym == '✗' else 'white', fontweight='bold')

    ax3.set_xlim(-0.5, 1.0)
    ax3.set_ylim(-0.5, 3.5)
    ax3.axis('off')
    ax3.set_title('C. Only Oscillatory Satisfies All', fontweight='bold')

    # Panel D: Oscillatory dynamics example
    ax4 = fig.add_subplot(gs[1, 0])

    t = np.linspace(0, 20, 500)
    y = np.sin(t) + 0.3*np.sin(2.5*t) + 0.15*np.sin(6*t)

    ax4.plot(t, y, color=COLORS['primary'], linewidth=2)
    ax4.fill_between(t, y, alpha=0.2, color=COLORS['primary'])
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Mark recurrence
    peaks = [3.14, 9.42, 15.7]
    for p in peaks:
        ax4.axvline(x=p, color=COLORS['secondary'], linestyle=':', alpha=0.7)

    ax4.set_xlabel('Time t')
    ax4.set_ylabel('Amplitude')
    ax4.set_title('D. Oscillatory Returns to Origin', fontweight='bold')
    ax4.set_xlim(0, 20)

    # Panel E: The derivation arrow
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.axis('off')

    # Left box: Premise
    premise_box = FancyBboxPatch((0.0, 0.3), 0.35, 0.4,
                                  boxstyle="round,pad=0.02",
                                  facecolor='#E8F5E9', edgecolor=COLORS['accent'], linewidth=2)
    ax5.add_patch(premise_box)
    ax5.text(0.175, 0.5, 'Bounded\nPhase Space\n+\nConsistency',
            ha='center', va='center', fontsize=11, fontweight='bold')

    # Arrow
    ax5.annotate('', xy=(0.6, 0.5), xytext=(0.4, 0.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=4))
    ax5.text(0.5, 0.62, 'IMPLIES', ha='center', fontsize=10, fontweight='bold')

    # Right box: Conclusion
    conclusion_box = FancyBboxPatch((0.65, 0.3), 0.35, 0.4,
                                    boxstyle="round,pad=0.02",
                                    facecolor='#FFF3E0', edgecolor=COLORS['secondary'], linewidth=2)
    ax5.add_patch(conclusion_box)
    ax5.text(0.825, 0.5, 'Oscillatory\nDynamics\n(Unique Mode)',
            ha='center', va='center', fontsize=11, fontweight='bold')

    ax5.set_xlim(-0.05, 1.05)
    ax5.set_ylim(0, 1)
    ax5.set_title('E. The Logical Implication', fontweight='bold')

    # Panel F: Physical consequence
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

    consequences = [
        '• Energy quantization: E = ℏω',
        '• Wave-like behaviour',
        '• Periodic phenomena',
        '• Recurrent states',
        '• Time-reversal symmetry'
    ]

    ax6.text(0.5, 0.85, 'Physical Consequences', ha='center', fontsize=12,
            fontweight='bold', color=COLORS['primary'])

    for i, cons in enumerate(consequences):
        ax6.text(0.1, 0.7 - i*0.12, cons, fontsize=11, va='center')

    # Summary box
    summary_box = FancyBboxPatch((0.05, 0.05), 0.9, 0.2,
                                  boxstyle="round,pad=0.02",
                                  facecolor='#FFEBEE', edgecolor=COLORS['danger'], linewidth=2)
    ax6.add_patch(summary_box)
    ax6.text(0.5, 0.15, 'Reality MUST oscillate\n(no other option)',
            ha='center', va='center', fontsize=11, fontweight='bold', color=COLORS['danger'])

    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
    ax6.set_title('F. Why Reality Oscillates', fontweight='bold')

    plt.suptitle('Figure 1: Poincaré Recurrence → Oscillation Necessity',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('../figures/fig1_poincare_oscillation.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('../figures/fig1_poincare_oscillation.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: fig1_poincare_oscillation.png/pdf")


def figure_2_categorical_temporal():
    """Figure 2: Categorical structure → temporal emergence"""
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Panel A: Continuous oscillation
    ax1 = fig.add_subplot(gs[0, 0])

    t = np.linspace(0, 10, 1000)
    y = np.sin(2*np.pi*t/2) + 0.3*np.sin(2*np.pi*t/0.7)

    ax1.plot(t, y, color=COLORS['primary'], linewidth=1.5, alpha=0.7)
    ax1.set_xlabel('Continuous Variable')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('A. Continuous Oscillation\n(Infinite Resolution)', fontweight='bold')
    ax1.set_xlim(0, 10)

    # Panel B: Categorical discretization
    ax2 = fig.add_subplot(gs[0, 1])

    # Discretize into categories
    n_cats = 6
    bounds = np.linspace(y.min(), y.max(), n_cats + 1)
    y_cat = np.digitize(y, bounds) - 1
    y_cat = np.clip(y_cat, 0, n_cats - 1)

    colors = plt.cm.Set2(np.linspace(0, 1, n_cats))

    for i in range(n_cats):
        mask = y_cat == i
        ax2.scatter(t[mask], y[mask], c=[colors[i]], s=3, alpha=0.8, label=f'C{i}')

    for bound in bounds[1:-1]:
        ax2.axhline(y=bound, color='gray', linestyle='--', alpha=0.3)

    ax2.set_xlabel('Continuous Variable')
    ax2.set_ylabel('Amplitude')
    ax2.set_title('B. Categorical Discretization\n(Finite Observer)', fontweight='bold')
    ax2.set_xlim(0, 10)
    ax2.legend(loc='upper right', fontsize=8, ncol=2)

    # Panel C: Completion order
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')

    # Hasse diagram
    nodes = {
        'C0': (0.5, 0.1),
        'C1': (0.25, 0.35), 'C2': (0.75, 0.35),
        'C3': (0.15, 0.6), 'C4': (0.5, 0.6), 'C5': (0.85, 0.6),
        'C6': (0.35, 0.85), 'C7': (0.65, 0.85)
    }

    edges = [('C0', 'C1'), ('C0', 'C2'), ('C1', 'C3'), ('C1', 'C4'),
             ('C2', 'C4'), ('C2', 'C5'), ('C4', 'C6'), ('C4', 'C7')]

    for e1, e2 in edges:
        ax3.plot([nodes[e1][0], nodes[e2][0]],
                [nodes[e1][1], nodes[e2][1]],
                'gray', linewidth=1.5, zorder=1)

    for node, (x, y) in nodes.items():
        circle = Circle((x, y), 0.06, color=COLORS['primary'], zorder=2)
        ax3.add_patch(circle)
        ax3.text(x, y, node.replace('C', ''), ha='center', va='center',
                fontsize=9, color='white', fontweight='bold', zorder=3)

    ax3.arrow(0.05, 0.1, 0, 0.75, head_width=0.03, head_length=0.03,
             fc='black', ec='black')
    ax3.text(0.02, 0.5, 'Time\n(emergent)', rotation=90, va='center',
            fontsize=10, fontweight='bold')

    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_title('C. Completion Order\n(Partial Ordering <)', fontweight='bold')

    # Panel D: Irreversibility
    ax4 = fig.add_subplot(gs[1, 0])

    t_comp = np.linspace(0, 10, 100)
    completion = np.zeros_like(t_comp)

    # Step function for completion
    transition_times = [1, 2.5, 4, 5.5, 7, 8.5]
    for tt in transition_times:
        completion[t_comp >= tt] += 1

    ax4.step(t_comp, completion, where='post', color=COLORS['accent'], linewidth=2.5)
    ax4.fill_between(t_comp, 0, completion, step='post', alpha=0.2, color=COLORS['accent'])

    # Arrow showing irreversibility
    ax4.annotate('', xy=(8, 5), xytext=(8, 2),
                arrowprops=dict(arrowstyle='-|>', color=COLORS['danger'], lw=2))
    ax4.text(8.3, 3.5, 'Cannot\ndecrease', fontsize=9, color=COLORS['danger'])

    ax4.set_xlabel('Completion Order')
    ax4.set_ylabel('Completed Categories')
    ax4.set_title('D. Categorical Irreversibility\n(Arrow of Time)', fontweight='bold')
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 7)

    # Panel E: The derivation
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.axis('off')

    # Derivation chain
    steps = [
        ('Continuous\nOscillations', '#E8F5E9', COLORS['accent']),
        ('Finite\nObserver', '#E3F2FD', COLORS['primary']),
        ('Categorical\nApproximation', '#FFF3E0', COLORS['secondary']),
        ('Completion\nOrder', '#FFEBEE', COLORS['danger']),
        ('TIME\nEmerges', '#F3E5F5', '#9C27B0')
    ]

    for i, (text, facecolor, edgecolor) in enumerate(steps):
        y = 0.85 - i * 0.18
        box = FancyBboxPatch((0.2, y - 0.07), 0.6, 0.12,
                             boxstyle="round,pad=0.02",
                             facecolor=facecolor, edgecolor=edgecolor, linewidth=2)
        ax5.add_patch(box)
        ax5.text(0.5, y, text, ha='center', va='center', fontsize=10, fontweight='bold')

        if i < len(steps) - 1:
            ax5.annotate('', xy=(0.5, y - 0.09), xytext=(0.5, y - 0.07),
                        arrowprops=dict(arrowstyle='->', color='black', lw=2))

    ax5.set_xlim(0, 1)
    ax5.set_ylim(0, 1)
    ax5.set_title('E. Derivation Chain', fontweight='bold')

    # Panel F: Time is not fundamental
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

    ax6.text(0.5, 0.9, 'KEY INSIGHT', ha='center', fontsize=13,
            fontweight='bold', color=COLORS['danger'])

    insight_text = (
        "Time is NOT a fundamental\n"
        "parameter of the universe.\n\n"
        "Time EMERGES from the\n"
        "completion order of\n"
        "categorical states.\n\n"
        "The 'arrow of time' is\n"
        "categorical irreversibility:\n"
        "μ(C, t₁) = 1 ⟹ μ(C, t₂) = 1\n"
        "for t₂ > t₁"
    )

    ax6.text(0.5, 0.45, insight_text, ha='center', va='center', fontsize=11,
            family='serif', bbox=dict(boxstyle='round', facecolor='#FFF8E1', edgecolor='orange'))

    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
    ax6.set_title('F. Time Emergence', fontweight='bold')

    plt.suptitle('Figure 2: Categorical Structure → Temporal Emergence',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('../figures/fig2_categorical_temporal.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('../figures/fig2_categorical_temporal.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: fig2_categorical_temporal.png/pdf")


def figure_3_partition_spatial():
    """Figure 3: Partition geometry → spatial coordinates"""
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

    # Panel A: Partition coordinate definition
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')

    # Generate valid (n, l, m) coordinates
    coords = []
    colors = []
    for n in range(1, 5):
        for l in range(0, n):
            for m in range(-l, l+1):
                coords.append([n, l, m])
                colors.append(n)

    coords = np.array(coords)
    scatter = ax1.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                         c=colors, cmap='viridis', s=60, edgecolor='black', alpha=0.8)

    ax1.set_xlabel('n (depth)')
    ax1.set_ylabel('l (angular)')
    ax1.set_zlabel('m (orientation)')
    ax1.set_title('A. Partition Coordinates\n(n, l, m, s)', fontweight='bold')

    # Panel B: Constraints
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')

    constraints_text = (
        "PARTITION CONSTRAINTS\n\n"
        "n ∈ ℤ⁺         (depth ≥ 1)\n\n"
        "0 ≤ l ≤ n-1   (angular limit)\n\n"
        "-l ≤ m ≤ +l   (orientation range)\n\n"
        "s = ±½        (chirality/spin)\n\n"
        "━━━━━━━━━━━━━━━━\n"
        "Capacity: 2n² states per shell"
    )

    box = FancyBboxPatch((0.05, 0.1), 0.9, 0.8,
                         boxstyle="round,pad=0.02",
                         facecolor='#E8F5E9', edgecolor=COLORS['accent'], linewidth=2)
    ax2.add_patch(box)
    ax2.text(0.5, 0.5, constraints_text, ha='center', va='center', fontsize=11,
            family='monospace')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_title('B. Geometric Constraints', fontweight='bold')

    # Panel C: Spherical harmonics (angular structure)
    ax3 = fig.add_subplot(gs[0, 2], projection='3d')

    theta = np.linspace(0, np.pi, 50)
    phi = np.linspace(0, 2*np.pi, 100)
    THETA, PHI = np.meshgrid(theta, phi)

    l, m = 2, 1
    Y = sph_harm(m, l, PHI, THETA).real
    R = np.abs(Y) + 0.2

    X = R * np.sin(THETA) * np.cos(PHI)
    Y_coord = R * np.sin(THETA) * np.sin(PHI)
    Z = R * np.cos(THETA)

    ax3.plot_surface(X, Y_coord, Z, cmap='RdBu', alpha=0.8, edgecolor='none')
    ax3.set_title(f'C. Angular Structure\nY₂¹(θ,φ) → 3D Space', fontweight='bold')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('z')

    # Panel D: The mapping to 3D
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.axis('off')

    # Show mapping
    mapping = [
        ('l ∈ {0,1,...,n-1}', '→', 'SO(3) representations'),
        ('m ∈ {-l,...,+l}', '→', '(2l+1) orientations'),
        ('(l, m) together', '→', 'Spherical harmonics'),
        ('n (radial)', '→', 'r ∝ n² extension'),
        ('', '', ''),
        ('RESULT:', '', '3D EUCLIDEAN SPACE')
    ]

    for i, (left, arrow, right) in enumerate(mapping):
        y = 0.85 - i * 0.12
        if arrow:
            ax4.text(0.1, y, left, fontsize=10, va='center', family='monospace')
            ax4.text(0.5, y, arrow, fontsize=12, va='center', ha='center')
            ax4.text(0.6, y, right, fontsize=10, va='center')
        else:
            ax4.text(0.1, y, left, fontsize=12, va='center', fontweight='bold',
                    color=COLORS['primary'])
            ax4.text(0.5, y, right, fontsize=12, va='center', fontweight='bold',
                    color=COLORS['secondary'])

    ax4.set_xlim(0, 1)
    ax4.set_ylim(0.2, 1)
    ax4.set_title('D. Mapping to Space', fontweight='bold')

    # Panel E: Radial shells
    ax5 = fig.add_subplot(gs[1, 1])

    theta = np.linspace(0, 2*np.pi, 100)
    for n in range(1, 5):
        r = n**2 / 16  # Scale for visibility
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        ax5.plot(x, y, linewidth=2, label=f'n={n}, r∝{n**2}')
        ax5.text(r*1.1, 0, f'n={n}', fontsize=9)

    ax5.set_xlim(-1.2, 1.2)
    ax5.set_ylim(-1.2, 1.2)
    ax5.set_aspect('equal')
    ax5.set_xlabel('x')
    ax5.set_ylabel('y')
    ax5.set_title('E. Radial Extension\nr ∝ n² (Bohr-like)', fontweight='bold')
    ax5.legend(loc='upper right', fontsize=8)

    # Panel F: Dimensionality proof
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

    proof_text = (
        "WHY D = 3?\n\n"
        "The constraint structure:\n"
        "• l ∈ {0, 1, ..., n-1}\n"
        "• m ∈ {-l, ..., +l}\n\n"
        "Has exactly 2 angular\n"
        "quantum numbers (l, m).\n\n"
        "This is the UNIQUE\n"
        "signature of SO(3).\n\n"
        "━━━━━━━━━━━━━━━━\n"
        "D = 3 is DERIVED,\n"
        "not assumed!"
    )

    box = FancyBboxPatch((0.05, 0.05), 0.9, 0.9,
                         boxstyle="round,pad=0.02",
                         facecolor='#FFF3E0', edgecolor=COLORS['secondary'], linewidth=2)
    ax6.add_patch(box)
    ax6.text(0.5, 0.5, proof_text, ha='center', va='center', fontsize=10,
            family='serif')
    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
    ax6.set_title('F. Dimensionality Theorem', fontweight='bold')

    plt.suptitle('Figure 3: Partition Geometry → Spatial Coordinates',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('../figures/fig3_partition_spatial.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('../figures/fig3_partition_spatial.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: fig3_partition_spatial.png/pdf")


def figure_4_mode_occupation():
    """Figure 4: Mode occupation → matter/dark matter split"""
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Panel A: Mode space visualization
    ax1 = fig.add_subplot(gs[0, 0])

    # Grid of modes
    np.random.seed(42)
    n_modes = 400
    occupied = np.random.random(n_modes) < 0.05  # 5% occupied

    grid_size = 20
    occupation_grid = occupied.reshape(grid_size, grid_size)

    cmap = LinearSegmentedColormap.from_list('occ', ['#ECEFF1', '#2196F3'])
    ax1.imshow(occupation_grid, cmap=cmap, aspect='equal')

    n_occ = occupied.sum()
    ax1.set_title(f'A. Mode Space\n{n_occ}/{n_modes} occupied ({100*n_occ/n_modes:.0f}%)',
                 fontweight='bold')
    ax1.set_xlabel('Mode index i')
    ax1.set_ylabel('Mode index j')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#ECEFF1', edgecolor='black', label='Unoccupied'),
                       Patch(facecolor='#2196F3', edgecolor='black', label='Occupied')]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=8)

    # Panel B: The logic
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')

    logic_text = (
        "MODE OCCUPATION LOGIC\n\n"
        "Total oscillatory modes: N_total\n\n"
        "Occupied modes: N_occ\n"
        "(carry energy E = ℏω)\n\n"
        "Unoccupied modes: N_unocc\n"
        "(zero-point energy only)\n\n"
        "━━━━━━━━━━━━━━━━━\n"
        "Occupied = VISIBLE MATTER\n"
        "Unoccupied = DARK SECTOR"
    )

    box = FancyBboxPatch((0.05, 0.1), 0.9, 0.8,
                         boxstyle="round,pad=0.02",
                         facecolor='#E3F2FD', edgecolor=COLORS['primary'], linewidth=2)
    ax2.add_patch(box)
    ax2.text(0.5, 0.5, logic_text, ha='center', va='center', fontsize=10,
            family='serif')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_title('B. The Logic', fontweight='bold')

    # Panel C: Pie chart
    ax3 = fig.add_subplot(gs[0, 2])

    sizes = [5, 27, 68]
    labels = ['Visible\nMatter\n5%', 'Dark\nMatter\n27%', 'Dark\nEnergy\n68%']
    colors_pie = ['#2196F3', '#9C27B0', '#424242']
    explode = (0.1, 0.02, 0.02)

    wedges, texts = ax3.pie(sizes, labels=labels, colors=colors_pie,
                            explode=explode, startangle=90,
                            textprops={'fontsize': 10, 'fontweight': 'bold'})

    ax3.set_title('C. Cosmic Composition\n(Observed)', fontweight='bold')

    # Panel D: Exclusion principle
    ax4 = fig.add_subplot(gs[1, 0])

    # Draw orbital boxes for Carbon
    orbitals = ['1s', '2s', '2p']
    capacities = [2, 2, 6]
    occupations = [2, 2, 2]  # Carbon ground state

    y_pos = 0
    for orb, cap, occ in zip(orbitals, capacities, occupations):
        # Draw boxes
        for i in range(cap//2):
            x = i * 0.6
            rect = Rectangle((x, y_pos), 0.5, 0.8,
                            facecolor='white', edgecolor='black', linewidth=2)
            ax4.add_patch(rect)

            # Add electrons
            if 2*i < occ:
                ax4.annotate('', xy=(x+0.25, y_pos+0.7), xytext=(x+0.25, y_pos+0.3),
                            arrowprops=dict(arrowstyle='->', color='blue', lw=2))
            if 2*i+1 < occ:
                ax4.annotate('', xy=(x+0.25, y_pos+0.3), xytext=(x+0.25, y_pos+0.7),
                            arrowprops=dict(arrowstyle='->', color='red', lw=2))

        ax4.text(-0.2, y_pos+0.4, orb, ha='right', va='center', fontsize=11, fontweight='bold')
        y_pos += 1.2

    ax4.set_xlim(-0.5, 2.5)
    ax4.set_ylim(-0.2, 4)
    ax4.axis('off')
    ax4.set_title('D. Exclusion Principle\n(Max 2 per orbital)', fontweight='bold')

    # Panel E: Fermi-Dirac distribution
    ax5 = fig.add_subplot(gs[1, 1])

    E = np.linspace(0, 5, 200)
    mu = 2.5

    for T, color, label in [(0.1, 'blue', 'T→0'), (0.5, 'green', 'T=0.5'), (1.0, 'red', 'T=1.0')]:
        f = 1 / (np.exp((E - mu)/T) + 1)
        ax5.plot(E, f, color=color, linewidth=2, label=label)

    ax5.axvline(x=mu, color='gray', linestyle='--', label=f'μ={mu}')
    ax5.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)

    ax5.fill_between(E, 0, 1/(np.exp((E-mu)/0.1)+1), alpha=0.2, color='blue')

    ax5.set_xlabel('Energy E')
    ax5.set_ylabel('Occupation f(E)')
    ax5.set_title('E. Fermi-Dirac Statistics\n(Fermion occupation)', fontweight='bold')
    ax5.legend(loc='upper right', fontsize=9)
    ax5.set_xlim(0, 5)
    ax5.set_ylim(0, 1.1)

    # Panel F: The prediction
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

    prediction_text = (
        "FRAMEWORK PREDICTION\n\n"
        "Visible matter fraction:\n"
        "~5% of total mode space\n\n"
        "Dark sector:\n"
        "~95% unoccupied modes\n\n"
        "━━━━━━━━━━━━━━━━━\n"
        "MATCHES OBSERVATION!\n\n"
        "Dark matter/energy are NOT\n"
        "exotic new substances—\n"
        "they are the unoccupied\n"
        "portion of oscillatory\n"
        "mode space."
    )

    box = FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
                         boxstyle="round,pad=0.02",
                         facecolor='#E8F5E9', edgecolor=COLORS['accent'], linewidth=2)
    ax6.add_patch(box)
    ax6.text(0.5, 0.5, prediction_text, ha='center', va='center', fontsize=10,
            family='serif')
    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
    ax6.set_title('F. The Prediction', fontweight='bold')

    plt.suptitle('Figure 4: Mode Occupation → Matter/Dark Matter Split',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('../figures/fig4_mode_occupation.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('../figures/fig4_mode_occupation.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: fig4_mode_occupation.png/pdf")


def figure_5_force_hierarchy():
    """Figure 5: Cross-scale coupling → force hierarchy"""
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Panel A: Hierarchical oscillatory modes
    ax1 = fig.add_subplot(gs[0, 0])

    t = np.linspace(0, 10, 1000)

    # Multiple scales
    y1 = np.sin(2*np.pi*t*10)  # Fast
    y2 = np.sin(2*np.pi*t*1)   # Medium
    y3 = np.sin(2*np.pi*t*0.1) # Slow

    ax1.plot(t, y1*0.2 + 2, 'b-', linewidth=1, alpha=0.7, label='Fast (ω₁)')
    ax1.plot(t, y2*0.5 + 1, 'g-', linewidth=1.5, label='Medium (ω₂)')
    ax1.plot(t, y3*0.8 + 0, 'r-', linewidth=2, label='Slow (ω₃)')

    ax1.set_xlabel('Time')
    ax1.set_ylabel('Amplitude (offset)')
    ax1.set_title('A. Hierarchical Oscillations\n(Multiple Timescales)', fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_xlim(0, 10)

    # Panel B: Resonance coupling
    ax2 = fig.add_subplot(gs[0, 1])

    omega_ratio = np.linspace(0.5, 2.0, 200)
    gamma = 0.05
    coupling = 1 / np.sqrt((1 - omega_ratio**2)**2 + (2*gamma*omega_ratio)**2)

    ax2.semilogy(omega_ratio, coupling, color=COLORS['primary'], linewidth=2.5)
    ax2.axvline(x=1.0, color=COLORS['danger'], linestyle='--', linewidth=2, label='Resonance')
    ax2.fill_between(omega_ratio, 1, coupling, where=(coupling>1), alpha=0.2, color=COLORS['primary'])

    ax2.set_xlabel('Frequency Ratio ω₁/ω₂')
    ax2.set_ylabel('Coupling Strength')
    ax2.set_title('B. Resonance Enhancement\n(Maximum at ω₁ = ω₂)', fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.set_xlim(0.5, 2.0)

    # Panel C: Force potentials
    ax3 = fig.add_subplot(gs[0, 2])

    r = np.logspace(-18, -8, 200)

    # Coulomb (1/r)
    V_em = 1 / r
    V_em = V_em / V_em[100]

    # Yukawa (strong)
    m_pion = 1e14
    V_strong = np.exp(-m_pion * r) / r
    V_strong = V_strong / np.nanmax(V_strong) * 100

    # Gravity
    V_grav = 1 / r
    V_grav = V_grav / V_grav[100] * 1e-36

    ax3.loglog(r, V_em, 'b-', linewidth=2, label='EM (1/r)')
    ax3.loglog(r, V_strong, 'r-', linewidth=2, label='Strong (Yukawa)')
    ax3.loglog(r, V_grav, 'purple', linewidth=2, label='Gravity (1/r)')

    ax3.set_xlabel('Distance r (m)')
    ax3.set_ylabel('Relative Strength')
    ax3.set_title('C. Force Range & Strength\n(Different Mediators)', fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.set_xlim(1e-18, 1e-8)
    ax3.set_ylim(1e-40, 1e3)

    # Panel D: Force hierarchy bar chart
    ax4 = fig.add_subplot(gs[1, 0])

    forces = ['Strong', 'EM', 'Weak', 'Gravity']
    log_strengths = [0, -2.1, -6, -39]  # log10 relative to strong
    colors_bar = ['red', 'blue', 'orange', 'purple']

    bars = ax4.barh(range(len(forces)), [40 + s for s in log_strengths],
                   color=colors_bar, edgecolor='black', linewidth=2)

    ax4.set_yticks(range(len(forces)))
    ax4.set_yticklabels(forces, fontsize=11, fontweight='bold')
    ax4.set_xlabel('log₁₀(Coupling Strength) + 40')
    ax4.set_title('D. Force Hierarchy\n(40 Orders of Magnitude!)', fontweight='bold')

    for i, s in enumerate(log_strengths):
        ax4.text(40 + s + 1, i, f'10^{s}', va='center', fontsize=10)

    # Panel E: The explanation
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.axis('off')

    explanation = (
        "WHY THE HIERARCHY?\n\n"
        "Coupling strength depends on:\n"
        "1. Mediator frequency ω_med\n"
        "2. Mode overlap integral\n\n"
        "α ∝ ω²_mediator\n\n"
        "High-frequency mediators:\n"
        "→ Strong coupling (local)\n\n"
        "Low-frequency mediators:\n"
        "→ Weak coupling (global)\n\n"
        "━━━━━━━━━━━━━━━━━\n"
        "Hierarchy is NECESSARY,\n"
        "not accidental!"
    )

    box = FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
                         boxstyle="round,pad=0.02",
                         facecolor='#E3F2FD', edgecolor=COLORS['primary'], linewidth=2)
    ax5.add_patch(box)
    ax5.text(0.5, 0.5, explanation, ha='center', va='center', fontsize=10,
            family='serif')
    ax5.set_xlim(0, 1)
    ax5.set_ylim(0, 1)
    ax5.set_title('E. The Explanation', fontweight='bold')

    # Panel F: Unified picture
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

    # Draw hierarchy
    levels = ['Planck Scale', 'Strong Force', 'EM Force', 'Weak Force', 'Gravity']
    scales = ['10⁻³⁵ m', '10⁻¹⁵ m', '10⁻¹⁰ m', '10⁻¹⁸ m', '∞']

    for i, (level, scale) in enumerate(zip(levels, scales)):
        y = 0.85 - i * 0.17

        # Left: scale
        ax6.text(0.1, y, scale, fontsize=10, va='center', family='monospace')

        # Box
        box = FancyBboxPatch((0.25, y-0.06), 0.5, 0.1,
                             boxstyle="round,pad=0.01",
                             facecolor=plt.cm.plasma(i/5), edgecolor='black', linewidth=1.5)
        ax6.add_patch(box)
        ax6.text(0.5, y, level, ha='center', va='center', fontsize=10,
                fontweight='bold', color='white' if i > 1 else 'black')

        # Arrow
        if i < len(levels) - 1:
            ax6.annotate('', xy=(0.5, y-0.08), xytext=(0.5, y-0.06),
                        arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    ax6.text(0.5, 0.02, 'Same physics, different scales', ha='center',
            fontsize=11, style='italic')

    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
    ax6.set_title('F. Unified Hierarchy', fontweight='bold')

    plt.suptitle('Figure 5: Cross-Scale Coupling → Force Hierarchy',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('../figures/fig5_force_hierarchy.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('../figures/fig5_force_hierarchy.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: fig5_force_hierarchy.png/pdf")


def figure_6_cyclic_cosmology():
    """Figure 6: Categorical exhaustion → cyclic cosmology"""
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Panel A: Configuration space exploration
    ax1 = fig.add_subplot(gs[0, 0])

    np.random.seed(42)
    n_steps = 3000
    steps = np.random.randn(n_steps, 2) * 0.1
    trajectory = np.cumsum(steps, axis=0)

    colors = np.linspace(0, 1, n_steps)
    ax1.scatter(trajectory[:, 0], trajectory[:, 1], c=colors, cmap='viridis',
               s=1, alpha=0.5)
    ax1.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=12,
            label='Start', zorder=5)
    ax1.plot(trajectory[-1, 0], trajectory[-1, 1], 'r^', markersize=10,
            label='Current', zorder=5)

    ax1.set_xlabel('Configuration 1')
    ax1.set_ylabel('Configuration 2')
    ax1.set_title('A. Configuration Exploration\n(Visiting States)', fontweight='bold')
    ax1.legend(loc='upper right')

    # Panel B: Monotonic vs cyclic exploration
    ax2 = fig.add_subplot(gs[0, 1])

    t = np.linspace(0, 10, 200)
    total_configs = 1000

    # Monotonic (polynomial)
    visited_mono = 50 * t**1.5
    visited_mono = np.minimum(visited_mono, total_configs * 0.3)  # Caps out

    # Cyclic (exponential approach)
    visited_cyclic = total_configs * (1 - np.exp(-t/3))

    ax2.plot(t, visited_mono, 'r--', linewidth=2.5, label='Monotonic (incomplete)')
    ax2.plot(t, visited_cyclic, 'g-', linewidth=2.5, label='Cyclic (exhaustive)')
    ax2.axhline(y=total_configs, color='gray', linestyle=':', linewidth=2, label='Total configs')

    ax2.fill_between(t, visited_mono, visited_cyclic, alpha=0.2, color='green')

    ax2.set_xlabel('Time')
    ax2.set_ylabel('Configurations Visited')
    ax2.set_title('B. Exploration Comparison\n(Cyclic Required)', fontweight='bold')
    ax2.legend(loc='lower right', fontsize=9)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 1100)

    # Panel C: The cycle
    ax3 = fig.add_subplot(gs[0, 2])

    theta = np.linspace(0, 2*np.pi, 100)
    r = 1.0
    ax3.plot(r*np.cos(theta), r*np.sin(theta), 'k-', linewidth=3)

    # Phases
    phases = [
        (0, 'Expansion', 'green', 0.3),
        (np.pi/2, 'Heat Death', 'blue', 0.15),
        (np.pi, 'Contraction', 'orange', 0.3),
        (3*np.pi/2, 'Big Bang', 'red', 0.15)
    ]

    for angle, label, color, width in phases:
        px, py = 1.4 * np.cos(angle), 1.4 * np.sin(angle)
        ax3.plot(np.cos(angle), np.sin(angle), 'o', markersize=15, color=color)
        ax3.text(px, py, label, ha='center', va='center', fontsize=10,
                fontweight='bold', color=color)

        # Arc for phase
        arc_start = angle - width * np.pi
        arc_end = angle + width * np.pi
        arc_theta = np.linspace(arc_start, arc_end, 30)
        ax3.plot(1.15*np.cos(arc_theta), 1.15*np.sin(arc_theta), color=color, linewidth=3)

    # Direction arrows
    for i in range(4):
        angle = i * np.pi/2 + np.pi/4
        ax3.annotate('', xy=(0.85*np.cos(angle+0.15), 0.85*np.sin(angle+0.15)),
                    xytext=(0.85*np.cos(angle-0.15), 0.85*np.sin(angle-0.15)),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2))

    ax3.set_xlim(-2, 2)
    ax3.set_ylim(-2, 2)
    ax3.set_aspect('equal')
    ax3.axis('off')
    ax3.set_title('C. The Cosmic Cycle\n(Categorical Exhaustion)', fontweight='bold')

    # Panel D: Entropy evolution
    ax4 = fig.add_subplot(gs[1, 0])

    t_cycle = np.linspace(0, 4, 400)  # Two cycles
    S = np.abs(np.sin(np.pi * t_cycle))

    ax4.plot(t_cycle, S, color=COLORS['primary'], linewidth=2.5)
    ax4.fill_between(t_cycle, 0, S, alpha=0.2, color=COLORS['primary'])

    # Mark phases
    for i in range(5):
        ax4.axvline(x=i, color='gray', linestyle='--', alpha=0.5)

    ax4.text(0.5, 0.9, 'Expansion', ha='center', fontsize=9)
    ax4.text(1.5, 0.9, 'Contract', ha='center', fontsize=9)
    ax4.text(2.5, 0.9, 'Expansion', ha='center', fontsize=9)
    ax4.text(3.5, 0.9, 'Contract', ha='center', fontsize=9)

    ax4.set_xlabel('Cosmic Time (cycles)')
    ax4.set_ylabel('Entropy S')
    ax4.set_title('D. Entropy is Cyclic\n(Not Monotonic)', fontweight='bold')
    ax4.set_xlim(0, 4)
    ax4.set_ylim(0, 1.1)

    # Panel E: Why heat death is not terminal
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.axis('off')

    text = (
        "WHY HEAT DEATH ≠ END\n\n"
        "Heat death = maximum entropy\n"
        "= maximum degeneracy\n\n"
        "But maximum degeneracy means:\n"
        "• Only HIGH-entropy states visited\n"
        "• LOW-entropy states never explored\n"
        "  (stars, galaxies, life...)\n\n"
        "━━━━━━━━━━━━━━━━━━━\n"
        "Categorical completeness\n"
        "requires ALL states visited.\n\n"
        "→ Contraction phase explores\n"
        "  low-entropy configurations."
    )

    box = FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
                         boxstyle="round,pad=0.02",
                         facecolor='#FFF3E0', edgecolor=COLORS['secondary'], linewidth=2)
    ax5.add_patch(box)
    ax5.text(0.5, 0.5, text, ha='center', va='center', fontsize=10,
            family='serif')
    ax5.set_xlim(0, 1)
    ax5.set_ylim(0, 1)
    ax5.set_title('E. Heat Death is Incomplete', fontweight='bold')

    # Panel F: The conclusion
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

    conclusion = (
        "DERIVED RESULT:\n\n"
        "The universe MUST be cyclic.\n\n"
        "This is not assumption—\n"
        "it follows from:\n\n"
        "1. Bounded phase space\n"
        "   (Poincaré recurrence)\n\n"
        "2. Categorical completeness\n"
        "   (all states must be visited)\n\n"
        "3. Consistency requirements\n"
        "   (no state left unexplored)\n\n"
        "━━━━━━━━━━━━━━━━━━━\n"
        "Big Bang → Heat Death →\n"
        "Big Crunch → Big Bang →..."
    )

    box = FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
                         boxstyle="round,pad=0.02",
                         facecolor='#E8F5E9', edgecolor=COLORS['accent'], linewidth=2)
    ax6.add_patch(box)
    ax6.text(0.5, 0.5, conclusion, ha='center', va='center', fontsize=10,
            family='serif')
    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
    ax6.set_title('F. Cyclic Universe Derived', fontweight='bold')

    plt.suptitle('Figure 6: Categorical Exhaustion → Cyclic Cosmology',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('../figures/fig6_cyclic_cosmology.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('../figures/fig6_cyclic_cosmology.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: fig6_cyclic_cosmology.png/pdf")


def figure_7_complete_flowchart():
    """Figure 7: Complete derivation flowchart (axioms → reality)"""
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111)
    ax.axis('off')

    # Define the flowchart structure
    # Level 0: Axiom
    # Level 1: Primary theorems
    # Level 2: Secondary results
    # Level 3: Physical phenomena
    # Level 4: Observable reality

    boxes = {
        # Level 0 - Axiom
        'axiom': {'pos': (0.5, 0.95), 'text': 'AXIOM\nBounded Phase Space\n+ Self-Consistency',
                  'color': '#B71C1C', 'width': 0.25},

        # Level 1 - Primary theorems
        'poincare': {'pos': (0.5, 0.82), 'text': 'Poincaré Recurrence\nTheorem',
                     'color': '#1565C0', 'width': 0.18},

        # Level 2 - Secondary results
        'oscillatory': {'pos': (0.25, 0.69), 'text': 'Oscillatory\nDynamics',
                        'color': '#2E7D32', 'width': 0.15},
        'categorical': {'pos': (0.5, 0.69), 'text': 'Categorical\nStructure',
                        'color': '#2E7D32', 'width': 0.15},
        'partition': {'pos': (0.75, 0.69), 'text': 'Partition\nGeometry',
                      'color': '#2E7D32', 'width': 0.15},

        # Level 3 - Derived structure
        'energy': {'pos': (0.12, 0.54), 'text': 'E = ℏω\nQuantization',
                   'color': '#F57C00', 'width': 0.12},
        'time': {'pos': (0.32, 0.54), 'text': 'Time\nEmergence',
                 'color': '#F57C00', 'width': 0.12},
        'space': {'pos': (0.52, 0.54), 'text': 'Space\n(D=3)',
                  'color': '#F57C00', 'width': 0.12},
        'matter': {'pos': (0.72, 0.54), 'text': 'Matter vs\nDark',
                   'color': '#F57C00', 'width': 0.12},
        'forces': {'pos': (0.88, 0.54), 'text': 'Force\nHierarchy',
                   'color': '#F57C00', 'width': 0.12},

        # Level 4 - Physical phenomena
        'qm': {'pos': (0.15, 0.38), 'text': 'Quantum\nMechanics',
               'color': '#7B1FA2', 'width': 0.13},
        'gr': {'pos': (0.35, 0.38), 'text': 'General\nRelativity',
               'color': '#7B1FA2', 'width': 0.13},
        'periodic': {'pos': (0.55, 0.38), 'text': 'Periodic\nTable',
                     'color': '#7B1FA2', 'width': 0.13},
        'cosmo': {'pos': (0.75, 0.38), 'text': 'Cyclic\nCosmology',
                  'color': '#7B1FA2', 'width': 0.13},

        # Level 5 - Observable reality
        'reality': {'pos': (0.5, 0.2), 'text': 'OBSERVABLE REALITY\nAtoms • Stars • Galaxies • Life',
                    'color': '#263238', 'width': 0.5}
    }

    # Draw boxes
    for name, props in boxes.items():
        x, y = props['pos']
        w = props['width']
        h = 0.08 if name not in ['axiom', 'reality'] else 0.1

        box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                             boxstyle="round,pad=0.01",
                             facecolor=props['color'], edgecolor='black',
                             linewidth=2, alpha=0.9)
        ax.add_patch(box)
        ax.text(x, y, props['text'], ha='center', va='center',
               fontsize=9, fontweight='bold', color='white')

    # Draw arrows
    arrows = [
        ('axiom', 'poincare'),
        ('poincare', 'oscillatory'),
        ('poincare', 'categorical'),
        ('poincare', 'partition'),
        ('oscillatory', 'energy'),
        ('oscillatory', 'time'),
        ('categorical', 'time'),
        ('categorical', 'space'),
        ('partition', 'space'),
        ('partition', 'matter'),
        ('partition', 'forces'),
        ('energy', 'qm'),
        ('time', 'qm'),
        ('time', 'gr'),
        ('space', 'gr'),
        ('space', 'periodic'),
        ('matter', 'periodic'),
        ('matter', 'cosmo'),
        ('forces', 'cosmo'),
        ('qm', 'reality'),
        ('gr', 'reality'),
        ('periodic', 'reality'),
        ('cosmo', 'reality')
    ]

    for start, end in arrows:
        x1, y1 = boxes[start]['pos']
        x2, y2 = boxes[end]['pos']

        # Adjust for box height
        h1 = 0.08 if start not in ['axiom', 'reality'] else 0.1
        h2 = 0.08 if end not in ['axiom', 'reality'] else 0.1

        ax.annotate('', xy=(x2, y2 + h2/2), xytext=(x1, y1 - h1/2),
                   arrowprops=dict(arrowstyle='->', color='gray', lw=1.5, alpha=0.7,
                                  connectionstyle='arc3,rad=0.0'))

    # Legend
    legend_items = [
        ('#B71C1C', 'Axiom (Starting Point)'),
        ('#1565C0', 'Foundational Theorem'),
        ('#2E7D32', 'Primary Structures'),
        ('#F57C00', 'Derived Properties'),
        ('#7B1FA2', 'Physical Theories'),
        ('#263238', 'Observable Reality')
    ]

    for i, (color, label) in enumerate(legend_items):
        x = 0.02
        y = 0.12 - i * 0.02
        rect = Rectangle((x, y - 0.008), 0.015, 0.015, facecolor=color, edgecolor='black')
        ax.add_patch(rect)
        ax.text(x + 0.02, y, label, fontsize=9, va='center')

    # Title and annotations
    ax.text(0.5, 0.02,
           'From ONE axiom (bounded phase space + consistency) → ALL of physics',
           ha='center', fontsize=12, style='italic', fontweight='bold')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.suptitle('Figure 7: Complete Derivation Flowchart (Axioms → Reality)',
                fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('../figures/fig7_derivation_flowchart.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('../figures/fig7_derivation_flowchart.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: fig7_derivation_flowchart.png/pdf")


if __name__ == "__main__":
    os.makedirs('../figures', exist_ok=True)

    print("Generating concept figures...")
    print("=" * 50)

    figure_1_poincare_to_oscillation()
    figure_2_categorical_temporal()
    figure_3_partition_spatial()
    figure_4_mode_occupation()
    figure_5_force_hierarchy()
    figure_6_cyclic_cosmology()
    figure_7_complete_flowchart()

    print("=" * 50)
    print("All concept figures generated!")

