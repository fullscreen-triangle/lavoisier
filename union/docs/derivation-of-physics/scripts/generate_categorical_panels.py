"""
Generate visualization panels for Sections 4-5:
- Categorical Structure and Temporal Emergence
- Partition Geometry and State Coordinates
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle, Wedge
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from scipy.special import sph_harm
import math

plt.style.use('default')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f8f8f8'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

def generate_categorical_structure_panel():
    """Visualize categorical structure and temporal emergence."""
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.35)
    
    # Panel 1: Continuous to categorical mapping
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.linspace(0, 10, 1000)
    y_continuous = np.sin(x) + 0.5*np.sin(3*x) + 0.3*np.sin(7*x)
    
    # Continuous curve
    ax1.plot(x, y_continuous, 'b-', linewidth=1, alpha=0.5, label='Continuous')
    
    # Categorical discretization
    n_categories = 8
    category_bounds = np.linspace(y_continuous.min(), y_continuous.max(), n_categories+1)
    y_categorical = np.digitize(y_continuous, category_bounds) - 1
    y_categorical = np.clip(y_categorical, 0, n_categories-1)
    
    colors = plt.cm.Set2(np.linspace(0, 1, n_categories))
    for i in range(n_categories):
        mask = y_categorical == i
        ax1.scatter(x[mask], y_continuous[mask], c=[colors[i]], s=1, alpha=0.8)
    
    # Category boundaries
    for bound in category_bounds[1:-1]:
        ax1.axhline(y=bound, color='gray', linestyle='--', alpha=0.3)
    
    ax1.set_xlabel('Continuous Variable')
    ax1.set_ylabel('Value')
    ax1.set_title('Continuous → Categorical\n(Finite Observer Resolution)')
    
    # Panel 2: Completion order (Hasse diagram)
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Draw a completion order diagram
    positions = {
        'C0': (0.5, 0),
        'C1': (0.25, 0.3), 'C2': (0.75, 0.3),
        'C3': (0.1, 0.6), 'C4': (0.4, 0.6), 'C5': (0.6, 0.6), 'C6': (0.9, 0.6),
        'C7': (0.3, 0.9), 'C8': (0.7, 0.9)
    }
    
    edges = [('C0', 'C1'), ('C0', 'C2'), ('C1', 'C3'), ('C1', 'C4'),
             ('C2', 'C5'), ('C2', 'C6'), ('C4', 'C7'), ('C5', 'C8')]
    
    # Draw edges
    for e1, e2 in edges:
        ax2.plot([positions[e1][0], positions[e2][0]], 
                [positions[e1][1], positions[e2][1]], 'gray', linewidth=1.5, zorder=1)
    
    # Draw nodes
    for node, (x, y) in positions.items():
        circle = Circle((x, y), 0.06, color='steelblue', zorder=2)
        ax2.add_patch(circle)
        ax2.text(x, y, node.replace('C', ''), ha='center', va='center', 
                fontsize=8, color='white', fontweight='bold', zorder=3)
    
    ax2.set_xlim(-0.1, 1.1)
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_title('Completion Order (≺)\n(Hasse Diagram)')
    ax2.arrow(0.05, 0, 0, 0.9, head_width=0.03, head_length=0.03, fc='black', ec='black')
    ax2.text(0.02, 0.5, 'Time', rotation=90, va='center', fontsize=10)
    
    # Panel 3: Temporal emergence from completion
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Completion trajectory
    t = np.linspace(0, 10, 100)
    n_completed = 1 - np.exp(-t/3)
    
    ax3.plot(t, n_completed * 100, 'b-', linewidth=2.5, label='Completed categories')
    ax3.fill_between(t, 0, n_completed * 100, alpha=0.2)
    
    # Mark categorical transitions
    transition_times = [1, 2.5, 4, 6, 8]
    for tt in transition_times:
        ax3.axvline(x=tt, color='red', linestyle='--', alpha=0.5)
        idx = np.argmin(np.abs(t - tt))
        ax3.plot(tt, n_completed[idx] * 100, 'ro', markersize=8)
    
    ax3.set_xlabel('Emergent Time (completion order)')
    ax3.set_ylabel('% Categories Completed')
    ax3.set_title('Temporal Emergence\n(Time from Completion)')
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 105)
    
    # Panel 4: Irreversibility
    ax4 = fig.add_subplot(gs[0, 3])
    
    # Forward trajectory
    t = np.linspace(0, 8, 100)
    states = np.floor(t * 1.2).astype(int)
    states = np.clip(states, 0, 9)
    
    ax4.step(t, states, 'b-', linewidth=2, where='post', label='μ(C,t)')
    ax4.fill_between(t, 0, states, step='post', alpha=0.2)
    
    # Arrow showing irreversibility
    ax4.annotate('', xy=(7, 7), xytext=(7, 3),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax4.text(7.3, 5, 'Irreversible\n(μ monotonic)', fontsize=9, color='red')
    
    ax4.set_xlabel('Time t')
    ax4.set_ylabel('Completion μ(C,t)')
    ax4.set_title('Categorical Irreversibility\n(Arrow of Time)')
    ax4.set_xlim(0, 8)
    ax4.set_ylim(0, 10)
    
    # Panel 5: Partition coordinate space (n, l, m)
    ax5 = fig.add_subplot(gs[1, 0], projection='3d')
    
    # Generate all valid (n, l, m) coordinates up to n=4
    coords = []
    colors = []
    for n in range(1, 5):
        for l in range(0, n):
            for m in range(-l, l+1):
                coords.append([n, l, m])
                colors.append(n)
    
    coords = np.array(coords)
    ax5.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=colors, 
               cmap='viridis', s=50, edgecolor='black', alpha=0.8)
    
    ax5.set_xlabel('n (depth)')
    ax5.set_ylabel('l (angular)')
    ax5.set_zlabel('m (orientation)')
    ax5.set_title('Partition Coordinates\n(n, l, m) Space')
    
    # Panel 6: Shell capacity 2n²
    ax6 = fig.add_subplot(gs[1, 1])
    
    n_values = np.arange(1, 8)
    capacity = 2 * n_values**2
    cumulative = np.cumsum(capacity)
    
    ax6.bar(n_values - 0.2, capacity, width=0.4, color='steelblue', 
           label='Shell capacity 2n²', edgecolor='black')
    ax6.bar(n_values + 0.2, cumulative, width=0.4, color='coral',
           label='Cumulative', edgecolor='black', alpha=0.7)
    
    # Add numbers on bars
    for i, (c, cum) in enumerate(zip(capacity, cumulative)):
        ax6.text(n_values[i] - 0.2, c + 2, str(c), ha='center', fontsize=9, fontweight='bold')
        ax6.text(n_values[i] + 0.2, cum + 2, str(cum), ha='center', fontsize=9)
    
    ax6.set_xlabel('Shell n')
    ax6.set_ylabel('Number of States')
    ax6.set_title('Shell Capacity Theorem\nN(n) = 2n²')
    ax6.legend(loc='upper left')
    ax6.set_xticks(n_values)
    
    # Panel 7: Energy ordering (n + αl)
    ax7 = fig.add_subplot(gs[1, 2])
    
    # All orbitals up to n=7
    orbitals = []
    for n in range(1, 8):
        for l in range(n):
            orbitals.append({'n': n, 'l': l, 'npl': n + l})
    
    # Sort by (n+l), then by n
    orbitals.sort(key=lambda x: (x['npl'], x['n']))
    
    # Plot as filling order
    y_positions = np.arange(len(orbitals))
    colors = [o['n'] for o in orbitals]
    labels = [f"{o['n']}{['s','p','d','f','g','h','i'][o['l']]}" for o in orbitals]
    
    bars = ax7.barh(y_positions[:20], [o['npl'] for o in orbitals[:20]], 
                   color=plt.cm.viridis(np.array(colors[:20])/7), edgecolor='black')
    ax7.set_yticks(y_positions[:20])
    ax7.set_yticklabels(labels[:20], fontsize=8)
    ax7.set_xlabel('n + l (energy ordering)')
    ax7.set_ylabel('Orbital')
    ax7.set_title('Energy Ordering Rule\n(n + αl), α ≈ 1')
    ax7.invert_yaxis()
    
    # Panel 8: Selection rules Δl = ±1
    ax8 = fig.add_subplot(gs[1, 3])
    
    # Energy levels
    levels = {
        '1s': (0, 1), '2s': (1, 2), '2p': (2, 2),
        '3s': (3, 3), '3p': (4, 3), '3d': (5, 3),
        '4s': (6, 4), '4p': (7, 4), '4d': (8, 4), '4f': (9, 4)
    }
    
    l_values = {'s': 0, 'p': 1, 'd': 2, 'f': 3}
    
    for name, (y, n) in levels.items():
        l = l_values[name[-1]]
        ax8.hlines(y, l-0.3, l+0.3, linewidth=4, color=plt.cm.viridis(n/4))
        ax8.text(l+0.35, y, name, fontsize=9, va='center')
    
    # Allowed transitions (Δl = ±1)
    allowed = [('2p', '1s'), ('3p', '2s'), ('3s', '2p'), ('3d', '2p'),
               ('4p', '3s'), ('4s', '3p'), ('4d', '3p'), ('4f', '3d')]
    
    for start, end in allowed:
        l1 = l_values[start[-1]]
        l2 = l_values[end[-1]]
        y1 = levels[start][0]
        y2 = levels[end][0]
        ax8.annotate('', xy=(l2, y2), xytext=(l1, y1),
                    arrowprops=dict(arrowstyle='->', color='green', lw=1.5, alpha=0.7))
    
    ax8.set_xlim(-0.5, 3.5)
    ax8.set_xticks([0, 1, 2, 3])
    ax8.set_xticklabels(['s (l=0)', 'p (l=1)', 'd (l=2)', 'f (l=3)'])
    ax8.set_ylabel('Energy Level')
    ax8.set_title('Selection Rules\nΔl = ±1 (allowed)')
    
    # Panel 9: Spherical harmonics visualization
    ax9 = fig.add_subplot(gs[2, 0], projection='3d')
    
    # Create spherical harmonic Y_2^0
    theta = np.linspace(0, np.pi, 50)
    phi = np.linspace(0, 2*np.pi, 100)
    THETA, PHI = np.meshgrid(theta, phi)
    
    l, m = 2, 0
    Y = sph_harm(m, l, PHI, THETA).real
    R = np.abs(Y)
    
    X = R * np.sin(THETA) * np.cos(PHI)
    Y_coord = R * np.sin(THETA) * np.sin(PHI)
    Z = R * np.cos(THETA)
    
    ax9.plot_surface(X, Y_coord, Z, cmap='RdBu', alpha=0.8, edgecolor='none')
    ax9.set_title(f'Spherical Harmonic\nY₂⁰(θ,φ)')
    ax9.set_xlabel('x')
    ax9.set_ylabel('y')
    ax9.set_zlabel('z')
    
    # Panel 10: All Y_l^m for l=0,1,2
    ax10 = fig.add_subplot(gs[2, 1])
    
    # Small multiples of spherical harmonics
    ax10.axis('off')
    ax10.set_title('Angular Momentum States\nl = 0, 1, 2', fontsize=12)
    
    # Create inset axes for each Y_l^m
    for l in range(3):
        for m in range(-l, l+1):
            # Calculate position
            col = m + 2
            row = l
            
            inset = ax10.inset_axes([0.1 + col*0.18, 0.7 - row*0.3, 0.15, 0.25])
            
            theta = np.linspace(0, np.pi, 30)
            phi = np.linspace(0, 2*np.pi, 60)
            THETA, PHI = np.meshgrid(theta, phi)
            
            Y = sph_harm(m, l, PHI, THETA).real
            R = np.abs(Y) + 0.1
            
            inset.contourf(PHI, THETA, R, levels=20, cmap='RdBu')
            inset.set_title(f'l={l}, m={m}', fontsize=7)
            inset.set_xticks([])
            inset.set_yticks([])
    
    # Panel 11: Chirality s = ±1/2
    ax11 = fig.add_subplot(gs[2, 2])
    
    # Draw two helices with opposite handedness
    t = np.linspace(0, 4*np.pi, 200)
    
    # Right-handed helix
    x1 = np.cos(t)
    y1 = np.sin(t)
    z1 = t / (2*np.pi)
    
    # Left-handed helix
    x2 = np.cos(-t) + 3
    y2 = np.sin(-t)
    z2 = t / (2*np.pi)
    
    ax11.plot(x1 + 0.5*z1, y1, 'b-', linewidth=2, label='s = +½')
    ax11.plot(x2 + 0.5*z2, y2, 'r-', linewidth=2, label='s = -½')
    
    ax11.fill_between([0, 1.5], [-2, -2], [2, 2], alpha=0.1, color='blue')
    ax11.fill_between([3, 4.5], [-2, -2], [2, 2], alpha=0.1, color='red')
    
    ax11.text(0.7, -1.7, 's = +½\n(right)', ha='center', fontsize=10, color='blue')
    ax11.text(3.7, -1.7, 's = -½\n(left)', ha='center', fontsize=10, color='red')
    
    ax11.set_xlim(-0.5, 5)
    ax11.set_ylim(-2, 2)
    ax11.set_xlabel('Projection')
    ax11.set_ylabel('Phase')
    ax11.set_title('Chirality\ns = ±½ (Spin)')
    ax11.legend(loc='upper right')
    
    # Panel 12: Degeneracy visualization
    ax12 = fig.add_subplot(gs[2, 3])
    
    # Show degeneracy for n=1,2,3,4
    n_vals = [1, 2, 3, 4]
    degeneracies = [2*n**2 for n in n_vals]
    
    # Draw stacked representations
    y_offset = 0
    for i, (n, deg) in enumerate(zip(n_vals, degeneracies)):
        # Draw rectangle representing degeneracy
        rect = Rectangle((0, y_offset), deg/10, 0.8, 
                         facecolor=plt.cm.viridis(i/4), edgecolor='black', linewidth=2)
        ax12.add_patch(rect)
        ax12.text(-0.3, y_offset + 0.4, f'n={n}', ha='right', va='center', fontsize=10)
        ax12.text(deg/10 + 0.1, y_offset + 0.4, f'{deg} states', ha='left', va='center', fontsize=10)
        
        # Show individual states as small squares
        state_size = min(0.15, deg/10/deg)
        for j in range(min(deg, 20)):
            small_rect = Rectangle((j*deg/10/deg, y_offset + 0.1), 
                                   deg/10/deg * 0.8, 0.6, 
                                   facecolor='white', edgecolor='gray', linewidth=0.5)
            ax12.add_patch(small_rect)
        
        y_offset += 1.2
    
    ax12.set_xlim(-0.5, 4)
    ax12.set_ylim(-0.2, 5)
    ax12.set_aspect('equal')
    ax12.axis('off')
    ax12.set_title('State Degeneracy\ng(n) = 2n²')
    
    plt.suptitle('Categorical Structure and Partition Geometry', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('../figures/categorical_partition_panel.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('../figures/categorical_partition_panel.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: categorical_partition_panel.png/pdf")

if __name__ == "__main__":
    import os
    os.makedirs('../figures', exist_ok=True)
    generate_categorical_structure_panel()

