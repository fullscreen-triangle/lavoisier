#!/usr/bin/env python3
"""
Mass Computing Figure Generation

Generates publication-quality figures for the Mass Computing paper.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
from pathlib import Path
import sys

# Import core modules
sys.path.insert(0, str(Path(__file__).parent))
from ternary_core import TernaryAddress, SpectrumExtractor, SEntropyCoord

# Set style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Output directory
FIGURES_DIR = Path(__file__).parent.parent / 'figures'
FIGURES_DIR.mkdir(exist_ok=True)


def draw_cube_wireframe(ax, alpha=0.3):
    """Draw S-space unit cube wireframe."""
    # Vertices of unit cube
    vertices = [
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
    ]

    # Edges
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # bottom
        [4, 5], [5, 6], [6, 7], [7, 4],  # top
        [0, 4], [1, 5], [2, 6], [3, 7]   # vertical
    ]

    for edge in edges:
        pts = [vertices[edge[0]], vertices[edge[1]]]
        ax.plot3D(*zip(*pts), 'k-', alpha=alpha, linewidth=0.5)


def draw_cell(ax, bounds, color='blue', alpha=0.2, label=None):
    """Draw a 3D cell as a semi-transparent box."""
    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]
    z_min, z_max = bounds[2]

    # Define vertices
    vertices = [
        [x_min, y_min, z_min], [x_max, y_min, z_min],
        [x_max, y_max, z_min], [x_min, y_max, z_min],
        [x_min, y_min, z_max], [x_max, y_min, z_max],
        [x_max, y_max, z_max], [x_min, y_max, z_max]
    ]

    # Define faces
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
        [vertices[0], vertices[3], vertices[7], vertices[4]],  # left
        [vertices[1], vertices[2], vertices[6], vertices[5]],  # right
    ]

    collection = Poly3DCollection(faces, alpha=alpha, facecolor=color,
                                   edgecolor='k', linewidth=0.3)
    ax.add_collection3d(collection)


def get_cell_bounds(addr: TernaryAddress):
    """Get the bounds of the cell addressed by a ternary address."""
    bounds = {
        0: [0.0, 1.0],
        1: [0.0, 1.0],
        2: [0.0, 1.0],
    }

    for i, trit in enumerate(addr.trits):
        axis = i % 3
        low, high = bounds[axis]
        width = (high - low) / 3
        new_low = low + trit * width
        new_high = new_low + width
        bounds[axis] = [new_low, new_high]

    return [[bounds[0][0], bounds[0][1]],
            [bounds[1][0], bounds[1][1]],
            [bounds[2][0], bounds[2][1]]]


def figure1_sspace_coordinates():
    """
    Figure 1: S-Entropy Coordinate Space

    Shows the unit cube [0,1]^3 with labeled axes and example molecules.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Draw wireframe cube
    draw_cube_wireframe(ax, alpha=0.5)

    # Axis labels with descriptions
    ax.set_xlabel(r'$S_k$ (Knowledge Entropy)', fontsize=12)
    ax.set_ylabel(r'$S_t$ (Temporal Entropy)', fontsize=12)
    ax.set_zlabel(r'$S_e$ (Evolution Entropy)', fontsize=12)

    # Example molecules in S-space
    molecules = [
        # (S_k, S_t, S_e, name, color)
        (0.1, 0.8, 0.2, 'Phospholipid\n(high mass, late RT)', 'red'),
        (0.7, 0.2, 0.3, 'Amino acid\n(low mass, early RT)', 'blue'),
        (0.4, 0.5, 0.8, 'Fragmented\nion', 'green'),
        (0.5, 0.5, 0.1, 'Intact\nmolecular ion', 'purple'),
    ]

    for s_k, s_t, s_e, name, color in molecules:
        ax.scatter([s_k], [s_t], [s_e], c=color, s=100, marker='o',
                   edgecolor='black', linewidth=1)
        ax.text(s_k + 0.05, s_t + 0.05, s_e + 0.05, name, fontsize=8)

    # Add annotation boxes
    ax.text2D(0.02, 0.98, r'$\mathcal{S} = [0,1]^3$', transform=ax.transAxes,
              fontsize=14, fontweight='bold', va='top')

    # Add coordinate meaning annotations
    annotations = [
        (0.02, 0.15, r'$S_k \downarrow \Rightarrow$ Higher mass'),
        (0.02, 0.10, r'$S_t \uparrow \Rightarrow$ Later elution'),
        (0.02, 0.05, r'$S_e \uparrow \Rightarrow$ More fragmented'),
    ]
    for x, y, text in annotations:
        ax.text2D(x, y, text, transform=ax.transAxes, fontsize=9)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.view_init(elev=20, azim=45)

    plt.title('S-Entropy Coordinate Space', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'figure1_sspace_coordinates.png')
    plt.close()
    print("Generated: figure1_sspace_coordinates.png")


def figure2_ternary_partitioning():
    """
    Figure 2: Ternary Partitioning Hierarchy

    Shows how ternary addresses recursively partition S-space.
    """
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))

    colors = ['#E6F2FF', '#FFE6E6', '#E6FFE6']  # Light blue, pink, green

    # Panel A: Depth 0 (whole cube - shown as 2D slice at S_e=0.5)
    ax = axes[0]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor='lightgray',
                                edgecolor='black', linewidth=2))
    ax.set_xlabel(r'$S_k$')
    ax.set_ylabel(r'$S_t$')
    ax.set_title('Depth 0\n$3^0 = 1$ cell', fontweight='bold')
    ax.set_aspect('equal')

    # Panel B: Depth 1 (3 cells along S_k)
    ax = axes[1]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    for i, c in enumerate(colors):
        ax.add_patch(plt.Rectangle((i/3, 0), 1/3, 1, facecolor=c,
                                    edgecolor='black', linewidth=1))
        ax.text((i + 0.5)/3, 0.5, f'{i}', ha='center', va='center',
                fontsize=14, fontweight='bold')
    ax.set_xlabel(r'$S_k$')
    ax.set_ylabel(r'$S_t$')
    ax.set_title('Depth 1\n$3^1 = 3$ cells', fontweight='bold')
    ax.set_aspect('equal')

    # Panel C: Depth 2 (9 cells - S_k then S_t)
    ax = axes[2]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    for i in range(3):
        for j in range(3):
            color = plt.cm.viridis(0.2 + 0.2 * i + 0.1 * j)
            ax.add_patch(plt.Rectangle((i/3, j/3), 1/3, 1/3,
                                        facecolor=color, edgecolor='black',
                                        linewidth=0.5, alpha=0.7))
            addr = f'{i}{j}'
            ax.text((i + 0.5)/3, (j + 0.5)/3, addr, ha='center', va='center',
                    fontsize=9, fontweight='bold', color='white')
    ax.set_xlabel(r'$S_k$')
    ax.set_ylabel(r'$S_t$')
    ax.set_title('Depth 2\n$3^2 = 9$ cells', fontweight='bold')
    ax.set_aspect('equal')

    # Panel D: Depth 3 (27 cells)
    ax = axes[3]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    for i in range(9):
        for j in range(3):
            # This is a simplification showing 27 cells in 2D
            x_idx = i % 3 + (j % 1) * 3
            y_idx = i // 3
            color = plt.cm.plasma(0.1 + 0.03 * (i + j * 3))
            ax.add_patch(plt.Rectangle((i/9, j/3), 1/9, 1/3,
                                        facecolor=color, edgecolor='black',
                                        linewidth=0.3, alpha=0.7))
    ax.set_xlabel(r'$S_k$')
    ax.set_ylabel(r'$S_t$')
    ax.set_title('Depth 3\n$3^3 = 27$ cells', fontweight='bold')
    ax.set_aspect('equal')

    plt.suptitle('Ternary Partitioning: Each Trit Divides Space by 3',
                 fontsize=13, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'figure2_ternary_partitioning.png')
    plt.close()
    print("Generated: figure2_ternary_partitioning.png")


def figure3_trajectory_position():
    """
    Figure 3: Trajectory-Position Equivalence

    Shows how a ternary address encodes BOTH position and trajectory.
    """
    fig = plt.figure(figsize=(12, 5))

    # Left panel: The trajectory as sequence of refinements
    ax1 = fig.add_subplot(121, projection='3d')

    # Example address: 012102
    address = "012102"
    addr = TernaryAddress.from_string(address)

    # Draw unit cube
    draw_cube_wireframe(ax1, alpha=0.3)

    # Track bounds and draw progressively smaller cells
    bounds = [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
    colors = plt.cm.Reds(np.linspace(0.2, 0.9, len(address)))

    trajectory_points = [(0.5, 0.5, 0.5)]  # Start at center

    for i, trit in enumerate(addr.trits):
        axis = i % 3
        low, high = bounds[axis]
        width = (high - low) / 3

        new_low = low + int(trit) * width
        new_high = new_low + width
        bounds[axis] = [new_low, new_high]

        # Calculate center of current cell
        center = [(bounds[j][0] + bounds[j][1]) / 2 for j in range(3)]
        trajectory_points.append(center)

        # Draw cell with decreasing alpha
        alpha = 0.3 - 0.04 * i
        draw_cell(ax1, bounds, color=colors[i], alpha=max(0.05, alpha))

    # Draw trajectory as arrows
    for i in range(len(trajectory_points) - 1):
        p1 = trajectory_points[i]
        p2 = trajectory_points[i + 1]
        ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                 'b-', linewidth=2, alpha=0.8)

    # Mark final position
    final = trajectory_points[-1]
    ax1.scatter([final[0]], [final[1]], [final[2]], c='red', s=150,
                marker='*', edgecolor='black', linewidth=1, zorder=10)

    ax1.set_xlabel(r'$S_k$')
    ax1.set_ylabel(r'$S_t$')
    ax1.set_zlabel(r'$S_e$')
    ax1.set_title(f'Address "{address}"\nTrajectory through S-Space', fontweight='bold')
    ax1.view_init(elev=20, azim=45)

    # Right panel: Address interpretation
    ax2 = fig.add_subplot(122)
    ax2.axis('off')

    # Draw the address with annotations
    y_start = 0.9
    ax2.text(0.5, y_start, 'Ternary Address:', ha='center', fontsize=14,
             fontweight='bold', transform=ax2.transAxes)

    y_start -= 0.12
    ax2.text(0.5, y_start, address, ha='center', fontsize=28, fontweight='bold',
             family='monospace', transform=ax2.transAxes)

    # Draw interpretation table
    y_start -= 0.15

    interpretations = [
        ('Position', 'Trit', 'Axis', 'Meaning'),
        ('1', '0', r'$S_k$', 'Select low-$S_k$ third (high mass)'),
        ('2', '1', r'$S_t$', 'Select mid-$S_t$ third (mid RT)'),
        ('3', '2', r'$S_e$', 'Select high-$S_e$ third (fragmented)'),
        ('4', '1', r'$S_k$', 'Refine: mid-$S_k$'),
        ('5', '0', r'$S_t$', 'Refine: low-$S_t$'),
        ('6', '2', r'$S_e$', 'Refine: high-$S_e$'),
    ]

    for i, (pos, trit, axis, meaning) in enumerate(interpretations):
        y = y_start - 0.07 * i
        if i == 0:  # Header
            ax2.text(0.05, y, pos, ha='left', fontsize=10, fontweight='bold',
                     transform=ax2.transAxes)
            ax2.text(0.2, y, trit, ha='center', fontsize=10, fontweight='bold',
                     transform=ax2.transAxes)
            ax2.text(0.35, y, axis, ha='center', fontsize=10, fontweight='bold',
                     transform=ax2.transAxes)
            ax2.text(0.7, y, meaning, ha='center', fontsize=10, fontweight='bold',
                     transform=ax2.transAxes)
        else:
            ax2.text(0.05, y, pos, ha='left', fontsize=10, transform=ax2.transAxes)
            ax2.text(0.2, y, trit, ha='center', fontsize=12, fontweight='bold',
                     family='monospace', transform=ax2.transAxes)
            ax2.text(0.35, y, axis, ha='center', fontsize=10, transform=ax2.transAxes)
            ax2.text(0.7, y, meaning, ha='center', fontsize=9, transform=ax2.transAxes)

    # Key insight box
    y_start -= 0.65
    box_text = "Key Insight:\nThe address IS the trajectory.\nPosition and path are\nencoded identically."
    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, edgecolor='orange')
    ax2.text(0.5, y_start, box_text, ha='center', va='center', fontsize=11,
             transform=ax2.transAxes, bbox=props)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'figure3_trajectory_position.png')
    plt.close()
    print("Generated: figure3_trajectory_position.png")


def figure4_observable_extraction():
    """
    Figure 4: Observable Extraction Functions

    Shows mapping from S-coordinates to mass spectrometric observables.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    extractor = SpectrumExtractor(mass_min=100, mass_max=1000, t0=0.5, t_max=20)

    # Panel A: S_k to m/z mapping
    ax = axes[0, 0]
    s_k = np.linspace(0.01, 0.99, 100)
    mz = [extractor.mass_from_scoord(sk) for sk in s_k]
    ax.plot(s_k, mz, 'b-', linewidth=2)
    ax.fill_between(s_k, mz, alpha=0.2)
    ax.set_xlabel(r'$S_k$ (Knowledge Entropy)')
    ax.set_ylabel(r'$m/z$')
    ax.set_title('A. Mass Function', fontweight='bold')
    ax.annotate('Low $S_k$\n= High mass', xy=(0.1, 800), fontsize=9)
    ax.annotate('High $S_k$\n= Low mass', xy=(0.7, 200), fontsize=9)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # Panel B: S_t to RT mapping
    ax = axes[0, 1]
    s_t = np.linspace(0, 1, 100)
    rt = [extractor.retention_time_from_scoord(st) for st in s_t]
    ax.plot(s_t, rt, 'g-', linewidth=2)
    ax.fill_between(s_t, rt, alpha=0.2, color='green')
    ax.set_xlabel(r'$S_t$ (Temporal Entropy)')
    ax.set_ylabel('Retention Time (min)')
    ax.set_title('B. Retention Function', fontweight='bold')
    ax.annotate('Early\neluters', xy=(0.1, 4), fontsize=9)
    ax.annotate('Late\neluters', xy=(0.7, 16), fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel C: Fragmentation vs S_e
    ax = axes[1, 0]
    s_e = np.linspace(0, 1, 100)
    n_frags = [int(se * 5) for se in s_e]
    ax.step(s_e, n_frags, 'r-', linewidth=2, where='post')
    ax.fill_between(s_e, n_frags, alpha=0.2, color='red', step='post')
    ax.set_xlabel(r'$S_e$ (Evolution Entropy)')
    ax.set_ylabel('Number of Fragments')
    ax.set_title('C. Fragmentation Function', fontweight='bold')
    ax.set_ylim(0, 6)
    ax.annotate('Intact ion', xy=(0.1, 0.5), fontsize=9)
    ax.annotate('Extensive\nfragmentation', xy=(0.7, 3.5), fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel D: Example spectrum synthesis
    ax = axes[1, 1]

    # Synthesize a spectrum
    addr = TernaryAddress.from_string("012102012102012102")
    spectrum = extractor.extract_spectrum(addr)

    # Plot parent ion
    ax.bar([spectrum.mz], [1.0], width=5, color='blue', alpha=0.8, label='Parent')

    # Plot fragments
    for frag_mz, frag_int in spectrum.fragments:
        ax.bar([frag_mz], [frag_int], width=3, color='red', alpha=0.7)

    # Plot isotopes
    for iso_mz, iso_int in spectrum.isotope_pattern:
        ax.bar([iso_mz], [iso_int * 0.8], width=2, color='green', alpha=0.5)

    ax.set_xlabel(r'$m/z$')
    ax.set_ylabel('Relative Intensity')
    ax.set_title('D. Synthesized Spectrum', fontweight='bold')
    ax.set_xlim(100, 500)
    ax.annotate(f'RT = {spectrum.retention_time:.1f} min', xy=(0.7, 0.9),
                xycoords='axes fraction', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Observable Extraction: S-Coordinates → Mass Spectrum',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'figure4_observable_extraction.png')
    plt.close()
    print("Generated: figure4_observable_extraction.png")


def figure5_massscript_workflow():
    """
    Figure 5: MassScript Workflow

    Shows how MassScript commands operate on ternary addresses.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis('off')

    # Title
    ax.text(0.5, 0.98, 'MassScript: Virtual Mass Spectrometry', ha='center',
            fontsize=14, fontweight='bold', transform=ax.transAxes)

    # Define workflow steps
    steps = [
        {
            'cmd': 'partition PC_34_1 201102012021',
            'desc': 'Define molecule by ternary address',
            'addr': '201102012021',
            'result': 'Address set'
        },
        {
            'cmd': 'observe',
            'desc': 'Extract observables (no computation)',
            'addr': '201102012021',
            'result': 'm/z=760.58, RT=14.2 min'
        },
        {
            'cmd': 'extend by 012',
            'desc': 'Increase fragmentation depth',
            'addr': '201102012021012',
            'result': 'Address refined'
        },
        {
            'cmd': 'fragment at 6',
            'desc': 'Split address at position 6',
            'addr': '201102 | 012021012',
            'result': 'Headgroup + fatty acid'
        },
        {
            'cmd': 'observe',
            'desc': 'Extract fragment spectra',
            'addr': '201102, 012021012',
            'result': 'm/z=184.07, m/z=577.52'
        },
    ]

    y_start = 0.85
    y_step = 0.15

    for i, step in enumerate(steps):
        y = y_start - i * y_step

        # Step number
        circle = plt.Circle((0.05, y), 0.025, color='steelblue',
                            transform=ax.transAxes)
        ax.add_patch(circle)
        ax.text(0.05, y, str(i+1), ha='center', va='center', color='white',
                fontweight='bold', fontsize=10, transform=ax.transAxes)

        # Command
        props = dict(boxstyle='round', facecolor='#f0f0f0', edgecolor='gray')
        ax.text(0.25, y, step['cmd'], ha='center', va='center', fontsize=10,
                family='monospace', transform=ax.transAxes, bbox=props)

        # Arrow
        ax.annotate('', xy=(0.45, y), xytext=(0.40, y),
                    arrowprops=dict(arrowstyle='->', color='gray'),
                    transform=ax.transAxes)

        # Description
        ax.text(0.55, y + 0.02, step['desc'], ha='left', va='center',
                fontsize=9, style='italic', transform=ax.transAxes)

        # Address state
        addr_box = dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange')
        ax.text(0.55, y - 0.03, f"τ = {step['addr']}", ha='left', va='center',
                fontsize=9, family='monospace', transform=ax.transAxes, bbox=addr_box)

        # Result
        ax.text(0.85, y, f"→ {step['result']}", ha='left', va='center',
                fontsize=9, color='darkgreen', transform=ax.transAxes)

    # Key insight box at bottom
    y = 0.08
    insight = (
        "Key: MassScript operates on addresses, not physical simulations.\n"
        "observe reads from partition structure; fragment splits addresses."
    )
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.5, edgecolor='blue')
    ax.text(0.5, y, insight, ha='center', va='center', fontsize=10,
            transform=ax.transAxes, bbox=props)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'figure5_massscript_workflow.png')
    plt.close()
    print("Generated: figure5_massscript_workflow.png")


def figure6_partition_determinism():
    """
    Figure 6: Partition Determinism Theorem Visualization

    Shows that ternary addresses uniquely determine spectra.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    extractor = SpectrumExtractor()

    # Generate multiple addresses at different depths
    base_addr = "012102"
    depths = [6, 12, 18]

    for idx, depth in enumerate(depths):
        ax = axes[idx]

        # Extend to target depth
        addr_str = (base_addr * 3)[:depth]
        addr = TernaryAddress.from_string(addr_str)
        s_k, s_t, s_e = addr.to_scoord()

        # Calculate resolution
        resolution = (1/3) ** (depth // 3)

        # Generate spectrum
        spectrum = extractor.extract_spectrum(addr)

        # Plot spectrum
        masses = [spectrum.mz]
        intensities = [1.0]
        for frag_mz, frag_int in spectrum.fragments:
            masses.append(frag_mz)
            intensities.append(frag_int)

        ax.bar(masses, intensities, width=max(2, 0.5 * depth),
               color='steelblue', alpha=0.8)

        ax.set_xlabel(r'$m/z$')
        ax.set_ylabel('Intensity')
        ax.set_title(f'Depth {depth}: {addr_str[:12]}...\n'
                     f'Resolution: {resolution:.2e}', fontweight='bold')
        ax.set_xlim(100, 600)
        ax.set_ylim(0, 1.1)

        # Add S-coordinate info
        info = f'$S_k$={s_k:.3f}\n$S_t$={s_t:.3f}\n$S_e$={s_e:.3f}'
        ax.text(0.95, 0.95, info, ha='right', va='top', fontsize=9,
                transform=ax.transAxes, family='monospace',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

        # Add m/z annotation
        ax.annotate(f'm/z = {spectrum.mz:.2f}', xy=(spectrum.mz, 1.0),
                    xytext=(spectrum.mz + 50, 1.05), fontsize=8,
                    arrowprops=dict(arrowstyle='->', color='red'))

    plt.suptitle('Partition Determinism: Address Depth → Spectral Resolution',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'figure6_partition_determinism.png')
    plt.close()
    print("Generated: figure6_partition_determinism.png")


def figure7_comparison_paradigms():
    """
    Figure 7: Forward Simulation vs Partition Synthesis

    Compares the two paradigms visually.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Forward Simulation
    ax = axes[0]
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.5, 0.95, 'Forward Simulation', ha='center', fontsize=13,
            fontweight='bold')

    # Flow boxes
    boxes = [
        (0.5, 0.8, 'Molecular\nStructure'),
        (0.5, 0.6, 'Physical Laws\n(Mathieu eq., QM)'),
        (0.5, 0.4, 'Numerical\nIntegration'),
        (0.5, 0.2, 'Predicted\nSpectrum'),
    ]

    for x, y, text in boxes:
        props = dict(boxstyle='round', facecolor='lightcoral',
                     edgecolor='darkred', alpha=0.7)
        ax.text(x, y, text, ha='center', va='center', fontsize=10,
                bbox=props)

    # Arrows
    for i in range(len(boxes) - 1):
        y1 = boxes[i][1] - 0.07
        y2 = boxes[i+1][1] + 0.07
        ax.annotate('', xy=(0.5, y2), xytext=(0.5, y1),
                    arrowprops=dict(arrowstyle='->', color='darkred', lw=2))

    # Annotations
    ax.text(0.8, 0.7, 'O(N³-N⁷)', fontsize=9, color='red')
    ax.text(0.8, 0.5, 'Parameter\nsensitive', fontsize=9, color='red')
    ax.text(0.8, 0.3, 'Approximations\naccumulate', fontsize=9, color='red')

    # Right: Partition Synthesis
    ax = axes[1]
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.5, 0.95, 'Partition Synthesis', ha='center', fontsize=13,
            fontweight='bold')

    boxes = [
        (0.5, 0.75, 'Ternary\nAddress'),
        (0.5, 0.45, 'S-Coordinate\nMapping'),
        (0.5, 0.2, 'Synthesized\nSpectrum'),
    ]

    for x, y, text in boxes:
        props = dict(boxstyle='round', facecolor='lightgreen',
                     edgecolor='darkgreen', alpha=0.7)
        ax.text(x, y, text, ha='center', va='center', fontsize=10,
                bbox=props)

    # Arrows
    for i in range(len(boxes) - 1):
        y1 = boxes[i][1] - 0.08
        y2 = boxes[i+1][1] + 0.08
        ax.annotate('', xy=(0.5, y2), xytext=(0.5, y1),
                    arrowprops=dict(arrowstyle='->', color='darkgreen', lw=2))

    # Annotations
    ax.text(0.8, 0.6, 'O(k)', fontsize=9, color='green')
    ax.text(0.8, 0.4, 'Deterministic', fontsize=9, color='green')
    ax.text(0.8, 0.25, 'Exact within\nresolution', fontsize=9, color='green')

    # Summary comparison
    ax.text(0.15, 0.05, '10⁶× speedup', fontsize=11, fontweight='bold',
            color='darkgreen')

    plt.suptitle('Paradigm Comparison: Simulation vs Synthesis',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'figure7_comparison_paradigms.png')
    plt.close()
    print("Generated: figure7_comparison_paradigms.png")


def generate_all_figures():
    """Generate all figures for the Mass Computing paper."""
    print("Generating figures for Mass Computing paper...")
    print("=" * 50)

    figure1_sspace_coordinates()
    figure2_ternary_partitioning()
    figure3_trajectory_position()
    figure4_observable_extraction()
    figure5_massscript_workflow()
    figure6_partition_determinism()
    figure7_comparison_paradigms()

    print("=" * 50)
    print(f"All figures saved to: {FIGURES_DIR}")


if __name__ == '__main__':
    generate_all_figures()
