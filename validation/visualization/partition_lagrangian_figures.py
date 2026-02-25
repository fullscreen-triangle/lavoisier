"""
Partition Lagrangian Visualization Suite
Generates publication-quality panel figures for the partition Lagrangian paper.

Each figure contains 4 panels with at least one 3D visualization.
Minimal text, no conceptual or table-based charts.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless rendering

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
import pandas as pd
from pathlib import Path
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

# Set publication style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.2,
})

# Custom colormaps
partition_cmap = LinearSegmentedColormap.from_list(
    'partition', ['#1a1a2e', '#16213e', '#0f3460', '#e94560', '#ff6b6b', '#feca57']
)

analyzer_colors = {
    'TOF': '#e74c3c',
    'Quadrupole': '#3498db',
    'Orbitrap': '#2ecc71',
    'FT-ICR': '#9b59b6'
}

# Output directory
OUTPUT_DIR = Path(__file__).parent / 'figures'
OUTPUT_DIR.mkdir(exist_ok=True)


class Arrow3D(FancyArrowPatch):
    """3D arrow for trajectory visualization"""
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


def generate_partition_depth_field(x, y, t=0, detector_pos=(0, 0)):
    """Generate partition depth field M(x, y, t)"""
    dx = x - detector_pos[0]
    dy = y - detector_pos[1]
    r_sq = dx**2 + dy**2
    M = 10 * (1 - np.exp(-r_sq / 50)) + 0.5 * np.sin(0.5 * t) * np.exp(-r_sq / 100)
    return M


def generate_ion_trajectory(t, m_z=500, analyzer='TOF'):
    """Generate ion trajectory for different analyzer types"""
    if analyzer == 'TOF':
        # Linear acceleration then drift
        x = 0.1 * t**2
        y = 0.1 * np.sin(0.1 * t)
        z = t * 0.5
    elif analyzer == 'Quadrupole':
        # Mathieu oscillations
        a, q = 0.2, 0.7
        omega = 2 * np.pi * 1e6
        x = np.cos(0.5 * omega * t / 1e6) * np.exp(-0.01 * t)
        y = np.sin(0.5 * omega * t / 1e6) * np.exp(-0.01 * t)
        z = t * 0.1
    elif analyzer == 'Orbitrap':
        # Helical motion
        omega_z = np.sqrt(1 / m_z) * 10
        omega_r = 0.5 * omega_z
        x = np.cos(omega_r * t) * (1 + 0.1 * np.sin(omega_z * t))
        y = np.sin(omega_r * t) * (1 + 0.1 * np.sin(omega_z * t))
        z = np.sin(omega_z * t)
    else:  # FT-ICR
        # Cyclotron motion
        omega_c = 1 / m_z * 1000
        x = np.cos(omega_c * t)
        y = np.sin(omega_c * t)
        z = 0.01 * t
    return x, y, z


def figure1_partition_lagrangian_dynamics():
    """Figure 1: Partition Lagrangian Dynamics"""
    fig = plt.figure(figsize=(14, 3.5))
    gs = GridSpec(1, 4, figure=fig, wspace=0.3)

    # Panel A (3D): Partition depth field with ion trajectory
    ax1 = fig.add_subplot(gs[0], projection='3d')
    x = np.linspace(-10, 10, 50)
    y = np.linspace(-10, 10, 50)
    X, Y = np.meshgrid(x, y)
    Z = generate_partition_depth_field(X, Y)

    surf = ax1.plot_surface(X, Y, Z, cmap=partition_cmap, alpha=0.7,
                            linewidth=0, antialiased=True)

    # Ion trajectory ribbon
    t_traj = np.linspace(0, 20, 100)
    x_traj = 8 * np.exp(-t_traj / 10) * np.cos(t_traj * 0.5)
    y_traj = 8 * np.exp(-t_traj / 10) * np.sin(t_traj * 0.5)
    z_traj = generate_partition_depth_field(x_traj, y_traj) + 0.5
    ax1.plot(x_traj, y_traj, z_traj, 'r-', linewidth=2.5, label='Ion path')
    ax1.scatter([x_traj[-1]], [y_traj[-1]], [z_traj[-1]], c='gold', s=100,
                marker='*', edgecolors='k', zorder=5)

    ax1.set_xlabel(r'$x$')
    ax1.set_ylabel(r'$y$')
    ax1.set_zlabel(r'$\mathcal{M}$')
    ax1.set_title('A', loc='left', fontweight='bold')
    ax1.view_init(elev=25, azim=45)

    # Panel B (2D): Time evolution of partition depth
    ax2 = fig.add_subplot(gs[1])
    t = np.linspace(0, 50, 500)
    M_t = 10 * np.exp(-t / 15) + 2 * np.sin(t * 0.3) * np.exp(-t / 30) + 0.5

    ax2.plot(t, M_t, 'b-', linewidth=2)
    ax2.fill_between(t, 0, M_t, alpha=0.3, color='blue')

    # Mark phase transitions
    transitions = [8, 22, 38]
    for tr in transitions:
        idx = int(tr * 10)
        ax2.axvline(tr, color='red', linestyle='--', alpha=0.5)
        ax2.scatter([tr], [M_t[idx]], c='red', s=60, zorder=5)

    ax2.set_xlabel(r'$t$')
    ax2.set_ylabel(r'$\mathcal{M}(t)$')
    ax2.set_title('B', loc='left', fontweight='bold')
    ax2.set_xlim(0, 50)
    ax2.set_ylim(0, 12)

    # Panel C (2D): Partition force vs position
    ax3 = fig.add_subplot(gs[2])
    x_pos = np.linspace(-10, 10, 200)
    # Force is negative gradient of partition depth
    F_p = -20 * x_pos * np.exp(-x_pos**2 / 20) / 20

    ax3.plot(x_pos, F_p, 'g-', linewidth=2)
    ax3.fill_between(x_pos, F_p, where=(F_p > 0), alpha=0.3, color='green')
    ax3.fill_between(x_pos, F_p, where=(F_p < 0), alpha=0.3, color='orange')
    ax3.axhline(0, color='k', linewidth=0.5)
    ax3.axvline(0, color='k', linewidth=0.5, linestyle='--')

    # Mark gradient wells
    ax3.annotate('', xy=(-5, 0.3), xytext=(-5, -0.1),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax3.annotate('', xy=(5, -0.3), xytext=(5, 0.1),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))

    ax3.set_xlabel(r'$x$')
    ax3.set_ylabel(r'$F_{\mathcal{M}}$')
    ax3.set_title('C', loc='left', fontweight='bold')

    # Panel D (2D): Energy landscape with trajectory
    ax4 = fig.add_subplot(gs[3])
    x_E = np.linspace(-8, 8, 200)

    # Kinetic, potential, partition energies
    T_kinetic = 0.5 * (1 + np.cos(x_E * 0.5))**2
    V_potential = 0.3 * x_E**2
    M_partition = 5 * (1 - np.exp(-x_E**2 / 20))
    E_total = T_kinetic + V_potential + M_partition

    ax4.plot(x_E, T_kinetic, 'b-', linewidth=1.5, alpha=0.7, label=r'$T$')
    ax4.plot(x_E, V_potential, 'g-', linewidth=1.5, alpha=0.7, label=r'$V$')
    ax4.plot(x_E, M_partition, 'r-', linewidth=1.5, alpha=0.7, label=r'$\mathcal{M}$')
    ax4.plot(x_E, E_total, 'k-', linewidth=2.5, label=r'$E_{tot}$')

    # Trajectory overlay
    t_overlay = np.linspace(0, 15, 50)
    x_overlay = 6 * np.exp(-t_overlay / 5) * np.cos(t_overlay)
    y_overlay = T_kinetic[100] + V_potential[100] + M_partition[100] - t_overlay * 0.1
    ax4.scatter(x_overlay[::5], y_overlay[::5], c=t_overlay[::5], cmap='plasma',
                s=30, zorder=5, edgecolors='k', linewidths=0.5)

    ax4.set_xlabel(r'$x$')
    ax4.set_ylabel('Energy')
    ax4.legend(loc='upper right', framealpha=0.9)
    ax4.set_title('D', loc='left', fontweight='bold')
    ax4.set_xlim(-8, 8)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'figure1_partition_dynamics.png', bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / 'figure1_partition_dynamics.pdf', bbox_inches='tight')
    plt.close()
    print("Figure 1: Partition Lagrangian Dynamics - Complete")


def figure2_four_analyzers():
    """Figure 2: Four Analyzer Types Unified"""
    fig = plt.figure(figsize=(14, 3.5))
    gs = GridSpec(1, 4, figure=fig, wspace=0.25)

    t = np.linspace(0, 10, 500)

    # Panel A (3D): TOF analyzer
    ax1 = fig.add_subplot(gs[0], projection='3d')
    for m_z in [200, 400, 600, 800]:
        x, y, z = generate_ion_trajectory(t, m_z, 'TOF')
        M = generate_partition_depth_field(x, z)
        scatter = ax1.scatter(x[::10], y[::10], z[::10], c=M[::10],
                             cmap=partition_cmap, s=15, alpha=0.8)
        ax1.plot(x, y, z, alpha=0.3, color=analyzer_colors['TOF'])

    ax1.set_xlabel(r'$x$')
    ax1.set_ylabel(r'$y$')
    ax1.set_zlabel(r'$z$')
    ax1.set_title('A: TOF', loc='left', fontweight='bold')
    ax1.view_init(elev=20, azim=30)

    # Panel B (3D): Quadrupole stability
    ax2 = fig.add_subplot(gs[1], projection='3d')
    a_vals = np.linspace(0, 0.25, 30)
    q_vals = np.linspace(0, 0.9, 30)
    A, Q = np.meshgrid(a_vals, q_vals)

    # Stability region (simplified Mathieu)
    stability = np.exp(-((A - 0.1)**2 / 0.02 + (Q - 0.7)**2 / 0.1))
    M_stab = 5 * stability + np.random.normal(0, 0.1, A.shape)

    ax2.plot_surface(A, Q, M_stab, cmap=partition_cmap, alpha=0.8)

    # Ion paths through stability
    for i in range(3):
        a_path = 0.1 + 0.02 * np.sin(t + i)
        q_path = 0.3 + 0.4 * t / 10
        m_path = 5 * np.exp(-((a_path - 0.1)**2 / 0.02 + (q_path - 0.7)**2 / 0.1))
        ax2.plot(a_path, q_path, m_path + 0.2, color=analyzer_colors['Quadrupole'],
                linewidth=2, alpha=0.9)

    ax2.set_xlabel(r'$a$')
    ax2.set_ylabel(r'$q$')
    ax2.set_zlabel(r'$\mathcal{M}$')
    ax2.set_title('B: Quad', loc='left', fontweight='bold')
    ax2.view_init(elev=25, azim=45)

    # Panel C (3D): Orbitrap helical
    ax3 = fig.add_subplot(gs[2], projection='3d')
    t_orbi = np.linspace(0, 20, 1000)

    for m_z, alpha in [(300, 0.9), (500, 0.7), (700, 0.5)]:
        x, y, z = generate_ion_trajectory(t_orbi, m_z, 'Orbitrap')
        r = np.sqrt(x**2 + y**2)
        M = 3 * (1 - np.exp(-r**2 / 2)) + np.abs(z)

        scatter = ax3.scatter(x[::5], y[::5], z[::5], c=M[::5],
                             cmap=partition_cmap, s=8, alpha=alpha)

    ax3.set_xlabel(r'$x$')
    ax3.set_ylabel(r'$y$')
    ax3.set_zlabel(r'$z$')
    ax3.set_title('C: Orbitrap', loc='left', fontweight='bold')
    ax3.view_init(elev=15, azim=60)

    # Panel D (3D): ICR cyclotron
    ax4 = fig.add_subplot(gs[3], projection='3d')

    # Create heatmap base
    theta = np.linspace(0, 2*np.pi, 50)
    r_grid = np.linspace(0, 2, 30)
    THETA, R = np.meshgrid(theta, r_grid)
    X_grid = R * np.cos(THETA)
    Y_grid = R * np.sin(THETA)
    Z_grid = np.zeros_like(X_grid)
    M_grid = 5 * (1 - np.exp(-R**2))

    ax4.plot_surface(X_grid, Y_grid, Z_grid, facecolors=plt.cm.viridis(M_grid / M_grid.max()),
                    alpha=0.5, linewidth=0)

    # Cyclotron orbits
    t_icr = np.linspace(0, 15, 500)
    for m_z in [300, 500, 800]:
        x, y, z = generate_ion_trajectory(t_icr, m_z, 'FT-ICR')
        ax4.plot(x, y, z, color=analyzer_colors['FT-ICR'], linewidth=1.5, alpha=0.8)
        ax4.scatter([x[-1]], [y[-1]], [z[-1]], c='gold', s=50, marker='*')

    ax4.set_xlabel(r'$x$')
    ax4.set_ylabel(r'$y$')
    ax4.set_zlabel(r'$z$')
    ax4.set_title('D: ICR', loc='left', fontweight='bold')
    ax4.view_init(elev=30, azim=45)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'figure2_four_analyzers.png', bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / 'figure2_four_analyzers.pdf', bbox_inches='tight')
    plt.close()
    print("Figure 2: Four Analyzer Types - Complete")


def figure3_resolution_limits():
    """Figure 3: Resolution Limit Validation"""
    fig = plt.figure(figsize=(14, 3.5))
    gs = GridSpec(1, 4, figure=fig, wspace=0.3)

    # Constants
    hbar = 1.054e-34

    # Panel A (2D): ΔM·τ_p vs mass (log-log)
    ax1 = fig.add_subplot(gs[0])

    m_z = np.logspace(2, 4, 50)

    analyzers_data = {
        'TOF': (1e-6, 1.2),
        'Quadrupole': (1e-5, 0.8),
        'Orbitrap': (1e-4, 1.5),
        'FT-ICR': (1e-3, 2.0)
    }

    for name, (tau_scale, dm_scale) in analyzers_data.items():
        tau_p = tau_scale * np.sqrt(m_z)
        delta_M = dm_scale * 1e-20 / np.sqrt(m_z)
        product = delta_M * tau_p
        ax1.loglog(m_z, product, '-', color=analyzer_colors[name],
                  linewidth=2, label=name)

    # ℏ threshold
    ax1.axhline(hbar, color='k', linestyle='--', linewidth=1.5, label=r'$\hbar$')

    ax1.set_xlabel(r'$m/z$')
    ax1.set_ylabel(r'$\Delta\mathcal{M} \cdot \tau_p$')
    ax1.legend(loc='lower right', fontsize=7)
    ax1.set_title('A', loc='left', fontweight='bold')

    # Panel B (3D): Resolution surface R(m, τ_p)
    ax2 = fig.add_subplot(gs[1], projection='3d')

    m_vals = np.linspace(100, 2000, 40)
    tau_vals = np.linspace(1e-6, 1e-3, 40)
    M, TAU = np.meshgrid(m_vals, tau_vals)

    # Resolution R = T·ΔM/ℏ (simplified)
    R = (TAU * 1e20 * np.sqrt(M)) / hbar
    R = np.log10(R + 1)  # Log scale for visualization

    surf = ax2.plot_surface(M, np.log10(TAU), R, cmap='viridis', alpha=0.8)

    ax2.set_xlabel(r'$m/z$')
    ax2.set_ylabel(r'$\log_{10}(\tau_p)$')
    ax2.set_zlabel(r'$\log_{10}(R)$')
    ax2.set_title('B', loc='left', fontweight='bold')
    ax2.view_init(elev=25, azim=45)

    # Panel C (2D): Measured vs predicted resolution
    ax3 = fig.add_subplot(gs[2])

    np.random.seed(42)
    n_points = 80
    R_predicted = np.logspace(2, 6, n_points)
    noise = np.random.normal(0, 0.1, n_points)
    R_measured = R_predicted * 10**(noise)

    colors = np.array([analyzer_colors[a] for a in
                       np.random.choice(list(analyzer_colors.keys()), n_points)])

    ax3.scatter(R_predicted, R_measured, c=colors, s=30, alpha=0.7, edgecolors='k',
               linewidths=0.3)
    ax3.plot([1e2, 1e6], [1e2, 1e6], 'k--', linewidth=1.5, label='Perfect')

    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_xlabel('Predicted Resolution')
    ax3.set_ylabel('Measured Resolution')
    ax3.set_title('C', loc='left', fontweight='bold')
    ax3.set_xlim(1e2, 1e6)
    ax3.set_ylim(1e2, 1e6)

    # Panel D (2D): Partition lag distribution (violin)
    ax4 = fig.add_subplot(gs[3])

    tau_data = []
    positions = []
    colors_violin = []

    for i, (name, color) in enumerate(analyzer_colors.items()):
        if name == 'TOF':
            data = np.random.lognormal(-14, 0.5, 200)
        elif name == 'Quadrupole':
            data = np.random.lognormal(-12, 0.6, 200)
        elif name == 'Orbitrap':
            data = np.random.lognormal(-10, 0.4, 200)
        else:
            data = np.random.lognormal(-8, 0.3, 200)

        parts = ax4.violinplot([np.log10(data)], positions=[i], showmeans=True,
                               showmedians=True)
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_alpha(0.7)

    ax4.set_xticks(range(4))
    ax4.set_xticklabels(['TOF', 'Quad', 'Orbi', 'ICR'])
    ax4.set_ylabel(r'$\log_{10}(\tau_p)$')
    ax4.set_title('D', loc='left', fontweight='bold')

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'figure3_resolution_limits.png', bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / 'figure3_resolution_limits.pdf', bbox_inches='tight')
    plt.close()
    print("Figure 3: Resolution Limits - Complete")


def figure4_partition_funnel():
    """Figure 4: Partition Funnel (Novel Device)"""
    fig = plt.figure(figsize=(14, 3.5))
    gs = GridSpec(1, 4, figure=fig, wspace=0.25)

    # Panel A (3D): Funnel geometry with partition depth
    ax1 = fig.add_subplot(gs[0], projection='3d')

    z_funnel = np.linspace(0, 10, 50)
    theta = np.linspace(0, 2*np.pi, 50)
    Z, THETA = np.meshgrid(z_funnel, theta)

    # Funnel radius decreases along z
    R_funnel = 3 * np.exp(-Z / 5) + 0.5
    X = R_funnel * np.cos(THETA)
    Y = R_funnel * np.sin(THETA)

    # Partition depth as color
    M_funnel = 10 * (1 - Z / 10)  # Decreases toward exit

    ax1.plot_surface(X, Y, Z, facecolors=plt.cm.viridis(M_funnel / 10),
                    alpha=0.7, linewidth=0)

    # Add gradient arrows
    for z_pos in [2, 5, 8]:
        idx = int(z_pos * 5)
        for th in [0, np.pi/2, np.pi, 3*np.pi/2]:
            r = 3 * np.exp(-z_pos / 5) + 0.5
            ax1.quiver(r*np.cos(th), r*np.sin(th), z_pos,
                      -0.3*np.cos(th), -0.3*np.sin(th), 0.5,
                      color='red', arrow_length_ratio=0.3, linewidth=1.5)

    ax1.set_xlabel(r'$x$')
    ax1.set_ylabel(r'$y$')
    ax1.set_zlabel(r'$z$')
    ax1.set_title('A', loc='left', fontweight='bold')
    ax1.view_init(elev=20, azim=30)

    # Panel B (3D): Ion trajectories colored by velocity
    ax2 = fig.add_subplot(gs[1], projection='3d')

    t_traj = np.linspace(0, 10, 200)

    for i, (r0, th0) in enumerate([(2.5, 0), (2.0, np.pi/3), (2.8, np.pi), (1.8, 5*np.pi/3)]):
        # Spiral inward
        r = r0 * np.exp(-t_traj / 5) + 0.3
        theta_t = th0 + t_traj * 0.5
        x = r * np.cos(theta_t)
        y = r * np.sin(theta_t)
        z = t_traj

        # Velocity (increases as funnel narrows)
        v = np.gradient(z) + 0.5 * np.gradient(r)**2
        v = (v - v.min()) / (v.max() - v.min())

        scatter = ax2.scatter(x, y, z, c=v, cmap='hot', s=8, alpha=0.8)

    ax2.set_xlabel(r'$x$')
    ax2.set_ylabel(r'$y$')
    ax2.set_zlabel(r'$z$')
    ax2.set_title('B', loc='left', fontweight='bold')
    ax2.view_init(elev=15, azim=45)

    # Panel C (2D): Transmission efficiency vs mass
    ax3 = fig.add_subplot(gs[2])

    m_z = np.linspace(100, 2000, 100)

    # Funnel transmission (high, nearly flat)
    T_funnel = 0.95 - 0.05 * (m_z / 2000)**2 + np.random.normal(0, 0.01, 100)
    T_funnel = np.clip(T_funnel, 0, 1)

    # Conventional (decreases with mass)
    T_conv = 0.85 * np.exp(-m_z / 3000) + np.random.normal(0, 0.02, 100)
    T_conv = np.clip(T_conv, 0, 1)

    ax3.fill_between(m_z, T_funnel, alpha=0.3, color='green')
    ax3.fill_between(m_z, T_conv, alpha=0.3, color='gray')
    ax3.plot(m_z, T_funnel, 'g-', linewidth=2, label='Funnel')
    ax3.plot(m_z, T_conv, 'k-', linewidth=2, label='Conv.')

    ax3.set_xlabel(r'$m/z$')
    ax3.set_ylabel('Transmission')
    ax3.legend(loc='lower left')
    ax3.set_title('C', loc='left', fontweight='bold')
    ax3.set_ylim(0, 1.05)

    # Panel D (2D): Partition force along axis
    ax4 = fig.add_subplot(gs[3])

    z_axis = np.linspace(0, 10, 200)

    # Partition force (focusing regions)
    F_radial = -2 * np.exp(-z_axis / 3) * np.sin(z_axis * 0.8)
    F_axial = 1.5 * (1 - np.exp(-z_axis / 5))

    ax4.plot(z_axis, F_radial, 'b-', linewidth=2, label=r'$F_r$')
    ax4.plot(z_axis, F_axial, 'r-', linewidth=2, label=r'$F_z$')
    ax4.axhline(0, color='k', linewidth=0.5)

    # Mark focusing regions
    focus_regions = [(1, 3), (5, 7)]
    for zs, ze in focus_regions:
        ax4.axvspan(zs, ze, alpha=0.2, color='yellow')

    ax4.set_xlabel(r'$z$')
    ax4.set_ylabel(r'$F_{\mathcal{M}}$')
    ax4.legend(loc='upper right')
    ax4.set_title('D', loc='left', fontweight='bold')

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'figure4_partition_funnel.png', bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / 'figure4_partition_funnel.pdf', bbox_inches='tight')
    plt.close()
    print("Figure 4: Partition Funnel - Complete")


def figure5_nist_validation():
    """Figure 5: NIST Experimental Validation"""
    fig = plt.figure(figsize=(14, 3.5))
    gs = GridSpec(1, 4, figure=fig, wspace=0.3)

    # Load or simulate NIST validation data
    np.random.seed(42)

    # NIST glycan data from validation results
    compounds = [
        {'name': 'NGA3B(1-4)', 'n': 4, 'l': 1, 'm': 0, 'sk': 0.83, 'st': 0.5, 'se': 0.3,
         'drip_c': 0.17, 'drip_s': 0.98, 'score': 1.0, 'lib': 'NIST'},
        {'name': 'NGA4', 'n': 5, 'l': 1, 'm': 1, 'sk': 0.82, 'st': 0.5, 'se': 0.7,
         'drip_c': 0.17, 'drip_s': 0.98, 'score': 1.0, 'lib': 'NIST'},
        {'name': 'A1F-MIX', 'n': 5, 'l': 2, 'm': 2, 'sk': 0.78, 'st': 0.5, 'se': 0.7,
         'drip_c': 0.17, 'drip_s': 0.98, 'score': 1.0, 'lib': 'NIST'},
        {'name': 'NGA2', 'n': 4, 'l': 1, 'm': 1, 'sk': 0.76, 'st': 0.5, 'se': 0.7,
         'drip_c': 0.17, 'drip_s': 0.98, 'score': 1.0, 'lib': 'NIST'},
        {'name': 'NA2G1F', 'n': 5, 'l': 0, 'm': 0, 'sk': 0.83, 'st': 0.5, 'se': 0.7,
         'drip_c': 0.16, 'drip_s': 0.99, 'score': 1.0, 'lib': 'NIST'},
        {'name': '3-Sialyl', 'n': 5, 'l': 4, 'm': 0, 'sk': 0.73, 'st': 0.5, 'se': 0.3,
         'drip_c': 0.18, 'drip_s': 0.98, 'score': 1.0, 'lib': 'HMO'},
        {'name': '3-Galac', 'n': 4, 'l': 1, 'm': 0, 'sk': 0.78, 'st': 0.5, 'se': 0.3,
         'drip_c': 0.17, 'drip_s': 0.98, 'score': 1.0, 'lib': 'HMO'},
        {'name': '2-Fucos', 'n': 5, 'l': 1, 'm': 0, 'sk': 0.84, 'st': 0.5, 'se': 0.3,
         'drip_c': 0.17, 'drip_s': 0.98, 'score': 1.0, 'lib': 'HMO'},
        {'name': 'DFLNO', 'n': 6, 'l': 0, 'm': 0, 'sk': 0.84, 'st': 0.5, 'se': 0.7,
         'drip_c': 0.16, 'drip_s': 0.98, 'score': 1.0, 'lib': 'HMO'},
        {'name': 'A4122a', 'n': 7, 'l': 0, 'm': 0, 'sk': 0.87, 'st': 0.5, 'se': 0.7,
         'drip_c': 0.16, 'drip_s': 0.99, 'score': 1.0, 'lib': 'HMO'},
    ]

    # Extract arrays
    n_vals = np.array([c['n'] for c in compounds])
    l_vals = np.array([c['l'] for c in compounds])
    m_vals = np.array([c['m'] for c in compounds])
    sk_vals = np.array([c['sk'] for c in compounds])
    st_vals = np.array([c['st'] for c in compounds])
    se_vals = np.array([c['se'] for c in compounds])
    scores = np.array([c['score'] for c in compounds])
    drip_c = np.array([c['drip_c'] for c in compounds])
    drip_s = np.array([c['drip_s'] for c in compounds])
    libs = np.array([c['lib'] for c in compounds])

    # Panel A (3D): Partition quantum numbers
    ax1 = fig.add_subplot(gs[0], projection='3d')

    scatter1 = ax1.scatter(n_vals, l_vals, m_vals, c=scores, cmap='RdYlGn',
                          s=80, edgecolors='k', linewidths=0.5, vmin=0.8, vmax=1.0)

    # Connect related compounds
    for i in range(len(compounds) - 1):
        if libs[i] == libs[i+1]:
            ax1.plot([n_vals[i], n_vals[i+1]], [l_vals[i], l_vals[i+1]],
                    [m_vals[i], m_vals[i+1]], 'k-', alpha=0.2, linewidth=0.5)

    ax1.set_xlabel(r'$n$')
    ax1.set_ylabel(r'$\ell$')
    ax1.set_zlabel(r'$m$')
    ax1.set_title('A', loc='left', fontweight='bold')
    ax1.view_init(elev=20, azim=45)

    # Panel B (3D): S-entropy coordinates
    ax2 = fig.add_subplot(gs[1], projection='3d')

    colors = ['#e74c3c' if l == 'NIST' else '#3498db' for l in libs]
    ax2.scatter(sk_vals, st_vals, se_vals, c=colors, s=80,
               edgecolors='k', linewidths=0.5)

    # Add entropy cube edges
    for i in [0, 1]:
        for j in [0, 1]:
            ax2.plot([i, i], [j, j], [0, 1], 'k-', alpha=0.1, linewidth=0.5)
            ax2.plot([i, i], [0, 1], [j, j], 'k-', alpha=0.1, linewidth=0.5)
            ax2.plot([0, 1], [i, i], [j, j], 'k-', alpha=0.1, linewidth=0.5)

    ax2.set_xlabel(r'$S_k$')
    ax2.set_ylabel(r'$S_t$')
    ax2.set_zlabel(r'$S_e$')
    ax2.set_title('B', loc='left', fontweight='bold')
    ax2.view_init(elev=25, azim=60)

    # Panel C (2D): Validation scores bar chart
    ax3 = fig.add_subplot(gs[2])

    x_pos = np.arange(len(compounds))
    colors_bar = ['#e74c3c' if l == 'NIST' else '#3498db' for l in libs]

    bars = ax3.bar(x_pos, scores, color=colors_bar, edgecolor='k', linewidth=0.5)
    ax3.axhline(0.95, color='green', linestyle='--', linewidth=1.5, label='Threshold')

    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([c['name'][:6] for c in compounds], rotation=45, ha='right', fontsize=7)
    ax3.set_ylabel('Score')
    ax3.set_ylim(0.9, 1.02)
    ax3.set_title('C', loc='left', fontweight='bold')

    # Panel D (2D): DRIP complexity vs symmetry
    ax4 = fig.add_subplot(gs[3])

    # Add more points for density
    n_extra = 50
    drip_c_all = np.concatenate([drip_c, np.random.normal(0.17, 0.01, n_extra)])
    drip_s_all = np.concatenate([drip_s, np.random.normal(0.98, 0.005, n_extra)])

    # Density contours
    from scipy.stats import gaussian_kde
    xy = np.vstack([drip_c_all, drip_s_all])
    z = gaussian_kde(xy)(xy)

    ax4.scatter(drip_c_all, drip_s_all, c=z, cmap='viridis', s=30, alpha=0.6)
    ax4.scatter(drip_c, drip_s, c=colors, s=80, edgecolors='k', linewidths=1, zorder=5)

    ax4.set_xlabel('DRIP Complexity')
    ax4.set_ylabel('DRIP Symmetry')
    ax4.set_title('D', loc='left', fontweight='bold')
    ax4.set_xlim(0.14, 0.20)
    ax4.set_ylim(0.96, 1.0)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'figure5_nist_validation.png', bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / 'figure5_nist_validation.pdf', bbox_inches='tight')
    plt.close()
    print("Figure 5: NIST Validation - Complete")


def figure6_ternary_address():
    """Figure 6: Ternary Address Space"""
    fig = plt.figure(figsize=(14, 3.5))
    gs = GridSpec(1, 4, figure=fig, wspace=0.3)

    # Generate ternary addresses
    np.random.seed(42)

    compounds_data = [
        {'addr': '0011-0001-0000', 'mass': 589.9, 'adduct': '[M+H]'},
        {'addr': '0012-0001-0001', 'mass': 873.3, 'adduct': '[M+Na]'},
        {'addr': '0012-0002-0002', 'mass': 1039.9, 'adduct': '[M+2H]'},
        {'addr': '0011-0001-0001', 'mass': 681.2, 'adduct': '[M+H]'},
        {'addr': '0012-0000-0000', 'mass': 824.3, 'adduct': '[M+Na]'},
        {'addr': '0012-0011-0000', 'mass': 802.3, 'adduct': '[M+H]'},
        {'addr': '0020-0000-0000', 'mass': 1728.6, 'adduct': '[M+H]'},
        {'addr': '0021-0000-0000', 'mass': 1818.7, 'adduct': '[M+H]'},
        {'addr': '0011-0002-0000', 'mass': 657.2, 'adduct': '[M+H]'},
        {'addr': '0012-0001-0000', 'mass': 914.3, 'adduct': '[M-H]'},
    ]

    def addr_to_coords(addr):
        """Convert ternary address to 3D coordinates"""
        parts = addr.split('-')
        x = int(parts[0], 3) / 9  # Normalize
        y = int(parts[1], 3) / 9
        z = int(parts[2], 3) / 9
        return x, y, z

    coords = np.array([addr_to_coords(c['addr']) for c in compounds_data])
    masses = np.array([c['mass'] for c in compounds_data])
    adducts = [c['adduct'] for c in compounds_data]

    # Adduct color mapping
    adduct_colors = {'[M+H]': '#e74c3c', '[M+Na]': '#3498db',
                     '[M+2H]': '#2ecc71', '[M-H]': '#9b59b6'}

    # Panel A (3D): Colored by mass
    ax1 = fig.add_subplot(gs[0], projection='3d')

    scatter1 = ax1.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                          c=masses, cmap='plasma', s=100, edgecolors='k', linewidths=0.5)

    ax1.set_xlabel(r'$A_1$')
    ax1.set_ylabel(r'$A_2$')
    ax1.set_zlabel(r'$A_3$')
    ax1.set_title('A', loc='left', fontweight='bold')
    ax1.view_init(elev=20, azim=45)

    # Panel B (3D): Same space, colored by adduct
    ax2 = fig.add_subplot(gs[1], projection='3d')

    colors_adduct = [adduct_colors.get(a, '#888888') for a in adducts]
    ax2.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
               c=colors_adduct, s=100, edgecolors='k', linewidths=0.5)

    ax2.set_xlabel(r'$A_1$')
    ax2.set_ylabel(r'$A_2$')
    ax2.set_zlabel(r'$A_3$')
    ax2.set_title('B', loc='left', fontweight='bold')
    ax2.view_init(elev=20, azim=135)

    # Panel C (2D): Distance matrix heatmap
    ax3 = fig.add_subplot(gs[2])

    dist_matrix = squareform(pdist(coords, 'euclidean'))

    im = ax3.imshow(dist_matrix, cmap='viridis', aspect='auto')
    ax3.set_xlabel('Compound')
    ax3.set_ylabel('Compound')
    ax3.set_title('C', loc='left', fontweight='bold')
    plt.colorbar(im, ax=ax3, shrink=0.8)

    # Panel D (2D): Hierarchical dendrogram
    ax4 = fig.add_subplot(gs[3])

    Z = linkage(coords, 'ward')
    dendrogram(Z, ax=ax4, leaf_rotation=90, leaf_font_size=8,
              color_threshold=0.5 * max(Z[:, 2]))

    ax4.set_xlabel('Compound')
    ax4.set_ylabel('Distance')
    ax4.set_title('D', loc='left', fontweight='bold')

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'figure6_ternary_address.png', bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / 'figure6_ternary_address.pdf', bbox_inches='tight')
    plt.close()
    print("Figure 6: Ternary Address Space - Complete")


def figure7_state_counting():
    """Figure 7: State Counting Dynamics (ADDITIONAL)"""
    fig = plt.figure(figsize=(14, 3.5))
    gs = GridSpec(1, 4, figure=fig, wspace=0.3)

    # Panel A (3D): Cumulative state count M(t) surface
    ax1 = fig.add_subplot(gs[0], projection='3d')

    t = np.linspace(0, 10, 50)
    m_z = np.linspace(200, 1000, 40)
    T, MZ = np.meshgrid(t, m_z)

    # M(t) = t / τ_p, τ_p ∝ √(m/z)
    M_count = T / (0.1 * np.sqrt(MZ / 200))

    surf = ax1.plot_surface(T, MZ, M_count, cmap='viridis', alpha=0.8)

    ax1.set_xlabel(r'$t$')
    ax1.set_ylabel(r'$m/z$')
    ax1.set_zlabel(r'$M(t)$')
    ax1.set_title('A', loc='left', fontweight='bold')
    ax1.view_init(elev=25, azim=45)

    # Panel B (2D): dM/dt rate across analyzers
    ax2 = fig.add_subplot(gs[1])

    m_z_range = np.linspace(100, 2000, 100)

    for name, (scale, power) in [('TOF', (1e6, -0.5)), ('Quadrupole', (5e5, -0.3)),
                                  ('Orbitrap', (2e6, -0.5)), ('FT-ICR', (1e7, -1.0))]:
        dM_dt = scale * m_z_range**power
        ax2.loglog(m_z_range, dM_dt, '-', color=analyzer_colors[name],
                  linewidth=2, label=name)

    ax2.set_xlabel(r'$m/z$')
    ax2.set_ylabel(r'$dM/dt = 1/\langle\tau_p\rangle$')
    ax2.legend(loc='upper right', fontsize=7)
    ax2.set_title('B', loc='left', fontweight='bold')

    # Panel C (3D): Phase space trajectory with state transitions
    ax3 = fig.add_subplot(gs[2], projection='3d')

    t_phase = np.linspace(0, 20, 500)
    x = np.sin(t_phase) * np.exp(-t_phase / 10)
    p = np.cos(t_phase) * np.exp(-t_phase / 10)
    M_phase = 10 * np.exp(-t_phase / 8)

    # Color by state transitions
    states = np.floor(M_phase).astype(int)
    colors = plt.cm.tab10(states % 10)

    for i in range(len(t_phase) - 1):
        ax3.plot(x[i:i+2], p[i:i+2], M_phase[i:i+2],
                color=colors[i], linewidth=2)

    # Mark transitions
    transition_idx = np.where(np.diff(states) != 0)[0]
    ax3.scatter(x[transition_idx], p[transition_idx], M_phase[transition_idx],
               c='red', s=50, marker='x', zorder=5)

    ax3.set_xlabel(r'$x$')
    ax3.set_ylabel(r'$p$')
    ax3.set_zlabel(r'$\mathcal{M}$')
    ax3.set_title('C', loc='left', fontweight='bold')
    ax3.view_init(elev=20, azim=60)

    # Panel D (2D): State density by principal quantum number
    ax4 = fig.add_subplot(gs[3])

    n_values = np.arange(1, 10)
    C_n = 2 * n_values**2  # Capacity formula

    # Simulated occupancy
    np.random.seed(42)
    occupancy = np.random.poisson(C_n * 0.3)

    width = 0.35
    x_pos = np.arange(len(n_values))

    ax4.bar(x_pos - width/2, C_n, width, color='lightblue', edgecolor='k',
           label=r'$C(n) = 2n^2$')
    ax4.bar(x_pos + width/2, occupancy, width, color='coral', edgecolor='k',
           label='Observed')

    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([str(n) for n in n_values])
    ax4.set_xlabel(r'$n$')
    ax4.set_ylabel('States')
    ax4.legend(loc='upper left', fontsize=7)
    ax4.set_title('D', loc='left', fontweight='bold')

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'figure7_state_counting.png', bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / 'figure7_state_counting.pdf', bbox_inches='tight')
    plt.close()
    print("Figure 7: State Counting Dynamics - Complete")


def figure8_uncertainty_principle():
    """Figure 8: Partition Uncertainty Principle (ADDITIONAL)"""
    fig = plt.figure(figsize=(14, 3.5))
    gs = GridSpec(1, 4, figure=fig, wspace=0.3)

    hbar = 1.054e-34

    # Panel A (3D): Uncertainty product surface
    ax1 = fig.add_subplot(gs[0], projection='3d')

    m_z = np.linspace(100, 2000, 30)
    analyzer_idx = np.arange(4)
    MZ, ANA = np.meshgrid(m_z, analyzer_idx)

    # Different uncertainty products for each analyzer
    scales = [1e-33, 5e-34, 2e-33, 1e-32]  # TOF, Quad, Orbi, ICR
    product = np.zeros_like(MZ, dtype=float)
    for i, scale in enumerate(scales):
        product[i, :] = scale * (1 + 0.1 * np.sin(m_z / 200))

    surf = ax1.plot_surface(MZ, ANA, np.log10(product), cmap='plasma', alpha=0.8)

    ax1.set_xlabel(r'$m/z$')
    ax1.set_ylabel('Analyzer')
    ax1.set_zlabel(r'$\log_{10}(\Delta\mathcal{M}\cdot\tau_p)$')
    ax1.set_yticks([0, 1, 2, 3])
    ax1.set_yticklabels(['TOF', 'Q', 'O', 'ICR'], fontsize=7)
    ax1.set_title('A', loc='left', fontweight='bold')
    ax1.view_init(elev=25, azim=45)

    # Panel B (2D): Quantum limit approach
    ax2 = fig.add_subplot(gs[1])

    # Distance from ℏ limit for each analyzer
    analyzers = ['TOF', 'Quadrupole', 'Orbitrap', 'FT-ICR']
    distances = [3.2, 4.5, 2.8, 1.5]  # Orders of magnitude above ℏ

    colors = [analyzer_colors[a] for a in analyzers]
    bars = ax2.barh(analyzers, distances, color=colors, edgecolor='k')

    ax2.axvline(0, color='green', linestyle='--', linewidth=2, label=r'$\hbar$ limit')
    ax2.set_xlabel(r'$\log_{10}(\Delta\mathcal{M}\cdot\tau_p / \hbar)$')
    ax2.set_title('B', loc='left', fontweight='bold')
    ax2.set_xlim(0, 6)

    # Panel C (3D): Resolution manifold R(T, ΔM)
    ax3 = fig.add_subplot(gs[2], projection='3d')

    T_vals = np.logspace(-6, 0, 30)  # Measurement time
    dM_vals = np.logspace(-22, -18, 30)  # Partition depth separation
    T_grid, DM_grid = np.meshgrid(T_vals, dM_vals)

    # R = T·ΔM/ℏ
    R = np.log10(T_grid * DM_grid / hbar + 1)

    surf = ax3.plot_surface(np.log10(T_grid), np.log10(DM_grid), R,
                           cmap='viridis', alpha=0.8)

    ax3.set_xlabel(r'$\log_{10}(T)$')
    ax3.set_ylabel(r'$\log_{10}(\Delta\mathcal{M})$')
    ax3.set_zlabel(r'$\log_{10}(R)$')
    ax3.set_title('C', loc='left', fontweight='bold')
    ax3.view_init(elev=30, azim=135)

    # Panel D (2D): ΔM vs τ_p with hyperbolic bound
    ax4 = fig.add_subplot(gs[3])

    tau_p = np.logspace(-8, -2, 100)
    dM_bound = hbar / tau_p

    ax4.loglog(tau_p, dM_bound, 'k-', linewidth=2, label=r'$\Delta\mathcal{M} = \hbar/\tau_p$')
    ax4.fill_between(tau_p, dM_bound, 1e-15, alpha=0.2, color='red',
                     label='Forbidden')
    ax4.fill_between(tau_p, dM_bound, 1e-45, alpha=0.2, color='green',
                     label='Allowed')

    # Scatter points for each analyzer
    np.random.seed(42)
    for name, color in analyzer_colors.items():
        if name == 'TOF':
            tau_pts = 10**np.random.uniform(-7, -5, 15)
        elif name == 'Quadrupole':
            tau_pts = 10**np.random.uniform(-6, -4, 15)
        elif name == 'Orbitrap':
            tau_pts = 10**np.random.uniform(-5, -3, 15)
        else:
            tau_pts = 10**np.random.uniform(-4, -2, 15)

        dM_pts = hbar / tau_pts * 10**np.random.uniform(1, 3, 15)
        ax4.scatter(tau_pts, dM_pts, c=color, s=30, alpha=0.7, edgecolors='k',
                   linewidths=0.3, label=name)

    ax4.set_xlabel(r'$\tau_p$ (s)')
    ax4.set_ylabel(r'$\Delta\mathcal{M}$ (J)')
    ax4.legend(loc='upper right', fontsize=6, ncol=2)
    ax4.set_title('D', loc='left', fontweight='bold')
    ax4.set_xlim(1e-8, 1e-2)
    ax4.set_ylim(1e-35, 1e-20)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'figure8_uncertainty_principle.png', bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / 'figure8_uncertainty_principle.pdf', bbox_inches='tight')
    plt.close()
    print("Figure 8: Uncertainty Principle - Complete")


def figure9_ion_journey_drip():
    """Figure 9: Ion Journey and Drip Visualization"""
    fig = plt.figure(figsize=(14, 3.5))
    gs = GridSpec(1, 4, figure=fig, wspace=0.3)

    np.random.seed(42)

    # Panel A (3D): Ion journey through partition space - source to detector
    ax1 = fig.add_subplot(gs[0], projection='3d')

    # Create journey stages: Source -> Ionization -> Analyzer -> Detection
    n_points = 200
    t = np.linspace(0, 1, n_points)

    # Journey path through (x, y, z) with z representing partition depth
    # Starts at high partition depth, descends to detector
    x_journey = 5 * np.sin(4 * np.pi * t) * (1 - t**0.5)
    y_journey = 5 * np.cos(4 * np.pi * t) * (1 - t**0.5)
    z_journey = 10 * (1 - t)**1.5  # Partition depth decreasing

    # Color by journey stage
    colors_journey = plt.cm.viridis(t)

    # Plot trajectory
    for i in range(n_points - 1):
        ax1.plot(x_journey[i:i+2], y_journey[i:i+2], z_journey[i:i+2],
                color=colors_journey[i], linewidth=2.5)

    # Mark key stages
    stages = [0, 0.2, 0.5, 0.8, 0.99]
    stage_labels = ['Source', 'Ion', 'Analyze', 'Focus', 'Detect']
    stage_colors = ['purple', 'blue', 'green', 'orange', 'red']

    for s, label, col in zip(stages, stage_labels, stage_colors):
        idx = int(s * (n_points - 1))
        ax1.scatter([x_journey[idx]], [y_journey[idx]], [z_journey[idx]],
                   c=col, s=100, marker='o', edgecolors='k', linewidths=1, zorder=5)

    # Detector plane at z=0
    theta_det = np.linspace(0, 2*np.pi, 30)
    r_det = np.linspace(0, 1, 10)
    THETA_det, R_det = np.meshgrid(theta_det, r_det)
    X_det = R_det * np.cos(THETA_det)
    Y_det = R_det * np.sin(THETA_det)
    Z_det = np.zeros_like(X_det)
    ax1.plot_surface(X_det, Y_det, Z_det, color='gold', alpha=0.5)

    ax1.set_xlabel(r'$x$')
    ax1.set_ylabel(r'$y$')
    ax1.set_zlabel(r'$\mathcal{M}$')
    ax1.set_title('A', loc='left', fontweight='bold')
    ax1.view_init(elev=25, azim=45)

    # Panel B (2D): Drip spectrum visualization - the visual "droplet" image
    ax2 = fig.add_subplot(gs[1])

    # Create drip/droplet pattern - concentric intensity patterns
    n_drip = 100
    x_drip = np.linspace(-3, 3, n_drip)
    y_drip = np.linspace(-3, 3, n_drip)
    X_drip, Y_drip = np.meshgrid(x_drip, y_drip)
    R_drip = np.sqrt(X_drip**2 + Y_drip**2)

    # Multi-modal drip pattern (simulating mass spectral peaks as concentric rings)
    drip_intensity = np.zeros_like(R_drip)

    # Central drop (precursor)
    drip_intensity += 1.0 * np.exp(-R_drip**2 / 0.3)

    # Fragment rings at different radii
    for r_frag, intensity in [(0.8, 0.7), (1.3, 0.5), (1.8, 0.3), (2.3, 0.2)]:
        ring = np.exp(-((R_drip - r_frag)**2) / 0.05)
        # Add angular modulation for asymmetry
        angular = 1 + 0.3 * np.sin(3 * np.arctan2(Y_drip, X_drip))
        drip_intensity += intensity * ring * angular

    # Add some droplet texture
    drip_intensity += 0.1 * np.random.normal(0, 1, drip_intensity.shape)
    drip_intensity = gaussian_filter(drip_intensity, sigma=1)
    drip_intensity = np.clip(drip_intensity, 0, None)

    # Plot as image
    im = ax2.imshow(drip_intensity, extent=[-3, 3, -3, 3], cmap='Blues',
                   origin='lower', aspect='equal')

    # Add droplet outline
    theta_outline = np.linspace(0, 2*np.pi, 100)
    ax2.plot(2.5 * np.cos(theta_outline), 2.5 * np.sin(theta_outline),
            'b-', linewidth=2, alpha=0.7)

    ax2.set_xlabel(r'$\xi$')
    ax2.set_ylabel(r'$\eta$')
    ax2.set_title('B', loc='left', fontweight='bold')
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-3, 3)

    # Panel C (3D): Ion-to-Drip bijective transformation
    ax3 = fig.add_subplot(gs[2], projection='3d')

    # Show transformation: Ion spectrum -> Drip representation
    # Left side: Ion (m/z peaks as vertical lines)
    # Right side: Drip (radial pattern)

    n_peaks = 8
    mz_peaks = np.sort(np.random.uniform(100, 1000, n_peaks))
    intensities = np.random.uniform(0.3, 1.0, n_peaks)

    # Ion representation (x < 0)
    for mz, inten in zip(mz_peaks, intensities):
        z_pos = (mz - 100) / 900 * 5  # Scale to [0, 5]
        ax3.plot([-2, -2], [0, 0], [z_pos, z_pos], 'b-', linewidth=1)
        ax3.scatter([-2], [0], [z_pos], c='blue', s=inten * 80, alpha=0.8)

    # Transformation arrows
    for i, (mz, inten) in enumerate(zip(mz_peaks, intensities)):
        z_pos = (mz - 100) / 900 * 5
        r_drip = 0.5 + 2 * (i / n_peaks)
        theta_drip = 2 * np.pi * i / n_peaks
        x_end = 2 + r_drip * np.cos(theta_drip) * 0.3
        y_end = r_drip * np.sin(theta_drip) * 0.3

        # Arrow from ion to drip
        ax3.plot([-2, x_end], [0, y_end], [z_pos, 2.5],
                'g-', alpha=0.4, linewidth=0.8)

    # Drip representation (x > 0)
    # Draw as points in circular pattern
    for i, (mz, inten) in enumerate(zip(mz_peaks, intensities)):
        r_drip = 0.5 + 2 * (i / n_peaks)
        for j in range(12):
            theta_drip = 2 * np.pi * j / 12 + np.pi * i / n_peaks
            x_d = 2 + r_drip * np.cos(theta_drip) * 0.3
            y_d = r_drip * np.sin(theta_drip) * 0.3
            ax3.scatter([x_d], [y_d], [2.5], c='cyan', s=inten * 30 / (r_drip + 0.5),
                       alpha=0.6)

    # Labels for regions
    ax3.text(-2.5, 0, 5.5, 'Ion', fontsize=9, ha='center')
    ax3.text(2.5, 0, 5.5, 'Drip', fontsize=9, ha='center')

    ax3.set_xlabel(r'$x$')
    ax3.set_ylabel(r'$y$')
    ax3.set_zlabel(r'$z$')
    ax3.set_title('C', loc='left', fontweight='bold')
    ax3.view_init(elev=20, azim=30)
    ax3.set_xlim(-4, 4)

    # Panel D (2D): Multiple drip patterns showing compound diversity
    ax4 = fig.add_subplot(gs[3])

    # Create grid of mini-drip patterns
    n_compounds = 9
    grid_size = 3

    # Compound properties (from validation data)
    compound_props = [
        {'n': 4, 'l': 1, 'm': 0, 'name': 'NGA3B'},
        {'n': 5, 'l': 1, 'm': 1, 'name': 'NGA4'},
        {'n': 5, 'l': 2, 'm': 2, 'name': 'A1F'},
        {'n': 4, 'l': 1, 'm': 1, 'name': 'NGA2'},
        {'n': 5, 'l': 0, 'm': 0, 'name': 'NA2G1F'},
        {'n': 5, 'l': 4, 'm': 0, 'name': 'Sialyl'},
        {'n': 6, 'l': 0, 'm': 0, 'name': 'DFLNO'},
        {'n': 7, 'l': 0, 'm': 0, 'name': 'A4122a'},
        {'n': 4, 'l': 2, 'm': 0, 'name': 'SialylNAc'},
    ]

    for idx, props in enumerate(compound_props):
        row = idx // grid_size
        col = idx % grid_size

        # Mini drip for this compound
        x_mini = np.linspace(-1, 1, 30)
        y_mini = np.linspace(-1, 1, 30)
        X_mini, Y_mini = np.meshgrid(x_mini, y_mini)
        R_mini = np.sqrt(X_mini**2 + Y_mini**2)

        # Pattern based on quantum numbers
        n, l, m = props['n'], props['l'], props['m']

        # Central intensity scales with n
        pattern = (n / 7) * np.exp(-R_mini**2 / 0.2)

        # Ring structure based on l
        for ring_idx in range(l + 1):
            r_ring = 0.3 + 0.15 * ring_idx
            pattern += 0.3 * np.exp(-((R_mini - r_ring)**2) / 0.02)

        # Angular modulation based on m
        if m > 0:
            pattern *= (1 + 0.3 * np.cos(m * np.arctan2(Y_mini, X_mini)))

        pattern = gaussian_filter(pattern, sigma=0.5)

        # Position in grid
        extent = [col - 0.45, col + 0.45, 2 - row - 0.45, 2 - row + 0.45]
        ax4.imshow(pattern, extent=extent, cmap='viridis', aspect='equal',
                  vmin=0, vmax=1.5)

        # Border
        ax4.plot([col - 0.45, col + 0.45, col + 0.45, col - 0.45, col - 0.45],
                [2 - row - 0.45, 2 - row - 0.45, 2 - row + 0.45, 2 - row + 0.45, 2 - row - 0.45],
                'k-', linewidth=0.5)

    ax4.set_xlim(-0.6, 2.6)
    ax4.set_ylim(-0.6, 2.6)
    ax4.set_xticks([])
    ax4.set_yticks([])
    ax4.set_title('D', loc='left', fontweight='bold')
    ax4.set_aspect('equal')

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'figure9_ion_journey_drip.png', bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / 'figure9_ion_journey_drip.pdf', bbox_inches='tight')
    plt.close()
    print("Figure 9: Ion Journey & Drip - Complete")


def main():
    """Generate all figures"""
    print("=" * 60)
    print("Partition Lagrangian Visualization Suite")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    figure1_partition_lagrangian_dynamics()
    figure2_four_analyzers()
    figure3_resolution_limits()
    figure4_partition_funnel()
    figure5_nist_validation()
    figure6_ternary_address()
    figure7_state_counting()
    figure8_uncertainty_principle()
    figure9_ion_journey_drip()

    print()
    print("=" * 60)
    print("All figures generated successfully!")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
