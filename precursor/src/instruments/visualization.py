"""
Visualization Module for Virtual Instruments

Creates publication-quality figures for:
- Phase-lock networks
- S-entropy space distributions
- Negation fields and potentials
- Shell structures
- Fragmentation topologies
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, Any, List, Optional, Tuple
import os

# Set publication-quality defaults
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['figure.dpi'] = 150


class InstrumentVisualizer:
    """
    Visualization tools for virtual instrument results.
    """
    
    def __init__(self, output_dir: str = "figures"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def save_figure(self, fig, name: str, formats: List[str] = ['png', 'pdf']):
        """Save figure in multiple formats"""
        for fmt in formats:
            path = os.path.join(self.output_dir, f"{name}.{fmt}")
            fig.savefig(path, dpi=300, bbox_inches='tight')
            print(f"Saved: {path}")
    
    def plot_s_entropy_space(self, states: List[Dict[str, float]],
                             title: str = "S-Entropy State Space") -> plt.Figure:
        """
        Plot molecules in S-entropy coordinate space.
        
        Args:
            states: List of dicts with 'S_k', 'S_t', 'S_e' keys
            title: Figure title
        """
        fig = plt.figure(figsize=(12, 10))
        
        # 3D scatter
        ax1 = fig.add_subplot(221, projection='3d')
        S_k = [s['S_k'] for s in states]
        S_t = [s['S_t'] for s in states]
        S_e = [s['S_e'] for s in states]
        
        scatter = ax1.scatter(S_k, S_t, S_e, c=S_e, cmap='viridis', alpha=0.6)
        ax1.set_xlabel(r'$S_k$ (Knowledge)')
        ax1.set_ylabel(r'$S_t$ (Temporal)')
        ax1.set_zlabel(r'$S_e$ (Entropy)')
        ax1.set_title('3D S-Space Distribution')
        plt.colorbar(scatter, ax=ax1, label=r'$S_e$')
        
        # 2D projections
        ax2 = fig.add_subplot(222)
        ax2.scatter(S_k, S_t, c=S_e, cmap='viridis', alpha=0.6)
        ax2.set_xlabel(r'$S_k$')
        ax2.set_ylabel(r'$S_t$')
        ax2.set_title(r'$S_k$-$S_t$ Projection')
        
        ax3 = fig.add_subplot(223)
        ax3.scatter(S_k, S_e, c=S_t, cmap='plasma', alpha=0.6)
        ax3.set_xlabel(r'$S_k$')
        ax3.set_ylabel(r'$S_e$')
        ax3.set_title(r'$S_k$-$S_e$ Projection')
        
        ax4 = fig.add_subplot(224)
        ax4.scatter(S_t, S_e, c=S_k, cmap='coolwarm', alpha=0.6)
        ax4.set_xlabel(r'$S_t$')
        ax4.set_ylabel(r'$S_e$')
        ax4.set_title(r'$S_t$-$S_e$ Projection')
        
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        return fig
    
    def plot_phase_lock_network(self, adjacency: np.ndarray,
                                 clusters: List[Dict],
                                 title: str = "Phase-Lock Network") -> plt.Figure:
        """
        Visualize phase-lock network topology.
        
        Args:
            adjacency: Adjacency matrix
            clusters: List of cluster info dicts
            title: Figure title
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Left: Network visualization
        ax1 = axes[0]
        n = adjacency.shape[0]
        
        # Position nodes in a circle
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        x = np.cos(angles)
        y = np.sin(angles)
        
        # Draw edges
        for i in range(n):
            for j in range(i + 1, n):
                if adjacency[i, j] > 0:
                    ax1.plot([x[i], x[j]], [y[i], y[j]], 
                            'b-', alpha=adjacency[i, j], linewidth=0.5)
        
        # Draw nodes with cluster colors
        cluster_colors = plt.cm.tab10(np.linspace(0, 1, len(clusters) + 1))
        node_colors = ['gray'] * n
        
        for idx, cluster in enumerate(clusters):
            for node_idx in cluster.get('node_indices', []):
                if node_idx < n:
                    node_colors[node_idx] = cluster_colors[idx]
        
        ax1.scatter(x, y, c=node_colors, s=50, zorder=5)
        ax1.set_xlim(-1.3, 1.3)
        ax1.set_ylim(-1.3, 1.3)
        ax1.set_aspect('equal')
        ax1.set_title('Network Topology')
        ax1.axis('off')
        
        # Right: Adjacency matrix
        ax2 = axes[1]
        im = ax2.imshow(adjacency, cmap='Blues')
        ax2.set_xlabel('Node')
        ax2.set_ylabel('Node')
        ax2.set_title('Adjacency Matrix')
        plt.colorbar(im, ax=ax2, label='Coupling Strength')
        
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        return fig
    
    def plot_negation_field(self, result: Dict[str, Any],
                            title: str = "Negation Field") -> plt.Figure:
        """
        Visualize negation field (electric-like potential and field).
        
        Args:
            result: Output from NegationFieldMapper.measure()
            title: Figure title
        """
        fig = plt.figure(figsize=(14, 10))
        
        grid = result['grid']
        X, Y = grid['X'], grid['Y']
        Z = result['Z']
        
        # Potential
        ax1 = fig.add_subplot(221)
        potential = np.clip(result['potential'], -10, 0)  # Clip for visualization
        im1 = ax1.contourf(X, Y, potential, levels=50, cmap='RdBu_r')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title(f'Potential φ(r) = -{Z}/r')
        ax1.set_aspect('equal')
        plt.colorbar(im1, ax=ax1, label='φ')
        
        # Field magnitude
        ax2 = fig.add_subplot(222)
        field_mag = np.clip(result['field_magnitude'], 0, 10)
        im2 = ax2.contourf(X, Y, field_mag, levels=50, cmap='hot')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_title(f'Field Magnitude |E| = {Z}/r²')
        ax2.set_aspect('equal')
        plt.colorbar(im2, ax=ax2, label='|E|')
        
        # Field vectors
        ax3 = fig.add_subplot(223)
        skip = max(1, grid['size'] // 15)
        ax3.quiver(X[::skip, ::skip], Y[::skip, ::skip],
                  result['field_x'][::skip, ::skip],
                  result['field_y'][::skip, ::skip],
                  color='blue', alpha=0.7)
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        ax3.set_title('Field Vectors')
        ax3.set_aspect('equal')
        ax3.set_xlim(-grid['r_max'], grid['r_max'])
        ax3.set_ylim(-grid['r_max'], grid['r_max'])
        
        # Wave functions
        ax4 = fig.add_subplot(224)
        for label, wf_data in result.get('wave_functions', {}).items():
            radii = wf_data['radii']
            prob = wf_data['probability']
            ax4.plot(radii, prob, label=label)
        ax4.set_xlabel('r')
        ax4.set_ylabel(r'$|\psi|^2$')
        ax4.set_title('Boundary Probability Distributions')
        ax4.legend()
        ax4.set_xlim(0, grid['r_max'])
        
        fig.suptitle(f"{title} (Z = {Z})", fontsize=16)
        plt.tight_layout()
        
        return fig
    
    def plot_shell_structure(self, result: Dict[str, Any],
                             title: str = "Non-Actualisation Shell Structure") -> plt.Figure:
        """
        Visualize non-actualisation shell structure.
        
        Args:
            result: Output from NonActualisationShellScanner.measure()
            title: Figure title
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        shells = result['shells']
        radii = list(shells.keys())
        
        # Shell sizes
        ax1 = axes[0, 0]
        theoretical = [shells[r]['theoretical_count'] for r in radii]
        measured = [shells[r]['measured_count'] for r in radii]
        ax1.bar(np.array(radii) - 0.2, theoretical, 0.4, label='Theoretical', alpha=0.7)
        ax1.bar(np.array(radii) + 0.2, measured, 0.4, label='Measured', alpha=0.7)
        ax1.set_xlabel('Shell Radius r')
        ax1.set_ylabel('Count')
        ax1.set_title('Shell Size: |N_r| = k^r')
        ax1.legend()
        ax1.set_yscale('log')
        
        # Paired fraction
        ax2 = axes[0, 1]
        paired = [shells[r]['paired_fraction'] for r in radii]
        colors = ['green' if not shells[r]['is_dark'] else 'purple' for r in radii]
        ax2.bar(radii, paired, color=colors, alpha=0.7)
        ax2.axhline(y=0.5, color='red', linestyle='--', label='50% threshold')
        ax2.set_xlabel('Shell Radius r')
        ax2.set_ylabel('Paired Fraction')
        ax2.set_title('Paired (Ordinary) vs Unpaired (Dark)')
        ax2.legend(['Threshold', 'Ordinary', 'Dark'])
        
        # Dark vs ordinary accumulation
        ax3 = axes[1, 0]
        cum_ordinary = []
        cum_dark = []
        ordinary_sum = 0
        dark_sum = 0
        for r in radii:
            n = shells[r]['measured_count']
            p = shells[r]['paired_fraction']
            if shells[r]['is_dark']:
                dark_sum += n
            else:
                ordinary_sum += n * p
                dark_sum += n * (1 - p)
            cum_ordinary.append(ordinary_sum)
            cum_dark.append(dark_sum)
        
        ax3.fill_between(radii, cum_ordinary, alpha=0.5, label='Ordinary Matter', color='green')
        ax3.fill_between(radii, cum_dark, alpha=0.5, label='Dark Matter', color='purple')
        ax3.set_xlabel('Shell Radius r')
        ax3.set_ylabel('Cumulative Count')
        ax3.set_title('Matter Accumulation')
        ax3.legend()
        
        # Ratio
        ax4 = axes[1, 1]
        ratio_text = (
            f"Dark/Ordinary Ratio\n\n"
            f"Measured: {result['dark_ordinary_ratio']:.2f}\n"
            f"Theoretical: {result['theoretical_ratio']:.2f}\n\n"
            f"Branching factor k = {result['branching_factor']}\n"
            f"Pairing radius = {result['pairing_radius']}\n\n"
            f"Ordinary matter: Paired non-actualisations\n"
            f"Dark matter: Unpaired non-actualisations"
        )
        ax4.text(0.5, 0.5, ratio_text, transform=ax4.transAxes,
                fontsize=12, va='center', ha='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax4.axis('off')
        
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        return fig
    
    def plot_entropy_unification(self, result: Dict[str, Any],
                                  title: str = "Entropy Unification") -> plt.Figure:
        """
        Visualize the fundamental equivalence: S_osc = S_cat = S_part
        
        Args:
            result: Output from CrossInstrumentConvergenceValidator
            title: Figure title
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Left: Bar chart of three entropies
        ax1 = axes[0]
        labels = ['Oscillatory\n$S_{osc}$', 'Categorical\n$S_{cat}$', 'Partition\n$S_{part}$']
        values = [result['S_oscillatory'], result['S_categorical'], result['S_partition']]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        bars = ax1.bar(labels, values, color=colors, alpha=0.8)
        ax1.axhline(y=result['S_unified_formula'], color='red', linestyle='--', 
                   label=f'Unified: S = k_B×M×ln(n)')
        ax1.set_ylabel('Entropy (J/K)')
        ax1.set_title(f'Three Derivations, One Formula\nM={result["M"]}, n={result["n"]}')
        ax1.legend()
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.2e}', ha='center', va='bottom', fontsize=9)
        
        # Right: Conceptual diagram
        ax2 = axes[1]
        
        # Draw three circles converging
        circle_centers = [(0.2, 0.7), (0.8, 0.7), (0.5, 0.2)]
        circle_labels = ['Oscillation\n(Modes)', 'Category\n(Dimensions)', 'Partition\n(Levels)']
        
        for (cx, cy), label, color in zip(circle_centers, circle_labels, colors):
            circle = Circle((cx, cy), 0.15, fill=True, alpha=0.3, color=color)
            ax2.add_patch(circle)
            ax2.text(cx, cy, label, ha='center', va='center', fontsize=10)
        
        # Central convergence point
        ax2.plot(0.5, 0.5, 'ko', markersize=15)
        ax2.text(0.5, 0.35, r'$S = k_B M \ln n$', ha='center', fontsize=14, fontweight='bold')
        
        # Arrows to center
        for (cx, cy) in circle_centers:
            ax2.annotate('', xy=(0.5, 0.5), xytext=(cx, cy),
                        arrowprops=dict(arrowstyle='->', color='gray', lw=2))
        
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.set_aspect('equal')
        ax2.axis('off')
        ax2.set_title('Fundamental Equivalence')
        
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        return fig
    
    def plot_heat_entropy_decoupling(self, result: Dict[str, Any],
                                      title: str = "Heat-Entropy Decoupling") -> plt.Figure:
        """
        Visualize that heat fluctuates while entropy always increases.
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Heat distribution (fluctuates around 0)
        ax1 = axes[0]
        heat_std = result['heat_flow_std']
        x = np.linspace(-3*heat_std, 3*heat_std, 100)
        ax1.plot(x, np.exp(-x**2 / (2*heat_std**2)), 'b-', linewidth=2)
        ax1.axvline(x=0, color='red', linestyle='--', label='Mean')
        ax1.fill_between(x, np.exp(-x**2 / (2*heat_std**2)), alpha=0.3)
        ax1.set_xlabel('Heat Flow (arbitrary units)')
        ax1.set_ylabel('Probability')
        ax1.set_title('Heat: Bidirectional\n(can be + or -)')
        ax1.legend()
        
        # Entropy (always positive)
        ax2 = axes[1]
        dS_mean = result['dS_total_mean']
        x = np.linspace(0, 3*dS_mean, 100)
        ax2.plot(x, x*np.exp(-x/dS_mean), 'g-', linewidth=2)
        ax2.fill_between(x, x*np.exp(-x/dS_mean), alpha=0.3, color='green')
        ax2.axvline(x=0, color='red', linestyle='--')
        ax2.set_xlabel('Entropy Change ΔS')
        ax2.set_ylabel('Probability')
        ax2.set_title('Entropy: Unidirectional\n(always > 0)')
        
        # Decoupling summary
        ax3 = axes[2]
        ax3.text(0.5, 0.8, 'DECOUPLING RESULT', ha='center', fontsize=14, fontweight='bold')
        ax3.text(0.5, 0.6, f'Heat correlation with entropy: {result["heat_entropy_correlation"]:.3f}', 
                ha='center', fontsize=11)
        ax3.text(0.5, 0.45, f'Heat fluctuates: {result["heat_fluctuates"]}', ha='center', fontsize=11)
        ax3.text(0.5, 0.3, f'Entropy always positive: {result["dS_total_all_positive"]}', 
                ha='center', fontsize=11)
        ax3.text(0.5, 0.1, 'The Second Law protects ENTROPY,\nnot heat direction.', 
                ha='center', fontsize=11, style='italic',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.axis('off')
        
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        return fig


def create_all_figures(output_dir: str = "precursor/results/instrument_figures"):
    """
    Generate all publication figures from instruments.
    """
    from .partition_coordinates import PartitionCoordinateMeasurer
    from .thermodynamic import CrossInstrumentConvergenceValidator, HeatEntropyDecoupler
    from .network import PhaseLockNetworkMapper
    from .categorical_navigation import NonActualisationShellScanner
    from .field import NegationFieldMapper
    
    visualizer = InstrumentVisualizer(output_dir)
    
    print("Generating instrument figures...")
    
    # 1. S-entropy space
    print("  - S-entropy space...")
    mapper = PhaseLockNetworkMapper()
    network_result = mapper.measure(n_molecules=100)
    states = [{'S_k': s.S_coords.S_k, 'S_t': s.S_coords.S_t, 'S_e': s.S_coords.S_e} 
              for s in [mapper.oscillator.create_categorical_state() for _ in range(200)]]
    fig = visualizer.plot_s_entropy_space(states)
    visualizer.save_figure(fig, "s_entropy_space")
    plt.close(fig)
    
    # 2. Phase-lock network
    print("  - Phase-lock network...")
    clusters = [{'node_indices': c.node_indices} for c in network_result['clusters']]
    fig = visualizer.plot_phase_lock_network(network_result['adjacency_matrix'], clusters)
    visualizer.save_figure(fig, "phase_lock_network")
    plt.close(fig)
    
    # 3. Negation field
    print("  - Negation field...")
    field_mapper = NegationFieldMapper()
    field_result = field_mapper.measure(Z=2, grid_size=50)
    fig = visualizer.plot_negation_field(field_result)
    visualizer.save_figure(fig, "negation_field")
    plt.close(fig)
    
    # 4. Shell structure
    print("  - Shell structure...")
    scanner = NonActualisationShellScanner()
    shell_result = scanner.measure(max_radius=8)
    fig = visualizer.plot_shell_structure(shell_result)
    visualizer.save_figure(fig, "shell_structure")
    plt.close(fig)
    
    # 5. Entropy unification
    print("  - Entropy unification...")
    validator = CrossInstrumentConvergenceValidator()
    entropy_result = validator.measure(M=3, n=3)
    fig = visualizer.plot_entropy_unification(entropy_result)
    visualizer.save_figure(fig, "entropy_unification")
    plt.close(fig)
    
    # 6. Heat-entropy decoupling
    print("  - Heat-entropy decoupling...")
    decoupler = HeatEntropyDecoupler()
    decoupling_result = decoupler.measure(n_transfers=500)
    fig = visualizer.plot_heat_entropy_decoupling(decoupling_result)
    visualizer.save_figure(fig, "heat_entropy_decoupling")
    plt.close(fig)
    
    print(f"\nAll figures saved to: {output_dir}")


if __name__ == "__main__":
    create_all_figures()

