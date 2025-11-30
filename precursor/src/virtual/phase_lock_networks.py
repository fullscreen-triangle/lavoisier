"""
Phase-Lock Network Visualization - REAL DATA

Builds and visualizes phase-lock networks from ACTUAL S-Entropy coordinates.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy.spatial import distance_matrix
from pathlib import Path
import sys

# Add parent directory to path
script_dir = Path(__file__).parent
src_dir = script_dir.parent
sys.path.insert(0, str(src_dir))

from virtual.load_real_data import load_comparison_data

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300


def build_phase_lock_network(coords, max_nodes=500, threshold_percentile=10):
    """
    Build phase-lock network from REAL S-Entropy coordinates

    Args:
        coords: Nx3 array of [s_k, s_t, s_e]
        max_nodes: Maximum nodes to include (sample if needed)
        threshold_percentile: Distance percentile for edge creation

    Returns:
        networkx.Graph
    """
    # Check minimum number of droplets
    if len(coords) < 2:
        print(f"  ! Warning: Too few droplets ({len(coords)}) to build network")
        G = nx.Graph()
        if len(coords) == 1:
            G.add_node(0, s_k=coords[0,0], s_t=coords[0,1], s_e=coords[0,2], pos=(coords[0,0], coords[0,1]))
        return G

    # Sample if too many nodes
    if len(coords) > max_nodes:
        indices = np.random.choice(len(coords), max_nodes, replace=False)
        coords = coords[indices]

    # Compute pairwise distances in S-Entropy space
    dist_matrix = distance_matrix(coords, coords)

    # Get non-zero distances
    nonzero_dists = dist_matrix[dist_matrix > 0]

    if len(nonzero_dists) == 0:
        print(f"  ! Warning: No non-zero distances found")
        G = nx.Graph()
        for i, (s_k, s_t, s_e) in enumerate(coords):
            G.add_node(i, s_k=s_k, s_t=s_t, s_e=s_e, pos=(s_k, s_t))
        return G

    # Create edges for close pairs (phase-locked)
    threshold = np.percentile(nonzero_dists, threshold_percentile)

    G = nx.Graph()

    # Add nodes with S-Entropy coordinates
    for i, (s_k, s_t, s_e) in enumerate(coords):
        G.add_node(i, s_k=s_k, s_t=s_t, s_e=s_e, pos=(s_k, s_t))

    # Add edges for phase-locked pairs
    for i in range(len(coords)):
        for j in range(i+1, len(coords)):
            if dist_matrix[i, j] <= threshold:
                G.add_edge(i, j, weight=1.0/dist_matrix[i, j] if dist_matrix[i, j] > 0 else 1.0)

    return G


def visualize_phase_lock_network(G, platform_name, output_dir):
    """
    Visualize phase-lock network
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f'Phase-Lock Network - {platform_name}\n{G.number_of_nodes()} nodes, {G.number_of_edges()} edges',
                 fontsize=14, fontweight='bold')

    # Get node positions from S_k and S_t
    pos = nx.get_node_attributes(G, 'pos')
    s_e_values = list(nx.get_node_attributes(G, 's_e').values())

    # Network visualization in S_k-S_t space
    ax1 = axes[0]
    nx.draw_networkx_edges(G, pos, ax=ax1, alpha=0.2, width=0.5, edge_color='gray')
    nodes = nx.draw_networkx_nodes(G, pos, ax=ax1, node_size=20,
                                     node_color=s_e_values, cmap='viridis',
                                     vmin=min(s_e_values), vmax=max(s_e_values))
    ax1.set_xlabel('S-Knowledge', fontsize=11, fontweight='bold')
    ax1.set_ylabel('S-Time', fontsize=11, fontweight='bold')
    ax1.set_title('Network in S_k-S_t Space', fontsize=12)
    plt.colorbar(nodes, ax=ax1, label='S-Entropy')

    # Degree distribution
    ax2 = axes[1]
    degrees = [G.degree(n) for n in G.nodes()]
    ax2.hist(degrees, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Node Degree', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax2.set_title(f'Degree Distribution\nMean={np.mean(degrees):.1f}, Max={np.max(degrees)}',
                  fontsize=12)
    ax2.grid(alpha=0.3)

    plt.tight_layout()

    output_file = output_dir / f"phase_lock_network_{platform_name}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_file.name}")
    return output_file


def main():
    """Main visualization workflow"""
    print("="*80)
    print("PHASE-LOCK NETWORK VISUALIZATION - REAL DATA")
    print("="*80)

    # Determine paths
    script_dir = Path(__file__).parent
    precursor_root = script_dir.parent.parent
    results_dir = precursor_root / "results" / "fragmentation_comparison"
    output_dir = precursor_root / "visualizations"
    output_dir.mkdir(exist_ok=True)

    # Load REAL data
    print("\nLoading REAL S-Entropy data...")
    data = load_comparison_data(str(results_dir))

    if not data:
        print("ERROR: No data loaded!")
        return

    # Create networks and visualizations
    for platform_name, platform_data in data.items():
        print(f"\n{platform_name}: Building network from {platform_data['n_droplets']} droplets...")

        # Stack coordinates
        coords = np.column_stack([
            platform_data['s_knowledge'],
            platform_data['s_time'],
            platform_data['s_entropy']
        ])

        # Build network
        G = build_phase_lock_network(coords, max_nodes=500)
        print(f"  Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        # Visualize
        visualize_phase_lock_network(G, platform_name, output_dir)

    print("\n" + "="*80)
    print("✓ PHASE-LOCK NETWORK VISUALIZATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
