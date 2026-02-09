"""
Graph Analysis for Proteomics Fragment Networks
=================================================

Comprehensive graph-based analysis of fragment ion networks, including:
- Fragment graph construction from MS/MS spectra
- Graph topology metrics (degree, centrality, clustering)
- Cardinal direction graphs from Empty Dictionary
- Community detection and motif analysis
- S-Entropy graph embedding
- Visualization generation

Author: Kundai Sachikonye
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict
import time
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# Standard amino acid masses (monoisotopic) - always available
AMINO_ACID_MASSES = {
    'A': 71.03711,   'R': 156.10111,  'N': 114.04293,
    'D': 115.02694,  'C': 103.00919,  'E': 129.04259,
    'Q': 128.05858,  'G': 57.02146,   'H': 137.05891,
    'I': 113.08406,  'L': 113.08406,  'K': 128.09496,
    'M': 131.04049,  'F': 147.06841,  'P': 97.05276,
    'S': 87.03203,   'T': 101.04768,  'W': 186.07931,
    'Y': 163.06333,  'V': 99.06841,
}

# Import from existing modules
try:
    from .st_stellas_sequence import (
        StStellasSequenceTransformer,
        AMINO_ACID_CARDINAL_COORDS
    )
    STELLAS_AVAILABLE = True
except ImportError:
    STELLAS_AVAILABLE = False

try:
    from .empty_dictionary_proteomics import (
        EmptyDictionaryTransformer,
        AMINO_ACID_TO_CODON,
        NUCLEOTIDE_CARDINAL
    )
    EMPTY_DICT_AVAILABLE = True
except ImportError:
    EMPTY_DICT_AVAILABLE = False


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class GraphMetrics:
    """Computed metrics for a graph."""
    n_nodes: int
    n_edges: int
    density: float
    avg_degree: float
    max_degree: int
    avg_clustering: float
    avg_path_length: float
    diameter: int
    n_components: int
    largest_component_size: int

    # Centrality metrics (averages)
    avg_betweenness: float = 0.0
    avg_closeness: float = 0.0
    avg_eigenvector: float = 0.0

    # Degree distribution
    degree_histogram: Dict[int, int] = field(default_factory=dict)

    # S-Entropy metrics
    graph_entropy: float = 0.0


@dataclass
class FragmentGraphAnalysis:
    """Complete analysis of a fragment graph."""
    spectrum_id: str
    precursor_mz: float
    n_fragments: int

    # Graph object
    graph: nx.DiGraph = None

    # Metrics
    metrics: GraphMetrics = None

    # Node attributes
    node_mz: Dict[float, float] = field(default_factory=dict)
    node_intensity: Dict[float, float] = field(default_factory=dict)
    node_sentropy: Dict[float, np.ndarray] = field(default_factory=dict)

    # Edge attributes
    edge_amino_acids: Dict[Tuple[float, float], str] = field(default_factory=dict)
    edge_confidence: Dict[Tuple[float, float], float] = field(default_factory=dict)

    # Paths
    longest_path: List[float] = field(default_factory=list)
    inferred_sequence: str = ""

    # Cardinal walk (from Empty Dictionary)
    cardinal_graph: nx.DiGraph = None
    cardinal_metrics: GraphMetrics = None


@dataclass
class CardinalWalkGraph:
    """Graph representation of cardinal direction walk."""
    sequence: str
    nucleotides: str

    # Graph with positions as nodes
    graph: nx.Graph = None

    # Positions
    positions: np.ndarray = None

    # Metrics
    metrics: GraphMetrics = None

    # Closure
    is_closed: bool = False
    closure_distance: float = 0.0


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

class FragmentGraphBuilder:
    """
    Build and analyze fragment ion graphs from MS/MS spectra.
    """

    def __init__(
        self,
        mass_tolerance: float = 0.5,
        min_intensity_ratio: float = 0.01
    ):
        self.mass_tolerance = mass_tolerance
        self.min_intensity_ratio = min_intensity_ratio

        if STELLAS_AVAILABLE:
            self.stellas = StStellasSequenceTransformer(mass_tolerance=mass_tolerance)
        else:
            self.stellas = None

        if EMPTY_DICT_AVAILABLE:
            self.empty_dict = EmptyDictionaryTransformer(mass_tolerance=mass_tolerance)
        else:
            self.empty_dict = None

    def build_fragment_graph(
        self,
        mz_array: np.ndarray,
        intensity_array: np.ndarray,
        precursor_mz: float,
        spectrum_id: str = "unknown"
    ) -> FragmentGraphAnalysis:
        """
        Build a directed fragment graph from MS/MS spectrum.

        Nodes: Fragment ions (m/z values)
        Edges: Mass differences corresponding to amino acids
        """
        # Filter low intensity peaks
        max_intensity = np.max(intensity_array)
        mask = intensity_array >= (self.min_intensity_ratio * max_intensity)
        mz_filtered = mz_array[mask]
        intensity_filtered = intensity_array[mask]

        # Sort by m/z
        sorted_idx = np.argsort(mz_filtered)
        mz_sorted = mz_filtered[sorted_idx]
        intensity_sorted = intensity_filtered[sorted_idx]

        # Build graph
        G = nx.DiGraph()

        # Add nodes
        node_mz = {}
        node_intensity = {}
        node_sentropy = {}

        for i, (mz, intensity) in enumerate(zip(mz_sorted, intensity_sorted)):
            G.add_node(mz)
            node_mz[mz] = mz
            node_intensity[mz] = intensity

            # Compute S-entropy coordinates
            mz_norm = mz / precursor_mz
            intensity_norm = np.log1p(intensity) / np.log1p(max_intensity)
            position_norm = i / max(1, len(mz_sorted) - 1)

            s_k = -np.log2(mz_norm + 0.01) * intensity_norm
            s_t = position_norm
            s_e = intensity_norm * (1 - position_norm)

            node_sentropy[mz] = np.array([s_k, s_t, s_e])

        # Add edges based on amino acid mass differences
        edge_amino_acids = {}
        edge_confidence = {}

        for i, mz_i in enumerate(mz_sorted):
            for j in range(i + 1, len(mz_sorted)):
                mz_j = mz_sorted[j]
                mass_diff = mz_j - mz_i

                # Check if matches any amino acid
                aa = self._match_mass_to_aa(mass_diff)

                if aa is not None:
                    # Compute edge confidence
                    intensity_factor = min(intensity_sorted[i], intensity_sorted[j]) / max(intensity_sorted[i], intensity_sorted[j])
                    mass_error = abs(mass_diff - AMINO_ACID_MASSES[aa])
                    mass_factor = 1.0 - (mass_error / self.mass_tolerance)
                    confidence = 0.5 * intensity_factor + 0.5 * mass_factor

                    G.add_edge(mz_i, mz_j, weight=1.0 - confidence, amino_acid=aa)
                    edge_amino_acids[(mz_i, mz_j)] = aa
                    edge_confidence[(mz_i, mz_j)] = confidence

        # Compute metrics
        metrics = self._compute_graph_metrics(G)

        # Find longest path and infer sequence
        longest_path, inferred_sequence = self._find_longest_path(G, edge_amino_acids)

        # Build cardinal walk graph if sequence found
        cardinal_graph = None
        cardinal_metrics = None

        if inferred_sequence and self.empty_dict:
            cardinal_graph, cardinal_metrics = self._build_cardinal_graph(inferred_sequence)

        return FragmentGraphAnalysis(
            spectrum_id=spectrum_id,
            precursor_mz=precursor_mz,
            n_fragments=len(mz_sorted),
            graph=G,
            metrics=metrics,
            node_mz=node_mz,
            node_intensity=node_intensity,
            node_sentropy=node_sentropy,
            edge_amino_acids=edge_amino_acids,
            edge_confidence=edge_confidence,
            longest_path=longest_path,
            inferred_sequence=inferred_sequence,
            cardinal_graph=cardinal_graph,
            cardinal_metrics=cardinal_metrics
        )

    def _match_mass_to_aa(self, mass_diff: float) -> Optional[str]:
        """Match mass difference to amino acid."""
        for aa, mass in AMINO_ACID_MASSES.items():
            if abs(mass_diff - mass) < self.mass_tolerance:
                return aa
        return None

    def _compute_graph_metrics(self, G: nx.DiGraph) -> GraphMetrics:
        """Compute comprehensive graph metrics."""
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()

        if n_nodes == 0:
            return GraphMetrics(
                n_nodes=0, n_edges=0, density=0, avg_degree=0,
                max_degree=0, avg_clustering=0, avg_path_length=0,
                diameter=0, n_components=0, largest_component_size=0
            )

        # Basic metrics
        density = nx.density(G)

        # Degree metrics
        degrees = [d for n, d in G.degree()]
        avg_degree = np.mean(degrees) if degrees else 0
        max_degree = max(degrees) if degrees else 0

        # Clustering (use undirected version)
        G_undirected = G.to_undirected()
        avg_clustering = nx.average_clustering(G_undirected)

        # Connected components
        components = list(nx.weakly_connected_components(G))
        n_components = len(components)
        largest_component_size = max(len(c) for c in components) if components else 0

        # Path metrics (on largest component)
        avg_path_length = 0
        diameter = 0

        if largest_component_size > 1:
            largest_component = max(components, key=len)
            subgraph = G.subgraph(largest_component)

            try:
                avg_path_length = nx.average_shortest_path_length(subgraph)
            except:
                avg_path_length = 0

            try:
                diameter = nx.diameter(subgraph.to_undirected())
            except:
                diameter = 0

        # Centrality metrics
        avg_betweenness = 0
        avg_closeness = 0
        avg_eigenvector = 0

        if n_nodes > 1:
            try:
                betweenness = nx.betweenness_centrality(G)
                avg_betweenness = np.mean(list(betweenness.values()))
            except:
                pass

            try:
                closeness = nx.closeness_centrality(G)
                avg_closeness = np.mean(list(closeness.values()))
            except:
                pass

            try:
                eigenvector = nx.eigenvector_centrality_numpy(G, max_iter=100)
                avg_eigenvector = np.mean(list(eigenvector.values()))
            except:
                pass

        # Degree histogram
        degree_histogram = defaultdict(int)
        for d in degrees:
            degree_histogram[d] += 1

        # Graph entropy (based on degree distribution)
        if n_nodes > 0:
            degree_probs = np.array(degrees) / (2 * n_edges + 1)
            degree_probs = degree_probs[degree_probs > 0]
            graph_entropy = -np.sum(degree_probs * np.log2(degree_probs + 1e-10))
        else:
            graph_entropy = 0

        return GraphMetrics(
            n_nodes=n_nodes,
            n_edges=n_edges,
            density=density,
            avg_degree=avg_degree,
            max_degree=max_degree,
            avg_clustering=avg_clustering,
            avg_path_length=avg_path_length,
            diameter=diameter,
            n_components=n_components,
            largest_component_size=largest_component_size,
            avg_betweenness=avg_betweenness,
            avg_closeness=avg_closeness,
            avg_eigenvector=avg_eigenvector,
            degree_histogram=dict(degree_histogram),
            graph_entropy=graph_entropy
        )

    def _find_longest_path(
        self,
        G: nx.DiGraph,
        edge_amino_acids: Dict
    ) -> Tuple[List[float], str]:
        """Find longest path through graph and infer sequence."""
        if len(G.nodes()) < 2:
            return [], ""

        nodes = sorted(G.nodes())

        # Try to find path from smallest to largest m/z
        try:
            path = nx.dijkstra_path(G, nodes[0], nodes[-1])
        except nx.NetworkXNoPath:
            # Find longest path in largest component
            components = list(nx.weakly_connected_components(G))
            if not components:
                return [], ""

            largest = max(components, key=len)
            subgraph = G.subgraph(largest)
            sub_nodes = sorted(subgraph.nodes())

            if len(sub_nodes) < 2:
                return list(sub_nodes), ""

            try:
                path = nx.dijkstra_path(subgraph, sub_nodes[0], sub_nodes[-1])
            except:
                path = sub_nodes

        # Extract sequence from path
        sequence = []
        for i in range(len(path) - 1):
            edge = (path[i], path[i + 1])
            if edge in edge_amino_acids:
                sequence.append(edge_amino_acids[edge])

        return path, ''.join(sequence)

    def _build_cardinal_graph(
        self,
        sequence: str
    ) -> Tuple[nx.DiGraph, GraphMetrics]:
        """Build cardinal direction graph from sequence."""
        if not self.empty_dict:
            return None, None

        # Get cardinal walk
        walk = self.empty_dict.peptide_cardinal_walk(sequence)

        # Build graph from walk positions
        G = nx.DiGraph()

        for i, pos in enumerate(walk.positions):
            G.add_node(i, x=pos[0], y=pos[1])

            if i > 0:
                G.add_edge(i - 1, i)

        # Add closure edge if path closes
        if walk.is_closed and len(walk.positions) > 2:
            G.add_edge(len(walk.positions) - 1, 0, closure=True)

        metrics = self._compute_graph_metrics(G)

        return G, metrics


# ============================================================================
# GRAPH VISUALIZATION
# ============================================================================

class GraphVisualizer:
    """Generate visualizations for fragment and cardinal graphs."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_fragment_graph(
        self,
        analysis: FragmentGraphAnalysis,
        filename: str = None
    ) -> Path:
        """Plot fragment ion graph with annotations."""
        G = analysis.graph

        if len(G.nodes()) == 0:
            return None

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(f"Fragment Graph Analysis: {analysis.spectrum_id}", fontsize=14, fontweight='bold')

        # 1. Fragment graph layout
        ax1 = axes[0, 0]

        # Position nodes by m/z
        pos = {}
        nodes = sorted(G.nodes())
        for i, node in enumerate(nodes):
            pos[node] = (node, i % 5)  # x = m/z, y = layer

        # Node colors by intensity
        intensities = [analysis.node_intensity.get(n, 0) for n in G.nodes()]
        max_int = max(intensities) if intensities else 1
        node_colors = [i / max_int for i in intensities]

        # Edge colors by confidence
        edge_colors = [analysis.edge_confidence.get((u, v), 0.5) for u, v in G.edges()]

        nx.draw_networkx_nodes(G, pos, ax=ax1, node_size=100,
                              node_color=node_colors, cmap='YlOrRd', alpha=0.8)
        nx.draw_networkx_edges(G, pos, ax=ax1, edge_color=edge_colors,
                              edge_cmap=plt.cm.Blues, arrows=True,
                              arrowsize=10, alpha=0.6)

        # Label edges with amino acids
        edge_labels = {(u, v): aa for (u, v), aa in analysis.edge_amino_acids.items()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax1, font_size=8)

        ax1.set_xlabel('m/z')
        ax1.set_ylabel('Layer')
        ax1.set_title(f'Fragment Graph (n={analysis.n_fragments}, edges={G.number_of_edges()})')

        # 2. Degree distribution
        ax2 = axes[0, 1]

        degrees = [d for n, d in G.degree()]
        if degrees:
            ax2.hist(degrees, bins=range(max(degrees) + 2), edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Degree')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Degree Distribution')

        # 3. S-Entropy space
        ax3 = axes[1, 0]

        if analysis.node_sentropy:
            sentropy_coords = np.array(list(analysis.node_sentropy.values()))
            if len(sentropy_coords) > 0:
                scatter = ax3.scatter(sentropy_coords[:, 0], sentropy_coords[:, 1],
                                     c=sentropy_coords[:, 2], cmap='viridis',
                                     s=50, alpha=0.7)
                plt.colorbar(scatter, ax=ax3, label='S_entropy')

        ax3.set_xlabel('S_knowledge')
        ax3.set_ylabel('S_time')
        ax3.set_title('S-Entropy Coordinates')

        # 4. Metrics summary
        ax4 = axes[1, 1]
        ax4.axis('off')

        metrics = analysis.metrics
        if metrics:
            text = f"""Graph Metrics Summary

Nodes: {metrics.n_nodes}
Edges: {metrics.n_edges}
Density: {metrics.density:.4f}
Avg Degree: {metrics.avg_degree:.2f}
Max Degree: {metrics.max_degree}
Avg Clustering: {metrics.avg_clustering:.4f}
Components: {metrics.n_components}
Largest Component: {metrics.largest_component_size}
Avg Path Length: {metrics.avg_path_length:.2f}
Diameter: {metrics.diameter}

Centrality Metrics:
  Betweenness: {metrics.avg_betweenness:.4f}
  Closeness: {metrics.avg_closeness:.4f}
  Eigenvector: {metrics.avg_eigenvector:.4f}

Graph Entropy: {metrics.graph_entropy:.4f}

Inferred Sequence: {analysis.inferred_sequence[:20] if analysis.inferred_sequence else 'N/A'}...
"""
            ax4.text(0.1, 0.9, text, transform=ax4.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace')

        plt.tight_layout()

        if filename is None:
            filename = f"fragment_graph_{analysis.spectrum_id.replace('/', '_').replace(';', '_')}.png"

        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        return output_path

    def plot_cardinal_walk(
        self,
        sequence: str,
        transformer: 'EmptyDictionaryTransformer',
        filename: str = None
    ) -> Path:
        """Plot cardinal direction walk for a sequence."""
        walk = transformer.peptide_cardinal_walk(sequence)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"Cardinal Walk: {sequence}", fontsize=14, fontweight='bold')

        # 1. 2D Walk trajectory
        ax1 = axes[0]

        positions = walk.positions
        if len(positions) > 1:
            # Plot path
            ax1.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2, alpha=0.7)

            # Color points by position
            colors = np.linspace(0, 1, len(positions))
            scatter = ax1.scatter(positions[:, 0], positions[:, 1], c=colors,
                                 cmap='coolwarm', s=100, zorder=5)

            # Mark start and end
            ax1.scatter(positions[0, 0], positions[0, 1], c='green', s=200,
                       marker='s', zorder=10, label='Start')
            ax1.scatter(positions[-1, 0], positions[-1, 1], c='red', s=200,
                       marker='^', zorder=10, label='End')

            # Draw closure arrow
            ax1.annotate('', xy=(0, 0), xytext=(walk.final_position[0], walk.final_position[1]),
                        arrowprops=dict(arrowstyle='->', color='red', linestyle='--', alpha=0.5))

            plt.colorbar(scatter, ax=ax1, label='Step')

        ax1.axhline(y=0, color='k', linestyle='-', alpha=0.2)
        ax1.axvline(x=0, color='k', linestyle='-', alpha=0.2)
        ax1.set_xlabel('X (East-West)')
        ax1.set_ylabel('Y (North-South)')
        ax1.set_title(f'Trajectory (closure={walk.closure_distance:.2f})')
        ax1.legend()
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)

        # 2. Direction sequence
        ax2 = axes[1]

        direction_colors = {'N': 'green', 'S': 'blue', 'E': 'red', 'W': 'orange'}
        directions = list(walk.direction_sequence)

        if directions:
            x_pos = range(len(directions))
            colors = [direction_colors.get(d, 'gray') for d in directions]
            ax2.bar(x_pos, [1] * len(directions), color=colors)
            ax2.set_xticks(x_pos[::3])  # Show every 3rd label
            ax2.set_xticklabels([directions[i] if i < len(directions) else ''
                                for i in range(0, len(directions), 3)], fontsize=8)

        ax2.set_xlabel('Step')
        ax2.set_ylabel('Direction')
        ax2.set_title('Direction Sequence')

        # Legend
        patches = [mpatches.Patch(color=c, label=d) for d, c in direction_colors.items()]
        ax2.legend(handles=patches, loc='upper right')

        # 3. S-Entropy trajectory
        ax3 = axes[2]

        if walk.s_entropy_trajectory is not None and len(walk.s_entropy_trajectory) > 0:
            steps = range(len(walk.s_entropy_trajectory))
            ax3.plot(steps, walk.s_entropy_trajectory[:, 0], 'b-', label='S_k', linewidth=2)
            ax3.plot(steps, walk.s_entropy_trajectory[:, 1], 'g-', label='S_t', linewidth=2)
            ax3.plot(steps, walk.s_entropy_trajectory[:, 2], 'r-', label='S_e', linewidth=2)
            ax3.legend()

        ax3.set_xlabel('Step')
        ax3.set_ylabel('S-Entropy')
        ax3.set_title('S-Entropy Trajectory')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()

        if filename is None:
            filename = f"cardinal_walk_{sequence[:10]}.png"

        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        return output_path

    def plot_graph_comparison(
        self,
        analyses: List[FragmentGraphAnalysis],
        filename: str = "graph_comparison.png"
    ) -> Path:
        """Plot comparison of multiple fragment graphs."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("Fragment Graph Comparison", fontsize=14, fontweight='bold')

        # Extract metrics
        n_nodes = [a.metrics.n_nodes for a in analyses if a.metrics]
        n_edges = [a.metrics.n_edges for a in analyses if a.metrics]
        densities = [a.metrics.density for a in analyses if a.metrics]
        clusterings = [a.metrics.avg_clustering for a in analyses if a.metrics]
        entropies = [a.metrics.graph_entropy for a in analyses if a.metrics]
        path_lengths = [a.metrics.avg_path_length for a in analyses if a.metrics]

        # 1. Nodes vs Edges scatter
        ax1 = axes[0, 0]
        ax1.scatter(n_nodes, n_edges, alpha=0.6, s=50)
        ax1.set_xlabel('Number of Nodes')
        ax1.set_ylabel('Number of Edges')
        ax1.set_title('Graph Size')

        # Fit line
        if len(n_nodes) > 1:
            z = np.polyfit(n_nodes, n_edges, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(n_nodes), max(n_nodes), 100)
            ax1.plot(x_line, p(x_line), 'r--', alpha=0.5, label=f'Trend')
            ax1.legend()

        # 2. Density distribution
        ax2 = axes[0, 1]
        ax2.hist(densities, bins=20, edgecolor='black', alpha=0.7)
        ax2.axvline(np.mean(densities), color='r', linestyle='--', label=f'Mean: {np.mean(densities):.3f}')
        ax2.set_xlabel('Density')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Graph Density Distribution')
        ax2.legend()

        # 3. Clustering coefficient
        ax3 = axes[0, 2]
        ax3.hist(clusterings, bins=20, edgecolor='black', alpha=0.7, color='green')
        ax3.axvline(np.mean(clusterings), color='r', linestyle='--', label=f'Mean: {np.mean(clusterings):.3f}')
        ax3.set_xlabel('Clustering Coefficient')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Clustering Distribution')
        ax3.legend()

        # 4. Graph entropy
        ax4 = axes[1, 0]
        ax4.hist(entropies, bins=20, edgecolor='black', alpha=0.7, color='purple')
        ax4.axvline(np.mean(entropies), color='r', linestyle='--', label=f'Mean: {np.mean(entropies):.3f}')
        ax4.set_xlabel('Graph Entropy')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Graph Entropy Distribution')
        ax4.legend()

        # 5. Density vs Clustering scatter
        ax5 = axes[1, 1]
        ax5.scatter(densities, clusterings, c=entropies, cmap='viridis', alpha=0.6, s=50)
        ax5.set_xlabel('Density')
        ax5.set_ylabel('Clustering')
        ax5.set_title('Density vs Clustering')
        plt.colorbar(ax5.collections[0], ax=ax5, label='Entropy')

        # 6. Path length distribution
        ax6 = axes[1, 2]
        valid_paths = [p for p in path_lengths if p > 0]
        if valid_paths:
            ax6.hist(valid_paths, bins=20, edgecolor='black', alpha=0.7, color='orange')
            ax6.axvline(np.mean(valid_paths), color='r', linestyle='--', label=f'Mean: {np.mean(valid_paths):.2f}')
        ax6.set_xlabel('Average Path Length')
        ax6.set_ylabel('Frequency')
        ax6.set_title('Path Length Distribution')
        ax6.legend()

        plt.tight_layout()

        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        return output_path

    def plot_amino_acid_network(
        self,
        analyses: List[FragmentGraphAnalysis],
        filename: str = "amino_acid_network.png"
    ) -> Path:
        """Plot network of amino acid transitions across all spectra."""
        # Count amino acid transitions
        aa_transitions = defaultdict(int)

        for analysis in analyses:
            for (u, v), aa in analysis.edge_amino_acids.items():
                # Find next amino acid
                G = analysis.graph
                if G.has_node(v):
                    successors = list(G.successors(v))
                    for succ in successors:
                        if (v, succ) in analysis.edge_amino_acids:
                            next_aa = analysis.edge_amino_acids[(v, succ)]
                            aa_transitions[(aa, next_aa)] += 1

        if not aa_transitions:
            return None

        # Build AA transition graph
        G_aa = nx.DiGraph()

        for (aa1, aa2), count in aa_transitions.items():
            if count > 0:
                G_aa.add_edge(aa1, aa2, weight=count)

        if len(G_aa.nodes()) == 0:
            return None

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Amino Acid Transition Network", fontsize=14, fontweight='bold')

        # 1. Network visualization
        ax1 = axes[0]

        pos = nx.spring_layout(G_aa, seed=42, k=2)

        # Node sizes by degree
        degrees = dict(G_aa.degree())
        node_sizes = [degrees[n] * 100 + 200 for n in G_aa.nodes()]

        # Edge widths by weight
        weights = [G_aa[u][v]['weight'] for u, v in G_aa.edges()]
        max_weight = max(weights) if weights else 1
        edge_widths = [w / max_weight * 3 + 0.5 for w in weights]

        nx.draw_networkx_nodes(G_aa, pos, ax=ax1, node_size=node_sizes,
                              node_color='lightblue', alpha=0.8)
        nx.draw_networkx_labels(G_aa, pos, ax=ax1, font_size=10, font_weight='bold')
        nx.draw_networkx_edges(G_aa, pos, ax=ax1, width=edge_widths,
                              alpha=0.6, edge_color='gray', arrows=True,
                              arrowsize=15, connectionstyle='arc3,rad=0.1')

        ax1.set_title('AA Transition Graph')
        ax1.axis('off')

        # 2. Transition heatmap
        ax2 = axes[1]

        all_aas = sorted(set(G_aa.nodes()))
        n_aa = len(all_aas)

        matrix = np.zeros((n_aa, n_aa))
        aa_to_idx = {aa: i for i, aa in enumerate(all_aas)}

        for (aa1, aa2), count in aa_transitions.items():
            if aa1 in aa_to_idx and aa2 in aa_to_idx:
                matrix[aa_to_idx[aa1], aa_to_idx[aa2]] = count

        im = ax2.imshow(matrix, cmap='YlOrRd')
        ax2.set_xticks(range(n_aa))
        ax2.set_yticks(range(n_aa))
        ax2.set_xticklabels(all_aas)
        ax2.set_yticklabels(all_aas)
        ax2.set_xlabel('To')
        ax2.set_ylabel('From')
        ax2.set_title('Transition Count Matrix')
        plt.colorbar(im, ax=ax2, label='Count')

        plt.tight_layout()

        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        return output_path


# ============================================================================
# MAIN ANALYSIS RUNNER
# ============================================================================

def run_graph_analysis(
    mz_arrays: List[np.ndarray],
    intensity_arrays: List[np.ndarray],
    precursor_mzs: List[float],
    spectrum_ids: List[str] = None,
    output_dir: Path = None,
    max_spectra: int = 100
) -> Dict:
    """
    Run complete graph analysis on multiple spectra.
    """
    print("\n" + "=" * 70)
    print("GRAPH ANALYSIS FOR FRAGMENT NETWORKS")
    print("=" * 70)

    # Set output directory
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent.parent / 'results' / 'graph_analysis'

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nOutput directory: {output_dir}")
    print(f"Analyzing {min(len(mz_arrays), max_spectra)} spectra...")

    start_time = time.time()

    # Initialize
    builder = FragmentGraphBuilder()
    visualizer = GraphVisualizer(output_dir)

    analyses = []

    # Analyze each spectrum
    for i in range(min(len(mz_arrays), max_spectra)):
        spectrum_id = spectrum_ids[i] if spectrum_ids else f"spectrum_{i}"

        analysis = builder.build_fragment_graph(
            mz_arrays[i],
            intensity_arrays[i],
            precursor_mzs[i],
            spectrum_id
        )
        analyses.append(analysis)

        if (i + 1) % 10 == 0:
            print(f"  Analyzed {i + 1} spectra...")

    print(f"\n  Completed {len(analyses)} analyses")

    # Generate visualizations
    print("\n  Generating visualizations...")

    # Plot first few individual graphs
    for i, analysis in enumerate(analyses[:5]):
        if analysis.graph and analysis.graph.number_of_nodes() > 0:
            visualizer.plot_fragment_graph(analysis, f"fragment_graph_{i}.png")

    # Comparison plot
    if len(analyses) > 1:
        visualizer.plot_graph_comparison(analyses)

    # Amino acid network
    visualizer.plot_amino_acid_network(analyses)

    # Cardinal walk plots for inferred sequences
    if EMPTY_DICT_AVAILABLE:
        transformer = EmptyDictionaryTransformer()

        for i, analysis in enumerate(analyses[:5]):
            if analysis.inferred_sequence:
                visualizer.plot_cardinal_walk(
                    analysis.inferred_sequence,
                    transformer,
                    f"cardinal_walk_{i}.png"
                )

    # Save metrics to CSV
    metrics_data = []

    for analysis in analyses:
        if analysis.metrics:
            metrics_data.append({
                'spectrum_id': analysis.spectrum_id,
                'precursor_mz': analysis.precursor_mz,
                'n_fragments': analysis.n_fragments,
                'n_nodes': analysis.metrics.n_nodes,
                'n_edges': analysis.metrics.n_edges,
                'density': analysis.metrics.density,
                'avg_degree': analysis.metrics.avg_degree,
                'max_degree': analysis.metrics.max_degree,
                'avg_clustering': analysis.metrics.avg_clustering,
                'avg_path_length': analysis.metrics.avg_path_length,
                'diameter': analysis.metrics.diameter,
                'n_components': analysis.metrics.n_components,
                'largest_component': analysis.metrics.largest_component_size,
                'avg_betweenness': analysis.metrics.avg_betweenness,
                'avg_closeness': analysis.metrics.avg_closeness,
                'avg_eigenvector': analysis.metrics.avg_eigenvector,
                'graph_entropy': analysis.metrics.graph_entropy,
                'inferred_sequence': analysis.inferred_sequence,
                'sequence_length': len(analysis.inferred_sequence),
            })

    metrics_df = pd.DataFrame(metrics_data)
    metrics_path = output_dir / 'graph_metrics.csv'
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n  Saved metrics: {metrics_path}")

    # Save summary
    total_time = time.time() - start_time

    summary = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'n_spectra': len(analyses),
        'total_time_seconds': total_time,
        'output_dir': str(output_dir),
        'mean_nodes': float(np.mean([a.metrics.n_nodes for a in analyses if a.metrics])),
        'mean_edges': float(np.mean([a.metrics.n_edges for a in analyses if a.metrics])),
        'mean_density': float(np.mean([a.metrics.density for a in analyses if a.metrics])),
        'mean_clustering': float(np.mean([a.metrics.avg_clustering for a in analyses if a.metrics])),
        'mean_entropy': float(np.mean([a.metrics.graph_entropy for a in analyses if a.metrics])),
        'n_with_sequences': sum(1 for a in analyses if a.inferred_sequence),
    }

    summary_path = output_dir / 'analysis_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary: {summary_path}")

    print(f"\n  Total time: {total_time:.2f} seconds")

    return {
        'analyses': analyses,
        'metrics_df': metrics_df,
        'summary': summary,
        'output_dir': output_dir
    }


def run_demo_analysis():
    """Run demonstration analysis with synthetic data."""
    print("\n" + "=" * 70)
    print("GRAPH ANALYSIS DEMONSTRATION")
    print("=" * 70)

    # Generate synthetic spectra
    np.random.seed(42)

    n_spectra = 20
    mz_arrays = []
    intensity_arrays = []
    precursor_mzs = []

    for i in range(n_spectra):
        # Random precursor
        precursor = np.random.uniform(500, 1000)

        # Generate fragments
        n_frags = np.random.randint(20, 80)
        mz = np.sort(np.random.uniform(100, precursor - 100, n_frags))
        intensity = np.random.exponential(1000, n_frags)

        mz_arrays.append(mz)
        intensity_arrays.append(intensity)
        precursor_mzs.append(precursor)

    # Run analysis
    results = run_graph_analysis(
        mz_arrays,
        intensity_arrays,
        precursor_mzs,
        max_spectra=20
    )

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {results['output_dir']}")

    return results


if __name__ == "__main__":
    run_demo_analysis()
