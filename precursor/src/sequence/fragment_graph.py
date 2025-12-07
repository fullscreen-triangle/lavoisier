#!/usr/bin/env python3
"""
Fragment Graph Construction
===========================

From st-stellas-sequence.tex Algorithm 2 Step 1:
Build directed graph where nodes are fragments and edges represent
sequential relationships.

Author: Kundai Sachikonye
"""

import numpy as np
import networkx as nx
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from core.EntropyTransformation import SEntropyCoordinates


@dataclass
class FragmentNode:
    """
    Fragment node in the reconstruction graph.

    Attributes:
        fragment_id: Unique identifier
        sequence: Partial sequence (if identified)
        s_entropy_coords: S-Entropy coordinates
        mass: Fragment mass
        ion_type: 'b' or 'y' (if known)
        position: Position in sequence (if known)
        confidence: Identification confidence
    """
    fragment_id: str
    sequence: Optional[str]
    s_entropy_coords: SEntropyCoordinates
    mass: float
    ion_type: Optional[str] = None
    position: Optional[int] = None
    confidence: float = 1.0

    def entropy(self) -> float:
        """Calculate S-Entropy magnitude for this fragment."""
        coords = self.s_entropy_coords.to_array()
        return float(np.linalg.norm(coords))


class FragmentGraph:
    """
    Directed graph of peptide fragments for sequence reconstruction.

    From st-stellas-sequence.tex:
    Nodes = fragments
    Edges = sequential relationships (i → j if j extends i by one residue)
    """

    def __init__(self):
        """Initialize empty fragment graph."""
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, FragmentNode] = {}
        self.precursor_mass: Optional[float] = None

    def add_fragment(self, node: FragmentNode):
        """Add fragment node to graph."""
        self.nodes[node.fragment_id] = node
        self.graph.add_node(node.fragment_id, fragment=node)

    def add_edge(
        self,
        from_id: str,
        to_id: str,
        mass_diff: float,
        sentropy_similarity: float
    ):
        """
        Add directed edge between fragments.

        Args:
            from_id: Source fragment ID
            to_id: Target fragment ID
            mass_diff: Mass difference (should match amino acid mass)
            sentropy_similarity: S-Entropy similarity [0, 1]
        """
        self.graph.add_edge(
            from_id,
            to_id,
            mass_diff=mass_diff,
            sentropy_similarity=sentropy_similarity,
            weight=sentropy_similarity  # For path finding
        )

    def find_sequential_pairs(self, mass_tolerance: float = 0.5):
        """
        Find fragment pairs that differ by one amino acid mass.

        From st-stellas-sequence.tex:
        Sequential b-ions: b_i → b_{i+1} differ by m(AA)
        Sequential y-ions: y_i → y_{i+1} differ by m(AA)

        Args:
            mass_tolerance: Mass tolerance for AA matching (Da)
        """
        from molecular_language.amino_acid_alphabet import STANDARD_AMINO_ACIDS

        # Get all amino acid masses
        aa_masses = [aa.mass for aa in STANDARD_AMINO_ACIDS.values()]

        # Compare all fragment pairs
        fragment_ids = list(self.nodes.keys())

        for i in range(len(fragment_ids)):
            for j in range(len(fragment_ids)):
                if i == j:
                    continue

                frag_i = self.nodes[fragment_ids[i]]
                frag_j = self.nodes[fragment_ids[j]]

                # Check if same ion type
                if frag_i.ion_type and frag_j.ion_type:
                    if frag_i.ion_type != frag_j.ion_type:
                        continue

                # Calculate mass difference
                mass_diff = abs(frag_j.mass - frag_i.mass)

                # Check if mass diff matches any amino acid
                for aa_mass in aa_masses:
                    if abs(mass_diff - aa_mass) <= mass_tolerance:
                        # Sequential relationship found!

                        # Calculate S-Entropy similarity
                        coords_i = frag_i.s_entropy_coords.to_array()
                        coords_j = frag_j.s_entropy_coords.to_array()
                        distance = np.linalg.norm(coords_i - coords_j)
                        similarity = np.exp(-distance / 0.3)  # Exponential similarity

                        # Add edge (larger fragment extends smaller)
                        if frag_j.mass > frag_i.mass:
                            self.add_edge(
                                fragment_ids[i],
                                fragment_ids[j],
                                mass_diff,
                                similarity
                            )
                        break

    def find_hamiltonian_path(self) -> Optional[List[str]]:
        """
        Find Hamiltonian path through fragment graph.

        From st-stellas-sequence.tex Algorithm 2 Step 3:
        "Find a Hamiltonian path that visits all fragments exactly once"

        This is NP-hard, so we use approximations.

        Returns:
            Ordered list of fragment IDs (or None if no path exists)
        """
        # Try to find longest path using topological sort (for DAGs)
        if nx.is_directed_acyclic_graph(self.graph):
            try:
                # Get topological ordering
                topo_order = list(nx.topological_sort(self.graph))

                # Find longest path through this ordering
                longest_path = self._find_longest_path_dag()

                if longest_path and len(longest_path) >= len(self.nodes) * 0.7:
                    # Accept if covers at least 70% of fragments
                    return longest_path
            except:
                pass

        # Fallback: greedy path construction
        return self._greedy_path_construction()

    def _find_longest_path_dag(self) -> Optional[List[str]]:
        """Find longest path in DAG using dynamic programming."""
        if len(self.nodes) == 0:
            return None

        # Initialize distance dict
        dist = {node: float('-inf') for node in self.graph.nodes()}
        parent = {node: None for node in self.graph.nodes()}

        # Topological sort
        topo_order = list(nx.topological_sort(self.graph))

        # Find node with no predecessors
        for node in topo_order:
            if self.graph.in_degree(node) == 0:
                dist[node] = 0

        # DP: longest path
        for node in topo_order:
            for successor in self.graph.successors(node):
                edge_weight = self.graph[node][successor].get('weight', 1.0)
                if dist[node] + edge_weight > dist[successor]:
                    dist[successor] = dist[node] + edge_weight
                    parent[successor] = node

        # Find ending node with max distance
        end_node = max(dist.keys(), key=lambda k: dist[k])

        # Reconstruct path
        path = []
        current = end_node
        while current is not None:
            path.append(current)
            current = parent[current]

        path.reverse()

        return path if len(path) > 1 else None

    def _greedy_path_construction(self) -> Optional[List[str]]:
        """Greedy construction of fragment path."""
        if len(self.nodes) == 0:
            return None

        # Start with fragment with lowest mass (likely N-terminal)
        start_id = min(self.nodes.keys(), key=lambda k: self.nodes[k].mass)

        path = [start_id]
        visited = {start_id}
        current = start_id

        while len(visited) < len(self.nodes):
            # Find best successor (highest similarity edge)
            best_next = None
            best_weight = -1

            for successor in self.graph.successors(current):
                if successor not in visited:
                    weight = self.graph[current][successor].get('weight', 0.0)
                    if weight > best_weight:
                        best_weight = weight
                        best_next = successor

            if best_next is None:
                # Dead end - can't complete path
                break

            path.append(best_next)
            visited.add(best_next)
            current = best_next

        return path if len(path) > 1 else None

    def calculate_path_entropy(self, path: List[str]) -> float:
        """
        Calculate total S-Entropy for a path.

        From st-stellas-sequence.tex:
        Minimize total S-Entropy = Σ S(fragment_i)

        Args:
            path: List of fragment IDs

        Returns:
            Total path entropy
        """
        total_entropy = 0.0

        for frag_id in path:
            if frag_id in self.nodes:
                total_entropy += self.nodes[frag_id].entropy()

        return total_entropy


def build_fragment_graph_from_spectra(
    fragments: List[FragmentNode],
    precursor_mass: Optional[float] = None
) -> FragmentGraph:
    """
    Build fragment graph from MS/MS spectrum.

    Args:
        fragments: List of identified fragments
        precursor_mass: Precursor mass (optional)

    Returns:
        FragmentGraph ready for sequence reconstruction
    """
    graph = FragmentGraph()
    graph.precursor_mass = precursor_mass

    # Add all fragments
    for frag in fragments:
        graph.add_fragment(frag)

    # Find sequential relationships
    graph.find_sequential_pairs()

    print(f"[FragmentGraph] Built graph with {len(fragments)} nodes, "
          f"{graph.graph.number_of_edges()} edges")

    return graph
