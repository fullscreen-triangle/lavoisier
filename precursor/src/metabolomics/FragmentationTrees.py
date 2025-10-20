#!/usr/bin/env python3
"""
S-Entropy Fragmentation Network for Metabolomics
=================================================

Resolves Gibbs' paradox in fragment assignment through network topology.
Transforms hierarchical tree structure to DAG (directed acyclic graph) where
fragments become distinguishable by their network position, not just m/z.

Theoretical Foundation:
-----------------------
From entropy-coordinates.tex (lines 1051-1389):
- Traditional trees treat fragments as indistinguishable (Gibbs paradox)
- S-Entropy coordinates place fragments in metric space
- Network edges connect similar ions (semantic distance < τ)
- Fragments distinguished by neighborhood structure

Key Innovation:
---------------
Precursor → Fragments is NOT one-to-many (hierarchical)
It's many-to-many (network): F_i ← P_j, P_k, P_ℓ (shared fragments)

Fragment assignment via network navigation:
- Find neighborhood N_τ(F_obs)
- Compute cluster coherence C(F_i, P_k)
- Assign to precursor with highest coherence

Author: Kundai Sachikonye
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import networkx as nx
from scipy.spatial.distance import euclidean
from scipy.spatial import KDTree

# Import S-Entropy and Phase-Lock frameworks
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from core.EntropyTransformation import SEntropyTransformer, SEntropyCoordinates, SEntropyFeatures
from core.PhaseLockNetworks import (
    PhaseLockMeasurementDevice,
    EnhancedPhaseLockMeasurementDevice,
    PhaseLockSignature
)


@dataclass
class FragmentIon:
    """
    Individual fragment ion with S-Entropy coordinates and phase-lock signature.

    Attributes:
        mz: Mass-to-charge ratio
        intensity: Intensity
        s_entropy_coords: S-Entropy 3D coordinates (S_knowledge, S_time, S_entropy)
        s_entropy_features: 14D feature vector
        phase_lock_signature: Phase-lock signature from measurement device
        categorical_state: Assigned categorical state
        precursor_mz: Parent precursor m/z (if known)
        annotation: Chemical formula or structure (if known)
    """
    mz: float
    intensity: float
    s_entropy_coords: Optional[SEntropyCoordinates] = None
    s_entropy_features: Optional[SEntropyFeatures] = None
    phase_lock_signature: Optional[PhaseLockSignature] = None
    categorical_state: Optional[int] = None
    precursor_mz: Optional[float] = None
    annotation: Optional[str] = None
    # Network properties
    network_neighbors: Set[str] = field(default_factory=set)
    cluster_id: Optional[int] = None


@dataclass
class PrecursorIon:
    """
    Precursor ion with associated fragments.

    Attributes:
        mz: Precursor m/z
        intensity: Precursor intensity
        rt: Retention time
        charge: Charge state
        fragments: List of fragment ions
        s_entropy_coords: S-Entropy coordinates of precursor intact spectrum
        annotation: Metabolite annotation (if known)
    """
    mz: float
    intensity: float
    rt: float
    charge: int = 1
    fragments: List[FragmentIon] = field(default_factory=list)
    s_entropy_coords: Optional[SEntropyCoordinates] = None
    s_entropy_features: Optional[SEntropyFeatures] = None
    annotation: Optional[str] = None


@dataclass
class NetworkEdge:
    """
    Edge in S-Entropy fragmentation network.

    Attributes:
        source_id: Source node ID
        target_id: Target node ID
        semantic_distance: Semantic distance d_sem(source, target)
        edge_weight: w = exp(-d_sem / σ)
        edge_type: 'precursor_fragment', 'fragment_fragment', 'precursor_precursor'
    """
    source_id: str
    target_id: str
    semantic_distance: float
    edge_weight: float
    edge_type: str


class SEntropyFragmentationNetwork:
    """
    S-Entropy Fragmentation Network (DAG, not tree!)

    Implements network-based fragment assignment (Algorithm from entropy-coordinates.tex
    lines 1141-1160).

    Key Properties:
    ---------------
    1. Vertices: V = P ∪ F (precursors and fragments)
    2. Edges: (u,v) ∈ E if d_sem(f(u), f(v)) < τ
    3. Non-hierarchical: fragments can have multiple precursor parents
    4. Distinguishability: fragments distinguished by network position
    """

    def __init__(
        self,
        similarity_threshold: float = 0.5,
        sigma: float = 0.2,
        feature_weights: Optional[np.ndarray] = None
    ):
        """
        Initialize S-Entropy fragmentation network.

        Args:
            similarity_threshold: τ for edge creation (d_sem < τ)
            sigma: Scale parameter for edge weights w = exp(-d / σ)
            feature_weights: Weights for semantic distance (14D)
        """
        self.similarity_threshold = similarity_threshold
        self.sigma = sigma

        # Feature weights (from feature importance analysis)
        # From entropy-coordinates.tex Table: Base peak m/z (23.4%), TIC (19.8%), etc.
        if feature_weights is None:
            self.feature_weights = np.array([
                0.234,  # f1: base peak m/z
                0.143,  # f2: peak count
                0.089,  # f3: m/z range
                0.032,  # f4: peak spacing variance
                0.198,  # f5: total ion current
                0.128,  # f6: intensity variance
                0.045,  # f7: intensity skewness
                0.034,  # f8: intensity kurtosis
                0.176,  # f9: spectral entropy
                0.032,  # f10: structural entropy
                0.023,  # f11: mutual information
                0.018,  # f12: conditional entropy
                0.012,  # f13: temporal coordinate
                0.010   # f14: phase coherence
            ])
        else:
            self.feature_weights = feature_weights

        # Network structure
        self.graph = nx.DiGraph()
        self.precursors: Dict[str, PrecursorIon] = {}
        self.fragments: Dict[str, FragmentIon] = {}
        self.edges: List[NetworkEdge] = []

        # S-Entropy transformer
        self.s_entropy_transformer = SEntropyTransformer()

        # Phase-lock measurement device
        self.phase_lock_device = EnhancedPhaseLockMeasurementDevice(
            enable_performance_tracking=True
        )

        # KD-Tree for efficient nearest neighbor search in 14D S-Entropy space
        self.kdtree: Optional[KDTree] = None
        self.kdtree_indices: List[str] = []

    def add_precursor(
        self,
        precursor: PrecursorIon,
        compute_s_entropy: bool = True
    ) -> str:
        """
        Add precursor ion to network.

        Args:
            precursor: PrecursorIon object
            compute_s_entropy: Whether to compute S-Entropy coordinates

        Returns:
            Precursor node ID
        """
        precursor_id = f"P_{precursor.mz:.4f}_{precursor.rt:.2f}"

        # Compute S-Entropy if needed and intact spectrum available
        if compute_s_entropy and len(precursor.fragments) > 0:
            # Use fragment spectrum as proxy for precursor
            mz_array = np.array([f.mz for f in precursor.fragments])
            intensity_array = np.array([f.intensity for f in precursor.fragments])

            precursor.s_entropy_features = self.s_entropy_transformer.calculate_spectrum_features(
                mz_array=mz_array,
                intensity_array=intensity_array,
                precursor_mz=precursor.mz,
                rt=precursor.rt
            )

        self.precursors[precursor_id] = precursor
        self.graph.add_node(precursor_id, node_type='precursor', data=precursor)

        return precursor_id

    def add_fragment(
        self,
        fragment: FragmentIon,
        precursor_mz: float,
        compute_s_entropy: bool = True,
        local_context: Optional[List[FragmentIon]] = None
    ) -> str:
        """
        Add fragment ion to network.

        Args:
            fragment: FragmentIon object
            precursor_mz: Precursor m/z for S-Entropy calculation
            compute_s_entropy: Whether to compute S-Entropy coordinates
            local_context: Other fragments for local entropy calculation

        Returns:
            Fragment node ID
        """
        fragment_id = f"F_{fragment.mz:.4f}"

        # Compute S-Entropy coordinates for single ion
        if compute_s_entropy:
            if local_context:
                local_intensities = np.array([f.intensity for f in local_context])
                mz_array = np.array([f.mz for f in local_context])
                intensity_array = np.array([f.intensity for f in local_context])
            else:
                local_intensities = None
                mz_array = None
                intensity_array = None

            fragment.s_entropy_coords = self.s_entropy_transformer.calculate_s_entropy(
                mz=fragment.mz,
                intensity=fragment.intensity,
                precursor_mz=precursor_mz,
                rt=None,
                local_intensities=local_intensities,
                mz_array=mz_array,
                intensity_array=intensity_array
            )

        self.fragments[fragment_id] = fragment
        self.graph.add_node(fragment_id, node_type='fragment', data=fragment)

        return fragment_id

    def compute_semantic_distance(
        self,
        features1: SEntropyFeatures,
        features2: SEntropyFeatures
    ) -> float:
        """
        Compute semantic distance with feature weighting.

        d_sem(f_i, f_j) = Σ_k w_k |f_ik - f_jk|

        Args:
            features1: First feature vector (14D)
            features2: Second feature vector (14D)

        Returns:
            Semantic distance
        """
        diff = np.abs(features1.features - features2.features)
        weighted_diff = self.feature_weights * diff
        return np.sum(weighted_diff)

    def build_network(self):
        """
        Build S-Entropy fragmentation network by computing edges.

        Creates edges for:
        1. Precursor → Fragment (if d_sem < τ)
        2. Fragment → Fragment (secondary fragmentation)
        3. Precursor ↔ Precursor (structural similarity)
        """
        print("[Network Building] Computing S-Entropy features for all nodes...")

        # Collect all nodes with S-Entropy features
        all_nodes: List[Tuple[str, SEntropyFeatures, str]] = []

        for precursor_id, precursor in self.precursors.items():
            if precursor.s_entropy_features:
                all_nodes.append((precursor_id, precursor.s_entropy_features, 'precursor'))

        # For fragments, we need to compute features from their context
        # Group fragments by precursor
        fragments_by_precursor = defaultdict(list)
        for fragment_id, fragment in self.fragments.items():
            if fragment.precursor_mz:
                fragments_by_precursor[fragment.precursor_mz].append((fragment_id, fragment))

        # Compute features for each fragment group
        for precursor_mz, fragment_list in fragments_by_precursor.items():
            if len(fragment_list) > 0:
                mz_array = np.array([f[1].mz for f in fragment_list])
                intensity_array = np.array([f[1].intensity for f in fragment_list])

                features = self.s_entropy_transformer.calculate_spectrum_features(
                    mz_array=mz_array,
                    intensity_array=intensity_array,
                    precursor_mz=precursor_mz,
                    rt=None
                )

                # Assign to each fragment (they share context features)
                for fragment_id, fragment in fragment_list:
                    fragment.s_entropy_features = features
                    all_nodes.append((fragment_id, features, 'fragment'))

        print(f"[Network Building] Total nodes: {len(all_nodes)}")

        # Build KD-Tree for efficient nearest neighbor search
        if len(all_nodes) > 0:
            feature_matrix = np.array([node[1].features for node in all_nodes])
            self.kdtree = KDTree(feature_matrix)
            self.kdtree_indices = [node[0] for node in all_nodes]
            print(f"[Network Building] Built KD-Tree with {len(all_nodes)} nodes")

        # Compute edges
        print("[Network Building] Computing edges...")
        edge_count = 0

        for i, (node_id1, features1, type1) in enumerate(all_nodes):
            # Query neighbors within threshold using KD-Tree
            # Convert semantic distance threshold to Euclidean (approximation)
            euclidean_threshold = self.similarity_threshold * np.sqrt(len(self.feature_weights))

            neighbor_indices = self.kdtree.query_ball_point(features1.features, euclidean_threshold)

            for j in neighbor_indices:
                if i == j:
                    continue

                node_id2 = self.kdtree_indices[j]
                features2 = all_nodes[j][1]
                type2 = all_nodes[j][2]

                # Compute semantic distance
                d_sem = self.compute_semantic_distance(features1, features2)

                if d_sem < self.similarity_threshold:
                    # Compute edge weight
                    edge_weight = np.exp(-d_sem / self.sigma)

                    # Determine edge type
                    if type1 == 'precursor' and type2 == 'fragment':
                        edge_type = 'precursor_fragment'
                    elif type1 == 'fragment' and type2 == 'fragment':
                        edge_type = 'fragment_fragment'
                    elif type1 == 'precursor' and type2 == 'precursor':
                        edge_type = 'precursor_precursor'
                    else:
                        edge_type = 'fragment_precursor'

                    # Add edge
                    edge = NetworkEdge(
                        source_id=node_id1,
                        target_id=node_id2,
                        semantic_distance=d_sem,
                        edge_weight=edge_weight,
                        edge_type=edge_type
                    )
                    self.edges.append(edge)
                    self.graph.add_edge(node_id1, node_id2,
                                       weight=edge_weight,
                                       distance=d_sem,
                                       edge_type=edge_type)
                    edge_count += 1

        print(f"[Network Building] Created {edge_count} edges")
        print(f"[Network Building] Network density: {nx.density(self.graph):.4f}")

    def assign_fragment_to_precursor(
        self,
        fragment_mz: float,
        fragment_intensity: float,
        candidate_precursors: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Network-based fragment assignment (Algorithm from entropy-coordinates.tex).

        Args:
            fragment_mz: Fragment m/z
            fragment_intensity: Fragment intensity
            candidate_precursors: List of candidate precursor IDs (None = all)

        Returns:
            Dictionary mapping precursor_id → probability P(P_k | F_obs)
        """
        # Create temporary fragment
        temp_fragment = FragmentIon(mz=fragment_mz, intensity=fragment_intensity)
        fragment_id = f"F_temp_{fragment_mz:.4f}"

        # Compute S-Entropy features (use first precursor as context)
        if candidate_precursors is None:
            candidate_precursors = list(self.precursors.keys())

        if len(candidate_precursors) == 0:
            return {}

        # Use first candidate's fragments as context
        first_precursor = self.precursors[candidate_precursors[0]]
        if len(first_precursor.fragments) > 0:
            local_context = first_precursor.fragments
            mz_array = np.array([f.mz for f in local_context] + [fragment_mz])
            intensity_array = np.array([f.intensity for f in local_context] + [fragment_intensity])

            temp_fragment.s_entropy_features = self.s_entropy_transformer.calculate_spectrum_features(
                mz_array=mz_array,
                intensity_array=intensity_array,
                precursor_mz=first_precursor.mz,
                rt=first_precursor.rt
            )
        else:
            return {}

        # Find neighborhood N_τ(F_obs)
        query_features = temp_fragment.s_entropy_features.features
        neighbor_indices = self.kdtree.query_ball_point(
            query_features,
            self.similarity_threshold * np.sqrt(len(self.feature_weights))
        )

        # Extract precursor candidates from neighborhood
        precursor_scores = {}

        for idx in neighbor_indices:
            neighbor_id = self.kdtree_indices[idx]

            if neighbor_id.startswith('P_'):  # Precursor node
                # Compute edge weight
                neighbor_data = self.graph.nodes[neighbor_id]['data']
                if neighbor_data.s_entropy_features:
                    d_sem = self.compute_semantic_distance(
                        temp_fragment.s_entropy_features,
                        neighbor_data.s_entropy_features
                    )
                    w_k = np.exp(-d_sem / self.sigma)

                    # Compute cluster coherence C(F_i, P_k)
                    cluster_coherence = self._compute_cluster_coherence(
                        fragment_id, neighbor_id, neighbor_indices
                    )

                    # Combined score
                    precursor_scores[neighbor_id] = w_k * cluster_coherence

        # Normalize to probabilities
        total_score = sum(precursor_scores.values())
        if total_score > 0:
            probabilities = {pid: score / total_score
                           for pid, score in precursor_scores.items()}
        else:
            probabilities = {}

        return probabilities

    def _compute_cluster_coherence(
        self,
        fragment_id: str,
        precursor_id: str,
        neighborhood_indices: List[int]
    ) -> float:
        """
        Compute cluster coherence C(F_i, P_k).

        C(F_i, P_k) = (1/|N_τ(F_i)|) Σ_{v ∈ N_τ(F_i)} 𝟙_{v from P_k} · w(F_i, v)

        Args:
            fragment_id: Fragment node ID
            precursor_id: Precursor node ID
            neighborhood_indices: Indices of neighbors in KD-tree

        Returns:
            Cluster coherence score
        """
        if len(neighborhood_indices) == 0:
            return 0.0

        precursor = self.precursors.get(precursor_id)
        if not precursor:
            return 0.0

        # Get fragments from this precursor
        precursor_fragment_ids = set(f"F_{f.mz:.4f}" for f in precursor.fragments)

        coherence_sum = 0.0
        for idx in neighborhood_indices:
            neighbor_id = self.kdtree_indices[idx]

            # Check if neighbor is from this precursor
            if neighbor_id in precursor_fragment_ids:
                # Compute edge weight (simplified: use 1.0 if in same cluster)
                coherence_sum += 1.0

        return coherence_sum / len(neighborhood_indices)

    def get_fragment_annotations(self, fragment_id: str) -> List[Tuple[str, float]]:
        """
        Get possible annotations for a fragment with confidence scores.

        Args:
            fragment_id: Fragment node ID

        Returns:
            List of (annotation, confidence) tuples
        """
        if fragment_id not in self.graph:
            return []

        # Find connected precursors
        annotations = []

        for precursor_id in self.graph.predecessors(fragment_id):
            if precursor_id.startswith('P_'):
                precursor = self.precursors[precursor_id]
                edge_data = self.graph.edges[precursor_id, fragment_id]

                if precursor.annotation:
                    confidence = edge_data['weight']
                    annotations.append((precursor.annotation, confidence))

        # Sort by confidence
        annotations.sort(key=lambda x: x[1], reverse=True)

        return annotations

    def export_network(self, filepath: str):
        """Export network to GraphML format for visualization."""
        nx.write_graphml(self.graph, filepath)
        print(f"[Export] Network exported to {filepath}")

    def get_network_statistics(self) -> Dict[str, any]:
        """Get network statistics."""
        stats = {
            'num_precursors': len(self.precursors),
            'num_fragments': len(self.fragments),
            'num_edges': len(self.edges),
            'network_density': nx.density(self.graph),
            'avg_degree': np.mean([d for n, d in self.graph.degree()]),
            'num_connected_components': nx.number_weakly_connected_components(self.graph),
            'largest_component_size': len(max(nx.weakly_connected_components(self.graph), key=len)) if len(self.graph) > 0 else 0
        }

        # Edge type distribution
        edge_types = defaultdict(int)
        for edge in self.edges:
            edge_types[edge.edge_type] += 1
        stats['edge_types'] = dict(edge_types)

        return stats
