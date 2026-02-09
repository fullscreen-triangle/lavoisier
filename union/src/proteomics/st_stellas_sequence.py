"""
St. Stella's Sequence Framework for Proteomics
===============================================

Implements cardinal direction coordinate transformation for peptide sequences
and fragment analysis, following the S-Entropy molecular language framework.

Key Concepts:
- Amino acid cardinal direction mapping (20 amino acids -> 20-dimensional rotation)
- Peptide coordinate path construction
- Fragment graph with categorical completion
- Phase-lock network analysis for b/y ion series
- State counting sequence reconstruction (integrated)
- Fragment-parent hierarchical validation

Based on:
- docs/publication/st-stellas-sequence.tex
- docs/publication/st-stellas-molecular-language.tex
- union/publication/state-counting/state-counting-mass-spectrometry.tex

Author: Kundai Sachikonye
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy as scipy_entropy
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
import networkx as nx
from collections import defaultdict
import re

# Import state counting module
try:
    from .state_counting import (
        StateCountingReconstructor,
        StateCountingDropletMapper,
        validate_fragment_parent_hierarchy,
        validate_fragment_hierarchy,
        PartitionState,
        capacity,
        total_capacity,
        mz_to_partition_depth,
        AMINO_ACID_STATES,
        FragmentParentValidation
    )
    STATE_COUNTING_AVAILABLE = True
except ImportError:
    STATE_COUNTING_AVAILABLE = False


# ============================================================================
# AMINO ACID CARDINAL DIRECTION MAPPING
# ============================================================================

# Standard amino acid masses (monoisotopic)
AMINO_ACID_MASSES = {
    'A': 71.03711,   # Alanine
    'R': 156.10111,  # Arginine
    'N': 114.04293,  # Asparagine
    'D': 115.02694,  # Aspartate
    'C': 103.00919,  # Cysteine
    'E': 129.04259,  # Glutamate
    'Q': 128.05858,  # Glutamine
    'G': 57.02146,   # Glycine
    'H': 137.05891,  # Histidine
    'I': 113.08406,  # Isoleucine
    'L': 113.08406,  # Leucine
    'K': 128.09496,  # Lysine
    'M': 131.04049,  # Methionine
    'F': 147.06841,  # Phenylalanine
    'P': 97.05276,   # Proline
    'S': 87.03203,   # Serine
    'T': 101.04768,  # Threonine
    'W': 186.07931,  # Tryptophan
    'Y': 163.06333,  # Tyrosine
    'V': 99.06841,   # Valine
}

# Cardinal direction mapping for amino acids
# Using physicochemical properties to assign 3D coordinates
# Based on: hydrophobicity (x), polarity (y), size (z)
AMINO_ACID_CARDINAL_COORDS = {
    # Hydrophobic, nonpolar (High x, low y)
    'A': (0.62, 0.0, 0.09),   # Alanine - small, hydrophobic
    'V': (1.08, 0.0, 0.16),   # Valine - branched, hydrophobic
    'I': (1.38, 0.0, 0.19),   # Isoleucine - branched, hydrophobic
    'L': (1.06, 0.0, 0.19),   # Leucine - branched, hydrophobic
    'M': (0.64, 0.0, 0.21),   # Methionine - sulfur-containing
    'F': (1.19, 0.0, 0.23),   # Phenylalanine - aromatic
    'W': (0.81, 0.0, 0.28),   # Tryptophan - large aromatic
    'P': (0.12, 0.0, 0.14),   # Proline - cyclic

    # Polar, uncharged (Medium x, medium y)
    'G': (0.48, 0.0, 0.00),   # Glycine - smallest
    'S': (-0.18, 0.5, 0.12),  # Serine - hydroxyl
    'T': (-0.05, 0.5, 0.14),  # Threonine - hydroxyl
    'C': (0.29, 0.5, 0.15),   # Cysteine - thiol
    'Y': (0.26, 0.5, 0.25),   # Tyrosine - aromatic hydroxyl
    'N': (-0.78, 0.5, 0.17),  # Asparagine - amide
    'Q': (-0.85, 0.5, 0.21),  # Glutamine - amide

    # Charged, positive (Low x, high y positive)
    'K': (-1.50, 1.0, 0.21),  # Lysine - amino
    'R': (-2.53, 1.0, 0.26),  # Arginine - guanidinium
    'H': (-0.40, 1.0, 0.21),  # Histidine - imidazole

    # Charged, negative (Low x, high y negative)
    'D': (-0.90, -1.0, 0.16), # Aspartate - carboxyl
    'E': (-0.74, -1.0, 0.20), # Glutamate - carboxyl
}

# Amino acid groups for pattern analysis
AA_GROUPS = {
    'hydrophobic': {'A', 'V', 'I', 'L', 'M', 'F', 'W', 'P'},
    'polar': {'S', 'T', 'C', 'Y', 'N', 'Q'},
    'positive': {'K', 'R', 'H'},
    'negative': {'D', 'E'},
    'aromatic': {'F', 'W', 'Y', 'H'},
    'small': {'G', 'A', 'S'},
    'proline': {'P'},  # Special - affects backbone
}


@dataclass
class StellaCoordinate:
    """
    St. Stella's Sequence coordinate for a single amino acid.

    Represents position in 3D S-Entropy space:
    - s_knowledge: Information content (from sequence context)
    - s_time: Temporal/sequential position
    - s_entropy: Local disorder measure
    """
    amino_acid: str
    position: int

    # Base coordinates (from cardinal mapping)
    base_x: float  # Hydrophobicity
    base_y: float  # Polarity
    base_z: float  # Size

    # S-Entropy weighted coordinates
    s_knowledge: float
    s_time: float
    s_entropy: float

    def to_base_array(self) -> np.ndarray:
        """Return base coordinates."""
        return np.array([self.base_x, self.base_y, self.base_z])

    def to_sentropy_array(self) -> np.ndarray:
        """Return S-Entropy coordinates."""
        return np.array([self.s_knowledge, self.s_time, self.s_entropy])

    def magnitude(self) -> float:
        """Euclidean magnitude of S-Entropy coordinates."""
        return np.linalg.norm(self.to_sentropy_array())


@dataclass
class PeptideCoordinatePath:
    """
    Complete coordinate path for a peptide sequence.

    Represents the peptide as a trajectory through S-Entropy space,
    enabling geometric analysis of sequence properties.
    """
    sequence: str
    coordinates: List[StellaCoordinate]

    # Path properties
    cumulative_path: np.ndarray = field(default=None)  # Cumulative sum of coords
    path_length: float = 0.0
    endpoint_distance: float = 0.0
    tortuosity: float = 0.0

    # S-Entropy path features
    mean_s_knowledge: float = 0.0
    mean_s_time: float = 0.0
    mean_s_entropy: float = 0.0

    # Geometric properties
    curvature_profile: np.ndarray = field(default=None)
    velocity_profile: np.ndarray = field(default=None)


@dataclass
class FragmentNode:
    """
    Node in the fragment graph representing a detected ion.
    """
    mz: float
    intensity: float
    ion_type: Optional[str] = None  # 'b', 'y', 'a', etc.
    ion_number: Optional[int] = None
    charge: int = 1

    # S-Entropy coordinates
    sentropy_coords: np.ndarray = field(default=None)

    # Inferred amino acid (if known)
    amino_acid: Optional[str] = None

    def __hash__(self):
        return hash((round(self.mz, 4), round(self.intensity, 2)))


@dataclass
class FragmentEdge:
    """
    Edge in the fragment graph representing amino acid transition.
    """
    source_mz: float
    target_mz: float
    mass_difference: float
    inferred_aa: Optional[str] = None
    confidence: float = 0.0

    # S-Entropy interpolation distance
    sentropy_distance: float = 0.0


class StStellasSequenceTransformer:
    """
    St. Stella's Sequence Transformer for Peptides.

    Implements cardinal direction coordinate transformation following
    the molecular language framework for proteomics.

    Key transformations:
    1. Amino acid -> Base coordinates (hydrophobicity, polarity, size)
    2. Base coordinates -> S-Entropy weighted coordinates
    3. Peptide sequence -> Coordinate path
    4. Fragment spectrum -> Fragment graph
    """

    def __init__(
        self,
        window_size: int = 5,
        tau_decay: float = 5.0,
        mass_tolerance: float = 0.5
    ):
        """
        Initialize St. Stella's Sequence transformer.

        Args:
            window_size: Context window for S-Entropy weighting
            tau_decay: Characteristic decay length for time weighting
            mass_tolerance: Mass tolerance for amino acid matching (Da)
        """
        self.window_size = window_size
        self.tau_decay = tau_decay
        self.mass_tolerance = mass_tolerance

        # Build amino acid mass lookup (for fragment matching)
        self._build_mass_lookup()

    def _build_mass_lookup(self):
        """Build reverse lookup from mass to amino acid."""
        self.mass_to_aa = {}
        for aa, mass in AMINO_ACID_MASSES.items():
            mass_key = round(mass, 1)
            if mass_key not in self.mass_to_aa:
                self.mass_to_aa[mass_key] = []
            self.mass_to_aa[mass_key].append(aa)

    # ========================================================================
    # PEPTIDE SEQUENCE TRANSFORMATION
    # ========================================================================

    def transform_peptide(self, sequence: str) -> PeptideCoordinatePath:
        """
        Transform peptide sequence to St. Stella's coordinate path.

        Args:
            sequence: Amino acid sequence (e.g., "PEPTIDE")

        Returns:
            PeptideCoordinatePath with complete coordinate representation
        """
        sequence = sequence.upper().strip()
        n = len(sequence)

        if n == 0:
            return PeptideCoordinatePath(
                sequence="",
                coordinates=[],
                cumulative_path=np.array([])
            )

        coordinates = []
        cumulative = np.zeros(3)
        cumulative_path = []

        for i, aa in enumerate(sequence):
            # Get base coordinates
            if aa in AMINO_ACID_CARDINAL_COORDS:
                base_x, base_y, base_z = AMINO_ACID_CARDINAL_COORDS[aa]
            else:
                base_x, base_y, base_z = (0.0, 0.0, 0.1)  # Unknown AA

            # Compute S-Entropy weights
            w_k = self._compute_knowledge_weight(sequence, i)
            w_t = self._compute_time_weight(sequence, i)
            w_e = self._compute_entropy_weight(sequence, i)

            # Create coordinate
            coord = StellaCoordinate(
                amino_acid=aa,
                position=i,
                base_x=base_x,
                base_y=base_y,
                base_z=base_z,
                s_knowledge=w_k * base_x,
                s_time=w_t * base_y,
                s_entropy=w_e * base_z
            )
            coordinates.append(coord)

            # Accumulate path
            cumulative += coord.to_sentropy_array()
            cumulative_path.append(cumulative.copy())

        cumulative_path = np.array(cumulative_path)

        # Compute path properties
        path = PeptideCoordinatePath(
            sequence=sequence,
            coordinates=coordinates,
            cumulative_path=cumulative_path
        )

        self._compute_path_properties(path)

        return path

    def _compute_knowledge_weight(self, sequence: str, position: int) -> float:
        """
        Knowledge weighting function: Shannon entropy of local context.

        w_k = -Σ p_j log₂(p_j) for j in window
        """
        n = len(sequence)
        window_start = max(0, position - self.window_size // 2)
        window_end = min(n, position + self.window_size // 2 + 1)
        window = sequence[window_start:window_end]

        if len(window) == 0:
            return 1.0

        # Count amino acid frequencies
        aa_counts = defaultdict(int)
        for aa in window:
            aa_counts[aa] += 1

        # Compute Shannon entropy
        total = len(window)
        probs = [count / total for count in aa_counts.values()]
        shannon = scipy_entropy(probs, base=2)

        # Normalize to [0.1, 2.0]
        # Max entropy for 20 AA is log2(20) ≈ 4.32
        normalized = 0.1 + (shannon / 4.32) * 1.9

        return normalized

    def _compute_time_weight(self, sequence: str, position: int) -> float:
        """
        Time weighting function: exponential decay from previous occurrences.

        w_t = Σ exp(-(i-j)/τ) for j < i where sequence[j] == sequence[i]
        """
        if position == 0:
            return 0.1

        current_aa = sequence[position]
        weight_sum = 0.0

        for j in range(position):
            if sequence[j] == current_aa:
                decay = np.exp(-(position - j) / self.tau_decay)
                weight_sum += decay

        # Normalize to [0.1, 2.0]
        normalized = 0.1 + min(weight_sum / 3.0, 1.9)

        return normalized

    def _compute_entropy_weight(self, sequence: str, position: int) -> float:
        """
        Entropy weighting function: local coordinate variance.

        w_e = sqrt(variance of coordinates in window)
        """
        n = len(sequence)
        window_start = max(0, position - self.window_size // 2)
        window_end = min(n, position + self.window_size // 2 + 1)

        # Get coordinates in window
        window_coords = []
        for i in range(window_start, window_end):
            aa = sequence[i]
            if aa in AMINO_ACID_CARDINAL_COORDS:
                window_coords.append(AMINO_ACID_CARDINAL_COORDS[aa])
            else:
                window_coords.append((0.0, 0.0, 0.1))

        if len(window_coords) < 2:
            return 1.0

        window_coords = np.array(window_coords)

        # Compute variance
        mean_coord = np.mean(window_coords, axis=0)
        variance = np.mean(np.sum((window_coords - mean_coord) ** 2, axis=1))
        std_dev = np.sqrt(variance)

        # Normalize to [0.1, 2.0]
        normalized = 0.1 + min(std_dev / 1.0, 1.9)

        return normalized

    def _compute_path_properties(self, path: PeptideCoordinatePath):
        """Compute geometric properties of coordinate path."""
        if len(path.coordinates) < 2:
            return

        coords = np.array([c.to_sentropy_array() for c in path.coordinates])

        # Path length (sum of segment lengths)
        diffs = np.diff(coords, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        path.path_length = float(np.sum(segment_lengths))

        # Endpoint distance
        path.endpoint_distance = float(np.linalg.norm(coords[-1] - coords[0]))

        # Tortuosity (path length / endpoint distance)
        if path.endpoint_distance > 0:
            path.tortuosity = path.path_length / path.endpoint_distance
        else:
            path.tortuosity = 1.0

        # Mean S-Entropy values
        path.mean_s_knowledge = float(np.mean([c.s_knowledge for c in path.coordinates]))
        path.mean_s_time = float(np.mean([c.s_time for c in path.coordinates]))
        path.mean_s_entropy = float(np.mean([c.s_entropy for c in path.coordinates]))

        # Curvature profile (discrete curvature at each point)
        if len(coords) >= 3:
            curvatures = []
            for i in range(1, len(coords) - 1):
                v1 = coords[i] - coords[i-1]
                v2 = coords[i+1] - coords[i]

                norm1 = np.linalg.norm(v1)
                norm2 = np.linalg.norm(v2)

                if norm1 > 0 and norm2 > 0:
                    cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                    cos_angle = np.clip(cos_angle, -1, 1)
                    angle = np.arccos(cos_angle)
                    curvatures.append(angle)
                else:
                    curvatures.append(0.0)

            path.curvature_profile = np.array(curvatures)

        # Velocity profile (segment lengths)
        path.velocity_profile = segment_lengths

    # ========================================================================
    # FRAGMENT SPECTRUM TO GRAPH
    # ========================================================================

    def build_fragment_graph(
        self,
        mz_array: np.ndarray,
        intensity_array: np.ndarray,
        precursor_mz: float,
        precursor_charge: int = 2
    ) -> nx.DiGraph:
        """
        Build directed fragment graph from MS/MS spectrum.

        Nodes: Detected fragment ions
        Edges: Mass differences corresponding to amino acids

        Args:
            mz_array: Array of fragment m/z values
            intensity_array: Array of fragment intensities
            precursor_mz: Precursor m/z
            precursor_charge: Precursor charge state

        Returns:
            NetworkX directed graph representing fragment relationships
        """
        G = nx.DiGraph()

        # Sort by m/z
        sorted_idx = np.argsort(mz_array)
        mz_sorted = mz_array[sorted_idx]
        intensity_sorted = intensity_array[sorted_idx]

        # Create nodes
        for i, (mz, intensity) in enumerate(zip(mz_sorted, intensity_sorted)):
            # Compute S-Entropy coordinates for fragment
            sentropy = self._compute_fragment_sentropy(
                mz, intensity, precursor_mz, i, len(mz_sorted)
            )

            node = FragmentNode(
                mz=mz,
                intensity=intensity,
                sentropy_coords=sentropy
            )

            G.add_node(mz, node=node)

        # Create edges based on amino acid mass differences
        for i, mz_i in enumerate(mz_sorted):
            for j in range(i + 1, len(mz_sorted)):
                mz_j = mz_sorted[j]
                mass_diff = mz_j - mz_i

                # Check if mass difference matches an amino acid
                inferred_aa = self._match_mass_to_aa(mass_diff)

                if inferred_aa is not None:
                    # Compute edge weight (confidence)
                    confidence = self._compute_edge_confidence(
                        intensity_sorted[i], intensity_sorted[j],
                        mass_diff, inferred_aa
                    )

                    # S-Entropy distance
                    node_i = G.nodes[mz_i]['node']
                    node_j = G.nodes[mz_j]['node']
                    sentropy_dist = np.linalg.norm(
                        node_i.sentropy_coords - node_j.sentropy_coords
                    )

                    edge = FragmentEdge(
                        source_mz=mz_i,
                        target_mz=mz_j,
                        mass_difference=mass_diff,
                        inferred_aa=inferred_aa,
                        confidence=confidence,
                        sentropy_distance=sentropy_dist
                    )

                    G.add_edge(mz_i, mz_j, edge=edge, weight=1.0 - confidence)

        return G

    def _compute_fragment_sentropy(
        self,
        mz: float,
        intensity: float,
        precursor_mz: float,
        position: int,
        total_fragments: int
    ) -> np.ndarray:
        """Compute S-Entropy coordinates for a fragment ion."""
        # Normalize values
        mz_norm = mz / precursor_mz
        intensity_norm = np.log1p(intensity)
        position_norm = position / max(1, total_fragments - 1)

        # S-Entropy transformation
        s_knowledge = -np.log2(mz_norm + 0.01) * intensity_norm / 10.0
        s_time = np.exp(-position_norm) * mz_norm
        s_entropy = intensity_norm * (1 - position_norm)

        return np.array([s_knowledge, s_time, s_entropy])

    def _match_mass_to_aa(self, mass_diff: float) -> Optional[str]:
        """Match mass difference to amino acid."""
        for aa, mass in AMINO_ACID_MASSES.items():
            if abs(mass_diff - mass) < self.mass_tolerance:
                return aa
        return None

    def _compute_edge_confidence(
        self,
        intensity_source: float,
        intensity_target: float,
        mass_diff: float,
        aa: str
    ) -> float:
        """Compute confidence score for edge."""
        # Intensity-based confidence
        intensity_factor = min(intensity_source, intensity_target) / max(intensity_source, intensity_target)

        # Mass accuracy
        expected_mass = AMINO_ACID_MASSES[aa]
        mass_error = abs(mass_diff - expected_mass)
        mass_factor = 1.0 - (mass_error / self.mass_tolerance)

        confidence = 0.5 * intensity_factor + 0.5 * mass_factor

        return float(np.clip(confidence, 0, 1))

    # ========================================================================
    # CATEGORICAL COMPLETION
    # ========================================================================

    def categorical_completion(
        self,
        fragment_graph: nx.DiGraph,
        precursor_mz: float,
        max_gap_size: int = 3
    ) -> List[str]:
        """
        Perform categorical completion to infer missing amino acids.

        Fills gaps in the fragment graph by enumerating possible
        amino acid combinations and selecting the one minimizing
        S-Entropy interpolation distance.

        Args:
            fragment_graph: Fragment graph from build_fragment_graph()
            precursor_mz: Precursor m/z
            max_gap_size: Maximum number of amino acids to fill in gap

        Returns:
            List of possible sequences (ranked by S-Entropy distance)
        """
        if len(fragment_graph) == 0:
            return []

        # Find longest path through graph
        try:
            # Get all simple paths from lowest to highest m/z
            mz_values = sorted(fragment_graph.nodes())

            if len(mz_values) < 2:
                return []

            start_node = mz_values[0]
            end_node = mz_values[-1]

            # Find shortest path (by weight = 1 - confidence)
            try:
                path = nx.dijkstra_path(fragment_graph, start_node, end_node)
            except nx.NetworkXNoPath:
                # No direct path - find longest connected component
                path = self._find_best_partial_path(fragment_graph)

            if len(path) < 2:
                return []

            # Extract sequence from path
            sequence = self._extract_sequence_from_path(fragment_graph, path)

            # Fill gaps using categorical completion
            completed_sequence = self._fill_sequence_gaps(
                sequence, path, fragment_graph, precursor_mz, max_gap_size
            )

            return completed_sequence

        except Exception as e:
            print(f"Warning: Categorical completion failed: {e}")
            return []

    def _find_best_partial_path(self, G: nx.DiGraph) -> List[float]:
        """Find the best partial path through the graph."""
        if len(G) == 0:
            return []

        # Find strongly connected components
        components = list(nx.weakly_connected_components(G))

        if len(components) == 0:
            return list(G.nodes())[:2] if len(G) >= 2 else []

        # Use largest component
        largest = max(components, key=len)
        subgraph = G.subgraph(largest)

        mz_values = sorted(subgraph.nodes())

        if len(mz_values) < 2:
            return mz_values

        # Try to find path in largest component
        try:
            return nx.dijkstra_path(subgraph, mz_values[0], mz_values[-1])
        except nx.NetworkXNoPath:
            return mz_values

    def _extract_sequence_from_path(
        self,
        G: nx.DiGraph,
        path: List[float]
    ) -> List[Optional[str]]:
        """Extract amino acid sequence from path through fragment graph."""
        sequence = []

        for i in range(len(path) - 1):
            source = path[i]
            target = path[i + 1]

            if G.has_edge(source, target):
                edge = G.edges[source, target]['edge']
                sequence.append(edge.inferred_aa)
            else:
                sequence.append(None)  # Gap

        return sequence

    def _fill_sequence_gaps(
        self,
        sequence: List[Optional[str]],
        path: List[float],
        G: nx.DiGraph,
        precursor_mz: float,
        max_gap_size: int
    ) -> List[str]:
        """Fill gaps in sequence using categorical completion."""
        filled_sequences = []

        # Find gaps
        gap_positions = [i for i, aa in enumerate(sequence) if aa is None]

        if len(gap_positions) == 0:
            # No gaps - return sequence as is
            return [''.join(sequence)]

        # For each gap, enumerate possible fillings
        for gap_pos in gap_positions:
            if gap_pos >= len(path) - 1:
                continue

            source_mz = path[gap_pos]
            target_mz = path[gap_pos + 1]
            mass_gap = target_mz - source_mz

            # Enumerate amino acid combinations
            candidates = self._enumerate_gap_candidates(mass_gap, max_gap_size)

            if candidates:
                # Select best candidate by S-Entropy distance
                best = self._select_best_candidate(
                    candidates, source_mz, target_mz, G
                )
                sequence[gap_pos] = best

        # Build final sequence (filter remaining Nones)
        final_seq = ''.join([aa for aa in sequence if aa is not None])
        filled_sequences.append(final_seq)

        return filled_sequences

    def _enumerate_gap_candidates(
        self,
        mass_gap: float,
        max_size: int
    ) -> List[str]:
        """Enumerate amino acid combinations matching mass gap."""
        candidates = []

        # Single amino acid
        for aa, mass in AMINO_ACID_MASSES.items():
            if abs(mass - mass_gap) < self.mass_tolerance:
                candidates.append(aa)

        # Two amino acids
        if max_size >= 2:
            for aa1, mass1 in AMINO_ACID_MASSES.items():
                for aa2, mass2 in AMINO_ACID_MASSES.items():
                    if abs(mass1 + mass2 - mass_gap) < self.mass_tolerance:
                        candidates.append(aa1 + aa2)

        # Three amino acids
        if max_size >= 3:
            for aa1, mass1 in AMINO_ACID_MASSES.items():
                for aa2, mass2 in AMINO_ACID_MASSES.items():
                    for aa3, mass3 in AMINO_ACID_MASSES.items():
                        if abs(mass1 + mass2 + mass3 - mass_gap) < self.mass_tolerance:
                            candidates.append(aa1 + aa2 + aa3)

        return candidates

    def _select_best_candidate(
        self,
        candidates: List[str],
        source_mz: float,
        target_mz: float,
        G: nx.DiGraph
    ) -> str:
        """Select best candidate by S-Entropy distance."""
        if len(candidates) == 0:
            return 'X'  # Unknown

        if len(candidates) == 1:
            return candidates[0]

        # Get S-Entropy coords of source and target
        source_node = G.nodes[source_mz]['node']
        target_node = G.nodes[target_mz]['node']

        source_coords = source_node.sentropy_coords
        target_coords = target_node.sentropy_coords

        best_candidate = candidates[0]
        best_distance = float('inf')

        for candidate in candidates:
            # Transform candidate to coordinates
            path = self.transform_peptide(candidate)

            if len(path.coordinates) == 0:
                continue

            # Compute interpolation distance
            candidate_coords = np.mean([
                c.to_sentropy_array() for c in path.coordinates
            ], axis=0)

            # Distance to midpoint of source-target
            midpoint = (source_coords + target_coords) / 2
            distance = np.linalg.norm(candidate_coords - midpoint)

            if distance < best_distance:
                best_distance = distance
                best_candidate = candidate

        return best_candidate

    # ========================================================================
    # SEQUENCE RECONSTRUCTION
    # ========================================================================

    def reconstruct_sequence(
        self,
        mz_array: np.ndarray,
        intensity_array: np.ndarray,
        precursor_mz: float,
        precursor_charge: int = 2,
        known_sequence: Optional[str] = None,
        use_state_counting: bool = True
    ) -> Dict:
        """
        Full de novo sequence reconstruction pipeline.

        Uses state counting framework for improved accuracy when available.

        Args:
            mz_array: Fragment m/z values
            intensity_array: Fragment intensities
            precursor_mz: Precursor m/z
            precursor_charge: Precursor charge
            known_sequence: Known sequence for validation (optional)
            use_state_counting: Use state counting reconstruction (default True)

        Returns:
            Dictionary with reconstruction results
        """
        # Use state counting if available and requested
        if use_state_counting and STATE_COUNTING_AVAILABLE:
            return self._reconstruct_with_state_counting(
                mz_array, intensity_array, precursor_mz, precursor_charge, known_sequence
            )

        # Fallback to graph-based reconstruction
        return self._reconstruct_graph_based(
            mz_array, intensity_array, precursor_mz, precursor_charge, known_sequence
        )

    def _reconstruct_with_state_counting(
        self,
        mz_array: np.ndarray,
        intensity_array: np.ndarray,
        precursor_mz: float,
        precursor_charge: int,
        known_sequence: Optional[str] = None
    ) -> Dict:
        """
        Reconstruct sequence using state counting framework.

        Combines:
        - State counting trajectory completion
        - Fragment-parent hierarchical validation
        - S-Entropy coordinate scoring
        """
        reconstructor = StateCountingReconstructor(
            mass_tolerance=self.mass_tolerance,
            min_intensity_ratio=0.01,
            epsilon_boundary=0.1
        )

        # State counting reconstruction
        sc_result = reconstructor.reconstruct(
            mz_array, intensity_array, precursor_mz, precursor_charge, known_sequence
        )

        # Fragment-parent hierarchical validation
        hierarchy_validation = validate_fragment_hierarchy(mz_array, intensity_array, precursor_mz)

        # Build fragment graph for additional scoring
        G = self.build_fragment_graph(mz_array, intensity_array, precursor_mz, precursor_charge)

        # Enhance candidates with S-Entropy scoring
        enhanced_candidates = []
        for candidate in sc_result.get('candidates', []):
            seq = candidate['sequence']
            if seq:
                path = self.transform_peptide(seq)

                # Combined score with S-Entropy
                sentropy_score = 1.0 / (1.0 + path.tortuosity) if path.tortuosity > 0 else 0.5

                enhanced = {
                    **candidate,
                    'path_length': path.path_length,
                    'tortuosity': path.tortuosity,
                    'mean_s_entropy': path.mean_s_entropy,
                    'mean_s_knowledge': path.mean_s_knowledge,
                    'mean_s_time': path.mean_s_time,
                    'sentropy_score': sentropy_score,
                    'combined_score': 0.6 * candidate.get('score', 0) + 0.2 * sentropy_score + 0.2 * hierarchy_validation.overall_score
                }
                enhanced_candidates.append(enhanced)

        # Sort by combined score
        enhanced_candidates.sort(key=lambda x: -x.get('combined_score', 0))

        # Compute validation metrics against known sequence
        validation_result = {}
        if known_sequence and enhanced_candidates:
            best_seq = enhanced_candidates[0]['sequence']
            validation_result = self._compute_validation_metrics(best_seq, known_sequence)

        return {
            'method': 'state_counting',
            'n_fragments': len(mz_array),
            'n_nodes': len(G.nodes()),
            'n_edges': len(G.edges()),
            'precursor_mass': sc_result.get('precursor_mass', 0),
            'target_length': sc_result.get('target_length', 0),
            'candidates': enhanced_candidates[:10],
            'best_sequence': enhanced_candidates[0]['sequence'] if enhanced_candidates else '',
            'best_score': enhanced_candidates[0].get('combined_score', 0) if enhanced_candidates else 0.0,
            'hierarchy_validation': {
                'overlap_score': hierarchy_validation.overlap_score,
                'wavelength_ratio': hierarchy_validation.wavelength_ratio,
                'energy_ratio': hierarchy_validation.energy_ratio,
                'phase_coherence': hierarchy_validation.phase_coherence,
                'is_valid': hierarchy_validation.is_valid,
                'overall_score': hierarchy_validation.overall_score
            },
            'validation': validation_result
        }

    def _reconstruct_graph_based(
        self,
        mz_array: np.ndarray,
        intensity_array: np.ndarray,
        precursor_mz: float,
        precursor_charge: int,
        known_sequence: Optional[str] = None
    ) -> Dict:
        """
        Fallback graph-based reconstruction.
        """
        # Build fragment graph
        G = self.build_fragment_graph(
            mz_array, intensity_array, precursor_mz, precursor_charge
        )

        # Categorical completion
        candidate_sequences = self.categorical_completion(
            G, precursor_mz, max_gap_size=3
        )

        # Score candidates
        scored_candidates = []
        for seq in candidate_sequences:
            if len(seq) > 0:
                path = self.transform_peptide(seq)

                # Compute theoretical mass
                theoretical_mass = sum(AMINO_ACID_MASSES.get(aa, 110) for aa in seq)

                # Compute mass error
                observed_mass = precursor_mz * precursor_charge - precursor_charge * 1.007276
                mass_error = abs(theoretical_mass - observed_mass)

                scored_candidates.append({
                    'sequence': seq,
                    'length': len(seq),
                    'theoretical_mass': theoretical_mass,
                    'mass_error': mass_error,
                    'path_length': path.path_length,
                    'tortuosity': path.tortuosity,
                    'mean_s_entropy': path.mean_s_entropy
                })

        # Sort by mass error
        scored_candidates.sort(key=lambda x: x['mass_error'])

        # Validation
        validation_result = {}
        if known_sequence and scored_candidates:
            best_seq = scored_candidates[0]['sequence']
            validation_result = self._compute_validation_metrics(best_seq, known_sequence)

        return {
            'method': 'graph_based',
            'n_fragments': len(mz_array),
            'n_nodes': len(G.nodes()),
            'n_edges': len(G.edges()),
            'candidates': scored_candidates[:10],
            'best_sequence': scored_candidates[0]['sequence'] if scored_candidates else '',
            'validation': validation_result
        }

    def _compute_validation_metrics(self, predicted: str, known: str) -> Dict:
        """
        Compute validation metrics comparing predicted to known sequence.
        """
        pred_upper = predicted.upper()
        known_upper = known.upper()

        # Exact match
        exact_match = pred_upper == known_upper

        # Longest common subsequence
        lcs_len = self._lcs_length(pred_upper, known_upper)
        partial_score = lcs_len / max(len(known_upper), 1)

        # Edit distance
        edit_dist = self._edit_distance(pred_upper, known_upper)
        edit_similarity = 1.0 - (edit_dist / max(len(known_upper), len(pred_upper), 1))

        # Amino acid overlap
        pred_set = set(pred_upper)
        known_set = set(known_upper)
        aa_overlap = len(pred_set & known_set) / max(len(known_set), 1)

        # Position-wise accuracy
        correct_positions = sum(1 for i, aa in enumerate(pred_upper)
                               if i < len(known_upper) and known_upper[i] == aa)
        position_accuracy = correct_positions / max(len(known_upper), 1)

        return {
            'exact_match': exact_match,
            'partial_score': partial_score,
            'edit_similarity': edit_similarity,
            'aa_overlap': aa_overlap,
            'position_accuracy': position_accuracy,
            'lcs_length': lcs_len,
            'edit_distance': edit_dist,
            'predicted_length': len(pred_upper),
            'known_length': len(known_upper)
        }

    def _lcs_length(self, s1: str, s2: str) -> int:
        """Compute length of longest common subsequence."""
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        return dp[m][n]

    def _edit_distance(self, s1: str, s2: str) -> int:
        """Compute Levenshtein edit distance."""
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

        return dp[m][n]


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def transform_peptide_to_stellas(sequence: str) -> PeptideCoordinatePath:
    """Quick peptide to St. Stella's coordinate transformation."""
    transformer = StStellasSequenceTransformer()
    return transformer.transform_peptide(sequence)


def build_fragment_graph_stellas(
    mz_array: np.ndarray,
    intensity_array: np.ndarray,
    precursor_mz: float
) -> nx.DiGraph:
    """Quick fragment graph construction."""
    transformer = StStellasSequenceTransformer()
    return transformer.build_fragment_graph(mz_array, intensity_array, precursor_mz)


def reconstruct_sequence_stellas(
    mz_array: np.ndarray,
    intensity_array: np.ndarray,
    precursor_mz: float,
    precursor_charge: int = 2
) -> Dict:
    """Quick sequence reconstruction."""
    transformer = StStellasSequenceTransformer()
    return transformer.reconstruct_sequence(
        mz_array, intensity_array, precursor_mz, precursor_charge
    )


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("St. Stella's Sequence Framework - Example")
    print("=" * 60)

    # Example: Transform peptide sequence
    sequence = "PEPTIDE"
    transformer = StStellasSequenceTransformer()
    path = transformer.transform_peptide(sequence)

    print(f"\nPeptide: {sequence}")
    print(f"Path length: {path.path_length:.4f}")
    print(f"Endpoint distance: {path.endpoint_distance:.4f}")
    print(f"Tortuosity: {path.tortuosity:.4f}")
    print(f"Mean S_knowledge: {path.mean_s_knowledge:.4f}")
    print(f"Mean S_time: {path.mean_s_time:.4f}")
    print(f"Mean S_entropy: {path.mean_s_entropy:.4f}")

    print("\nCoordinate details:")
    for coord in path.coordinates:
        print(f"  {coord.amino_acid} (pos {coord.position}): "
              f"S_k={coord.s_knowledge:.3f}, "
              f"S_t={coord.s_time:.3f}, "
              f"S_e={coord.s_entropy:.3f}")

    # Example: Fragment graph
    print("\n" + "=" * 60)
    print("Fragment Graph Example")
    print("=" * 60)

    # Simulated b-ion series for PEPTIDE
    b_ions = np.array([97.05, 226.09, 323.14, 436.23, 551.26, 664.34])
    intensities = np.array([1000, 5000, 8000, 12000, 6000, 3000])
    precursor = 800.35

    G = transformer.build_fragment_graph(b_ions, intensities, precursor)

    print(f"\nNodes: {len(G.nodes())}")
    print(f"Edges: {len(G.edges())}")

    # Sequence reconstruction
    result = transformer.reconstruct_sequence(b_ions, intensities, precursor, 1)

    print(f"\nReconstructed: {result['best_sequence']}")
    print(f"Candidates: {len(result['candidates'])}")

    print("\n" + "=" * 60)
