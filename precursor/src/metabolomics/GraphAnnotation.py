#!/usr/bin/env python3
"""
Graph-Based Annotation for Mass Spectrometry
=============================================

Extends DatabaseSearch.py with graph network architecture based on:
- S-Entropy coordinate transformation
- Phase-lock signature matching
- Bidirectional graph completion (numerical + visual modalities)
- Empty Dictionary dynamic synthesis

Theoretical Foundation:
-----------------------
Instead of hierarchical database organization, both library and experimental
spectra form random graph networks based on S-Entropy proximity. The dual
modality (numerical + visual) creates new categorical states through graph
intersection, resolving Gibbs' paradox for fragment disambiguation.

Key Concepts:
- Library Graph: Reference spectra connected by S-Entropy similarity
- Experimental Graph: Query spectra connected by phase-lock relationships
- Bidirectional Flow: Graph-to-graph matching creates categorical states
- Empty Dictionary: Dynamic synthesis without explicit storage

References:
-----------
- docs/oscillatory/categorical-completion.tex
- docs/oscillatory/entropy-coordinates.tex
- docs/oscillatory/tandem-mass-spec.tex

Author: Kundai Chinyamakobvu
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from collections import defaultdict
import logging

# Import S-Entropy and Vector transformation
from precursor.src.core.EntropyTransformation import (
    SEntropyTransformer,
    SEntropyCoordinates,
    SEntropyFeatures,
    PhaseLockSignatureComputer
)
from precursor.src.core.VectorTransformation import (
    VectorTransformer,
    SpectrumEmbedding,
    MSDataContainerIntegration
)
from precursor.src.core.DataStructure import MSDataContainer

# Import original annotation components
from .DatabaseSearch import MSAnnotator, AnnotationParameters


@dataclass
class GraphNode:
    """
    Node in the spectral graph network.

    Attributes:
        node_id: Unique identifier
        spectrum_embedding: Vector embedding
        s_entropy_coords: 3D S-Entropy coordinates
        s_entropy_features: 14D feature vector
        phase_lock_signature: 64D phase-lock signature
        categorical_state: Categorical completion state
        metadata: Spectrum metadata
        neighbors: Connected nodes
        edge_weights: Weights to neighbors
    """
    node_id: str
    spectrum_embedding: SpectrumEmbedding
    s_entropy_coords: np.ndarray
    s_entropy_features: SEntropyFeatures
    phase_lock_signature: np.ndarray
    categorical_state: int
    metadata: Dict[str, Any]
    neighbors: List[str] = field(default_factory=list)
    edge_weights: Dict[str, float] = field(default_factory=dict)

    def add_neighbor(self, neighbor_id: str, weight: float):
        """Add a neighboring node."""
        if neighbor_id not in self.neighbors:
            self.neighbors.append(neighbor_id)
        self.edge_weights[neighbor_id] = weight


@dataclass
class CategoricalState:
    """
    Represents a categorical completion state from dual-modality intersection.

    When a library node and experimental node create a match, they form
    a new categorical state that contains more information than either alone.
    This is the resolution of Gibbs' paradox.
    """
    state_id: int
    library_node_id: str
    experimental_node_id: str
    numerical_similarity: float  # S-Entropy distance
    visual_similarity: float  # Phase-lock similarity
    dual_modality_score: float  # Combined score
    entropy_increase: float  # Information gain from intersection
    confidence: float


class SpectralGraphNetwork:
    """
    Graph network for spectral organization.

    Instead of hierarchical trees, spectra are organized as a random graph
    where edges connect spectra with similar S-Entropy coordinates and
    phase-lock signatures.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.7,
        max_neighbors: int = 10,
        use_categorical_states: bool = True
    ):
        """
        Initialize spectral graph network.

        Args:
            similarity_threshold: Minimum similarity to connect nodes
            max_neighbors: Maximum neighbors per node
            use_categorical_states: Enable categorical state tracking
        """
        self.graph = nx.Graph()
        self.nodes: Dict[str, GraphNode] = {}
        self.similarity_threshold = similarity_threshold
        self.max_neighbors = max_neighbors
        self.use_categorical_states = use_categorical_states

        # Categorical state tracking
        self.categorical_states: Dict[int, List[str]] = defaultdict(list)
        self.next_state_id = 0

        self.logger = logging.getLogger(__name__)

    def add_node(self, node: GraphNode):
        """Add a node to the graph."""
        self.nodes[node.node_id] = node
        self.graph.add_node(
            node.node_id,
            categorical_state=node.categorical_state,
            s_entropy_coords=node.s_entropy_coords,
            phase_lock_sig=node.phase_lock_signature
        )

        # Track categorical state
        if self.use_categorical_states:
            self.categorical_states[node.categorical_state].append(node.node_id)

    def connect_nodes(
        self,
        node1_id: str,
        node2_id: str,
        similarity: float
    ):
        """Connect two nodes with weighted edge."""
        if node1_id not in self.nodes or node2_id not in self.nodes:
            return

        # Add edge to graph
        self.graph.add_edge(node1_id, node2_id, weight=similarity)

        # Update node neighbors
        self.nodes[node1_id].add_neighbor(node2_id, similarity)
        self.nodes[node2_id].add_neighbor(node1_id, similarity)

    def build_graph_from_embeddings(
        self,
        embeddings: List[SpectrumEmbedding],
        node_ids: List[str],
        metadata_list: List[Dict]
    ):
        """
        Build graph from spectrum embeddings.

        Connects nodes based on S-Entropy similarity and phase-lock signatures.
        """
        self.logger.info(f"Building graph from {len(embeddings)} spectra...")

        # Add all nodes
        for i, (embedding, node_id, metadata) in enumerate(zip(embeddings, node_ids, metadata_list)):
            node = GraphNode(
                node_id=node_id,
                spectrum_embedding=embedding,
                s_entropy_coords=embedding.s_entropy_features.to_array()[:3],  # First 3 dims
                s_entropy_features=embedding.s_entropy_features,
                phase_lock_signature=embedding.phase_lock_signature,
                categorical_state=embedding.categorical_state,
                metadata=metadata
            )
            self.add_node(node)

        # Compute pairwise similarities
        n = len(embeddings)
        for i in range(n):
            # Find k nearest neighbors
            similarities = []
            for j in range(n):
                if i == j:
                    continue

                # Calculate dual-modality similarity
                sim = embeddings[i].similarity_to(embeddings[j], metric='dual')
                if sim >= self.similarity_threshold:
                    similarities.append((j, sim))

            # Sort by similarity and keep top k
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_neighbors = similarities[:self.max_neighbors]

            # Connect to neighbors
            for j, sim in top_neighbors:
                self.connect_nodes(node_ids[i], node_ids[j], sim)

        self.logger.info(f"Graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")

    def find_nearest_neighbors(
        self,
        query_embedding: SpectrumEmbedding,
        k: int = 10,
        metric: str = 'dual'
    ) -> List[Tuple[str, float]]:
        """
        Find k nearest neighbors to query.

        Args:
            query_embedding: Query spectrum embedding
            k: Number of neighbors
            metric: Similarity metric

        Returns:
            List of (node_id, similarity) tuples
        """
        similarities = []

        for node_id, node in self.nodes.items():
            sim = query_embedding.similarity_to(node.spectrum_embedding, metric=metric)
            similarities.append((node_id, sim))

        # Sort and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]

    def get_categorical_state_members(self, state_id: int) -> List[GraphNode]:
        """Get all nodes in a categorical state."""
        node_ids = self.categorical_states.get(state_id, [])
        return [self.nodes[nid] for nid in node_ids if nid in self.nodes]

    def get_subgraph_around_node(
        self,
        node_id: str,
        radius: int = 2
    ) -> nx.Graph:
        """
        Extract subgraph around a node.

        Args:
            node_id: Center node
            radius: How many hops to include

        Returns:
            Subgraph
        """
        # Get all nodes within radius
        nodes_in_subgraph = set([node_id])
        current_layer = set([node_id])

        for _ in range(radius):
            next_layer = set()
            for node in current_layer:
                if node in self.graph:
                    neighbors = set(self.graph.neighbors(node))
                    next_layer.update(neighbors)
            nodes_in_subgraph.update(next_layer)
            current_layer = next_layer

        return self.graph.subgraph(nodes_in_subgraph)


class GraphBasedAnnotator:
    """
    Graph-based annotation system using dual-modality graph intersection.

    This extends MSAnnotator with:
    1. Library as random graph (not hierarchical tree)
    2. Experimental spectra as random graph
    3. Bidirectional graph-to-graph matching
    4. Empty Dictionary dynamic synthesis
    5. Categorical state-based disambiguation
    """

    def __init__(
        self,
        params: AnnotationParameters,
        vector_transformer: Optional[VectorTransformer] = None,
        use_original_annotator: bool = True
    ):
        """
        Initialize graph-based annotator.

        Args:
            params: Annotation parameters
            vector_transformer: Vector transformer (creates default if None)
            use_original_annotator: Whether to also use original MSAnnotator
        """
        self.params = params
        self.logger = logging.getLogger(__name__)

        # Initialize transformers
        self.s_entropy_transformer = SEntropyTransformer()
        self.vector_transformer = vector_transformer or VectorTransformer(
            embedding_method='enhanced_entropy',
            embedding_dim=256,
            normalize=True
        )

        # Initialize graph networks
        self.library_graph = SpectralGraphNetwork(
            similarity_threshold=0.7,
            max_neighbors=10
        )
        self.experimental_graph = SpectralGraphNetwork(
            similarity_threshold=0.7,
            max_neighbors=10
        )

        # Categorical states from intersections
        self.intersection_states: List[CategoricalState] = []
        self.next_intersection_state_id = 0

        # Original annotator (for comparison/fallback)
        self.original_annotator = None
        if use_original_annotator:
            try:
                self.original_annotator = MSAnnotator(params)
            except Exception as e:
                self.logger.warning(f"Could not initialize original annotator: {e}")

        # Empty Dictionary state
        self.empty_dictionary: Dict[int, Dict] = {}  # categorical_state -> synthesized info

    def load_library_from_datacontainer(
        self,
        library_container: MSDataContainer,
        ms_level: int = 2
    ):
        """
        Load spectral library from MSDataContainer into graph network.

        Args:
            library_container: Library data
            ms_level: MS level to use (1 or 2)
        """
        self.logger.info("Loading library into graph network...")

        # Transform all library spectra
        embeddings = []
        node_ids = []
        metadata_list = []

        for spec_idx, spectrum in library_container.spectra_dict.items():
            metadata_obj = library_container.get_spectrum_metadata(spec_idx)

            if metadata_obj.ms_level != ms_level:
                continue

            # Extract data
            mz_array = spectrum['mz'].values
            intensity_array = spectrum['i'].values
            precursor_mz = metadata_obj.precursor_mz if metadata_obj.ms_level == 2 else None
            rt = metadata_obj.scan_time

            # Transform to embedding
            embedding = self.vector_transformer.transform_spectrum(
                mz_array, intensity_array, precursor_mz, rt,
                metadata={
                    'spec_index': spec_idx,
                    'source': 'library',
                    'sample_name': library_container.sample_name
                }
            )

            embeddings.append(embedding)
            node_ids.append(f"lib_{spec_idx}")
            metadata_list.append({
                'spec_index': spec_idx,
                'precursor_mz': precursor_mz,
                'rt': rt,
                'sample_name': library_container.sample_name
            })

        # Build graph
        self.library_graph.build_graph_from_embeddings(
            embeddings, node_ids, metadata_list
        )

        self.logger.info(f"Library graph loaded: {len(embeddings)} spectra")

    def annotate_datacontainer(
        self,
        query_container: MSDataContainer,
        ms_level: int = 2,
        enable_original_annotator: bool = False
    ) -> pd.DataFrame:
        """
        Annotate spectra from MSDataContainer using graph-based approach.

        Args:
            query_container: Experimental data
            ms_level: MS level to annotate
            enable_original_annotator: Also run original annotator for comparison

        Returns:
            DataFrame with annotations
        """
        self.logger.info("Starting graph-based annotation...")

        # Step 1: Build experimental graph
        exp_embeddings = []
        exp_node_ids = []
        exp_metadata_list = []
        exp_spec_indices = []

        for spec_idx, spectrum in query_container.spectra_dict.items():
            metadata_obj = query_container.get_spectrum_metadata(spec_idx)

            if metadata_obj.ms_level != ms_level:
                continue

            # Extract data
            mz_array = spectrum['mz'].values
            intensity_array = spectrum['i'].values
            precursor_mz = metadata_obj.precursor_mz if metadata_obj.ms_level == 2 else None
            rt = metadata_obj.scan_time

            # Transform to embedding
            embedding = self.vector_transformer.transform_spectrum(
                mz_array, intensity_array, precursor_mz, rt,
                metadata={
                    'spec_index': spec_idx,
                    'source': 'experimental',
                    'sample_name': query_container.sample_name
                }
            )

            exp_embeddings.append(embedding)
            exp_node_ids.append(f"exp_{spec_idx}")
            exp_metadata_list.append({
                'spec_index': spec_idx,
                'precursor_mz': precursor_mz,
                'rt': rt,
                'sample_name': query_container.sample_name
            })
            exp_spec_indices.append(spec_idx)

        # Build experimental graph
        self.experimental_graph.build_graph_from_embeddings(
            exp_embeddings, exp_node_ids, exp_metadata_list
        )

        # Step 2: Bidirectional graph completion
        # Match experimental graph → library graph
        annotations = self._bidirectional_graph_matching(
            exp_embeddings, exp_node_ids, exp_spec_indices
        )

        # Step 3: Run original annotator if enabled (for comparison)
        if enable_original_annotator and self.original_annotator:
            self.logger.info("Running original annotator for comparison...")
            # TODO: Convert to format expected by original annotator
            # original_annotations = self.original_annotator.annotate(...)

        # Convert to DataFrame
        results_df = pd.DataFrame(annotations)

        self.logger.info(f"Annotation complete: {len(results_df)} results")
        return results_df

    def _bidirectional_graph_matching(
        self,
        query_embeddings: List[SpectrumEmbedding],
        query_node_ids: List[str],
        query_spec_indices: List[int]
    ) -> List[Dict]:
        """
        Perform bidirectional matching between experimental and library graphs.

        This is the core of the dual-modality system:
        - Forward: Experimental → Library (standard search)
        - Backward: Library → Experimental (validate matches)
        - Intersection: Creates new categorical states
        """
        annotations = []

        for query_embedding, query_node_id, spec_idx in zip(
            query_embeddings, query_node_ids, query_spec_indices
        ):
            # Forward matching: Find nearest library neighbors
            library_matches = self.library_graph.find_nearest_neighbors(
                query_embedding, k=10, metric='dual'
            )

            # For each library match, perform backward validation
            validated_matches = []
            for lib_node_id, forward_sim in library_matches:
                lib_node = self.library_graph.nodes[lib_node_id]

                # Backward: Does this library spectrum also find the query?
                backward_matches = self.experimental_graph.find_nearest_neighbors(
                    lib_node.spectrum_embedding, k=5, metric='dual'
                )

                # Check if query is in backward matches
                backward_sim = 0.0
                query_in_backward = False
                for back_node_id, back_sim in backward_matches:
                    if back_node_id == query_node_id:
                        backward_sim = back_sim
                        query_in_backward = True
                        break

                # Bidirectional consistency score
                if query_in_backward:
                    consistency_score = (forward_sim + backward_sim) / 2.0
                else:
                    consistency_score = forward_sim * 0.5  # Penalty for no backward match

                validated_matches.append({
                    'library_node_id': lib_node_id,
                    'forward_similarity': forward_sim,
                    'backward_similarity': backward_sim if query_in_backward else 0.0,
                    'bidirectional_consistency': consistency_score,
                    'library_metadata': lib_node.metadata
                })

            # Sort by consistency
            validated_matches.sort(
                key=lambda x: x['bidirectional_consistency'],
                reverse=True
            )

            # Create categorical states for top matches
            for match in validated_matches[:5]:  # Top 5
                categorical_state = self._create_categorical_state(
                    query_embedding, query_node_id, match
                )

                # Check Empty Dictionary for dynamic synthesis
                synthesized_info = self._empty_dictionary_lookup(categorical_state)

                # Create annotation record
                annotation = {
                    'query_spec_index': spec_idx,
                    'query_node_id': query_node_id,
                    'library_node_id': match['library_node_id'],
                    'categorical_state_id': categorical_state.state_id,
                    'numerical_similarity': categorical_state.numerical_similarity,
                    'visual_similarity': categorical_state.visual_similarity,
                    'dual_modality_score': categorical_state.dual_modality_score,
                    'bidirectional_consistency': match['bidirectional_consistency'],
                    'entropy_increase': categorical_state.entropy_increase,
                    'confidence': categorical_state.confidence,
                    'synthesized_annotation': synthesized_info,
                    **match['library_metadata']
                }

                annotations.append(annotation)

        return annotations

    def _create_categorical_state(
        self,
        query_embedding: SpectrumEmbedding,
        query_node_id: str,
        match: Dict
    ) -> CategoricalState:
        """
        Create a new categorical state from graph intersection.

        This is the Gibbs' paradox resolution: combining two modalities
        (experimental + library) creates a new categorical state with
        more information than either alone.
        """
        # Numerical similarity (S-Entropy distance)
        numerical_sim = 1.0 - cosine(
            query_embedding.s_entropy_features.to_array(),
            self.library_graph.nodes[match['library_node_id']].s_entropy_features.to_array()
        )

        # Visual similarity (Phase-lock signature)
        visual_sim = 1.0 - cosine(
            query_embedding.phase_lock_signature,
            self.library_graph.nodes[match['library_node_id']].phase_lock_signature
        )

        # Dual modality score (weighted combination)
        dual_score = 0.6 * numerical_sim + 0.4 * visual_sim

        # Entropy increase (information gain from intersection)
        # This is always positive, per categorical completion theory
        base_entropy = -np.log2(max(numerical_sim, 0.001))  # Entropy of single modality
        dual_entropy = -np.log2(max(dual_score, 0.001))  # Entropy with both modalities
        entropy_increase = abs(dual_entropy - base_entropy)

        # Confidence (based on bidirectional consistency)
        confidence = match['bidirectional_consistency'] * dual_score

        # Create state
        state = CategoricalState(
            state_id=self.next_intersection_state_id,
            library_node_id=match['library_node_id'],
            experimental_node_id=query_node_id,
            numerical_similarity=numerical_sim,
            visual_similarity=visual_sim,
            dual_modality_score=dual_score,
            entropy_increase=entropy_increase,
            confidence=confidence
        )

        self.intersection_states.append(state)
        self.next_intersection_state_id += 1

        return state

    def _empty_dictionary_lookup(
        self,
        categorical_state: CategoricalState
    ) -> Dict[str, Any]:
        """
        Empty Dictionary: Dynamic annotation synthesis without storage.

        Instead of storing all possible annotations, we synthesize them
        on-the-fly using the categorical state and phase-lock relationships.

        This implements the "Empty Dictionary" concept from the theory.
        """
        # Check if we've already synthesized for this state
        if categorical_state.state_id in self.empty_dictionary:
            return self.empty_dictionary[categorical_state.state_id]

        # Synthesize annotation from categorical state
        lib_node = self.library_graph.nodes[categorical_state.library_node_id]
        exp_node = self.experimental_graph.nodes[categorical_state.experimental_node_id]

        # Extract information from both nodes
        synthesized_info = {
            'compound_name': lib_node.metadata.get('compound_name', f'Unknown_{categorical_state.state_id}'),
            'formula': lib_node.metadata.get('formula', ''),
            'source': 'empty_dictionary',
            'synthesis_method': 'categorical_state_intersection',
            'categorical_state_id': categorical_state.state_id,
            'phase_lock_strength': float(np.linalg.norm(lib_node.phase_lock_signature)),
            'entropy_contribution': categorical_state.entropy_increase
        }

        # Store for future lookups
        self.empty_dictionary[categorical_state.state_id] = synthesized_info

        return synthesized_info

    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get statistics about the graph networks."""
        stats = {
            'library_graph': {
                'num_nodes': self.library_graph.graph.number_of_nodes(),
                'num_edges': self.library_graph.graph.number_of_edges(),
                'density': nx.density(self.library_graph.graph),
                'num_categorical_states': len(self.library_graph.categorical_states)
            },
            'experimental_graph': {
                'num_nodes': self.experimental_graph.graph.number_of_nodes(),
                'num_edges': self.experimental_graph.graph.number_of_edges(),
                'density': nx.density(self.experimental_graph.graph),
                'num_categorical_states': len(self.experimental_graph.categorical_states)
            },
            'intersection': {
                'num_intersection_states': len(self.intersection_states),
                'empty_dictionary_size': len(self.empty_dictionary)
            }
        }
        return stats


# Helper function for scipy import
from scipy.spatial.distance import cosine


# ============================================================================
# Example Usage
# ============================================================================

def example_graph_annotation():
    """Example: Annotate spectra using graph-based approach."""
    print("=" * 70)
    print("Graph-Based Annotation Example")
    print("=" * 70)

    # This would be used with real MSDataContainer objects
    print("\nUsage:")
    print("------")
    print("# Initialize annotator")
    print("params = AnnotationParameters()")
    print("annotator = GraphBasedAnnotator(params)")
    print()
    print("# Load library")
    print("library_container = MSDataContainer(...)")
    print("annotator.load_library_from_datacontainer(library_container, ms_level=2)")
    print()
    print("# Annotate experimental data")
    print("query_container = MSDataContainer(...)")
    print("results = annotator.annotate_datacontainer(query_container, ms_level=2)")
    print()
    print("# Get graph statistics")
    print("stats = annotator.get_graph_statistics()")
    print("print(stats)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    example_graph_annotation()
