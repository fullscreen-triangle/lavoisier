"""
Mzekezeke: Bayesian Evidence Network with Fuzzy Logic for MS Annotations

This module implements a sophisticated annotation system that combines:
- Bayesian evidence networks for probabilistic reasoning
- Fuzzy logic for handling uncertainty in m/z ratios
- Network-based identification system
- Dynamic evidence updating
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict
import json

logger = logging.getLogger(__name__)

class EvidenceType(Enum):
    """Types of evidence in the Bayesian network"""
    MASS_MATCH = "mass_match"
    ISOTOPE_PATTERN = "isotope_pattern"
    FRAGMENTATION = "fragmentation"
    RETENTION_TIME = "retention_time"
    ADDUCT_FORMATION = "adduct_formation"
    NEUTRAL_LOSS = "neutral_loss"
    DATABASE_MATCH = "database_match"
    SPECTRAL_SIMILARITY = "spectral_similarity"

@dataclass
class FuzzyMembership:
    """Fuzzy membership function for m/z annotations"""
    center: float  # Central m/z value
    width: float   # Fuzzy width (uncertainty)
    shape: str     # 'triangular', 'gaussian', 'trapezoidal'
    confidence: float  # Base confidence level

    def membership_degree(self, mz_value: float) -> float:
        """Calculate membership degree for given m/z value"""
        if self.shape == 'gaussian':
            return np.exp(-0.5 * ((mz_value - self.center) / self.width) ** 2)
        elif self.shape == 'triangular':
            distance = abs(mz_value - self.center)
            if distance <= self.width:
                return 1.0 - (distance / self.width)
            else:
                return 0.0
        elif self.shape == 'trapezoidal':
            distance = abs(mz_value - self.center)
            if distance <= self.width * 0.5:
                return 1.0
            elif distance <= self.width:
                return 1.0 - 2 * (distance - self.width * 0.5) / self.width
            else:
                return 0.0
        else:
            raise ValueError(f"Unknown fuzzy shape: {self.shape}")

@dataclass
class EvidenceNode:
    """Node in the Bayesian evidence network"""
    node_id: str
    evidence_type: EvidenceType
    mz_value: float
    intensity: float
    fuzzy_membership: FuzzyMembership
    prior_probability: float
    likelihood: float = 0.0
    posterior_probability: float = 0.0
    connected_nodes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AnnotationCandidate:
    """Candidate annotation with fuzzy logic and evidence"""
    compound_name: str
    molecular_formula: str
    exact_mass: float
    adduct_type: str
    fuzzy_score: float
    evidence_score: float
    network_score: float
    final_confidence: float
    supporting_evidence: List[EvidenceNode] = field(default_factory=list)

class MzekezekeBayesianNetwork:
    """
    Advanced Bayesian evidence network with fuzzy logic for MS annotations.

    This system builds a probabilistic network of evidence for compound identification,
    using fuzzy logic to handle m/z uncertainty and Bayesian inference for evidence
    integration.
    """

    def __init__(self,
                 mass_tolerance_ppm: float = 5.0,
                 fuzzy_width_multiplier: float = 2.0,
                 min_evidence_nodes: int = 2,
                 network_convergence_threshold: float = 1e-6):
        self.mass_tolerance_ppm = mass_tolerance_ppm
        self.fuzzy_width_multiplier = fuzzy_width_multiplier
        self.min_evidence_nodes = min_evidence_nodes
        self.network_convergence_threshold = network_convergence_threshold

        # Network components
        self.evidence_graph = nx.DiGraph()
        self.evidence_nodes: Dict[str, EvidenceNode] = {}
        self.annotation_candidates: List[AnnotationCandidate] = []
        self.fuzzy_rules: Dict[str, Any] = {}

        # Learning components
        self.prior_distributions: Dict[EvidenceType, stats.rv_continuous] = {}
        self.evidence_correlations: Dict[Tuple[EvidenceType, EvidenceType], float] = {}

        self._initialize_prior_distributions()
        self._initialize_fuzzy_rules()

    def _initialize_prior_distributions(self):
        """Initialize prior probability distributions for different evidence types"""
        # Based on typical MS analysis patterns
        self.prior_distributions = {
            EvidenceType.MASS_MATCH: stats.beta(a=2, b=5),  # Conservative for mass alone
            EvidenceType.ISOTOPE_PATTERN: stats.beta(a=5, b=2),  # Strong evidence
            EvidenceType.FRAGMENTATION: stats.beta(a=8, b=2),  # Very strong evidence
            EvidenceType.RETENTION_TIME: stats.beta(a=3, b=4),  # Moderate evidence
            EvidenceType.ADDUCT_FORMATION: stats.beta(a=4, b=3),  # Good evidence
            EvidenceType.NEUTRAL_LOSS: stats.beta(a=4, b=4),  # Moderate evidence
            EvidenceType.DATABASE_MATCH: stats.beta(a=6, b=3),  # Strong evidence
            EvidenceType.SPECTRAL_SIMILARITY: stats.beta(a=7, b=3)  # Very strong evidence
        }

    def _initialize_fuzzy_rules(self):
        """Initialize fuzzy logic rules for evidence combination"""
        self.fuzzy_rules = {
            'mass_accuracy_excellent': {'ppm_threshold': 1.0, 'confidence_boost': 1.5},
            'mass_accuracy_good': {'ppm_threshold': 3.0, 'confidence_boost': 1.2},
            'mass_accuracy_acceptable': {'ppm_threshold': 5.0, 'confidence_boost': 1.0},
            'mass_accuracy_poor': {'ppm_threshold': 10.0, 'confidence_boost': 0.5},

            'intensity_high': {'percentile_threshold': 90, 'importance_weight': 1.3},
            'intensity_medium': {'percentile_threshold': 50, 'importance_weight': 1.0},
            'intensity_low': {'percentile_threshold': 10, 'importance_weight': 0.7},

            'evidence_convergence': {'min_supporting': 3, 'convergence_bonus': 1.4},
            'evidence_conflict': {'conflict_threshold': 0.3, 'conflict_penalty': 0.6}
        }

    def add_evidence_node(self,
                         mz_value: float,
                         intensity: float,
                         evidence_type: EvidenceType,
                         metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add an evidence node to the Bayesian network with fuzzy membership.
        """
        node_id = f"{evidence_type.value}_{mz_value:.4f}_{len(self.evidence_nodes)}"

        # Calculate fuzzy membership parameters
        mass_error_ppm = self.mass_tolerance_ppm
        fuzzy_width = (mz_value * mass_error_ppm * self.fuzzy_width_multiplier) / 1e6

        fuzzy_membership = FuzzyMembership(
            center=mz_value,
            width=fuzzy_width,
            shape='gaussian',  # Default to gaussian for mass spectrometry
            confidence=0.8  # Base confidence
        )

        # Calculate prior probability
        prior_prob = self.prior_distributions[evidence_type].mean()

        evidence_node = EvidenceNode(
            node_id=node_id,
            evidence_type=evidence_type,
            mz_value=mz_value,
            intensity=intensity,
            fuzzy_membership=fuzzy_membership,
            prior_probability=prior_prob,
            metadata=metadata or {}
        )

        self.evidence_nodes[node_id] = evidence_node
        self.evidence_graph.add_node(node_id, **evidence_node.__dict__)

        logger.info(f"Added evidence node {node_id} with fuzzy width {fuzzy_width:.6f}")

        return node_id

    def connect_evidence_nodes(self, node1_id: str, node2_id: str,
                              connection_strength: float):
        """
        Create connections between evidence nodes based on chemical relationships.
        """
        if node1_id in self.evidence_nodes and node2_id in self.evidence_nodes:
            self.evidence_graph.add_edge(node1_id, node2_id, weight=connection_strength)
            self.evidence_nodes[node1_id].connected_nodes.append(node2_id)
            self.evidence_nodes[node2_id].connected_nodes.append(node1_id)

            logger.debug(f"Connected {node1_id} to {node2_id} with strength {connection_strength}")

    def auto_connect_related_evidence(self, correlation_threshold: float = 0.5):
        """
        Automatically connect related evidence nodes based on chemical relationships.
        """
        node_ids = list(self.evidence_nodes.keys())

        for i, node1_id in enumerate(node_ids):
            for node2_id in node_ids[i+1:]:
                node1 = self.evidence_nodes[node1_id]
                node2 = self.evidence_nodes[node2_id]

                # Calculate correlation based on m/z relationships
                correlation = self._calculate_evidence_correlation(node1, node2)

                if correlation >= correlation_threshold:
                    self.connect_evidence_nodes(node1_id, node2_id, correlation)

    def _calculate_evidence_correlation(self, node1: EvidenceNode,
                                      node2: EvidenceNode) -> float:
        """
        Calculate correlation between two evidence nodes.
        """
        # Mass difference analysis
        mass_diff = abs(node1.mz_value - node2.mz_value)

        # Common adduct/neutral loss differences
        common_differences = [
            1.0078,   # H+
            17.0027,  # NH3 loss
            18.0106,  # H2O loss
            22.9898,  # Na+
            28.0061,  # CO loss
            38.9637,  # K+
            44.0262   # CO2 loss
        ]

        # Check for common chemical relationships
        relationship_score = 0.0
        for diff in common_differences:
            if abs(mass_diff - diff) < 0.01:  # 10 mDa tolerance
                relationship_score = 0.8
                break

        # Isotope pattern correlation
        isotope_correlation = 0.0
        if node1.evidence_type == EvidenceType.ISOTOPE_PATTERN or \
           node2.evidence_type == EvidenceType.ISOTOPE_PATTERN:
            isotope_correlation = 0.6

        # Evidence type compatibility
        type_compatibility = self._get_evidence_type_compatibility(
            node1.evidence_type, node2.evidence_type
        )

        # Combined correlation score
        correlation = max(relationship_score, isotope_correlation) * type_compatibility

        return min(1.0, correlation)

    def _get_evidence_type_compatibility(self, type1: EvidenceType,
                                       type2: EvidenceType) -> float:
        """
        Get compatibility score between evidence types.
        """
        compatibility_matrix = {
            (EvidenceType.MASS_MATCH, EvidenceType.ISOTOPE_PATTERN): 0.9,
            (EvidenceType.MASS_MATCH, EvidenceType.FRAGMENTATION): 0.8,
            (EvidenceType.ISOTOPE_PATTERN, EvidenceType.FRAGMENTATION): 0.9,
            (EvidenceType.FRAGMENTATION, EvidenceType.NEUTRAL_LOSS): 0.9,
            (EvidenceType.ADDUCT_FORMATION, EvidenceType.MASS_MATCH): 0.8,
            (EvidenceType.DATABASE_MATCH, EvidenceType.SPECTRAL_SIMILARITY): 0.9,
            (EvidenceType.RETENTION_TIME, EvidenceType.DATABASE_MATCH): 0.7,
        }

        # Check both directions
        key = (type1, type2)
        reverse_key = (type2, type1)

        if key in compatibility_matrix:
            return compatibility_matrix[key]
        elif reverse_key in compatibility_matrix:
            return compatibility_matrix[reverse_key]
        else:
            return 0.5  # Default moderate compatibility

    def update_bayesian_network(self, max_iterations: int = 100) -> bool:
        """
        Perform Bayesian inference to update posterior probabilities.
        """
        logger.info("Starting Bayesian network inference...")

        converged = False
        iteration = 0

        while not converged and iteration < max_iterations:
            old_posteriors = {node_id: node.posterior_probability
                            for node_id, node in self.evidence_nodes.items()}

            # Update each node's posterior probability
            for node_id, node in self.evidence_nodes.items():
                self._update_node_posterior(node)

            # Check convergence
            max_change = 0.0
            for node_id, node in self.evidence_nodes.items():
                change = abs(node.posterior_probability - old_posteriors[node_id])
                max_change = max(max_change, change)

            converged = max_change < self.network_convergence_threshold
            iteration += 1

            if iteration % 10 == 0:
                logger.debug(f"Iteration {iteration}, max change: {max_change:.8f}")

        logger.info(f"Bayesian inference {'converged' if converged else 'stopped'} after {iteration} iterations")
        return converged

    def _update_node_posterior(self, node: EvidenceNode):
        """
        Update posterior probability of a single evidence node using Bayesian inference.
        """
        # Start with prior
        posterior = node.prior_probability

        # Incorporate evidence from connected nodes
        if node.connected_nodes:
            # Calculate likelihood based on connected evidence
            likelihood_product = 1.0

            for connected_id in node.connected_nodes:
                if connected_id in self.evidence_nodes:
                    connected_node = self.evidence_nodes[connected_id]
                    edge_weight = self.evidence_graph[node.node_id][connected_id]['weight']

                    # Likelihood contribution from connected node
                    likelihood_contribution = (connected_node.posterior_probability * edge_weight +
                                             (1 - edge_weight) * 0.5)
                    likelihood_product *= likelihood_contribution

            # Apply fuzzy logic modulation
            fuzzy_modifier = self._calculate_fuzzy_modifier(node)
            likelihood_product *= fuzzy_modifier

            # Bayesian update: P(H|E) = P(E|H) * P(H) / P(E)
            # Simplified version using likelihood ratio
            posterior = (likelihood_product * node.prior_probability) / \
                       (likelihood_product * node.prior_probability +
                        (1 - likelihood_product) * (1 - node.prior_probability))

        node.likelihood = likelihood_product if 'likelihood_product' in locals() else node.prior_probability
        node.posterior_probability = posterior

    def _calculate_fuzzy_modifier(self, node: EvidenceNode) -> float:
        """
        Calculate fuzzy logic modifier based on node properties.
        """
        modifier = 1.0

        # Mass accuracy fuzzy rules
        if node.evidence_type == EvidenceType.MASS_MATCH:
            # Calculate mass error (simplified)
            mass_error_ppm = 2.0  # Placeholder - would be calculated from actual data

            if mass_error_ppm <= self.fuzzy_rules['mass_accuracy_excellent']['ppm_threshold']:
                modifier *= self.fuzzy_rules['mass_accuracy_excellent']['confidence_boost']
            elif mass_error_ppm <= self.fuzzy_rules['mass_accuracy_good']['ppm_threshold']:
                modifier *= self.fuzzy_rules['mass_accuracy_good']['confidence_boost']
            elif mass_error_ppm <= self.fuzzy_rules['mass_accuracy_acceptable']['ppm_threshold']:
                modifier *= self.fuzzy_rules['mass_accuracy_acceptable']['confidence_boost']
            else:
                modifier *= self.fuzzy_rules['mass_accuracy_poor']['confidence_boost']

        # Intensity-based fuzzy rules
        intensity_percentile = 75  # Placeholder - would be calculated from spectrum

        if intensity_percentile >= self.fuzzy_rules['intensity_high']['percentile_threshold']:
            modifier *= self.fuzzy_rules['intensity_high']['importance_weight']
        elif intensity_percentile >= self.fuzzy_rules['intensity_medium']['percentile_threshold']:
            modifier *= self.fuzzy_rules['intensity_medium']['importance_weight']
        else:
            modifier *= self.fuzzy_rules['intensity_low']['importance_weight']

        return modifier

    def generate_annotations(self,
                           compound_database: List[Dict[str, Any]],
                           min_confidence: float = 0.3) -> List[AnnotationCandidate]:
        """
        Generate fuzzy logic-based annotations using the Bayesian evidence network.
        """
        logger.info(f"Generating annotations from {len(compound_database)} compounds...")

        self.annotation_candidates.clear()

        for compound in compound_database:
            # Calculate fuzzy scores for this compound
            fuzzy_score = self._calculate_compound_fuzzy_score(compound)

            # Calculate evidence network score
            evidence_score = self._calculate_compound_evidence_score(compound)

            # Calculate network topology score
            network_score = self._calculate_network_topology_score(compound)

            # Combine scores using fuzzy logic
            final_confidence = self._combine_scores_fuzzy(fuzzy_score, evidence_score, network_score)

            if final_confidence >= min_confidence:
                # Find supporting evidence nodes
                supporting_evidence = self._find_supporting_evidence(compound)

                candidate = AnnotationCandidate(
                    compound_name=compound.get('name', 'Unknown'),
                    molecular_formula=compound.get('formula', ''),
                    exact_mass=compound.get('exact_mass', 0.0),
                    adduct_type=compound.get('adduct', '[M+H]+'),
                    fuzzy_score=fuzzy_score,
                    evidence_score=evidence_score,
                    network_score=network_score,
                    final_confidence=final_confidence,
                    supporting_evidence=supporting_evidence
                )

                self.annotation_candidates.append(candidate)

        # Sort by confidence
        self.annotation_candidates.sort(key=lambda x: x.final_confidence, reverse=True)

        logger.info(f"Generated {len(self.annotation_candidates)} annotation candidates")
        return self.annotation_candidates

    def _calculate_compound_fuzzy_score(self, compound: Dict[str, Any]) -> float:
        """
        Calculate fuzzy logic score for compound annotation.
        """
        compound_mass = compound.get('exact_mass', 0.0)
        fuzzy_scores = []

        for node in self.evidence_nodes.values():
            if node.evidence_type in [EvidenceType.MASS_MATCH, EvidenceType.DATABASE_MATCH]:
                # Calculate fuzzy membership
                membership = node.fuzzy_membership.membership_degree(compound_mass)
                fuzzy_scores.append(membership * node.posterior_probability)

        return np.mean(fuzzy_scores) if fuzzy_scores else 0.0

    def _calculate_compound_evidence_score(self, compound: Dict[str, Any]) -> float:
        """
        Calculate evidence network score for compound.
        """
        evidence_contributions = []

        for node in self.evidence_nodes.values():
            # Weight evidence by posterior probability and network connectivity
            connectivity_bonus = 1.0 + 0.1 * len(node.connected_nodes)
            contribution = node.posterior_probability * connectivity_bonus
            evidence_contributions.append(contribution)

        return np.mean(evidence_contributions) if evidence_contributions else 0.0

    def _calculate_network_topology_score(self, compound: Dict[str, Any]) -> float:
        """
        Calculate score based on network topology and connectivity.
        """
        if len(self.evidence_graph.nodes) < 2:
            return 0.5

        # Network metrics
        avg_clustering = nx.average_clustering(self.evidence_graph.to_undirected())
        density = nx.density(self.evidence_graph)

        # Combine metrics
        topology_score = 0.6 * avg_clustering + 0.4 * density

        return min(1.0, topology_score)

    def _combine_scores_fuzzy(self, fuzzy_score: float,
                             evidence_score: float,
                             network_score: float) -> float:
        """
        Combine scores using fuzzy logic operations.
        """
        # Fuzzy AND operation (minimum)
        fuzzy_and = min(fuzzy_score, evidence_score, network_score)

        # Fuzzy OR operation (maximum)
        fuzzy_or = max(fuzzy_score, evidence_score, network_score)

        # Weighted average (compromise between AND/OR)
        weighted_avg = (0.4 * fuzzy_score + 0.4 * evidence_score + 0.2 * network_score)

        # Final combination using fuzzy aggregation
        final_score = 0.3 * fuzzy_and + 0.3 * fuzzy_or + 0.4 * weighted_avg

        return min(1.0, max(0.0, final_score))

    def _find_supporting_evidence(self, compound: Dict[str, Any]) -> List[EvidenceNode]:
        """
        Find evidence nodes that support this compound annotation.
        """
        supporting_nodes = []
        compound_mass = compound.get('exact_mass', 0.0)

        for node in self.evidence_nodes.values():
            # Check if node supports this compound
            if node.evidence_type == EvidenceType.MASS_MATCH:
                membership = node.fuzzy_membership.membership_degree(compound_mass)
                if membership > 0.5 and node.posterior_probability > 0.3:
                    supporting_nodes.append(node)
            elif node.posterior_probability > 0.5:
                supporting_nodes.append(node)

        return supporting_nodes

    def get_network_summary(self) -> Dict[str, Any]:
        """
        Generate comprehensive network analysis summary.
        """
        if not self.evidence_nodes:
            return {"error": "No evidence nodes in network"}

        # Basic network statistics
        num_nodes = len(self.evidence_nodes)
        num_edges = len(self.evidence_graph.edges)
        avg_degree = 2 * num_edges / num_nodes if num_nodes > 0 else 0

        # Evidence type distribution
        evidence_types = [node.evidence_type.value for node in self.evidence_nodes.values()]
        type_counts = pd.Series(evidence_types).value_counts().to_dict()

        # Confidence statistics
        posteriors = [node.posterior_probability for node in self.evidence_nodes.values()]
        confidence_stats = {
            'mean_confidence': np.mean(posteriors),
            'std_confidence': np.std(posteriors),
            'min_confidence': np.min(posteriors),
            'max_confidence': np.max(posteriors)
        }

        # Network connectivity analysis
        if num_nodes > 1:
            connectivity_analysis = {
                'density': nx.density(self.evidence_graph),
                'average_clustering': nx.average_clustering(self.evidence_graph.to_undirected()),
                'is_connected': nx.is_strongly_connected(self.evidence_graph)
            }
        else:
            connectivity_analysis = {'density': 0, 'average_clustering': 0, 'is_connected': False}

        summary = {
            'network_size': {
                'nodes': num_nodes,
                'edges': num_edges,
                'average_degree': avg_degree
            },
            'evidence_distribution': type_counts,
            'confidence_statistics': confidence_stats,
            'network_connectivity': connectivity_analysis,
            'annotation_candidates': len(self.annotation_candidates),
            'fuzzy_rules_active': len(self.fuzzy_rules)
        }

        return summary

    def export_network(self, filename: str):
        """
        Export the Bayesian evidence network for visualization and analysis.
        """
        # Prepare network data
        network_data = {
            'nodes': [
                {
                    'id': node_id,
                    'evidence_type': node.evidence_type.value,
                    'mz_value': node.mz_value,
                    'intensity': node.intensity,
                    'prior_probability': node.prior_probability,
                    'posterior_probability': node.posterior_probability,
                    'fuzzy_center': node.fuzzy_membership.center,
                    'fuzzy_width': node.fuzzy_membership.width,
                    'fuzzy_shape': node.fuzzy_membership.shape
                }
                for node_id, node in self.evidence_nodes.items()
            ],
            'edges': [
                {
                    'source': edge[0],
                    'target': edge[1],
                    'weight': self.evidence_graph[edge[0]][edge[1]]['weight']
                }
                for edge in self.evidence_graph.edges()
            ],
            'annotations': [
                {
                    'compound_name': candidate.compound_name,
                    'molecular_formula': candidate.molecular_formula,
                    'exact_mass': candidate.exact_mass,
                    'final_confidence': candidate.final_confidence,
                    'fuzzy_score': candidate.fuzzy_score,
                    'evidence_score': candidate.evidence_score,
                    'network_score': candidate.network_score
                }
                for candidate in self.annotation_candidates
            ]
        }

        with open(filename, 'w') as f:
            json.dump(network_data, f, indent=2)

        logger.info(f"Network exported to {filename}")
