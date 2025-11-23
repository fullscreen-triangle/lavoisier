"""
Frequency Hierarchy Architecture for Virtual Mass Spectrometry
===============================================================

Implements the 8-scale hardware oscillatory hierarchy mapped to
molecular properties for Molecular Maxwell Demon materialization.

From oscillatory_computation_readme.md:
Scale 1 (10^15 Hz): CPU clock → Quantum molecular properties
Scale 2 (10^6 Hz):  Memory bus → Fragment coupling
Scale 3 (10^2 Hz):  Disk I/O → Conformational dynamics
Scale 4 (sub-Hz):   Network → Ensemble dynamics
Scale 5 (mHz):      USB → Validation rhythm
Scale 6 (μHz):      Display → Spectroscopic features
Scale 7 (μHz):      Timers → Physiological rhythms
Scale 8 (nHz):      Process → Global coherence

Key Concept: Gear ratios enable O(1) hierarchical navigation between scales.

Author: Kundai Farai Sachikonye
Date: 2025
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum


class HardwareScale(Enum):
    """8-scale hardware oscillatory hierarchy"""
    SCALE_1_QUANTUM = 1      # CPU clock (10^15 Hz) → Quantum membrane
    SCALE_2_FRAGMENT = 2     # Memory bus (10^6 Hz) → Intracellular circuit
    SCALE_3_CONFORM = 3      # Disk I/O (10^2 Hz) → Cellular information
    SCALE_4_ENSEMBLE = 4     # Network (sub-Hz) → Tissue integration
    SCALE_5_VALIDATION = 5   # USB (mHz) → Microbiome community
    SCALE_6_SPECTRO = 6      # Display (μHz) → Organ coordination
    SCALE_7_PHYSIO = 7       # Timers (μHz) → Physiological systems
    SCALE_8_GLOBAL = 8       # Process (nHz) → Allometric organism


@dataclass
class FrequencyHierarchyNode:
    """
    Node in frequency hierarchy = finite observer's view.

    Each node represents a categorical state in the phase-lock network.
    Nodes are connected through gear ratios enabling O(1) navigation.
    """
    node_id: str
    hierarchical_level: HardwareScale

    # Frequency information
    frequency_hz: float
    frequency_range: Tuple[float, float]  # (min, max) observable at this scale

    # Hardware source
    hardware_source: str  # 'clock', 'memory', 'network', etc.

    # Observation window (finite observer's view)
    observation_window_start: float
    observation_window_end: float

    # Phase-lock signatures detected in this window
    phase_lock_signatures: List[Dict[str, Any]] = field(default_factory=list)

    # S-entropy coordinates (if computed)
    s_coordinates: Optional[Tuple[float, float, float]] = None  # (S_k, S_t, S_e)

    # Categorical state ID
    categorical_state_id: Optional[str] = None

    # Hierarchy navigation
    parent_node: Optional['FrequencyHierarchyNode'] = None
    child_nodes: List['FrequencyHierarchyNode'] = field(default_factory=list)
    gear_ratio_to_parent: Optional[float] = None  # ω_parent / ω_this

    # Convergence metrics
    convergence_score: float = 0.0  # Higher = better convergence node
    is_convergence_node: bool = False

    def add_child(self, child_node: 'FrequencyHierarchyNode', gear_ratio: float):
        """Add child node with gear ratio"""
        child_node.parent_node = self
        child_node.gear_ratio_to_parent = gear_ratio
        self.child_nodes.append(child_node)

    def compute_convergence_score(self) -> float:
        """
        Compute convergence score (high phase-lock density).

        Convergence nodes are where multiple categorical paths intersect,
        making them optimal sites for MMD materialization.
        """
        if not self.phase_lock_signatures:
            self.convergence_score = 0.0
            return 0.0

        # Number of phase-locks
        n_locks = len(self.phase_lock_signatures)

        # Average coherence strength
        avg_coherence = np.mean([
            sig.get('coherence', 0) for sig in self.phase_lock_signatures
        ])

        # Frequency clustering (many frequencies → high convergence)
        frequencies = [sig.get('frequency', 0) for sig in self.phase_lock_signatures]
        freq_std = np.std(frequencies) if len(frequencies) > 1 else 0
        freq_clustering = 1.0 / (1.0 + freq_std)  # Higher when frequencies cluster

        # Combined score
        self.convergence_score = n_locks * avg_coherence * freq_clustering

        return self.convergence_score

    def navigate_to_parent_via_gear_ratio(self) -> Optional['FrequencyHierarchyNode']:
        """
        Navigate to parent node via gear ratio (O(1) operation).

        This is the key to hierarchical navigation: gear ratios enable
        instant jumps between scales without traversing intermediate nodes.
        """
        return self.parent_node

    def navigate_to_child_via_gear_ratio(self, target_frequency: float) -> Optional['FrequencyHierarchyNode']:
        """
        Navigate to child node closest to target frequency via gear ratio.

        O(log N) where N = number of children, but typically O(1) since
        each node has only a few children.
        """
        if not self.child_nodes:
            return None

        # Find child with frequency closest to target
        best_child = min(
            self.child_nodes,
            key=lambda c: abs(c.frequency_hz - target_frequency)
        )

        return best_child


class FrequencyHierarchyTree:
    """
    Complete frequency hierarchy tree for MMD navigation.

    8 levels (hardware scales) with gear ratios between levels.
    Finite observers deployed at each level.
    Convergence nodes identified for MMD materialization.
    """

    def __init__(self):
        self.root: Optional[FrequencyHierarchyNode] = None
        self.nodes_by_level: Dict[HardwareScale, List[FrequencyHierarchyNode]] = {
            scale: [] for scale in HardwareScale
        }
        self.convergence_nodes: List[FrequencyHierarchyNode] = []

    def build_from_hardware_oscillations(self, hardware_measurements: Dict[str, Any]):
        """
        Build hierarchy from hardware oscillation measurements.

        Args:
            hardware_measurements: Dict with keys:
                'clock': {frequency, phase, coherence}
                'memory': {frequency, phase, coherence}
                'network': {frequency, phase, coherence}
                ... for all 8 hardware sources
        """
        # Create root node (Scale 8: Global)
        global_freq = hardware_measurements.get('global', {}).get('frequency', 1e-9)
        self.root = FrequencyHierarchyNode(
            node_id="scale_8_global",
            hierarchical_level=HardwareScale.SCALE_8_GLOBAL,
            frequency_hz=global_freq,
            frequency_range=(global_freq * 0.9, global_freq * 1.1),
            hardware_source='process_scheduling',
            observation_window_start=0.0,
            observation_window_end=1e10  # Very large window for global scale
        )
        self.nodes_by_level[HardwareScale.SCALE_8_GLOBAL].append(self.root)

        # Build hierarchy downward (Scale 7 → Scale 1)
        parent = self.root

        hardware_map = [
            (HardwareScale.SCALE_7_PHYSIO, 'timers', 1e-6),
            (HardwareScale.SCALE_6_SPECTRO, 'led', 1e-3),
            (HardwareScale.SCALE_5_VALIDATION, 'usb', 1e-3),
            (HardwareScale.SCALE_4_ENSEMBLE, 'network', 1.0),
            (HardwareScale.SCALE_3_CONFORM, 'disk', 1e2),
            (HardwareScale.SCALE_2_FRAGMENT, 'memory', 1e6),
            (HardwareScale.SCALE_1_QUANTUM, 'clock', 1e15),
        ]

        for scale, hw_source, typical_freq in hardware_map:
            hw_data = hardware_measurements.get(hw_source, {})
            freq = hw_data.get('frequency', typical_freq)

            node = FrequencyHierarchyNode(
                node_id=f"scale_{scale.value}_{hw_source}",
                hierarchical_level=scale,
                frequency_hz=freq,
                frequency_range=(freq * 0.9, freq * 1.1),
                hardware_source=hw_source,
                observation_window_start=0.0,
                observation_window_end=1.0 / freq  # One period
            )

            # Compute gear ratio
            gear_ratio = parent.frequency_hz / freq if freq > 0 else 1.0
            parent.add_child(node, gear_ratio)

            self.nodes_by_level[scale].append(node)
            parent = node

    def deploy_finite_observers(self, spectrum_data: Dict[str, Any]) -> Dict[HardwareScale, List[Dict[str, Any]]]:
        """
        Deploy finite observers at each hierarchical level.

        Each finite observer:
        - Observes exactly one level (one observation window)
        - Detects phase-locks within its window
        - Reports to transcendent observer

        Args:
            spectrum_data: Molecular data (mz, intensity, rt, fragments)

        Returns:
            Dict mapping scale → list of phase-lock signatures
        """
        measurements_by_scale = {}

        for scale in HardwareScale:
            nodes = self.nodes_by_level[scale]
            scale_measurements = []

            for node in nodes:
                # Finite observer at this node
                phase_locks = self._observe_phase_locks_at_node(node, spectrum_data)
                node.phase_lock_signatures = phase_locks
                scale_measurements.extend(phase_locks)

            measurements_by_scale[scale] = scale_measurements

        return measurements_by_scale

    def _observe_phase_locks_at_node(self, node: FrequencyHierarchyNode,
                                    spectrum_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Finite observer measures phase-locks within observation window.

        Looks for molecular oscillations phase-locked to hardware at this scale.
        """
        phase_locks = []

        # Extract molecular signals
        mz = spectrum_data.get('mz', np.array([]))
        intensity = spectrum_data.get('intensity', np.array([]))

        if len(mz) == 0:
            return phase_locks

        # Convert m/z to approximate frequencies (via vibrational modes)
        # Higher m/z → lower frequency (heavier molecules vibrate slower)
        molecular_frequencies = 1e13 / np.sqrt(mz)  # Rough approximation

        # Check which frequencies fall in this node's observation window
        in_window = (molecular_frequencies >= node.frequency_range[0]) & \
                   (molecular_frequencies <= node.frequency_range[1])

        for i in np.where(in_window)[0]:
            phase_lock = {
                'frequency': molecular_frequencies[i],
                'mz': mz[i],
                'intensity': intensity[i],
                'phase': np.random.uniform(0, 2*np.pi),  # Would measure from hardware
                'coherence': np.random.uniform(0.5, 1.0),  # Would measure from phase-lock quality
                'node_id': node.node_id,
                'scale': node.hierarchical_level.value
            }
            phase_locks.append(phase_lock)

        return phase_locks

    def identify_convergence_nodes(self, top_fraction: float = 0.1) -> List[FrequencyHierarchyNode]:
        """
        Identify convergence nodes (high phase-lock density).

        These are optimal sites for MMD materialization because multiple
        categorical paths intersect here.

        Args:
            top_fraction: Fraction of nodes to consider as convergence (e.g., top 10%)

        Returns:
            List of convergence nodes, sorted by score (highest first)
        """
        # Compute convergence scores for all nodes
        all_nodes = []
        for scale_nodes in self.nodes_by_level.values():
            all_nodes.extend(scale_nodes)

        for node in all_nodes:
            node.compute_convergence_score()

        # Sort by convergence score
        sorted_nodes = sorted(all_nodes, key=lambda n: n.convergence_score, reverse=True)

        # Take top fraction
        n_convergence = max(1, int(len(sorted_nodes) * top_fraction))
        convergence_nodes = sorted_nodes[:n_convergence]

        # Mark as convergence nodes
        for node in convergence_nodes:
            node.is_convergence_node = True

        self.convergence_nodes = convergence_nodes
        return convergence_nodes

    def navigate_via_gear_ratios(self, start_node: FrequencyHierarchyNode,
                                 target_scale: HardwareScale) -> Optional[FrequencyHierarchyNode]:
        """
        Navigate from start node to target scale via gear ratios.

        O(1) operation per level (not traversing nodes, using gear ratios).

        Key innovation: Gear ratios enable instant hierarchical jumps.
        """
        current = start_node

        # Navigate up or down?
        if target_scale.value > current.hierarchical_level.value:
            # Navigate up to coarser scale
            while current and current.hierarchical_level != target_scale:
                current = current.navigate_to_parent_via_gear_ratio()
        else:
            # Navigate down to finer scale
            # This is more complex - need to choose which child
            while current and current.hierarchical_level != target_scale:
                if not current.child_nodes:
                    break
                # For now, take first child (could be smarter)
                current = current.child_nodes[0] if current.child_nodes else None

        return current

    def get_phase_locked_ensemble(self, convergence_node: FrequencyHierarchyNode) -> List[Dict[str, Any]]:
        """
        Get phase-locked ensemble at convergence node.

        This is the data that MMD will read to materialize categorical state.
        """
        return convergence_node.phase_lock_signatures

    def get_statistics(self) -> Dict[str, Any]:
        """Get hierarchy statistics"""
        return {
            'total_nodes': sum(len(nodes) for nodes in self.nodes_by_level.values()),
            'nodes_per_scale': {
                scale.name: len(nodes) for scale, nodes in self.nodes_by_level.items()
            },
            'convergence_nodes': len(self.convergence_nodes),
            'convergence_scores': [n.convergence_score for n in self.convergence_nodes],
            'max_convergence_score': max([n.convergence_score for n in self.convergence_nodes]) if self.convergence_nodes else 0
        }


# Module exports
__all__ = [
    'HardwareScale',
    'FrequencyHierarchyNode',
    'FrequencyHierarchyTree'
]
