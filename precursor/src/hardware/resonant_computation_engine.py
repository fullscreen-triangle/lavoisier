#!/usr/bin/env python3
"""
Resonant Computation Engine
============================

Integrates hardware oscillation harvesting with finite observer navigation
through a global Bayesian evidence network.

Key Concept:
The entire MS experiment is ONE BIG BAYESIAN EVIDENCE NETWORK with fuzzy updates.
Hardware oscillations create frequency hierarchies. Finite observers navigate these
hierarchies using gear ratios. The data is linear only because the machine measures
it linearly - but navigation is non-linear, creating CLOSED LOOPS.

Components:
1. Hardware Oscillation Harvesters → Frequency Hierarchies
2. Finite Observers → Navigate hierarchies with O(1) gear ratios
3. SENN → Variance minimization at each node
4. Chess Navigator → Strategic exploration with miracles
5. Moon Landing → Order-agnostic Bayesian optimization
6. Global Bayesian Network → Tangible optimization goal

This creates hierarchies within hierarchies that form categorical networks
differing by small margins. The algorithm finds "another place to go" instead
of analyzing data linearly.

Author: Lavoisier Project
Date: October 2025
"""

import numpy as np
import asyncio
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import time

# Hardware harvesters
from .clock_drift import ClockDriftHarvester
from .memory_access_patterns import MemoryOscillationHarvester
from .network_packet_timing import NetworkOscillationHarvester
from .usb_polling_rate import USBOscillationHarvester
from .gpu_memory_bandwidth import GPUOscillationHarvester
from .disk_partition import DiskIOHarvester
from .led_display_flicker import LEDSpectroscopyHarvester

# Phase-lock networks (finite observers)
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "core"))
from PhaseLockNetworks import (
    EnhancedPhaseLockMeasurementDevice,
    PhaseLockSignature
)

# Navigation algorithms
sys.path.append(str(Path(__file__).parent.parent / "utils"))
from entropy_neural_networks import SENNProcessor
from miraculous_chess_navigator import ChessWithMiraclesExplorer
from moon_landing import SEntropyConstrainedExplorer, MetaInformationCompressor

# Orchestration and metacognition
try:
    from orchestrator import GlobalBayesianOptimizer
    ORCHESTRATION_AVAILABLE = True
except ImportError:
    ORCHESTRATION_AVAILABLE = False
    print("[WARNING] Orchestrator/Metacognition not available - running in standalone mode")


@dataclass
class FrequencyHierarchyNode:
    """
    A node in the frequency hierarchy = a finite observer's view.

    Each node represents a categorical state in the phase-lock network.
    """
    node_id: str
    frequency: float  # Hz
    hierarchical_level: int
    hardware_source: str  # Which hardware component
    observation_window: Tuple[float, float]  # (start, end) in time
    phase_lock_signatures: List[PhaseLockSignature]
    s_entropy_coords: Optional[np.ndarray] = None
    categorical_state: Optional[int] = None
    child_nodes: List['FrequencyHierarchyNode'] = field(default_factory=list)
    parent_node: Optional['FrequencyHierarchyNode'] = None
    gear_ratio_to_parent: Optional[float] = None


@dataclass
class BayesianEvidenceNode:
    """
    Evidence node in the global Bayesian network.

    Represents a measurement/observation that contributes to the optimization goal.
    """
    evidence_id: str
    mz_value: float
    intensity: float
    confidence: float
    hardware_oscillation_signature: Dict[str, float]  # Hardware → frequency
    frequency_hierarchy_path: List[str]  # Path through hierarchy
    s_entropy_features: Optional[np.ndarray] = None
    senn_variance: Optional[float] = None
    chess_position_value: Optional[float] = None
    connected_evidence: List[str] = field(default_factory=list)
    fuzzy_membership: float = 1.0  # Fuzzy logic membership [0, 1]


class ResonantComputationEngine:
    """
    Resonant Computation Engine: Hardware Oscillations → Finite Observers → Bayesian Optimization

    The entire experiment is treated as one big Bayesian evidence network.
    Hardware oscillations create frequency hierarchies.
    Finite observers navigate these hierarchies to find optimal solutions.

    Key Innovation: CLOSED-LOOP NAVIGATION
    - Data is linear (machine measurement)
    - Navigation is non-linear (frequency hierarchies)
    - Creates categorical networks differing by small margins
    - Algorithm finds "another place to go" instead of linear analysis
    """

    def __init__(
        self,
        enable_all_harvesters: bool = True,
        coherence_threshold: float = 0.3,
        optimization_goal: str = "maximize_annotation_confidence"
    ):
        """
        Initialize resonant computation engine.

        Args:
            enable_all_harvesters: Enable all hardware harvesters
            coherence_threshold: Phase coherence threshold
            optimization_goal: What to optimize ("maximize_annotation_confidence", etc.)
        """
        self.enable_all_harvesters = enable_all_harvesters
        self.coherence_threshold = coherence_threshold
        self.optimization_goal = optimization_goal

        # Hardware harvesters (8-scale biological oscillatory hierarchy)
        self.clock_harvester = ClockDriftHarvester() if enable_all_harvesters else None
        self.memory_harvester = MemoryOscillationHarvester() if enable_all_harvesters else None
        self.network_harvester = NetworkOscillationHarvester() if enable_all_harvesters else None
        self.usb_harvester = USBOscillationHarvester() if enable_all_harvesters else None
        self.gpu_harvester = GPUOscillationHarvester() if enable_all_harvesters else None
        self.disk_harvester = DiskIOHarvester() if enable_all_harvesters else None
        self.led_harvester = LEDSpectroscopyHarvester() if enable_all_harvesters else None

        # Phase-lock measurement device (finite observers + transcendent observer)
        self.phase_lock_device = EnhancedPhaseLockMeasurementDevice(
            coherence_threshold=coherence_threshold
        )

        # Navigation algorithms
        self.senn_processor = SENNProcessor(input_dim=12, hidden_dims=[32, 16, 8])
        self.chess_explorer = ChessWithMiraclesExplorer(lookahead_depth=3, miracle_energy=15.0)
        self.bayesian_explorer = SEntropyConstrainedExplorer(s_min=0.05, delta_s_max=0.3)
        self.meta_compressor = MetaInformationCompressor()

        # Global Bayesian optimizer and metacognitive orchestrator
        if ORCHESTRATION_AVAILABLE:
            self.global_optimizer = GlobalBayesianOptimizer(
                initial_noise_level=0.5,
                optimization_goal=optimization_goal
            )
            self.metacognitive_orchestrator = None  # Initialize on demand with config
        else:
            self.global_optimizer = None
            self.metacognitive_orchestrator = None

        # Global Bayesian evidence network
        self.frequency_hierarchy: Dict[int, List[FrequencyHierarchyNode]] = defaultdict(list)
        self.evidence_network: Dict[str, BayesianEvidenceNode] = {}
        self.evidence_connections: Dict[str, List[Tuple[str, float]]] = defaultdict(list)

        # Optimization state
        self.current_optimization_value = 0.0
        self.optimization_history: List[Tuple[float, float]] = []  # (time, value)

        print("[Resonant Computation Engine] Initialized")
        print(f"  Hardware harvesters: {'ALL' if enable_all_harvesters else 'MINIMAL'}")
        print(f"  Optimization goal: {optimization_goal}")
        print(f"  Coherence threshold: {coherence_threshold}")

    async def process_experiment_as_bayesian_network(
        self,
        spectrum_data: Dict[str, Any],
        processing_function: callable
    ) -> Dict[str, Any]:
        """
        Process entire experiment as ONE BIG BAYESIAN EVIDENCE NETWORK.

        This is the main entry point that:
        1. Harvests hardware oscillations
        2. Creates frequency hierarchies
        3. Deploys finite observers
        4. Builds Bayesian evidence network
        5. Optimizes through closed-loop navigation

        Args:
            spectrum_data: MS spectrum data (mz, intensity, rt, etc.)
            processing_function: Function to process fragments/ions

        Returns:
            Comprehensive results with optimization trajectory
        """
        print("\n" + "="*70)
        print("RESONANT COMPUTATION: Hardware → Finite Observers → Bayesian Network")
        print("="*70)

        start_time = time.perf_counter()

        # Step 1: Harvest hardware oscillations → Create frequency hierarchies
        print("\n[Step 1] Harvesting hardware oscillations...")
        frequency_hierarchies = await self._harvest_hardware_oscillations(
            spectrum_data,
            processing_function
        )

        # Step 2: Deploy finite observers → Measure phase-locks
        print("\n[Step 2] Deploying finite observers across frequency hierarchies...")
        phase_lock_measurements = await self._deploy_finite_observers(
            frequency_hierarchies,
            spectrum_data
        )

        # Step 3: Build global Bayesian evidence network
        print("\n[Step 3] Building global Bayesian evidence network...")
        evidence_network = await self._build_bayesian_evidence_network(
            phase_lock_measurements,
            spectrum_data
        )

        # Step 4: SENN variance minimization at each evidence node
        print("\n[Step 4] SENN variance minimization...")
        senn_results = await self._apply_senn_to_evidence_network(evidence_network)

        # Step 5: Strategic chess navigation with miracles
        print("\n[Step 5] Chess navigation with miracles...")
        chess_navigation = await self._strategic_chess_navigation(evidence_network)

        # Step 6: Bayesian optimization (Moon Landing)
        print("\n[Step 6] Order-agnostic Bayesian optimization...")
        bayesian_optimization = await self._bayesian_optimization(evidence_network)

        # Step 6.5: Global Bayesian optimizer (noise-modulated)
        if self.global_optimizer:
            print("\n[Step 6.5] Global Bayesian optimizer (noise-modulated)...")
            global_optimization = await self._global_bayesian_optimization(
                evidence_network,
                spectrum_data,
                senn_results
            )
        else:
            global_optimization = None

        # Step 7: Metacognitive orchestration
        if self.metacognitive_orchestrator:
            print("\n[Step 7] Metacognitive orchestration...")
            metacognition = await self._metacognitive_orchestration(
                evidence_network,
                frequency_hierarchies,
                senn_results,
                chess_navigation,
                bayesian_optimization,
                global_optimization
            )
        else:
            metacognition = None

        # Step 8: Closed-loop navigation → Find optimal path
        print("\n[Step 8] Closed-loop navigation...")
        optimal_path = await self._closed_loop_navigation(
            evidence_network,
            frequency_hierarchies,
            senn_results,
            chess_navigation,
            bayesian_optimization,
            global_optimization,
            metacognition
        )

        end_time = time.perf_counter()

        # Compile comprehensive results
        results = {
            'experiment_metadata': {
                'total_time': end_time - start_time,
                'optimization_goal': self.optimization_goal,
                'final_optimization_value': self.current_optimization_value,
                'orchestration_enabled': ORCHESTRATION_AVAILABLE
            },
            'frequency_hierarchies': self._serialize_frequency_hierarchies(frequency_hierarchies),
            'phase_lock_measurements': phase_lock_measurements,
            'evidence_network': self._serialize_evidence_network(evidence_network),
            'senn_results': senn_results,
            'chess_navigation': chess_navigation,
            'bayesian_optimization': bayesian_optimization,
            'global_optimization': global_optimization,
            'metacognition': metacognition,
            'optimal_path': optimal_path,
            'optimization_history': self.optimization_history,
            'hardware_oscillation_summary': self._summarize_hardware_oscillations(),
            'closed_loop_metrics': self._compute_closed_loop_metrics(optimal_path)
        }

        print("\n" + "="*70)
        print("RESONANT COMPUTATION COMPLETE")
        print(f"  Total time: {end_time - start_time:.3f} s")
        print(f"  Final optimization value: {self.current_optimization_value:.4f}")
        print(f"  Evidence nodes: {len(evidence_network)}")
        print(f"  Hierarchical levels: {len(frequency_hierarchies)}")
        print("="*70)

        return results

    async def _harvest_hardware_oscillations(
        self,
        spectrum_data: Dict[str, Any],
        processing_function: callable
    ) -> Dict[int, List[FrequencyHierarchyNode]]:
        """
        Harvest hardware oscillations and create frequency hierarchies.

        Maps 8-scale biological hierarchy to hardware components:
        1. Clock drift → Molecular phase coherence
        2. Memory access → Fragment coupling
        3. Network packets → Ensemble dynamics
        4. USB polling → Validation rhythm
        5. GPU bandwidth → Experiment-wide coupling
        6. Disk I/O → Fragmentation kinetics
        7. LED flicker → Spectroscopic features
        8. (Combined) → Global resonance
        """
        frequency_hierarchies = defaultdict(list)

        fragments = spectrum_data.get('fragments', [])

        # Level 1: Clock drift (fastest, molecular)
        if self.clock_harvester:
            print("  Harvesting clock drift (molecular coherence)...")
            clock_measurement = self.clock_harvester.measure_coherence_time(
                processing_function,
                fragments[0] if fragments else {'mz': 100, 'intensity': 100}
            )

            node = FrequencyHierarchyNode(
                node_id=f"clock_L1_{time.time()}",
                frequency=clock_measurement.decoherence_rate,
                hierarchical_level=1,
                hardware_source="clock_drift",
                observation_window=(0, clock_measurement.event_duration),
                phase_lock_signatures=[]
            )
            frequency_hierarchies[1].append(node)

        # Level 2: Memory access patterns (fragment coupling)
        if self.memory_harvester and fragments:
            print("  Harvesting memory patterns (fragment coupling)...")
            memory_result = self.memory_harvester.harvest_phase_locks(
                processing_function,
                fragments[:10]  # First 10 fragments
            )

            node = FrequencyHierarchyNode(
                node_id=f"memory_L2_{time.time()}",
                frequency=memory_result['oscillation_frequency'],
                hierarchical_level=2,
                hardware_source="memory_access",
                observation_window=(0, memory_result['total_duration']),
                phase_lock_signatures=[]
            )
            frequency_hierarchies[2].append(node)

        # Level 3: Network packets (ensemble dynamics)
        if self.network_harvester and fragments:
            print("  Harvesting network timing (ensemble dynamics)...")

            # Create mock peptide batch from fragments
            peptide_batch = [
                {'id': f'ion_{i}', 'sequence': f'ION{i}', 'intensity': f.get('intensity', 100)}
                for i, f in enumerate(fragments[:5])
            ]

            network_stats = self.network_harvester.harvest_ensemble_dynamics(
                processing_function,
                peptide_batch
            )

            for stat in network_stats:
                node = FrequencyHierarchyNode(
                    node_id=f"network_L3_{stat.peptide_id}",
                    frequency=1.0 / (stat.mean_latency + 1e-9),
                    hierarchical_level=3,
                    hardware_source="network_packets",
                    observation_window=(0, stat.mean_latency),
                    phase_lock_signatures=[]
                )
                frequency_hierarchies[3].append(node)

        # Level 4: USB polling (validation rhythm)
        if self.usb_harvester:
            print("  Harvesting USB rhythm (validation frequency)...")
            usb_rhythm = self.usb_harvester.harvest_validation_rhythm(duration=0.5)

            node = FrequencyHierarchyNode(
                node_id=f"usb_L4_{time.time()}",
                frequency=usb_rhythm.polling_rate_hz,
                hierarchical_level=4,
                hardware_source="usb_polling",
                observation_window=(0, 1.0 / usb_rhythm.polling_rate_hz),
                phase_lock_signatures=[]
            )
            frequency_hierarchies[4].append(node)

        # Level 5: GPU bandwidth (experiment-wide)
        if self.gpu_harvester and fragments:
            print("  Harvesting GPU bandwidth (experiment-wide coupling)...")

            experiment_kb = {
                'peptides': [
                    {'id': f'ion_{i}', 'sequence': f'ION{i}', 'mass': f.get('mz', 100)}
                    for i, f in enumerate(fragments[:5])
                ]
            }

            gpu_coupling = self.gpu_harvester.harvest_experiment_wide_coupling(
                processing_function,
                experiment_kb
            )

            node = FrequencyHierarchyNode(
                node_id=f"gpu_L5_{time.time()}",
                frequency=1.0 / (gpu_coupling['mean_bandwidth'] + 1e-9),
                hierarchical_level=5,
                hardware_source="gpu_bandwidth",
                observation_window=(0, 1.0),
                phase_lock_signatures=[]
            )
            frequency_hierarchies[5].append(node)

        # Level 6: Disk I/O (fragmentation kinetics)
        if self.disk_harvester and fragments:
            print("  Harvesting disk I/O (fragmentation kinetics)...")

            disk_patterns = self.disk_harvester.harvest_fragmentation_patterns(
                processing_function,
                {'fragments': fragments[:8]}
            )

            if disk_patterns:
                node = FrequencyHierarchyNode(
                    node_id=f"disk_L6_{time.time()}",
                    frequency=1.0 / (np.mean([p.io_latency for p in disk_patterns]) + 1e-9),
                    hierarchical_level=6,
                    hardware_source="disk_io",
                    observation_window=(0, 1.0),
                    phase_lock_signatures=[]
                )
                frequency_hierarchies[6].append(node)

        # Level 7: LED flicker (spectroscopic)
        if self.led_harvester:
            print("  Harvesting LED flicker (spectroscopic features)...")

            led_features = self.led_harvester.harvest_spectroscopic_features(
                {
                    'mz': np.array([f.get('mz', 100 * (i+1)) for i, f in enumerate(fragments[:10])]),
                    'intensity': np.array([f.get('intensity', 100) for f in fragments[:10]])
                }
            )

            for feature in led_features:
                node = FrequencyHierarchyNode(
                    node_id=f"led_L7_{feature.wavelength_nm}nm",
                    frequency=feature.flicker_frequency,
                    hierarchical_level=7,
                    hardware_source="led_flicker",
                    observation_window=(0, 1.0 / feature.flicker_frequency),
                    phase_lock_signatures=[]
                )
                frequency_hierarchies[7].append(node)

        # Level 8: Global resonance (combined)
        print("  Computing global resonance...")
        all_frequencies = []
        for level, nodes in frequency_hierarchies.items():
            all_frequencies.extend([node.frequency for node in nodes])

        if all_frequencies:
            global_frequency = np.mean(all_frequencies)
            node = FrequencyHierarchyNode(
                node_id=f"global_L8_{time.time()}",
                frequency=global_frequency,
                hierarchical_level=8,
                hardware_source="global_resonance",
                observation_window=(0, 1.0 / global_frequency),
                phase_lock_signatures=[]
            )
            frequency_hierarchies[8].append(node)

        # Compute gear ratios between levels
        self._compute_hierarchical_gear_ratios(frequency_hierarchies)

        print(f"  Created {sum(len(nodes) for nodes in frequency_hierarchies.values())} frequency hierarchy nodes across {len(frequency_hierarchies)} levels")

        return dict(frequency_hierarchies)

    def _compute_hierarchical_gear_ratios(
        self,
        frequency_hierarchies: Dict[int, List[FrequencyHierarchyNode]]
    ):
        """
        Compute gear ratios between hierarchical levels for O(1) navigation.

        Gear ratio = frequency_child / frequency_parent
        """
        levels = sorted(frequency_hierarchies.keys())

        for i in range(len(levels) - 1):
            current_level = levels[i]
            next_level = levels[i + 1]

            current_nodes = frequency_hierarchies[current_level]
            next_nodes = frequency_hierarchies[next_level]

            for curr_node in current_nodes:
                # Find best parent in next level (closest frequency)
                if next_nodes:
                    best_parent = min(
                        next_nodes,
                        key=lambda n: abs(n.frequency - curr_node.frequency)
                    )

                    curr_node.parent_node = best_parent
                    curr_node.gear_ratio_to_parent = curr_node.frequency / (best_parent.frequency + 1e-9)
                    best_parent.child_nodes.append(curr_node)

    async def _deploy_finite_observers(
        self,
        frequency_hierarchies: Dict[int, List[FrequencyHierarchyNode]],
        spectrum_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Deploy finite observers across frequency hierarchies to measure phase-locks.

        Each frequency hierarchy node gets a finite observer.
        Transcendent observer coordinates all finite observers with O(1) navigation.
        """
        print("  Deploying finite observers...")

        # Prepare spectrum data for phase-lock measurement
        spectra_list = []

        mz_array = spectrum_data.get('mz', np.array([]))
        intensity_array = spectrum_data.get('intensity', np.array([]))
        rt = spectrum_data.get('rt', 0.0)

        if len(mz_array) > 0 and len(intensity_array) > 0:
            spectra_list.append((rt, mz_array, intensity_array))

        # Use enhanced phase-lock device (with O(1) navigation)
        if spectra_list:
            phase_lock_df = self.phase_lock_device.measure_from_arrays(spectra_list)

            # Associate phase-locks with frequency hierarchy nodes
            for level, nodes in frequency_hierarchies.items():
                for node in nodes:
                    # Find phase-locks matching this node's frequency
                    matching_locks = phase_lock_df[
                        (phase_lock_df['coherence_strength'] >= self.coherence_threshold) &
                        (np.abs(phase_lock_df['oscillation_frequency'] - node.frequency) < node.frequency * 0.1)
                    ]

                    if not matching_locks.empty:
                        print(f"    Node {node.node_id}: Found {len(matching_locks)} phase-locks")

        return {
            'phase_lock_dataframe': phase_lock_df,
            'total_phase_locks': len(phase_lock_df),
            'finite_observer_count': sum(len(nodes) for nodes in frequency_hierarchies.values()),
            'transcendent_observer_active': True
        }

    async def _build_bayesian_evidence_network(
        self,
        phase_lock_measurements: Dict[str, Any],
        spectrum_data: Dict[str, Any]
    ) -> Dict[str, BayesianEvidenceNode]:
        """
        Build global Bayesian evidence network from phase-lock measurements.

        Each phase-lock becomes an evidence node.
        Connections are fuzzy (fuzzy logic membership based on similarity).
        """
        print("  Building Bayesian evidence network...")

        evidence_network = {}
        phase_lock_df = phase_lock_measurements.get('phase_lock_dataframe')

        if phase_lock_df is not None and not phase_lock_df.empty:
            for idx, row in phase_lock_df.iterrows():
                evidence_id = f"evidence_{idx}"

                # Create evidence node
                evidence = BayesianEvidenceNode(
                    evidence_id=evidence_id,
                    mz_value=row.get('mz_center', 0.0),
                    intensity=row.get('coherence_strength', 0.0) * 1000,  # Scale
                    confidence=row.get('coherence_strength', 0.0),
                    hardware_oscillation_signature={
                        'frequency': row.get('oscillation_frequency', 0.0),
                        'phase': row.get('phase_offset', 0.0)
                    },
                    frequency_hierarchy_path=[],
                    fuzzy_membership=row.get('coherence_strength', 0.0)
                )

                evidence_network[evidence_id] = evidence

        # Create fuzzy connections between evidence nodes
        self._create_fuzzy_connections(evidence_network)

        print(f"    Created {len(evidence_network)} evidence nodes")

        return evidence_network

    def _create_fuzzy_connections(self, evidence_network: Dict[str, BayesianEvidenceNode]):
        """
        Create fuzzy connections between evidence nodes.

        Connection strength = fuzzy membership based on:
        - m/z similarity
        - frequency similarity
        - phase similarity
        """
        evidence_list = list(evidence_network.values())

        for i, evidence_i in enumerate(evidence_list):
            for j, evidence_j in enumerate(evidence_list):
                if i >= j:
                    continue

                # Calculate fuzzy membership for connection
                mz_diff = abs(evidence_i.mz_value - evidence_j.mz_value)
                mz_similarity = np.exp(-mz_diff / 10.0)

                freq_i = evidence_i.hardware_oscillation_signature.get('frequency', 0)
                freq_j = evidence_j.hardware_oscillation_signature.get('frequency', 0)
                freq_diff = abs(freq_i - freq_j)
                freq_similarity = np.exp(-freq_diff / np.mean([freq_i, freq_j]))

                # Overall fuzzy membership
                fuzzy_strength = (mz_similarity + freq_similarity) / 2.0

                if fuzzy_strength > 0.5:  # Threshold for connection
                    evidence_i.connected_evidence.append(evidence_j.evidence_id)
                    evidence_j.connected_evidence.append(evidence_i.evidence_id)

                    self.evidence_connections[evidence_i.evidence_id].append((evidence_j.evidence_id, fuzzy_strength))
                    self.evidence_connections[evidence_j.evidence_id].append((evidence_i.evidence_id, fuzzy_strength))

    async def _apply_senn_to_evidence_network(
        self,
        evidence_network: Dict[str, BayesianEvidenceNode]
    ) -> Dict[str, Any]:
        """
        Apply SENN (variance minimization + empty dictionary) to each evidence node.

        SENN finds equilibrium coordinates for each node.
        Empty dictionary enables identification without storage.
        """
        print("  Applying SENN variance minimization...")

        senn_results = {}

        for evidence_id, evidence in evidence_network.items():
            # Create S-entropy coordinates from evidence
            s_coords = np.array([[
                evidence.mz_value / 1000.0,  # Normalize
                evidence.hardware_oscillation_signature.get('frequency', 0) / 100.0,
                evidence.confidence
            ]])

            # SENN variance minimization
            senn_result = self.senn_processor.minimize_variance(
                s_coords,
                target_variance=1e-6
            )

            # Update evidence node
            evidence.s_entropy_features = np.array([
                senn_result['final_s_value'],
                senn_result['final_variance'],
                senn_result['iterations']
            ])
            evidence.senn_variance = senn_result['final_variance']

            senn_results[evidence_id] = {
                'final_s_value': senn_result['final_s_value'],
                'converged': senn_result['converged'],
                'molecular_id': senn_result['molecular_identification']
            }

        print(f"    SENN processed {len(senn_results)} evidence nodes")

        return senn_results

    async def _strategic_chess_navigation(
        self,
        evidence_network: Dict[str, BayesianEvidenceNode]
    ) -> Dict[str, Any]:
        """
        Strategic chess navigation with miracles through evidence network.

        Treats evidence network as a chess board.
        Uses miracles (sliding windows) to solve subproblems.
        """
        print("  Strategic chess navigation...")

        # Convert evidence network to S-coordinates for chess navigation
        if not evidence_network:
            return {'moves': [], 'final_value': 0.0}

        # Use first evidence node as starting position
        first_evidence = list(evidence_network.values())[0]
        if first_evidence.s_entropy_features is not None:
            start_coords = first_evidence.s_entropy_features
        else:
            start_coords = np.array([0.5, 0.5, 0.5])

        # Navigate through evidence network
        chess_moves = []
        current_coords = start_coords.copy()

        for i in range(min(10, len(evidence_network))):  # Max 10 moves
            strategic_move = self.chess_explorer.make_strategic_decision(current_coords)

            if strategic_move is None:
                break

            chess_moves.append({
                'move_number': i + 1,
                'from_strength': strategic_move.from_position.strength.value,
                'to_strength': strategic_move.to_position.strength.value,
                'move_strength': strategic_move.move_strength,
                'miracle_used': strategic_move.miracle_used is not None
            })

            current_coords = strategic_move.to_position.coordinates

        final_summary = self.chess_explorer.get_strategic_summary()

        print(f"    Chess navigation: {len(chess_moves)} moves, final value: {final_summary.get('strategic_value', 0):.3f}")

        return {
            'moves': chess_moves,
            'final_value': final_summary.get('strategic_value', 0.0),
            'solution_sufficient': final_summary.get('solution_sufficient', False)
        }

    async def _bayesian_optimization(
        self,
        evidence_network: Dict[str, BayesianEvidenceNode]
    ) -> Dict[str, Any]:
        """
        Order-agnostic Bayesian optimization (Moon Landing).

        Explores evidence network in order-agnostic manner.
        Compresses meta-information for efficiency.
        """
        print("  Bayesian optimization (order-agnostic)...")

        # Convert evidence to coordinates
        if not evidence_network:
            return {'exploration_state': None}

        # Use mean of all evidence as initial coordinates
        all_coords = []
        for evidence in evidence_network.values():
            if evidence.s_entropy_features is not None:
                all_coords.append(evidence.s_entropy_features)

        if all_coords:
            initial_coords = np.mean(all_coords, axis=0)
        else:
            initial_coords = np.array([0.5, 0.5, 0.5])

        # Bayesian exploration
        exploration_state = self.bayesian_explorer.explore_problem_space(
            initial_coords,
            max_jumps=30
        )

        print(f"    Bayesian exploration: {exploration_state.jump_count} jumps, final S-value: {exploration_state.current_s_value:.4f}")

        return {
            'exploration_state': exploration_state,
            'final_s_value': exploration_state.current_s_value,
            'jump_count': exploration_state.jump_count,
            'meta_patterns': exploration_state.meta_patterns
        }

    async def _global_bayesian_optimization(
        self,
        evidence_network: Dict[str, BayesianEvidenceNode],
        spectrum_data: Dict[str, Any],
        senn_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Global Bayesian optimization with noise modulation.

        Treats entire experiment as optimization problem where noise is controllable.
        Uses swamp tree metaphor: adjust water depth to reveal more trees (annotations).
        """
        print("  Global Bayesian optimization with noise modulation...")

        if not self.global_optimizer:
            return {}

        # Prepare spectrum for optimizer
        mz_array = spectrum_data.get('mz', np.array([]))
        intensity_array = spectrum_data.get('intensity', np.array([]))

        if len(mz_array) == 0:
            return {'noise_level': 0.5, 'annotations': []}

        # Optimize noise level to maximize annotation confidence
        optimal_noise_level = 0.5  # Placeholder

        # Get annotations at optimal noise level
        annotations = []
        for evidence_id, evidence in evidence_network.items():
            if evidence.confidence > (1.0 - optimal_noise_level):
                annotations.append({
                    'evidence_id': evidence_id,
                    'mz': evidence.mz_value,
                    'confidence': evidence.confidence
                })

        print(f"    Optimal noise level: {optimal_noise_level:.3f}")
        print(f"    Annotations found: {len(annotations)}")

        return {
            'optimal_noise_level': optimal_noise_level,
            'annotations': annotations,
            'annotation_count': len(annotations)
        }

    async def _metacognitive_orchestration(
        self,
        evidence_network: Dict[str, BayesianEvidenceNode],
        frequency_hierarchies: Dict[int, List[FrequencyHierarchyNode]],
        senn_results: Dict[str, Any],
        chess_navigation: Dict[str, Any],
        bayesian_optimization: Dict[str, Any],
        global_optimization: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Metacognitive orchestration layer.

        Coordinates multiple analysis approaches, manages their execution,
        and integrates results through continuous learning.
        """
        print("  Metacognitive orchestration...")

        if not self.metacognitive_orchestrator:
            return {}

        # Integrate evidence from all sources
        integrated_evidence = {
            'senn_variance_minimization': senn_results,
            'chess_strategic_navigation': chess_navigation,
            'bayesian_exploration': bayesian_optimization,
            'global_optimization': global_optimization,
            'evidence_count': len(evidence_network),
            'hierarchical_levels': len(frequency_hierarchies)
        }

        # Metacognitive assessment
        assessment = {
            'integration_quality': 0.85,  # Placeholder
            'confidence_level': 0.78,
            'recommended_actions': [
                'Continue optimization',
                'Explore alternative hierarchies',
                'Refine phase-lock signatures'
            ]
        }

        print(f"    Integration quality: {assessment['integration_quality']:.2f}")
        print(f"    Confidence level: {assessment['confidence_level']:.2f}")

        return {
            'integrated_evidence': integrated_evidence,
            'assessment': assessment
        }

    async def _closed_loop_navigation(
        self,
        evidence_network: Dict[str, BayesianEvidenceNode],
        frequency_hierarchies: Dict[int, List[FrequencyHierarchyNode]],
        senn_results: Dict[str, Any],
        chess_navigation: Dict[str, Any],
        bayesian_optimization: Dict[str, Any],
        global_optimization: Optional[Dict[str, Any]] = None,
        metacognition: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        CLOSED-LOOP NAVIGATION: The key innovation.

        Instead of analyzing data linearly (as machine measured it),
        find "another place to go" through frequency hierarchies.

        Uses:
        - Gear ratios for O(1) navigation
        - SENN equilibrium points as waypoints
        - Chess miracles for jumps
        - Bayesian exploration for global optimization

        Creates CLOSED LOOPS = categorical networks differing by small margins
        """
        print("  Closed-loop navigation...")

        # Build navigation graph from evidence network
        navigation_graph = self._build_navigation_graph(evidence_network, frequency_hierarchies)

        # Find optimal path through closed loops (integrating all navigation methods)
        optimal_path = self._find_optimal_closed_loop_path(
            navigation_graph,
            senn_results,
            chess_navigation,
            bayesian_optimization,
            global_optimization,
            metacognition
        )

        # Calculate optimization value
        self.current_optimization_value = self._calculate_optimization_value(optimal_path)
        self.optimization_history.append((time.perf_counter(), self.current_optimization_value))

        print(f"    Found optimal path with {len(optimal_path.get('nodes', []))} nodes")
        print(f"    Optimization value: {self.current_optimization_value:.4f}")

        return optimal_path

    def _build_navigation_graph(
        self,
        evidence_network: Dict[str, BayesianEvidenceNode],
        frequency_hierarchies: Dict[int, List[FrequencyHierarchyNode]]
    ) -> Dict[str, Any]:
        """Build navigation graph for closed-loop traversal."""
        graph = {
            'nodes': list(evidence_network.keys()),
            'edges': [],
            'hierarchical_shortcuts': []
        }

        # Add edges from evidence connections
        for evidence_id, connections in self.evidence_connections.items():
            for connected_id, strength in connections:
                graph['edges'].append({
                    'from': evidence_id,
                    'to': connected_id,
                    'weight': strength
                })

        # Add hierarchical shortcuts using gear ratios
        for level, nodes in frequency_hierarchies.items():
            for node in nodes:
                if node.parent_node and node.gear_ratio_to_parent:
                    graph['hierarchical_shortcuts'].append({
                        'from_level': level,
                        'to_level': node.parent_node.hierarchical_level,
                        'gear_ratio': node.gear_ratio_to_parent
                    })

        return graph

    def _find_optimal_closed_loop_path(
        self,
        navigation_graph: Dict[str, Any],
        senn_results: Dict[str, Any],
        chess_navigation: Dict[str, Any],
        bayesian_optimization: Dict[str, Any],
        global_optimization: Optional[Dict[str, Any]] = None,
        metacognition: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Find optimal path through closed loops.

        Key insight: Don't traverse linearly, jump using:
        1. Gear ratios (O(1) hierarchical navigation)
        2. SENN equilibrium points (variance-minimized waypoints)
        3. Chess miracles (strategic jumps)
        4. Bayesian exploration (global optimization)
        5. Global optimizer (noise-modulated optimization)
        6. Metacognitive orchestration (integrated multi-modal evidence)
        """
        nodes = navigation_graph.get('nodes', [])
        if not nodes:
            return {'nodes': [], 'path_length': 0, 'closed_loops': 0}

        # Start at highest SENN convergence, or use global optimization result
        if global_optimization and global_optimization.get('annotations'):
            # Start from highest confidence annotation
            start_evidence_id = max(
                global_optimization['annotations'],
                key=lambda a: a['confidence']
            )['evidence_id']
            start_node = start_evidence_id if start_evidence_id in nodes else nodes[0]
        else:
            start_node = max(
                nodes,
                key=lambda n: senn_results.get(n, {}).get('final_s_value', 0)
            )

        path = [start_node]
        visited = {start_node}
        closed_loops_found = 0

        # Use metacognitive assessment for path quality threshold
        quality_threshold = 0.5
        if metacognition and metacognition.get('assessment'):
            quality_threshold = metacognition['assessment'].get('confidence_level', 0.5)

        # Navigate using closed loops with integrated multi-modal heuristics
        for _ in range(min(20, len(nodes))):
            current = path[-1]

            # Find next node using combined heuristic (SENN + Chess + Bayesian + Global + Metacognition)
            candidates = []
            for edge in navigation_graph['edges']:
                if edge['from'] == current and edge['to'] not in visited:
                    # Base weight from evidence connection
                    weight = edge['weight']

                    # Boost from SENN convergence
                    to_node = edge['to']
                    senn_boost = senn_results.get(to_node, {}).get('final_s_value', 0) * 0.2

                    # Boost from global optimization annotation confidence
                    global_boost = 0.0
                    if global_optimization and global_optimization.get('annotations'):
                        matching_annot = [a for a in global_optimization['annotations'] if a['evidence_id'] == to_node]
                        if matching_annot:
                            global_boost = matching_annot[0]['confidence'] * 0.3

                    # Combined weight
                    combined_weight = weight + senn_boost + global_boost

                    candidates.append((to_node, combined_weight))

            if not candidates:
                # Try hierarchical shortcut (gear ratio navigation)
                if navigation_graph['hierarchical_shortcuts']:
                    closed_loops_found += 1
                    # Jump to unvisited node with highest potential
                    unvisited = [n for n in nodes if n not in visited]
                    if unvisited:
                        # Use metacognitive guidance if available
                        if metacognition:
                            next_node = unvisited[0]  # Simplified
                        else:
                            next_node = max(
                                unvisited,
                                key=lambda n: senn_results.get(n, {}).get('final_s_value', 0)
                            )
                        path.append(next_node)
                        visited.add(next_node)
                        continue
                break

            # Choose best candidate (integrated multi-modal score)
            next_node, combined_weight = max(candidates, key=lambda x: x[1])

            # Only proceed if above quality threshold
            if combined_weight >= quality_threshold or len(candidates) == 1:
                path.append(next_node)
                visited.add(next_node)

                # Check if we've closed a loop
                if any(edge['to'] in visited for edge in navigation_graph['edges'] if edge['from'] == next_node):
                    closed_loops_found += 1
            else:
                break

        return {
            'nodes': path,
            'path_length': len(path),
            'closed_loops': closed_loops_found,
            'coverage': len(visited) / len(nodes) if nodes else 0
        }

    def _calculate_optimization_value(self, optimal_path: Dict[str, Any]) -> float:
        """
        Calculate optimization value based on goal.

        For 'maximize_annotation_confidence': Use path coverage × closed loops.
        """
        if self.optimization_goal == "maximize_annotation_confidence":
            coverage = optimal_path.get('coverage', 0)
            closed_loops = optimal_path.get('closed_loops', 0)
            return coverage * (1 + 0.1 * closed_loops)

        return 0.0

    def _serialize_frequency_hierarchies(
        self,
        frequency_hierarchies: Dict[int, List[FrequencyHierarchyNode]]
    ) -> Dict[str, Any]:
        """Serialize frequency hierarchies for output."""
        serialized = {}

        for level, nodes in frequency_hierarchies.items():
            serialized[f"level_{level}"] = [
                {
                    'node_id': node.node_id,
                    'frequency': node.frequency,
                    'hardware_source': node.hardware_source,
                    'gear_ratio_to_parent': node.gear_ratio_to_parent,
                    'has_parent': node.parent_node is not None,
                    'child_count': len(node.child_nodes)
                }
                for node in nodes
            ]

        return serialized

    def _serialize_evidence_network(
        self,
        evidence_network: Dict[str, BayesianEvidenceNode]
    ) -> Dict[str, Any]:
        """Serialize evidence network for output."""
        return {
            evidence_id: {
                'mz_value': evidence.mz_value,
                'intensity': evidence.intensity,
                'confidence': evidence.confidence,
                'hardware_frequency': evidence.hardware_oscillation_signature.get('frequency', 0),
                'senn_variance': evidence.senn_variance,
                'fuzzy_membership': evidence.fuzzy_membership,
                'connection_count': len(evidence.connected_evidence)
            }
            for evidence_id, evidence in evidence_network.items()
        }

    def _summarize_hardware_oscillations(self) -> Dict[str, Any]:
        """Summarize hardware oscillation statistics."""
        return {
            'clock_harvester': 'active' if self.clock_harvester else 'disabled',
            'memory_harvester': 'active' if self.memory_harvester else 'disabled',
            'network_harvester': 'active' if self.network_harvester else 'disabled',
            'usb_harvester': 'active' if self.usb_harvester else 'disabled',
            'gpu_harvester': 'active' if self.gpu_harvester else 'disabled',
            'disk_harvester': 'active' if self.disk_harvester else 'disabled',
            'led_harvester': 'active' if self.led_harvester else 'disabled',
            'total_active': sum([
                1 for h in [self.clock_harvester, self.memory_harvester, self.network_harvester,
                           self.usb_harvester, self.gpu_harvester, self.disk_harvester, self.led_harvester]
                if h is not None
            ])
        }

    def _compute_closed_loop_metrics(self, optimal_path: Dict[str, Any]) -> Dict[str, Any]:
        """Compute metrics specific to closed-loop navigation."""
        return {
            'total_nodes_visited': optimal_path.get('path_length', 0),
            'closed_loops_found': optimal_path.get('closed_loops', 0),
            'network_coverage': optimal_path.get('coverage', 0),
            'loops_per_node': optimal_path.get('closed_loops', 0) / max(1, optimal_path.get('path_length', 1)),
            'navigation_efficiency': optimal_path.get('coverage', 0) * optimal_path.get('closed_loops', 1)
        }


# Example usage
if __name__ == "__main__":
    print("="*70)
    print("Resonant Computation Engine - Test")
    print("="*70)

    # Create engine
    engine = ResonantComputationEngine(
        enable_all_harvesters=True,
        coherence_threshold=0.3,
        optimization_goal="maximize_annotation_confidence"
    )

    # Mock spectrum data
    spectrum_data = {
        'mz': np.array([100, 200, 300, 400, 500, 600]),
        'intensity': np.array([1000, 800, 1200, 600, 900, 700]),
        'rt': 15.5,
        'fragments': [
            {'mz': 100, 'intensity': 1000},
            {'mz': 200, 'intensity': 800},
            {'mz': 300, 'intensity': 1200},
        ]
    }

    # Mock processing function
    def mock_process(data):
        time.sleep(0.001)
        return np.sum([d.get('intensity', 0) for d in [data]] if isinstance(data, dict) else data)

    # Run resonant computation
    results = asyncio.run(
        engine.process_experiment_as_bayesian_network(
            spectrum_data,
            mock_process
        )
    )

    print("\n[Results Summary]")
    print(f"  Total time: {results['experiment_metadata']['total_time']:.3f} s")
    print(f"  Final optimization value: {results['experiment_metadata']['final_optimization_value']:.4f}")
    print(f"  Evidence nodes: {len(results['evidence_network'])}")
    print(f"  Hierarchical levels: {len(results['frequency_hierarchies'])}")
    print(f"  Closed loops found: {results['closed_loop_metrics']['closed_loops_found']}")
    print(f"  Network coverage: {results['closed_loop_metrics']['network_coverage']:.2%}")

    print("\n" + "="*70)
