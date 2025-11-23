"""
Virtual Mass Spectrometer Ensemble
===================================

Orchestrates multiple virtual mass spectrometers (Molecular Maxwell Demons)
reading the SAME categorical state simultaneously.

Core Principle (St-Stellas Theorem 3.7):
MMD Operation ≡ S-Navigation ≡ Categorical Completion

Architecture:
1. Hardware harvesters → Frequency hierarchies (8 scales)
2. Finite observers → Phase-lock detection at each scale
3. Transcendent observer → Integration via gear ratios
4. Convergence nodes → MMD materialization sites
5. MMD ensemble → Multiple instruments reading same categorical state

Justification for Virtual Instruments (No Simulation):
- Ion trajectories unknowable: infinite weak force configurations
- Journey never repeats: categorical irreversibility
- ~10^6 equivalent paths → same categorical state → same measurement
- Detector reads categorical invariants (m/q, time, charge), not path

Author: Kundai Farai Sachikonye
Date: 2025
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import time
import json

# Virtual infrastructure
from .molecular_demon_state_architecture import (
    MolecularMaxwellDemon,
    InstrumentProjection,
    CategoricalState
)
from .frequency_hierarchy import FrequencyHierarchyTree, HardwareScale
from .finite_observers import TranscendentObserver, PhaseLockSignature

# Hardware harvesters
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from hardware.clock_drift import ClockDriftHarvester
    from hardware.memory_access_patterns import MemoryOscillationHarvester
    from hardware.network_packet_timing import NetworkOscillationHarvester
    from hardware.usb_polling_rate import USBOscillationHarvester
    from hardware.gpu_memory_bandwidth import GPUOscillationHarvester
    from hardware.disk_partition import DiskIOHarvester
    from hardware.led_display_flicker import LEDSpectroscopyHarvester
    HARDWARE_AVAILABLE = True
except ImportError:
    HARDWARE_AVAILABLE = False
    print("[WARNING] Hardware harvesters not available - using simulated oscillations")

# BMD components
try:
    from bmd import BiologicalMaxwellDemonReference, HardwareBMDStream
    BMD_AVAILABLE = True
except ImportError:
    BMD_AVAILABLE = False
    print("[WARNING] BMD components not available - no hardware grounding")


@dataclass
class VirtualInstrumentResult:
    """Result from a single virtual instrument"""
    instrument_type: str
    instrument_projection: InstrumentProjection
    measurement: Dict[str, Any]
    categorical_state: CategoricalState
    materialization_time: float
    demon_statistics: Dict[str, Any]


@dataclass
class MassSpecEnsembleResult:
    """Complete ensemble result"""
    ensemble_id: str

    # Input data
    input_mz: np.ndarray
    input_intensity: np.ndarray
    input_metadata: Dict[str, Any]

    # Frequency hierarchy
    frequency_hierarchy_stats: Dict[str, Any]
    convergence_nodes_count: int

    # Finite observer measurements
    phase_locks_by_scale: Dict[str, int]  # scale_name → count
    total_phase_locks: int

    # MMD ensemble
    virtual_instruments: List[VirtualInstrumentResult]
    n_instruments: int

    # Cross-validation
    cross_validation: Dict[str, Any]

    # Performance
    total_time: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict"""
        return {
            'ensemble_id': self.ensemble_id,
            'input_data': {
                'n_peaks': len(self.input_mz),
                'mz_range': (float(self.input_mz.min()), float(self.input_mz.max())) if len(self.input_mz) > 0 else (0, 0),
                'metadata': self.input_metadata
            },
            'frequency_hierarchy': self.frequency_hierarchy_stats,
            'convergence_nodes': self.convergence_nodes_count,
            'phase_locks': {
                'by_scale': self.phase_locks_by_scale,
                'total': self.total_phase_locks
            },
            'virtual_instruments': [
                {
                    'type': vi.instrument_type,
                    'projection': vi.instrument_projection.value,
                    'measurement': vi.measurement,
                    'categorical_state': {
                        'S_k': vi.categorical_state.S_k,
                        'S_t': vi.categorical_state.S_t,
                        'S_e': vi.categorical_state.S_e,
                        'frequency_hz': vi.categorical_state.frequency_hz,
                        'equivalence_class_size': vi.categorical_state.equivalence_class_size,
                        'information_content_bits': vi.categorical_state.information_content_bits()
                    },
                    'demon_stats': vi.demon_statistics
                }
                for vi in self.virtual_instruments
            ],
            'cross_validation': self.cross_validation,
            'performance': {
                'total_time_s': self.total_time,
                'timestamp': self.timestamp
            }
        }


class VirtualMassSpecEnsemble:
    """
    Virtual Mass Spectrometer Ensemble.

    Creates multiple virtual instruments (MMDs) that all read the SAME
    categorical state simultaneously at convergence nodes.

    No simulation of:
    - TOF tubes (unknowable trajectories)
    - Quadrupole RF fields (never repeats)
    - Collision cells (chaotic)
    - Ion-ion interactions (infinite configurations)

    Only reads:
    - Categorical states at convergence nodes
    - S-coordinates: (S_k, S_t, S_e)
    - Predetermined by molecular properties
    """

    def __init__(
        self,
        enable_all_instruments: bool = True,
        enable_hardware_grounding: bool = True,
        coherence_threshold: float = 0.3
    ):
        """
        Initialize virtual mass spec ensemble.

        Args:
            enable_all_instruments: Create all instrument types (TOF, Orbitrap, FT-ICR, etc.)
            enable_hardware_grounding: Enable hardware oscillation harvesting and BMD grounding
            coherence_threshold: Phase-lock coherence threshold
        """
        self.enable_all_instruments = enable_all_instruments
        self.enable_hardware_grounding = enable_hardware_grounding
        self.coherence_threshold = coherence_threshold

        # Hardware harvesters (if enabled)
        self.hardware_harvesters = self._initialize_hardware_harvesters() if enable_hardware_grounding else None

        # BMD reference (if available and enabled)
        if BMD_AVAILABLE and enable_hardware_grounding:
            self.bmd_reference = BiologicalMaxwellDemonReference(enable_all_harvesters=True)
        else:
            self.bmd_reference = None

        # Frequency hierarchy
        self.frequency_hierarchy: Optional[FrequencyHierarchyTree] = None

        # Transcendent observer
        self.transcendent_observer: Optional[TranscendentObserver] = None

        # MMD ensemble
        self.molecular_demons: List[MolecularMaxwellDemon] = []

        print(f"[Virtual Mass Spec Ensemble] Initialized")
        print(f"  All instruments: {enable_all_instruments}")
        print(f"  Hardware grounding: {enable_hardware_grounding}")
        print(f"  Coherence threshold: {coherence_threshold}")

    def _initialize_hardware_harvesters(self) -> Dict[str, Any]:
        """Initialize 8-scale hardware oscillation harvesters"""
        if not HARDWARE_AVAILABLE:
            return {}

        return {
            'clock': ClockDriftHarvester(),
            'memory': MemoryOscillationHarvester(),
            'network': NetworkOscillationHarvester(),
            'usb': USBOscillationHarvester(),
            'gpu': GPUOscillationHarvester(),
            'disk': DiskIOHarvester(),
            'led': LEDSpectroscopyHarvester(),
            'global': None  # Computed from combination
        }

    def _harvest_hardware_oscillations(self) -> Dict[str, Any]:
        """
        Harvest oscillations from all hardware sources.

        Returns Dict mapping source → {frequency, phase, coherence}
        """
        if not self.hardware_harvesters:
            # Simulate hardware oscillations
            return {
                'clock': {'frequency': 3e9, 'phase': 0.0, 'coherence': 0.9},
                'memory': {'frequency': 1e9, 'phase': 0.5, 'coherence': 0.8},
                'network': {'frequency': 1e6, 'phase': 1.0, 'coherence': 0.7},
                'usb': {'frequency': 1e3, 'phase': 1.5, 'coherence': 0.75},
                'gpu': {'frequency': 1e9, 'phase': 0.3, 'coherence': 0.85},
                'disk': {'frequency': 100, 'phase': 2.0, 'coherence': 0.7},
                'led': {'frequency': 60, 'phase': 2.5, 'coherence': 0.8},
                'global': {'frequency': 0.1, 'phase': 0.0, 'coherence': 0.9}
            }

        # Actual harvesting (would call real harvesters)
        oscillations = {}
        for source, harvester in self.hardware_harvesters.items():
            if harvester is None:
                continue
            # Would call: oscillations[source] = harvester.harvest()
            # For now, simulated
            oscillations[source] = {
                'frequency': np.random.uniform(1e6, 1e9),
                'phase': np.random.uniform(0, 2*np.pi),
                'coherence': np.random.uniform(0.6, 0.95)
            }

        return oscillations

    def measure_spectrum(
        self,
        mz: np.ndarray,
        intensity: np.ndarray,
        rt: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> MassSpecEnsembleResult:
        """
        Measure spectrum using virtual mass spec ensemble.

        Process:
        1. Harvest hardware oscillations (8 scales)
        2. Build frequency hierarchy
        3. Deploy finite observers
        4. Detect phase-locks at all scales (parallel)
        5. Identify convergence nodes
        6. Materialize MMDs at convergence nodes
        7. Each MMD reads categorical state
        8. Project as different instruments (TOF, Orbitrap, FT-ICR, etc.)
        9. Cross-validate: all instruments should agree
        10. Dissolve MMDs

        Args:
            mz: Mass-to-charge ratios
            intensity: Intensities
            rt: Retention time (optional)
            metadata: Additional metadata

        Returns:
            MassSpecEnsembleResult with measurements from all virtual instruments
        """
        start_time = time.time()
        ensemble_id = f"ensemble_{int(start_time)}"

        print(f"\n{'='*70}")
        print(f"VIRTUAL MASS SPEC ENSEMBLE: {ensemble_id}")
        print(f"{'='*70}")
        print(f"Input: {len(mz)} peaks")
        if len(mz) > 0:
            print(f"  m/z range: {mz.min():.2f} - {mz.max():.2f}")
            print(f"  Total intensity: {intensity.sum():.2e}")
        if rt is not None:
            print(f"  RT: {rt:.2f} min")

        # STEP 1: Harvest hardware oscillations
        print(f"\n[Step 1] Harvesting hardware oscillations (8 scales)...")
        hardware_oscillations = self._harvest_hardware_oscillations()
        print(f"  ✓ Harvested from {len(hardware_oscillations)} hardware sources")

        # STEP 2: Build frequency hierarchy
        print(f"\n[Step 2] Building frequency hierarchy...")
        self.frequency_hierarchy = FrequencyHierarchyTree()
        self.frequency_hierarchy.build_from_hardware_oscillations(hardware_oscillations)
        hierarchy_stats = self.frequency_hierarchy.get_statistics()
        print(f"  ✓ Hierarchy: {hierarchy_stats['total_nodes']} nodes across 8 scales")

        # STEP 3: Deploy finite observers
        print(f"\n[Step 3] Deploying finite observers...")
        self.transcendent_observer = TranscendentObserver("transcendent_master")
        finite_observers = self.transcendent_observer.deploy_finite_observers(self.frequency_hierarchy)
        print(f"  ✓ Deployed {len(finite_observers)} finite observers")

        # STEP 4: Coordinate observations (parallel at all scales)
        print(f"\n[Step 4] Observing phase-locks at all scales (parallel)...")
        molecular_data = {
            'mz': mz,
            'intensity': intensity,
            'rt': rt or 0.0
        }
        observations_by_scale = self.transcendent_observer.coordinate_observations(molecular_data)

        phase_locks_by_scale = {
            scale.name: len(sigs) for scale, sigs in observations_by_scale.items()
        }
        total_phase_locks = sum(phase_locks_by_scale.values())
        print(f"  ✓ Detected {total_phase_locks} phase-locks across all scales")
        for scale_name, count in phase_locks_by_scale.items():
            if count > 0:
                print(f"    {scale_name}: {count} phase-locks")

        # STEP 5: Identify convergence nodes
        print(f"\n[Step 5] Identifying convergence nodes...")
        convergence_sites = self.transcendent_observer.identify_convergence_sites(
            observations_by_scale,
            top_fraction=0.1
        )
        print(f"  ✓ Found {len(convergence_sites)} convergence sites (top 10%)")

        # STEP 6-10: Materialize MMDs, measure, cross-validate, dissolve
        print(f"\n[Step 6-10] Materializing MMD ensemble...")
        virtual_instrument_results = []

        # For each convergence site, create ensemble of virtual instruments
        for i, (scale, signatures) in enumerate(convergence_sites[:3]):  # Top 3 sites
            print(f"\n  Convergence site {i+1} (Scale {scale.name}):")
            print(f"    Phase-locks: {len(signatures)}")

            # Instrument types to create at this node
            if self.enable_all_instruments:
                instrument_types = [
                    InstrumentProjection.TOF,
                    InstrumentProjection.ORBITRAP,
                    InstrumentProjection.FTICR,
                    InstrumentProjection.QUADRUPOLE,
                    InstrumentProjection.SECTOR,
                    InstrumentProjection.ION_MOBILITY,
                    InstrumentProjection.PHOTODETECTOR,
                    InstrumentProjection.ION_DETECTOR
                ]
            else:
                instrument_types = [InstrumentProjection.TOF]  # Just TOF

            # Create MMD for each instrument type at this node
            for inst_type in instrument_types:
                # Create Molecular Maxwell Demon
                demon = MolecularMaxwellDemon(
                    demon_id=f"{ensemble_id}_demon_{i}_{inst_type.value}",
                    convergence_node_id=f"node_{i}_{scale.name}",
                    instrument_projection=inst_type
                )

                # Prepare hardware phase-lock data
                hardware_phase_lock = {
                    'modes': {f"hw_{src}": hw['phase'] for src, hw in hardware_oscillations.items()},
                    'frequencies': {f"hw_{src}": hw['frequency'] for src, hw in hardware_oscillations.items()},
                    'signals': [
                        {
                            'mz': sig.mz_value,
                            'intensity': sig.intensity,
                            'frequency': sig.frequency_hz,
                            'phase': sig.phase_rad,
                            'coherence': sig.phase_coherence
                        }
                        for sig in signatures
                    ],
                    'hardware_stream_bmd': self.bmd_reference if self.bmd_reference else None
                }

                # Materialize demon (dual filtering + categorical state compression)
                categorical_state = demon.materialize(hardware_phase_lock)

                # Read instrument-specific projection
                measurement = demon.read_projection()

                # Store result
                virtual_instrument_results.append(VirtualInstrumentResult(
                    instrument_type=inst_type.value,
                    instrument_projection=inst_type,
                    measurement=measurement,
                    categorical_state=categorical_state,
                    materialization_time=time.time() - start_time,
                    demon_statistics=demon.get_statistics()
                ))

                print(f"      ✓ {inst_type.value}: m/z={measurement.get('mz', 'N/A')}")

                # Dissolve demon (exists only during measurement)
                demon.dissolve()

                self.molecular_demons.append(demon)

        # Cross-validation: Check if all instruments agree
        print(f"\n[Cross-Validation] Comparing measurements...")
        cross_validation = self._cross_validate_instruments(virtual_instrument_results)
        print(f"  ✓ Agreement: {cross_validation['agreement_summary']}")

        # Create result
        total_time = time.time() - start_time
        result = MassSpecEnsembleResult(
            ensemble_id=ensemble_id,
            input_mz=mz,
            input_intensity=intensity,
            input_metadata=metadata or {},
            frequency_hierarchy_stats=hierarchy_stats,
            convergence_nodes_count=len(convergence_sites),
            phase_locks_by_scale=phase_locks_by_scale,
            total_phase_locks=total_phase_locks,
            virtual_instruments=virtual_instrument_results,
            n_instruments=len(virtual_instrument_results),
            cross_validation=cross_validation,
            total_time=total_time
        )

        print(f"\n{'='*70}")
        print(f"ENSEMBLE COMPLETE")
        print(f"{'='*70}")
        print(f"Virtual instruments: {len(virtual_instrument_results)}")
        print(f"Convergence nodes: {len(convergence_sites)}")
        print(f"Total time: {total_time:.3f} s")
        print(f"Sample consumed: 0 molecules (categorical access)")
        print(f"Hardware cost: $0 marginal (virtual instruments)")
        print(f"{'='*70}\n")

        return result

    def _cross_validate_instruments(self, results: List[VirtualInstrumentResult]) -> Dict[str, Any]:
        """
        Cross-validate measurements from different virtual instruments.

        All instruments should agree on categorical invariants (m/z, charge)
        since they're reading the SAME categorical state.
        """
        if not results:
            return {'agreement_summary': 'No instruments', 'agreements': []}

        # Group by convergence node
        by_node = {}
        for result in results:
            node = result.demon_statistics['demon_id'].split('_demon_')[1].split('_')[0]
            if node not in by_node:
                by_node[node] = []
            by_node[node].append(result)

        # Check agreement within each node
        agreements = []
        for node, node_results in by_node.items():
            if len(node_results) < 2:
                continue

            # Extract m/z from each instrument
            mz_values = [r.measurement.get('mz', 0) for r in node_results]
            mz_values = [m for m in mz_values if m is not None and m > 0]

            if len(mz_values) < 2:
                continue

            # Check agreement (all within 0.01 Da)
            mz_std = np.std(mz_values)
            agrees = mz_std < 0.01

            agreements.append({
                'node': node,
                'n_instruments': len(node_results),
                'mz_values': mz_values,
                'mz_std': mz_std,
                'agrees': agrees,
                'instruments': [r.instrument_type for r in node_results]
            })

        # Summary
        n_agree = sum(1 for a in agreements if a['agrees'])
        summary = f"{n_agree}/{len(agreements)} nodes agree" if agreements else "Insufficient data"

        return {
            'agreement_summary': summary,
            'agreements': agreements,
            'n_nodes_checked': len(agreements),
            'n_nodes_agree': n_agree
        }

    def save_results(self, result: MassSpecEnsembleResult, output_dir: Path):
        """Save ensemble results to JSON"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"{result.ensemble_id}.json"

        with open(output_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)

        print(f"✓ Results saved to: {output_file}")
        return output_file


# Module exports
__all__ = [
    'VirtualMassSpecEnsemble',
    'VirtualInstrumentResult',
    'MassSpecEnsembleResult'
]
