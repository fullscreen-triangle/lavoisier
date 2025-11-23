"""
Finite Observer Framework for Virtual Mass Spectrometry
========================================================

Implements finite observers that detect phase-locks at specific
hierarchical levels, coordinated by a transcendent observer.

From resonant_computation_readme.md:
"Always introduce a finite observer"
- Finite observer = observes exactly ONE hierarchical level
- Transcendent observer = observes other finite observers
- Navigation = gear ratio transitions between observers
- Observation = phase-lock detection within observer's window

Key Concept: Finite observers enable parallel measurement at all scales,
with transcendent observer coordinating via gear ratios.

Author: Kundai Farai Sachikonye
Date: 2025
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import time

from .frequency_hierarchy import FrequencyHierarchyNode, HardwareScale


class ObserverType(Enum):
    """Types of observers in the hierarchy"""
    FINITE = "finite"            # Observes one level
    TRANSCENDENT = "transcendent"  # Observes finite observers


@dataclass
class PhaseLockSignature:
    """
    Phase-lock signature detected by finite observer.

    Represents a molecular oscillation phase-locked to hardware at a specific scale.
    """
    signature_id: str

    # Molecular properties
    mz_value: float
    intensity: float
    frequency_hz: float

    # Phase information
    phase_rad: float  # Phase relative to hardware oscillation
    phase_coherence: float  # [0, 1] quality of phase-lock

    # Hardware context
    hardware_scale: HardwareScale
    hardware_source: str  # 'clock', 'memory', 'network', etc.
    hardware_frequency: float

    # Observation window
    observation_start: float
    observation_end: float

    # Temporal
    detection_time: float = field(default_factory=time.time)

    def is_phase_locked(self, threshold: float = np.pi/4) -> bool:
        """
        Check if truly phase-locked.

        Criterion from phase-lock theory: |φ_i - φ_j| < π/4
        """
        return self.phase_coherence > 0.7  # Coherence > 70% → phase-locked

    def gear_ratio_to_molecular(self) -> float:
        """Compute gear ratio between hardware and molecular frequency"""
        if self.frequency_hz == 0:
            return 0.0
        return self.hardware_frequency / self.frequency_hz


class FiniteObserver:
    """
    Finite Observer: Observes exactly ONE hierarchical level.

    Responsibility:
    - Monitor specific observation window (one hardware scale)
    - Detect phase-locks between molecular and hardware oscillations
    - Report findings to transcendent observer
    - No knowledge of other levels (strictly local view)

    Key Principle: "Always introduce a finite observer"
    - Creates clean separation between scales
    - Enables parallel observation at all levels
    - Simplifies complexity (each observer sees only its window)
    """

    def __init__(
        self,
        observer_id: str,
        hierarchical_level: HardwareScale,
        observation_node: FrequencyHierarchyNode
    ):
        """
        Initialize finite observer at specific hierarchical level.

        Args:
            observer_id: Unique observer identifier
            hierarchical_level: Which hardware scale this observer monitors
            observation_node: Frequency hierarchy node (defines window)
        """
        self.observer_id = observer_id
        self.hierarchical_level = hierarchical_level
        self.observation_node = observation_node

        # Observation window (finite = limited view)
        self.window_start = observation_node.observation_window_start
        self.window_end = observation_node.observation_window_end

        # Hardware reference frequency
        self.hardware_frequency = observation_node.frequency_hz
        self.hardware_source = observation_node.hardware_source

        # Detected phase-locks
        self.phase_lock_signatures: List[PhaseLockSignature] = []

        # Observation statistics
        self.total_observations: int = 0
        self.phase_locks_detected: int = 0
        self.observation_start_time: Optional[float] = None

    def observe(self, molecular_signals: List[Dict[str, Any]]) -> List[PhaseLockSignature]:
        """
        Observe molecular signals within this observer's window.

        Detects which signals are phase-locked to hardware at this scale.

        Args:
            molecular_signals: List of molecular data (mz, intensity, frequency, phase)

        Returns:
            List of phase-lock signatures detected
        """
        if self.observation_start_time is None:
            self.observation_start_time = time.time()

        signatures = []

        for signal in molecular_signals:
            self.total_observations += 1

            # Check if signal frequency is in this observer's window
            signal_freq = signal.get('frequency', 0)

            # Frequency window check
            in_window = (self.observation_node.frequency_range[0] <= signal_freq <=
                        self.observation_node.frequency_range[1])

            if not in_window:
                continue

            # Measure phase-lock quality
            signal_phase = signal.get('phase', 0)
            hardware_phase = self._measure_hardware_phase()

            # Phase difference
            phase_diff = abs(signal_phase - hardware_phase)
            phase_diff = min(phase_diff, 2*np.pi - phase_diff)  # Wrap to [0, π]

            # Phase coherence (closer to 0 → higher coherence)
            phase_coherence = 1.0 - (phase_diff / np.pi)

            # Threshold check (π/4 criterion)
            if phase_diff < np.pi/4:
                # Phase-locked!
                signature = PhaseLockSignature(
                    signature_id=f"{self.observer_id}_sig_{self.phase_locks_detected}",
                    mz_value=signal.get('mz', 0),
                    intensity=signal.get('intensity', 0),
                    frequency_hz=signal_freq,
                    phase_rad=signal_phase,
                    phase_coherence=phase_coherence,
                    hardware_scale=self.hierarchical_level,
                    hardware_source=self.hardware_source,
                    hardware_frequency=self.hardware_frequency,
                    observation_start=self.window_start,
                    observation_end=self.window_end
                )

                signatures.append(signature)
                self.phase_lock_signatures.append(signature)
                self.phase_locks_detected += 1

        return signatures

    def _measure_hardware_phase(self) -> float:
        """
        Measure current hardware phase.

        In real implementation, would read from hardware oscillation harvester.
        For now, simulate based on time and frequency.
        """
        current_time = time.time()
        if self.observation_start_time is None:
            return 0.0

        elapsed = current_time - self.observation_start_time
        phase = (2 * np.pi * self.hardware_frequency * elapsed) % (2 * np.pi)
        return phase

    def get_observation_report(self) -> Dict[str, Any]:
        """
        Generate observation report for transcendent observer.

        Finite observer reports:
        - How many signals observed
        - How many phase-locks detected
        - Phase-lock signatures
        - Observation window
        """
        return {
            'observer_id': self.observer_id,
            'observer_type': ObserverType.FINITE.value,
            'hierarchical_level': self.hierarchical_level.name,
            'hardware_source': self.hardware_source,
            'observation_window': (self.window_start, self.window_end),
            'total_observations': self.total_observations,
            'phase_locks_detected': self.phase_locks_detected,
            'detection_rate': self.phase_locks_detected / self.total_observations if self.total_observations > 0 else 0,
            'signatures': [
                {
                    'signature_id': sig.signature_id,
                    'mz': sig.mz_value,
                    'frequency': sig.frequency_hz,
                    'coherence': sig.phase_coherence
                }
                for sig in self.phase_lock_signatures
            ]
        }


class TranscendentObserver:
    """
    Transcendent Observer: Observes other finite observers.

    Responsibility:
    - Coordinate finite observers across all scales
    - Integrate observations via gear ratios
    - Identify convergence nodes for MMD materialization
    - Navigate hierarchically using O(1) gear ratio jumps

    Key Principle: Transcendent observer sees the STRUCTURE of observations,
    not the observations themselves. It navigates between finite observers
    using gear ratios, synthesizing their reports into unified view.
    """

    def __init__(self, transcendent_id: str):
        """
        Initialize transcendent observer.

        Args:
            transcendent_id: Unique transcendent observer identifier
        """
        self.transcendent_id = transcendent_id
        self.finite_observers: Dict[HardwareScale, List[FiniteObserver]] = {
            scale: [] for scale in HardwareScale
        }
        self.integration_start_time: Optional[float] = None

    def deploy_finite_observers(self, frequency_hierarchy: Any) -> List[FiniteObserver]:
        """
        Deploy finite observers at all hierarchical levels.

        One finite observer per frequency hierarchy node.

        Args:
            frequency_hierarchy: FrequencyHierarchyTree instance

        Returns:
            List of deployed finite observers
        """
        deployed = []

        for scale in HardwareScale:
            nodes = frequency_hierarchy.nodes_by_level[scale]

            for node in nodes:
                observer = FiniteObserver(
                    observer_id=f"finite_{scale.name}_{node.node_id}",
                    hierarchical_level=scale,
                    observation_node=node
                )

                self.finite_observers[scale].append(observer)
                deployed.append(observer)

        return deployed

    def coordinate_observations(self, molecular_data: Dict[str, Any]) -> Dict[HardwareScale, List[PhaseLockSignature]]:
        """
        Coordinate observations across all finite observers.

        Each finite observer observes in parallel within its window.
        Transcendent observer collects and organizes results.

        Args:
            molecular_data: Molecular signals to observe

        Returns:
            Dict mapping scale → phase-lock signatures detected at that scale
        """
        if self.integration_start_time is None:
            self.integration_start_time = time.time()

        # Prepare molecular signals
        mz = molecular_data.get('mz', np.array([]))
        intensity = molecular_data.get('intensity', np.array([]))

        # Convert m/z to frequencies (rough approximation)
        molecular_frequencies = 1e13 / np.sqrt(mz) if len(mz) > 0 else np.array([])

        # Simulate phases (would measure from hardware)
        phases = np.random.uniform(0, 2*np.pi, len(mz))

        # Create signal list
        signals = [
            {
                'mz': mz[i],
                'intensity': intensity[i],
                'frequency': molecular_frequencies[i],
                'phase': phases[i]
            }
            for i in range(len(mz))
        ]

        # Each finite observer observes (parallel operation)
        observations_by_scale = {}

        for scale in HardwareScale:
            scale_signatures = []

            for observer in self.finite_observers[scale]:
                signatures = observer.observe(signals)
                scale_signatures.extend(signatures)

            observations_by_scale[scale] = scale_signatures

        return observations_by_scale

    def integrate_via_gear_ratios(self, observations_by_scale: Dict[HardwareScale, List[PhaseLockSignature]]) -> List[Dict[str, Any]]:
        """
        Integrate observations across scales using gear ratios.

        Key innovation: Gear ratios enable O(1) integration between scales.

        Finds patterns that span multiple scales (e.g., fragment at Scale 2
        phase-locked to precursor at Scale 4 via gear ratio).

        Args:
            observations_by_scale: Observations from all finite observers

        Returns:
            List of integrated observations (cross-scale patterns)
        """
        integrated = []

        # For each pair of adjacent scales, check for gear ratio relationships
        scales = sorted(HardwareScale, key=lambda s: s.value)

        for i in range(len(scales) - 1):
            scale_coarse = scales[i+1]  # Coarser (lower frequency)
            scale_fine = scales[i]      # Finer (higher frequency)

            sigs_coarse = observations_by_scale.get(scale_coarse, [])
            sigs_fine = observations_by_scale.get(scale_fine, [])

            # Find gear ratio relationships
            for sig_c in sigs_coarse:
                for sig_f in sigs_fine:
                    # Compute gear ratio
                    gear_ratio = sig_c.frequency_hz / sig_f.frequency_hz if sig_f.frequency_hz > 0 else 0

                    # Check if gear ratio is close to integer (indicates hierarchical coupling)
                    nearest_int = round(gear_ratio)
                    gear_error = abs(gear_ratio - nearest_int)

                    if gear_error < 0.1 and nearest_int > 0:  # Within 10% of integer
                        # Coupled via gear ratio!
                        integrated.append({
                            'coarse_scale': scale_coarse.name,
                            'fine_scale': scale_fine.name,
                            'gear_ratio': gear_ratio,
                            'coarse_signature': sig_c.signature_id,
                            'fine_signature': sig_f.signature_id,
                            'coarse_mz': sig_c.mz_value,
                            'fine_mz': sig_f.mz_value,
                            'coupling_quality': 1.0 - gear_error
                        })

        return integrated

    def identify_convergence_sites(self, observations_by_scale: Dict[HardwareScale, List[PhaseLockSignature]],
                                  top_fraction: float = 0.1) -> List[Tuple[HardwareScale, List[PhaseLockSignature]]]:
        """
        Identify convergence sites (high phase-lock density).

        These are optimal for MMD materialization because many categorical
        paths intersect at these sites.

        Args:
            observations_by_scale: Observations from all scales
            top_fraction: Fraction to consider as convergence (top 10%)

        Returns:
            List of (scale, signatures) for convergence sites
        """
        # Compute density at each scale
        densities = []

        for scale in HardwareScale:
            signatures = observations_by_scale.get(scale, [])

            if not signatures:
                continue

            # Density = number of signatures × average coherence
            n_sigs = len(signatures)
            avg_coherence = np.mean([sig.phase_coherence for sig in signatures])
            density = n_sigs * avg_coherence

            densities.append((density, scale, signatures))

        # Sort by density
        densities.sort(reverse=True, key=lambda x: x[0])

        # Take top fraction
        n_convergence = max(1, int(len(densities) * top_fraction))
        convergence_sites = [(scale, sigs) for _, scale, sigs in densities[:n_convergence]]

        return convergence_sites

    def get_transcendent_report(self) -> Dict[str, Any]:
        """
        Generate transcendent observation report.

        Integrates reports from all finite observers into unified view.
        """
        # Collect all finite observer reports
        finite_reports = []
        total_phase_locks = 0
        total_observations = 0

        for scale in HardwareScale:
            for observer in self.finite_observers[scale]:
                report = observer.get_observation_report()
                finite_reports.append(report)
                total_phase_locks += report['phase_locks_detected']
                total_observations += report['total_observations']

        return {
            'transcendent_id': self.transcendent_id,
            'observer_type': ObserverType.TRANSCENDENT.value,
            'n_finite_observers': len(finite_reports),
            'total_observations': total_observations,
            'total_phase_locks': total_phase_locks,
            'global_detection_rate': total_phase_locks / total_observations if total_observations > 0 else 0,
            'finite_observer_reports': finite_reports,
            'scales_monitored': [scale.name for scale in HardwareScale]
        }


# Module exports
__all__ = [
    'ObserverType',
    'PhaseLockSignature',
    'FiniteObserver',
    'TranscendentObserver'
]
