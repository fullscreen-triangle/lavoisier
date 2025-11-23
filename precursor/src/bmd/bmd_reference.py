"""
Biological Maxwell Demon Reference from Hardware

Implements hardware BMD stream that grounds all spectral processing.

Key Concepts:
- Hardware oscillations (clock, memory, network, USB, GPU, disk, LED) form unified BMD stream
- Stream is phase-locked through physical coupling (AC power, EM fields, etc.)
- Provides reality grounding through intersection of compatible states
- All devices are EQUIVALENT components of unified stream (not independent)
"""

import numpy as np
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from .bmd_state import BMDState, PhaseStructure, OscillatoryHole
from .categorical_state import CategoricalState

# Import hardware harvesters from precursor
try:
    from ..hardware.clock_drift import ClockDriftHarvester
    from ..hardware.memory_access_patterns import MemoryAccessHarvester
    from ..hardware.network_latency import NetworkLatencyHarvester
    from ..hardware.usb_timing import USBTimingHarvester
    from ..hardware.gpu_thermal import GPUThermalHarvester
    from ..hardware.disk_access_patterns import DiskAccessHarvester
    from ..hardware.led_blink import LEDBlinkHarvester
except ImportError:
    # Fallback if hardware module not available
    ClockDriftHarvester = None
    MemoryAccessHarvester = None
    NetworkLatencyHarvester = None
    USBTimingHarvester = None
    GPUThermalHarvester = None
    DiskAccessHarvester = None
    LEDBlinkHarvester = None


@dataclass
class HardwareBMDStream:
    """
    Unified hardware BMD stream representing physical reality.

    β^(stream)_hardware = β_clock ⊛ β_memory ⊛ β_network ⊛ β_USB ⊛ β_GPU ⊛ β_disk ⊛ β_LED

    All devices are phase-locked through physical environment:
    - Clock driven by AC power line (50/60 Hz)
    - Network packet timing couples to EM fields
    - Display backlight couples to acoustic pressure
    - Accelerometer couples to vibrations from disk/fan

    Attributes:
        unified_bmd: Single BMD state representing hierarchical composition
        device_bmds: Individual device BMDs (for diagnostics)
        measurement_timestamp: When stream was measured
        phase_lock_quality: Quality of phase-locking across devices
    """
    unified_bmd: BMDState
    device_bmds: Dict[str, BMDState]
    measurement_timestamp: float
    phase_lock_quality: float  # [0, 1] where 1 = perfect phase-lock

    def get_categorical_richness(self) -> int:
        """
        Categorical richness of stream.

        R(β^stream) = |⋂_devices C_device| ≪ Π_devices R(β_device)

        Intersection dramatically reduces richness - only states compatible
        across ALL devices are physically real.
        """
        return self.unified_bmd.categorical_richness

    def get_phase_structure(self) -> PhaseStructure:
        """Get unified phase structure"""
        return self.unified_bmd.phase_structure

    def is_coherent(self, threshold: float = 0.7) -> bool:
        """Check if stream is coherent (high quality phase-lock)"""
        return self.phase_lock_quality >= threshold


class BiologicalMaxwellDemonReference:
    """
    Hardware BMD reference that grounds spectral processing in physical reality.

    Implements dual filtering at hardware level:
    - Input: Phase-lock coherence determines signal vs noise
    - Output: Stream coherence determines physical vs unphysical interpretations

    Usage:
        bmd_ref = BiologicalMaxwellDemonReference()
        hardware_stream = bmd_ref.measure_stream()

        # Use for grounding
        filtered_peaks = stage.observe_with_bmd_grounding(
            spectrum, hardware_bmd=hardware_stream.unified_bmd
        )
    """

    def __init__(self, enable_all_harvesters: bool = True,
                 enable_specific: Optional[List[str]] = None):
        """
        Initialize hardware BMD reference.

        Args:
            enable_all_harvesters: Enable all available hardware harvesters
            enable_specific: List of specific harvesters to enable
                           (e.g., ['clock', 'memory', 'network'])
        """
        self.harvesters: Dict[str, Any] = {}
        self._init_harvesters(enable_all_harvesters, enable_specific)

        # AC power line frequency (for clock synchronization)
        self.ac_line_frequency = 60.0  # Hz (50 Hz in Europe/Asia)

        # Stream history for tracking evolution
        self.stream_history: List[HardwareBMDStream] = []
        self.max_history_size = 1000

    def _init_harvesters(self, enable_all: bool, enable_specific: Optional[List[str]]):
        """Initialize hardware harvesters"""
        available_harvesters = {
            'clock': ClockDriftHarvester,
            'memory': MemoryAccessHarvester,
            'network': NetworkLatencyHarvester,
            'usb': USBTimingHarvester,
            'gpu': GPUThermalHarvester,
            'disk': DiskAccessHarvester,
            'led': LEDBlinkHarvester,
        }

        to_enable = enable_specific if enable_specific else list(available_harvesters.keys())

        if not enable_all:
            to_enable = enable_specific or []

        for name, harvester_class in available_harvesters.items():
            if name in to_enable and harvester_class is not None:
                try:
                    self.harvesters[name] = harvester_class()
                except Exception as e:
                    print(f"Warning: Could not initialize {name} harvester: {e}")

    def measure_stream(self) -> HardwareBMDStream:
        """
        Measure unified hardware BMD stream.

        Returns hierarchical composition of all device BMDs with phase-lock coupling:
        β^(stream) = β_clock ⊛ β_memory ⊛ ... ⊛ β_LED

        Returns:
            HardwareBMDStream with unified BMD and device-level BMDs
        """
        timestamp = time.time()

        # Measure individual device BMDs
        device_bmds = {}
        for name, harvester in self.harvesters.items():
            try:
                device_bmds[name] = self._measure_device_bmd(name, harvester)
            except Exception as e:
                print(f"Warning: Could not measure {name}: {e}")

        if not device_bmds:
            # No devices available - return minimal BMD
            return self._create_fallback_stream(timestamp)

        # Phase-lock devices through hierarchical composition
        unified_bmd = self._phase_lock_composition(device_bmds)

        # Compute phase-lock quality
        phase_lock_quality = self._compute_phase_lock_quality(device_bmds)

        # Create stream
        stream = HardwareBMDStream(
            unified_bmd=unified_bmd,
            device_bmds=device_bmds,
            measurement_timestamp=timestamp,
            phase_lock_quality=phase_lock_quality
        )

        # Store in history
        self.stream_history.append(stream)
        if len(self.stream_history) > self.max_history_size:
            self.stream_history.pop(0)

        return stream

    def _measure_device_bmd(self, device_name: str, harvester: Any) -> BMDState:
        """
        Measure BMD state from a single hardware device.

        Args:
            device_name: Name of device
            harvester: Hardware harvester instance

        Returns:
            BMD state for this device
        """
        # Harvest oscillation data
        measurement = harvester.harvest()

        # Extract phase information
        phase = measurement.get('phase', 0.0)
        frequency = measurement.get('frequency', 1.0)
        coherence_time = measurement.get('coherence_time', 1e-6)

        # Create phase structure
        phase_structure = PhaseStructure()
        phase_structure.add_mode(device_name, phase, frequency, coherence_time)

        # Create oscillatory hole
        # Each device creates hole with ~100-1000 possible configurations
        hole_size = measurement.get('hole_size', 100)
        hole = OscillatoryHole(
            hole_id=f"{device_name}_hole",
            possible_configurations=set([f"{device_name}_config_{i}" for i in range(hole_size)]),
            selection_frequency=frequency,
            categorical_possibilities=hole_size
        )

        # Create BMD state
        bmd = BMDState(
            bmd_id=f"bmd_{device_name}_{int(time.time()*1e6)}",
            oscillatory_hole=hole,
            phase_structure=phase_structure,
            device_source=device_name,
            created_at=time.time()
        )

        return bmd

    def _phase_lock_composition(self, device_bmds: Dict[str, BMDState]) -> BMDState:
        """
        Hierarchically compose device BMDs through phase-lock coupling.

        β^(stream) = β_clock ⊛ β_memory ⊛ ... ⊛ β_LED

        Phase-lock structure:
        - Clock drives everything (AC power line 50-60 Hz)
        - Other devices phase-lock to clock + each other
        - Coupling through: EM fields, acoustic pressure, mechanical vibration

        Args:
            device_bmds: Dictionary of device BMDs

        Returns:
            Unified BMD through hierarchical composition
        """
        if not device_bmds:
            return self._create_default_bmd()

        # Start with clock as base (if available)
        if 'clock' in device_bmds:
            base_bmd = device_bmds['clock']
            base_phase = base_bmd.phase_structure.modes.get('clock', 0.0)
        else:
            # Use first available device
            base_bmd = list(device_bmds.values())[0]
            base_phase = 0.0

        # Align all devices to base phase (AC power line coupling)
        aligned_bmds = []
        for name, bmd in device_bmds.items():
            # Phase-lock to base
            aligned = self._align_to_base_phase(bmd, base_phase, name)
            aligned_bmds.append(aligned)

        # Hierarchically merge all aligned BMDs
        unified = aligned_bmds[0]
        for bmd in aligned_bmds[1:]:
            unified = unified.hierarchical_merge(bmd)

        # Update categorical richness as INTERSECTION
        # R(β^stream) = |⋂_devices C_device| ≪ Π_devices R(β_device)
        unified.categorical_richness = self._compute_stream_richness(device_bmds)

        unified.bmd_id = f"hardware_stream_{int(time.time()*1e6)}"

        return unified

    def _align_to_base_phase(self, bmd: BMDState, base_phase: float,
                            device_name: str) -> BMDState:
        """
        Align device BMD to base phase (AC power line).

        Simulates physical coupling through:
        - Display refresh ↔ AC line frequency
        - Network timing ↔ EM fields
        - Disk/GPU ↔ mechanical/thermal coupling

        Args:
            bmd: Device BMD to align
            base_phase: Base phase (from clock/AC line)
            device_name: Name of device

        Returns:
            Aligned BMD
        """
        aligned = BMDState(
            bmd_id=bmd.bmd_id,
            current_categorical_state=bmd.current_categorical_state,
            oscillatory_hole=bmd.oscillatory_hole,
            phase_structure=PhaseStructure(),
            device_source=bmd.device_source,
            history=bmd.history.copy()
        )

        # Copy and adjust phases
        for mode_name, phase in bmd.phase_structure.modes.items():
            # Add coupling to base phase
            coupling_strength = 0.1  # Weak coupling
            aligned_phase = phase + coupling_strength * base_phase
            aligned_phase = aligned_phase % (2 * np.pi)

            aligned.phase_structure.add_mode(
                mode_name,
                aligned_phase,
                bmd.phase_structure.frequencies.get(mode_name, 1.0),
                bmd.phase_structure.coherence_times.get(mode_name, 1e-6)
            )

        return aligned

    def _compute_phase_lock_quality(self, device_bmds: Dict[str, BMDState]) -> float:
        """
        Compute phase-lock quality across all devices.

        Quality = average pairwise phase coherence

        Args:
            device_bmds: Dictionary of device BMDs

        Returns:
            Quality in [0, 1] where 1 = perfect phase-lock
        """
        if len(device_bmds) < 2:
            return 1.0

        devices = list(device_bmds.values())
        total_coherence = 0.0
        count = 0

        for i in range(len(devices)):
            for j in range(i+1, len(devices)):
                phase_distance = devices[i].phase_structure.phase_lock_distance(
                    devices[j].phase_structure
                )
                # Convert distance to coherence (0 = perfect)
                coherence = max(0.0, 1.0 - phase_distance / np.pi)
                total_coherence += coherence
                count += 1

        return total_coherence / count if count > 0 else 0.0

    def _compute_stream_richness(self, device_bmds: Dict[str, BMDState]) -> int:
        """
        Compute categorical richness as INTERSECTION.

        R(β^stream) = |⋂_devices C_device| ≪ Π_devices R(β_device)

        Intersection dramatically reduces richness - only states compatible
        across ALL devices are physically real.

        Args:
            device_bmds: Dictionary of device BMDs

        Returns:
            Stream categorical richness
        """
        if not device_bmds:
            return 1

        # Product would be astronomical
        product = 1
        for bmd in device_bmds.values():
            product *= bmd.categorical_richness

        # Intersection is MUCH smaller (estimate as geometric mean / N)
        N = len(device_bmds)
        geometric_mean = product ** (1.0 / N)
        intersection_estimate = int(geometric_mean / N)

        return max(intersection_estimate, 1)

    def _create_fallback_stream(self, timestamp: float) -> HardwareBMDStream:
        """Create minimal fallback stream when no harvesters available"""
        default_bmd = self._create_default_bmd()

        return HardwareBMDStream(
            unified_bmd=default_bmd,
            device_bmds={},
            measurement_timestamp=timestamp,
            phase_lock_quality=0.0
        )

    def _create_default_bmd(self) -> BMDState:
        """Create default BMD with minimal structure"""
        phase_structure = PhaseStructure()
        phase_structure.add_mode('system', 0.0, 1.0, 1e-6)

        hole = OscillatoryHole(
            hole_id="default_hole",
            possible_configurations=set(['default']),
            categorical_possibilities=1
        )

        return BMDState(
            bmd_id=f"default_{int(time.time()*1e6)}",
            oscillatory_hole=hole,
            phase_structure=phase_structure,
            device_source='system'
        )

    def get_stream_evolution(self, window_size: int = 10) -> List[HardwareBMDStream]:
        """
        Get recent stream evolution.

        Args:
            window_size: Number of recent measurements

        Returns:
            List of recent hardware streams
        """
        return self.stream_history[-window_size:]

    def compute_stream_drift_rate(self) -> float:
        """
        Compute rate of stream drift (phase evolution).

        Returns:
            Drift rate (radians/second)
        """
        if len(self.stream_history) < 2:
            return 0.0

        recent = self.stream_history[-2:]

        dt = recent[1].measurement_timestamp - recent[0].measurement_timestamp
        if dt <= 0:
            return 0.0

        phase_distance = recent[0].unified_bmd.phase_structure.phase_lock_distance(
            recent[1].unified_bmd.phase_structure
        )

        return phase_distance / dt
