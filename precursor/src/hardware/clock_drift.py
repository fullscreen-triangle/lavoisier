#!/usr/bin/env python3
"""
Clock Drift Harvester
====================

Hardware clock drift = molecular phase coherence decay!

Perfect mapping:
- Clock drift over time = coherence time
- Drift rate = decoherence rate
- Synchronization corrections = phase lock maintenance

For proteomics: All fragments from a collision event are frequency-coupled.
The hardware clock drift during processing measures how long this coupling persists.

Author: Lavoisier Project
Date: October 2025
"""

import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import platform

if platform.system() == 'Windows':
    import ctypes

    # Windows high-resolution performance counter
    _kernel32 = ctypes.windll.kernel32
    _qpc_frequency = ctypes.c_int64()
    _kernel32.QueryPerformanceFrequency(ctypes.byref(_qpc_frequency))
    _qpc_frequency = _qpc_frequency.value

    def _qpc():
        """Query Performance Counter (Windows)"""
        counter = ctypes.c_int64()
        _kernel32.QueryPerformanceCounter(ctypes.byref(counter))
        return counter.value / _qpc_frequency
else:
    # Unix: use time.monotonic() and time.time()
    def _qpc():
        return time.perf_counter()


@dataclass
class ClockDriftMeasurement:
    """Measurement of clock drift during processing."""
    start_time_monotonic: float
    start_time_realtime: float
    end_time_monotonic: float
    end_time_realtime: float
    drift: float  # Drift in seconds
    coherence_time: float  # Estimated coherence time
    decoherence_rate: float  # Rate of coherence loss (1/s)
    event_duration: float  # Duration of the event


class ClockDriftHarvester:
    """
    Hardware clock drift = molecular phase coherence decay!

    Perfect mapping:
    - Clock drift over time = coherence time
    - Drift rate = decoherence rate
    - Synchronization corrections = phase lock maintenance

    For proteomics:
    All fragments from a peptide collision are frequency-coupled because
    they emerge at the same time. Clock drift during processing measures
    how long this temporal coupling persists in the computational domain.
    """

    def __init__(self):
        """Initialize clock drift harvester."""
        self.drift_measurements: List[ClockDriftMeasurement] = []
        self.baseline_drift = self._measure_baseline_drift()

        print("[Clock Drift Harvester] Initialized")
        print(f"  Baseline drift: {self.baseline_drift*1e6:.3f} μs over 1 second")

    def _measure_baseline_drift(self, duration: float = 1.0) -> float:
        """Measure baseline clock drift without processing."""
        t0_mono = time.monotonic()
        t0_real = time.time()

        time.sleep(duration)

        t1_mono = time.monotonic()
        t1_real = time.time()

        # Drift = difference in elapsed time between two clocks
        drift = abs((t1_real - t0_real) - (t1_mono - t0_mono))

        return drift / duration  # Drift per second

    def measure_coherence_time(
        self,
        processing_function: Callable,
        *args,
        collision_energy: Optional[float] = None,
        **kwargs
    ) -> ClockDriftMeasurement:
        """
        Measure how long hardware clocks stay synchronized during processing.

        This measures how long fragments stay frequency-coupled!

        Args:
            processing_function: Function that processes peptide fragmentation
            *args: Arguments for processing_function
            collision_energy: Collision energy (if available)
            **kwargs: Keyword arguments for processing_function

        Returns:
            ClockDriftMeasurement with coherence time
        """
        # Start clocks
        t0_mono = time.monotonic()
        t0_real = time.time()
        t0_perf = time.perf_counter()

        # Process collision event
        result = processing_function(*args, **kwargs)

        # End clocks
        t1_mono = time.monotonic()
        t1_real = time.time()
        t1_perf = time.perf_counter()

        # Calculate drift
        elapsed_mono = t1_mono - t0_mono
        elapsed_real = t1_real - t0_real
        elapsed_perf = t1_perf - t0_perf

        # Drift = desynchronization between clocks
        drift_mono_real = abs(elapsed_real - elapsed_mono)

        # Correct for baseline drift
        drift_corrected = drift_mono_real - (self.baseline_drift * elapsed_mono)

        # Event duration (use most precise clock)
        event_duration = elapsed_perf

        # Coherence time: how long before significant drift
        # Significant drift threshold: 1 microsecond
        coherence_threshold = 1e-6  # 1 μs

        if abs(drift_corrected) > coherence_threshold:
            coherence_time = event_duration * (coherence_threshold / abs(drift_corrected))
        else:
            # Very low drift - excellent coherence
            coherence_time = event_duration * 1000  # Extrapolate

        # Decoherence rate
        decoherence_rate = 1.0 / coherence_time if coherence_time > 0 else 0.0

        measurement = ClockDriftMeasurement(
            start_time_monotonic=t0_mono,
            start_time_realtime=t0_real,
            end_time_monotonic=t1_mono,
            end_time_realtime=t1_real,
            drift=drift_corrected,
            coherence_time=coherence_time,
            decoherence_rate=decoherence_rate,
            event_duration=event_duration
        )

        self.drift_measurements.append(measurement)

        return measurement

    def measure_fragment_coupling_coherence(
        self,
        fragment_processing_functions: List[Callable],
        fragment_data: List[Dict]
    ) -> Dict:
        """
        Measure coherence time for multiple fragments from same collision.

        All fragments should have similar coherence times if they're
        truly frequency-coupled from the same event.

        Args:
            fragment_processing_functions: List of processing functions (one per fragment)
            fragment_data: List of data for each fragment

        Returns:
            Dictionary with coupling coherence statistics
        """
        fragment_coherences = []

        for func, data in zip(fragment_processing_functions, fragment_data):
            measurement = self.measure_coherence_time(func, data)
            fragment_coherences.append(measurement.coherence_time)

        # All fragments from same collision should have similar coherence
        mean_coherence = np.mean(fragment_coherences)
        std_coherence = np.std(fragment_coherences)

        # Coupling quality = how uniform the coherence times are
        coupling_quality = 1.0 / (1.0 + std_coherence / (mean_coherence + 1e-9))

        return {
            'mean_coherence_time': mean_coherence,
            'std_coherence_time': std_coherence,
            'coupling_quality': coupling_quality,
            'fragment_coherences': fragment_coherences,
            'num_fragments': len(fragment_coherences)
        }

    def get_coherence_statistics(self) -> Dict:
        """Get statistics of all coherence measurements."""
        if not self.drift_measurements:
            return {}

        coherence_times = [m.coherence_time for m in self.drift_measurements]
        drifts = [m.drift for m in self.drift_measurements]
        decoherence_rates = [m.decoherence_rate for m in self.drift_measurements]

        return {
            'num_measurements': len(self.drift_measurements),
            'mean_coherence_time': np.mean(coherence_times),
            'std_coherence_time': np.std(coherence_times),
            'median_coherence_time': np.median(coherence_times),
            'mean_drift': np.mean(drifts),
            'std_drift': np.std(drifts),
            'mean_decoherence_rate': np.mean(decoherence_rates),
            'baseline_drift': self.baseline_drift
        }

    def estimate_phase_lock_lifetime(
        self,
        collision_energy: float,
        fragment_count: int
    ) -> float:
        """
        Estimate how long phase-lock persists based on collision parameters.

        Higher energy → shorter coherence (more violent collision)
        More fragments → shorter coherence (more complex system)

        Args:
            collision_energy: Collision energy in eV
            fragment_count: Number of fragments

        Returns:
            Estimated phase-lock lifetime in seconds
        """
        # Base coherence time from measurements
        if self.drift_measurements:
            base_coherence = np.median([m.coherence_time for m in self.drift_measurements])
        else:
            base_coherence = 1e-3  # 1 ms default

        # Energy scaling: higher energy → shorter coherence
        energy_factor = np.exp(-collision_energy / 50.0)  # Decay constant ~50 eV

        # Fragment count scaling: more fragments → shorter coherence
        fragment_factor = 1.0 / np.sqrt(fragment_count)

        estimated_lifetime = base_coherence * energy_factor * fragment_factor

        return estimated_lifetime

    def synchronization_correction_frequency(
        self,
        target_coherence_time: float
    ) -> float:
        """
        Calculate how often clocks need synchronization to maintain coherence.

        This is the "phase lock maintenance" frequency.

        Args:
            target_coherence_time: Desired coherence time

        Returns:
            Required synchronization frequency in Hz
        """
        # Need to sync before significant drift accumulates
        # Safety factor: sync at 10x the drift rate
        safety_factor = 10.0

        if target_coherence_time > 0:
            sync_frequency = safety_factor / target_coherence_time
        else:
            sync_frequency = 1000.0  # Default 1 kHz

        return sync_frequency


if __name__ == "__main__":
    print("="*70)
    print("Clock Drift Harvester - Coherence Time Measurement")
    print("="*70)

    harvester = ClockDriftHarvester()

    # Mock peptide fragmentation processing
    def mock_fragmentation_processing(fragment_mz, collision_energy):
        """Simulate fragment processing."""
        # Simulate some computation
        result = 0.0
        for i in range(10000):
            result += np.sin(fragment_mz * i) * np.exp(-collision_energy / 100.0)
        return result

    print("\n[Test 1] Single collision event coherence")
    measurement = harvester.measure_coherence_time(
        mock_fragmentation_processing,
        300.0,  # fragment m/z
        25.0    # collision energy
    )

    print(f"  Event duration: {measurement.event_duration*1e3:.3f} ms")
    print(f"  Clock drift: {measurement.drift*1e6:.3f} μs")
    print(f"  Coherence time: {measurement.coherence_time*1e3:.3f} ms")
    print(f"  Decoherence rate: {measurement.decoherence_rate:.2f} Hz")

    print("\n[Test 2] Multiple fragments from same collision")
    fragment_funcs = [
        lambda d: mock_fragmentation_processing(d['mz'], d['energy'])
        for _ in range(5)
    ]
    fragment_data = [
        {'mz': mz, 'energy': 25.0}
        for mz in [100, 200, 300, 400, 500]
    ]

    coupling_stats = harvester.measure_fragment_coupling_coherence(
        fragment_funcs,
        fragment_data
    )

    print(f"  Mean coherence: {coupling_stats['mean_coherence_time']*1e3:.3f} ms")
    print(f"  Std coherence: {coupling_stats['std_coherence_time']*1e6:.3f} μs")
    print(f"  Coupling quality: {coupling_stats['coupling_quality']:.3f}")

    print("\n[Test 3] Phase-lock lifetime estimation")
    for energy in [10, 25, 50]:
        for n_frags in [3, 5, 10]:
            lifetime = harvester.estimate_phase_lock_lifetime(energy, n_frags)
            print(f"  Energy={energy}eV, Frags={n_frags}: {lifetime*1e3:.3f} ms")

    print("\n[Test 4] Synchronization frequency")
    for target_coherence in [1e-3, 1e-4, 1e-5]:
        sync_freq = harvester.synchronization_correction_frequency(target_coherence)
        print(f"  Target coherence={target_coherence*1e3:.3f}ms → Sync @ {sync_freq:.1f} Hz")

    print("\n[Statistics]")
    stats = harvester.get_coherence_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            if 'time' in key or 'drift' in key:
                print(f"  {key}: {value*1e3:.3f} ms")
            else:
                print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")

    print("\n" + "="*70)
