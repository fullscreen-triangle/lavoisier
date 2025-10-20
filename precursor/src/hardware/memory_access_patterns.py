#!/usr/bin/env python3
"""
Memory Access Pattern Harvester
================================

Memory bandwidth oscillations = ensemble dynamics!

Memory access patterns during spectrum processing = phase-lock signatures!

For proteomics: The temporal pattern of memory accesses while processing
fragments reveals the frequency coupling between them. Fragments from the
same collision event create correlated memory access patterns.

Author: Lavoisier Project
Date: October 2025
"""

import time
import numpy as np
import psutil
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import deque
import threading


@dataclass
class MemoryAccessEvent:
    """Single memory access event during processing."""
    timestamp: float
    fragment_mz: float
    fragment_intensity: float
    bytes_accessed: int
    cache_hits: int
    cache_misses: int
    page_faults: int
    access_duration: float


@dataclass
class MemoryAccessTrace:
    """Complete trace of memory accesses during spectrum processing."""
    events: List[MemoryAccessEvent] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    total_bytes_accessed: int = 0
    total_cache_hits: int = 0
    total_cache_misses: int = 0

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def cache_hit_rate(self) -> float:
        total = self.total_cache_hits + self.total_cache_misses
        return self.total_cache_hits / total if total > 0 else 0.0


class MemoryOscillationHarvester:
    """
    Harvests memory access patterns as oscillatory signatures.

    Memory bandwidth oscillations = ensemble dynamics!
    Memory access patterns = phase-lock signatures!
    """

    def __init__(self, sampling_interval: float = 0.0001):
        """
        Initialize memory oscillation harvester.

        Args:
            sampling_interval: Interval for memory monitoring (seconds)
        """
        self.sampling_interval = sampling_interval
        self.traces: List[MemoryAccessTrace] = []
        self._monitoring = False
        self._monitor_thread = None
        self._current_trace: Optional[MemoryAccessTrace] = None
        self._memory_buffer = deque(maxlen=10000)

        # Baseline memory stats
        self.baseline_memory = psutil.virtual_memory()

        print("[Memory Oscillation Harvester] Initialized")
        print(f"  Sampling interval: {sampling_interval*1e6:.1f} μs")

    def start_monitoring(self):
        """Start continuous memory monitoring."""
        if self._monitoring:
            return

        self._monitoring = True
        self._current_trace = MemoryAccessTrace()
        self._current_trace.start_time = time.perf_counter()

        # Start monitor thread
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self._monitor_thread.start()

    def _monitor_loop(self):
        """Background thread for memory monitoring."""
        while self._monitoring:
            try:
                timestamp = time.perf_counter()
                mem = psutil.virtual_memory()

                # Store memory snapshot
                self._memory_buffer.append({
                    'timestamp': timestamp,
                    'available': mem.available,
                    'used': mem.used,
                    'percent': mem.percent
                })

                time.sleep(self.sampling_interval)
            except Exception:
                continue

    def stop_monitoring(self):
        """Stop memory monitoring."""
        if not self._monitoring:
            return

        self._monitoring = False

        if self._current_trace:
            self._current_trace.end_time = time.perf_counter()
            self.traces.append(self._current_trace)
            self._current_trace = None

        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)

    def harvest_phase_locks(
        self,
        processing_function: callable,
        fragments: List[Dict],
        spectrum_metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Memory access patterns during spectrum processing = phase-lock signatures!

        Args:
            processing_function: Function that processes fragments
            fragments: List of fragment dictionaries with 'mz', 'intensity'
            spectrum_metadata: Optional metadata about the spectrum

        Returns:
            Dictionary with phase-lock signatures from memory patterns
        """
        # Start monitoring
        self.start_monitoring()

        memory_trace = []
        start_time = time.perf_counter()

        # Get initial memory state
        mem_start = psutil.virtual_memory()

        # Process spectrum - memory accesses are REAL oscillations
        for i, fragment in enumerate(fragments):
            # Timestamp before processing
            t_before = time.perf_counter()
            mem_before = psutil.virtual_memory()

            # Process fragment
            result = processing_function(fragment)

            # Timestamp after processing
            t_after = time.perf_counter()
            mem_after = psutil.virtual_memory()

            # Calculate memory access metrics
            bytes_accessed = abs(mem_after.used - mem_before.used)
            access_duration = t_after - t_before

            # Estimate cache performance from memory change speed
            # Fast changes = cache hits, slow = cache misses
            access_speed = bytes_accessed / (access_duration + 1e-9)
            cache_hit_threshold = 1e9  # ~1 GB/s

            if access_speed > cache_hit_threshold:
                cache_hits = 1
                cache_misses = 0
            else:
                cache_hits = 0
                cache_misses = 1

            event = MemoryAccessEvent(
                timestamp=t_before,
                fragment_mz=fragment.get('mz', 0.0),
                fragment_intensity=fragment.get('intensity', 0.0),
                bytes_accessed=bytes_accessed,
                cache_hits=cache_hits,
                cache_misses=cache_misses,
                page_faults=0,  # Would need OS-specific API
                access_duration=access_duration
            )

            memory_trace.append(event)

            if self._current_trace:
                self._current_trace.events.append(event)
                self._current_trace.total_bytes_accessed += bytes_accessed
                self._current_trace.total_cache_hits += cache_hits
                self._current_trace.total_cache_misses += cache_misses

        # Stop monitoring
        self.stop_monitoring()

        end_time = time.perf_counter()

        # Memory access pattern = frequency coupling matrix!
        coupling_matrix = self.compute_temporal_correlations(memory_trace)

        # Extract oscillatory features
        access_times = np.array([e.timestamp - start_time for e in memory_trace])
        access_durations = np.array([e.access_duration for e in memory_trace])
        bytes_accessed = np.array([e.bytes_accessed for e in memory_trace])

        # Compute oscillation frequency from access pattern
        if len(access_times) > 1:
            access_intervals = np.diff(access_times)
            mean_interval = np.mean(access_intervals)
            oscillation_freq = 1.0 / mean_interval if mean_interval > 0 else 0.0
        else:
            oscillation_freq = 0.0

        # Phase coherence from regularity of access pattern
        if len(access_intervals) > 1:
            phase_coherence = 1.0 / (1.0 + np.std(access_intervals) / (mean_interval + 1e-9))
        else:
            phase_coherence = 1.0

        return {
            'coupling_matrix': coupling_matrix,
            'memory_trace': memory_trace,
            'oscillation_frequency': oscillation_freq,
            'phase_coherence': phase_coherence,
            'total_duration': end_time - start_time,
            'num_fragments': len(memory_trace),
            'mean_access_duration': np.mean(access_durations),
            'std_access_duration': np.std(access_durations),
            'total_bytes_accessed': np.sum(bytes_accessed),
            'cache_hit_rate': sum(e.cache_hits for e in memory_trace) / len(memory_trace)
        }

    def compute_temporal_correlations(
        self,
        memory_trace: List[MemoryAccessEvent]
    ) -> np.ndarray:
        """
        Compute frequency coupling matrix from temporal correlations.

        Fragments accessed close in time = strongly coupled.
        Similar access patterns = frequency-coupled.
        """
        n = len(memory_trace)
        if n == 0:
            return np.array([])

        coupling_matrix = np.zeros((n, n))

        # Extract features for each access
        timestamps = np.array([e.timestamp for e in memory_trace])
        durations = np.array([e.access_duration for e in memory_trace])
        bytes_arr = np.array([e.bytes_accessed for e in memory_trace])

        for i in range(n):
            for j in range(n):
                if i == j:
                    coupling_matrix[i, j] = 1.0
                    continue

                # Temporal proximity coupling
                time_diff = abs(timestamps[i] - timestamps[j])
                temporal_coupling = np.exp(-time_diff / 0.001)  # 1 ms decay

                # Duration similarity coupling
                duration_diff = abs(durations[i] - durations[j])
                duration_coupling = np.exp(-duration_diff / np.mean(durations))

                # Memory size similarity coupling
                bytes_diff = abs(bytes_arr[i] - bytes_arr[j])
                bytes_coupling = np.exp(-bytes_diff / np.mean(bytes_arr))

                # Overall coupling
                coupling = (temporal_coupling + duration_coupling + bytes_coupling) / 3.0
                coupling_matrix[i, j] = coupling

        return coupling_matrix

    def extract_ensemble_dynamics(
        self,
        coupling_matrix: np.ndarray,
        memory_trace: List[MemoryAccessEvent]
    ) -> Dict:
        """
        Extract ensemble dynamics from memory access patterns.

        Memory bandwidth oscillations = molecular ensemble statistics!
        """
        if len(memory_trace) == 0:
            return {}

        # Ensemble size = number of strongly coupled accesses
        coupling_threshold = 0.7
        strong_couplings = coupling_matrix > coupling_threshold
        ensemble_size = np.sum(strong_couplings) / len(memory_trace)

        # Ensemble coherence = average coupling strength
        ensemble_coherence = np.mean(coupling_matrix[np.triu_indices_from(coupling_matrix, k=1)])

        # Temporal spread of ensemble
        timestamps = np.array([e.timestamp for e in memory_trace])
        temporal_spread = np.max(timestamps) - np.min(timestamps)

        # Memory bandwidth oscillation
        bytes_accessed = np.array([e.bytes_accessed for e in memory_trace])
        if len(bytes_accessed) > 1:
            bandwidth_mean = np.mean(bytes_accessed) / temporal_spread if temporal_spread > 0 else 0
            bandwidth_std = np.std(bytes_accessed) / temporal_spread if temporal_spread > 0 else 0
        else:
            bandwidth_mean = 0.0
            bandwidth_std = 0.0

        return {
            'ensemble_size': ensemble_size,
            'ensemble_coherence': ensemble_coherence,
            'temporal_spread': temporal_spread,
            'bandwidth_mean': bandwidth_mean,
            'bandwidth_std': bandwidth_std,
            'bandwidth_oscillation_amplitude': bandwidth_std / (bandwidth_mean + 1e-9)
        }

    def get_memory_oscillation_spectrum(self, duration: float = 1.0) -> np.ndarray:
        """
        Get frequency spectrum of memory oscillations.

        Args:
            duration: Time window for spectrum calculation

        Returns:
            Frequency spectrum of memory usage oscillations
        """
        if not self._memory_buffer:
            return np.zeros(100)

        current_time = time.perf_counter()
        recent_memory = [
            m for m in self._memory_buffer
            if current_time - m['timestamp'] <= duration
        ]

        if len(recent_memory) < 2:
            return np.zeros(100)

        # Extract time series
        times = np.array([m['timestamp'] for m in recent_memory])
        memory_used = np.array([m['used'] for m in recent_memory])

        # Compute FFT
        memory_normalized = (memory_used - np.mean(memory_used)) / (np.std(memory_used) + 1e-9)
        fft = np.fft.rfft(memory_normalized)
        power_spectrum = np.abs(fft)**2

        # Bin into 100 frequency bins
        if len(power_spectrum) > 100:
            indices = np.linspace(0, len(power_spectrum)-1, 100, dtype=int)
            spectrum = power_spectrum[indices]
        else:
            spectrum = np.pad(power_spectrum, (0, 100-len(power_spectrum)))

        return spectrum


if __name__ == "__main__":
    print("="*70)
    print("Memory Access Pattern Harvester")
    print("="*70)

    harvester = MemoryOscillationHarvester(sampling_interval=0.0001)

    # Mock fragment processing
    def mock_process_fragment(fragment: Dict) -> float:
        """Simulate fragment processing with memory allocation."""
        # Allocate some memory
        data = np.random.rand(1000)  # ~8 KB
        result = np.sum(data * fragment['intensity'])
        del data
        return result

    # Mock fragments
    fragments = [
        {'mz': 100.0 + i*50, 'intensity': 100.0 - i*10}
        for i in range(10)
    ]

    print("\n[Test 1] Harvesting phase-locks from memory patterns")
    phase_lock_result = harvester.harvest_phase_locks(
        mock_process_fragment,
        fragments
    )

    print(f"  Oscillation frequency: {phase_lock_result['oscillation_frequency']:.2f} Hz")
    print(f"  Phase coherence: {phase_lock_result['phase_coherence']:.3f}")
    print(f"  Total duration: {phase_lock_result['total_duration']*1e3:.3f} ms")
    print(f"  Mean access duration: {phase_lock_result['mean_access_duration']*1e6:.3f} μs")
    print(f"  Total bytes: {phase_lock_result['total_bytes_accessed']/1024:.1f} KB")
    print(f"  Cache hit rate: {phase_lock_result['cache_hit_rate']:.3f}")

    print("\n[Test 2] Coupling matrix")
    coupling_matrix = phase_lock_result['coupling_matrix']
    print(f"  Matrix shape: {coupling_matrix.shape}")
    print(f"  Mean coupling: {np.mean(coupling_matrix):.3f}")
    print(f"  Max off-diagonal: {np.max(coupling_matrix[np.triu_indices_from(coupling_matrix, k=1)]):.3f}")

    print("\n[Test 3] Ensemble dynamics")
    ensemble = harvester.extract_ensemble_dynamics(
        coupling_matrix,
        phase_lock_result['memory_trace']
    )

    for key, value in ensemble.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")

    print("\n[Test 4] Memory oscillation spectrum")
    spectrum = harvester.get_memory_oscillation_spectrum(duration=0.5)
    print(f"  Spectrum shape: {spectrum.shape}")
    print(f"  Peak power: {np.max(spectrum):.2e}")
    print(f"  Mean power: {np.mean(spectrum):.2e}")

    print("\n" + "="*70)
