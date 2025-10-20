#!/usr/bin/env python3
"""
Disk I/O Pattern Harvester
===========================

Disk I/O timing = peptide fragmentation sequences!

- Sequential reads = linear fragmentation
- Random access = complex fragmentation patterns
- I/O latency = fragmentation kinetics

Author: Lavoisier Project
Date: October 2025
"""

import time
import numpy as np
import psutil
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque
import threading


@dataclass
class DiskIOEvent:
    """Single disk I/O event."""
    timestamp: float
    read_bytes: int
    write_bytes: int
    read_count: int
    write_count: int
    read_time: float  # ms
    write_time: float  # ms
    is_sequential: bool


@dataclass
class FragmentationPattern:
    """Fragmentation pattern derived from I/O."""
    fragment_mz: float
    io_latency: float
    sequential_score: float
    fragmentation_order: int


class DiskIOHarvester:
    """
    Disk I/O timing = peptide fragmentation sequences!

    - Sequential reads = linear fragmentation
    - Random access = complex fragmentation patterns
    - I/O latency = fragmentation kinetics
    """

    def __init__(self, sampling_interval: float = 0.01):
        """
        Initialize disk I/O harvester.

        Args:
            sampling_interval: I/O monitoring interval (seconds)
        """
        self.sampling_interval = sampling_interval
        self._monitoring = False
        self._monitor_thread = None
        self._io_buffer = deque(maxlen=10000)
        self.baseline_io = psutil.disk_io_counters()

        print("[Disk I/O Harvester] Initialized")
        print(f"  Sampling interval: {sampling_interval*1e3:.1f} ms")

    def start_monitoring(self):
        """Start disk I/O monitoring."""
        if self._monitoring:
            return

        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self._monitor_thread.start()

    def _monitor_loop(self):
        """Background I/O monitoring."""
        last_io = psutil.disk_io_counters()
        last_time = time.perf_counter()
        last_read_bytes = last_io.read_bytes
        last_write_bytes = last_io.write_bytes

        while self._monitoring:
            try:
                time.sleep(self.sampling_interval)

                current_time = time.perf_counter()
                current_io = psutil.disk_io_counters()

                elapsed = current_time - last_time

                # Calculate deltas
                read_bytes = current_io.read_bytes - last_io.read_bytes
                write_bytes = current_io.write_bytes - last_io.write_bytes
                read_count = current_io.read_count - last_io.read_count
                write_count = current_io.write_count - last_io.write_count
                read_time = current_io.read_time - last_io.read_time
                write_time = current_io.write_time - last_io.write_time

                # Detect sequential vs random access
                # Sequential if large bytes per operation
                if read_count > 0:
                    bytes_per_read = read_bytes / read_count
                    is_sequential = bytes_per_read > 64 * 1024  # > 64 KB per read = sequential
                else:
                    is_sequential = False

                event = DiskIOEvent(
                    timestamp=current_time,
                    read_bytes=read_bytes,
                    write_bytes=write_bytes,
                    read_count=read_count,
                    write_count=write_count,
                    read_time=read_time,
                    write_time=write_time,
                    is_sequential=is_sequential
                )

                self._io_buffer.append(event)

                last_io = current_io
                last_time = current_time

            except Exception as e:
                print(f"[Disk I/O Monitor] Error: {e}")
                continue

    def stop_monitoring(self):
        """Stop I/O monitoring."""
        if not self._monitoring:
            return

        self._monitoring = False

        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)

    def harvest_fragmentation_patterns(
        self,
        processing_function: callable,
        spectrum_data: Dict
    ) -> List[FragmentationPattern]:
        """
        Monitor disk I/O while reading MS/MS data = fragmentation sequence timing!

        Args:
            processing_function: Function that reads/processes spectrum
            spectrum_data: Spectrum data including fragments

        Returns:
            List of fragmentation patterns
        """
        # Start monitoring
        self.start_monitoring()

        # Process spectrum - I/O happens during this
        start_time = time.perf_counter()
        start_buffer_len = len(self._io_buffer)

        result = processing_function(spectrum_data)

        end_time = time.perf_counter()

        # Extract I/O events during processing
        io_events = []
        for event in self._io_buffer:
            if start_time <= event.timestamp <= end_time:
                io_events.append(event)

        # Stop monitoring
        self.stop_monitoring()

        # I/O pattern = fragmentation sequence
        fragments = spectrum_data.get('fragments', [])
        fragmentation_patterns = self._extract_fragmentation_from_io(
            fragments,
            io_events
        )

        return fragmentation_patterns

    def _extract_fragmentation_from_io(
        self,
        fragments: List[Dict],
        io_events: List[DiskIOEvent]
    ) -> List[FragmentationPattern]:
        """
        Extract fragmentation patterns from I/O events.

        I/O timing encodes fragmentation sequence!
        """
        if len(fragments) == 0 or len(io_events) == 0:
            return []

        patterns = []

        # Sort fragments by m/z
        fragments_sorted = sorted(fragments, key=lambda f: f.get('mz', 0.0))

        # Assign I/O events to fragments
        events_per_fragment = max(1, len(io_events) // len(fragments_sorted))

        for i, fragment in enumerate(fragments_sorted):
            # Get I/O events for this fragment
            event_start = i * events_per_fragment
            event_end = min((i + 1) * events_per_fragment, len(io_events))
            fragment_events = io_events[event_start:event_end]

            if len(fragment_events) == 0:
                continue

            # Compute I/O latency
            latencies = []
            sequential_count = 0

            for event in fragment_events:
                if event.read_count > 0:
                    latency = event.read_time / event.read_count
                    latencies.append(latency)

                if event.is_sequential:
                    sequential_count += 1

            if len(latencies) > 0:
                io_latency = np.mean(latencies)
            else:
                io_latency = 0.0

            # Sequential score
            sequential_score = sequential_count / len(fragment_events) if fragment_events else 0.0

            pattern = FragmentationPattern(
                fragment_mz=fragment.get('mz', 0.0),
                io_latency=io_latency,
                sequential_score=sequential_score,
                fragmentation_order=i
            )

            patterns.append(pattern)

        return patterns

    def analyze_fragmentation_kinetics(
        self,
        patterns: List[FragmentationPattern]
    ) -> Dict:
        """
        Analyze fragmentation kinetics from I/O patterns.

        I/O latency = fragmentation kinetics!
        """
        if len(patterns) == 0:
            return {}

        latencies = np.array([p.io_latency for p in patterns])
        sequential_scores = np.array([p.sequential_score for p in patterns])

        # Linear fragmentation = high sequential scores, uniform latencies
        is_linear = np.mean(sequential_scores) > 0.7

        # Complex fragmentation = low sequential scores, varying latencies
        is_complex = np.mean(sequential_scores) < 0.3 and np.std(latencies) > np.mean(latencies) * 0.5

        # Fragmentation rate from latency trend
        if len(latencies) > 1:
            # Fit linear trend to latencies vs order
            orders = np.arange(len(latencies))
            slope, intercept = np.polyfit(orders, latencies, 1)
            fragmentation_rate = -slope  # Negative slope = faster over time
        else:
            fragmentation_rate = 0.0

        return {
            'num_fragments': len(patterns),
            'mean_latency': float(np.mean(latencies)),
            'std_latency': float(np.std(latencies)),
            'mean_sequential_score': float(np.mean(sequential_scores)),
            'is_linear_fragmentation': is_linear,
            'is_complex_fragmentation': is_complex,
            'fragmentation_rate': fragmentation_rate,
            'patterns': patterns
        }

    def get_io_oscillation_spectrum(self, duration: float = 1.0) -> np.ndarray:
        """
        Get frequency spectrum of disk I/O oscillations.

        Args:
            duration: Time window for spectrum calculation

        Returns:
            Power spectrum of I/O oscillations
        """
        if not self._io_buffer:
            return np.zeros(100)

        current_time = time.perf_counter()
        recent_events = [
            e for e in self._io_buffer
            if current_time - e.timestamp <= duration
        ]

        if len(recent_events) < 2:
            return np.zeros(100)

        # Extract I/O rate time series
        read_bytes = np.array([e.read_bytes for e in recent_events])

        # Normalize
        read_normalized = (read_bytes - np.mean(read_bytes)) / (np.std(read_bytes) + 1e-9)

        # FFT
        fft = np.fft.rfft(read_normalized)
        power_spectrum = np.abs(fft)**2

        # Bin
        if len(power_spectrum) > 100:
            indices = np.linspace(0, len(power_spectrum)-1, 100, dtype=int)
            spectrum = power_spectrum[indices]
        else:
            spectrum = np.pad(power_spectrum, (0, 100-len(power_spectrum)))

        return spectrum


if __name__ == "__main__":
    print("="*70)
    print("Disk I/O Pattern Harvester")
    print("="*70)

    harvester = DiskIOHarvester(sampling_interval=0.01)

    # Mock spectrum processing
    def mock_read_spectrum(spectrum_data: Dict) -> Dict:
        """Simulate reading spectrum from disk."""
        # Simulate I/O operations
        fragments = spectrum_data.get('fragments', [])

        result = []
        for fragment in fragments:
            # Simulate reading fragment data
            time.sleep(0.01)
            data = np.random.rand(100)
            result.append(np.sum(data))

        return {'processed': len(result)}

    # Mock spectrum
    spectrum_data = {
        'fragments': [
            {'mz': 100.0 + i*50, 'intensity': 100.0 - i*10}
            for i in range(8)
        ]
    }

    print("\n[Test 1] Harvesting fragmentation patterns")
    patterns = harvester.harvest_fragmentation_patterns(
        mock_read_spectrum,
        spectrum_data
    )

    print(f"  Extracted {len(patterns)} fragmentation patterns")
    for i, pattern in enumerate(patterns[:5]):  # Show first 5
        print(f"    Fragment {i}: m/z={pattern.fragment_mz:.1f}, "
              f"latency={pattern.io_latency:.3f}ms, "
              f"sequential={pattern.sequential_score:.2f}")

    print("\n[Test 2] Analyzing fragmentation kinetics")
    kinetics = harvester.analyze_fragmentation_kinetics(patterns)

    for key, value in kinetics.items():
        if key == 'patterns':
            continue
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")

    print("\n[Test 3] I/O oscillation spectrum")
    harvester.start_monitoring()
    time.sleep(0.5)
    spectrum = harvester.get_io_oscillation_spectrum(duration=0.5)
    harvester.stop_monitoring()

    print(f"  Spectrum shape: {spectrum.shape}")
    print(f"  Peak power: {np.max(spectrum):.2e}")
    print(f"  Mean power: {np.mean(spectrum):.2e}")

    print("\n" + "="*70)
