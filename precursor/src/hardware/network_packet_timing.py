#!/usr/bin/env python3
"""
Network Packet Timing Harvester
================================

Network packet arrival times = molecular ensemble statistics!

- Packet jitter = collision event variation
- Bandwidth oscillations = ensemble size fluctuations
- Latency = phase propagation time

For proteomics: Network timing during batch processing reveals
ensemble formation dynamics and inter-peptide coupling.

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
class NetworkPacketTiming:
    """Network packet timing measurement."""
    timestamp: float
    bytes_sent: int
    bytes_recv: int
    packets_sent: int
    packets_recv: int
    latency: float  # Estimated from timing
    jitter: float   # Variation in arrival times


@dataclass
class EnsembleStatistics:
    """Ensemble statistics derived from network timing."""
    peptide_id: str
    ensemble_size: float
    coherence: float
    timing_signature: List[float]
    mean_latency: float
    jitter: float
    bandwidth_oscillation: float


class NetworkOscillationHarvester:
    """
    Network packet arrival times = molecular ensemble statistics!

    - Packet jitter = collision event variation
    - Bandwidth oscillations = ensemble size fluctuations
    - Latency = phase propagation time
    """

    def __init__(self, sampling_interval: float = 0.01):
        """
        Initialize network oscillation harvester.

        Args:
            sampling_interval: Network monitoring interval (seconds)
        """
        self.sampling_interval = sampling_interval
        self._monitoring = False
        self._monitor_thread = None
        self._packet_buffer = deque(maxlen=10000)
        self.baseline_network = psutil.net_io_counters()

        print("[Network Oscillation Harvester] Initialized")
        print(f"  Sampling interval: {sampling_interval*1e3:.1f} ms")

    def start_monitoring(self):
        """Start continuous network monitoring."""
        if self._monitoring:
            return

        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self._monitor_thread.start()

    def _monitor_loop(self):
        """Background network monitoring."""
        last_net = psutil.net_io_counters()
        last_time = time.perf_counter()

        while self._monitoring:
            try:
                time.sleep(self.sampling_interval)

                current_time = time.perf_counter()
                current_net = psutil.net_io_counters()

                elapsed = current_time - last_time

                # Calculate delta
                bytes_sent_delta = current_net.bytes_sent - last_net.bytes_sent
                bytes_recv_delta = current_net.bytes_recv - last_net.bytes_recv
                packets_sent_delta = current_net.packets_sent - last_net.packets_sent
                packets_recv_delta = current_net.packets_recv - last_net.packets_recv

                # Estimate latency from packet rate
                total_packets = packets_sent_delta + packets_recv_delta
                if total_packets > 0:
                    latency = elapsed / total_packets
                else:
                    latency = elapsed

                timing = NetworkPacketTiming(
                    timestamp=current_time,
                    bytes_sent=bytes_sent_delta,
                    bytes_recv=bytes_recv_delta,
                    packets_sent=packets_sent_delta,
                    packets_recv=packets_recv_delta,
                    latency=latency,
                    jitter=0.0  # Will be computed from buffer
                )

                self._packet_buffer.append(timing)

                last_net = current_net
                last_time = current_time

            except Exception:
                continue

    def stop_monitoring(self):
        """Stop network monitoring."""
        if not self._monitoring:
            return

        self._monitoring = False

        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)

    def harvest_ensemble_dynamics(
        self,
        processing_function: callable,
        peptide_batch: List[Dict]
    ) -> List[EnsembleStatistics]:
        """
        Process peptide batch - network timing = ensemble behavior!

        Args:
            processing_function: Function to process each peptide
            peptide_batch: List of peptide dictionaries

        Returns:
            List of ensemble statistics for each peptide
        """
        ensemble_stats = []

        # Start monitoring
        self.start_monitoring()

        for peptide in peptide_batch:
            # Network timing during processing
            packet_times = self._monitor_network_during_processing(
                processing_function,
                peptide
            )

            # Packet arrival pattern = ensemble formation
            ensemble_size = self._estimate_ensemble_from_packet_pattern(packet_times)
            coherence = self._compute_packet_coherence(packet_times)
            bandwidth_osc = self._compute_bandwidth_oscillation(packet_times)

            # Compute jitter
            if len(packet_times) > 1:
                intervals = np.diff(packet_times)
                jitter = np.std(intervals)
                mean_latency = np.mean(intervals)
            else:
                jitter = 0.0
                mean_latency = 0.0

            stats = EnsembleStatistics(
                peptide_id=peptide.get('id', str(peptide.get('sequence', ''))),
                ensemble_size=ensemble_size,
                coherence=coherence,
                timing_signature=packet_times,
                mean_latency=mean_latency,
                jitter=jitter,
                bandwidth_oscillation=bandwidth_osc
            )

            ensemble_stats.append(stats)

        # Stop monitoring
        self.stop_monitoring()

        return ensemble_stats

    def _monitor_network_during_processing(
        self,
        processing_function: callable,
        peptide: Dict
    ) -> List[float]:
        """
        Monitor network timing during peptide processing.

        Returns:
            List of packet arrival times
        """
        start_time = time.perf_counter()
        start_buffer_size = len(self._packet_buffer)

        # Process peptide
        result = processing_function(peptide)

        end_time = time.perf_counter()

        # Extract packet times from buffer
        packet_times = []
        for timing in self._packet_buffer:
            if start_time <= timing.timestamp <= end_time:
                packet_times.append(timing.timestamp - start_time)

        return packet_times

    def _estimate_ensemble_from_packet_pattern(
        self,
        packet_times: List[float]
    ) -> float:
        """
        Estimate ensemble size from packet arrival pattern.

        Bursty arrivals = large ensemble
        Uniform arrivals = small ensemble
        """
        if len(packet_times) < 2:
            return 1.0

        intervals = np.diff(packet_times)

        # Burstiness metric
        if np.mean(intervals) > 0:
            burstiness = np.std(intervals) / np.mean(intervals)
        else:
            burstiness = 0.0

        # Map burstiness to ensemble size (heuristic)
        ensemble_size = 1.0 + 10.0 * burstiness

        return ensemble_size

    def _compute_packet_coherence(
        self,
        packet_times: List[float]
    ) -> float:
        """
        Compute coherence from packet timing regularity.

        Regular timing = high coherence
        Irregular timing = low coherence
        """
        if len(packet_times) < 2:
            return 1.0

        intervals = np.diff(packet_times)

        if np.mean(intervals) > 0:
            regularity = 1.0 / (1.0 + np.std(intervals) / np.mean(intervals))
        else:
            regularity = 0.0

        return regularity

    def _compute_bandwidth_oscillation(
        self,
        packet_times: List[float]
    ) -> float:
        """
        Compute bandwidth oscillation amplitude.

        Large oscillations = dynamic ensemble
        Small oscillations = stable ensemble
        """
        if len(packet_times) < 3:
            return 0.0

        # Compute instantaneous bandwidth
        intervals = np.diff(packet_times)
        bandwidth = 1.0 / (intervals + 1e-9)  # Packets per second

        # Oscillation = normalized std
        if np.mean(bandwidth) > 0:
            oscillation = np.std(bandwidth) / np.mean(bandwidth)
        else:
            oscillation = 0.0

        return oscillation

    def compute_inter_peptide_coupling(
        self,
        ensemble_stats: List[EnsembleStatistics]
    ) -> np.ndarray:
        """
        Compute coupling matrix between peptides based on network timing.

        Similar timing patterns = coupled peptides
        """
        n = len(ensemble_stats)
        if n == 0:
            return np.array([])

        coupling_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    coupling_matrix[i, j] = 1.0
                    continue

                # Similarity in ensemble size
                size_diff = abs(ensemble_stats[i].ensemble_size - ensemble_stats[j].ensemble_size)
                size_coupling = np.exp(-size_diff / 5.0)

                # Similarity in coherence
                coherence_diff = abs(ensemble_stats[i].coherence - ensemble_stats[j].coherence)
                coherence_coupling = np.exp(-coherence_diff / 0.5)

                # Similarity in jitter
                jitter_diff = abs(ensemble_stats[i].jitter - ensemble_stats[j].jitter)
                mean_jitter = (ensemble_stats[i].jitter + ensemble_stats[j].jitter) / 2.0 + 1e-9
                jitter_coupling = np.exp(-jitter_diff / mean_jitter)

                # Overall coupling
                coupling = (size_coupling + coherence_coupling + jitter_coupling) / 3.0
                coupling_matrix[i, j] = coupling

        return coupling_matrix

    def get_network_oscillation_spectrum(self, duration: float = 1.0) -> np.ndarray:
        """
        Get frequency spectrum of network packet arrivals.

        Args:
            duration: Time window for spectrum calculation

        Returns:
            Power spectrum of packet arrival times
        """
        if not self._packet_buffer:
            return np.zeros(100)

        current_time = time.perf_counter()
        recent_packets = [
            p for p in self._packet_buffer
            if current_time - p.timestamp <= duration
        ]

        if len(recent_packets) < 2:
            return np.zeros(100)

        # Extract packet rate time series
        times = np.array([p.timestamp for p in recent_packets])
        packets_total = np.array([p.packets_sent + p.packets_recv for p in recent_packets])

        # Normalize
        packets_normalized = (packets_total - np.mean(packets_total)) / (np.std(packets_total) + 1e-9)

        # FFT
        fft = np.fft.rfft(packets_normalized)
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
    print("Network Packet Timing Harvester")
    print("="*70)

    harvester = NetworkOscillationHarvester(sampling_interval=0.01)

    # Mock peptide processing
    def mock_process_peptide(peptide: Dict) -> float:
        """Simulate peptide processing."""
        time.sleep(0.01)  # Simulate work
        result = np.sum([ord(c) for c in peptide.get('sequence', 'ABC')]) * peptide.get('intensity', 1.0)
        return result

    # Mock peptide batch
    peptides = [
        {'id': f'pep_{i}', 'sequence': f'PEPTIDE{i}', 'intensity': 100.0 - i*5}
        for i in range(5)
    ]

    print("\n[Test 1] Harvesting ensemble dynamics")
    ensemble_stats = harvester.harvest_ensemble_dynamics(
        mock_process_peptide,
        peptides
    )

    for stats in ensemble_stats:
        print(f"\n  Peptide: {stats.peptide_id}")
        print(f"    Ensemble size: {stats.ensemble_size:.3f}")
        print(f"    Coherence: {stats.coherence:.3f}")
        print(f"    Mean latency: {stats.mean_latency*1e3:.3f} ms")
        print(f"    Jitter: {stats.jitter*1e6:.3f} μs")
        print(f"    Bandwidth osc: {stats.bandwidth_oscillation:.3f}")

    print("\n[Test 2] Inter-peptide coupling matrix")
    coupling_matrix = harvester.compute_inter_peptide_coupling(ensemble_stats)
    print(f"  Matrix shape: {coupling_matrix.shape}")
    print(f"  Mean coupling: {np.mean(coupling_matrix):.3f}")

    print("\n[Test 3] Network oscillation spectrum")
    spectrum = harvester.get_network_oscillation_spectrum(duration=0.5)
    print(f"  Spectrum shape: {spectrum.shape}")
    print(f"  Peak power: {np.max(spectrum):.2e}")

    print("\n" + "="*70)
