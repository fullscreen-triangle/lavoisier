#!/usr/bin/env python3
"""
GPU Memory Bandwidth Harvester
===============================

GPU memory bandwidth oscillations = large-scale frequency coupling across entire experiment!

The GPU processes entire experiments in parallel, creating bandwidth oscillations
that reveal experiment-wide coupling patterns.

Author: Lavoisier Project
Date: October 2025
"""

import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque
import threading

try:
    import pynvml
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("[Warning] pynvml not available - GPU monitoring will use simulation")


@dataclass
class GPUBandwidthMeasurement:
    """GPU memory bandwidth measurement."""
    timestamp: float
    memory_used: int  # bytes
    memory_total: int  # bytes
    utilization_gpu: float  # Percentage
    utilization_memory: float  # Percentage
    bandwidth: float  # GB/s


@dataclass
class GPUProcessingTrace:
    """Trace of GPU processing for an experiment."""
    peptide_id: str
    bandwidth_samples: List[float]
    memory_pattern: np.ndarray
    mean_bandwidth: float
    bandwidth_oscillation: float


class GPUOscillationHarvester:
    """
    GPU memory bandwidth oscillations = large-scale frequency coupling across entire experiment!

    GPU parallel processing creates bandwidth oscillations that reveal how
    different peptides/molecules couple across the entire experimental space.
    """

    def __init__(self):
        """Initialize GPU oscillation harvester."""
        self.gpu_available = GPU_AVAILABLE
        self._monitoring = False
        self._monitor_thread = None
        self._bandwidth_buffer = deque(maxlen=10000)

        if self.gpu_available:
            try:
                pynvml.nvmlInit()
                self.device_count = pynvml.nvmlDeviceGetCount()
                self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(self.device_count)]
                print(f"[GPU Oscillation Harvester] Initialized with {self.device_count} GPU(s)")
            except Exception as e:
                print(f"[GPU Oscillation Harvester] Init failed: {e}")
                self.gpu_available = False

        if not self.gpu_available:
            print("[GPU Oscillation Harvester] Running in simulation mode")

    def start_monitoring(self):
        """Start GPU monitoring."""
        if self._monitoring:
            return

        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self._monitor_thread.start()

    def _monitor_loop(self):
        """Background GPU monitoring."""
        while self._monitoring:
            try:
                timestamp = time.perf_counter()

                if self.gpu_available and self.handles:
                    # Real GPU monitoring
                    handle = self.handles[0]  # Primary GPU

                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)

                    memory_used = memory_info.used
                    memory_total = memory_info.total
                    util_gpu = utilization.gpu
                    util_memory = utilization.memory

                    # Estimate bandwidth from utilization
                    # Typical GPU memory bandwidth: ~500 GB/s for modern GPUs
                    max_bandwidth = 500.0  # GB/s
                    bandwidth = max_bandwidth * (util_memory / 100.0)
                else:
                    # Simulation
                    memory_used = int(4 * 1024**3)  # 4 GB
                    memory_total = int(8 * 1024**3)  # 8 GB
                    util_gpu = 50.0 + 20.0 * np.sin(timestamp)
                    util_memory = 40.0 + 15.0 * np.sin(timestamp * 1.5)
                    bandwidth = 200.0 * (util_memory / 100.0)

                measurement = GPUBandwidthMeasurement(
                    timestamp=timestamp,
                    memory_used=memory_used,
                    memory_total=memory_total,
                    utilization_gpu=util_gpu,
                    utilization_memory=util_memory,
                    bandwidth=bandwidth
                )

                self._bandwidth_buffer.append(measurement)

                time.sleep(0.01)  # 100 Hz sampling

            except Exception as e:
                print(f"[GPU Monitor] Error: {e}")
                continue

    def stop_monitoring(self):
        """Stop GPU monitoring."""
        if not self._monitoring:
            return

        self._monitoring = False

        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)

    def harvest_experiment_wide_coupling(
        self,
        processing_function: callable,
        experiment_kb: Dict
    ) -> Dict:
        """
        GPU processing of entire experiment → harvest bandwidth oscillations = coupling matrix!

        Args:
            processing_function: Function to process each peptide on GPU
            experiment_kb: Experiment knowledge base with peptides

        Returns:
            Dictionary with experiment-wide coupling information
        """
        peptides = experiment_kb.get('peptides', [])

        # Start monitoring
        self.start_monitoring()

        gpu_trace = []

        print(f"[GPU Harvesting] Processing {len(peptides)} peptides...")

        # Process all peptides - GPU does parallel processing
        for peptide in peptides:
            # Capture GPU state before processing
            start_time = time.perf_counter()
            start_buffer_len = len(self._bandwidth_buffer)

            # Process peptide (potentially on GPU)
            result = processing_function(peptide)

            # Capture GPU state after processing
            end_time = time.perf_counter()

            # Extract bandwidth measurements during this peptide
            bandwidth_samples = []
            memory_pattern = []

            for measurement in self._bandwidth_buffer:
                if start_time <= measurement.timestamp <= end_time:
                    bandwidth_samples.append(measurement.bandwidth)
                    memory_pattern.append(measurement.utilization_memory)

            if len(bandwidth_samples) > 0:
                mean_bandwidth = np.mean(bandwidth_samples)
                bandwidth_oscillation = np.std(bandwidth_samples) / (mean_bandwidth + 1e-9)
            else:
                mean_bandwidth = 0.0
                bandwidth_oscillation = 0.0

            trace = GPUProcessingTrace(
                peptide_id=peptide.get('id', str(peptide.get('sequence', ''))),
                bandwidth_samples=bandwidth_samples,
                memory_pattern=np.array(memory_pattern),
                mean_bandwidth=mean_bandwidth,
                bandwidth_oscillation=bandwidth_oscillation
            )

            gpu_trace.append(trace)

        # Stop monitoring
        self.stop_monitoring()

        # GPU bandwidth oscillations = experiment-wide coupling!
        coupling_distribution = self._analyze_bandwidth_oscillations(gpu_trace)

        return coupling_distribution

    def _analyze_bandwidth_oscillations(
        self,
        gpu_trace: List[GPUProcessingTrace]
    ) -> Dict:
        """
        Analyze bandwidth oscillations to extract coupling patterns.

        Args:
            gpu_trace: List of GPU processing traces

        Returns:
            Coupling distribution across experiment
        """
        if len(gpu_trace) == 0:
            return {}

        # Build coupling matrix from bandwidth correlations
        n = len(gpu_trace)
        coupling_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    coupling_matrix[i, j] = 1.0
                    continue

                # Coupling from bandwidth similarity
                bandwidth_diff = abs(gpu_trace[i].mean_bandwidth - gpu_trace[j].mean_bandwidth)
                mean_bandwidth = (gpu_trace[i].mean_bandwidth + gpu_trace[j].mean_bandwidth) / 2.0 + 1e-9
                bandwidth_coupling = np.exp(-bandwidth_diff / mean_bandwidth)

                # Coupling from oscillation similarity
                osc_diff = abs(gpu_trace[i].bandwidth_oscillation - gpu_trace[j].bandwidth_oscillation)
                oscillation_coupling = np.exp(-osc_diff / 0.5)

                # Overall coupling
                coupling = (bandwidth_coupling + oscillation_coupling) / 2.0
                coupling_matrix[i, j] = coupling

        # Extract distribution statistics
        bandwidths = [trace.mean_bandwidth for trace in gpu_trace]
        oscillations = [trace.bandwidth_oscillation for trace in gpu_trace]

        return {
            'coupling_matrix': coupling_matrix,
            'mean_coupling': np.mean(coupling_matrix[np.triu_indices_from(coupling_matrix, k=1)]),
            'num_peptides': n,
            'mean_bandwidth': np.mean(bandwidths),
            'std_bandwidth': np.std(bandwidths),
            'mean_oscillation': np.mean(oscillations),
            'std_oscillation': np.std(oscillations),
            'gpu_trace': gpu_trace
        }

    def get_bandwidth_spectrum(self, duration: float = 1.0) -> np.ndarray:
        """
        Get frequency spectrum of GPU bandwidth oscillations.

        Args:
            duration: Time window for spectrum calculation

        Returns:
            Power spectrum of bandwidth oscillations
        """
        if not self._bandwidth_buffer:
            return np.zeros(100)

        current_time = time.perf_counter()
        recent_measurements = [
            m for m in self._bandwidth_buffer
            if current_time - m.timestamp <= duration
        ]

        if len(recent_measurements) < 2:
            return np.zeros(100)

        # Extract bandwidth time series
        bandwidths = np.array([m.bandwidth for m in recent_measurements])

        # Normalize
        bandwidths_normalized = (bandwidths - np.mean(bandwidths)) / (np.std(bandwidths) + 1e-9)

        # FFT
        fft = np.fft.rfft(bandwidths_normalized)
        power_spectrum = np.abs(fft)**2

        # Bin to 100 frequencies
        if len(power_spectrum) > 100:
            indices = np.linspace(0, len(power_spectrum)-1, 100, dtype=int)
            spectrum = power_spectrum[indices]
        else:
            spectrum = np.pad(power_spectrum, (0, 100-len(power_spectrum)))

        return spectrum

    def shutdown(self):
        """Shutdown GPU monitoring and cleanup."""
        self.stop_monitoring()

        if self.gpu_available:
            try:
                pynvml.nvmlShutdown()
            except:
                pass


if __name__ == "__main__":
    print("="*70)
    print("GPU Memory Bandwidth Harvester")
    print("="*70)

    harvester = GPUOscillationHarvester()

    # Mock peptide processing
    def mock_gpu_process(peptide: Dict) -> np.ndarray:
        """Simulate GPU-accelerated peptide processing."""
        # Simulate GPU computation
        sequence_len = len(peptide.get('sequence', 'PEPTIDE'))
        data = np.random.rand(1000 * sequence_len)  # Large array for GPU
        result = np.fft.fft(data)  # FFT as mock GPU operation
        time.sleep(0.05)  # Simulate processing time
        return result

    # Mock experiment
    experiment_kb = {
        'peptides': [
            {'id': f'pep_{i}', 'sequence': f'PEPTIDE{i}', 'mass': 500.0 + i*50}
            for i in range(5)
        ]
    }

    print("\n[Test 1] Harvesting experiment-wide coupling")
    coupling_dist = harvester.harvest_experiment_wide_coupling(
        mock_gpu_process,
        experiment_kb
    )

    print(f"  Peptides processed: {coupling_dist['num_peptides']}")
    print(f"  Mean bandwidth: {coupling_dist['mean_bandwidth']:.2f} GB/s")
    print(f"  Std bandwidth: {coupling_dist['std_bandwidth']:.2f} GB/s")
    print(f"  Mean oscillation: {coupling_dist['mean_oscillation']:.3f}")
    print(f"  Mean coupling: {coupling_dist['mean_coupling']:.3f}")

    print("\n[Test 2] Coupling matrix")
    coupling_matrix = coupling_dist['coupling_matrix']
    print(f"  Matrix shape: {coupling_matrix.shape}")
    print(f"  Diagonal: {np.diag(coupling_matrix)}")

    print("\n[Test 3] Bandwidth spectrum")
    harvester.start_monitoring()
    time.sleep(0.5)  # Collect data
    spectrum = harvester.get_bandwidth_spectrum(duration=0.5)
    harvester.stop_monitoring()

    print(f"  Spectrum shape: {spectrum.shape}")
    print(f"  Peak power: {np.max(spectrum):.2e}")
    print(f"  Mean power: {np.mean(spectrum):.2e}")

    harvester.shutdown()

    print("\n" + "="*70)
