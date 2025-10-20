"""
Hardware Integration Module - Python Bindings

High-performance hardware harvesting using Rust backend for
computational hardware oscillation capture and analysis.
"""

import asyncio
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass

try:
    # Import the Rust module
    import lavoisier_computational
    RUST_AVAILABLE = True
except ImportError:
    # Fallback to pure Python implementation
    RUST_AVAILABLE = False
    import psutil
    import time
    import threading
    from collections import deque


@dataclass
class HardwareOscillation:
    """Represents a captured hardware oscillation pattern"""
    timestamp: float
    source: str
    frequency: float
    amplitude: float
    phase: float
    metadata: Dict[str, Any]


class HardwareHarvester:
    """
    Advanced hardware oscillation harvesting system.

    Uses high-performance Rust backend when available, falls back to Python.
    """

    def __init__(self, sample_rate: float = 1000.0, buffer_size: int = 10000):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size

        if RUST_AVAILABLE:
            self._rust_harvester = lavoisier_computational.PyHardwareHarvester(
                sample_rate, buffer_size
            )
            self._use_rust = True
        else:
            self._use_rust = False
            self._initialize_python_fallback()

    def _initialize_python_fallback(self):
        """Initialize Python fallback implementation"""
        self.oscillation_buffer = deque(maxlen=self.buffer_size)
        self.is_harvesting = False
        self.harvest_thread = None

    async def start_harvesting(self) -> None:
        """Begin continuous hardware oscillation harvesting"""
        if self._use_rust:
            self._rust_harvester.start_harvesting()
        else:
            await self._start_python_harvesting()

    async def _start_python_harvesting(self) -> None:
        """Python fallback harvesting implementation"""
        if self.is_harvesting:
            return

        self.is_harvesting = True
        self.harvest_thread = threading.Thread(target=self._harvest_loop, daemon=True)
        self.harvest_thread.start()

    def _harvest_loop(self) -> None:
        """Main harvesting loop for Python fallback"""
        while self.is_harvesting:
            try:
                timestamp = time.time()

                # Capture CPU oscillations
                cpu_usage = psutil.cpu_percent(interval=None)
                cpu_osc = HardwareOscillation(
                    timestamp=timestamp,
                    source="cpu",
                    frequency=cpu_usage,
                    amplitude=cpu_usage / 100.0,
                    phase=(timestamp * cpu_usage) % 1.0,
                    metadata={"usage": cpu_usage}
                )
                self.oscillation_buffer.append(cpu_osc)

                # Capture memory oscillations
                memory = psutil.virtual_memory()
                mem_osc = HardwareOscillation(
                    timestamp=timestamp,
                    source="memory",
                    frequency=memory.percent,
                    amplitude=memory.percent / 100.0,
                    phase=(memory.available / memory.total),
                    metadata={"percent": memory.percent}
                )
                self.oscillation_buffer.append(mem_osc)

                time.sleep(1.0 / self.sample_rate)

            except Exception:
                continue

    def stop_harvesting(self) -> None:
        """Stop hardware oscillation harvesting"""
        if self._use_rust:
            self._rust_harvester.stop_harvesting()
        else:
            self.is_harvesting = False
            if self.harvest_thread:
                self.harvest_thread.join(timeout=1.0)

    def get_oscillation_spectrum(self, duration: float = 1.0) -> np.ndarray:
        """Get frequency spectrum of recent oscillations"""
        if self._use_rust:
            spectrum = self._rust_harvester.get_oscillation_spectrum(duration)
            return np.array(spectrum)
        else:
            return self._get_python_spectrum(duration)

    def _get_python_spectrum(self, duration: float) -> np.ndarray:
        """Python fallback spectrum calculation"""
        if not self.oscillation_buffer:
            return np.zeros(100)

        current_time = time.time()
        recent_oscillations = [
            osc for osc in self.oscillation_buffer
            if current_time - osc.timestamp <= duration
        ]

        if not recent_oscillations:
            return np.zeros(100)

        frequencies = np.array([osc.frequency for osc in recent_oscillations])
        amplitudes = np.array([osc.amplitude for osc in recent_oscillations])

        spectrum, _ = np.histogram(frequencies, weights=amplitudes, bins=100)
        return spectrum

    def get_resonance_signature(self, target_frequency: float, tolerance: float = 0.1) -> float:
        """Get resonance signature strength at target frequency"""
        if self._use_rust:
            return self._rust_harvester.get_resonance_signature(target_frequency, tolerance)
        else:
            return self._get_python_resonance_signature(target_frequency, tolerance)

    def _get_python_resonance_signature(self, target_frequency: float, tolerance: float) -> float:
        """Python fallback resonance signature calculation"""
        if not self.oscillation_buffer:
            return 0.0

        resonant_oscillations = [
            osc for osc in self.oscillation_buffer
            if abs(osc.frequency - target_frequency) <= tolerance
        ]

        if not resonant_oscillations:
            return 0.0

        amplitudes = [osc.amplitude for osc in resonant_oscillations]
        return np.mean(amplitudes)


class SystemOscillationProfiler:
    """
    High-level system oscillation profiler that coordinates
    multiple hardware profilers and provides unified analysis.
    """

    def __init__(self, sample_rate: float = 1000.0, buffer_size: int = 10000):
        if RUST_AVAILABLE:
            self._rust_profiler = lavoisier_computational.PySystemOscillationProfiler(
                sample_rate, buffer_size
            )
            self._use_rust = True
        else:
            self._use_rust = False
            self.harvester = HardwareHarvester(sample_rate, buffer_size)

    async def start_profiling(self) -> None:
        """Start system-wide oscillation profiling"""
        if self._use_rust:
            self._rust_profiler.start_profiling()
        else:
            await self.harvester.start_harvesting()

    def stop_profiling(self) -> None:
        """Stop system-wide oscillation profiling"""
        if self._use_rust:
            self._rust_profiler.stop_profiling()
        else:
            self.harvester.stop_harvesting()

    def get_system_resonance_map(self) -> Dict[str, np.ndarray]:
        """Get comprehensive resonance map of all system oscillations"""
        if self._use_rust:
            return self._rust_profiler.get_system_resonance_map()
        else:
            return self._get_python_resonance_map()

    def _get_python_resonance_map(self) -> Dict[str, np.ndarray]:
        """Python fallback resonance map calculation"""
        resonance_map = {}

        # Simple implementation for Python fallback
        for source in ["cpu", "memory"]:
            if hasattr(self.harvester, 'oscillation_buffer'):
                source_oscillations = [
                    osc for osc in self.harvester.oscillation_buffer
                    if osc.source == source
                ]

                if source_oscillations:
                    frequencies = [osc.frequency for osc in source_oscillations]
                    amplitudes = [osc.amplitude for osc in source_oscillations]

                    spectrum, _ = np.histogram(frequencies, weights=amplitudes, bins=50)
                    resonance_map[source] = spectrum

        return resonance_map


# High-level convenience functions
def get_system_oscillations(duration: float = 1.0) -> List[Tuple[str, float, float, float]]:
    """
    Get system oscillations for a specified duration

    Returns:
        List of tuples (source, frequency, amplitude, phase)
    """
    if RUST_AVAILABLE:
        return lavoisier_computational.py_get_system_oscillations(duration)
    else:
        # Python fallback
        harvester = HardwareHarvester()
        asyncio.run(harvester.start_harvesting())
        time.sleep(duration)
        harvester.stop_harvesting()

        oscillations = []
        if hasattr(harvester, 'oscillation_buffer'):
            for osc in harvester.oscillation_buffer:
                oscillations.append((osc.source, osc.frequency, osc.amplitude, osc.phase))

        return oscillations


def analyze_hardware_resonance(
    target_frequencies: List[float],
    duration: float = 5.0,
    tolerance: float = 0.1
) -> Dict[float, float]:
    """
    Analyze hardware resonance for target frequencies

    Args:
        target_frequencies: List of frequencies to analyze
        duration: Analysis duration in seconds
        tolerance: Frequency tolerance for matching

    Returns:
        Dictionary mapping target frequencies to resonance strengths
    """
    harvester = HardwareHarvester(sample_rate=100.0, buffer_size=5000)

    async def run_analysis():
        await harvester.start_harvesting()
        await asyncio.sleep(duration)

        results = {}
        for freq in target_frequencies:
            strength = harvester.get_resonance_signature(freq, tolerance)
            results[freq] = strength

        harvester.stop_harvesting()
        return results

    return asyncio.run(run_analysis())


# Export main classes and functions
__all__ = [
    'HardwareHarvester',
    'SystemOscillationProfiler',
    'HardwareOscillation',
    'get_system_oscillations',
    'analyze_hardware_resonance',
    'RUST_AVAILABLE'
]
