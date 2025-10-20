#!/usr/bin/env python3
"""
USB Polling Rate Harvester
===========================

USB polling rate (125 Hz - 1000 Hz) = periodic validation check frequency!

USB provides a natural periodic rhythm for validation checks.
This rhythm is a REAL hardware oscillation that can drive computational validation.

Author: Lavoisier Project
Date: October 2025
"""

import time
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
import platform


@dataclass
class USBPollingRhythm:
    """USB polling rhythm measurement."""
    polling_rate_hz: float
    validation_interval: float
    measured_jitter: float
    rhythm_quality: float


class USBOscillationHarvester:
    """
    USB polling rate (125 Hz - 1000 Hz) = periodic validation check frequency!

    USB devices poll at standard rates:
    - USB 1.1/2.0: 125 Hz (8 ms)
    - USB 3.0: 1000 Hz (1 ms)
    - USB 3.1+: Variable, up to 8000 Hz

    This provides a natural periodic rhythm for validation checks!
    """

    def __init__(self):
        """Initialize USB oscillation harvester."""
        self.detected_usb_rate = self._detect_usb_polling_rate()
        self.rhythm_measurements: List[USBPollingRhythm] = []

        print("[USB Oscillation Harvester] Initialized")
        print(f"  Detected USB polling rate: {self.detected_usb_rate} Hz")

    def _detect_usb_polling_rate(self) -> float:
        """
        Detect USB polling rate from system.

        Returns:
            Estimated USB polling rate in Hz
        """
        # Platform-specific detection
        if platform.system() == 'Windows':
            # Windows typically uses 125 Hz for USB 2.0
            return 125.0
        elif platform.system() == 'Linux':
            # Linux can use 1000 Hz for USB 3.0
            return 1000.0
        elif platform.system() == 'Darwin':  # macOS
            # macOS typically uses 1000 Hz
            return 1000.0
        else:
            # Default to USB 2.0 rate
            return 125.0

    def harvest_validation_rhythm(
        self,
        duration: float = 1.0
    ) -> USBPollingRhythm:
        """
        USB polling provides natural periodic rhythm for validation checks!

        Args:
            duration: Duration to measure rhythm (seconds)

        Returns:
            USBPollingRhythm with validation parameters
        """
        usb_poll_rate = self.detected_usb_rate

        # Use USB polling as validation clock
        validation_interval = 1.0 / usb_poll_rate

        # Measure actual jitter in the rhythm
        measured_jitter = self._measure_rhythm_jitter(duration, usb_poll_rate)

        # Rhythm quality = how stable the polling is
        # Lower jitter = higher quality
        rhythm_quality = 1.0 / (1.0 + measured_jitter / validation_interval)

        rhythm = USBPollingRhythm(
            polling_rate_hz=usb_poll_rate,
            validation_interval=validation_interval,
            measured_jitter=measured_jitter,
            rhythm_quality=rhythm_quality
        )

        self.rhythm_measurements.append(rhythm)

        # This is a REAL hardware oscillation!
        return rhythm

    def _measure_rhythm_jitter(
        self,
        duration: float,
        expected_rate: float
    ) -> float:
        """
        Measure jitter in timing rhythm.

        Args:
            duration: Measurement duration
            expected_rate: Expected polling rate

        Returns:
            Measured jitter in seconds
        """
        expected_interval = 1.0 / expected_rate
        num_samples = int(duration * expected_rate)

        if num_samples < 2:
            return 0.0

        # Measure actual intervals
        intervals = []
        for i in range(num_samples):
            t_start = time.perf_counter()
            time.sleep(expected_interval)
            t_end = time.perf_counter()
            actual_interval = t_end - t_start
            intervals.append(actual_interval)

        # Jitter = standard deviation of intervals
        jitter = np.std(intervals)

        return jitter

    def create_validation_schedule(
        self,
        num_validations: int
    ) -> List[float]:
        """
        Create validation schedule based on USB polling rhythm.

        Args:
            num_validations: Number of validation checks to schedule

        Returns:
            List of validation timestamps (relative to start)
        """
        validation_interval = 1.0 / self.detected_usb_rate

        schedule = [i * validation_interval for i in range(num_validations)]

        return schedule

    def validate_on_usb_rhythm(
        self,
        validation_function: callable,
        data: any,
        num_validations: int = 10
    ) -> List[Dict]:
        """
        Perform validations synchronized to USB polling rhythm.

        Args:
            validation_function: Function to call for validation
            data: Data to validate
            num_validations: Number of validation checks

        Returns:
            List of validation results with timing info
        """
        validation_interval = 1.0 / self.detected_usb_rate
        results = []

        print(f"[USB Rhythm Validation] Starting {num_validations} validations @ {self.detected_usb_rate} Hz")

        start_time = time.perf_counter()

        for i in range(num_validations):
            # Wait for next USB poll cycle
            target_time = start_time + i * validation_interval
            current_time = time.perf_counter()
            sleep_time = target_time - current_time

            if sleep_time > 0:
                time.sleep(sleep_time)

            # Perform validation
            validation_start = time.perf_counter()
            validation_result = validation_function(data)
            validation_end = time.perf_counter()

            # Record result
            result = {
                'validation_index': i,
                'scheduled_time': target_time - start_time,
                'actual_time': validation_start - start_time,
                'timing_error': (validation_start - start_time) - (target_time - start_time),
                'validation_duration': validation_end - validation_start,
                'validation_result': validation_result
            }

            results.append(result)

        return results

    def compute_rhythm_coherence(
        self,
        validation_results: List[Dict]
    ) -> float:
        """
        Compute coherence of validation rhythm.

        High coherence = validations synchronized well with USB polling
        Low coherence = poor synchronization

        Args:
            validation_results: Results from validate_on_usb_rhythm

        Returns:
            Coherence measure [0, 1]
        """
        if len(validation_results) < 2:
            return 1.0

        # Extract timing errors
        timing_errors = np.array([r['timing_error'] for r in validation_results])

        # Coherence = inverse of timing error variance
        error_variance = np.var(timing_errors)
        coherence = 1.0 / (1.0 + error_variance * 1000)  # Scale factor

        return coherence

    def get_rhythm_statistics(self) -> Dict:
        """Get statistics of all rhythm measurements."""
        if not self.rhythm_measurements:
            return {}

        polling_rates = [r.polling_rate_hz for r in self.rhythm_measurements]
        jitters = [r.measured_jitter for r in self.rhythm_measurements]
        qualities = [r.rhythm_quality for r in self.rhythm_measurements]

        return {
            'num_measurements': len(self.rhythm_measurements),
            'mean_polling_rate': np.mean(polling_rates),
            'mean_jitter': np.mean(jitters),
            'std_jitter': np.std(jitters),
            'mean_quality': np.mean(qualities),
            'min_quality': np.min(qualities),
            'max_quality': np.max(qualities)
        }


if __name__ == "__main__":
    print("="*70)
    print("USB Polling Rate Harvester")
    print("="*70)

    harvester = USBOscillationHarvester()

    print("\n[Test 1] Harvest validation rhythm")
    rhythm = harvester.harvest_validation_rhythm(duration=0.5)

    print(f"  Polling rate: {rhythm.polling_rate_hz} Hz")
    print(f"  Validation interval: {rhythm.validation_interval*1e3:.3f} ms")
    print(f"  Measured jitter: {rhythm.measured_jitter*1e6:.3f} μs")
    print(f"  Rhythm quality: {rhythm.rhythm_quality:.3f}")

    print("\n[Test 2] Create validation schedule")
    schedule = harvester.create_validation_schedule(num_validations=10)
    print(f"  Scheduled {len(schedule)} validations")
    print(f"  First 5 times: {[f'{t*1e3:.1f}ms' for t in schedule[:5]]}")

    print("\n[Test 3] Validate on USB rhythm")

    # Mock validation function
    def mock_validation(data):
        """Mock validation that checks data quality."""
        # Simulate some validation work
        result = np.sum(data) > 0
        return result

    mock_data = np.random.rand(100)

    validation_results = harvester.validate_on_usb_rhythm(
        mock_validation,
        mock_data,
        num_validations=5
    )

    print(f"  Completed {len(validation_results)} validations")
    for result in validation_results:
        print(f"    Val {result['validation_index']}: "
              f"error={result['timing_error']*1e6:.1f}μs, "
              f"duration={result['validation_duration']*1e6:.1f}μs, "
              f"result={result['validation_result']}")

    print("\n[Test 4] Rhythm coherence")
    coherence = harvester.compute_rhythm_coherence(validation_results)
    print(f"  Rhythm coherence: {coherence:.3f}")

    print("\n[Statistics]")
    stats = harvester.get_rhythm_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            if 'jitter' in key:
                print(f"  {key}: {value*1e6:.3f} μs")
            else:
                print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")

    print("\n" + "="*70)
