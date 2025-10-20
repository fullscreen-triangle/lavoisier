#!/usr/bin/env python3
"""
LED Display Flicker Harvester
==============================

Harvest LED flicker as spectroscopic training data!

LED refresh rate oscillations = molecular excitation patterns.

LED wavelengths map to molecular features, and the flicker frequency
provides a natural oscillatory encoding.

Author: Lavoisier Project
Date: October 2025
"""

import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


# LED wavelengths and corresponding frequencies
LED_FREQUENCIES = {
    'blue': {
        'wavelength_nm': 470,
        'wavelength_m': 470e-9,
        'frequency_hz': 6.38e14,  # c / λ
        'energy_ev': 2.64
    },
    'green': {
        'wavelength_nm': 525,
        'wavelength_m': 525e-9,
        'frequency_hz': 5.71e14,
        'energy_ev': 2.36
    },
    'red': {
        'wavelength_nm': 625,
        'wavelength_m': 625e-9,
        'frequency_hz': 4.80e14,
        'energy_ev': 1.98
    }
}

# Typical display refresh rates
DISPLAY_REFRESH_RATES = {
    'standard': 60.0,  # Hz
    'gaming': 144.0,
    'high_end': 240.0,
    'oled': 120.0
}


@dataclass
class LEDSpectroscopicFeature:
    """LED-derived spectroscopic feature."""
    wavelength_nm: float
    flicker_frequency: float
    molecular_intensity: float
    coupling_strength: float
    phase_modulation: float


class LEDSpectroscopyHarvester:
    """
    Harvest LED flicker as spectroscopic training data!

    LED refresh rate oscillations = molecular excitation patterns.

    The display refresh rate creates a natural oscillatory encoding
    where molecular features are modulated by LED flicker.
    """

    def __init__(self, refresh_rate: float = 60.0):
        """
        Initialize LED spectroscopy harvester.

        Args:
            refresh_rate: Display refresh rate in Hz
        """
        self.refresh_rate = refresh_rate
        self.flicker_period = 1.0 / refresh_rate

        print("[LED Spectroscopy Harvester] Initialized")
        print(f"  Refresh rate: {refresh_rate} Hz")
        print(f"  Flicker period: {self.flicker_period*1e3:.3f} ms")

    def harvest_spectroscopic_features(
        self,
        spectrum: Dict
    ) -> List[LEDSpectroscopicFeature]:
        """
        Use LED flicker to encode spectroscopic information!

        Args:
            spectrum: Mass spectrum with m/z and intensity arrays

        Returns:
            List of LED spectroscopic features
        """
        led_oscillations = []

        # Process each LED color channel
        for color_name, led_data in LED_FREQUENCIES.items():
            wavelength_nm = led_data['wavelength_nm']

            # LED flicker frequency
            flicker_freq = self._get_led_flicker_frequency(wavelength_nm)

            # Map spectrum to this wavelength
            molecular_feature = self._map_spectrum_to_wavelength(
                spectrum,
                wavelength_nm
            )

            # Coupling strength = flicker frequency × molecular intensity
            coupling_strength = flicker_freq * molecular_feature

            # Phase modulation from wavelength
            phase_modulation = self._calculate_phase_modulation(
                wavelength_nm,
                flicker_freq
            )

            feature = LEDSpectroscopicFeature(
                wavelength_nm=wavelength_nm,
                flicker_frequency=flicker_freq,
                molecular_intensity=molecular_feature,
                coupling_strength=coupling_strength,
                phase_modulation=phase_modulation
            )

            led_oscillations.append(feature)

        return led_oscillations

    def _get_led_flicker_frequency(self, wavelength_nm: float) -> float:
        """
        Get LED flicker frequency for wavelength.

        Different wavelengths may have different effective flicker rates
        due to phosphor persistence and color mixing.

        Args:
            wavelength_nm: Wavelength in nanometers

        Returns:
            Effective flicker frequency in Hz
        """
        # Base flicker = display refresh rate
        base_flicker = self.refresh_rate

        # Wavelength-dependent modulation
        # Shorter wavelengths (blue) have faster phosphor decay
        # Longer wavelengths (red) have slower decay
        if wavelength_nm < 500:  # Blue range
            wavelength_factor = 1.2
        elif wavelength_nm < 575:  # Green range
            wavelength_factor = 1.0
        else:  # Red range
            wavelength_factor = 0.8

        return base_flicker * wavelength_factor

    def _map_spectrum_to_wavelength(
        self,
        spectrum: Dict,
        wavelength_nm: float
    ) -> float:
        """
        Map mass spectrum features to spectroscopic wavelength.

        This is a heuristic mapping from m/z space to optical wavelength.

        Args:
            spectrum: Mass spectrum
            wavelength_nm: Target wavelength

        Returns:
            Molecular feature intensity
        """
        mz_array = spectrum.get('mz', np.array([]))
        intensity_array = spectrum.get('intensity', np.array([]))

        if len(mz_array) == 0:
            return 0.0

        # Map wavelength to m/z range (heuristic)
        # Shorter wavelengths → higher energy → higher m/z
        # λ(nm) 400-700 maps roughly to m/z 100-1000
        mz_center = (wavelength_nm - 400) / 300 * 900 + 100
        mz_width = 100.0

        # Extract features near this m/z
        mask = (mz_array >= mz_center - mz_width) & (mz_array <= mz_center + mz_width)

        if np.any(mask):
            molecular_intensity = np.mean(intensity_array[mask])
        else:
            # Fallback: use overall intensity scaled by wavelength
            molecular_intensity = np.mean(intensity_array) * (wavelength_nm / 525.0)

        return molecular_intensity

    def _calculate_phase_modulation(
        self,
        wavelength_nm: float,
        flicker_freq: float
    ) -> float:
        """
        Calculate phase modulation from wavelength and flicker.

        Args:
            wavelength_nm: Wavelength in nm
            flicker_freq: Flicker frequency in Hz

        Returns:
            Phase modulation angle in radians
        """
        # Phase from wavelength (within visible range)
        wavelength_phase = (wavelength_nm - 400) / 300 * 2 * np.pi

        # Phase from flicker timing
        flicker_phase = (flicker_freq / 1000.0) * 2 * np.pi

        # Combined phase
        phase_modulation = (wavelength_phase + flicker_phase) % (2 * np.pi)

        return phase_modulation

    def create_led_training_data(
        self,
        spectra: List[Dict],
        labels: List[str]
    ) -> Dict:
        """
        Create training data from LED spectroscopic features.

        Args:
            spectra: List of mass spectra
            labels: List of labels for each spectrum

        Returns:
            Training data dictionary
        """
        training_features = []
        training_labels = []

        for spectrum, label in zip(spectra, labels):
            led_features = self.harvest_spectroscopic_features(spectrum)

            # Flatten LED features into training vector
            feature_vector = []
            for led_feature in led_features:
                feature_vector.extend([
                    led_feature.flicker_frequency,
                    led_feature.molecular_intensity,
                    led_feature.coupling_strength,
                    led_feature.phase_modulation
                ])

            training_features.append(feature_vector)
            training_labels.append(label)

        return {
            'features': np.array(training_features),
            'labels': np.array(training_labels),
            'feature_names': [
                f'{color}_{feat}'
                for color in ['blue', 'green', 'red']
                for feat in ['flicker_freq', 'molecular_intensity', 'coupling', 'phase']
            ],
            'num_features': len(training_features[0]) if training_features else 0,
            'num_samples': len(training_features)
        }

    def visualize_led_encoding(
        self,
        spectrum: Dict
    ) -> Dict:
        """
        Visualize how LED encoding represents the spectrum.

        Args:
            spectrum: Mass spectrum

        Returns:
            Visualization data
        """
        led_features = self.harvest_spectroscopic_features(spectrum)

        # Create RGB color representation
        rgb_values = []
        for feature in led_features:
            # Normalize intensity to [0, 1]
            normalized_intensity = feature.molecular_intensity / (np.max([f.molecular_intensity for f in led_features]) + 1e-9)

            # Modulate by flicker phase
            phase_modulated = normalized_intensity * (1.0 + 0.5 * np.cos(feature.phase_modulation))
            rgb_values.append(phase_modulated)

        # RGB tuple
        rgb = tuple(rgb_values)

        # Flicker pattern over time
        time_points = np.linspace(0, 1.0 / self.refresh_rate, 100)
        flicker_patterns = {}

        for i, feature in enumerate(led_features):
            color_name = list(LED_FREQUENCIES.keys())[i]
            pattern = feature.molecular_intensity * (1.0 + 0.5 * np.sin(2 * np.pi * feature.flicker_frequency * time_points))
            flicker_patterns[color_name] = pattern

        return {
            'rgb': rgb,
            'led_features': led_features,
            'time_points': time_points,
            'flicker_patterns': flicker_patterns
        }


if __name__ == "__main__":
    print("="*70)
    print("LED Display Flicker Harvester")
    print("="*70)

    harvester = LEDSpectroscopyHarvester(refresh_rate=60.0)

    # Mock spectrum
    spectrum = {
        'mz': np.array([100, 200, 300, 400, 500, 600, 700, 800]),
        'intensity': np.array([100, 80, 60, 90, 70, 50, 40, 30])
    }

    print("\n[Test 1] Harvesting spectroscopic features")
    led_features = harvester.harvest_spectroscopic_features(spectrum)

    for feature in led_features:
        print(f"\n  Wavelength: {feature.wavelength_nm} nm")
        print(f"    Flicker frequency: {feature.flicker_frequency:.2f} Hz")
        print(f"    Molecular intensity: {feature.molecular_intensity:.2f}")
        print(f"    Coupling strength: {feature.coupling_strength:.2e}")
        print(f"    Phase modulation: {feature.phase_modulation:.3f} rad")

    print("\n[Test 2] Create training data")
    spectra = [spectrum for _ in range(5)]
    labels = [f'molecule_{i}' for i in range(5)]

    training_data = harvester.create_led_training_data(spectra, labels)

    print(f"  Features shape: {training_data['features'].shape}")
    print(f"  Labels shape: {training_data['labels'].shape}")
    print(f"  Num features: {training_data['num_features']}")
    print(f"  Num samples: {training_data['num_samples']}")
    print(f"  Feature names: {training_data['feature_names'][:4]}...")

    print("\n[Test 3] Visualize LED encoding")
    viz_data = harvester.visualize_led_encoding(spectrum)

    print(f"  RGB representation: ({viz_data['rgb'][0]:.3f}, {viz_data['rgb'][1]:.3f}, {viz_data['rgb'][2]:.3f})")
    print(f"  Flicker patterns available: {list(viz_data['flicker_patterns'].keys())}")

    for color, pattern in viz_data['flicker_patterns'].items():
        print(f"    {color}: min={np.min(pattern):.2f}, max={np.max(pattern):.2f}, mean={np.mean(pattern):.2f}")

    print("\n" + "="*70)
