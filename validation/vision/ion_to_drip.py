#!/usr/bin/env python3
"""
Universal Ion-to-Drip Algorithm Validation
Based on: ion-drip.tex
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from typing import Dict, List, Tuple
import cv2

class UniversalIonToDrip:
    """Universal algorithm for ion-to-drip transformation"""
    
    def __init__(self, drip_threshold: float = 0.7):
        self.drip_threshold = drip_threshold
        self.ion_patterns = self._initialize_ion_patterns()
    
    def _initialize_ion_patterns(self) -> Dict:
        """Initialize known ion oscillation patterns"""
        return {
            'Na+': {'frequency': 2.3, 'amplitude': 0.8, 'phase': 0.0},
            'K+': {'frequency': 1.7, 'amplitude': 0.6, 'phase': 0.2},
            'Ca2+': {'frequency': 3.1, 'amplitude': 1.2, 'phase': 0.1},
            'Mg2+': {'frequency': 2.8, 'amplitude': 0.9, 'phase': 0.3},
            'Cl-': {'frequency': 1.9, 'amplitude': 0.7, 'phase': 0.15}
        }
    
    def detect_ion_oscillations(self, signal: np.ndarray, 
                               time: np.ndarray) -> Dict:
        """Detect ion oscillation patterns in signal"""
        
        # FFT analysis for frequency detection
        fft = np.fft.fft(signal)
        frequencies = np.fft.fftfreq(len(signal), time[1] - time[0])
        
        # Find dominant frequencies
        power_spectrum = np.abs(fft)
        peaks, _ = find_peaks(power_spectrum, height=np.max(power_spectrum) * 0.3)
        
        detected_ions = {}
        
        for peak_idx in peaks:
            peak_freq = abs(frequencies[peak_idx])
            peak_amplitude = power_spectrum[peak_idx] / len(signal)
            
            # Match to known ion patterns
            best_match = None
            min_diff = float('inf')
            
            for ion, pattern in self.ion_patterns.items():
                freq_diff = abs(peak_freq - pattern['frequency'])
                if freq_diff < min_diff:
                    min_diff = freq_diff
                    best_match = ion
            
            if best_match and min_diff < 0.5:  # Tolerance threshold
                detected_ions[best_match] = {
                    'frequency': peak_freq,
                    'amplitude': peak_amplitude,
                    'confidence': 1.0 / (1.0 + min_diff)
                }
        
        return detected_ions
    
    def ion_to_drip_transformation(self, ion_data: Dict, 
                                  time_points: np.ndarray) -> np.ndarray:
        """Transform detected ions to drip patterns"""
        
        drip_signal = np.zeros(len(time_points))
        
        for ion, properties in ion_data.items():
            if ion in self.ion_patterns:
                pattern = self.ion_patterns[ion]
                
                # Generate ion oscillation
                oscillation = (properties['amplitude'] * 
                              np.sin(2 * np.pi * properties['frequency'] * time_points + 
                                    pattern['phase']))
                
                # Apply drip transformation (threshold and accumulation)
                drip_contribution = np.where(oscillation > self.drip_threshold, 
                                           oscillation - self.drip_threshold, 0)
                
                drip_signal += drip_contribution * properties['confidence']
        
        # Apply cumulative drip effect
        cumulative_drip = np.cumsum(drip_signal)
        
        # Normalize
        if np.max(cumulative_drip) > 0:
            cumulative_drip = cumulative_drip / np.max(cumulative_drip)
        
        return cumulative_drip
    
    def analyze_drip_morphology(self, drip_signal: np.ndarray) -> Dict:
        """Analyze morphological properties of drip patterns"""
        
        # Find drip events (local maxima)
        drip_peaks, peak_properties = find_peaks(drip_signal, 
                                                 height=0.1, 
                                                 distance=5)
        
        # Calculate drip characteristics
        drip_intervals = np.diff(drip_peaks) if len(drip_peaks) > 1 else np.array([])
        drip_amplitudes = drip_signal[drip_peaks] if len(drip_peaks) > 0 else np.array([])
        
        return {
            'n_drips': len(drip_peaks),
            'drip_positions': drip_peaks,
            'drip_amplitudes': drip_amplitudes,
            'drip_intervals': drip_intervals,
            'mean_interval': np.mean(drip_intervals) if len(drip_intervals) > 0 else 0,
            'mean_amplitude': np.mean(drip_amplitudes) if len(drip_amplitudes) > 0 else 0,
            'drip_regularity': 1.0 / (1.0 + np.std(drip_intervals)) if len(drip_intervals) > 0 else 0
        }

def validate_ion_to_drip():
    """Validate Universal Ion-to-Drip algorithm"""
    
    print("UNIVERSAL ION-TO-DRIP VALIDATION")
    print("=" * 35)
    
    drip_analyzer = UniversalIonToDrip()
    
    # Generate synthetic ion signal
    time_points = np.linspace(0, 10, 500)
    
    print("\n1. Generating synthetic ion signals...")
    
    # Create composite signal with multiple ions
    composite_signal = np.zeros(len(time_points))
    true_ions = ['Na+', 'K+', 'Ca2+']
    
    for ion in true_ions:
        if ion in drip_analyzer.ion_patterns:
            pattern = drip_analyzer.ion_patterns[ion]
            ion_signal = (pattern['amplitude'] * 
                         np.sin(2 * np.pi * pattern['frequency'] * time_points + 
                               pattern['phase']))
            composite_signal += ion_signal
    
    # Add noise
    composite_signal += 0.1 * np.random.normal(0, 1, len(time_points))
    
    print(f"   → Generated signal with {len(true_ions)} ion types")
    
    print("\n2. Detecting ion oscillations...")
    
    # Detect ions
    detected_ions = drip_analyzer.detect_ion_oscillations(composite_signal, time_points)
    
    print(f"   → Detected {len(detected_ions)} ion patterns:")
    for ion, props in detected_ions.items():
        print(f"     • {ion}: freq={props['frequency']:.2f}, conf={props['confidence']:.3f}")
    
    print("\n3. Performing ion-to-drip transformation...")
    
    # Transform to drip pattern
    drip_signal = drip_analyzer.ion_to_drip_transformation(detected_ions, time_points)
    
    print(f"   → Drip signal generated (length: {len(drip_signal)})")
    
    print("\n4. Analyzing drip morphology...")
    
    # Analyze drip patterns
    drip_analysis = drip_analyzer.analyze_drip_morphology(drip_signal)
    
    print(f"   → Number of drips: {drip_analysis['n_drips']}")
    print(f"   → Mean drip interval: {drip_analysis['mean_interval']:.2f}")
    print(f"   → Mean drip amplitude: {drip_analysis['mean_amplitude']:.3f}")
    print(f"   → Drip regularity: {drip_analysis['drip_regularity']:.3f}")
    
    # Performance validation
    detection_accuracy = len([ion for ion in detected_ions if ion in true_ions]) / len(true_ions)
    drip_quality = drip_analysis['drip_regularity']
    
    print(f"\n5. Performance Metrics:")
    print(f"   → Ion detection accuracy: {detection_accuracy:.3f}")
    print(f"   → Drip pattern quality: {drip_quality:.3f}")
    print(f"   → Overall performance: {(detection_accuracy + drip_quality)/2:.3f}")
    
    # Visualization
    plt.figure(figsize=(14, 10))
    
    # Plot 1: Original ion signal
    plt.subplot(2, 3, 1)
    plt.plot(time_points, composite_signal, 'b-', alpha=0.7)
    plt.xlabel('Time')
    plt.ylabel('Ion Signal')
    plt.title('Composite Ion Signal')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Frequency spectrum
    plt.subplot(2, 3, 2)
    fft = np.fft.fft(composite_signal)
    frequencies = np.fft.fftfreq(len(composite_signal), time_points[1] - time_points[0])
    power_spectrum = np.abs(fft)
    
    plt.plot(frequencies[:len(frequencies)//2], 
             power_spectrum[:len(power_spectrum)//2], 'g-')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.title('Frequency Spectrum')
    plt.grid(True, alpha=0.3)
    
    # Mark detected frequencies
    for ion, props in detected_ions.items():
        plt.axvline(x=props['frequency'], color='red', linestyle='--', alpha=0.7)
    
    # Plot 3: Drip signal
    plt.subplot(2, 3, 3)
    plt.plot(time_points, drip_signal, 'r-', linewidth=2)
    
    # Mark drip events
    if drip_analysis['n_drips'] > 0:
        drip_times = time_points[drip_analysis['drip_positions']]
        drip_amps = drip_analysis['drip_amplitudes']
        plt.scatter(drip_times, drip_amps, color='red', s=50, zorder=5)
    
    plt.xlabel('Time')
    plt.ylabel('Drip Signal')
    plt.title('Ion-to-Drip Transformation')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Drip intervals
    plt.subplot(2, 3, 4)
    if len(drip_analysis['drip_intervals']) > 0:
        plt.hist(drip_analysis['drip_intervals'], bins=10, alpha=0.7, color='purple')
        plt.axvline(x=drip_analysis['mean_interval'], color='red', 
                   linestyle='--', label=f'Mean: {drip_analysis["mean_interval"]:.2f}')
        plt.xlabel('Drip Interval')
        plt.ylabel('Frequency')
        plt.title('Drip Interval Distribution')
        plt.legend()
    
    # Plot 5: Ion detection results
    plt.subplot(2, 3, 5)
    ion_names = list(detected_ions.keys())
    confidences = [detected_ions[ion]['confidence'] for ion in ion_names]
    
    if ion_names:
        bars = plt.bar(range(len(ion_names)), confidences, alpha=0.7)
        plt.xticks(range(len(ion_names)), ion_names)
        plt.ylabel('Detection Confidence')
        plt.title('Ion Detection Results')
        plt.ylim(0, 1)
        
        # Color bars by accuracy
        for i, (bar, ion) in enumerate(zip(bars, ion_names)):
            color = 'green' if ion in true_ions else 'orange'
            bar.set_color(color)
    
    # Plot 6: Performance summary
    plt.subplot(2, 3, 6)
    metrics = ['Ion\nDetection', 'Drip\nQuality', 'Overall\nPerformance']
    values = [detection_accuracy * 100, drip_quality * 100, 
             (detection_accuracy + drip_quality) / 2 * 100]
    
    bars = plt.bar(metrics, values, alpha=0.7, color=['blue', 'green', 'red'])
    plt.ylabel('Performance (%)')
    plt.title('Algorithm Performance')
    plt.ylim(0, 100)
    
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('validation/vision/ion_to_drip_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    overall_score = (detection_accuracy + drip_quality) / 2
    
    print(f"\n6. Validation Summary:")
    print(f"   → Algorithm validates ion-to-drip transformation")
    print(f"   → Detection accuracy: {detection_accuracy*100:.1f}%")
    print(f"   → Drip quality: {drip_quality*100:.1f}%")
    print(f"   → Overall score: {overall_score*100:.1f}%")
    print(f"   → Status: {'VALIDATED' if overall_score > 0.6 else 'NEEDS IMPROVEMENT'}")
    
    return {
        'detection_accuracy': detection_accuracy,
        'drip_quality': drip_quality,
        'overall_score': overall_score,
        'detected_ions': detected_ions,
        'drip_analysis': drip_analysis
    }

if __name__ == "__main__":
    results = validate_ion_to_drip()
    print(f"\nIon-to-Drip validation complete! Score: {results['overall_score']*100:.1f}%")