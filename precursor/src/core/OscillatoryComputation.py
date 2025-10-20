#!/usr/bin/env python3
"""
Oscillatory Computation for Mass Spectrometry
==============================================

Replaces traditional numerical computation with oscillatory computation
using harvested hardware frequencies.

Key Principle:
-------------
Instead of calculating S-Entropy, frequency coupling, phase-locks, etc.,
we ACCESS MEMORY at hierarchical frequencies, letting hardware oscillations
perform the computation through resonant coupling.

This is not an optimization - it's a fundamentally different computational
model where analysis happens through physical oscillations rather than
arithmetic operations.

Integration Points:
------------------
1. EntropyTransformation.py → Use oscillatory S-Entropy computation
2. PhaseLockNetworks.py → Detect phase-locks through hardware resonance
3. VectorTransformation.py → Vector operations via oscillatory modulation
4. FragmentationTrees.py → Build trees through multi-scale coupling
5. LLM Training → Training data from actual hardware measurements

Author: Lavoisier Project
Date: October 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import sys
from pathlib import Path

# Import hardware harvester
sys.path.append(str(Path(__file__).parent.parent))
from hardware.oscillatory_hierarchy import (
    EightScaleHardwareHarvester,
    OscillatoryComputationEngine
)

# Import existing frameworks (to replace their computation)
from core.EntropyTransformation import SEntropyCoordinates, SEntropyFeatures
from core.PhaseLockNetworks import PhaseLockSignature


class OscillatorySEntropyTransformer:
    """
    S-Entropy transformer that uses hardware oscillations instead of calculation.
    
    Replaces EntropyTransformation.SEntropyTransformer with oscillatory version.
    """
    
    def __init__(self):
        """Initialize with hardware harvester."""
        self.engine = OscillatoryComputationEngine()
        print("[Oscillatory S-Entropy] Initialized")
        print("  Using hardware oscillations instead of calculation")
    
    def transform_spectrum(
        self,
        mz_array: np.ndarray,
        intensity_array: np.ndarray,
        precursor_mz: Optional[float] = None,
        rt: Optional[float] = None
    ) -> Tuple[List[SEntropyCoordinates], np.ndarray]:
        """
        Transform spectrum using OSCILLATORY computation.
        
        Instead of calculating S-entropy, we access memory at different
        frequencies to create the oscillatory signature.
        """
        # Normalize intensities
        intensity_norm = intensity_array / np.sum(intensity_array)
        
        # Compute S-Entropy through oscillatory coupling
        osc_result = self.engine.compute_s_entropy_oscillatory(
            mz_array,
            intensity_norm
        )
        
        # Create S-Entropy coordinates for each peak
        coords_list = []
        coord_matrix = np.zeros((len(mz_array), 3))
        
        for i in range(len(mz_array)):
            # Map peak to oscillatory scale
            scale = self._map_mz_to_scale(mz_array[i])
            
            # Read from memory at this scale's frequency
            memory_data = self.engine.harvester.access_memory_at_scale(scale, num_bytes=3)
            
            # Decode oscillatory pattern as S-Entropy coordinates
            s_knowledge = (memory_data[0] / 255.0) * osc_result['s_knowledge']
            s_time = (memory_data[1] / 255.0) * osc_result['s_time']
            s_entropy = (memory_data[2] / 255.0) * osc_result['s_entropy']
            
            coord = SEntropyCoordinates(
                s_knowledge=float(s_knowledge),
                s_time=float(s_time),
                s_entropy=float(s_entropy)
            )
            
            coords_list.append(coord)
            coord_matrix[i] = coord.to_array()
        
        return coords_list, coord_matrix
    
    def _map_mz_to_scale(self, mz: float) -> int:
        """Map m/z to oscillatory scale (1-8)."""
        # Same mapping as in OscillatoryComputationEngine
        if mz < 100:
            return 1
        elif mz < 200:
            return 2
        elif mz < 400:
            return 3
        elif mz < 600:
            return 4
        elif mz < 800:
            return 5
        elif mz < 1000:
            return 6
        elif mz < 1500:
            return 7
        else:
            return 8


class OscillatoryPhaseLockDetector:
    """
    Phase-lock detector using hardware resonance.
    
    Instead of calculating phase-lock signatures, we detect them through
    resonance in the hardware oscillatory hierarchy.
    """
    
    def __init__(self):
        """Initialize with hardware harvester."""
        self.engine = OscillatoryComputationEngine()
        print("[Oscillatory Phase-Lock] Initialized")
        print("  Detecting phase-locks through hardware resonance")
    
    def detect_phase_lock(
        self,
        mz_array: np.ndarray,
        intensity_array: np.ndarray,
        rt: float
    ) -> PhaseLockSignature:
        """
        Detect phase-lock through hardware resonance.
        
        Phase-locks appear as resonances between oscillatory scales.
        """
        # Map fragments to scales
        scales = [self._map_mz_to_scale(mz) for mz in mz_array]
        
        # Measure resonance between scales
        coupling_strengths = []
        for i in range(len(scales) - 1):
            coupling = self.engine.harvester.get_coupling_strength(
                scales[i],
                scales[i + 1]
            )
            coupling_strengths.append(coupling)
        
        # Phase-lock strength = average coupling
        coherence_strength = np.mean(coupling_strengths) if coupling_strengths else 0.0
        
        # Dominant scale
        dominant_scale = max(set(scales), key=scales.count)
        scale_info = self.engine.harvester.scales[dominant_scale]
        
        # Build phase-lock signature from hardware measurements
        signature = PhaseLockSignature(
            mz_center=float(np.mean(mz_array)),
            mz_range=(float(np.min(mz_array)), float(np.max(mz_array))),
            rt_center=rt,
            rt_range=(rt - 0.1, rt + 0.1),  # ±0.1 min window
            coherence_strength=coherence_strength,
            coupling_modality="hardware_resonance",
            oscillation_frequency=scale_info.measured_frequency,
            phase_offset=scale_info.phase,
            ensemble_size=len(mz_array),
            temperature_signature=scale_info.amplitude,  # Use amplitude as proxy
            pressure_signature=coherence_strength,
            categorical_state=dominant_scale  # Use scale as categorical state
        )
        
        return signature
    
    def _map_mz_to_scale(self, mz: float) -> int:
        """Map m/z to oscillatory scale."""
        if mz < 100:
            return 1
        elif mz < 200:
            return 2
        elif mz < 400:
            return 3
        elif mz < 600:
            return 4
        elif mz < 800:
            return 5
        elif mz < 1000:
            return 6
        elif mz < 1500:
            return 7
        else:
            return 8


class OscillatoryFrequencyCoupling:
    """
    Frequency coupling computation using hardware oscillations.
    
    For proteomics: all peptide fragments are frequency-coupled because
    they emerge from the same collision event. We measure this coupling
    through hardware resonance.
    """
    
    def __init__(self):
        """Initialize with hardware harvester."""
        self.engine = OscillatoryComputationEngine()
        print("[Oscillatory Frequency Coupling] Initialized")
        print("  Measuring coupling through hardware resonance")
    
    def compute_coupling_matrix(
        self,
        fragment_mzs: List[float],
        fragment_intensities: List[float]
    ) -> np.ndarray:
        """
        Compute frequency coupling matrix using hardware.
        
        Coupling = resonance between hardware scales mapped from m/z values.
        """
        return self.engine.compute_frequency_coupling_oscillatory(
            fragment_mzs,
            fragment_intensities
        )
    
    def compute_collision_event_signature(
        self,
        fragment_mzs: List[float],
        fragment_intensities: List[float],
        collision_energy: float
    ) -> Dict:
        """
        Compute collision event signature from hardware.
        
        All fragments from same collision → coupled in hardware hierarchy.
        """
        # Map fragments to scales
        scales = [self._map_mz_to_scale(mz) for mz in fragment_mzs]
        
        # Measure inter-scale coupling
        coupling_matrix = self.compute_coupling_matrix(fragment_mzs, fragment_intensities)
        
        # Collision energy determines dominant scale
        energy_scale = self._map_energy_to_scale(collision_energy)
        energy_scale_info = self.engine.harvester.scales[energy_scale]
        
        signature = {
            'coupling_matrix': coupling_matrix,
            'mean_coupling': float(np.mean(coupling_matrix[np.triu_indices_from(coupling_matrix, k=1)])),
            'dominant_frequency': energy_scale_info.measured_frequency,
            'phase_coherence': energy_scale_info.amplitude,
            'ensemble_size': len(fragment_mzs),
            'oscillatory_scales': scales,
            'collision_scale': energy_scale
        }
        
        return signature
    
    def _map_mz_to_scale(self, mz: float) -> int:
        """Map m/z to oscillatory scale."""
        if mz < 100:
            return 1
        elif mz < 200:
            return 2
        elif mz < 400:
            return 3
        elif mz < 600:
            return 4
        elif mz < 800:
            return 5
        elif mz < 1000:
            return 6
        elif mz < 1500:
            return 7
        else:
            return 8
    
    def _map_energy_to_scale(self, collision_energy: float) -> int:
        """Map collision energy to oscillatory scale."""
        # Higher energy → higher frequency scale
        if collision_energy > 50:
            return 1
        elif collision_energy > 40:
            return 2
        elif collision_energy > 30:
            return 3
        elif collision_energy > 20:
            return 4
        elif collision_energy > 15:
            return 5
        elif collision_energy > 10:
            return 6
        elif collision_energy > 5:
            return 7
        else:
            return 8


class OscillatoryMemoryManager:
    """
    Manages memory access at different oscillatory frequencies.
    
    This is the CORE of oscillatory computation: by constantly accessing
    memory at different rates, we create the oscillatory coupling that
    performs molecular analysis.
    """
    
    def __init__(self):
        """Initialize memory manager."""
        self.harvester = EightScaleHardwareHarvester()
        self.harvester.start_harvesting()
        
        print("[Oscillatory Memory] Initialized")
        print("  Memory access creates oscillatory coupling")
    
    def access_at_frequency(
        self,
        data: np.ndarray,
        target_frequency: float
    ) -> np.ndarray:
        """
        Access memory at target frequency to modulate data.
        
        This is how oscillatory computation works: data is modulated
        by accessing memory at specific frequencies.
        """
        # Map frequency to scale
        scale = self._map_frequency_to_scale(target_frequency)
        
        # Access memory at this scale
        memory_data = self.harvester.access_memory_at_scale(scale, num_bytes=len(data))
        
        # Convert to modulation pattern
        pattern = np.frombuffer(memory_data, dtype=np.uint8).astype(float) / 255.0
        
        # Apply oscillatory modulation
        if len(pattern) < len(data):
            # Tile pattern
            repeats = int(np.ceil(len(data) / len(pattern)))
            pattern = np.tile(pattern, repeats)[:len(data)]
        else:
            pattern = pattern[:len(data)]
        
        result = data * (0.5 + 0.5 * pattern)
        
        return result
    
    def multi_scale_access(
        self,
        data: np.ndarray,
        scales: List[int]
    ) -> Dict[int, np.ndarray]:
        """
        Access memory at multiple scales simultaneously.
        
        This creates multi-scale oscillatory coupling.
        """
        results = {}
        
        for scale in scales:
            memory_data = self.harvester.access_memory_at_scale(scale, num_bytes=len(data))
            pattern = np.frombuffer(memory_data, dtype=np.uint8).astype(float) / 255.0
            
            if len(pattern) < len(data):
                repeats = int(np.ceil(len(data) / len(pattern)))
                pattern = np.tile(pattern, repeats)[:len(data)]
            else:
                pattern = pattern[:len(data)]
            
            results[scale] = data * (0.5 + 0.5 * pattern)
        
        return results
    
    def _map_frequency_to_scale(self, frequency: float) -> int:
        """Map frequency to oscillatory scale (1-8)."""
        for scale_id, scale in self.harvester.scales.items():
            if scale.frequency_range[0] <= frequency <= scale.frequency_range[1]:
                return scale_id
        
        # Fallback: find closest scale
        closest_scale = 1
        min_distance = float('inf')
        
        for scale_id, scale in self.harvester.scales.items():
            mid_freq = np.sqrt(scale.frequency_range[0] * scale.frequency_range[1])
            distance = abs(np.log10(frequency) - np.log10(mid_freq))
            
            if distance < min_distance:
                min_distance = distance
                closest_scale = scale_id
        
        return closest_scale
    
    def shutdown(self):
        """Shutdown memory manager."""
        self.harvester.stop_harvesting()


# Global oscillatory engine instance
_GLOBAL_ENGINE = None


def get_oscillatory_engine() -> OscillatoryComputationEngine:
    """Get global oscillatory computation engine."""
    global _GLOBAL_ENGINE
    if _GLOBAL_ENGINE is None:
        _GLOBAL_ENGINE = OscillatoryComputationEngine()
    return _GLOBAL_ENGINE


def shutdown_oscillatory_engine():
    """Shutdown global oscillatory engine."""
    global _GLOBAL_ENGINE
    if _GLOBAL_ENGINE is not None:
        _GLOBAL_ENGINE.shutdown()
        _GLOBAL_ENGINE = None


if __name__ == "__main__":
    print("="*70)
    print("Oscillatory Computation for Mass Spectrometry")
    print("="*70)
    
    print("\n[Demo] Testing oscillatory S-Entropy transformation...")
    
    # Mock spectrum
    mz = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    intensity = np.array([100.0, 80.0, 60.0, 40.0, 20.0])
    
    # Traditional vs Oscillatory
    print("\n1. S-Entropy Transformation")
    print("   (Using hardware oscillations instead of calculation)")
    
    transformer = OscillatorySEntropyTransformer()
    coords_list, coord_matrix = transformer.transform_spectrum(mz, intensity)
    
    print(f"\n   Transformed {len(coords_list)} peaks")
    print(f"   Coordinate matrix shape: {coord_matrix.shape}")
    print(f"   Mean S-knowledge: {np.mean(coord_matrix[:, 0]):.6f}")
    print(f"   Mean S-time: {np.mean(coord_matrix[:, 1]):.6f}")
    print(f"   Mean S-entropy: {np.mean(coord_matrix[:, 2]):.6f}")
    
    print("\n2. Phase-Lock Detection")
    print("   (Detecting through hardware resonance)")
    
    detector = OscillatoryPhaseLockDetector()
    signature = detector.detect_phase_lock(mz, intensity, rt=5.0)
    
    print(f"\n   Coherence strength: {signature.coherence_strength:.6f}")
    print(f"   Oscillation frequency: {signature.oscillation_frequency:.2e} Hz")
    print(f"   Ensemble size: {signature.ensemble_size}")
    
    print("\n3. Frequency Coupling (Proteomics)")
    print("   (All fragments coupled in hardware hierarchy)")
    
    coupling = OscillatoryFrequencyCoupling()
    coupling_matrix = coupling.compute_coupling_matrix(
        mz.tolist(),
        intensity.tolist()
    )
    
    print(f"\n   Coupling matrix shape: {coupling_matrix.shape}")
    print(f"   Mean coupling: {np.mean(coupling_matrix):.6f}")
    print(f"   Max coupling: {np.max(coupling_matrix):.6f}")
    
    collision_sig = coupling.compute_collision_event_signature(
        mz.tolist(),
        intensity.tolist(),
        collision_energy=25.0
    )
    
    print(f"   Collision event mean coupling: {collision_sig['mean_coupling']:.6f}")
    print(f"   Dominant frequency: {collision_sig['dominant_frequency']:.2e} Hz")
    
    print("\n4. Memory Access Patterns")
    print("   (Creating oscillatory coupling through memory)")
    
    memory_mgr = OscillatoryMemoryManager()
    
    # Access at different scales
    multi_scale_data = memory_mgr.multi_scale_access(
        intensity,
        scales=[1, 2, 3]
    )
    
    print(f"\n   Accessed {len(multi_scale_data)} scales")
    for scale, data in multi_scale_data.items():
        print(f"   Scale {scale}: mean = {np.mean(data):.3f}")
    
    # Cleanup
    memory_mgr.shutdown()
    shutdown_oscillatory_engine()
    
    print("\n[Demo] Complete!")
    print("="*70)

