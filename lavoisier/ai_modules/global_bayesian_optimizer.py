"""
Global Bayesian Evidence Network with Noise-Modulated Optimization

This module implements the revolutionary "swamp trees" approach where:
- The entire analysis becomes a single Bayesian evidence network
- Noise level becomes a controllable optimization parameter
- Different noise levels reveal different annotation "trees"
- CV and numerical evidence are weighted probabilistically
- The system optimizes noise level to maximize annotation confidence

Metaphor: Trees in a swamp - adjust water depth (noise) to see different tree heights (annotations)
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize, differential_evolution
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import seaborn as sns

from .mzekezeke import MzekezekeBayesianNetwork, EvidenceType, EvidenceNode, AnnotationCandidate
from ..numerical.numeric import NumericPipeline
from ..visual.visual import VisualPipeline

logger = logging.getLogger(__name__)

class EvidenceSource(Enum):
    """Sources of evidence in the global network"""
    NUMERICAL_PIPELINE = "numerical"
    VISUAL_PIPELINE = "visual"
    CROSS_VALIDATION = "cross_validation"
    NOISE_PATTERN = "noise_pattern"
    TEMPORAL_CONSISTENCY = "temporal"

@dataclass
class PrecisionNoiseModel:
    """Ultra-high fidelity noise model for different environmental complexity levels"""
    level: float  # 0.0 to 1.0, water depth in swamp metaphor
    thermal_model_params: Dict[str, float]
    electromagnetic_model_params: Dict[str, float] 
    chemical_background_model_params: Dict[str, float]
    instrumental_drift_model_params: Dict[str, float]
    stochastic_model_params: Dict[str, float]
    
    def __post_init__(self):
        """Calculate precise noise model parameters based on level"""
        # Ultra-precise thermal noise modeling
        self.thermal_model_params = {
            'base_variance': 0.001 * self.level,
            'temperature_coefficient': 2.3e-4,  # Based on Johnson-Nyquist noise
            'frequency_dependence': 1.2,
            'intensity_scaling_factor': 0.707  # sqrt(2)/2 for RMS
        }
        
        # Electromagnetic interference precise modeling
        self.electromagnetic_model_params = {
            'mains_frequency': 50.0,  # 50 Hz mains frequency
            'harmonic_frequencies': [100.0, 150.0, 200.0],  # Harmonics
            'amplitude_decay': 0.6,  # Amplitude decay for harmonics
            'phase_shifts': [0, np.pi/4, np.pi/2, 3*np.pi/4],  # Phase relationships
            'coupling_strength': 0.002 * self.level
        }
        
        # Chemical background precise modeling  
        self.chemical_background_model_params = {
            'exponential_decay_constant': 500.0,  # m/z units
            'baseline_offset': 10.0 * self.level,
            'contamination_peaks': [78.9, 149.0, 207.1, 279.2],  # Common contaminants
            'contamination_intensities': [0.1, 0.05, 0.03, 0.02],
            'solvent_cluster_pattern': 'exponential'
        }
        
        # Instrumental drift precise modeling
        self.instrumental_drift_model_params = {
            'linear_drift_rate': 1e-6 * self.level,  # ppm per acquisition
            'thermal_expansion_coefficient': 2.1e-5,
            'voltage_stability_factor': 0.999995,
            'time_constant': 3600.0  # seconds
        }
        
        # Stochastic components precise modeling
        self.stochastic_model_params = {
            'shot_noise_factor': np.sqrt(self.level),  # Poisson statistics
            'flicker_noise_alpha': 1.2,  # 1/f^alpha noise
            'white_noise_density': 1e-12 * self.level,
            'correlation_length': 0.1  # m/z units
        }
    
    def generate_expected_noise_spectrum(self, mz_array: np.ndarray, 
                                       acquisition_time: float = 1.0,
                                       temperature: float = 298.15) -> np.ndarray:
        """
        Generate the PRECISE expected noise spectrum at this water level.
        This is the key: we model exactly what noise should look like.
        """
        expected_noise = np.zeros_like(mz_array)
        
        # 1. Thermal noise (Johnson-Nyquist) - precisely modeled
        thermal_variance = (self.thermal_model_params['base_variance'] * 
                          self.thermal_model_params['temperature_coefficient'] * temperature)
        thermal_component = thermal_variance * np.power(mz_array, 
                          -self.thermal_model_params['frequency_dependence'])
        expected_noise += thermal_component
        
        # 2. Electromagnetic interference - precisely modeled periodic components
        em_params = self.electromagnetic_model_params
        for i, freq in enumerate([em_params['mains_frequency']] + em_params['harmonic_frequencies']):
            amplitude = (em_params['coupling_strength'] * 
                        np.power(em_params['amplitude_decay'], i))
            phase = em_params['phase_shifts'][i % len(em_params['phase_shifts'])]
            
            # EM interference creates periodic modulation in m/z space
            em_component = amplitude * np.sin(2 * np.pi * mz_array / freq + phase)
            expected_noise += em_component
        
        # 3. Chemical background - precisely modeled based on known chemistry
        bg_params = self.chemical_background_model_params
        
        # Exponential decay baseline
        baseline = (bg_params['baseline_offset'] * 
                   np.exp(-mz_array / bg_params['exponential_decay_constant']))
        expected_noise += baseline
        
        # Known contamination peaks
        for peak_mz, intensity in zip(bg_params['contamination_peaks'], 
                                    bg_params['contamination_intensities']):
            # Add Gaussian contamination peaks
            contamination = (intensity * self.level * 
                           np.exp(-0.5 * ((mz_array - peak_mz) / 0.1)**2))
            expected_noise += contamination
        
        # 4. Instrumental drift - precisely modeled systematic effects
        drift_params = self.instrumental_drift_model_params
        
        # Linear drift across m/z range
        linear_drift = (drift_params['linear_drift_rate'] * 
                       (mz_array - np.mean(mz_array)) * acquisition_time)
        
        # Thermal expansion effects
        thermal_expansion = (drift_params['thermal_expansion_coefficient'] * 
                           (temperature - 273.15) * mz_array / 1000.0)
        
        expected_noise += linear_drift + thermal_expansion
        
        # 5. Stochastic components - precisely modeled statistical noise
        stoch_params = self.stochastic_model_params
        
        # Shot noise (Poisson statistics)
        shot_noise_level = stoch_params['shot_noise_factor'] * np.sqrt(np.maximum(expected_noise, 1.0))
        
        # Flicker noise (1/f^alpha)
        freq_array = np.fft.fftfreq(len(mz_array), d=mz_array[1]-mz_array[0])
        freq_array[0] = 1e-10  # Avoid division by zero
        flicker_spectrum = stoch_params['white_noise_density'] / np.power(np.abs(freq_array), 
                                                                        stoch_params['flicker_noise_alpha'])
        flicker_noise = np.fft.irfft(np.sqrt(flicker_spectrum[:len(flicker_spectrum)//2+1]) * 
                                   np.random.normal(size=len(flicker_spectrum)//2+1))
        if len(flicker_noise) != len(expected_noise):
            flicker_noise = np.resize(flicker_noise, len(expected_noise))
        
        expected_noise += shot_noise_level + flicker_noise
        
        return np.maximum(expected_noise, 0.0)  # Ensure non-negative
    
    def calculate_deviation_significance(self, mz_array: np.ndarray, 
                                       observed_intensity: np.ndarray,
                                       acquisition_time: float = 1.0,
                                       temperature: float = 298.15) -> np.ndarray:
        """
        Calculate how significantly each point deviates from expected noise.
        THIS IS THE KEY: deviations from precise noise model = true peaks!
        """
        # Generate expected noise spectrum
        expected_noise = self.generate_expected_noise_spectrum(mz_array, acquisition_time, temperature)
        
        # Calculate deviation
        deviation = observed_intensity - expected_noise
        
        # Calculate statistical significance of deviation
        # Use local variance estimate for significance testing
        window_size = max(5, len(mz_array) // 1000)
        local_variance = np.zeros_like(expected_noise)
        
        for i in range(len(expected_noise)):
            start_idx = max(0, i - window_size//2)
            end_idx = min(len(expected_noise), i + window_size//2 + 1)
            local_variance[i] = np.var(expected_noise[start_idx:end_idx])
        
        # Z-score: how many standard deviations from expected noise
        z_scores = deviation / np.sqrt(local_variance + 1e-10)
        
        # Convert to significance (probability that this is NOT noise)
        from scipy import stats as scipy_stats
        significance = 1.0 - scipy_stats.norm.cdf(np.abs(z_scores))
        
        return significance
    
    def detect_true_peaks(self, mz_array: np.ndarray, 
                         observed_intensity: np.ndarray,
                         significance_threshold: float = 0.001,  # p < 0.001
                         acquisition_time: float = 1.0,
                         temperature: float = 298.15) -> List[Dict[str, Any]]:
        """
        Detect true peaks as significant deviations from precise noise model.
        This is where the magic happens: anything not explained by noise IS signal!
        """
        # Calculate significance of deviations
        significance = self.calculate_deviation_significance(
            mz_array, observed_intensity, acquisition_time, temperature
        )
        
        # Find peaks that significantly deviate from noise model
        true_peak_indices = np.where(significance < significance_threshold)[0]
        
        # Group nearby significant points into peaks
        peaks = []
        if len(true_peak_indices) > 0:
            # Use clustering to group nearby significant points
            from sklearn.cluster import DBSCAN
            
            # Cluster based on m/z proximity
            clustering = DBSCAN(eps=0.1, min_samples=1).fit(
                true_peak_indices.reshape(-1, 1)
            )
            
            for cluster_id in np.unique(clustering.labels_):
                if cluster_id == -1:  # Skip noise points
                    continue
                
                cluster_indices = true_peak_indices[clustering.labels_ == cluster_id]
                
                # Find peak maximum within cluster
                peak_idx = cluster_indices[np.argmax(observed_intensity[cluster_indices])]
                
                peaks.append({
                    'mz': mz_array[peak_idx],
                    'intensity': observed_intensity[peak_idx],
                    'significance': significance[peak_idx],
                    'deviation_from_noise': (observed_intensity[peak_idx] - 
                                           self.generate_expected_noise_spectrum(
                                               mz_array, acquisition_time, temperature)[peak_idx]),
                    'confidence': 1.0 - significance[peak_idx],  # High confidence = low p-value
                    'noise_level': self.level,
                    'detection_method': 'precision_noise_modeling'
                })
        
        # Sort by significance (most significant first)
        peaks.sort(key=lambda x: x['significance'])
        
        return peaks

@dataclass
class MultiSourceEvidence:
    """Evidence from multiple sources with probabilistic weighting"""
    node_id: str
    mz_value: float
    numerical_confidence: float
    visual_confidence: float
    cross_validation_score: float
    noise_sensitivity: float  # How much this evidence depends on noise level
    optimal_noise_level: float  # Noise level where this evidence is strongest
    source_weights: Dict[EvidenceSource, float] = field(default_factory=dict)
    
    def calculate_weighted_confidence(self, current_noise_level: float) -> float:
        """Calculate confidence based on current noise level and source weights"""
        # Noise level penalty/bonus
        noise_factor = 1.0 - abs(current_noise_level - self.optimal_noise_level)
        noise_factor = max(0.1, noise_factor)  # Minimum 10% confidence
        
        # Weighted combination of sources
        base_confidence = (
            self.source_weights.get(EvidenceSource.NUMERICAL_PIPELINE, 0.4) * self.numerical_confidence +
            self.source_weights.get(EvidenceSource.VISUAL_PIPELINE, 0.3) * self.visual_confidence +
            self.source_weights.get(EvidenceSource.CROSS_VALIDATION, 0.3) * self.cross_validation_score
        )
        
        return base_confidence * noise_factor * (1.0 + self.noise_sensitivity * (1.0 - current_noise_level))

class GlobalBayesianOptimizer:
    """
    Global Bayesian Evidence Network with Noise-Modulated Optimization
    
    This system transforms the entire MS analysis into a single optimization problem
    where noise level is dynamically adjusted to maximize annotation confidence.
    """
    
    def __init__(self,
                 numerical_pipeline: NumericPipeline,
                 visual_pipeline: VisualPipeline,
                 base_noise_levels: List[float] = None,
                 optimization_method: str = "differential_evolution",
                 max_optimization_iterations: int = 100,
                 convergence_threshold: float = 1e-4):
        
        self.numerical_pipeline = numerical_pipeline
        self.visual_pipeline = visual_pipeline
        self.base_noise_levels = base_noise_levels or np.linspace(0.1, 0.9, 9).tolist()
        self.optimization_method = optimization_method
        self.max_optimization_iterations = max_optimization_iterations
        self.convergence_threshold = convergence_threshold
        
        # Core Bayesian network (extended from Mzekezeke)
        self.global_network = MzekezekeBayesianNetwork(
            mass_tolerance_ppm=3.0,
            fuzzy_width_multiplier=1.5,
            min_evidence_nodes=1,
            network_convergence_threshold=1e-6
        )
        
        # Multi-source evidence storage
        self.multi_source_evidence: Dict[str, MultiSourceEvidence] = {}
        
        # Noise optimization state
        self.current_noise_level = 0.5
        self.noise_optimization_history: List[Tuple[float, float]] = []  # (noise_level, objective_value)
        self.optimal_noise_level = 0.5
        self.annotation_confidence_curve: List[Tuple[float, List[float]]] = []  # (noise_level, confidences)
        
        # Analysis cache for efficiency
        self.analysis_cache: Dict[str, Any] = {}
        
        logger.info("Global Bayesian Optimizer initialized with noise-modulated architecture")
    
    def create_precision_noise_model(self, level: float) -> PrecisionNoiseModel:
        """Create a precision noise model for a given level (0.0 to 1.0)"""
        # Create ultra-high fidelity noise model
        return PrecisionNoiseModel(
            level=level,
            thermal_model_params={},  # Will be populated in __post_init__
            electromagnetic_model_params={},
            chemical_background_model_params={},
            instrumental_drift_model_params={},
            stochastic_model_params={}
        )
    
    async def analyze_with_global_optimization(self,
                                             mz_array: np.ndarray,
                                             intensity_array: np.ndarray,
                                             compound_database: List[Dict[str, Any]],
                                             spectrum_id: str = "unknown") -> Dict[str, Any]:
        """
        Perform global analysis with noise-modulated optimization
        
        This is the main entry point that converts the entire analysis into
        a single optimization problem.
        """
        logger.info(f"Starting global Bayesian optimization for spectrum {spectrum_id}")
        
        # Step 1: Generate multi-noise evidence profiles
        logger.info("Generating evidence profiles across noise levels...")
        evidence_profiles = await self._generate_evidence_profiles(
            mz_array, intensity_array, spectrum_id
        )
        
        # Step 2: Build global Bayesian network
        logger.info("Building global Bayesian evidence network...")
        await self._build_global_network(evidence_profiles)
        
        # Step 3: Define objective function for optimization
        def objective_function(noise_level: float) -> float:
            """Objective function to maximize: total annotation confidence"""
            return -self._calculate_total_annotation_confidence(noise_level, compound_database)
        
        # Step 4: Optimize noise level
        logger.info("Optimizing noise level using swamp tree metaphor...")
        optimization_result = await self._optimize_noise_level(objective_function)
        
        # Step 5: Generate final annotations at optimal noise level
        logger.info(f"Generating final annotations at optimal noise level: {self.optimal_noise_level:.3f}")
        final_annotations = await self._generate_optimal_annotations(
            compound_database, self.optimal_noise_level
        )
        
        # Step 6: Create comprehensive analysis report
        analysis_report = self._create_analysis_report(
            spectrum_id, evidence_profiles, optimization_result, final_annotations
        )
        
        logger.info(f"Global optimization complete. Found {len(final_annotations)} annotations at optimal noise level {self.optimal_noise_level:.3f}")
        
        return analysis_report
    
    async def _generate_evidence_profiles(self,
                                        mz_array: np.ndarray,
                                        intensity_array: np.ndarray,
                                        spectrum_id: str) -> Dict[float, Dict[str, Any]]:
        """Generate evidence profiles across different noise levels"""
        evidence_profiles = {}
        
        # Process each noise level in parallel
        tasks = []
        for noise_level in self.base_noise_levels:
            task = self._analyze_at_noise_level(mz_array, intensity_array, noise_level, spectrum_id)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        for i, noise_level in enumerate(self.base_noise_levels):
            evidence_profiles[noise_level] = results[i]
        
        return evidence_profiles
    
    async def _analyze_at_noise_level(self,
                                    mz_array: np.ndarray,
                                    intensity_array: np.ndarray,
                                    noise_level: float,
                                    spectrum_id: str) -> Dict[str, Any]:
        """
        Analyze spectrum at a specific noise level using PRECISION NOISE MODELING.
        
        Key insight: By modeling exactly what noise should look like at this level,
        anything that deviates from the model becomes a true peak!
        """
        # Create precision noise model for this water level
        precision_noise_model = self.create_precision_noise_model(noise_level)
        
        # Generate expected noise spectrum at this level
        expected_noise = precision_noise_model.generate_expected_noise_spectrum(
            mz_array, acquisition_time=1.0, temperature=298.15
        )
        
        # Detect true peaks as deviations from expected noise
        true_peaks = precision_noise_model.detect_true_peaks(
            mz_array, intensity_array, significance_threshold=0.001
        )
        
        # Calculate significance map
        significance_map = precision_noise_model.calculate_deviation_significance(
            mz_array, intensity_array
        )
        
        # Run pipelines on the significance-filtered data
        # Only analyze points that significantly deviate from noise model
        significant_indices = np.where(significance_map < 0.01)[0]  # p < 0.01
        
        if len(significant_indices) > 0:
            filtered_mz = mz_array[significant_indices]
            filtered_intensity = intensity_array[significant_indices]
            
            numerical_results = await self._run_numerical_pipeline(
                filtered_mz, filtered_intensity, spectrum_id
            )
            visual_results = await self._run_visual_pipeline(
                filtered_mz, filtered_intensity, spectrum_id
            )
        else:
            # No significant deviations found
            numerical_results = {'annotations': [], 'analysis_confidence': 0.0}
            visual_results = {'annotations': [], 'analysis_confidence': 0.0}
        
        # Calculate cross-validation scores
        cross_validation_scores = self._calculate_cross_validation_scores(
            numerical_results, visual_results
        )
        
        # Add noise model insights to results
        numerical_results['true_peaks_from_noise_model'] = true_peaks
        numerical_results['noise_model_confidence'] = 1.0 - np.mean(significance_map)
        
        return {
            'noise_level': noise_level,
            'precision_noise_model': precision_noise_model,
            'expected_noise_spectrum': expected_noise,
            'significance_map': significance_map,
            'true_peaks_detected': true_peaks,
            'significant_points_ratio': len(significant_indices) / len(mz_array),
            'numerical_results': numerical_results,
            'visual_results': visual_results,
            'cross_validation_scores': cross_validation_scores,
            'noise_modeling_method': 'ultra_high_fidelity_precision'
        }
    
    async def _run_numerical_pipeline(self,
                                    mz_array: np.ndarray,
                                    intensity_array: np.ndarray,
                                    spectrum_id: str) -> Dict[str, Any]:
        """Run numerical pipeline analysis"""
        # This would integrate with the existing numerical pipeline
        # For now, we'll simulate the results
        
        # Detect peaks
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(intensity_array, height=np.percentile(intensity_array, 90))
        
        # Create numerical annotations
        annotations = []
        for i, peak_idx in enumerate(peaks[:20]):  # Top 20 peaks
            annotations.append({
                'mz': mz_array[peak_idx],
                'intensity': intensity_array[peak_idx],
                'confidence': min(1.0, intensity_array[peak_idx] / np.max(intensity_array)),
                'method': 'numerical'
            })
        
        return {
            'peaks': peaks,
            'annotations': annotations,
            'total_peaks': len(peaks),
            'analysis_confidence': 0.8
        }
    
    async def _run_visual_pipeline(self,
                                 mz_array: np.ndarray,
                                 intensity_array: np.ndarray,
                                 spectrum_id: str) -> Dict[str, Any]:
        """Run visual pipeline analysis"""
        # This would integrate with the existing visual pipeline
        # For now, we'll simulate complementary results
        
        # Visual analysis tends to find different patterns
        # Simulate finding patterns that numerical might miss
        visual_annotations = []
        
        # Look for patterns in the noise structure
        noise_patterns = self._detect_visual_patterns(mz_array, intensity_array)
        
        for pattern in noise_patterns:
            visual_annotations.append({
                'mz': pattern['center_mz'],
                'intensity': pattern['pattern_strength'],
                'confidence': pattern['visual_confidence'],
                'method': 'visual',
                'pattern_type': pattern['type']
            })
        
        return {
            'visual_patterns': noise_patterns,
            'annotations': visual_annotations,
            'pattern_count': len(noise_patterns),
            'analysis_confidence': 0.7
        }
    
    def _detect_visual_patterns(self, mz_array: np.ndarray, intensity_array: np.ndarray) -> List[Dict[str, Any]]:
        """Detect visual patterns in spectral data"""
        patterns = []
        
        # Sliding window pattern detection
        window_size = max(10, len(mz_array) // 50)
        
        for i in range(0, len(mz_array) - window_size, window_size // 2):
            window_mz = mz_array[i:i+window_size]
            window_intensity = intensity_array[i:i+window_size]
            
            # Look for visual patterns
            if len(window_intensity) > 0:
                pattern_strength = np.std(window_intensity) / (np.mean(window_intensity) + 1e-10)
                
                if pattern_strength > 0.5:  # Threshold for pattern detection
                    patterns.append({
                        'center_mz': np.mean(window_mz),
                        'pattern_strength': pattern_strength * 1000,  # Scale for intensity
                        'visual_confidence': min(1.0, pattern_strength),
                        'type': 'noise_structure',
                        'window_range': (window_mz[0], window_mz[-1])
                    })
        
        return patterns[:15]  # Return top 15 patterns
    
    def _calculate_cross_validation_scores(self,
                                         numerical_results: Dict[str, Any],
                                         visual_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate cross-validation scores between pipelines"""
        num_annotations = numerical_results['annotations']
        vis_annotations = visual_results['annotations']
        
        # Find overlapping m/z regions
        overlaps = 0
        total_comparisons = 0
        
        for num_ann in num_annotations:
            for vis_ann in vis_annotations:
                total_comparisons += 1
                mz_diff = abs(num_ann['mz'] - vis_ann['mz'])
                if mz_diff < 0.01:  # 10 mDa tolerance
                    overlaps += 1
        
        overlap_score = overlaps / max(1, total_comparisons)
        
        # Complementarity score (how well they complement each other)
        num_confidence = numerical_results.get('analysis_confidence', 0.5)
        vis_confidence = visual_results.get('analysis_confidence', 0.5)
        
        complementarity = 1.0 - abs(num_confidence - vis_confidence)
        
        return {
            'overlap_score': overlap_score,
            'complementarity_score': complementarity,
            'combined_confidence': (num_confidence + vis_confidence) / 2.0,
            'total_annotations': len(num_annotations) + len(vis_annotations)
        }
    
    async def _build_global_network(self, evidence_profiles: Dict[float, Dict[str, Any]]):
        """Build the global Bayesian evidence network from multi-noise profiles"""
        logger.info("Building global evidence network from multi-noise analysis...")
        
        # Clear existing network
        self.global_network = MzekezekeBayesianNetwork()
        self.multi_source_evidence.clear()
        
        # Process evidence from each noise level
        for noise_level, profile in evidence_profiles.items():
            await self._integrate_noise_level_evidence(noise_level, profile)
        
        # Auto-connect related evidence across noise levels
        self.global_network.auto_connect_related_evidence(correlation_threshold=0.4)
        
        # Update the global network
        self.global_network.update_bayesian_network(max_iterations=200)
        
        logger.info(f"Global network built with {len(self.multi_source_evidence)} multi-source evidence nodes")
    
    async def _integrate_noise_level_evidence(self, noise_level: float, profile: Dict[str, Any]):
        """Integrate evidence from a specific noise level into the global network"""
        numerical_annotations = profile['numerical_results']['annotations']
        visual_annotations = profile['visual_results']['annotations']
        cross_val_scores = profile['cross_validation_scores']
        
        # Create multi-source evidence nodes
        all_mz_values = set()
        
        # Collect all m/z values
        for ann in numerical_annotations:
            all_mz_values.add(ann['mz'])
        for ann in visual_annotations:
            all_mz_values.add(ann['mz'])
        
        # Create evidence nodes for each unique m/z
        for mz_value in all_mz_values:
            # Find corresponding annotations
            num_ann = next((ann for ann in numerical_annotations if abs(ann['mz'] - mz_value) < 0.01), None)
            vis_ann = next((ann for ann in visual_annotations if abs(ann['mz'] - mz_value) < 0.01), None)
            
            # Create multi-source evidence
            evidence_id = f"multisource_{mz_value:.4f}_{noise_level:.2f}"
            
            multi_evidence = MultiSourceEvidence(
                node_id=evidence_id,
                mz_value=mz_value,
                numerical_confidence=num_ann['confidence'] if num_ann else 0.0,
                visual_confidence=vis_ann['confidence'] if vis_ann else 0.0,
                cross_validation_score=cross_val_scores['combined_confidence'],
                noise_sensitivity=self._calculate_noise_sensitivity(mz_value, evidence_profiles),
                optimal_noise_level=noise_level,
                source_weights={
                    EvidenceSource.NUMERICAL_PIPELINE: 0.5 if num_ann else 0.0,
                    EvidenceSource.VISUAL_PIPELINE: 0.3 if vis_ann else 0.0,
                    EvidenceSource.CROSS_VALIDATION: 0.2
                }
            )
            
            self.multi_source_evidence[evidence_id] = multi_evidence
            
            # Add to Bayesian network
            intensity = (num_ann['intensity'] if num_ann else 0) + (vis_ann['intensity'] if vis_ann else 0)
            node_id = self.global_network.add_evidence_node(
                mz_value=mz_value,
                intensity=intensity,
                evidence_type=EvidenceType.MASS_MATCH,
                metadata={
                    'noise_level': noise_level,
                    'source': 'multi_source',
                    'evidence_id': evidence_id
                }
            )
    
    def _calculate_noise_sensitivity(self, mz_value: float, evidence_profiles: Dict[float, Dict[str, Any]]) -> float:
        """Calculate how sensitive an m/z value is to noise level"""
        confidences = []
        
        for noise_level, profile in evidence_profiles.items():
            # Find annotation for this m/z at this noise level
            total_confidence = 0.0
            count = 0
            
            for ann in profile['numerical_results']['annotations']:
                if abs(ann['mz'] - mz_value) < 0.01:
                    total_confidence += ann['confidence']
                    count += 1
            
            for ann in profile['visual_results']['annotations']:
                if abs(ann['mz'] - mz_value) < 0.01:
                    total_confidence += ann['confidence']
                    count += 1
            
            if count > 0:
                confidences.append(total_confidence / count)
            else:
                confidences.append(0.0)
        
        # Sensitivity is the standard deviation of confidence across noise levels
        return np.std(confidences) if len(confidences) > 1 else 0.0
    
    def _calculate_total_annotation_confidence(self, noise_level: float, compound_database: List[Dict[str, Any]]) -> float:
        """Calculate total annotation confidence at a given noise level"""
        total_confidence = 0.0
        annotation_count = 0
        
        # Calculate confidence for each multi-source evidence
        for evidence in self.multi_source_evidence.values():
            confidence = evidence.calculate_weighted_confidence(noise_level)
            
            # Check if this evidence matches any compound in database
            for compound in compound_database:
                compound_mass = compound.get('exact_mass', 0.0)
                if abs(evidence.mz_value - compound_mass) < 0.01:  # 10 mDa tolerance
                    total_confidence += confidence
                    annotation_count += 1
                    break
        
        # Return average confidence, with penalty for low annotation count
        if annotation_count == 0:
            return 0.0
        
        avg_confidence = total_confidence / annotation_count
        
        # Bonus for having more annotations (more trees visible in swamp)
        annotation_bonus = min(1.0, annotation_count / 10.0)  # Bonus up to 10 annotations
        
        return avg_confidence * (0.7 + 0.3 * annotation_bonus)
    
    async def _optimize_noise_level(self, objective_function: Callable[[float], float]) -> Dict[str, Any]:
        """Optimize noise level using the swamp tree metaphor"""
        logger.info("Starting noise level optimization (adjusting swamp water depth)...")
        
        # Clear optimization history
        self.noise_optimization_history.clear()
        self.annotation_confidence_curve.clear()
        
        def wrapped_objective(noise_level_array):
            """Wrapper for optimization function"""
            noise_level = noise_level_array[0]
            
            # Clamp noise level to valid range
            noise_level = max(0.05, min(0.95, noise_level))
            
            obj_value = objective_function(noise_level)
            
            # Store history
            self.noise_optimization_history.append((noise_level, -obj_value))  # Store as positive
            
            # Calculate annotation confidences at this level
            confidences = []
            for evidence in self.multi_source_evidence.values():
                conf = evidence.calculate_weighted_confidence(noise_level)
                confidences.append(conf)
            
            self.annotation_confidence_curve.append((noise_level, confidences))
            
            logger.debug(f"Noise level {noise_level:.3f}: objective = {-obj_value:.4f}, annotations = {len([c for c in confidences if c > 0.5])}")
            
            return obj_value
        
        # Perform optimization
        if self.optimization_method == "differential_evolution":
            result = differential_evolution(
                wrapped_objective,
                bounds=[(0.05, 0.95)],
                maxiter=self.max_optimization_iterations,
                popsize=15,
                tol=self.convergence_threshold,
                seed=42
            )
        else:
            # Fallback to scipy minimize
            result = minimize(
                wrapped_objective,
                x0=[0.5],
                bounds=[(0.05, 0.95)],
                method='L-BFGS-B',
                options={'maxiter': self.max_optimization_iterations}
            )
        
        self.optimal_noise_level = result.x[0]
        optimal_objective_value = -result.fun
        
        logger.info(f"Optimization complete. Optimal noise level: {self.optimal_noise_level:.3f}, max confidence: {optimal_objective_value:.4f}")
        
        return {
            'optimal_noise_level': self.optimal_noise_level,
            'optimal_objective_value': optimal_objective_value,
            'optimization_success': result.success,
            'optimization_iterations': len(self.noise_optimization_history),
            'convergence_message': result.message if hasattr(result, 'message') else "N/A"
        }
    
    async def _generate_optimal_annotations(self,
                                          compound_database: List[Dict[str, Any]],
                                          optimal_noise_level: float) -> List[Dict[str, Any]]:
        """Generate final annotations at the optimal noise level"""
        annotations = []
        
        for evidence in self.multi_source_evidence.values():
            confidence = evidence.calculate_weighted_confidence(optimal_noise_level)
            
            # Find matching compounds
            for compound in compound_database:
                compound_mass = compound.get('exact_mass', 0.0)
                if abs(evidence.mz_value - compound_mass) < 0.01:  # 10 mDa tolerance
                    
                    annotation = {
                        'compound_name': compound.get('name', 'Unknown'),
                        'molecular_formula': compound.get('formula', ''),
                        'exact_mass': compound_mass,
                        'observed_mz': evidence.mz_value,
                        'mass_error_ppm': ((evidence.mz_value - compound_mass) / compound_mass) * 1e6,
                        'confidence': confidence,
                        'numerical_confidence': evidence.numerical_confidence,
                        'visual_confidence': evidence.visual_confidence,
                        'cross_validation_score': evidence.cross_validation_score,
                        'noise_sensitivity': evidence.noise_sensitivity,
                        'optimal_noise_level': evidence.optimal_noise_level,
                        'source_weights': evidence.source_weights,
                        'evidence_node_id': evidence.node_id
                    }
                    
                    annotations.append(annotation)
                    break
        
        # Sort by confidence
        annotations.sort(key=lambda x: x['confidence'], reverse=True)
        
        return annotations
    
    def _create_analysis_report(self,
                              spectrum_id: str,
                              evidence_profiles: Dict[float, Dict[str, Any]],
                              optimization_result: Dict[str, Any],
                              final_annotations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create comprehensive analysis report"""
        
        # Calculate statistics
        high_confidence_annotations = [ann for ann in final_annotations if ann['confidence'] > 0.7]
        medium_confidence_annotations = [ann for ann in final_annotations if 0.3 < ann['confidence'] <= 0.7]
        
        # Network statistics
        network_summary = self.global_network.get_network_summary()
        
        report = {
            'spectrum_id': spectrum_id,
            'analysis_timestamp': pd.Timestamp.now().isoformat(),
            
            # Optimization results
            'noise_optimization': optimization_result,
            'optimal_noise_level': self.optimal_noise_level,
            'noise_level_range_tested': self.base_noise_levels,
            'optimization_history': self.noise_optimization_history,
            
            # Evidence analysis
            'total_evidence_nodes': len(self.multi_source_evidence),
            'noise_levels_analyzed': len(evidence_profiles),
            'evidence_profiles': evidence_profiles,
            
            # Annotation results
            'total_annotations': len(final_annotations),
            'high_confidence_annotations': len(high_confidence_annotations),
            'medium_confidence_annotations': len(medium_confidence_annotations),
            'annotations': final_annotations,
            
            # Network analysis
            'global_network_summary': network_summary,
            'bayesian_convergence': network_summary.get('network_connectivity', {}).get('is_connected', False),
            
            # Performance metrics
            'average_annotation_confidence': np.mean([ann['confidence'] for ann in final_annotations]) if final_annotations else 0.0,
            'confidence_std': np.std([ann['confidence'] for ann in final_annotations]) if final_annotations else 0.0,
            'pipeline_complementarity': self._calculate_pipeline_complementarity(),
            
            # Swamp tree metaphor metrics
            'swamp_depth_optimal': self.optimal_noise_level,
            'trees_visible_at_optimal': len(high_confidence_annotations),
            'total_trees_in_swamp': len(self.multi_source_evidence),
            'water_depth_sensitivity': np.std([evidence.noise_sensitivity for evidence in self.multi_source_evidence.values()]),
        }
        
        return report
    
    def _calculate_pipeline_complementarity(self) -> float:
        """Calculate how well the numerical and visual pipelines complement each other"""
        if not self.multi_source_evidence:
            return 0.0
        
        complementarity_scores = []
        
        for evidence in self.multi_source_evidence.values():
            # High complementarity when both pipelines contribute but differ
            if evidence.numerical_confidence > 0 and evidence.visual_confidence > 0:
                diff = abs(evidence.numerical_confidence - evidence.visual_confidence)
                # Complementarity is high when there's moderate difference (each sees something different)
                complementarity = 1.0 - min(1.0, diff / 0.5)  # Normalize to 0-1
                complementarity_scores.append(complementarity)
        
        return np.mean(complementarity_scores) if complementarity_scores else 0.0
    
    def visualize_swamp_optimization(self, save_path: str = None):
        """Visualize the swamp tree optimization process"""
        if not self.noise_optimization_history:
            logger.warning("No optimization history available for visualization")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Swamp Tree Optimization: Noise-Modulated Bayesian Evidence Network', fontsize=16)
        
        # 1. Optimization convergence
        noise_levels, objective_values = zip(*self.noise_optimization_history)
        axes[0, 0].plot(noise_levels, objective_values, 'b-', alpha=0.6, label='Optimization path')
        axes[0, 0].scatter([self.optimal_noise_level], [max(objective_values)], color='red', s=100, zorder=5, label='Optimal point')
        axes[0, 0].set_xlabel('Noise Level (Swamp Water Depth)')
        axes[0, 0].set_ylabel('Total Annotation Confidence')
        axes[0, 0].set_title('Swamp Water Depth Optimization')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Confidence curve across noise levels
        if self.annotation_confidence_curve:
            noise_levels_curve = [x[0] for x in self.annotation_confidence_curve]
            avg_confidences = [np.mean(x[1]) if x[1] else 0 for x in self.annotation_confidence_curve]
            max_confidences = [np.max(x[1]) if x[1] else 0 for x in self.annotation_confidence_curve]
            
            axes[0, 1].plot(noise_levels_curve, avg_confidences, 'g-', label='Average confidence')
            axes[0, 1].plot(noise_levels_curve, max_confidences, 'r--', label='Max confidence')
            axes[0, 1].axvline(x=self.optimal_noise_level, color='red', linestyle=':', alpha=0.7, label='Optimal level')
            axes[0, 1].set_xlabel('Noise Level')
            axes[0, 1].set_ylabel('Annotation Confidence')
            axes[0, 1].set_title('Tree Visibility vs Water Depth')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Evidence sensitivity heatmap
        evidence_data = []
        for evidence in self.multi_source_evidence.values():
            evidence_data.append([
                evidence.mz_value,
                evidence.noise_sensitivity,
                evidence.optimal_noise_level,
                evidence.numerical_confidence + evidence.visual_confidence
            ])
        
        if evidence_data:
            evidence_df = pd.DataFrame(evidence_data, columns=['m/z', 'Noise Sensitivity', 'Optimal Noise', 'Total Confidence'])
            
            # Create scatter plot
            scatter = axes[1, 0].scatter(evidence_df['m/z'], evidence_df['Optimal Noise'], 
                                       c=evidence_df['Noise Sensitivity'], s=evidence_df['Total Confidence']*100,
                                       alpha=0.6, cmap='viridis')
            axes[1, 0].set_xlabel('m/z')
            axes[1, 0].set_ylabel('Optimal Noise Level')
            axes[1, 0].set_title('Evidence Noise Sensitivity Map')
            plt.colorbar(scatter, ax=axes[1, 0], label='Noise Sensitivity')
        
        # 4. Pipeline complementarity
        num_confidences = [evidence.numerical_confidence for evidence in self.multi_source_evidence.values()]
        vis_confidences = [evidence.visual_confidence for evidence in self.multi_source_evidence.values()]
        
        axes[1, 1].scatter(num_confidences, vis_confidences, alpha=0.6)
        axes[1, 1].plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Perfect correlation')
        axes[1, 1].set_xlabel('Numerical Pipeline Confidence')
        axes[1, 1].set_ylabel('Visual Pipeline Confidence')
        axes[1, 1].set_title('Pipeline Complementarity')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Swamp optimization visualization saved to {save_path}")
        
        plt.show()
    
    def export_global_network(self, filename: str):
        """Export the global Bayesian network with multi-source evidence"""
        network_data = {
            'global_network': {
                'optimal_noise_level': self.optimal_noise_level,
                'total_evidence_nodes': len(self.multi_source_evidence),
                'bayesian_network': self.global_network.export_network(filename + '_bayesian.json')
            },
            'multi_source_evidence': {
                evidence_id: {
                    'mz_value': evidence.mz_value,
                    'numerical_confidence': evidence.numerical_confidence,
                    'visual_confidence': evidence.visual_confidence,
                    'cross_validation_score': evidence.cross_validation_score,
                    'noise_sensitivity': evidence.noise_sensitivity,
                    'optimal_noise_level': evidence.optimal_noise_level,
                    'source_weights': evidence.source_weights
                }
                for evidence_id, evidence in self.multi_source_evidence.items()
            },
            'optimization_history': self.noise_optimization_history,
            'annotation_confidence_curve': self.annotation_confidence_curve
        }
        
        with open(filename, 'w') as f:
            json.dump(network_data, f, indent=2, default=str)
        
        logger.info(f"Global Bayesian network exported to {filename}")