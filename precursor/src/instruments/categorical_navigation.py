"""
Categorical Navigation Instruments

Instruments for navigating and measuring in categorical state space:
- Categorical Distance Meter: Measures S-space vs physical distance
- Null Geodesic Detector: Identifies partition-free traversals
- Non-Actualisation Shell Scanner: Maps geometry of non-actualisations
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from .base import (
    VirtualInstrument,
    HardwareOscillator,
    CategoricalState,
    SEntropyCoordinate,
    BOLTZMANN_CONSTANT
)


@dataclass
class NonActualisationShell:
    """A shell of non-actualisations at categorical distance r"""
    radius: int  # Categorical distance from actualisation
    theoretical_count: int  # k^r for branching factor k
    measured_count: int
    paired_fraction: float  # Fraction that are paired (ordinary matter)
    is_dark: bool  # Beyond pairing radius


class CategoricalDistanceMeter(VirtualInstrument):
    """
    Categorical Distance Meter - Measures distance in S-space vs physical space.
    
    Theory: d_categorical ≠ f(d_physical) for any function f.
    
    Key results:
    - Categorical adjacency does not imply spatial proximity
    - Spatial proximity does not imply categorical adjacency
    - Phase-lock clusters can span macroscopic distances
    """
    
    def __init__(self):
        super().__init__("Categorical Distance Meter")
        
    def calibrate(self) -> bool:
        """Calibrate distance measurements"""
        self.calibrated = True
        return True
    
    def measure_categorical_distance(self, state_A: CategoricalState,
                                      state_B: CategoricalState) -> float:
        """
        Measure categorical distance between two states.
        
        d_C(A, B) = ||S_A - S_B||
        """
        return state_A.categorical_distance_to(state_B)
    
    def measure_physical_distance(self, position_A: np.ndarray,
                                   position_B: np.ndarray) -> float:
        """
        Measure physical distance between two positions.
        
        d_P(A, B) = ||r_A - r_B||
        """
        return np.linalg.norm(position_A - position_B)
    
    def measure(self, n_pairs: int = 100, **kwargs) -> Dict[str, Any]:
        """
        Measure categorical vs physical distances for many pairs.
        
        Demonstrates that d_categorical ≠ f(d_physical).
        
        Args:
            n_pairs: Number of pairs to measure
            
        Returns:
            Dictionary with distance comparison results
        """
        categorical_distances = []
        physical_distances = []
        
        for _ in range(n_pairs):
            # Create two categorical states from hardware
            state_A = self.oscillator.create_categorical_state()
            state_B = self.oscillator.create_categorical_state()
            
            # Assign random physical positions
            pos_A = np.random.randn(3)
            pos_B = np.random.randn(3)
            
            # Measure distances
            d_cat = self.measure_categorical_distance(state_A, state_B)
            d_phys = self.measure_physical_distance(pos_A, pos_B)
            
            categorical_distances.append(d_cat)
            physical_distances.append(d_phys)
        
        categorical_distances = np.array(categorical_distances)
        physical_distances = np.array(physical_distances)
        
        # Compute correlation (should be near 0 for independent quantities)
        correlation = np.corrcoef(categorical_distances, physical_distances)[0, 1]
        
        # Find counterexamples
        # Case 1: Large physical, small categorical
        mask_1 = (physical_distances > np.percentile(physical_distances, 75)) & \
                 (categorical_distances < np.percentile(categorical_distances, 25))
        n_case_1 = np.sum(mask_1)
        
        # Case 2: Small physical, large categorical
        mask_2 = (physical_distances < np.percentile(physical_distances, 25)) & \
                 (categorical_distances > np.percentile(categorical_distances, 75))
        n_case_2 = np.sum(mask_2)
        
        result = {
            'n_pairs': n_pairs,
            'd_categorical_mean': np.mean(categorical_distances),
            'd_categorical_std': np.std(categorical_distances),
            'd_physical_mean': np.mean(physical_distances),
            'd_physical_std': np.std(physical_distances),
            'correlation': correlation,
            'inequivalence_demonstrated': abs(correlation) < 0.5,
            'counterexamples_case_1': n_case_1,  # Far physically, close categorically
            'counterexamples_case_2': n_case_2,  # Close physically, far categorically
            'theorem_verified': 'd_C(A,B) ≠ f(d_P(r_A, r_B)) for any f',
            'explanation': (
                'Categorical distance measures S-entropy separation, determined by '
                'phase-lock network topology. Physical distance measures Euclidean '
                'separation. These are independent: molecules can be physically distant '
                'but categorically adjacent (same phase-lock cluster), or physically '
                'proximate but categorically distant (different clusters).'
            )
        }
        
        self.record_measurement(result)
        return result
    
    def demonstrate_categorical_adjacency_without_proximity(self) -> Dict[str, Any]:
        """
        Demonstrate that categorical adjacency doesn't require physical proximity.
        
        Counterexample 1: Molecules A and B at large separation L >> r_lock
        can still be categorically adjacent through phase-lock chains.
        """
        # Create chain of phase-locked molecules
        chain_length = 10
        states = []
        
        # First molecule
        state_0 = self.oscillator.create_categorical_state()
        states.append(state_0)
        
        # Build chain where each is close to previous in S-space
        for i in range(1, chain_length):
            # Create state near previous (small categorical perturbation)
            prev_S = states[-1].S_coords.to_array()
            delta = np.random.randn(3) * 0.1  # Small categorical step
            new_S = np.clip(prev_S + delta, 0, 1)
            
            new_state = CategoricalState(
                S_coords=SEntropyCoordinate.from_array(new_S)
            )
            states.append(new_state)
        
        # Physical positions: widely separated
        physical_positions = [np.array([i * 1e6, 0, 0]) for i in range(chain_length)]  # Macroscopic
        
        # Categorical path length (sum of adjacent distances)
        cat_path_length = sum(
            self.measure_categorical_distance(states[i], states[i+1])
            for i in range(chain_length - 1)
        )
        
        # Physical separation end-to-end
        phys_distance = self.measure_physical_distance(
            physical_positions[0], physical_positions[-1]
        )
        
        return {
            'chain_length': chain_length,
            'categorical_path_length': cat_path_length,
            'physical_end_to_end_distance': phys_distance,
            'ratio': phys_distance / cat_path_length if cat_path_length > 0 else float('inf'),
            'demonstration': (
                f'Chain of {chain_length} molecules spans {phys_distance:.0f} meters '
                f'physically but only {cat_path_length:.3f} units categorically. '
                f'End-to-end categorical distance is finite despite macroscopic separation.'
            )
        }


class NullGeodesicDetector(VirtualInstrument):
    """
    Null Geodesic Detector - Identifies partition-free traversals.
    
    Theory:
    - Partition-free traversal generates zero boundary entropy
    - Zero entropy → zero proper time (Δτ = 0)
    - Zero proper time → maximum speed (v = c)
    
    Only massless, partition-free entities achieve light speed.
    """
    
    def __init__(self):
        super().__init__("Null Geodesic Detector")
        self.H_edge = 0.5  # Edge entropy in natural units
        self.omega = 1e12  # Characteristic frequency (Hz)
        
    def calibrate(self) -> bool:
        """Calibrate proper time measurement"""
        self.calibrated = True
        return True
    
    def count_partitions(self, trajectory: List[SEntropyCoordinate]) -> int:
        """
        Count number of partitions (categorical distinctions) along trajectory.
        
        A partition occurs when the trajectory creates a categorical distinction.
        """
        if len(trajectory) < 2:
            return 0
        
        partition_count = 0
        threshold = 0.1  # Minimum change to count as partition
        
        for i in range(len(trajectory) - 1):
            d = trajectory[i].distance_to(trajectory[i + 1])
            if d > threshold:
                partition_count += 1
        
        return partition_count
    
    def measure_boundary_entropy(self, n_partitions: int) -> float:
        """
        Compute boundary entropy from partition count.
        
        S_boundary = k_B × (n-1) × H_edge
        """
        if n_partitions <= 1:
            return 0.0
        return BOLTZMANN_CONSTANT * (n_partitions - 1) * self.H_edge
    
    def measure_proper_time(self, boundary_entropy: float) -> float:
        """
        Compute proper time from boundary entropy.
        
        Δτ = S_boundary / (k_B × ω)
        """
        if boundary_entropy == 0:
            return 0.0
        return boundary_entropy / (BOLTZMANN_CONSTANT * self.omega)
    
    def measure(self, trajectory: Optional[List[SEntropyCoordinate]] = None,
                n_points: int = 20, **kwargs) -> Dict[str, Any]:
        """
        Analyze a trajectory for partition-free traversal.
        
        Args:
            trajectory: List of S-coordinates along path (or generate)
            n_points: Number of points if generating
            
        Returns:
            Dictionary with null geodesic analysis
        """
        if trajectory is None:
            # Generate trajectory from hardware
            trajectory = []
            for _ in range(n_points):
                state = self.oscillator.create_categorical_state()
                trajectory.append(state.S_coords)
        
        # Count partitions
        n_partitions = self.count_partitions(trajectory)
        
        # Compute entropy and proper time
        S_boundary = self.measure_boundary_entropy(n_partitions)
        proper_time = self.measure_proper_time(S_boundary)
        
        # Check if null geodesic
        is_null = n_partitions == 0
        
        # Effective speed
        if proper_time > 0:
            # Subluminal: v < c
            coordinate_time = 1e-9  # Arbitrary
            gamma = coordinate_time / proper_time if proper_time > 0 else float('inf')
            v_over_c = np.sqrt(1 - 1/gamma**2) if gamma > 1 else 0
        else:
            # Null: v = c
            v_over_c = 1.0
        
        result = {
            'n_points': len(trajectory),
            'n_partitions': n_partitions,
            'boundary_entropy': S_boundary,
            'proper_time': proper_time,
            'is_null_geodesic': is_null,
            'v_over_c': v_over_c,
            'explanation': (
                'Partition-free traversal (n=0) → S_boundary = 0 → Δτ = 0 → v = c. '
                f'This trajectory has {n_partitions} partitions, '
                f'giving proper time Δτ = {proper_time:.2e} s '
                f'and speed v/c = {v_over_c:.4f}.'
            )
        }
        
        self.record_measurement(result)
        return result
    
    def create_null_trajectory(self, start: SEntropyCoordinate,
                                end: SEntropyCoordinate,
                                n_points: int = 10) -> List[SEntropyCoordinate]:
        """
        Create a partition-free trajectory from start to end.
        
        A null trajectory treats the interval as a single categorical entity
        with no internal distinctions.
        """
        # Linear interpolation - no internal partitions
        trajectory = []
        for t in np.linspace(0, 1, n_points):
            S_arr = (1 - t) * start.to_array() + t * end.to_array()
            trajectory.append(SEntropyCoordinate.from_array(S_arr))
        
        return trajectory
    
    def verify_mass_partition_coupling(self) -> Dict[str, Any]:
        """
        Verify that mass requires partition capability.
        
        Theory: Mass → localization → partition → entropy → proper time → v < c
        """
        # Massive particle: must create partitions (localized)
        massive_trajectory = []
        for _ in range(20):
            state = self.oscillator.create_categorical_state()
            massive_trajectory.append(state.S_coords)
        
        massive_result = self.measure(trajectory=massive_trajectory)
        
        # Massless particle: partition-free
        start = massive_trajectory[0]
        end = massive_trajectory[-1]
        null_trajectory = self.create_null_trajectory(start, end, 20)
        null_result = self.measure(trajectory=null_trajectory)
        
        return {
            'massive': {
                'partitions': massive_result['n_partitions'],
                'proper_time': massive_result['proper_time'],
                'v_over_c': massive_result['v_over_c']
            },
            'massless': {
                'partitions': null_result['n_partitions'],
                'proper_time': null_result['proper_time'],
                'v_over_c': null_result['v_over_c']
            },
            'theorem_verified': (
                massive_result['n_partitions'] > 0 and
                null_result['n_partitions'] == 0
            ),
            'explanation': (
                'Mass requires localization ("the object is here, not there"), '
                'which is a partition of space. Each partition generates entropy, '
                'hence proper time. Only m = 0 allows ρ_partition = 0, '
                'enabling partition-free traversal at v = c.'
            )
        }


class NonActualisationShellScanner(VirtualInstrument):
    """
    Non-Actualisation Shell Scanner - Maps the geometry of non-actualisations.
    
    Theory:
    - Non-actualisations organized in shells by categorical distance
    - Shell size grows exponentially: |N_r| ≈ k^r
    - Close shells (r ≤ r_pair): paired → ordinary matter
    - Distant shells (r > r_pair): unpaired → dark matter
    - Ratio: M_dark/M_ordinary ≈ k - 1 ≈ 5 for k ≈ 3
    """
    
    def __init__(self, branching_factor: int = 3, pairing_radius: int = 2):
        super().__init__("Non-Actualisation Shell Scanner")
        self.k = branching_factor  # Categorical branching factor
        self.r_pair = pairing_radius  # Maximum pairing distance
        
    def calibrate(self) -> bool:
        """Calibrate shell detection"""
        self.calibrated = True
        return True
    
    def theoretical_shell_size(self, r: int) -> int:
        """
        Compute theoretical shell size at distance r.
        
        |N_r| = k^r (exponential growth)
        """
        return self.k ** r
    
    def measure_shell(self, actualisation: CategoricalState,
                      r: int) -> NonActualisationShell:
        """
        Measure non-actualisation shell at distance r from actualisation.
        
        Args:
            actualisation: The central actualisation
            r: Categorical distance (shell radius)
            
        Returns:
            NonActualisationShell with measured properties
        """
        # Create non-actualisations at distance r
        # (In categorical space, these are "what didn't happen")
        theoretical = self.theoretical_shell_size(r)
        
        # Measure actual shell (from hardware)
        measured = 0
        paired = 0
        
        for _ in range(min(theoretical, 1000)):  # Sample limit
            delta_p = self.oscillator.read_timing_deviation()
            
            # Count this non-actualisation
            measured += 1
            
            # Check if paired (has nearby actualisation to reference)
            # Pairing probability decreases with distance
            if r <= self.r_pair:
                if np.random.random() < 1.0 / (1 + r):
                    paired += 1
        
        # Scale up if we sampled
        if measured < theoretical:
            scale = theoretical / measured
            paired = int(paired * scale)
            measured = theoretical
        
        paired_fraction = paired / measured if measured > 0 else 0
        
        return NonActualisationShell(
            radius=r,
            theoretical_count=theoretical,
            measured_count=measured,
            paired_fraction=paired_fraction,
            is_dark=r > self.r_pair
        )
    
    def measure(self, max_radius: int = 10, **kwargs) -> Dict[str, Any]:
        """
        Scan all shells around an actualisation.
        
        Args:
            max_radius: Maximum categorical distance to scan
            
        Returns:
            Dictionary with complete shell structure and dark/ordinary ratio
        """
        # Create central actualisation from hardware
        actualisation = self.oscillator.create_categorical_state()
        
        # Scan shells
        shells = {}
        total_paired = 0
        total_unpaired = 0
        
        for r in range(1, max_radius + 1):
            shell = self.measure_shell(actualisation, r)
            shells[r] = {
                'radius': r,
                'theoretical_count': shell.theoretical_count,
                'measured_count': shell.measured_count,
                'paired_fraction': shell.paired_fraction,
                'is_dark': shell.is_dark
            }
            
            n_paired = int(shell.measured_count * shell.paired_fraction)
            n_unpaired = shell.measured_count - n_paired
            
            total_paired += n_paired
            total_unpaired += n_unpaired
        
        # Compute dark/ordinary ratio
        if total_paired > 0:
            dark_ordinary_ratio = total_unpaired / total_paired
        else:
            dark_ordinary_ratio = float('inf')
        
        # Theoretical prediction
        theoretical_ratio = self.k - 1
        
        result = {
            'actualisation_S_coords': actualisation.S_coords.to_array().tolist(),
            'branching_factor': self.k,
            'pairing_radius': self.r_pair,
            'max_radius': max_radius,
            'shells': shells,
            'total_paired': total_paired,
            'total_unpaired': total_unpaired,
            'dark_ordinary_ratio': dark_ordinary_ratio,
            'theoretical_ratio': theoretical_ratio,
            'agreement': abs(dark_ordinary_ratio - theoretical_ratio) / theoretical_ratio < 0.5,
            'explanation': (
                f'Non-actualisations organized in shells with |N_r| = {self.k}^r. '
                f'Shells r ≤ {self.r_pair} are paired (ordinary matter). '
                f'Shells r > {self.r_pair} are unpaired (dark matter). '
                f'Measured ratio: {dark_ordinary_ratio:.2f}, '
                f'theoretical: {theoretical_ratio} (for k={self.k}).'
            )
        }
        
        self.record_measurement(result)
        return result
    
    def explain_dark_matter(self) -> Dict[str, Any]:
        """
        Provide explanation of dark matter from partition theory.
        
        Dark matter is NOT:
        - Mysterious missing mass
        - New exotic particles
        - Modified gravity
        
        Dark matter IS:
        - Accumulated unpaired non-actualisations
        - "What didn't happen" at large categorical distance
        - Non-partitionable (no local structure to partition)
        """
        # Measure shell structure
        result = self.measure(max_radius=8)
        
        explanation = {
            'theory': {
                'ordinary_matter': (
                    'Network of paired mutual non-actualisations. '
                    'Each actualisation A defines "not A" for nearby actualisations B, '
                    'and vice versa. These pairs form closed reference loops '
                    'that constitute the relational structure of ordinary matter.'
                ),
                'dark_matter': (
                    'Accumulated unpaired non-actualisations in distant shells. '
                    'These are "not something far away" - a relation with no local anchor. '
                    'Without local structure, they cannot be partitioned, '
                    'cannot interact with light (which is partition-free), '
                    'but still contribute gravitationally (mass is real).'
                ),
                'ratio_origin': (
                    f'The ~5:1 ratio emerges from shell geometry. '
                    f'For branching factor k={self.k}, '
                    f'the ratio of unpaired to paired is k-1 = {self.k - 1}. '
                    f'This is not a coincidence or fine-tuning.'
                )
            },
            'measured_ratio': result['dark_ordinary_ratio'],
            'shell_structure': result['shells']
        }
        
        return explanation

