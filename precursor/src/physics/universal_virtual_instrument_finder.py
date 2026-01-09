"""
Universal Virtual Instrument Finder (UVIF)

Implementation of Algorithm 1 from "Partition Coordinate Geometry in Bounded Oscillatory Systems"

This module provides a systematic procedure for constructing optimal virtual instruments
from arbitrary hardware oscillators. Given available hardware, target partition coordinates,
and precision requirements, the algorithm outputs an optimal instrument configuration,
measurement protocol, and coordinate extraction procedure.

Author: Categorical Physics Framework
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import sys

# Handle Unicode output on Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


@dataclass
class Hardware:
    """Represents a hardware oscillator.
    
    Attributes:
        name: Human-readable name of the hardware
        frequencies: Oscillation signature Omega(h) - frequencies it can generate/detect
        noise: Intrinsic noise level sigma_noise(h) in eV
        cost: Relative cost of using this hardware
        time: Time required for measurement in seconds
    """
    name: str
    frequencies: np.ndarray
    noise: float
    cost: float
    time: float


@dataclass
class Target:
    """Represents a target partition coordinate.
    
    Attributes:
        name: Name of the coordinate (n, l, m, s, s_c)
        char_freq: Characteristic frequency omega_t of this coordinate
        precision: Required measurement precision sigma_required
        weight: Importance weight for optimization
    """
    name: str
    char_freq: float
    precision: float
    weight: float = 1.0


@dataclass
class InstrumentConfiguration:
    """Output of the UVIF algorithm.
    
    Attributes:
        hardware_indices: Indices of selected hardware in optimal configuration
        hardware_names: Names of selected hardware
        protocol: Measurement protocol dictionary
        extraction: Coordinate extraction procedure dictionary
        uncertainties: Achieved uncertainty bounds for each target
        quality: Overall quality score of configuration
    """
    hardware_indices: List[int]
    hardware_names: List[str]
    protocol: Dict
    extraction: Dict
    uncertainties: Dict[str, float]
    quality: float


def compute_coupling(hardware: Hardware, target: Target) -> float:
    """Compute coupling strength <omega|t> (Theorem: Coordinate-Frequency Coupling).
    
    The coupling depends on the overlap between hardware frequencies and 
    the target coordinate's characteristic frequency.
    
    Args:
        hardware: Hardware oscillator
        target: Target partition coordinate
        
    Returns:
        Coupling strength in [0, 1]
    """
    if target.char_freq == 0:
        # Special case for l=0 ground states
        return 0.1 if len(hardware.frequencies) > 0 else 0.0
    
    # Find closest frequency match
    freq_diffs = np.abs(hardware.frequencies - target.char_freq)
    min_diff = np.min(freq_diffs)
    
    # Coupling decays exponentially with frequency mismatch
    # Normalized by characteristic frequency to make dimensionless
    coupling = np.exp(-min_diff / target.char_freq)
    return float(coupling)


def compute_accessibility(hardware: Hardware, target: Target) -> float:
    """Compute accessibility A(h,t) (Definition: Accessibility Function).
    
    A(h,t) = max_{omega in Omega(h)} |<omega|t>|^2
    
    Args:
        hardware: Hardware oscillator
        target: Target partition coordinate
        
    Returns:
        Accessibility value in [0, 1]
    """
    coupling = compute_coupling(hardware, target)
    return coupling ** 2


def compute_precision(hardware: Hardware, target: Target, 
                     integration_time: Optional[float] = None) -> float:
    """Estimate measurement precision (Theorem: Precision Scaling).
    
    sigma(h,t) = sigma_noise(h) / sqrt(A(h,t) * T_int)
    
    Args:
        hardware: Hardware oscillator
        target: Target partition coordinate
        integration_time: Integration time (uses hardware.time if not specified)
        
    Returns:
        Estimated precision (uncertainty)
    """
    if integration_time is None:
        integration_time = hardware.time
        
    accessibility = compute_accessibility(hardware, target)
    
    if accessibility == 0:
        return np.inf
    
    return hardware.noise / np.sqrt(accessibility * integration_time)


def build_accessibility_matrix(hardware_list: List[Hardware],
                               target_list: List[Target]) -> np.ndarray:
    """Construct accessibility matrix (Corollary: Accessibility Matrix).
    
    A[i,j] quantifies how well hardware h_i can measure coordinate t_j.
    
    Args:
        hardware_list: List of available hardware
        target_list: List of target coordinates
        
    Returns:
        N x M accessibility matrix
    """
    N = len(hardware_list)
    M = len(target_list)
    A = np.zeros((N, M))
    
    for i, h in enumerate(hardware_list):
        for j, t in enumerate(target_list):
            A[i, j] = compute_accessibility(h, t)
    
    return A


def compute_quality(hardware_list: List[Hardware],
                   target_list: List[Target],
                   config: List[int],
                   A: np.ndarray) -> float:
    """Compute quality function Q(I, T) (Definition: Instrument Quality Function).
    
    Q(I, T) = sum_t w_t * max_{h in I} [A(h,t) / sigma(h,t)]
    
    Args:
        hardware_list: List of all hardware
        target_list: List of target coordinates
        config: Indices of hardware in configuration
        A: Accessibility matrix
        
    Returns:
        Quality score (higher is better)
    """
    if not config:
        return -np.inf
    
    quality = 0.0
    for j, t in enumerate(target_list):
        # Find best hardware for this target
        best_ratio = 0.0
        for i in config:
            h = hardware_list[i]
            sigma = compute_precision(h, t)
            if sigma < np.inf and sigma > 0:
                ratio = A[i, j] / sigma
                best_ratio = max(best_ratio, ratio)
        
        quality += t.weight * best_ratio
    
    return quality


def optimize_configuration(hardware_list: List[Hardware],
                          target_list: List[Target],
                          constraints: Dict,
                          A: np.ndarray) -> Optional[List[int]]:
    """Find optimal instrument configuration (Step 3 of Algorithm).
    
    Uses exhaustive search for small N, would use heuristics for large N.
    
    Args:
        hardware_list: List of available hardware
        target_list: List of target coordinates
        constraints: Dictionary with 'cost_budget' and 'time_budget'
        A: Accessibility matrix
        
    Returns:
        List of hardware indices in optimal configuration, or None if infeasible
    """
    N = len(hardware_list)
    M = len(target_list)
    
    best_config = None
    best_quality = -np.inf
    
    # Exhaustive search over all subsets
    for mask in range(1, 2**N):
        config = [i for i in range(N) if mask & (1 << i)]
        
        # Check constraints
        total_cost = sum(hardware_list[i].cost for i in config)
        total_time = sum(hardware_list[i].time for i in config)
        
        if total_cost > constraints.get('cost_budget', np.inf):
            continue
        if total_time > constraints.get('time_budget', np.inf):
            continue
        
        # Check precision requirements
        feasible = True
        for j, t in enumerate(target_list):
            # Check if any hardware in config can measure this target
            max_acc = max((A[i, j] for i in config), default=0)
            if max_acc == 0:
                feasible = False
                break
            
            # Check precision for best hardware
            best_sigma = np.inf
            for i in config:
                sigma = compute_precision(hardware_list[i], t)
                best_sigma = min(best_sigma, sigma)
            
            if best_sigma > t.precision:
                feasible = False
                break
        
        if not feasible:
            continue
        
        # Compute quality
        quality = compute_quality(hardware_list, target_list, config, A)
        
        if quality > best_quality:
            best_quality = quality
            best_config = config
    
    return best_config


def generate_protocol(hardware_list: List[Hardware],
                     config: List[int]) -> Dict:
    """Generate measurement protocol (Step 4 of Algorithm).
    
    Args:
        hardware_list: List of all hardware
        config: Indices of selected hardware
        
    Returns:
        Protocol dictionary with settings for each instrument
    """
    protocol = {}
    
    for i in config:
        h = hardware_list[i]
        freq_range = f"{h.frequencies.min():.2e} - {h.frequencies.max():.2e} Hz"
        sample_rate = 2 * np.max(h.frequencies)
        
        protocol[h.name] = {
            'excitation': f'Scan frequency range {freq_range}',
            'acquisition': f'Sample at {sample_rate:.2e} Hz (Nyquist)',
            'integration_time': h.time,
            'calibration': 'Standard calibration against known reference'
        }
    
    return protocol


def create_extraction(hardware_list: List[Hardware],
                     target_list: List[Target],
                     config: List[int],
                     A: np.ndarray) -> Tuple[Dict, Dict[str, float]]:
    """Create coordinate extraction procedure (Step 5 of Algorithm).
    
    Args:
        hardware_list: List of all hardware
        target_list: List of target coordinates
        config: Indices of selected hardware
        A: Accessibility matrix
        
    Returns:
        Tuple of (extraction procedure dict, uncertainty dict)
    """
    extraction = {}
    uncertainties = {}
    
    for j, t in enumerate(target_list):
        # Find hardware that can measure this target
        contributing = [i for i in config if A[i, j] > 0]
        
        if not contributing:
            extraction[t.name] = None
            uncertainties[t.name] = np.inf
            continue
        
        # Multi-hardware fusion: weighted average by accessibility
        weights = np.array([A[i, j] for i in contributing])
        weights = weights / weights.sum()
        
        # Combined uncertainty from multi-instrument fusion
        variances = []
        for i in contributing:
            sigma = compute_precision(hardware_list[i], t)
            variances.append(sigma ** 2)
        
        # Weighted combination of variances
        combined_variance = np.sum(weights ** 2 * np.array(variances))
        uncertainties[t.name] = np.sqrt(combined_variance)
        
        extraction[t.name] = {
            'sources': [hardware_list[i].name for i in contributing],
            'weights': weights.tolist(),
            'algorithm': 'weighted_average',
            'corrections': 'screening_correction' if t.name == 'n' else 'none'
        }
    
    return extraction, uncertainties


def universal_virtual_instrument_finder(
    hardware_list: List[Hardware],
    target_list: List[Target],
    constraints: Dict
) -> Optional[InstrumentConfiguration]:
    """Universal Virtual Instrument Finder - Main Algorithm.
    
    Implementation of Algorithm 1 (UVIF) from the paper.
    
    Args:
        hardware_list: List of available hardware oscillators
        target_list: List of target partition coordinates to measure
        constraints: Dictionary with 'cost_budget' and 'time_budget'
        
    Returns:
        InstrumentConfiguration with optimal settings, or None if no feasible config
    """
    print("=" * 60)
    print("Universal Virtual Instrument Finder (UVIF)")
    print("=" * 60)
    
    # Step 1: Hardware Characterization (already done in Hardware objects)
    print("\nStep 1: Hardware Characterization")
    for h in hardware_list:
        print(f"  {h.name}: {len(h.frequencies)} frequencies, "
              f"noise={h.noise} eV, cost={h.cost}, time={h.time}s")
    
    # Step 2: Accessibility Analysis
    print("\nStep 2: Accessibility Analysis")
    A = build_accessibility_matrix(hardware_list, target_list)
    
    print("  Accessibility Matrix:")
    header = "  " + " " * 20 + " ".join(f"{t.name:>8}" for t in target_list)
    print(header)
    for i, h in enumerate(hardware_list):
        row = f"  {h.name:20}" + " ".join(f"{A[i,j]:>8.4f}" for j in range(len(target_list)))
        print(row)
    
    # Step 3: Instrument Optimization
    print("\nStep 3: Instrument Optimization")
    config = optimize_configuration(hardware_list, target_list, constraints, A)
    
    if config is None:
        print("  ERROR: No feasible configuration found!")
        return None
    
    selected_names = [hardware_list[i].name for i in config]
    print(f"  Optimal configuration: {selected_names}")
    
    # Step 4: Protocol Generation
    print("\nStep 4: Protocol Generation")
    protocol = generate_protocol(hardware_list, config)
    for name, settings in protocol.items():
        print(f"  {name}:")
        for key, value in settings.items():
            print(f"    {key}: {value}")
    
    # Step 5: Extraction Procedure
    print("\nStep 5: Extraction Procedure")
    extraction, uncertainties = create_extraction(hardware_list, target_list, config, A)
    
    for t in target_list:
        if extraction[t.name]:
            print(f"  {t.name}: sources={extraction[t.name]['sources']}, "
                  f"uncertainty={uncertainties[t.name]:.4f}")
    
    # Step 6: Validation
    print("\nStep 6: Validation")
    all_valid = True
    for t in target_list:
        achieved = uncertainties[t.name]
        required = t.precision
        valid = achieved <= required
        status = "OK" if valid else "FAIL"
        print(f"  {t.name}: achieved={achieved:.4f}, required={required:.4f} [{status}]")
        all_valid = all_valid and valid
    
    quality = compute_quality(hardware_list, target_list, config, A)
    
    print("\n" + "=" * 60)
    if all_valid:
        print(f"SUCCESS: All precision requirements met (Quality={quality:.4f})")
    else:
        print("WARNING: Some precision requirements not met")
    print("=" * 60)
    
    return InstrumentConfiguration(
        hardware_indices=config,
        hardware_names=selected_names,
        protocol=protocol,
        extraction=extraction,
        uncertainties=uncertainties,
        quality=quality
    )


# =============================================================================
# Example Hardware and Targets
# =============================================================================

def create_standard_hardware() -> List[Hardware]:
    """Create standard analytical chemistry hardware set."""
    
    mass_spec = Hardware(
        name="Mass Spectrometer",
        frequencies=np.array([1e6, 1e7, 1e8]),  # RF, ion cyclotron, detector
        noise=0.01,  # eV
        cost=100,
        time=1.0
    )
    
    uv_vis = Hardware(
        name="UV-Vis Spectrometer",
        frequencies=np.linspace(1e14, 1e15, 100),  # UV-Vis range
        noise=0.001,  # eV
        cost=50,
        time=10.0
    )
    
    nmr = Hardware(
        name="NMR Spectrometer",
        frequencies=np.array([1.42e9, 3e8, 6e8]),  # 1420 MHz + common NMR
        noise=1e-6,  # eV (very precise)
        cost=200,
        time=60.0
    )
    
    xps = Hardware(
        name="XPS",
        frequencies=np.linspace(1e17, 1e18, 50),  # X-ray frequencies
        noise=0.1,  # eV
        cost=150,
        time=30.0
    )
    
    esr = Hardware(
        name="ESR Spectrometer",
        frequencies=np.array([9.5e9, 35e9]),  # X-band, Q-band
        noise=1e-5,  # eV
        cost=120,
        time=20.0
    )
    
    return [mass_spec, uv_vis, nmr, xps, esr]


def create_hydrogen_targets() -> List[Target]:
    """Create targets for hydrogen ground state measurement."""
    
    # Characteristic frequencies for hydrogen
    # n: Related to ionization energy E = 13.6 eV -> freq ~ E/h ~ 3.3e15 Hz
    # l: Angular modes (l=0 for ground state)
    # m: Orientation (m=0 for ground state)
    # s: Electron spin -> Larmor frequency
    # s_c: Center (proton) spin -> hyperfine at 1420 MHz
    
    return [
        Target(name="n", char_freq=3.3e15, precision=0.01),
        Target(name="l", char_freq=0, precision=0.01),  # l=0 ground state
        Target(name="m", char_freq=0, precision=0.01),  # m=0 ground state
        Target(name="s", char_freq=28e9, precision=0.01),  # Larmor at ~1T
        Target(name="s_c", char_freq=1.42e9, precision=0.01),  # 21 cm line
    ]


def create_carbon_targets() -> List[Target]:
    """Create targets for carbon measurement."""
    return [
        Target(name="n", char_freq=3.3e15/4, precision=0.01),  # n=2 shell
        Target(name="l", char_freq=1e14, precision=0.01),  # l=1 (p orbital)
        Target(name="s", char_freq=28e9, precision=0.01),
    ]


# =============================================================================
# Main Demonstration
# =============================================================================

def demonstrate_hydrogen():
    """Demonstrate UVIF for hydrogen ground state."""
    print("\n" + "=" * 70)
    print("DEMONSTRATION: Hydrogen Ground State Measurement")
    print("=" * 70)
    
    hardware = create_standard_hardware()
    targets = create_hydrogen_targets()
    constraints = {'cost_budget': 500, 'time_budget': 200}
    
    result = universal_virtual_instrument_finder(hardware, targets, constraints)
    
    if result:
        print("\n" + "-" * 40)
        print("FINAL CONFIGURATION SUMMARY")
        print("-" * 40)
        print(f"Selected instruments: {result.hardware_names}")
        print(f"Overall quality: {result.quality:.4f}")
        print(f"Uncertainties: {result.uncertainties}")


def demonstrate_carbon():
    """Demonstrate UVIF for carbon."""
    print("\n" + "=" * 70)
    print("DEMONSTRATION: Carbon Measurement")
    print("=" * 70)
    
    hardware = create_standard_hardware()
    targets = create_carbon_targets()
    constraints = {'cost_budget': 300, 'time_budget': 100}
    
    result = universal_virtual_instrument_finder(hardware, targets, constraints)
    
    if result:
        print("\n" + "-" * 40)
        print("FINAL CONFIGURATION SUMMARY")
        print("-" * 40)
        print(f"Selected instruments: {result.hardware_names}")
        print(f"Overall quality: {result.quality:.4f}")


if __name__ == "__main__":
    demonstrate_hydrogen()
    demonstrate_carbon()

