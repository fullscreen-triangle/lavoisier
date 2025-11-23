"""
BMD Algebra Operations

Implements comparison, generation, and integration operations for BMD states.

Key Operations:
1. Comparison: A(β, R) - measure ambiguity between BMD and target
2. Generation: β' = Generate(β, R) - create new BMD through categorical completion
3. Stream Divergence: D_stream(β^network, β^hardware) - measure drift from hardware
4. Hierarchical Integration: β^(network)_{i+1} = IntegrateHierarchical(...)
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from scipy.stats import entropy as kl_divergence_scipy
from .bmd_state import BMDState, OscillatoryHole, PhaseStructure
from .categorical_state import CategoricalState


def compute_ambiguity(bmd: BMDState, target: Any,
                     target_categorical_states: Optional[List[CategoricalState]] = None) -> float:
    """
    Compute ambiguity measure A(β, R).

    A(β, R) = Σ_{c ∈ C(R)} P(c|R) · D_KL(P_complete(c|β) || P_image(c|R))

    High ambiguity = many incompatible completion pathways
    Low ambiguity = strong alignment

    Args:
        bmd: BMD state
        target: Target to compare (spectrum, region, etc.)
        target_categorical_states: Optional list of categorical states for target

    Returns:
        Ambiguity value (higher = more uncertain)
    """
    # If no categorical states provided, use categorical richness as proxy
    if target_categorical_states is None or not target_categorical_states:
        # Simple ambiguity: based on categorical richness and hole size
        if bmd.oscillatory_hole:
            hole_info = bmd.oscillatory_hole.information_content()
            return hole_info * np.log(bmd.categorical_richness)
        return float(bmd.categorical_richness)

    # Compute proper ambiguity with categorical states
    total_ambiguity = 0.0

    for cat_state in target_categorical_states:
        # P(c|R): Probability target is in this categorical state
        # For now, assume uniform over compatible states
        p_c_given_R = 1.0 / len(target_categorical_states)

        # D_KL(P_complete(c|β) || P_image(c|R))
        # Measures information needed to update BMD's distribution to match target

        # P_complete(c|β): Probability of completing BMD into state c
        # Depends on phase-lock coherence
        phase_coherence = 0.0
        if bmd.current_categorical_state:
            phase_coherence = 1.0 if bmd.current_categorical_state.is_phase_locked_with(cat_state) else 0.1
        else:
            phase_coherence = 0.5  # Neutral prior

        p_complete = phase_coherence

        # P_image(c|R): Probability based on target data
        p_image = p_c_given_R

        # KL divergence (handle edge cases)
        if p_complete > 0 and p_image > 0:
            kl_div = p_complete * np.log(p_complete / p_image)
        else:
            kl_div = 1.0  # High divergence if distributions don't overlap

        total_ambiguity += p_c_given_R * kl_div

    return total_ambiguity


def generate_bmd_from_comparison(bmd: BMDState, target: Any,
                                 hardware_bmd: BMDState,
                                 lambda_coupling: float = 0.1) -> BMDState:
    """
    Generate new BMD state from comparison.

    β' = Generate(β, R) = ⟨c_new, H(c_new), Φ'⟩

    Completes oscillatory hole by selecting one configuration from possibilities,
    then creates new hole for continued cascade.

    Args:
        bmd: Current BMD state
        target: Target being compared
        hardware_bmd: Hardware BMD for grounding
        lambda_coupling: Coupling parameter for energy/information trade-off

    Returns:
        New BMD state after completion
    """
    # Select new categorical state by completing hole
    # c_new = argmin_c [E_fill(c_current → c) + λ · A(β_c, R)]

    # For now, create new state with updated phase structure
    new_phase_structure = bmd.phase_structure.evolve(dt=1e-6)  # Evolve 1 microsecond

    # Create new oscillatory hole
    # After completion, new hole emerges with different configurations
    new_hole = OscillatoryHole(
        hole_id=f"{bmd.bmd_id}_generated",
        possible_configurations=set([f"config_{i}" for i in range(100)]),  # Placeholder
        categorical_possibilities=bmd.oscillatory_hole.categorical_possibilities if bmd.oscillatory_hole else 100
    )

    # Create new BMD state
    new_bmd = BMDState(
        bmd_id=f"{bmd.bmd_id}_gen",
        current_categorical_state=bmd.current_categorical_state,  # Update in full implementation
        oscillatory_hole=new_hole,
        phase_structure=new_phase_structure,
        history=bmd.history + [bmd.bmd_id],
        device_source=bmd.device_source
    )

    return new_bmd


def compute_stream_divergence(network_bmd: BMDState, hardware_stream: BMDState,
                              device_names: Optional[List[str]] = None) -> float:
    """
    Compute stream divergence D_stream(β^network, β^stream_hardware).

    D_stream = Σ_device D_KL(P_phase^network || P_phase^hardware,device)

    Measures how far network BMD has drifted from physical hardware reality.

    Args:
        network_bmd: Network BMD from processing
        hardware_stream: Hardware BMD stream (reality reference)
        device_names: List of device names to check (or all if None)

    Returns:
        Stream divergence (higher = more drift from reality)
    """
    if device_names is None:
        # Use all devices in hardware stream
        device_names = list(hardware_stream.phase_structure.modes.keys())

    total_divergence = 0.0

    for device in device_names:
        # Get phases for this device
        if device not in network_bmd.phase_structure.modes:
            # Network missing this device - high penalty
            total_divergence += 10.0
            continue

        if device not in hardware_stream.phase_structure.modes:
            # Hardware missing this device - skip
            continue

        network_phase = network_bmd.phase_structure.modes[device]
        hardware_phase = hardware_stream.phase_structure.modes[device]

        # Phase difference (handling wrapping)
        phase_diff = abs(network_phase - hardware_phase)
        phase_diff = min(phase_diff, 2*np.pi - phase_diff)

        # Convert to divergence (normalized)
        divergence = phase_diff / np.pi  # Range [0, 1]

        total_divergence += divergence

    return total_divergence


def integrate_hierarchical(network_bmd: BMDState, new_bmd: BMDState,
                           processing_sequence: List[str]) -> BMDState:
    """
    Hierarchically integrate new BMD into network BMD.

    β^(network)_{i+1} = IntegrateHierarchical(β^(network)_i, β_{i+1}, σ ∪ (R_next))

    This creates:
    1. Pairwise compound BMDs with previously processed regions
    2. Higher-order compound BMDs from longer sequences
    3. Global BMD encoding complete categorical history

    Args:
        network_bmd: Current network BMD
        new_bmd: New BMD to integrate
        processing_sequence: Sequence of processed regions/stages

    Returns:
        Updated network BMD
    """
    # Hierarchical merge through ⊛ operation
    integrated = network_bmd.hierarchical_merge(new_bmd)

    # Update history to include processing sequence
    integrated.history = processing_sequence.copy()

    # Categorical richness grows with integration
    # O(2^n) compound BMDs through hierarchical composition
    n_processed = len(processing_sequence)
    integrated.categorical_richness = network_bmd.categorical_richness * (2 ** min(n_processed, 10))

    return integrated


def compare_bmd_with_region(bmd: BMDState, region_data: Any,
                            hardware_bmd: BMDState,
                            lambda_coupling: float = 0.1) -> Tuple[float, float]:
    """
    Full comparison of BMD with data region.

    Computes both:
    1. Ambiguity A(β, R)
    2. Stream divergence D_stream(β ⊛ R, β^hardware)

    Used for region selection in iterative BMD algorithm:
    R_next = argmax_R [A(β, R) - λ · D_stream(β ⊛ R, β^hardware)]

    Args:
        bmd: Current BMD state
        region_data: Data region to compare
        hardware_bmd: Hardware BMD stream
        lambda_coupling: Coupling parameter

    Returns:
        (ambiguity, stream_divergence) tuple
    """
    # Compute ambiguity
    ambiguity = compute_ambiguity(bmd, region_data)

    # Hypothetically integrate region and compute stream divergence
    hypothetical_bmd = generate_bmd_from_comparison(bmd, region_data, hardware_bmd)
    stream_div = compute_stream_divergence(hypothetical_bmd, hardware_bmd)

    return ambiguity, stream_div


def select_next_region(network_bmd: BMDState,
                      available_regions: List[Any],
                      hardware_bmd: BMDState,
                      lambda_coupling: float = 0.1) -> Tuple[Any, float]:
    """
    Select next region for processing.

    R_next = argmax_R [A(β^network, R) - λ · D_stream(β^network ⊛ R, β^hardware)]

    Dual objective:
    - Maximize ambiguity (explore high-categorical-richness regions)
    - Maintain hardware coherence (stay grounded in reality)

    Args:
        network_bmd: Current network BMD
        available_regions: List of unprocessed regions
        hardware_bmd: Hardware BMD stream
        lambda_coupling: Trade-off parameter

    Returns:
        (selected_region, selection_score) tuple
    """
    best_region = None
    best_score = -np.inf

    for region in available_regions:
        ambiguity, stream_div = compare_bmd_with_region(
            network_bmd, region, hardware_bmd, lambda_coupling
        )

        # Dual objective score
        score = ambiguity - lambda_coupling * stream_div

        if score > best_score:
            best_score = score
            best_region = region

    return best_region, best_score


def check_termination_criterion(network_bmd: BMDState,
                                available_regions: List[Any],
                                hardware_bmd: BMDState,
                                coherence_threshold: float = 1.0) -> bool:
    """
    Check if processing should terminate.

    Terminates when network coherence achieved:
    A(β^network, R) < A_coherence for all R

    Args:
        network_bmd: Current network BMD
        available_regions: Unprocessed regions
        hardware_bmd: Hardware BMD stream
        coherence_threshold: Ambiguity threshold for coherence

    Returns:
        True if should terminate
    """
    if not available_regions:
        return True

    for region in available_regions:
        ambiguity = compute_ambiguity(network_bmd, region)

        if ambiguity >= coherence_threshold:
            return False  # Still have high-ambiguity regions

    return True  # All regions below threshold


def check_revisitation(network_bmd_current: BMDState,
                      network_bmd_previous: BMDState,
                      region: Any,
                      previous_step: int) -> bool:
    """
    Check if region should be revisited.

    Revisit if: A(β^network_current, R) > A(β^network_previous, R)

    Network evolution can INCREASE ambiguity for previously processed regions
    through new categorical connections.

    Args:
        network_bmd_current: Current network BMD
        network_bmd_previous: Network BMD when region was first processed
        region: Region to check
        previous_step: Step when region was processed

    Returns:
        True if should revisit
    """
    current_ambiguity = compute_ambiguity(network_bmd_current, region)
    previous_ambiguity = compute_ambiguity(network_bmd_previous, region)

    return current_ambiguity > previous_ambiguity


def compute_information_dissipation(bmd_initial: BMDState, bmd_final: BMDState,
                                   temperature: float = 300.0) -> float:
    """
    Compute energy dissipation from BMD transition.

    E_dissipate = k_B T log(R(β_initial) / R(β_final))

    Landauer's principle: Information reduction requires energy dissipation.

    Args:
        bmd_initial: Initial BMD state
        bmd_final: Final BMD state
        temperature: Temperature in Kelvin

    Returns:
        Energy dissipated (meV)
    """
    k_B = 8.617e-5  # eV/K

    R_initial = max(bmd_initial.categorical_richness, 1)
    R_final = max(bmd_final.categorical_richness, 1)

    if R_initial <= R_final:
        return 0.0  # No reduction, no required dissipation

    energy = k_B * temperature * np.log(R_initial / R_final) * 1e3  # Convert to meV

    return energy


def compute_phase_lock_ensemble(bmd: BMDState,
                                candidate_bmds: List[BMDState],
                                phase_threshold: float = np.pi/4) -> List[BMDState]:
    """
    Find phase-locked ensemble of BMDs.

    Ensemble = {β_i : |φ_i - φ_j| < π/4 for all j}

    Forms ~10³-10⁴ BMDs with mutual phase coherence.

    Args:
        bmd: Reference BMD
        candidate_bmds: Candidate BMDs to check
        phase_threshold: Phase coherence threshold

    Returns:
        List of phase-locked BMDs
    """
    ensemble = [bmd]  # Include reference

    for candidate in candidate_bmds:
        if bmd.phase_structure.is_phase_locked(candidate.phase_structure, phase_threshold):
            ensemble.append(candidate)

    return ensemble
