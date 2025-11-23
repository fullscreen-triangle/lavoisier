"""
S-Entropy Integration with Categorical States

Integrates S-Entropy transformation with BMD categorical state framework.

Key Insight:
S-Entropy coordinates ARE categorical states! Each 14D S-Entropy coordinate
represents a position in the oscillatory completion sequence for MS data.

The 14 dimensions encode:
1. Structural entropy (S) - spectral complexity
2. Shannon entropy (H) - information content
3. Temporal coordination (T) - time-domain structure
4-14. Statistical, geometric, and information-theoretic features

These naturally map to categorical states in phase-locked molecular ensembles.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from .categorical_state import CategoricalState, CategoricalStateSpace
from .bmd_state import BMDState, OscillatoryHole, PhaseStructure


def sentropy_to_categorical_state(
    s_entropy_coords: np.ndarray,
    features: Optional[Dict[str, float]] = None,
    mz_range: Optional[Tuple[float, float]] = None,
    rt_value: Optional[float] = None,
    state_id: Optional[str] = None
) -> CategoricalState:
    """
    Convert S-Entropy coordinates to categorical state.

    S-Entropy 14D coordinates directly encode position in oscillatory
    completion sequence. Higher entropy = later in cascade (more holes filled).

    Args:
        s_entropy_coords: 14D S-Entropy feature vector
        features: Optional feature dictionary
        mz_range: Optional m/z range for this spectrum
        rt_value: Optional retention time
        state_id: Optional state ID (generated if not provided)

    Returns:
        CategoricalState representing this S-Entropy coordinate
    """
    if len(s_entropy_coords) != 14:
        raise ValueError(f"S-Entropy coords must be 14D, got {len(s_entropy_coords)}")

    # Generate state ID if not provided
    if state_id is None:
        # Use hash of coordinates for unique ID
        coord_hash = hash(tuple(s_entropy_coords))
        state_id = f"sentropy_{coord_hash:016x}"

    # Extract key features for phase relationships
    structural_entropy = s_entropy_coords[0] if len(s_entropy_coords) > 0 else 0.0
    shannon_entropy = s_entropy_coords[1] if len(s_entropy_coords) > 1 else 0.0

    # Position in completion sequence = ratio of entropies
    # High S/H ratio = early in sequence (high structure, low info)
    # Low S/H ratio = late in sequence (low structure, high info)
    completion_ratio = structural_entropy / (shannon_entropy + 1e-9)

    # Map to phase relationships (normalized to [0, 2π))
    # Different features map to different oscillatory modes
    phase_relationships = {
        'structural': (s_entropy_coords[0] % 1.0) * 2 * np.pi,
        'shannon': (s_entropy_coords[1] % 1.0) * 2 * np.pi,
        'temporal': (s_entropy_coords[2] % 1.0) * 2 * np.pi if len(s_entropy_coords) > 2 else 0.0,
        'spectral_complexity': (np.mean(s_entropy_coords[3:7]) % 1.0) * 2 * np.pi if len(s_entropy_coords) > 6 else 0.0,
        'geometric': (np.mean(s_entropy_coords[7:11]) % 1.0) * 2 * np.pi if len(s_entropy_coords) > 10 else 0.0,
        'information': (np.mean(s_entropy_coords[11:]) % 1.0) * 2 * np.pi if len(s_entropy_coords) > 11 else 0.0,
    }

    # Estimate categorical richness from Shannon entropy
    # Higher entropy = more categorical possibilities
    categorical_richness = int(np.exp(shannon_entropy * 10) + 1)

    # Estimate binding energy from structural entropy
    # Higher structure = higher binding
    binding_energy = structural_entropy * 10.0  # meV

    # Determine completion position
    from .categorical_state import CompletionPosition
    if completion_ratio > 2.0:
        position = CompletionPosition.INITIAL
    elif completion_ratio > 0.5:
        position = CompletionPosition.INTERMEDIATE
    elif completion_ratio > 0.1:
        position = CompletionPosition.BRANCHING
    else:
        position = CompletionPosition.TERMINAL

    # Create categorical state
    cat_state = CategoricalState(
        state_id=state_id,
        s_entropy_coords=s_entropy_coords,
        phase_relationships=phase_relationships,
        completion_position=position,
        categorical_richness=categorical_richness,
        binding_energy=binding_energy,
        spectral_features=features,
        mz_range=mz_range,
        rt_value=rt_value,
        metadata={'sentropy_integration': True}
    )

    return cat_state


def categorical_state_to_bmd(
    cat_state: CategoricalState,
    hole_size: int = 1000
) -> BMDState:
    """
    Convert categorical state to BMD state.

    Creates oscillatory hole with configurations based on categorical richness.

    Args:
        cat_state: Categorical state
        hole_size: Number of possible configurations in oscillatory hole

    Returns:
        BMD state with hole requiring completion
    """
    # Create phase structure from categorical state
    phase_structure = PhaseStructure()

    for mode_name, phase in cat_state.phase_relationships.items():
        # Estimate frequency from phase (simplified)
        frequency = 1e6  # 1 MHz default
        coherence_time = 1e-6  # 1 microsecond

        phase_structure.add_mode(mode_name, phase, frequency, coherence_time)

    # Create oscillatory hole
    hole = OscillatoryHole(
        hole_id=f"hole_{cat_state.state_id}",
        possible_configurations=set([f"config_{i}" for i in range(hole_size)]),
        selection_frequency=1e14,  # 100 THz (typical for molecular oscillations)
        filling_energy=cat_state.binding_energy,
        categorical_possibilities=cat_state.categorical_richness
    )

    # Create BMD state
    bmd = BMDState(
        bmd_id=f"bmd_{cat_state.state_id}",
        current_categorical_state=cat_state,
        oscillatory_hole=hole,
        phase_structure=phase_structure,
        categorical_richness=cat_state.categorical_richness
    )

    return bmd


def spectrum_to_categorical_space(
    mz_array: np.ndarray,
    intensity_array: np.ndarray,
    rt: float,
    precursor_mz: Optional[float] = None
) -> Tuple[List[CategoricalState], CategoricalStateSpace]:
    """
    Transform mass spectrum to categorical state space.

    Uses S-Entropy transformation to convert spectrum into network of
    categorical states connected by completion sequences.

    Args:
        mz_array: m/z values
        intensity_array: Intensity values
        rt: Retention time
        precursor_mz: Optional precursor m/z

    Returns:
        (list of categorical states, categorical state space)
    """
    # Import S-Entropy transformer
    try:
        from ..core.EntropyTransformation import SEntropyTransformer
    except ImportError:
        raise ImportError("SEntropyTransformer not available. Install precursor.core")

    # Create transformer
    transformer = SEntropyTransformer()

    # Transform spectrum
    coords_list, features = transformer.transform_and_extract(
        mz_array=mz_array,
        intensity_array=intensity_array,
        precursor_mz=precursor_mz,
        rt=rt
    )

    # Create categorical states from coordinates
    categorical_states = []
    state_space = CategoricalStateSpace()

    # Global state from aggregated features
    global_coords = features.to_array()
    global_state = sentropy_to_categorical_state(
        s_entropy_coords=global_coords,
        features=features.to_dict() if hasattr(features, 'to_dict') else None,
        rt_value=rt,
        state_id=f"spectrum_rt{rt:.2f}"
    )

    categorical_states.append(global_state)
    state_space.add_state(global_state)

    # Individual peak states from coordinate list
    for i, coords in enumerate(coords_list):
        if coords is not None and len(coords) == 14:
            # Estimate m/z for this peak
            peak_mz = mz_array[i] if i < len(mz_array) else None
            peak_intensity = intensity_array[i] if i < len(intensity_array) else None

            peak_state = sentropy_to_categorical_state(
                s_entropy_coords=coords,
                mz_range=(peak_mz, peak_mz) if peak_mz else None,
                rt_value=rt,
                state_id=f"peak_rt{rt:.2f}_mz{peak_mz:.4f}" if peak_mz else None
            )

            categorical_states.append(peak_state)
            state_space.add_state(peak_state)

            # Create completion sequence: global → peak
            state_space.add_completion_sequence([global_state.state_id, peak_state.state_id])

    return categorical_states, state_space


def build_spectrum_bmd_network(
    mz_array: np.ndarray,
    intensity_array: np.ndarray,
    rt: float,
    precursor_mz: Optional[float] = None
) -> Tuple[BMDState, List[BMDState]]:
    """
    Build BMD network from mass spectrum.

    Creates hierarchical network of BMD states through categorical completion.

    Args:
        mz_array: m/z values
        intensity_array: Intensity values
        rt: Retention time
        precursor_mz: Optional precursor m/z

    Returns:
        (global BMD, list of peak BMDs)
    """
    # Get categorical states
    cat_states, state_space = spectrum_to_categorical_space(
        mz_array, intensity_array, rt, precursor_mz
    )

    if not cat_states:
        raise ValueError("No categorical states generated from spectrum")

    # Create BMDs from categorical states
    bmds = [categorical_state_to_bmd(state) for state in cat_states]

    # Global BMD (first state)
    global_bmd = bmds[0] if bmds else None

    # Peak BMDs (remaining states)
    peak_bmds = bmds[1:] if len(bmds) > 1 else []

    # Hierarchically merge peak BMDs into global BMD
    if global_bmd and peak_bmds:
        for peak_bmd in peak_bmds[:10]:  # Limit to first 10 for efficiency
            global_bmd = global_bmd.hierarchical_merge(peak_bmd)

    return global_bmd, peak_bmds


def compute_spectrum_ambiguity(
    bmd: BMDState,
    mz_array: np.ndarray,
    intensity_array: np.ndarray
) -> float:
    """
    Compute ambiguity of spectrum interpretation using BMD.

    Args:
        bmd: BMD state for comparison
        mz_array: m/z values
        intensity_array: Intensity values

    Returns:
        Ambiguity measure
    """
    from .bmd_algebra import compute_ambiguity

    # Package spectrum data
    spectrum_data = {
        'mz': mz_array,
        'intensity': intensity_array,
        'n_peaks': len(mz_array)
    }

    # Compute ambiguity
    ambiguity = compute_ambiguity(bmd, spectrum_data)

    return ambiguity


# Example usage
if __name__ == "__main__":
    print("\n" + "="*70)
    print("S-ENTROPY → CATEGORICAL STATE → BMD INTEGRATION")
    print("="*70)

    # Example spectrum
    mz = np.array([100.0, 150.0, 200.0, 250.0, 300.0])
    intensity = np.array([1000.0, 800.0, 1200.0, 900.0, 500.0])
    rt = 15.5

    print(f"\nExample Spectrum:")
    print(f"  Peaks: {len(mz)}")
    print(f"  m/z range: {mz.min():.1f} - {mz.max():.1f}")
    print(f"  RT: {rt:.2f} min")

    try:
        # Transform to categorical space
        print("\n1. Transforming to categorical state space...")
        cat_states, state_space = spectrum_to_categorical_space(mz, intensity, rt)
        print(f"   Generated {len(cat_states)} categorical states")

        # Build BMD network
        print("\n2. Building BMD network...")
        global_bmd, peak_bmds = build_spectrum_bmd_network(mz, intensity, rt)
        print(f"   Global BMD richness: {global_bmd.categorical_richness}")
        print(f"   Peak BMDs: {len(peak_bmds)}")

        # Compute ambiguity
        print("\n3. Computing ambiguity...")
        ambiguity = compute_spectrum_ambiguity(global_bmd, mz, intensity)
        print(f"   Ambiguity: {ambiguity:.3f}")

        print("\n" + "="*70)
        print("SUCCESS: S-Entropy fully integrated with BMD framework")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\nIntegration test incomplete: {e}")
        print("(Expected if S-Entropy transformer not available)")
