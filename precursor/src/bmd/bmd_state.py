"""
BMD State Definition

A Biological Maxwell Demon (BMD) state is an oscillatory hole requiring completion.

Key Concepts:
- BMD = oscillatory hole where one weak force configuration must be selected
  from millions of possibilities to enable cascade continuation
- BMD performs dual filtering: input (noise rejection) + output (targeting)
- BMD is characterized by (current_state, oscillatory_hole, phase_structure)
- Categorical richness R(β) = |hole| × Π_k N_k(Φ) determines information capacity
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any
from .categorical_state import CategoricalState


@dataclass
class PhaseStructure:
    """
    Phase structure of coupled oscillatory modes.

    Represents the collective phase configuration of all oscillatory modes
    coupled to the system (hardware oscillations, molecular vibrations, etc.).

    Attributes:
        modes: Dictionary mapping mode name to phase value (radians)
        frequencies: Dictionary mapping mode name to frequency (Hz)
        coherence_times: Dictionary mapping mode name to coherence lifetime (seconds)
        coupling_strengths: Dictionary of mode-mode coupling strengths
    """
    modes: Dict[str, float] = field(default_factory=dict)  # mode_name -> phase (rad)
    frequencies: Dict[str, float] = field(default_factory=dict)  # mode_name -> freq (Hz)
    coherence_times: Dict[str, float] = field(default_factory=dict)  # mode_name -> τ (s)
    coupling_strengths: Dict[tuple, float] = field(default_factory=dict)  # (mode1, mode2) -> strength

    def add_mode(self, name: str, phase: float, frequency: float,
                 coherence_time: float = 1e-6):
        """Add an oscillatory mode to the structure"""
        self.modes[name] = phase % (2 * np.pi)  # Wrap to [0, 2π)
        self.frequencies[name] = frequency
        self.coherence_times[name] = coherence_time

    def evolve(self, dt: float) -> 'PhaseStructure':
        """
        Evolve phase structure forward in time.

        φ'_k = φ_k + ω_k * dt (mod 2π)

        Args:
            dt: Time step (seconds)

        Returns:
            New PhaseStructure after evolution
        """
        evolved = PhaseStructure(
            modes={},
            frequencies=self.frequencies.copy(),
            coherence_times=self.coherence_times.copy(),
            coupling_strengths=self.coupling_strengths.copy()
        )

        for mode_name, phase in self.modes.items():
            omega = self.frequencies.get(mode_name, 0.0)
            new_phase = (phase + 2 * np.pi * omega * dt) % (2 * np.pi)
            evolved.modes[mode_name] = new_phase

        return evolved

    def phase_lock_distance(self, other: 'PhaseStructure') -> float:
        """
        Compute phase-lock distance to another structure.

        Lower distance indicates stronger phase coherence.

        Args:
            other: Another phase structure

        Returns:
            Phase distance (0 = perfect phase-lock)
        """
        if not self.modes or not other.modes:
            return np.inf

        common_modes = set(self.modes.keys()) & set(other.modes.keys())

        if not common_modes:
            return np.inf

        total_distance = 0.0

        for mode in common_modes:
            phase_diff = abs(self.modes[mode] - other.modes[mode])
            # Handle wrapping
            phase_diff = min(phase_diff, 2*np.pi - phase_diff)
            total_distance += phase_diff ** 2

        return np.sqrt(total_distance / len(common_modes))

    def is_phase_locked(self, other: 'PhaseStructure',
                       threshold: float = np.pi/4) -> bool:
        """
        Check if phase-locked with another structure.

        Args:
            other: Another phase structure
            threshold: Maximum phase difference for phase-lock

        Returns:
            True if phase-locked
        """
        return self.phase_lock_distance(other) < threshold

    def merge(self, other: 'PhaseStructure') -> 'PhaseStructure':
        """
        Merge with another phase structure through hierarchical composition.

        This is the ⊛ operation: β₁ ⊛ β₂

        Args:
            other: Another phase structure

        Returns:
            Merged phase structure
        """
        merged = PhaseStructure(
            modes=self.modes.copy(),
            frequencies=self.frequencies.copy(),
            coherence_times=self.coherence_times.copy(),
            coupling_strengths=self.coupling_strengths.copy()
        )

        # Add modes from other (averaging if overlapping)
        for mode_name, phase in other.modes.items():
            if mode_name in merged.modes:
                # Average phases (accounting for circular nature)
                phase1 = merged.modes[mode_name]
                phase2 = phase
                # Convert to complex numbers, average, convert back
                avg_complex = (np.exp(1j * phase1) + np.exp(1j * phase2)) / 2
                merged.modes[mode_name] = np.angle(avg_complex) % (2 * np.pi)
            else:
                merged.modes[mode_name] = phase
                if mode_name in other.frequencies:
                    merged.frequencies[mode_name] = other.frequencies[mode_name]
                if mode_name in other.coherence_times:
                    merged.coherence_times[mode_name] = other.coherence_times[mode_name]

        # Merge coupling strengths
        merged.coupling_strengths.update(other.coupling_strengths)

        return merged


@dataclass
class OscillatoryHole:
    """
    Oscillatory hole requiring completion.

    A hole is a physical absence in oscillatory cascade where one weak force
    configuration must be selected from many possibilities.

    Attributes:
        hole_id: Unique identifier
        possible_configurations: Set of weak force configurations that can fill hole
        selection_frequency: Frequency at which selections occur (Hz)
        filling_energy: Energy required to fill hole (meV)
        categorical_possibilities: Number of distinct categorical outcomes
    """
    hole_id: str
    possible_configurations: Set[str] = field(default_factory=set)
    selection_frequency: float = 1e14  # Hz (typical for O₂ at 300K)
    filling_energy: float = 10.0  # meV
    categorical_possibilities: int = 1

    # Metadata
    created_from_state: Optional[str] = None  # State ID that created this hole
    metadata: Dict[str, Any] = field(default_factory=dict)

    def size(self) -> int:
        """Number of possible configurations"""
        return len(self.possible_configurations)

    def information_content(self) -> float:
        """
        Information content of hole in bits.

        I = log₂(N) where N is number of possibilities
        """
        N = max(self.size(), 1)
        return np.log2(N)

    def landauer_bound(self, temperature: float = 300.0) -> float:
        """
        Landauer bound on energy dissipation for hole filling.

        E_fill ≥ k_B T ln(N)

        Args:
            temperature: Temperature in Kelvin

        Returns:
            Minimum energy dissipation (meV)
        """
        k_B = 8.617e-5  # eV/K
        N = max(self.size(), 1)
        return k_B * temperature * np.log(N) * 1e3  # Convert to meV


@dataclass
class BMDState:
    """
    Biological Maxwell Demon state.

    A BMD is an oscillatory hole requiring completion. It performs dual filtering:
    - Input filter ℑ_input: Y_↓^(in) → Y_↑^(in) (select signal from noise)
    - Output filter ℑ_output: Z_↓^(fin) → Z_↑^(fin) (target physical interpretations)

    Attributes:
        bmd_id: Unique identifier
        current_categorical_state: Current position in completion sequence
        oscillatory_hole: Hole requiring completion
        phase_structure: Phase configuration of coupled modes
        categorical_richness: R(β) = |hole| × Π_k N_k(Φ)
        history: Completion history (sequence of states)
    """
    bmd_id: str
    current_categorical_state: Optional[CategoricalState] = None
    oscillatory_hole: Optional[OscillatoryHole] = None
    phase_structure: PhaseStructure = field(default_factory=PhaseStructure)
    categorical_richness: int = 1
    history: List[str] = field(default_factory=list)  # State IDs in completion order

    # Filters
    input_filter_threshold: float = 0.5  # Phase coherence threshold for input
    output_filter_threshold: float = 0.3  # Stream coherence threshold for output

    # Metadata
    created_at: float = 0.0  # Timestamp
    device_source: Optional[str] = None  # Which hardware device created this
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Compute derived quantities"""
        self._update_categorical_richness()

    def _update_categorical_richness(self):
        """
        Compute categorical richness R(β).

        R(β) = |H(c_current)| × Π_k N_k(Φ)

        where |H| is hole size and N_k is accessible phase configs in mode k.
        """
        hole_size = self.oscillatory_hole.size() if self.oscillatory_hole else 1

        # Estimate accessible phase configurations per mode
        # Assuming ~10 distinguishable phases per mode (0.1 rad resolution)
        phase_configs = len(self.phase_structure.modes) ** 10 if self.phase_structure.modes else 1

        self.categorical_richness = hole_size * phase_configs

    def input_filter(self, candidates: List[Any],
                    criterion: str = 'phase_lock') -> List[Any]:
        """
        Input filter: Select signal from noise.

        ℑ_input: Y_↓^(in) → Y_↑^(in)

        Filters based on phase-lock coherence with current BMD state.

        Args:
            candidates: List of candidate signals (peaks, features, etc.)
            criterion: Filtering criterion ('phase_lock', 'energy', 'frequency')

        Returns:
            Filtered candidates that pass input filter
        """
        if criterion == 'phase_lock':
            # Filter by phase coherence
            filtered = []
            for candidate in candidates:
                # Check if candidate has phase information
                if hasattr(candidate, 'phase') or isinstance(candidate, dict) and 'phase' in candidate:
                    candidate_phase = candidate.phase if hasattr(candidate, 'phase') else candidate['phase']

                    # Check coherence with any mode in phase structure
                    is_coherent = False
                    for mode_phase in self.phase_structure.modes.values():
                        phase_diff = abs(candidate_phase - mode_phase)
                        phase_diff = min(phase_diff, 2*np.pi - phase_diff)

                        if phase_diff < self.input_filter_threshold * np.pi:
                            is_coherent = True
                            break

                    if is_coherent:
                        filtered.append(candidate)

            return filtered

        return candidates  # Default: pass all

    def output_filter(self, interpretations: List[Any],
                     hardware_bmd: 'BMDState') -> List[Any]:
        """
        Output filter: Target physically-grounded interpretations.

        ℑ_output: Z_↓^(fin) → Z_↑^(fin)

        Only accepts interpretations coherent with hardware BMD stream.

        Args:
            interpretations: List of possible interpretations
            hardware_bmd: Hardware BMD stream for reality grounding

        Returns:
            Filtered interpretations that are physically grounded
        """
        filtered = []

        for interp in interpretations:
            # Check stream coherence
            # If interpretation has phase structure, check phase-lock with hardware
            if hasattr(interp, 'phase_structure'):
                if self.phase_structure.is_phase_locked(hardware_bmd.phase_structure,
                                                       self.output_filter_threshold * np.pi):
                    filtered.append(interp)
            else:
                # Default: accept if no phase info
                filtered.append(interp)

        return filtered

    def compare_with(self, target: Any) -> float:
        """
        Compare BMD state with a target (spectrum, region, etc.).

        Returns ambiguity measure A(β, R).

        Args:
            target: Target to compare against

        Returns:
            Ambiguity (high = many possible interpretations)
        """
        # This will be implemented in bmd_algebra.py
        # For now, return categorical richness as proxy
        return float(self.categorical_richness)

    def generate_from_comparison(self, target: Any,
                                hardware_bmd: 'BMDState') -> 'BMDState':
        """
        Generate new BMD state from comparison.

        β' = Generate(β, R)

        Completes current oscillatory hole and creates new hole.

        Args:
            target: Target that was compared
            hardware_bmd: Hardware BMD for grounding

        Returns:
            New BMD state after completion
        """
        # This will be implemented in bmd_algebra.py
        # For now, return self (placeholder)
        return self

    def hierarchical_merge(self, other: 'BMDState') -> 'BMDState':
        """
        Hierarchical merge with another BMD state.

        β_compound = β₁ ⊛ β₂

        Creates compound BMD through phase-lock coupling.

        Args:
            other: Another BMD state

        Returns:
            Compound BMD state
        """
        compound = BMDState(
            bmd_id=f"{self.bmd_id}⊛{other.bmd_id}",
            current_categorical_state=self.current_categorical_state,  # Keep first state
            oscillatory_hole=self.oscillatory_hole,  # Merge holes
            phase_structure=self.phase_structure.merge(other.phase_structure),
            history=self.history + other.history
        )

        # Merge oscillatory holes
        if self.oscillatory_hole and other.oscillatory_hole:
            merged_hole = OscillatoryHole(
                hole_id=f"{self.oscillatory_hole.hole_id}⊛{other.oscillatory_hole.hole_id}",
                possible_configurations=self.oscillatory_hole.possible_configurations |
                                      other.oscillatory_hole.possible_configurations,
                categorical_possibilities=self.oscillatory_hole.categorical_possibilities *
                                        other.oscillatory_hole.categorical_possibilities
            )
            compound.oscillatory_hole = merged_hole

        compound._update_categorical_richness()

        return compound

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return {
            'bmd_id': self.bmd_id,
            'current_categorical_state': self.current_categorical_state.to_dict()
                                        if self.current_categorical_state else None,
            'oscillatory_hole': {
                'hole_id': self.oscillatory_hole.hole_id,
                'possible_configurations': list(self.oscillatory_hole.possible_configurations),
                'selection_frequency': self.oscillatory_hole.selection_frequency,
                'filling_energy': self.oscillatory_hole.filling_energy,
                'categorical_possibilities': self.oscillatory_hole.categorical_possibilities,
            } if self.oscillatory_hole else None,
            'phase_structure': {
                'modes': self.phase_structure.modes,
                'frequencies': self.phase_structure.frequencies,
                'coherence_times': self.phase_structure.coherence_times,
            },
            'categorical_richness': self.categorical_richness,
            'history': self.history,
            'input_filter_threshold': self.input_filter_threshold,
            'output_filter_threshold': self.output_filter_threshold,
            'created_at': self.created_at,
            'device_source': self.device_source,
            'metadata': self.metadata,
        }
