"""
Categorical State Definition

A categorical state is an equivalence class of physical configurations that share
identical positions in an oscillatory completion sequence.

Key Concepts:
- Configurations belong to same categorical state if they have identical phase
  relationships in all coupled oscillatory modes
- Categorical states form completion sequences through oscillatory hole filling
- Entropy defined by probability of oscillatory cascade termination at state
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum


class CompletionPosition(Enum):
    """Position in categorical completion sequence"""
    INITIAL = "initial"           # Starting state
    INTERMEDIATE = "intermediate" # Mid-sequence
    TERMINAL = "terminal"         # End state
    BRANCHING = "branching"       # Multiple paths available


@dataclass
class CategoricalState:
    """
    A categorical state in the completion sequence.

    Represents equivalence class of physical configurations with identical
    phase relationships in all coupled oscillatory modes.

    Attributes:
        state_id: Unique identifier for this categorical state
        s_entropy_coords: S-Entropy coordinates (platform-independent 14D feature space)
        phase_relationships: Phase values for each coupled oscillatory mode
        completion_position: Position in oscillatory completion sequence
        accessible_states: Set of states accessible from this one via hole completion
        binding_energy: Energy required to reach this state (meV)
        categorical_richness: Number of distinct completion pathways available
    """
    state_id: str
    s_entropy_coords: Optional[np.ndarray] = None  # 14D from S-Entropy
    phase_relationships: Dict[str, float] = field(default_factory=dict)
    completion_position: CompletionPosition = CompletionPosition.INTERMEDIATE
    accessible_states: List[str] = field(default_factory=list)
    binding_energy: float = 0.0  # meV
    categorical_richness: int = 1

    # Additional spectral information
    spectral_features: Optional[Dict[str, float]] = None
    mz_range: Optional[Tuple[float, float]] = None
    rt_value: Optional[float] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and compute derived quantities"""
        if self.s_entropy_coords is not None:
            if len(self.s_entropy_coords) != 14:
                raise ValueError(f"S-Entropy coords must be 14D, got {len(self.s_entropy_coords)}")

    def is_phase_locked_with(self, other: 'CategoricalState',
                            phase_threshold: float = np.pi/4) -> bool:
        """
        Check if this state is phase-locked with another.

        Phase-locking criterion: |φ_i - φ_j| < π/4 for all modes

        Args:
            other: Another categorical state
            phase_threshold: Maximum phase difference (default π/4 rad = 45°)

        Returns:
            True if states are phase-locked
        """
        if not self.phase_relationships or not other.phase_relationships:
            return False

        common_modes = set(self.phase_relationships.keys()) & set(other.phase_relationships.keys())

        if not common_modes:
            return False

        for mode in common_modes:
            phase_diff = abs(self.phase_relationships[mode] - other.phase_relationships[mode])
            # Handle phase wrapping (0 ≈ 2π)
            phase_diff = min(phase_diff, 2*np.pi - phase_diff)

            if phase_diff > phase_threshold:
                return False

        return True

    def compute_entropy(self, completion_sequences: List[List[str]],
                       temperature: float = 300.0) -> float:
        """
        Compute oscillatory entropy of this state.

        S(c) = -k_B Σ_σ P(σ) log P(σ)

        where sum is over all completion sequences containing this state.

        Args:
            completion_sequences: List of sequences containing this state
            temperature: Temperature in Kelvin

        Returns:
            Entropy in units of k_B
        """
        k_B = 8.617e-5  # eV/K

        if not completion_sequences:
            return 0.0

        # Compute probability of each sequence
        Z = 0.0  # Partition function
        sequence_probs = []

        for seq in completion_sequences:
            # P(σ) ∝ exp(-E_total / k_B T)
            # For now, assume uniform energy per transition
            energy = len(seq) * self.binding_energy
            prob = np.exp(-energy / (k_B * temperature))
            sequence_probs.append(prob)
            Z += prob

        # Normalize probabilities
        sequence_probs = [p/Z for p in sequence_probs]

        # Compute entropy
        entropy = -sum(p * np.log(p) if p > 0 else 0 for p in sequence_probs)

        return entropy

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return {
            'state_id': self.state_id,
            's_entropy_coords': self.s_entropy_coords.tolist() if self.s_entropy_coords is not None else None,
            'phase_relationships': self.phase_relationships,
            'completion_position': self.completion_position.value,
            'accessible_states': self.accessible_states,
            'binding_energy': self.binding_energy,
            'categorical_richness': self.categorical_richness,
            'spectral_features': self.spectral_features,
            'mz_range': self.mz_range,
            'rt_value': self.rt_value,
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CategoricalState':
        """Deserialize from dictionary"""
        data_copy = data.copy()

        # Handle enum
        if 'completion_position' in data_copy:
            data_copy['completion_position'] = CompletionPosition(data_copy['completion_position'])

        # Handle numpy array
        if 's_entropy_coords' in data_copy and data_copy['s_entropy_coords'] is not None:
            data_copy['s_entropy_coords'] = np.array(data_copy['s_entropy_coords'])

        return cls(**data_copy)


@dataclass
class CategoricalStateSpace:
    """
    Complete space of categorical states for a system.

    Manages the network of categorical states and completion sequences.
    """
    states: Dict[str, CategoricalState] = field(default_factory=dict)
    completion_sequences: List[List[str]] = field(default_factory=list)

    def add_state(self, state: CategoricalState):
        """Add a categorical state to the space"""
        self.states[state.state_id] = state

    def add_completion_sequence(self, sequence: List[str]):
        """
        Add a completion sequence.

        Args:
            sequence: List of state IDs forming a completion path
        """
        # Validate all states exist
        for state_id in sequence:
            if state_id not in self.states:
                raise ValueError(f"State {state_id} not in state space")

        self.completion_sequences.append(sequence)

        # Update accessible_states for each state in sequence
        for i in range(len(sequence) - 1):
            current_state = self.states[sequence[i]]
            next_state_id = sequence[i + 1]

            if next_state_id not in current_state.accessible_states:
                current_state.accessible_states.append(next_state_id)

    def find_phase_locked_ensemble(self, reference_state: CategoricalState,
                                   phase_threshold: float = np.pi/4) -> List[CategoricalState]:
        """
        Find all states phase-locked with reference state.

        Forms ensemble of ~10^3-10^4 states with mutual phase coherence.

        Args:
            reference_state: State to find ensemble around
            phase_threshold: Phase coherence threshold

        Returns:
            List of phase-locked states
        """
        ensemble = []

        for state in self.states.values():
            if state.is_phase_locked_with(reference_state, phase_threshold):
                ensemble.append(state)

        return ensemble

    def compute_transition_probability(self, from_state_id: str, to_state_id: str,
                                      temperature: float = 300.0) -> float:
        """
        Compute probability of transition from one state to another.

        P(c_i → c_j) ∝ exp(-E_fill / k_B T)

        where E_fill is energy required to fill oscillatory hole.

        Args:
            from_state_id: Starting state
            to_state_id: Target state
            temperature: Temperature in Kelvin

        Returns:
            Transition probability
        """
        if from_state_id not in self.states or to_state_id not in self.states:
            return 0.0

        from_state = self.states[from_state_id]
        to_state = self.states[to_state_id]

        # Check if transition is allowed
        if to_state_id not in from_state.accessible_states:
            return 0.0

        k_B = 8.617e-5  # eV/K

        # Energy to fill hole (Landauer bound)
        # E_fill ≥ k_B T log(N) where N is categorical richness
        N_hole = from_state.categorical_richness
        E_fill = k_B * temperature * np.log(N_hole) * 1e3  # Convert to meV

        # Boltzmann factor
        prob = np.exp(-E_fill / (k_B * temperature * 1e3))

        return prob

    def get_completion_paths(self, from_state_id: str, to_state_id: str) -> List[List[str]]:
        """
        Find all completion paths from one state to another.

        Args:
            from_state_id: Starting state
            to_state_id: Target state

        Returns:
            List of paths (each path is list of state IDs)
        """
        paths = []

        def dfs(current_id: str, target_id: str, path: List[str], visited: set):
            """Depth-first search for paths"""
            if current_id == target_id:
                paths.append(path.copy())
                return

            if current_id not in self.states:
                return

            current_state = self.states[current_id]

            for next_id in current_state.accessible_states:
                if next_id not in visited:
                    visited.add(next_id)
                    path.append(next_id)
                    dfs(next_id, target_id, path, visited)
                    path.pop()
                    visited.remove(next_id)

        visited = {from_state_id}
        dfs(from_state_id, to_state_id, [from_state_id], visited)

        return paths

    def compute_total_categorical_richness(self) -> int:
        """
        Compute total categorical richness of state space.

        This is the number of possible completion sequences.
        """
        return len(self.completion_sequences)

    def save(self, filepath: str):
        """Save state space to JSON file"""
        import json

        data = {
            'states': {state_id: state.to_dict() for state_id, state in self.states.items()},
            'completion_sequences': self.completion_sequences,
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'CategoricalStateSpace':
        """Load state space from JSON file"""
        import json

        with open(filepath, 'r') as f:
            data = json.load(f)

        space = cls()

        # Load states
        for state_id, state_data in data['states'].items():
            space.add_state(CategoricalState.from_dict(state_data))

        # Load sequences
        for seq in data['completion_sequences']:
            space.completion_sequences.append(seq)

        return space
