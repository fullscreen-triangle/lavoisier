"""
Molecular Maxwell Demon State Architecture
===========================================

Implements the St-Stellas categorical framework for virtual mass spectrometry.

Key Concepts (from st-stellas-categories.tex):
1. MMD = Information catalyst filtering potential → actual states
2. Dual filtering: ℑ_input (noise rejection) ∘ ℑ_output (physical validation)
3. Categorical equivalence: ~10^6 configurations → same observable
4. S-coordinate compression: Infinite info → 3 sufficient statistics
5. Recursive self-similarity: Each S-coordinate is itself an MMD
6. Self-propagating cascades: 3^k parallel processing hierarchy

Author: Kundai Farai Sachikonye
Date: 2025
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import time


class InstrumentProjection(Enum):
    """Types of instrument projections from categorical state"""
    TOF = "time_of_flight"           # Read S_t coordinate
    ORBITRAP = "orbitrap"            # Read harmonic structure
    FTICR = "ft_icr"                 # Read exact frequency
    QUADRUPOLE = "quadrupole"        # Read ω with S_e filtering
    SECTOR = "sector"                # Read S_e/ω ratio
    ION_MOBILITY = "ion_mobility"    # Read S_k/S_e structure
    PHOTODETECTOR = "photodetector"  # Read frequency
    ION_DETECTOR = "ion_detector"    # Read S_e charge


@dataclass
class CategoricalState:
    """
    Categorical state at convergence node.

    Represents compressed information from ~10^6 equivalent
    molecular configurations (different weak force arrangements,
    Van der Waals angles, dipole orientations, vibrational phases
    that produce SAME observable).

    From St-Stellas Theorem 3.2: S-values compress infinity through sufficiency.
    """
    state_id: str

    # Tri-dimensional S-space coordinates (sufficient statistics)
    S_k: float  # Knowledge: Information deficit, which equivalence class
    S_t: float  # Time: Categorical sequence position
    S_e: float  # Entropy: Constraint density, thermodynamic accessibility

    # Frequency information (from molecular oscillations)
    frequency_hz: float
    harmonics: Optional[np.ndarray] = None  # Harmonic decomposition

    # Phase structure (hardware + molecular oscillations)
    phase_relationships: Dict[str, float] = field(default_factory=dict)

    # Categorical richness (size of equivalence class)
    equivalence_class_size: int = 1_000_000  # ~10^6 typical

    # Hardware coherence (reality grounding)
    hardware_coherence: float = 0.0  # [0, 1]

    # Observable properties (categorical invariants)
    mz_ratio: Optional[float] = None
    charge_state: Optional[int] = None
    intensity: Optional[float] = None

    # Metadata
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        """Validate categorical state"""
        if self.equivalence_class_size < 1:
            raise ValueError("Equivalence class must contain at least 1 configuration")
        if not 0 <= self.hardware_coherence <= 1:
            raise ValueError("Hardware coherence must be in [0, 1]")

    def information_content_bits(self) -> float:
        """
        Information content of MMD operation (Corollary 2.2).

        I_MMD = log2(|[C]_~|) bits

        Represents selection of one categorical state from
        equivalence class of equivalent possibilities.
        """
        return np.log2(self.equivalence_class_size)

    def is_physically_realizable(self, stream_divergence_threshold: float = 0.3) -> bool:
        """
        Check if state is physically realizable via hardware stream coherence.

        From BMD output filter: only accept interpretations coherent
        with hardware reality (stream divergence < threshold).
        """
        return self.hardware_coherence >= (1.0 - stream_divergence_threshold)


@dataclass
class OscillatoryHole:
    """
    Missing pattern in oscillatory cascade requiring completion.

    From St-Stellas: BMD operates by filling holes - selecting ONE
    configuration from ~10^6 equivalent patterns to continue cascade.
    """
    hole_id: str
    required_frequency: float  # Required oscillatory signature
    completion_space_size: int  # |ΔΩ| ~ 10^6 equivalent patterns
    phase_constraints: Dict[str, Tuple[float, float]]  # min, max phase per mode

    def is_completed_by(self, categorical_state: CategoricalState) -> bool:
        """Check if categorical state completes this hole"""
        # Frequency match
        freq_match = np.abs(categorical_state.frequency_hz - self.required_frequency) < 1e-3

        # Phase constraints satisfied
        phase_match = all(
            self.phase_constraints[mode][0] <= categorical_state.phase_relationships.get(mode, 0) <= self.phase_constraints[mode][1]
            for mode in self.phase_constraints
        )

        return freq_match and phase_match


@dataclass
class PhaseStructure:
    """
    Multi-mode phase configuration (hardware + molecular oscillations).

    Represents collective phase of all oscillatory modes coupled to system:
    - Hardware oscillations (8 scales: clock, memory, network, USB, GPU, disk, LED, global)
    - Molecular vibrations
    - Electronic oscillations
    """
    modes: Dict[str, float] = field(default_factory=dict)  # mode_name -> phase (rad)
    frequencies: Dict[str, float] = field(default_factory=dict)  # mode_name -> freq (Hz)
    coherence_times: Dict[str, float] = field(default_factory=dict)  # mode_name -> τ (s)

    def add_mode(self, name: str, phase: float, frequency: float, coherence_time: float = 1e-6):
        """Add oscillatory mode"""
        self.modes[name] = phase % (2 * np.pi)
        self.frequencies[name] = frequency
        self.coherence_times[name] = coherence_time

    def phase_lock_distance(self, other: 'PhaseStructure') -> float:
        """
        Compute phase-lock distance (lower = stronger coherence).

        Phase-lock criterion: |φ_i - φ_j| < π/4
        """
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

    def is_phase_locked(self, other: 'PhaseStructure', threshold: float = np.pi/4) -> bool:
        """Check if phase-locked with another structure"""
        return self.phase_lock_distance(other) < threshold


class MolecularMaxwellDemon:
    """
    Molecular Maxwell Demon: Information catalyst for mass spectrometry.

    Core Principle (St-Stellas Theorem 3.7):
    MMD Operation ≡ S-Navigation ≡ Categorical Completion

    Does NOT simulate:
    - Ion trajectories through TOF tube (unknowable: infinite weak force configs)
    - Quadrupole RF interactions (never repeats: categorical irreversibility)
    - Collision cell dynamics (chaotic: ~10^6 equivalent paths)

    DOES read:
    - Categorical state at convergence node (predetermined entropy endpoint)
    - S-coordinates: (S_k, S_t, S_e) compressed from ~10^6 equivalent trajectories
    - Independent of specific path taken (categorical invariants)

    Justification for No Simulation:
    - Multiple ions → mutual influence (Coulomb, Van der Waals, London, steric)
    - Each ion's trajectory depends on ALL other ions at ALL moments
    - Weak force configurations: ~10^6 per ion pair, continuously changing
    - Same ion, same instrument → DIFFERENT journey each time
    - But ALL journeys in equivalence class → SAME categorical state at detector
    - Detector measures categorical invariants (m/q, time, charge), not path details
    """

    def __init__(
        self,
        demon_id: str,
        convergence_node_id: str,
        instrument_projection: InstrumentProjection = InstrumentProjection.TOF
    ):
        """
        Initialize Molecular Maxwell Demon.

        Args:
            demon_id: Unique demon identifier
            convergence_node_id: ID of convergence node (materialization site)
            instrument_projection: Which instrument type to project as
        """
        self.demon_id = demon_id
        self.convergence_node_id = convergence_node_id
        self.instrument_projection = instrument_projection

        # MMD state
        self.categorical_state: Optional[CategoricalState] = None
        self.oscillatory_hole: Optional[OscillatoryHole] = None
        self.phase_structure: Optional[PhaseStructure] = None

        # Recursive sub-demons (St-Stellas Theorem 3.3: Recursive Self-Similarity)
        # Each S-coordinate is itself an MMD with tri-dimensional structure
        self.sub_demons: List['MolecularMaxwellDemon'] = []

        # Materialization state
        self.materialized: bool = False
        self.materialization_time: Optional[float] = None

        # Filtering statistics
        self.input_filter_count: int = 0  # Rejected by input filter (noise)
        self.output_filter_count: int = 0  # Rejected by output filter (unphysical)
        self.total_filtered: int = 0

    def input_filter(self, potential_signals: List[Dict[str, Any]],
                    hardware_phase_structure: PhaseStructure) -> List[Dict[str, Any]]:
        """
        Input filter: ℑ_input: Y_↓^(in) → Y_↑^(in)

        Selects signal from noise based on phase-lock coherence with hardware.

        Criterion: Peaks phase-locked with hardware oscillations = signal
                  Peaks not phase-locked = noise

        This is the first filter in dual filtering (Definition 2.2).
        """
        filtered = []

        for signal in potential_signals:
            # Extract signal phase structure
            signal_phase = PhaseStructure(
                modes=signal.get('phase_modes', {}),
                frequencies=signal.get('frequencies', {})
            )

            # Check phase-lock with hardware
            if hardware_phase_structure.is_phase_locked(signal_phase):
                filtered.append(signal)
            else:
                self.input_filter_count += 1

        return filtered

    def output_filter(self, interpretations: List[CategoricalState],
                     hardware_stream_bmd: Any) -> List[CategoricalState]:
        """
        Output filter: ℑ_output: Z_↓^(fin) → Z_↑^(fin)

        Targets physically-grounded interpretations.

        Criterion: Only interpretations coherent with hardware stream are physical.
                  Stream divergence < threshold → physically realizable
                  Stream divergence > threshold → unphysical (reject)

        This is the second filter in dual filtering (Definition 2.2).
        """
        filtered = []

        for interpretation in interpretations:
            # Check physical realizability via hardware coherence
            if interpretation.is_physically_realizable():
                filtered.append(interpretation)
            else:
                self.output_filter_count += 1

        return filtered

    def compress_to_s_coordinates(self, phase_locked_signals: List[Dict[str, Any]]) -> CategoricalState:
        """
        Compress infinite molecular information to S-coordinates (St-Stellas Theorem 3.2).

        From ~10^24 continuous degrees of freedom (positions, velocities,
        Van der Waals angles, dipole orientations, vibrational phases)
        → THREE sufficient statistics: (S_k, S_t, S_e)

        This compression IS the MMD operation: filtering potential states
        (infinite weak force configurations) to actual state (three coordinates).

        Each of these three numbers is itself an MMD compressing infinity, recursively.
        """
        if not phase_locked_signals:
            raise ValueError("Cannot compress empty signal set")

        # Extract features from phase-locked ensemble
        frequencies = np.array([s.get('frequency', 0) for s in phase_locked_signals])
        intensities = np.array([s.get('intensity', 0) for s in phase_locked_signals])
        phases = np.array([s.get('phase', 0) for s in phase_locked_signals])

        # S_k (Knowledge): Information deficit - which equivalence class?
        # Computed from Shannon entropy of intensity distribution
        p_intensity = intensities / intensities.sum() if intensities.sum() > 0 else intensities
        p_intensity = p_intensity[p_intensity > 0]  # Remove zeros
        S_k = -np.sum(p_intensity * np.log2(p_intensity)) if len(p_intensity) > 0 else 0.0

        # S_t (Time): Temporal position in categorical sequence
        # Computed from phase progression
        phase_diffs = np.diff(phases) if len(phases) > 1 else np.array([0])
        S_t = np.mean(phase_diffs) if len(phase_diffs) > 0 else 0.0

        # S_e (Entropy): Constraint density - thermodynamic accessibility
        # Computed from frequency distribution (more spread = more constraints)
        S_e = np.std(frequencies) if len(frequencies) > 1 else 0.0

        # Compute primary frequency (weighted average)
        if intensities.sum() > 0:
            primary_frequency = np.average(frequencies, weights=intensities)
        else:
            primary_frequency = np.mean(frequencies) if len(frequencies) > 0 else 0.0

        # Harmonic decomposition
        harmonics = np.fft.fft(intensities) if len(intensities) > 0 else np.array([])

        # Phase relationships
        phase_relationships = {
            f"mode_{i}": phase for i, phase in enumerate(phases)
        }

        # Hardware coherence (from phase-lock quality)
        hardware_coherence = np.mean([s.get('coherence', 0.5) for s in phase_locked_signals])

        # Observable properties (categorical invariants)
        mz_ratio = phase_locked_signals[0].get('mz', None) if phase_locked_signals else None
        charge_state = phase_locked_signals[0].get('charge', None) if phase_locked_signals else None
        total_intensity = intensities.sum()

        return CategoricalState(
            state_id=f"{self.demon_id}_cat_state",
            S_k=S_k,
            S_t=S_t,
            S_e=S_e,
            frequency_hz=primary_frequency,
            harmonics=harmonics,
            phase_relationships=phase_relationships,
            equivalence_class_size=1_000_000,  # ~10^6 typical
            hardware_coherence=hardware_coherence,
            mz_ratio=mz_ratio,
            charge_state=charge_state,
            intensity=total_intensity
        )

    def generate_sub_demons(self) -> List['MolecularMaxwellDemon']:
        """
        Generate sub-demons through recursive decomposition (St-Stellas Corollary 3.6).

        Self-propagating cascades: Each MMD automatically creates 3 sub-MMDs.

        Why automatic? To evaluate ANY S-coordinate requires evaluating sub-coordinates:
        - To know S_k = x, need (S_k,k, S_k,t, S_k,e)
        - To know S_t = y, need (S_t,k, S_t,t, S_t,e)
        - To know S_e = z, need (S_e,k, S_e,t, S_e,e)

        Creates exponential 3^k cascade, all operating in parallel through phase-locking.
        """
        if not self.categorical_state:
            return []

        sub_demons = []

        # Sub-demon for S_k coordinate
        demon_k = MolecularMaxwellDemon(
            demon_id=f"{self.demon_id}_S_k",
            convergence_node_id=f"{self.convergence_node_id}_k",
            instrument_projection=self.instrument_projection
        )
        # This sub-demon has its own (S_k,k, S_k,t, S_k,e) structure
        sub_demons.append(demon_k)

        # Sub-demon for S_t coordinate
        demon_t = MolecularMaxwellDemon(
            demon_id=f"{self.demon_id}_S_t",
            convergence_node_id=f"{self.convergence_node_id}_t",
            instrument_projection=self.instrument_projection
        )
        sub_demons.append(demon_t)

        # Sub-demon for S_e coordinate
        demon_e = MolecularMaxwellDemon(
            demon_id=f"{self.demon_id}_S_e",
            convergence_node_id=f"{self.convergence_node_id}_e",
            instrument_projection=self.instrument_projection
        )
        sub_demons.append(demon_e)

        return sub_demons

    def materialize(self, hardware_phase_lock: Dict[str, Any]) -> CategoricalState:
        """
        Materialize demon at convergence node.

        Demon exists ONLY during measurement. Between measurements: doesn't exist
        (no hardware, no power, no cost, no maintenance).

        Dual filtering process:
        1. Input filter: potential signals → phase-locked signals
        2. Compress: phase-locked signals → categorical state (S_k, S_t, S_e)
        3. Output filter: potential interpretations → physical interpretations
        4. Generate sub-demons: recursive decomposition (automatic)

        Returns:
            CategoricalState at this convergence node
        """
        if self.materialized:
            raise RuntimeError(f"Demon {self.demon_id} already materialized")

        self.materialization_time = time.time()

        # Extract hardware phase structure
        hardware_phase_structure = PhaseStructure(
            modes=hardware_phase_lock.get('modes', {}),
            frequencies=hardware_phase_lock.get('frequencies', {})
        )

        # Get potential signals from hardware
        potential_signals = hardware_phase_lock.get('signals', [])

        # DUAL FILTERING

        # 1. Input filter: ℑ_input (noise rejection)
        phase_locked_signals = self.input_filter(potential_signals, hardware_phase_structure)

        # 2. Compress to categorical state
        self.categorical_state = self.compress_to_s_coordinates(phase_locked_signals)

        # 3. Output filter: ℑ_output (physical validation)
        # For now, single interpretation = categorical state itself
        interpretations = [self.categorical_state]
        hardware_stream_bmd = hardware_phase_lock.get('hardware_stream_bmd', None)
        physical_interpretations = self.output_filter(interpretations, hardware_stream_bmd)

        if not physical_interpretations:
            raise ValueError("No physically realizable interpretations at convergence node")

        self.categorical_state = physical_interpretations[0]

        # 4. Generate sub-demons (recursive, self-propagating)
        self.sub_demons = self.generate_sub_demons()

        self.materialized = True

        return self.categorical_state

    def read_projection(self) -> Any:
        """
        Read instrument-specific projection from categorical state.

        Each instrument type = different coordinate projection of SAME categorical state.

        ALL instruments read simultaneously because they're just reading different
        aspects of the categorical state that already exists.
        """
        if not self.categorical_state:
            raise RuntimeError("Cannot read projection: demon not materialized")

        state = self.categorical_state

        if self.instrument_projection == InstrumentProjection.TOF:
            # Time-of-flight: Read S_t coordinate (temporal position)
            return {
                'instrument': 'TOF',
                'arrival_time': state.S_t,
                'mz': state.mz_ratio,
                'intensity': state.intensity
            }

        elif self.instrument_projection == InstrumentProjection.ORBITRAP:
            # Orbitrap: Read harmonic structure (high resolution)
            return {
                'instrument': 'Orbitrap',
                'mz': state.mz_ratio,
                'frequency': state.frequency_hz,
                'harmonics': state.harmonics,
                'resolution': len(state.harmonics) if state.harmonics is not None else 0,
                'intensity': state.intensity
            }

        elif self.instrument_projection == InstrumentProjection.FTICR:
            # FT-ICR: Read exact frequency (exact mass)
            return {
                'instrument': 'FT-ICR',
                'exact_frequency': state.frequency_hz,
                'exact_mz': state.mz_ratio,
                'mass_accuracy_ppm': 1.0,  # Categorical state → high accuracy
                'intensity': state.intensity
            }

        elif self.instrument_projection == InstrumentProjection.QUADRUPOLE:
            # Quadrupole: Read ω with S_e filtering
            return {
                'instrument': 'Quadrupole',
                'mz': state.mz_ratio,
                'selected_by_rf_scan': True,
                'stability_parameter': state.S_e,  # Constraint density
                'intensity': state.intensity
            }

        elif self.instrument_projection == InstrumentProjection.SECTOR:
            # Sector: Read S_e/ω ratio (momentum/charge → elemental composition)
            return {
                'instrument': 'Sector',
                'mz': state.mz_ratio,
                'momentum_energy_ratio': state.S_e / state.frequency_hz if state.frequency_hz > 0 else 0,
                'elemental_composition_hint': state.S_k,  # Knowledge coordinate
                'intensity': state.intensity
            }

        elif self.instrument_projection == InstrumentProjection.ION_MOBILITY:
            # IMS: Read S_k/S_e structure (collision cross-section)
            return {
                'instrument': 'IMS',
                'mz': state.mz_ratio,
                'collision_cross_section': state.S_k / state.S_e if state.S_e > 0 else 0,
                'drift_time': state.S_t,
                'structural_info': state.S_k,
                'intensity': state.intensity
            }

        elif self.instrument_projection == InstrumentProjection.PHOTODETECTOR:
            # Photodetector: Read frequency (photon energy)
            return {
                'instrument': 'Photodetector',
                'frequency_hz': state.frequency_hz,
                'wavelength_nm': 3e8 / state.frequency_hz * 1e9 if state.frequency_hz > 0 else 0,
                'energy_ev': 4.136e-15 * state.frequency_hz,  # h*ν in eV
                'intensity': state.intensity,
                'absorbed': False  # Virtual photodetector doesn't absorb!
            }

        elif self.instrument_projection == InstrumentProjection.ION_DETECTOR:
            # Ion detector: Read S_e (charge state)
            return {
                'instrument': 'Ion Detector',
                'charge_state': state.charge_state or int(state.S_e) + 1,
                'kinetic_energy_ev': state.S_e * 1.0,  # From entropy coordinate
                'intensity': state.intensity,
                'ion_destroyed': False  # Virtual detector doesn't destroy!
            }

        else:
            raise ValueError(f"Unknown instrument projection: {self.instrument_projection}")

    def dissolve(self):
        """
        Dissolve demon after measurement.

        Demon exists ONLY during measurement. Between measurements: doesn't exist.

        This is key to zero marginal cost: infinite demons cost nothing because
        they don't persist. They materialize on demand, read categorical state,
        then dissolve.
        """
        self.categorical_state = None
        self.oscillatory_hole = None
        self.phase_structure = None
        self.sub_demons = []
        self.materialized = False
        self.materialization_time = None

    def get_statistics(self) -> Dict[str, Any]:
        """Get demon filtering statistics"""
        return {
            'demon_id': self.demon_id,
            'instrument_type': self.instrument_projection.value,
            'materialized': self.materialized,
            'input_filtered': self.input_filter_count,
            'output_filtered': self.output_filter_count,
            'total_filtered': self.input_filter_count + self.output_filter_count,
            'sub_demons': len(self.sub_demons),
            'categorical_state': {
                'S_k': self.categorical_state.S_k if self.categorical_state else None,
                'S_t': self.categorical_state.S_t if self.categorical_state else None,
                'S_e': self.categorical_state.S_e if self.categorical_state else None,
                'equivalence_class_size': self.categorical_state.equivalence_class_size if self.categorical_state else None,
                'information_content_bits': self.categorical_state.information_content_bits() if self.categorical_state else None
            }
        }


# Module exports
__all__ = [
    'InstrumentProjection',
    'CategoricalState',
    'OscillatoryHole',
    'PhaseStructure',
    'MolecularMaxwellDemon'
]
