"""
Transport Phenomena in Chromatography as Computation

This module implements chromatography as an electric trap array where:
- Column = Array of electric traps
- Retention time = Partition lag τ_p(S_k, S_t, S_e)
- Volume reduction = 10²¹× (from mL to nm³)
- Separation = Categorical state assignment

Key insight: Chromatographic retention IS partition lag!
    t_R = τ_p(S_k, S_t, S_e)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)

# Physical constants
K_B = 1.380649e-23  # Boltzmann constant (J/K)
E_CHARGE = 1.602176634e-19  # Elementary charge (C)
AMU = 1.66053906660e-27  # Atomic mass unit (kg)


class TrapState(Enum):
    """State of an electric trap in the chromatographic array."""
    EMPTY = "empty"
    CAPTURING = "capturing"
    TRAPPED = "trapped"
    PARTITIONING = "partitioning"
    RELEASING = "releasing"


@dataclass
class SEntropyCoordinate:
    """
    S-Entropy coordinates - platform-independent representation.

    From the categorical memory framework:
    - S_k: Knowledge entropy (charge configuration)
    - S_t: Temporal entropy (timing uncertainty)
    - S_e: Evolution entropy (trajectory uncertainty)

    These coordinates form the "memory address" in categorical space.
    """
    S_k: float  # Knowledge entropy [0, 1]
    S_t: float  # Temporal entropy [0, 1]
    S_e: float  # Evolution entropy [0, 1]

    def __post_init__(self):
        """Validate entropy coordinates are in valid range."""
        for name, val in [('S_k', self.S_k), ('S_t', self.S_t), ('S_e', self.S_e)]:
            if not 0 <= val <= 1:
                logger.warning(f"{name} = {val} outside [0,1], clipping")
        self.S_k = np.clip(self.S_k, 0, 1)
        self.S_t = np.clip(self.S_t, 0, 1)
        self.S_e = np.clip(self.S_e, 0, 1)

    @property
    def total_entropy(self) -> float:
        """Total entropy magnitude."""
        return np.sqrt(self.S_k**2 + self.S_t**2 + self.S_e**2)

    @property
    def address(self) -> Tuple[float, float, float]:
        """Memory address in categorical space."""
        return (self.S_k, self.S_t, self.S_e)

    def distance_to(self, other: 'SEntropyCoordinate') -> float:
        """Euclidean distance in S-entropy space."""
        return np.sqrt(
            (self.S_k - other.S_k)**2 +
            (self.S_t - other.S_t)**2 +
            (self.S_e - other.S_e)**2
        )

    def to_partition_depth(self) -> int:
        """
        Convert S-entropy to partition depth n.
        Higher entropy → higher partition depth needed.
        """
        # n ≥ 1, scales with total entropy
        return max(1, int(np.ceil(self.total_entropy * 10)))


@dataclass
class PartitionCoordinates:
    """
    Partition coordinates (n, ℓ, m, s) - quantum-like state description.

    - n: Principal depth (n ≥ 1) - which trap in array
    - ℓ: Angular complexity (0 ≤ ℓ < n) - cyclotron orbit
    - m: Orientation (-ℓ ≤ m ≤ ℓ) - orbit phase
    - s: Chirality (±1/2) - internal state/spin

    Capacity at depth n: C(n) = 2n² states
    """
    n: int  # Principal depth
    l: int  # Angular complexity (using 'l' for code compatibility)
    m: int  # Orientation
    s: float  # Chirality (+0.5 or -0.5)

    def __post_init__(self):
        """Validate partition coordinate constraints."""
        if self.n < 1:
            raise ValueError(f"n must be ≥ 1, got {self.n}")
        if not 0 <= self.l < self.n:
            raise ValueError(f"ℓ must be 0 ≤ ℓ < n={self.n}, got {self.l}")
        if not -self.l <= self.m <= self.l:
            raise ValueError(f"m must be -{self.l} ≤ m ≤ {self.l}, got {self.m}")
        if self.s not in [-0.5, 0.5]:
            raise ValueError(f"s must be ±1/2, got {self.s}")

    @property
    def capacity(self) -> int:
        """Capacity at this partition depth: C(n) = 2n²."""
        return 2 * self.n * self.n

    @classmethod
    def from_s_entropy(cls, s_entropy: SEntropyCoordinate, mz: float = None) -> 'PartitionCoordinates':
        """
        Convert S-entropy coordinates to partition coordinates.

        Mapping:
        - n: From total entropy magnitude
        - ℓ: From S_k (knowledge determines angular complexity)
        - m: From S_e (evolution determines orientation)
        - s: From S_t sign relative to threshold
        """
        n = max(1, int(np.ceil(s_entropy.total_entropy * 10)))
        l = min(int(s_entropy.S_k * n), n - 1)
        # Calculate m and clamp to valid range [-l, l]
        m_raw = int(s_entropy.S_e * (2 * l + 1)) - l if l > 0 else 0
        m = max(-l, min(l, m_raw))  # Ensure -l ≤ m ≤ l
        s = 0.5 if s_entropy.S_t > 0.5 else -0.5

        return cls(n=n, l=l, m=m, s=s)


@dataclass
class ElectricTrap:
    """
    Single electric trap in the chromatographic array.

    Models the trap as a Penning trap potential:
        Φ(r, z) = (V₀/2d²)(z² - r²/2)

    Volume for single ion confinement: ~3 nm³
    """
    trap_id: int
    position: float  # Position along column (mm)
    state: TrapState = TrapState.EMPTY

    # Trap parameters
    voltage: float = 100.0  # Trapping voltage V₀ (V)
    magnetic_field: float = 10.0  # B field (T)
    trap_depth: float = 1.0  # d parameter (mm)

    # Trapped ion properties (if any)
    ion_mz: Optional[float] = None
    s_entropy: Optional[SEntropyCoordinate] = None
    partition_coords: Optional[PartitionCoordinates] = None

    # Timing
    capture_time: Optional[float] = None
    partition_lag: Optional[float] = None

    @property
    def trap_volume_nm3(self) -> float:
        """
        Trap volume in nm³.
        For single ion confinement: V ~ πr²z where r,z ~ 1nm
        """
        # Cyclotron radius in nm (approximate)
        r_nm = 1.0  # nm
        z_nm = 1.0  # nm
        return np.pi * r_nm**2 * z_nm

    def cyclotron_frequency(self, mz: float) -> float:
        """
        Calculate cyclotron frequency for given m/z.

        ω_c = qB/m

        Returns frequency in MHz.
        """
        mass_kg = mz * AMU
        omega = E_CHARGE * self.magnetic_field / mass_kg
        freq_mhz = omega / (2 * np.pi * 1e6)
        return freq_mhz

    def capture_ion(self, mz: float, s_entropy: SEntropyCoordinate, time: float):
        """Capture an ion in this trap."""
        self.state = TrapState.CAPTURING
        self.ion_mz = mz
        self.s_entropy = s_entropy
        self.capture_time = time
        self.state = TrapState.TRAPPED

    def perform_partition(self) -> PartitionCoordinates:
        """
        Perform partition operation on trapped ion.

        The trap IS the partition operator!

        Returns the determined partition coordinates.
        """
        if self.state != TrapState.TRAPPED or self.s_entropy is None:
            raise ValueError("No ion trapped to partition")

        self.state = TrapState.PARTITIONING

        # Convert S-entropy to partition coordinates
        self.partition_coords = PartitionCoordinates.from_s_entropy(
            self.s_entropy, self.ion_mz
        )

        # Calculate partition lag
        self.partition_lag = self._calculate_partition_lag()

        return self.partition_coords

    def _calculate_partition_lag(self) -> float:
        """
        Calculate partition lag τ_p.

        τ_p is the time to complete categorical assignment.
        Higher entropy → longer partition lag.
        """
        if self.s_entropy is None:
            return 0.0

        # Base lag from cyclotron period
        base_lag = 1.0 / self.cyclotron_frequency(self.ion_mz or 100.0)  # ms

        # Scale by total entropy
        entropy_factor = 1.0 + self.s_entropy.total_entropy * 10

        return base_lag * entropy_factor

    def release_ion(self) -> Dict[str, Any]:
        """Release trapped ion and return its properties."""
        if self.state not in [TrapState.TRAPPED, TrapState.PARTITIONING]:
            raise ValueError("No ion to release")

        self.state = TrapState.RELEASING

        result = {
            'trap_id': self.trap_id,
            'mz': self.ion_mz,
            's_entropy': self.s_entropy,
            'partition_coords': self.partition_coords,
            'capture_time': self.capture_time,
            'partition_lag': self.partition_lag,
            'cyclotron_freq_mhz': self.cyclotron_frequency(self.ion_mz) if self.ion_mz else None
        }

        # Clear trap
        self.ion_mz = None
        self.s_entropy = None
        self.partition_coords = None
        self.capture_time = None
        self.partition_lag = None
        self.state = TrapState.EMPTY

        return result


@dataclass
class ChromatographicTrapArray:
    """
    Chromatographic column as an array of electric traps.

    Chromatographic Column = Array of Electric Traps

    Volume reduction: V_initial → V_single ~ 10²¹× (from mL to nm³)
    """
    column_length_mm: float = 150.0  # Typical HPLC column
    trap_spacing_mm: float = 0.1  # One trap every 0.1 mm
    magnetic_field: float = 10.0  # Tesla

    # Computed properties
    traps: List[ElectricTrap] = field(default_factory=list)

    def __post_init__(self):
        """Initialize trap array."""
        n_traps = int(self.column_length_mm / self.trap_spacing_mm)
        self.traps = [
            ElectricTrap(
                trap_id=i,
                position=i * self.trap_spacing_mm,
                magnetic_field=self.magnetic_field
            )
            for i in range(n_traps)
        ]
        logger.info(f"Initialized chromatographic trap array with {n_traps} traps")

    @property
    def n_traps(self) -> int:
        """Number of traps in array."""
        return len(self.traps)

    @property
    def volume_reduction_factor(self) -> float:
        """
        Volume reduction factor from initial to single ion.

        V_initial ~ 1 mL = 10⁹ nm³
        V_single ~ 3 nm³
        Factor ~ 3 × 10⁸ per trap
        With many traps: Total ~ 10²¹
        """
        initial_volume_nm3 = 1e9  # 1 mL
        single_ion_volume_nm3 = 3.0  # nm³
        return initial_volume_nm3 / single_ion_volume_nm3

    def find_trap_for_retention_time(self, retention_time: float, flow_rate: float = 0.3) -> int:
        """
        Find appropriate trap for given retention time.

        Args:
            retention_time: Retention time in minutes
            flow_rate: Flow rate in mL/min

        Returns:
            Trap index
        """
        # Linear velocity (mm/min)
        velocity = (flow_rate * 1000) / (np.pi * 2.1**2 / 4)  # 2.1mm ID column

        # Position along column
        position_mm = velocity * retention_time

        # Find nearest trap
        trap_idx = min(int(position_mm / self.trap_spacing_mm), self.n_traps - 1)
        return max(0, trap_idx)

    def capture_at_retention_time(
        self,
        mz: float,
        s_entropy: SEntropyCoordinate,
        retention_time: float,
        flow_rate: float = 0.3
    ) -> ElectricTrap:
        """
        Capture ion at specified retention time.

        Retention time = Partition lag for categorical assignment
        t_R = τ_p(S_k, S_t, S_e)
        """
        trap_idx = self.find_trap_for_retention_time(retention_time, flow_rate)
        trap = self.traps[trap_idx]

        if trap.state != TrapState.EMPTY:
            # Find next available trap
            for i in range(trap_idx, min(trap_idx + 10, self.n_traps)):
                if self.traps[i].state == TrapState.EMPTY:
                    trap = self.traps[i]
                    break

        trap.capture_ion(mz, s_entropy, retention_time)
        return trap

    def process_all_trapped_ions(self) -> List[Dict[str, Any]]:
        """Perform partition operation on all trapped ions."""
        results = []

        for trap in self.traps:
            if trap.state == TrapState.TRAPPED:
                trap.perform_partition()
                results.append(trap.release_ion())

        return results


class PartitionLagCalculator:
    """
    Calculate partition lag from chromatographic parameters.

    Key equation:
        t_R = τ_p(S_k, S_t, S_e)

    Where:
        t_R = retention time
        τ_p = partition lag
        S_k, S_t, S_e = S-entropy coordinates
    """

    def __init__(
        self,
        dead_time: float = 1.0,  # Column dead time (min)
        temperature: float = 298.15,  # Temperature (K)
        column_length: float = 150.0,  # mm
        particle_size: float = 1.7  # μm
    ):
        self.dead_time = dead_time
        self.temperature = temperature
        self.column_length = column_length
        self.particle_size = particle_size

        # Theoretical plate height (van Deemter minimum)
        self.plate_height = 2 * particle_size / 1000  # mm
        self.n_plates = column_length / self.plate_height

    def retention_time_from_s_entropy(
        self,
        s_entropy: SEntropyCoordinate,
        k_factor: float = 1.0  # Retention factor scaling
    ) -> float:
        """
        Calculate retention time from S-entropy coordinates.

        Retention time = Dead time × (1 + k')
        Where k' depends on S-entropy coordinates.
        """
        # Retention factor from entropy
        # Higher entropy → longer retention
        k_prime = k_factor * (
            s_entropy.S_k * 2.0 +  # Charge configuration
            s_entropy.S_t * 1.0 +  # Temporal uncertainty
            s_entropy.S_e * 0.5    # Evolution uncertainty
        )

        return self.dead_time * (1 + k_prime)

    def partition_lag_from_retention(
        self,
        retention_time: float,
        s_entropy: SEntropyCoordinate
    ) -> float:
        """
        Extract partition lag from retention time.

        τ_p = t_R - t_0 - τ_transport

        Where τ_transport is non-partition transport time.
        """
        # Transport time (purely physical, no partition)
        transport_time = self.dead_time * (1 - s_entropy.S_t)

        # Partition lag
        partition_lag = max(0, retention_time - transport_time)

        return partition_lag

    def s_entropy_from_retention(
        self,
        retention_time: float,
        mz: float,
        logp: float = 0.0  # Partition coefficient
    ) -> SEntropyCoordinate:
        """
        Infer S-entropy coordinates from retention time.

        Inverse of retention_time_from_s_entropy.
        """
        # Retention factor
        k_prime = (retention_time / self.dead_time) - 1

        # Estimate S-entropy components
        # Use heuristics based on typical chromatographic behavior

        # S_k from logP (hydrophobicity correlates with charge distribution)
        S_k = 1.0 / (1.0 + np.exp(-logp))  # Sigmoid transform

        # S_t from peak width / retention time ratio
        expected_width = retention_time / np.sqrt(self.n_plates)
        S_t = min(1.0, expected_width / retention_time)

        # S_e from residual after accounting for S_k and S_t
        estimated_retention = self.dead_time * (1 + S_k * 2.0 + S_t * 1.0)
        residual = retention_time - estimated_retention
        S_e = np.clip(residual / (self.dead_time * 0.5), 0, 1)

        return SEntropyCoordinate(S_k=S_k, S_t=S_t, S_e=S_e)

    def undetermined_residue(
        self,
        retention_time: float,
        peak_width: float,
        s_entropy: SEntropyCoordinate
    ) -> float:
        """
        Calculate undetermined residue from chromatographic peak.

        Residue = states that cannot be assigned during partition lag.
        Higher peak width → more residue → more entropy generated.
        """
        # Ideal peak width (plate theory)
        ideal_width = retention_time / np.sqrt(self.n_plates)

        # Excess width = undetermined residue
        excess_width = max(0, peak_width - ideal_width)

        # Number of unresolved states
        n_residue = 1 + int(excess_width * s_entropy.total_entropy * 100)

        return n_residue

    def entropy_generated(self, n_residue: int) -> float:
        """
        Entropy generated by partition operation.

        ΔS = k_B × ln(n_residue)
        """
        if n_residue <= 1:
            return 0.0
        return K_B * np.log(n_residue)


class ChromatographicQuantumComputer:
    """
    The complete chromatographic quantum computer.

    Stages:
    1. Chromatographic separation → Memory addressing
    2. Electric trap array → Volume reduction
    3. Partition operation → Categorical state calculation
    4. Categorical memory → Information storage
    5. SQUID readout → State reading
    """

    def __init__(
        self,
        column_length: float = 150.0,
        trap_spacing: float = 0.1,
        magnetic_field: float = 10.0,
        dead_time: float = 1.0,
        temperature: float = 298.15
    ):
        self.trap_array = ChromatographicTrapArray(
            column_length_mm=column_length,
            trap_spacing_mm=trap_spacing,
            magnetic_field=magnetic_field
        )

        self.partition_calculator = PartitionLagCalculator(
            dead_time=dead_time,
            temperature=temperature,
            column_length=column_length
        )

        self.categorical_memory: Dict[Tuple[float, float, float], Dict] = {}
        self.computation_log: List[Dict] = []

    def process_chromatographic_peak(
        self,
        mz: float,
        retention_time: float,
        intensity: float,
        peak_width: float = 0.1,
        logp: float = 0.0
    ) -> Dict[str, Any]:
        """
        Process a single chromatographic peak through the full pipeline.

        Returns complete computational result.
        """
        # Stage 1: Calculate S-entropy from retention
        s_entropy = self.partition_calculator.s_entropy_from_retention(
            retention_time, mz, logp
        )

        # Stage 2: Capture in trap array
        trap = self.trap_array.capture_at_retention_time(
            mz, s_entropy, retention_time
        )

        # Stage 3: Perform partition operation
        partition_coords = trap.perform_partition()

        # Stage 4: Calculate undetermined residue
        n_residue = self.partition_calculator.undetermined_residue(
            retention_time, peak_width, s_entropy
        )

        # Stage 5: Calculate entropy generated
        entropy_generated = self.partition_calculator.entropy_generated(n_residue)

        # Stage 6: Store in categorical memory
        memory_address = s_entropy.address
        memory_entry = {
            'mz': mz,
            'retention_time': retention_time,
            'intensity': intensity,
            's_entropy': s_entropy,
            'partition_coords': partition_coords,
            'cyclotron_freq_mhz': trap.cyclotron_frequency(mz),
            'undetermined_residue': n_residue,
            'entropy_generated_J_K': entropy_generated,
            'trap_id': trap.trap_id,
            'partition_lag': trap.partition_lag
        }
        self.categorical_memory[memory_address] = memory_entry

        # Stage 7: Release ion and log
        release_data = trap.release_ion()

        result = {
            'input': {
                'mz': mz,
                'retention_time': retention_time,
                'intensity': intensity,
                'peak_width': peak_width
            },
            's_entropy': {
                'S_k': s_entropy.S_k,
                'S_t': s_entropy.S_t,
                'S_e': s_entropy.S_e,
                'total': s_entropy.total_entropy
            },
            'partition': {
                'n': partition_coords.n,
                'l': partition_coords.l,
                'm': partition_coords.m,
                's': partition_coords.s,
                'capacity': partition_coords.capacity
            },
            'physics': {
                'cyclotron_freq_mhz': memory_entry['cyclotron_freq_mhz'],
                'partition_lag_ms': trap.partition_lag,
                'trap_volume_nm3': self.trap_array.traps[0].trap_volume_nm3,
                'volume_reduction': self.trap_array.volume_reduction_factor
            },
            'thermodynamics': {
                'undetermined_residue': n_residue,
                'entropy_generated_J_K': entropy_generated,
                'entropy_generated_bits': entropy_generated / (K_B * np.log(2)) if entropy_generated > 0 else 0
            },
            'memory_address': memory_address
        }

        self.computation_log.append(result)
        return result

    def navigate_memory(
        self,
        current: SEntropyCoordinate,
        target: SEntropyCoordinate
    ) -> List[int]:
        """
        Navigate from current to target address in categorical memory.

        Uses precision-by-difference in 3^k hierarchy.
        """
        path = []
        distance = current.distance_to(target)

        # Navigate through hierarchy levels
        k = 0
        while distance > 0.01 and k < 10:  # Max 10 levels
            # Calculate precision-by-difference
            delta_S_k = target.S_k - current.S_k
            delta_S_t = target.S_t - current.S_t
            delta_S_e = target.S_e - current.S_e

            # Determine branch (0, 1, or 2) based on largest delta
            deltas = [abs(delta_S_k), abs(delta_S_t), abs(delta_S_e)]
            branch = np.argmax(deltas)
            path.append(branch)

            # Move towards target
            step_size = 0.5 ** (k + 1)
            new_S_k = current.S_k + np.sign(delta_S_k) * step_size * (branch == 0)
            new_S_t = current.S_t + np.sign(delta_S_t) * step_size * (branch == 1)
            new_S_e = current.S_e + np.sign(delta_S_e) * step_size * (branch == 2)

            current = SEntropyCoordinate(
                S_k=np.clip(new_S_k, 0, 1),
                S_t=np.clip(new_S_t, 0, 1),
                S_e=np.clip(new_S_e, 0, 1)
            )

            distance = current.distance_to(target)
            k += 1

        return path

    def query_memory(self, s_entropy: SEntropyCoordinate, radius: float = 0.1) -> List[Dict]:
        """
        Query categorical memory for entries near given S-entropy coordinates.
        """
        results = []

        for address, entry in self.categorical_memory.items():
            stored_entropy = entry['s_entropy']
            if s_entropy.distance_to(stored_entropy) <= radius:
                results.append(entry)

        return results


class PlateTheoryValidator:
    """
    Validate chromatographic results against traditional plate theory.

    This provides a bridge between the categorical framework
    and classical chromatography theory.
    """

    def __init__(self, n_plates: float = 10000):
        self.n_plates = n_plates

    def expected_peak_width(self, retention_time: float) -> float:
        """
        Expected peak width from plate theory.

        σ = t_R / √N
        """
        return retention_time / np.sqrt(self.n_plates)

    def resolution(
        self,
        rt1: float,
        rt2: float,
        width1: float,
        width2: float
    ) -> float:
        """
        Chromatographic resolution between two peaks.

        R_s = 2(t_R2 - t_R1) / (w1 + w2)
        """
        if width1 + width2 == 0:
            return float('inf')
        return 2 * abs(rt2 - rt1) / (width1 + width2)

    def validate_partition_assignment(
        self,
        s_entropy1: SEntropyCoordinate,
        s_entropy2: SEntropyCoordinate,
        resolution: float
    ) -> bool:
        """
        Validate that partition coordinates are distinguishable.

        If resolution < 1.5, partition coordinates may not be unique.
        """
        entropy_distance = s_entropy1.distance_to(s_entropy2)

        # Resolution threshold for partition uniqueness
        # Higher entropy distance requires less chromatographic resolution
        required_resolution = 1.5 / (1 + entropy_distance)

        return resolution >= required_resolution

    def efficiency_from_entropy(self, s_entropy: SEntropyCoordinate) -> float:
        """
        Calculate effective plate count from S-entropy.

        Lower entropy → higher efficiency (more information preserved).
        """
        efficiency_factor = 1 - s_entropy.total_entropy / np.sqrt(3)
        return self.n_plates * max(0.1, efficiency_factor)


def compute_chromatographic_trajectory(
    peaks: List[Dict[str, float]],
    column_params: Dict[str, float] = None
) -> Dict[str, Any]:
    """
    Compute full chromatographic trajectory for a set of peaks.

    This is the main entry point for chromatography-as-computation.

    Args:
        peaks: List of dicts with 'mz', 'retention_time', 'intensity', 'peak_width'
        column_params: Optional column parameters

    Returns:
        Complete computational trajectory
    """
    if column_params is None:
        column_params = {
            'column_length': 150.0,
            'trap_spacing': 0.1,
            'magnetic_field': 10.0,
            'dead_time': 1.0,
            'temperature': 298.15
        }

    computer = ChromatographicQuantumComputer(**column_params)

    results = []
    for peak in peaks:
        result = computer.process_chromatographic_peak(
            mz=peak.get('mz', 100.0),
            retention_time=peak.get('retention_time', 1.0),
            intensity=peak.get('intensity', 1000.0),
            peak_width=peak.get('peak_width', 0.1),
            logp=peak.get('logp', 0.0)
        )
        results.append(result)

    # Summary statistics
    summary = {
        'n_peaks_processed': len(results),
        'total_entropy_generated': sum(
            r['thermodynamics']['entropy_generated_J_K'] for r in results
        ),
        'total_entropy_bits': sum(
            r['thermodynamics']['entropy_generated_bits'] for r in results
        ),
        'memory_utilization': len(computer.categorical_memory),
        'volume_reduction': computer.trap_array.volume_reduction_factor
    }

    return {
        'peaks': results,
        'summary': summary,
        'memory': {
            str(k): {
                'mz': v['mz'],
                's_entropy': (v['s_entropy'].S_k, v['s_entropy'].S_t, v['s_entropy'].S_e),
                'partition': (v['partition_coords'].n, v['partition_coords'].l,
                             v['partition_coords'].m, v['partition_coords'].s)
            }
            for k, v in computer.categorical_memory.items()
        }
    }
