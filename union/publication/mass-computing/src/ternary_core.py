#!/usr/bin/env python3
"""
Mass Computing Core: Ternary Partition Synthesis Framework

This module implements the core data structures and algorithms for
partition synthesis in mass spectrometry using ternary representation.
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict, Any
import numpy as np
from enum import IntEnum


class TritAxis(IntEnum):
    """Mapping from trit values to S-entropy axes."""
    S_K = 0  # Knowledge entropy
    S_T = 1  # Temporal entropy
    S_E = 2  # Evolution entropy


@dataclass(frozen=True)
class TernaryAddress:
    """
    Ternary address in S-entropy space.

    A k-trit address specifies one of 3^k cells in S-space.
    The address encodes both position AND trajectory.
    """
    trits: Tuple[int, ...]

    def __post_init__(self):
        # Validate all trits are in {0, 1, 2}
        for t in self.trits:
            if t not in (0, 1, 2):
                raise ValueError(f"Invalid trit value: {t}. Must be 0, 1, or 2.")

    @classmethod
    def from_string(cls, s: str) -> 'TernaryAddress':
        """Create address from string of digits."""
        trits = tuple(int(c) for c in s if c in '012')
        return cls(trits)

    @classmethod
    def from_scoord(cls, s_k: float, s_t: float, s_e: float,
                    depth: int = 18) -> 'TernaryAddress':
        """
        Create ternary address from S-entropy coordinates.

        Uses interleaved encoding: each trit refines one axis,
        cycling through S_k, S_t, S_e.
        """
        coords = [s_k, s_t, s_e]
        trits = []

        for i in range(depth):
            axis = i % 3
            # Find which third the coordinate falls into
            c = coords[axis]
            if c < 1/3:
                trit = 0
                coords[axis] = c * 3
            elif c < 2/3:
                trit = 1
                coords[axis] = (c - 1/3) * 3
            else:
                trit = 2
                coords[axis] = (c - 2/3) * 3
            trits.append(trit)

        return cls(tuple(trits))

    def to_scoord(self) -> Tuple[float, float, float]:
        """
        Convert ternary address to S-entropy coordinates.

        Returns the center of the cell addressed by this trit sequence.
        """
        # Track bounds for each axis
        bounds = {
            0: [0.0, 1.0],  # S_k bounds
            1: [0.0, 1.0],  # S_t bounds
            2: [0.0, 1.0],  # S_e bounds
        }

        for i, trit in enumerate(self.trits):
            axis = i % 3
            low, high = bounds[axis]
            width = (high - low) / 3

            # Update bounds based on trit value
            new_low = low + trit * width
            new_high = new_low + width
            bounds[axis] = [new_low, new_high]

        # Return centers
        s_k = (bounds[0][0] + bounds[0][1]) / 2
        s_t = (bounds[1][0] + bounds[1][1]) / 2
        s_e = (bounds[2][0] + bounds[2][1]) / 2

        return (s_k, s_t, s_e)

    def extend(self, extension: 'TernaryAddress') -> 'TernaryAddress':
        """Extend this address with additional trits."""
        return TernaryAddress(self.trits + extension.trits)

    def fragment_at(self, k: int) -> Tuple['TernaryAddress', 'TernaryAddress']:
        """Fragment address at position k."""
        if k < 0 or k > len(self.trits):
            raise ValueError(f"Invalid fragment position: {k}")
        return (TernaryAddress(self.trits[:k]),
                TernaryAddress(self.trits[k:]))

    def depth(self) -> int:
        """Return the number of trits in the address."""
        return len(self.trits)

    def cell_volume(self) -> float:
        """Return the volume of the cell addressed."""
        return (1/3) ** self.depth()

    def __str__(self) -> str:
        return ''.join(str(t) for t in self.trits)

    def __repr__(self) -> str:
        return f"TernaryAddress('{self}')"


@dataclass
class PartitionState:
    """
    Partition coordinates (n, l, m, s) derived from ternary address.
    """
    n: int      # Principal partition number (mass scale)
    l: int      # Angular partition number (chromatographic)
    m: int      # Magnetic partition number (isotope)
    s: float    # Spin partition number (polarity)

    @classmethod
    def from_address(cls, addr: TernaryAddress) -> 'PartitionState':
        """Derive partition state from ternary address."""
        s_k, s_t, s_e = addr.to_scoord()
        return cls.from_scoord(s_k, s_t, s_e)

    @classmethod
    def from_scoord(cls, s_k: float, s_t: float, s_e: float) -> 'PartitionState':
        """Derive partition state directly from S-coordinates."""
        # Principal quantum number: inverse of S_k, scaled for mass range
        n = max(1, int(1 / max(s_k, 0.01)))

        # Angular quantum number from S_t
        l = min(int(n * s_t), n - 1) if n > 0 else 0

        # Magnetic quantum number from S_e
        m_range = 2 * l + 1
        m = int(m_range * s_e) - l

        # Spin from polarity
        s = 0.5 if s_e >= 0.5 else -0.5

        return cls(n=n, l=l, m=m, s=s)

    def capacity(self) -> int:
        """Return the capacity C(n) = 2n^2."""
        return 2 * self.n ** 2

    def to_tuple(self) -> Tuple[int, int, int, float]:
        return (self.n, self.l, self.m, self.s)

    def __repr__(self) -> str:
        return f"PartitionState(n={self.n}, l={self.l}, m={self.m}, s={self.s:+.1f})"


@dataclass
class SEntropyCoord:
    """S-entropy coordinates (S_k, S_t, S_e)."""
    s_k: float
    s_t: float
    s_e: float

    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.s_k, self.s_t, self.s_e)

    def to_address(self, depth: int = 18) -> TernaryAddress:
        return TernaryAddress.from_scoord(self.s_k, self.s_t, self.s_e, depth)

    def distance(self, other: 'SEntropyCoord') -> float:
        return np.sqrt(
            (self.s_k - other.s_k)**2 +
            (self.s_t - other.s_t)**2 +
            (self.s_e - other.s_e)**2
        )

    @classmethod
    def from_address(cls, addr: TernaryAddress) -> 'SEntropyCoord':
        s_k, s_t, s_e = addr.to_scoord()
        return cls(s_k=s_k, s_t=s_t, s_e=s_e)


@dataclass
class Spectrum:
    """Mass spectrum representation."""
    mz: float
    retention_time: float
    intensity: float = 1.0
    fragments: List[Tuple[float, float]] = field(default_factory=list)
    isotope_pattern: List[Tuple[float, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'mz': self.mz,
            'rt': self.retention_time,
            'intensity': self.intensity,
            'fragments': self.fragments,
            'isotopes': self.isotope_pattern
        }


class SpectrumExtractor:
    """
    Extract mass spectrometric observables from S-entropy coordinates.

    Key insight: We extract directly from S-coordinates, NOT from
    partition state. This ensures proper calibration.
    """

    def __init__(self,
                 mass_min: float = 100.0,
                 mass_max: float = 1000.0,
                 t0: float = 0.5,
                 t_max: float = 20.0):
        """
        Initialize extractor with calibrated parameters.
        """
        self.mass_min = mass_min
        self.mass_max = mass_max
        self.t0 = t0
        self.t_max = t_max

    def mass_from_scoord(self, s_k: float) -> float:
        """
        Extract m/z directly from S_k coordinate.

        S_k=0 -> mass_max, S_k=1 -> mass_min (inverse relationship)
        """
        # Log scale for mass
        log_min = np.log10(self.mass_min)
        log_max = np.log10(self.mass_max)
        log_mass = log_max - s_k * (log_max - log_min)
        return 10 ** log_mass

    def retention_time_from_scoord(self, s_t: float) -> float:
        """
        Extract retention time directly from S_t coordinate.

        S_t is directly proportional to retention time.
        """
        return self.t0 + s_t * (self.t_max - self.t0)

    def fragments_from_scoord(self, s_k: float, s_e: float) -> List[Tuple[float, float]]:
        """
        Extract fragments from S_k and S_e coordinates.

        Higher S_e means more fragmentation.
        """
        parent_mass = self.mass_from_scoord(s_k)
        fragments = []

        # Number of fragments depends on S_e
        n_frags = max(0, int(s_e * 5))

        for i in range(1, n_frags + 1):
            # Fragments are fractions of parent mass
            frag_ratio = 0.8 - 0.15 * i
            if frag_ratio > 0.1:
                frag_mass = parent_mass * frag_ratio
                intensity = 1.0 / (i ** 1.2)
                fragments.append((frag_mass, intensity))

        return fragments

    def isotope_pattern_from_scoord(self, s_k: float) -> List[Tuple[float, float]]:
        """Extract isotope pattern."""
        base_mass = self.mass_from_scoord(s_k)

        # Simplified isotope pattern
        pattern = [
            (base_mass, 1.0),
            (base_mass + 1.003, 0.35),
            (base_mass + 2.006, 0.08),
        ]
        return pattern

    def extract_from_scoord(self, s_k: float, s_t: float, s_e: float) -> Spectrum:
        """Extract spectrum directly from S-coordinates."""
        return Spectrum(
            mz=self.mass_from_scoord(s_k),
            retention_time=self.retention_time_from_scoord(s_t),
            fragments=self.fragments_from_scoord(s_k, s_e),
            isotope_pattern=self.isotope_pattern_from_scoord(s_k)
        )

    def extract_spectrum(self, addr: TernaryAddress) -> Spectrum:
        """Extract spectrum from ternary address."""
        s_k, s_t, s_e = addr.to_scoord()
        return self.extract_from_scoord(s_k, s_t, s_e)


class MoleculeEncoder:
    """
    Encode molecular structures as ternary addresses.

    This is the INVERSE of SpectrumExtractor: given molecular
    properties, compute the S-coordinates and ternary address.
    """

    def __init__(self,
                 mass_min: float = 100.0,
                 mass_max: float = 1000.0,
                 t0: float = 0.5,
                 t_max: float = 20.0):
        self.mass_min = mass_min
        self.mass_max = mass_max
        self.t0 = t0
        self.t_max = t_max

    def encode(self,
               exact_mass: float,
               retention_time: float = 10.0,
               fragmentation: float = 0.5,
               depth: int = 18) -> TernaryAddress:
        """
        Encode molecular properties as ternary address.
        """
        # S_k from mass (inverse log relationship)
        log_min = np.log10(self.mass_min)
        log_max = np.log10(self.mass_max)
        log_mass = np.log10(np.clip(exact_mass, self.mass_min, self.mass_max))
        s_k = (log_max - log_mass) / (log_max - log_min)
        s_k = np.clip(s_k, 0.01, 0.99)

        # S_t from retention time
        s_t = (retention_time - self.t0) / (self.t_max - self.t0)
        s_t = np.clip(s_t, 0.01, 0.99)

        # S_e from fragmentation
        s_e = np.clip(fragmentation, 0.01, 0.99)

        return TernaryAddress.from_scoord(s_k, s_t, s_e, depth)

    def encode_from_formula(self, formula: str,
                           retention_time: float = None,
                           depth: int = 18) -> TernaryAddress:
        """Encode from molecular formula."""
        mass = self._estimate_mass(formula)

        # Estimate RT from formula if not provided
        if retention_time is None:
            logp = self._estimate_logp(formula)
            # RT correlates with logP
            retention_time = self.t0 + (logp + 2) / 10 * (self.t_max - self.t0)
            retention_time = np.clip(retention_time, self.t0, self.t_max)

        frag = self._estimate_fragmentation(formula)

        return self.encode(mass, retention_time, frag, depth)

    def _estimate_mass(self, formula: str) -> float:
        """Estimate mass from formula."""
        masses = {
            'C': 12.0000, 'H': 1.00794, 'O': 15.9994, 'N': 14.0067,
            'P': 30.9738, 'S': 32.065, 'F': 18.9984, 'Cl': 35.453,
            'Br': 79.904, 'I': 126.904
        }

        total = 0.0
        i = 0
        while i < len(formula):
            if formula[i].isupper():
                element = formula[i]
                if i + 1 < len(formula) and formula[i+1].islower():
                    element += formula[i+1]
                    i += 1

                i += 1
                count_str = ''
                while i < len(formula) and formula[i].isdigit():
                    count_str += formula[i]
                    i += 1
                count = int(count_str) if count_str else 1

                total += masses.get(element, 12.0) * count
            else:
                i += 1

        return total if total > 0 else 200.0

    def _estimate_logp(self, formula: str) -> float:
        """Estimate logP from formula."""
        c_count = sum(1 for c in formula if c == 'C')
        h_count = sum(1 for c in formula if c == 'H')
        o_count = sum(1 for c in formula if c == 'O')
        n_count = sum(1 for c in formula if c == 'N')

        # Simplified Wildman-Crippen-like estimate
        logp = 0.3 * c_count - 0.5 * o_count - 0.3 * n_count - 0.05 * h_count
        return logp

    def _estimate_fragmentation(self, formula: str) -> float:
        """Estimate fragmentation propensity."""
        mass = self._estimate_mass(formula)
        return np.clip(0.3 + 0.0005 * mass, 0.1, 0.9)


def tryte_to_cell(tryte: Tuple[int, ...]) -> int:
    """Convert 6-trit tryte to cell index (0-728)."""
    if len(tryte) != 6:
        raise ValueError("Tryte must have exactly 6 trits")

    index = 0
    for i, t in enumerate(tryte):
        index += t * (3 ** (5 - i))
    return index


def cell_to_tryte(cell: int) -> Tuple[int, ...]:
    """Convert cell index (0-728) to 6-trit tryte."""
    if cell < 0 or cell >= 729:
        raise ValueError("Cell index must be 0-728")

    trits = []
    for _ in range(6):
        trits.append(cell % 3)
        cell //= 3
    return tuple(reversed(trits))


def synthesize(address_str: str,
               mass_min: float = 100.0,
               mass_max: float = 1000.0) -> Spectrum:
    """Synthesize spectrum from ternary address string."""
    addr = TernaryAddress.from_string(address_str)
    extractor = SpectrumExtractor(mass_min=mass_min, mass_max=mass_max)
    return extractor.extract_spectrum(addr)


def observe(address_str: str) -> Dict[str, Any]:
    """Observe (extract) all observables from ternary address."""
    return synthesize(address_str).to_dict()


if __name__ == '__main__':
    print("Mass Computing Framework - Ternary Partition Synthesis")
    print("=" * 60)

    # Test encode-extract roundtrip
    print("\n--- Encode-Extract Roundtrip Test ---")

    encoder = MoleculeEncoder(mass_min=100, mass_max=1000)
    extractor = SpectrumExtractor(mass_min=100, mass_max=1000)

    test_cases = [
        ("PC 34:1", "C42H82NO8P", 760.59, 14.2),
        ("Glucose", "C6H12O6", 180.06, 0.5),
        ("Phenylalanine", "C9H11NO2", 165.08, 2.8),
    ]

    for name, formula, expected_mass, expected_rt in test_cases:
        # Encode
        addr = encoder.encode(expected_mass, expected_rt, 0.5)

        # Extract
        spectrum = extractor.extract_spectrum(addr)

        # Calculate errors
        mz_error = abs(spectrum.mz - expected_mass) / expected_mass * 1e6
        rt_error = abs(spectrum.retention_time - expected_rt)

        print(f"\n{name}:")
        print(f"  Expected: m/z={expected_mass:.2f}, RT={expected_rt:.1f}")
        print(f"  Got:      m/z={spectrum.mz:.2f}, RT={spectrum.retention_time:.1f}")
        print(f"  Errors:   {mz_error:.0f} ppm, {rt_error:.2f} min")
