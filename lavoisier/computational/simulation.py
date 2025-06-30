"""
Virtual Molecular Simulation Module - Python Bindings

High-performance molecular simulation using Rust backend for
virtual molecular structure generation and spectral prediction.
"""

import asyncio
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
from dataclasses import dataclass

try:
    # Import the Rust module
    import lavoisier_computational
    RUST_AVAILABLE = True
except ImportError:
    # Fallback to pure Python implementation
    RUST_AVAILABLE = False
    import random
    import math


@dataclass
class VirtualMolecule:
    """Represents a virtual molecule with its properties"""
    id: str
    formula: str
    mass: float
    energy: float
    mz_peaks: List[float]
    intensities: List[float]
    retention_time: float
    metadata: Dict[str, Any]


class VirtualMolecularSimulator:
    """
    High-performance virtual molecular simulator.

    Uses Rust backend when available for maximum performance.
    """

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed

        if RUST_AVAILABLE:
            self._rust_simulator = lavoisier_computational.PyVirtualMolecularSimulator(seed)
            self._use_rust = True
        else:
            self._use_rust = False
            if seed:
                random.seed(seed)
                np.random.seed(seed)

    def generate_molecule(self, formula: str, target_mass: float) -> VirtualMolecule:
        """Generate a virtual molecule with specified properties"""
        if self._use_rust:
            result = self._rust_simulator.generate_molecule(formula, target_mass)
            return VirtualMolecule(
                id=result["id"],
                formula=result["formula"],
                mass=result["mass"],
                energy=result["energy"],
                mz_peaks=result["mz_peaks"],
                intensities=result["intensities"],
                retention_time=result["retention_time"],
                metadata={}
            )
        else:
            return self._generate_python_molecule(formula, target_mass)

    def _generate_python_molecule(self, formula: str, target_mass: float) -> VirtualMolecule:
        """Python fallback molecular generation"""
        mol_id = f"VM_{random.randint(10000, 99999)}"

        # Simple spectral simulation
        num_peaks = random.randint(5, 15)
        mz_peaks = sorted([target_mass * random.uniform(0.1, 1.0) for _ in range(num_peaks)])
        mz_peaks.append(target_mass)  # Molecular ion peak

        intensities = [random.uniform(5, 100) for _ in range(len(mz_peaks))]
        intensities[-1] = 100.0  # Molecular ion is base peak

        # Simple retention time model
        carbon_count = formula.count('C')
        oxygen_count = formula.count('O')
        rt = carbon_count * 0.5 - oxygen_count * 0.3 + random.uniform(-0.5, 0.5)
        rt = max(0.1, rt)

        return VirtualMolecule(
            id=mol_id,
            formula=formula,
            mass=target_mass,
            energy=random.uniform(100, 1000),
            mz_peaks=mz_peaks,
            intensities=intensities,
            retention_time=rt,
            metadata={"generator": "python_fallback"}
        )

    def generate_molecular_library(self,
                                 formulas: List[str],
                                 masses: List[float]) -> List[VirtualMolecule]:
        """Generate a library of virtual molecules"""
        if len(formulas) != len(masses):
            raise ValueError("Formulas and masses lists must have the same length")

        molecules = []
        for formula, mass in zip(formulas, masses):
            molecule = self.generate_molecule(formula, mass)
            molecules.append(molecule)

        return molecules

    def simulate_fragmentation(self, molecule: VirtualMolecule) -> Dict[str, List[float]]:
        """Simulate molecular fragmentation patterns"""
        fragments = {}

        # Common neutral losses
        neutral_losses = [1.0, 15.0, 17.0, 18.0, 28.0, 44.0, 45.0]

        for loss in neutral_losses:
            if molecule.mass > loss:
                fragment_mz = molecule.mass - loss
                fragment_intensity = random.uniform(10, 80)

                loss_name = {
                    1.0: "H",
                    15.0: "CH3",
                    17.0: "OH",
                    18.0: "H2O",
                    28.0: "CO",
                    44.0: "CO2",
                    45.0: "COOH"
                }.get(loss, f"NL_{loss}")

                fragments[loss_name] = [fragment_mz, fragment_intensity]

        return fragments


class MolecularResonanceEngine:
    """
    Engine for analyzing resonance between virtual molecules and hardware oscillations.
    """

    def __init__(self, resonance_threshold: float = 0.5):
        self.resonance_threshold = resonance_threshold

        if RUST_AVAILABLE:
            self._rust_engine = lavoisier_computational.PyMolecularResonanceEngine(resonance_threshold)
            self._use_rust = True
        else:
            self._use_rust = False
            self.molecules = []

    def add_molecule(self, molecule: VirtualMolecule) -> None:
        """Add a virtual molecule to the resonance engine"""
        if not self._use_rust:
            self.molecules.append(molecule)

    def calculate_resonance(self,
                          hardware_spectrum: np.ndarray,
                          hardware_frequencies: np.ndarray) -> List[Tuple[str, float]]:
        """Calculate resonance between molecular vibrations and hardware oscillations"""
        if self._use_rust:
            return self._rust_engine.calculate_resonance(
                hardware_spectrum.tolist(),
                hardware_frequencies.tolist()
            )
        else:
            return self._calculate_python_resonance(hardware_spectrum, hardware_frequencies)

    def _calculate_python_resonance(self,
                                   hardware_spectrum: np.ndarray,
                                   hardware_frequencies: np.ndarray) -> List[Tuple[str, float]]:
        """Python fallback resonance calculation"""
        resonance_scores = []

        for molecule in self.molecules:
            total_resonance = 0.0
            match_count = 0

            # Compare molecular peaks with hardware frequencies
            for mz, intensity in zip(molecule.mz_peaks, molecule.intensities):
                # Convert m/z to frequency (simplified model)
                molecular_freq = mz / 100.0  # Simple conversion

                for i, hw_freq in enumerate(hardware_frequencies):
                    freq_match = math.exp(-abs(molecular_freq - hw_freq) / molecular_freq)

                    if freq_match > self.resonance_threshold:
                        hw_amplitude = hardware_spectrum[i] if i < len(hardware_spectrum) else 0.0
                        resonance = freq_match * (intensity / 100.0) * hw_amplitude
                        total_resonance += resonance
                        match_count += 1

            if match_count > 0:
                avg_resonance = total_resonance / match_count
                resonance_scores.append((molecule.id, avg_resonance))

        resonance_scores.sort(key=lambda x: x[1], reverse=True)
        return resonance_scores

    def find_best_matches(self,
                         hardware_spectrum: np.ndarray,
                         hardware_frequencies: np.ndarray,
                         top_n: int = 5) -> List[Dict[str, Any]]:
        """Find the best molecular matches for given hardware patterns"""
        resonance_scores = self.calculate_resonance(hardware_spectrum, hardware_frequencies)

        best_matches = []
        for i, (mol_id, score) in enumerate(resonance_scores[:top_n]):
            if self._use_rust:
                # In Rust version, we'd need to query molecule details
                match_info = {
                    "molecule_id": mol_id,
                    "resonance_score": score,
                    "rank": i + 1,
                    "confidence": min(score * 100, 100)
                }
            else:
                # Find the molecule in our Python list
                molecule = next((m for m in self.molecules if m.id == mol_id), None)
                if molecule:
                    match_info = {
                        "molecule_id": mol_id,
                        "formula": molecule.formula,
                        "mass": molecule.mass,
                        "resonance_score": score,
                        "rank": i + 1,
                        "confidence": min(score * 100, 100),
                        "retention_time": molecule.retention_time
                    }
                else:
                    match_info = {
                        "molecule_id": mol_id,
                        "resonance_score": score,
                        "rank": i + 1,
                        "confidence": min(score * 100, 100)
                    }

            best_matches.append(match_info)

        return best_matches


class MolecularDatabase:
    """Database of virtual molecules for analysis"""

    def __init__(self):
        self.molecules = {}
        self.simulator = VirtualMolecularSimulator()

    def add_molecule(self, molecule: VirtualMolecule) -> None:
        """Add a molecule to the database"""
        self.molecules[molecule.id] = molecule

    def get_molecule(self, mol_id: str) -> Optional[VirtualMolecule]:
        """Get a molecule by ID"""
        return self.molecules.get(mol_id)

    def search_by_formula(self, formula: str) -> List[VirtualMolecule]:
        """Search molecules by molecular formula"""
        return [mol for mol in self.molecules.values() if mol.formula == formula]

    def search_by_mass(self, target_mass: float, tolerance: float = 0.01) -> List[VirtualMolecule]:
        """Search molecules by mass within tolerance"""
        return [
            mol for mol in self.molecules.values()
            if abs(mol.mass - target_mass) <= tolerance
        ]

    def generate_common_metabolites(self) -> None:
        """Generate common metabolites for the database"""
        common_metabolites = [
            ("C6H12O6", 180.156),  # Glucose
            ("C3H6O3", 90.078),    # Lactate
            ("C3H4O3", 88.062),    # Pyruvate
            ("C4H6O5", 134.088),   # Malate
            ("C4H6O4", 118.088),   # Succinate
            ("C6H8O7", 192.124),   # Citrate
            ("C2H5NO2", 75.067),   # Glycine
            ("C3H7NO2", 89.093),   # Alanine
            ("C6H13NO2", 131.173), # Leucine
            ("C5H11NO2", 117.146), # Valine
            ("C3H7NO3", 105.093),  # Serine
            ("C10H13N5O4", 267.241), # Adenosine (AMP base)
            ("C10H15N5O10P3", 507.181), # ATP
            ("C10H15N5O7P2", 427.201),  # ADP
        ]

        for formula, mass in common_metabolites:
            molecule = self.simulator.generate_molecule(formula, mass)
            self.add_molecule(molecule)

    def export_library(self) -> Dict[str, Any]:
        """Export the molecular library"""
        export_data = {
            "molecules": [],
            "count": len(self.molecules),
            "timestamp": np.datetime64('now').item().isoformat()
        }

        for molecule in self.molecules.values():
            mol_data = {
                "id": molecule.id,
                "formula": molecule.formula,
                "mass": molecule.mass,
                "energy": molecule.energy,
                "mz_peaks": molecule.mz_peaks,
                "intensities": molecule.intensities,
                "retention_time": molecule.retention_time,
                "metadata": molecule.metadata
            }
            export_data["molecules"].append(mol_data)

        return export_data


# High-level convenience functions
def simulate_molecule(formula: str, target_mass: float, seed: Optional[int] = None) -> VirtualMolecule:
    """
    Simulate a single molecule

    Args:
        formula: Molecular formula
        target_mass: Target molecular mass
        seed: Random seed for reproducibility

    Returns:
        VirtualMolecule object
    """
    if RUST_AVAILABLE:
        result = lavoisier_computational.py_simulate_molecule(formula, target_mass, seed)
        return VirtualMolecule(
            id=result["id"],
            formula=result["formula"],
            mass=result["mass"],
            energy=result["energy"],
            mz_peaks=result["mz_peaks"],
            intensities=result["intensities"],
            retention_time=result["retention_time"],
            metadata={}
        )
    else:
        simulator = VirtualMolecularSimulator(seed)
        return simulator.generate_molecule(formula, target_mass)


def create_metabolite_library(include_common: bool = True) -> MolecularDatabase:
    """
    Create a library of metabolite molecules

    Args:
        include_common: Whether to include common metabolites

    Returns:
        MolecularDatabase with metabolites
    """
    database = MolecularDatabase()

    if include_common:
        database.generate_common_metabolites()

    return database


def analyze_molecular_resonance(molecules: List[VirtualMolecule],
                              hardware_spectrum: np.ndarray,
                              hardware_frequencies: np.ndarray,
                              threshold: float = 0.5) -> Dict[str, Any]:
    """
    Analyze resonance between molecules and hardware oscillations

    Args:
        molecules: List of virtual molecules
        hardware_spectrum: Hardware oscillation spectrum
        hardware_frequencies: Hardware frequency array
        threshold: Resonance threshold

    Returns:
        Analysis results dictionary
    """
    engine = MolecularResonanceEngine(threshold)

    for molecule in molecules:
        engine.add_molecule(molecule)

    resonance_scores = engine.calculate_resonance(hardware_spectrum, hardware_frequencies)
    best_matches = engine.find_best_matches(hardware_spectrum, hardware_frequencies)

    return {
        "resonance_scores": resonance_scores,
        "best_matches": best_matches,
        "total_molecules": len(molecules),
        "significant_matches": len([s for s in resonance_scores if s[1] > threshold]),
        "analysis_threshold": threshold
    }


# Export main classes and functions
__all__ = [
    'VirtualMolecularSimulator',
    'MolecularResonanceEngine',
    'MolecularDatabase',
    'VirtualMolecule',
    'simulate_molecule',
    'create_metabolite_library',
    'analyze_molecular_resonance',
    'RUST_AVAILABLE'
]
