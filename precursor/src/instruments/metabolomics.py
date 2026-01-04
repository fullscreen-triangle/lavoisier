"""
Metabolomics Instruments

Specialized instruments for applying partition theory to metabolomics:
- Fragmentation Topology Mapper: Maps MS2 as categorical completion
- S-Entropy Virtual Mass Spectrometer: Virtual MS in S-coordinates

Theory Application:
- Molecular fragmentation = categorical completion through bond topology
- Fragment intensity = phase-lock degeneracy (how many ways to reach fragment)
- MS2 spectra reveal molecular categorical structure
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
from .base import (
    VirtualInstrument,
    HardwareOscillator,
    CategoricalState,
    SEntropyCoordinate,
    BOLTZMANN_CONSTANT
)


@dataclass
class MolecularBond:
    """Represents a bond in molecular topology"""
    atom_i: int
    atom_j: int
    bond_type: str  # 'single', 'double', 'triple', 'aromatic'
    phase_lock_strength: float = 1.0
    
    def get_dissociation_energy(self) -> float:
        """Get relative dissociation energy (categorical completion barrier)"""
        energies = {
            'single': 1.0,
            'double': 2.0,
            'triple': 3.0,
            'aromatic': 1.5
        }
        return energies.get(self.bond_type, 1.0)


@dataclass
class FragmentNode:
    """A node in the fragmentation tree"""
    fragment_id: int
    mass: float
    atom_indices: Set[int]
    parent_id: Optional[int] = None
    intensity: float = 0.0
    categorical_completion_path: List[int] = field(default_factory=list)


class FragmentationTopologyMapper(VirtualInstrument):
    """
    Fragmentation Topology Mapper - Maps MS2 as categorical completion.
    
    Theory: Molecular fragmentation is NOT random bond breaking.
    It is categorical completion through bond topology:
    - Bonds = phase-lock relationships between atoms
    - Fragmentation follows phase-lock adjacency
    - Intensity reflects degeneracy (number of paths to fragment)
    
    The "Molecular Maxwell Demon" is actually categorical completion
    through molecular phase-lock networks - no demon required.
    """
    
    def __init__(self):
        super().__init__("Fragmentation Topology Mapper")
        
    def calibrate(self) -> bool:
        """Calibrate fragmentation detection"""
        self.calibrated = True
        return True
    
    def build_bond_network(self, n_atoms: int, 
                            bonds: List[Tuple[int, int, str]]) -> Dict[str, Any]:
        """
        Build molecular bond network (phase-lock topology).
        
        Args:
            n_atoms: Number of atoms
            bonds: List of (atom_i, atom_j, bond_type) tuples
            
        Returns:
            Dictionary with network structure
        """
        adjacency = np.zeros((n_atoms, n_atoms))
        bond_objects = []
        
        for atom_i, atom_j, bond_type in bonds:
            bond = MolecularBond(
                atom_i=atom_i,
                atom_j=atom_j,
                bond_type=bond_type,
                phase_lock_strength=1.0 / MolecularBond(0, 0, bond_type).get_dissociation_energy()
            )
            bond_objects.append(bond)
            
            # Stronger bonds have stronger phase-lock
            adjacency[atom_i, atom_j] = bond.phase_lock_strength
            adjacency[atom_j, atom_i] = bond.phase_lock_strength
        
        return {
            'n_atoms': n_atoms,
            'bonds': bond_objects,
            'adjacency': adjacency,
            'n_bonds': len(bonds)
        }
    
    def find_weakest_phase_lock(self, network: Dict[str, Any],
                                 available_bonds: List[MolecularBond]) -> Optional[MolecularBond]:
        """
        Find the weakest phase-lock (most likely to break).
        
        Categorical completion follows the path of least resistance
        through phase-lock network.
        """
        if not available_bonds:
            return None
        
        # Sort by phase-lock strength (weak first)
        sorted_bonds = sorted(available_bonds, key=lambda b: b.phase_lock_strength)
        return sorted_bonds[0]
    
    def predict_fragmentation_path(self, network: Dict[str, Any],
                                    max_fragments: int = 10) -> List[FragmentNode]:
        """
        Predict fragmentation as categorical completion through bond topology.
        
        Algorithm:
        1. Start with intact molecule
        2. Find weakest phase-lock (categorical boundary)
        3. Break to create fragments
        4. Repeat for each fragment
        """
        n_atoms = network['n_atoms']
        bonds = network['bonds'].copy()
        
        # Initial molecule
        root = FragmentNode(
            fragment_id=0,
            mass=n_atoms * 12.0,  # Simplified: all carbon
            atom_indices=set(range(n_atoms))
        )
        
        fragments = [root]
        fragment_id = 1
        
        # Categorical completion: break weakest phase-locks
        available_bonds = bonds.copy()
        
        while len(fragments) < max_fragments and available_bonds:
            # Find weakest bond (categorical boundary)
            weakest = self.find_weakest_phase_lock(network, available_bonds)
            if weakest is None:
                break
            
            available_bonds.remove(weakest)
            
            # Find which fragment this bond belongs to
            for frag in fragments:
                if weakest.atom_i in frag.atom_indices and weakest.atom_j in frag.atom_indices:
                    # Break this fragment into two
                    # Simplified: just split atom indices
                    atoms_1 = {a for a in frag.atom_indices if a <= weakest.atom_i}
                    atoms_2 = frag.atom_indices - atoms_1
                    
                    if atoms_1 and atoms_2:
                        frag_1 = FragmentNode(
                            fragment_id=fragment_id,
                            mass=len(atoms_1) * 12.0,
                            atom_indices=atoms_1,
                            parent_id=frag.fragment_id,
                            categorical_completion_path=frag.categorical_completion_path + [fragment_id]
                        )
                        fragment_id += 1
                        
                        frag_2 = FragmentNode(
                            fragment_id=fragment_id,
                            mass=len(atoms_2) * 12.0,
                            atom_indices=atoms_2,
                            parent_id=frag.fragment_id,
                            categorical_completion_path=frag.categorical_completion_path + [fragment_id]
                        )
                        fragment_id += 1
                        
                        fragments.append(frag_1)
                        fragments.append(frag_2)
                    break
        
        return fragments
    
    def compute_degeneracy(self, fragments: List[FragmentNode]) -> Dict[int, int]:
        """
        Compute phase-lock degeneracy for each fragment.
        
        Degeneracy = number of categorical paths reaching the fragment
        This determines intensity in the spectrum.
        """
        degeneracy = defaultdict(int)
        
        for frag in fragments:
            # Count paths to this fragment
            path_length = len(frag.categorical_completion_path)
            # More paths = higher degeneracy = higher intensity
            degeneracy[frag.fragment_id] = max(1, 2 ** path_length)
        
        return dict(degeneracy)
    
    def measure(self, n_atoms: int = 10, 
                bonds: Optional[List[Tuple[int, int, str]]] = None,
                **kwargs) -> Dict[str, Any]:
        """
        Map molecular fragmentation as categorical completion.
        
        Args:
            n_atoms: Number of atoms in molecule
            bonds: Bond list, or generate random if None
            
        Returns:
            Dictionary with fragmentation analysis
        """
        # Generate bonds if not provided
        if bonds is None:
            bonds = []
            for i in range(n_atoms - 1):
                bond_type = np.random.choice(['single', 'double', 'aromatic'])
                bonds.append((i, i + 1, bond_type))
            # Add some branching
            for _ in range(n_atoms // 3):
                i = np.random.randint(0, n_atoms - 2)
                j = np.random.randint(i + 2, n_atoms)
                if j < n_atoms:
                    bonds.append((i, j, 'single'))
        
        # Build network
        network = self.build_bond_network(n_atoms, bonds)
        
        # Predict fragmentation
        fragments = self.predict_fragmentation_path(network)
        
        # Compute degeneracies (intensities)
        degeneracy = self.compute_degeneracy(fragments)
        
        # Assign intensities
        max_deg = max(degeneracy.values()) if degeneracy else 1
        for frag in fragments:
            frag.intensity = degeneracy[frag.fragment_id] / max_deg
        
        # Create spectrum
        spectrum = [(frag.mass, frag.intensity) for frag in fragments]
        spectrum.sort(key=lambda x: x[0])
        
        result = {
            'n_atoms': n_atoms,
            'n_bonds': len(bonds),
            'network': network,
            'fragments': fragments,
            'degeneracy': degeneracy,
            'spectrum': spectrum,
            'explanation': (
                'Fragmentation proceeds by categorical completion through bond '
                'topology. Weakest phase-lock bonds (categorical boundaries) '
                'break first. Fragment intensity reflects degeneracy - the '
                'number of categorical paths reaching that fragment. '
                'No "demon" is required; fragmentation is automatic topology.'
            )
        }
        
        self.record_measurement(result)
        return result
    
    def explain_molecular_maxwell_demon(self) -> Dict[str, Any]:
        """
        Explain why there is no "Molecular Maxwell Demon".
        
        The term is pedagogical - it describes how categorical completion
        LOOKS like intelligent sorting but requires no agent.
        """
        return {
            'demon_does_not_exist': True,
            'what_looks_like_sorting': (
                'Molecular fragmentation appears to "sort" fragments by stability, '
                'as if an intelligent agent chose which bonds to break.'
            ),
            'what_actually_happens': (
                'Categorical completion through phase-lock bond topology. '
                'Weak bonds (low phase-lock strength) are categorical boundaries. '
                'The molecule follows the path of minimum categorical distance. '
                'This is automatic, deterministic, requires no information.'
            ),
            'why_keep_the_name': (
                'The "Maxwell Demon" terminology explains the "hard maths" - '
                'how categorical completion dynamics appear when projected '
                'onto the observable intensity/mass face. The demon is the '
                'shadow of categorical structure, not an actual agent.'
            )
        }


class SEntropyMassSpectrometer(VirtualInstrument):
    """
    S-Entropy Virtual Mass Spectrometer
    
    A mass spectrometer that operates in categorical coordinates (S_k, S_t, S_e)
    rather than physical m/z space.
    
    Key features:
    - Scans categorical state space
    - Each measurement creates (not discovers) a state
    - All instruments converge to same categorical entropy
    """
    
    def __init__(self, resolution: float = 0.01):
        super().__init__("S-Entropy Virtual Mass Spectrometer")
        self.resolution = resolution  # S-space resolution
        
    def calibrate(self) -> bool:
        """Calibrate S-space scanning"""
        self.calibrated = True
        return True
    
    def S_to_mz(self, S_k: float, S_t: float, S_e: float) -> float:
        """
        Convert S-coordinates to approximate m/z.
        
        This is a projection from categorical to physical space.
        """
        # Simplified mapping (would be calibrated in practice)
        mz = 100 + 1000 * S_k + 500 * S_t + 200 * S_e
        return mz
    
    def scan_S_space(self, S_range: Dict[str, Tuple[float, float]],
                      threshold: float = 0.01) -> List[Dict[str, Any]]:
        """
        Scan a region of S-entropy space.
        
        Args:
            S_range: Dictionary with 'S_k', 'S_t', 'S_e' ranges as (min, max) tuples
            threshold: Intensity threshold for reporting
            
        Returns:
            List of detected peaks in S-space
        """
        peaks = []
        
        S_k_range = S_range.get('S_k', (0, 1))
        S_t_range = S_range.get('S_t', (0, 1))
        S_e_range = S_range.get('S_e', (0, 1))
        
        for S_k in np.arange(S_k_range[0], S_k_range[1], self.resolution):
            for S_t in np.arange(S_t_range[0], S_t_range[1], self.resolution):
                for S_e in np.arange(S_e_range[0], S_e_range[1], self.resolution):
                    # Create categorical state from hardware
                    delta_p = self.oscillator.read_timing_deviation()
                    
                    # Check if this state is "occupied"
                    # (hardware timing determines what we "find")
                    occupation = np.exp(-(S_k**2 + S_t**2 + S_e**2))
                    occupation *= (1 + 0.1 * np.sin(delta_p / 100))
                    
                    if occupation > threshold:
                        peaks.append({
                            'S_coords': (S_k, S_t, S_e),
                            'intensity': occupation,
                            'mz_equivalent': self.S_to_mz(S_k, S_t, S_e)
                        })
        
        return peaks
    
    def measure(self, scan_mode: str = 'full', **kwargs) -> Dict[str, Any]:
        """
        Perform S-entropy mass spectrometry measurement.
        
        Args:
            scan_mode: 'full' for complete scan, 'targeted' for specific region
            
        Returns:
            Dictionary with S-space spectrum
        """
        if scan_mode == 'full':
            S_range = {
                'S_k': (0, 1),
                'S_t': (0, 1),
                'S_e': (0, 1)
            }
        else:
            # Targeted scan around a specific region
            S_range = kwargs.get('S_range', {
                'S_k': (0.3, 0.7),
                'S_t': (0.3, 0.7),
                'S_e': (0.3, 0.7)
            })
        
        # Scan
        peaks = self.scan_S_space(S_range)
        
        # Compute categorical entropy of detected states
        n_peaks = len(peaks)
        if n_peaks > 0:
            intensities = np.array([p['intensity'] for p in peaks])
            probs = intensities / intensities.sum()
            categorical_entropy = -BOLTZMANN_CONSTANT * np.sum(probs * np.log(probs + 1e-10))
        else:
            categorical_entropy = 0
        
        result = {
            'scan_mode': scan_mode,
            'S_range': S_range,
            'resolution': self.resolution,
            'n_peaks': n_peaks,
            'peaks': peaks,
            'categorical_entropy': categorical_entropy,
            'mz_spectrum': [(p['mz_equivalent'], p['intensity']) for p in peaks],
            'explanation': (
                'S-Entropy MS operates in categorical coordinates (S_k, S_t, S_e). '
                'Each scan position creates (not discovers) a categorical state. '
                'The spectrum reveals categorical structure of the sample, '
                'with m/z being a projection onto physical observables.'
            )
        }
        
        self.record_measurement(result)
        return result
    
    def compare_to_physical_MS(self) -> Dict[str, Any]:
        """
        Compare S-Entropy MS to physical mass spectrometry.
        """
        return {
            'physical_MS': {
                'measures': 'm/z (mass-to-charge ratio)',
                'domain': 'Physical space',
                'interpretation': 'Ion detection at specific m/z',
                'limitation': 'Spatial sampling required'
            },
            'S_entropy_MS': {
                'measures': '(S_k, S_t, S_e) categorical coordinates',
                'domain': 'Categorical state space',
                'interpretation': 'Categorical state instantiation',
                'advantage': 'No spatial limitation (categorical distance independence)'
            },
            'equivalence_theorem': (
                'Both should yield identical categorical entropy S = k_B × M × ln(n). '
                'Physical MS and S-Entropy MS are different projections of the '
                'same underlying categorical structure.'
            ),
            'practical_advantage': (
                'S-Entropy MS can access categorical states independent of '
                'physical location. A molecule at Jupiter\'s core categorical '
                'coordinates can be measured from Earth by configuring the '
                'apparatus to access those S-coordinates.'
            )
        }

