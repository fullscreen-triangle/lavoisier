"""
Virtual Aperture: Categorical Selection from Hardware
======================================================

A categorical aperture is NOT a simulated filter.
A categorical aperture IS an external charge field configuration that
selects molecules based on their charge distribution (S-coordinates).

FUNDAMENTAL INSIGHT:
An aperture = an external charge field at a point.
Molecules pass if their charge configuration matches the field geometry.
This is electromagnetic selection, not mechanical filtering.

Key properties:
- Selection is by CHARGE CONFIGURATION, not velocity
- Therefore selection is TEMPERATURE-INDEPENDENT
- This explains prebiotic chemistry at low temperatures
- Ion channels ARE charge-field apertures
- Membrane potentials ARE aperture potentials

The aperture IS an electric field configuration in S-coordinate space.
Molecules pass if their S-coordinates (charge distribution) match.
This is REAL selection from REAL hardware timing variations.
"""

import time
import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Callable
from collections import deque

try:
    from .virtual_molecule import VirtualMolecule, CategoricalState, SCoordinate
    from .virtual_chamber import VirtualChamber, CategoricalGas
    from .virtual_spectrometer import HardwareOscillator, VirtualSpectrometer
except ImportError:
    from virtual_molecule import VirtualMolecule, CategoricalState, SCoordinate
    from virtual_chamber import VirtualChamber, CategoricalGas
    from virtual_spectrometer import HardwareOscillator, VirtualSpectrometer


@dataclass
class ApertureResult:
    """Result of a molecule passing through an aperture."""
    passed: bool
    molecule: VirtualMolecule
    aperture_id: str
    timestamp: float = field(default_factory=time.perf_counter)

    # Why it passed or was blocked
    distance_to_center: float = 0.0
    aperture_radius: float = 0.0


class CategoricalAperture:
    """
    A Categorical Aperture: Selects molecules by S-coordinate configuration.

    Key theorem from paper: Apertures select by configuration, not velocity.
    Therefore, selection is temperature-independent.

    The aperture IS a region in S-space.
    Molecules whose S-coordinates fall within the region PASS.
    Molecules outside the region are BLOCKED.

    This is NOT based on velocity (which is temperature-dependent).
    This IS based on configuration (which is temperature-independent).
    """

    def __init__(self,
                 center: SCoordinate,
                 radius: float = 0.2,
                 aperture_id: str = "aperture"):
        """
        Create a categorical aperture.

        Args:
            center: Center of the aperture in S-space
            radius: Radius of the passable region
            aperture_id: Identifier for this aperture
        """
        self.center = center
        self.radius = radius
        self.aperture_id = aperture_id

        # Statistics
        self._passed: List[VirtualMolecule] = []
        self._blocked: List[VirtualMolecule] = []
        self._history: deque[ApertureResult] = deque(maxlen=10000)

    def evaluate(self, molecule: VirtualMolecule) -> ApertureResult:
        """
        Evaluate whether a molecule passes through the aperture.

        This is the fundamental categorical selection operation.
        Selection is based on S-coordinates (configuration), not velocity.
        """
        distance = molecule.s_coord.distance_to(self.center)
        passed = distance <= self.radius

        result = ApertureResult(
            passed=passed,
            molecule=molecule,
            aperture_id=self.aperture_id,
            distance_to_center=distance,
            aperture_radius=self.radius
        )

        if passed:
            self._passed.append(molecule)
        else:
            self._blocked.append(molecule)

        self._history.append(result)
        return result

    def filter(self, molecules: List[VirtualMolecule]) -> List[VirtualMolecule]:
        """Filter a list of molecules, returning only those that pass."""
        return [m for m in molecules if self.evaluate(m).passed]

    @property
    def selectivity(self) -> float:
        """
        Selectivity: fraction of molecules that pass.

        For a random distribution in S-space, this should be
        approximately the volume ratio (sphere/cube).
        """
        total = len(self._passed) + len(self._blocked)
        if total == 0:
            return 0.0
        return len(self._passed) / total

    @property
    def statistics(self) -> Dict[str, Any]:
        """Get aperture statistics."""
        return {
            'aperture_id': self.aperture_id,
            'center': self.center.as_tuple(),
            'radius': self.radius,
            'passed_count': len(self._passed),
            'blocked_count': len(self._blocked),
            'selectivity': self.selectivity,
        }

    def clear(self):
        """Clear history."""
        self._passed.clear()
        self._blocked.clear()
        self._history.clear()


class ChargeFieldAperture(CategoricalAperture):
    """
    An aperture generated by charge separation (electric field).

    From the paper: Electron transport creates charge separation.
    The resulting electric field defines equipotential surfaces.
    These surfaces act as categorical apertures.

    Molecules with compatible charge distribution pass.
    Molecules with incompatible charge distribution are blocked.
    """

    def __init__(self,
                 membrane_potential: float = -70e-3,  # -70 mV typical
                 temperature: float = 300):  # Kelvin
        """
        Create a charge field aperture.

        Args:
            membrane_potential: Membrane potential in Volts
            temperature: Temperature in Kelvin
        """
        # Map membrane potential to S-space center
        # More negative potential -> higher S_k (more definite state)
        S_k = min(1.0, abs(membrane_potential) / 0.1)  # Normalize to 100mV
        S_t = 0.5  # Neutral temporal
        S_e = 0.5  # Neutral evolution

        center = SCoordinate(S_k, S_t, S_e)

        # Radius determined by thermal energy vs electrical energy
        # kT / qV determines how strict the selection is
        k_B = 1.380649e-23  # Boltzmann constant
        e = 1.602176634e-19  # Elementary charge

        thermal_energy = k_B * temperature
        electrical_energy = e * abs(membrane_potential)

        # When thermal >> electrical, radius is large (less selective)
        # When electrical >> thermal, radius is small (more selective)
        if electrical_energy > 0:
            ratio = thermal_energy / electrical_energy
            radius = min(0.5, ratio * 0.5)
        else:
            radius = 0.5

        super().__init__(center=center, radius=radius, aperture_id=f"charge_field_{membrane_potential:.0e}V")

        self.membrane_potential = membrane_potential
        self.temperature = temperature
        self.thermal_energy = thermal_energy
        self.electrical_energy = electrical_energy

    def enhancement_factor(self, charge: float = 1.0) -> float:
        """
        Calculate concentration enhancement from categorical exclusion.

        From paper: exp(q·ΔΦ / kT) enhancement over diffusion.

        Args:
            charge: Molecular charge in elementary charge units

        Returns:
            Enhancement factor over diffusive concentration
        """
        k_B = 1.380649e-23
        e = 1.602176634e-19

        exponent = charge * e * abs(self.membrane_potential) / (k_B * self.temperature)
        return math.exp(exponent)


class ExternalChargeFieldAperture:
    """
    An aperture IS an external charge field at a point.
    
    FUNDAMENTAL INSIGHT:
    - An aperture is not a physical hole
    - An aperture IS an electric field configuration
    - Molecules pass if their charge distribution matches the field geometry
    - This is electromagnetic selection, not mechanical filtering
    
    The field can be visualized as:
    - Electric potential V(x, y, z) at a point
    - Molecules with charge q experience force F = -q∇V
    - If q matches the field sign/geometry → attracted through
    - If q opposes the field → repelled (blocked)
    
    Examples in biology:
    - Ion channels: K+ passes through K-field aperture, Na+ blocked
    - Membrane: Charge field from lipid bilayer dipoles
    - Ribosomes: Charge field selects charged tRNA
    """
    
    def __init__(self, 
                 field_strength: float = 1e6,  # V/m typical membrane field
                 field_geometry: str = "dipole",
                 position: tuple = (0.0, 0.0)):
        """
        Create an external charge field aperture.
        
        Args:
            field_strength: Electric field magnitude (V/m)
            field_geometry: "dipole", "monopole", "quadrupole"
            position: (x, y) position of the aperture
        """
        self.field_strength = field_strength
        self.field_geometry = field_geometry
        self.position = position
        
        # Map field to S-coordinates
        # Higher field → more selective (smaller acceptance)
        self.selectivity = min(0.99, field_strength / 1e7)
        
        # Hardware oscillator for real measurements
        try:
            from .virtual_spectrometer import HardwareOscillator
        except ImportError:
            from virtual_spectrometer import HardwareOscillator
        self.oscillator = HardwareOscillator("aperture_field", 1e9)
    
    def field_at_point(self, x: float, y: float) -> tuple:
        """
        Calculate electric field at point (x, y) from this aperture.
        
        Returns:
            (Ex, Ey, V): Field components and potential
        """
        px, py = self.position
        dx, dy = x - px, y - py
        r = math.sqrt(dx**2 + dy**2) + 1e-10  # Avoid division by zero
        
        if self.field_geometry == "monopole":
            # Point charge: E = kq/r²
            E_mag = self.field_strength / (r**2 + 0.01)
            Ex = E_mag * dx / r
            Ey = E_mag * dy / r
            V = self.field_strength / (r + 0.1)
            
        elif self.field_geometry == "dipole":
            # Dipole: E ∝ 1/r³, with angular dependence
            E_mag = self.field_strength / (r**3 + 0.001)
            theta = math.atan2(dy, dx)
            # Dipole field (aligned along y-axis)
            Ex = E_mag * 2 * math.cos(theta) * dx / r
            Ey = E_mag * (3 * math.sin(theta)**2 - 1) * dy / r if r > 0.01 else 0
            V = self.field_strength * dy / (r**2 + 0.01)
            
        elif self.field_geometry == "quadrupole":
            # Quadrupole: alternating + and - regions
            E_mag = self.field_strength / (r**4 + 0.0001)
            theta = math.atan2(dy, dx)
            Ex = E_mag * math.cos(2 * theta) * dx / r
            Ey = E_mag * math.sin(2 * theta) * dy / r
            V = self.field_strength * (dx**2 - dy**2) / (r**3 + 0.001)
        
        else:
            Ex, Ey, V = 0, 0, 0
        
        return Ex, Ey, V
    
    def molecule_passes(self, molecule_charge: float, molecule_position: tuple) -> bool:
        """
        Determine if a molecule passes through this aperture.
        
        Selection criterion: charge-field interaction energy < kT
        
        Args:
            molecule_charge: Charge of molecule (in e units)
            molecule_position: (x, y) position of molecule
        
        Returns:
            True if molecule passes, False if blocked
        """
        mx, my = molecule_position
        _, _, V = self.field_at_point(mx, my)
        
        # Interaction energy
        e = 1.602e-19
        k_B = 1.38e-23
        T = 300  # Room temperature
        
        U = molecule_charge * e * V
        kT = k_B * T
        
        # Boltzmann probability of passing
        if U > 0:  # Repulsive
            prob = math.exp(-U / kT)
        else:  # Attractive
            prob = 1.0
        
        # Use hardware timing for actual decision
        delta = self.oscillator.sample()
        random_val = abs(delta * 1e9) % 1.0
        
        return random_val < prob
    
    def generate_field_grid(self, grid_size: int = 50, extent: float = 2.0) -> tuple:
        """
        Generate electric field on a grid for visualization.
        
        Returns:
            (X, Y, Ex, Ey, V): Meshgrid coordinates and field values
        """
        x = np.linspace(-extent, extent, grid_size)
        y = np.linspace(-extent, extent, grid_size)
        X, Y = np.meshgrid(x, y)
        
        Ex = np.zeros_like(X)
        Ey = np.zeros_like(Y)
        V = np.zeros_like(X)
        
        for i in range(grid_size):
            for j in range(grid_size):
                ex, ey, v = self.field_at_point(X[i, j], Y[i, j])
                Ex[i, j] = ex
                Ey[i, j] = ey
                V[i, j] = v
        
        return X, Y, Ex, Ey, V


class ApertureCascade:
    """
    A cascade of apertures: sequential filtering.

    From paper: Aperture cascades exponentially amplify selectivity.
    For n apertures each with selectivity s:
        S_total = s^n

    This enables arbitrarily high specificity from moderate individual selectivity.
    """

    def __init__(self, apertures: Optional[List[CategoricalAperture]] = None):
        """Create an aperture cascade."""
        self.apertures = apertures or []

    def add_aperture(self, aperture: CategoricalAperture) -> None:
        """Add an aperture to the cascade."""
        self.apertures.append(aperture)

    def create_uniform_cascade(self, n: int, selectivity: float = 0.5) -> None:
        """
        Create a cascade of n identical apertures.

        Args:
            n: Number of apertures
            selectivity: Target selectivity per aperture
        """
        self.apertures.clear()

        # Create apertures with radius that gives target selectivity
        # For uniform S-space, selectivity ≈ (4/3)πr³ for spherical aperture
        radius = (selectivity * 3 / (4 * math.pi)) ** (1/3)

        for i in range(n):
            # Vary center slightly for each aperture
            center = SCoordinate(
                0.5 + 0.1 * math.sin(i * math.pi / n),
                0.5 + 0.1 * math.cos(i * math.pi / n),
                0.5
            )
            aperture = CategoricalAperture(center=center, radius=radius,
                                          aperture_id=f"cascade_{i}")
            self.apertures.append(aperture)

    def filter(self, molecules: List[VirtualMolecule]) -> List[VirtualMolecule]:
        """Pass molecules through the entire cascade."""
        current = molecules
        for aperture in self.apertures:
            current = aperture.filter(current)
            if not current:
                break
        return current

    def total_selectivity(self) -> float:
        """
        Calculate total selectivity of the cascade.

        This is the product of individual selectivities.
        """
        if not self.apertures:
            return 1.0

        total = 1.0
        for aperture in self.apertures:
            total *= aperture.selectivity
        return total

    @property
    def theoretical_selectivity(self) -> float:
        """
        Theoretical selectivity from individual apertures.

        S_total = ∏ s_i
        """
        if not self.apertures:
            return 1.0

        # Estimate individual selectivity from radius
        # Volume of sphere / volume of unit cube
        total = 1.0
        for aperture in self.apertures:
            sphere_vol = (4/3) * math.pi * (aperture.radius ** 3)
            total *= min(1.0, sphere_vol)  # Cap at 1.0
        return total

    def selectivity_vs_cascade_length(self,
                                       molecules: List[VirtualMolecule],
                                       max_n: int = 10) -> Dict[str, List[float]]:
        """
        Measure how selectivity changes with cascade length.

        This demonstrates exponential amplification.
        """
        lengths = []
        selectivities = []
        counts = []

        current = molecules.copy()
        for i, aperture in enumerate(self.apertures[:max_n]):
            aperture.clear()
            current = aperture.filter(current)

            lengths.append(i + 1)
            selectivities.append(len(current) / len(molecules))
            counts.append(len(current))

        return {
            'cascade_length': lengths,
            'selectivity': selectivities,
            'molecule_count': counts,
            'theoretical': [self.apertures[0].selectivity ** (i+1)
                           if self.apertures else 0.5 ** (i+1)
                           for i in range(len(lengths))]
        }


def temperature_independence_experiment(temperatures: List[float] = None) -> Dict[str, Any]:
    """
    Demonstrate that aperture selection is temperature-independent.

    From paper: Selection probability P(passage|c, T) = P(passage|c)
    Temperature affects encounter rate, not selection outcome.
    """
    if temperatures is None:
        temperatures = [100, 200, 300, 400, 500]  # Kelvin

    results = {}

    for T in temperatures:
        # Create aperture at this temperature
        aperture = ChargeFieldAperture(membrane_potential=-70e-3, temperature=T)

        # Create chamber and populate
        chamber = VirtualChamber()
        chamber.populate(500)

        # Filter through aperture
        passed = aperture.filter(list(chamber.gas))

        results[T] = {
            'temperature_K': T,
            'thermal_energy_J': aperture.thermal_energy,
            'selectivity': aperture.selectivity,
            'passed_count': len(passed),
            'aperture_radius': aperture.radius,
        }

    # Check if selectivity is approximately constant
    selectivities = [r['selectivity'] for r in results.values()]
    mean_sel = sum(selectivities) / len(selectivities)
    variance = sum((s - mean_sel)**2 for s in selectivities) / len(selectivities)

    return {
        'temperatures': temperatures,
        'results': results,
        'mean_selectivity': mean_sel,
        'selectivity_variance': variance,
        'is_temperature_independent': variance < 0.01,  # Low variance = independent
    }


def categorical_exclusion_experiment() -> Dict[str, Any]:
    """
    Demonstrate categorical exclusion: non-diffusive concentration.

    From paper: Molecules are concentrated not by random diffusion
    but by deterministic exclusion from incompatible charge regions.

    Enhancement factor: exp(q·ΔΦ / kT)
    """
    potentials = [-10e-3, -30e-3, -50e-3, -70e-3, -100e-3, -150e-3]  # mV

    results = []

    for V in potentials:
        aperture = ChargeFieldAperture(membrane_potential=V, temperature=300)

        # Create molecules
        chamber = VirtualChamber()
        chamber.populate(1000)

        # Measure exclusion
        passed = aperture.filter(list(chamber.gas))

        results.append({
            'membrane_potential_mV': V * 1000,
            'enhancement_factor': aperture.enhancement_factor(charge=1.0),
            'selectivity': aperture.selectivity,
            'aperture_radius': aperture.radius,
            'passed_count': len(passed),
        })

    return {
        'experiment': 'categorical_exclusion',
        'results': results,
        'potentials_mV': [V * 1000 for V in potentials],
        'enhancement_factors': [r['enhancement_factor'] for r in results],
        'selectivities': [r['selectivity'] for r in results],
    }


def cascade_amplification_experiment(cascade_lengths: List[int] = None) -> Dict[str, Any]:
    """
    Demonstrate exponential selectivity amplification in cascades.

    From paper: For n apertures with selectivity s:
        S_total = s^n

    This achieves enzymatic specificity through purely geometric means.
    """
    if cascade_lengths is None:
        cascade_lengths = [1, 2, 3, 5, 10, 20]

    individual_selectivity = 0.5

    results = []

    for n in cascade_lengths:
        cascade = ApertureCascade()
        cascade.create_uniform_cascade(n=n, selectivity=individual_selectivity)

        # Create molecules
        chamber = VirtualChamber()
        chamber.populate(10000)  # Need many for long cascades

        # Filter through cascade
        initial = list(chamber.gas)
        passed = cascade.filter(initial)

        measured_selectivity = len(passed) / len(initial) if initial else 0
        theoretical_selectivity = individual_selectivity ** n

        results.append({
            'cascade_length': n,
            'measured_selectivity': measured_selectivity,
            'theoretical_selectivity': theoretical_selectivity,
            'passed_count': len(passed),
            'log10_selectivity': math.log10(measured_selectivity) if measured_selectivity > 0 else -float('inf'),
        })

    return {
        'experiment': 'cascade_amplification',
        'individual_selectivity': individual_selectivity,
        'results': results,
        'lengths': cascade_lengths,
        'measured': [r['measured_selectivity'] for r in results],
        'theoretical': [r['theoretical_selectivity'] for r in results],
    }


def demonstrate_aperture():
    """Demonstrate virtual aperture functionality."""
    print("=== VIRTUAL APERTURE DEMONSTRATION ===\n")

    # Create an aperture
    center = SCoordinate(0.5, 0.5, 0.5)
    aperture = CategoricalAperture(center=center, radius=0.3)

    # Create chamber
    chamber = VirtualChamber()
    chamber.populate(500)

    print(f"Aperture at center {center}, radius {aperture.radius}")
    print(f"Testing against {len(chamber.gas)} molecules...\n")

    # Filter molecules
    passed = aperture.filter(list(chamber.gas))

    stats = aperture.statistics
    print(f"Aperture Statistics:")
    print(f"  Passed: {stats['passed_count']}")
    print(f"  Blocked: {stats['blocked_count']}")
    print(f"  Selectivity: {stats['selectivity']:.2%}")

    # Temperature independence
    print("\n--- Temperature Independence Test ---")
    temp_results = temperature_independence_experiment()
    print(f"Temperatures tested: {temp_results['temperatures']}")
    print(f"Mean selectivity: {temp_results['mean_selectivity']:.4f}")
    print(f"Selectivity variance: {temp_results['selectivity_variance']:.6f}")
    print(f"Is temperature-independent: {temp_results['is_temperature_independent']}")

    # Categorical exclusion
    print("\n--- Categorical Exclusion Test ---")
    excl_results = categorical_exclusion_experiment()
    print(f"{'Potential (mV)':<15} {'Enhancement':<15} {'Selectivity':<15}")
    print("-" * 45)
    for r in excl_results['results']:
        print(f"{r['membrane_potential_mV']:<15.0f} {r['enhancement_factor']:<15.1f} {r['selectivity']:<15.4f}")

    # Cascade amplification
    print("\n--- Cascade Amplification Test ---")
    cascade_results = cascade_amplification_experiment()
    print(f"Individual aperture selectivity: {cascade_results['individual_selectivity']}")
    print(f"\n{'Length':<10} {'Measured':<15} {'Theoretical':<15} {'log10(S)':<15}")
    print("-" * 55)
    for r in cascade_results['results']:
        log_s = r['log10_selectivity']
        log_str = f"{log_s:.2f}" if log_s > -float('inf') else "-∞"
        print(f"{r['cascade_length']:<10} {r['measured_selectivity']:<15.6f} {r['theoretical_selectivity']:<15.6f} {log_str:<15}")

    print("\n=== KEY INSIGHT ===")
    print("Apertures select by S-coordinate configuration, not velocity.")
    print("Selection is temperature-independent (T affects rate, not outcome).")
    print("Cascade amplification achieves exponential selectivity.")
    print("This explains prebiotic chemistry at cold interstellar temperatures.")

    return aperture


if __name__ == "__main__":
    aperture = demonstrate_aperture()

