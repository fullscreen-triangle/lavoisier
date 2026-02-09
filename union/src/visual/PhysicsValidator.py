"""
Physics-Based Validation for Ion-to-Droplet Conversion
======================================================

Implements physics verification inspired by high-speed movement detection
to ensure ion-to-droplet transformations are physically plausible.

Validates:
- Ion flight time consistency (TOF principles)
- Energy conservation in droplet formation
- Thermodynamic parameter bounds (Weber number, Reynolds number)
- Signal detection plausibility (instrument response, dynamic range)
- Trajectory feasibility (velocity, acceleration constraints)

This adds a quality score to each conversion and filters spurious signals.

Inspired by: github.com/fullscreen-triangle/vibrio (high-speed movement detection)

Author: Kundai Chinyamakobvu
"""

import numpy as np
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
import warnings


@dataclass
class PhysicsConstraints:
    """Physical constraints for MS and droplet formation."""
    # Ion flight constraints
    max_ion_velocity: float = 1e6  # m/s (relativistic limit check)
    min_flight_time: float = 1e-6  # seconds (instrument response time)
    max_flight_time: float = 1.0   # seconds (typical TOF-MS range)

    # Droplet formation constraints
    min_droplet_velocity: float = 0.1  # m/s
    max_droplet_velocity: float = 10.0  # m/s (supersonic limit)
    min_droplet_radius: float = 0.05  # mm (spray instability limit)
    max_droplet_radius: float = 5.0   # mm (breakup limit)

    # Surface tension bounds (water-like liquids)
    min_surface_tension: float = 0.01  # N/m
    max_surface_tension: float = 0.1   # N/m

    # Temperature bounds
    min_temperature: float = 200.0  # K (cryogenic lower bound)
    max_temperature: float = 500.0  # K (thermal decomposition upper bound)

    # Signal detection constraints
    min_detectable_intensity: float = 1e2   # Minimum signal-to-noise
    max_saturation_intensity: float = 1e10  # Detector saturation

    # Mass spectrometry constraints
    min_mz: float = 10.0     # m/z lower limit
    max_mz: float = 10000.0  # m/z upper limit
    charge_states: List[int] = None  # Allowed charge states

    def __post_init__(self):
        if self.charge_states is None:
            self.charge_states = list(range(1, 6))  # Default: +1 to +5


@dataclass
class PhysicsValidationResult:
    """Result of physics validation."""
    is_valid: bool
    quality_score: float  # 0.0 to 1.0
    violations: List[str]
    warnings: List[str]
    metrics: Dict[str, float]


class PhysicsValidator:
    """
    Validates ion-to-droplet conversions using physics constraints.

    Inspired by high-speed movement detection: checks if particles
    can physically reach the detector given their properties.
    """

    def __init__(self, constraints: Optional[PhysicsConstraints] = None):
        """
        Initialize physics validator.

        Args:
            constraints: Physical constraints (uses defaults if None)
        """
        self.constraints = constraints or PhysicsConstraints()

        # Physical constants
        self.c = 299792458  # Speed of light (m/s)
        self.m_proton = 1.672621898e-27  # Proton mass (kg)
        self.e = 1.602176634e-19  # Elementary charge (C)
        self.k_B = 1.380649e-23  # Boltzmann constant (J/K)
        self.N_A = 6.02214076e23  # Avogadro constant

        # Fluid dynamics constants
        self.rho_water = 1000  # Water density (kg/m³)
        self.mu_water = 1e-3   # Water dynamic viscosity (Pa·s)
        self.g = 9.81          # Gravitational acceleration (m/s²)

    def validate_ion_properties(
        self,
        mz: float,
        intensity: float,
        rt: Optional[float] = None,
        charge: int = 1
    ) -> PhysicsValidationResult:
        """
        Validate basic ion properties before conversion.

        Args:
            mz: Mass-to-charge ratio
            intensity: Ion intensity
            rt: Retention time (optional)
            charge: Charge state

        Returns:
            PhysicsValidationResult with validation outcome
        """
        violations = []
        warnings_list = []
        metrics = {}

        # 1. Mass-to-charge ratio bounds
        if mz < self.constraints.min_mz:
            violations.append(f"m/z ({mz:.2f}) below minimum ({self.constraints.min_mz})")
        elif mz > self.constraints.max_mz:
            violations.append(f"m/z ({mz:.2f}) above maximum ({self.constraints.max_mz})")
        metrics['mz'] = mz

        # 2. Intensity bounds
        if intensity < self.constraints.min_detectable_intensity:
            violations.append(f"Intensity ({intensity:.2e}) below detection limit")
        elif intensity > self.constraints.max_saturation_intensity:
            warnings_list.append(f"Intensity ({intensity:.2e}) may saturate detector")
        metrics['intensity'] = intensity

        # 3. Charge state validation
        if charge not in self.constraints.charge_states:
            warnings_list.append(f"Unusual charge state: {charge}")
        metrics['charge'] = charge

        # 4. Calculate ion mass
        ion_mass = mz * charge * self.m_proton  # Approximate mass in kg
        metrics['ion_mass_kg'] = ion_mass

        # 5. Check if mass is physically reasonable
        if ion_mass < 10 * self.m_proton:  # Less than 10 Da
            warnings_list.append(f"Very low mass: {ion_mass/self.m_proton:.1f} Da")
        elif ion_mass > 1e6 * self.m_proton:  # Greater than 1 MDa
            warnings_list.append(f"Very high mass: {ion_mass/self.m_proton:.1f} Da")

        # 6. Retention time validation (if provided)
        if rt is not None:
            if rt < 0:
                violations.append(f"Negative retention time: {rt}")
            elif rt > 200:  # Typical LC-MS run < 200 min
                warnings_list.append(f"Unusually long retention time: {rt:.1f} min")
            metrics['retention_time_min'] = rt

        # Calculate quality score
        quality_score = self._calculate_quality_score(
            violations, warnings_list, metrics
        )

        return PhysicsValidationResult(
            is_valid=len(violations) == 0,
            quality_score=quality_score,
            violations=violations,
            warnings=warnings_list,
            metrics=metrics
        )

    def validate_droplet_parameters(
        self,
        velocity: float,
        radius: float,
        surface_tension: float,
        temperature: float,
        phase_coherence: float
    ) -> PhysicsValidationResult:
        """
        Validate droplet parameters using fluid dynamics principles.

        Checks:
        - Weber number (inertial vs surface tension forces)
        - Reynolds number (inertial vs viscous forces)
        - Capillary number (viscous vs surface tension)
        - Thermal energy consistency

        Args:
            velocity: Droplet velocity (m/s)
            radius: Droplet radius (mm)
            surface_tension: Surface tension (N/m)
            temperature: Temperature (K)
            phase_coherence: Phase coherence [0, 1]

        Returns:
            PhysicsValidationResult
        """
        violations = []
        warnings_list = []
        metrics = {}

        # Convert radius to meters
        radius_m = radius * 1e-3

        # 1. Velocity bounds
        if velocity < self.constraints.min_droplet_velocity:
            violations.append(f"Velocity ({velocity:.2f} m/s) too low")
        elif velocity > self.constraints.max_droplet_velocity:
            violations.append(f"Velocity ({velocity:.2f} m/s) exceeds limit")
        metrics['velocity_ms'] = velocity

        # 2. Radius bounds
        if radius < self.constraints.min_droplet_radius:
            violations.append(f"Radius ({radius:.3f} mm) below stability limit")
        elif radius > self.constraints.max_droplet_radius:
            violations.append(f"Radius ({radius:.3f} mm) exceeds breakup limit")
        metrics['radius_mm'] = radius

        # 3. Surface tension bounds
        if surface_tension < self.constraints.min_surface_tension:
            violations.append(f"Surface tension ({surface_tension:.4f} N/m) too low")
        elif surface_tension > self.constraints.max_surface_tension:
            violations.append(f"Surface tension ({surface_tension:.4f} N/m) too high")
        metrics['surface_tension_Nm'] = surface_tension

        # 4. Temperature bounds
        if temperature < self.constraints.min_temperature:
            violations.append(f"Temperature ({temperature:.1f} K) below physical limit")
        elif temperature > self.constraints.max_temperature:
            violations.append(f"Temperature ({temperature:.1f} K) above stability limit")
        metrics['temperature_K'] = temperature

        # 5. Weber number (We = ρ v² d / σ)
        # Measures ratio of inertial to surface tension forces
        weber_number = (self.rho_water * velocity**2 * 2*radius_m) / surface_tension
        metrics['weber_number'] = weber_number

        if weber_number < 1:
            # Surface tension dominant - stable spherical droplet
            pass
        elif 1 <= weber_number < 12:
            # Transition regime - slight deformation
            pass
        elif weber_number >= 12:
            # Droplet breakup likely
            warnings_list.append(f"High Weber number ({weber_number:.1f}) - breakup likely")

        # 6. Reynolds number (Re = ρ v d / μ)
        # Measures ratio of inertial to viscous forces
        reynolds_number = (self.rho_water * velocity * 2*radius_m) / self.mu_water
        metrics['reynolds_number'] = reynolds_number

        if reynolds_number < 1:
            # Viscous flow (Stokes regime)
            pass
        elif 1 <= reynolds_number < 1000:
            # Transitional flow
            pass
        elif reynolds_number >= 1000:
            # Turbulent flow
            warnings_list.append(f"High Reynolds number ({reynolds_number:.1f}) - turbulent")

        # 7. Capillary number (Ca = μ v / σ)
        # Measures ratio of viscous to surface tension forces
        capillary_number = (self.mu_water * velocity) / surface_tension
        metrics['capillary_number'] = capillary_number

        if capillary_number > 1:
            warnings_list.append(f"High Capillary number ({capillary_number:.3f}) - viscous dominant")

        # 8. Bond number (Bo = ρ g L² / σ)
        # Measures gravity vs surface tension (for vertical motion)
        bond_number = (self.rho_water * self.g * (2*radius_m)**2) / surface_tension
        metrics['bond_number'] = bond_number

        if bond_number > 1:
            warnings_list.append(f"Gravity effects significant (Bo = {bond_number:.3f})")

        # 9. Thermal energy consistency
        # Check if temperature is consistent with kinetic energy
        droplet_mass = (4/3) * np.pi * radius_m**3 * self.rho_water
        kinetic_energy = 0.5 * droplet_mass * velocity**2
        thermal_energy = 1.5 * self.k_B * temperature  # Per molecule

        # Number of molecules in droplet (approximate)
        n_molecules = droplet_mass / (18e-3 / self.N_A)  # Water molecules
        total_thermal_energy = thermal_energy * n_molecules

        energy_ratio = kinetic_energy / total_thermal_energy
        metrics['kinetic_thermal_ratio'] = energy_ratio

        if energy_ratio > 1e3:
            warnings_list.append(f"Kinetic energy >> thermal energy (ratio = {energy_ratio:.1e})")
        elif energy_ratio < 1e-3:
            warnings_list.append(f"Thermal energy >> kinetic energy (ratio = {energy_ratio:.1e})")

        # 10. Phase coherence validation
        if not (0 <= phase_coherence <= 1):
            violations.append(f"Phase coherence ({phase_coherence:.2f}) out of range [0,1]")
        metrics['phase_coherence'] = phase_coherence

        # Phase coherence should correlate with temperature
        # Higher temp = more thermal noise = lower coherence
        expected_coherence = np.exp(-(temperature - self.constraints.min_temperature) /
                                    (self.constraints.max_temperature - self.constraints.min_temperature))
        coherence_deviation = abs(phase_coherence - expected_coherence)

        if coherence_deviation > 0.5:
            warnings_list.append(f"Phase coherence inconsistent with temperature")
        metrics['expected_coherence'] = expected_coherence
        metrics['coherence_deviation'] = coherence_deviation

        # Calculate quality score
        quality_score = self._calculate_quality_score(
            violations, warnings_list, metrics
        )

        return PhysicsValidationResult(
            is_valid=len(violations) == 0,
            quality_score=quality_score,
            violations=violations,
            warnings=warnings_list,
            metrics=metrics
        )

    def validate_ion_flight_time(
        self,
        mz: float,
        flight_distance: float = 1.0,  # meters
        accelerating_voltage: float = 20000.0,  # volts
        charge: int = 1
    ) -> PhysicsValidationResult:
        """
        Validate ion flight time using TOF-MS principles.

        Similar to high-speed movement detection: can the particle
        reach the detector in the observed time?

        Args:
            mz: Mass-to-charge ratio
            flight_distance: Distance to detector (m)
            accelerating_voltage: Acceleration voltage (V)
            charge: Charge state

        Returns:
            PhysicsValidationResult
        """
        violations = []
        warnings_list = []
        metrics = {}

        # Calculate ion mass
        ion_mass = mz * charge * self.m_proton  # kg

        # Calculate ion kinetic energy after acceleration
        kinetic_energy = charge * self.e * accelerating_voltage  # Joules

        # Calculate ion velocity (non-relativistic)
        velocity = np.sqrt(2 * kinetic_energy / ion_mass)  # m/s

        # Check if relativistic effects are significant (v > 0.1c)
        if velocity > 0.1 * self.c:
            warnings_list.append(f"Relativistic effects may be significant (v = {velocity/self.c:.3f}c)")
            # Apply relativistic correction
            gamma = 1 / np.sqrt(1 - (velocity/self.c)**2)
            velocity = velocity / gamma

        metrics['ion_velocity_ms'] = velocity
        metrics['ion_velocity_c'] = velocity / self.c

        # Calculate flight time
        flight_time = flight_distance / velocity  # seconds
        metrics['flight_time_s'] = flight_time
        metrics['flight_time_us'] = flight_time * 1e6

        # Validate flight time
        if flight_time < self.constraints.min_flight_time:
            violations.append(f"Flight time ({flight_time*1e6:.2f} μs) below instrument response")
        elif flight_time > self.constraints.max_flight_time:
            violations.append(f"Flight time ({flight_time:.4f} s) exceeds typical range")

        # Check velocity bounds
        if velocity > self.constraints.max_ion_velocity:
            violations.append(f"Ion velocity ({velocity:.2e} m/s) exceeds physical limit")

        # Calculate momentum
        momentum = ion_mass * velocity
        metrics['momentum_kg_ms'] = momentum

        # Calculate de Broglie wavelength (quantum mechanical)
        h = 6.62607015e-34  # Planck constant
        wavelength = h / momentum
        metrics['de_broglie_wavelength_m'] = wavelength

        if wavelength > 1e-10:  # > 0.1 nm
            warnings_list.append(f"Quantum effects may be significant (λ = {wavelength*1e9:.2f} nm)")

        # Calculate quality score
        quality_score = self._calculate_quality_score(
            violations, warnings_list, metrics
        )

        return PhysicsValidationResult(
            is_valid=len(violations) == 0,
            quality_score=quality_score,
            violations=violations,
            warnings=warnings_list,
            metrics=metrics
        )

    def validate_energy_conservation(
        self,
        ion_mass: float,  # kg
        ion_velocity: float,  # m/s
        droplet_velocity: float,  # m/s
        droplet_radius: float,  # mm
        surface_tension: float,  # N/m
        temperature: float  # K
    ) -> PhysicsValidationResult:
        """
        Validate energy conservation in ion-to-droplet transformation.

        Ensures total energy is conserved across the transformation.

        Args:
            ion_mass: Ion mass (kg)
            ion_velocity: Ion velocity (m/s)
            droplet_velocity: Droplet velocity (m/s)
            droplet_radius: Droplet radius (mm)
            surface_tension: Surface tension (N/m)
            temperature: Temperature (K)

        Returns:
            PhysicsValidationResult
        """
        violations = []
        warnings_list = []
        metrics = {}

        radius_m = droplet_radius * 1e-3

        # Ion kinetic energy
        ion_kinetic = 0.5 * ion_mass * ion_velocity**2
        metrics['ion_kinetic_energy_J'] = ion_kinetic

        # Droplet mass (assume water density)
        droplet_mass = (4/3) * np.pi * radius_m**3 * self.rho_water
        metrics['droplet_mass_kg'] = droplet_mass

        # Droplet kinetic energy
        droplet_kinetic = 0.5 * droplet_mass * droplet_velocity**2
        metrics['droplet_kinetic_energy_J'] = droplet_kinetic

        # Droplet surface energy
        surface_area = 4 * np.pi * radius_m**2
        surface_energy = surface_tension * surface_area
        metrics['surface_energy_J'] = surface_energy

        # Thermal energy
        n_molecules = droplet_mass / (18e-3 / self.N_A)
        thermal_energy = 1.5 * self.k_B * temperature * n_molecules
        metrics['thermal_energy_J'] = thermal_energy

        # Total energy
        total_initial = ion_kinetic
        total_final = droplet_kinetic + surface_energy + thermal_energy

        metrics['total_initial_energy_J'] = total_initial
        metrics['total_final_energy_J'] = total_final

        # Energy difference
        energy_diff = abs(total_final - total_initial)
        energy_ratio = energy_diff / max(total_initial, 1e-20)

        metrics['energy_difference_J'] = energy_diff
        metrics['energy_ratio'] = energy_ratio

        # Validate energy conservation (allow 10% discrepancy for dissipation)
        if energy_ratio > 0.1:
            warnings_list.append(f"Energy not conserved: {energy_ratio*100:.1f}% difference")

        # Check if energy is reasonable
        if total_final > total_initial * 10:
            violations.append(f"Final energy > 10× initial (non-physical)")

        # Calculate quality score
        quality_score = self._calculate_quality_score(
            violations, warnings_list, metrics
        )

        return PhysicsValidationResult(
            is_valid=len(violations) == 0,
            quality_score=quality_score,
            violations=violations,
            warnings=warnings_list,
            metrics=metrics
        )

    def _calculate_quality_score(
        self,
        violations: List[str],
        warnings: List[str],
        metrics: Dict[str, float]
    ) -> float:
        """
        Calculate overall quality score from validation results.

        Args:
            violations: List of hard violations
            warnings: List of soft warnings
            metrics: Validation metrics

        Returns:
            Quality score in [0, 1]
        """
        if violations:
            return 0.0  # Hard failures get zero quality

        # Start with perfect score
        score = 1.0

        # Deduct for each warning
        score -= 0.1 * len(warnings)

        # Bonus for good metrics (if available)
        if 'weber_number' in metrics:
            # Ideal Weber number around 5-10
            we = metrics['weber_number']
            if 5 <= we <= 10:
                score += 0.05

        if 'reynolds_number' in metrics:
            # Moderate Reynolds number is good
            re = metrics['reynolds_number']
            if 10 <= re <= 100:
                score += 0.05

        if 'phase_coherence' in metrics:
            # High phase coherence is good
            coherence = metrics['phase_coherence']
            score += 0.1 * coherence

        return np.clip(score, 0.0, 1.0)

    def comprehensive_validation(
        self,
        mz: float,
        intensity: float,
        velocity: float,
        radius: float,
        surface_tension: float,
        temperature: float,
        phase_coherence: float,
        rt: Optional[float] = None,
        charge: int = 1
    ) -> Dict[str, PhysicsValidationResult]:
        """
        Perform comprehensive physics validation.

        Args:
            mz: Mass-to-charge ratio
            intensity: Ion intensity
            velocity: Droplet velocity
            radius: Droplet radius
            surface_tension: Surface tension
            temperature: Temperature
            phase_coherence: Phase coherence
            rt: Retention time (optional)
            charge: Charge state

        Returns:
            Dictionary of validation results by category
        """
        results = {}

        # 1. Validate ion properties
        results['ion'] = self.validate_ion_properties(mz, intensity, rt, charge)

        # 2. Validate droplet parameters
        results['droplet'] = self.validate_droplet_parameters(
            velocity, radius, surface_tension, temperature, phase_coherence
        )

        # 3. Validate flight time
        results['flight'] = self.validate_ion_flight_time(mz, charge=charge)

        # 4. Validate energy conservation
        ion_mass = mz * charge * self.m_proton
        ion_velocity = 1e5  # Approximate (would calculate from TOF in practice)
        results['energy'] = self.validate_energy_conservation(
            ion_mass, ion_velocity, velocity, radius, surface_tension, temperature
        )

        return results

    def get_overall_quality(
        self,
        validation_results: Dict[str, PhysicsValidationResult]
    ) -> Tuple[float, bool]:
        """
        Get overall quality score and validity from multiple validations.

        Args:
            validation_results: Dictionary of validation results

        Returns:
            Tuple of (overall_quality_score, is_valid)
        """
        quality_scores = [r.quality_score for r in validation_results.values()]
        is_valid = all(r.is_valid for r in validation_results.values())

        # Weighted average (can adjust weights)
        weights = {'ion': 0.3, 'droplet': 0.3, 'flight': 0.2, 'energy': 0.2}
        overall_quality = sum(
            weights.get(k, 1.0) * r.quality_score
            for k, r in validation_results.items()
        ) / sum(weights.values())

        return overall_quality, is_valid
