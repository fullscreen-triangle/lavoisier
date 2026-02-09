"""
Ion-to-Droplet Thermodynamic Pixel Converter
==============================================

Implements the visual modality of the dual-graph system by converting
mass spectrometry ions into thermodynamic droplet impacts that encode:
- S-Entropy coordinates (S_knowledge, S_time, S_entropy)
- Phase-lock relationships
- Categorical state information
- Thermodynamic wave propagation

This creates the "visual graph" that intersects with the "numerical graph"
for dual-modality phase-lock signature annotation.

Based on:
- docs/oscillatory/tandem-mass-spec.tex
- docs/oscillatory/entropy-coordinates.tex
- docs/oscillatory/categorical-completion.tex
- precursor/src/utils/molecule_to-drip.py

Author: Kundai Sachikonye
"""

import numpy as np
from typing import Tuple, List, Dict, Optional, Any
from dataclasses import dataclass
import warnings

# Optional imports
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from scipy.ndimage import gaussian_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    # Fallback gaussian filter using numpy
    def gaussian_filter(image, sigma):
        """Simple gaussian blur fallback using numpy."""
        return image  # Return unchanged if scipy not available


@dataclass
class SEntropyCoordinates:
    """S-Entropy 3D coordinates for an ion."""
    s_knowledge: float  # Information content from intensity and m/z
    s_time: float       # Temporal/sequential ordering
    s_entropy: float    # Local entropy from intensity distribution


@dataclass
class DropletParameters:
    """Thermodynamic droplet parameters derived from S-Entropy."""
    velocity: float           # Impact velocity (relates to S_knowledge)
    radius: float            # Droplet radius (relates to S_entropy)
    surface_tension: float   # Surface tension (relates to S_time)
    impact_angle: float      # Impact angle in degrees
    temperature: float       # Thermodynamic temperature
    phase_coherence: float   # Phase-lock strength [0, 1]


@dataclass
class IonDroplet:
    """Complete ion-to-droplet transformation."""
    mz: float
    intensity: float
    s_entropy_coords: SEntropyCoordinates
    droplet_params: DropletParameters
    categorical_state: int  # Categorical completion state
    phase_lock_signature: Optional[np.ndarray] = None
    # Physics validation results
    physics_quality: float = 1.0  # Quality score [0, 1]
    is_physically_valid: bool = True
    validation_warnings: Optional[List[str]] = None


class SEntropyCalculator:
    """
    Calculate S-Entropy coordinates from ion properties.

    Implements the information-theoretic framework from tandem-mass-spec.tex:
    - S_knowledge: Information content (intensity, m/z precision)
    - S_time: Temporal coordination (RT, sequence position)
    - S_entropy: Distributional entropy (local intensity variation)
    """

    def __init__(self):
        # Physical constants
        self.k_B = 1.380649e-23  # Boltzmann constant
        self.T_room = 298.15      # Room temperature (K)

    def calculate_s_entropy(
        self,
        mz: float,
        intensity: float,
        rt: Optional[float] = None,
        local_intensities: Optional[np.ndarray] = None,
        mz_precision: float = 50e-6
    ) -> SEntropyCoordinates:
        """
        Calculate S-Entropy coordinates for an ion.

        Args:
            mz: Mass-to-charge ratio
            intensity: Ion intensity
            rt: Retention time (optional, for S_time)
            local_intensities: Neighboring intensities (for S_entropy)
            mz_precision: Mass precision (ppm)

        Returns:
            SEntropyCoordinates with (S_knowledge, S_time, S_entropy)
        """
        # S_knowledge: Information content from intensity and m/z
        # Higher intensity = more information
        # Higher m/z = more molecular complexity
        intensity_info = np.log1p(intensity) / np.log1p(1e10)  # Normalize to [0,1]
        mz_info = np.tanh(mz / 1000.0)  # Normalize m/z contribution
        precision_info = 1.0 / (1.0 + mz_precision * mz)  # Better precision = more info

        s_knowledge = (0.5 * intensity_info + 0.3 * mz_info + 0.2 * precision_info)
        s_knowledge = np.clip(s_knowledge, 0, 1)

        # S_time: Temporal coordination
        if rt is not None:
            # Normalize RT to [0, 1] assuming typical LC-MS run (0-60 min)
            s_time = np.clip(rt / 60.0, 0, 1)
        else:
            # Use m/z as proxy for fragmentation sequence
            # Smaller fragments appear "later" in conceptual time
            s_time = 1.0 - np.exp(-mz / 500.0)

        # S_entropy: Local distributional entropy
        if local_intensities is not None and len(local_intensities) > 1:
            # Shannon entropy of local intensity distribution
            intensities_norm = local_intensities / np.sum(local_intensities)
            intensities_norm = intensities_norm[intensities_norm > 0]  # Remove zeros
            shannon_entropy = -np.sum(intensities_norm * np.log2(intensities_norm))
            # Normalize by maximum possible entropy
            max_entropy = np.log2(len(intensities_norm))
            s_entropy = shannon_entropy / max_entropy if max_entropy > 0 else 0.5
        else:
            # Use intensity distribution as proxy
            # High intensity = low entropy (concentrated), low intensity = high entropy
            s_entropy = 1.0 - (intensity_info ** 0.5)

        s_entropy = np.clip(s_entropy, 0, 1)

        return SEntropyCoordinates(
            s_knowledge=float(s_knowledge),
            s_time=float(s_time),
            s_entropy=float(s_entropy)
        )


class DropletMapper:
    """
    Map S-Entropy coordinates to thermodynamic droplet parameters.

    Implements the physical transformation from information-theoretic
    coordinates to fluid dynamics parameters that encode molecular phase-lock.
    """

    def __init__(self):
        # Physical parameter ranges
        self.velocity_range = (1.0, 5.0)      # m/s
        self.radius_range = (0.3, 3.0)        # mm
        self.surface_tension_range = (0.02, 0.08)  # N/m
        self.temperature_range = (273.15, 373.15)  # K

    def map_to_droplet(
        self,
        s_coords: SEntropyCoordinates,
        intensity: float = 1.0
    ) -> DropletParameters:
        """
        Map S-Entropy coordinates to droplet parameters.

        The mapping encodes:
        - S_knowledge → velocity (higher info = faster impact)
        - S_entropy → radius (higher entropy = larger droplet)
        - S_time → surface_tension (temporal coherence)
        - Intensity → temperature (thermodynamic energy)

        Args:
            s_coords: S-Entropy coordinates
            intensity: Ion intensity (for temperature)

        Returns:
            DropletParameters with physical properties
        """
        # Velocity from S_knowledge
        velocity = (
            self.velocity_range[0] +
            s_coords.s_knowledge * (self.velocity_range[1] - self.velocity_range[0])
        )

        # Radius from S_entropy
        radius = (
            self.radius_range[0] +
            s_coords.s_entropy * (self.radius_range[1] - self.radius_range[0])
        )

        # Surface tension from S_time
        # Higher S_time = better temporal coherence = lower surface tension
        surface_tension = (
            self.surface_tension_range[1] -
            s_coords.s_time * (self.surface_tension_range[1] - self.surface_tension_range[0])
        )

        # Impact angle from S_knowledge and S_entropy interaction
        impact_angle = 45.0 * (s_coords.s_knowledge * s_coords.s_entropy)

        # Temperature from intensity (thermodynamic energy)
        intensity_norm = np.log1p(intensity) / np.log1p(1e10)
        temperature = (
            self.temperature_range[0] +
            intensity_norm * (self.temperature_range[1] - self.temperature_range[0])
        )

        # Phase coherence from coordinate product
        # High coherence when all coordinates are balanced
        phase_coherence = np.exp(-((s_coords.s_knowledge - 0.5)**2 +
                                   (s_coords.s_time - 0.5)**2 +
                                   (s_coords.s_entropy - 0.5)**2))

        return DropletParameters(
            velocity=float(velocity),
            radius=float(radius),
            surface_tension=float(surface_tension),
            impact_angle=float(impact_angle),
            temperature=float(temperature),
            phase_coherence=float(phase_coherence)
        )


class ThermodynamicWaveGenerator:
    """
    Generate thermodynamic wave patterns from droplet impacts.

    Implements wave propagation that encodes:
    - Droplet impact physics
    - Phase-lock relationships
    - Categorical state information
    - Temporal evolution
    """

    def __init__(self, resolution: Tuple[int, int] = (512, 512)):
        self.resolution = resolution

    def generate_wave_pattern(
        self,
        droplet: IonDroplet,
        center: Tuple[int, int],
        canvas: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Generate thermodynamic wave pattern from droplet impact.

        The wave pattern encodes:
        - Amplitude: Droplet velocity and intensity
        - Wavelength: Droplet radius and surface tension
        - Decay rate: Temperature and phase coherence
        - Directionality: Impact angle

        Args:
            droplet: IonDroplet with complete transformation
            center: Impact center coordinates (x, y)
            canvas: Existing canvas to add to (or create new)

        Returns:
            2D wave pattern with encoded thermodynamic information
        """
        if canvas is None:
            canvas = np.zeros(self.resolution, dtype=np.float32)

        # Create coordinate grid
        y, x = np.ogrid[:self.resolution[0], :self.resolution[1]]
        cx, cy = center

        # Distance from impact center
        distance = np.sqrt((x - cx)**2 + (y - cy)**2)

        # Thermodynamic parameters
        params = droplet.droplet_params

        # Wave amplitude from velocity and intensity
        amplitude = params.velocity * np.log1p(droplet.intensity) / 10.0

        # Wavelength from radius and surface tension
        # Higher surface tension = shorter wavelength (stiffer waves)
        wavelength = params.radius * (1.0 + params.surface_tension * 10.0)

        # Decay rate from temperature and phase coherence
        # Higher temperature = faster decay (more thermal noise)
        # Higher coherence = slower decay (more stable)
        decay_rate = (params.temperature / 373.15) / (params.phase_coherence + 0.1)

        # Generate concentric wave pattern
        wave = amplitude * np.exp(-distance / (params.radius * 30.0 * decay_rate))
        wave *= np.cos(2 * np.pi * distance / (wavelength * 5.0))

        # Apply directional bias from impact angle
        if params.impact_angle > 0:
            angle_rad = np.deg2rad(params.impact_angle)
            angle_grid = np.arctan2(y - cy, x - cx)
            directional_factor = 1.0 + 0.3 * np.cos(angle_grid - angle_rad)
            wave *= directional_factor

        # Encode categorical state as phase offset
        if droplet.categorical_state > 0:
            phase_offset = (droplet.categorical_state * np.pi / 10.0)
            wave *= np.cos(phase_offset)

        # Add to canvas
        canvas += wave

        return canvas

    def generate_spectrum_image(
        self,
        ion_droplets: List[IonDroplet],
        mz_range: Optional[Tuple[float, float]] = None
    ) -> np.ndarray:
        """
        Generate complete thermodynamic image from spectrum.

        Args:
            ion_droplets: List of transformed ions
            mz_range: Optional (mz_min, mz_max) for coordinate mapping

        Returns:
            2D thermodynamic image encoding all ions
        """
        canvas = np.zeros(self.resolution, dtype=np.float32)

        if not ion_droplets:
            return canvas

        # Determine m/z range
        if mz_range is None:
            mzs = [d.mz for d in ion_droplets]
            mz_range = (min(mzs), max(mzs))

        # Map each ion to impact position
        for droplet in ion_droplets:
            # x-position from m/z
            x = int(np.interp(
                droplet.mz,
                mz_range,
                [0, self.resolution[1] - 1]
            ))

            # y-position from S_time (temporal coordinate)
            y = int(droplet.s_entropy_coords.s_time * (self.resolution[0] - 1))

            # Generate wave pattern at this position
            canvas = self.generate_wave_pattern(droplet, (x, y), canvas)

        # Normalize to [0, 255]
        canvas = canvas - canvas.min()
        if canvas.max() > 0:
            canvas = 255.0 * canvas / canvas.max()

        return canvas.astype(np.uint8)


class IonToDropletConverter:
    """
    Complete ion-to-droplet thermodynamic pixel converter.

    This is the visual modality component of the dual-graph system.
    It transforms mass spectrometry ions into thermodynamic droplet impacts
    that encode phase-lock relationships and categorical states.

    Enhanced with physics validation to ensure transformations are physically plausible.
    """

    def __init__(
        self,
        resolution: Tuple[int, int] = (512, 512),
        enable_physics_validation: bool = True,
        validation_threshold: float = 0.3
    ):
        """
        Initialize converter.

        Args:
            resolution: Output image resolution (height, width)
            enable_physics_validation: Whether to validate physics
            validation_threshold: Minimum quality score to accept (0.0 - 1.0)
        """
        self.resolution = resolution
        self.s_entropy_calculator = SEntropyCalculator()
        self.droplet_mapper = DropletMapper()
        self.wave_generator = ThermodynamicWaveGenerator(resolution)

        # Physics validation
        self.enable_physics_validation = enable_physics_validation
        self.validation_threshold = validation_threshold
        if enable_physics_validation:
            from .PhysicsValidator import PhysicsValidator
            self.physics_validator = PhysicsValidator()
        else:
            self.physics_validator = None

        # Categorical state counter
        self.categorical_state_counter = 0

        # Statistics
        self.validation_stats = {
            'total_ions': 0,
            'valid_ions': 0,
            'filtered_ions': 0,
            'warnings_issued': 0
        }

    def convert_ion_to_droplet(
        self,
        mz: float,
        intensity: float,
        rt: Optional[float] = None,
        local_intensities: Optional[np.ndarray] = None,
        mz_precision: float = 50e-6,
        charge: int = 1
    ) -> Optional[IonDroplet]:
        """
        Convert single ion to thermodynamic droplet with physics validation.

        Args:
            mz: Mass-to-charge ratio
            intensity: Ion intensity
            rt: Retention time (optional)
            local_intensities: Neighboring intensities (for entropy calculation)
            mz_precision: Mass precision (ppm)
            charge: Charge state

        Returns:
            IonDroplet with complete transformation, or None if filtered
        """
        self.validation_stats['total_ions'] += 1

        # Calculate S-Entropy coordinates
        s_coords = self.s_entropy_calculator.calculate_s_entropy(
            mz=mz,
            intensity=intensity,
            rt=rt,
            local_intensities=local_intensities,
            mz_precision=mz_precision
        )

        # Map to droplet parameters
        droplet_params = self.droplet_mapper.map_to_droplet(s_coords, intensity)

        # Physics validation
        physics_quality = 1.0
        is_valid = True
        validation_warnings = []

        if self.enable_physics_validation and self.physics_validator is not None:
            # Comprehensive validation
            validation_results = self.physics_validator.comprehensive_validation(
                mz=mz,
                intensity=intensity,
                velocity=droplet_params.velocity,
                radius=droplet_params.radius,
                surface_tension=droplet_params.surface_tension,
                temperature=droplet_params.temperature,
                phase_coherence=droplet_params.phase_coherence,
                rt=rt,
                charge=charge
            )

            # Get overall quality and validity
            physics_quality, is_valid = self.physics_validator.get_overall_quality(
                validation_results
            )

            # Collect warnings from all validations
            for category, result in validation_results.items():
                validation_warnings.extend(result.warnings)
                if result.violations:
                    validation_warnings.extend([f"[{category}] {v}" for v in result.violations])

            # Update statistics
            if validation_warnings:
                self.validation_stats['warnings_issued'] += len(validation_warnings)

            # Filter low-quality conversions
            if physics_quality < self.validation_threshold:
                self.validation_stats['filtered_ions'] += 1
                if not warnings.filters:  # Only warn if warnings are enabled
                    warnings.warn(
                        f"Ion at m/z {mz:.2f} filtered: quality {physics_quality:.2f} "
                        f"< threshold {self.validation_threshold:.2f}",
                        UserWarning
                    )
                return None  # Filter out low-quality ion

        self.validation_stats['valid_ions'] += 1

        # Assign categorical state (increments with each unique ion)
        self.categorical_state_counter += 1
        categorical_state = self.categorical_state_counter

        return IonDroplet(
            mz=mz,
            intensity=intensity,
            s_entropy_coords=s_coords,
            droplet_params=droplet_params,
            categorical_state=categorical_state,
            physics_quality=physics_quality,
            is_physically_valid=is_valid,
            validation_warnings=validation_warnings if validation_warnings else None
        )

    def convert_spectrum_to_image(
        self,
        mzs: np.ndarray,
        intensities: np.ndarray,
        rt: Optional[float] = None,
        mz_range: Optional[Tuple[float, float]] = None,
        normalize: bool = True
    ) -> Tuple[np.ndarray, List[IonDroplet]]:
        """
        Convert complete spectrum to thermodynamic droplet image.

        Args:
            mzs: Array of m/z values
            intensities: Array of intensity values
            rt: Retention time (optional)
            mz_range: Optional (mz_min, mz_max) for visualization range
            normalize: Whether to normalize intensities

        Returns:
            Tuple of (thermodynamic_image, ion_droplets)
        """
        if len(mzs) != len(intensities):
            raise ValueError("mzs and intensities must have same length")

        if len(mzs) == 0:
            return np.zeros(self.resolution, dtype=np.uint8), []

        # Normalize intensities if requested
        if normalize:
            intensities = intensities / np.max(intensities)

        # Reset categorical state counter for new spectrum
        self.categorical_state_counter = 0

        # Convert each ion to droplet
        ion_droplets = []
        for i, (mz, intensity) in enumerate(zip(mzs, intensities)):
            # Get local intensities for entropy calculation
            window_size = 5
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(intensities), i + window_size // 2 + 1)
            local_intensities = intensities[start_idx:end_idx]

            droplet = self.convert_ion_to_droplet(
                mz=float(mz),
                intensity=float(intensity),
                rt=rt,
                local_intensities=local_intensities
            )

            # Only add if not filtered by physics validation
            if droplet is not None:
                ion_droplets.append(droplet)

        # Generate thermodynamic image
        image = self.wave_generator.generate_spectrum_image(
            ion_droplets,
            mz_range=mz_range
        )

        return image, ion_droplets

    def extract_phase_lock_features(
        self,
        image: np.ndarray,
        ion_droplets: List[IonDroplet]
    ) -> np.ndarray:
        """
        Extract phase-lock signature features from thermodynamic image.

        This creates the "visual graph" features for dual-modality intersection.

        Args:
            image: Thermodynamic droplet image
            ion_droplets: List of ion droplets

        Returns:
            Feature vector encoding phase-lock relationships
        """
        features = []

        # 1. Image-based features
        # Statistical features
        features.extend([
            np.mean(image),
            np.std(image),
            np.median(image),
            np.percentile(image, 25),
            np.percentile(image, 75)
        ])

        # 2. Frequency domain features (wave patterns)
        fft = np.fft.fft2(image)
        fft_magnitude = np.abs(fft)
        features.extend([
            np.mean(fft_magnitude),
            np.std(fft_magnitude),
            np.max(fft_magnitude)
        ])

        # 3. Droplet-based features (thermodynamic properties)
        if ion_droplets:
            velocities = [d.droplet_params.velocity for d in ion_droplets]
            radii = [d.droplet_params.radius for d in ion_droplets]
            surface_tensions = [d.droplet_params.surface_tension for d in ion_droplets]
            phase_coherences = [d.droplet_params.phase_coherence for d in ion_droplets]

            features.extend([
                np.mean(velocities),
                np.std(velocities),
                np.mean(radii),
                np.std(radii),
                np.mean(surface_tensions),
                np.mean(phase_coherences)
            ])
        else:
            features.extend([0, 0, 0, 0, 0, 0])

        # 4. S-Entropy coordinate features
        if ion_droplets:
            s_knowledge_vals = [d.s_entropy_coords.s_knowledge for d in ion_droplets]
            s_time_vals = [d.s_entropy_coords.s_time for d in ion_droplets]
            s_entropy_vals = [d.s_entropy_coords.s_entropy for d in ion_droplets]

            features.extend([
                np.mean(s_knowledge_vals),
                np.std(s_knowledge_vals),
                np.mean(s_time_vals),
                np.std(s_time_vals),
                np.mean(s_entropy_vals),
                np.std(s_entropy_vals)
            ])
        else:
            features.extend([0, 0, 0, 0, 0, 0])

        # Total: 23 features
        return np.array(features, dtype=np.float32)

    def get_droplet_summary(self, ion_droplets: List[IonDroplet]) -> Dict[str, Any]:
        """Get summary statistics of droplet transformation including physics validation."""
        if not ion_droplets:
            return {'num_ions': 0}

        summary = {
            'num_ions': len(ion_droplets),
            'mz_range': (min(d.mz for d in ion_droplets), max(d.mz for d in ion_droplets)),
            'intensity_range': (min(d.intensity for d in ion_droplets), max(d.intensity for d in ion_droplets)),
            's_entropy_coords': {
                's_knowledge_mean': np.mean([d.s_entropy_coords.s_knowledge for d in ion_droplets]),
                's_time_mean': np.mean([d.s_entropy_coords.s_time for d in ion_droplets]),
                's_entropy_mean': np.mean([d.s_entropy_coords.s_entropy for d in ion_droplets]),
            },
            'droplet_params': {
                'velocity_mean': np.mean([d.droplet_params.velocity for d in ion_droplets]),
                'radius_mean': np.mean([d.droplet_params.radius for d in ion_droplets]),
                'surface_tension_mean': np.mean([d.droplet_params.surface_tension for d in ion_droplets]),
                'phase_coherence_mean': np.mean([d.droplet_params.phase_coherence for d in ion_droplets]),
            },
            'categorical_states': list(range(1, len(ion_droplets) + 1))
        }

        # Add physics validation statistics
        if self.enable_physics_validation:
            summary['physics_validation'] = {
                'quality_mean': np.mean([d.physics_quality for d in ion_droplets]),
                'quality_min': min(d.physics_quality for d in ion_droplets),
                'quality_max': max(d.physics_quality for d in ion_droplets),
                'num_valid': sum(1 for d in ion_droplets if d.is_physically_valid),
                'num_with_warnings': sum(1 for d in ion_droplets if d.validation_warnings),
                'validation_stats': dict(self.validation_stats)
            }

        return summary

    def get_validation_report(self) -> str:
        """Get human-readable validation report."""
        stats = self.validation_stats
        total = stats['total_ions']

        if total == 0:
            return "No ions processed yet."

        report = f"""
Physics Validation Report
=========================
Total ions processed: {total}
Valid ions: {stats['valid_ions']} ({100*stats['valid_ions']/total:.1f}%)
Filtered ions: {stats['filtered_ions']} ({100*stats['filtered_ions']/total:.1f}%)
Total warnings: {stats['warnings_issued']}

Threshold: {self.validation_threshold:.2f}
Validation enabled: {self.enable_physics_validation}
"""
        return report
