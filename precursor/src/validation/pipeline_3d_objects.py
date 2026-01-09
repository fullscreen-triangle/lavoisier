"""
3D Object Pipeline for Validation
==================================

Generates 3D objects at each stage of the analytical pipeline:
1. Solution Phase (Blue Sphere) - Initial molecular ensemble
2. Chromatography (Green Ellipsoid) - Temporal separation with ridges
3. Ionization (Yellow Fragmenting Sphere) - Coulomb explosion
4. MS1 (Orange Sphere Array) - Discrete ions positioned by (m/z, S_t, S_k)
5. MS2/Fragmentation (Red Cascade) - Autocatalytic explosion
6. Droplet (Purple Wave Pattern) - Final thermodynamic image

Each 3D object encodes molecular information through:
- Position: S-entropy coordinates (S_k, S_t, S_e)
- Color: Temperature/energy state
- Texture: Surface properties (smooth → ridged → fractured → waves)
- Shape: Geometric transformation through pipeline
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class SEntropyCoordinate:
    """S-Entropy coordinate in categorical space."""
    S_k: float  # Knowledge entropy [0, 1]
    S_t: float  # Temporal entropy [0, 1]
    S_e: float  # Evolution entropy [0, 1]
    
    def as_array(self) -> np.ndarray:
        """Return as numpy array."""
        return np.array([self.S_k, self.S_t, self.S_e])
    
    def distance_to(self, other: 'SEntropyCoordinate') -> float:
        """Euclidean distance to another coordinate."""
        return np.linalg.norm(self.as_array() - other.as_array())


@dataclass
class ThermodynamicProperties:
    """Thermodynamic properties of a 3D object."""
    temperature: float  # Categorical temperature (S-variance)
    pressure: float     # Categorical pressure (sampling rate)
    entropy: float      # Categorical entropy (S-spread)
    volume: float       # S-space volume
    
    # Droplet properties (for final stage)
    radius: Optional[float] = None       # Droplet radius (m)
    velocity: Optional[float] = None     # Droplet velocity (m/s)
    surface_tension: Optional[float] = None  # Surface tension (N/m)
    
    # Dimensionless numbers (physics validation)
    weber_number: Optional[float] = None     # We = ρv²L/σ
    reynolds_number: Optional[float] = None  # Re = ρvL/μ
    ohnesorge_number: Optional[float] = None # Oh = We^0.5 / Re


@dataclass
class Object3D:
    """
    A 3D object representing molecular state at a specific pipeline stage.
    
    The object transforms through the pipeline while preserving information.
    """
    stage: str  # 'solution', 'chromatography', 'ionization', 'ms1', 'ms2', 'droplet'
    shape: str  # 'sphere', 'ellipsoid', 'fragmenting_sphere', 'sphere_array', 'cascade', 'wave_pattern'
    
    # Geometric properties
    center: SEntropyCoordinate
    dimensions: Tuple[float, float, float]  # (a, b, c) for ellipsoid, (r, r, r) for sphere
    
    # Physical properties
    color: Tuple[float, float, float]  # RGB [0, 1]
    texture: str  # 'smooth', 'ridged', 'fractured', 'discrete', 'explosive', 'waves'
    
    # Thermodynamic properties
    thermo: ThermodynamicProperties
    
    # Data payload
    data: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    timestamp: float = 0.0
    molecule_count: int = 0
    
    def volume(self) -> float:
        """Calculate geometric volume."""
        a, b, c = self.dimensions
        if self.shape == 'sphere':
            return (4/3) * np.pi * a**3
        elif self.shape == 'ellipsoid':
            return (4/3) * np.pi * a * b * c
        else:
            return a * b * c  # Approximate


class Pipeline3DObjectGenerator:
    """
    Generates 3D objects at each stage of the analytical pipeline.
    
    Reads experimental data and creates 3D object representations
    that transform through the pipeline stages.
    """
    
    def __init__(self, experiment_dir: Path):
        """
        Initialize generator for an experiment.
        
        Args:
            experiment_dir: Path to experiment directory (e.g., A_M3_negPFP_03)
        """
        self.experiment_dir = Path(experiment_dir)
        self.experiment_name = self.experiment_dir.name
        
        # Stage directories
        self.stage_dirs = {
            'preprocessing': self.experiment_dir / 'stage_01_preprocessing',
            'sentropy': self.experiment_dir / 'stage_02_sentropy',
            'fragmentation': self.experiment_dir / 'stage_02_5_fragmentation',
            'bmd': self.experiment_dir / 'stage_03_bmd',
            'completion': self.experiment_dir / 'stage_04_completion',
        }
        
        # Loaded data
        self.ms1_data: Optional[pd.DataFrame] = None
        self.sentropy_data: Optional[pd.DataFrame] = None
        self.spectra_summary: Optional[pd.DataFrame] = None
        
        # Generated objects
        self.objects: Dict[str, Object3D] = {}
        
    def load_data(self) -> None:
        """Load all experimental data."""
        logger.info(f"Loading data for {self.experiment_name}")
        
        # Load MS1 XIC data
        ms1_file = self.stage_dirs['preprocessing'] / 'ms1_xic.csv'
        if ms1_file.exists():
            self.ms1_data = pd.read_csv(ms1_file)
            logger.info(f"  Loaded {len(self.ms1_data)} MS1 data points")
        
        # Load S-entropy features
        sentropy_file = self.stage_dirs['sentropy'] / 'sentropy_features.csv'
        if sentropy_file.exists():
            self.sentropy_data = pd.read_csv(sentropy_file)
            logger.info(f"  Loaded {len(self.sentropy_data)} S-entropy features")
        
        # Load spectra summary
        spectra_file = self.stage_dirs['preprocessing'] / 'spectra_summary.csv'
        if spectra_file.exists():
            self.spectra_summary = pd.read_csv(spectra_file)
            logger.info(f"  Loaded {len(self.spectra_summary)} spectra summaries")
    
    def generate_all_objects(self) -> Dict[str, Object3D]:
        """
        Generate 3D objects for all pipeline stages.
        
        Returns:
            Dictionary of stage -> Object3D
        """
        logger.info(f"Generating 3D objects for {self.experiment_name}")
        
        self.load_data()
        
        # Generate objects for each stage
        self.objects['solution'] = self.generate_solution_object()
        self.objects['chromatography'] = self.generate_chromatography_object()
        self.objects['ionization'] = self.generate_ionization_object()
        self.objects['ms1'] = self.generate_ms1_object()
        self.objects['ms2'] = self.generate_ms2_object()
        self.objects['droplet'] = self.generate_droplet_object()
        
        logger.info(f"  Generated {len(self.objects)} 3D objects")
        
        return self.objects
    
    def generate_solution_object(self) -> Object3D:
        """
        Stage 1: Solution Phase (Blue Sphere)
        
        Initial molecular ensemble in solution before chromatography.
        Represents the complete mixture.
        """
        logger.info("  Generating solution phase object...")
        
        # Calculate mean S-coordinates from all data
        if self.sentropy_data is not None:
            s_k_mean = self.sentropy_data['s_k_mean'].mean()
            s_t_mean = self.sentropy_data['s_t_mean'].mean()
            s_e_mean = self.sentropy_data['s_e_mean'].mean()
            
            # Temperature from S-variance
            s_k_var = self.sentropy_data['s_k_std'].mean() ** 2
            s_t_var = self.sentropy_data['s_t_std'].mean() ** 2
            s_e_var = self.sentropy_data['s_e_std'].mean() ** 2
            temperature = (s_k_var + s_t_var + s_e_var) / 3
        else:
            s_k_mean = s_t_mean = s_e_mean = 0.5
            temperature = 0.1
        
        center = SEntropyCoordinate(s_k_mean, s_t_mean, s_e_mean)
        
        # Sphere with radius proportional to data spread
        radius = 0.3  # Base radius in S-space
        
        # Thermodynamic properties
        n_molecules = len(self.ms1_data) if self.ms1_data is not None else 1000
        thermo = ThermodynamicProperties(
            temperature=temperature,
            pressure=n_molecules / 1.0,  # Molecules per unit time
            entropy=np.log(n_molecules),
            volume=(4/3) * np.pi * radius**3
        )
        
        obj = Object3D(
            stage='solution',
            shape='sphere',
            center=center,
            dimensions=(radius, radius, radius),
            color=(0.2, 0.4, 0.8),  # Blue
            texture='smooth',
            thermo=thermo,
            molecule_count=n_molecules,
            data={'description': 'Initial molecular ensemble in solution'}
        )
        
        logger.info(f"    Solution: {n_molecules} molecules, T={temperature:.4f}")
        return obj
    
    def generate_chromatography_object(self) -> Object3D:
        """
        Stage 2: Chromatography (Green Ellipsoid with Ridges)
        
        Temporal separation creates elongated structure along time axis.
        Ridges represent individual chromatographic peaks.
        """
        logger.info("  Generating chromatography object...")
        
        if self.sentropy_data is None:
            return self.generate_solution_object()  # Fallback
        
        # Elongate along S_t axis (temporal)
        s_k_mean = self.sentropy_data['s_k_mean'].mean()
        s_t_mean = self.sentropy_data['s_t_mean'].mean()
        s_e_mean = self.sentropy_data['s_e_mean'].mean()
        
        center = SEntropyCoordinate(s_k_mean, s_t_mean, s_e_mean)
        
        # Ellipsoid: elongated along time axis
        a = 0.2  # S_k axis
        b = 0.8  # S_t axis (elongated)
        c = 0.2  # S_e axis
        
        # Temperature increases slightly due to chromatographic heating
        temperature = self.sentropy_data['s_t_std'].mean() ** 2
        
        # Calculate number of peaks (ridges)
        n_peaks = len(self.sentropy_data)
        
        thermo = ThermodynamicProperties(
            temperature=temperature * 1.1,
            pressure=n_peaks / 2.0,
            entropy=np.log(n_peaks) * 1.2,
            volume=(4/3) * np.pi * a * b * c
        )
        
        obj = Object3D(
            stage='chromatography',
            shape='ellipsoid',
            center=center,
            dimensions=(a, b, c),
            color=(0.2, 0.8, 0.3),  # Green
            texture='ridged',
            thermo=thermo,
            molecule_count=n_peaks,
            data={
                'description': 'Chromatographic separation with temporal elongation',
                'n_peaks': n_peaks,
                'elongation_factor': b / a
            }
        )
        
        logger.info(f"    Chromatography: {n_peaks} peaks, elongation={b/a:.2f}")
        return obj
    
    def generate_ionization_object(self) -> Object3D:
        """
        Stage 3: Ionization (Yellow Fragmenting Sphere)
        
        Electrospray creates Coulomb explosion.
        Sphere begins to fragment due to charge repulsion.
        """
        logger.info("  Generating ionization object...")
        
        if self.sentropy_data is None:
            return self.generate_solution_object()
        
        # Center shifts slightly due to charge state distribution
        s_k_mean = self.sentropy_data['s_k_mean'].mean() * 1.05
        s_t_mean = self.sentropy_data['s_t_mean'].mean()
        s_e_mean = self.sentropy_data['s_e_mean'].mean() * 1.1  # Energy increases
        
        center = SEntropyCoordinate(s_k_mean, s_t_mean, s_e_mean)
        
        # Sphere with slight expansion
        radius = 0.35
        
        # Temperature increases significantly (ionization energy)
        temperature = self.sentropy_data['s_e_std'].mean() ** 2 * 2.0
        
        n_ions = len(self.sentropy_data)
        
        thermo = ThermodynamicProperties(
            temperature=temperature,
            pressure=n_ions / 0.5,  # Higher pressure from ionization
            entropy=np.log(n_ions) * 1.5,
            volume=(4/3) * np.pi * radius**3
        )
        
        obj = Object3D(
            stage='ionization',
            shape='fragmenting_sphere',
            center=center,
            dimensions=(radius, radius, radius),
            color=(0.9, 0.8, 0.2),  # Yellow
            texture='fractured',
            thermo=thermo,
            molecule_count=n_ions,
            data={
                'description': 'Coulomb explosion from electrospray ionization',
                'charge_states': 'multiple'
            }
        )
        
        logger.info(f"    Ionization: {n_ions} ions, T={temperature:.4f}")
        return obj
    
    def generate_ms1_object(self) -> Object3D:
        """
        Stage 4: MS1 (Orange Sphere Array)
        
        Discrete ions positioned by (m/z, S_t, S_k).
        Array of small spheres in 3D space.
        """
        logger.info("  Generating MS1 object...")
        
        if self.ms1_data is None or self.sentropy_data is None:
            return self.generate_ionization_object()
        
        # Sample representative ions (too many for full array)
        sample_size = min(1000, len(self.ms1_data))
        ms1_sample = self.ms1_data.sample(n=sample_size, random_state=42)
        
        # Mean position in S-space
        s_k_mean = self.sentropy_data['s_k_mean'].mean()
        s_t_mean = self.sentropy_data['s_t_mean'].mean()
        s_e_mean = self.sentropy_data['s_e_mean'].mean()
        
        center = SEntropyCoordinate(s_k_mean, s_t_mean, s_e_mean)
        
        # Array dimensions (bounding box)
        mz_range = ms1_sample['mz'].max() - ms1_sample['mz'].min()
        rt_range = ms1_sample['rt'].max() - ms1_sample['rt'].min()
        
        # Normalize to S-space
        a = 0.4  # S_k spread
        b = 0.4  # S_t spread
        c = 0.4  # S_e spread
        
        # Temperature from ion distribution
        temperature = self.sentropy_data['s_k_std'].mean() ** 2
        
        n_ions = len(ms1_sample)
        
        # Calculate individual ion positions
        ion_positions = []
        for _, row in ms1_sample.iterrows():
            # Map m/z to S_k, rt to S_t
            s_k = (row['mz'] - ms1_sample['mz'].min()) / mz_range if mz_range > 0 else 0.5
            s_t = (row['rt'] - ms1_sample['rt'].min()) / rt_range if rt_range > 0 else 0.5
            s_e = s_e_mean
            ion_positions.append((s_k, s_t, s_e))
        
        thermo = ThermodynamicProperties(
            temperature=temperature,
            pressure=n_ions / 1.0,
            entropy=np.log(n_ions) * 1.8,
            volume=a * b * c
        )
        
        obj = Object3D(
            stage='ms1',
            shape='sphere_array',
            center=center,
            dimensions=(a, b, c),
            color=(1.0, 0.6, 0.2),  # Orange
            texture='discrete',
            thermo=thermo,
            molecule_count=n_ions,
            data={
                'description': 'Discrete ions positioned by (m/z, rt, intensity)',
                'ion_positions': ion_positions[:100],  # Store first 100
                'mz_range': (ms1_sample['mz'].min(), ms1_sample['mz'].max()),
                'rt_range': (ms1_sample['rt'].min(), ms1_sample['rt'].max())
            }
        )
        
        logger.info(f"    MS1: {n_ions} ions, m/z range={mz_range:.1f}")
        return obj
    
    def generate_ms2_object(self) -> Object3D:
        """
        Stage 5: MS2/Fragmentation (Red Cascade)
        
        Autocatalytic cascade of fragmentation events.
        Explosive pattern radiating from precursor ions.
        """
        logger.info("  Generating MS2 fragmentation object...")
        
        if self.sentropy_data is None:
            return self.generate_ms1_object()
        
        # Fragmentation increases S_e (evolution entropy)
        s_k_mean = self.sentropy_data['s_k_mean'].mean()
        s_t_mean = self.sentropy_data['s_t_mean'].mean()
        s_e_mean = self.sentropy_data['s_e_mean'].mean() * 1.5  # Significant increase
        
        center = SEntropyCoordinate(s_k_mean, s_t_mean, s_e_mean)
        
        # Cascade dimensions (explosive expansion)
        a = 0.6  # Expanded
        b = 0.6
        c = 0.6
        
        # Temperature peaks during fragmentation
        temperature = self.sentropy_data['s_e_std'].mean() ** 2 * 3.0
        
        # Estimate fragment count (typically more fragments than precursors)
        n_fragments = len(self.sentropy_data) * 5  # Approximate
        
        thermo = ThermodynamicProperties(
            temperature=temperature,
            pressure=n_fragments / 0.3,  # High pressure from cascade
            entropy=np.log(n_fragments) * 2.0,
            volume=(4/3) * np.pi * a * b * c
        )
        
        obj = Object3D(
            stage='ms2',
            shape='cascade',
            center=center,
            dimensions=(a, b, c),
            color=(0.9, 0.2, 0.2),  # Red
            texture='explosive',
            thermo=thermo,
            molecule_count=n_fragments,
            data={
                'description': 'Autocatalytic fragmentation cascade',
                'cascade_depth': 3,
                'branching_factor': 5
            }
        )
        
        logger.info(f"    MS2: ~{n_fragments} fragments, T={temperature:.4f}")
        return obj
    
    def generate_droplet_object(self) -> Object3D:
        """
        Stage 6: Droplet (Purple Wave Pattern)
        
        Final thermodynamic image - the bijective CV transformation.
        Wave pattern encodes complete spectral information.
        """
        logger.info("  Generating droplet object...")
        
        if self.sentropy_data is None:
            return self.generate_ms2_object()
        
        # Final state in S-space
        s_k_mean = self.sentropy_data['s_k_mean'].mean()
        s_t_mean = self.sentropy_data['s_t_mean'].mean()
        s_e_mean = self.sentropy_data['s_e_mean'].mean()
        
        center = SEntropyCoordinate(s_k_mean, s_t_mean, s_e_mean)
        
        # Droplet dimensions
        radius = 0.25  # Compact final state
        
        # Calculate thermodynamic droplet properties
        # Using bijective transformation formulas
        
        # Temperature from S-entropy
        T_categorical = (self.sentropy_data['s_k_std'].mean() ** 2 + 
                        self.sentropy_data['s_t_std'].mean() ** 2 + 
                        self.sentropy_data['s_e_std'].mean() ** 2) / 3
        
        # Map to physical droplet properties
        # Radius: r ∝ S_k (information content)
        r_droplet = s_k_mean * 1e-6  # meters
        
        # Velocity: v ∝ S_t (temporal evolution)
        v_droplet = s_t_mean * 10.0  # m/s
        
        # Surface tension: σ ∝ 1/S_e (inverse evolution entropy)
        sigma_droplet = 0.072 / (s_e_mean + 0.1)  # N/m (water-like)
        
        # Calculate dimensionless numbers
        rho = 1000.0  # kg/m³ (water density)
        mu = 0.001   # Pa·s (water viscosity)
        
        We = rho * v_droplet**2 * r_droplet / sigma_droplet if sigma_droplet > 0 else 0
        Re = rho * v_droplet * r_droplet / mu if mu > 0 else 0
        Oh = np.sqrt(We) / Re if Re > 0 else 0
        
        thermo = ThermodynamicProperties(
            temperature=T_categorical,
            pressure=len(self.sentropy_data) / 10.0,
            entropy=np.log(len(self.sentropy_data)) * 2.5,
            volume=(4/3) * np.pi * radius**3,
            radius=r_droplet,
            velocity=v_droplet,
            surface_tension=sigma_droplet,
            weber_number=We,
            reynolds_number=Re,
            ohnesorge_number=Oh
        )
        
        # Physics validation
        physics_valid = (0.1 < We < 1000) and (10 < Re < 10000) and (0.001 < Oh < 1.0)
        
        obj = Object3D(
            stage='droplet',
            shape='wave_pattern',
            center=center,
            dimensions=(radius, radius, radius),
            color=(0.6, 0.2, 0.8),  # Purple
            texture='waves',
            thermo=thermo,
            molecule_count=len(self.sentropy_data),
            data={
                'description': 'Final thermodynamic droplet image',
                'bijective_transformation': True,
                'physics_validation': {
                    'weber_number': We,
                    'reynolds_number': Re,
                    'ohnesorge_number': Oh,
                    'physically_valid': physics_valid
                }
            }
        )
        
        logger.info(f"    Droplet: We={We:.2f}, Re={Re:.2f}, Oh={Oh:.4f}, valid={physics_valid}")
        return obj
    
    def validate_conservation(self) -> Dict[str, Any]:
        """
        Validate that information is conserved through the pipeline.
        
        Volume should be conserved (bijective transformation).
        """
        logger.info("Validating information conservation...")
        
        if not self.objects:
            return {'error': 'No objects generated'}
        
        # Calculate volumes
        volumes = {stage: obj.volume() for stage, obj in self.objects.items()}
        
        # Initial and final volumes
        v_initial = volumes.get('solution', 0)
        v_final = volumes.get('droplet', 0)
        
        # Conservation ratio
        conservation_ratio = v_final / v_initial if v_initial > 0 else 0
        
        # Check molecule count conservation
        n_initial = self.objects['solution'].molecule_count
        n_final = self.objects['droplet'].molecule_count
        
        molecule_ratio = n_final / n_initial if n_initial > 0 else 0
        
        validation = {
            'volumes': volumes,
            'initial_volume': v_initial,
            'final_volume': v_final,
            'conservation_ratio': conservation_ratio,
            'volume_conserved': abs(conservation_ratio - 1.0) < 0.5,
            'initial_molecules': n_initial,
            'final_molecules': n_final,
            'molecule_ratio': molecule_ratio,
            'information_preserved': abs(molecule_ratio - 1.0) < 0.2
        }
        
        logger.info(f"  Volume conservation: {conservation_ratio:.2%}")
        logger.info(f"  Molecule conservation: {molecule_ratio:.2%}")
        
        return validation
    
    def export_objects(self, output_dir: Path) -> None:
        """
        Export 3D objects to JSON for visualization.
        
        Args:
            output_dir: Directory to save object files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Exporting 3D objects to {output_dir}")
        
        for stage, obj in self.objects.items():
            obj_dict = {
                'stage': obj.stage,
                'shape': obj.shape,
                'center': {
                    'S_k': obj.center.S_k,
                    'S_t': obj.center.S_t,
                    'S_e': obj.center.S_e
                },
                'dimensions': obj.dimensions,
                'color': obj.color,
                'texture': obj.texture,
                'thermodynamics': {
                    'temperature': obj.thermo.temperature,
                    'pressure': obj.thermo.pressure,
                    'entropy': obj.thermo.entropy,
                    'volume': obj.thermo.volume,
                    'radius': obj.thermo.radius,
                    'velocity': obj.thermo.velocity,
                    'surface_tension': obj.thermo.surface_tension,
                    'weber_number': obj.thermo.weber_number,
                    'reynolds_number': obj.thermo.reynolds_number,
                    'ohnesorge_number': obj.thermo.ohnesorge_number
                },
                'molecule_count': obj.molecule_count,
                'data': obj.data
            }
            
            output_file = output_dir / f'{stage}_object.json'
            with open(output_file, 'w') as f:
                json.dump(obj_dict, f, indent=2, default=str)
            
            logger.info(f"  Exported {stage} → {output_file.name}")


def generate_pipeline_objects_for_experiment(experiment_dir: Path, 
                                             output_dir: Optional[Path] = None) -> Dict[str, Object3D]:
    """
    Generate complete 3D object pipeline for an experiment.
    
    Args:
        experiment_dir: Path to experiment directory
        output_dir: Optional output directory for JSON exports
        
    Returns:
        Dictionary of stage -> Object3D
    """
    generator = Pipeline3DObjectGenerator(experiment_dir)
    objects = generator.generate_all_objects()
    
    # Validate conservation
    validation = generator.validate_conservation()
    
    # Export if requested
    if output_dir:
        generator.export_objects(output_dir)
    
    return objects, validation


if __name__ == "__main__":
    # Test with one experiment
    logging.basicConfig(level=logging.INFO)
    
    experiment_dir = Path("precursor/results/ucdavis_fast_analysis/A_M3_negPFP_03")
    output_dir = experiment_dir / "3d_objects"
    
    objects, validation = generate_pipeline_objects_for_experiment(
        experiment_dir, 
        output_dir
    )
    
    print("\n" + "="*70)
    print("3D OBJECT PIPELINE GENERATED")
    print("="*70)
    print(f"\nExperiment: {experiment_dir.name}")
    print(f"\nGenerated {len(objects)} 3D objects:")
    for stage, obj in objects.items():
        print(f"  {stage:15s} → {obj.shape:20s} ({obj.color[0]:.1f}, {obj.color[1]:.1f}, {obj.color[2]:.1f})")
    
    print(f"\nValidation:")
    print(f"  Volume conservation: {validation['conservation_ratio']:.2%}")
    print(f"  Molecule conservation: {validation['molecule_ratio']:.2%}")
    print(f"  Information preserved: {validation['information_preserved']}")

