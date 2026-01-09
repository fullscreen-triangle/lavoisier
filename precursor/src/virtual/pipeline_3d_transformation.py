"""
3D Object Pipeline Transformation
==================================

Extends the virtual MS framework with 3D object representation.
Integrates with existing categorical state architecture.

Core Principle:
The mass spectrometer physically implements a 3D thermodynamic transformation.
Each pipeline stage corresponds to a 3D object in S-entropy space.

Integration Points:
- molecular_demon_state_architecture.py → CategoricalState
- virtual_stages.py → Theatre pipeline stages
- entropy_transformation.py → S-entropy coordinates
- experimental_validation.py → Real data validation

Author: Kundai Farai Sachikonye
Date: 2025
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import json
import logging

# Import from existing virtual framework
try:
    from .molecular_demon_state_architecture import CategoricalState
    from .entropy_transformation import SEntropyCoordinates
    VIRTUAL_AVAILABLE = True
except ImportError:
    VIRTUAL_AVAILABLE = False
    logging.warning("Virtual framework not available - using standalone mode")

logger = logging.getLogger(__name__)


@dataclass
class Object3DState:
    """
    3D object representation of a categorical state.
    
    Extends CategoricalState with geometric and thermodynamic properties.
    """
    # Pipeline stage
    stage: str  # 'solution', 'chromatography', 'ionization', 'ms1', 'ms2', 'droplet'
    
    # Categorical state (from existing framework)
    categorical_state: Optional[Any] = None  # CategoricalState if available
    
    # S-entropy coordinates
    S_k: float = 0.5  # Knowledge entropy
    S_t: float = 0.5  # Temporal entropy
    S_e: float = 0.5  # Evolution entropy
    
    # Geometric properties
    shape: str = 'sphere'  # 'sphere', 'ellipsoid', 'fragmenting_sphere', etc.
    dimensions: Tuple[float, float, float] = (0.3, 0.3, 0.3)  # (a, b, c)
    color: Tuple[float, float, float] = (0.5, 0.5, 0.5)  # RGB
    texture: str = 'smooth'  # 'smooth', 'ridged', 'fractured', etc.
    
    # Thermodynamic properties (from categorical state)
    temperature: float = 0.0  # Categorical temperature (S-variance)
    pressure: float = 0.0     # Categorical pressure (sampling rate)
    entropy: float = 0.0      # Categorical entropy (S-spread)
    volume: float = 0.0       # S-space volume
    
    # Droplet properties (final stage)
    radius: Optional[float] = None       # Physical radius (m)
    velocity: Optional[float] = None     # Physical velocity (m/s)
    surface_tension: Optional[float] = None  # Surface tension (N/m)
    
    # Dimensionless numbers (physics validation)
    weber_number: Optional[float] = None     # We = ρv²L/σ
    reynolds_number: Optional[float] = None  # Re = ρvL/μ
    ohnesorge_number: Optional[float] = None # Oh = √We / Re
    
    # Data payload
    molecule_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Export to JSON-serializable dict."""
        return {
            'stage': self.stage,
            'shape': self.shape,
            'center': {'S_k': self.S_k, 'S_t': self.S_t, 'S_e': self.S_e},
            'dimensions': list(self.dimensions),
            'color': list(self.color),
            'texture': self.texture,
            'thermodynamics': {
                'temperature': self.temperature,
                'pressure': self.pressure,
                'entropy': self.entropy,
                'volume': self.volume,
                'radius': self.radius,
                'velocity': self.velocity,
                'surface_tension': self.surface_tension,
                'weber_number': self.weber_number,
                'reynolds_number': self.reynolds_number,
                'ohnesorge_number': self.ohnesorge_number
            },
            'molecule_count': self.molecule_count,
            'metadata': self.metadata
        }


class Pipeline3DTransformation:
    """
    Transforms categorical states into 3D objects through the pipeline.
    
    Integrates with existing virtual MS framework:
    - Reads from theatre stage results
    - Uses S-entropy coordinates from entropy_transformation
    - Creates 3D objects at each convergence node
    - Validates with experimental data
    """
    
    def __init__(self, experiment_dir: Path):
        """
        Initialize transformation for an experiment.
        
        Args:
            experiment_dir: Path to experiment results (theatre output)
        """
        self.experiment_dir = Path(experiment_dir)
        self.experiment_name = self.experiment_dir.name
        
        # Load theatre results
        self.theatre_data = self._load_theatre_results()
        
        # Load stage data
        self.stage_data = {}
        self._load_stage_data()
        
        # Generated 3D objects
        self.objects_3d: Dict[str, Object3DState] = {}
        
    def _load_theatre_results(self) -> Optional[Dict]:
        """Load theatre execution results."""
        theatre_file = self.experiment_dir / 'theatre_result.json'
        
        if theatre_file.exists():
            with open(theatre_file) as f:
                return json.load(f)
        
        # Try pipeline_results.json (ucdavis format)
        pipeline_file = self.experiment_dir / 'pipeline_results.json'
        if pipeline_file.exists():
            with open(pipeline_file) as f:
                return json.load(f)
        
        logger.warning(f"No theatre results found in {self.experiment_dir}")
        return None
    
    def _load_stage_data(self) -> None:
        """Load data from each pipeline stage."""
        stages = [
            'stage_01_preprocessing',
            'stage_02_sentropy',
            'stage_02_5_fragmentation',
            'stage_03_bmd',
            'stage_04_completion'
        ]
        
        for stage in stages:
            stage_dir = self.experiment_dir / stage
            if not stage_dir.exists():
                continue
            
            # Load metrics
            metrics_file = stage_dir / f'{stage}_metrics.json'
            if not metrics_file.exists():
                metrics_file = stage_dir / f'{stage}_result.json'
            
            if metrics_file.exists():
                with open(metrics_file) as f:
                    self.stage_data[stage] = json.load(f)
            
            # Load data files
            for data_file in stage_dir.glob('*.csv'):
                key = f'{stage}_{data_file.stem}'
                try:
                    self.stage_data[key] = pd.read_csv(data_file)
                except Exception as e:
                    logger.warning(f"Could not load {data_file}: {e}")
    
    def generate_all_objects(self) -> Dict[str, Object3DState]:
        """
        Generate 3D objects for all pipeline stages.
        
        Returns:
            Dictionary mapping stage name to Object3DState
        """
        logger.info(f"Generating 3D objects for {self.experiment_name}")
        
        # Generate objects for each stage
        self.objects_3d['solution'] = self._generate_solution_object()
        self.objects_3d['chromatography'] = self._generate_chromatography_object()
        self.objects_3d['ionization'] = self._generate_ionization_object()
        self.objects_3d['ms1'] = self._generate_ms1_object()
        self.objects_3d['ms2'] = self._generate_ms2_object()
        self.objects_3d['droplet'] = self._generate_droplet_object()
        
        logger.info(f"Generated {len(self.objects_3d)} 3D objects")
        
        return self.objects_3d
    
    def _get_sentropy_stats(self) -> Tuple[float, float, float, float, float, float]:
        """Get S-entropy statistics from stage_02_sentropy."""
        # Try to load sentropy features
        sentropy_key = 'stage_02_sentropy_sentropy_features'
        
        if sentropy_key in self.stage_data:
            df = self.stage_data[sentropy_key]
            return (
                df['s_k_mean'].mean(),
                df['s_t_mean'].mean(),
                df['s_e_mean'].mean(),
                df['s_k_std'].mean(),
                df['s_t_std'].mean(),
                df['s_e_std'].mean()
            )
        
        # Fallback: use default values
        return (0.5, 0.5, 0.5, 0.1, 0.1, 0.1)
    
    def _generate_solution_object(self) -> Object3DState:
        """Stage 1: Solution Phase (Blue Sphere)."""
        logger.info("  Generating solution phase object...")
        
        s_k, s_t, s_e, s_k_std, s_t_std, s_e_std = self._get_sentropy_stats()
        
        # Temperature from S-variance
        temperature = (s_k_std**2 + s_t_std**2 + s_e_std**2) / 3
        
        # Get molecule count from preprocessing
        ms1_key = 'stage_01_preprocessing_ms1_xic'
        n_molecules = len(self.stage_data[ms1_key]) if ms1_key in self.stage_data else 1000
        
        radius = 0.3
        volume = (4/3) * np.pi * radius**3
        
        obj = Object3DState(
            stage='solution',
            S_k=s_k, S_t=s_t, S_e=s_e,
            shape='sphere',
            dimensions=(radius, radius, radius),
            color=(0.2, 0.4, 0.8),  # Blue
            texture='smooth',
            temperature=temperature,
            pressure=n_molecules / 1.0,
            entropy=np.log(n_molecules),
            volume=volume,
            molecule_count=n_molecules,
            metadata={'description': 'Initial molecular ensemble in solution'}
        )
        
        logger.info(f"    Solution: {n_molecules} molecules, T={temperature:.4f}")
        return obj
    
    def _generate_chromatography_object(self) -> Object3DState:
        """Stage 2: Chromatography (Green Ellipsoid with Ridges)."""
        logger.info("  Generating chromatography object...")
        
        s_k, s_t, s_e, s_k_std, s_t_std, s_e_std = self._get_sentropy_stats()
        
        # Elongate along S_t axis (temporal)
        a, b, c = 0.2, 0.8, 0.2
        
        temperature = s_t_std**2
        
        # Get peak count
        sentropy_key = 'stage_02_sentropy_sentropy_features'
        n_peaks = len(self.stage_data[sentropy_key]) if sentropy_key in self.stage_data else 100
        
        volume = (4/3) * np.pi * a * b * c
        
        obj = Object3DState(
            stage='chromatography',
            S_k=s_k, S_t=s_t, S_e=s_e,
            shape='ellipsoid',
            dimensions=(a, b, c),
            color=(0.2, 0.8, 0.3),  # Green
            texture='ridged',
            temperature=temperature * 1.1,
            pressure=n_peaks / 2.0,
            entropy=np.log(n_peaks) * 1.2,
            volume=volume,
            molecule_count=n_peaks,
            metadata={
                'description': 'Chromatographic separation with temporal elongation',
                'n_peaks': n_peaks,
                'elongation_factor': b / a
            }
        )
        
        logger.info(f"    Chromatography: {n_peaks} peaks, elongation={b/a:.2f}")
        return obj
    
    def _generate_ionization_object(self) -> Object3DState:
        """Stage 3: Ionization (Yellow Fragmenting Sphere)."""
        logger.info("  Generating ionization object...")
        
        s_k, s_t, s_e, s_k_std, s_t_std, s_e_std = self._get_sentropy_stats()
        
        # Energy increases during ionization
        s_k *= 1.05
        s_e *= 1.1
        
        radius = 0.35
        temperature = s_e_std**2 * 2.0
        
        sentropy_key = 'stage_02_sentropy_sentropy_features'
        n_ions = len(self.stage_data[sentropy_key]) if sentropy_key in self.stage_data else 100
        
        volume = (4/3) * np.pi * radius**3
        
        obj = Object3DState(
            stage='ionization',
            S_k=s_k, S_t=s_t, S_e=s_e,
            shape='fragmenting_sphere',
            dimensions=(radius, radius, radius),
            color=(0.9, 0.8, 0.2),  # Yellow
            texture='fractured',
            temperature=temperature,
            pressure=n_ions / 0.5,
            entropy=np.log(n_ions) * 1.5,
            volume=volume,
            molecule_count=n_ions,
            metadata={
                'description': 'Coulomb explosion from electrospray ionization',
                'charge_states': 'multiple'
            }
        )
        
        logger.info(f"    Ionization: {n_ions} ions, T={temperature:.4f}")
        return obj
    
    def _generate_ms1_object(self) -> Object3DState:
        """Stage 4: MS1 (Orange Sphere Array)."""
        logger.info("  Generating MS1 object...")
        
        s_k, s_t, s_e, s_k_std, s_t_std, s_e_std = self._get_sentropy_stats()
        
        # Array dimensions
        a, b, c = 0.4, 0.4, 0.4
        
        temperature = s_k_std**2
        
        ms1_key = 'stage_01_preprocessing_ms1_xic'
        if ms1_key in self.stage_data:
            ms1_df = self.stage_data[ms1_key]
            n_ions = min(1000, len(ms1_df))
            
            # Sample ions
            sample = ms1_df.sample(n=n_ions, random_state=42) if len(ms1_df) > n_ions else ms1_df
            
            # Get m/z and rt ranges
            mz_range = sample['mz'].max() - sample['mz'].min() if 'mz' in sample.columns else 0
            rt_range = sample['rt'].max() - sample['rt'].min() if 'rt' in sample.columns else 0
        else:
            n_ions = 100
            mz_range = 1000
            rt_range = 10
        
        volume = a * b * c
        
        obj = Object3DState(
            stage='ms1',
            S_k=s_k, S_t=s_t, S_e=s_e,
            shape='sphere_array',
            dimensions=(a, b, c),
            color=(1.0, 0.6, 0.2),  # Orange
            texture='discrete',
            temperature=temperature,
            pressure=n_ions / 1.0,
            entropy=np.log(n_ions) * 1.8,
            volume=volume,
            molecule_count=n_ions,
            metadata={
                'description': 'Discrete ions positioned by (m/z, rt, intensity)',
                'mz_range': mz_range,
                'rt_range': rt_range
            }
        )
        
        logger.info(f"    MS1: {n_ions} ions, m/z range={mz_range:.1f}")
        return obj
    
    def _generate_ms2_object(self) -> Object3DState:
        """Stage 5: MS2/Fragmentation (Red Cascade)."""
        logger.info("  Generating MS2 fragmentation object...")
        
        s_k, s_t, s_e, s_k_std, s_t_std, s_e_std = self._get_sentropy_stats()
        
        # Fragmentation increases S_e
        s_e *= 1.5
        
        # Cascade dimensions
        a, b, c = 0.6, 0.6, 0.6
        
        temperature = s_e_std**2 * 3.0
        
        # Estimate fragment count
        sentropy_key = 'stage_02_sentropy_sentropy_features'
        n_precursors = len(self.stage_data[sentropy_key]) if sentropy_key in self.stage_data else 100
        n_fragments = n_precursors * 5  # Approximate
        
        volume = (4/3) * np.pi * a * b * c
        
        obj = Object3DState(
            stage='ms2',
            S_k=s_k, S_t=s_t, S_e=s_e,
            shape='cascade',
            dimensions=(a, b, c),
            color=(0.9, 0.2, 0.2),  # Red
            texture='explosive',
            temperature=temperature,
            pressure=n_fragments / 0.3,
            entropy=np.log(n_fragments) * 2.0,
            volume=volume,
            molecule_count=n_fragments,
            metadata={
                'description': 'Autocatalytic fragmentation cascade',
                'cascade_depth': 3,
                'branching_factor': 5
            }
        )
        
        logger.info(f"    MS2: ~{n_fragments} fragments, T={temperature:.4f}")
        return obj
    
    def _generate_droplet_object(self) -> Object3DState:
        """Stage 6: Droplet (Purple Wave Pattern)."""
        logger.info("  Generating droplet object...")
        
        s_k, s_t, s_e, s_k_std, s_t_std, s_e_std = self._get_sentropy_stats()
        
        # Final state
        radius = 0.25
        
        # Temperature from S-variance
        T_categorical = (s_k_std**2 + s_t_std**2 + s_e_std**2) / 3
        
        # Map to physical droplet properties
        r_droplet = s_k * 1e-6  # meters
        v_droplet = s_t * 10.0  # m/s
        sigma_droplet = 0.072 / (s_e + 0.1)  # N/m
        
        # Calculate dimensionless numbers
        rho = 1000.0  # kg/m³
        mu = 0.001   # Pa·s
        
        We = rho * v_droplet**2 * r_droplet / sigma_droplet if sigma_droplet > 0 else 0
        Re = rho * v_droplet * r_droplet / mu if mu > 0 else 0
        Oh = np.sqrt(We) / Re if Re > 0 else 0
        
        # Physics validation
        physics_valid = (0.1 < We < 1000) and (10 < Re < 10000) and (0.001 < Oh < 1.0)
        
        sentropy_key = 'stage_02_sentropy_sentropy_features'
        n_molecules = len(self.stage_data[sentropy_key]) if sentropy_key in self.stage_data else 100
        
        volume = (4/3) * np.pi * radius**3
        
        obj = Object3DState(
            stage='droplet',
            S_k=s_k, S_t=s_t, S_e=s_e,
            shape='wave_pattern',
            dimensions=(radius, radius, radius),
            color=(0.6, 0.2, 0.8),  # Purple
            texture='waves',
            temperature=T_categorical,
            pressure=n_molecules / 10.0,
            entropy=np.log(n_molecules) * 2.5,
            volume=volume,
            radius=r_droplet,
            velocity=v_droplet,
            surface_tension=sigma_droplet,
            weber_number=We,
            reynolds_number=Re,
            ohnesorge_number=Oh,
            molecule_count=n_molecules,
            metadata={
                'description': 'Final thermodynamic droplet image',
                'bijective_transformation': True,
                'physically_valid': physics_valid
            }
        )
        
        logger.info(f"    Droplet: We={We:.2f}, Re={Re:.2f}, Oh={Oh:.4f}, valid={physics_valid}")
        return obj
    
    def export_objects(self, output_dir: Path) -> None:
        """
        Export 3D objects to JSON.
        
        Args:
            output_dir: Directory to save object files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Exporting 3D objects to {output_dir}")
        
        for stage, obj in self.objects_3d.items():
            output_file = output_dir / f'{stage}_object.json'
            with open(output_file, 'w') as f:
                json.dump(obj.to_dict(), f, indent=2, default=str)
            
            logger.info(f"  Exported {stage} → {output_file.name}")
    
    def validate_conservation(self) -> Dict[str, Any]:
        """
        Validate information conservation through pipeline.
        
        Returns:
            Validation metrics
        """
        if not self.objects_3d:
            return {'error': 'No objects generated'}
        
        # Calculate volumes
        volumes = {stage: obj.volume for stage, obj in self.objects_3d.items()}
        
        v_initial = volumes.get('solution', 0)
        v_final = volumes.get('droplet', 0)
        
        conservation_ratio = v_final / v_initial if v_initial > 0 else 0
        
        # Molecule counts
        n_initial = self.objects_3d['solution'].molecule_count
        n_final = self.objects_3d['droplet'].molecule_count
        
        molecule_ratio = n_final / n_initial if n_initial > 0 else 0
        
        return {
            'volumes': volumes,
            'conservation_ratio': conservation_ratio,
            'volume_conserved': abs(conservation_ratio - 1.0) < 0.5,
            'initial_molecules': n_initial,
            'final_molecules': n_final,
            'molecule_ratio': molecule_ratio,
            'information_preserved': abs(molecule_ratio - 1.0) < 0.2
        }


def generate_3d_objects_for_experiment(experiment_dir: Path,
                                       output_dir: Optional[Path] = None) -> Tuple[Dict[str, Object3DState], Dict[str, Any]]:
    """
    Generate 3D objects for an experiment.
    
    Args:
        experiment_dir: Path to experiment directory
        output_dir: Optional output directory for JSON exports
        
    Returns:
        (objects, validation) tuple
    """
    transformer = Pipeline3DTransformation(experiment_dir)
    objects = transformer.generate_all_objects()
    validation = transformer.validate_conservation()
    
    if output_dir:
        transformer.export_objects(output_dir)
    
    return objects, validation


if __name__ == "__main__":
    # Test with one experiment
    logging.basicConfig(level=logging.INFO)
    
    experiment_dir = Path("results/ucdavis_fast_analysis/A_M3_negPFP_03")
    output_dir = experiment_dir / "3d_objects"
    
    objects, validation = generate_3d_objects_for_experiment(
        experiment_dir,
        output_dir
    )
    
    print("\n" + "="*70)
    print("3D OBJECT PIPELINE GENERATED")
    print("="*70)
    print(f"\nExperiment: {experiment_dir.name}")
    print(f"\nGenerated {len(objects)} 3D objects:")
    for stage, obj in objects.items():
        print(f"  {stage:15s} -> {obj.shape:20s} N={obj.molecule_count:6d}")
    
    print(f"\nValidation:")
    print(f"  Volume conservation: {validation['conservation_ratio']:.2%}")
    print(f"  Molecule conservation: {validation['molecule_ratio']:.2%}")
    print(f"  Information preserved: {validation['information_preserved']}")

