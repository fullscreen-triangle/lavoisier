"""
3D Pipeline Visualization
==========================

Visualize the 3D object pipeline transformation.
Creates publication-quality figures showing the molecular journey.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import logging

logger = logging.getLogger(__name__)


class Pipeline3DVisualizer:
    """
    Visualizer for 3D object pipeline.
    
    Creates various visualization types:
    1. 2D projection grid (all stages)
    2. 3D interactive view (single stage)
    3. Animation (transformation through stages)
    4. Property evolution charts
    """
    
    def __init__(self, objects_dir: Path):
        """
        Initialize visualizer.
        
        Args:
            objects_dir: Directory containing 3D object JSON files
        """
        self.objects_dir = Path(objects_dir)
        self.objects: Dict[str, Dict] = {}
        self.load_objects()
        
    def load_objects(self) -> None:
        """Load all 3D objects from JSON files."""
        logger.info(f"Loading 3D objects from {self.objects_dir}")
        
        stage_files = {
            'solution': 'solution_object.json',
            'chromatography': 'chromatography_object.json',
            'ionization': 'ionization_object.json',
            'ms1': 'ms1_object.json',
            'ms2': 'ms2_object.json',
            'droplet': 'droplet_object.json'
        }
        
        for stage, filename in stage_files.items():
            filepath = self.objects_dir / filename
            if filepath.exists():
                with open(filepath, 'r') as f:
                    self.objects[stage] = json.load(f)
                logger.info(f"  Loaded {stage}")
        
        logger.info(f"Loaded {len(self.objects)} objects")
    
    def plot_2d_grid(self, output_file: Optional[Path] = None) -> None:
        """
        Create 2×3 grid showing all pipeline stages.
        
        Each subplot shows a 2D projection of the 3D object.
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('3D Object Pipeline: Molecular Journey Through Mass Spectrometry', 
                     fontsize=16, fontweight='bold')
        
        stages = ['solution', 'chromatography', 'ionization', 'ms1', 'ms2', 'droplet']
        
        for idx, stage in enumerate(stages):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]
            
            if stage not in self.objects:
                ax.text(0.5, 0.5, f'{stage}\n(not available)', 
                       ha='center', va='center', fontsize=12)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
                continue
            
            obj = self.objects[stage]
            self._plot_object_2d(ax, obj)
            
            # Title with stage info
            title = f"{stage.upper()}\n{obj['shape'].replace('_', ' ').title()}"
            ax.set_title(title, fontsize=12, fontweight='bold')
            
            # Add molecule count
            ax.text(0.02, 0.98, f"N = {obj['molecule_count']}", 
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved 2D grid to {output_file}")
        
        plt.show()
    
    def _plot_object_2d(self, ax, obj: Dict) -> None:
        """
        Plot a single 3D object as 2D projection.
        
        Args:
            ax: Matplotlib axis
            obj: Object dictionary
        """
        center = obj['center']
        dims = obj['dimensions']
        color = obj['color']
        shape = obj['shape']
        
        # S_k (x-axis) vs S_t (y-axis) projection
        cx, cy = center['S_k'], center['S_t']
        
        if shape == 'sphere':
            circle = Circle((cx, cy), dims[0], color=color, alpha=0.6)
            ax.add_patch(circle)
            
        elif shape == 'ellipsoid':
            ellipse = Ellipse((cx, cy), dims[0]*2, dims[1]*2, 
                            color=color, alpha=0.6)
            ax.add_patch(ellipse)
            
            # Add ridges for chromatography
            if obj['texture'] == 'ridged':
                n_ridges = 5
                for i in range(n_ridges):
                    y_ridge = cy - dims[1] + (2*dims[1]*i/(n_ridges-1))
                    ax.plot([cx-dims[0], cx+dims[0]], [y_ridge, y_ridge], 
                           'k-', alpha=0.3, linewidth=1)
        
        elif shape == 'fragmenting_sphere':
            # Sphere with fracture lines
            circle = Circle((cx, cy), dims[0], color=color, alpha=0.6)
            ax.add_patch(circle)
            
            # Add fracture lines
            angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
            for angle in angles:
                dx = dims[0] * np.cos(angle)
                dy = dims[0] * np.sin(angle)
                ax.plot([cx, cx+dx], [cy, cy+dy], 'k-', alpha=0.5, linewidth=2)
        
        elif shape == 'sphere_array':
            # Draw bounding box
            rect = mpatches.Rectangle((cx-dims[0]/2, cy-dims[1]/2), 
                                     dims[0], dims[1],
                                     fill=False, edgecolor=color, linewidth=2)
            ax.add_patch(rect)
            
            # Draw sample ions
            if 'ion_positions' in obj['data']:
                positions = obj['data']['ion_positions'][:50]  # First 50
                for pos in positions:
                    s_k, s_t, _ = pos
                    ax.plot(s_k, s_t, 'o', color=color, markersize=3, alpha=0.5)
        
        elif shape == 'cascade':
            # Explosive pattern
            circle = Circle((cx, cy), dims[0], fill=False, 
                          edgecolor=color, linewidth=2, linestyle='--')
            ax.add_patch(circle)
            
            # Radiating lines
            n_rays = 12
            angles = np.linspace(0, 2*np.pi, n_rays, endpoint=False)
            for angle in angles:
                dx = dims[0] * np.cos(angle)
                dy = dims[0] * np.sin(angle)
                ax.plot([cx, cx+dx*1.2], [cy, cy+dy*1.2], 
                       color=color, alpha=0.6, linewidth=2)
        
        elif shape == 'wave_pattern':
            # Droplet with wave pattern
            circle = Circle((cx, cy), dims[0], color=color, alpha=0.6)
            ax.add_patch(circle)
            
            # Add concentric waves
            n_waves = 5
            for i in range(1, n_waves+1):
                r = dims[0] * i / n_waves
                wave_circle = Circle((cx, cy), r, fill=False, 
                                   edgecolor='purple', linewidth=1, alpha=0.4)
                ax.add_patch(wave_circle)
        
        # Set limits
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('S_k (Knowledge Entropy)', fontsize=10)
        ax.set_ylabel('S_t (Temporal Entropy)', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    def plot_property_evolution(self, output_file: Optional[Path] = None) -> None:
        """
        Plot how properties evolve through the pipeline.
        
        Creates line plots showing:
        - Temperature evolution
        - Pressure evolution
        - Entropy evolution
        - Volume evolution
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Property Evolution Through Pipeline', 
                     fontsize=14, fontweight='bold')
        
        stages = ['solution', 'chromatography', 'ionization', 'ms1', 'ms2', 'droplet']
        stage_nums = list(range(len(stages)))
        
        # Extract properties
        temperatures = []
        pressures = []
        entropies = []
        volumes = []
        
        for stage in stages:
            if stage in self.objects:
                thermo = self.objects[stage]['thermodynamics']
                temperatures.append(thermo['temperature'])
                pressures.append(thermo['pressure'])
                entropies.append(thermo['entropy'])
                volumes.append(thermo['volume'])
            else:
                temperatures.append(np.nan)
                pressures.append(np.nan)
                entropies.append(np.nan)
                volumes.append(np.nan)
        
        # Temperature
        ax = axes[0, 0]
        ax.plot(stage_nums, temperatures, 'o-', color='red', linewidth=2, markersize=8)
        ax.set_ylabel('Temperature (S-variance)', fontsize=11)
        ax.set_title('Temperature Evolution', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(stage_nums)
        ax.set_xticklabels([s[:4] for s in stages], rotation=45)
        
        # Pressure
        ax = axes[0, 1]
        ax.plot(stage_nums, pressures, 'o-', color='blue', linewidth=2, markersize=8)
        ax.set_ylabel('Pressure (sampling rate)', fontsize=11)
        ax.set_title('Pressure Evolution', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(stage_nums)
        ax.set_xticklabels([s[:4] for s in stages], rotation=45)
        
        # Entropy
        ax = axes[1, 0]
        ax.plot(stage_nums, entropies, 'o-', color='green', linewidth=2, markersize=8)
        ax.set_ylabel('Entropy (S-spread)', fontsize=11)
        ax.set_title('Entropy Evolution', fontweight='bold')
        ax.set_xlabel('Pipeline Stage', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(stage_nums)
        ax.set_xticklabels([s[:4] for s in stages], rotation=45)
        
        # Volume
        ax = axes[1, 1]
        ax.plot(stage_nums, volumes, 'o-', color='purple', linewidth=2, markersize=8)
        ax.set_ylabel('Volume (S-space)', fontsize=11)
        ax.set_title('Volume Conservation', fontweight='bold')
        ax.set_xlabel('Pipeline Stage', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(stage_nums)
        ax.set_xticklabels([s[:4] for s in stages], rotation=45)
        
        # Add horizontal line at initial volume
        if volumes[0] is not np.nan:
            ax.axhline(volumes[0], color='gray', linestyle='--', alpha=0.5, 
                      label='Initial volume')
            ax.legend()
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved property evolution to {output_file}")
        
        plt.show()
    
    def plot_physics_validation(self, output_file: Optional[Path] = None) -> None:
        """
        Plot physics validation for droplet stage.
        
        Shows Weber, Reynolds, and Ohnesorge numbers.
        """
        if 'droplet' not in self.objects:
            logger.warning("No droplet object available for physics validation")
            return
        
        droplet = self.objects['droplet']
        if 'physics_validation' not in droplet['data']:
            logger.warning("No physics validation data in droplet object")
            return
        
        phys = droplet['data']['physics_validation']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Physics Validation: Dimensionless Numbers', 
                     fontsize=14, fontweight='bold')
        
        # Weber number
        ax = axes[0]
        We = phys['weber_number']
        ax.barh(['Weber'], [We], color='blue', alpha=0.7)
        ax.axvline(0.1, color='red', linestyle='--', alpha=0.5, label='Valid range')
        ax.axvline(1000, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Weber Number (We)', fontsize=11)
        ax.set_title(f'We = {We:.2f}', fontweight='bold')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')
        
        # Reynolds number
        ax = axes[1]
        Re = phys['reynolds_number']
        ax.barh(['Reynolds'], [Re], color='green', alpha=0.7)
        ax.axvline(10, color='red', linestyle='--', alpha=0.5, label='Valid range')
        ax.axvline(10000, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Reynolds Number (Re)', fontsize=11)
        ax.set_title(f'Re = {Re:.2f}', fontweight='bold')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')
        
        # Ohnesorge number
        ax = axes[2]
        Oh = phys['ohnesorge_number']
        ax.barh(['Ohnesorge'], [Oh], color='purple', alpha=0.7)
        ax.axvline(0.001, color='red', linestyle='--', alpha=0.5, label='Valid range')
        ax.axvline(1.0, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Ohnesorge Number (Oh)', fontsize=11)
        ax.set_title(f'Oh = {Oh:.4f}', fontweight='bold')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add validation status
        valid = phys['physically_valid']
        status_text = "✓ PHYSICALLY VALID" if valid else "✗ NOT PHYSICALLY VALID"
        status_color = 'green' if valid else 'red'
        fig.text(0.5, 0.02, status_text, ha='center', fontsize=14, 
                fontweight='bold', color=status_color)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved physics validation to {output_file}")
        
        plt.show()


def visualize_experiment(experiment_dir: Path, output_dir: Optional[Path] = None) -> None:
    """
    Create all visualizations for an experiment.
    
    Args:
        experiment_dir: Path to experiment directory
        output_dir: Optional output directory for figures
    """
    objects_dir = experiment_dir / '3d_objects'
    
    if not objects_dir.exists():
        logger.error(f"3D objects directory not found: {objects_dir}")
        return
    
    visualizer = Pipeline3DVisualizer(objects_dir)
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate all visualizations
    print(f"Generating visualizations for {experiment_dir.name}")
    
    # 2D grid
    grid_file = output_dir / f'{experiment_dir.name}_grid.png' if output_dir else None
    visualizer.plot_2d_grid(grid_file)
    
    # Property evolution
    prop_file = output_dir / f'{experiment_dir.name}_properties.png' if output_dir else None
    visualizer.plot_property_evolution(prop_file)
    
    # Physics validation
    phys_file = output_dir / f'{experiment_dir.name}_physics.png' if output_dir else None
    visualizer.plot_physics_validation(phys_file)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test with one experiment
    experiment_dir = Path("precursor/results/ucdavis_fast_analysis/A_M3_negPFP_03")
    output_dir = experiment_dir / 'visualizations'
    
    visualize_experiment(experiment_dir, output_dir)

