"""
Batch 3D Object Generation
===========================

Generate 3D objects for all experiments in the results directories.
Creates a complete pipeline visualization dataset.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any
import json
import pandas as pd

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Fallback: simple progress indicator
    def tqdm(iterable, desc=None):
        """Simple fallback for tqdm."""
        total = len(list(iterable)) if hasattr(iterable, '__len__') else 0
        for i, item in enumerate(iterable, 1):
            if desc and total:
                print(f"\r{desc}: {i}/{total}", end='', flush=True)
            yield item
        if desc:
            print()  # New line after progress

from .pipeline_3d_objects import generate_pipeline_objects_for_experiment

logger = logging.getLogger(__name__)


class Batch3DObjectGenerator:
    """
    Batch processor for generating 3D objects across multiple experiments.
    """
    
    def __init__(self, results_base_dir: Path):
        """
        Initialize batch generator.
        
        Args:
            results_base_dir: Base directory containing experiment results
        """
        self.results_base_dir = Path(results_base_dir)
        self.experiments: List[Path] = []
        self.results: List[Dict[str, Any]] = []
        
    def discover_experiments(self) -> List[Path]:
        """
        Discover all experiment directories.
        
        Returns:
            List of experiment directory paths
        """
        logger.info(f"Discovering experiments in {self.results_base_dir}")
        
        experiments = []
        
        # Look for directories with stage_01_preprocessing
        for item in self.results_base_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                stage_01 = item / 'stage_01_preprocessing'
                if stage_01.exists():
                    experiments.append(item)
        
        self.experiments = sorted(experiments)
        logger.info(f"  Found {len(self.experiments)} experiments")
        
        return self.experiments
    
    def generate_all(self, export_json: bool = True) -> pd.DataFrame:
        """
        Generate 3D objects for all experiments.
        
        Args:
            export_json: Whether to export JSON files for each object
            
        Returns:
            DataFrame with summary statistics
        """
        if not self.experiments:
            self.discover_experiments()
        
        logger.info(f"Generating 3D objects for {len(self.experiments)} experiments")
        
        results = []
        
        for exp_dir in tqdm(self.experiments, desc="Processing experiments"):
            try:
                # Output directory for this experiment
                output_dir = exp_dir / '3d_objects' if export_json else None
                
                # Generate objects
                objects, validation = generate_pipeline_objects_for_experiment(
                    exp_dir, 
                    output_dir
                )
                
                # Collect statistics
                result = {
                    'experiment': exp_dir.name,
                    'n_objects': len(objects),
                    'volume_conservation': validation['conservation_ratio'],
                    'molecule_conservation': validation['molecule_ratio'],
                    'information_preserved': validation['information_preserved'],
                    'initial_volume': validation['initial_volume'],
                    'final_volume': validation['final_volume'],
                    'initial_molecules': validation['initial_molecules'],
                    'final_molecules': validation['final_molecules'],
                }
                
                # Add droplet physics validation
                if 'droplet' in objects:
                    droplet = objects['droplet']
                    if 'physics_validation' in droplet.data:
                        phys = droplet.data['physics_validation']
                        result.update({
                            'weber_number': phys['weber_number'],
                            'reynolds_number': phys['reynolds_number'],
                            'ohnesorge_number': phys['ohnesorge_number'],
                            'physically_valid': phys['physically_valid']
                        })
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"  Error processing {exp_dir.name}: {e}")
                results.append({
                    'experiment': exp_dir.name,
                    'error': str(e)
                })
        
        self.results = results
        df = pd.DataFrame(results)
        
        # Save summary
        summary_file = self.results_base_dir / '3d_objects_summary.csv'
        df.to_csv(summary_file, index=False)
        logger.info(f"Saved summary to {summary_file}")
        
        return df
    
    def generate_master_report(self) -> Dict[str, Any]:
        """
        Generate master report with aggregate statistics.
        
        Returns:
            Dictionary with aggregate statistics
        """
        if not self.results:
            logger.warning("No results to report")
            return {}
        
        df = pd.DataFrame(self.results)
        
        # Filter out errors
        df_valid = df[~df['experiment'].str.contains('error', na=False)]
        
        report = {
            'total_experiments': len(self.results),
            'successful': len(df_valid),
            'failed': len(self.results) - len(df_valid),
            
            'volume_conservation': {
                'mean': df_valid['volume_conservation'].mean(),
                'std': df_valid['volume_conservation'].std(),
                'min': df_valid['volume_conservation'].min(),
                'max': df_valid['volume_conservation'].max(),
                'within_50_percent': (df_valid['volume_conservation'].between(0.5, 1.5)).sum()
            },
            
            'molecule_conservation': {
                'mean': df_valid['molecule_conservation'].mean(),
                'std': df_valid['molecule_conservation'].std(),
                'min': df_valid['molecule_conservation'].min(),
                'max': df_valid['molecule_conservation'].max(),
                'within_20_percent': (df_valid['molecule_conservation'].between(0.8, 1.2)).sum()
            },
            
            'information_preservation': {
                'count_preserved': df_valid['information_preserved'].sum(),
                'percentage': df_valid['information_preserved'].mean() * 100
            }
        }
        
        # Physics validation (if available)
        if 'physically_valid' in df_valid.columns:
            report['physics_validation'] = {
                'count_valid': df_valid['physically_valid'].sum(),
                'percentage': df_valid['physically_valid'].mean() * 100,
                'weber_number': {
                    'mean': df_valid['weber_number'].mean(),
                    'std': df_valid['weber_number'].std()
                },
                'reynolds_number': {
                    'mean': df_valid['reynolds_number'].mean(),
                    'std': df_valid['reynolds_number'].std()
                },
                'ohnesorge_number': {
                    'mean': df_valid['ohnesorge_number'].mean(),
                    'std': df_valid['ohnesorge_number'].std()
                }
            }
        
        # Save report
        report_file = self.results_base_dir / '3d_objects_master_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Saved master report to {report_file}")
        
        return report


def main():
    """Main entry point for batch generation."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("="*70)
    print("BATCH 3D OBJECT GENERATION")
    print("="*70)
    
    # Process both result directories
    result_dirs = [
        Path("precursor/results/ucdavis_fast_analysis"),
        Path("precursor/results/metabolomics_analysis")
    ]
    
    all_results = []
    
    for results_dir in result_dirs:
        if not results_dir.exists():
            logger.warning(f"Directory not found: {results_dir}")
            continue
        
        print(f"\nProcessing: {results_dir}")
        print("-"*70)
        
        generator = Batch3DObjectGenerator(results_dir)
        df = generator.generate_all(export_json=True)
        
        print(f"\nGenerated 3D objects for {len(df)} experiments")
        print(f"  Successful: {len(df[~df.get('error', pd.Series()).notna()])}")
        
        # Generate report
        report = generator.generate_master_report()
        
        print(f"\nVolume Conservation:")
        if 'volume_conservation' in report:
            vc = report['volume_conservation']
            print(f"  Mean: {vc['mean']:.2%}")
            print(f"  Std:  {vc['std']:.2%}")
            print(f"  Within 50%: {vc['within_50_percent']}/{report['successful']}")
        
        print(f"\nMolecule Conservation:")
        if 'molecule_conservation' in report:
            mc = report['molecule_conservation']
            print(f"  Mean: {mc['mean']:.2%}")
            print(f"  Std:  {mc['std']:.2%}")
            print(f"  Within 20%: {mc['within_20_percent']}/{report['successful']}")
        
        print(f"\nInformation Preservation:")
        if 'information_preservation' in report:
            ip = report['information_preservation']
            print(f"  Preserved: {ip['count_preserved']}/{report['successful']} ({ip['percentage']:.1f}%)")
        
        if 'physics_validation' in report:
            pv = report['physics_validation']
            print(f"\nPhysics Validation:")
            print(f"  Valid: {pv['count_valid']}/{report['successful']} ({pv['percentage']:.1f}%)")
            print(f"  Weber number: {pv['weber_number']['mean']:.2f} ± {pv['weber_number']['std']:.2f}")
            print(f"  Reynolds number: {pv['reynolds_number']['mean']:.2f} ± {pv['reynolds_number']['std']:.2f}")
            print(f"  Ohnesorge number: {pv['ohnesorge_number']['mean']:.4f} ± {pv['ohnesorge_number']['std']:.4f}")
        
        all_results.append(df)
    
    # Combine all results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        combined_file = Path("precursor/results/all_3d_objects_summary.csv")
        combined_df.to_csv(combined_file, index=False)
        print(f"\n{'='*70}")
        print(f"Combined summary saved to: {combined_file}")
        print(f"Total experiments processed: {len(combined_df)}")
        print("="*70)


if __name__ == "__main__":
    main()

