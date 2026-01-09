"""
Batch 3D Pipeline Processing
=============================

Batch processes multiple experiments through the 3D object pipeline.
Integrates with existing virtual MS framework.

Author: Kundai Farai Sachikonye
Date: 2025
"""

import sys
from pathlib import Path
import logging
import json
from typing import List, Dict, Any
import time

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline_3d_transformation import Pipeline3DTransformation, generate_3d_objects_for_experiment
from pipeline_3d_visualization import visualize_experiment

# Optional progress bar
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("[INFO] tqdm not available - no progress bars")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Batch3DPipelineProcessor:
    """
    Batch processes experiments through 3D object pipeline.
    
    Integrates with existing theatre/virtual MS framework.
    """
    
    def __init__(self, results_base_dir: Path):
        """
        Initialize batch processor.
        
        Args:
            results_base_dir: Base directory containing experiment results
        """
        self.results_base_dir = Path(results_base_dir)
        self.experiments: List[Path] = []
        self.results: Dict[str, Any] = {}
        
    def discover_experiments(self, pattern: str = "*") -> List[Path]:
        """
        Discover experiments in results directory.
        
        Args:
            pattern: Glob pattern for experiment directories
            
        Returns:
            List of experiment directories
        """
        logger.info(f"Discovering experiments in {self.results_base_dir}")
        
        experiments = []
        
        # Look for directories with theatre_result.json or pipeline_results.json
        for exp_dir in self.results_base_dir.glob(pattern):
            if not exp_dir.is_dir():
                continue
            
            # Check for theatre results
            has_theatre = (exp_dir / 'theatre_result.json').exists()
            has_pipeline = (exp_dir / 'pipeline_results.json').exists()
            
            # Check for stage directories
            has_stages = any([
                (exp_dir / 'stage_01_preprocessing').exists(),
                (exp_dir / 'stage_02_sentropy').exists()
            ])
            
            if has_theatre or has_pipeline or has_stages:
                experiments.append(exp_dir)
                logger.info(f"  Found: {exp_dir.name}")
        
        self.experiments = sorted(experiments)
        logger.info(f"Discovered {len(self.experiments)} experiments")
        
        return self.experiments
    
    def process_experiment(self, experiment_dir: Path) -> Dict[str, Any]:
        """
        Process a single experiment.
        
        Args:
            experiment_dir: Path to experiment directory
            
        Returns:
            Processing results
        """
        experiment_name = experiment_dir.name
        logger.info(f"Processing {experiment_name}...")
        
        start_time = time.time()
        
        try:
            # Generate 3D objects
            output_dir = experiment_dir / '3d_objects'
            objects, validation = generate_3d_objects_for_experiment(
                experiment_dir,
                output_dir
            )
            
            # Generate visualizations
            viz_dir = experiment_dir / 'visualizations'
            figures = visualize_experiment(experiment_dir, viz_dir)
            
            elapsed = time.time() - start_time
            
            result = {
                'experiment': experiment_name,
                'status': 'success',
                'n_objects': len(objects),
                'n_figures': len(figures),
                'validation': validation,
                'output_dirs': {
                    '3d_objects': str(output_dir),
                    'visualizations': str(viz_dir)
                },
                'figures': {name: str(path) for name, path in figures.items()},
                'elapsed_time': elapsed
            }
            
            logger.info(f"  ✓ Success: {len(objects)} objects, {len(figures)} figures ({elapsed:.2f}s)")
            
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"  ✗ Failed: {e}")
            
            result = {
                'experiment': experiment_name,
                'status': 'failed',
                'error': str(e),
                'elapsed_time': elapsed
            }
        
        return result
    
    def process_all(self, save_summary: bool = True) -> Dict[str, Any]:
        """
        Process all discovered experiments.
        
        Args:
            save_summary: Whether to save processing summary
            
        Returns:
            Summary of all processing results
        """
        if not self.experiments:
            logger.warning("No experiments to process. Run discover_experiments() first.")
            return {}
        
        logger.info(f"Processing {len(self.experiments)} experiments...")
        
        results = []
        
        # Use tqdm if available
        iterator = tqdm(self.experiments, desc="Processing") if TQDM_AVAILABLE else self.experiments
        
        for exp_dir in iterator:
            result = self.process_experiment(exp_dir)
            results.append(result)
            self.results[exp_dir.name] = result
        
        # Summary statistics
        n_success = sum(1 for r in results if r['status'] == 'success')
        n_failed = sum(1 for r in results if r['status'] == 'failed')
        total_time = sum(r['elapsed_time'] for r in results)
        
        summary = {
            'total_experiments': len(results),
            'successful': n_success,
            'failed': n_failed,
            'total_time': total_time,
            'results': results
        }
        
        logger.info("\n" + "="*70)
        logger.info("BATCH PROCESSING COMPLETE")
        logger.info("="*70)
        logger.info(f"Total experiments: {len(results)}")
        logger.info(f"Successful:        {n_success}")
        logger.info(f"Failed:            {n_failed}")
        logger.info(f"Total time:        {total_time:.2f}s")
        logger.info(f"Average time:      {total_time/len(results):.2f}s per experiment")
        
        # Save summary
        if save_summary:
            summary_file = self.results_base_dir / '3d_pipeline_summary.json'
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            logger.info(f"\nSummary saved: {summary_file}")
        
        return summary


def main():
    """Main entry point for batch processing."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch 3D pipeline processing')
    parser.add_argument('results_dir', type=str, 
                       help='Base directory containing experiment results')
    parser.add_argument('--pattern', type=str, default='*',
                       help='Glob pattern for experiment directories (default: *)')
    parser.add_argument('--no-summary', action='store_true',
                       help='Do not save summary JSON')
    
    args = parser.parse_args()
    
    # Create processor
    processor = Batch3DPipelineProcessor(Path(args.results_dir))
    
    # Discover experiments
    experiments = processor.discover_experiments(args.pattern)
    
    if not experiments:
        print("No experiments found!")
        return
    
    # Process all
    summary = processor.process_all(save_summary=not args.no_summary)
    
    print("\n" + "="*70)
    print("PROCESSING COMPLETE")
    print("="*70)
    print(f"\nSuccessful: {summary['successful']}/{summary['total_experiments']}")
    print(f"Failed:     {summary['failed']}/{summary['total_experiments']}")
    print(f"Total time: {summary['total_time']:.2f}s")


if __name__ == "__main__":
    # Test mode - process ucdavis experiments
    import sys
    
    if len(sys.argv) > 1:
        main()
    else:
        # Default test
        print("="*70)
        print("BATCH 3D PIPELINE PROCESSING - TEST MODE")
        print("="*70)
        
        # Process ucdavis experiments
        results_dir = Path("results/ucdavis_fast_analysis")
        
        if not results_dir.exists():
            print(f"Results directory not found: {results_dir}")
            sys.exit(1)
        
        processor = Batch3DPipelineProcessor(results_dir)
        experiments = processor.discover_experiments()
        
        if experiments:
            # Process first 3 experiments for testing
            processor.experiments = experiments[:3]
            summary = processor.process_all()
        else:
            print("No experiments found!")

