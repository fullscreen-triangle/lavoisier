"""
Test Batch Processing (Small Sample)
=====================================

Test batch processing on a small sample of experiments.
"""

import sys
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Import
sys.path.insert(0, str(Path(__file__).parent))

from pipeline_3d_objects import generate_pipeline_objects_for_experiment


def test_batch_small(results_dir: str, max_experiments: int = 3):
    """
    Test batch processing on a small sample.
    
    Args:
        results_dir: Path to results directory
        max_experiments: Maximum number of experiments to process
    """
    results_dir = Path(results_dir)
    
    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return False
    
    # Discover experiments
    experiments = []
    for item in results_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            stage_01 = item / 'stage_01_preprocessing'
            if stage_01.exists():
                experiments.append(item)
    
    experiments = sorted(experiments)[:max_experiments]
    
    logger.info(f"Found {len(experiments)} experiments to process")
    logger.info("="*70)
    
    results = []
    
    for i, exp_dir in enumerate(experiments, 1):
        logger.info(f"\n[{i}/{len(experiments)}] Processing: {exp_dir.name}")
        logger.info("-"*70)
        
        try:
            output_dir = exp_dir / '3d_objects'
            objects, validation = generate_pipeline_objects_for_experiment(
                exp_dir,
                output_dir
            )
            
            result = {
                'experiment': exp_dir.name,
                'success': True,
                'n_objects': len(objects),
                'volume_conservation': validation['conservation_ratio'],
                'molecule_conservation': validation['molecule_ratio'],
                'information_preserved': validation['information_preserved']
            }
            
            # Get droplet physics if available
            if 'droplet' in objects:
                droplet = objects['droplet']
                if 'physics_validation' in droplet.data:
                    phys = droplet.data['physics_validation']
                    result['weber_number'] = phys['weber_number']
                    result['reynolds_number'] = phys['reynolds_number']
                    result['ohnesorge_number'] = phys['ohnesorge_number']
                    result['physically_valid'] = phys['physically_valid']
            
            results.append(result)
            
            logger.info(f"  Volume conservation: {validation['conservation_ratio']:.2%}")
            logger.info(f"  Molecule conservation: {validation['molecule_ratio']:.2%}")
            logger.info(f"  Success!")
            
        except Exception as e:
            logger.error(f"  Error: {e}")
            results.append({
                'experiment': exp_dir.name,
                'success': False,
                'error': str(e)
            })
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info(" BATCH TEST SUMMARY")
    logger.info("="*70)
    
    successful = sum(1 for r in results if r.get('success', False))
    logger.info(f"\nProcessed: {len(results)} experiments")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {len(results) - successful}")
    
    if successful > 0:
        # Calculate averages
        vol_cons = [r['volume_conservation'] for r in results if 'volume_conservation' in r]
        mol_cons = [r['molecule_conservation'] for r in results if 'molecule_conservation' in r]
        
        if vol_cons:
            logger.info(f"\nAverage Volume Conservation: {sum(vol_cons)/len(vol_cons):.2%}")
        if mol_cons:
            logger.info(f"Average Molecule Conservation: {sum(mol_cons)/len(mol_cons):.2%}")
        
        # Physics validation
        phys_valid = [r for r in results if r.get('physically_valid')]
        if phys_valid:
            logger.info(f"\nPhysically Valid: {len(phys_valid)}/{successful}")
    
    logger.info("\n" + "="*70)
    
    return successful == len(results)


if __name__ == "__main__":
    print("="*70)
    print(" BATCH TEST (SMALL SAMPLE)")
    print("="*70)
    print()
    
    # Default: process 3 experiments from ucdavis
    default_dir = "results/ucdavis_fast_analysis"
    max_exp = 3
    
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        results_dir = default_dir
        print(f"Using default directory: {results_dir}")
        print(f"Processing first {max_exp} experiments")
        print()
    
    if len(sys.argv) > 2:
        max_exp = int(sys.argv[2])
    
    success = test_batch_small(results_dir, max_exp)
    
    if success:
        print("\nBatch test completed successfully!")
        print("\nNext steps:")
        print("  1. Review the generated 3d_objects/ directories")
        print("  2. Check validation metrics")
        print("  3. Run full validation on all experiments")
    else:
        print("\nSome experiments failed - check the logs above")
    
    sys.exit(0 if success else 1)

