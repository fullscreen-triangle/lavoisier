"""
Test Single Experiment
======================

Simple test script to validate a single experiment.
Use this to test the pipeline before running full validation.
"""

import sys
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Import from current directory
sys.path.insert(0, str(Path(__file__).parent))

try:
    from pipeline_3d_objects import generate_pipeline_objects_for_experiment
    from visualize_3d_pipeline import visualize_experiment
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.info("Trying alternative import method...")
    from src.validation.pipeline_3d_objects import generate_pipeline_objects_for_experiment
    from src.validation.visualize_3d_pipeline import visualize_experiment


def test_single_experiment(experiment_path: str):
    """
    Test 3D object generation for a single experiment.
    
    Args:
        experiment_path: Path to experiment directory
    """
    experiment_dir = Path(experiment_path)
    
    if not experiment_dir.exists():
        logger.error(f"Experiment directory not found: {experiment_dir}")
        return False
    
    logger.info(f"Testing experiment: {experiment_dir.name}")
    logger.info("="*70)
    
    # Check required files
    stage_01 = experiment_dir / 'stage_01_preprocessing'
    if not stage_01.exists():
        logger.error(f"stage_01_preprocessing not found in {experiment_dir}")
        return False
    
    ms1_file = stage_01 / 'ms1_xic.csv'
    sentropy_file = experiment_dir / 'stage_02_sentropy' / 'sentropy_features.csv'
    
    logger.info("Checking required files:")
    logger.info(f"  ms1_xic.csv: {'✓' if ms1_file.exists() else '✗'}")
    logger.info(f"  sentropy_features.csv: {'✓' if sentropy_file.exists() else '✗'}")
    
    if not ms1_file.exists() or not sentropy_file.exists():
        logger.error("Required files missing!")
        return False
    
    # Generate 3D objects
    logger.info("\nGenerating 3D objects...")
    logger.info("-"*70)
    
    output_dir = experiment_dir / '3d_objects'
    
    try:
        objects, validation = generate_pipeline_objects_for_experiment(
            experiment_dir,
            output_dir
        )
        
        logger.info(f"\n✓ Successfully generated {len(objects)} 3D objects")
        
        # Print validation results
        logger.info("\nValidation Results:")
        logger.info("="*70)
        logger.info(f"Volume Conservation: {validation['conservation_ratio']:.2%}")
        logger.info(f"Molecule Conservation: {validation['molecule_ratio']:.2%}")
        logger.info(f"Information Preserved: {validation['information_preserved']}")
        
        # Print object details
        logger.info("\nGenerated Objects:")
        logger.info("-"*70)
        for stage, obj in objects.items():
            logger.info(f"{stage:15s} → {obj.shape:20s} "
                       f"N={obj.molecule_count:6d} "
                       f"T={obj.thermo.temperature:.4f}")
        
        # Check droplet physics
        if 'droplet' in objects:
            droplet = objects['droplet']
            if 'physics_validation' in droplet.data:
                phys = droplet.data['physics_validation']
                logger.info("\nPhysics Validation:")
                logger.info("-"*70)
                logger.info(f"Weber Number:     {phys['weber_number']:.2f}")
                logger.info(f"Reynolds Number:  {phys['reynolds_number']:.2f}")
                logger.info(f"Ohnesorge Number: {phys['ohnesorge_number']:.4f}")
                logger.info(f"Physically Valid: {phys['physically_valid']}")
        
        # Try to generate visualizations
        logger.info("\nGenerating visualizations...")
        logger.info("-"*70)
        
        try:
            viz_dir = experiment_dir / 'visualizations'
            visualize_experiment(experiment_dir, viz_dir)
            logger.info(f"✓ Visualizations saved to {viz_dir}")
        except Exception as e:
            logger.warning(f"Could not generate visualizations: {e}")
            logger.info("(This is optional - 3D objects were generated successfully)")
        
        logger.info("\n" + "="*70)
        logger.info("TEST PASSED ✓")
        logger.info("="*70)
        
        return True
        
    except Exception as e:
        logger.error(f"\n✗ Error generating 3D objects: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("="*70)
    print(" SINGLE EXPERIMENT TEST")
    print("="*70)
    print()
    
    # Default test experiment
    default_exp = "results/ucdavis_fast_analysis/A_M3_negPFP_03"
    
    if len(sys.argv) > 1:
        experiment_path = sys.argv[1]
    else:
        experiment_path = default_exp
        print(f"Using default experiment: {experiment_path}")
        print(f"(You can specify a different path as argument)")
        print()
    
    success = test_single_experiment(experiment_path)
    
    if success:
        print("\nTest completed successfully!")
        print("\nNext steps:")
        print("  1. Check the generated 3d_objects/ directory")
        print("  2. Review the JSON files")
        print("  3. If visualizations were generated, check visualizations/ directory")
        print("  4. Run full validation: python -m src.validation.run_validation")
    else:
        print("\nTest failed!")
        print("\nTroubleshooting:")
        print("  1. Check that the experiment directory exists")
        print("  2. Verify required files are present (ms1_xic.csv, sentropy_features.csv)")
        print("  3. Check the error messages above")
    
    sys.exit(0 if success else 1)

