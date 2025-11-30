#!/usr/bin/env python3
"""
Test Fragmentation Network Stage Integration
==============================================

Tests the newly integrated Stage 2.5: Fragmentation Network Analysis
in the metabolomics pipeline.

This script:
1. Runs the full metabolomics pipeline with fragmentation analysis
2. Validates that fragmentation networks are built correctly
3. Checks phase-lock detection
4. Analyzes intensity-entropy correlation
5. Validates platform independence

Expected outputs:
- Fragmentation network with precursor-fragment relationships
- Phase-lock network statistics
- Intensity-entropy correlation metrics
- Platform independence scores
"""

import sys
from pathlib import Path

# Add precursor to path
sys.path.insert(0, str(Path(__file__).parent))

import logging
import json
import numpy as np

from src.pipeline.metabolomics import (
    MetabolomicsTheatre,
    run_metabolomics_analysis
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_fragmentation_stage_single_file():
    """
    Test fragmentation stage on a single mzML file.
    """
    logger.info("="*80)
    logger.info("TEST: Fragmentation Network Stage - Single File")
    logger.info("="*80)

    # Use real data file
    data_dir = Path(__file__).parent / "public" / "metabolomics"
    mzml_file = data_dir / "TG_Pos_Thermo_Orbi.mzML"

    if not mzml_file.exists():
        logger.warning(f"Test file not found: {mzml_file}")
        logger.info("Using alternative file...")
        mzml_file = data_dir / "PL_Neg_Waters_qTOF.mzML"

        if not mzml_file.exists():
            logger.error("No test files available!")
            return False

    logger.info(f"Processing: {mzml_file.name}")

    # Output directory
    output_dir = Path(__file__).parent / "results" / "fragmentation_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize theatre with fragmentation stage
    theatre = MetabolomicsTheatre(
        output_dir=output_dir,
        enable_bmd_grounding=False,  # Focus on fragmentation
        preprocessing={
            'acquisition': {
                'rt_range': [0, 50],  # First 50 min
                'ms1_threshold': 1000,
                'ms2_threshold': 10,
                'vendor': 'thermo'
            },
            'peak_detection': {
                'min_intensity': 100.0,
                'min_snr': 3.0
            }
        },
        fragmentation={
            'similarity_threshold': 0.5,
            'sigma': 0.2,
            'harmonic_tolerance': 0.01
        }
    )

    logger.info("\nExecuting pipeline with fragmentation analysis...")

    # Run pipeline
    result = theatre.observe_all_stages(input_data=mzml_file)

    # Save results
    result_file = output_dir / "fragmentation_test_result.json"
    result.save(result_file)

    logger.info(f"\n{'='*80}")
    logger.info("RESULTS SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Theatre status: {result.status.value}")
    logger.info(f"Total execution time: {result.execution_time:.2f}s")
    logger.info(f"Stages completed: {len([s for s in result.stage_results.values() if s.status.value == 'completed'])}/{len(result.stage_results)}")

    # Extract fragmentation metrics
    if 'stage_02_5_fragmentation' in result.stage_results:
        frag_result = result.stage_results['stage_02_5_fragmentation']

        logger.info(f"\n{'='*80}")
        logger.info("FRAGMENTATION NETWORK METRICS")
        logger.info(f"{'='*80}")
        logger.info(f"Status: {frag_result.status.value}")
        logger.info(f"Execution time: {frag_result.execution_time:.2f}s")

        metrics = frag_result.metrics
        logger.info(f"\nNetwork Statistics:")
        logger.info(f"  Precursors: {metrics.get('n_precursors', 0)}")
        logger.info(f"  Fragments: {metrics.get('n_fragments', 0)}")
        logger.info(f"  Edges: {metrics.get('n_edges', 0)}")
        logger.info(f"  Network density: {metrics.get('network_density', 0):.4f}")
        logger.info(f"  Average degree: {metrics.get('avg_degree', 0):.2f}")

        logger.info(f"\nPhase-Lock Analysis:")
        logger.info(f"  Phase-locks detected: {metrics.get('n_phase_locks', 0)}")
        logger.info(f"  Clustering coefficient: {metrics.get('clustering_coefficient', 0):.3f}")
        logger.info(f"  Average path length: {metrics.get('avg_path_length', 0):.2f}")
        logger.info(f"  Max degree: {metrics.get('max_degree', 0)}")

        logger.info(f"\nIntensity-Entropy Correlation:")
        logger.info(f"  Fragments analyzed: {metrics.get('n_fragments_analyzed', 0)}")
        logger.info(f"  Correlation coefficient: {metrics.get('intensity_entropy_correlation', 0):.3f}")
        logger.info(f"  Avg termination prob: {metrics.get('avg_termination_prob', 0):.3f}")

        logger.info(f"\nPlatform Independence:")
        logger.info(f"  Degree CV: {metrics.get('degree_cv', 0):.3f}")
        logger.info(f"  S-Entropy CV: {metrics.get('sentropy_cv', 0):.3f}")
        logger.info(f"  Edge weight CV: {metrics.get('edge_weight_cv', 0):.3f}")
        logger.info(f"  Independence score: {metrics.get('platform_independence_score', 0):.3f}")

        # Validation
        success = True
        if metrics.get('n_precursors', 0) == 0:
            logger.warning("WARNING: No precursors found!")
            success = False
        if metrics.get('n_edges', 0) == 0:
            logger.warning("WARNING: No network edges created!")
            success = False
        if metrics.get('n_phase_locks', 0) == 0:
            logger.warning("WARNING: No phase-locks detected!")

        return success
    else:
        logger.error("Fragmentation stage not found in results!")
        return False


def test_fragmentation_stage_comparison():
    """
    Test fragmentation stage on multiple files for platform independence.
    """
    logger.info("="*80)
    logger.info("TEST: Fragmentation Network Stage - Platform Comparison")
    logger.info("="*80)

    # Use Waters and Thermo files
    data_dir = Path(__file__).parent / "public" / "metabolomics"
    mzml_files = [
        data_dir / "PL_Neg_Waters_qTOF.mzML",
        data_dir / "TG_Pos_Thermo_Orbi.mzML"
    ]

    # Filter existing files
    existing_files = [f for f in mzml_files if f.exists()]

    if len(existing_files) < 2:
        logger.warning("Not enough files for comparison test")
        return False

    logger.info(f"Comparing {len(existing_files)} files:")
    for f in existing_files:
        logger.info(f"  - {f.name}")

    # Output directory
    output_dir = Path(__file__).parent / "results" / "fragmentation_comparison"

    # Run analysis on all files
    results = run_metabolomics_analysis(
        mzml_files=existing_files,
        output_dir=output_dir,
        enable_bmd=False,
        preprocessing={
            'acquisition': {
                'rt_range': [0, 30],  # First 30 min
                'ms1_threshold': 1000,
                'ms2_threshold': 10
            }
        },
        fragmentation={
            'similarity_threshold': 0.5,
            'sigma': 0.2,
            'harmonic_tolerance': 0.01
        }
    )

    # Compare fragmentation metrics across platforms
    logger.info(f"\n{'='*80}")
    logger.info("PLATFORM COMPARISON")
    logger.info(f"{'='*80}")

    frag_metrics = {}
    for file_name, result in results.items():
        if 'stage_02_5_fragmentation' in result.stage_results:
            frag_metrics[file_name] = result.stage_results['stage_02_5_fragmentation'].metrics

    if len(frag_metrics) < 2:
        logger.warning("Not enough fragmentation results for comparison")
        return False

    # Compare key metrics
    logger.info("\nComparison of Network Topology:")
    logger.info(f"{'File':<40} {'Density':>10} {'Avg Degree':>12} {'Phase-locks':>12}")
    logger.info("-"*80)

    for file_name, metrics in frag_metrics.items():
        density = metrics.get('network_density', 0)
        avg_degree = metrics.get('avg_degree', 0)
        phase_locks = metrics.get('n_phase_locks', 0)
        logger.info(f"{file_name:<40} {density:>10.4f} {avg_degree:>12.2f} {phase_locks:>12}")

    # Compute CV across platforms (should be low)
    densities = [m.get('network_density', 0) for m in frag_metrics.values()]
    degrees = [m.get('avg_degree', 0) for m in frag_metrics.values()]

    density_cv = np.std(densities) / np.mean(densities) if np.mean(densities) > 0 else 0
    degree_cv = np.std(degrees) / np.mean(degrees) if np.mean(degrees) > 0 else 0

    logger.info(f"\nCross-Platform Consistency:")
    logger.info(f"  Density CV: {density_cv:.3f} (lower is better)")
    logger.info(f"  Degree CV: {degree_cv:.3f} (lower is better)")

    if density_cv < 0.2 and degree_cv < 0.2:
        logger.info("  ✓ Platform independence VALIDATED (CV < 0.2)")
        return True
    else:
        logger.warning("  ⚠ Platform independence needs investigation")
        return False


if __name__ == "__main__":
    """
    Run fragmentation stage tests.
    """
    print("\n" + "="*80)
    print("FRAGMENTATION NETWORK STAGE - TEST SUITE")
    print("="*80)
    print("\nThis test validates the integration of categorical fragmentation")
    print("theory into the metabolomics pipeline.")
    print("\nExpected results:")
    print("  1. S-Entropy fragmentation networks from MS2 spectra")
    print("  2. Phase-lock network formation (tree → network)")
    print("  3. Intensity-entropy correlation validation")
    print("  4. Platform independence confirmation")
    print("="*80 + "\n")

    # Test 1: Single file
    success_1 = test_fragmentation_stage_single_file()

    print("\n" + "="*80 + "\n")

    # Test 2: Platform comparison
    success_2 = test_fragmentation_stage_comparison()

    # Summary
    print("\n" + "="*80)
    print("TEST SUITE SUMMARY")
    print("="*80)
    print(f"Test 1 (Single file): {'✓ PASS' if success_1 else '✗ FAIL'}")
    print(f"Test 2 (Platform comparison): {'✓ PASS' if success_2 else '✗ FAIL'}")
    print("="*80 + "\n")

    if success_1 and success_2:
        print("✓ All tests passed! Fragmentation stage successfully integrated.")
        sys.exit(0)
    else:
        print("⚠ Some tests failed. Review logs for details.")
        sys.exit(1)
