#!/usr/bin/env python3
"""
Database-Free Proteomics Demonstration
======================================

Demonstrates the complete Molecular Maxwell Demon system for
database-free peptide sequencing.

Usage:
    python demo_database_free_proteomics.py [--data PATH] [--output DIR]

Author: Kundai Sachikonye
"""

import numpy as np
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent / 'src'))

from mmdsystem.mmd_orchestrator import MolecularMaxwellDemonSystem, MMDConfig
from dictionary.sentropy_dictionary import create_standard_proteomics_dictionary
from molecular_language.fragmentation_grammar import PROTEOMICS_GRAMMAR


def generate_synthetic_spectrum(
    sequence: str,
    precursor_charge: int = 2,
    noise_level: float = 0.1
) -> tuple:
    """
    Generate synthetic MS/MS spectrum for a peptide sequence.

    Args:
        sequence: Peptide sequence
        precursor_charge: Precursor charge state
        noise_level: Noise level [0, 1]

    Returns:
        (mz_array, intensity_array)
    """
    print(f"\n[Demo] Generating synthetic spectrum for: {sequence}")

    # Generate theoretical fragments
    fragments = PROTEOMICS_GRAMMAR.generate_all_fragments(
        sequence=sequence,
        charge=precursor_charge,
        include_b_ions=True,
        include_y_ions=True
    )

    mz_list = []
    intensity_list = []

    for ion_type, pos, frag_seq, neutral_loss in fragments:
        # Calculate m/z
        mz = PROTEOMICS_GRAMMAR.calculate_fragment_mass(
            frag_seq, ion_type, neutral_loss, charge=1
        )

        # Simulate intensity (decay with position)
        intensity = 1000.0 * np.exp(-0.05 * pos) * np.random.uniform(0.5, 1.5)

        mz_list.append(mz)
        intensity_list.append(intensity)

    # Add noise peaks
    n_noise = int(len(mz_list) * noise_level)
    for _ in range(n_noise):
        noise_mz = np.random.uniform(100, 2000)
        noise_intensity = np.random.uniform(10, 100)
        mz_list.append(noise_mz)
        intensity_list.append(noise_intensity)

    # Sort by m/z
    indices = np.argsort(mz_list)
    mz_array = np.array(mz_list)[indices]
    intensity_array = np.array(intensity_list)[indices]

    print(f"[Demo] Generated {len(mz_array)} peaks ({len(fragments)} theoretical, {n_noise} noise)")

    return mz_array, intensity_array


def demo_single_peptide():
    """Demonstrate analysis of a single peptide."""
    print("\n" + "="*70)
    print("DEMO 1: Single Peptide Analysis")
    print("="*70)

    # Test peptide
    true_sequence = "PEPTIDE"
    precursor_charge = 2

    # Generate synthetic spectrum
    mz_array, intensity_array = generate_synthetic_spectrum(
        sequence=true_sequence,
        precursor_charge=precursor_charge,
        noise_level=0.2
    )

    # Calculate precursor m/z
    from molecular_language.amino_acid_alphabet import STANDARD_AMINO_ACIDS
    precursor_mass = sum(STANDARD_AMINO_ACIDS[aa].mass for aa in true_sequence if aa in STANDARD_AMINO_ACIDS)
    precursor_mz = (precursor_mass + precursor_charge * 1.008) / precursor_charge

    print(f"\n[Demo] True sequence: {true_sequence}")
    print(f"[Demo] Precursor m/z: {precursor_mz:.4f}")

    # Initialize MMD system
    config = MMDConfig(
        enable_dynamic_learning=True,
        enable_cross_modal=True,
        enable_bmd_filtering=False
    )

    mmd_system = MolecularMaxwellDemonSystem(config)

    # Analyze
    result = mmd_system.analyze_spectrum(
        mz_array=mz_array,
        intensity_array=intensity_array,
        precursor_mz=precursor_mz,
        precursor_charge=precursor_charge,
        rt=10.5
    )

    # Display results
    print("\n" + "="*70)
    print("RECONSTRUCTION RESULTS")
    print("="*70)
    print(f"True sequence:          {true_sequence}")
    print(f"Reconstructed sequence: {result.sequence}")
    print(f"Confidence:             {result.confidence:.3f}")
    print(f"Fragment coverage:      {result.fragment_coverage:.1%}")
    print(f"Gaps filled:            {len(result.gap_filled_regions)}")
    print(f"Total S-Entropy:        {result.total_entropy:.3f}")

    # Validation scores
    print(f"\nValidation Scores:")
    for key, value in result.validation_scores.items():
        print(f"  {key}: {value:.3f}")

    # Check if correct
    if result.sequence == true_sequence:
        print(f"\n✓ PERFECT MATCH!")
    else:
        # Calculate similarity
        from difflib import SequenceMatcher
        similarity = SequenceMatcher(None, true_sequence, result.sequence).ratio()
        print(f"\n~ Sequence similarity: {similarity:.1%}")


def demo_batch_peptides():
    """Demonstrate batch analysis of multiple peptides."""
    print("\n" + "="*70)
    print("DEMO 2: Batch Peptide Analysis")
    print("="*70)

    # Test peptides (common tryptic peptides)
    test_sequences = [
        "PEPTIDE",
        "SEQUENCE",
        "SAMPLE",
        "ANALYSIS",
        "PROTEIN"
    ]

    print(f"\n[Demo] Analyzing {len(test_sequences)} peptides...")

    # Generate spectra
    spectra = []
    for seq in test_sequences:
        mz, intensity = generate_synthetic_spectrum(seq, precursor_charge=2, noise_level=0.15)

        from molecular_language.amino_acid_alphabet import STANDARD_AMINO_ACIDS
        mass = sum(STANDARD_AMINO_ACIDS[aa].mass for aa in seq if aa in STANDARD_AMINO_ACIDS)
        precursor_mz = (mass + 2 * 1.008) / 2

        spectra.append((mz, intensity, precursor_mz, 2, None))

    # Initialize MMD system
    config = MMDConfig(enable_dynamic_learning=True)
    mmd_system = MolecularMaxwellDemonSystem(config)

    # Batch analyze
    results = mmd_system.batch_analyze(spectra)

    # Display results
    print("\n" + "="*70)
    print("BATCH RESULTS")
    print("="*70)
    print(f"{'True':<15} {'Reconstructed':<15} {'Conf':<8} {'Coverage':<10}")
    print("-"*70)

    for i, (true_seq, result) in enumerate(zip(test_sequences, results)):
        print(f"{true_seq:<15} {result.sequence:<15} {result.confidence:.3f}   {result.fragment_coverage:.1%}")


def demo_from_real_data(data_path: str):
    """Demonstrate analysis using real MS data."""
    print("\n" + "="*70)
    print("DEMO 3: Real Data Analysis")
    print("="*70)

    from pipeline.database_free_proteomics import create_database_free_pipeline

    # Run pipeline
    reconstruction_results, summary = create_database_free_pipeline(
        data_path=data_path,
        output_dir="results/database_free_demo",
        mmd_config=MMDConfig(
            enable_dynamic_learning=True,
            enable_cross_modal=True
        )
    )

    # Display top results
    if not summary.empty:
        print("\nTop 10 Reconstructions:")
        top_results = summary.nlargest(10, 'confidence')
        print(top_results[['scan_id', 'sequence', 'confidence', 'coverage']])


def main():
    """Main demonstration script."""
    parser = argparse.ArgumentParser(
        description="Database-Free Proteomics Demonstration"
    )
    parser.add_argument(
        '--data',
        type=str,
        help='Path to real MS data file (optional)'
    )
    parser.add_argument(
        '--demo',
        type=str,
        choices=['single', 'batch', 'real', 'all'],
        default='all',
        help='Which demo to run'
    )

    args = parser.parse_args()

    print("\n" + "="*70)
    print("DATABASE-FREE PROTEOMICS DEMONSTRATION")
    print("Molecular Maxwell Demon System")
    print("="*70)

    # Run demos
    if args.demo in ['single', 'all']:
        demo_single_peptide()

    if args.demo in ['batch', 'all']:
        demo_batch_peptides()

    if args.demo == 'real' and args.data:
        demo_from_real_data(args.data)
    elif args.demo == 'real':
        print("\n[Demo] Real data demo requires --data argument")

    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
