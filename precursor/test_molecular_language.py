#!/usr/bin/env python3
"""
Test Molecular Language Module
===============================

Tests amino acid alphabet, fragmentation grammar, and coordinate mapping.
Saves all results to results/tests/molecular_language/

Author: Kundai Sachikonye
"""

import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np

# Add paths like existing project
PRECURSOR_ROOT = Path(__file__).parent
sys.path.insert(0, str(PRECURSOR_ROOT / 'src'))

# Now import
from molecular_language.amino_acid_alphabet import AminoAcidAlphabet, STANDARD_AMINO_ACIDS
from molecular_language.fragmentation_grammar import PROTEOMICS_GRAMMAR
from molecular_language.coordinate_mapping import (
    amino_acid_to_sentropy,
    sequence_to_sentropy_path,
    calculate_sequence_entropy,
    calculate_sequence_complexity
)


def test_amino_acid_alphabet():
    """Test amino acid alphabet and S-Entropy mapping."""
    print("\n[Test 1] Amino Acid Alphabet")
    print("="*60)

    results = []

    for symbol, aa in STANDARD_AMINO_ACIDS.items():
        coords = aa.s_entropy_coords
        results.append({
            'symbol': symbol,
            'name': aa.name,
            'mass': aa.mass,
            'hydrophobicity': aa.hydrophobicity,
            'volume': aa.volume,
            'charge': aa.charge,
            'polarity': aa.polarity,
            's_knowledge': coords.s_knowledge,
            's_time': coords.s_time,
            's_entropy': coords.s_entropy
        })

        print(f"  {symbol} ({aa.name:15s}): S-Entropy ({coords.s_knowledge:.3f}, {coords.s_time:.3f}, {coords.s_entropy:.3f})")

    # Save to CSV
    df = pd.DataFrame(results)
    output_dir = PRECURSOR_ROOT / "results" / "tests" / "molecular_language"
    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_dir / "amino_acid_alphabet.csv", index=False)
    print(f"\n  ✓ Saved to {output_dir / 'amino_acid_alphabet.csv'}")

    return df


def test_fragmentation_grammar():
    """Test fragmentation grammar."""
    print("\n[Test 2] Fragmentation Grammar")
    print("="*60)

    test_sequences = [
        "PEPTIDE",
        "SEQUENCE",
        "PROTEIN",
        "SAMPLE"
    ]

    all_results = []

    for seq in test_sequences:
        print(f"\n  Sequence: {seq}")

        # Generate fragments
        fragments = PROTEOMICS_GRAMMAR.generate_all_fragments(
            sequence=seq,
            charge=2,
            include_b_ions=True,
            include_y_ions=True
        )

        print(f"    Generated {len(fragments)} fragments")

        # Calculate masses
        for ion_type, pos, frag_seq, neutral_loss in fragments[:5]:  # Show first 5
            mz = PROTEOMICS_GRAMMAR.calculate_fragment_mass(
                frag_seq, ion_type, neutral_loss, charge=1
            )

            result = {
                'parent_sequence': seq,
                'ion_type': ion_type.value,
                'position': pos,
                'fragment_sequence': frag_seq,
                'neutral_loss': neutral_loss,
                'mz': mz
            }
            all_results.append(result)

            nl_str = f"-{neutral_loss}" if neutral_loss else ""
            print(f"      {ion_type.value}{pos}{nl_str}: {frag_seq} (m/z {mz:.4f})")

    # Save to CSV
    df = pd.DataFrame(all_results)
    output_dir = PRECURSOR_ROOT / "results" / "tests" / "molecular_language"
    df.to_csv(output_dir / "fragmentation_grammar.csv", index=False)
    print(f"\n  ✓ Saved to {output_dir / 'fragmentation_grammar.csv'}")

    return df


def test_sequence_to_sentropy():
    """Test sequence to S-Entropy path conversion."""
    print("\n[Test 3] Sequence to S-Entropy Path")
    print("="*60)

    test_sequences = [
        "PEPTIDE",
        "SAMPLE",
        "PROTEIN"
    ]

    all_results = []

    for seq in test_sequences:
        print(f"\n  Sequence: {seq}")

        # Convert to S-Entropy path
        coords_path = sequence_to_sentropy_path(seq)

        # Calculate entropy and complexity
        entropy = calculate_sequence_entropy(coords_path)
        complexity = calculate_sequence_complexity(seq)

        print(f"    Entropy: {entropy:.3f}")
        print(f"    Complexity: {complexity:.3f}")

        # Save each residue's coordinates
        for i, (aa, coords) in enumerate(zip(seq, coords_path)):
            all_results.append({
                'sequence': seq,
                'position': i,
                'amino_acid': aa,
                's_knowledge': coords.s_knowledge,
                's_time': coords.s_time,
                's_entropy': coords.s_entropy,
                'sequence_entropy': entropy,
                'sequence_complexity': complexity
            })

    # Save to CSV
    df = pd.DataFrame(all_results)
    output_dir = PRECURSOR_ROOT / "results" / "tests" / "molecular_language"
    df.to_csv(output_dir / "sequence_sentropy_paths.csv", index=False)
    print(f"\n  ✓ Saved to {output_dir / 'sequence_sentropy_paths.csv'}")

    return df


def main():
    """Run all molecular language tests."""
    print("\n" + "="*60)
    print("MOLECULAR LANGUAGE MODULE TEST")
    print("="*60)

    results = {}

    try:
        results['alphabet'] = test_amino_acid_alphabet()
        results['grammar'] = test_fragmentation_grammar()
        results['sentropy_paths'] = test_sequence_to_sentropy()

        # Summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print(f"  Amino acids tested: {len(results['alphabet'])}")
        print(f"  Fragments generated: {len(results['grammar'])}")
        print(f"  S-Entropy paths: {len(results['sentropy_paths'])}")
        print(f"\n  ✓ All results saved to results/tests/molecular_language/")

        return True

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
