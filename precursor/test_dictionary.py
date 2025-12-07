#!/usr/bin/env python3
"""
Test S-Entropy Dictionary Module
=================================

Tests dictionary creation, zero-shot identification, and dynamic learning.
Uses real data from: precursor/public/proteomics/BSA1.mzML
Saves all results to: results/tests/dictionary/

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
from dictionary.sentropy_dictionary import create_standard_proteomics_dictionary
from dictionary.zero_shot_identification import ZeroShotIdentifier
from core.EntropyTransformation import SEntropyTransformer, SEntropyCoordinates

try:
    from pyteomics import mzml
    HAS_PYTEOMICS = True
except ImportError:
    HAS_PYTEOMICS = False
    print("[Warning] pyteomics not installed - will skip real data test")

# Test data path
TEST_DATA = Path("public/proteomics/BSA1.mzML")


def test_dictionary_creation():
    """Test standard proteomics dictionary creation."""
    print("\n[Test 1] Dictionary Creation")
    print("="*60)

    dictionary = create_standard_proteomics_dictionary()

    results = []
    for symbol, entry in dictionary.entries.items():
        results.append({
            'symbol': entry.symbol,
            'name': entry.name,
            'mass': entry.mass,
            's_knowledge': entry.s_entropy_coords.s_knowledge,
            's_time': entry.s_entropy_coords.s_time,
            's_entropy': entry.s_entropy_coords.s_entropy,
            'discovery_method': entry.discovery_method,
            'confidence': entry.confidence
        })
        print(f"  {entry.symbol}: {entry.name} (mass={entry.mass:.3f})")

    df = pd.DataFrame(results)
    output_dir = PRECURSOR_ROOT / "results" / "tests" / "dictionary"
    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_dir / "dictionary_entries.csv", index=False)

    # Save dictionary to JSON
    dictionary.save(str(output_dir / "standard_dictionary.json"))

    print(f"\n  ✓ Created dictionary with {len(results)} entries")
    print(f"  ✓ Saved to {output_dir}")

    return dictionary, df


def test_zero_shot_identification(dictionary):
    """Test zero-shot identification with synthetic queries."""
    print("\n[Test 2] Zero-Shot Identification")
    print("="*60)

    # Test with known amino acids (should identify correctly)
    test_amino_acids = ['A', 'G', 'L', 'K', 'R', 'D', 'E', 'F', 'W', 'Y']

    identifier = ZeroShotIdentifier(dictionary)

    results = []

    for aa_symbol in test_amino_acids:
        # Get true entry
        true_entry = dictionary.get_entry(aa_symbol)
        if not true_entry:
            continue

        # Identify using S-Entropy coords (with small noise)
        noise = np.random.normal(0, 0.02, 3)
        noisy_coords = SEntropyCoordinates(
            s_knowledge=true_entry.s_entropy_coords.s_knowledge + noise[0],
            s_time=true_entry.s_entropy_coords.s_time + noise[1],
            s_entropy=true_entry.s_entropy_coords.s_entropy + noise[2]
        )

        # Identify
        result = identifier.identify(noisy_coords, true_entry.mass)

        match_symbol = result.top_match.symbol if result.top_match else "?"
        correct = (match_symbol == aa_symbol)

        results.append({
            'true_aa': aa_symbol,
            'true_mass': true_entry.mass,
            'identified_aa': match_symbol,
            'confidence': result.confidence,
            'distance': result.distance,
            'is_novel': result.is_novel,
            'correct': correct
        })

        status = "✓" if correct else "✗"
        print(f"  {status} {aa_symbol} -> {match_symbol} (conf={result.confidence:.3f}, dist={result.distance:.3f})")

    df = pd.DataFrame(results)
    output_dir = PRECURSOR_ROOT / "results" / "tests" / "dictionary"
    df.to_csv(output_dir / "zero_shot_identification.csv", index=False)

    accuracy = df['correct'].sum() / len(df)
    print(f"\n  Accuracy: {accuracy:.1%}")
    print(f"  ✓ Saved to {output_dir / 'zero_shot_identification.csv'}")

    return df


def test_with_real_data(dictionary):
    """Test with real MS data from BSA1.mzML."""
    print("\n[Test 3] Real Data Analysis")
    print("="*60)

    if not HAS_PYTEOMICS:
        print("  ⚠ Skipping - pyteomics not installed")
        return pd.DataFrame()

    if not TEST_DATA.exists():
        print(f"  ⚠ Test data not found: {TEST_DATA}")
        return pd.DataFrame()

    print(f"  Loading: {TEST_DATA}")

    # Load spectra
    spectra = list(mzml.MzML(str(TEST_DATA)))
    print(f"  Loaded {len(spectra)} spectra")

    # Process first 10 MS2 spectra
    transformer = SEntropyTransformer()
    identifier = ZeroShotIdentifier(dictionary)

    results = []
    n_processed = 0

    for spec in spectra[:100]:  # First 100 scans
        if spec.get('ms level', 1) != 2:
            continue

        # Extract data
        mz = np.array(spec['m/z array'])
        intensity = np.array(spec['intensity array'])

        if len(mz) < 5:
            continue

        # Get precursor info
        precursor_info = spec.get('precursorList', {}).get('precursor', [{}])[0]
        selected_ion = precursor_info.get('selectedIonList', {}).get('selectedIon', [{}])[0]
        precursor_mz = selected_ion.get('selected ion m/z', 500.0)

        # Transform to S-Entropy
        coords_list, coord_matrix = transformer.transform_spectrum(
            mz_array=mz,
            intensity_array=intensity,
            precursor_mz=precursor_mz
        )

        # Identify top 5 peaks
        top_indices = np.argsort(intensity)[-5:]

        for idx in top_indices:
            if idx < len(coords_list):
                coords = coords_list[idx]
                mass = mz[idx]

                # Identify
                result = identifier.identify(coords, mass)

                if result.top_match:
                    results.append({
                        'scan_id': spec.get('id', f'scan_{n_processed}'),
                        'peak_mz': mz[idx],
                        'peak_intensity': intensity[idx],
                        'identified_aa': result.top_match.symbol,
                        'identified_name': result.top_match.name,
                        'confidence': result.confidence,
                        'distance': result.distance,
                        'is_novel': result.is_novel
                    })

        n_processed += 1
        if n_processed >= 10:
            break

    df = pd.DataFrame(results)
    output_dir = PRECURSOR_ROOT / "results" / "tests" / "dictionary"
    df.to_csv(output_dir / "real_data_identifications.csv", index=False)

    print(f"\n  Processed {n_processed} MS2 spectra")
    print(f"  Identified {len(df)} peaks")
    if len(df) > 0:
        print(f"  Mean confidence: {df['confidence'].mean():.3f}")
        print(f"  Novel entities: {df['is_novel'].sum()}")
    print(f"  ✓ Saved to {output_dir / 'real_data_identifications.csv'}")

    return df


def main():
    """Run all dictionary tests."""
    print("\n" + "="*60)
    print("S-ENTROPY DICTIONARY MODULE TEST")
    print(f"Test data: {TEST_DATA}")
    print("="*60)

    results = {}

    try:
        # Test 1: Dictionary creation
        dictionary, dict_df = test_dictionary_creation()
        results['dictionary'] = dict_df

        # Test 2: Zero-shot identification
        zeroshot_df = test_zero_shot_identification(dictionary)
        results['zero_shot'] = zeroshot_df

        # Test 3: Real data
        real_data_df = test_with_real_data(dictionary)
        results['real_data'] = real_data_df

        # Summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print(f"  Dictionary entries: {len(results['dictionary'])}")
        print(f"  Zero-shot tests: {len(results['zero_shot'])}")
        print(f"  Real data identifications: {len(results['real_data'])}")
        print(f"\n  ✓ All results saved to results/tests/dictionary/")

        return True

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
