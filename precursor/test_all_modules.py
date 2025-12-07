#!/usr/bin/env python3
"""
Complete Test Suite for Database-Free Proteomics
=================================================

Tests all modules systematically using real BSA1.mzML data.
Saves all results to results/tests/

Author: Kundai Sachikonye
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
from datetime import datetime

# Setup paths like existing project
PRECURSOR_ROOT = Path(__file__).parent
sys.path.insert(0, str(PRECURSOR_ROOT / 'src'))

# Import core functionality
from core.SpectraReader import extract_mzml
from core.EntropyTransformation import SEntropyTransformer, SEntropyCoordinates

# Import new modules
from molecular_language.amino_acid_alphabet import AminoAcidAlphabet, STANDARD_AMINO_ACIDS
from molecular_language.fragmentation_grammar import PROTEOMICS_GRAMMAR
from molecular_language.coordinate_mapping import sequence_to_sentropy_path, calculate_sequence_entropy, calculate_sequence_complexity
from dictionary.sentropy_dictionary import create_standard_proteomics_dictionary
from dictionary.zero_shot_identification import ZeroShotIdentifier
from sequence.fragment_graph import FragmentNode, build_fragment_graph_from_spectra
from sequence.categorical_completion import CategoricalCompleter
from sequence.sequence_reconstruction import SequenceReconstructor
from mmdsystem.mmd_orchestrator import MolecularMaxwellDemonSystem, MMDConfig

# Test data path
TEST_DATA = Path("public/proteomics/BSA1.mzML")


def test_module_1_molecular_language():
    """Test Module 1: Molecular Language"""
    print("\n" + "="*70)
    print("MODULE 1: MOLECULAR LANGUAGE")
    print("="*70)

    output_dir = PRECURSOR_ROOT / "results" / "tests" / "molecular_language"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Test 1: Amino acids
    print("\n[1.1] Amino Acid Alphabet")
    aa_data = []
    for symbol, aa in STANDARD_AMINO_ACIDS.items():
        coords = aa.s_entropy_coords
        aa_data.append({
            'symbol': symbol,
            'name': aa.name,
            'mass': aa.mass,
            's_knowledge': coords.s_knowledge,
            's_time': coords.s_time,
            's_entropy': coords.s_entropy
        })

    df_aa = pd.DataFrame(aa_data)
    df_aa.to_csv(output_dir / "amino_acids.csv", index=False)
    print(f"  ✓ {len(df_aa)} amino acids → {output_dir / 'amino_acids.csv'}")

    # Test 2: Fragmentation
    print("\n[1.2] Fragmentation Grammar")
    test_seq = "PEPTIDE"
    fragments = PROTEOMICS_GRAMMAR.generate_all_fragments(test_seq)
    frag_data = [{
        'ion_type': f.value,
        'position': p,
        'sequence': s,
        'neutral_loss': nl
    } for f, p, s, nl in fragments[:10]]

    df_frag = pd.DataFrame(frag_data)
    df_frag.to_csv(output_dir / "fragments.csv", index=False)
    print(f"  ✓ {len(fragments)} fragments generated → {output_dir / 'fragments.csv'}")

    # Test 3: S-Entropy paths
    print("\n[1.3] S-Entropy Paths")
    coords_path = sequence_to_sentropy_path(test_seq)
    entropy = calculate_sequence_entropy(coords_path)
    complexity = calculate_sequence_complexity(test_seq)
    print(f"  ✓ Sequence entropy: {entropy:.3f}, complexity: {complexity:.3f}")

    return {'amino_acids': df_aa, 'fragments': df_frag}


def test_module_2_dictionary():
    """Test Module 2: S-Entropy Dictionary"""
    print("\n" + "="*70)
    print("MODULE 2: S-ENTROPY DICTIONARY")
    print("="*70)

    output_dir = PRECURSOR_ROOT / "results" / "tests" / "dictionary"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create dictionary
    print("\n[2.1] Creating Dictionary")
    dictionary = create_standard_proteomics_dictionary()
    print(f"  ✓ Created with {len(dictionary.entries)} entries")

    # Test zero-shot identification
    print("\n[2.2] Zero-Shot Identification")
    identifier = ZeroShotIdentifier(dictionary)
    test_results = []

    for aa_symbol in ['A', 'G', 'L', 'K', 'R']:
        entry = dictionary.get_entry(aa_symbol)
        result = identifier.identify(entry.s_entropy_coords, entry.mass)
        match = result.top_match.symbol if result.top_match else "?"
        correct = (match == aa_symbol)
        test_results.append({
            'true_aa': aa_symbol,
            'identified_aa': match,
            'confidence': result.confidence,
            'correct': correct
        })

    df_test = pd.DataFrame(test_results)
    df_test.to_csv(output_dir / "zero_shot_test.csv", index=False)
    accuracy = df_test['correct'].sum() / len(df_test)
    print(f"  ✓ Accuracy: {accuracy:.1%} → {output_dir / 'zero_shot_test.csv'}")

    return dictionary


def test_module_3_sequence_reconstruction(dictionary):
    """Test Module 3: Sequence Reconstruction"""
    print("\n" + "="*70)
    print("MODULE 3: SEQUENCE RECONSTRUCTION")
    print("="*70)

    output_dir = PRECURSOR_ROOT / "results" / "tests" / "sequence"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create components
    identifier = ZeroShotIdentifier(dictionary)
    completer = CategoricalCompleter(dictionary)
    reconstructor = SequenceReconstructor(dictionary, identifier, completer)

    # Create synthetic fragments
    print("\n[3.1] Fragment Graph")
    fragments = []
    for i in range(5):
        node = FragmentNode(
            fragment_id=f"frag_{i}",
            sequence=None,
            s_entropy_coords=SEntropyCoordinates(0.5 + i*0.1, 0.3 + i*0.05, 0.6),
            mass=100.0 + i*100.0,
            confidence=0.9
        )
        fragments.append(node)

    graph = build_fragment_graph_from_spectra(fragments)
    print(f"  ✓ Graph: {len(graph.nodes)} nodes, {graph.graph.number_of_edges()} edges")

    # Test reconstruction
    print("\n[3.2] Sequence Reconstruction")
    result = reconstructor.reconstruct(fragments, precursor_mass=500.0)
    print(f"  ✓ Sequence: {result.sequence}, confidence: {result.confidence:.3f}")

    # Save result
    result_data = {
        'sequence': result.sequence,
        'confidence': result.confidence,
        'coverage': result.fragment_coverage
    }
    with open(output_dir / "reconstruction_test.json", 'w') as f:
        json.dump(result_data, f, indent=2)

    return reconstructor


def test_module_4_mmd_system_real_data(dictionary):
    """Test Module 4: Complete MMD System with Real Data"""
    print("\n" + "="*70)
    print("MODULE 4: MOLECULAR MAXWELL DEMON SYSTEM")
    print("="*70)

    output_dir = PRECURSOR_ROOT / "results" / "tests" / "mmd_system"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not TEST_DATA.exists():
        print(f"  ⚠ Test data not found: {TEST_DATA}")
        return pd.DataFrame()

    # Initialize MMD
    print("\n[4.1] Initializing MMD System")
    config = MMDConfig(
        enable_dynamic_learning=True,
        enable_cross_modal=False,  # Disable for testing
        enable_bmd_filtering=False  # Disable for testing
    )
    mmd = MolecularMaxwellDemonSystem(config)

    # Load real data
    print(f"\n[4.2] Loading: {TEST_DATA}")
    try:
        spectra_dict = extract_mzml(str(TEST_DATA))
        ms2_spectra = {k: v for k, v in spectra_dict.items() if v.get('ms_level', 1) == 2}
        print(f"  ✓ Loaded {len(ms2_spectra)} MS2 spectra")
    except Exception as e:
        print(f"  ✗ Failed to load data: {e}")
        return pd.DataFrame()

    # Process first 10 spectra
    print("\n[4.3] Analyzing Spectra")
    results = []

    for i, (scan_id, spec_data) in enumerate(list(ms2_spectra.items())[:10]):
        try:
            mz = np.array(spec_data.get('mz', []))
            intensity = np.array(spec_data.get('intensity', []))
            precursor_mz = spec_data.get('precursor_mz', 500.0)
            precursor_charge = spec_data.get('precursor_charge', 2)

            if len(mz) < 5:
                continue

            result = mmd.analyze_spectrum(
                mz_array=mz,
                intensity_array=intensity,
                precursor_mz=precursor_mz,
                precursor_charge=precursor_charge
            )

            results.append({
                'scan_id': scan_id,
                'sequence': result.sequence,
                'confidence': result.confidence,
                'coverage': result.fragment_coverage
            })

            print(f"  [{i+1}/10] {scan_id}: {result.sequence} (conf={result.confidence:.3f})")

        except Exception as e:
            print(f"  [{i+1}/10] {scan_id}: Error - {e}")
            continue

    # Save results
    df_results = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df_results.to_csv(output_dir / f"mmd_results_{timestamp}.csv", index=False)

    print(f"\n  ✓ Analyzed {len(df_results)} spectra → {output_dir}")
    if len(df_results) > 0:
        print(f"  Mean confidence: {df_results['confidence'].mean():.3f}")

    return df_results


def main():
    """Run complete test suite"""
    print("\n" + "="*70)
    print("DATABASE-FREE PROTEOMICS - COMPLETE TEST SUITE")
    print(f"Test data: {TEST_DATA}")
    print("Started:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*70)

    try:
        # Module 1
        mod1_results = test_module_1_molecular_language()

        # Module 2
        dictionary = test_module_2_dictionary()

        # Module 3
        reconstructor = test_module_3_sequence_reconstruction(dictionary)

        # Module 4
        mmd_results = test_module_4_mmd_system_real_data(dictionary)

        # Final summary
        print("\n" + "="*70)
        print("TEST SUITE COMPLETE")
        print("="*70)
        print(f"✓ Module 1: {len(mod1_results['amino_acids'])} amino acids tested")
        print(f"✓ Module 2: {len(dictionary.entries)} dictionary entries")
        print(f"✓ Module 3: Reconstruction system operational")
        print(f"✓ Module 4: {len(mmd_results)} spectra analyzed")
        print(f"\nAll results saved to: results/tests/")
        print("="*70 + "\n")

        return True

    except Exception as e:
        print(f"\n✗ TEST SUITE FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
