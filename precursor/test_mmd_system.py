#!/usr/bin/env python3
"""
MMD System Integration Test
============================

Quick test to verify all modules load and integrate correctly.

Author: Kundai Sachikonye
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / 'src'))

def test_imports():
    """Test that all modules import correctly."""
    print("\n[Test] Testing module imports...")

    try:
        # Module 1: Molecular Language
        from molecular_language import (
            AminoAcid, AminoAcidAlphabet, STANDARD_AMINO_ACIDS,
            FragmentationRule, MolecularGrammar, PROTEOMICS_GRAMMAR,
            amino_acid_to_sentropy, sequence_to_sentropy_path
        )
        print("  ✓ molecular_language")

        # Module 2: Dictionary
        from dictionary import (
            DictionaryEntry, EquivalenceClass,
            SEntropyDictionary, create_standard_proteomics_dictionary,
            ZeroShotIdentifier, IdentificationResult
        )
        print("  ✓ dictionary")

        # Module 3: Sequence Reconstruction
        from sequence import (
            FragmentNode, FragmentGraph,
            CategoricalCompleter, GapFiller,
            SequenceReconstructor, ReconstructionResult
        )
        print("  ✓ sequence")

        # Module 4: MMD System
        from mmdsystem import (
            MolecularMaxwellDemonSystem, MMDConfig,
            StrategicIntelligence, MiracleEngine,
            SemanticNavigator, CrossModalValidator
        )
        print("  ✓ mmdsystem")

        # Module 5: Pipeline Integration
        from pipeline.database_free_proteomics import (
            DatabaseFreeProteomicsProcess,
            create_database_free_pipeline
        )
        print("  ✓ pipeline integration")

        print("\n[Test] All imports successful! ✓")
        return True

    except Exception as e:
        print(f"\n[Test] Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_basic_functionality():
    """Test basic functionality of key components."""
    print("\n[Test] Testing basic functionality...")

    try:
        from dictionary import create_standard_proteomics_dictionary
        from mmdsystem import MolecularMaxwellDemonSystem, MMDConfig
        import numpy as np

        # Test 1: Dictionary creation
        dictionary = create_standard_proteomics_dictionary()
        assert len(dictionary.entries) == 20, "Should have 20 amino acids"
        print("  ✓ Dictionary creation")

        # Test 2: MMD system initialization
        config = MMDConfig(
            enable_dynamic_learning=False,
            enable_cross_modal=False,
            enable_bmd_filtering=False
        )
        mmd = MolecularMaxwellDemonSystem(config)
        print("  ✓ MMD system initialization")

        # Test 3: Simple spectrum analysis
        mz = np.array([100.0, 200.0, 300.0])
        intensity = np.array([1000.0, 500.0, 200.0])

        result = mmd.analyze_spectrum(
            mz_array=mz,
            intensity_array=intensity,
            precursor_mz=500.0,
            precursor_charge=2
        )

        print("  ✓ Spectrum analysis")
        print(f"    Result: sequence='{result.sequence}', confidence={result.confidence:.3f}")

        print("\n[Test] Basic functionality tests passed! ✓")
        return True

    except Exception as e:
        print(f"\n[Test] Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("MMD SYSTEM INTEGRATION TEST")
    print("="*60)

    success = True

    # Test imports
    if not test_imports():
        success = False

    # Test functionality
    if not test_basic_functionality():
        success = False

    print("\n" + "="*60)
    if success:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("="*60 + "\n")

    return success


if __name__ == '__main__':
    import sys
    success = main()
    sys.exit(0 if success else 1)
