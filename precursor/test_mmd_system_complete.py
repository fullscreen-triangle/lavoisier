#!/usr/bin/env python3
"""
Complete MMD System Test
=========================

Full end-to-end test of the Molecular Maxwell Demon system.
Uses real data from: public/proteomics/BSA1.mzML
Saves all results to: results/tests/mmd_system/

Author: Kundai Sachikonye
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add paths like existing project
PRECURSOR_ROOT = Path(__file__).parent
sys.path.insert(0, str(PRECURSOR_ROOT / 'src'))

# Now import
from mmdsystem.mmd_orchestrator import MolecularMaxwellDemonSystem, MMDConfig
from core.SpectraReader import extract_mzml

# Test data path
TEST_DATA = Path("public/proteomics/BSA1.mzML")


def test_mmd_system_real_data():
    """Test complete MMD system with real proteomics data."""
    print("\n[Test] Complete MMD System Analysis")
    print("="*60)

    if not TEST_DATA.exists():
        print(f"  ✗ Test data not found: {TEST_DATA}")
        print(f"     Expected: {TEST_DATA.absolute()}")
        return pd.DataFrame(), pd.DataFrame()

    print(f"  Loading: {TEST_DATA}")

    # Initialize MMD system
    config = MMDConfig(
        enable_dynamic_learning=True,
        enable_cross_modal=False,  # Disable for simpler testing
        enable_bmd_filtering=False,  # Disable for simpler testing
        distance_threshold=0.15,
        mass_tolerance=0.5,
        max_gap_size=5
    )

    mmd = MolecularMaxwellDemonSystem(config)

    # Load spectra using YOUR extract_mzml function
    try:
        scan_info, spectra_dict, _ = extract_mzml(
            str(TEST_DATA),
            rt_range=[0, 100],
            ms1_threshold=1000,
            ms2_threshold=10
        )
        print(f"  Loaded {len(spectra_dict)} total spectra")
        print(f"  scan_info columns: {list(scan_info.columns)}")
    except Exception as e:
        print(f"  ✗ Failed to load: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(), pd.DataFrame()

    # All entries in scan_info are MS2 spectra (extract_mzml only returns MS2 info)
    # The scan_info contains: spec_index, scan_time, dda_event_idx, DDA_rank, scan_number, MS2_PR_mz
    ms2_scans = scan_info['scan_number'].tolist()
    print(f"  Found {len(ms2_scans)} MS2 spectra")
    print(f"  Processing first 20 MS2 spectra...")

    results = []
    detailed_results = []
    spectra_data = []  # Store spectra for saving

    # Create lookup by scan_number
    scan_info_lookup = scan_info.set_index('scan_number')

    for i, scan_id in enumerate(ms2_scans[:20]):

        try:
            # Get spectrum DataFrame
            if scan_id not in spectra_dict:
                continue

            spec_df = spectra_dict[scan_id]

            # Extract data
            if isinstance(spec_df, pd.DataFrame):
                if 'mz' in spec_df.columns and 'i' in spec_df.columns:
                    mz = spec_df['mz'].values
                    intensity = spec_df['i'].values
                else:
                    continue
            else:
                continue

            if len(mz) < 5:
                continue

            # Get precursor info from scan_info
            # Columns: spec_index, scan_time, dda_event_idx, DDA_rank, scan_number, MS2_PR_mz
            precursor_mz = scan_info_lookup.loc[scan_id, 'MS2_PR_mz']
            # Charge not directly available - estimate from precursor m/z and mass range
            precursor_charge = 2  # Default assumption for proteomics
            rt = scan_info_lookup.loc[scan_id, 'scan_time']

            print(f"\n  [{i+1}/20] Scan: {scan_id}")
            print(f"          Precursor: {precursor_mz:.4f} m/z, charge {precursor_charge}")
            print(f"          Peaks: {len(mz)}, RT: {rt:.2f} min")

            # Store spectrum data for saving
            for j, (m, intens) in enumerate(zip(mz, intensity)):
                spectra_data.append({
                    'scan_id': scan_id,
                    'peak_index': j,
                    'mz': float(m),
                    'intensity': float(intens),
                    'precursor_mz': float(precursor_mz),
                    'rt': float(rt)
                })

            # Analyze with MMD system
            result = mmd.analyze_spectrum(
                mz_array=mz,
                intensity_array=intensity,
                precursor_mz=precursor_mz,
                precursor_charge=precursor_charge,
                rt=rt
            )

            # Store results
            results.append({
                'scan_id': scan_id,
                'scan_index': i,
                'precursor_mz': float(precursor_mz),
                'precursor_charge': precursor_charge,
                'rt': float(rt),
                'n_peaks': len(mz),
                'sequence': result.sequence,
                'sequence_length': len(result.sequence) if result.sequence else 0,
                'confidence': float(result.confidence),
                'coverage': float(result.fragment_coverage),
                'n_gaps_filled': len(result.gap_filled_regions),
                'total_entropy': float(result.total_entropy)
            })

            # Store detailed validation scores
            for score_name, score_value in result.validation_scores.items():
                detailed_results.append({
                    'scan_id': scan_id,
                    'score_type': score_name,
                    'score_value': float(score_value) if score_value is not None else 0.0
                })

            print(f"          Result: {result.sequence}")
            print(f"          Confidence: {result.confidence:.3f}, Coverage: {result.fragment_coverage:.1%}")

        except Exception as e:
            print(f"          ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Create DataFrames
    df_results = pd.DataFrame(results)
    df_detailed = pd.DataFrame(detailed_results)
    df_spectra = pd.DataFrame(spectra_data)

    # Save results
    output_dir = PRECURSOR_ROOT / "results" / "tests" / "mmd_system"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save all CSVs
    df_results.to_csv(output_dir / f"mmd_analysis_{timestamp}.csv", index=False)
    df_detailed.to_csv(output_dir / f"mmd_validation_scores_{timestamp}.csv", index=False)
    df_spectra.to_csv(output_dir / f"spectra_data_{timestamp}.csv", index=False)

    # Save dictionary entries as CSV (instead of JSON to avoid serialization issues)
    dict_entries = []
    for symbol, entry in mmd.dictionary.entries.items():
        dict_entries.append({
            'symbol': entry.symbol,
            'name': entry.name,
            'mass': entry.mass,
            's_knowledge': entry.s_entropy_coords.s_knowledge,
            's_time': entry.s_entropy_coords.s_time,
            's_entropy': entry.s_entropy_coords.s_entropy,
            'confidence': entry.confidence,
            'discovery_method': entry.discovery_method
        })
    df_dictionary = pd.DataFrame(dict_entries)
    df_dictionary.to_csv(output_dir / f"learned_dictionary_{timestamp}.csv", index=False)

    # Generate summary statistics
    sequences_reconstructed = 0
    mean_confidence = 0.0
    mean_coverage = 0.0
    mean_sequence_length = 0.0
    high_confidence_count = 0

    if len(df_results) > 0:
        sequences_reconstructed = int((df_results['sequence'].notna() & (df_results['sequence'] != '')).sum())
        mean_confidence = float(df_results['confidence'].mean())
        mean_coverage = float(df_results['coverage'].mean())
        mean_sequence_length = float(df_results['sequence_length'].mean())
        high_confidence_count = int((df_results['confidence'] > 0.7).sum())

    # Save summary as CSV
    summary_df = pd.DataFrame([{
        'test_timestamp': timestamp,
        'test_data': str(TEST_DATA),
        'total_spectra': len(ms2_scans),
        'processed_spectra': len(df_results),
        'sequences_reconstructed': sequences_reconstructed,
        'mean_confidence': mean_confidence,
        'mean_coverage': mean_coverage,
        'mean_sequence_length': mean_sequence_length,
        'high_confidence_count': high_confidence_count,
        'dictionary_size': len(mmd.dictionary.entries),
        'config_dynamic_learning': config.enable_dynamic_learning,
        'config_cross_modal': config.enable_cross_modal,
        'config_distance_threshold': config.distance_threshold,
        'config_mass_tolerance': config.mass_tolerance,
        'config_max_gap_size': config.max_gap_size
    }])
    summary_df.to_csv(output_dir / f"summary_{timestamp}.csv", index=False)

    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"  Total MS2 spectra: {len(ms2_scans)}")
    print(f"  Processed: {len(df_results)}")
    print(f"  Sequences reconstructed: {sequences_reconstructed}")
    if len(df_results) > 0:
        print(f"  Mean confidence: {mean_confidence:.3f}")
        print(f"  Mean coverage: {mean_coverage:.1%}")
        print(f"  Mean sequence length: {mean_sequence_length:.1f}")
        print(f"  High confidence (>0.7): {high_confidence_count}")
    print(f"  Dictionary size: {len(mmd.dictionary.entries)}")
    print(f"  Spectra peaks saved: {len(df_spectra)}")
    print(f"\n  ✓ Results saved to {output_dir}")
    print(f"     - mmd_analysis_{timestamp}.csv (reconstructed sequences)")
    print(f"     - mmd_validation_scores_{timestamp}.csv (validation details)")
    print(f"     - spectra_data_{timestamp}.csv (all spectrum peaks)")
    print(f"     - learned_dictionary_{timestamp}.csv (dictionary entries)")
    print(f"     - summary_{timestamp}.csv (test summary)")

    return df_results, df_detailed


def main():
    """Run complete MMD system test."""
    print("\n" + "="*60)
    print("MOLECULAR MAXWELL DEMON SYSTEM TEST")
    print(f"Test data: {TEST_DATA}")
    print("="*60)

    try:
        df_results, df_detailed = test_mmd_system_real_data()

        if len(df_results) > 0:
            return True
        else:
            print("\n⚠ No results generated")
            return False

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
