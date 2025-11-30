#!/usr/bin/env python3
"""
Simple Virtual Mass Spectrometry - Step by Step
================================================

Starting SIMPLE with MS1 precursor ions only:
1. Load MS1 data (precursor m/z + retention times)
2. Convert to S-Entropy coordinates using existing infrastructure
3. Create virtual instrument projections
4. Save detailed results at every step

NO MS2 fragments yet - just precursor ions!
Uses existing SpectraReader, EntropyTransformation, etc.

Author: Kundai Farai Sachikonye
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import json

# Add precursor root to path
precursor_root = Path(__file__).parent
sys.path.insert(0, str(precursor_root))

from src.core.SpectraReader import extract_mzml
from src.core.EntropyTransformation import SEntropyTransformer
from src.virtual import VirtualMassSpecEnsemble


def test_ms1_precursors_simple():
    """
    Test with MS1 precursor ions only - SIMPLE approach

    Steps:
    1. Extract MS1 precursor ions with RT
    2. Convert to S-Entropy coordinates
    3. Create virtual instrument projections
    4. Save results at each step
    """
    print("\n" + "="*80)
    print("SIMPLE VIRTUAL SPECTROMETRY - MS1 PRECURSORS ONLY")
    print("="*80)

    # STEP 1: Load MS1 data
    print("\n[STEP 1] Loading MS1 precursor data...")
    mzml_file = precursor_root / "public" / "metabolomics" / "PL_Neg_Waters_qTOF.mzML"

    if not mzml_file.exists():
        print(f"  ⚠ File not found: {mzml_file}")
        print("  Creating synthetic MS1 data for demo...")

        # Create synthetic MS1 data
        ms1_data = pd.DataFrame({
            'rt': [10.5, 10.6, 10.7, 10.8],
            'mz': [256.2634, 385.1923, 512.3456, 678.4567],
            'intensity': [1e6, 8e5, 6e5, 4e5]
        })

        print(f"  ✓ Created {len(ms1_data)} synthetic MS1 precursors")
    else:
        # Load real MS1 data
        print(f"  Reading: {mzml_file.name}")
        scan_info, spectra_dict, ms1_xic = extract_mzml(
            str(mzml_file),
            rt_range=[10, 15],  # 5 minute window
            ms1_threshold=1000,
            vendor='waters'
        )

        # Extract MS1 precursors (DDA_rank == 0)
        ms1_scans = scan_info[scan_info['DDA_rank'] == 0].copy()

        if len(ms1_scans) == 0:
            print("  ⚠ No MS1 scans found, using XIC data...")
            # Use MS1 XIC instead (has columns: mz, i, rt, spec_idx, dda_event_idx, DDA_rank, scan_number)
            if ms1_xic.empty:
                print("  ⚠ MS1 XIC also empty, creating synthetic data...")
                ms1_data = pd.DataFrame({
                    'rt': [10.5, 11.0, 11.5, 12.0],
                    'mz': [256.2634, 385.1923, 512.3456, 678.4567],
                    'intensity': [1e6, 8e5, 6e5, 4e5]
                })
            else:
                ms1_data = ms1_xic.copy()
                ms1_data = ms1_data.rename(columns={'i': 'intensity'})
                # ms1_xic already has 'rt' and 'mz' columns
        else:
            # Extract from scan_info (has columns: dda_event_idx, spec_index, scan_time, DDA_rank, scan_number, MS2_PR_mz)
            ms1_data = ms1_scans[['scan_time', 'scan_number']].copy()
            ms1_data = ms1_data.rename(columns={'scan_time': 'rt', 'scan_number': 'scan'})

            # Get m/z and intensity from actual spectra
            mz_list = []
            intensity_list = []
            for idx, row in ms1_data.iterrows():
                scan_idx = ms1_scans.loc[idx, 'spec_index']
                if scan_idx in spectra_dict:
                    spectrum = spectra_dict[scan_idx]
                    # Use base peak (highest intensity)
                    max_idx = spectrum['i'].idxmax()
                    mz_list.append(spectrum['mz'].iloc[max_idx])
                    intensity_list.append(spectrum['i'].iloc[max_idx])
                else:
                    mz_list.append(0)
                    intensity_list.append(0)

            ms1_data['mz'] = mz_list
            ms1_data['intensity'] = intensity_list

            # Remove zeros
            ms1_data = ms1_data[ms1_data['mz'] > 0].reset_index(drop=True)

        print(f"  ✓ Loaded {len(ms1_data)} MS1 precursors")

    print(f"\nMS1 Data Summary:")
    print(f"  RT range: {ms1_data['rt'].min():.2f} - {ms1_data['rt'].max():.2f} min")
    print(f"  m/z range: {ms1_data['mz'].min():.2f} - {ms1_data['mz'].max():.2f}")
    print(f"  Intensity range: {ms1_data['intensity'].min():.2e} - {ms1_data['intensity'].max():.2e}")

    # Save Step 1
    output_dir = precursor_root / "results" / "virtual_simple"
    output_dir.mkdir(parents=True, exist_ok=True)

    step1_file = output_dir / "step1_ms1_precursors.json"
    ms1_data_dict = {
        'n_precursors': len(ms1_data),
        'rt_range': [float(ms1_data['rt'].min()), float(ms1_data['rt'].max())],
        'mz_range': [float(ms1_data['mz'].min()), float(ms1_data['mz'].max())],
        'precursors': ms1_data.to_dict(orient='records')
    }
    with open(step1_file, 'w') as f:
        json.dump(ms1_data_dict, f, indent=2)
    print(f"  ✓ Saved to: {step1_file}")


    # STEP 2: Convert to S-Entropy coordinates
    print("\n[STEP 2] Converting to S-Entropy coordinates...")
    print("  Using existing EntropyTransformation infrastructure...")

    # Create S-Entropy transformer
    transformer = SEntropyTransformer()

    # Convert MS1 data to S-Entropy
    # For MS1, we create a "spectrum" with just the precursor peaks
    s_entropy_coords = []
    s_entropy_features_list = []

    for idx, row in ms1_data.iterrows():
        # Create mini-spectrum for this precursor
        mini_spectrum = pd.DataFrame({
            'mz': [row['mz']],
            'intensity': [row['intensity']]
        })

        # Transform to S-Entropy
        coords = transformer.transform_spectrum(mini_spectrum)
        features = transformer.extract_features(coords)

        s_entropy_coords.append({
            'precursor_mz': row['mz'],
            'rt': row['rt'],
            'S_k': float(features.mean_knowledge),
            'S_t': float(features.mean_time),
            'S_e': float(features.mean_entropy),
            'features_14d': features.to_array().tolist()
        })
        s_entropy_features_list.append(features)

    print(f"  ✓ Converted {len(s_entropy_coords)} precursors to S-Entropy")

    # Save Step 2
    step2_file = output_dir / "step2_s_entropy_coordinates.json"
    with open(step2_file, 'w') as f:
        json.dump({
            'n_precursors': len(s_entropy_coords),
            's_entropy_coords': s_entropy_coords
        }, f, indent=2)
    print(f"  ✓ Saved to: {step2_file}")

    # Display S-Entropy coordinates
    print(f"\nS-Entropy Coordinates:")
    for i, coord in enumerate(s_entropy_coords[:5]):  # Show first 5
        print(f"  Precursor {i+1}: m/z={coord['precursor_mz']:.4f}, RT={coord['rt']:.2f}")
        print(f"    (S_k={coord['S_k']:.3f}, S_t={coord['S_t']:.3f}, S_e={coord['S_e']:.3f})")
    if len(s_entropy_coords) > 5:
        print(f"  ... and {len(s_entropy_coords) - 5} more")


    # STEP 3: Create virtual instrument projections
    print("\n[STEP 3] Creating virtual instrument projections...")
    print("  Using Virtual Mass Spec Ensemble...")

    # Create ensemble (simplified - just use the m/z and intensity)
    ensemble = VirtualMassSpecEnsemble(
        enable_all_instruments=True,
        enable_hardware_grounding=True,
        coherence_threshold=0.3
    )

    # Measure with virtual ensemble
    result = ensemble.measure_spectrum(
        mz=ms1_data['mz'].values,
        intensity=ms1_data['intensity'].values,
        rt=ms1_data['rt'].mean(),  # Use mean RT
        metadata={
            'test': 'ms1_precursors_simple',
            'n_precursors': len(ms1_data),
            'rt_range': [float(ms1_data['rt'].min()), float(ms1_data['rt'].max())]
        }
    )

    print(f"  ✓ Created {result.n_instruments} virtual instruments")
    print(f"  ✓ Detected {result.total_phase_locks} phase-locks")
    print(f"  ✓ Found {result.convergence_nodes_count} convergence nodes")

    # Save Step 3 with detailed step-by-step data
    ensemble.save_results(result, output_dir, save_detailed_steps=True)


    # STEP 4: Analyze and summarize
    print("\n[STEP 4] Analysis Summary...")
    print("="*80)
    print(f"MS1 Precursors: {len(ms1_data)}")
    print(f"S-Entropy Coordinates: {len(s_entropy_coords)}")
    print(f"Virtual Instruments: {result.n_instruments}")
    print(f"Phase-locks: {result.total_phase_locks}")
    print(f"Convergence Nodes: {result.convergence_nodes_count}")
    print(f"\nResults saved to: {output_dir}")
    print("="*80)

    return result, s_entropy_coords, ms1_data


if __name__ == "__main__":
    print("\n" + "="*80)
    print("SIMPLE VIRTUAL MASS SPECTROMETRY TEST")
    print("="*80)
    print("\nApproach:")
    print("  1. Start with MS1 precursors ONLY (no MS2 fragments yet)")
    print("  2. Use existing SpectraReader to extract data")
    print("  3. Use existing EntropyTransformation for S-Entropy")
    print("  4. Create virtual instrument projections")
    print("  5. Save results at EVERY step")

    try:
        result, s_coords, ms1 = test_ms1_precursors_simple()
        print("\n✓ Test completed successfully!")
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
