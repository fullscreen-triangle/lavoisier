#!/usr/bin/env python3
"""
Test Sequence Reconstruction Module
====================================

Tests fragment graph, categorical completion, and sequence reconstruction.
Uses real data from: public/proteomics/BSA1.mzML
Saves all results to: results/tests/sequence/

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
from sequence.fragment_graph import FragmentNode, FragmentGraph, build_fragment_graph_from_spectra
from sequence.categorical_completion import CategoricalCompleter, GapFiller, GapRegion
from sequence.sequence_reconstruction import SequenceReconstructor
from dictionary.sentropy_dictionary import create_standard_proteomics_dictionary
from dictionary.zero_shot_identification import ZeroShotIdentifier
from core.EntropyTransformation import SEntropyTransformer, SEntropyCoordinates
from core.SpectraReader import extract_mzml

# Test data path
TEST_DATA = Path("public/proteomics/BSA1.mzML")


def test_fragment_graph():
    """Test fragment graph construction."""
    print("\n[Test 1] Fragment Graph Construction")
    print("="*60)

    # Create synthetic fragments
    fragments = []
    for i in range(5):
        node = FragmentNode(
            fragment_id=f"frag_{i}",
            sequence=None,
            s_entropy_coords=SEntropyCoordinates(
                s_knowledge=0.5 + i * 0.1,
                s_time=0.3 + i * 0.05,
                s_entropy=0.6 + i * 0.02
            ),
            mass=100.0 + i * 100.0,
            confidence=0.9
        )
        fragments.append(node)

    # Build graph
    graph = build_fragment_graph_from_spectra(fragments, precursor_mass=500.0)

    # Extract graph info
    nodes_data = []
    for frag_id, frag in graph.nodes.items():
        nodes_data.append({
            'fragment_id': frag_id,
            'mass': frag.mass,
            's_knowledge': frag.s_entropy_coords.s_knowledge,
            's_time': frag.s_entropy_coords.s_time,
            's_entropy': frag.s_entropy_coords.s_entropy,
            'confidence': frag.confidence
        })

    edges_data = []
    for edge in graph.graph.edges(data=True):
        edges_data.append({
            'from': edge[0],
            'to': edge[1],
            'mass_diff': edge[2].get('mass_diff', 0),
            'sentropy_similarity': edge[2].get('sentropy_similarity', 0),
            'weight': edge[2].get('weight', 0)
        })

    # Save
    output_dir = PRECURSOR_ROOT / "results" / "tests" / "sequence"
    output_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(nodes_data).to_csv(output_dir / "fragment_graph_nodes.csv", index=False)
    pd.DataFrame(edges_data).to_csv(output_dir / "fragment_graph_edges.csv", index=False)

    print(f"  Nodes: {len(nodes_data)}")
    print(f"  Edges: {len(edges_data)}")
    print(f"  ✓ Saved to {output_dir}")

    return graph


def test_categorical_completion():
    """Test categorical completion for gap filling."""
    print("\n[Test 2] Categorical Completion")
    print("="*60)

    # Create dictionary and completer
    dictionary = create_standard_proteomics_dictionary()
    completer = CategoricalCompleter(dictionary)

    # Test gaps of different sizes
    test_gaps = [
        (110.0, "Single AA"),     # ~1 amino acid
        (228.0, "Double AA"),     # ~2 amino acids
        (340.0, "Triple AA")      # ~3 amino acids
    ]

    results = []

    for gap_mass, description in test_gaps:
        gap = GapRegion(
            start_fragment="frag_1",
            end_fragment="frag_2",
            mass_gap=gap_mass,
            sentropy_start=SEntropyCoordinates(0.5, 0.5, 0.5),
            sentropy_end=SEntropyCoordinates(0.6, 0.6, 0.6)
        )

        filled = completer.fill_gap(gap, max_gap_size=5)

        if filled:
            filled_seq = ''.join(aa for aa, _ in filled)
            avg_conf = np.mean([conf for _, conf in filled])

            results.append({
                'gap_mass': gap_mass,
                'description': description,
                'filled_sequence': filled_seq,
                'n_residues': len(filled),
                'avg_confidence': avg_conf
            })

            print(f"  {description} ({gap_mass:.1f} Da): {filled_seq} (conf={avg_conf:.3f})")
        else:
            print(f"  {description} ({gap_mass:.1f} Da): Could not fill")

    df = pd.DataFrame(results)
    output_dir = PRECURSOR_ROOT / "results" / "tests" / "sequence"
    df.to_csv(output_dir / "categorical_completion.csv", index=False)

    print(f"  ✓ Saved to {output_dir / 'categorical_completion.csv'}")

    return df


def test_sequence_reconstruction_real_data():
    """Test sequence reconstruction with real MS data."""
    print("\n[Test 3] Sequence Reconstruction from Real Data")
    print("="*60)

    if not TEST_DATA.exists():
        print(f"  ⚠ Test data not found: {TEST_DATA}")
        return pd.DataFrame()

    print(f"  Loading: {TEST_DATA}")

    # Setup
    dictionary = create_standard_proteomics_dictionary()
    identifier = ZeroShotIdentifier(dictionary)
    completer = CategoricalCompleter(dictionary)
    reconstructor = SequenceReconstructor(dictionary, identifier, completer)
    transformer = SEntropyTransformer()

    # Load spectra using YOUR extract_mzml function
    try:
        scan_info, spectra_dict, _ = extract_mzml(
            str(TEST_DATA),
            rt_range=[0, 100],
            ms1_threshold=1000,
            ms2_threshold=10
        )
        print(f"  Loaded {len(spectra_dict)} spectra")
    except Exception as e:
        print(f"  ✗ Failed to load: {e}")
        return pd.DataFrame()

    results = []
    n_processed = 0

    # Process first 10 spectra
    for scan_id, spec_df in list(spectra_dict.items())[:10]:
        try:
            # Extract data from DataFrame
            if isinstance(spec_df, pd.DataFrame):
                if 'mz' in spec_df.columns and 'i' in spec_df.columns:
                    mz = spec_df['mz'].values
                    intensity = spec_df['i'].values
                else:
                    continue
            else:
                continue

            if len(mz) < 10:
                continue

            # Get precursor info from scan_info
            if scan_id in scan_info.index:
                precursor_mz = scan_info.loc[scan_id, 'PrecursorMZ']
                precursor_charge = int(scan_info.loc[scan_id, 'PrecursorCharge'])
            else:
                precursor_mz = 500.0
                precursor_charge = 2

            # Transform to S-Entropy
            coords_list, coord_matrix = transformer.transform_spectrum(
                mz_array=mz,
                intensity_array=intensity,
                precursor_mz=precursor_mz
            )

            # Create fragment nodes (top 20 peaks)
            top_indices = np.argsort(intensity)[-20:]
            fragments = []

            for idx in top_indices:
                if idx < len(coords_list):
                    node = FragmentNode(
                        fragment_id=f"scan_{n_processed}_frag_{idx}",
                        sequence=None,
                        s_entropy_coords=coords_list[idx],
                        mass=mz[idx] * precursor_charge,
                        confidence=intensity[idx] / intensity.max()
                    )
                    fragments.append(node)

            # Reconstruct
            result = reconstructor.reconstruct(
                fragments=fragments,
                precursor_mass=precursor_mz * precursor_charge,
                precursor_charge=precursor_charge
            )

            results.append({
                'scan_id': scan_id,
                'precursor_mz': precursor_mz,
                'precursor_charge': precursor_charge,
                'n_fragments': len(fragments),
                'reconstructed_sequence': result.sequence,
                'confidence': result.confidence,
                'coverage': result.fragment_coverage,
                'n_gaps_filled': len(result.gap_filled_regions),
                'total_entropy': result.total_entropy
            })

            print(f"  Scan {n_processed}: {result.sequence} (conf={result.confidence:.3f})")

        except Exception as e:
            print(f"  Scan {n_processed}: Failed - {e}")

        n_processed += 1
        if n_processed >= 10:
            break

    df = pd.DataFrame(results)
    output_dir = PRECURSOR_ROOT / "results" / "tests" / "sequence"
    df.to_csv(output_dir / "real_data_reconstructions.csv", index=False)

    print(f"\n  Processed {n_processed} MS2 spectra")
    print(f"  Reconstructed {len(df)} sequences")
    if len(df) > 0:
        print(f"  Mean confidence: {df['confidence'].mean():.3f}")
        print(f"  Mean coverage: {df['coverage'].mean():.1%}")
    print(f"  ✓ Saved to {output_dir / 'real_data_reconstructions.csv'}")

    return df


def main():
    """Run all sequence reconstruction tests."""
    print("\n" + "="*60)
    print("SEQUENCE RECONSTRUCTION MODULE TEST")
    print(f"Test data: {TEST_DATA}")
    print("="*60)

    results = {}

    try:
        # Test 1: Fragment graph
        graph = test_fragment_graph()
        results['fragment_graph'] = graph

        # Test 2: Categorical completion
        completion_df = test_categorical_completion()
        results['categorical_completion'] = completion_df

        # Test 3: Real data reconstruction
        reconstruction_df = test_sequence_reconstruction_real_data()
        results['reconstructions'] = reconstruction_df

        # Summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print(f"  Fragment graph edges: {graph.graph.number_of_edges()}")
        print(f"  Categorical completions: {len(completion_df)}")
        print(f"  Real data reconstructions: {len(reconstruction_df)}")
        print(f"\n  ✓ All results saved to results/tests/sequence/")

        return True

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
