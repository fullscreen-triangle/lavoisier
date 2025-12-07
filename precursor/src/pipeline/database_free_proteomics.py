#!/usr/bin/env python3
"""
Database-Free Proteomics Pipeline
==================================

Integration of Molecular Maxwell Demon system with the existing
theatre-based pipeline architecture.

This pipeline performs complete database-free peptide sequencing
using S-Entropy, categorical completion, and zero-shot identification.

Author: Kundai Sachikonye
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from core.Theatre import TheatreProcess, ProcessState

# Import MMD system
from mmdsystem.mmd_orchestrator import MolecularMaxwellDemonSystem, MMDConfig
from sequence.sequence_reconstruction import ReconstructionResult


class DatabaseFreeProteomicsProcess(TheatreProcess):
    """
    Theatre process for database-free proteomics sequencing.

    Integrates MMD system into the pipeline framework.
    """

    def __init__(self, config: Optional[MMDConfig] = None):
        """
        Initialize process.

        Args:
            config: MMD system configuration
        """
        super().__init__()
        self.config = config if config else MMDConfig()
        self.mmd_system: Optional[MolecularMaxwellDemonSystem] = None

    def execute(self, input_data: dict) -> dict:
        """
        Execute database-free sequencing.

        Args:
            input_data: Dict containing:
                - spectra: Dict of {scan_id: (mz_array, intensity_array)}
                - scan_info: DataFrame with precursor info
                - sentropy_features: S-Entropy transformed features (optional)

        Returns:
            Dict containing:
                - sequences: Dict of {scan_id: reconstructed_sequence}
                - reconstruction_results: Dict of {scan_id: ReconstructionResult}
                - summary: Summary DataFrame
        """
        self.state = ProcessState.RUNNING

        try:
            # Initialize MMD system
            if self.mmd_system is None:
                print("\n[DatabaseFree] Initializing Molecular Maxwell Demon System...")
                self.mmd_system = MolecularMaxwellDemonSystem(self.config)

            # Extract input data
            spectra = input_data.get('spectra', input_data.get('filtered_spectra', {}))
            scan_info = input_data.get('scan_info', pd.DataFrame())

            if len(spectra) == 0:
                print("[DatabaseFree] WARNING: No spectra provided!")
                self.state = ProcessState.FAILED
                return {
                    'sequences': {},
                    'reconstruction_results': {},
                    'summary': pd.DataFrame()
                }

            print(f"\n[DatabaseFree] Processing {len(spectra)} spectra...")

            # Process each spectrum
            sequences = {}
            reconstruction_results = {}

            for scan_id, spectrum_data in spectra.items():
                try:
                    # Extract spectrum
                    if isinstance(spectrum_data, tuple) and len(spectrum_data) == 2:
                        mz_array, intensity_array = spectrum_data
                    elif isinstance(spectrum_data, dict):
                        mz_array = np.array(spectrum_data.get('mz', []))
                        intensity_array = np.array(spectrum_data.get('intensity', []))
                    else:
                        print(f"[DatabaseFree] WARNING: Invalid spectrum format for scan {scan_id}")
                        continue

                    # Get precursor info
                    if not scan_info.empty and scan_id in scan_info.index:
                        scan_row = scan_info.loc[scan_id]
                        precursor_mz = float(scan_row.get('precursor_mz', scan_row.get('PrecursorMZ', 0.0)))
                        precursor_charge = int(scan_row.get('precursor_charge', scan_row.get('PrecursorCharge', 2)))
                        rt = float(scan_row.get('rt', scan_row.get('RT', 0.0)))
                    else:
                        # Estimate from data
                        precursor_mz = 500.0  # Default
                        precursor_charge = 2
                        rt = None

                    # Analyze spectrum
                    result = self.mmd_system.analyze_spectrum(
                        mz_array=mz_array,
                        intensity_array=intensity_array,
                        precursor_mz=precursor_mz,
                        precursor_charge=precursor_charge,
                        rt=rt
                    )

                    sequences[scan_id] = result.sequence
                    reconstruction_results[scan_id] = result

                except Exception as e:
                    print(f"[DatabaseFree] ERROR processing scan {scan_id}: {e}")
                    continue

            # Create summary DataFrame
            summary_data = []
            for scan_id, result in reconstruction_results.items():
                summary_data.append({
                    'scan_id': scan_id,
                    'sequence': result.sequence,
                    'confidence': result.confidence,
                    'coverage': result.fragment_coverage,
                    'n_gaps_filled': len(result.gap_filled_regions),
                    'total_entropy': result.total_entropy
                })

            summary = pd.DataFrame(summary_data)

            # Print summary statistics
            print(f"\n{'='*60}")
            print("DATABASE-FREE PROTEOMICS SUMMARY")
            print(f"{'='*60}")
            print(f"Total spectra: {len(spectra)}")
            print(f"Sequences reconstructed: {len(sequences)}")
            if len(summary) > 0:
                print(f"Mean confidence: {summary['confidence'].mean():.3f}")
                print(f"Mean coverage: {summary['coverage'].mean():.1%}")
                print(f"High confidence (>0.7): {sum(summary['confidence'] > 0.7)}")

            self.state = ProcessState.COMPLETED

            return {
                'sequences': sequences,
                'reconstruction_results': reconstruction_results,
                'summary': summary,
                'mmd_system': self.mmd_system  # Pass system for further analysis
            }

        except Exception as e:
            print(f"\n[DatabaseFree] CRITICAL ERROR: {e}")
            import traceback
            traceback.print_exc()
            self.state = ProcessState.FAILED
            return {
                'sequences': {},
                'reconstruction_results': {},
                'summary': pd.DataFrame()
            }


def create_database_free_pipeline(
    data_path: str,
    output_dir: str,
    mmd_config: Optional[MMDConfig] = None
) -> Tuple[Dict, pd.DataFrame]:
    """
    Create and run complete database-free proteomics pipeline.

    Args:
        data_path: Path to MS data file
        output_dir: Output directory for results
        mmd_config: MMD system configuration

    Returns:
        (reconstruction_results, summary_dataframe)
    """
    from core.Theatre import Theatre
    from pipeline.metabolomics import SpectralAcquisitionProcess, SEntropyTransformProcess

    print("\n" + "="*70)
    print("DATABASE-FREE PROTEOMICS PIPELINE")
    print("Powered by Molecular Maxwell Demon")
    print("="*70 + "\n")

    # Create theatre
    theatre = Theatre(name="DatabaseFreeProteomics")

    # Define stages
    stages = {
        'acquisition': SpectralAcquisitionProcess(),
        'sentropy_transform': SEntropyTransformProcess(),
        'database_free_sequencing': DatabaseFreeProteomicsProcess(mmd_config)
    }

    # Add stages to theatre
    for stage_name, process in stages.items():
        theatre.add_stage(stage_name, process)

    # Run theatre
    initial_data = {'data_file': data_path}
    results = theatre.execute(initial_data)

    # Extract results
    sequencing_results = results.get('database_free_sequencing', {})
    reconstruction_results = sequencing_results.get('reconstruction_results', {})
    summary = sequencing_results.get('summary', pd.DataFrame())

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not summary.empty:
        summary_file = output_path / "database_free_results.csv"
        summary.to_csv(summary_file, index=False)
        print(f"\n[Pipeline] Results saved to {summary_file}")

    return reconstruction_results, summary
