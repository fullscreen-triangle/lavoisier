#!/usr/bin/env python3
"""
Virtual Mass Spectrometer Pipeline - Full Analysis
===================================================

Implements complete metabolomics analysis using virtual mass spectrometers.

Pipeline Stages:
1. Spectral Acquisition & Preprocessing (ALL MS1+MS2, full RT range)
2. S-Entropy Transformation (categorical state extraction)
3. Hardware BMD Grounding (phase-lock validation)
4. Virtual Instrument Ensemble (multi-instrument materialization)
5. Categorical Completion (database search & annotation)
6. Cross-Platform Validation (Waters vs Thermo)

Expected Output: 200MB+ of comprehensive results per file

Author: Kundai Farai Sachikonye
Date: 2025
"""

import sys
from pathlib import Path
import logging
import json

# Add precursor root
precursor_root = Path(__file__).parent
sys.path.insert(0, str(precursor_root))

from src.pipeline.metabolomics import run_metabolomics_analysis
from src.pipeline.theatre import Theatre, NavigationMode
from src.pipeline.stages import StageObserver, ProcessObserver, ProcessResult, StageStatus
from src.core.SpectraReader import extract_mzml
from src.core.EntropyTransformation import SEntropyTransformer
from src.virtual import VirtualMassSpecEnsemble

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('virtual_instrument_pipeline.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class VirtualInstrumentProcess(ProcessObserver):
    """
    Process: Create virtual instrument ensemble and measure all spectra

    Materializes multiple virtual instruments (TOF, Orbitrap, FT-ICR, IMS, etc.)
    at convergence nodes identified from phase-lock signatures.
    """

    def __init__(self):
        super().__init__("virtual_instrument_ensemble")
        self.ensemble = VirtualMassSpecEnsemble(
            enable_all_instruments=True,
            enable_hardware_grounding=True,
            coherence_threshold=0.3
        )

    def observe(self, input_data, **kwargs):
        """
        Create virtual instrument ensemble for all spectra.

        Args:
            input_data: Dict with 'scan_info', 'spectra', 's_entropy_features'

        Returns:
            ProcessResult with virtual instrument measurements
        """
        import time
        import numpy as np
        start_time = time.time()

        scan_info = input_data['scan_info']
        spectra_dict = input_data['spectra']
        s_entropy_features = input_data.get('s_entropy_features', {})

        virtual_results = []
        total_phase_locks = 0
        total_convergence_nodes = 0
        total_instruments = 0

        # Process each spectrum
        for scan_id in spectra_dict.keys():
            spectrum = spectra_dict[scan_id]

            if spectrum is None or len(spectrum) == 0:
                continue

            # Get RT for this scan
            scan_row = scan_info[scan_info['spec_index'] == scan_id]
            if len(scan_row) == 0:
                continue
            rt = scan_row.iloc[0]['scan_time']

            # Measure with virtual ensemble
            result = self.ensemble.measure_spectrum(
                mz=spectrum['mz'].values,
                intensity=spectrum['intensity'].values if 'intensity' in spectrum.columns else spectrum['i'].values,
                rt=rt,
                metadata={
                    'scan_id': int(scan_id),
                    'dda_rank': int(scan_row.iloc[0]['DDA_rank'])
                }
            )

            virtual_results.append({
                'scan_id': int(scan_id),
                'rt': float(rt),
                'n_instruments': result.n_instruments,
                'phase_locks': result.total_phase_locks,
                'convergence_nodes': result.convergence_nodes_count,
                'virtual_instruments': [
                    {
                        'type': vi.instrument_type,
                        'mz': vi.measurement.get('mz'),
                        'S_k': vi.categorical_state.S_k,
                        'S_t': vi.categorical_state.S_t,
                        'S_e': vi.categorical_state.S_e,
                    }
                    for vi in result.virtual_instruments
                ]
            })

            total_phase_locks += result.total_phase_locks
            total_convergence_nodes += result.convergence_nodes_count
            total_instruments += result.n_instruments

        execution_time = time.time() - start_time

        return ProcessResult(
            process_name=self.observer_id,
            status=StageStatus.COMPLETED,
            execution_time=execution_time,
            output_data={'virtual_results': virtual_results},
            metrics={
                'n_spectra_processed': len(virtual_results),
                'total_phase_locks': total_phase_locks,
                'total_convergence_nodes': total_convergence_nodes,
                'total_virtual_instruments': total_instruments,
                'avg_instruments_per_spectrum': total_instruments / len(virtual_results) if virtual_results else 0,
                'throughput_spec_per_sec': len(virtual_results) / execution_time if execution_time > 0 else 0
            },
            metadata={}
        )


def run_virtual_instrument_pipeline(mzml_files, output_dir, enable_bmd=True, **config):
    """
    Run complete virtual instrument pipeline on experimental files.

    This extends the standard metabolomics pipeline with virtual instrument stages.

    Args:
        mzml_files: List of paths to mzML files
        output_dir: Directory for results
        enable_bmd: Enable hardware BMD grounding
        **config: Pipeline configuration

    Returns:
        Dict of results per file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for mzml_file in mzml_files:
        mzml_path = Path(mzml_file)
        file_stem = mzml_path.stem

        logger.info(f"\n{'='*80}")
        logger.info(f"PROCESSING: {file_stem}")
        logger.info(f"{'='*80}\n")

        file_output_dir = output_dir / file_stem
        file_output_dir.mkdir(parents=True, exist_ok=True)

        # First: Run standard metabolomics pipeline
        logger.info("Phase 1: Standard Metabolomics Pipeline")
        logger.info("-"*80)

        standard_results = run_metabolomics_analysis(
            mzml_files=[mzml_file],
            output_dir=output_dir,
            enable_bmd=enable_bmd,
            **config
        )

        # Second: Add virtual instrument stages
        logger.info("\nPhase 2: Virtual Instrument Ensemble")
        logger.info("-"*80)

        # Load the preprocessed data
        stage1_data_file = file_output_dir / "stage_01_preprocessing" / "stage_01_preprocessing_data.tab"
        if not stage1_data_file.exists():
            logger.error(f"Stage 1 data not found: {stage1_data_file}")
            continue

        # Create virtual instrument stage
        virtual_stage = StageObserver(
            stage_name="virtual_instrument_ensemble",
            stage_id="stage_05_virtual_instruments"
        )

        # Add virtual instrument process
        virtual_process = VirtualInstrumentProcess()
        virtual_stage.add_process(virtual_process)

        # Execute virtual instrument stage
        # Load previous stage data
        import pandas as pd
        scan_info = pd.read_csv(
            file_output_dir / "stage_01_preprocessing" / "scan_info.tsv",
            sep="\t"
        )

        # Load spectra
        spectra_dir = file_output_dir / "stage_01_preprocessing" / "spectra"
        spectra_dict = {}
        if spectra_dir.exists():
            for spec_file in spectra_dir.glob("*.tsv"):
                scan_id = int(spec_file.stem.replace("spectrum_", ""))
                spectra_dict[scan_id] = pd.read_csv(spec_file, sep="\t")

        stage_input = {
            'scan_info': scan_info,
            'spectra': spectra_dict,
            's_entropy_features': {}
        }

        virtual_result = virtual_stage.observe(stage_input)

        # Save virtual instrument results
        virtual_output_dir = file_output_dir / "stage_05_virtual_instruments"
        virtual_output_dir.mkdir(parents=True, exist_ok=True)

        # Save stage result
        result_file = virtual_output_dir / "stage_05_virtual_instruments_result.json"
        with open(result_file, 'w') as f:
            json.dump(virtual_result.to_dict(), f, indent=2)

        logger.info(f"\nVirtual Instrument Stage Complete:")
        logger.info(f"  Spectra processed: {virtual_result.metrics['n_spectra_processed']}")
        logger.info(f"  Total phase-locks: {virtual_result.metrics['total_phase_locks']}")
        logger.info(f"  Convergence nodes: {virtual_result.metrics['total_convergence_nodes']}")
        logger.info(f"  Virtual instruments: {virtual_result.metrics['total_virtual_instruments']}")
        logger.info(f"  Avg instruments/spectrum: {virtual_result.metrics['avg_instruments_per_spectrum']:.1f}")
        logger.info(f"  Throughput: {virtual_result.metrics['throughput_spec_per_sec']:.1f} spec/s")

        results[file_stem] = {
            'standard_pipeline': standard_results[file_stem],
            'virtual_instruments': virtual_result
        }

    return results


def main():
    """Execute virtual instrument pipeline."""

    # Define paths
    project_root = Path(__file__).parent
    data_dir = project_root / "public" / "metabolomics"
    output_dir = project_root / "results" / "virtual_instrument_analysis"

    # Experimental files
    mzml_files = [
        data_dir / "PL_Neg_Waters_qTOF.mzML",
        data_dir / "TG_Pos_Thermo_Orbi.mzML"
    ]

    # Verify files exist
    for mzml_file in mzml_files:
        if not mzml_file.exists():
            logger.error(f"File not found: {mzml_file}")
            logger.error(f"Please ensure experimental files are in {data_dir}")
            return 1

    logger.info("="*80)
    logger.info("VIRTUAL MASS SPECTROMETER PIPELINE - FULL ANALYSIS")
    logger.info("="*80)
    logger.info(f"\nFiles to process:")
    for f in mzml_files:
        logger.info(f"  - {f.name}")
    logger.info(f"\nOutput directory: {output_dir}")
    logger.info(f"\nPipeline Configuration:")
    logger.info(f"  • Full RT range (0-100 min)")
    logger.info(f"  • All MS1 + MS2 spectra")
    logger.info(f"  • S-Entropy transformation")
    logger.info(f"  • Hardware BMD grounding")
    logger.info(f"  • Virtual instrument ensemble (8 types)")
    logger.info(f"  • Categorical completion")
    logger.info(f"  • Expected output: 200MB+ per file\n")

    # Configuration
    config = {
        'preprocessing': {
            'acquisition': {
                'rt_range': [0, 100],  # FULL RT range
                'ms1_threshold': 1000,
                'ms2_threshold': 10,
            },
            'peak_detection': {
                'min_intensity': 100.0,
                'min_snr': 3.0
            }
        },
        'sentropy': {
            'categorical': {
                'epsilon': 0.1
            }
        },
        'bmd_grounding': {
            'coherence': {
                'divergence_threshold': 0.3
            }
        },
        'completion': {
            'temporal': {
                'database_path': None
            }
        }
    }

    try:
        # Run pipeline
        results = run_virtual_instrument_pipeline(
            mzml_files=mzml_files,
            output_dir=output_dir,
            enable_bmd=True,
            **config
        )

        # Summary
        print("\n" + "="*80)
        print("VIRTUAL INSTRUMENT PIPELINE COMPLETE")
        print("="*80)

        for file_name, result in results.items():
            print(f"\nFILE: {file_name}")
            print(f"{'─'*80}")

            # Standard pipeline summary
            std_result = result['standard_pipeline']
            print(f"Standard Pipeline:")
            print(f"  Status: {std_result.status.value}")
            print(f"  Time: {std_result.execution_time:.2f}s")
            print(f"  Stages: {len(std_result.execution_order)}")

            # Virtual instrument summary
            virt_result = result['virtual_instruments']
            print(f"\nVirtual Instruments:")
            print(f"  Spectra processed: {virt_result.metrics['n_spectra_processed']}")
            print(f"  Virtual instruments: {virt_result.metrics['total_virtual_instruments']}")
            print(f"  Phase-locks: {virt_result.metrics['total_phase_locks']}")
            print(f"  Convergence nodes: {virt_result.metrics['total_convergence_nodes']}")
            print(f"  Throughput: {virt_result.metrics['throughput_spec_per_sec']:.1f} spec/s")

        print(f"\n{'='*80}")
        print(f"Results saved to: {output_dir}")
        print("="*80)

        return 0

    except Exception as e:
        logger.error(f"\nPipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
