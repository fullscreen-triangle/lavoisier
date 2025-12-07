#!/usr/bin/env python3
"""
UC DAVIS METABOLOMICS - RESUME FROM STAGE 2B
=============================================

This script resumes the analysis from Stage 2B (Computer Vision),
processing only a FRACTION of spectra for droplet conversion to
avoid hanging on large datasets.

Author: Kundai Farai Sachikonye
Date: 2025
"""

import sys
import os
from pathlib import Path
import logging
import json
import time
from datetime import datetime
import traceback

# Add precursor root to path
precursor_root = Path(__file__).parent
sys.path.insert(0, str(precursor_root))
sys.path.insert(0, str(precursor_root / 'src'))

import numpy as np
import pandas as pd

# Configure logging
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = precursor_root / f'ucdavis_resume_{timestamp}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


# === Configuration ===
MAX_SPECTRA_FOR_CV = 50  # Only process this many spectra for CV droplet conversion
SKIP_STAGES = ['preprocessing', 'sentropy']  # Already completed


def save_dataframe(df: pd.DataFrame, path: Path, formats: list = ['csv', 'tsv']):
    """Save DataFrame in multiple formats."""
    path.parent.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        if fmt == 'csv':
            df.to_csv(path.with_suffix('.csv'), index=False)
        elif fmt == 'tsv':
            df.to_csv(path.with_suffix('.tsv'), sep='\t', index=False)

    logger.info(f"Saved: {path.stem} ({len(df)} rows)")


def save_json(data: dict, path: Path):
    """Save dictionary as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)

    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        elif hasattr(obj, '__dict__'):
            return convert(obj.__dict__)
        return obj

    with open(path, 'w') as f:
        json.dump(convert(data), f, indent=2, default=str)

    logger.info(f"Saved: {path.name}")


def load_stage1_results(output_dir: Path) -> dict:
    """Load existing Stage 1 results."""
    stage_dir = output_dir / 'stage_01_preprocessing'

    scan_info = pd.read_csv(stage_dir / 'scan_info.csv')
    logger.info(f"Loaded scan_info: {len(scan_info)} rows")

    spectra_dir = stage_dir / 'spectra'
    spectra_dict = {}

    if spectra_dir.exists():
        for spectrum_file in spectra_dir.glob('spectrum_*.tsv'):
            scan_id = int(spectrum_file.stem.replace('spectrum_', ''))
            spectra_dict[scan_id] = pd.read_csv(spectrum_file, sep='\t')

    logger.info(f"Loaded {len(spectra_dict)} spectra")

    xic_path = stage_dir / 'ms1_xic.csv'
    xic_df = pd.read_csv(xic_path) if xic_path.exists() else pd.DataFrame()

    return {
        'scan_info': scan_info,
        'spectra': spectra_dict,
        'xic': xic_df
    }


def load_stage2_results(output_dir: Path) -> dict:
    """Load existing Stage 2 results."""
    stage_dir = output_dir / 'stage_02_sentropy'

    sentropy_path = stage_dir / 'sentropy_features.csv'
    sentropy_df = pd.read_csv(sentropy_path) if sentropy_path.exists() else pd.DataFrame()
    logger.info(f"Loaded sentropy: {len(sentropy_df)} rows")

    matrices_dir = stage_dir / 'matrices'
    sentropy_features = {}

    if matrices_dir.exists():
        for matrix_file in matrices_dir.glob('sentropy_*.tsv'):
            scan_id = int(matrix_file.stem.replace('sentropy_', ''))
            sentropy_features[scan_id] = pd.read_csv(matrix_file, sep='\t').values

    return {
        'sentropy_df': sentropy_df,
        'sentropy_features': sentropy_features
    }


def run_stage2b_cv_limited(stage1_data: dict, output_dir: Path, max_spectra: int = 50) -> dict:
    """
    Stage 2B: Computer Vision Ion-to-Droplet Conversion (LIMITED)
    Only process a fraction of spectra to avoid hanging.
    """
    logger.info("=" * 60)
    logger.info(f"STAGE 2B: COMPUTER VISION (Limited to {max_spectra} spectra)")
    logger.info("=" * 60)

    start_time = time.time()

    try:
        from src.core.IonToDropletConverter import IonToDropletConverter
        import cv2
        cv_available = True
    except ImportError as e:
        logger.warning(f"CV module not available: {e}")
        cv_available = False

    if not cv_available:
        return {'cv_available': False, 'metrics': {'cv_available': False}}

    spectra_dict = stage1_data['spectra']
    scan_info = stage1_data['scan_info']

    # Select a subset of spectra (evenly distributed)
    scan_ids = list(spectra_dict.keys())
    if len(scan_ids) > max_spectra:
        step = len(scan_ids) // max_spectra
        selected_ids = scan_ids[::step][:max_spectra]
    else:
        selected_ids = scan_ids

    logger.info(f"Processing {len(selected_ids)} of {len(scan_ids)} spectra")

    converter = IonToDropletConverter(
        resolution=(256, 256),  # Smaller resolution for speed
        enable_physics_validation=True,
        validation_threshold=0.3
    )

    cv_results = []
    cv_images_dir = output_dir / 'stage_02_cv' / 'images'
    cv_images_dir.mkdir(parents=True, exist_ok=True)

    droplets_dir = output_dir / 'stage_02_cv' / 'droplets'
    droplets_dir.mkdir(parents=True, exist_ok=True)

    for i, scan_id in enumerate(selected_ids):
        if i % 10 == 0:
            logger.info(f"  Processing spectrum {i+1}/{len(selected_ids)}")

        spectrum_df = spectra_dict.get(scan_id)
        if spectrum_df is None or len(spectrum_df) == 0:
            continue

        mz_array = spectrum_df['mz'].values
        intensity_col = 'intensity' if 'intensity' in spectrum_df.columns else 'i'
        intensity_array = spectrum_df[intensity_col].values

        # Get RT
        rt = None
        scan_row = scan_info[scan_info['spec_index'] == scan_id]
        if len(scan_row) > 0:
            rt = scan_row.iloc[0]['scan_time']

        try:
            image, ion_droplets = converter.convert_spectrum_to_image(
                mzs=mz_array,
                intensities=intensity_array,
                rt=rt,
                normalize=True
            )

            if ion_droplets and len(ion_droplets) > 0:
                cv2.imwrite(str(cv_images_dir / f'droplet_{scan_id}.png'), image)

                droplet_data = []
                for idx, droplet in enumerate(ion_droplets):
                    droplet_data.append({
                        'droplet_idx': idx,
                        'mz': droplet.mz,
                        'intensity': droplet.intensity,
                        's_knowledge': droplet.s_entropy_coords.s_knowledge,
                        's_time': droplet.s_entropy_coords.s_time,
                        's_entropy': droplet.s_entropy_coords.s_entropy,
                        'velocity': droplet.droplet_params.velocity,
                        'radius': droplet.droplet_params.radius,
                        'phase_coherence': droplet.droplet_params.phase_coherence,
                        'categorical_state': droplet.categorical_state,
                        'physics_quality': droplet.physics_quality
                    })

                droplet_df = pd.DataFrame(droplet_data)
                droplet_df.to_csv(droplets_dir / f'droplets_{scan_id}.tsv', sep='\t', index=False)

                cv_results.append({
                    'scan_id': scan_id,
                    'n_droplets': len(ion_droplets),
                    'avg_physics_quality': np.mean([d.physics_quality for d in ion_droplets]),
                    'avg_phase_coherence': np.mean([d.droplet_params.phase_coherence for d in ion_droplets]),
                    'n_valid': sum(1 for d in ion_droplets if d.is_physically_valid)
                })
        except Exception as e:
            logger.warning(f"CV conversion failed for scan {scan_id}: {e}")

    execution_time = time.time() - start_time

    logger.info(f"Converted {len(cv_results)} spectra to droplets")
    logger.info(f"Execution time: {execution_time:.2f}s")

    stage_dir = output_dir / 'stage_02_cv'
    cv_summary_df = pd.DataFrame(cv_results)
    if len(cv_results) > 0:
        save_dataframe(cv_summary_df, stage_dir / 'cv_summary')

    metrics = {
        'n_converted': len(cv_results),
        'n_total_available': len(scan_ids),
        'fraction_processed': len(cv_results) / len(scan_ids) if scan_ids else 0,
        'avg_physics_quality': cv_summary_df['avg_physics_quality'].mean() if len(cv_results) > 0 else 0,
        'avg_phase_coherence': cv_summary_df['avg_phase_coherence'].mean() if len(cv_results) > 0 else 0,
        'execution_time': execution_time
    }
    save_json(metrics, stage_dir / 'stage_02_cv_metrics.json')

    return {
        'cv_results': cv_results,
        'cv_summary': cv_summary_df,
        'metrics': metrics
    }


def run_stage25_fragmentation(stage1_data: dict, stage2_data: dict, output_dir: Path) -> dict:
    """Stage 2.5: Fragmentation Network Analysis"""
    logger.info("=" * 60)
    logger.info("STAGE 2.5: FRAGMENTATION NETWORK ANALYSIS")
    logger.info("=" * 60)

    start_time = time.time()

    try:
        from src.metabolomics.FragmentationTrees import (
            SEntropyFragmentationNetwork,
            PrecursorIon,
            FragmentIon
        )
        frag_available = True
    except ImportError as e:
        logger.warning(f"Fragmentation module not available: {e}")
        frag_available = False

    if not frag_available:
        return {'fragmentation_available': False, 'metrics': {'fragmentation_available': False}}

    spectra_dict = stage1_data['spectra']
    scan_info = stage1_data['scan_info']

    network = SEntropyFragmentationNetwork(similarity_threshold=0.5, sigma=0.2)

    n_precursors = 0
    n_fragments = 0

    for scan_id in spectra_dict.keys():
        scan_row = scan_info[scan_info['spec_index'] == scan_id]
        if len(scan_row) == 0:
            continue

        scan_data = scan_row.iloc[0]
        dda_rank = scan_data.get('DDA_rank', 0)
        spectrum_df = spectra_dict[scan_id]

        if spectrum_df is None or len(spectrum_df) == 0:
            continue

        intensity_col = 'intensity' if 'intensity' in spectrum_df.columns else 'i'

        if dda_rank == 0:
            precursor_mz = spectrum_df['mz'].values[0]
            precursor = PrecursorIon(
                mz=precursor_mz,
                intensity=spectrum_df[intensity_col].max(),
                rt=scan_data.get('scan_time', 0),
                fragments=[]
            )
            network.add_precursor(precursor, compute_s_entropy=True)
            n_precursors += 1
        else:
            for _, row in spectrum_df.iterrows():
                fragment = FragmentIon(
                    mz=row['mz'],
                    intensity=row[intensity_col],
                    precursor_mz=scan_data.get('MS2_PR_mz', 0)
                )
                n_fragments += 1

    if n_precursors > 0:
        try:
            network.build_network()
            stats = network.get_network_statistics()
        except Exception as e:
            logger.warning(f"Network build failed: {e}")
            stats = {'num_precursors': n_precursors, 'num_fragments': n_fragments, 'num_edges': 0}
    else:
        stats = {'num_precursors': 0, 'num_fragments': 0, 'num_edges': 0}

    execution_time = time.time() - start_time

    logger.info(f"Precursors: {stats['num_precursors']}")
    logger.info(f"Fragments: {stats['num_fragments']}")
    logger.info(f"Execution time: {execution_time:.2f}s")

    stage_dir = output_dir / 'stage_02_5_fragmentation'
    stage_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        'n_precursors': stats['num_precursors'],
        'n_fragments': stats['num_fragments'],
        'n_edges': stats.get('num_edges', 0),
        'execution_time': execution_time
    }
    save_json(metrics, stage_dir / 'stage_02_5_metrics.json')

    return {'network_stats': stats, 'metrics': metrics}


def run_stage3_bmd(stage1_data: dict, stage2_data: dict, output_dir: Path) -> dict:
    """Stage 3: Hardware BMD Grounding"""
    logger.info("=" * 60)
    logger.info("STAGE 3: HARDWARE BMD GROUNDING")
    logger.info("=" * 60)

    start_time = time.time()

    sentropy_features = stage2_data.get('sentropy_features', {})
    coherence_results = []

    for scan_id, features in sentropy_features.items():
        if features is None or len(features) == 0:
            continue

        if len(features.shape) > 1 and features.shape[0] > 1:
            variance = np.var(features, axis=0)
            coherence = 1.0 / (1.0 + np.sum(variance))
        else:
            coherence = 1.0

        coherence_results.append({
            'scan_id': scan_id,
            'coherence': coherence,
            'divergence': 1.0 - coherence
        })

    execution_time = time.time() - start_time

    coherence_df = pd.DataFrame(coherence_results)

    mean_coherence = coherence_df['coherence'].mean() if len(coherence_df) > 0 else 0
    mean_divergence = coherence_df['divergence'].mean() if len(coherence_df) > 0 else 0

    logger.info(f"Mean coherence: {mean_coherence:.4f}")
    logger.info(f"Mean divergence: {mean_divergence:.4f}")
    logger.info(f"Execution time: {execution_time:.2f}s")

    stage_dir = output_dir / 'stage_03_bmd'
    stage_dir.mkdir(parents=True, exist_ok=True)

    if len(coherence_df) > 0:
        save_dataframe(coherence_df, stage_dir / 'coherence_results')

    metrics = {
        'n_analyzed': len(coherence_results),
        'mean_coherence': mean_coherence,
        'mean_divergence': mean_divergence,
        'execution_time': execution_time
    }
    save_json(metrics, stage_dir / 'stage_03_metrics.json')

    return {'coherence_df': coherence_df, 'metrics': metrics}


def run_stage4_completion(stage1_data: dict, stage2_data: dict, stage3_data: dict, output_dir: Path) -> dict:
    """Stage 4: Categorical Completion"""
    logger.info("=" * 60)
    logger.info("STAGE 4: CATEGORICAL COMPLETION")
    logger.info("=" * 60)

    start_time = time.time()

    sentropy_df = stage2_data.get('sentropy_df', pd.DataFrame())
    coherence_df = stage3_data.get('coherence_df', pd.DataFrame())

    if len(sentropy_df) > 0 and len(coherence_df) > 0:
        completion_df = sentropy_df.merge(coherence_df, on='scan_id', how='left')
    else:
        completion_df = sentropy_df.copy() if len(sentropy_df) > 0 else pd.DataFrame()

    completion_results = []

    for _, row in completion_df.iterrows():
        s_k = row.get('s_k_mean', 0)
        s_t = row.get('s_t_mean', 0)
        s_e = row.get('s_e_mean', 0)
        coherence = row.get('coherence', 1.0)

        confidence = coherence * (1.0 - row.get('s_e_std', 0))

        completion_results.append({
            'scan_id': row['scan_id'],
            's_k': s_k,
            's_t': s_t,
            's_e': s_e,
            'coherence': coherence,
            'completion_confidence': confidence
        })

    execution_time = time.time() - start_time

    completion_df = pd.DataFrame(completion_results)

    logger.info(f"Completion candidates: {len(completion_results)}")
    logger.info(f"Execution time: {execution_time:.2f}s")

    stage_dir = output_dir / 'stage_04_completion'
    stage_dir.mkdir(parents=True, exist_ok=True)

    if len(completion_df) > 0:
        save_dataframe(completion_df, stage_dir / 'completion_results')

    metrics = {
        'n_candidates': len(completion_results),
        'avg_confidence': completion_df['completion_confidence'].mean() if len(completion_df) > 0 else 0,
        'execution_time': execution_time
    }
    save_json(metrics, stage_dir / 'stage_04_metrics.json')

    return {'completion_df': completion_df, 'metrics': metrics}


def run_stage5_virtual(stage1_data: dict, stage2_data: dict, output_dir: Path) -> dict:
    """Stage 5: Virtual Instrument Ensemble"""
    logger.info("=" * 60)
    logger.info("STAGE 5: VIRTUAL INSTRUMENT ENSEMBLE")
    logger.info("=" * 60)

    start_time = time.time()

    try:
        from src.virtual import VirtualMassSpecEnsemble
        virtual_available = True
    except ImportError as e:
        logger.warning(f"Virtual module not available: {e}")
        virtual_available = False

    if not virtual_available:
        return {'virtual_available': False, 'metrics': {'virtual_available': False}}

    spectra_dict = stage1_data['spectra']
    scan_info = stage1_data['scan_info']

    # Limit to subset for speed
    scan_ids = list(spectra_dict.keys())[:100]

    ensemble = VirtualMassSpecEnsemble(
        enable_all_instruments=True,
        enable_hardware_grounding=True,
        coherence_threshold=0.3
    )

    virtual_results = []

    for scan_id in scan_ids:
        spectrum_df = spectra_dict.get(scan_id)
        if spectrum_df is None or len(spectrum_df) == 0:
            continue

        mz_array = spectrum_df['mz'].values
        intensity_col = 'intensity' if 'intensity' in spectrum_df.columns else 'i'
        intensity_array = spectrum_df[intensity_col].values

        scan_row = scan_info[scan_info['spec_index'] == scan_id]
        rt = scan_row.iloc[0]['scan_time'] if len(scan_row) > 0 else 0

        try:
            result = ensemble.measure_spectrum(
                mz=mz_array,
                intensity=intensity_array,
                rt=rt,
                metadata={'scan_id': int(scan_id)}
            )

            virtual_results.append({
                'scan_id': scan_id,
                'n_instruments': result.n_instruments,
                'phase_locks': result.total_phase_locks,
                'convergence_nodes': result.convergence_nodes_count
            })
        except Exception as e:
            logger.warning(f"Virtual measurement failed for scan {scan_id}: {e}")

    execution_time = time.time() - start_time

    virtual_df = pd.DataFrame(virtual_results)

    logger.info(f"Virtual measurements: {len(virtual_results)}")
    if len(virtual_df) > 0:
        logger.info(f"Total phase-locks: {virtual_df['phase_locks'].sum()}")
    logger.info(f"Execution time: {execution_time:.2f}s")

    stage_dir = output_dir / 'stage_05_virtual'
    stage_dir.mkdir(parents=True, exist_ok=True)

    if len(virtual_df) > 0:
        save_dataframe(virtual_df, stage_dir / 'virtual_results')

    metrics = {
        'n_measurements': len(virtual_results),
        'total_phase_locks': int(virtual_df['phase_locks'].sum()) if len(virtual_df) > 0 else 0,
        'execution_time': execution_time
    }
    save_json(metrics, stage_dir / 'stage_05_metrics.json')

    return {'virtual_df': virtual_df, 'metrics': metrics}


def resume_file_processing(file_dir: Path) -> dict:
    """Resume processing a single file from Stage 2B."""

    file_name = file_dir.name
    logger.info("\n" + "=" * 80)
    logger.info(f"RESUMING: {file_name}")
    logger.info("=" * 80 + "\n")

    file_start = time.time()

    results = {
        'file': file_name,
        'path': str(file_dir),
        'stages': {},
        'resumed_from': 'stage_2b'
    }

    try:
        # Load existing Stage 1 results
        logger.info("Loading Stage 1 results...")
        stage1_data = load_stage1_results(file_dir)
        results['stages']['preprocessing'] = {'loaded': True, 'n_spectra': len(stage1_data['spectra'])}

        # Load existing Stage 2 results
        logger.info("Loading Stage 2 results...")
        stage2_data = load_stage2_results(file_dir)
        results['stages']['sentropy'] = {'loaded': True, 'n_rows': len(stage2_data['sentropy_df'])}

        # Stage 2B: Computer Vision (LIMITED)
        cv_data = run_stage2b_cv_limited(stage1_data, file_dir, max_spectra=MAX_SPECTRA_FOR_CV)
        results['stages']['cv'] = cv_data.get('metrics', {})

        # Stage 2.5: Fragmentation
        frag_data = run_stage25_fragmentation(stage1_data, stage2_data, file_dir)
        results['stages']['fragmentation'] = frag_data.get('metrics', {})

        # Stage 3: BMD Grounding
        stage3_data = run_stage3_bmd(stage1_data, stage2_data, file_dir)
        results['stages']['bmd'] = stage3_data['metrics']

        # Stage 4: Completion
        stage4_data = run_stage4_completion(stage1_data, stage2_data, stage3_data, file_dir)
        results['stages']['completion'] = stage4_data['metrics']

        # Stage 5: Virtual Instruments
        stage5_data = run_stage5_virtual(stage1_data, stage2_data, file_dir)
        results['stages']['virtual'] = stage5_data.get('metrics', {})

        results['status'] = 'completed'
        results['total_time'] = time.time() - file_start

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        logger.error(traceback.format_exc())
        results['status'] = 'failed'
        results['error'] = str(e)
        results['total_time'] = time.time() - file_start

    save_json(results, file_dir / 'pipeline_resume_results.json')

    return results


def main():
    """Resume UC Davis analysis from Stage 2B."""

    print("=" * 80)
    print("UC DAVIS METABOLOMICS - RESUME FROM STAGE 2B")
    print("=" * 80)
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log file: {log_file}")
    print(f"Max spectra for CV: {MAX_SPECTRA_FOR_CV}\n")

    project_root = Path(__file__).parent
    output_base = project_root / 'results' / 'ucdavis_complete_analysis'

    # Find directories with completed Stage 1
    file_dirs = [d for d in output_base.iterdir() if d.is_dir()]

    if not file_dirs:
        logger.error(f"No result directories found in {output_base}")
        return 1

    logger.info(f"Found {len(file_dirs)} files to resume:")
    for d in file_dirs:
        logger.info(f"  - {d.name}")

    all_results = []

    for i, file_dir in enumerate(file_dirs):
        logger.info(f"\n{'#' * 80}")
        logger.info(f"FILE {i + 1}/{len(file_dirs)}: {file_dir.name}")
        logger.info(f"{'#' * 80}\n")

        # Check if Stage 1 exists
        stage1_dir = file_dir / 'stage_01_preprocessing'
        if not stage1_dir.exists():
            logger.warning(f"Skipping {file_dir.name} - no Stage 1 results")
            continue

        result = resume_file_processing(file_dir)
        all_results.append(result)

    # Generate summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY REPORT")
    logger.info("=" * 80 + "\n")

    summary_data = []
    for result in all_results:
        summary = {
            'file': result['file'],
            'status': result['status'],
            'total_time': result.get('total_time', 0)
        }

        for stage_name, stage_metrics in result.get('stages', {}).items():
            if isinstance(stage_metrics, dict):
                for key, value in stage_metrics.items():
                    if isinstance(value, (int, float)):
                        summary[f"{stage_name}_{key}"] = value

        summary_data.append(summary)

    summary_df = pd.DataFrame(summary_data)
    save_dataframe(summary_df, output_base / 'resume_summary')

    print("\n" + "=" * 80)
    print("RESUME COMPLETE")
    print("=" * 80)

    for result in all_results:
        status_icon = "✓" if result['status'] == 'completed' else "✗"
        print(f"\n{status_icon} {result['file']}")
        print(f"  Status: {result['status']}")
        print(f"  Time: {result.get('total_time', 0):.2f}s")

    print(f"\n{'=' * 80}")
    print(f"Results saved to: {output_base}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    save_json({
        'timestamp': timestamp,
        'n_files': len(file_dirs),
        'max_spectra_cv': MAX_SPECTRA_FOR_CV,
        'results': all_results
    }, output_base / 'resume_master_results.json')

    return 0


if __name__ == "__main__":
    sys.exit(main())
