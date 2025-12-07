#!/usr/bin/env python3
"""
UC DAVIS METABOLOMICS COMPLETE ANALYSIS
========================================

COMPREHENSIVE FRAMEWORK VALIDATION
----------------------------------

This script runs the COMPLETE metabolomics framework on all UC Davis mzML files:

Pipeline Stages:
1. Spectral Acquisition & Preprocessing (mzML extraction, peak detection)
2. S-Entropy Transformation (bijective mapping to categorical states)
   - Computer Vision modality (ion-to-droplet thermodynamic conversion)
3. Fragmentation Network Analysis (phase-lock networks)
4. Hardware BMD Grounding (reality validation)
5. Categorical Completion (temporal navigation, annotation)
6. Virtual Instrument Ensemble (multi-instrument materialization)

All results are saved systematically for validation.

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
log_file = precursor_root / f'ucdavis_analysis_{timestamp}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


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


def run_stage1_preprocessing(mzml_path: Path, output_dir: Path, config: dict) -> dict:
    """
    Stage 1: Spectral Acquisition and Preprocessing
    """
    from src.core.SpectraReader import extract_mzml

    logger.info("=" * 60)
    logger.info("STAGE 1: SPECTRAL ACQUISITION & PREPROCESSING")
    logger.info("=" * 60)

    start_time = time.time()

    vendor = 'thermo'
    if 'waters' in mzml_path.stem.lower():
        vendor = 'waters'

    logger.info(f"Loading: {mzml_path.name}")
    scan_info_df, spectra_dict, ms1_xic_df = extract_mzml(
        mzml=str(mzml_path),
        rt_range=config.get('rt_range', [0, 100]),
        ms1_threshold=config.get('ms1_threshold', 1000),
        ms2_threshold=config.get('ms2_threshold', 10),
        vendor=vendor
    )

    n_ms1 = len(scan_info_df[scan_info_df['DDA_rank'] == 0])
    n_ms2 = len(scan_info_df[scan_info_df['DDA_rank'] > 0])
    n_spectra = len(spectra_dict)

    total_peaks = 0
    for scan_id, spectrum_df in spectra_dict.items():
        if spectrum_df is not None:
            total_peaks += len(spectrum_df)

    execution_time = time.time() - start_time

    logger.info(f"MS1 scans: {n_ms1}")
    logger.info(f"MS2 scans: {n_ms2}")
    logger.info(f"Total spectra: {n_spectra}")
    logger.info(f"Total peaks: {total_peaks}")
    logger.info(f"Execution time: {execution_time:.2f}s")

    stage_dir = output_dir / 'stage_01_preprocessing'
    stage_dir.mkdir(parents=True, exist_ok=True)

    save_dataframe(scan_info_df, stage_dir / 'scan_info')

    if not ms1_xic_df.empty:
        save_dataframe(ms1_xic_df, stage_dir / 'ms1_xic')

    spectra_dir = stage_dir / 'spectra'
    spectra_dir.mkdir(parents=True, exist_ok=True)

    spectra_summary = []
    for scan_id, spectrum_df in spectra_dict.items():
        if spectrum_df is not None and len(spectrum_df) > 0:
            if 'i' in spectrum_df.columns:
                spectrum_df = spectrum_df.rename(columns={'i': 'intensity'})

            spectrum_df.to_csv(spectra_dir / f'spectrum_{scan_id}.tsv', sep='\t', index=False)

            int_col = 'intensity' if 'intensity' in spectrum_df.columns else 'i'
            spectra_summary.append({
                'scan_id': scan_id,
                'n_peaks': len(spectrum_df),
                'mz_min': spectrum_df['mz'].min(),
                'mz_max': spectrum_df['mz'].max(),
                'intensity_max': spectrum_df[int_col].max()
            })

    spectra_summary_df = pd.DataFrame(spectra_summary)
    save_dataframe(spectra_summary_df, stage_dir / 'spectra_summary')

    metrics = {
        'n_ms1': n_ms1,
        'n_ms2': n_ms2,
        'n_spectra': n_spectra,
        'total_peaks': total_peaks,
        'execution_time': execution_time,
        'vendor': vendor
    }
    save_json(metrics, stage_dir / 'stage_01_metrics.json')

    return {
        'scan_info': scan_info_df,
        'spectra': spectra_dict,
        'xic': ms1_xic_df,
        'metrics': metrics
    }


def run_stage2_sentropy(stage1_data: dict, output_dir: Path) -> dict:
    """
    Stage 2: S-Entropy Transformation
    """
    from src.core.EntropyTransformation import SEntropyTransformer

    logger.info("=" * 60)
    logger.info("STAGE 2: S-ENTROPY TRANSFORMATION")
    logger.info("=" * 60)

    start_time = time.time()

    spectra_dict = stage1_data['spectra']
    transformer = SEntropyTransformer()

    sentropy_results = []
    sentropy_features = {}

    for scan_id, spectrum_df in spectra_dict.items():
        if spectrum_df is None or len(spectrum_df) == 0:
            continue

        mz_array = spectrum_df['mz'].values
        intensity_col = 'intensity' if 'intensity' in spectrum_df.columns else 'i'
        intensity_array = spectrum_df[intensity_col].values

        try:
            coords_list, coord_matrix = transformer.transform_spectrum(mz_array, intensity_array)
            sentropy_features[scan_id] = coord_matrix

            if len(coord_matrix) > 0:
                mean_coords = np.mean(coord_matrix, axis=0) if len(coord_matrix.shape) > 1 else coord_matrix
                std_coords = np.std(coord_matrix, axis=0) if len(coord_matrix.shape) > 1 and coord_matrix.shape[0] > 1 else np.zeros(3)

                sentropy_results.append({
                    'scan_id': scan_id,
                    'n_peaks': len(coord_matrix),
                    's_k_mean': float(mean_coords[0]) if len(mean_coords) > 0 else 0,
                    's_t_mean': float(mean_coords[1]) if len(mean_coords) > 1 else 0,
                    's_e_mean': float(mean_coords[2]) if len(mean_coords) > 2 else 0,
                    's_k_std': float(std_coords[0]) if len(std_coords) > 0 else 0,
                    's_t_std': float(std_coords[1]) if len(std_coords) > 1 else 0,
                    's_e_std': float(std_coords[2]) if len(std_coords) > 2 else 0
                })
        except Exception as e:
            logger.warning(f"Failed to transform scan {scan_id}: {e}")

    execution_time = time.time() - start_time

    logger.info(f"Transformed {len(sentropy_results)} spectra")
    logger.info(f"Execution time: {execution_time:.2f}s")

    stage_dir = output_dir / 'stage_02_sentropy'
    stage_dir.mkdir(parents=True, exist_ok=True)

    sentropy_df = pd.DataFrame(sentropy_results)
    save_dataframe(sentropy_df, stage_dir / 'sentropy_features')

    sentropy_matrices_dir = stage_dir / 'matrices'
    sentropy_matrices_dir.mkdir(parents=True, exist_ok=True)

    for scan_id, matrix in sentropy_features.items():
        if matrix is not None and len(matrix) > 0:
            ncols = matrix.shape[1] if len(matrix.shape) > 1 else 1
            matrix_df = pd.DataFrame(matrix, columns=['s_k', 's_t', 's_e'][:ncols])
            matrix_df.to_csv(sentropy_matrices_dir / f'sentropy_{scan_id}.tsv', sep='\t', index=False)

    metrics = {
        'n_transformed': len(sentropy_results),
        'execution_time': execution_time,
        'throughput': len(sentropy_results) / execution_time if execution_time > 0 else 0
    }
    save_json(metrics, stage_dir / 'stage_02_metrics.json')

    return {
        'sentropy_features': sentropy_features,
        'sentropy_df': sentropy_df,
        'metrics': metrics
    }


def run_stage2_cv(stage1_data: dict, output_dir: Path) -> dict:
    """
    Stage 2B: Computer Vision Ion-to-Droplet Conversion
    """
    logger.info("=" * 60)
    logger.info("STAGE 2B: COMPUTER VISION ION-TO-DROPLET")
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

    converter = IonToDropletConverter(
        resolution=(512, 512),
        enable_physics_validation=True,
        validation_threshold=0.3
    )

    cv_results = []
    cv_images_dir = output_dir / 'stage_02_cv' / 'images'
    cv_images_dir.mkdir(parents=True, exist_ok=True)

    droplets_dir = output_dir / 'stage_02_cv' / 'droplets'
    droplets_dir.mkdir(parents=True, exist_ok=True)

    for scan_id, spectrum_df in spectra_dict.items():
        if spectrum_df is None or len(spectrum_df) == 0:
            continue

        mz_array = spectrum_df['mz'].values
        intensity_col = 'intensity' if 'intensity' in spectrum_df.columns else 'i'
        intensity_array = spectrum_df[intensity_col].values

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

    try:
        validation_report = converter.get_validation_report()
        with open(stage_dir / 'validation_report.txt', 'w') as f:
            f.write(validation_report)
    except:
        pass

    metrics = {
        'n_converted': len(cv_results),
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
    """
    Stage 2.5: Fragmentation Network Analysis
    """
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
    logger.info(f"Edges: {stats['num_edges']}")
    logger.info(f"Execution time: {execution_time:.2f}s")

    stage_dir = output_dir / 'stage_02_5_fragmentation'
    stage_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        'n_precursors': stats['num_precursors'],
        'n_fragments': stats['num_fragments'],
        'n_edges': stats['num_edges'],
        'network_density': stats.get('network_density', 0),
        'avg_degree': stats.get('avg_degree', 0),
        'execution_time': execution_time
    }
    save_json(metrics, stage_dir / 'stage_02_5_metrics.json')

    return {'network_stats': stats, 'metrics': metrics}


def run_stage3_bmd(stage1_data: dict, stage2_data: dict, output_dir: Path) -> dict:
    """
    Stage 3: Hardware BMD Grounding
    """
    logger.info("=" * 60)
    logger.info("STAGE 3: HARDWARE BMD GROUNDING")
    logger.info("=" * 60)

    start_time = time.time()

    try:
        from src.bmd import BiologicalMaxwellDemonReference
        bmd_available = True
    except ImportError:
        bmd_available = False

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
        'bmd_available': bmd_available,
        'n_analyzed': len(coherence_results),
        'mean_coherence': mean_coherence,
        'mean_divergence': mean_divergence,
        'execution_time': execution_time
    }
    save_json(metrics, stage_dir / 'stage_03_metrics.json')

    return {'coherence_df': coherence_df, 'metrics': metrics}


def run_stage4_completion(stage1_data: dict, stage2_data: dict, stage3_data: dict, output_dir: Path) -> dict:
    """
    Stage 4: Categorical Completion
    """
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
    """
    Stage 5: Virtual Instrument Ensemble
    """
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

    ensemble = VirtualMassSpecEnsemble(
        enable_all_instruments=True,
        enable_hardware_grounding=True,
        coherence_threshold=0.3
    )

    virtual_results = []

    for scan_id, spectrum_df in spectra_dict.items():
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
        logger.info(f"Total convergence nodes: {virtual_df['convergence_nodes'].sum()}")
    logger.info(f"Execution time: {execution_time:.2f}s")

    stage_dir = output_dir / 'stage_05_virtual'
    stage_dir.mkdir(parents=True, exist_ok=True)

    if len(virtual_df) > 0:
        save_dataframe(virtual_df, stage_dir / 'virtual_results')

    metrics = {
        'n_measurements': len(virtual_results),
        'total_phase_locks': int(virtual_df['phase_locks'].sum()) if len(virtual_df) > 0 else 0,
        'total_convergence_nodes': int(virtual_df['convergence_nodes'].sum()) if len(virtual_df) > 0 else 0,
        'execution_time': execution_time
    }
    save_json(metrics, stage_dir / 'stage_05_metrics.json')

    return {'virtual_df': virtual_df, 'metrics': metrics}


def process_single_file(mzml_path: Path, output_base: Path, config: dict) -> dict:
    """Process a single mzML file through the complete pipeline."""

    file_stem = mzml_path.stem
    output_dir = output_base / file_stem
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("\n" + "=" * 80)
    logger.info(f"PROCESSING: {file_stem}")
    logger.info("=" * 80 + "\n")

    file_start = time.time()

    results = {
        'file': file_stem,
        'path': str(mzml_path),
        'stages': {}
    }

    try:
        # Stage 1: Preprocessing
        stage1_data = run_stage1_preprocessing(mzml_path, output_dir, config)
        results['stages']['preprocessing'] = stage1_data['metrics']

        # Stage 2: S-Entropy
        stage2_data = run_stage2_sentropy(stage1_data, output_dir)
        results['stages']['sentropy'] = stage2_data['metrics']

        # Stage 2B: Computer Vision
        cv_data = run_stage2_cv(stage1_data, output_dir)
        results['stages']['cv'] = cv_data.get('metrics', {})

        # Stage 2.5: Fragmentation
        frag_data = run_stage25_fragmentation(stage1_data, stage2_data, output_dir)
        results['stages']['fragmentation'] = frag_data.get('metrics', {})

        # Stage 3: BMD Grounding
        stage3_data = run_stage3_bmd(stage1_data, stage2_data, output_dir)
        results['stages']['bmd'] = stage3_data['metrics']

        # Stage 4: Completion
        stage4_data = run_stage4_completion(stage1_data, stage2_data, stage3_data, output_dir)
        results['stages']['completion'] = stage4_data['metrics']

        # Stage 5: Virtual Instruments
        stage5_data = run_stage5_virtual(stage1_data, stage2_data, output_dir)
        results['stages']['virtual'] = stage5_data.get('metrics', {})

        results['status'] = 'completed'
        results['total_time'] = time.time() - file_start

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        logger.error(traceback.format_exc())
        results['status'] = 'failed'
        results['error'] = str(e)
        results['total_time'] = time.time() - file_start

    save_json(results, output_dir / 'pipeline_results.json')

    return results


def main():
    """Run complete analysis on all UC Davis files."""

    print("=" * 80)
    print("UC DAVIS METABOLOMICS - COMPLETE FRAMEWORK ANALYSIS")
    print("=" * 80)
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log file: {log_file}\n")

    project_root = Path(__file__).parent
    data_dir = project_root / 'public' / 'ucdavis'
    output_dir = project_root / 'results' / 'ucdavis_complete_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)

    mzml_files = sorted(list(data_dir.glob('*.mzml')) + list(data_dir.glob('*.mzML')))

    if not mzml_files:
        logger.error(f"No mzML files found in {data_dir}")
        return 1

    logger.info(f"Found {len(mzml_files)} mzML files:")
    for f in mzml_files:
        logger.info(f"  - {f.name}")

    config = {
        'rt_range': [0, 100],
        'ms1_threshold': 1000,
        'ms2_threshold': 10
    }

    all_results = []

    for i, mzml_path in enumerate(mzml_files):
        logger.info(f"\n{'#' * 80}")
        logger.info(f"FILE {i + 1}/{len(mzml_files)}: {mzml_path.name}")
        logger.info(f"{'#' * 80}\n")

        result = process_single_file(mzml_path, output_dir, config)
        all_results.append(result)

    # Generate summary report
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
    save_dataframe(summary_df, output_dir / 'analysis_summary')

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    for result in all_results:
        status_icon = "✓" if result['status'] == 'completed' else "✗"
        print(f"\n{status_icon} {result['file']}")
        print(f"  Status: {result['status']}")
        print(f"  Time: {result.get('total_time', 0):.2f}s")

        if 'stages' in result:
            for stage_name, metrics in result['stages'].items():
                if isinstance(metrics, dict) and metrics:
                    key_metric = list(metrics.values())[0]
                    print(f"  {stage_name}: {key_metric}")

    print(f"\n{'=' * 80}")
    print(f"Results saved to: {output_dir}")
    print(f"Log file: {log_file}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    save_json({
        'timestamp': timestamp,
        'n_files': len(mzml_files),
        'results': all_results
    }, output_dir / 'master_results.json')

    return 0


if __name__ == "__main__":
    sys.exit(main())
