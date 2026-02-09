#!/usr/bin/env python3
"""
Mass Computing Pipeline: Real Data Processing

This pipeline demonstrates the Mass Computing framework on real mzML data:
1. Reads mass spectra from mzML files
2. Encodes each spectrum as a ternary address
3. Synthesizes spectra from addresses (without the original data)
4. Compares measured vs synthesized spectra
5. Validates the partition determinism theorem
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import json
from datetime import datetime
import sys

# Add parent paths
sys.path.insert(0, str(Path(__file__).parent))

# Import Mass Computing core
from ternary_core import (
    TernaryAddress, SEntropyCoord, SpectrumExtractor,
    MoleculeEncoder, Spectrum as SynthesizedSpectrum
)

# Try to import pyteomics for real mzML parsing
try:
    from pyteomics import mzml
    HAS_PYTEOMICS = True
except ImportError:
    HAS_PYTEOMICS = False
    print("Warning: pyteomics not found. Install with: pip install pyteomics")


@dataclass
class MeasuredPeak:
    """A measured peak from real data"""
    mz: float
    intensity: float
    retention_time: float
    scan_id: str
    ms_level: int = 1


@dataclass
class EncodedSpectrum:
    """A spectrum encoded in the Mass Computing framework"""
    # Original measurement
    measured_mz: float
    measured_rt: float
    measured_intensity: float
    scan_id: str

    # Ternary encoding
    address: str
    s_k: float
    s_t: float
    s_e: float

    # Synthesized from address
    synthesized_mz: float
    synthesized_rt: float

    # Errors
    mz_error_ppm: float
    rt_error_min: float


@dataclass
class PipelineResult:
    """Results from running the pipeline"""
    input_file: str
    n_spectra: int
    n_peaks_processed: int
    encoding_depth: int

    # Accuracy metrics
    mz_accuracy_ppm: float
    rt_accuracy_min: float
    mz_within_5ppm: float
    rt_within_05min: float

    # Detailed results
    encoded_spectra: List[EncodedSpectrum] = field(default_factory=list)

    # Timing
    processing_time_sec: float = 0.0
    throughput_spectra_per_sec: float = 0.0


class MzMLReader:
    """Reader for mzML files using pyteomics"""

    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.spectra = []
        self.metadata = {}

    def read(self, max_spectra: int = None) -> List[Dict]:
        """Read mzML file and return list of spectrum dicts"""
        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {self.filepath}")

        if not HAS_PYTEOMICS:
            raise ImportError("pyteomics is required. Install with: pip install pyteomics")

        spectra = []

        with mzml.read(str(self.filepath)) as reader:
            for i, spectrum in enumerate(reader):
                if max_spectra and i >= max_spectra:
                    break

                try:
                    parsed = self._parse_spectrum(spectrum)
                    if parsed:
                        spectra.append(parsed)
                except Exception as e:
                    print(f"Warning: Could not parse spectrum {i}: {e}")
                    continue

        self.spectra = spectra
        return spectra

    def _parse_spectrum(self, spectrum: Dict) -> Optional[Dict]:
        """Parse a single spectrum from pyteomics format"""
        # Get m/z and intensity arrays
        mz_array = spectrum.get('m/z array', np.array([]))
        intensity_array = spectrum.get('intensity array', np.array([]))

        if len(mz_array) == 0 or len(intensity_array) == 0:
            return None

        # Get retention time
        scan_time = spectrum.get('scanList', {}).get('scan', [{}])[0]
        rt = scan_time.get('scan start time', 0.0)

        # Get MS level
        ms_level = spectrum.get('ms level', 1)

        # Get scan ID
        scan_id = spectrum.get('id', f'scan_{hash(str(mz_array[:5]))}')

        return {
            'mz_array': mz_array,
            'intensity_array': intensity_array,
            'retention_time': rt,
            'ms_level': ms_level,
            'scan_id': scan_id,
            'base_peak_mz': mz_array[np.argmax(intensity_array)] if len(intensity_array) > 0 else 0,
            'base_peak_intensity': np.max(intensity_array) if len(intensity_array) > 0 else 0,
            'tic': np.sum(intensity_array)
        }


class MassComputingPipeline:
    """Main pipeline for processing mass spectrometry data"""

    def __init__(self,
                 mass_min: float = 100.0,
                 mass_max: float = 1200.0,
                 rt_min: float = 0.0,
                 rt_max: float = 30.0,
                 encoding_depth: int = 24):
        """
        Initialize the pipeline.

        Args:
            mass_min: Minimum m/z for calibration
            mass_max: Maximum m/z for calibration
            rt_min: Minimum retention time (minutes)
            rt_max: Maximum retention time (minutes)
            encoding_depth: Ternary address depth (higher = more precision)
        """
        self.mass_min = mass_min
        self.mass_max = mass_max
        self.rt_min = rt_min
        self.rt_max = rt_max
        self.encoding_depth = encoding_depth

        # Initialize encoder and extractor with calibration
        self.encoder = MoleculeEncoder(
            mass_min=mass_min,
            mass_max=mass_max,
            t0=rt_min,
            t_max=rt_max
        )

        self.extractor = SpectrumExtractor(
            mass_min=mass_min,
            mass_max=mass_max,
            t0=rt_min,
            t_max=rt_max
        )

    def process_file(self,
                     filepath: str,
                     max_spectra: int = None,
                     intensity_threshold: float = 1000.0) -> PipelineResult:
        """
        Process an mzML file through the Mass Computing pipeline.

        Args:
            filepath: Path to mzML file
            max_spectra: Maximum number of spectra to process
            intensity_threshold: Minimum intensity for peaks

        Returns:
            PipelineResult with accuracy metrics
        """
        import time
        start_time = time.time()

        # Read the file
        print(f"Reading: {filepath}")
        reader = MzMLReader(filepath)
        spectra = reader.read(max_spectra=max_spectra)
        print(f"Loaded {len(spectra)} spectra")

        if not spectra:
            return PipelineResult(
                input_file=str(filepath),
                n_spectra=0,
                n_peaks_processed=0,
                encoding_depth=self.encoding_depth,
                mz_accuracy_ppm=0,
                rt_accuracy_min=0,
                mz_within_5ppm=0,
                rt_within_05min=0
            )

        # Calibrate RT range from data
        rt_values = [s['retention_time'] for s in spectra if s['retention_time'] > 0]
        if rt_values:
            self.rt_min = min(rt_values)
            self.rt_max = max(rt_values)
            # Update encoder/extractor
            self.encoder = MoleculeEncoder(
                mass_min=self.mass_min,
                mass_max=self.mass_max,
                t0=self.rt_min,
                t_max=self.rt_max
            )
            self.extractor = SpectrumExtractor(
                mass_min=self.mass_min,
                mass_max=self.mass_max,
                t0=self.rt_min,
                t_max=self.rt_max
            )

        # Process each spectrum
        encoded_spectra = []
        n_peaks = 0

        for spectrum in spectra:
            # Get base peak for encoding
            mz = spectrum['base_peak_mz']
            rt = spectrum['retention_time']
            intensity = spectrum['base_peak_intensity']

            if mz < self.mass_min or mz > self.mass_max:
                continue
            if intensity < intensity_threshold:
                continue

            n_peaks += 1

            # Encode to ternary address
            # Estimate fragmentation from intensity distribution
            intensities = spectrum['intensity_array']
            frag_estimate = min(0.9, len(intensities[intensities > intensity_threshold * 0.1]) / 100)

            try:
                address = self.encoder.encode(
                    exact_mass=mz,
                    retention_time=rt,
                    fragmentation=frag_estimate,
                    depth=self.encoding_depth
                )

                # Get S-coordinates
                s_k, s_t, s_e = address.to_scoord()

                # Synthesize from address (this is the key step!)
                synthesized = self.extractor.extract_spectrum(address)

                # Calculate errors
                mz_error_ppm = abs(synthesized.mz - mz) / mz * 1e6
                rt_error_min = abs(synthesized.retention_time - rt)

                encoded_spectra.append(EncodedSpectrum(
                    measured_mz=mz,
                    measured_rt=rt,
                    measured_intensity=intensity,
                    scan_id=spectrum['scan_id'],
                    address=str(address),
                    s_k=s_k,
                    s_t=s_t,
                    s_e=s_e,
                    synthesized_mz=synthesized.mz,
                    synthesized_rt=synthesized.retention_time,
                    mz_error_ppm=mz_error_ppm,
                    rt_error_min=rt_error_min
                ))

            except Exception as e:
                print(f"Warning: Could not encode spectrum: {e}")
                continue

        end_time = time.time()
        processing_time = end_time - start_time

        # Calculate summary statistics
        if encoded_spectra:
            mz_errors = [s.mz_error_ppm for s in encoded_spectra]
            rt_errors = [s.rt_error_min for s in encoded_spectra]

            result = PipelineResult(
                input_file=str(filepath),
                n_spectra=len(spectra),
                n_peaks_processed=len(encoded_spectra),
                encoding_depth=self.encoding_depth,
                mz_accuracy_ppm=np.mean(mz_errors),
                rt_accuracy_min=np.mean(rt_errors),
                mz_within_5ppm=sum(1 for e in mz_errors if e < 5.0) / len(mz_errors) * 100,
                rt_within_05min=sum(1 for e in rt_errors if e < 0.5) / len(rt_errors) * 100,
                encoded_spectra=encoded_spectra,
                processing_time_sec=processing_time,
                throughput_spectra_per_sec=len(encoded_spectra) / processing_time if processing_time > 0 else 0
            )
        else:
            result = PipelineResult(
                input_file=str(filepath),
                n_spectra=len(spectra),
                n_peaks_processed=0,
                encoding_depth=self.encoding_depth,
                mz_accuracy_ppm=0,
                rt_accuracy_min=0,
                mz_within_5ppm=0,
                rt_within_05min=0,
                processing_time_sec=processing_time
            )

        return result

    def generate_validation_figure(self,
                                   result: PipelineResult,
                                   output_path: str = None) -> str:
        """Generate validation figure comparing measured vs synthesized"""
        if not result.encoded_spectra:
            print("No encoded spectra to visualize")
            return None

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Extract data
        measured_mz = [s.measured_mz for s in result.encoded_spectra]
        synthesized_mz = [s.synthesized_mz for s in result.encoded_spectra]
        measured_rt = [s.measured_rt for s in result.encoded_spectra]
        synthesized_rt = [s.synthesized_rt for s in result.encoded_spectra]
        mz_errors = [s.mz_error_ppm for s in result.encoded_spectra]
        rt_errors = [s.rt_error_min for s in result.encoded_spectra]

        # Panel A: m/z correlation
        ax = axes[0, 0]
        ax.scatter(measured_mz, synthesized_mz, alpha=0.5, s=10)
        ax.plot([min(measured_mz), max(measured_mz)],
                [min(measured_mz), max(measured_mz)], 'r--', label='Perfect')
        ax.set_xlabel('Measured m/z')
        ax.set_ylabel('Synthesized m/z')
        ax.set_title(f'A. m/z Correlation (R² = {np.corrcoef(measured_mz, synthesized_mz)[0,1]**2:.4f})')
        ax.legend()

        # Panel B: RT correlation
        ax = axes[0, 1]
        ax.scatter(measured_rt, synthesized_rt, alpha=0.5, s=10, color='green')
        ax.plot([min(measured_rt), max(measured_rt)],
                [min(measured_rt), max(measured_rt)], 'r--', label='Perfect')
        ax.set_xlabel('Measured RT (min)')
        ax.set_ylabel('Synthesized RT (min)')
        ax.set_title(f'B. RT Correlation (R² = {np.corrcoef(measured_rt, synthesized_rt)[0,1]**2:.4f})')
        ax.legend()

        # Panel C: m/z error distribution
        ax = axes[1, 0]
        ax.hist(mz_errors, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(mz_errors), color='red', linestyle='--',
                   label=f'Mean: {np.mean(mz_errors):.1f} ppm')
        ax.axvline(5.0, color='green', linestyle=':', label='5 ppm threshold')
        ax.set_xlabel('m/z Error (ppm)')
        ax.set_ylabel('Count')
        ax.set_title(f'C. m/z Error Distribution ({result.mz_within_5ppm:.1f}% < 5 ppm)')
        ax.legend()

        # Panel D: RT error distribution
        ax = axes[1, 1]
        ax.hist(rt_errors, bins=50, edgecolor='black', alpha=0.7, color='green')
        ax.axvline(np.mean(rt_errors), color='red', linestyle='--',
                   label=f'Mean: {np.mean(rt_errors):.2f} min')
        ax.axvline(0.5, color='orange', linestyle=':', label='0.5 min threshold')
        ax.set_xlabel('RT Error (min)')
        ax.set_ylabel('Count')
        ax.set_title(f'D. RT Error Distribution ({result.rt_within_05min:.1f}% < 0.5 min)')
        ax.legend()

        plt.suptitle(f'Mass Computing Validation: {Path(result.input_file).name}\n'
                     f'{result.n_peaks_processed} spectra, depth={result.encoding_depth}',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()

        if output_path is None:
            output_path = Path(__file__).parent.parent / 'figures' / 'pipeline_validation.png'

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved validation figure to: {output_path}")
        return str(output_path)

    def save_results(self, result: PipelineResult, output_path: str = None) -> str:
        """Save results to JSON"""
        if output_path is None:
            output_path = Path(__file__).parent.parent / 'results' / 'pipeline_results.json'

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict for JSON serialization
        result_dict = {
            'input_file': result.input_file,
            'n_spectra': result.n_spectra,
            'n_peaks_processed': result.n_peaks_processed,
            'encoding_depth': result.encoding_depth,
            'mz_accuracy_ppm': result.mz_accuracy_ppm,
            'rt_accuracy_min': result.rt_accuracy_min,
            'mz_within_5ppm': result.mz_within_5ppm,
            'rt_within_05min': result.rt_within_05min,
            'processing_time_sec': result.processing_time_sec,
            'throughput_spectra_per_sec': result.throughput_spectra_per_sec,
            'timestamp': datetime.now().isoformat(),
            'encoded_spectra': [
                {
                    'measured_mz': s.measured_mz,
                    'measured_rt': s.measured_rt,
                    'address': s.address,
                    's_k': s.s_k,
                    's_t': s.s_t,
                    's_e': s.s_e,
                    'synthesized_mz': s.synthesized_mz,
                    'synthesized_rt': s.synthesized_rt,
                    'mz_error_ppm': s.mz_error_ppm,
                    'rt_error_min': s.rt_error_min
                }
                for s in result.encoded_spectra[:100]  # Limit to first 100 for file size
            ]
        }

        with open(output_path, 'w') as f:
            json.dump(result_dict, f, indent=2)

        print(f"Saved results to: {output_path}")
        return str(output_path)


def run_pipeline(mzml_path: str,
                 max_spectra: int = None,
                 encoding_depth: int = 24) -> PipelineResult:
    """
    Run the complete Mass Computing pipeline on an mzML file.

    Args:
        mzml_path: Path to mzML file
        max_spectra: Maximum number of spectra to process
        encoding_depth: Ternary address depth

    Returns:
        PipelineResult with all metrics
    """
    print("=" * 60)
    print("Mass Computing Pipeline")
    print("=" * 60)

    pipeline = MassComputingPipeline(encoding_depth=encoding_depth)
    result = pipeline.process_file(mzml_path, max_spectra=max_spectra)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"File: {Path(result.input_file).name}")
    print(f"Spectra processed: {result.n_peaks_processed}")
    print(f"Encoding depth: {result.encoding_depth} trits")
    print(f"Processing time: {result.processing_time_sec:.2f} sec")
    print(f"Throughput: {result.throughput_spectra_per_sec:.0f} spectra/sec")
    print()
    print("Accuracy Metrics:")
    print(f"  m/z MAE: {result.mz_accuracy_ppm:.2f} ppm")
    print(f"  m/z < 5 ppm: {result.mz_within_5ppm:.1f}%")
    print(f"  RT MAE: {result.rt_accuracy_min:.3f} min")
    print(f"  RT < 0.5 min: {result.rt_within_05min:.1f}%")
    print("=" * 60)

    # Generate figure
    fig_path = pipeline.generate_validation_figure(result)

    # Save results
    results_path = pipeline.save_results(result)

    return result


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Mass Computing Pipeline')
    parser.add_argument('mzml_file', nargs='?',
                        default='../../public/PL_Neg_Waters_qTOF.mzML',
                        help='Path to mzML file')
    parser.add_argument('--max-spectra', type=int, default=None,
                        help='Maximum number of spectra to process')
    parser.add_argument('--depth', type=int, default=24,
                        help='Ternary address encoding depth')

    args = parser.parse_args()

    # Resolve path
    mzml_path = Path(args.mzml_file)
    if not mzml_path.is_absolute():
        mzml_path = Path(__file__).parent / args.mzml_file

    result = run_pipeline(str(mzml_path),
                          max_spectra=args.max_spectra,
                          encoding_depth=args.depth)
