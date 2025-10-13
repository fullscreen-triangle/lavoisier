#!/usr/bin/env python3
"""
STEP 1: DATA LOADING VALIDATION EXPERIMENT
=========================================

OBJECTIVE: Validate mzML data loading capabilities
HYPOTHESIS: The standalone mzML reader can successfully load and parse mass spectrometry data

EXPERIMENT PROCEDURE:
1. Load mzML files (real or synthetic)
2. Validate data structure integrity
3. Analyze spectrum properties
4. Generate loading performance metrics
5. Create data quality visualizations

Run: python step_01_data_loading_experiment.py
"""

import sys
import os
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Ensure core modules are available
sys.path.insert(0, str(Path(__file__).parent))

def main():
    """Step 1: Data Loading Validation Experiment"""

    print("🧪 STEP 1: DATA LOADING VALIDATION EXPERIMENT")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Objective: Validate mzML data loading and parsing")
    print("=" * 60)

    # Create step-specific results directory
    step_dir = Path("step_results") / "step_01_data_loading"
    step_dir.mkdir(parents=True, exist_ok=True)

    # Initialize step log
    log_file = step_dir / "data_loading_log.txt"

    def log_and_print(message):
        """Log message to both console and file"""
        print(message)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"{datetime.now().isoformat()}: {message}\n")

    log_and_print("🔬 STEP 1 EXPERIMENTAL SETUP")
    log_and_print("-" * 30)

    try:
        # Import data loading components
        log_and_print("📦 Loading mzML reader...")
        from core.mzml_reader import StandaloneMzMLReader, Spectrum

        reader = StandaloneMzMLReader()
        log_and_print("✅ mzML reader initialized successfully")

        # Define test datasets
        test_files = [
            "PL_Neg_Waters_qTOF.mzML",
            "TG_Pos_Thermo_Orbi.mzML"
        ]

        log_and_print(f"📊 Testing {len(test_files)} datasets:")
        for i, file in enumerate(test_files, 1):
            log_and_print(f"  {i}. {file}")

        # Initialize results
        step_results = {
            'step_metadata': {
                'step_number': 1,
                'step_name': 'Data Loading Validation',
                'start_time': datetime.now().isoformat(),
                'objective': 'Validate mzML data loading and parsing capabilities'
            },
            'loading_results': {},
            'data_quality_metrics': {},
            'performance_metrics': {},
            'step_conclusion': {}
        }

        log_and_print("\n🚀 STARTING DATA LOADING VALIDATION")
        log_and_print("-" * 45)

        all_loading_times = []
        all_spectrum_counts = []
        all_data_quality_scores = []

        # Test each dataset
        for dataset_num, test_file in enumerate(test_files, 1):
            log_and_print(f"\n📂 DATASET {dataset_num}: {test_file}")
            log_and_print("-" * 40)

            dataset_start_time = time.time()

            try:
                # STEP 1.1: Attempt to load mzML file
                log_and_print("STEP 1.1: Loading mzML file...")
                load_start = time.time()

                spectra = reader.load_mzml(test_file)
                loading_time = time.time() - load_start

                log_and_print(f"  ✅ Loaded {len(spectra)} spectra in {loading_time:.2f}s")

                # STEP 1.2: Validate data structure
                log_and_print("STEP 1.2: Validating data structure...")

                structure_valid = True
                invalid_spectra = 0

                for spectrum in spectra[:10]:  # Check first 10 spectra
                    if not isinstance(spectrum, Spectrum):
                        structure_valid = False
                        invalid_spectra += 1
                        continue

                    # Check required attributes
                    required_attrs = ['scan_id', 'ms_level', 'mz_array', 'intensity_array']
                    for attr in required_attrs:
                        if not hasattr(spectrum, attr):
                            structure_valid = False
                            invalid_spectra += 1
                            break

                log_and_print(f"  📊 Structure validation: {'✅ PASSED' if structure_valid else '❌ FAILED'}")
                if invalid_spectra > 0:
                    log_and_print(f"  ⚠️  Invalid spectra found: {invalid_spectra}")

                # STEP 1.3: Analyze spectrum properties
                log_and_print("STEP 1.3: Analyzing spectrum properties...")

                ms1_count = len([s for s in spectra if s.ms_level == 1])
                ms2_count = len([s for s in spectra if s.ms_level == 2])

                # Peak statistics
                peak_counts = [len(s.mz_array) for s in spectra]
                avg_peaks = np.mean(peak_counts) if peak_counts else 0
                max_peaks = max(peak_counts) if peak_counts else 0
                min_peaks = min(peak_counts) if peak_counts else 0

                # Mass range analysis
                all_mz = []
                for spectrum in spectra[:50]:  # Sample first 50 spectra
                    all_mz.extend(spectrum.mz_array)

                mz_range_min = min(all_mz) if all_mz else 0
                mz_range_max = max(all_mz) if all_mz else 0

                log_and_print(f"  📈 Spectrum Analysis:")
                log_and_print(f"     MS1 spectra: {ms1_count}")
                log_and_print(f"     MS2 spectra: {ms2_count}")
                log_and_print(f"     Average peaks per spectrum: {avg_peaks:.1f}")
                log_and_print(f"     Peak count range: {min_peaks} - {max_peaks}")
                log_and_print(f"     m/z range: {mz_range_min:.1f} - {mz_range_max:.1f}")

                # STEP 1.4: Calculate data quality score
                log_and_print("STEP 1.4: Calculating data quality score...")

                quality_score = 0.0

                # Factor 1: Loading success (30%)
                if len(spectra) > 0:
                    quality_score += 0.3

                # Factor 2: Structure integrity (25%)
                if structure_valid and invalid_spectra == 0:
                    quality_score += 0.25

                # Factor 3: Data completeness (25%)
                if ms1_count > 0 and avg_peaks > 10:
                    quality_score += 0.25

                # Factor 4: Reasonable data ranges (20%)
                if mz_range_max > 100 and len(all_mz) > 100:
                    quality_score += 0.2

                log_and_print(f"  🎯 Data Quality Score: {quality_score:.2f}/1.00")

                # STEP 1.5: Performance assessment
                log_and_print("STEP 1.5: Performance assessment...")

                spectra_per_second = len(spectra) / max(0.001, loading_time)

                if quality_score >= 0.9 and spectra_per_second >= 100:
                    performance_grade = "🟢 EXCELLENT"
                elif quality_score >= 0.7 and spectra_per_second >= 50:
                    performance_grade = "🟡 GOOD"
                elif quality_score >= 0.5 and spectra_per_second >= 20:
                    performance_grade = "🟠 ACCEPTABLE"
                else:
                    performance_grade = "🔴 NEEDS IMPROVEMENT"

                log_and_print(f"  🏆 Performance Grade: {performance_grade}")

                # Store results
                step_results['loading_results'][test_file] = {
                    'loading_success': True,
                    'loading_time': loading_time,
                    'spectra_loaded': len(spectra),
                    'spectra_per_second': spectra_per_second,
                    'structure_valid': structure_valid,
                    'invalid_spectra_count': invalid_spectra,
                    'spectrum_analysis': {
                        'ms1_count': ms1_count,
                        'ms2_count': ms2_count,
                        'avg_peaks_per_spectrum': avg_peaks,
                        'peak_count_range': [min_peaks, max_peaks],
                        'mz_range': [mz_range_min, mz_range_max]
                    },
                    'quality_score': quality_score,
                    'performance_grade': performance_grade
                }

                # Track overall metrics
                all_loading_times.append(loading_time)
                all_spectrum_counts.append(len(spectra))
                all_data_quality_scores.append(quality_score)

                log_and_print(f"✅ Dataset {dataset_num} loading validation completed")

            except Exception as e:
                log_and_print(f"❌ Dataset {dataset_num} loading failed: {e}")

                step_results['loading_results'][test_file] = {
                    'loading_success': False,
                    'error': str(e),
                    'loading_time': time.time() - dataset_start_time
                }

        # STEP 1.6: Overall analysis and conclusions
        log_and_print(f"\n" + "=" * 60)
        log_and_print("📊 STEP 1 OVERALL ANALYSIS")
        log_and_print("=" * 60)

        successful_loads = len([r for r in step_results['loading_results'].values() if r.get('loading_success', False)])
        total_datasets = len(test_files)
        success_rate = successful_loads / max(1, total_datasets)

        if successful_loads > 0:
            avg_loading_time = np.mean(all_loading_times)
            avg_spectrum_count = np.mean(all_spectrum_counts)
            avg_quality_score = np.mean(all_data_quality_scores)
            total_spectra_loaded = sum(all_spectrum_counts)

            step_results['performance_metrics'] = {
                'successful_loads': successful_loads,
                'total_datasets': total_datasets,
                'success_rate': success_rate,
                'avg_loading_time': avg_loading_time,
                'avg_spectrum_count': avg_spectrum_count,
                'avg_quality_score': avg_quality_score,
                'total_spectra_loaded': total_spectra_loaded,
                'avg_spectra_per_second': avg_spectrum_count / max(0.001, avg_loading_time)
            }

            log_and_print(f"🔢 STEP 1 PERFORMANCE METRICS:")
            log_and_print(f"   Successful loads: {successful_loads}/{total_datasets}")
            log_and_print(f"   Success rate: {success_rate:.1%}")
            log_and_print(f"   Average loading time: {avg_loading_time:.2f}s")
            log_and_print(f"   Total spectra loaded: {total_spectra_loaded}")
            log_and_print(f"   Average quality score: {avg_quality_score:.3f}")
            log_and_print(f"   Loading speed: {avg_spectrum_count / max(0.001, avg_loading_time):.1f} spectra/s")

            # Step conclusion
            if success_rate >= 0.8 and avg_quality_score >= 0.7:
                step_conclusion = "🟢 DATA LOADING VALIDATION PASSED - mzML reader is highly effective"
                step_status = "validated"
            elif success_rate >= 0.6 and avg_quality_score >= 0.5:
                step_conclusion = "🟡 DATA LOADING VALIDATION PARTIAL - Good performance with minor issues"
                step_status = "functional"
            else:
                step_conclusion = "🔴 DATA LOADING VALIDATION FAILED - Significant issues detected"
                step_status = "problematic"

            log_and_print(f"\n🎯 STEP 1 CONCLUSION:")
            log_and_print(f"   {step_conclusion}")

            step_results['step_conclusion'] = {
                'overall_assessment': step_conclusion,
                'step_status': step_status,
                'success_rate': success_rate,
                'avg_quality_score': avg_quality_score,
                'key_findings': [
                    f"mzML loading success rate: {success_rate:.1%}",
                    f"Average data quality: {avg_quality_score:.3f}",
                    f"Loading performance: {avg_spectrum_count / max(0.001, avg_loading_time):.1f} spectra/s",
                    f"Total spectra processed: {total_spectra_loaded}"
                ],
                'recommendations': [
                    "mzML reader is reliable" if success_rate >= 0.8 else "Improve error handling in mzML reader",
                    "Data quality is acceptable" if avg_quality_score >= 0.6 else "Enhance data validation procedures",
                    "Loading speed is satisfactory" if avg_loading_time <= 5.0 else "Optimize file parsing algorithms"
                ]
            }

        else:
            log_and_print("❌ No successful data loading - critical failure in Step 1")
            step_results['step_conclusion'] = {
                'overall_assessment': '🔴 CRITICAL FAILURE - No data could be loaded',
                'step_status': 'failed'
            }

        # STEP 1.7: Save results and generate visualizations
        log_and_print(f"\n💾 SAVING STEP 1 RESULTS")
        log_and_print("-" * 30)

        # JSON results
        results_file = step_dir / "step_01_data_loading_results.json"
        with open(results_file, 'w') as f:
            json.dump(step_results, f, indent=2)
        log_and_print(f"📄 Saved JSON results: {results_file}")

        # CSV summary
        if successful_loads > 0:
            csv_data = []
            for dataset, result in step_results['loading_results'].items():
                if result.get('loading_success', False):
                    csv_data.append({
                        'Dataset': dataset,
                        'Loading_Time_s': result['loading_time'],
                        'Spectra_Loaded': result['spectra_loaded'],
                        'Spectra_Per_Second': result['spectra_per_second'],
                        'Quality_Score': result['quality_score'],
                        'Performance_Grade': result['performance_grade'],
                        'MS1_Count': result['spectrum_analysis']['ms1_count'],
                        'MS2_Count': result['spectrum_analysis']['ms2_count'],
                        'Avg_Peaks_Per_Spectrum': result['spectrum_analysis']['avg_peaks_per_spectrum']
                    })

            if csv_data:
                csv_file = step_dir / "step_01_data_loading_summary.csv"
                pd.DataFrame(csv_data).to_csv(csv_file, index=False)
                log_and_print(f"📊 Saved CSV summary: {csv_file}")

        # Generate visualizations
        try:
            from visualization.panel import create_data_loading_panel
            import matplotlib.pyplot as plt

            if successful_loads > 0:
                viz_data = {
                    'loading_times': all_loading_times,
                    'spectrum_counts': all_spectrum_counts,
                    'quality_scores': all_data_quality_scores,
                    'dataset_names': [f.replace('.mzML', '') for f in test_files]
                }

                fig = create_data_loading_panel(viz_data, "Step 1: Data Loading Performance")
                viz_file = step_dir / "step_01_data_loading_performance.png"
                fig.savefig(viz_file, dpi=300, bbox_inches='tight')
                plt.close(fig)
                log_and_print(f"📊 Saved visualization: {viz_file}")

        except ImportError as e:
            log_and_print(f"⚠️  Visualization generation failed: {e}")

        log_and_print(f"\n🧪 STEP 1: DATA LOADING VALIDATION COMPLETE 🧪")
        log_and_print(f"📁 Results saved to: {step_dir}")
        log_and_print(f"📋 Step log: {log_file}")

        return step_results

    except Exception as e:
        log_and_print(f"💥 CRITICAL STEP 1 FAILURE: {e}")
        import traceback
        log_and_print(f"Error details: {traceback.format_exc()}")
        return None


if __name__ == "__main__":
    print("🚀 Starting Step 1: Data Loading Validation Experiment...")

    results = main()

    if results and results.get('step_conclusion', {}).get('step_status') in ['validated', 'functional']:
        print("\n✅ STEP 1 SUCCESSFUL - Data loading validated!")
        sys.exit(0)
    else:
        print("\n❌ STEP 1 FAILED - Check results for details")
        sys.exit(1)
