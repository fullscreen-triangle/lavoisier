#!/usr/bin/env python3
"""
STEP 6: ION EXTRACTION VALIDATION EXPERIMENT
==========================================

OBJECTIVE: Validate ion extraction from mass spectra for visual processing
HYPOTHESIS: The ion extraction system can effectively convert spectra to ion representations

EXPERIMENT PROCEDURE:
1. Load mass spectrometry data
2. Extract ions from spectra using different methods
3. Analyze ion type distributions and extraction rates
4. Evaluate ion quality and diversity
5. Generate ion extraction performance visualizations

Run: python step_06_ion_extraction_experiment.py
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
    """Step 6: Ion Extraction Validation Experiment"""

    print("🧪 STEP 6: ION EXTRACTION VALIDATION EXPERIMENT")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Objective: Validate ion extraction from mass spectra")
    print("=" * 60)

    # Create step-specific results directory
    step_dir = Path("step_results") / "step_06_ion_extraction"
    step_dir.mkdir(parents=True, exist_ok=True)

    # Initialize step log
    log_file = step_dir / "ion_extraction_log.txt"

    def log_and_print(message):
        """Log message to both console and file"""
        print(message)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"{datetime.now().isoformat()}: {message}\n")

    log_and_print("🔬 STEP 6 EXPERIMENTAL SETUP")
    log_and_print("-" * 30)

    try:
        # Import required components
        log_and_print("📦 Loading ion extraction components...")
        from core.mzml_reader import StandaloneMzMLReader
        from core.visual_pipeline import IonDripConverter, Ion
        from core.numerical_pipeline import QualityControlModule

        reader = StandaloneMzMLReader()
        ion_converter = IonDripConverter()
        qc_module = QualityControlModule()
        log_and_print("✅ Ion extraction components loaded successfully")

        # Define test datasets
        test_files = [
            "PL_Neg_Waters_qTOF.mzML",
            "TG_Pos_Thermo_Orbi.mzML"
        ]

        log_and_print(f"📊 Testing ion extraction on {len(test_files)} datasets:")
        for i, file in enumerate(test_files, 1):
            log_and_print(f"  {i}. {file}")

        # Initialize results
        step_results = {
            'step_metadata': {
                'step_number': 6,
                'step_name': 'Ion Extraction Validation',
                'start_time': datetime.now().isoformat(),
                'objective': 'Validate ion extraction from mass spectra for visual processing'
            },
            'ion_extraction_results': {},
            'ion_analysis': {},
            'extraction_performance_metrics': {},
            'step_conclusion': {}
        }

        log_and_print("\n🚀 STARTING ION EXTRACTION VALIDATION")
        log_and_print("-" * 45)

        all_extraction_times = []
        all_ion_extraction_rates = []
        all_ion_type_distributions = []
        total_ions_extracted = 0

        # Test each dataset
        for dataset_num, test_file in enumerate(test_files, 1):
            log_and_print(f"\n⚛️  DATASET {dataset_num}: {test_file}")
            log_and_print("-" * 40)

            dataset_start_time = time.time()

            try:
                # STEP 6.1: Load and prepare data for ion extraction
                log_and_print("STEP 6.1: Loading data for ion extraction...")
                spectra = reader.load_mzml(test_file)

                # Filter for high-quality spectra
                high_quality_spectra = []
                for spectrum in spectra:
                    quality_metrics = qc_module.assess_spectrum_quality(spectrum)
                    if quality_metrics.get('quality_score', 0) >= 0.2:
                        high_quality_spectra.append(spectrum)

                # Limit for performance testing
                extraction_spectra = high_quality_spectra[:25]  # Test first 25 spectra

                log_and_print(f"  📂 Loaded {len(spectra)} total spectra")
                log_and_print(f"  🎯 Selected {len(extraction_spectra)} spectra for ion extraction")

                # STEP 6.2: Ion extraction performance testing
                log_and_print("STEP 6.2: Performing ion extraction...")

                extraction_start = time.time()
                extracted_ions = []
                extraction_errors = 0
                ion_type_counts = {}

                for spectrum in extraction_spectra:
                    try:
                        # Extract ions from spectrum
                        ions = ion_converter.spectrum_to_ions(spectrum)

                        if ions:
                            extracted_ions.extend(ions)

                            # Count ion types
                            for ion in ions:
                                ion_type = ion.ion_type
                                ion_type_counts[ion_type] = ion_type_counts.get(ion_type, 0) + 1

                    except Exception as e:
                        extraction_errors += 1
                        log_and_print(f"      ⚠️  Ion extraction failed for {spectrum.scan_id}: {e}")

                extraction_time = time.time() - extraction_start

                # STEP 6.3: Ion extraction analysis
                log_and_print("STEP 6.3: Analyzing extracted ions...")

                ions_per_spectrum = len(extracted_ions) / max(1, len(extraction_spectra))
                extraction_success_rate = (len(extraction_spectra) - extraction_errors) / max(1, len(extraction_spectra))

                # Ion type distribution analysis
                total_ions_this_dataset = len(extracted_ions)
                ion_type_percentages = {}

                for ion_type, count in ion_type_counts.items():
                    percentage = (count / max(1, total_ions_this_dataset)) * 100
                    ion_type_percentages[ion_type] = percentage

                log_and_print(f"  📊 ION EXTRACTION RESULTS:")
                log_and_print(f"     Total ions extracted: {len(extracted_ions)}")
                log_and_print(f"     Ions per spectrum: {ions_per_spectrum:.1f}")
                log_and_print(f"     Extraction success rate: {extraction_success_rate:.1%}")
                log_and_print(f"     Extraction errors: {extraction_errors}")
                log_and_print(f"     Extraction time: {extraction_time:.2f}s")

                if ion_type_counts:
                    log_and_print(f"  ⚛️  ION TYPE DISTRIBUTION:")
                    sorted_ion_types = sorted(ion_type_counts.items(), key=lambda x: x[1], reverse=True)
                    for ion_type, count in sorted_ion_types:
                        percentage = ion_type_percentages[ion_type]
                        log_and_print(f"     {ion_type}: {count} ions ({percentage:.1f}%)")

                # STEP 6.4: Ion quality assessment
                log_and_print("STEP 6.4: Assessing ion quality...")

                if extracted_ions:
                    # Analyze ion properties
                    ion_masses = [ion.mass for ion in extracted_ions]
                    ion_intensities = [ion.intensity for ion in extracted_ions]

                    # Mass distribution analysis
                    mass_range = max(ion_masses) - min(ion_masses) if ion_masses else 0
                    avg_mass = np.mean(ion_masses) if ion_masses else 0
                    mass_std = np.std(ion_masses) if len(ion_masses) > 1 else 0

                    # Intensity distribution analysis
                    avg_intensity = np.mean(ion_intensities) if ion_intensities else 0
                    intensity_std = np.std(ion_intensities) if len(ion_intensities) > 1 else 0

                    # Ion diversity assessment
                    unique_ion_types = len(set(ion.ion_type for ion in extracted_ions))
                    ion_diversity_score = unique_ion_types / max(1, len(ion_type_counts))

                    # Quality metrics
                    mass_diversity = mass_std / max(1, avg_mass) if avg_mass > 0 else 0
                    intensity_diversity = intensity_std / max(1, avg_intensity) if avg_intensity > 0 else 0

                    ion_quality_score = 0.0

                    # Factor 1: Extraction success (30%)
                    ion_quality_score += extraction_success_rate * 0.3

                    # Factor 2: Ion yield (25%)
                    if ions_per_spectrum >= 5.0:
                        ion_quality_score += 0.25
                    elif ions_per_spectrum >= 2.0:
                        ion_quality_score += 0.15

                    # Factor 3: Ion diversity (25%)
                    ion_quality_score += ion_diversity_score * 0.25

                    # Factor 4: Data distribution quality (20%)
                    if mass_diversity >= 0.1 and intensity_diversity >= 0.2:
                        ion_quality_score += 0.2
                    elif mass_diversity >= 0.05 or intensity_diversity >= 0.1:
                        ion_quality_score += 0.1

                    log_and_print(f"  🎯 ION QUALITY ASSESSMENT:")
                    log_and_print(f"     Mass range: {mass_range:.1f} Da")
                    log_and_print(f"     Average mass: {avg_mass:.1f} Da (std: {mass_std:.1f})")
                    log_and_print(f"     Average intensity: {avg_intensity:.0f} (std: {intensity_std:.0f})")
                    log_and_print(f"     Ion type diversity: {unique_ion_types} types")
                    log_and_print(f"     Ion quality score: {ion_quality_score:.3f}")

                else:
                    ion_quality_score = 0.0
                    log_and_print(f"  ❌ No ions extracted - quality score: 0.000")

                # STEP 6.5: Extraction efficiency analysis
                log_and_print("STEP 6.5: Extraction efficiency analysis...")

                extraction_rate = len(extracted_ions) / max(0.001, extraction_time)  # ions/second
                spectrum_processing_rate = len(extraction_spectra) / max(0.001, extraction_time)  # spectra/second

                # Efficiency score based on speed and yield
                efficiency_score = 0.0

                # Speed component (50%)
                if extraction_rate >= 50:  # 50 ions/second is excellent
                    efficiency_score += 0.5
                elif extraction_rate >= 20:
                    efficiency_score += 0.3
                elif extraction_rate >= 5:
                    efficiency_score += 0.1

                # Yield component (50%)
                if ions_per_spectrum >= 10:
                    efficiency_score += 0.5
                elif ions_per_spectrum >= 5:
                    efficiency_score += 0.3
                elif ions_per_spectrum >= 2:
                    efficiency_score += 0.1

                log_and_print(f"  ⚡ EXTRACTION EFFICIENCY:")
                log_and_print(f"     Extraction rate: {extraction_rate:.1f} ions/s")
                log_and_print(f"     Processing rate: {spectrum_processing_rate:.1f} spectra/s")
                log_and_print(f"     Efficiency score: {efficiency_score:.3f}")

                # STEP 6.6: Performance classification
                log_and_print("STEP 6.6: Ion extraction performance classification...")

                if (ion_quality_score >= 0.7 and efficiency_score >= 0.6 and
                    extraction_success_rate >= 0.8):
                    performance_grade = "🟢 EXCELLENT"
                elif ion_quality_score >= 0.5 and efficiency_score >= 0.4:
                    performance_grade = "🟡 GOOD"
                elif ion_quality_score >= 0.3 and efficiency_score >= 0.2:
                    performance_grade = "🟠 ACCEPTABLE"
                else:
                    performance_grade = "🔴 NEEDS IMPROVEMENT"

                log_and_print(f"  🏆 Ion Extraction Performance: {performance_grade}")

                # Store results
                step_results['ion_extraction_results'][test_file] = {
                    'spectra_processed': len(extraction_spectra),
                    'extraction_time': extraction_time,
                    'extraction_rate': extraction_rate,
                    'spectrum_processing_rate': spectrum_processing_rate,
                    'extraction_statistics': {
                        'total_ions_extracted': len(extracted_ions),
                        'ions_per_spectrum': ions_per_spectrum,
                        'extraction_success_rate': extraction_success_rate,
                        'extraction_errors': extraction_errors,
                        'ion_type_counts': ion_type_counts,
                        'ion_type_percentages': ion_type_percentages
                    },
                    'ion_quality_metrics': {
                        'ion_quality_score': ion_quality_score,
                        'mass_range': mass_range if extracted_ions else 0,
                        'avg_mass': avg_mass if extracted_ions else 0,
                        'avg_intensity': avg_intensity if extracted_ions else 0,
                        'unique_ion_types': unique_ion_types if extracted_ions else 0,
                        'ion_diversity_score': ion_diversity_score if extracted_ions else 0
                    },
                    'efficiency_metrics': {
                        'efficiency_score': efficiency_score,
                        'extraction_rate': extraction_rate,
                        'spectrum_processing_rate': spectrum_processing_rate
                    },
                    'performance_grade': performance_grade
                }

                # Track overall metrics
                all_extraction_times.append(extraction_time)
                all_ion_extraction_rates.append(ions_per_spectrum)
                all_ion_type_distributions.append(ion_type_counts)
                total_ions_extracted += len(extracted_ions)

                log_and_print(f"✅ Dataset {dataset_num} ion extraction validation completed")

            except Exception as e:
                log_and_print(f"❌ Dataset {dataset_num} ion extraction failed: {e}")

                step_results['ion_extraction_results'][test_file] = {
                    'extraction_success': False,
                    'error': str(e),
                    'processing_time': time.time() - dataset_start_time
                }

        # STEP 6.7: Overall analysis and conclusions
        log_and_print(f"\n" + "=" * 60)
        log_and_print("📊 STEP 6 OVERALL ION EXTRACTION ANALYSIS")
        log_and_print("=" * 60)

        successful_extractions = len([r for r in step_results['ion_extraction_results'].values()
                                    if 'extraction_success' not in r or r.get('extraction_success', True)])
        total_datasets = len(test_files)
        success_rate = successful_extractions / max(1, total_datasets)

        if successful_extractions > 0:
            avg_extraction_time = np.mean(all_extraction_times)
            avg_ion_extraction_rate = np.mean(all_ion_extraction_rates)
            overall_extraction_rate = total_ions_extracted / max(0.001, sum(all_extraction_times))

            # Combined ion type analysis
            combined_ion_types = {}
            for ion_dist in all_ion_type_distributions:
                for ion_type, count in ion_dist.items():
                    combined_ion_types[ion_type] = combined_ion_types.get(ion_type, 0) + count

            # Most and least common ion types
            if combined_ion_types:
                most_common_ion = max(combined_ion_types.keys(), key=lambda x: combined_ion_types[x])
                least_common_ion = min(combined_ion_types.keys(), key=lambda x: combined_ion_types[x])
            else:
                most_common_ion = None
                least_common_ion = None

            step_results['ion_analysis'] = {
                'total_ions_extracted': total_ions_extracted,
                'avg_ion_extraction_rate': avg_ion_extraction_rate,
                'combined_ion_type_distribution': combined_ion_types,
                'most_common_ion_type': most_common_ion,
                'least_common_ion_type': least_common_ion,
                'ion_type_diversity': len(combined_ion_types)
            }

            step_results['extraction_performance_metrics'] = {
                'successful_extractions': successful_extractions,
                'total_datasets': total_datasets,
                'success_rate': success_rate,
                'avg_extraction_time': avg_extraction_time,
                'avg_ion_extraction_rate': avg_ion_extraction_rate,
                'overall_extraction_rate': overall_extraction_rate,
                'total_ions_extracted': total_ions_extracted
            }

            log_and_print(f"🔢 STEP 6 PERFORMANCE METRICS:")
            log_and_print(f"   Successful ion extractions: {successful_extractions}/{total_datasets}")
            log_and_print(f"   Success rate: {success_rate:.1%}")
            log_and_print(f"   Total ions extracted: {total_ions_extracted}")
            log_and_print(f"   Average extraction time: {avg_extraction_time:.2f}s per dataset")
            log_and_print(f"   Average ions per spectrum: {avg_ion_extraction_rate:.1f}")
            log_and_print(f"   Overall extraction rate: {overall_extraction_rate:.1f} ions/s")

            # Ion type distribution summary
            if combined_ion_types:
                log_and_print(f"\n⚛️  OVERALL ION TYPE DISTRIBUTION:")
                sorted_combined = sorted(combined_ion_types.items(), key=lambda x: x[1], reverse=True)
                for ion_type, total_count in sorted_combined:
                    percentage = (total_count / max(1, total_ions_extracted)) * 100
                    log_and_print(f"   {ion_type}: {total_count} ions ({percentage:.1f}%)")

            # Step conclusion
            if (success_rate >= 0.8 and avg_ion_extraction_rate >= 5.0 and
                total_ions_extracted >= 50):
                step_conclusion = "🟢 ION EXTRACTION VALIDATION PASSED - Effective ion extraction capabilities"
                step_status = "validated"
            elif success_rate >= 0.6 and avg_ion_extraction_rate >= 3.0:
                step_conclusion = "🟡 ION EXTRACTION VALIDATION PARTIAL - Good extraction with room for optimization"
                step_status = "functional"
            else:
                step_conclusion = "🔴 ION EXTRACTION VALIDATION FAILED - Poor extraction performance"
                step_status = "problematic"

            log_and_print(f"\n🎯 STEP 6 CONCLUSION:")
            log_and_print(f"   {step_conclusion}")

            step_results['step_conclusion'] = {
                'overall_assessment': step_conclusion,
                'step_status': step_status,
                'success_rate': success_rate,
                'avg_ion_extraction_rate': avg_ion_extraction_rate,
                'total_ions_extracted': total_ions_extracted,
                'ion_type_diversity': len(combined_ion_types),
                'key_findings': [
                    f"Ion extraction success rate: {success_rate:.1%}",
                    f"Average ions per spectrum: {avg_ion_extraction_rate:.1f}",
                    f"Total ions extracted: {total_ions_extracted}",
                    f"Ion type diversity: {len(combined_ion_types)} different types"
                ],
                'recommendations': [
                    "Ion extraction is effective" if avg_ion_extraction_rate >= 5.0 else "Improve ion extraction algorithms",
                    "Processing speed is acceptable" if avg_extraction_time <= 10.0 else "Optimize extraction computation",
                    "Ion diversity is good" if len(combined_ion_types) >= 3 else "Enhance ion type recognition",
                    f"Focus on {most_common_ion} ion type" if most_common_ion else "Improve overall ion detection"
                ]
            }

        else:
            log_and_print("❌ No successful ion extractions - critical failure in Step 6")
            step_results['step_conclusion'] = {
                'overall_assessment': '🔴 CRITICAL FAILURE - Ion extraction system completely non-functional',
                'step_status': 'failed'
            }

        # STEP 6.8: Save results and generate visualizations
        log_and_print(f"\n💾 SAVING STEP 6 RESULTS")
        log_and_print("-" * 30)

        # JSON results
        results_file = step_dir / "step_06_ion_extraction_results.json"
        with open(results_file, 'w') as f:
            json.dump(step_results, f, indent=2)
        log_and_print(f"📄 Saved JSON results: {results_file}")

        # CSV summary
        if successful_extractions > 0:
            csv_data = []
            for dataset, result in step_results['ion_extraction_results'].items():
                if 'spectra_processed' in result:
                    extraction_stats = result['extraction_statistics']
                    quality_metrics = result['ion_quality_metrics']
                    efficiency_metrics = result['efficiency_metrics']

                    csv_data.append({
                        'Dataset': dataset,
                        'Spectra_Processed': result['spectra_processed'],
                        'Extraction_Time_s': result['extraction_time'],
                        'Total_Ions_Extracted': extraction_stats['total_ions_extracted'],
                        'Ions_Per_Spectrum': extraction_stats['ions_per_spectrum'],
                        'Extraction_Success_Rate': extraction_stats['extraction_success_rate'],
                        'Ion_Quality_Score': quality_metrics['ion_quality_score'],
                        'Unique_Ion_Types': quality_metrics['unique_ion_types'],
                        'Extraction_Rate': efficiency_metrics['extraction_rate'],
                        'Efficiency_Score': efficiency_metrics['efficiency_score'],
                        'Performance_Grade': result['performance_grade']
                    })

            if csv_data:
                csv_file = step_dir / "step_06_ion_extraction_summary.csv"
                pd.DataFrame(csv_data).to_csv(csv_file, index=False)
                log_and_print(f"📊 Saved CSV summary: {csv_file}")

        # Generate visualizations
        try:
            from visualization.panel import create_ion_extraction_panel
            import matplotlib.pyplot as plt

            if successful_extractions > 0:
                viz_data = {
                    'extraction_times': all_extraction_times,
                    'ion_extraction_rates': all_ion_extraction_rates,
                    'ion_type_distribution': combined_ion_types,
                    'dataset_names': [f.replace('.mzML', '') for f in test_files]
                }

                fig = create_ion_extraction_panel(viz_data, "Step 6: Ion Extraction Performance")
                viz_file = step_dir / "step_06_ion_extraction_performance.png"
                fig.savefig(viz_file, dpi=300, bbox_inches='tight')
                plt.close(fig)
                log_and_print(f"📊 Saved visualization: {viz_file}")

        except ImportError as e:
            log_and_print(f"⚠️  Visualization generation failed: {e}")

        log_and_print(f"\n🧪 STEP 6: ION EXTRACTION VALIDATION COMPLETE 🧪")
        log_and_print(f"📁 Results saved to: {step_dir}")
        log_and_print(f"📋 Step log: {log_file}")

        return step_results

    except Exception as e:
        log_and_print(f"💥 CRITICAL STEP 6 FAILURE: {e}")
        import traceback
        log_and_print(f"Error details: {traceback.format_exc()}")
        return None


if __name__ == "__main__":
    print("🚀 Starting Step 6: Ion Extraction Validation Experiment...")

    results = main()

    if results and results.get('step_conclusion', {}).get('step_status') in ['validated', 'functional']:
        print("\n✅ STEP 6 SUCCESSFUL - Ion extraction validated!")
        sys.exit(0)
    else:
        print("\n❌ STEP 6 FAILED - Check results for details")
        sys.exit(1)
