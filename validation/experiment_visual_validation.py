#!/usr/bin/env python3
"""
VISUAL VALIDATION SCIENCE EXPERIMENT
===================================

This script runs as a standalone science experiment to validate
visual mass spectrometry processing capabilities.

EXPERIMENT OBJECTIVES:
- Validate Ion-to-Drip conversion accuracy
- Test LipidMaps annotation performance
- Measure visual spectrum analysis quality
- Assess drip image generation
- Benchmark visual processing speed

OUTPUTS:
- JSON results file
- CSV data summary
- Visual performance panels
- Drip conversion analysis
- Processing logs

Run: python experiment_visual_validation.py
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
    """Main visual validation experiment"""

    print("🧪 VISUAL VALIDATION SCIENCE EXPERIMENT")
    print("=" * 60)
    print(f"Experiment started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Objective: Validate visual mass spectrometry & Ion-to-Drip processing")
    print("=" * 60)

    # Create experiment directory
    experiment_dir = Path("experiment_results") / "visual_validation"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Initialize experiment log
    log_file = experiment_dir / "experiment_log.txt"

    def log_and_print(message):
        """Log message to both console and file"""
        print(message)
        with open(log_file, 'a') as f:
            f.write(f"{datetime.now().isoformat()}: {message}\n")

    log_and_print("🔬 VISUAL EXPERIMENTAL SETUP")
    log_and_print("-" * 35)

    try:
        # Import visual pipeline
        log_and_print("📦 Loading visual validation pipeline...")
        from core.visual_pipeline import VisualPipelineOrchestrator

        orchestrator = VisualPipelineOrchestrator()
        log_and_print("✅ Visual pipeline loaded successfully")
        log_and_print("  🌊 Ion-to-Drip converter initialized")
        log_and_print("  🧬 LipidMaps annotator ready")

        # Define experimental datasets
        datasets = [
            "PL_Neg_Waters_qTOF.mzML",  # Phospholipids, negative mode
            "TG_Pos_Thermo_Orbi.mzML"  # Triglycerides, positive mode
        ]

        log_and_print(f"📊 Visual experimental datasets: {len(datasets)}")
        for i, dataset in enumerate(datasets, 1):
            log_and_print(f"  {i}. {dataset}")

        # Initialize experiment results
        experiment_results = {
            'experiment_metadata': {
                'experiment_name': 'Visual Validation Experiment',
                'start_time': datetime.now().isoformat(),
                'datasets': datasets,
                'pipeline_version': 'standalone_visual_1.0',
                'objective': 'Validate visual MS processing & Ion-to-Drip conversion'
            },
            'dataset_results': {},
            'conversion_analysis': {},
            'annotation_performance': {},
            'visual_quality_metrics': {},
            'conclusions': {}
        }

        log_and_print("\n🚀 STARTING VISUAL EXPERIMENTAL VALIDATION")
        log_and_print("=" * 55)

        total_start_time = time.time()
        all_conversion_rates = []
        all_annotation_rates = []
        all_processing_times = []
        all_ion_extraction_rates = []

        # Process each dataset
        for experiment_num, dataset_name in enumerate(datasets, 1):
            log_and_print(f"\n🎨 VISUAL EXPERIMENT {experiment_num}/{len(datasets)}: {dataset_name}")
            log_and_print("-" * 60)

            dataset_start_time = time.time()

            try:
                # Step 1: Visual Data Processing
                log_and_print("STEP 1: Visual data processing & Ion-to-Drip conversion")
                log_and_print(f"  🔄 Processing visual dataset: {dataset_name}")

                # Run visual pipeline with Ion-to-Drip conversion
                results = orchestrator.process_dataset(
                    dataset_name,
                    create_visualizations=True,
                    save_drip_images=True
                )

                dataset_processing_time = time.time() - dataset_start_time

                # Step 2: Ion-to-Drip Conversion Analysis
                log_and_print("STEP 2: Ion-to-Drip conversion performance analysis")

                pipeline_info = results.get('pipeline_info', {})
                ion_conversion = results.get('ion_conversion', {})
                visual_summary = results.get('visual_processing_summary', {})
                lipid_annotation = results.get('lipidmaps_annotation', {})

                # Extract conversion metrics
                ion_stats = ion_conversion.get('statistics', {})
                spectra_processed = visual_summary.get('spectra_processed', 0)
                ions_extracted = visual_summary.get('ions_extracted', 0)
                drip_images_created = visual_summary.get('drip_images_created', 0)
                annotations_generated = visual_summary.get('annotations_generated', 0)
                processing_time = pipeline_info.get('processing_time', 0)

                # Calculate key performance indicators
                ion_extraction_rate = ions_extracted / max(1, spectra_processed)
                drip_conversion_rate = drip_images_created / max(1, spectra_processed)
                annotation_rate = annotations_generated / max(1, drip_images_created)

                log_and_print(f"  📈 ION-TO-DRIP CONVERSION RESULTS:")
                log_and_print(f"     Spectra processed: {spectra_processed}")
                log_and_print(f"     Ions extracted: {ions_extracted} ({ion_extraction_rate:.1f} ions/spectrum)")
                log_and_print(f"     Drip images created: {drip_images_created} ({drip_conversion_rate:.1%})")
                log_and_print(f"     Processing time: {processing_time:.2f}s")

                # Step 3: Ion Type Distribution Analysis
                log_and_print("STEP 3: Ion type distribution analysis")

                ion_type_distribution = ion_stats.get('ion_type_distribution', {})
                total_ions_by_type = sum(ion_type_distribution.values())

                log_and_print(f"  ⚛️  Ion type analysis (Total: {total_ions_by_type}):")
                for ion_type, count in ion_type_distribution.items():
                    percentage = (count / max(1, total_ions_by_type)) * 100
                    log_and_print(f"     {ion_type}: {count} ions ({percentage:.1f}%)")

                # Step 4: LipidMaps Annotation Performance
                log_and_print("STEP 4: LipidMaps annotation performance")

                annotated_spectra = lipid_annotation.get('annotated_spectra', 0)
                total_annotations = lipid_annotation.get('total_annotations', 0)

                log_and_print(f"  🧬 LIPIDMAPS ANNOTATION RESULTS:")
                log_and_print(f"     Annotated drip spectra: {annotated_spectra}")
                log_and_print(f"     Total annotations: {total_annotations}")
                log_and_print(f"     Annotation rate: {annotation_rate:.1%}")

                # Sample annotations analysis
                sample_annotations = lipid_annotation.get('sample_annotations', [])
                if sample_annotations:
                    log_and_print(f"     Sample annotations (first 3):")
                    for i, ann in enumerate(sample_annotations[:3], 1):
                        if isinstance(ann, dict):
                            lipid_name = ann.get('lipid_name', 'Unknown')
                            confidence = ann.get('confidence_score', 0)
                            log_and_print(f"       {i}. {lipid_name}: confidence={confidence:.3f}")

                # Step 5: Visual Quality Assessment
                log_and_print("STEP 5: Visual processing quality assessment")

                # Quality indicators
                drip_generation_success = drip_images_created > 0
                ion_extraction_success = ions_extracted > 0
                annotation_success = annotated_spectra > 0

                quality_score = 0.0
                if drip_generation_success:
                    quality_score += 0.4
                if ion_extraction_success:
                    quality_score += 0.3
                if annotation_success:
                    quality_score += 0.3

                log_and_print(f"  🎯 VISUAL QUALITY METRICS:")
                log_and_print(f"     Drip generation: {'✅ SUCCESS' if drip_generation_success else '❌ FAILED'}")
                log_and_print(f"     Ion extraction: {'✅ SUCCESS' if ion_extraction_success else '❌ FAILED'}")
                log_and_print(f"     LipidMaps annotation: {'✅ SUCCESS' if annotation_success else '❌ FAILED'}")
                log_and_print(f"     Overall quality score: {quality_score:.2f}/1.00")

                # Step 6: Performance Classification
                log_and_print("STEP 6: Visual processing performance assessment")

                # Classify performance based on multiple criteria
                if drip_conversion_rate >= 0.8 and annotation_rate >= 0.6 and ion_extraction_rate >= 5.0:
                    performance_grade = "🟢 EXCELLENT"
                elif drip_conversion_rate >= 0.6 and annotation_rate >= 0.4 and ion_extraction_rate >= 3.0:
                    performance_grade = "🟡 GOOD"
                elif drip_conversion_rate >= 0.4 and annotation_rate >= 0.2 and ion_extraction_rate >= 2.0:
                    performance_grade = "🟠 ACCEPTABLE"
                else:
                    performance_grade = "🔴 NEEDS IMPROVEMENT"

                log_and_print(f"  🏆 Visual Performance Grade: {performance_grade}")

                # Store detailed results
                experiment_results['dataset_results'][dataset_name] = {
                    'processing_metrics': {
                        'spectra_processed': spectra_processed,
                        'processing_time': processing_time,
                        'spectra_per_second': spectra_processed / max(0.001, processing_time)
                    },
                    'conversion_metrics': {
                        'ions_extracted': ions_extracted,
                        'ion_extraction_rate': ion_extraction_rate,
                        'drip_images_created': drip_images_created,
                        'drip_conversion_rate': drip_conversion_rate,
                        'ion_type_distribution': ion_type_distribution,
                        'conversion_success': drip_generation_success
                    },
                    'annotation_metrics': {
                        'annotated_spectra': annotated_spectra,
                        'total_annotations': total_annotations,
                        'annotation_rate': annotation_rate,
                        'annotation_success': annotation_success
                    },
                    'quality_metrics': {
                        'overall_quality_score': quality_score,
                        'drip_generation_success': drip_generation_success,
                        'ion_extraction_success': ion_extraction_success,
                        'annotation_success': annotation_success
                    },
                    'performance_assessment': {
                        'grade': performance_grade,
                        'conversion_efficiency': drip_conversion_rate,
                        'annotation_efficiency': annotation_rate,
                        'processing_speed': spectra_processed / max(0.001, processing_time)
                    },
                    'raw_results': results
                }

                # Track overall metrics
                all_conversion_rates.append(drip_conversion_rate)
                all_annotation_rates.append(annotation_rate)
                all_processing_times.append(processing_time)
                all_ion_extraction_rates.append(ion_extraction_rate)

                log_and_print(f"✅ Visual Experiment {experiment_num} completed successfully")

            except Exception as e:
                log_and_print(f"❌ Visual Experiment {experiment_num} failed: {e}")
                import traceback
                error_details = traceback.format_exc()
                log_and_print(f"Error details: {error_details}")

                # Store failure information
                experiment_results['dataset_results'][dataset_name] = {
                    'status': 'failed',
                    'error': str(e),
                    'error_details': error_details
                }

        # Final Analysis and Conclusions
        total_experiment_time = time.time() - total_start_time

        log_and_print(f"\n" + "=" * 60)
        log_and_print("📊 VISUAL EXPERIMENTAL CONCLUSIONS")
        log_and_print("=" * 60)

        successful_experiments = len([r for r in experiment_results['dataset_results'].values()
                                    if 'status' not in r or r.get('status') != 'failed'])

        if successful_experiments > 0:
            # Calculate overall performance metrics
            avg_conversion_rate = np.mean(all_conversion_rates) if all_conversion_rates else 0
            avg_annotation_rate = np.mean(all_annotation_rates) if all_annotation_rates else 0
            avg_processing_time = np.mean(all_processing_times) if all_processing_times else 0
            avg_ion_extraction_rate = np.mean(all_ion_extraction_rates) if all_ion_extraction_rates else 0

            experiment_results['conversion_analysis'] = {
                'average_conversion_rate': avg_conversion_rate,
                'average_ion_extraction_rate': avg_ion_extraction_rate,
                'conversion_consistency': np.std(all_conversion_rates) if len(all_conversion_rates) > 1 else 0,
                'total_converted_spectra': sum([r.get('conversion_metrics', {}).get('drip_images_created', 0)
                                              for r in experiment_results['dataset_results'].values()
                                              if 'conversion_metrics' in r])
            }

            experiment_results['annotation_performance'] = {
                'average_annotation_rate': avg_annotation_rate,
                'annotation_consistency': np.std(all_annotation_rates) if len(all_annotation_rates) > 1 else 0,
                'total_annotations': sum([r.get('annotation_metrics', {}).get('total_annotations', 0)
                                        for r in experiment_results['dataset_results'].values()
                                        if 'annotation_metrics' in r])
            }

            experiment_results['visual_quality_metrics'] = {
                'successful_experiments': successful_experiments,
                'total_experiments': len(datasets),
                'success_rate': successful_experiments / len(datasets),
                'average_processing_time': avg_processing_time,
                'total_experiment_time': total_experiment_time
            }

            log_and_print(f"🔢 OVERALL VISUAL EXPERIMENTAL STATISTICS:")
            log_and_print(f"   Successful experiments: {successful_experiments}/{len(datasets)}")
            log_and_print(f"   Success rate: {(successful_experiments/len(datasets)):.1%}")
            log_and_print(f"   Average drip conversion rate: {avg_conversion_rate:.1%}")
            log_and_print(f"   Average ion extraction rate: {avg_ion_extraction_rate:.1f} ions/spectrum")
            log_and_print(f"   Average annotation rate: {avg_annotation_rate:.1%}")
            log_and_print(f"   Average processing time: {avg_processing_time:.2f}s per dataset")
            log_and_print(f"   Total experiment duration: {total_experiment_time:.2f}s")

            # Scientific conclusions
            if avg_conversion_rate >= 0.7 and avg_annotation_rate >= 0.5 and avg_ion_extraction_rate >= 4.0:
                overall_conclusion = "🟢 VISUAL PIPELINE VALIDATED - Ion-to-Drip conversion is highly effective"
            elif avg_conversion_rate >= 0.5 and avg_annotation_rate >= 0.3 and avg_ion_extraction_rate >= 2.5:
                overall_conclusion = "🟡 VISUAL PIPELINE FUNCTIONAL - Good performance with room for optimization"
            else:
                overall_conclusion = "🔴 VISUAL PIPELINE REQUIRES IMPROVEMENT - Conversion efficiency below threshold"

            log_and_print(f"\n🎯 VISUAL SCIENTIFIC CONCLUSION:")
            log_and_print(f"   {overall_conclusion}")

            experiment_results['conclusions'] = {
                'overall_assessment': overall_conclusion,
                'visual_pipeline_status': 'validated' if avg_conversion_rate >= 0.7 else 'needs_improvement',
                'ion_drip_conversion_status': 'effective' if avg_conversion_rate >= 0.6 else 'ineffective',
                'lipidmaps_annotation_status': 'functional' if avg_annotation_rate >= 0.4 else 'problematic',
                'key_findings': [
                    f"Ion-to-Drip conversion rate: {avg_conversion_rate:.1%}",
                    f"Ion extraction efficiency: {avg_ion_extraction_rate:.1f} ions/spectrum",
                    f"LipidMaps annotation rate: {avg_annotation_rate:.1%}",
                    f"Visual processing speed: {np.mean([r.get('processing_metrics', {}).get('spectra_per_second', 0) for r in experiment_results['dataset_results'].values() if 'processing_metrics' in r]):.1f} spectra/s"
                ],
                'recommendations': [
                    "Ion-to-Drip conversion is effective" if avg_conversion_rate >= 0.6 else "Improve ion extraction algorithms",
                    "LipidMaps integration is satisfactory" if avg_annotation_rate >= 0.4 else "Enhance lipid annotation coverage",
                    "Processing speed is acceptable" if avg_processing_time <= 15.0 else "Optimize visual processing pipeline",
                    "Ion type distribution analysis successful" if all_ion_extraction_rates else "Ion classification needs improvement"
                ]
            }

        else:
            log_and_print("❌ No successful visual experiments - unable to draw conclusions")
            experiment_results['conclusions'] = {
                'overall_assessment': '🔴 VISUAL EXPERIMENTAL FAILURE - All tests failed',
                'status': 'failed'
            }

        # Save comprehensive results
        log_and_print(f"\n💾 SAVING VISUAL EXPERIMENTAL RESULTS")
        log_and_print("-" * 45)

        # JSON results
        results_file = experiment_dir / "visual_validation_results.json"
        with open(results_file, 'w') as f:
            json.dump(experiment_results, f, indent=2)
        log_and_print(f"📄 Saved JSON results: {results_file}")

        # CSV summary
        if successful_experiments > 0:
            summary_data = []
            for dataset_name, result in experiment_results['dataset_results'].items():
                if 'processing_metrics' in result:
                    summary_data.append({
                        'Dataset': dataset_name,
                        'Spectra_Processed': result['processing_metrics']['spectra_processed'],
                        'Ions_Extracted': result['conversion_metrics']['ions_extracted'],
                        'Ion_Extraction_Rate': result['conversion_metrics']['ion_extraction_rate'],
                        'Drip_Images_Created': result['conversion_metrics']['drip_images_created'],
                        'Drip_Conversion_Rate': result['conversion_metrics']['drip_conversion_rate'],
                        'Annotations_Generated': result['annotation_metrics']['total_annotations'],
                        'Annotation_Rate': result['annotation_metrics']['annotation_rate'],
                        'Processing_Time_s': result['processing_metrics']['processing_time'],
                        'Performance_Grade': result['performance_assessment']['grade'],
                        'Quality_Score': result['quality_metrics']['overall_quality_score']
                    })

            if summary_data:
                csv_file = experiment_dir / "visual_validation_summary.csv"
                pd.DataFrame(summary_data).to_csv(csv_file, index=False)
                log_and_print(f"📊 Saved CSV summary: {csv_file}")

        # Generate visualizations
        try:
            from visualization.panel import create_visual_performance_panel
            import matplotlib.pyplot as plt

            if successful_experiments > 0:
                # Visual performance visualization
                performance_data = {
                    'conversion_rates': all_conversion_rates,
                    'annotation_rates': all_annotation_rates,
                    'ion_extraction_rates': all_ion_extraction_rates,
                    'processing_times': all_processing_times,
                    'dataset_names': [name for name in datasets if name in experiment_results['dataset_results'] and 'processing_metrics' in experiment_results['dataset_results'][name]]
                }

                fig = create_visual_performance_panel(performance_data, "Visual Validation Experimental Results")
                viz_file = experiment_dir / "visual_validation_performance.png"
                fig.savefig(viz_file, dpi=300, bbox_inches='tight')
                plt.close(fig)
                log_and_print(f"📊 Saved visualization: {viz_file}")

        except ImportError as e:
            log_and_print(f"⚠️  Visualization generation failed: {e}")

        log_and_print(f"\n🧪 VISUAL VALIDATION EXPERIMENT COMPLETE 🧪")
        log_and_print(f"📁 All results saved to: {experiment_dir}")
        log_and_print(f"📋 Experiment log: {log_file}")

        return experiment_results

    except Exception as e:
        log_and_print(f"💥 CRITICAL VISUAL EXPERIMENTAL FAILURE: {e}")
        import traceback
        log_and_print(f"Error details: {traceback.format_exc()}")
        return None


if __name__ == "__main__":
    print("🚀 Starting Visual Validation Science Experiment...")

    results = main()

    if results and results.get('conclusions', {}).get('visual_pipeline_status') == 'validated':
        print("\n✅ EXPERIMENT SUCCESSFUL - Visual pipeline & Ion-to-Drip conversion validated!")
        sys.exit(0)
    else:
        print("\n❌ EXPERIMENT FAILED - Check results for details")
        sys.exit(1)
