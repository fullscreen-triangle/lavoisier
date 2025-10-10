#!/usr/bin/env python3
"""
NUMERICAL VALIDATION SCIENCE EXPERIMENT
======================================

This script runs as a standalone science experiment to validate
numerical mass spectrometry processing capabilities.

EXPERIMENT OBJECTIVES:
- Validate numerical pipeline performance
- Test database annotation accuracy
- Measure spectrum embedding quality
- Assess quality control metrics
- Benchmark processing speed

OUTPUTS:
- JSON results file
- CSV data summary
- Performance visualization panels
- Processing logs

Run: python experiment_numerical_validation.py
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
    """Main numerical validation experiment"""

    print("🧪 NUMERICAL VALIDATION SCIENCE EXPERIMENT")
    print("=" * 60)
    print(f"Experiment started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Objective: Validate numerical mass spectrometry processing")
    print("=" * 60)

    # Create experiment directory
    experiment_dir = Path("experiment_results") / "numerical_validation"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Initialize experiment log
    log_file = experiment_dir / "experiment_log.txt"

    def log_and_print(message):
        """Log message to both console and file"""
        print(message)
        with open(log_file, 'a') as f:
            f.write(f"{datetime.now().isoformat()}: {message}\n")

    log_and_print("🔬 EXPERIMENTAL SETUP")
    log_and_print("-" * 30)

    try:
        # Import numerical pipeline
        log_and_print("📦 Loading numerical validation pipeline...")
        from core.numerical_pipeline import NumericalPipelineOrchestrator

        orchestrator = NumericalPipelineOrchestrator()
        log_and_print("✅ Numerical pipeline loaded successfully")

        # Define experimental datasets
        datasets = [
            "PL_Neg_Waters_qTOF.mzML",  # Phospholipids, negative mode
            "TG_Pos_Thermo_Orbi.mzML"  # Triglycerides, positive mode
        ]

        log_and_print(f"📊 Experimental datasets: {len(datasets)}")
        for i, dataset in enumerate(datasets, 1):
            log_and_print(f"  {i}. {dataset}")

        # Initialize experiment results
        experiment_results = {
            'experiment_metadata': {
                'experiment_name': 'Numerical Validation Experiment',
                'start_time': datetime.now().isoformat(),
                'datasets': datasets,
                'pipeline_version': 'standalone_1.0',
                'objective': 'Validate numerical MS processing capabilities'
            },
            'dataset_results': {},
            'performance_summary': {},
            'quality_metrics': {},
            'conclusions': {}
        }

        log_and_print("\n🚀 STARTING EXPERIMENTAL VALIDATION")
        log_and_print("=" * 50)

        total_start_time = time.time()
        all_processing_times = []
        all_annotation_rates = []
        all_quality_scores = []

        # Process each dataset
        for experiment_num, dataset_name in enumerate(datasets, 1):
            log_and_print(f"\n📊 EXPERIMENT {experiment_num}/{len(datasets)}: {dataset_name}")
            log_and_print("-" * 50)

            dataset_start_time = time.time()

            try:
                # Step 1: Data Loading and Initial Assessment
                log_and_print("STEP 1: Loading and initial assessment")
                log_and_print(f"  🔄 Processing dataset: {dataset_name}")

                # Run numerical pipeline with detailed logging
                results = orchestrator.process_dataset(dataset_name, use_stellas=True)

                dataset_processing_time = time.time() - dataset_start_time

                # Step 2: Extract Key Metrics
                log_and_print("STEP 2: Extracting validation metrics")

                pipeline_info = results.get('pipeline_info', {})
                spectra_info = results.get('spectra_processed', {})
                db_annotations = results.get('database_annotations', {})
                qc_stats = results.get('quality_control', {})
                embeddings = results.get('spectrum_embeddings', {})
                clustering = results.get('feature_clustering', {})

                # Calculate key performance indicators
                total_spectra = spectra_info.get('total_input', 0)
                high_quality_spectra = spectra_info.get('high_quality', 0)
                annotated_spectra = db_annotations.get('total_annotated_spectra', 0)
                processing_time = pipeline_info.get('processing_time', 0)

                quality_ratio = high_quality_spectra / max(1, total_spectra)
                annotation_rate = annotated_spectra / max(1, spectra_info.get('ms1_count', 1))

                # Quality control metrics
                qc_metrics = qc_stats.get('quality_metrics', {})
                mean_quality_score = qc_metrics.get('mean_quality_score', 0)

                log_and_print(f"  📈 EXPERIMENTAL RESULTS:")
                log_and_print(f"     Total spectra processed: {total_spectra}")
                log_and_print(f"     High-quality spectra: {high_quality_spectra} ({quality_ratio:.1%})")
                log_and_print(f"     MS1 spectra annotated: {annotated_spectra} ({annotation_rate:.1%})")
                log_and_print(f"     Processing time: {processing_time:.2f}s")
                log_and_print(f"     Mean quality score: {mean_quality_score:.3f}")

                # Step 3: Database Performance Analysis
                log_and_print("STEP 3: Database annotation performance")

                annotations_per_db = db_annotations.get('annotations_per_database', {})
                active_databases = {db: count for db, count in annotations_per_db.items() if count > 0}

                log_and_print(f"  🗃️  Active databases: {len(active_databases)}/{len(annotations_per_db)}")
                for db_name, count in active_databases.items():
                    percentage = (count / max(1, annotated_spectra)) * 100
                    log_and_print(f"     {db_name}: {count} annotations ({percentage:.1f}%)")

                # Step 4: Embedding Quality Assessment
                log_and_print("STEP 4: Spectrum embedding analysis")

                embedding_methods = embeddings.get('methods_used', [])
                embeddings_per_method = embeddings.get('embeddings_per_method', {})

                log_and_print(f"  🧠 Embedding methods: {', '.join(embedding_methods)}")
                for method, count in embeddings_per_method.items():
                    dimension = embeddings.get('embedding_dimensions', {}).get(method, 0)
                    log_and_print(f"     {method}: {count} embeddings (dim={dimension})")

                # Step 5: Clustering Performance
                log_and_print("STEP 5: Feature clustering validation")

                if 'n_clusters' in clustering:
                    clusters = clustering.get('n_clusters', 0)
                    clustered_spectra = clustering.get('n_spectra_clustered', 0)
                    log_and_print(f"  🔬 Clustering: {clusters} clusters for {clustered_spectra} spectra")
                else:
                    log_and_print(f"  ⚠️  Clustering not performed or failed")

                # Step 6: Performance Classification
                log_and_print("STEP 6: Performance assessment")

                # Classify performance
                if annotation_rate >= 0.8 and quality_ratio >= 0.7 and mean_quality_score >= 0.6:
                    performance_grade = "🟢 EXCELLENT"
                elif annotation_rate >= 0.6 and quality_ratio >= 0.5 and mean_quality_score >= 0.4:
                    performance_grade = "🟡 GOOD"
                elif annotation_rate >= 0.4 and quality_ratio >= 0.3 and mean_quality_score >= 0.2:
                    performance_grade = "🟠 ACCEPTABLE"
                else:
                    performance_grade = "🔴 NEEDS IMPROVEMENT"

                log_and_print(f"  🎯 Performance Grade: {performance_grade}")

                # Store detailed results
                experiment_results['dataset_results'][dataset_name] = {
                    'processing_metrics': {
                        'total_spectra': total_spectra,
                        'high_quality_spectra': high_quality_spectra,
                        'quality_ratio': quality_ratio,
                        'processing_time': processing_time,
                        'spectra_per_second': total_spectra / max(0.001, processing_time)
                    },
                    'annotation_metrics': {
                        'total_annotated': annotated_spectra,
                        'annotation_rate': annotation_rate,
                        'active_databases': len(active_databases),
                        'database_breakdown': active_databases
                    },
                    'quality_metrics': {
                        'mean_quality_score': mean_quality_score,
                        'high_quality_ratio': quality_ratio,
                        'quality_control_passed': quality_ratio >= 0.5
                    },
                    'embedding_metrics': {
                        'methods_used': embedding_methods,
                        'total_embeddings': sum(embeddings_per_method.values()),
                        'embedding_success': len(embedding_methods) > 0
                    },
                    'clustering_metrics': {
                        'clusters_created': clustering.get('n_clusters', 0),
                        'spectra_clustered': clustering.get('n_spectra_clustered', 0),
                        'clustering_success': 'n_clusters' in clustering
                    },
                    'performance_assessment': {
                        'grade': performance_grade,
                        'annotation_rate': annotation_rate,
                        'quality_score': mean_quality_score,
                        'processing_efficiency': total_spectra / max(0.001, processing_time)
                    },
                    'raw_results': results
                }

                # Track overall metrics
                all_processing_times.append(processing_time)
                all_annotation_rates.append(annotation_rate)
                all_quality_scores.append(mean_quality_score)

                log_and_print(f"✅ Experiment {experiment_num} completed successfully")

            except Exception as e:
                log_and_print(f"❌ Experiment {experiment_num} failed: {e}")
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
        log_and_print("📊 EXPERIMENTAL CONCLUSIONS")
        log_and_print("=" * 60)

        successful_experiments = len([r for r in experiment_results['dataset_results'].values()
                                    if 'status' not in r or r.get('status') != 'failed'])

        if successful_experiments > 0:
            # Calculate overall performance metrics
            avg_processing_time = np.mean(all_processing_times) if all_processing_times else 0
            avg_annotation_rate = np.mean(all_annotation_rates) if all_annotation_rates else 0
            avg_quality_score = np.mean(all_quality_scores) if all_quality_scores else 0

            experiment_results['performance_summary'] = {
                'successful_experiments': successful_experiments,
                'total_experiments': len(datasets),
                'success_rate': successful_experiments / len(datasets),
                'average_processing_time': avg_processing_time,
                'average_annotation_rate': avg_annotation_rate,
                'average_quality_score': avg_quality_score,
                'total_experiment_time': total_experiment_time
            }

            log_and_print(f"🔢 OVERALL EXPERIMENTAL STATISTICS:")
            log_and_print(f"   Successful experiments: {successful_experiments}/{len(datasets)}")
            log_and_print(f"   Success rate: {(successful_experiments/len(datasets)):.1%}")
            log_and_print(f"   Average processing time: {avg_processing_time:.2f}s per dataset")
            log_and_print(f"   Average annotation rate: {avg_annotation_rate:.1%}")
            log_and_print(f"   Average quality score: {avg_quality_score:.3f}")
            log_and_print(f"   Total experiment duration: {total_experiment_time:.2f}s")

            # Scientific conclusions
            if avg_annotation_rate >= 0.7 and avg_quality_score >= 0.6:
                overall_conclusion = "🟢 NUMERICAL PIPELINE VALIDATED - Ready for production use"
            elif avg_annotation_rate >= 0.5 and avg_quality_score >= 0.4:
                overall_conclusion = "🟡 NUMERICAL PIPELINE FUNCTIONAL - Minor optimizations recommended"
            else:
                overall_conclusion = "🔴 NUMERICAL PIPELINE REQUIRES IMPROVEMENT - Significant issues detected"

            log_and_print(f"\n🎯 SCIENTIFIC CONCLUSION:")
            log_and_print(f"   {overall_conclusion}")

            experiment_results['conclusions'] = {
                'overall_assessment': overall_conclusion,
                'numerical_pipeline_status': 'validated' if avg_annotation_rate >= 0.7 else 'needs_improvement',
                'key_findings': [
                    f"Annotation rate: {avg_annotation_rate:.1%}",
                    f"Quality score: {avg_quality_score:.3f}",
                    f"Processing efficiency: {np.mean([r.get('processing_metrics', {}).get('spectra_per_second', 0) for r in experiment_results['dataset_results'].values() if 'processing_metrics' in r]):.1f} spectra/s"
                ],
                'recommendations': [
                    "Database coverage is satisfactory" if avg_annotation_rate >= 0.6 else "Improve database coverage",
                    "Quality control is effective" if avg_quality_score >= 0.5 else "Enhance quality control filters",
                    "Processing speed is acceptable" if avg_processing_time <= 10.0 else "Optimize processing algorithms"
                ]
            }

        else:
            log_and_print("❌ No successful experiments - unable to draw conclusions")
            experiment_results['conclusions'] = {
                'overall_assessment': '🔴 EXPERIMENTAL FAILURE - All tests failed',
                'status': 'failed'
            }

        # Save comprehensive results
        log_and_print(f"\n💾 SAVING EXPERIMENTAL RESULTS")
        log_and_print("-" * 40)

        # JSON results
        results_file = experiment_dir / "numerical_validation_results.json"
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
                        'Total_Spectra': result['processing_metrics']['total_spectra'],
                        'High_Quality_Spectra': result['processing_metrics']['high_quality_spectra'],
                        'Quality_Ratio': result['processing_metrics']['quality_ratio'],
                        'Annotation_Rate': result['annotation_metrics']['annotation_rate'],
                        'Processing_Time_s': result['processing_metrics']['processing_time'],
                        'Performance_Grade': result['performance_assessment']['grade'],
                        'Active_Databases': result['annotation_metrics']['active_databases']
                    })

            if summary_data:
                csv_file = experiment_dir / "numerical_validation_summary.csv"
                pd.DataFrame(summary_data).to_csv(csv_file, index=False)
                log_and_print(f"📊 Saved CSV summary: {csv_file}")

        # Generate visualizations
        try:
            from visualization.panel import create_performance_panel
            import matplotlib.pyplot as plt

            if successful_experiments > 0:
                # Performance visualization
                performance_data = {
                    'annotation_rates': all_annotation_rates,
                    'quality_scores': all_quality_scores,
                    'processing_times': all_processing_times,
                    'dataset_names': [name for name in datasets if name in experiment_results['dataset_results'] and 'processing_metrics' in experiment_results['dataset_results'][name]]
                }

                fig = create_performance_panel(performance_data, "Numerical Validation Experimental Results")
                viz_file = experiment_dir / "numerical_validation_performance.png"
                fig.savefig(viz_file, dpi=300, bbox_inches='tight')
                plt.close(fig)
                log_and_print(f"📊 Saved visualization: {viz_file}")

        except ImportError as e:
            log_and_print(f"⚠️  Visualization generation failed: {e}")

        log_and_print(f"\n🧪 NUMERICAL VALIDATION EXPERIMENT COMPLETE 🧪")
        log_and_print(f"📁 All results saved to: {experiment_dir}")
        log_and_print(f"📋 Experiment log: {log_file}")

        return experiment_results

    except Exception as e:
        log_and_print(f"💥 CRITICAL EXPERIMENTAL FAILURE: {e}")
        import traceback
        log_and_print(f"Error details: {traceback.format_exc()}")
        return None


if __name__ == "__main__":
    print("🚀 Starting Numerical Validation Science Experiment...")

    results = main()

    if results and results.get('conclusions', {}).get('numerical_pipeline_status') == 'validated':
        print("\n✅ EXPERIMENT SUCCESSFUL - Numerical pipeline validated!")
        sys.exit(0)
    else:
        print("\n❌ EXPERIMENT FAILED - Check results for details")
        sys.exit(1)
