#!/usr/bin/env python3
"""
STEP 3: DATABASE SEARCH VALIDATION EXPERIMENT
===========================================

OBJECTIVE: Validate database search and annotation capabilities
HYPOTHESIS: The database search engine can effectively annotate spectra using multiple databases

EXPERIMENT PROCEDURE:
1. Load mass spectrometry data
2. Extract MS1 spectra for annotation
3. Search against multiple databases
4. Analyze annotation coverage and accuracy
5. Generate database performance visualizations

Run: python step_03_database_search_experiment.py
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
    """Step 3: Database Search Validation Experiment"""

    print("🧪 STEP 3: DATABASE SEARCH VALIDATION EXPERIMENT")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Objective: Validate database search and annotation performance")
    print("=" * 60)

    # Create step-specific results directory
    step_dir = Path("step_results") / "step_03_database_search"
    step_dir.mkdir(parents=True, exist_ok=True)

    # Initialize step log
    log_file = step_dir / "database_search_log.txt"

    def log_and_print(message):
        """Log message to both console and file"""
        print(message)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"{datetime.now().isoformat()}: {message}\n")

    log_and_print("🔬 STEP 3 EXPERIMENTAL SETUP")
    log_and_print("-" * 30)

    try:
        # Import required components
        log_and_print("📦 Loading database search components...")
        from core.mzml_reader import StandaloneMzMLReader
        from core.numerical_pipeline import DatabaseSearchEngine, QualityControlModule

        reader = StandaloneMzMLReader()
        database_engine = DatabaseSearchEngine()
        qc_module = QualityControlModule()
        log_and_print("✅ Database search components loaded successfully")

        # Get available databases
        available_databases = list(database_engine.databases.keys())
        log_and_print(f"🗃️  Available databases: {available_databases}")

        # Define test datasets
        test_files = [
            "PL_Neg_Waters_qTOF.mzML",
            "TG_Pos_Thermo_Orbi.mzML"
        ]

        log_and_print(f"📊 Testing database search on {len(test_files)} datasets:")
        for i, file in enumerate(test_files, 1):
            log_and_print(f"  {i}. {file}")

        # Initialize results
        step_results = {
            'step_metadata': {
                'step_number': 3,
                'step_name': 'Database Search Validation',
                'start_time': datetime.now().isoformat(),
                'objective': 'Validate database search and annotation performance',
                'available_databases': available_databases
            },
            'database_search_results': {},
            'annotation_coverage_analysis': {},
            'database_performance_metrics': {},
            'step_conclusion': {}
        }

        log_and_print("\n🚀 STARTING DATABASE SEARCH VALIDATION")
        log_and_print("-" * 45)

        all_search_times = []
        all_annotation_rates = []
        database_hit_counts = {db: 0 for db in available_databases}
        total_queries = 0

        # Test each dataset
        for dataset_num, test_file in enumerate(test_files, 1):
            log_and_print(f"\n🗃️  DATASET {dataset_num}: {test_file}")
            log_and_print("-" * 40)

            dataset_start_time = time.time()

            try:
                # STEP 3.1: Load and prepare data for database search
                log_and_print("STEP 3.1: Loading data for database search...")
                spectra = reader.load_mzml(test_file)

                # Filter for high-quality MS1 spectra
                ms1_spectra = [s for s in spectra if s.ms_level == 1]

                # Apply quality filtering
                high_quality_ms1 = []
                for spectrum in ms1_spectra:
                    quality_metrics = qc_module.assess_spectrum_quality(spectrum)
                    if quality_metrics.get('quality_score', 0) >= 0.3:  # Minimum quality threshold
                        high_quality_ms1.append(spectrum)

                # Limit for performance testing
                search_spectra = high_quality_ms1[:20]  # Test first 20 high-quality spectra

                log_and_print(f"  📂 Loaded {len(spectra)} total spectra")
                log_and_print(f"  📊 Found {len(ms1_spectra)} MS1 spectra")
                log_and_print(f"  🎯 Selected {len(search_spectra)} high-quality MS1 for database search")

                # STEP 3.2: Individual database performance testing
                log_and_print("STEP 3.2: Testing individual database performance...")

                database_performance = {}

                for db_name in available_databases:
                    db_start_time = time.time()
                    db_annotations = []
                    db_hit_count = 0

                    log_and_print(f"    🔍 Testing {db_name} database...")

                    for spectrum in search_spectra:
                        base_peak_mz, _ = spectrum.base_peak

                        # Search this specific database
                        db_results = database_engine.search_by_mass(
                            base_peak_mz, db_name,
                            tolerance=database_engine.databases[db_name]['search_params'].get('mz_tolerance', 0.01)
                        )

                        if db_results:
                            db_hit_count += 1
                            db_annotations.extend(db_results[:3])  # Top 3 results per spectrum

                    db_search_time = time.time() - db_start_time
                    db_hit_rate = db_hit_count / max(1, len(search_spectra))

                    database_performance[db_name] = {
                        'search_time': db_search_time,
                        'hit_count': db_hit_count,
                        'hit_rate': db_hit_rate,
                        'total_annotations': len(db_annotations),
                        'avg_annotations_per_hit': len(db_annotations) / max(1, db_hit_count),
                        'search_rate': len(search_spectra) / max(0.001, db_search_time)
                    }

                    log_and_print(f"      📊 {db_name}: {db_hit_count}/{len(search_spectra)} hits ({db_hit_rate:.1%})")
                    log_and_print(f"      ⏱️  Search time: {db_search_time:.2f}s ({len(search_spectra) / max(0.001, db_search_time):.1f} spectra/s)")

                # STEP 3.3: Combined database search analysis
                log_and_print("STEP 3.3: Combined database search analysis...")

                combined_search_start = time.time()
                combined_annotations = {}

                for spectrum in search_spectra:
                    base_peak_mz, _ = spectrum.base_peak

                    # Search all databases
                    all_db_results = database_engine.search_all_databases(base_peak_mz)

                    if all_db_results:
                        combined_annotations[spectrum.scan_id] = {
                            'query_mz': base_peak_mz,
                            'database_results': all_db_results,
                            'total_databases_hit': len(all_db_results),
                            'total_annotations': sum(len(results) for results in all_db_results.values())
                        }

                combined_search_time = time.time() - combined_search_start

                # STEP 3.4: Annotation coverage analysis
                log_and_print("STEP 3.4: Analyzing annotation coverage...")

                annotated_spectra = len(combined_annotations)
                annotation_rate = annotated_spectra / max(1, len(search_spectra))

                # Database coverage analysis
                databases_per_spectrum = []
                annotations_per_spectrum = []

                for annotation_data in combined_annotations.values():
                    databases_per_spectrum.append(annotation_data['total_databases_hit'])
                    annotations_per_spectrum.append(annotation_data['total_annotations'])

                avg_databases_per_hit = np.mean(databases_per_spectrum) if databases_per_spectrum else 0
                avg_annotations_per_spectrum = np.mean(annotations_per_spectrum) if annotations_per_spectrum else 0

                # Database contribution analysis
                database_contributions = {}
                for db_name in available_databases:
                    db_contribution = sum(1 for ann in combined_annotations.values()
                                        if db_name in ann['database_results'])
                    database_contributions[db_name] = {
                        'contribution_count': db_contribution,
                        'contribution_rate': db_contribution / max(1, len(search_spectra))
                    }

                log_and_print(f"  📈 ANNOTATION COVERAGE ANALYSIS:")
                log_and_print(f"     Spectra annotated: {annotated_spectra}/{len(search_spectra)} ({annotation_rate:.1%})")
                log_and_print(f"     Avg databases per hit: {avg_databases_per_hit:.1f}")
                log_and_print(f"     Avg annotations per spectrum: {avg_annotations_per_spectrum:.1f}")
                log_and_print(f"     Combined search time: {combined_search_time:.2f}s")

                log_and_print(f"  🗃️  DATABASE CONTRIBUTIONS:")
                for db_name, contrib in database_contributions.items():
                    log_and_print(f"     {db_name}: {contrib['contribution_count']} hits ({contrib['contribution_rate']:.1%})")

                # STEP 3.5: Search efficiency assessment
                log_and_print("STEP 3.5: Database search efficiency assessment...")

                # Calculate efficiency metrics
                total_individual_search_time = sum(perf['search_time'] for perf in database_performance.values())
                search_efficiency = combined_search_time / max(0.001, total_individual_search_time)

                # Coverage efficiency (how well databases complement each other)
                max_individual_hit_rate = max(perf['hit_rate'] for perf in database_performance.values())
                coverage_improvement = annotation_rate / max(0.001, max_individual_hit_rate)

                # Annotation quality (multiple databases providing results)
                multi_db_hits = len([ann for ann in combined_annotations.values()
                                   if ann['total_databases_hit'] > 1])
                multi_db_rate = multi_db_hits / max(1, annotated_spectra)

                log_and_print(f"  ⚡ SEARCH EFFICIENCY METRICS:")
                log_and_print(f"     Search efficiency ratio: {search_efficiency:.2f}")
                log_and_print(f"     Coverage improvement: {coverage_improvement:.2f}x")
                log_and_print(f"     Multi-database hits: {multi_db_hits}/{annotated_spectra} ({multi_db_rate:.1%})")

                # STEP 3.6: Performance classification
                log_and_print("STEP 3.6: Database search performance classification...")

                if (annotation_rate >= 0.7 and avg_databases_per_hit >= 2.0 and
                    combined_search_time <= 10.0):
                    performance_grade = "🟢 EXCELLENT"
                elif annotation_rate >= 0.5 and avg_databases_per_hit >= 1.5:
                    performance_grade = "🟡 GOOD"
                elif annotation_rate >= 0.3 and avg_databases_per_hit >= 1.0:
                    performance_grade = "🟠 ACCEPTABLE"
                else:
                    performance_grade = "🔴 NEEDS IMPROVEMENT"

                log_and_print(f"  🏆 Database Search Performance: {performance_grade}")

                # Store results
                step_results['database_search_results'][test_file] = {
                    'spectra_searched': len(search_spectra),
                    'annotated_spectra': annotated_spectra,
                    'annotation_rate': annotation_rate,
                    'combined_search_time': combined_search_time,
                    'search_rate': len(search_spectra) / max(0.001, combined_search_time),
                    'database_performance': database_performance,
                    'coverage_metrics': {
                        'avg_databases_per_hit': avg_databases_per_hit,
                        'avg_annotations_per_spectrum': avg_annotations_per_spectrum,
                        'multi_db_rate': multi_db_rate,
                        'database_contributions': database_contributions
                    },
                    'efficiency_metrics': {
                        'search_efficiency': search_efficiency,
                        'coverage_improvement': coverage_improvement,
                        'total_individual_search_time': total_individual_search_time
                    },
                    'performance_grade': performance_grade,
                    'sample_annotations': dict(list(combined_annotations.items())[:3])  # First 3 for reference
                }

                # Track overall metrics
                all_search_times.append(combined_search_time)
                all_annotation_rates.append(annotation_rate)
                total_queries += len(search_spectra)

                for db_name, contrib in database_contributions.items():
                    database_hit_counts[db_name] += contrib['contribution_count']

                log_and_print(f"✅ Dataset {dataset_num} database search validation completed")

            except Exception as e:
                log_and_print(f"❌ Dataset {dataset_num} database search failed: {e}")

                step_results['database_search_results'][test_file] = {
                    'search_success': False,
                    'error': str(e),
                    'processing_time': time.time() - dataset_start_time
                }

        # STEP 3.7: Overall analysis and conclusions
        log_and_print(f"\n" + "=" * 60)
        log_and_print("📊 STEP 3 OVERALL DATABASE SEARCH ANALYSIS")
        log_and_print("=" * 60)

        successful_searches = len([r for r in step_results['database_search_results'].values()
                                 if 'search_success' not in r or r.get('search_success', True)])
        total_datasets = len(test_files)
        success_rate = successful_searches / max(1, total_datasets)

        if successful_searches > 0:
            avg_search_time = np.mean(all_search_times)
            avg_annotation_rate = np.mean(all_annotation_rates)
            overall_search_rate = total_queries / max(0.001, sum(all_search_times))

            # Database effectiveness analysis
            database_effectiveness = {}
            for db_name in available_databases:
                total_hits = database_hit_counts[db_name]
                effectiveness = total_hits / max(1, total_queries)
                database_effectiveness[db_name] = {
                    'total_hits': total_hits,
                    'effectiveness': effectiveness,
                    'hit_percentage': effectiveness * 100
                }

            step_results['annotation_coverage_analysis'] = {
                'total_queries': total_queries,
                'avg_annotation_rate': avg_annotation_rate,
                'database_effectiveness': database_effectiveness,
                'most_effective_database': max(database_effectiveness.keys(),
                                              key=lambda x: database_effectiveness[x]['effectiveness']),
                'least_effective_database': min(database_effectiveness.keys(),
                                               key=lambda x: database_effectiveness[x]['effectiveness'])
            }

            step_results['database_performance_metrics'] = {
                'successful_searches': successful_searches,
                'total_datasets': total_datasets,
                'success_rate': success_rate,
                'avg_search_time': avg_search_time,
                'avg_annotation_rate': avg_annotation_rate,
                'overall_search_rate': overall_search_rate,
                'total_queries_processed': total_queries
            }

            log_and_print(f"🔢 STEP 3 PERFORMANCE METRICS:")
            log_and_print(f"   Successful database searches: {successful_searches}/{total_datasets}")
            log_and_print(f"   Success rate: {success_rate:.1%}")
            log_and_print(f"   Total spectra queried: {total_queries}")
            log_and_print(f"   Average annotation rate: {avg_annotation_rate:.1%}")
            log_and_print(f"   Average search time: {avg_search_time:.2f}s per dataset")
            log_and_print(f"   Overall search rate: {overall_search_rate:.1f} spectra/s")

            # Database effectiveness ranking
            log_and_print(f"\n🏆 DATABASE EFFECTIVENESS RANKING:")
            sorted_databases = sorted(database_effectiveness.items(),
                                    key=lambda x: x[1]['effectiveness'], reverse=True)
            for i, (db_name, metrics) in enumerate(sorted_databases, 1):
                log_and_print(f"   {i}. {db_name}: {metrics['effectiveness']:.1%} ({metrics['total_hits']} hits)")

            # Step conclusion
            if (success_rate >= 0.8 and avg_annotation_rate >= 0.6 and
                avg_search_time <= 15.0):
                step_conclusion = "🟢 DATABASE SEARCH VALIDATION PASSED - Effective multi-database annotation"
                step_status = "validated"
            elif success_rate >= 0.6 and avg_annotation_rate >= 0.4:
                step_conclusion = "🟡 DATABASE SEARCH VALIDATION PARTIAL - Good performance with optimization potential"
                step_status = "functional"
            else:
                step_conclusion = "🔴 DATABASE SEARCH VALIDATION FAILED - Poor annotation coverage or performance"
                step_status = "problematic"

            log_and_print(f"\n🎯 STEP 3 CONCLUSION:")
            log_and_print(f"   {step_conclusion}")

            best_db = max(database_effectiveness.keys(), key=lambda x: database_effectiveness[x]['effectiveness'])
            worst_db = min(database_effectiveness.keys(), key=lambda x: database_effectiveness[x]['effectiveness'])

            step_results['step_conclusion'] = {
                'overall_assessment': step_conclusion,
                'step_status': step_status,
                'success_rate': success_rate,
                'avg_annotation_rate': avg_annotation_rate,
                'best_performing_database': best_db,
                'worst_performing_database': worst_db,
                'key_findings': [
                    f"Database search success rate: {success_rate:.1%}",
                    f"Average annotation rate: {avg_annotation_rate:.1%}",
                    f"Search performance: {overall_search_rate:.1f} spectra/s",
                    f"Most effective database: {best_db} ({database_effectiveness[best_db]['effectiveness']:.1%})"
                ],
                'recommendations': [
                    "Database coverage is comprehensive" if avg_annotation_rate >= 0.6 else "Expand database coverage or improve search algorithms",
                    f"Prioritize {best_db} database" if database_effectiveness[best_db]['effectiveness'] >= 0.5 else "All databases have low hit rates - review data quality",
                    "Search speed is acceptable" if avg_search_time <= 10.0 else "Optimize database search algorithms",
                    f"Consider removing {worst_db}" if database_effectiveness[worst_db]['effectiveness'] < 0.1 else "All databases contribute meaningfully"
                ]
            }

        else:
            log_and_print("❌ No successful database searches - critical failure in Step 3")
            step_results['step_conclusion'] = {
                'overall_assessment': '🔴 CRITICAL FAILURE - Database search completely non-functional',
                'step_status': 'failed'
            }

        # STEP 3.8: Save results and generate visualizations
        log_and_print(f"\n💾 SAVING STEP 3 RESULTS")
        log_and_print("-" * 30)

        # JSON results
        results_file = step_dir / "step_03_database_search_results.json"
        with open(results_file, 'w') as f:
            json.dump(step_results, f, indent=2)
        log_and_print(f"📄 Saved JSON results: {results_file}")

        # CSV summary
        if successful_searches > 0:
            csv_data = []
            for dataset, result in step_results['database_search_results'].items():
                if 'spectra_searched' in result:
                    csv_data.append({
                        'Dataset': dataset,
                        'Spectra_Searched': result['spectra_searched'],
                        'Annotated_Spectra': result['annotated_spectra'],
                        'Annotation_Rate': result['annotation_rate'],
                        'Search_Time_s': result['combined_search_time'],
                        'Search_Rate': result['search_rate'],
                        'Avg_Databases_Per_Hit': result['coverage_metrics']['avg_databases_per_hit'],
                        'Multi_DB_Rate': result['coverage_metrics']['multi_db_rate'],
                        'Performance_Grade': result['performance_grade']
                    })

            if csv_data:
                csv_file = step_dir / "step_03_database_search_summary.csv"
                pd.DataFrame(csv_data).to_csv(csv_file, index=False)
                log_and_print(f"📊 Saved CSV summary: {csv_file}")

        # Generate visualizations
        try:
            from visualization.panel import create_database_search_panel
            import matplotlib.pyplot as plt

            if successful_searches > 0:
                viz_data = {
                    'annotation_rates': all_annotation_rates,
                    'search_times': all_search_times,
                    'database_effectiveness': database_effectiveness,
                    'dataset_names': [f.replace('.mzML', '') for f in test_files]
                }

                fig = create_database_search_panel(viz_data, "Step 3: Database Search Performance")
                viz_file = step_dir / "step_03_database_search_performance.png"
                fig.savefig(viz_file, dpi=300, bbox_inches='tight')
                plt.close(fig)
                log_and_print(f"📊 Saved visualization: {viz_file}")

        except ImportError as e:
            log_and_print(f"⚠️  Visualization generation failed: {e}")

        log_and_print(f"\n🧪 STEP 3: DATABASE SEARCH VALIDATION COMPLETE 🧪")
        log_and_print(f"📁 Results saved to: {step_dir}")
        log_and_print(f"📋 Step log: {log_file}")

        return step_results

    except Exception as e:
        log_and_print(f"💥 CRITICAL STEP 3 FAILURE: {e}")
        import traceback
        log_and_print(f"Error details: {traceback.format_exc()}")
        return None


if __name__ == "__main__":
    print("🚀 Starting Step 3: Database Search Validation Experiment...")

    results = main()

    if results and results.get('step_conclusion', {}).get('step_status') in ['validated', 'functional']:
        print("\n✅ STEP 3 SUCCESSFUL - Database search validated!")
        sys.exit(0)
    else:
        print("\n❌ STEP 3 FAILED - Check results for details")
        sys.exit(1)
