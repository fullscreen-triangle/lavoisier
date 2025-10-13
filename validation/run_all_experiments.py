#!/usr/bin/env python3
"""
MASTER SCIENCE EXPERIMENT RUNNER
===============================

This script orchestrates all validation experiments as a comprehensive
scientific study of the mass spectrometry validation framework.

EXPERIMENT SUITE:
1. Numerical Validation Experiment
2. Visual Validation Experiment
3. Benchmark Performance Experiment

OUTPUTS:
- Master experiment report (JSON + HTML)
- Comparative analysis charts
- Executive summary
- Publication-ready results

Run: python run_all_experiments.py
"""

import sys
import os
import time
import json
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import importlib.util

def main():
    """Master experiment orchestrator"""

    print("🧪 MASTER SCIENCE EXPERIMENT SUITE")
    print("=" * 60)
    print(f"Experiment suite started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Objective: Comprehensive validation of MS processing framework")
    print("=" * 60)

    # Create master experiment directory
    master_dir = Path("experiment_results") / "master_validation_suite"
    master_dir.mkdir(parents=True, exist_ok=True)

    # Initialize master experiment log
    master_log_file = master_dir / "master_experiment_log.txt"

    def log_and_print(message):
        """Log message to both console and file"""
        print(message)
        with open(master_log_file, 'a', encoding='utf-8') as f:
            f.write(f"{datetime.now().isoformat()}: {message}\n")

    log_and_print("🔬 MASTER EXPERIMENTAL SETUP")
    log_and_print("-" * 35)

    # Define experiment suite
    experiments = [
        {
            'name': 'Numerical Validation',
            'script': 'experiment_numerical_validation.py',
            'description': 'Validates numerical MS processing pipeline',
            'expected_outputs': ['numerical_validation_results.json', 'numerical_validation_summary.csv']
        },
        {
            'name': 'Visual Validation',
            'script': 'experiment_visual_validation.py',
            'description': 'Validates visual MS processing & Ion-to-Drip conversion',
            'expected_outputs': ['visual_validation_results.json', 'visual_validation_summary.csv']
        },
        {
            'name': 'Benchmark Performance',
            'script': 'core/simple_benchmark.py',
            'description': 'Benchmarks overall system performance',
            'expected_outputs': ['benchmark_results.json', 'benchmark_summary.csv']
        }
    ]

    log_and_print(f"📊 Experiment suite contains {len(experiments)} experiments:")
    for i, exp in enumerate(experiments, 1):
        log_and_print(f"  {i}. {exp['name']}: {exp['description']}")

    # Initialize master results
    master_results = {
        'suite_metadata': {
            'suite_name': 'Master MS Validation Experiment Suite',
            'start_time': datetime.now().isoformat(),
            'experiments_planned': len(experiments),
            'framework_version': 'standalone_validation_1.0'
        },
        'experiment_results': {},
        'suite_summary': {},
        'comparative_analysis': {},
        'conclusions': {},
        'publication_summary': {}
    }

    log_and_print("\n🚀 STARTING MASTER EXPERIMENT SUITE")
    log_and_print("=" * 50)

    suite_start_time = time.time()
    experiment_outcomes = []

    # Execute each experiment
    for exp_num, experiment in enumerate(experiments, 1):
        log_and_print(f"\n🔬 EXPERIMENT {exp_num}/{len(experiments)}: {experiment['name']}")
        log_and_print("-" * 60)

        exp_start_time = time.time()

        try:
            log_and_print(f"📋 Executing: {experiment['script']}")
            log_and_print(f"📝 Description: {experiment['description']}")

            # Run the experiment script
            script_path = Path(__file__).parent / experiment['script']

            if not script_path.exists():
                log_and_print(f"❌ Experiment script not found: {script_path}")
                experiment_outcomes.append({
                    'name': experiment['name'],
                    'status': 'failed',
                    'error': f"Script not found: {experiment['script']}",
                    'execution_time': 0
                })
                continue

            # Execute experiment as module
            log_and_print(f"🔄 Running experiment...")

            # Load and run the experiment module
            spec = importlib.util.spec_from_file_location(
                f"experiment_{exp_num}", script_path
            )
            experiment_module = importlib.util.module_from_spec(spec)

            # Capture experiment output
            experiment_result = None
            try:
                spec.loader.exec_module(experiment_module)
                if hasattr(experiment_module, 'main'):
                    experiment_result = experiment_module.main()
            except SystemExit as e:
                # Handle sys.exit() calls in experiment scripts
                if e.code == 0:
                    log_and_print(f"✅ Experiment completed successfully (exit code 0)")
                    experiment_status = 'success'
                else:
                    log_and_print(f"❌ Experiment failed (exit code {e.code})")
                    experiment_status = 'failed'
            except Exception as e:
                log_and_print(f"❌ Experiment execution failed: {e}")
                experiment_status = 'failed'
                experiment_result = None
            else:
                experiment_status = 'success' if experiment_result else 'unknown'

            exp_execution_time = time.time() - exp_start_time

            # Store experiment outcome
            outcome = {
                'name': experiment['name'],
                'script': experiment['script'],
                'status': experiment_status,
                'execution_time': exp_execution_time,
                'result_data': experiment_result
            }

            # Look for expected output files
            expected_files_found = []
            for output_file in experiment['expected_outputs']:
                # Check in various possible locations
                possible_paths = [
                    Path("experiment_results") / experiment['name'].lower().replace(' ', '_') / output_file,
                    Path("benchmark_validation_results") / output_file,
                    Path("numerical_validation_results") / output_file,
                    Path("visual_validation_results") / output_file
                ]

                for path in possible_paths:
                    if path.exists():
                        expected_files_found.append(str(path))
                        break

            outcome['output_files'] = expected_files_found
            outcome['outputs_found'] = len(expected_files_found)
            outcome['expected_outputs'] = len(experiment['expected_outputs'])

            log_and_print(f"⏱️  Execution time: {exp_execution_time:.2f} seconds")
            log_and_print(f"📄 Output files found: {len(expected_files_found)}/{len(experiment['expected_outputs'])}")

            if experiment_status == 'success':
                log_and_print(f"✅ Experiment {exp_num} completed successfully")
            else:
                log_and_print(f"❌ Experiment {exp_num} failed or had issues")

            experiment_outcomes.append(outcome)

        except Exception as e:
            exp_execution_time = time.time() - exp_start_time
            log_and_print(f"💥 Critical failure in experiment {exp_num}: {e}")
            import traceback
            log_and_print(f"Error details: {traceback.format_exc()}")

            experiment_outcomes.append({
                'name': experiment['name'],
                'status': 'critical_failure',
                'error': str(e),
                'execution_time': exp_execution_time
            })

    # Master Analysis
    total_suite_time = time.time() - suite_start_time

    log_and_print(f"\n" + "=" * 60)
    log_and_print("📊 MASTER EXPERIMENT SUITE ANALYSIS")
    log_and_print("=" * 60)

    # Calculate overall statistics
    successful_experiments = len([exp for exp in experiment_outcomes if exp['status'] == 'success'])
    failed_experiments = len([exp for exp in experiment_outcomes if exp['status'] in ['failed', 'critical_failure']])
    total_experiments = len(experiment_outcomes)
    success_rate = successful_experiments / max(1, total_experiments)

    total_execution_time = sum([exp['execution_time'] for exp in experiment_outcomes])
    avg_execution_time = total_execution_time / max(1, total_experiments)

    log_and_print(f"🔢 MASTER SUITE STATISTICS:")
    log_and_print(f"   Total experiments planned: {len(experiments)}")
    log_and_print(f"   Experiments executed: {total_experiments}")
    log_and_print(f"   Successful experiments: {successful_experiments}")
    log_and_print(f"   Failed experiments: {failed_experiments}")
    log_and_print(f"   Overall success rate: {success_rate:.1%}")
    log_and_print(f"   Total execution time: {total_execution_time:.2f}s")
    log_and_print(f"   Average execution time: {avg_execution_time:.2f}s per experiment")
    log_and_print(f"   Suite duration: {total_suite_time:.2f}s")

    # Detailed experiment analysis
    log_and_print(f"\n📋 DETAILED EXPERIMENT OUTCOMES:")
    for i, outcome in enumerate(experiment_outcomes, 1):
        status_icon = "✅" if outcome['status'] == 'success' else "❌"
        log_and_print(f"  {status_icon} {i}. {outcome['name']}: {outcome['status']}")
        log_and_print(f"     Execution time: {outcome['execution_time']:.2f}s")
        if 'outputs_found' in outcome:
            log_and_print(f"     Output files: {outcome['outputs_found']}/{outcome['expected_outputs']}")
        if outcome['status'] in ['failed', 'critical_failure'] and 'error' in outcome:
            log_and_print(f"     Error: {outcome['error']}")

    # Store master results
    master_results['experiment_results'] = experiment_outcomes
    master_results['suite_summary'] = {
        'total_experiments': total_experiments,
        'successful_experiments': successful_experiments,
        'failed_experiments': failed_experiments,
        'success_rate': success_rate,
        'total_execution_time': total_execution_time,
        'average_execution_time': avg_execution_time,
        'suite_duration': total_suite_time,
        'completion_time': datetime.now().isoformat()
    }

    # Generate overall conclusions
    if success_rate >= 0.8:
        overall_assessment = "🟢 VALIDATION FRAMEWORK COMPREHENSIVELY VALIDATED"
        framework_status = "production_ready"
    elif success_rate >= 0.6:
        overall_assessment = "🟡 VALIDATION FRAMEWORK MOSTLY FUNCTIONAL"
        framework_status = "needs_minor_improvements"
    elif success_rate >= 0.3:
        overall_assessment = "🟠 VALIDATION FRAMEWORK PARTIALLY FUNCTIONAL"
        framework_status = "needs_major_improvements"
    else:
        overall_assessment = "🔴 VALIDATION FRAMEWORK REQUIRES SIGNIFICANT WORK"
        framework_status = "not_ready"

    log_and_print(f"\n🎯 MASTER SCIENTIFIC CONCLUSION:")
    log_and_print(f"   {overall_assessment}")
    log_and_print(f"   Framework Status: {framework_status}")

    master_results['conclusions'] = {
        'overall_assessment': overall_assessment,
        'framework_status': framework_status,
        'success_rate': success_rate,
        'key_findings': [
            f"Experiment success rate: {success_rate:.1%}",
            f"Average processing time: {avg_execution_time:.1f}s per experiment",
            f"Suite completion time: {total_suite_time:.1f}s",
            f"Framework reliability: {'High' if success_rate >= 0.8 else 'Medium' if success_rate >= 0.5 else 'Low'}"
        ],
        'recommendations': [
            "Framework is ready for deployment" if success_rate >= 0.8 else "Address failing experiments before deployment",
            "Performance is acceptable" if avg_execution_time <= 30 else "Optimize processing speed",
            "Experiment suite is comprehensive" if total_experiments >= 3 else "Add more validation experiments"
        ]
    }

    # Generate publication summary
    master_results['publication_summary'] = {
        'title': 'Comprehensive Validation of Mass Spectrometry Processing Framework',
        'abstract': f"We conducted a comprehensive validation study of a standalone mass spectrometry processing framework. The study included {total_experiments} independent experiments testing numerical processing, visual analysis, and performance benchmarking. Overall success rate was {success_rate:.1%} with an average processing time of {avg_execution_time:.1f} seconds per experiment.",
        'methods': f"Three independent validation experiments were conducted: (1) Numerical validation testing database annotation and spectrum embedding, (2) Visual validation testing Ion-to-Drip conversion and LipidMaps annotation, and (3) Performance benchmarking testing system reliability and memory usage.",
        'results': f"The validation framework achieved a {success_rate:.1%} success rate across all experiments. {successful_experiments} out of {total_experiments} experiments completed successfully within a total execution time of {total_suite_time:.1f} seconds.",
        'conclusion': f"The mass spectrometry processing framework demonstrates {'excellent' if success_rate >= 0.8 else 'good' if success_rate >= 0.6 else 'moderate'} performance and is {'recommended for production use' if success_rate >= 0.7 else 'suitable for further development'}."
    }

    # Save master results
    log_and_print(f"\n💾 SAVING MASTER EXPERIMENT SUITE RESULTS")
    log_and_print("-" * 50)

    # JSON results
    master_results_file = master_dir / "master_validation_results.json"
    with open(master_results_file, 'w') as f:
        json.dump(master_results, f, indent=2)
    log_and_print(f"📄 Saved master JSON results: {master_results_file}")

    # CSV summary
    summary_data = []
    for outcome in experiment_outcomes:
        summary_data.append({
            'Experiment_Name': outcome['name'],
            'Status': outcome['status'],
            'Execution_Time_s': outcome['execution_time'],
            'Expected_Outputs': outcome.get('expected_outputs', 0),
            'Outputs_Found': outcome.get('outputs_found', 0),
            'Success': 1 if outcome['status'] == 'success' else 0
        })

    csv_file = master_dir / "master_experiment_summary.csv"
    pd.DataFrame(summary_data).to_csv(csv_file, index=False)
    log_and_print(f"📊 Saved master CSV summary: {csv_file}")

    # Generate HTML report
    try:
        html_report = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Master Validation Experiment Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background: #f0f8ff; padding: 20px; border-radius: 10px; }}
                .success {{ color: #28a745; }}
                .failure {{ color: #dc3545; }}
                .warning {{ color: #ffc107; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🧪 Master Validation Experiment Report</h1>
                <p><strong>Framework:</strong> Standalone Mass Spectrometry Validation</p>
                <p><strong>Execution Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Success Rate:</strong> <span class="{'success' if success_rate >= 0.7 else 'warning' if success_rate >= 0.5 else 'failure'}">{success_rate:.1%}</span></p>
            </div>

            <h2>📊 Experiment Summary</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Total Experiments</td><td>{total_experiments}</td></tr>
                <tr><td>Successful Experiments</td><td class="success">{successful_experiments}</td></tr>
                <tr><td>Failed Experiments</td><td class="failure">{failed_experiments}</td></tr>
                <tr><td>Success Rate</td><td class="{'success' if success_rate >= 0.7 else 'warning' if success_rate >= 0.5 else 'failure'}">{success_rate:.1%}</td></tr>
                <tr><td>Total Execution Time</td><td>{total_suite_time:.1f} seconds</td></tr>
            </table>

            <h2>🔬 Individual Experiment Results</h2>
            <table>
                <tr><th>Experiment</th><th>Status</th><th>Time (s)</th><th>Outputs</th></tr>
        """

        for outcome in experiment_outcomes:
            status_class = 'success' if outcome['status'] == 'success' else 'failure'
            html_report += f"""
                <tr>
                    <td>{outcome['name']}</td>
                    <td class="{status_class}">{outcome['status']}</td>
                    <td>{outcome['execution_time']:.2f}</td>
                    <td>{outcome.get('outputs_found', 0)}/{outcome.get('expected_outputs', 0)}</td>
                </tr>
            """

        html_report += f"""
            </table>

            <h2>🎯 Conclusions</h2>
            <div class="{'success' if framework_status == 'production_ready' else 'warning'}">
                <h3>{overall_assessment}</h3>
                <p><strong>Framework Status:</strong> {framework_status}</p>
                <p><strong>Recommendation:</strong> {master_results['conclusions']['recommendations'][0]}</p>
            </div>

            <h2>📄 Publication Abstract</h2>
            <p>{master_results['publication_summary']['abstract']}</p>

            <h2>🔬 Methods</h2>
            <p>{master_results['publication_summary']['methods']}</p>

            <h2>📈 Results</h2>
            <p>{master_results['publication_summary']['results']}</p>

            <h2>💡 Conclusion</h2>
            <p>{master_results['publication_summary']['conclusion']}</p>

            <hr>
            <p><small>Generated by Master Experiment Suite at {datetime.now().isoformat()}</small></p>
        </body>
        </html>
        """

        html_file = master_dir / "master_experiment_report.html"
        with open(html_file, 'w') as f:
            f.write(html_report)
        log_and_print(f"📄 Generated HTML report: {html_file}")

    except Exception as e:
        log_and_print(f"⚠️  HTML report generation failed: {e}")

    log_and_print(f"\n🧪 MASTER EXPERIMENT SUITE COMPLETE 🧪")
    log_and_print(f"📁 All results saved to: {master_dir}")
    log_and_print(f"📋 Master log: {master_log_file}")
    log_and_print(f"🎯 Framework Status: {framework_status}")

    return master_results, success_rate >= 0.6


if __name__ == "__main__":
    print("🚀 Starting Master Science Experiment Suite...")

    master_results, suite_success = main()

    if suite_success:
        print("\n✅ MASTER EXPERIMENT SUITE SUCCESSFUL!")
        print("🎉 Validation framework demonstrates good performance")
        sys.exit(0)
    else:
        print("\n❌ MASTER EXPERIMENT SUITE HAD ISSUES")
        print("🔍 Check individual experiment results for details")
        sys.exit(1)
