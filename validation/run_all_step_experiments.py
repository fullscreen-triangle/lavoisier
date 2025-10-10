#!/usr/bin/env python3
"""
MASTER STEP-BY-STEP VALIDATION EXPERIMENT RUNNER
===============================================

This script orchestrates ALL individual step validation experiments in sequence.
Each step is run as an independent scientific experiment with complete isolation.

AVAILABLE STEPS:
Step 1: Data Loading Validation
Step 2: Quality Control Validation
Step 3: Database Search Validation
Step 4: Spectrum Embedding Validation
Step 5: Feature Clustering Validation
Step 6: Ion Extraction Validation

EXPERIMENT PHILOSOPHY:
- Each step runs in complete isolation
- Each step saves its own results and visualizations
- Each step documents every action taken
- Each step provides pass/fail validation
- Steps can be run individually or in sequence

Run: python run_all_step_experiments.py
"""

import sys
import os
import time
import json
import subprocess
import importlib.util
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

def main():
    """Master step-by-step validation experiment orchestrator"""

    print("🧪 MASTER STEP-BY-STEP VALIDATION EXPERIMENT SUITE")
    print("=" * 70)
    print(f"Experiment suite started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Objective: Comprehensive step-by-step validation of MS processing")
    print("=" * 70)

    # Create master results directory
    master_dir = Path("step_results") / "master_step_validation_suite"
    master_dir.mkdir(parents=True, exist_ok=True)

    # Initialize master log
    master_log_file = master_dir / "master_step_log.txt"

    def log_and_print(message):
        """Log message to both console and file"""
        print(message)
        with open(master_log_file, 'a') as f:
            f.write(f"{datetime.now().isoformat()}: {message}\n")

    log_and_print("🔬 MASTER STEP EXPERIMENTAL SETUP")
    log_and_print("-" * 40)

    # Define all validation steps
    validation_steps = [
        {
            'step_number': 1,
            'step_name': 'Data Loading Validation',
            'script_name': 'step_01_data_loading_experiment.py',
            'description': 'Validates mzML data loading and parsing capabilities',
            'category': 'Core Infrastructure',
            'critical': True,
            'expected_outputs': ['step_01_data_loading_results.json', 'step_01_data_loading_summary.csv']
        },
        {
            'step_number': 2,
            'step_name': 'Quality Control Validation',
            'script_name': 'step_02_quality_control_experiment.py',
            'description': 'Validates spectrum quality assessment and filtering',
            'category': 'Data Processing',
            'critical': True,
            'expected_outputs': ['step_02_quality_control_results.json', 'step_02_quality_control_summary.csv']
        },
        {
            'step_number': 3,
            'step_name': 'Database Search Validation',
            'script_name': 'step_03_database_search_experiment.py',
            'description': 'Validates database search and annotation performance',
            'category': 'Annotation',
            'critical': True,
            'expected_outputs': ['step_03_database_search_results.json', 'step_03_database_search_summary.csv']
        },
        {
            'step_number': 4,
            'step_name': 'Spectrum Embedding Validation',
            'script_name': 'step_04_spectrum_embedding_experiment.py',
            'description': 'Validates spectrum embedding and similarity analysis',
            'category': 'Machine Learning',
            'critical': False,
            'expected_outputs': ['step_04_spectrum_embedding_results.json', 'step_04_spectrum_embedding_summary.csv']
        },
        {
            'step_number': 5,
            'step_name': 'Feature Clustering Validation',
            'script_name': 'step_05_feature_clustering_experiment.py',
            'description': 'Validates feature extraction and spectrum clustering',
            'category': 'Analysis',
            'critical': False,
            'expected_outputs': ['step_05_feature_clustering_results.json', 'step_05_feature_clustering_summary.csv']
        },
        {
            'step_number': 6,
            'step_name': 'Ion Extraction Validation',
            'script_name': 'step_06_ion_extraction_experiment.py',
            'description': 'Validates ion extraction for visual processing',
            'category': 'Visual Processing',
            'critical': True,
            'expected_outputs': ['step_06_ion_extraction_results.json', 'step_06_ion_extraction_summary.csv']
        }
    ]

    log_and_print(f"📊 Master experiment suite contains {len(validation_steps)} validation steps:")
    for step in validation_steps:
        critical_marker = "⚠️  CRITICAL" if step['critical'] else "📋 Optional"
        log_and_print(f"  Step {step['step_number']}: {step['step_name']} ({step['category']}) - {critical_marker}")
        log_and_print(f"    📝 {step['description']}")

    # Initialize master results
    master_results = {
        'suite_metadata': {
            'suite_name': 'Master Step-by-Step Validation Experiment Suite',
            'start_time': datetime.now().isoformat(),
            'total_steps_planned': len(validation_steps),
            'critical_steps': len([s for s in validation_steps if s['critical']]),
            'optional_steps': len([s for s in validation_steps if not s['critical']]),
            'framework_version': 'standalone_step_validation_1.0'
        },
        'step_results': {},
        'suite_summary': {},
        'critical_step_analysis': {},
        'step_dependencies': {},
        'conclusions': {}
    }

    log_and_print("\n🚀 STARTING MASTER STEP-BY-STEP VALIDATION SUITE")
    log_and_print("=" * 60)

    suite_start_time = time.time()
    step_outcomes = []
    critical_step_failures = []

    # Execute each step in sequence
    for step in validation_steps:
        step_num = step['step_number']
        step_name = step['step_name']
        script_name = step['script_name']

        log_and_print(f"\n🔬 STEP {step_num}: {step_name}")
        log_and_print("-" * 70)

        step_start_time = time.time()

        try:
            log_and_print(f"📋 Executing: {script_name}")
            log_and_print(f"📝 Description: {step['description']}")
            log_and_print(f"🏷️  Category: {step['category']}")
            log_and_print(f"⚠️  Critical: {'Yes' if step['critical'] else 'No'}")

            # Check if script exists
            script_path = Path(__file__).parent / script_name

            if not script_path.exists():
                log_and_print(f"❌ Step script not found: {script_path}")
                step_outcome = {
                    'step_number': step_num,
                    'step_name': step_name,
                    'status': 'script_not_found',
                    'error': f"Script not found: {script_name}",
                    'execution_time': 0,
                    'critical': step['critical']
                }
                step_outcomes.append(step_outcome)

                if step['critical']:
                    critical_step_failures.append(step_num)

                continue

            # Execute step as module
            log_and_print(f"🔄 Running step experiment...")

            # Load and run the step module
            spec = importlib.util.spec_from_file_location(
                f"step_{step_num}", script_path
            )
            step_module = importlib.util.module_from_spec(spec)

            # Capture step output
            step_result = None
            step_status = 'unknown'

            try:
                spec.loader.exec_module(step_module)
                if hasattr(step_module, 'main'):
                    step_result = step_module.main()

                    # Determine step status from result
                    if step_result:
                        step_conclusion = step_result.get('step_conclusion', {})
                        step_status = step_conclusion.get('step_status', 'unknown')

                        if step_status in ['validated', 'functional']:
                            log_and_print(f"✅ Step {step_num} completed successfully - Status: {step_status}")
                        else:
                            log_and_print(f"⚠️  Step {step_num} completed with issues - Status: {step_status}")
                    else:
                        step_status = 'no_result'
                        log_and_print(f"⚠️  Step {step_num} returned no results")

            except SystemExit as e:
                # Handle sys.exit() calls in step scripts
                if e.code == 0:
                    log_and_print(f"✅ Step {step_num} completed successfully (exit code 0)")
                    step_status = 'validated'
                else:
                    log_and_print(f"❌ Step {step_num} failed (exit code {e.code})")
                    step_status = 'failed'
            except Exception as e:
                log_and_print(f"❌ Step {step_num} execution failed: {e}")
                step_status = 'error'
                step_result = {'error': str(e)}

            step_execution_time = time.time() - step_start_time

            # Store step outcome
            step_outcome = {
                'step_number': step_num,
                'step_name': step_name,
                'script_name': script_name,
                'status': step_status,
                'execution_time': step_execution_time,
                'critical': step['critical'],
                'category': step['category'],
                'result_data': step_result
            }

            # Check for expected output files
            expected_files_found = []
            step_result_dir = Path("step_results") / f"step_{step_num:02d}_{step_name.lower().replace(' ', '_')}"

            for output_file in step['expected_outputs']:
                output_path = step_result_dir / output_file
                if output_path.exists():
                    expected_files_found.append(str(output_path))

            step_outcome['output_files'] = expected_files_found
            step_outcome['outputs_found'] = len(expected_files_found)
            step_outcome['expected_outputs'] = len(step['expected_outputs'])

            log_and_print(f"⏱️  Step {step_num} execution time: {step_execution_time:.2f} seconds")
            log_and_print(f"📄 Output files found: {len(expected_files_found)}/{len(step['expected_outputs'])}")

            # Track critical step failures
            if step['critical'] and step_status not in ['validated', 'functional']:
                critical_step_failures.append(step_num)
                log_and_print(f"🚨 CRITICAL STEP {step_num} FAILED - This may affect subsequent steps")

            step_outcomes.append(step_outcome)

        except Exception as e:
            step_execution_time = time.time() - step_start_time
            log_and_print(f"💥 Critical failure in Step {step_num}: {e}")
            import traceback
            log_and_print(f"Error details: {traceback.format_exc()}")

            step_outcome = {
                'step_number': step_num,
                'step_name': step_name,
                'status': 'critical_failure',
                'error': str(e),
                'execution_time': step_execution_time,
                'critical': step['critical']
            }

            step_outcomes.append(step_outcome)

            if step['critical']:
                critical_step_failures.append(step_num)

    # Master Suite Analysis
    total_suite_time = time.time() - suite_start_time

    log_and_print(f"\n" + "=" * 70)
    log_and_print("📊 MASTER STEP-BY-STEP VALIDATION ANALYSIS")
    log_and_print("=" * 70)

    # Calculate overall statistics
    total_steps = len(step_outcomes)
    successful_steps = len([s for s in step_outcomes if s['status'] in ['validated', 'functional']])
    failed_steps = len([s for s in step_outcomes if s['status'] not in ['validated', 'functional']])

    critical_steps_total = len([s for s in step_outcomes if s['critical']])
    critical_steps_successful = len([s for s in step_outcomes if s['critical'] and s['status'] in ['validated', 'functional']])
    critical_steps_failed = critical_steps_total - critical_steps_successful

    overall_success_rate = successful_steps / max(1, total_steps)
    critical_success_rate = critical_steps_successful / max(1, critical_steps_total)

    total_execution_time = sum([s['execution_time'] for s in step_outcomes])
    avg_execution_time = total_execution_time / max(1, total_steps)

    log_and_print(f"🔢 MASTER SUITE STATISTICS:")
    log_and_print(f"   Total steps executed: {total_steps}")
    log_and_print(f"   Successful steps: {successful_steps}")
    log_and_print(f"   Failed steps: {failed_steps}")
    log_and_print(f"   Overall success rate: {overall_success_rate:.1%}")
    log_and_print(f"   Critical steps success rate: {critical_success_rate:.1%}")
    log_and_print(f"   Total execution time: {total_execution_time:.2f}s")
    log_and_print(f"   Average execution time: {avg_execution_time:.2f}s per step")
    log_and_print(f"   Suite duration: {total_suite_time:.2f}s")

    # Detailed step analysis
    log_and_print(f"\n📋 DETAILED STEP OUTCOMES:")
    for outcome in step_outcomes:
        status_icon = "✅" if outcome['status'] in ['validated', 'functional'] else "❌"
        critical_marker = "⚠️  CRITICAL" if outcome['critical'] else "📋"

        log_and_print(f"  {status_icon} Step {outcome['step_number']}: {outcome['step_name']} - {outcome['status']} {critical_marker}")
        log_and_print(f"     Execution time: {outcome['execution_time']:.2f}s")
        log_and_print(f"     Category: {outcome['category']}")

        if 'outputs_found' in outcome:
            log_and_print(f"     Output files: {outcome['outputs_found']}/{outcome['expected_outputs']}")

        if outcome['status'] not in ['validated', 'functional'] and 'error' in outcome:
            log_and_print(f"     Error: {outcome['error']}")

    # Critical step analysis
    log_and_print(f"\n🚨 CRITICAL STEP ANALYSIS:")
    if critical_step_failures:
        log_and_print(f"   Failed critical steps: {critical_step_failures}")
        log_and_print(f"   Critical failure impact: High - Core functionality compromised")
    else:
        log_and_print(f"   All critical steps passed: ✅")
        log_and_print(f"   Core functionality: Validated")

    # Store master results
    master_results['step_results'] = step_outcomes
    master_results['suite_summary'] = {
        'total_steps': total_steps,
        'successful_steps': successful_steps,
        'failed_steps': failed_steps,
        'overall_success_rate': overall_success_rate,
        'critical_success_rate': critical_success_rate,
        'total_execution_time': total_execution_time,
        'average_execution_time': avg_execution_time,
        'suite_duration': total_suite_time,
        'completion_time': datetime.now().isoformat()
    }

    master_results['critical_step_analysis'] = {
        'critical_steps_total': critical_steps_total,
        'critical_steps_successful': critical_steps_successful,
        'critical_steps_failed': critical_steps_failed,
        'failed_critical_steps': critical_step_failures,
        'core_functionality_status': 'validated' if not critical_step_failures else 'compromised'
    }

    # Generate overall conclusions
    if overall_success_rate >= 0.9 and not critical_step_failures:
        overall_assessment = "🟢 STEP-BY-STEP VALIDATION COMPREHENSIVELY PASSED"
        framework_status = "fully_validated"
    elif overall_success_rate >= 0.7 and critical_success_rate >= 0.8:
        overall_assessment = "🟡 STEP-BY-STEP VALIDATION MOSTLY PASSED"
        framework_status = "largely_validated"
    elif critical_success_rate >= 0.6:
        overall_assessment = "🟠 STEP-BY-STEP VALIDATION PARTIALLY PASSED"
        framework_status = "partially_validated"
    else:
        overall_assessment = "🔴 STEP-BY-STEP VALIDATION FAILED"
        framework_status = "validation_failed"

    log_and_print(f"\n🎯 MASTER STEP-BY-STEP CONCLUSION:")
    log_and_print(f"   {overall_assessment}")
    log_and_print(f"   Framework Status: {framework_status}")

    # Step dependency analysis
    dependency_issues = []

    # Check if data loading (Step 1) failed - affects all other steps
    step_1_outcome = next((s for s in step_outcomes if s['step_number'] == 1), None)
    if step_1_outcome and step_1_outcome['status'] not in ['validated', 'functional']:
        dependency_issues.append("Step 1 (Data Loading) failure affects all subsequent steps")

    # Check if quality control (Step 2) failed - affects annotation steps
    step_2_outcome = next((s for s in step_outcomes if s['step_number'] == 2), None)
    if step_2_outcome and step_2_outcome['status'] not in ['validated', 'functional']:
        dependency_issues.append("Step 2 (Quality Control) failure affects annotation accuracy")

    master_results['step_dependencies'] = {
        'dependency_issues': dependency_issues,
        'dependency_impact': 'high' if dependency_issues else 'none'
    }

    master_results['conclusions'] = {
        'overall_assessment': overall_assessment,
        'framework_status': framework_status,
        'overall_success_rate': overall_success_rate,
        'critical_success_rate': critical_success_rate,
        'key_findings': [
            f"Step success rate: {overall_success_rate:.1%}",
            f"Critical step success rate: {critical_success_rate:.1%}",
            f"Average execution time: {avg_execution_time:.1f}s per step",
            f"Suite completion time: {total_suite_time:.1f}s"
        ],
        'recommendations': [
            "Framework is ready for production" if framework_status == 'fully_validated' else "Address failed steps before production deployment",
            "All critical components functional" if not critical_step_failures else f"Fix critical steps: {critical_step_failures}",
            "Step execution performance is acceptable" if avg_execution_time <= 60 else "Optimize step execution speed",
            "Step isolation is effective" if len(dependency_issues) == 0 else "Address step dependency issues"
        ]
    }

    # Save master results
    log_and_print(f"\n💾 SAVING MASTER STEP VALIDATION RESULTS")
    log_and_print("-" * 50)

    # JSON results
    master_results_file = master_dir / "master_step_validation_results.json"
    with open(master_results_file, 'w') as f:
        json.dump(master_results, f, indent=2)
    log_and_print(f"📄 Saved master JSON results: {master_results_file}")

    # CSV summary
    csv_data = []
    for outcome in step_outcomes:
        csv_data.append({
            'Step_Number': outcome['step_number'],
            'Step_Name': outcome['step_name'],
            'Category': outcome['category'],
            'Status': outcome['status'],
            'Critical': outcome['critical'],
            'Execution_Time_s': outcome['execution_time'],
            'Expected_Outputs': outcome.get('expected_outputs', 0),
            'Outputs_Found': outcome.get('outputs_found', 0),
            'Success': 1 if outcome['status'] in ['validated', 'functional'] else 0
        })

    csv_file = master_dir / "master_step_validation_summary.csv"
    pd.DataFrame(csv_data).to_csv(csv_file, index=False)
    log_and_print(f"📊 Saved master CSV summary: {csv_file}")

    # Generate HTML report
    try:
        html_report = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Master Step-by-Step Validation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background: #f0f8ff; padding: 20px; border-radius: 10px; }}
                .success {{ color: #28a745; }}
                .failure {{ color: #dc3545; }}
                .warning {{ color: #ffc107; }}
                .critical {{ background: #fff3cd; padding: 5px; border-radius: 3px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .step-success {{ background-color: #d4edda; }}
                .step-failure {{ background-color: #f8d7da; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🧪 Master Step-by-Step Validation Report</h1>
                <p><strong>Framework:</strong> Standalone MS Validation - Step-by-Step Analysis</p>
                <p><strong>Execution Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Overall Success Rate:</strong> <span class="{'success' if overall_success_rate >= 0.8 else 'warning' if overall_success_rate >= 0.6 else 'failure'}">{overall_success_rate:.1%}</span></p>
                <p><strong>Critical Steps Success Rate:</strong> <span class="{'success' if critical_success_rate >= 0.8 else 'failure'}">{critical_success_rate:.1%}</span></p>
            </div>

            <h2>📊 Step Execution Summary</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Total Steps</td><td>{total_steps}</td></tr>
                <tr><td>Successful Steps</td><td class="success">{successful_steps}</td></tr>
                <tr><td>Failed Steps</td><td class="failure">{failed_steps}</td></tr>
                <tr><td>Critical Steps</td><td>{critical_steps_total}</td></tr>
                <tr><td>Failed Critical Steps</td><td class="{'failure' if critical_step_failures else 'success'}">{len(critical_step_failures)}</td></tr>
                <tr><td>Total Execution Time</td><td>{total_suite_time:.1f} seconds</td></tr>
            </table>

            <h2>🔬 Individual Step Results</h2>
            <table>
                <tr><th>Step</th><th>Name</th><th>Category</th><th>Status</th><th>Critical</th><th>Time (s)</th><th>Outputs</th></tr>
        """

        for outcome in step_outcomes:
            status_class = 'step-success' if outcome['status'] in ['validated', 'functional'] else 'step-failure'
            critical_marker = '⚠️ CRITICAL' if outcome['critical'] else 'Optional'

            html_report += f"""
                <tr class="{status_class}">
                    <td>{outcome['step_number']}</td>
                    <td>{outcome['step_name']}</td>
                    <td>{outcome['category']}</td>
                    <td>{outcome['status']}</td>
                    <td class="{'critical' if outcome['critical'] else ''}">{critical_marker}</td>
                    <td>{outcome['execution_time']:.2f}</td>
                    <td>{outcome.get('outputs_found', 0)}/{outcome.get('expected_outputs', 0)}</td>
                </tr>
            """

        html_report += f"""
            </table>

            <h2>🎯 Validation Conclusions</h2>
            <div class="{'success' if framework_status == 'fully_validated' else 'warning' if 'validated' in framework_status else 'failure'}">
                <h3>{overall_assessment}</h3>
                <p><strong>Framework Status:</strong> {framework_status}</p>
                <p><strong>Critical Steps Status:</strong> {'All Passed ✅' if not critical_step_failures else f'Failed: {critical_step_failures} ❌'}</p>
                <p><strong>Recommendation:</strong> {master_results['conclusions']['recommendations'][0]}</p>
            </div>

            <hr>
            <p><small>Generated by Master Step-by-Step Validation Suite at {datetime.now().isoformat()}</small></p>
        </body>
        </html>
        """

        html_file = master_dir / "master_step_validation_report.html"
        with open(html_file, 'w') as f:
            f.write(html_report)
        log_and_print(f"📄 Generated HTML report: {html_file}")

    except Exception as e:
        log_and_print(f"⚠️  HTML report generation failed: {e}")

    log_and_print(f"\n🧪 MASTER STEP-BY-STEP VALIDATION SUITE COMPLETE 🧪")
    log_and_print(f"📁 All results saved to: {master_dir}")
    log_and_print(f"📋 Master log: {master_log_file}")
    log_and_print(f"🎯 Framework Status: {framework_status}")

    if critical_step_failures:
        log_and_print(f"🚨 CRITICAL FAILURES DETECTED: Steps {critical_step_failures}")
        log_and_print(f"⚠️  These failures compromise core functionality")

    return master_results, framework_status in ['fully_validated', 'largely_validated']


if __name__ == "__main__":
    print("🚀 Starting Master Step-by-Step Validation Experiment Suite...")

    master_results, suite_success = main()

    if suite_success:
        print("\n✅ MASTER STEP-BY-STEP VALIDATION SUITE SUCCESSFUL!")
        print("🎉 Framework demonstrates strong step-by-step validation")
        sys.exit(0)
    else:
        print("\n❌ MASTER STEP-BY-STEP VALIDATION SUITE HAD CRITICAL ISSUES")
        print("🔍 Check individual step results for detailed analysis")
        sys.exit(1)
