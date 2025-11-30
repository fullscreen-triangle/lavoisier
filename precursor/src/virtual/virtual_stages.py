"""
Virtual Stages Visualization - REAL DATA

Visualizes the complete pipeline process using ACTUAL theatre results:
Preprocessing → S-Entropy → Fragmentation → BMD → Completion

Shows the transformation at each stage with detailed metrics from real experiments.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
from pathlib import Path
import sys

# Add parent directory to path
script_dir = Path(__file__).parent
src_dir = script_dir.parent
sys.path.insert(0, str(src_dir))

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300


def load_complete_pipeline_data(results_dir):
    """
    Load complete pipeline data from theatre results
    """
    results_dir = Path(results_dir)

    # Load theatre result (has all stages)
    theatre_json = results_dir / "theatre_result.json"

    if not theatre_json.exists():
        print(f"Warning: No theatre_result.json found in {results_dir}")
        return None

    with open(theatre_json) as f:
        theatre_data = json.load(f)

    return theatre_data


def create_pipeline_flow_diagram(theatre_data, platform_name, output_dir):
    """
    Create a visual flowchart of the pipeline execution
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.suptitle(f'Pipeline Execution Flow - {platform_name}',
                 fontsize=16, fontweight='bold')

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # Get stage results
    stage_results = theatre_data.get('stage_results', {})
    execution_order = theatre_data.get('execution_order', [])

    # Define stage positions
    stages = [
        ('stage_01_preprocessing', 'Spectral\nPreprocessing', 5, 10),
        ('stage_02_sentropy', 'S-Entropy\nTransformation', 5, 8),
        ('stage_02_5_fragmentation', 'Fragmentation\nNetwork', 5, 6),
        ('stage_03_bmd', 'BMD\nGrounding', 5, 4),
        ('stage_04_completion', 'Categorical\nCompletion', 5, 2)
    ]

    for stage_id, stage_name, x, y in stages:
        if stage_id in stage_results:
            stage_data = stage_results[stage_id]
            status = stage_data.get('status', 'unknown')
            exec_time = stage_data.get('execution_time', 0)

            # Color based on status
            if status == 'completed':
                color = '#2ecc71'
            elif status == 'failed':
                color = '#e74c3c'
            else:
                color = '#95a5a6'

            # Draw stage box
            rect = plt.Rectangle((x-1.5, y-0.6), 3, 1.2,
                                facecolor=color, edgecolor='black',
                                linewidth=3, alpha=0.3)
            ax.add_patch(rect)

            # Stage name
            ax.text(x, y+0.2, stage_name, ha='center', va='center',
                   fontsize=12, fontweight='bold')

            # Execution time
            ax.text(x, y-0.3, f'{exec_time:.2f}s', ha='center', va='center',
                   fontsize=10, style='italic')

            # Metrics
            metrics = stage_data.get('metrics', {})
            if metrics:
                metrics_text = ''
                key_metrics = ['total_processes', 'completed_processes', 'failed_processes']
                for key in key_metrics:
                    if key in metrics:
                        metrics_text += f'{key}: {metrics[key]}\n'

                if metrics_text:
                    ax.text(x+1.8, y, metrics_text, ha='left', va='center',
                           fontsize=8, family='monospace',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        else:
            # Stage not executed
            rect = plt.Rectangle((x-1.5, y-0.6), 3, 1.2,
                                facecolor='#ecf0f1', edgecolor='gray',
                                linewidth=2, alpha=0.3, linestyle='--')
            ax.add_patch(rect)

            ax.text(x, y, stage_name, ha='center', va='center',
                   fontsize=12, color='gray')

        # Draw arrow to next stage
        if y > 2:
            ax.arrow(x, y-0.7, 0, -1.2, head_width=0.3, head_length=0.2,
                    fc='black', ec='black', linewidth=2)

    # Overall statistics
    total_time = theatre_data.get('execution_time', 0)
    n_stages = theatre_data.get('metrics', {}).get('total_stages', 0)
    completed = theatre_data.get('metrics', {}).get('completed_stages', 0)
    failed = theatre_data.get('metrics', {}).get('failed_stages', 0)

    summary_text = f"""
THEATRE EXECUTION SUMMARY

Total Execution Time: {total_time:.2f}s
Total Stages: {n_stages}
Completed: {completed}
Failed: {failed}

Navigation Mode: {theatre_data.get('navigation_mode', 'N/A')}
Status: {theatre_data.get('status', 'N/A').upper()}
"""

    ax.text(0.5, 11, summary_text, ha='left', va='top',
           fontsize=11, family='monospace',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    plt.tight_layout()

    output_file = output_dir / f'{platform_name}_pipeline_flow.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    return output_file


def create_stage_metrics_comparison(all_theatre_data, output_dir):
    """
    Compare metrics across platforms
    """
    if not all_theatre_data:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Pipeline Metrics - Cross-Platform Comparison',
                 fontsize=16, fontweight='bold')

    platforms = list(all_theatre_data.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(platforms)))

    # Collect stage execution times
    stage_names = ['Preprocessing', 'S-Entropy', 'Fragmentation', 'BMD', 'Completion']
    stage_ids = ['stage_01_preprocessing', 'stage_02_sentropy', 'stage_02_5_fragmentation',
                 'stage_03_bmd', 'stage_04_completion']

    # Plot 1: Stage execution times
    ax = axes[0, 0]
    x = np.arange(len(stage_names))
    width = 0.8 / len(platforms)

    for idx, platform in enumerate(platforms):
        theatre_data = all_theatre_data[platform]
        stage_results = theatre_data.get('stage_results', {})

        times = []
        for stage_id in stage_ids:
            if stage_id in stage_results:
                times.append(stage_results[stage_id].get('execution_time', 0))
            else:
                times.append(0)

        offset = (idx - len(platforms)/2) * width
        ax.bar(x + offset, times, width, label=platform,
               color=colors[idx], alpha=0.7, edgecolor='black')

    ax.set_xlabel('Pipeline Stage', fontsize=11, fontweight='bold')
    ax.set_ylabel('Execution Time (s)', fontsize=11, fontweight='bold')
    ax.set_title('Stage Execution Times', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(stage_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Plot 2: Process success rates
    ax = axes[0, 1]

    for idx, platform in enumerate(platforms):
        theatre_data = all_theatre_data[platform]
        stage_results = theatre_data.get('stage_results', {})

        success_rates = []
        for stage_id in stage_ids:
            if stage_id in stage_results:
                metrics = stage_results[stage_id].get('metrics', {})
                total = metrics.get('total_processes', 0)
                completed = metrics.get('completed_processes', 0)
                rate = (completed / total * 100) if total > 0 else 0
                success_rates.append(rate)
            else:
                success_rates.append(0)

        offset = (idx - len(platforms)/2) * width
        ax.bar(x + offset, success_rates, width, label=platform,
               color=colors[idx], alpha=0.7, edgecolor='black')

    ax.set_xlabel('Pipeline Stage', fontsize=11, fontweight='bold')
    ax.set_ylabel('Success Rate (%)', fontsize=11, fontweight='bold')
    ax.set_title('Process Success Rates', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(stage_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 105)

    # Plot 3: Total execution time comparison
    ax = axes[1, 0]

    platform_times = [all_theatre_data[p].get('execution_time', 0) for p in platforms]
    bars = ax.bar(range(len(platforms)), platform_times, color=colors,
                  alpha=0.7, edgecolor='black', linewidth=2)

    # Add value labels on bars
    for bar, time in zip(bars, platform_times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.1f}s',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel('Platform', fontsize=11, fontweight='bold')
    ax.set_ylabel('Total Execution Time (s)', fontsize=11, fontweight='bold')
    ax.set_title('Total Pipeline Execution Time', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(platforms)))
    ax.set_xticklabels(platforms, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    # Plot 4: Stage completion status
    ax = axes[1, 1]
    ax.axis('off')

    summary_text = "PIPELINE COMPLETION STATUS\n\n"
    for platform in platforms:
        theatre_data = all_theatre_data[platform]
        metrics = theatre_data.get('metrics', {})

        total_stages = metrics.get('total_stages', 0)
        completed_stages = metrics.get('completed_stages', 0)
        failed_stages = metrics.get('failed_stages', 0)

        summary_text += f"{platform}:\n"
        summary_text += f"  Total Stages: {total_stages}\n"
        summary_text += f"  Completed: {completed_stages}\n"
        summary_text += f"  Failed: {failed_stages}\n"
        summary_text += f"  Status: {theatre_data.get('status', 'N/A').upper()}\n\n"

    ax.text(0.1, 0.95, summary_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.5))

    plt.tight_layout()

    output_file = output_dir / 'pipeline_metrics_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    return output_file


def main():
    """Main visualization workflow"""
    print("="*80)
    print("VIRTUAL STAGES - COMPLETE PIPELINE VISUALIZATION - REAL DATA")
    print("="*80)

    # Determine paths
    script_dir = Path(__file__).parent
    precursor_root = script_dir.parent.parent
    results_dir = precursor_root / "results" / "fragmentation_comparison"
    output_dir = precursor_root / "visualizations"
    output_dir.mkdir(exist_ok=True)

    print(f"\nResults directory: {results_dir}")
    print(f"Output directory: {output_dir}\n")

    if not results_dir.exists():
        print(f"✗ Error: Results directory not found: {results_dir}")
        print("Please run the fragmentation pipeline first!")
        sys.exit(1)

    # Find available platforms
    platforms = [d for d in results_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]

    if not platforms:
        print("✗ Error: No platform directories found")
        sys.exit(1)

    platform_names = [p.name for p in platforms]
    print(f"Found {len(platforms)} platform(s): {', '.join(platform_names)}\n")

    # Load theatre data from all platforms
    all_theatre_data = {}

    for platform, platform_name in zip(platforms, platform_names):
        print(f"Loading REAL pipeline data from: {platform_name}")
        theatre_data = load_complete_pipeline_data(platform)

        if theatre_data:
            print(f"  ✓ Loaded theatre result (REAL execution data)")
            all_theatre_data[platform_name] = theatre_data

            # Create flow diagram for this platform
            output_file = create_pipeline_flow_diagram(theatre_data, platform_name, output_dir)
            print(f"  ✓ Saved: {output_file.name}")
        else:
            print(f"  ✗ No theatre data found")

        print()

    # Create cross-platform comparison
    if len(all_theatre_data) > 1:
        print("Creating cross-platform metrics comparison...")
        output_file = create_stage_metrics_comparison(all_theatre_data, output_dir)
        if output_file:
            print(f"  ✓ Saved: {output_file.name}")

    print("\n" + "="*80)
    print("✓ VIRTUAL STAGES VISUALIZATION COMPLETE")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print(f"Total platforms visualized: {len(all_theatre_data)}")


if __name__ == "__main__":
    main()
