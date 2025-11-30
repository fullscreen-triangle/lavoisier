"""
Visualize Preprocessing Stage Results

Shows what data was successfully extracted even if downstream stages failed.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

def load_stage_data(results_dir: Path):
    """Load saved stage data from .tab files"""

    # Load Stage 1 result JSON
    stage1_json = results_dir / "stage_01_preprocessing" / "stage_01_preprocessing_result.json"
    with open(stage1_json) as f:
        stage1_result = json.load(f)

    # Load tabular data if it exists
    stage1_data_tab = results_dir / "stage_01_preprocessing" / "stage_01_preprocessing_data.tab"
    if stage1_data_tab.exists():
        try:
            # Try to load as pandas DataFrame
            df = pd.read_csv(stage1_data_tab, sep='\t')
            return stage1_result, df
        except:
            pass

    return stage1_result, None

def visualize_preprocessing_results(results_dir: Path, output_dir: Path):
    """Create comprehensive visualizations of preprocessing results"""

    print("\n" + "="*80)
    print("PREPROCESSING RESULTS VISUALIZATION")
    print("="*80)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    stage1_result, df = load_stage_data(results_dir)

    # Extract metrics
    metrics = stage1_result['process_results'][0]['metrics']  # Spectral acquisition
    n_ms1 = metrics['n_ms1_spectra']
    n_ms2 = metrics['n_ms2_spectra']
    total = metrics['total_scans']

    print(f"\nData Extracted:")
    print(f"  MS1 spectra: {n_ms1}")
    print(f"  MS2 spectra: {n_ms2}")
    print(f"  Total scans: {total}")

    peak_metrics = stage1_result['process_results'][2]['metrics']  # Peak detection
    peaks_before = peak_metrics['peaks_before']
    peaks_after = peak_metrics['peaks_after']
    filter_rate = peak_metrics['filter_rate']

    print(f"\nPeak Filtering:")
    print(f"  Peaks before: {peaks_before:,}")
    print(f"  Peaks after: {peaks_after:,}")
    print(f"  Filter rate: {filter_rate:.2%}")

    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Spectrum type distribution
    ax = axes[0, 0]
    categories = ['MS1\n(Precursors)', 'MS2\n(Fragments)']
    counts = [n_ms1, n_ms2]
    colors = ['#2ecc71', '#3498db']
    bars = ax.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\nspectra',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('Number of Spectra', fontsize=12, fontweight='bold')
    ax.set_title('Spectrum Type Distribution', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(counts) * 1.2)

    # 2. Processing pipeline flowchart
    ax = axes[0, 1]
    ax.axis('off')

    # Draw flowchart
    stages = [
        ('Spectral\nAcquisition', 'completed', f'{total} scans'),
        ('Spectra\nAlignment', 'completed', f'{n_ms1+n_ms2} aligned'),
        ('Peak\nDetection', 'completed', f'{peaks_after:,} peaks'),
    ]

    y_pos = 0.9
    for i, (stage_name, status, info) in enumerate(stages):
        # Stage box
        color = '#2ecc71' if status == 'completed' else '#e74c3c'
        rect = plt.Rectangle((0.2, y_pos-0.15), 0.6, 0.12,
                            facecolor=color, edgecolor='black', linewidth=2, alpha=0.3)
        ax.add_patch(rect)

        # Stage text
        ax.text(0.5, y_pos-0.09, stage_name, ha='center', va='center',
                fontsize=11, fontweight='bold')
        ax.text(0.5, y_pos-0.12, info, ha='center', va='center',
                fontsize=9, style='italic')

        # Arrow to next stage
        if i < len(stages) - 1:
            ax.arrow(0.5, y_pos-0.17, 0, -0.08, head_width=0.05, head_length=0.02,
                    fc='black', ec='black', linewidth=2)

        y_pos -= 0.25

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Processing Pipeline', fontsize=14, fontweight='bold')

    # 3. Peak filtering efficiency
    ax = axes[1, 0]
    categories = ['Before\nFiltering', 'After\nFiltering']
    counts = [peaks_before, peaks_after]
    colors = ['#e74c3c', '#2ecc71']
    bars = ax.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

    # Add value labels
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count:,}\npeaks',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add filter rate annotation
    ax.text(0.5, max(counts) * 0.5, f'Filtered: {filter_rate:.2%}',
            ha='center', va='center', fontsize=13, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    ax.set_ylabel('Number of Peaks', fontsize=12, fontweight='bold')
    ax.set_title('Peak Detection & Filtering', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(counts) * 1.3)

    # 4. Stage execution times
    ax = axes[1, 1]
    stage_names = ['Acquisition', 'Alignment', 'Peak Det.']
    times = [pr['execution_time'] for pr in stage1_result['process_results']]
    colors_time = ['#3498db', '#9b59b6', '#e67e22']
    bars = ax.barh(stage_names, times, color=colors_time, alpha=0.7, edgecolor='black', linewidth=2)

    # Add time labels
    for bar, time in zip(bars, times):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
                f' {time:.2f}s',
                ha='left', va='center', fontsize=10, fontweight='bold')

    ax.set_xlabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Stage Execution Times', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / "preprocessing_summary.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_file}")
    plt.close()

    # Create detailed metrics plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # MS1 vs MS2 ratio pie chart
    ax = axes[0]
    sizes = [n_ms1, n_ms2]
    labels = [f'MS1\n{n_ms1} ({n_ms1/total*100:.1f}%)',
              f'MS2\n{n_ms2} ({n_ms2/total*100:.1f}%)']
    colors_pie = ['#2ecc71', '#3498db']
    explode = (0.05, 0.05)

    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors_pie, autopct='',
                                       explode=explode, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
    ax.set_title('MS1 vs MS2 Distribution', fontsize=14, fontweight='bold')

    # Data quality metrics
    ax = axes[1]
    ax.axis('off')

    metrics_text = f"""
    PREPROCESSING SUCCESS ✓

    Data Acquisition:
      • MS1 Spectra: {n_ms1}
      • MS2 Spectra: {n_ms2}
      • Total Scans: {total}
      • RT Range: {metrics['rt_range']}

    Peak Detection:
      • Peaks Detected: {peaks_before:,}
      • Peaks Filtered: {peaks_after:,}
      • Filter Rate: {filter_rate:.2%}
      • Avg Peaks/Spectrum: {peaks_after/total:.1f}

    Stage Status:
      • Spectral Acquisition: ✓ COMPLETED
      • Spectra Alignment: ✓ COMPLETED
      • Peak Detection: ✓ COMPLETED

    Next Steps:
      • Rerun with fixed code for
        fragmentation analysis
      • S-Entropy transformation
      • Network building
    """

    ax.text(0.1, 0.95, metrics_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

    plt.tight_layout()
    output_file = output_dir / "preprocessing_details.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print("\nNext: Rerun test with fixed code to get fragmentation results!")

if __name__ == "__main__":
    # Find latest results
    results_base = Path("results/fragmentation_test")
    if results_base.exists():
        output_dir = Path("results/visualizations/preprocessing")
        visualize_preprocessing_results(results_base, output_dir)
    else:
        print(f"Results directory not found: {results_base}")
        print("Run test_fragmentation_stage.py first!")
