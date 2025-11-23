#!/usr/bin/env python3
"""
Quality Control Validation Results
===================================

Visualizes quality assessment and filtering capabilities across datasets.

4-panel figure showing:
- Quality score distributions
- Threshold filtering effectiveness
- Assessment performance
- Dataset comparison

Author: Lavoisier Team
Date: 2025-10-27
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path

# Configure plotting
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams.update({
    'figure.dpi': 300,
    'font.size': 11,
    'font.family': 'sans-serif',
})

# Embedded validation results
RESULTS = {
    "PL_Neg_Waters_qTOF.mzML": {
        "spectra_assessed": 100,
        "assessment_rate": 4244.56,
        "mean_quality_score": 0.677,
        "std_quality_score": 0.103,
        "quality_range": 0.474,
        "high_quality_count": 85,
        "low_quality_count": 0,
        "thresholds": {
            "0.1": {"passed": 100, "failed": 0, "pass_rate": 1.0},
            "0.3": {"passed": 100, "failed": 0, "pass_rate": 1.0},
            "0.5": {"passed": 88, "failed": 12, "pass_rate": 0.88},
            "0.7": {"passed": 62, "failed": 38, "pass_rate": 0.62},
            "0.9": {"passed": 0, "failed": 100, "pass_rate": 0.0},
        }
    },
    "TG_Pos_Thermo_Orbi.mzML": {
        "spectra_assessed": 100,
        "assessment_rate": 4132.23,
        "mean_quality_score": 0.661,
        "std_quality_score": 0.112,
        "quality_range": 0.503,
        "high_quality_count": 79,
        "low_quality_count": 0,
        "thresholds": {
            "0.1": {"passed": 100, "failed": 0, "pass_rate": 1.0},
            "0.3": {"passed": 100, "failed": 0, "pass_rate": 1.0},
            "0.5": {"passed": 85, "failed": 15, "pass_rate": 0.85},
            "0.7": {"passed": 56, "failed": 44, "pass_rate": 0.56},
            "0.9": {"passed": 0, "failed": 100, "pass_rate": 0.0},
        }
    }
}


def create_quality_control_figure():
    """Create 4-panel quality control validation figure"""
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Panel A: Quality score distributions
    ax1 = fig.add_subplot(gs[0, 0])
    datasets = list(RESULTS.keys())
    colors = ['#3498DB', '#E74C3C']

    for i, (dataset, color) in enumerate(zip(datasets, colors)):
        mean = RESULTS[dataset]['mean_quality_score']
        std = RESULTS[dataset]['std_quality_score']

        # Simulate distribution
        x = np.linspace(0, 1, 200)
        y = np.exp(-0.5 * ((x - mean) / std) ** 2)
        y = y / np.max(y)  # Normalize

        ax1.plot(x, y, color=color, linewidth=2.5, label=dataset.split('.')[0], alpha=0.8)
        ax1.axvline(mean, color=color, linestyle='--', linewidth=2, alpha=0.6)
        ax1.fill_between(x, y, alpha=0.2, color=color)

    ax1.set_xlabel('Quality Score', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Normalized Density', fontweight='bold', fontsize=12)
    ax1.set_title('A. Quality Score Distributions', fontweight='bold', fontsize=14, pad=15)
    ax1.legend(fontsize=10, frameon=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)

    # Panel B: Threshold filtering effectiveness
    ax2 = fig.add_subplot(gs[0, 1])
    thresholds = ['0.1', '0.3', '0.5', '0.7', '0.9']
    x_pos = np.arange(len(thresholds))
    width = 0.35

    pass_rates_1 = [RESULTS[datasets[0]]['thresholds'][t]['pass_rate'] for t in thresholds]
    pass_rates_2 = [RESULTS[datasets[1]]['thresholds'][t]['pass_rate'] for t in thresholds]

    bars1 = ax2.bar(x_pos - width/2, pass_rates_1, width, label=datasets[0].split('.')[0],
                    color=colors[0], alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax2.bar(x_pos + width/2, pass_rates_2, width, label=datasets[1].split('.')[0],
                    color=colors[1], alpha=0.8, edgecolor='black', linewidth=1.5)

    ax2.set_xlabel('Quality Threshold', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Pass Rate', fontweight='bold', fontsize=12)
    ax2.set_title('B. Threshold Filtering Effectiveness', fontweight='bold', fontsize=14, pad=15)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(thresholds)
    ax2.legend(fontsize=10, frameon=True, shadow=True)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 1.1)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Panel C: Assessment performance metrics
    ax3 = fig.add_subplot(gs[1, 0])

    metrics = ['Mean\nQuality', 'High Quality\nRatio', 'Assessment\nRate (x1000)']
    data1 = [
        RESULTS[datasets[0]]['mean_quality_score'],
        RESULTS[datasets[0]]['high_quality_count'] / RESULTS[datasets[0]]['spectra_assessed'],
        RESULTS[datasets[0]]['assessment_rate'] / 1000
    ]
    data2 = [
        RESULTS[datasets[1]]['mean_quality_score'],
        RESULTS[datasets[1]]['high_quality_count'] / RESULTS[datasets[1]]['spectra_assessed'],
        RESULTS[datasets[1]]['assessment_rate'] / 1000
    ]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax3.bar(x - width/2, data1, width, label=datasets[0].split('.')[0],
                    color=colors[0], alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax3.bar(x + width/2, data2, width, label=datasets[1].split('.')[0],
                    color=colors[1], alpha=0.8, edgecolor='black', linewidth=1.5)

    ax3.set_ylabel('Value', fontweight='bold', fontsize=12)
    ax3.set_title('C. Assessment Performance Metrics', fontweight='bold', fontsize=14, pad=15)
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics, fontsize=10)
    ax3.legend(fontsize=10, frameon=True, shadow=True)
    ax3.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Panel D: Dataset comparison summary
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    table_data = [
        ['Metric', datasets[0].split('.')[0], datasets[1].split('.')[0]],
        ['', '', ''],
        ['Spectra Assessed',
         f"{RESULTS[datasets[0]]['spectra_assessed']}",
         f"{RESULTS[datasets[1]]['spectra_assessed']}"],
        ['Mean Quality Score',
         f"{RESULTS[datasets[0]]['mean_quality_score']:.3f}",
         f"{RESULTS[datasets[1]]['mean_quality_score']:.3f}"],
        ['Std Quality Score',
         f"{RESULTS[datasets[0]]['std_quality_score']:.3f}",
         f"{RESULTS[datasets[1]]['std_quality_score']:.3f}"],
        ['', '', ''],
        ['High Quality Count',
         f"{RESULTS[datasets[0]]['high_quality_count']}",
         f"{RESULTS[datasets[1]]['high_quality_count']}"],
        ['High Quality Ratio',
         f"{RESULTS[datasets[0]]['high_quality_count']/RESULTS[datasets[0]]['spectra_assessed']:.2f}",
         f"{RESULTS[datasets[1]]['high_quality_count']/RESULTS[datasets[1]]['spectra_assessed']:.2f}"],
        ['', '', ''],
        ['Assessment Rate',
         f"{RESULTS[datasets[0]]['assessment_rate']:.0f} spec/s",
         f"{RESULTS[datasets[1]]['assessment_rate']:.0f} spec/s"],
        ['', '', ''],
        ['Optimal Threshold', '0.5', '0.5'],
        ['Pass Rate @ 0.5',
         f"{RESULTS[datasets[0]]['thresholds']['0.5']['pass_rate']:.0%}",
         f"{RESULTS[datasets[1]]['thresholds']['0.5']['pass_rate']:.0%}"],
    ]

    table = ax4.table(cellText=table_data, cellLoc='left', loc='center',
                     colWidths=[0.4, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.2)

    # Style header
    for i in range(3):
        table[(0, i)].set_facecolor('#2C3E50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style section breaks
    for row_idx in [1, 5, 8, 10]:
        for col_idx in range(3):
            table[(row_idx, col_idx)].set_facecolor('#ECF0F1')

    ax4.set_title('D. Dataset Comparison Summary', fontweight='bold', fontsize=14, pad=15)

    fig.suptitle('Quality Control Validation Results\nSpectrum Quality Assessment and Filtering',
                fontsize=18, fontweight='bold', y=0.98)

    output_path = Path(__file__).parent / 'quality_control_validation.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✓ Quality control figure saved: {output_path}")


if __name__ == "__main__":
    print("="*80)
    print("QUALITY CONTROL VALIDATION - 4-PANEL FIGURE")
    print("="*80)
    create_quality_control_figure()
    print("\nVisualization complete!")
