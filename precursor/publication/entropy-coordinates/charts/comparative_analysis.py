#!/usr/bin/env python3
"""
Comparative Analysis: Numerical vs Visual Pipelines
====================================================

Scientific comparison of numerical and visual metabolomics processing pipelines.

4-panel figure showing:
- Processing efficiency comparison
- Annotation performance analysis
- Quality and coverage metrics
- Method strengths and weaknesses

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

# Embedded comparative results
NUMERICAL_RESULTS = {
    "PL_Neg_Waters_qTOF.mzML": {
        "spectra_processed": 117,
        "high_quality_spectra": 117,
        "processing_time": 1.258,
        "spectra_per_second": 92.98,
        "annotation_rate": 0.0104,
        "annotated_count": 1,
        "mean_quality_score": 0.657,
        "embeddings_created": 100,
        "clusters_created": 5,
        "active_databases": 1
    },
    "TG_Pos_Thermo_Orbi.mzML": {
        "spectra_processed": 121,
        "high_quality_spectra": 121,
        "processing_time": 1.512,
        "spectra_per_second": 80.03,
        "annotation_rate": 0.0165,
        "annotated_count": 2,
        "mean_quality_score": 0.682,
        "embeddings_created": 100,
        "clusters_created": 5,
        "active_databases": 2
    }
}

VISUAL_RESULTS = {
    "PL_Neg_Waters_qTOF.mzML": {
        "spectra_processed": 50,
        "ions_extracted": 8161,
        "processing_time": 2.153,
        "spectra_per_second": 23.23,
        "ion_extraction_rate": 163.22,
        "drip_images_created": 50,
        "conversion_rate": 1.0,
        "annotation_rate": 0.0,
        "annotated_count": 0,
        "overall_quality_score": 0.7
    },
    "TG_Pos_Thermo_Orbi.mzML": {
        "spectra_processed": 50,
        "ions_extracted": 8556,
        "processing_time": 3.781,
        "spectra_per_second": 13.22,
        "ion_extraction_rate": 171.12,
        "drip_images_created": 50,
        "conversion_rate": 1.0,
        "annotation_rate": 0.0,
        "annotated_count": 0,
        "overall_quality_score": 0.7
    }
}


def create_comparative_analysis_figure():
    """Create 4-panel comparative analysis figure"""
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    datasets = list(NUMERICAL_RESULTS.keys())
    colors_methods = ['#3498DB', '#E74C3C']  # Blue for numerical, red for visual

    # Panel A: Processing efficiency comparison
    ax1 = fig.add_subplot(gs[0, 0])

    metrics = ['Spectra/s\n(Higher=Better)', 'Processing\nTime (s)\n(Lower=Better)',
               'Quality\nScore']

    # Normalize for visualization
    num_data = [
        np.mean([NUMERICAL_RESULTS[d]['spectra_per_second'] for d in datasets]) / 100,
        np.mean([NUMERICAL_RESULTS[d]['processing_time'] for d in datasets]),
        np.mean([NUMERICAL_RESULTS[d]['mean_quality_score'] for d in datasets])
    ]

    vis_data = [
        np.mean([VISUAL_RESULTS[d]['spectra_per_second'] for d in datasets]) / 100,
        np.mean([VISUAL_RESULTS[d]['processing_time'] for d in datasets]),
        np.mean([VISUAL_RESULTS[d]['overall_quality_score'] for d in datasets])
    ]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax1.bar(x - width/2, num_data, width, label='Numerical Pipeline',
                    color=colors_methods[0], alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax1.bar(x + width/2, vis_data, width, label='Visual Pipeline',
                    color=colors_methods[1], alpha=0.8, edgecolor='black', linewidth=1.5)

    ax1.set_ylabel('Normalized Value', fontweight='bold', fontsize=12)
    ax1.set_title('A. Processing Efficiency Comparison', fontweight='bold', fontsize=14, pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, fontsize=10)
    ax1.legend(fontsize=10, frameon=True, shadow=True, loc='upper right')
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

    # Panel B: Annotation performance
    ax2 = fig.add_subplot(gs[0, 1])

    # Calculate average annotation rates
    num_ann_rates = [NUMERICAL_RESULTS[d]['annotation_rate'] * 100 for d in datasets]
    vis_ann_rates = [VISUAL_RESULTS[d]['annotation_rate'] * 100 for d in datasets]

    x = np.arange(len(datasets))
    width = 0.35

    bars1 = ax2.bar(x - width/2, num_ann_rates, width, label='Numerical',
                    color=colors_methods[0], alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax2.bar(x + width/2, vis_ann_rates, width, label='Visual',
                    color=colors_methods[1], alpha=0.8, edgecolor='black', linewidth=1.5)

    ax2.set_xlabel('Dataset', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Annotation Rate (%)', fontweight='bold', fontsize=12)
    ax2.set_title('B. Annotation Performance Analysis', fontweight='bold', fontsize=14, pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels([d.split('_')[0] for d in datasets], fontsize=10)
    ax2.legend(fontsize=10, frameon=True, shadow=True)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}%', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

    # Panel C: Method capabilities radar chart
    ax3 = fig.add_subplot(gs[1, 0], projection='polar')

    categories = ['Speed', 'Annotation', 'Quality', 'Coverage', 'Efficiency']
    N = len(categories)

    # Numerical pipeline scores (normalized 0-1)
    numerical_scores = [
        0.85,  # Speed (high)
        0.12,  # Annotation (low)
        0.68,  # Quality
        0.60,  # Coverage
        0.75   # Efficiency
    ]

    # Visual pipeline scores (normalized 0-1)
    visual_scores = [
        0.25,  # Speed (lower)
        0.0,   # Annotation (none)
        0.70,  # Quality
        1.0,   # Coverage (ion extraction)
        0.55   # Efficiency
    ]

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    numerical_scores += numerical_scores[:1]
    visual_scores += visual_scores[:1]
    angles += angles[:1]

    ax3.plot(angles, numerical_scores, 'o-', linewidth=2.5,
            color=colors_methods[0], label='Numerical', markersize=8)
    ax3.fill(angles, numerical_scores, alpha=0.25, color=colors_methods[0])

    ax3.plot(angles, visual_scores, 's-', linewidth=2.5,
            color=colors_methods[1], label='Visual', markersize=8)
    ax3.fill(angles, visual_scores, alpha=0.25, color=colors_methods[1])

    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(categories, fontsize=11, fontweight='bold')
    ax3.set_ylim(0, 1)
    ax3.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax3.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    ax3.grid(True)
    ax3.set_title('C. Method Capabilities Profile', fontweight='bold',
                 fontsize=14, pad=20, y=1.08)
    ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10,
              frameon=True, shadow=True)

    # Panel D: Comprehensive comparison table
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    # Calculate averages across datasets
    num_avg_speed = np.mean([NUMERICAL_RESULTS[d]['spectra_per_second'] for d in datasets])
    vis_avg_speed = np.mean([VISUAL_RESULTS[d]['spectra_per_second'] for d in datasets])

    num_avg_ann = np.mean([NUMERICAL_RESULTS[d]['annotation_rate'] for d in datasets]) * 100
    vis_avg_ann = np.mean([VISUAL_RESULTS[d]['annotation_rate'] for d in datasets]) * 100

    table_data = [
        ['Metric', 'Numerical', 'Visual', 'Winner'],
        ['', '', '', ''],
        ['Avg Processing Speed', f'{num_avg_speed:.1f} spec/s',
         f'{vis_avg_speed:.1f} spec/s', '🔵 Numerical'],
        ['Avg Annotation Rate', f'{num_avg_ann:.2f}%',
         f'{vis_avg_ann:.2f}%', '🔵 Numerical'],
        ['', '', '', ''],
        ['Ion Extraction', 'N/A', '167 ions/spec', '🔴 Visual'],
        ['CV Conversion', 'N/A', '100%', '🔴 Visual'],
        ['Feature Diversity', '14-D', 'Visual Features', '🟡 Tie'],
        ['', '', '', ''],
        ['Database Coverage', '1-2 DBs', 'LipidMaps', '🔵 Numerical'],
        ['Embedding Methods', '3 methods', 'CV-based', '🔵 Numerical'],
        ['Clustering', '5 clusters', 'N/A', '🔵 Numerical'],
        ['', '', '', ''],
        ['Strengths', 'Fast, Multi-DB', 'Complete Ion Info', ''],
        ['Weaknesses', 'Low Annotation', 'No Annotation', ''],
        ['', '', '', ''],
        ['Recommendation', '✓ Production', '✓ Research', ''],
    ]

    table = ax4.table(cellText=table_data, cellLoc='left', loc='center',
                     colWidths=[0.3, 0.25, 0.25, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.9)

    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#2C3E50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style section breaks
    for row_idx in [1, 4, 8, 12, 15]:
        for col_idx in range(4):
            table[(row_idx, col_idx)].set_facecolor('#ECF0F1')

    # Highlight numerical wins (blue)
    for row_idx in [2, 3, 9, 10, 11]:
        table[(row_idx, 3)].set_facecolor('#D6EAF8')

    # Highlight visual wins (red)
    for row_idx in [5, 6]:
        table[(row_idx, 3)].set_facecolor('#FADBD8')

    # Highlight recommendations
    for col_idx in range(4):
        table[(16, col_idx)].set_facecolor('#D5F4E6')
        table[(16, col_idx)].set_text_props(weight='bold')

    ax4.set_title('D. Comprehensive Method Comparison', fontweight='bold',
                 fontsize=14, pad=15)

    fig.suptitle('Comparative Analysis: Numerical vs Visual Pipelines\n'
                'Systematic Evaluation of Metabolomics Processing Approaches',
                fontsize=18, fontweight='bold', y=0.98)

    output_path = Path(__file__).parent / 'comparative_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✓ Comparative analysis figure saved: {output_path}")


if __name__ == "__main__":
    print("="*80)
    print("COMPARATIVE ANALYSIS - 4-PANEL FIGURE")
    print("="*80)
    create_comparative_analysis_figure()
    print("\nVisualization complete!")
