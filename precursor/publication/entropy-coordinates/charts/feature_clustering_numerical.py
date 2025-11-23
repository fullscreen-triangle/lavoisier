#!/usr/bin/env python3
"""
Feature Clustering Validation Results
======================================

Visualizes feature extraction and clustering performance across different configurations.

4-panel figure showing:
- Clustering quality across configurations
- Feature extraction statistics
- Cluster size distributions
- Performance metrics

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
        "spectra_processed": 50,
        "feature_dimensions": 14,
        "feature_diversity": 0.555,
        "clusters_tested": [3, 5, 8, 10],
        "clustering_results": {
            3: {"quality_score": 0.867, "balance": 0.667, "rate": 2059.40},
            5: {"quality_score": 0.770, "balance": 0.424, "rate": 1905.06},
            8: {"quality_score": 0.851, "balance": 0.627, "rate": 2154.26},
            10: {"quality_score": 0.807, "balance": 0.518, "rate": 1923.96}
        },
        "best_cluster_count": 3,
        "best_quality_score": 0.867,
        "effectiveness_score": 0.845
    },
    "TG_Pos_Thermo_Orbi.mzML": {
        "spectra_processed": 50,
        "feature_dimensions": 14,
        "feature_diversity": 0.570,
        "clusters_tested": [3, 5, 8, 10],
        "clustering_results": {
            3: {"quality_score": 0.909, "balance": 0.774, "rate": 1731.97},
            5: {"quality_score": 0.796, "balance": 0.490, "rate": 2195.74},
            8: {"quality_score": -0.233, "balance": -1.646, "rate": 704.61},  # Failed
            10: {"quality_score": 0.794, "balance": 0.486, "rate": 2419.89}
        },
        "best_cluster_count": 3,
        "best_quality_score": 0.909,
        "effectiveness_score": 0.784
    }
}


def create_feature_clustering_figure():
    """Create 4-panel feature clustering validation figure"""
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    datasets = list(RESULTS.keys())
    colors = ['#3498DB', '#E74C3C']

    # Panel A: Clustering quality scores across configurations
    ax1 = fig.add_subplot(gs[0, 0])

    cluster_counts = RESULTS[datasets[0]]['clusters_tested']

    for i, dataset in enumerate(datasets):
        quality_scores = [RESULTS[dataset]['clustering_results'][k]['quality_score']
                         for k in cluster_counts]
        # Clip negative values for visualization
        quality_scores_clipped = [max(0, q) for q in quality_scores]

        ax1.plot(cluster_counts, quality_scores_clipped,
                marker='o', markersize=10, linewidth=2.5,
                color=colors[i], label=dataset.split('.')[0], alpha=0.8)

        # Mark best configuration
        best_k = RESULTS[dataset]['best_cluster_count']
        best_q = RESULTS[dataset]['best_quality_score']
        ax1.scatter([best_k], [best_q], s=300, marker='*',
                   color='gold', edgecolors='black', linewidth=2,
                   zorder=10, label=f'Best: k={best_k}' if i == 0 else '')

    ax1.set_xlabel('Number of Clusters (k)', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Clustering Quality Score', fontweight='bold', fontsize=12)
    ax1.set_title('A. Clustering Quality vs Configuration', fontweight='bold', fontsize=14, pad=15)
    ax1.legend(fontsize=10, frameon=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(cluster_counts)
    ax1.set_ylim(0, 1.0)

    # Panel B: Cluster balance comparison
    ax2 = fig.add_subplot(gs[0, 1])

    x = np.arange(len(cluster_counts))
    width = 0.35

    balance1 = [max(0, RESULTS[datasets[0]]['clustering_results'][k]['balance'])
                for k in cluster_counts]
    balance2 = [max(0, RESULTS[datasets[1]]['clustering_results'][k]['balance'])
                for k in cluster_counts]

    bars1 = ax2.bar(x - width/2, balance1, width, label=datasets[0].split('.')[0],
                    color=colors[0], alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax2.bar(x + width/2, balance2, width, label=datasets[1].split('.')[0],
                    color=colors[1], alpha=0.8, edgecolor='black', linewidth=1.5)

    ax2.set_xlabel('Number of Clusters (k)', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Cluster Balance Score', fontweight='bold', fontsize=12)
    ax2.set_title('B. Cluster Balance Analysis', fontweight='bold', fontsize=14, pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'k={k}' for k in cluster_counts], fontsize=10)
    ax2.legend(fontsize=10, frameon=True, shadow=True)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom',
                        fontsize=8, fontweight='bold')

    # Panel C: Clustering rate performance
    ax3 = fig.add_subplot(gs[1, 0])

    for i, dataset in enumerate(datasets):
        rates = [RESULTS[dataset]['clustering_results'][k]['rate']
                for k in cluster_counts]

        ax3.plot(cluster_counts, rates,
                marker='s', markersize=8, linewidth=2.5,
                color=colors[i], label=dataset.split('.')[0], alpha=0.8)

    ax3.set_xlabel('Number of Clusters (k)', fontweight='bold', fontsize=12)
    ax3.set_ylabel('Clustering Rate (spec/s)', fontweight='bold', fontsize=12)
    ax3.set_title('C. Clustering Performance', fontweight='bold', fontsize=14, pad=15)
    ax3.legend(fontsize=10, frameon=True, shadow=True)
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(cluster_counts)

    # Panel D: Summary statistics table
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    table_data = [
        ['Metric', datasets[0].split('.')[0], datasets[1].split('.')[0]],
        ['', '', ''],
        ['Spectra Processed',
         f"{RESULTS[datasets[0]]['spectra_processed']}",
         f"{RESULTS[datasets[1]]['spectra_processed']}"],
        ['Feature Dimensions',
         f"{RESULTS[datasets[0]]['feature_dimensions']}",
         f"{RESULTS[datasets[1]]['feature_dimensions']}"],
        ['Feature Diversity',
         f"{RESULTS[datasets[0]]['feature_diversity']:.3f}",
         f"{RESULTS[datasets[1]]['feature_diversity']:.3f}"],
        ['', '', ''],
        ['Configurations Tested', '4', '4'],
        ['Successful Configs', '4', '3'],
        ['', '', ''],
        ['Best Cluster Count',
         f"{RESULTS[datasets[0]]['best_cluster_count']}",
         f"{RESULTS[datasets[1]]['best_cluster_count']}"],
        ['Best Quality Score',
         f"{RESULTS[datasets[0]]['best_quality_score']:.3f}",
         f"{RESULTS[datasets[1]]['best_quality_score']:.3f}"],
        ['', '', ''],
        ['Effectiveness Score',
         f"{RESULTS[datasets[0]]['effectiveness_score']:.3f}",
         f"{RESULTS[datasets[1]]['effectiveness_score']:.3f}"],
        ['Success Rate', '100%', '75%'],
        ['', '', ''],
        ['Performance Grade', '🟢 Excellent', '🟢 Excellent'],
    ]

    table = ax4.table(cellText=table_data, cellLoc='left', loc='center',
                     colWidths=[0.45, 0.275, 0.275])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.0)

    # Style header
    for i in range(3):
        table[(0, i)].set_facecolor('#2C3E50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style section breaks
    for row_idx in [1, 5, 8, 11, 14]:
        for col_idx in range(3):
            table[(row_idx, col_idx)].set_facecolor('#ECF0F1')

    # Highlight best configurations
    for col_idx in range(3):
        table[(9, col_idx)].set_facecolor('#FFF9C4')
        table[(10, col_idx)].set_facecolor('#FFF9C4')

    # Highlight assessment
    for col_idx in range(3):
        table[(15, col_idx)].set_facecolor('#D5F4E6')
        table[(15, col_idx)].set_text_props(weight='bold')

    ax4.set_title('D. Clustering Summary', fontweight='bold', fontsize=14, pad=15)

    fig.suptitle('Feature Clustering Validation Results\n14-Dimensional Feature Space Analysis',
                fontsize=18, fontweight='bold', y=0.98)

    output_path = Path(__file__).parent / 'feature_clustering_validation.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✓ Feature clustering figure saved: {output_path}")


if __name__ == "__main__":
    print("="*80)
    print("FEATURE CLUSTERING VALIDATION - 4-PANEL FIGURE")
    print("="*80)
    create_feature_clustering_figure()
    print("\nVisualization complete!")
