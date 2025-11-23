#!/usr/bin/env python3
"""
Spectrum Embedding Validation Results
======================================

Visualizes spectrum embedding and similarity analysis capabilities.

4-panel figure showing:
- Embedding method comparison
- Similarity score distributions
- Method performance metrics
- Embedding quality analysis

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
        "spectra_processed": 30,
        "embedding_rate": 830.39,
        "methods": {
            "spec2vec": {
                "embeddings_created": 30,
                "dimension": 100,
                "embedding_rate": 1832.85,
                "avg_magnitude": 5.63,
                "similarity_score": 0.781,
                "success_rate": 1.0
            },
            "stellas": {
                "embeddings_created": 30,
                "dimension": 128,
                "embedding_rate": 344.22,
                "avg_magnitude": 36.42,
                "similarity_score": 0.615,
                "success_rate": 1.0
            },
            "fingerprint": {
                "embeddings_created": 30,
                "dimension": 256,
                "embedding_rate": 6171.42,
                "avg_magnitude": 2.04,
                "similarity_score": 0.131,
                "success_rate": 1.0
            }
        }
    },
    "TG_Pos_Thermo_Orbi.mzML": {
        "spectra_processed": 30,
        "embedding_rate": 866.74,
        "methods": {
            "spec2vec": {
                "embeddings_created": 30,
                "dimension": 100,
                "embedding_rate": 2336.05,
                "avg_magnitude": 5.67,
                "similarity_score": 0.802,
                "success_rate": 1.0
            },
            "stellas": {
                "embeddings_created": 30,
                "dimension": 128,
                "embedding_rate": 346.04,
                "avg_magnitude": 26.90,
                "similarity_score": 0.417,
                "success_rate": 1.0
            },
            "fingerprint": {
                "embeddings_created": 30,
                "dimension": 256,
                "embedding_rate": 6976.55,
                "avg_magnitude": 1.82,
                "similarity_score": 0.281,
                "success_rate": 1.0
            }
        }
    }
}


def create_spectrum_embedding_figure():
    """Create 4-panel spectrum embedding validation figure"""
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    datasets = list(RESULTS.keys())
    methods = list(RESULTS[datasets[0]]['methods'].keys())
    colors = ['#3498DB', '#E74C3C', '#2ECC71']

    # Panel A: Embedding rate comparison
    ax1 = fig.add_subplot(gs[0, 0])

    x = np.arange(len(methods))
    width = 0.35

    rates1 = [RESULTS[datasets[0]]['methods'][m]['embedding_rate']/1000 for m in methods]
    rates2 = [RESULTS[datasets[1]]['methods'][m]['embedding_rate']/1000 for m in methods]

    bars1 = ax1.bar(x - width/2, rates1, width, label=datasets[0].split('.')[0],
                    color='#3498DB', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax1.bar(x + width/2, rates2, width, label=datasets[1].split('.')[0],
                    color='#E74C3C', alpha=0.8, edgecolor='black', linewidth=1.5)

    ax1.set_xlabel('Embedding Method', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Embedding Rate (×1000 spec/s)', fontweight='bold', fontsize=12)
    ax1.set_title('A. Embedding Performance Comparison', fontweight='bold', fontsize=14, pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=11)
    ax1.legend(fontsize=10, frameon=True, shadow=True)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_yscale('log')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Panel B: Similarity score distributions
    ax2 = fig.add_subplot(gs[0, 1])

    # Extract similarity scores
    x_pos = np.arange(len(methods))
    width = 0.25

    for i, dataset in enumerate(datasets):
        scores = [RESULTS[dataset]['methods'][m]['similarity_score'] for m in methods]
        offset = (i - 0.5) * width
        bars = ax2.bar(x_pos + offset, scores, width,
                      label=dataset.split('.')[0],
                      color=['#3498DB', '#E74C3C'][i],
                      alpha=0.8, edgecolor='black', linewidth=1.5)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom',
                    fontsize=8, fontweight='bold', rotation=0)

    ax2.set_xlabel('Embedding Method', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Avg Similarity Score', fontweight='bold', fontsize=12)
    ax2.set_title('B. Similarity Search Performance', fontweight='bold', fontsize=14, pad=15)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(methods, fontsize=11)
    ax2.legend(fontsize=10, frameon=True, shadow=True)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 1.0)

    # Panel C: Embedding dimensionality vs performance
    ax3 = fig.add_subplot(gs[1, 0])

    # Get dimensions and rates for both datasets
    dimensions = [RESULTS[datasets[0]]['methods'][m]['dimension'] for m in methods]

    for i, dataset in enumerate(datasets):
        rates = [RESULTS[dataset]['methods'][m]['embedding_rate'] for m in methods]
        similarities = [RESULTS[dataset]['methods'][m]['similarity_score'] for m in methods]

        # Plot with size proportional to similarity
        scatter = ax3.scatter(dimensions, rates,
                            s=[sim*500 for sim in similarities],
                            c=[['#3498DB', '#E74C3C'][i]]*len(methods),
                            alpha=0.6, edgecolors='black', linewidth=2,
                            label=dataset.split('.')[0])

        # Add method labels
        for j, method in enumerate(methods):
            ax3.annotate(method, (dimensions[j], rates[j]),
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3',
                                facecolor=['lightblue', 'lightcoral'][i],
                                alpha=0.7))

    ax3.set_xlabel('Embedding Dimension', fontweight='bold', fontsize=12)
    ax3.set_ylabel('Embedding Rate (spec/s)', fontweight='bold', fontsize=12)
    ax3.set_title('C. Dimensionality vs Performance\n(bubble size = similarity score)',
                 fontweight='bold', fontsize=14, pad=15)
    ax3.legend(fontsize=10, frameon=True, shadow=True)
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')

    # Panel D: Method effectiveness comparison (grouped bar chart)
    ax4 = fig.add_subplot(gs[1, 1])

    # Create effectiveness metrics for each method
    metrics = ['Embedding\nRate\n(×1000)', 'Similarity\nScore', 'Dimension\n(÷100)']

    # Normalize data for visualization
    x = np.arange(len(metrics))
    width = 0.15

    method_colors = ['#3498DB', '#E74C3C', '#2ECC71']

    for i, method in enumerate(methods):
        offset = (i - 1) * width

        # Get average data across both datasets
        avg_rate = np.mean([RESULTS[d]['methods'][method]['embedding_rate'] for d in datasets]) / 1000
        avg_sim = np.mean([RESULTS[d]['methods'][method]['similarity_score'] for d in datasets])
        avg_dim = RESULTS[datasets[0]]['methods'][method]['dimension'] / 100

        values = [avg_rate, avg_sim, avg_dim]

        bars = ax4.bar(x + offset, values, width,
                      label=method,
                      color=method_colors[i],
                      alpha=0.8, edgecolor='black', linewidth=1)

        # Add value labels on top
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}' if val < 10 else f'{val:.0f}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax4.set_ylabel('Normalized Value', fontweight='bold', fontsize=12)
    ax4.set_title('D. Method Effectiveness Comparison\n(Averaged Across Datasets)',
                 fontweight='bold', fontsize=14, pad=15)
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics, fontsize=10)
    ax4.legend(fontsize=10, frameon=True, shadow=True, loc='upper left')
    ax4.grid(True, alpha=0.3, axis='y')

    # Add annotation for winner
    ax4.text(0.98, 0.98, 'Best Overall:\nspec2vec',
            transform=ax4.transAxes, fontsize=11, fontweight='bold',
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#D5F4E6',
                     edgecolor='green', linewidth=2, alpha=0.9))

    fig.suptitle('Spectrum Embedding Validation Results\nMulti-Method Embedding and Similarity Analysis',
                fontsize=18, fontweight='bold', y=0.98)

    output_path = Path(__file__).parent / 'spectrum_embedding_validation.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✓ Spectrum embedding figure saved: {output_path}")


if __name__ == "__main__":
    print("="*80)
    print("SPECTRUM EMBEDDING VALIDATION - 4-PANEL FIGURE")
    print("="*80)
    create_spectrum_embedding_figure()
    print("\nVisualization complete!")
