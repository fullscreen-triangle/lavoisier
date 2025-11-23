#!/usr/bin/env python3
"""
Database Search Validation Results
===================================

Visualizes database search and annotation performance across multiple databases.

4-panel figure showing:
- Database hit rates
- Search performance comparison
- Coverage analysis
- Annotation effectiveness

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
        "spectra_searched": 20,
        "annotation_rate": 0.0,
        "search_rate": 6110.58,
        "databases": {
            "LIPIDMAPS": {"hit_rate": 0.0, "search_rate": 6629.21, "annotations": 0},
            "MSLIPIDS": {"hit_rate": 0.0, "search_rate": 14789.51, "annotations": 0},
            "PUBCHEM": {"hit_rate": 0.0, "search_rate": 20000.0, "annotations": 0},
            "METLIN": {"hit_rate": 0.0, "search_rate": 20000.0, "annotations": 0},
            "MASSBANK": {"hit_rate": 0.0, "search_rate": 20000.0, "annotations": 0},
            "MZCLOUD": {"hit_rate": 0.0, "search_rate": 20000.0, "annotations": 0},
            "KEGG": {"hit_rate": 0.0, "search_rate": 20000.0, "annotations": 0},
            "HUMANCYC": {"hit_rate": 0.0, "search_rate": 18791.68, "annotations": 0},
        }
    },
    "TG_Pos_Thermo_Orbi.mzML": {
        "spectra_searched": 20,
        "annotation_rate": 0.0,
        "search_rate": 5972.24,
        "databases": {
            "LIPIDMAPS": {"hit_rate": 0.0, "search_rate": 15543.09, "annotations": 0},
            "MSLIPIDS": {"hit_rate": 0.0, "search_rate": 16175.49, "annotations": 0},
            "PUBCHEM": {"hit_rate": 0.0, "search_rate": 20000.0, "annotations": 0},
            "METLIN": {"hit_rate": 0.0, "search_rate": 20000.0, "annotations": 0},
            "MASSBANK": {"hit_rate": 0.0, "search_rate": 20000.0, "annotations": 0},
            "MZCLOUD": {"hit_rate": 0.0, "search_rate": 20000.0, "annotations": 0},
            "KEGG": {"hit_rate": 0.0, "search_rate": 17225.07, "annotations": 0},
            "HUMANCYC": {"hit_rate": 0.0, "search_rate": 20000.0, "annotations": 0},
        }
    }
}


def create_database_search_figure():
    """Create 4-panel database search validation figure"""
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    datasets = list(RESULTS.keys())
    colors = ['#3498DB', '#E74C3C']

    # Panel A: Database search rates comparison
    ax1 = fig.add_subplot(gs[0, 0])

    databases = list(RESULTS[datasets[0]]['databases'].keys())
    x = np.arange(len(databases))
    width = 0.35

    rates1 = [RESULTS[datasets[0]]['databases'][db]['search_rate']/1000 for db in databases]
    rates2 = [RESULTS[datasets[1]]['databases'][db]['search_rate']/1000 for db in databases]

    bars1 = ax1.bar(x - width/2, rates1, width, label=datasets[0].split('.')[0],
                    color=colors[0], alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax1.bar(x + width/2, rates2, width, label=datasets[1].split('.')[0],
                    color=colors[1], alpha=0.8, edgecolor='black', linewidth=1)

    ax1.set_xlabel('Database', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Search Rate (×1000 spec/s)', fontweight='bold', fontsize=12)
    ax1.set_title('A. Database Search Performance', fontweight='bold', fontsize=14, pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(databases, rotation=45, ha='right', fontsize=9)
    ax1.legend(fontsize=10, frameon=True, shadow=True)
    ax1.grid(True, alpha=0.3, axis='y')

    # Panel B: Hit rates by database (heatmap-style)
    ax2 = fig.add_subplot(gs[0, 1])

    # Create synthetic hit rate data for visualization (since actual is 0)
    # Show what the visualization would look like with real data
    hit_matrix = np.array([
        [0.05, 0.03, 0.12, 0.08, 0.15, 0.06, 0.04, 0.02],  # Dataset 1
        [0.04, 0.02, 0.10, 0.07, 0.13, 0.05, 0.03, 0.02],  # Dataset 2
    ]) * 0  # Multiply by 0 since actual hit rates are 0

    im = ax2.imshow(hit_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.2)
    ax2.set_xticks(np.arange(len(databases)))
    ax2.set_yticks(np.arange(len(datasets)))
    ax2.set_xticklabels(databases, rotation=45, ha='right', fontsize=9)
    ax2.set_yticklabels([d.split('.')[0] for d in datasets], fontsize=10)

    # Add text annotations
    for i in range(len(datasets)):
        for j in range(len(databases)):
            text = ax2.text(j, i, f'{hit_matrix[i, j]:.0%}',
                          ha="center", va="center", color="black", fontsize=9)

    ax2.set_title('B. Database Hit Rate Matrix', fontweight='bold', fontsize=14, pad=15)
    plt.colorbar(im, ax=ax2, label='Hit Rate', fraction=0.046)

    # Panel C: Search efficiency metrics
    ax3 = fig.add_subplot(gs[1, 0])

    # Average search rates per database type
    db_categories = {
        'Lipid-Specific': ['LIPIDMAPS', 'MSLIPIDS'],
        'General': ['PUBCHEM', 'METLIN', 'MASSBANK', 'MZCLOUD'],
        'Pathway': ['KEGG', 'HUMANCYC']
    }

    categories = list(db_categories.keys())
    avg_rates1 = []
    avg_rates2 = []

    for cat in categories:
        dbs_in_cat = db_categories[cat]
        avg1 = np.mean([RESULTS[datasets[0]]['databases'][db]['search_rate'] for db in dbs_in_cat])
        avg2 = np.mean([RESULTS[datasets[1]]['databases'][db]['search_rate'] for db in dbs_in_cat])
        avg_rates1.append(avg1 / 1000)
        avg_rates2.append(avg2 / 1000)

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax3.bar(x - width/2, avg_rates1, width, label=datasets[0].split('.')[0],
                    color=colors[0], alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax3.bar(x + width/2, avg_rates2, width, label=datasets[1].split('.')[0],
                    color=colors[1], alpha=0.8, edgecolor='black', linewidth=1.5)

    ax3.set_xlabel('Database Category', fontweight='bold', fontsize=12)
    ax3.set_ylabel('Avg Search Rate (×1000 spec/s)', fontweight='bold', fontsize=12)
    ax3.set_title('C. Search Efficiency by Category', fontweight='bold', fontsize=14, pad=15)
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories, fontsize=10)
    ax3.legend(fontsize=10, frameon=True, shadow=True)
    ax3.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Panel D: Summary statistics table
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    table_data = [
        ['Metric', datasets[0].split('.')[0], datasets[1].split('.')[0]],
        ['', '', ''],
        ['Spectra Searched',
         f"{RESULTS[datasets[0]]['spectra_searched']}",
         f"{RESULTS[datasets[1]]['spectra_searched']}"],
        ['Overall Search Rate',
         f"{RESULTS[datasets[0]]['search_rate']:.0f} spec/s",
         f"{RESULTS[datasets[1]]['search_rate']:.0f} spec/s"],
        ['', '', ''],
        ['Databases Queried', '8', '8'],
        ['Total Annotations', '0', '0'],
        ['Annotation Rate', '0.0%', '0.0%'],
        ['', '', ''],
        ['Fastest Database', 'PUBCHEM', 'PUBCHEM'],
        ['Slowest Database', 'LIPIDMAPS', 'LIPIDMAPS'],
        ['', '', ''],
        ['Search Success Rate', '100%', '100%'],
        ['Coverage Status', '🔴 Low', '🔴 Low'],
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
    for row_idx in [1, 4, 8, 11]:
        for col_idx in range(3):
            table[(row_idx, col_idx)].set_facecolor('#ECF0F1')

    # Highlight low coverage
    table[(13, 0)].set_facecolor('#FADBD8')
    table[(13, 1)].set_facecolor('#FADBD8')
    table[(13, 2)].set_facecolor('#FADBD8')

    ax4.set_title('D. Search Performance Summary', fontweight='bold', fontsize=14, pad=15)

    fig.suptitle('Database Search Validation Results\n8-Database Annotation Performance Analysis',
                fontsize=18, fontweight='bold', y=0.98)

    output_path = Path(__file__).parent / 'database_search_validation.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✓ Database search figure saved: {output_path}")


if __name__ == "__main__":
    print("="*80)
    print("DATABASE SEARCH VALIDATION - 4-PANEL FIGURE")
    print("="*80)
    create_database_search_figure()
    print("\nVisualization complete!")
