#!/usr/bin/env python3
"""
sequence_visualisation.py

Visualizations for sequence S-entropy trajectories.

Author: Kundai Farai Sachikonye (with AI assistance)
Date: December 2025
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
PRECURSOR_ROOT = SCRIPT_DIR.parent.parent
RESULTS_DIR = PRECURSOR_ROOT / "results" / "tests" / "molecular_language"
SEQ_RESULTS_DIR = PRECURSOR_ROOT / "results" / "tests" / "sequence"

# Publication settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.linewidth'] = 1.5


def create_sequence_trajectory_figure(output_dir=None):
    """Create sequence trajectory visualization figure."""

    if output_dir is None:
        output_dir = PRECURSOR_ROOT / 'results' / 'visualizations' / 'sequences'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from: {RESULTS_DIR}")

    # Load data
    seq_paths = pd.read_csv(RESULTS_DIR / 'sequence_sentropy_paths.csv')
    aa_data = pd.read_csv(RESULTS_DIR / 'amino_acid_alphabet.csv')

    # Extract sequences
    peptide_path = seq_paths[seq_paths['sequence'] == 'PEPTIDE'].sort_values('position')
    sample_path = seq_paths[seq_paths['sequence'] == 'SAMPLE'].sort_values('position')

    # Create figure with 2x2 panel layout
    fig = plt.figure(figsize=(12, 10))

    # Panel A: PEPTIDE 3D Trajectory (top-left)
    ax1 = plt.subplot2grid((2, 2), (0, 0), projection='3d')

    # Plot amino acid space (faded)
    ax1.scatter(aa_data['s_knowledge'], aa_data['s_time'], aa_data['s_entropy'],
                c='lightgray', s=30, alpha=0.3)

    # Plot PEPTIDE path
    colors_gradient = plt.cm.viridis(np.linspace(0, 1, len(peptide_path)))
    for i in range(len(peptide_path)-1):
        ax1.plot(peptide_path['s_knowledge'].iloc[i:i+2],
                 peptide_path['s_time'].iloc[i:i+2],
                 peptide_path['s_entropy'].iloc[i:i+2],
                 color='#D55E00', linewidth=2.5, alpha=0.8)

    # Add position markers
    for idx, row in peptide_path.iterrows():
        ax1.scatter(row['s_knowledge'], row['s_time'], row['s_entropy'],
                    c=[colors_gradient[row['position']]], s=120, edgecolors='black', linewidths=1.5)
        ax1.text(row['s_knowledge'], row['s_time'], row['s_entropy'],
                 row['amino_acid'], fontsize=7, ha='center', va='bottom', fontweight='bold')

    ax1.set_xlabel('Sk', fontsize=9, fontweight='bold', labelpad=3)
    ax1.set_ylabel('St', fontsize=9, fontweight='bold', labelpad=3)
    ax1.set_zlabel('Se', fontsize=9, fontweight='bold', labelpad=3)
    ax1.set_title('A', fontsize=12, fontweight='bold', loc='left', pad=10)
    ax1.view_init(elev=25, azim=45)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=7)

    # Panel B: SAMPLE 3D Trajectory (top-right)
    ax2 = plt.subplot2grid((2, 2), (0, 1), projection='3d')

    # Plot amino acid space (faded)
    ax2.scatter(aa_data['s_knowledge'], aa_data['s_time'], aa_data['s_entropy'],
                c='lightgray', s=30, alpha=0.3)

    # Plot SAMPLE path
    colors_gradient2 = plt.cm.plasma(np.linspace(0, 1, len(sample_path)))
    for i in range(len(sample_path)-1):
        ax2.plot(sample_path['s_knowledge'].iloc[i:i+2],
                 sample_path['s_time'].iloc[i:i+2],
                 sample_path['s_entropy'].iloc[i:i+2],
                 color='#0072B2', linewidth=2.5, alpha=0.8)

    # Add position markers
    for idx, row in sample_path.iterrows():
        ax2.scatter(row['s_knowledge'], row['s_time'], row['s_entropy'],
                    c=[colors_gradient2[row['position']]], s=120, edgecolors='black', linewidths=1.5)
        ax2.text(row['s_knowledge'], row['s_time'], row['s_entropy'],
                 row['amino_acid'], fontsize=7, ha='center', va='bottom', fontweight='bold')

    ax2.set_xlabel('Sk', fontsize=9, fontweight='bold', labelpad=3)
    ax2.set_ylabel('St', fontsize=9, fontweight='bold', labelpad=3)
    ax2.set_zlabel('Se', fontsize=9, fontweight='bold', labelpad=3)
    ax2.set_title('B', fontsize=12, fontweight='bold', loc='left', pad=10)
    ax2.view_init(elev=25, azim=45)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=7)

    # Panel C: Path Metrics Comparison (bottom-left)
    ax3 = plt.subplot2grid((2, 2), (1, 0))

    metrics = pd.DataFrame({
        'Sequence': ['PEPTIDE', 'SAMPLE'],
        'Entropy': [peptide_path['sequence_entropy'].iloc[0], sample_path['sequence_entropy'].iloc[0]],
        'Complexity': [peptide_path['sequence_complexity'].iloc[0], sample_path['sequence_complexity'].iloc[0]]
    })

    x = np.arange(len(metrics['Sequence']))
    width = 0.35

    bars1 = ax3.bar(x - width/2, metrics['Entropy'], width, label='Sequence Entropy',
                    color='#E69F00', edgecolor='black', linewidth=1.2, alpha=0.8)
    bars2 = ax3.bar(x + width/2, metrics['Complexity']*5, width, label='Complexity (×5)',
                    color='#56B4E9', edgecolor='black', linewidth=1.2, alpha=0.8)

    ax3.set_ylabel('Value', fontsize=9, fontweight='bold')
    ax3.set_title('C', fontsize=12, fontweight='bold', loc='left')
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics['Sequence'], fontsize=8)
    ax3.legend(fontsize=7, loc='upper left')
    ax3.grid(True, axis='y', alpha=0.3)
    ax3.tick_params(labelsize=7)

    # Add values on bars
    for bar in bars1:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                 f'{height:.2f}', ha='center', va='bottom', fontsize=7, fontweight='bold')
    for bar in bars2:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                 f'{height/5:.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

    # Panel D: Position-by-Position Entropy (bottom-right)
    ax4 = plt.subplot2grid((2, 2), (1, 1))

    # Calculate S-entropy magnitude at each position
    peptide_path = peptide_path.copy()
    sample_path = sample_path.copy()
    peptide_path['s_magnitude'] = np.sqrt(peptide_path['s_knowledge']**2 +
                                           peptide_path['s_time']**2 +
                                           peptide_path['s_entropy']**2)
    sample_path['s_magnitude'] = np.sqrt(sample_path['s_knowledge']**2 +
                                          sample_path['s_time']**2 +
                                          sample_path['s_entropy']**2)

    ax4.plot(peptide_path['position'], peptide_path['s_magnitude'],
             'o-', linewidth=2, markersize=7, color='#D55E00', label='PEPTIDE', alpha=0.8)
    ax4.plot(sample_path['position'], sample_path['s_magnitude'],
             's-', linewidth=2, markersize=7, color='#0072B2', label='SAMPLE', alpha=0.8)

    # Add amino acid labels
    for idx, row in peptide_path.iterrows():
        ax4.text(row['position'], row['s_magnitude'] + 0.02, row['amino_acid'],
                 fontsize=7, ha='center', va='bottom', fontweight='bold', color='#D55E00')
    for idx, row in sample_path.iterrows():
        ax4.text(row['position'], row['s_magnitude'] - 0.02, row['amino_acid'],
                 fontsize=7, ha='center', va='top', fontweight='bold', color='#0072B2')

    ax4.set_xlabel('Sequence Position', fontsize=9, fontweight='bold')
    ax4.set_ylabel('S-Entropy Magnitude', fontsize=9, fontweight='bold')
    ax4.set_title('D', fontsize=12, fontweight='bold', loc='left')
    ax4.legend(fontsize=7, loc='upper left')
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(labelsize=7)

    plt.tight_layout()

    output_png = output_dir / 'Figure2_Sequence_Trajectories.png'
    output_pdf = output_dir / 'Figure2_Sequence_Trajectories.pdf'
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_pdf, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Figure 2 saved:")
    print(f"   - {output_png}")
    print(f"   - {output_pdf}")


def main():
    """Main execution."""
    print("="*60)
    print("SEQUENCE TRAJECTORY VISUALIZATION")
    print("="*60)

    create_sequence_trajectory_figure()

    print("\n✅ Visualization complete!")


if __name__ == '__main__':
    main()
