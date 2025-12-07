#!/usr/bin/env python3
"""
Figure 2: Sequence Trajectories Through S-Entropy Space
Creates a 4-panel figure showing peptide sequence paths through S-Entropy coordinates.
"""

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
PRECURSOR_ROOT = SCRIPT_DIR.parent.parent
RESULTS_DIR = PRECURSOR_ROOT / "results" / "tests" / "dictionary"
ML_RESULTS_DIR = PRECURSOR_ROOT / "results" / "tests" / "molecular_language"
OUTPUT_DIR = PRECURSOR_ROOT / "results" / "visualizations" / "trajectories"

# Publication-quality settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 8
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2


def load_data():
    """Load sequence path and amino acid data."""
    try:
        print(f"Loading data from: {ML_RESULTS_DIR}")
        seq_paths = pd.read_csv(ML_RESULTS_DIR / 'sequence_sentropy_paths.csv')
        aa_data = pd.read_csv(ML_RESULTS_DIR / 'amino_acid_alphabet.csv')
        print(f"✅ Loaded {len(seq_paths)} sequence path points")
        print(f"✅ Loaded {len(aa_data)} amino acids")
        return seq_paths, aa_data
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


def create_figure2(seq_paths, aa_data, output_dir=None):
    """Create Figure 2: Sequence Trajectories Through S-Entropy Space."""

    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    # Save figure
    output_png = os.path.join(output_dir, 'Figure2_Sequence_Trajectories.png')
    output_pdf = os.path.join(output_dir, 'Figure2_Sequence_Trajectories.pdf')
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_pdf, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Figure 2 saved:")
    print(f"   - {output_png}")
    print(f"   - {output_pdf}")


def main():
    """Main execution function."""
    print("=" * 60)
    print("Figure 2: Sequence Trajectories Through S-Entropy Space")
    print("=" * 60)

    # Load data
    seq_paths, aa_data = load_data()

    # Create figure
    create_figure2(seq_paths, aa_data)

    print("\n✅ Figure 2 generation complete!")


if __name__ == "__main__":
    main()



#!/usr/bin/env python3
"""
Figure 1: S-Entropy Coordinate Space Architecture
Creates a 4-panel figure showing the S-Entropy coordinate system and amino acid mapping.
"""

import matplotlib
matplotlib.use('Agg')  # Headless backend for server environments

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import sys
import os

# Publication-quality settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 8
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2

# Color-blind friendly palette
COLORS = {
    'charged': '#D55E00',      # Orange-red
    'polar': '#0072B2',        # Blue
    'hydrophobic': '#009E73',  # Green
    'special': '#F0E442',      # Yellow
    'aromatic': '#CC79A7'      # Purple
}


def classify_aa(row):
    """Classify amino acid by physicochemical property."""
    if row['charge'] != 0:
        return 'charged'
    elif row['polarity']:
        return 'polar'
    elif row['symbol'] in ['F', 'W', 'Y']:
        return 'aromatic'
    elif row['symbol'] in ['G', 'P']:
        return 'special'
    else:
        return 'hydrophobic'


def load_aa_data():
    """Load amino acid data from CSV files."""
    try:
        print(f"Loading data from: {ML_RESULTS_DIR}")
        aa_data = pd.read_csv(ML_RESULTS_DIR / 'amino_acid_alphabet.csv')
        print(f"✅ Loaded {len(aa_data)} amino acids")
        return aa_data
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print(f"Please ensure 'amino_acid_alphabet.csv' is in {ML_RESULTS_DIR}")
        sys.exit(1)


def create_figure1(aa_data, output_dir=None):
    """Create Figure 1: S-Entropy Coordinate Space Architecture."""

    # Create output directory
    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Classify amino acids
    aa_data['category'] = aa_data.apply(classify_aa, axis=1)

    # Create figure with 2x2 panel layout
    fig = plt.figure(figsize=(12, 10))

    # Panel A: 3D S-Entropy Space (top-left)
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=1, rowspan=1, projection='3d')

    for category in ['charged', 'polar', 'hydrophobic', 'aromatic', 'special']:
        mask = aa_data['category'] == category
        subset = aa_data[mask]
        ax1.scatter(subset['s_knowledge'], subset['s_time'], subset['s_entropy'],
                    c=COLORS[category], s=150, alpha=0.8, edgecolors='black',
                    linewidths=1.2, label=category.capitalize())

        # Add amino acid labels
        for _, row in subset.iterrows():
            ax1.text(row['s_knowledge'], row['s_time'], row['s_entropy'],
                     row['symbol'], fontsize=7, ha='center', va='center',
                     fontweight='bold')

    ax1.set_xlabel('Sk (Knowledge)', fontsize=9, fontweight='bold', labelpad=5)
    ax1.set_ylabel('St (Time)', fontsize=9, fontweight='bold', labelpad=5)
    ax1.set_zlabel('Se (Entropy)', fontsize=9, fontweight='bold', labelpad=5)
    ax1.set_title('A', fontsize=12, fontweight='bold', loc='left', pad=10)
    ax1.legend(loc='upper left', fontsize=7, framealpha=0.9)
    ax1.view_init(elev=20, azim=45)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=7)

    # Panel B: Sk vs St projection (top-right)
    ax2 = plt.subplot2grid((2, 2), (0, 1))

    for category in ['charged', 'polar', 'hydrophobic', 'aromatic', 'special']:
        mask = aa_data['category'] == category
        subset = aa_data[mask]
        ax2.scatter(subset['s_knowledge'], subset['s_time'],
                    c=COLORS[category], s=80, alpha=0.8, edgecolors='black', linewidths=1)
        for _, row in subset.iterrows():
            ax2.text(row['s_knowledge'], row['s_time'], row['symbol'],
                     fontsize=6, ha='center', va='center', fontweight='bold')

    ax2.set_xlabel('Sk (Knowledge)', fontsize=9, fontweight='bold')
    ax2.set_ylabel('St (Time)', fontsize=9, fontweight='bold')
    ax2.set_title('B', fontsize=12, fontweight='bold', loc='left')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.1, 1.1)
    ax2.set_ylim(-0.1, 0.7)
    ax2.tick_params(labelsize=7)

    # Panel C: Sk vs Se projection (bottom-left)
    ax3 = plt.subplot2grid((2, 2), (1, 0))

    for category in ['charged', 'polar', 'hydrophobic', 'aromatic', 'special']:
        mask = aa_data['category'] == category
        subset = aa_data[mask]
        ax3.scatter(subset['s_knowledge'], subset['s_entropy'],
                    c=COLORS[category], s=80, alpha=0.8, edgecolors='black', linewidths=1)
        for _, row in subset.iterrows():
            ax3.text(row['s_knowledge'], row['s_entropy'], row['symbol'],
                     fontsize=6, ha='center', va='center', fontweight='bold')

    ax3.set_xlabel('Sk (Knowledge)', fontsize=9, fontweight='bold')
    ax3.set_ylabel('Se (Entropy)', fontsize=9, fontweight='bold')
    ax3.set_title('C', fontsize=12, fontweight='bold', loc='left')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-0.1, 1.1)
    ax3.set_ylim(-0.1, 1.1)
    ax3.tick_params(labelsize=7)

    # Panel D: Property Correlations (bottom-right)
    ax4 = plt.subplot2grid((2, 2), (1, 1))

    # Calculate correlations
    corr_hydro = np.corrcoef(aa_data['hydrophobicity'], aa_data['s_knowledge'])[0,1]
    corr_vol = np.corrcoef(aa_data['volume'], aa_data['s_time'])[0,1]
    corr_charge = np.corrcoef(aa_data['charge'].abs(), aa_data['s_entropy'])[0,1]

    corr_data = pd.DataFrame({
        'Property': ['Hydrophobicity\n→ Sk', 'Volume\n→ St', 'Charge\n→ Se'],
        'Correlation': [corr_hydro, corr_vol, corr_charge]
    })

    bars = ax4.bar(corr_data['Property'], corr_data['Correlation'],
                   color=['#E69F00', '#56B4E9', '#CC79A7'],
                   edgecolor='black', linewidth=1.2, alpha=0.8)
    ax4.set_ylabel('Pearson Correlation (r)', fontsize=9, fontweight='bold')
    ax4.set_title('D', fontsize=12, fontweight='bold', loc='left')
    ax4.set_ylim(0, 1)
    ax4.grid(True, axis='y', alpha=0.3)
    ax4.axhline(y=0.7, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
    ax4.tick_params(labelsize=7)

    # Add correlation values on bars
    for bar, val in zip(bars, corr_data['Correlation']):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()

    # Save figure
    output_png = os.path.join(output_dir, 'Figure1_SEntropy_Space.png')
    output_pdf = os.path.join(output_dir, 'Figure1_SEntropy_Space.pdf')
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_pdf, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Figure 1 saved:")
    print(f"   - {output_png}")
    print(f"   - {output_pdf}")


def main():
    """Main execution function."""
    print("=" * 60)
    print("Figure 1: S-Entropy Coordinate Space Architecture")
    print("=" * 60)

    # Load data
    aa_data = load_aa_data()

    # Create figure
    create_figure1(aa_data)

    print("\n✅ Figure 1 generation complete!")

    # Also create Figure 2
    print("\n" + "=" * 60)
    print("Figure 2: Sequence Trajectories Through S-Entropy Space")
    print("=" * 60)

    seq_paths, aa_data = load_data()
    create_figure2(seq_paths, aa_data)

    print("\n✅ Figure 2 generation complete!")


if __name__ == "__main__":
    main()
