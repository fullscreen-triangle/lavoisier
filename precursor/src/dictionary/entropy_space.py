#!/usr/bin/env python3
"""
entropy_space.py

S-Entropy Space Visualizations for Dictionary Module.
Creates Figure 1 (S-Entropy Space), Figure 4 (Fragmentation Grammar),
and Figure 5 (Cross-Modal Validation).

Author: Kundai Farai Sachikonye (with AI assistance)
Date: December 2025
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import seaborn as sns
import json
import sys
import os
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
PRECURSOR_ROOT = SCRIPT_DIR.parent.parent
RESULTS_DIR = PRECURSOR_ROOT / "results" / "tests" / "dictionary"
ML_RESULTS_DIR = PRECURSOR_ROOT / "results" / "tests" / "molecular_language"
MMD_RESULTS_DIR = PRECURSOR_ROOT / "results" / "tests" / "mmd_system"
OUTPUT_DIR = PRECURSOR_ROOT / "results" / "visualizations" / "entropy_space"

# Set publication-quality defaults
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
    """Load amino acid and dictionary data."""
    print(f"Loading data from: {ML_RESULTS_DIR}")
    aa_data = pd.read_csv(ML_RESULTS_DIR / 'amino_acid_alphabet.csv')
    dict_entries = pd.read_csv(RESULTS_DIR / 'dictionary_entries.csv')
    print(f"✅ Loaded {len(aa_data)} amino acids")
    print(f"✅ Loaded {len(dict_entries)} dictionary entries")
    return aa_data, dict_entries


def load_spectra_data():
    """Load spectra and fragmentation data."""
    print(f"Loading spectra from: {MMD_RESULTS_DIR}")
    spectra = pd.read_csv(MMD_RESULTS_DIR / 'spectra_data_20251203_003550.csv')
    frag_grammar = pd.read_csv(ML_RESULTS_DIR / 'fragmentation_grammar.csv')
    mmd_analysis = pd.read_csv(MMD_RESULTS_DIR / 'mmd_analysis_20251203_003550.csv')

    print(f"✅ Loaded {len(spectra)} spectral peaks")
    print(f"✅ Loaded {len(frag_grammar)} theoretical fragments")
    print(f"✅ Loaded {len(mmd_analysis)} MMD analysis results")

    return spectra, frag_grammar, mmd_analysis


def load_frag_data():
    """Load fragmentation grammar data."""
    print(f"Loading fragmentation grammar from: {ML_RESULTS_DIR}")
    frag_grammar = pd.read_csv(ML_RESULTS_DIR / 'fragmentation_grammar.csv')
    print(f"✅ Loaded {len(frag_grammar)} fragmentation rules")
    return frag_grammar


def create_figure1(aa_data, output_dir=None):
    """Create Figure 1: S-Entropy Coordinate Space Architecture."""

    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Classify amino acids
    aa_data = aa_data.copy()
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

    output_png = output_dir / 'Figure1_SEntropy_Space.png'
    output_pdf = output_dir / 'Figure1_SEntropy_Space.pdf'
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_pdf, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Figure 1 saved:")
    print(f"   - {output_png}")
    print(f"   - {output_pdf}")


def create_figure4(frag_grammar, output_dir=None):
    """Create Figure 4: Molecular Fragmentation Grammar & Theoretical Ladders."""

    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create figure with 2x2 panel layout
    fig = plt.figure(figsize=(12, 10))

    # Panel A: b-Ion Ladder for PEPTIDE (top-left)
    ax1 = plt.subplot2grid((2, 2), (0, 0))

    peptide_frags = frag_grammar[frag_grammar['parent_sequence'] == 'PEPTIDE']
    peptide_b = peptide_frags[peptide_frags['ion_type'] == 'b']

    if len(peptide_b) > 0:
        peptide_b_main = peptide_b[peptide_b['neutral_loss'].isna()]
        peptide_b_h2o = peptide_b[peptide_b['neutral_loss'] == 'H2O']

        ax1.plot(peptide_b_main['position'], peptide_b_main['mz'],
                 'o-', linewidth=2.5, markersize=9, color='#D55E00',
                 label='b-ions', alpha=0.8)
        if len(peptide_b_h2o) > 0:
            ax1.plot(peptide_b_h2o['position'], peptide_b_h2o['mz'],
                     's--', linewidth=1.5, markersize=7, color='#0072B2',
                     label='b-ions - H₂O', alpha=0.7)

        # Add fragment labels
        for _, row in peptide_b_main.iterrows():
            ax1.text(row['position'], row['mz'] + 8,
                     f"b{row['position']}\n{row['mz']:.1f}",
                     ha='center', va='bottom', fontsize=6, fontweight='bold')

        ax1.set_xlim(0.5, peptide_b_main['position'].max() + 0.5)
    else:
        ax1.text(0.5, 0.5, 'No PEPTIDE data available',
                 ha='center', va='center', transform=ax1.transAxes)

    ax1.set_xlabel('Fragment Position', fontsize=9, fontweight='bold')
    ax1.set_ylabel('m/z', fontsize=9, fontweight='bold')
    ax1.set_title('A', fontsize=12, fontweight='bold', loc='left')
    ax1.legend(fontsize=7, loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=7)

    # Panel B: b-Ion Ladder for SEQUENCE (top-right)
    ax2 = plt.subplot2grid((2, 2), (0, 1))

    sequence_frags = frag_grammar[frag_grammar['parent_sequence'] == 'SEQUENCE']
    sequence_b = sequence_frags[sequence_frags['ion_type'] == 'b']

    if len(sequence_b) > 0:
        sequence_b_main = sequence_b[sequence_b['neutral_loss'].isna()]
        sequence_b_h2o = sequence_b[sequence_b['neutral_loss'] == 'H2O']

        ax2.plot(sequence_b_main['position'], sequence_b_main['mz'],
                 'o-', linewidth=2.5, markersize=9, color='#009E73',
                 label='b-ions', alpha=0.8)
        if len(sequence_b_h2o) > 0:
            ax2.plot(sequence_b_h2o['position'], sequence_b_h2o['mz'],
                     's--', linewidth=1.5, markersize=7, color='#CC79A7',
                     label='b-ions - H₂O', alpha=0.7)

        # Add fragment labels
        for _, row in sequence_b_main.iterrows():
            ax2.text(row['position'], row['mz'] + 10,
                     f"b{row['position']}\n{row['mz']:.1f}",
                     ha='center', va='bottom', fontsize=6, fontweight='bold')

        ax2.set_xlim(0.5, sequence_b_main['position'].max() + 0.5)
    else:
        ax2.text(0.5, 0.5, 'No SEQUENCE data available',
                 ha='center', va='center', transform=ax2.transAxes)

    ax2.set_xlabel('Fragment Position', fontsize=9, fontweight='bold')
    ax2.set_ylabel('m/z', fontsize=9, fontweight='bold')
    ax2.set_title('B', fontsize=12, fontweight='bold', loc='left')
    ax2.legend(fontsize=7, loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=7)

    # Panel C: Complementarity (b/y ions for PEPTIDE) (bottom-left)
    ax3 = plt.subplot2grid((2, 2), (1, 0))

    peptide_y = peptide_frags[peptide_frags['ion_type'] == 'y']

    if len(peptide_b) > 0 and len(peptide_y) > 0:
        peptide_b_main = peptide_b[peptide_b['neutral_loss'].isna()]
        peptide_y_main = peptide_y[peptide_y['neutral_loss'].isna()]

        # Mirror plot
        ax3.bar(peptide_b_main['position'], peptide_b_main['mz'],
                color='#D55E00', alpha=0.7, edgecolor='black', linewidth=1.2, label='b-ions')
        ax3.bar(peptide_y_main['position'], -peptide_y_main['mz'],
                color='#0072B2', alpha=0.7, edgecolor='black', linewidth=1.2, label='y-ions')

        ax3.axhline(y=0, color='black', linewidth=2)
    else:
        ax3.text(0.5, 0.5, 'No complementary ion data',
                 ha='center', va='center', transform=ax3.transAxes)

    ax3.set_xlabel('Fragment Position', fontsize=9, fontweight='bold')
    ax3.set_ylabel('m/z (b +, y -)', fontsize=9, fontweight='bold')
    ax3.set_title('C', fontsize=12, fontweight='bold', loc='left')
    ax3.legend(fontsize=7, loc='upper right')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.tick_params(labelsize=7)

    # Panel D: Grammar Production Rules (bottom-right)
    ax4 = plt.subplot2grid((2, 2), (1, 1))
    ax4.axis('off')

    # Precursor
    precursor_box = FancyBboxPatch((0.15, 0.75), 0.7, 0.18,
                                    boxstyle="round,pad=0.02",
                                    edgecolor='black', facecolor='#E69F00',
                                    linewidth=2, alpha=0.8)
    ax4.add_patch(precursor_box)
    ax4.text(0.5, 0.84, 'Precursor: PEPTIDE', ha='center', va='center',
             fontsize=9, fontweight='bold')

    # Arrow 1
    arrow1 = FancyArrowPatch((0.5, 0.75), (0.5, 0.60),
                              arrowstyle='->', mutation_scale=25,
                              linewidth=2, color='black')
    ax4.add_patch(arrow1)
    ax4.text(0.55, 0.675, 'Fragment', ha='left', va='center',
             fontsize=7, style='italic')

    # b1 + y6
    b1_box = FancyBboxPatch((0.05, 0.45), 0.35, 0.12,
                             boxstyle="round,pad=0.02",
                             edgecolor='black', facecolor='#D55E00',
                             linewidth=1.5, alpha=0.8)
    ax4.add_patch(b1_box)
    ax4.text(0.225, 0.51, 'b₁ (P)', ha='center', va='center',
             fontsize=8, fontweight='bold')

    y6_box = FancyBboxPatch((0.6, 0.45), 0.35, 0.12,
                             boxstyle="round,pad=0.02",
                             edgecolor='black', facecolor='#0072B2',
                             linewidth=1.5, alpha=0.8)
    ax4.add_patch(y6_box)
    ax4.text(0.775, 0.51, 'y₆ (EPTIDE)', ha='center', va='center',
             fontsize=8, fontweight='bold')

    # Arrow 2
    arrow2 = FancyArrowPatch((0.225, 0.45), (0.225, 0.30),
                              arrowstyle='->', mutation_scale=25,
                              linewidth=2, color='black')
    ax4.add_patch(arrow2)
    ax4.text(0.27, 0.375, '+E', ha='left', va='center',
             fontsize=7, style='italic')

    # b2
    b2_box = FancyBboxPatch((0.05, 0.15), 0.35, 0.12,
                             boxstyle="round,pad=0.02",
                             edgecolor='black', facecolor='#D55E00',
                             linewidth=1.5, alpha=0.8)
    ax4.add_patch(b2_box)
    ax4.text(0.225, 0.21, 'b₂ (PE)', ha='center', va='center',
             fontsize=8, fontweight='bold')

    # Sequential rule
    ax4.text(0.5, 0.02, 'Rule: bᵢ₊₁ = bᵢ + AAᵢ₊₁',
             ha='center', va='center', fontsize=8,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontweight='bold')

    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.set_title('D', fontsize=12, fontweight='bold', loc='left', pad=10)

    plt.tight_layout()

    output_png = output_dir / 'Figure4_Fragmentation_Grammar.png'
    output_pdf = output_dir / 'Figure4_Fragmentation_Grammar.pdf'
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_pdf, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Figure 4 saved:")
    print(f"   - {output_png}")
    print(f"   - {output_pdf}")


def match_peaks(observed_mz, theoretical_mz, tolerance_ppm=20):
    """Match observed peaks to theoretical fragments."""
    matches = []
    mass_errors = []

    for theo_mz in theoretical_mz:
        tolerance_da = theo_mz * tolerance_ppm / 1e6

        # Find closest observed peak within tolerance
        diffs = np.abs(observed_mz - theo_mz)
        min_idx = np.argmin(diffs)
        min_diff = diffs[min_idx]

        if min_diff <= tolerance_da:
            matches.append((theo_mz, observed_mz[min_idx]))
            mass_error_ppm = (observed_mz[min_idx] - theo_mz) / theo_mz * 1e6
            mass_errors.append(mass_error_ppm)

    return matches, mass_errors


def create_figure5(spectra, frag_grammar, mmd_analysis, output_dir=None):
    """Create Figure 5: Cross-Modal Validation with Real Spectra."""

    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get first scan with good data
    scan_ids = spectra['scan_id'].unique()
    selected_scan = scan_ids[0] if len(scan_ids) > 0 else 1

    scan_data = spectra[spectra['scan_id'] == selected_scan].sort_values('mz')

    print(f"📊 Analyzing scan {selected_scan} with {len(scan_data)} peaks")

    # Get theoretical fragments for PEPTIDE
    peptide_frags = frag_grammar[frag_grammar['parent_sequence'] == 'PEPTIDE']
    peptide_b = peptide_frags[peptide_frags['ion_type'] == 'b']
    peptide_b_main = peptide_b[peptide_b['neutral_loss'].isna()]

    # Create figure with 2x2 panel layout
    fig = plt.figure(figsize=(12, 10))

    # Panel A: Mirror Plot - Theoretical vs Observed (top)
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)

    # Plot observed spectrum (top, positive)
    ax1.vlines(scan_data['mz'], 0, scan_data['intensity'] / scan_data['intensity'].max(),
               colors='#0072B2', linewidth=1.5, alpha=0.7, label='Observed')

    # Plot theoretical spectrum (bottom, negative)
    if len(peptide_b_main) > 0:
        theo_intensities = np.ones(len(peptide_b_main)) * 0.5
        ax1.vlines(peptide_b_main['mz'], 0, -theo_intensities,
                   colors='#D55E00', linewidth=1.5, alpha=0.7, label='Theoretical (PEPTIDE)')

        # Match peaks
        matches, mass_errors = match_peaks(scan_data['mz'].values, peptide_b_main['mz'].values)

        # Highlight matched peaks
        for theo_mz, obs_mz in matches:
            ax1.plot([obs_mz, obs_mz], [0, 0.05], 'g-', linewidth=2, alpha=0.5)
            ax1.scatter([obs_mz], [0], c='green', s=50, marker='o', zorder=5)

    ax1.axhline(y=0, color='black', linewidth=2)
    ax1.set_xlabel('m/z', fontsize=9, fontweight='bold')
    ax1.set_ylabel('Relative Intensity', fontsize=9, fontweight='bold')
    ax1.set_title('A', fontsize=12, fontweight='bold', loc='left')
    ax1.legend(fontsize=7, loc='upper right')
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.tick_params(labelsize=7)
    ax1.set_ylim(-0.6, 1.1)

    # Panel B: Mass Error Distribution (bottom-left)
    ax2 = plt.subplot2grid((2, 2), (1, 0))

    if len(peptide_b_main) > 0:
        matches, mass_errors = match_peaks(scan_data['mz'].values, peptide_b_main['mz'].values)

        if len(mass_errors) > 0:
            ax2.hist(mass_errors, bins=15, color='#009E73',
                     edgecolor='black', linewidth=1.2, alpha=0.7)
            mean_error = np.mean(mass_errors)
            std_error = np.std(mass_errors)
            ax2.axvline(mean_error, color='red', linestyle='--', linewidth=2,
                        label=f'Mean: {mean_error:.2f} ppm')
            ax2.axvline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)

            ax2.text(0.05, 0.95, f'σ = {std_error:.2f} ppm\nn = {len(mass_errors)}',
                     transform=ax2.transAxes, fontsize=7, va='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax2.text(0.5, 0.5, 'No matches found', ha='center', va='center',
                     transform=ax2.transAxes, fontsize=9)
    else:
        ax2.text(0.5, 0.5, 'No theoretical data', ha='center', va='center',
                 transform=ax2.transAxes, fontsize=9)

    ax2.set_xlabel('Mass Error (ppm)', fontsize=9, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=9, fontweight='bold')
    ax2.set_title('B', fontsize=12, fontweight='bold', loc='left')
    ax2.legend(fontsize=7)
    ax2.grid(True, axis='y', alpha=0.3)
    ax2.tick_params(labelsize=7)

    # Panel C: Fragment Coverage (bottom-right)
    ax3 = plt.subplot2grid((2, 2), (1, 1))

    if len(peptide_b_main) > 0:
        matches, mass_errors = match_peaks(scan_data['mz'].values, peptide_b_main['mz'].values)

        coverage_pct = (len(matches) / len(peptide_b_main)) * 100

        # Bar chart showing coverage
        categories = ['Matched', 'Unmatched']
        values = [len(matches), len(peptide_b_main) - len(matches)]
        colors = ['#009E73', '#D55E00']

        bars = ax3.bar(categories, values, color=colors, edgecolor='black',
                       linewidth=1.2, alpha=0.8)

        # Add percentage labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            pct = (val / len(peptide_b_main)) * 100
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                     f'{val}\n({pct:.1f}%)', ha='center', va='bottom',
                     fontsize=8, fontweight='bold')

        ax3.text(0.5, 0.95, f'Total Coverage: {coverage_pct:.1f}%',
                 transform=ax3.transAxes, ha='center', va='top', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
                 fontweight='bold')
    else:
        ax3.text(0.5, 0.5, 'No theoretical data', ha='center', va='center',
                 transform=ax3.transAxes, fontsize=9)

    ax3.set_ylabel('Number of Fragments', fontsize=9, fontweight='bold')
    ax3.set_title('C', fontsize=12, fontweight='bold', loc='left')
    ax3.grid(True, axis='y', alpha=0.3)
    ax3.tick_params(labelsize=7)

    plt.tight_layout()

    output_png = output_dir / 'Figure5_CrossModal_Validation.png'
    output_pdf = output_dir / 'Figure5_CrossModal_Validation.pdf'
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_pdf, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Figure 5 saved:")
    print(f"   - {output_png}")
    print(f"   - {output_pdf}")


def main():
    """Main execution function - generates all figures."""
    print("=" * 60)
    print("S-ENTROPY SPACE VISUALIZATION SUITE")
    print("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    # Figure 1: S-Entropy Coordinate Space
    print("\n" + "-" * 60)
    print("Figure 1: S-Entropy Coordinate Space Architecture")
    print("-" * 60)
    try:
        aa_data, dict_entries = load_aa_data()
        create_figure1(aa_data)
    except Exception as e:
        print(f"⚠ Could not generate Figure 1: {e}")

    # Figure 4: Fragmentation Grammar
    print("\n" + "-" * 60)
    print("Figure 4: Molecular Fragmentation Grammar")
    print("-" * 60)
    try:
        frag_grammar = load_frag_data()
        create_figure4(frag_grammar)
    except Exception as e:
        print(f"⚠ Could not generate Figure 4: {e}")

    # Figure 5: Cross-Modal Validation
    print("\n" + "-" * 60)
    print("Figure 5: Cross-Modal Validation with Real Spectra")
    print("-" * 60)
    try:
        spectra, frag_grammar, mmd_analysis = load_spectra_data()
        create_figure5(spectra, frag_grammar, mmd_analysis)
    except Exception as e:
        print(f"⚠ Could not generate Figure 5: {e}")

    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)
    print(f"\nGenerated figures saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
