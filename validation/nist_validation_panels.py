#!/usr/bin/env python3
"""
Partition Framework Validation — 6-Panel Visualization
Each panel: 4 charts in a row, at least one 3D chart, minimal text, no tables.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import matplotlib.cm as cm


# ============================================================================
# Styling
# ============================================================================

plt.rcParams.update({
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 7,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'axes.linewidth': 0.6,
    'lines.linewidth': 0.8,
    'patch.linewidth': 0.5,
})

COLORS = {
    'primary': '#2563eb',
    'secondary': '#7c3aed',
    'accent': '#059669',
    'warm': '#dc2626',
    'orange': '#ea580c',
    'gold': '#d97706',
    'pass': '#16a34a',
    'fail': '#dc2626',
    'bg': '#f8fafc',
}

results_dir = Path(r'c:\Users\kundai\Documents\bioinformatics\lavoisier\validation\experiment_results')
figures_dir = results_dir / 'figures'
figures_dir.mkdir(parents=True, exist_ok=True)


def load_json(name):
    with open(results_dir / name) as f:
        return json.load(f)


# ============================================================================
# Panel 1: Spike Protein MS/MS — Partition Coordinates & Validation
# ============================================================================

def panel_1_spike_protein():
    data = load_json('spike_protein_detailed_validation.json')
    spectra = data['spectra']

    fig = plt.figure(figsize=(16, 3.8))
    fig.patch.set_facecolor('white')
    gs = GridSpec(1, 4, figure=fig, wspace=0.32, left=0.04, right=0.97, top=0.85, bottom=0.15)

    # --- Chart A: 3D partition coordinates (n, l, m) colored by validation score ---
    ax1 = fig.add_subplot(gs[0], projection='3d')
    n_vals = [s['partition_coords']['n'] for s in spectra]
    l_vals = [s['partition_coords']['l'] for s in spectra]
    m_vals = [s['partition_coords']['m'] for s in spectra]
    scores = [s['validation_score'] for s in spectra]

    sc = ax1.scatter(n_vals, l_vals, m_vals, c=scores, cmap='RdYlGn', s=50,
                     edgecolors='k', linewidths=0.3, vmin=0.85, vmax=1.0, alpha=0.9)
    ax1.set_xlabel(r'$n$')
    ax1.set_ylabel(r'$\ell$')
    ax1.set_zlabel(r'$m$')
    ax1.set_title('A', loc='left', fontweight='bold', fontsize=10)
    ax1.view_init(elev=25, azim=40)
    plt.colorbar(sc, ax=ax1, shrink=0.5, pad=0.1, label='Score')

    # --- Chart B: S-entropy coordinates scatter (Sk vs Se) colored by instrument ---
    ax2 = fig.add_subplot(gs[1])
    for s in spectra:
        c = COLORS['primary'] if s['instrument'] == 'HCD' else COLORS['warm']
        mk = 'o' if s['instrument'] == 'HCD' else 's'
        ax2.scatter(s['sentropy_coords']['sk'], s['sentropy_coords']['se'],
                    c=c, marker=mk, s=35, edgecolors='k', linewidths=0.3, alpha=0.8)

    ax2.set_xlabel(r'$S_k$')
    ax2.set_ylabel(r'$S_e$')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    # Legend
    ax2.scatter([], [], c=COLORS['primary'], marker='o', s=25, label='HCD')
    ax2.scatter([], [], c=COLORS['warm'], marker='s', s=25, label='IT-FT')
    ax2.legend(fontsize=5, loc='lower right', framealpha=0.7)
    ax2.set_title('B', loc='left', fontweight='bold', fontsize=10)

    # --- Chart C: Partition depth vs residue ratio ---
    ax3 = fig.add_subplot(gs[2])
    M_vals = [s['partition_depth']['M'] for s in spectra]
    R_vals = [s['partition_depth']['residue_ratio'] for s in spectra]
    ax3.scatter(M_vals, R_vals, c=[COLORS['secondary']] * len(spectra),
                s=40, edgecolors='k', linewidths=0.3, alpha=0.8)
    ax3.set_xlabel(r'Partition Depth $\mathcal{M}$')
    ax3.set_ylabel('Residue Ratio')
    ax3.set_title('C', loc='left', fontweight='bold', fontsize=10)

    # --- Chart D: Dispersion residual (log scale) ---
    ax4 = fig.add_subplot(gs[3])
    resids = [s['dispersion']['dispersion_residual'] for s in spectra]
    mzs = [s['precursor_mz'] for s in spectra]
    colors_d = [COLORS['pass'] if s['dispersion']['conforms'] else COLORS['fail'] for s in spectra]
    ax4.scatter(mzs, resids, c=colors_d, s=40, edgecolors='k', linewidths=0.3, alpha=0.8)
    ax4.set_yscale('log')
    ax4.axhline(1e-6, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax4.set_xlabel(r'Precursor $m/z$')
    ax4.set_ylabel(r'$|\omega^2 - \omega_0^2 - c^2k^2|/\omega^2$')
    ax4.set_title('D', loc='left', fontweight='bold', fontsize=10)

    fig.suptitle('SARS-CoV-2 Spike Protein Glycopeptide Validation', fontsize=11, fontweight='bold', y=0.97)
    fig.savefig(figures_dir / 'panel_1_spike_protein.png', bbox_inches='tight')
    plt.close(fig)
    print('  Panel 1 saved.')


# ============================================================================
# Panel 2: Source Libraries A–K — Cross-Lab Partition Consistency
# ============================================================================

def panel_2_source_libraries():
    data = load_json('nist_spike_igg_validation_results.json')
    entries = data['datasets']['spike_source_libraries']['entries']
    summary = data['datasets']['spike_source_libraries']['summary']

    fig = plt.figure(figsize=(16, 3.8))
    fig.patch.set_facecolor('white')
    gs = GridSpec(1, 4, figure=fig, wspace=0.32, left=0.04, right=0.97, top=0.85, bottom=0.15)

    # Source labels
    source_labels = sorted(set(e['source'] for e in entries))
    source_colors = cm.tab10(np.linspace(0, 1, len(source_labels)))
    source_cmap = {s: source_colors[i] for i, s in enumerate(source_labels)}

    # --- Chart A: 3D S-entropy coordinates colored by source ---
    ax1 = fig.add_subplot(gs[0], projection='3d')
    # Subsample for clarity
    np.random.seed(42)
    idx = np.random.choice(len(entries), min(600, len(entries)), replace=False)
    for i in idx:
        e = entries[i]
        c = source_cmap[e['source']]
        ax1.scatter(e['sentropy_coords']['sk'], e['sentropy_coords']['st'],
                    e['sentropy_coords']['se'], c=[c], s=8, alpha=0.5)

    ax1.set_xlabel(r'$S_k$')
    ax1.set_ylabel(r'$S_t$')
    ax1.set_zlabel(r'$S_e$')
    ax1.set_title('A', loc='left', fontweight='bold', fontsize=10)
    ax1.view_init(elev=20, azim=55)

    # --- Chart B: Entries per source (bar chart) ---
    ax2 = fig.add_subplot(gs[1])
    counts = summary['source_counts']
    labels = [k.replace('Source_', '') for k in sorted(counts.keys())]
    vals = [counts[f'Source_{l}'] for l in labels]
    bars = ax2.bar(labels, vals, color=[source_cmap[f'Source_{l}'] for l in labels],
                   edgecolor='k', linewidth=0.3)
    ax2.set_xlabel('Source Lab')
    ax2.set_ylabel('Entries')
    ax2.set_title('B', loc='left', fontweight='bold', fontsize=10)

    # --- Chart C: m/z distribution by source (violin-like scatter) ---
    ax3 = fig.add_subplot(gs[2])
    for i, src in enumerate(sorted(source_labels)):
        src_entries = [e for e in entries if e['source'] == src]
        mzs = [e['precursor_mz'] for e in src_entries]
        jitter = np.random.normal(0, 0.12, len(mzs))
        ax3.scatter([i + jitter[j] for j in range(len(mzs))], mzs,
                    c=[source_cmap[src]], s=3, alpha=0.3)

    ax3.set_xticks(range(len(source_labels)))
    ax3.set_xticklabels([s.replace('Source_', '') for s in sorted(source_labels)])
    ax3.set_xlabel('Source Lab')
    ax3.set_ylabel(r'Precursor $m/z$')
    ax3.set_title('C', loc='left', fontweight='bold', fontsize=10)

    # --- Chart D: Partition n distribution histogram by source ---
    ax4 = fig.add_subplot(gs[3])
    for src in sorted(source_labels):
        src_entries = [e for e in entries if e['source'] == src]
        n_vals = [e['partition_coords']['n'] for e in src_entries]
        ax4.hist(n_vals, bins=range(1, 10), alpha=0.4, label=src.replace('Source_', ''),
                 color=source_cmap[src], edgecolor='k', linewidth=0.2)

    ax4.set_xlabel(r'$n$ (principal)')
    ax4.set_ylabel('Count')
    ax4.legend(fontsize=4, ncol=3, loc='upper right', framealpha=0.6)
    ax4.set_title('D', loc='left', fontweight='bold', fontsize=10)

    fig.suptitle('Spike Protein Source Libraries (A–K) — Cross-Lab Partition Consistency', fontsize=11, fontweight='bold', y=0.97)
    fig.savefig(figures_dir / 'panel_2_source_libraries.png', bbox_inches='tight')
    plt.close(fig)
    print('  Panel 2 saved.')


# ============================================================================
# Panel 3: SPL Glycopeptide Library — Lactoferrin Glycan Landscape
# ============================================================================

def panel_3_spl_glycopeptides():
    data = load_json('spl_glycopeptide_validation.json')
    entries = data['entries']

    fig = plt.figure(figsize=(16, 3.8))
    fig.patch.set_facecolor('white')
    gs = GridSpec(1, 4, figure=fig, wspace=0.32, left=0.04, right=0.97, top=0.85, bottom=0.15)

    # --- Chart A: 3D partition coordinates colored by glycan mass ---
    ax1 = fig.add_subplot(gs[0], projection='3d')
    n_vals = [e['partition_coords']['n'] for e in entries]
    l_vals = [e['partition_coords']['l'] for e in entries]
    m_vals = [e['partition_coords']['m'] for e in entries]
    masses = [e['glycan_mass'] for e in entries]

    sc = ax1.scatter(n_vals, l_vals, m_vals, c=masses, cmap='viridis', s=60,
                     edgecolors='k', linewidths=0.3, alpha=0.9)
    ax1.set_xlabel(r'$n$')
    ax1.set_ylabel(r'$\ell$')
    ax1.set_zlabel(r'$m$')
    ax1.set_title('A', loc='left', fontweight='bold', fontsize=10)
    ax1.view_init(elev=30, azim=45)
    plt.colorbar(sc, ax=ax1, shrink=0.5, pad=0.1, label=r'$M_{glycan}$ (Da)')

    # --- Chart B: m/z vs glycan mass scatter ---
    ax2 = fig.add_subplot(gs[1])
    charges = [e['charge'] for e in entries]
    colors_ch = [COLORS['primary'] if c == 2 else COLORS['warm'] if c == 3 else COLORS['accent'] for c in charges]
    ax2.scatter([e['precursor_mz'] for e in entries], masses, c=colors_ch,
                s=45, edgecolors='k', linewidths=0.3, alpha=0.8)
    ax2.set_xlabel(r'Precursor $m/z$')
    ax2.set_ylabel(r'Glycan Mass (Da)')
    ax2.scatter([], [], c=COLORS['primary'], s=20, label='z=2')
    ax2.scatter([], [], c=COLORS['warm'], s=20, label='z=3')
    ax2.legend(fontsize=5, loc='upper left', framealpha=0.7)
    ax2.set_title('B', loc='left', fontweight='bold', fontsize=10)

    # --- Chart C: S-entropy Sk vs St colored by score ---
    ax3 = fig.add_subplot(gs[2])
    sc3 = ax3.scatter([e['sentropy_coords']['sk'] for e in entries],
                      [e['sentropy_coords']['st'] for e in entries],
                      c=[e['score'] for e in entries], cmap='plasma', s=50,
                      edgecolors='k', linewidths=0.3, alpha=0.8)
    ax3.set_xlabel(r'$S_k$')
    ax3.set_ylabel(r'$S_t$')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1.1)
    plt.colorbar(sc3, ax=ax3, shrink=0.7, label='Score')
    ax3.set_title('C', loc='left', fontweight='bold', fontsize=10)

    # --- Chart D: Capacity C(n) = 2n² vs precursor m/z ---
    ax4 = fig.add_subplot(gs[3])
    caps = [e['partition_coords']['capacity'] for e in entries]
    mzs = [e['precursor_mz'] for e in entries]
    ax4.scatter(mzs, caps, c=COLORS['secondary'], s=45, edgecolors='k',
                linewidths=0.3, alpha=0.8)
    # Overlay theoretical C(n) = 2n² lines
    for n_line in [2, 3, 4, 5]:
        ax4.axhline(2 * n_line**2, color='gray', linestyle=':', linewidth=0.4, alpha=0.5)
        ax4.text(max(mzs) * 0.98, 2 * n_line**2 + 0.5, f'$n={n_line}$', fontsize=5,
                 ha='right', va='bottom', color='gray')
    ax4.set_xlabel(r'Precursor $m/z$')
    ax4.set_ylabel(r'$C(n) = 2n^2$')
    ax4.set_title('D', loc='left', fontweight='bold', fontsize=10)

    fig.suptitle('NIST Lactoferrin Glycopeptide Library — Partition Landscape', fontsize=11, fontweight='bold', y=0.97)
    fig.savefig(figures_dir / 'panel_3_spl_glycopeptides.png', bbox_inches='tight')
    plt.close(fig)
    print('  Panel 3 saved.')


# ============================================================================
# Panel 4: Existing Glycan MS/MS Library — Comparison Baseline
# ============================================================================

def panel_4_glycan_msms():
    data = load_json('nist_spike_igg_validation_results.json')
    entries = data['datasets']['nist_glycan_msms']['entries']

    fig = plt.figure(figsize=(16, 3.8))
    fig.patch.set_facecolor('white')
    gs = GridSpec(1, 4, figure=fig, wspace=0.32, left=0.04, right=0.97, top=0.85, bottom=0.15)

    # Separate by library
    libs = set(e['library'] for e in entries)
    lib_colors = {lib: c for lib, c in zip(sorted(libs), [COLORS['primary'], COLORS['warm'], COLORS['accent']])}

    # --- Chart A: 3D partition coords (n, l, m) ---
    ax1 = fig.add_subplot(gs[0], projection='3d')
    for e in entries:
        c = lib_colors.get(e['library'], COLORS['primary'])
        ax1.scatter(e['partition_coords']['n'], e['partition_coords']['l'],
                    e['partition_coords']['m'], c=[c], s=70,
                    edgecolors='k', linewidths=0.3, alpha=0.9)
    ax1.set_xlabel(r'$n$')
    ax1.set_ylabel(r'$\ell$')
    ax1.set_zlabel(r'$m$')
    ax1.set_title('A', loc='left', fontweight='bold', fontsize=10)
    ax1.view_init(elev=25, azim=50)

    # --- Chart B: m/z histogram by charge state ---
    ax2 = fig.add_subplot(gs[1])
    charges = set(e['charge'] for e in entries)
    ch_colors = {1: COLORS['accent'], 2: COLORS['primary'], 3: COLORS['warm']}
    for ch in sorted(charges):
        mzs = [e['precursor_mz'] for e in entries if e['charge'] == ch]
        ax2.hist(mzs, bins=12, alpha=0.6, color=ch_colors.get(ch, 'gray'),
                 edgecolor='k', linewidth=0.3, label=f'z={ch}')
    ax2.set_xlabel(r'$m/z$')
    ax2.set_ylabel('Count')
    ax2.legend(fontsize=5, framealpha=0.7)
    ax2.set_title('B', loc='left', fontweight='bold', fontsize=10)

    # --- Chart C: n vs m/z showing mass shell structure ---
    ax3 = fig.add_subplot(gs[2])
    for e in entries:
        c = lib_colors.get(e['library'], COLORS['primary'])
        ax3.scatter(e['precursor_mz'], e['partition_coords']['n'],
                    c=[c], s=55, edgecolors='k', linewidths=0.3, alpha=0.8)
    ax3.set_xlabel(r'Precursor $m/z$')
    ax3.set_ylabel(r'$n$ (principal)')
    ax3.set_title('C', loc='left', fontweight='bold', fontsize=10)

    # --- Chart D: l vs charge (angular momentum vs charge) ---
    ax4 = fig.add_subplot(gs[3])
    for e in entries:
        c = lib_colors.get(e['library'], COLORS['primary'])
        jx = np.random.normal(0, 0.06)
        jy = np.random.normal(0, 0.06)
        ax4.scatter(e['charge'] + jx, e['partition_coords']['l'] + jy,
                    c=[c], s=55, edgecolors='k', linewidths=0.3, alpha=0.8)

    for lib in sorted(libs):
        ax4.scatter([], [], c=lib_colors[lib], s=30, label=lib.replace('_', ' '))
    ax4.legend(fontsize=4, loc='upper left', framealpha=0.7)
    ax4.set_xlabel('Charge State $z$')
    ax4.set_ylabel(r'$\ell$ (angular)')
    ax4.set_title('D', loc='left', fontweight='bold', fontsize=10)

    fig.suptitle('NIST Glycan MS/MS Libraries — Baseline Partition Mapping', fontsize=11, fontweight='bold', y=0.97)
    fig.savefig(figures_dir / 'panel_4_glycan_msms.png', bbox_inches='tight')
    plt.close(fig)
    print('  Panel 4 saved.')


# ============================================================================
# Panel 5: Cross-Dataset Analysis — Framework Universality
# ============================================================================

def panel_5_cross_dataset():
    full = load_json('nist_spike_igg_validation_results.json')
    spike = full['datasets']['sars_cov2_spike_protein']
    source = full['datasets']['spike_source_libraries']
    spl = full['datasets']['nist_gads_spl_glycopeptides']
    glycan = full['datasets']['nist_glycan_msms']
    cross = full['cross_dataset_analysis']

    fig = plt.figure(figsize=(16, 3.8))
    fig.patch.set_facecolor('white')
    gs = GridSpec(1, 4, figure=fig, wspace=0.35, left=0.04, right=0.97, top=0.85, bottom=0.18)

    # --- Chart A: 3D — all datasets overlaid in S-entropy space ---
    ax1 = fig.add_subplot(gs[0], projection='3d')

    # Spike spectra
    for s in spike['spectra']:
        ax1.scatter(s['sentropy_coords']['sk'], s['sentropy_coords']['st'],
                    s['sentropy_coords']['se'], c=[COLORS['warm']], s=20, alpha=0.7)

    # Source libraries (subsample)
    np.random.seed(42)
    src_idx = np.random.choice(len(source['entries']), min(300, len(source['entries'])), replace=False)
    for i in src_idx:
        e = source['entries'][i]
        ax1.scatter(e['sentropy_coords']['sk'], e['sentropy_coords']['st'],
                    e['sentropy_coords']['se'], c=[COLORS['primary']], s=5, alpha=0.3)

    # SPL
    for e in spl['entries']:
        ax1.scatter(e['sentropy_coords']['sk'], e['sentropy_coords']['st'],
                    e['sentropy_coords']['se'], c=[COLORS['accent']], s=25, alpha=0.8)

    ax1.set_xlabel(r'$S_k$')
    ax1.set_ylabel(r'$S_t$')
    ax1.set_zlabel(r'$S_e$')
    ax1.set_title('A', loc='left', fontweight='bold', fontsize=10)
    ax1.view_init(elev=20, azim=60)

    # --- Chart B: Conformance rates by dataset ---
    ax2 = fig.add_subplot(gs[1])
    datasets = ['Spike\nMS/MS', 'Source\nA–K', 'SPL\nGlycopep.', 'Glycan\nMS/MS']
    conformance = [
        spike['summary']['partition_coords_valid'] / spike['summary']['total_spectra'],
        source['summary']['partition_coords_valid'] / source['summary']['validated_entries'],
        spl['summary']['partition_coords_valid'] / spl['summary']['validated_entries'],
        glycan['summary']['partition_coords_valid'] / glycan['summary']['total_entries'],
    ]
    bar_colors = [COLORS['warm'], COLORS['primary'], COLORS['accent'], COLORS['gold']]
    bars = ax2.bar(datasets, conformance, color=bar_colors, edgecolor='k', linewidth=0.3)
    ax2.set_ylim(0.9, 1.02)
    ax2.axhline(1.0, color='gray', linestyle='--', linewidth=0.4)
    ax2.set_ylabel('Conformance')
    ax2.set_title('B', loc='left', fontweight='bold', fontsize=10)

    # --- Chart C: m/z range comparison (box-like) ---
    ax3 = fig.add_subplot(gs[2])

    # Collect m/z per dataset
    spike_mz = [s['precursor_mz'] for s in spike['spectra']]
    source_mz = [e['precursor_mz'] for e in source['entries']]
    spl_mz = [e['precursor_mz'] for e in spl['entries']]
    glycan_mz = [e['precursor_mz'] for e in glycan['entries']]

    bp = ax3.boxplot([spike_mz, source_mz, spl_mz, glycan_mz],
                     labels=['Spike', 'Source', 'SPL', 'Glycan'],
                     patch_artist=True, widths=0.5,
                     boxprops=dict(linewidth=0.5),
                     medianprops=dict(color='black', linewidth=1),
                     whiskerprops=dict(linewidth=0.5),
                     capprops=dict(linewidth=0.5),
                     flierprops=dict(markersize=2))
    for patch, color in zip(bp['boxes'], bar_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax3.set_ylabel(r'$m/z$')
    ax3.set_title('C', loc='left', fontweight='bold', fontsize=10)

    # --- Chart D: Partition n distribution stacked ---
    ax4 = fig.add_subplot(gs[3])

    all_n = {
        'Spike': [s['partition_coords']['n'] for s in spike['spectra']],
        'Source': [e['partition_coords']['n'] for e in source['entries']],
        'SPL': [e['partition_coords']['n'] for e in spl['entries']],
        'Glycan': [e['partition_coords']['n'] for e in glycan['entries']],
    }
    bins = range(1, 10)
    bottom = np.zeros(len(bins) - 1)
    for (name, n_vals), color in zip(all_n.items(), bar_colors):
        hist, _ = np.histogram(n_vals, bins=bins)
        ax4.bar(np.array(list(bins[:-1])) + 0.5, hist, bottom=bottom, width=0.7,
                color=color, edgecolor='k', linewidth=0.2, alpha=0.7, label=name)
        bottom += hist

    ax4.set_xlabel(r'$n$ (principal)')
    ax4.set_ylabel('Count')
    ax4.legend(fontsize=5, ncol=2, framealpha=0.7)
    ax4.set_title('D', loc='left', fontweight='bold', fontsize=10)

    fig.suptitle('Cross-Dataset Analysis — Partition Framework Universality', fontsize=11, fontweight='bold', y=0.97)
    fig.savefig(figures_dir / 'panel_5_cross_dataset.png', bbox_inches='tight')
    plt.close(fig)
    print('  Panel 5 saved.')


# ============================================================================
# Panel 6: Full Results — Deep Validation Metrics
# ============================================================================

def panel_6_full_validation():
    data = load_json('nist_spike_igg_validation_results.json')
    spectra = data['datasets']['sars_cov2_spike_protein']['spectra']

    fig = plt.figure(figsize=(16, 3.8))
    fig.patch.set_facecolor('white')
    gs = GridSpec(1, 4, figure=fig, wspace=0.35, left=0.04, right=0.97, top=0.85, bottom=0.15)

    # --- Chart A: 3D — partition depth M vs DRIP coherence vs DRIP symmetry ---
    ax1 = fig.add_subplot(gs[0], projection='3d')
    M_vals = [s['partition_depth']['M'] for s in spectra]
    drip_c = [s['lagrangian']['drip_coherence'] for s in spectra]
    drip_s = [s['lagrangian']['drip_symmetry'] for s in spectra]
    scores = [s['validation_score'] for s in spectra]

    sc = ax1.scatter(M_vals, drip_c, drip_s, c=scores, cmap='RdYlGn',
                     s=50, edgecolors='k', linewidths=0.3, vmin=0.85, vmax=1.0)
    ax1.set_xlabel(r'$\mathcal{M}$')
    ax1.set_ylabel('DRIP coh.')
    ax1.set_zlabel('DRIP sym.')
    ax1.set_title('A', loc='left', fontweight='bold', fontsize=10)
    ax1.view_init(elev=25, azim=45)

    # --- Chart B: A/R/V ternary-style (stacked bar) ---
    ax2 = fig.add_subplot(gs[1])
    A_vals = [s['partition_depth']['actualized'] for s in spectra]
    R_vals = [s['partition_depth']['residue'] for s in spectra]
    V_vals = [s['partition_depth']['potential'] for s in spectra]

    x = range(len(spectra))
    ax2.bar(x, A_vals, color=COLORS['primary'], edgecolor='none', width=1.0, label='A')
    ax2.bar(x, R_vals, bottom=A_vals, color=COLORS['warm'], edgecolor='none', width=1.0, label='R')
    bottoms = [a + r for a, r in zip(A_vals, R_vals)]
    ax2.bar(x, V_vals, bottom=bottoms, color=COLORS['accent'], edgecolor='none', width=1.0, label='V')
    ax2.set_xlabel('Spectrum Index')
    ax2.set_ylabel('Fraction')
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=5, loc='upper right', framealpha=0.7)
    ax2.set_title('B', loc='left', fontweight='bold', fontsize=10)

    # --- Chart C: Charge emergence — observed vs predicted ---
    ax3 = fig.add_subplot(gs[2])
    obs_charge = [s['charge_emergence']['charge_state'] for s in spectra]
    pred_charge = [s['charge_emergence']['charge_from_partition'] for s in spectra]
    matches = [s['charge_emergence']['charge_matches'] for s in spectra]
    colors_m = [COLORS['pass'] if m else COLORS['fail'] for m in matches]

    jitter_x = np.random.normal(0, 0.05, len(spectra))
    jitter_y = np.random.normal(0, 0.05, len(spectra))
    ax3.scatter([o + j for o, j in zip(obs_charge, jitter_x)],
                [p + j for p, j in zip(pred_charge, jitter_y)],
                c=colors_m, s=50, edgecolors='k', linewidths=0.3, alpha=0.8)
    ax3.plot([0, 5], [0, 5], 'k--', linewidth=0.5, alpha=0.3)
    ax3.set_xlabel('Observed $z$')
    ax3.set_ylabel('Predicted $z$')
    ax3.set_xlim(0.5, 5)
    ax3.set_ylim(0.5, 5)
    ax3.set_aspect('equal')
    ax3.set_title('C', loc='left', fontweight='bold', fontsize=10)

    # --- Chart D: Lagrangian terms (kinetic, gauge, potential) ---
    ax4 = fig.add_subplot(gs[3])
    kinetic = [s['lagrangian']['kinetic'] for s in spectra]
    gauge = [s['lagrangian']['gauge'] for s in spectra]
    mzs = [s['precursor_mz'] for s in spectra]

    ax4.scatter(mzs, kinetic, c=COLORS['primary'], s=30, alpha=0.7, label='Kinetic')
    ax4.scatter(mzs, gauge, c=COLORS['warm'], s=30, alpha=0.7, label='Gauge')
    ax4.set_xlabel(r'Precursor $m/z$')
    ax4.set_ylabel('Lagrangian Term')
    ax4.set_yscale('log')
    ax4.legend(fontsize=5, framealpha=0.7)
    ax4.set_title('D', loc='left', fontweight='bold', fontsize=10)

    fig.suptitle('Partition Framework Deep Validation Metrics', fontsize=11, fontweight='bold', y=0.97)
    fig.savefig(figures_dir / 'panel_6_full_validation.png', bbox_inches='tight')
    plt.close(fig)
    print('  Panel 6 saved.')


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print('Generating 6 validation panels...\n')
    panel_1_spike_protein()
    panel_2_source_libraries()
    panel_3_spl_glycopeptides()
    panel_4_glycan_msms()
    panel_5_cross_dataset()
    panel_6_full_validation()
    print(f'\nAll panels saved to: {figures_dir}')
