#!/usr/bin/env python3
"""
Generate 8 publication panels (4 per paper).
Each panel: white background, 4 charts in a row, minimal text,
at least one 3D chart per panel. No conceptual/text/table charts.
"""

import json
import csv
import math
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
from matplotlib import cm

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 6,
    'figure.dpi': 300,
})

BASE = os.path.dirname(__file__)
P1_RES = os.path.join(BASE, "prompt-based-spectral-database", "results")
P1_FIG = os.path.join(BASE, "prompt-based-spectral-database", "figures")
P2_RES = os.path.join(BASE, "purpose-based-analysis", "results")
P2_FIG = os.path.join(BASE, "purpose-based-analysis", "figures")


def load_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def load_json(path):
    with open(path) as f:
        return json.load(f)


# ============================================================================
# Color maps for molecular types
# ============================================================================
TYPE_COLORS = {
    'diatomic': '#2196F3',
    'triatomic': '#4CAF50',
    'tetra': '#FF9800',
    'poly': '#F44336',
}

def get_color(mol_type):
    return TYPE_COLORS.get(mol_type, '#9E9E9E')


# ============================================================================
# PAPER 1 PANELS
# ============================================================================

def paper1_panel1():
    """Panel 1: S-Entropy Space and Ternary Encoding"""
    data = load_csv(os.path.join(P1_RES, "01_sentropy_coordinates.csv"))
    res = load_csv(os.path.join(P1_RES, "02_resolution_cascade.csv"))

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    fig.subplots_adjust(left=0.04, right=0.97, top=0.88, bottom=0.13, wspace=0.35)

    # A: 3D S-entropy space
    ax = fig.add_subplot(141, projection='3d')
    axes[0].set_visible(False)
    for d in data:
        sk, st, se = float(d['Sk']), float(d['St']), float(d['Se'])
        c = get_color(d['type'])
        ms = max(4, float(d['mass']) / 15)
        ax.scatter(sk, st, se, c=c, s=ms, alpha=0.85, edgecolors='k', linewidth=0.3)
    ax.set_xlabel('$S_k$')
    ax.set_ylabel('$S_t$')
    ax.set_zlabel('$S_e$')
    ax.set_title('(A) S-Entropy Space', fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)
    ax.view_init(elev=25, azim=135)

    # B: Sk distribution by type
    ax2 = axes[1]
    for t, c in TYPE_COLORS.items():
        vals = [float(d['Sk']) for d in data if d['type'] == t]
        if vals:
            ax2.hist(vals, bins=10, alpha=0.6, color=c, label=t, range=(0, 1))
    ax2.set_xlabel('$S_k$ (Knowledge Entropy)')
    ax2.set_ylabel('Count')
    ax2.set_title('(B) $S_k$ Distribution')
    ax2.legend(loc='upper left', framealpha=0.7)

    # C: Resolution cascade
    ax3 = axes[2]
    depths = [int(r['depth']) for r in res]
    unique = [int(r['uniquely_resolved']) for r in res]
    occupied = [int(r['cells_occupied']) for r in res]
    ax3.plot(depths, unique, 'o-', color='#2196F3', markersize=4, label='Unique')
    ax3.plot(depths, occupied, 's--', color='#FF9800', markersize=3, label='Occupied cells')
    ax3.axhline(39, color='gray', linestyle=':', alpha=0.5)
    ax3.axvline(11, color='gray', linestyle=':', alpha=0.3)
    ax3.set_xlabel('Trit Depth $k$')
    ax3.set_ylabel('Count')
    ax3.set_title('(C) Resolution Cascade')
    ax3.legend(loc='center right', framealpha=0.7)

    # D: Se vs type (strip plot)
    ax4 = axes[3]
    type_order = ['diatomic', 'triatomic', 'tetra', 'poly']
    for i, t in enumerate(type_order):
        vals = [float(d['Se']) for d in data if d['type'] == t]
        jitter = np.random.uniform(-0.15, 0.15, len(vals))
        ax4.scatter([i + j for j in jitter], vals, c=TYPE_COLORS[t],
                    s=30, alpha=0.8, edgecolors='k', linewidth=0.3)
    ax4.set_xticks(range(4))
    ax4.set_xticklabels(['Di', 'Tri', 'Tetra', 'Poly'])
    ax4.set_ylabel('$S_e$ (Evolution Entropy)')
    ax4.set_title('(D) $S_e$ by Type')

    fig.savefig(os.path.join(P1_FIG, "panel_1_sentropy_space.png"), dpi=300)
    plt.close()
    print("  Paper 1 Panel 1 saved")


def paper1_panel2():
    """Panel 2: Analyzer Trajectory Generation"""
    traj = load_csv(os.path.join(P1_RES, "04_trajectories.csv"))
    data = load_csv(os.path.join(P1_RES, "01_sentropy_coordinates.csv"))

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    fig.subplots_adjust(left=0.05, right=0.97, top=0.88, bottom=0.15, wspace=0.32)

    mz = np.array([float(t['mass']) for t in traj])
    types = [next(d['type'] for d in data if d['name'] == t['name']) for t in traj]
    colors = [get_color(t) for t in types]

    # A: 3D - TOF vs Orbitrap vs FT-ICR
    ax = fig.add_subplot(141, projection='3d')
    axes[0].set_visible(False)
    tof_t = np.array([float(t['TOF_flight_time_us']) for t in traj])
    orbi_f = np.array([float(t['Orbitrap_freq_kHz']) for t in traj])
    icr_f = np.array([float(t['FTICR_freq_kHz']) for t in traj])
    ax.scatter(tof_t, np.log10(orbi_f), np.log10(icr_f), c=colors,
               s=mz * 0.5, alpha=0.8, edgecolors='k', linewidth=0.3)
    ax.set_xlabel('TOF ($\\mu$s)')
    ax.set_ylabel('log$_{10}$(Orbi kHz)')
    ax.set_zlabel('log$_{10}$(ICR kHz)')
    ax.set_title('(A) Multi-Analyzer Space')
    ax.view_init(elev=20, azim=45)

    # B: TOF flight time vs sqrt(m/z)
    ax2 = axes[1]
    sqrt_mz = np.sqrt(mz)
    ax2.scatter(sqrt_mz, tof_t, c=colors, s=30, alpha=0.8,
                edgecolors='k', linewidth=0.3)
    fit = np.polyfit(sqrt_mz, tof_t, 1)
    x_fit = np.linspace(sqrt_mz.min(), sqrt_mz.max(), 50)
    ax2.plot(x_fit, np.polyval(fit, x_fit), 'r--', linewidth=1, alpha=0.7)
    ax2.set_xlabel('$\\sqrt{m/z}$')
    ax2.set_ylabel('Flight Time ($\\mu$s)')
    ax2.set_title('(B) TOF: $T \\propto \\sqrt{m/z}$')

    # C: Orbitrap freq vs 1/sqrt(m/z)
    ax3 = axes[2]
    inv_sqrt = 1.0 / sqrt_mz
    ax3.scatter(inv_sqrt, orbi_f, c=colors, s=30, alpha=0.8,
                edgecolors='k', linewidth=0.3)
    fit2 = np.polyfit(inv_sqrt, orbi_f, 1)
    x_fit2 = np.linspace(inv_sqrt.min(), inv_sqrt.max(), 50)
    ax3.plot(x_fit2, np.polyval(fit2, x_fit2), 'r--', linewidth=1, alpha=0.7)
    ax3.set_xlabel('$1/\\sqrt{m/z}$')
    ax3.set_ylabel('Axial Freq (kHz)')
    ax3.set_title('(C) Orbitrap: $\\omega \\propto \\sqrt{z/m}$')

    # D: FT-ICR freq vs 1/(m/z)
    ax4 = axes[3]
    inv_mz = 1.0 / mz
    ax4.scatter(inv_mz, icr_f, c=colors, s=30, alpha=0.8,
                edgecolors='k', linewidth=0.3)
    fit3 = np.polyfit(inv_mz, icr_f, 1)
    x_fit3 = np.linspace(inv_mz.min(), inv_mz.max(), 50)
    ax4.plot(x_fit3, np.polyval(fit3, x_fit3), 'r--', linewidth=1, alpha=0.7)
    ax4.set_xlabel('$1/(m/z)$')
    ax4.set_ylabel('Cyclotron Freq (kHz)')
    ax4.set_title('(D) FT-ICR: $\\omega_c \\propto z/m$')

    fig.savefig(os.path.join(P1_FIG, "panel_2_analyzer_trajectories.png"), dpi=300)
    plt.close()
    print("  Paper 1 Panel 2 saved")


def paper1_panel3():
    """Panel 3: Chemical Cohesion and Emergent Clustering"""
    coh = load_csv(os.path.join(P1_RES, "03_chemical_cohesion.csv"))
    clust = load_csv(os.path.join(P1_RES, "07_depth3_clustering.csv"))
    sim = load_csv(os.path.join(P1_RES, "08_pairwise_similarity.csv"))
    data = load_csv(os.path.join(P1_RES, "01_sentropy_coordinates.csv"))

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    fig.subplots_adjust(left=0.04, right=0.97, top=0.88, bottom=0.15, wspace=0.35)

    # A: 3D with depth-3 cell coloring
    ax = fig.add_subplot(141, projection='3d')
    axes[0].set_visible(False)
    cell_colors_map = {}
    cmap = plt.cm.Set3
    for i, cl in enumerate(clust):
        cell_colors_map[cl['cell_prefix']] = cmap(i / max(len(clust), 1))
    for d in data:
        sk, st, se = float(d['Sk']), float(d['St']), float(d['Se'])
        prefix = d['ternary_12'][:3]
        c = cell_colors_map.get(prefix, (0.5, 0.5, 0.5, 1))
        ax.scatter(sk, st, se, c=[c], s=35, alpha=0.9, edgecolors='k', linewidth=0.3)
    ax.set_xlabel('$S_k$'); ax.set_ylabel('$S_t$'); ax.set_zlabel('$S_e$')
    ax.set_title('(A) Depth-3 Clusters')
    ax.view_init(elev=25, azim=135)

    # B: Cohesion ratio bar chart
    ax2 = axes[1]
    families = [c['family'] for c in coh]
    ratios = [float(c['cohesion_ratio_R']) for c in coh]
    bar_colors = ['#4CAF50' if r > 1.0 else '#F44336' for r in ratios]
    bars = ax2.barh(range(len(families)), ratios, color=bar_colors, edgecolor='k', linewidth=0.3)
    ax2.axvline(1.0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    ax2.set_yticks(range(len(families)))
    ax2.set_yticklabels([f.replace(' ', '\n') for f in families], fontsize=6)
    ax2.set_xlabel('Cohesion Ratio $R$')
    ax2.set_title('(B) Chemical Cohesion')

    # C: Pairwise similarity heatmap
    ax3 = axes[2]
    names = sorted(set(s['compound_A'] for s in sim) | set(s['compound_B'] for s in sim))
    n = len(names)
    idx = {name: i for i, name in enumerate(names)}
    mat = np.zeros((n, n))
    for s in sim:
        i, j = idx[s['compound_A']], idx[s['compound_B']]
        v = int(s['common_prefix_length'])
        mat[i, j] = v; mat[j, i] = v
    np.fill_diagonal(mat, 18)
    im = ax3.imshow(mat, cmap='viridis', aspect='auto')
    ax3.set_title('(C) Ternary Similarity')
    ax3.set_xlabel('Compound'); ax3.set_ylabel('Compound')
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04, label='Prefix')

    # D: Depth-3 cell occupancy
    ax4 = axes[3]
    cell_names = [cl['cell_prefix'] for cl in clust]
    cell_counts = [int(cl['n_compounds']) for cl in clust]
    sorted_idx = np.argsort(cell_counts)[::-1]
    cell_names_sorted = [cell_names[i] for i in sorted_idx]
    cell_counts_sorted = [cell_counts[i] for i in sorted_idx]
    cell_cols = [cell_colors_map.get(cn, (0.5, 0.5, 0.5, 1)) for cn in cell_names_sorted]
    ax4.bar(range(len(cell_names_sorted)), cell_counts_sorted,
            color=cell_cols, edgecolor='k', linewidth=0.3)
    ax4.set_xticks(range(len(cell_names_sorted)))
    ax4.set_xticklabels(cell_names_sorted, rotation=45, fontsize=6)
    ax4.set_ylabel('Compounds')
    ax4.set_title('(D) Cell Occupancy (Depth 3)')

    fig.savefig(os.path.join(P1_FIG, "panel_3_cohesion_clustering.png"), dpi=300)
    plt.close()
    print("  Paper 1 Panel 3 saved")


def paper1_panel4():
    """Panel 4: Ion-Droplet Bijection and Complexity"""
    bij = load_json(os.path.join(P1_RES, "05_ion_droplet_bijection.json"))
    comp = load_csv(os.path.join(P1_RES, "06_complexity_analysis.csv"))
    sim = load_csv(os.path.join(P1_RES, "08_pairwise_similarity.csv"))
    data = load_csv(os.path.join(P1_RES, "01_sentropy_coordinates.csv"))

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    fig.subplots_adjust(left=0.05, right=0.97, top=0.88, bottom=0.15, wspace=0.35)

    # A: 3D droplet space (Weber, Ohnesorge, Sk)
    ax = fig.add_subplot(141, projection='3d')
    axes[0].set_visible(False)
    we = [b['Weber'] for b in bij]
    oh = [b['Ohnesorge'] for b in bij]
    sk_ion = [b['Sk_ion'] for b in bij]
    type_map = {d['name']: d['type'] for d in data}
    colors = [get_color(type_map.get(b['name'], 'poly')) for b in bij]
    ax.scatter(we, oh, sk_ion, c=colors, s=30, alpha=0.8,
               edgecolors='k', linewidth=0.3)
    ax.set_xlabel('Weber')
    ax.set_ylabel('Ohnesorge')
    ax.set_zlabel('$S_k$ (ion)')
    ax.set_title('(A) Droplet Parameter Space')
    ax.view_init(elev=20, azim=60)

    # B: Sk_ion vs Sk_drip
    ax2 = axes[1]
    sk_drip = [b['Sk_drip'] for b in bij]
    ax2.scatter(sk_ion, sk_drip, c=colors, s=30, alpha=0.8,
                edgecolors='k', linewidth=0.3)
    ax2.plot([0, 1], [0, 1], 'r--', linewidth=0.8, alpha=0.5)
    ax2.set_xlabel('$S_k$ (Ion Path)')
    ax2.set_ylabel('$S_k$ (Drip Path)')
    ax2.set_title('(B) Dual Path $S_k$')
    ax2.set_xlim(0, 1.05); ax2.set_ylim(0, 1.05)

    # C: Prefix length vs Euclidean distance
    ax3 = axes[2]
    pls = [int(s['common_prefix_length']) for s in sim]
    eucs = [float(s['euclidean_distance']) for s in sim]
    ax3.scatter(pls, eucs, c='#2196F3', s=8, alpha=0.3)
    # Fit exponential decay
    unique_pls = sorted(set(pls))
    means = [np.mean([e for p, e in zip(pls, eucs) if p == pl]) for pl in unique_pls]
    ax3.plot(unique_pls, means, 'r-o', markersize=4, linewidth=1.5, label='Mean')
    ax3.set_xlabel('Common Prefix Length')
    ax3.set_ylabel('Euclidean Distance')
    ax3.set_title('(C) Distance Preservation')
    ax3.legend(framealpha=0.7)

    # D: Speedup scaling (log-log)
    ax4 = axes[3]
    for k in [6, 12, 18]:
        subset = [c for c in comp if int(c['trit_depth_k']) == k]
        ns = [int(c['N_compounds']) for c in subset]
        sp = [float(c['speedup']) for c in subset]
        ax4.plot(ns, sp, 'o-', markersize=4, label=f'$k={k}$')
    ax4.set_xscale('log'); ax4.set_yscale('log')
    ax4.set_xlabel('Database Size $N$')
    ax4.set_ylabel('Speedup')
    ax4.set_title('(D) Speedup vs $N$')
    ax4.legend(framealpha=0.7)

    fig.savefig(os.path.join(P1_FIG, "panel_4_bijection_complexity.png"), dpi=300)
    plt.close()
    print("  Paper 1 Panel 4 saved")


# ============================================================================
# PAPER 2 PANELS
# ============================================================================

def paper2_panel1():
    """Panel 1: Oscillatory Resonance vs Algorithmic Comparison"""
    res = load_csv(os.path.join(P2_RES, "01_resonance_vs_algorithmic.csv"))
    dual = load_csv(os.path.join(P2_RES, "02_dual_path_convergence.csv"))

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    fig.subplots_adjust(left=0.05, right=0.97, top=0.88, bottom=0.15, wspace=0.35)

    # A: 3D - prefix length, euclidean distance, tanimoto
    ax = fig.add_subplot(141, projection='3d')
    axes[0].set_visible(False)
    pls = np.array([int(r['prefix_length']) for r in res])
    eucs = np.array([float(r['euclidean_distance']) for r in res])
    tans = np.array([float(r['tanimoto']) for r in res])
    sc = ax.scatter(pls, eucs, tans, c=pls, cmap='viridis', s=5, alpha=0.5)
    ax.set_xlabel('Prefix Len')
    ax.set_ylabel('Eucl. Dist')
    ax.set_zlabel('Tanimoto')
    ax.set_title('(A) Resonance vs Algorithm')
    ax.view_init(elev=25, azim=45)

    # B: Resonance ops vs Tanimoto ops
    ax2 = axes[1]
    res_ops = np.array([int(r['resonance_ops']) for r in res])
    tan_ops = np.array([int(r['tanimoto_ops']) for r in res])
    ax2.scatter(res_ops, tan_ops, c='#2196F3', s=8, alpha=0.4)
    ax2.plot([0, 18], [1024, 1024], 'r--', linewidth=1, alpha=0.7, label='Tanimoto O(d)')
    ax2.set_xlabel('Resonance Ops (prefix check)')
    ax2.set_ylabel('Tanimoto Ops (bit ops)')
    ax2.set_title('(B) Operation Count')
    ax2.legend(framealpha=0.7)

    # C: Dual path Sk comparison
    ax3 = axes[2]
    sk_ion = [float(d['Sk_ion']) for d in dual]
    sk_drip = [float(d['Sk_drip']) for d in dual]
    types = [d['type'] for d in dual]
    colors = [get_color(t) for t in types]
    ax3.scatter(sk_ion, sk_drip, c=colors, s=40, alpha=0.8,
                edgecolors='k', linewidth=0.3)
    ax3.plot([0, 1], [0, 1], 'r--', linewidth=0.8, alpha=0.5)
    ax3.set_xlabel('$S_k$ (Ion Path)')
    ax3.set_ylabel('$S_k$ (Droplet Path)')
    ax3.set_title('(C) Dual Path Convergence')

    # D: Common prefix histogram (ion vs drip)
    ax4 = axes[3]
    cpls = [int(d['common_prefix']) for d in dual]
    ax4.hist(cpls, bins=range(0, 20), color='#4CAF50', edgecolor='k',
             linewidth=0.3, alpha=0.8)
    ax4.axvline(np.mean(cpls), color='r', linestyle='--', linewidth=1,
                label=f'Mean={np.mean(cpls):.1f}')
    ax4.set_xlabel('Common Prefix Length')
    ax4.set_ylabel('Count')
    ax4.set_title('(D) Dual Path Agreement')
    ax4.legend(framealpha=0.7)

    fig.savefig(os.path.join(P2_FIG, "panel_1_resonance_comparison.png"), dpi=300)
    plt.close()
    print("  Paper 2 Panel 1 saved")


def paper2_panel2():
    """Panel 2: Purpose-Based Constraint Reduction"""
    purp = load_csv(os.path.join(P2_RES, "03_purpose_constraints.csv"))
    cont = load_csv(os.path.join(P2_RES, "05_prompt_contraction.csv"))
    land = load_csv(os.path.join(P2_RES, "06_landauer_cost.csv"))

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    fig.subplots_adjust(left=0.05, right=0.97, top=0.88, bottom=0.15, wspace=0.38)

    # A: 3D - domain constraints in S-entropy space
    # Show the constrained regions as boxes
    ax = fig.add_subplot(141, projection='3d')
    axes[0].set_visible(False)
    # Plot the full [0,1]^3 wireframe
    for s in [0, 1]:
        for t in [0, 1]:
            ax.plot([0, 1], [s, s], [t, t], 'k-', alpha=0.1, linewidth=0.3)
            ax.plot([s, s], [0, 1], [t, t], 'k-', alpha=0.1, linewidth=0.3)
            ax.plot([s, s], [t, t], [0, 1], 'k-', alpha=0.1, linewidth=0.3)
    # Metabolomics constraint box (Sk: 0.1-1.0, Se: 0-1, St: 0-1)
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    verts_meta = [
        [[0.1, 0, 0], [1, 0, 0], [1, 1, 0], [0.1, 1, 0]],
        [[0.1, 0, 1], [1, 0, 1], [1, 1, 1], [0.1, 1, 1]],
    ]
    # Diatomic constraint (Se: 0-0.05)
    verts_di = [
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        [[0, 0, 0.05], [1, 0, 0.05], [1, 1, 0.05], [0, 1, 0.05]],
    ]
    pc_meta = Poly3DCollection(verts_meta, alpha=0.15, facecolor='#2196F3', edgecolor='#2196F3')
    pc_di = Poly3DCollection(verts_di, alpha=0.25, facecolor='#FF9800', edgecolor='#FF9800')
    ax.add_collection3d(pc_meta)
    ax.add_collection3d(pc_di)
    ax.set_xlabel('$S_k$'); ax.set_ylabel('$S_t$'); ax.set_zlabel('$S_e$')
    ax.set_title('(A) Domain Constraints')
    ax.view_init(elev=20, azim=135)

    # B: Reduction ratio by domain
    ax2 = axes[1]
    domains = [p['domain'] for p in purp if p['domain'] != 'No constraints']
    rhos = [float(p['rho_percent']) for p in purp if p['domain'] != 'No constraints']
    bar_colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
    ax2.barh(range(len(domains)), rhos, color=bar_colors[:len(domains)],
             edgecolor='k', linewidth=0.3)
    ax2.set_yticks(range(len(domains)))
    ax2.set_yticklabels(domains, fontsize=7)
    ax2.set_xlabel('Reduction $\\rho$ (%)')
    ax2.set_title('(B) Phase Space Reduction')
    ax2.set_xlim(85, 100.5)

    # C: Prompt contraction
    ax3 = axes[2]
    labels = [c['constraint'] for c in cont]
    counts = [int(c['n_matching']) for c in cont]
    ax3.plot(range(len(labels)), counts, 'o-', color='#F44336',
             markersize=6, linewidth=2)
    ax3.fill_between(range(len(labels)), counts, alpha=0.15, color='#F44336')
    ax3.set_xticks(range(len(labels)))
    ax3.set_xticklabels([l[:12] for l in labels], rotation=30, fontsize=6)
    ax3.set_ylabel('Matching Compounds')
    ax3.set_title('(C) Monotonic Contraction')

    # D: Landauer cost curve
    ax4 = axes[3]
    r_cells = [int(l['r_cells']) for l in land]
    info_bits = [float(l['information_gained_bits']) for l in land]
    rho_vals = [float(l['reduction_rho']) for l in land]
    ax4.plot(r_cells, info_bits, 'o-', color='#9C27B0', markersize=5)
    ax4.set_xscale('log')
    ax4.set_xlabel('Constrained Cells $r$')
    ax4.set_ylabel('Information Gained (bits)')
    ax4.set_title('(D) Landauer Cost')

    fig.savefig(os.path.join(P2_FIG, "panel_2_purpose_constraints.png"), dpi=300)
    plt.close()
    print("  Paper 2 Panel 2 saved")


def paper2_panel3():
    """Panel 3: Selective Generation Efficiency"""
    sel = load_csv(os.path.join(P2_RES, "04_selective_generation.csv"))
    scal = load_csv(os.path.join(P2_RES, "07_scaling_analysis.csv"))

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    fig.subplots_adjust(left=0.05, right=0.97, top=0.88, bottom=0.15, wspace=0.35)

    # A: 3D - query mass, Se, reduction %
    ax = fig.add_subplot(141, projection='3d')
    axes[0].set_visible(False)
    masses = [float(s['query_mass']) for s in sel]
    ses = [float(s['query_Se']) for s in sel]
    reds = [float(s['best_reduction_pct']) for s in sel]
    type_map = {'diatomic': '#2196F3', 'triatomic': '#4CAF50', 'tetra': '#FF9800', 'poly': '#F44336'}
    colors = [type_map.get(s['query_type'], '#999') for s in sel]
    ax.scatter(masses, ses, reds, c=colors, s=30, alpha=0.8,
               edgecolors='k', linewidth=0.3)
    ax.set_xlabel('Mass (Da)')
    ax.set_ylabel('$S_e$')
    ax.set_zlabel('Reduction %')
    ax.set_title('(A) Reduction Landscape')
    ax.view_init(elev=25, azim=45)

    # B: Exhaustive vs selective ops per compound
    ax2 = axes[1]
    exh = [float(s['ops_exhaustive']) for s in sel]
    type_ops = [float(s['ops_type']) for s in sel]
    mass_ops = [float(s['ops_mass']) for s in sel]
    se_ops = [float(s['ops_se']) for s in sel]
    x = range(len(sel))
    ax2.fill_between(x, exh, alpha=0.1, color='red')
    ax2.scatter(x, type_ops, s=12, alpha=0.7, label='Type', marker='o', color='#2196F3')
    ax2.scatter(x, mass_ops, s=12, alpha=0.7, label='Mass', marker='s', color='#4CAF50')
    ax2.scatter(x, se_ops, s=12, alpha=0.7, label='$S_e$', marker='^', color='#FF9800')
    ax2.set_yscale('log')
    ax2.set_xlabel('Compound Index')
    ax2.set_ylabel('Operations')
    ax2.set_title('(B) Selective vs Exhaustive')
    ax2.legend(framealpha=0.7, fontsize=5, ncol=2)

    # C: Speedup vs N at different domain fractions
    ax3 = axes[2]
    for frac in [0.01, 0.05, 0.10]:
        subset = [s for s in scal if float(s['domain_fraction']) == frac]
        ns = [int(s['N_database']) for s in subset]
        sp = [float(s['speedup_vs_traditional']) for s in subset]
        ax3.plot(ns, sp, 'o-', markersize=3, label=f'{frac:.0%} domain')
    ax3.set_xscale('log'); ax3.set_yscale('log')
    ax3.set_xlabel('Database Size $N$')
    ax3.set_ylabel('Speedup vs Traditional')
    ax3.set_title('(C) Scaling by Domain Size')
    ax3.legend(framealpha=0.7, fontsize=6)

    # D: Reduction rho vs domain fraction
    ax4 = axes[3]
    for N in [1000, 100000, 100000000]:
        subset = [s for s in scal if int(s['N_database']) == N]
        fracs = [float(s['domain_fraction']) for s in subset]
        rhos = [float(s['reduction_rho']) for s in subset]
        label = f'N={N:.0e}' if N > 999 else f'N={N}'
        ax4.plot(fracs, rhos, 'o-', markersize=4, label=label)
    ax4.set_xlabel('Domain Fraction')
    ax4.set_ylabel('Reduction $\\rho$')
    ax4.set_title('(D) $\\rho$ vs Specificity')
    ax4.legend(framealpha=0.7, fontsize=6)

    fig.savefig(os.path.join(P2_FIG, "panel_3_selective_generation.png"), dpi=300)
    plt.close()
    print("  Paper 2 Panel 3 saved")


def paper2_panel4():
    """Panel 4: Cross-Domain and Maxwell Demon"""
    fnr = load_csv(os.path.join(P2_RES, "08_cross_domain_fnr.csv"))
    land = load_csv(os.path.join(P2_RES, "06_landauer_cost.csv"))
    dual = load_csv(os.path.join(P2_RES, "02_dual_path_convergence.csv"))

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    fig.subplots_adjust(left=0.05, right=0.97, top=0.88, bottom=0.15, wspace=0.38)

    # A: 3D dual path (Sk, St, Se) ion vs drip
    ax = fig.add_subplot(141, projection='3d')
    axes[0].set_visible(False)
    for d in dual:
        sk_i, st_i, se_i = float(d['Sk_ion']), float(d['St_ion']), float(d['Se_ion'])
        sk_d, st_d, se_d = float(d['Sk_drip']), float(d['St_drip']), float(d['Se_drip'])
        c = get_color(d['type'])
        ax.scatter(sk_i, st_i, se_i, c=c, s=25, alpha=0.8, marker='o',
                   edgecolors='k', linewidth=0.2)
        ax.scatter(sk_d, st_d, se_d, c=c, s=15, alpha=0.5, marker='^',
                   edgecolors='k', linewidth=0.2)
        ax.plot([sk_i, sk_d], [st_i, st_d], [se_i, se_d],
                color=c, alpha=0.3, linewidth=0.5)
    ax.set_xlabel('$S_k$'); ax.set_ylabel('$S_t$'); ax.set_zlabel('$S_e$')
    ax.set_title('(A) Ion vs Drip Paths')
    ax.view_init(elev=25, azim=135)

    # B: False negative rate by domain
    ax2 = axes[1]
    domains = [f['domain'] for f in fnr]
    fnrs = [float(f['false_negative_rate']) * 100 for f in fnr]
    n_in = [int(f['n_in_domain']) for f in fnr]
    bar_colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0', '#795548']
    ax2.bar(range(len(domains)), fnrs, color=bar_colors[:len(domains)],
            edgecolor='k', linewidth=0.3)
    ax2.set_xticks(range(len(domains)))
    ax2.set_xticklabels([d.replace(' ', '\n') for d in domains], fontsize=5.5)
    ax2.set_ylabel('False Negative Rate (%)')
    ax2.set_title('(B) Wrong Domain FNR')

    # C: Landauer energy vs reduction
    ax3 = axes[2]
    rhos = [float(l['reduction_rho']) for l in land]
    info = [float(l['information_gained_bits']) for l in land]
    ax3.fill_between(rhos, info, alpha=0.2, color='#9C27B0')
    ax3.plot(rhos, info, 'o-', color='#9C27B0', markersize=5)
    ax3.set_xlabel('Reduction $\\rho$')
    ax3.set_ylabel('Information Gained (bits)')
    ax3.set_title('(C) Knowledge = Entropy Reduction')

    # D: Maxwell demon schematic as data visualization
    # Show the filtering cascade: potential -> filtered -> actual
    ax4 = axes[3]
    stages = ['Potential\n$3^{12}$', 'Metabol.\n$r$=4000', 'Glycan\n$r$=1500', 'Diatomic\n$r$=50']
    vals = [531441, 4000, 1500, 50]
    bar_colors_d = ['#CFD8DC', '#2196F3', '#4CAF50', '#FF9800']
    ax4.bar(range(len(stages)), [math.log10(v) for v in vals],
            color=bar_colors_d, edgecolor='k', linewidth=0.3)
    ax4.set_xticks(range(len(stages)))
    ax4.set_xticklabels(stages, fontsize=6)
    ax4.set_ylabel('$\\log_{10}$(States)')
    ax4.set_title('(D) MMD Filtering Cascade')

    fig.savefig(os.path.join(P2_FIG, "panel_4_crossdomain_mmd.png"), dpi=300)
    plt.close()
    print("  Paper 2 Panel 4 saved")


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    print("Generating all 8 panels...")
    print("\n--- Paper 1: Context-Based Spectral Database ---")
    paper1_panel1()
    paper1_panel2()
    paper1_panel3()
    paper1_panel4()
    print("\n--- Paper 2: Purpose-Based Spectral Analysis ---")
    paper2_panel1()
    paper2_panel2()
    paper2_panel3()
    paper2_panel4()
    print("\nAll 8 panels generated.")
