"""
MAFG Figure Panels Generator
Produces 8 publication-quality PNG panels (4 charts per row, 1 row each, 20x5 inches).
All data hardcoded from JSON validation results.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

OUT = r"c:\Users\kunda\Documents\bioinformatics\lavoisier\oxford\publications\minimum-analytic-finite-graph\figures"
os.makedirs(OUT, exist_ok=True)

C1 = '#2E86AB'
C2 = '#E84855'
C3 = '#3BB273'
C4 = '#F4A261'
C5 = '#9B5DE5'

def clean_ax(ax, grid=False):
    ax.set_facecolor('#ffffff')
    if grid:
        ax.grid(True, alpha=0.2, lw=0.5)
    else:
        ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)

def new_fig():
    fig = plt.figure(figsize=(20, 5), facecolor='#ffffff')
    return fig

# ── PANEL 1 ──────────────────────────────────────────────────────────────────
def panel_01():
    fig = new_fig()

    bmin_vals  = np.array([0.001, 0.5, 0.1, 1e-9, 0.5])
    weights    = np.arange(1, 9, dtype=float)
    angles     = np.linspace(0, 2*np.pi, 8, endpoint=False)
    node_labels = [f'v{i}' for i in range(8)]

    # Chart 1 – bar chart bmin log y
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.bar(range(5), bmin_vals, color=C1, width=0.6)
    ax1.set_yscale('log')
    ax1.set_xlabel('graph index', fontsize=9); ax1.set_ylabel('b_min', fontsize=9)
    ax1.set_xticks(range(5))
    clean_ax(ax1, grid=True)

    # Chart 2 – wheel graph: medium at centre, v0..v7 around
    ax2 = fig.add_subplot(1, 4, 2)
    ax2.set_facecolor('#ffffff')
    cmap = plt.cm.Blues
    norm = plt.Normalize(1, 8)
    for i, (ang, w) in enumerate(zip(angles, weights)):
        x_n, y_n = np.cos(ang), np.sin(ang)
        col = cmap(norm(w))
        ax2.scatter(x_n, y_n, color=col, s=120, zorder=3)
        ax2.plot([0, x_n], [0, y_n], color='#aaaaaa', lw=1.0, zorder=1)
        ax2.text(x_n*1.15, y_n*1.15, node_labels[i], ha='center', va='center', fontsize=7)
    ax2.scatter(0, 0, color=C4, s=150, zorder=4, marker='*')
    ax2.text(0, 0.12, 'M', ha='center', fontsize=8)
    ax2.set_xlim(-1.4, 1.4); ax2.set_ylim(-1.4, 1.4)
    ax2.set_aspect('equal'); ax2.axis('off')

    # Chart 3 – horizontal bar chart edge weights
    ax3 = fig.add_subplot(1, 4, 3)
    ax3.barh(node_labels, weights, color=C3, height=0.6)
    ax3.set_xlabel('edge weight', fontsize=9)
    clean_ax(ax3)

    # Chart 4 (3D) – 3D scatter: angle, weight, radius=1
    ax4 = fig.add_subplot(1, 4, 4, projection='3d')
    ax4.set_facecolor('#ffffff')
    x3 = np.cos(angles)
    y3 = np.sin(angles)
    z3 = weights
    ax4.scatter(x3, y3, z3, c=weights, cmap='Blues', s=60, depthshade=True)
    # Medium node at origin
    ax4.scatter([0], [0], [0], color=C4, s=120, marker='*')
    for xi, yi, zi, lbl in zip(x3, y3, z3, node_labels):
        ax4.plot([0, xi], [0, yi], [0, zi], color='#cccccc', lw=0.8)
    ax4.set_xlabel('x', fontsize=7); ax4.set_ylabel('y', fontsize=7); ax4.set_zlabel('weight', fontsize=7)
    ax4.tick_params(labelsize=6)

    plt.tight_layout()
    path = os.path.join(OUT, 'mafg_panel_01.png')
    fig.savefig(path, dpi=150, facecolor='#ffffff', bbox_inches='tight')
    plt.close(fig)
    return path

# ── PANEL 2 ──────────────────────────────────────────────────────────────────
def panel_02():
    fig = new_fig()

    sigma_all = np.array([2.5,3.8,3.8,6.5,5.0,6.0,3.5,3.5,5.5,5.5,5.5])
    bmin_all  = np.array([0.8,0.8,0.8,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5])

    sigma_graph1 = np.array([6.5, 5.0, 6.0, 3.5])
    node_labels  = ['a', 'b', 'c', 'd']
    bmin_g1      = 0.5

    sigma_mono = np.array([2.5, 2.8, 3.5, 4.5])
    n_neigh    = np.array([1, 2, 3, 4])

    # Chart 1 – scatter sigma vs bmin
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.scatter(sigma_all, bmin_all, color=C1, s=50, zorder=3)
    lim = np.array([0, 7])
    ax1.plot(lim, lim, '--', color='k', lw=0.8, label='σ=b')
    ax1.set_xlabel('σ', fontsize=9); ax1.set_ylabel('b_min', fontsize=9)
    ax1.legend(frameon=False, fontsize=8)
    clean_ax(ax1, grid=True)

    # Chart 2 – horizontal bars for nodes a,b,c,d
    ax2 = fig.add_subplot(1, 4, 2)
    ax2.barh(node_labels, sigma_graph1, color=C2, height=0.5)
    ax2.axvline(bmin_g1, color='k', lw=1.0, ls='--', label=f'b_min={bmin_g1}')
    ax2.set_xlabel('σ', fontsize=9)
    ax2.legend(frameon=False, fontsize=8)
    clean_ax(ax2)

    # Chart 3 – sigma vs n_neighbours
    ax3 = fig.add_subplot(1, 4, 3)
    ax3.plot(n_neigh, sigma_mono, 'o-', color=C3, lw=1.5, ms=6)
    ax3.set_xlabel('neighbours', fontsize=9); ax3.set_ylabel('σ', fontsize=9)
    clean_ax(ax3, grid=True)

    # Chart 4 (3D) – surface sigma(bmin, n_neighbours)
    bmin_grid = np.linspace(0.2, 1.0, 30)
    n_grid    = np.linspace(1, 5, 30)
    BM, NG    = np.meshgrid(bmin_grid, n_grid)
    SIGMA_S   = BM * (1 + 0.5 * NG)
    ax4 = fig.add_subplot(1, 4, 4, projection='3d')
    ax4.set_facecolor('#ffffff')
    ax4.plot_surface(BM, NG, SIGMA_S, cmap='viridis', edgecolor='none', alpha=0.85)
    ax4.set_xlabel('b_min', fontsize=7); ax4.set_ylabel('n', fontsize=7); ax4.set_zlabel('σ', fontsize=7)
    ax4.tick_params(labelsize=6)

    plt.tight_layout()
    path = os.path.join(OUT, 'mafg_panel_02.png')
    fig.savefig(path, dpi=150, facecolor='#ffffff', bbox_inches='tight')
    plt.close(fig)
    return path

# ── PANEL 3 ──────────────────────────────────────────────────────────────────
def panel_03():
    fig = new_fig()

    node_labels = ['a', 'b', 'c', 'd']
    sigma_tc    = np.array([3.9, 2.9, 4.0, 2.5])
    cut_w       = np.array([3.0, 3.0, 3.0, 3.0])  # equal cut weights from bmin=0.4 context
    cut_ec      = np.array([3, 3, 3, 3])
    bmin_tc     = 0.4

    # All 11 nodes across 3 graphs — assign graph indices
    sigma_all  = np.array([2.5,3.8,3.8, 6.5,5.0,6.0,3.5, 3.5,5.5,5.5,5.5])
    graph_idx  = np.array([0,0,0, 1,1,1,1, 2,2,2,2])
    colours_g  = [C1, C2, C3]

    # Chart 1 – grouped bars sigma and cut_weight
    x = np.arange(4)
    w = 0.35
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.bar(x - w/2, sigma_tc, width=w, color=C1, label='σ')
    ax1.bar(x + w/2, cut_w,   width=w, color=C2, label='cut_w')
    ax1.axhline(bmin_tc, color='k', lw=1.0, ls='--', label=f'b_min={bmin_tc}')
    ax1.set_xticks(x); ax1.set_xticklabels(node_labels, fontsize=8)
    ax1.legend(frameon=False, fontsize=7)
    clean_ax(ax1)

    # Chart 2 – cut_edges_count bars
    ax2 = fig.add_subplot(1, 4, 2)
    ax2.bar(x, cut_ec, color=C3, width=0.5)
    ax2.axhline(3, color='k', lw=0.8, ls='--')
    ax2.set_xticks(x); ax2.set_xticklabels(node_labels, fontsize=8)
    ax2.set_ylabel('cut edges', fontsize=9)
    ax2.text(3.5, 3.1, 'floor', fontsize=7)
    clean_ax(ax2)

    # Chart 3 – scatter sigma coloured by graph index
    ax3 = fig.add_subplot(1, 4, 3)
    for gi, col in zip([0,1,2], colours_g):
        mask = graph_idx == gi
        ax3.scatter(np.where(mask)[0], sigma_all[mask], color=col, s=50, label=f'G{gi}', zorder=3)
    ax3.set_xlabel('node idx', fontsize=9); ax3.set_ylabel('σ', fontsize=9)
    ax3.legend(frameon=False, fontsize=7)
    clean_ax(ax3, grid=True)

    # Chart 4 (3D) – 3D bars: node, graph_idx, sigma
    ax4 = fig.add_subplot(1, 4, 4, projection='3d')
    ax4.set_facecolor('#ffffff')
    for ni in range(len(sigma_all)):
        gi = graph_idx[ni]
        node_i = ni
        ax4.bar3d(node_i - 0.3, gi - 0.3, 0, 0.6, 0.6, sigma_all[ni],
                  color=colours_g[gi], alpha=0.7, shade=True)
    ax4.set_xlabel('node', fontsize=7); ax4.set_ylabel('graph', fontsize=7); ax4.set_zlabel('σ', fontsize=7)
    ax4.tick_params(labelsize=6)

    plt.tight_layout()
    path = os.path.join(OUT, 'mafg_panel_03.png')
    fig.savefig(path, dpi=150, facecolor='#ffffff', bbox_inches='tight')
    plt.close(fig)
    return path

# ── PANEL 4 ──────────────────────────────────────────────────────────────────
def panel_04():
    fig = new_fig()

    bmin_r  = np.array([0.6, 0.678, 0.618, 0.642, 0.560, 0.574])
    sigma_a = np.array([4.2, 4.28, 4.60, 4.67, 4.84, 4.64])
    tw      = np.array([12.6]*6)
    idx     = np.arange(6)

    # Chart 1 – bmin across reshufficings
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.plot(idx, bmin_r, 'o-', color=C1, lw=1.5, ms=5)
    ax1.axhline(0.6, color=C2, lw=1.0, ls='--', label='floor=0.6')
    ax1.set_xlabel('reshuffling', fontsize=9); ax1.set_ylabel('b_min', fontsize=9)
    ax1.legend(frameon=False, fontsize=8)
    clean_ax(ax1, grid=True)

    # Chart 2 – sigma_a
    ax2 = fig.add_subplot(1, 4, 2)
    ax2.plot(idx, sigma_a, 's-', color=C3, lw=1.5, ms=5)
    ax2.set_xlabel('reshuffling', fontsize=9); ax2.set_ylabel('σ_a', fontsize=9)
    clean_ax(ax2)

    # Chart 3 – total_weight (conserved)
    ax3 = fig.add_subplot(1, 4, 3)
    ax3.plot(idx, tw, 'D-', color=C4, lw=1.5, ms=5)
    ax3.set_ylim(0, 15)
    ax3.set_xlabel('reshuffling', fontsize=9); ax3.set_ylabel('total weight', fontsize=9)
    clean_ax(ax3)

    # Chart 4 (3D) – parametric (reshuffling_idx, bmin, sigma_a)
    ax4 = fig.add_subplot(1, 4, 4, projection='3d')
    ax4.set_facecolor('#ffffff')
    ax4.plot(idx, bmin_r, sigma_a, 'o-', color=C5, lw=1.5, ms=5)
    for xi, bi, si in zip(idx, bmin_r, sigma_a):
        ax4.scatter([xi], [bi], [si], color=C5, s=30, zorder=5)
    ax4.set_xlabel('idx', fontsize=7); ax4.set_ylabel('b_min', fontsize=7); ax4.set_zlabel('σ_a', fontsize=7)
    ax4.tick_params(labelsize=6)

    plt.tight_layout()
    path = os.path.join(OUT, 'mafg_panel_04.png')
    fig.savefig(path, dpi=150, facecolor='#ffffff', bbox_inches='tight')
    plt.close(fig)
    return path

# ── PANEL 5 ──────────────────────────────────────────────────────────────────
def panel_05():
    fig = new_fig()

    floor_seq = np.array([0.5, 0.25, 0.167, 0.125, 0.1, 0.083, 0.071, 0.063,
                          0.056, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
    floors_exp = np.array([0.5]*16)
    stable_fl   = np.array([1.0]*10)
    dissolving  = 1.0 / (np.arange(1, 21))  # approximation to given 1/n pattern

    # Chart 1 – floor_sequence
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.plot(range(16), floor_seq, 'o-', color=C1, lw=1.5, ms=4)
    ax1.axhline(0.05, color=C2, lw=1.0, ls='--', label='floor=0.05')
    ax1.set_xlabel('step', fontsize=9); ax1.set_ylabel('floor', fontsize=9)
    ax1.legend(frameon=False, fontsize=8)
    clean_ax(ax1, grid=True)

    # Chart 2 – floors_across_expansions (constant)
    ax2 = fig.add_subplot(1, 4, 2)
    ax2.plot(range(16), floors_exp, 'o-', color=C3, lw=1.5, ms=4)
    ax2.set_ylim(0, 1.2)
    ax2.set_xlabel('expansion', fontsize=9); ax2.set_ylabel('floor', fontsize=9)
    clean_ax(ax2)

    # Chart 3 – stable vs dissolving
    ax3 = fig.add_subplot(1, 4, 3)
    ax3.plot(range(10), stable_fl, 'o-', color=C1, lw=1.5, ms=4, label='stable')
    ax3.plot(range(20), dissolving, 's-', color=C2, lw=1.5, ms=3, label='dissolving')
    ax3.set_xlabel('n', fontsize=9); ax3.set_ylabel('floor', fontsize=9)
    ax3.legend(frameon=False, fontsize=8)
    clean_ax(ax3)

    # Chart 4 (3D) – surface floor(n_exp, init_w)
    n_exp_arr  = np.linspace(0, 15, 40)
    init_w_arr = np.linspace(0.5, 2.0, 40)
    NE, IW     = np.meshgrid(n_exp_arr, init_w_arr)
    min_asy    = 0.05
    FLOOR_S    = np.maximum(IW / (NE + 1), min_asy)
    ax4 = fig.add_subplot(1, 4, 4, projection='3d')
    ax4.set_facecolor('#ffffff')
    ax4.plot_surface(NE, IW, FLOOR_S, cmap='Blues', edgecolor='none', alpha=0.85)
    ax4.set_xlabel('n_exp', fontsize=7); ax4.set_ylabel('init_w', fontsize=7); ax4.set_zlabel('floor', fontsize=7)
    ax4.tick_params(labelsize=6)

    plt.tight_layout()
    path = os.path.join(OUT, 'mafg_panel_05.png')
    fig.savefig(path, dpi=150, facecolor='#ffffff', bbox_inches='tight')
    plt.close(fig)
    return path

# ── PANEL 6 ──────────────────────────────────────────────────────────────────
def panel_06():
    fig = new_fig()

    cut_w_10 = np.array([3.6, 3.824, 3.761, 3.988, 4.082, 4.185, 4.050, 4.108, 4.006, 4.013])
    common_lb = 3.6

    sizes_25  = np.arange(2, 27)
    sigmas_25 = np.array([2.5,3.0,3.333,3.583,3.783,3.95,4.093,4.218,4.329,4.429,
                          4.529,4.629,4.729,4.829,4.929,5.029,5.129,5.229,5.329,5.429,
                          5.529,5.629,5.729,5.829,5.929])

    gens     = np.arange(6)
    not_seq  = np.array([4, 7, 10, 13, 16, 19])

    # Chart 1 – cut_weights + lower bound
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.plot(range(10), cut_w_10, 'o-', color=C1, lw=1.5, ms=5)
    ax1.axhline(common_lb, color=C2, lw=1.0, ls='--', label=f'lb={common_lb}')
    ax1.set_xlabel('step', fontsize=9); ax1.set_ylabel('cut weight', fontsize=9)
    ax1.legend(frameon=False, fontsize=8)
    clean_ax(ax1, grid=True)

    # Chart 2 – sigma vs NOT-sequence size
    ax2 = fig.add_subplot(1, 4, 2)
    ax2.plot(sizes_25, sigmas_25, color=C3, lw=1.5)
    ax2.set_xlabel('NOT-seq size', fontsize=9); ax2.set_ylabel('σ', fontsize=9)
    clean_ax(ax2)

    # Chart 3 – not_sequence_grows: size vs generation
    ax3 = fig.add_subplot(1, 4, 3)
    ax3.plot(gens, not_seq, 'o-', color=C4, lw=1.5, ms=5)
    # linear fit
    m_fit, b_fit = np.polyfit(gens, not_seq, 1)
    ax3.plot(gens, m_fit*gens + b_fit, '--', color=C2, lw=1.0, label=f'slope={m_fit:.1f}')
    ax3.set_xlabel('generation', fontsize=9); ax3.set_ylabel('NOT-seq size', fontsize=9)
    ax3.legend(frameon=False, fontsize=8)
    clean_ax(ax3, grid=True)

    # Chart 4 (3D) – helix: (NOT-seq_size, sigma, generation)
    # Expand to continuous for helix look
    gen_ext  = np.repeat(gens, 4) + np.tile(np.linspace(0, 0.9, 4), 6)
    ns_ext   = np.interp(gen_ext, gens, not_seq)
    sig_ext  = np.interp(ns_ext, sizes_25, sigmas_25)
    ax4 = fig.add_subplot(1, 4, 4, projection='3d')
    ax4.set_facecolor('#ffffff')
    ax4.plot(ns_ext, sig_ext, gen_ext, color=C5, lw=1.5)
    ax4.scatter(not_seq, [sigmas_25[ns-2] for ns in not_seq], gens, color=C2, s=30, zorder=5)
    ax4.set_xlabel('NOT-seq', fontsize=7); ax4.set_ylabel('σ', fontsize=7); ax4.set_zlabel('gen', fontsize=7)
    ax4.tick_params(labelsize=6)

    plt.tight_layout()
    path = os.path.join(OUT, 'mafg_panel_06.png')
    fig.savefig(path, dpi=150, facecolor='#ffffff', bbox_inches='tight')
    plt.close(fig)
    return path

# ── PANEL 7 ──────────────────────────────────────────────────────────────────
def panel_07():
    fig = new_fig()

    gens_full    = np.arange(21)
    mass_edge    = np.array([6.941]*21)
    # frag_floor at gens 0,5,10,15,20 → interpolate
    frag_gens    = np.array([0, 5, 10, 15, 20])
    frag_floor_s = np.array([3.5, 3.75, 4.0, 4.25, 4.5])
    frag_interp  = np.interp(gens_full, frag_gens, frag_floor_s)

    sigma_6gens  = np.array([5.0, 7.25, 8.75, 9.875, 10.775, 11.525])
    gens_6       = np.arange(6)

    sigma_scale  = np.array([6.941 + 2*i for i in range(11)])
    cut_sizes    = np.arange(1, 12)

    # Chart 1 – mass_edge and frag_floor vs generation
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.plot(gens_full, mass_edge,  color=C1, lw=1.5, label='mass_edge')
    ax1.plot(gens_full, frag_interp, color=C2, lw=1.5, ls='--', label='frag_floor')
    ax1.set_xlabel('generation', fontsize=9)
    ax1.legend(frameon=False, fontsize=8)
    clean_ax(ax1)

    # Chart 2 – sigma vs generation
    ax2 = fig.add_subplot(1, 4, 2)
    ax2.plot(gens_6, sigma_6gens, 'o-', color=C3, lw=1.5, ms=5)
    ax2.axhline(sigma_6gens[0], color='k', lw=0.8, ls='--', label='floor')
    ax2.set_xlabel('generation', fontsize=9); ax2.set_ylabel('σ', fontsize=9)
    ax2.legend(frameon=False, fontsize=8)
    clean_ax(ax2, grid=True)

    # Chart 3 – cut_size vs sigma
    ax3 = fig.add_subplot(1, 4, 3)
    ax3.scatter(sigma_scale, cut_sizes, color=C4, s=50, zorder=3)
    ax3.plot(sigma_scale, cut_sizes, color=C4, lw=1.0, alpha=0.5)
    ax3.set_xlabel('σ', fontsize=9); ax3.set_ylabel('cut size', fontsize=9)
    clean_ax(ax3, grid=True)

    # Chart 4 (3D) – surface sigma(generation, bmin)
    bmin_arr = np.linspace(0.3, 1.0, 30)
    gen_arr  = np.linspace(0, 5, 30)
    BM, GN   = np.meshgrid(bmin_arr, gen_arr)
    SIGMA_GEN = sigma_6gens[0] + (sigma_6gens[-1]-sigma_6gens[0]) * (GN/5) * (BM/0.3)**0.5
    ax4 = fig.add_subplot(1, 4, 4, projection='3d')
    ax4.set_facecolor('#ffffff')
    ax4.plot_surface(GN, BM, SIGMA_GEN, cmap='plasma', edgecolor='none', alpha=0.85)
    ax4.set_xlabel('gen', fontsize=7); ax4.set_ylabel('b_min', fontsize=7); ax4.set_zlabel('σ', fontsize=7)
    ax4.tick_params(labelsize=6)

    plt.tight_layout()
    path = os.path.join(OUT, 'mafg_panel_07.png')
    fig.savefig(path, dpi=150, facecolor='#ffffff', bbox_inches='tight')
    plt.close(fig)
    return path

# ── PANEL 8 ──────────────────────────────────────────────────────────────────
def panel_08():
    fig = new_fig()

    elements   = ['Li', 'Na', 'F', 'Cl', 'He', 'Ne']
    sep_costs  = np.array([13.241, 27.79, 27.398, 40.553, 10.503, 23.28])
    cat_colours = [C1, C1, C2, C2, C3, C3]  # alkali=blue, halogen=coral, noble=green

    step_floors = np.array([0.5, 0.4, 0.3, 0.2, 0.15])
    chain_floor = 0.15
    steps_idx   = np.arange(5)

    # sigma_k = chain_floor + sum of floors up to step k
    cumsum_fl = np.cumsum(step_floors)
    sigma_k   = chain_floor + cumsum_fl

    # Chart 1 – horizontal bar: separation costs coloured by category
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.barh(elements, sep_costs, color=cat_colours, height=0.6)
    ax1.set_xlabel('separation cost', fontsize=9)
    from matplotlib.patches import Patch
    legend_els = [Patch(color=C1, label='alkali'), Patch(color=C2, label='halogen'),
                  Patch(color=C3, label='noble')]
    ax1.legend(handles=legend_els, frameon=False, fontsize=7)
    clean_ax(ax1)

    # Chart 2 – step_floors staircase
    ax2 = fig.add_subplot(1, 4, 2)
    ax2.step(steps_idx, step_floors, where='post', color=C1, lw=1.5)
    ax2.plot(steps_idx, step_floors, 'o', color=C1, ms=5)
    ax2.axhline(chain_floor, color=C2, lw=1.0, ls='--', label=f'chain_floor={chain_floor}')
    ax2.set_xlabel('step', fontsize=9); ax2.set_ylabel('floor', fontsize=9)
    ax2.legend(frameon=False, fontsize=8)
    clean_ax(ax2, grid=True)

    # Chart 3 – bar chart sigma_k at each step
    ax3 = fig.add_subplot(1, 4, 3)
    ax3.bar(steps_idx, sigma_k, color=C3, width=0.5)
    ax3.axhline(chain_floor, color=C2, lw=1.0, ls='--', label=f'chain_floor')
    ax3.set_xlabel('step', fontsize=9); ax3.set_ylabel('σ_k', fontsize=9)
    ax3.set_xticks(steps_idx)
    ax3.legend(frameon=False, fontsize=8)
    clean_ax(ax3)

    # Chart 4 (3D) – 3D funnel: (step, floor_k, cumulative_sigma)
    ax4 = fig.add_subplot(1, 4, 4, projection='3d')
    ax4.set_facecolor('#ffffff')
    ax4.scatter(steps_idx, step_floors, sigma_k, c=np.arange(5), cmap='viridis', s=80, zorder=5)
    ax4.plot(steps_idx, step_floors, sigma_k, color=C5, lw=1.5)
    # vertical drops to floor
    for xi, fi, si in zip(steps_idx, step_floors, sigma_k):
        ax4.plot([xi, xi], [fi, fi], [fi, si], color='#cccccc', lw=0.8)
    ax4.set_xlabel('step', fontsize=7); ax4.set_ylabel('floor', fontsize=7); ax4.set_zlabel('σ_cum', fontsize=7)
    ax4.tick_params(labelsize=6)

    plt.tight_layout()
    path = os.path.join(OUT, 'mafg_panel_08.png')
    fig.savefig(path, dpi=150, facecolor='#ffffff', bbox_inches='tight')
    plt.close(fig)
    return path

if __name__ == '__main__':
    panels = [panel_01, panel_02, panel_03, panel_04,
              panel_05, panel_06, panel_07, panel_08]
    for fn in panels:
        p = fn()
        print(f"  saved {os.path.basename(p)}")
    print("Done: 8 panels saved")
