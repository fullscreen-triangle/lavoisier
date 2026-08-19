"""Shared style and data loading for the Shapeshifter panels.

Every panel: white background, four charts in a row, at least one 3-D,
minimal text, no tables and no conceptual diagrams.
"""
from __future__ import annotations

import json
import statistics as st
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------- style

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "lines.linewidth": 1.6,
    "legend.frameon": False,
    "legend.fontsize": 7.5,
})

# Palette: colour-blind safe, consistent across all six panels.
BLUE = "#2166AC"
RED = "#B2182B"
GREEN = "#1B7837"
ORANGE = "#E08214"
PURPLE = "#762A83"
GREY = "#808080"
LIGHT = "#BFBFBF"

AXCOL = {"s_k": BLUE, "s_t": GREEN, "s_e": RED}
AXLAB = {"s_k": r"$S_k$", "s_t": r"$S_t$", "s_e": r"$S_e$"}

NCE = [10, 15, 20, 25, 30, 40, 50, 60, 80]


def new_panel(n=4, w=15.2, h=3.7, d3=()):
    """Figure with n axes in a row; indices in d3 are 3-D projections."""
    fig = plt.figure(figsize=(w, h))
    axes = []
    for i in range(n):
        if i in d3:
            ax = fig.add_subplot(1, n, i + 1, projection="3d")
            ax.set_facecolor("white")
            try:
                for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
                    pane.pane.set_facecolor("white")
                    pane.pane.set_edgecolor(LIGHT)
                    pane.pane.set_alpha(1.0)
                ax.grid(True, color=LIGHT, linewidth=0.4)
            except Exception:
                pass
        else:
            ax = fig.add_subplot(1, n, i + 1)
            ax.grid(True, color=LIGHT, linewidth=0.4, alpha=0.6)
            ax.set_axisbelow(True)
        axes.append(ax)
    return fig, axes


def tag(ax, letter, d3=False):
    """Single-letter subplot tag, no other text furniture."""
    if d3:
        ax.text2D(-0.06, 1.03, letter, transform=ax.transAxes,
                  fontsize=11, fontweight="bold", va="top")
    else:
        ax.text(-0.12, 1.04, letter, transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top")


def save(fig, path):
    fig.subplots_adjust(wspace=0.34)
    try:
        fig.tight_layout(w_pad=2.2)
    except Exception:
        pass
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", path)


# ---------------------------------------------------------------- data

import os
_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def _res(name):
    return os.path.join(_ROOT, "results", name)


def load():
    coords = json.load(open(_res("_coords_full.json")))
    meta = json.load(open(_res("_scans_meta.json")))
    return coords, meta


def groups(coords, min_n=9):
    g = defaultdict(list)
    for c in coords:
        g[c["compound"]].append(c)
    return {k: v for k, v in g.items() if len(v) >= min_n}


def axis_stats(full):
    """Per-axis within-sd, between-sd, ratio."""
    out = {}
    for a in ("s_k", "s_t", "s_e"):
        wit = [st.pstdev([i[a] for i in v]) for v in full.values()]
        cen = [st.mean([i[a] for i in v]) for v in full.values()]
        out[a] = (st.mean(wit), st.pstdev(cen),
                  st.pstdev(cen) / st.mean(wit))
    return out


def by_nce(coords, key):
    return [st.mean([c[key] for c in coords if c["nce"] == n]) for n in NCE]


def sd_by_nce(coords, key):
    return [st.pstdev([c[key] for c in coords if c["nce"] == n]) for n in NCE]


def fit(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m, b = np.polyfit(x, y, 1)
    r = np.corrcoef(x, y)[0, 1]
    return m, b, r
