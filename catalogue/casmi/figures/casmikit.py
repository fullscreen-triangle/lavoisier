"""casmikit.py --- shared styling for the CASMI panels.

Same contract as catalogue/validation/figures/panelkit.py: four charts in
a row on a white ground, at least one three-dimensional, and every mark a
number measured from the 58 CASMI challenges resolvable in the seventeen
local mzML files.  No panel contains a table, a text-only cell, or a
conceptual diagram.

Palette is the dataviz reference instance in fixed slot order.  Aqua sits
below the 3:1 contrast gate against white, so the relief rule applies:
any series carrying aqua is also named in a legend or a direct label.
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, ".."))
PAPER = os.path.abspath(os.path.join(
    HERE, "..", "..", "publications", "uc-davis-casmi-catalogue"))

C1 = "#2a78d6"   # blue
C2 = "#eb6834"   # orange
C3 = "#1baf7a"   # aqua  (relief rule: always directly labelled)
C4 = "#4a3aa7"   # violet
SERIES = [C1, C2, C3, C4]

GOOD = "#1baf7a"
BAD = "#e34948"

INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8985"
GRID = "#e3e3e0"
SURFACE = "#ffffff"

# Verdict -> colour.  Licensed is the blue slot, the two decline classes
# take orange and violet.  Never the status colours: a decline is not a
# failure, it is a measurement.
VCOL = {
    "licensed": C1,
    "decline-ambiguous": C2,
    "decline-unsupported": C4,
}
VLAB = {
    "licensed": "licensed",
    "decline-ambiguous": "decline (ambiguous)",
    "decline-unsupported": "decline (unsupported)",
}
VORDER = ["licensed", "decline-ambiguous", "decline-unsupported"]


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK2,
        "axes.titlesize": 9.5,
        "axes.titleweight": "medium",
        "axes.titlecolor": INK,
        "axes.labelsize": 8.5,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.0,
        "legend.frameon": False,
        "lines.linewidth": 2.0,
        "lines.markersize": 5.0,
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.14,
    })


def panel(three_d=(), figsize=(15.2, 3.9)):
    style()
    fig = plt.figure(figsize=figsize)
    axes = []
    for i in range(4):
        if i in three_d:
            ax = fig.add_subplot(1, 4, i + 1, projection="3d")
            ax.set_facecolor(SURFACE)
            for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
                pane.pane.set_facecolor(SURFACE)
                pane.pane.set_edgecolor(GRID)
                pane.pane.set_alpha(1.0)
            ax.grid(True, color=GRID, linewidth=0.5)
            ax.tick_params(labelsize=6.5, colors=INK2, pad=0.5)
        else:
            ax = fig.add_subplot(1, 4, i + 1)
        axes.append(ax)
    fig.subplots_adjust(wspace=0.42 if three_d else 0.32,
                        left=0.035, right=0.985)
    return fig, axes


def tag(ax, letter, three_d=False):
    if three_d:
        ax.text2D(-0.06, 1.06, letter, transform=ax.transAxes,
                  fontsize=10, fontweight="bold", color=INK, va="top")
    else:
        ax.text(-0.16, 1.08, letter, transform=ax.transAxes,
                fontsize=10, fontweight="bold", color=INK, va="top")


def load():
    with open(os.path.join(DATA, "panel_data.json"), encoding="utf8") as fh:
        return json.load(fh)


def save(fig, stem):
    out = os.path.join(PAPER, "figures")
    if not os.path.isdir(out):
        os.makedirs(out)
    p = os.path.join(out, stem + ".png")
    fig.savefig(p)
    plt.close(fig)
    print("  wrote", stem + ".png")
    return p
