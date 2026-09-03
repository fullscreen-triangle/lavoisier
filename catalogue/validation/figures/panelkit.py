"""
panelkit.py --- shared styling for the manuscript validation panels.

Every panel is four charts in a row on a white ground, at least one of
them three-dimensional, and none of them conceptual: each mark is a
number a validation experiment measured and wrote to
validation/results/*.json. Nothing here invents data, and no panel
contains a table or a text-only cell.

The categorical palette is the dataviz reference instance, used in its
fixed slot order and validated with the skill's own checker rather than
by eye:

    slot 1 blue    #2a78d6
    slot 2 orange  #eb6834
    slot 3 aqua    #1baf7a
    slot 4 violet  #4a3aa7

Checked with `validate_palette.js --mode light --pairs all` (the
all-pairs list, since several panels are scatter/3-D rather than stacked
bars): CVD dE 9.2 worst pair, normal-vision dE 16.3 worst pair, both
above their floors. Aqua sits at 2.74:1 against the white surface, below
the 3:1 contrast gate, so the relief rule applies --- every series
carrying aqua is also named in a legend or a direct label, never
identified by colour alone.

PASS/FAIL status uses the reserved status colours, never a categorical
slot, and always ships with a word or a marker shape as well as a hue.
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))
PUBS = os.path.abspath(os.path.join(HERE, "..", "..", "publications"))

# Categorical slots, fixed order, never cycled.
C1 = "#2a78d6"   # blue
C2 = "#eb6834"   # orange
C3 = "#1baf7a"   # aqua   (relief rule: always directly labelled)
C4 = "#4a3aa7"   # violet
SERIES = [C1, C2, C3, C4]

# Reserved status colours. Never reused as a fifth series.
GOOD = "#1baf7a"
BAD = "#e34948"

INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8985"
GRID = "#e3e3e0"
SURFACE = "#ffffff"


def style():
    """Print-figure rcParams: white ground, recessive grid and axes."""
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
        "legend.fontsize": 7.5,
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
    """A row of four axes on white. `three_d` lists which of 0..3 are 3-D.

    Returns (fig, [ax0..ax3]).
    """
    style()
    fig = plt.figure(figsize=figsize)
    axes = []
    for i in range(4):
        if i in three_d:
            ax = fig.add_subplot(1, 4, i + 1, projection="3d")
            ax.set_facecolor(SURFACE)
            # Recessive panes: the data should carry the ink, not the box.
            for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
                pane.pane.set_facecolor(SURFACE)
                pane.pane.set_edgecolor(GRID)
                pane.pane.set_alpha(1.0)
            ax.grid(True, color=GRID, linewidth=0.5)
            ax.tick_params(labelsize=6.5, colors=INK2, pad=0.5)
        else:
            ax = fig.add_subplot(1, 4, i + 1)
        axes.append(ax)
    # 3-D axes need a wider gutter: their z-label sits outside the axes
    # box and collides with the next panel's y-label at the default
    # spacing. Set it here so every panel inherits the fix.
    fig.subplots_adjust(wspace=0.42 if three_d else 0.32,
                        left=0.035, right=0.985)
    return fig, axes


def tag(ax, letter, three_d=False):
    """Panel letter, top-left, in ink rather than a series colour."""
    if three_d:
        ax.text2D(-0.06, 1.06, letter, transform=ax.transAxes,
                  fontsize=10, fontweight="bold", color=INK, va="top")
    else:
        ax.text(-0.16, 1.08, letter, transform=ax.transAxes,
                fontsize=10, fontweight="bold", color=INK, va="top")


def load(name):
    with open(os.path.join(RESULTS, name + ".json"), encoding="utf8") as fh:
        return json.load(fh)


def save(fig, paper, stem):
    """Write the panel into the paper's own figures/ directory."""
    out = os.path.join(PUBS, paper, "figures")
    if not os.path.isdir(out):
        os.makedirs(out)
    p = os.path.join(out, stem + ".png")
    fig.savefig(p)
    plt.close(fig)
    print("  wrote", os.path.relpath(p, PUBS))
    return p
