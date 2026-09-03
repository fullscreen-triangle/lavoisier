"""
Panels 3 and 4 for "Coordinates Without Provenance".

These spend the records panels 1 and 2 left: `minimal_record`,
`minimal_record_factorisation`, `comparability`, `decline_sound` and
`record_floor_control`. Every one of them is a scalar or a boolean at a
single fixed cell (|X| = 3, |Ctx| = 3, two contexts collapsed), so
plotting the records alone would give four bar-per-number charts.

Instead the experiment's own definitions --- `rec_min`,
`sufficient_for_comparison`, `behaviour`, `all_maps` --- are evaluated
over a FAMILY generated the way exp3 generated its instance: nc contexts
of which the first k behave identically. exp3's recorded cell is
(nc = 3, k = 2), and it is marked in every chart, where it reproduces the
recorded numbers exactly:

    sufficient_maps_with_smaller_image      0   -> panel 3 (a), (b)
    sufficient_maps_not_factoring           6   -> panel 3 (c)
    declined_context_pairs                  4   -> panel 4 (a)
    total_comparator_unlicensed_answers     4   -> panel 4 (a), (b)
    partial_comparator_unlicensed_answers   0   -> panel 4 (a), (b)

The sweep also shows the finding is not local to that cell: the
image-size bound holds in every cell (smaller image: 0 everywhere),
while the printed factorisation fails in every cell with a nontrivial
collapse, and by a growing margin.
"""
from __future__ import annotations

import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..")))

from exp3_coordinate_provenance import (          # noqa: E402
    all_maps, behaviour, rec_min, sufficient_for_comparison)
from panelkit import (C1, C2, C3, C4, GOOD, BAD, INK, INK2, MUTED,   # noqa: E402
                      panel, tag, load, save)

PAPER = "coordinate-provenance"
D = load("exp3_coordinate_provenance")
R = D["records"]

X = ["x1", "x2", "x3"]                     # exp3's measurement set
NCS = list(range(2, 6))                    # context counts swept
CELL = (3, 2)                              # exp3's own recorded cell


def family(nc, k):
    """exp3's instance shape: nc contexts, the first k behaving alike."""
    C = ["c%d" % (i + 1) for i in range(nc)]
    phi = {}
    for x in X:
        for i, c in enumerate(C):
            phi[(x, c)] = ("v" + x) if i < k else ("w%d" % i) + x
    return C, phi


def record_stats(nc, k):
    """Sufficiency, minimality and factorisation over every record map."""
    C, phi = family(nc, k)
    rmin = rec_min(phi, X, C)
    ncl = len(set(rmin.values()))
    total = suff = smaller = split = 0
    for size in range(1, nc + 1):
        for rec in all_maps(C, list(range(size))):
            total += 1
            if not sufficient_for_comparison(rec, phi, X, C):
                continue
            suff += 1
            if len(set(rec.values())) < ncl:
                smaller += 1
            seen, factors = {}, True
            for c in C:
                if rmin[c] in seen and seen[rmin[c]] != rec[c]:
                    factors = False
                    break
                seen[rmin[c]] = rec[c]
            if not factors:
                split += 1
    return {"classes": ncl, "maps": total, "sufficient": suff,
            "smaller": smaller, "not_factoring": split}


def comparator_stats(nc, k):
    """The total comparator against the declining one, same data."""
    C, phi = family(nc, k)
    rmin = rec_min(phi, X, C)
    tw = pw = dec = adm = 0
    for c1, c2 in itertools.product(C, C):
        agree = behaviour(phi, X, c1) == behaviour(phi, X, c2)
        if not agree:
            tw += 1
        if rmin[c1] == rmin[c2]:
            adm += 1
            if not agree:
                pw += 1
        else:
            dec += 1
    return {"total_wrong": tw, "partial_wrong": pw, "declined": dec,
            "admitted": adm}


REC = {(nc, k): record_stats(nc, k)
       for nc in NCS for k in range(1, nc + 1)}
CMP = {(nc, k): comparator_stats(nc, k)
       for nc in NCS for k in range(1, nc + 1)}

# The sweep must reproduce the artefact at exp3's own cell, or it is
# measuring a different family than the one the experiment ran.
assert REC[CELL]["smaller"] == R["minimal_record"][
    "sufficient_maps_with_smaller_image"]
assert REC[CELL]["not_factoring"] == R["minimal_record_factorisation"][
    "sufficient_maps_not_factoring_through_rec_min"]
assert REC[CELL]["classes"] == R["minimal_record"]["n_equivalence_classes"]
assert CMP[CELL]["declined"] == R["comparability"]["declined_context_pairs"]
assert CMP[CELL]["total_wrong"] == R["decline_sound"][
    "total_comparator_unlicensed_answers"]
assert CMP[CELL]["partial_wrong"] == R["decline_sound"][
    "partial_comparator_unlicensed_answers"]


# =====================================================================
# Panel 3 --- the record is minimal, but it does not factor as printed
# =====================================================================
def panel3():
    fig, ax = panel(three_d=(1,))

    ks = list(range(1, 6))

    # (a) how many record maps are sufficient at all, against how many
    # exist. Sufficiency is rare and gets rarer as contexts multiply.
    a = ax[0]
    for i, nc in enumerate(NCS):
        kk = list(range(1, nc + 1))
        a.plot(kk, [REC[(nc, k)]["sufficient"] for k in kk], "-o",
               color=[C1, C2, C4, MUTED][i], label="|Ctx| = %d" % nc,
               markersize=4)
    a.set_yscale("log")
    a.set_xticks(ks)
    a.set_xlabel("contexts collapsed  $k$")
    a.set_ylabel("sufficient record maps")
    a.set_title("Sufficiency thins as contexts split")
    a.legend(loc="upper left", ncol=2)
    a.scatter([CELL[1]], [REC[CELL]["sufficient"]], s=110,
              facecolor="white", edgecolor=BAD, linewidth=1.7, zorder=6)
    a.text(0.97, 0.05, "circle: the recorded cell", transform=a.transAxes,
           ha="right", fontsize=7, color=BAD)
    tag(a, "a")

    # (b) 3-D: sufficient record maps with an image SMALLER than
    # |Ctx/=|. thm:minimal-record says there are none, and the surface
    # is a floor at zero across the whole family --- the bound the
    # proof actually establishes holds everywhere, not just at the
    # recorded cell.
    b = ax[1]
    KK, NN = np.meshgrid(np.arange(1, 6), np.array(NCS), indexing="ij")
    Zs = np.full(KK.shape, np.nan)
    Zf = np.full(KK.shape, np.nan)
    for i, k in enumerate(range(1, 6)):
        for j, nc in enumerate(NCS):
            if k <= nc:
                Zs[i, j] = REC[(nc, k)]["smaller"]
                Zf[i, j] = REC[(nc, k)]["not_factoring"]
    b.view_init(elev=24, azim=-58)
    b.plot_surface(KK, NN, np.log10(np.where(Zf > 0, Zf, np.nan)),
                   cmap="Blues", linewidth=0, alpha=0.78,
                   rstride=1, cstride=1)
    b.plot_surface(KK, NN, np.where(np.isnan(Zs), np.nan, Zs),
                   color=GOOD, alpha=0.55, linewidth=0, shade=False)
    b.set_xlabel("collapsed  $k$", labelpad=-2)
    b.set_ylabel("|Ctx|", labelpad=-2)
    b.set_xticks(range(1, 6))
    b.set_yticks(NCS)
    b.set_zlabel("")
    b.set_title("Green floor: 0 smaller images, everywhere", y=1.04)
    b.text2D(0.00, 0.80, "blue: $\\log_{10}$ maps that do not" + chr(10)
             + "factor through the quotient", transform=b.transAxes,
             fontsize=6.4, color=C1)
    b.set_box_aspect(None, zoom=1.15)
    tag(b, "b", three_d=True)

    # (c) the two claims side by side across the family: the image-size
    # bound (0 in every cell) and the printed factorisation (fails in
    # every cell with a nontrivial collapse).
    c = ax[2]
    cells = [(nc, k) for nc in NCS for k in range(1, nc + 1)]
    xs = np.arange(len(cells))
    nf = [REC[cc]["not_factoring"] for cc in cells]
    sm = [REC[cc]["smaller"] for cc in cells]
    c.bar(xs, nf, width=0.62, color=BAD, zorder=3,
          label="do not factor (claim fails)")
    c.plot(xs, sm, "-o", color=GOOD, markersize=4, zorder=4,
           label="smaller image (bound holds)")
    c.set_yscale("symlog", linthresh=1.0)
    c.set_xticks(xs)
    c.set_xticklabels(["%d,%d" % cc for cc in cells], fontsize=5.6,
                      rotation=90)
    c.set_xlabel("cell  (|Ctx|, $k$)")
    c.set_ylabel("record maps")
    c.set_title("The bound holds; the factorisation does not")
    c.legend(loc="upper left")
    jc = cells.index(CELL)
    c.scatter([xs[jc]], [nf[jc]], s=95, facecolor="white", edgecolor=INK,
              linewidth=1.5, zorder=7)
    c.annotate("recorded: %d" % nf[jc], xy=(xs[jc], nf[jc]),
               xytext=(0.30, 0.60), textcoords="axes fraction",
               fontsize=7, color=INK,
               arrowprops=dict(arrowstyle="->", color=INK, lw=0.8,
                               shrinkB=8))
    tag(c, "c")

    # (d) where thm:minimal-record actually bites: the image sizes that
    # sufficient record maps take. The first version of this chart
    # plotted classes against k, but that is the arithmetic identity
    # |Ctx/=| = nc - k + 1 in every series --- a straight line, not a
    # measurement. The distribution is the measured quantity: its
    # minimum is the class count in every cell, and nothing lies to the
    # left of it, which is the theorem.
    # Swept at the widest context count, over every collapse depth: one
    # curve per k, each ending abruptly at its own class count. (Holding
    # k fixed and varying |Ctx| instead mixes collapse regimes --- at
    # k = 2 the family with |Ctx| = 2 is totally collapsed while the one
    # with |Ctx| = 5 is barely collapsed --- and gives fragments of two
    # or three points rather than distributions.)
    d = ax[3]
    ncd = NCS[-1]
    cols = [C1, C2, C4, MUTED, INK2]
    for i, k in enumerate(range(1, ncd + 1)):
        C, phi = family(ncd, k)
        rmin = rec_min(phi, X, C)
        ncl = len(set(rmin.values()))
        hist = {}
        for size in range(1, ncd + 1):
            for rec in all_maps(C, list(range(size))):
                if sufficient_for_comparison(rec, phi, X, C):
                    im = len(set(rec.values()))
                    hist[im] = hist.get(im, 0) + 1
        xsz = sorted(hist)
        d.plot(xsz, [hist[s] for s in xsz], "-o", color=cols[i],
               markersize=4, label="$k$ = %d  (min %d)" % (k, ncl))
        d.scatter([ncl], [hist[ncl]], s=100, facecolor="white",
                  edgecolor=cols[i], linewidth=1.6, zorder=6)
    d.set_yscale("log")
    d.set_xticks(ks)
    d.set_xlabel("image size of the record map")
    d.set_ylabel("sufficient record maps")
    d.set_title("No sufficient map is smaller than the quotient")
    d.legend(loc="upper left", fontsize=6.6, ncol=2)
    d.text(0.97, 0.05, "|Ctx| = %d; circles: each minimum, at $|Ctx/{\\sim}|$"
           % ncd, transform=d.transAxes, ha="right", fontsize=6.8,
           color=INK2)
    tag(d, "d")

    save(fig, PAPER, "panel3_minimality")


# =====================================================================
# Panel 4 --- declining beats answering, measured against a total one
# =====================================================================
def panel4():
    fig, ax = panel(three_d=(2,))

    cells = [(nc, k) for nc in NCS for k in range(1, nc + 1)]
    xs = np.arange(len(cells))

    # (a) the two comparators on identical data: a total comparator
    # answers unlicensed comparisons, the declining one answers none.
    a = ax[0]
    tw = [CMP[cc]["total_wrong"] for cc in cells]
    pw = [CMP[cc]["partial_wrong"] for cc in cells]
    a.bar(xs, tw, width=0.62, color=BAD, zorder=3,
          label="total comparator: unlicensed answers")
    a.plot(xs, pw, "-o", color=GOOD, markersize=4, zorder=5,
           label="declining comparator: unlicensed answers")
    a.set_xticks(xs)
    a.set_xticklabels(["%d,%d" % cc for cc in cells], fontsize=5.6,
                      rotation=90)
    a.set_xlabel("cell  (|Ctx|, $k$)")
    a.set_ylabel("comparisons")
    a.set_ylim(-0.8, max(tw) * 1.30)
    a.set_title("Declining is never unlicensed")
    a.legend(loc="upper left")
    ja = cells.index(CELL)
    a.scatter([xs[ja]], [tw[ja]], s=95, facecolor="white", edgecolor=INK,
              linewidth=1.5, zorder=7)
    a.text(0.03, 0.72, "circle: recorded cell" + chr(10) + "(%d vs %d)"
           % (tw[ja], pw[ja]), transform=a.transAxes, ha="left",
           fontsize=7, color=INK2)
    tag(a, "a")

    # (b) what declining costs, as an absolute trade rather than a
    # share. The first version stacked admitted and declined to a
    # constant 1.0 --- a 100%-stacked bar whose upper segment carries no
    # information the lower one does not, over the same cell axis as
    # (a). Plotted against each other instead, every declined pair is
    # one the total comparator would have got wrong: the two series
    # coincide exactly, which is the recorded `declined` == recorded
    # `total_comparator_unlicensed_answers` at every cell, not just at
    # the one exp3 ran.
    b = ax[1]
    dec = np.array([CMP[cc]["declined"] for cc in cells], float)
    adm = np.array([CMP[cc]["admitted"] for cc in cells], float)
    twb = np.array([CMP[cc]["total_wrong"] for cc in cells], float)
    b.plot(xs, adm, "-s", color=C1, markersize=4, label="admitted")
    b.plot(xs, dec, "-o", color=C2, markersize=4, label="declined")
    b.plot(xs, twb, ":", color=BAD, lw=2.4, zorder=5,
           label="wrong if answered anyway")
    b.set_xticks(xs)
    b.set_xticklabels(["%d,%d" % cc for cc in cells], fontsize=5.6,
                      rotation=90)
    b.set_xlabel("cell  (|Ctx|, $k$)")
    b.set_ylabel("context pairs")
    b.set_ylim(-1.0, max(adm.max(), dec.max()) * 1.32)
    b.set_title("Every declined pair is one it would fail")
    b.legend(loc="upper left", fontsize=6.8)
    jb = cells.index(CELL)
    b.scatter([xs[jb]], [dec[jb]], s=95, facecolor="white",
              edgecolor=INK, linewidth=1.5, zorder=7)
    tag(b, "b")

    # (c) 3-D: the total comparator's error surface over the family. It
    # rises with contexts and falls with collapse; the declining
    # comparator's surface is the zero plane beneath it.
    c = ax[2]
    KK, NN = np.meshgrid(np.arange(1, 6), np.array(NCS), indexing="ij")
    Zt = np.full(KK.shape, np.nan)
    Zp = np.full(KK.shape, np.nan)
    for i, k in enumerate(range(1, 6)):
        for j, nc in enumerate(NCS):
            if k <= nc:
                Zt[i, j] = CMP[(nc, k)]["total_wrong"]
                Zp[i, j] = CMP[(nc, k)]["partial_wrong"]
    c.view_init(elev=24, azim=-58)
    c.plot_surface(KK, NN, Zt, cmap="Reds", linewidth=0, alpha=0.78,
                   rstride=1, cstride=1)
    c.plot_surface(KK, NN, Zp, color=GOOD, alpha=0.60, linewidth=0,
                   shade=False)
    c.scatter([CELL[1]], [CELL[0]], [CMP[CELL]["total_wrong"]], s=44,
              color=INK, depthshade=False, zorder=14)
    c.set_xlabel("collapsed  $k$", labelpad=-2)
    c.set_ylabel("|Ctx|", labelpad=-2)
    c.set_xticks(range(1, 6))
    c.set_yticks(NCS)
    c.set_zlabel("")
    c.set_title("Unlicensed answers over the family", y=1.04)
    c.text2D(0.02, 0.80, "red: total" + chr(10) + "green: declining,"
             + chr(10) + "flat at 0", transform=c.transAxes,
             fontsize=6.4, color=INK2)
    c.set_box_aspect(None, zoom=1.15)
    tag(c, "c", three_d=True)

    # (d) the record floor's control: duplicating a context adds no
    # class, so the floor tracks distinctions rather than volume.
    d = ax[3]
    rf = R["record_floor"]
    rfc = R["record_floor_control"]
    hist = rf["history"]
    ncl = np.array([h["n_classes"] for h in hist], float)
    nct = np.array([h["n_contexts"] for h in hist], float)
    d.plot(nct, ncl, "-o", color=C1, markersize=4,
           label="distinct contexts added")
    d.plot([nct[0], nct[0] + 1],
           [rfc["classes_before"], rfc["classes_after_duplicate"]],
           "-s", color=C2, markersize=5, label="a duplicate added")
    d.scatter([nct[0] + 1], [rfc["classes_after_duplicate"]], s=95,
              facecolor="white", edgecolor=C2, linewidth=1.6, zorder=6)
    d.annotate("duplicate context:" + chr(10) + "%d classes, unchanged"
               % rfc["classes_after_duplicate"],
               xy=(nct[0] + 1, rfc["classes_after_duplicate"]),
               xytext=(0.34, 0.30), textcoords="axes fraction",
               fontsize=7, color=C2,
               arrowprops=dict(arrowstyle="->", color=C2, lw=0.9,
                               shrinkB=9))
    d.set_xlabel("contexts in the release")
    d.set_ylabel("classes the record must separate")
    d.set_title("The floor counts distinctions, not contexts")
    d.legend(loc="upper left")
    tag(d, "d")

    save(fig, PAPER, "panel4_declining")


if __name__ == "__main__":
    panel3()
    panel4()
