"""Panels 3 and 4 for "Mass Invariance" --- exp6, on the NIST library.

Panel 3 is thm:closure and thm:no-selector: whether a closed
determination names a single item or a region, and what the medium's
contact scale does to that. Panel 4 is prop:three, which the suite did
NOT reproduce --- and the failure is drawn as a mechanism rather than a
tally, because the measurement locates the defect precisely.

prop:three says three catalysts suffice to recover the true region. The
proposition's own wording contains two incompatible readings:

  literal --- "THE erroneous catalyst" is definite, so it is identified
      and removed, and the majority is taken over the survivors. Under
      this reading k = 2 already suffices (measured 1.000).

  unaided --- the erroneous catalyst is not identified, and the majority
      runs over all k. This is the case the proof's reason describes,
      and it needs k >= 3.

The two readings agree at every k except k = 2, where they differ by
exactly 1. That single point is the defect: it is a fault in the
STATEMENT, not in the proof, and it is identical at 3, 4 and 5
candidates, so it is not an artefact of the candidate pool.

As elsewhere, exp6's own definitions are swept over the parameters exp6
fixed (sweep_exp6.py), and every recorded scalar is asserted against the
sweep at module level before anything is drawn.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..")))

from panelkit import (C1, C2, C3, C4, GOOD, BAD, INK, INK2, MUTED,   # noqa: E402
                      RESULTS, panel, tag, load, save)

PAPER = "peptide-mass-invariance"
D = load("exp6_peptide_mass_invariance")
R = D["records"]

with open(os.path.join(RESULTS, "exp6_sweeps.json"), encoding="utf8") as _fh:
    SW = json.load(_fh)

NS = SW["no_selector"]
CL = SW["closure"]["rows"]
TH = SW["three"]["rows"]

# --- the sweeps must reproduce the artefact's VERDICT ----------------
# exp6 draws its own subgraphs from a shared rng, so the counts differ;
# the verdicts do not.

# T5: a closed determination is USUALLY a singleton but not always ---
# plurality is rare and must be found by a powered search, not asserted
# to occur in every sample. exp6 found 2 of 210; the sweep must also
# find some, and the mean region must sit just above 1.
assert R["T5_closed_plural"]["closed_with_plural_region"] > 0
assert sum(r["plural"] for r in CL) > 0
assert all(r["mean_region"] >= 1.0 for r in CL)
assert max(r["max_region"] for r in CL) > 1

# T4: without the medium, separation is free --- at every medium weight.
assert R["T4_no_selector"]["zero_cost_separation_without_medium"] == \
    R["T4_no_selector"]["graphs"]
assert all(r["zero_without_medium"] == r["graphs"] for r in NS["rows"])
assert all(r["positive_with_medium"] == r["graphs"] for r in NS["rows"])
assert all(r["min_sep"] >= r["beta"] - 1e-9 for r in NS["rows"])

# T7: the two readings of prop:three, and the single k at which they
# differ. This is the failed claim, so the assertion is on the SHAPE of
# the disagreement, which is what identifies it as a defect in the
# statement.
_rl = R["T7a_three_catalysts_literal"]["by_k"]
_ru = R["T7b_three_catalysts_unaided"]["by_k"]
assert _rl["2"]["fraction"] == 1.0 and _ru["2"]["fraction"] == 0.0
assert _rl["3"]["fraction"] == _ru["3"]["fraction"] == 1.0

# The regime sweep: delta is exactly 1 --- a singleton minimiser ---
# throughout the local contact regime, at every size, and exceeds 1
# once the contact scale passes the medium.
_MC = np.array(NS["regime_mc"], float)
_LO = [j for j, v in enumerate(NS["regime_mc"]) if v <= 1.0]
for _r in NS["regime"]:
    assert all(abs(_r["mean_delta"][j] - 1.0) < 1e-12 for j in _LO), _r
    assert _r["mean_delta"][-1] > 1.0, _r


# =====================================================================
# Panel 3 --- closure, and what the medium's scale buys
# =====================================================================
def panel3():
    fig, ax = panel(three_d=(2,))

    cn = np.array([r["n"] for r in CL], float)

    # (a) thm:closure. A determination that closes names a region, not
    # necessarily an item. Plurality is rare --- a few percent --- but
    # it is present at almost every size, and the one size where the
    # sweep found none (n = 7) is a sampling gap rather than a
    # structural fact: the mean region there is exactly 1.0 over 280
    # determinations, while its neighbours both exceed it.
    a = ax[0]
    rate = np.array([r["rate"] for r in CL], float)
    pl = np.array([r["plural"] for r in CL], float)
    det = np.array([r["determinations"] for r in CL], float)
    a.bar(cn, rate, width=0.56, color=C1, zorder=3)
    for x, y, k, m in zip(cn, rate, pl, det):
        a.text(x, y + rate.max() * 0.05, "%d/%d" % (int(k), int(m)),
               ha="center", fontsize=6.2, color=INK2)
    a.axhline(0.0, color=INK, lw=1.0, zorder=4)
    a.set_xticks(cn)
    a.set_ylim(0, rate.max() * 1.48)
    a.set_xlabel("items  $n$")
    a.set_ylabel("share of closed determinations naming a region")
    a.set_title("Closure names a region, not always an item")
    a.text(0.97, 0.95, "%d plural of %d closures"
           % (int(pl.sum()), int(det.sum())),
           transform=a.transAxes, ha="right", va="top", fontsize=6.6,
           color=INK2)
    tag(a, "a")

    # (b) how big the region gets when it is plural. The mean sits just
    # above one and the maximum reaches three: closure is nearly but not
    # quite injective, which is the same shape as the key's near-
    # injectivity in panel 1 and for the same reason.
    b = ax[1]
    mr = np.array([r["mean_region"] for r in CL], float)
    xr = np.array([r["max_region"] for r in CL], float)
    b.plot(cn, xr, "--s", color=C2, markersize=6, zorder=4,
           label="largest region observed")
    b.plot(cn, mr, "-o", color=C1, markersize=6, zorder=5,
           label="mean region size")
    b.axhline(1.0, color=MUTED, lw=1.2, ls=":", zorder=2)
    b.fill_between(cn, np.ones(len(cn)), mr, color=C1, alpha=0.14,
                   zorder=1)
    b.set_xticks(cn)
    b.set_ylim(0.72, 3.55)
    b.set_xlabel("items  $n$")
    b.set_ylabel("items named by one closed determination")
    b.set_title("The region is small but not a singleton")
    b.legend(loc="upper right", fontsize=6.6)
    # The note goes upper-left: the mean series hugs the dotted
    # reference at 1.0, near the bottom of the ylim, so a low note runs
    # straight through it. The upper-left quadrant is empty --- both
    # series are low there and the legend is upper-right.
    b.text(0.03, 0.97, "dotted: a singleton return",
           transform=b.transAxes, va="top", fontsize=6.6, color=INK2)
    tag(b, "b")

    # (c) 3-D: the regime transition, over (contact scale, subgraph
    # size). Height is the mean size of the minimising set |S*|. An
    # earlier version of this chart drew the transition over (contact
    # scale, MEDIUM WEIGHT) by tiling one measured row across the weight
    # axis, on the reasoning that delta is scale-invariant in the medium
    # because the contact scale is defined relative to it. That
    # reasoning is correct, but it is not a measurement, and a tiled
    # axis carries no data. The second axis is therefore the size of the
    # graph, where delta genuinely varies and was swept.
    #
    # The floor at exactly 1.0 across the whole local regime is what
    # ax:noncomplete's normalisation buys: while no item-item contact
    # exceeds the medium, every item's cheapest separation is its own
    # incident star, so the minimiser is a singleton at every size. Past
    # the medium the floor breaks and the minimiser grows with n.
    c = ax[2]
    rn = np.array(NS["regime_ns"], float)
    Zd = np.array([r["mean_delta"] for r in NS["regime"]], float)
    MM, NN = np.meshgrid(np.log2(_MC), rn, indexing="ij")
    c.view_init(elev=22, azim=-56)
    c.plot_surface(MM, NN, Zd.T, cmap="Blues", linewidth=0.25,
                   edgecolor="white", alpha=0.96, rstride=1, cstride=1)
    # The unit floor, drawn as the plane the surface lies on throughout
    # the local regime, so the departure is read against it.
    c.plot_surface(MM, NN, np.ones_like(Zd.T), color=MUTED, alpha=0.30,
                   linewidth=0, shade=False)
    c.set_xticks(np.log2(_MC))
    c.set_xticklabels(["%g" % v for v in _MC], fontsize=5.6)
    c.set_yticks(rn)
    c.set_xlabel("contact scale / medium", labelpad=8)
    c.set_ylabel("items  $n$", labelpad=-2)
    c.set_zlabel("")
    c.set_title("A singleton below the medium", y=1.02)
    c.text2D(0.50, -0.19, "height: mean $|S^*|$    "
             + "grey plane: a singleton minimiser    "
             + "exactly 1.0 in all %d cells at scale $\\leq 1$"
             % (len(_LO) * len(rn)),
             transform=c.transAxes, ha="center", fontsize=6.2,
             color=INK2)
    c.set_box_aspect(None, zoom=1.12)
    tag(c, "c", three_d=True)

    # (d) the same transition in section, at exp6's own size, with both
    # quantities the sweep measured: the separation itself, which rises
    # smoothly, and the size of the minimiser, which is pinned at 1
    # until the contact scale crosses the medium and then breaks away.
    # The break is the degenerate regime the paper's ax:noncomplete
    # excludes by hypothesis, shaded here.
    d = ax[3]
    sc = np.array([r["max_contact"] for r in NS["scale"]], float)
    sd = np.array([r["mean_delta"] for r in NS["scale"]], float)
    ss = np.array([r["mean_sep"] for r in NS["scale"]], float)
    d.axvspan(1.0, sc.max() * 1.08, color=BAD, alpha=0.07, zorder=0)
    d.plot(sc, ss, "-o", color=C2, markersize=6, zorder=5,
           label="mean separation")
    d.plot(sc, sd, "--s", color=C4, markersize=6, zorder=6,
           label="mean $|S^*|$")
    d.axvline(1.0, color=INK, lw=1.2, zorder=4)
    d.set_xlim(0, sc.max() * 1.08)
    d.set_ylim(0, max(ss.max(), sd.max()) * 1.30)
    d.set_xlabel("largest item-item contact / medium weight")
    d.set_ylabel("separation,   size of the minimiser")
    d.set_title("Past the medium, the minimiser grows")
    d.legend(loc="upper left", fontsize=6.6)
    d.text(0.985, 0.06, "shaded: excluded by hypothesis",
           transform=d.transAxes, ha="right", fontsize=6.4, color=INK2)
    tag(d, "d")

    save(fig, PAPER, "panel3_closure")


# =====================================================================
# Panel 4 --- prop:three, the claim the suite did not reproduce
# =====================================================================
def panel4():
    fig, ax = panel(three_d=(2,))

    ks = np.array([d_["k"] for d_ in TH[0]["literal"]], float)

    # (a) the two readings of the proposition, swept over k.
    #
    # The three candidate counts give IDENTICAL curves --- both readings
    # are flat in the candidate count, which (c) shows over the whole
    # grid. Drawing three overlapping pairs here would be six marks
    # carrying two measurements, and only the last drawn would be
    # visible. So one pair is drawn, and the invariance is ASSERTED
    # against every count rather than illustrated by overplotting.
    a = ax[0]
    lit = np.array([d_["fraction"] for d_ in TH[0]["literal"]], float)
    una = np.array([d_["fraction"] for d_ in TH[0]["unaided"]], float)
    for row in TH:
        assert np.allclose([d_["fraction"] for d_ in row["literal"]], lit)
        assert np.allclose([d_["fraction"] for d_ in row["unaided"]], una)
    a.fill_between(ks, una, lit, color=BAD, alpha=0.13, zorder=1,
                   label="the disagreement")
    a.plot(ks, lit, "-o", color=C1, markersize=6.5, zorder=6,
           label="literal: the catalyst is identified")
    a.plot(ks, una, "--s", color=C2, markersize=6.5, zorder=5,
           label="unaided: it is not")
    a.axvline(2.0, color=INK, lw=1.1, ls=":", zorder=3)
    a.set_xticks(ks)
    a.set_ylim(-0.08, 1.42)
    a.set_xlabel("catalysts  $k$")
    a.set_ylabel("share of runs recovering the true region")
    a.set_title("Two readings of one sentence")
    a.legend(loc="lower right", fontsize=6.2)
    a.text(0.97, 0.97, "identical at 3, 4 and 5 candidates",
           transform=a.transAxes, ha="right", va="top", fontsize=6.4,
           color=INK2)
    tag(a, "a")

    # (b) the gap between the readings, which is the defect itself. It
    # is exactly 1 at k = 2 and exactly 0 at every other k --- one
    # point, not a trend. Drawing three side-by-side bars per k would
    # put three adjacent slivers at a single x and read as one bar, so
    # the gap is a single series over k and the candidate counts are
    # overlaid as the points that must coincide on it.
    b = ax[1]
    gap = lit - una
    b.bar(ks, gap, width=0.55, color=C4, zorder=3,
          label="literal $-$ unaided")
    for j, (row, col, mk) in enumerate(zip(TH, (C1, C2, GOOD),
                                          ("o", "s", "^"))):
        g_ = (np.array([d_["fraction"] for d_ in row["literal"]], float)
              - np.array([d_["fraction"] for d_ in row["unaided"]], float))
        b.plot(ks + (j - 1) * 0.17, g_, mk, color=col, markersize=6,
               markerfacecolor="white", markeredgewidth=1.5, zorder=7,
               label="%d candidates" % row["candidates"])
    b.axhline(0.0, color=INK, lw=1.0, zorder=5)
    b.set_xticks(ks)
    b.set_ylim(-0.10, 1.42)
    b.set_xlabel("catalysts  $k$")
    b.set_ylabel("literal reading $-$ unaided reading")
    b.set_title("The readings disagree at one point only")
    b.legend(loc="upper right", fontsize=6.0, ncol=2)
    b.text(0.62, 0.62, "exactly 1 at $k=2$," + chr(10)
           + "exactly 0 elsewhere",
           transform=b.transAxes, va="top", fontsize=6.6, color=INK2)
    tag(b, "b")

    # (c) 3-D: both readings as surfaces over (k, candidate count). The
    # two sheets are two steps one apart in k and FLAT in the candidate
    # count, which is the measurement that rules out the obvious
    # alternative explanation --- that k = 2 succeeds only because the
    # candidate pool is small. It does not: the literal reading closes
    # at k = 2 whether there are three candidates or five.
    #
    # The unaided sheet is offset by a hair in z so that where the two
    # coincide (k >= 3) both remain visible; the offset is cosmetic and
    # smaller than any measured difference, which is 1.
    c = ax[2]
    cand = np.array([row["candidates"] for row in TH], float)
    Zl = np.array([[d_["fraction"] for d_ in row["literal"]]
                   for row in TH], float)
    Zu = np.array([[d_["fraction"] for d_ in row["unaided"]]
                   for row in TH], float)
    KK, CC = np.meshgrid(ks, cand, indexing="ij")
    c.view_init(elev=20, azim=-60)
    c.plot_surface(KK, CC, Zl.T, color=C1, alpha=0.90, linewidth=0.25,
                   edgecolor="white", rstride=1, cstride=1)
    c.plot_surface(KK, CC, Zu.T - 0.012, color=C2, alpha=0.80,
                   linewidth=0.25, edgecolor="white", rstride=1,
                   cstride=1)
    c.set_xticks(ks)
    c.set_yticks(cand)
    c.set_zlim(-0.06, 1.10)
    c.set_xlabel("catalysts  $k$", labelpad=-2)
    c.set_ylabel("candidates", labelpad=-2)
    c.set_zlabel("")
    c.set_title("Flat in the candidate count", y=1.04)
    c.text2D(0.50, -0.12, "height: share recovering the true region    "
             + "blue: literal    orange: unaided",
             transform=c.transAxes, ha="center", fontsize=6.2,
             color=INK2)
    c.set_box_aspect(None, zoom=1.12)
    tag(c, "c", three_d=True)

    # (d) the swept curves against exp6's own recorded points. The
    # artefact records five values of k for each reading; the sweep runs
    # seven. The recorded points must lie on the swept curves, and they
    # do --- so the disagreement in (b) is a property of the
    # proposition, not of this sweep's sampling.
    d = ax[3]
    d.plot(ks, lit, "-", color=C1, lw=2.0, zorder=4,
           label="swept: literal")
    d.plot(ks, una, "--", color=C2, lw=2.0, zorder=4,
           label="swept: unaided")
    rkl = np.array(sorted(int(k) for k in _rl), float)
    rvl = np.array([_rl[str(int(k))]["fraction"] for k in rkl], float)
    rvu = np.array([_ru[str(int(k))]["fraction"] for k in rkl], float)
    d.plot(rkl, rvl, "o", color=C1, markersize=8, zorder=6,
           markerfacecolor="white", markeredgewidth=1.7,
           label="exp6 recorded: literal")
    d.plot(rkl, rvu, "s", color=C2, markersize=8, zorder=6,
           markerfacecolor="white", markeredgewidth=1.7,
           label="exp6 recorded: unaided")
    # the recorded points must sit ON the swept curves
    for _k, _v in zip(rkl, rvl):
        assert abs(lit[int(_k) - 1] - _v) < 1e-12, (_k, _v)
    for _k, _v in zip(rkl, rvu):
        assert abs(una[int(_k) - 1] - _v) < 1e-12, (_k, _v)
    d.set_xticks(ks)
    d.set_ylim(-0.08, 1.42)
    d.set_xlabel("catalysts  $k$")
    d.set_ylabel("share of runs recovering the true region")
    d.set_title("The sweep reproduces the recorded values")
    d.legend(loc="lower right", fontsize=6.0)
    d.text(0.03, 0.97, "%d trials per recorded point"
           % _rl["2"]["trials"],
           transform=d.transAxes, va="top", fontsize=6.4, color=INK2)
    tag(d, "d")

    save(fig, PAPER, "panel4_three")


if __name__ == "__main__":
    panel3()
    panel4()
