"""Panels 1 and 2 for "Mass Invariance" --- exp6, on the NIST library.

exp6's records are scalars at fixed cells: 0 key mismatches over 40
relabellings of ONE 8-compound subgraph, 7 ambiguous weights over 420
cuts, 17 key-sharing pairs of 5600, 44 mass-ambiguous compounds at
100 ppm. Drawing those alone gives four charts of one number each.

As with the other papers, exp6's own definitions are imported and swept
over the parameters exp6 fixed; the sweeps live in results/exp6_sweeps.json
(built by sweep_exp6.py, which asserts every sweep against the recorded
scalar before writing anything).

Panel 1 is thm:invariance and thm:region --- what the cut key is
invariant UNDER, and whether a scalar can stand in for the region.
Panel 2 is cor:mass, and it draws the distinction the corollary turns on
and which the paper must not conflate:

  (a) pairs sharing a CUT KEY while differing in MASS. Refining the mass
      cannot create key agreement, so such a pair survives every
      refinement. Measured: 19 such pairs of 6000, and they are not
      near-degenerate --- the SMALLEST mass gap among them is 6448 ppm.

  (b) how well mass alone resolves this library, which is complete by
      9 ppm. That is the honest converse exp6 records as T3b, and it is
      NOT a refutation of (a): it is a fact about the library, while (a)
      is a fact about the key.
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

IV = SW["invariance"]["rows"]
RG = SW["region"]["rows"]
MS = SW["mass"]
NS = SW["no_selector"]

# The sweeps must reproduce the artefact's VERDICT. exp6 draws its own
# subgraphs from a shared rng, so the counts differ; the verdicts do not.
assert all(r["mismatches"] == 0 for r in IV)
assert R["T1_invariance"]["key_mismatches"] == 0
assert all(r["control_moved"] == r["graphs"] for r in IV)
assert sum(r["ambiguous_weights"] for r in RG) > 0
assert R["T2_region"]["weights_carrying_several_distinct_cuts"] > 0
assert len(MS["key_share_ppm_gaps"]) > 0
assert R["T3_mass_vs_key"]["pairs_sharing_a_cut_key"] > 0
assert R["T3b_library_mass_degeneracy"]["resolved_at_ppm"] == 10.0


# =====================================================================
# Panel 1 --- what the key is invariant under, and what it is not
# =====================================================================
def panel1():
    fig, ax = panel(three_d=(2,))

    ns = np.array([r["n"] for r in IV], float)

    # (a) thm:invariance swept over subgraph size, WITH the control that
    # makes it non-vacuous. Relabelling the items never moves a key ---
    # 0 mismatches over 288 relabellings across six sizes --- while
    # exp6's own control, scaling one medium edge by 0.25, moves a key in
    # every graph at every size. An invariance claim whose control never
    # fires is a claim that the key is constant, which is a different and
    # much weaker statement.
    a = ax[0]
    mism = np.array([r["mismatches"] for r in IV], float)
    ctrl = np.array([r["control_moved"] / float(r["graphs"])
                     for r in IV], float)
    rel = np.array([r["relabellings"] for r in IV], float)
    a.bar(ns + 0.18, ctrl, width=0.34, color=BAD, zorder=3,
          label="a real weight change moves a key")
    # The relabelling rate is exactly zero at every size, and a
    # zero-height bar is an absent mark rather than a small one --- it
    # would show as a gap and be read as missing data. The measured zero
    # is the whole claim, so it is drawn as a marker sitting ON the
    # baseline, where it can be seen to have been measured.
    a.plot(ns - 0.18, mism / rel, "o", color=GOOD, markersize=7,
           markeredgecolor=INK, markeredgewidth=0.8, zorder=6,
           label="relabelling moves a key (0 at every size)")
    a.axhline(0.0, color=INK, lw=1.0, zorder=4)
    a.set_xticks(ns)
    a.set_ylim(-0.09, 1.34)
    a.set_xlabel("items  $n$")
    a.set_ylabel("share of trials in which a key moved")
    a.set_title("Invariant under names, not under weights")
    a.legend(loc="upper center", fontsize=6.4)
    a.text(0.50, 0.055, "%d relabellings, %d mismatches"
           % (int(rel.sum()), int(mism.sum())) + chr(10)
           + "control fires in %d of %d graphs"
           % (int(sum(r["control_moved"] for r in IV)),
              int(sum(r["graphs"] for r in IV))),
           transform=a.transAxes, ha="center", fontsize=6.6, color=INK2)
    tag(a, "a")

    # (b) how much the key actually distinguishes. An invariant that
    # collapses everything to one value is invariant and useless, so the
    # number of DISTINCT keys is plotted against the number of items. It
    # tracks n almost exactly --- the key separates nearly every item
    # from every other --- which is what makes the invariance in (a)
    # worth having.
    b = ax[1]
    dk = np.array([r["mean_distinct_keys"] for r in IV], float)
    b.plot(ns, ns, "--", color=MUTED, lw=1.5, zorder=2,
           label="one key per item (the ceiling)")
    b.plot(ns, dk, "-o", color=C1, markersize=6, zorder=5,
           label="distinct keys measured")
    b.fill_between(ns, dk, ns, color=C1, alpha=0.12, zorder=1)
    b.set_xticks(ns)
    b.set_xlabel("items  $n$")
    b.set_ylabel("distinct cut keys in the subgraph")
    b.set_title("The invariant is nearly injective")
    b.legend(loc="upper left", fontsize=6.6)
    b.text(0.97, 0.06, "mean over 6 subgraphs per size",
           transform=b.transAxes, ha="right", fontsize=6.6, color=INK2)
    tag(b, "b")

    # (c) 3-D: thm:no-selector, which is what makes any of this cost
    # anything. Height is the mean separation over (subgraph size, the
    # medium's weight), drawn as two surfaces: with the medium present,
    # and with its edges deleted. The lower sheet is identically zero at
    # every one of the 30 cells --- deleting the medium does not make
    # separation cheap, it makes it FREE --- while the upper sheet rises
    # linearly in the medium's weight and is flat in n. Separation is
    # bought entirely from the medium.
    c = ax[2]
    gn = np.array(NS["grid_ns"], float)
    gm = np.array(NS["grid_mw"], float)
    Zw = np.array([r["with_medium"] for r in NS["surface"]], float)
    Zo = np.array([r["without"] for r in NS["surface"]], float)
    assert np.max(np.abs(Zo)) < 1e-12, "a zero-cost cut is no longer free"
    NN, MM = np.meshgrid(gn, np.log2(gm), indexing="ij")
    c.view_init(elev=20, azim=-58)
    c.plot_surface(NN, MM, Zw, cmap="Blues", linewidth=0.25,
                   edgecolor="white", alpha=0.96, rstride=1, cstride=1)
    c.plot_surface(NN, MM, Zo, color=BAD, alpha=0.55, linewidth=0,
                   shade=False)
    c.set_xlabel("items  $n$", labelpad=-2)
    c.set_ylabel("medium weight", labelpad=2)
    c.set_xticks(gn)
    c.set_yticks(np.log2(gm))
    c.set_yticklabels(["%g" % v for v in gm], fontsize=5.8)
    c.set_zlabel("")
    c.set_title("Without the medium, separation is free", y=1.04)
    c.text2D(0.50, -0.11, "height: mean separation    "
             + "blue: medium present    red: medium deleted (all %d cells 0)"
             % Zo.size, transform=c.transAxes, ha="center", fontsize=6.2,
             color=INK2)
    c.set_box_aspect(None, zoom=1.12)
    tag(c, "c", three_d=True)

    # (d) thm:region. A scalar separation can stand in for the whole
    # separating region only if distinct minimum cuts carry distinct
    # weights, and they do not: a small but persistent share of weights
    # carries several genuinely different cuts, at every size. The rate
    # is a couple of percent, which is exactly why exp6 had to power this
    # search rather than sample it --- 24 subgraphs per size finds none
    # at some sizes and would report the shortfall as a refutation.
    d = ax[3]
    rn = np.array([r["n"] for r in RG], float)
    sh = np.array([r["share"] for r in RG], float)
    amb = np.array([r["ambiguous_weights"] for r in RG], float)
    wt = np.array([r["weights"] for r in RG], float)
    d.bar(rn, sh, width=0.56, color=C4, zorder=3)
    for x, y, k, w in zip(rn, sh, amb, wt):
        d.text(x, y + 0.0013, "%d/%d" % (int(k), int(w)), ha="center",
               fontsize=6.2, color=INK2)
    d.axhline(0.0, color=INK, lw=1.0, zorder=4)
    d.set_xticks(rn)
    d.set_ylim(0, sh.max() * 1.42)
    d.set_xlabel("items  $n$")
    d.set_ylabel("share of weights carrying several cuts")
    d.set_title("A weight does not determine its cut")
    d.text(0.97, 0.90, "%d ambiguous of %d weights"
           % (int(amb.sum()), int(wt.sum())) + chr(10)
           + "over %d minimum cuts"
           % int(sum(r["cuts"] for r in RG)),
           transform=d.transAxes, ha="right", va="top", fontsize=6.6,
           color=INK2)
    tag(d, "d")

    save(fig, PAPER, "panel1_invariance")


# =====================================================================
# Panel 2 --- the key against the mass
# =====================================================================
def panel2():
    fig, ax = panel(three_d=(1,))

    # (a) cor:mass's actual content. Pairs of compounds that share a cut
    # key are plotted by the mass gap between them. Every one of the 19
    # is grossly mass-distinct --- the smallest gap is 6448 ppm, five
    # orders of magnitude above the 0.01 ppm at which exp6's tolerance
    # sweep bottoms out --- so no refinement of the mass will ever bring
    # such a pair into agreement, and no sharpening of the mass will ever
    # separate them by key. The two quantities are independent.
    a = ax[0]
    g = np.array(MS["key_share_ppm_gaps"], float)
    a.hist(g, bins=np.logspace(np.log10(g.min() * 0.7),
                               np.log10(g.max() * 1.4), 22),
           color=C1, alpha=0.88, zorder=3)
    a.axvline(g.min(), color=BAD, lw=1.6, zorder=6)
    a.set_xscale("log")
    a.set_xlim(g.min() * 0.45, g.max() * 1.8)
    a.set_ylim(0, 3.9)
    a.set_xlabel("mass gap within the pair (ppm)")
    a.set_ylabel("pairs sharing a cut key")
    a.set_title("Key-sharing pairs are grossly mass-distinct")
    a.text(0.97, 0.97, "%d pairs of %d" % (len(g), MS["pairs"]) + chr(10)
           + "red: the smallest gap, %.0f ppm" % g.min(),
           transform=a.transAxes, ha="right", va="top", fontsize=6.8,
           color=INK2)
    tag(a, "a")

    # (b) 3-D: the two quantities as surfaces over the library, which is
    # the comparison the corollary is about. Height is the share of
    # compounds left ambiguous, over (mass tolerance, subgraph size),
    # for each of the two determinations. Mass ambiguity collapses to
    # zero as the tolerance sharpens --- the library resolves completely
    # by 9 ppm --- while key ambiguity is flat in the tolerance, because
    # the key does not read the mass at all. The sheets cross; below the
    # crossing the key is the more discriminating of the two.
    b = ax[1]
    tol = MS["tolerance"]
    tp = np.array([t["tol_ppm"] for t in tol], float)
    tf = np.array([t["fraction"] for t in tol], float)
    kn = np.array([r["n"] for r in MS["per_n"]], float)
    kr = np.array([r["rate"] for r in MS["per_n"]], float)
    TT, KK = np.meshgrid(np.log10(tp), kn, indexing="ij")
    Zm = np.tile(tf[:, None], (1, len(kn)))
    Zk = np.tile(kr[None, :], (len(tp), 1))
    b.view_init(elev=19, azim=-62)
    b.plot_surface(TT, KK, Zm, cmap="Oranges", linewidth=0.2,
                   edgecolor="white", alpha=0.94, rstride=1, cstride=1)
    b.plot_surface(TT, KK, Zk, color=C1, alpha=0.75, linewidth=0,
                   shade=False)
    b.set_zlim(-0.055, max(float(Zm.max()), float(Zk.max())) * 1.05)
    b.set_xlabel(r"$\log_{10}$ tolerance (ppm)", labelpad=-2)
    b.set_ylabel("items  $n$", labelpad=-2)
    b.set_yticks(kn)
    b.set_zlabel("")
    b.set_title("Mass ambiguity vanishes; key ambiguity does not",
                y=1.04)
    b.text2D(0.50, -0.11, "height: share left ambiguous    "
             + "orange: by mass    blue: by cut key",
             transform=b.transAxes, ha="center", fontsize=6.2, color=INK2)
    b.set_box_aspect(None, zoom=1.10)
    tag(b, "b", three_d=True)

    # (c) T3b as measured, and the honest converse it records: mass alone
    # resolves THIS library completely once the tolerance is sharp
    # enough. exp6's five recorded tolerances are overlaid as the points
    # the swept curve must pass through. This is not a refutation of (a)
    # --- it is a property of a 244-compound library, and the pairs in
    # (a) survive it untouched.
    c = ax[2]
    c.plot(tp, tf, "-", color=C2, lw=2.0, zorder=4,
           label="swept: share of compounds mass-ambiguous")
    rec = R["T3b_library_mass_degeneracy"]["by_tolerance"]
    rp = np.array([r["tol_ppm"] for r in rec], float)
    rf = np.array([r["fraction"] for r in rec], float)
    c.plot(rp, rf, "s", color=BAD, markersize=7, zorder=6,
           markerfacecolor="white", markeredgewidth=1.6,
           label="exp6's recorded tolerances")
    # Resolution is monotone in the tolerance --- if the library resolves
    # at t it resolves at every t' < t --- so the threshold is the
    # LARGEST tolerance that resolves. Taking the smallest would report
    # the sweep's own lower endpoint instead of a property of the library.
    _clear = [t["tol_ppm"] for t in tol if t["ambiguous"] == 0]
    _amb = [t["tol_ppm"] for t in tol if t["ambiguous"] > 0]
    assert min(_amb) > max(_clear), "resolution is not monotone in ppm"
    res = max(_clear)
    c.axvline(res, color=INK, lw=1.3, ls="--", zorder=5)
    c.set_xscale("log")
    c.set_xlim(min(rp.min(), tp.min()) * 0.4, tp.max() * 2.0)
    c.set_ylim(-0.03, 0.72)
    c.set_xlabel("mass tolerance (ppm)")
    c.set_ylabel("share of the library left ambiguous")
    c.set_title("Mass resolves this library below %.0f ppm" % res)
    c.legend(loc="upper left", fontsize=6.4)
    c.text(res * 0.85, 0.40, "resolved  ", fontsize=6.8, color=INK,
           ha="right", rotation=90, va="center")
    tag(c, "c")

    # (d) the same two determinations put side by side per subgraph size:
    # the rate at which a pair shares a cut key, against the rate at
    # which a pair is mass-ambiguous at the library's own working
    # tolerance. The key's rate is small and non-zero at every size; the
    # mass's is zero once resolved. Neither dominates the other, which is
    # the point --- they are different determinations, and cor:mass says
    # the key's is not recoverable from the mass.
    d = ax[3]
    kp = np.array([r["pairs"] for r in MS["per_n"]], float)
    kk = np.array([r["sharing_key"] for r in MS["per_n"]], float)
    d.bar(kn, kr, width=0.52, color=C1, zorder=3,
          label="pairs sharing a cut key")
    for x, y, k, p in zip(kn, kr, kk, kp):
        d.text(x, y + kr.max() * 0.045, "%d/%d" % (int(k), int(p)),
               ha="center", fontsize=6.2, color=INK2)
    d.axhline(0.0, color=INK, lw=1.0, zorder=5)
    d.plot(kn, np.zeros(len(kn)), "s", color=C2, markersize=8, zorder=7,
           markerfacecolor="white", markeredgewidth=1.8,
           label="pairs mass-ambiguous below %.0f ppm (0)" % res)
    d.set_xticks(kn)
    d.set_ylim(-kr.max() * 0.22, kr.max() * 1.62)
    d.set_xlabel("items  $n$")
    d.set_ylabel("share of pairs left undetermined")
    d.set_title("Two determinations, neither reducible to the other")
    d.legend(loc="upper center", fontsize=6.2)
    tag(d, "d")

    save(fig, PAPER, "panel2_mass")


if __name__ == "__main__":
    panel1()
    panel2()
