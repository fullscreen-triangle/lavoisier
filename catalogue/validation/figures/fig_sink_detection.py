"""Panels 1 and 2 for "Detecting a Sink".

exp5's records are scalars and booleans at fixed cells --- 0 band
breaches over 6 items, a single harmless/fatal pair at n = 9, a constant
min_z_fraction of 0.64516 --- so plotting the records alone would give
four charts of one number each.

As in the other papers' panels, exp5's own definitions are imported and
evaluated over the parameters exp5 fixed. The sweeps are cached in
results/exp5_sweeps.json (built by sweep_exp5.py, which asserts every
sweep against the recorded scalar before writing).

Two of these panels carry a FAILED claim. exp5 grades sink-detection at
13 of 15, and the two failures are real defects in the manuscript rather
than defects in the experiment; they are drawn as measured, not softened.
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

PAPER = "sink-detection"
D = load("exp5_sink_detection")
R = D["records"]

with open(os.path.join(RESULTS, "exp5_sweeps.json"), encoding="utf8") as _fh:
    SW = json.load(_fh)

LAM = 2.0

# The sweeps reproduce the artefact; sweep_exp5.py asserts this at build
# time, and these repeat the two that the drawings depend on directly.
_j = [r for r in SW["collapse"]["rows"] if r["lam"] == LAM][0]
assert _j["beta"] == R["collapse"]["beta"]
assert sorted(_j["sep"]) == sorted(
    v for k, v in R["collapse"]["seps"].items() if k != "z")
_d9 = [r for r in SW["degree"]["rows"] if r["n"] == 9][0]
assert _d9["deg_h"] == R["degree_fails"]["harmless"]["degree"]
assert _d9["deg_z"] == R["degree_fails"]["fatal"]["degree"]


# =====================================================================
# Panel 1 --- a sink pins every separation into a band
# =====================================================================
def panel1():
    fig, ax = panel(three_d=(1,))

    cl = SW["collapse"]
    rows = cl["rows"]
    lams = np.array([r["lam"] for r in rows], float)

    # (a) thm:collapse's two-sided band, swept over the sink level. The
    # record is one lambda and 0 breaches over 6 items; the claim is
    # about the band, and the band is a function of lambda. The measured
    # separations sit strictly inside the floor and the ceiling at every
    # level, and the band closes as the sink strengthens.
    a = ax[0]
    lo = np.array([r["beta"] for r in rows], float)
    hi = np.array([max(r["upper"]) for r in rows], float)
    sm = np.array([min(r["sep"]) for r in rows], float)
    sM = np.array([max(r["sep"]) for r in rows], float)
    a.fill_between(lams, lo, hi, color=C1, alpha=0.15, zorder=1,
                   label="the band, floor to ceiling")
    a.plot(lams, hi, "-", color=C1, lw=1.8, zorder=3)
    a.plot(lams, lo, "-", color=C1, lw=1.8, zorder=3)
    a.fill_between(lams, sm, sM, color=INK, alpha=0.35, zorder=4)
    a.plot(lams, sM, "-o", color=INK, markersize=4.5, zorder=5,
           label="separations measured")
    a.plot(lams, sm, "-o", color=INK, markersize=4.5, zorder=5)
    jr = int(np.argmin(np.abs(lams - LAM)))
    a.axvline(LAM, color=MUTED, lw=0.9, ls=":")
    a.scatter([LAM], [sM[jr]], s=130, facecolor="white", edgecolor=BAD,
              linewidth=1.8, zorder=8)
    a.annotate("recorded: %d of %d outside"
               % (R["collapse"]["band_breaches"], len(_j["sep"])),
               xy=(LAM, sM[jr]), xytext=(0.30, 0.80),
               textcoords="axes fraction", fontsize=7, color=BAD,
               arrowprops=dict(arrowstyle="->", color=BAD, lw=0.9,
                               shrinkB=10))
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_ylim(lo.min() * 0.80, hi.max() * 1.5)
    a.set_xlabel(r"sink level  $\lambda$")
    a.set_ylabel("separation")
    a.set_title("Ceiling attained, floor never reached")
    a.legend(loc="upper left", fontsize=6.8)
    a.text(0.97, 0.06, "%d breaches over %d sink levels" % (
        sum(r["breaches"] for r in rows), len(rows)) + chr(10)
        + "the upper edge is attained, not merely respected",
        transform=a.transAxes, ha="right", fontsize=6.6, color=INK2)
    tag(a, "a")

    # (b) 3-D: the sink's own weighted spread over both quantities that
    # set it.
    #
    # Three earlier versions of this chart tried to draw thm:collapse's
    # two-sided band in 3-D and all three came out as flat plates.
    # Dumping the numbers said why, and it is a finding rather than a
    # drawing problem: on this family sep == upper in all 54 cells to
    # full precision. The theorem's ceiling is not loose, it is ATTAINED
    # --- an ordinary item's minimiser is always its own incident star,
    # so the cut equals the sum the theorem writes down. Any quantity
    # normalised by that gap is therefore identically 1 by construction,
    # which is exactly what the flat plates were reporting.
    #
    # The measured quantity with content over BOTH axes is the paper's
    # own detector, wspread(z): it sweeps 0.20 -> 0.99 as the sink
    # strengthens and falls monotonically in n at every level, because
    # more items dilute the share any single contact carries.
    b = ax[1]
    gl = np.array(cl["grid_lams"], float)
    gn = np.array(cl["grid_ns"], float)
    Zs = np.array([r["sep"] for r in cl["surface"]], float)
    Zu = np.array([r["upper"] for r in cl["surface"]], float)
    assert cl["ceiling_exact_cells"] == cl["ceiling_total_cells"]
    assert np.max(np.abs(Zs - Zu)) < 1e-9, "ceiling no longer exact"
    Zw = np.array([r["wspread_z"] for r in cl["surface"]], float)
    NN, LL = np.meshgrid(gn, np.log10(gl), indexing="ij")
    b.view_init(elev=24, azim=-58)
    b.plot_surface(NN, LL, Zw, cmap="Blues", linewidth=0.25,
                   edgecolor="white", alpha=0.97, rstride=1, cstride=1,
                   vmin=0.0, vmax=1.0)
    jl = int(np.argmin(np.abs(gl - LAM)))
    b.plot(gn, np.full(len(gn), np.log10(LAM)), Zw[:, jl], "-o",
           color=INK, lw=2.4, markersize=4.5, zorder=16)
    b.set_zlim(0.0, 1.0)
    b.set_xlabel("items  $n$", labelpad=-2)
    b.set_ylabel(r"$\log_{10}\lambda$", labelpad=-2)
    b.set_xticks(gn)
    b.set_yticks([-0.6, 0.0, 0.6, 1.2, 1.8])
    b.set_zticks([0.0, 0.5, 1.0])
    b.set_zlabel("")
    b.set_title("The sink's weighted spread", y=1.04)
    b.text2D(-0.05, 0.86, "height: wspread($z$)" + chr(10)
             + "rises with the sink, falls with $n$" + chr(10)
             + "black: as recorded", transform=b.transAxes,
             fontsize=6.2, color=INK2)
    b.set_box_aspect(None, zoom=1.12)
    tag(b, "b", three_d=True)

    # (c) how tight the band is, which is what a user of thm:collapse
    # actually needs. The recorded band_width_vs_lambda is six points;
    # this is the same quantity computed on the same family, with the
    # recorded series overlaid as the points the curve must pass
    # through. A strong sink does not merely bound the separations --- it
    # squeezes them to a point.
    c = ax[2]
    bw = R["band_width_vs_lambda"]
    rl = np.array([r["lambda"] for r in bw], float)
    rw = np.array([r["rel_width"] for r in bw], float)
    swl = lams
    sww = np.array([(max(r["sep"]) - min(r["sep"])) / max(r["sep"])
                    for r in rows], float)
    c.plot(swl, sww, "-o", color=C1, markersize=5, zorder=4,
           label="spread of the measured separations")
    pos = rl > 0
    c.plot(rl[pos], rw[pos], "s", color=BAD, markersize=7, zorder=6,
           markerfacecolor="white", markeredgewidth=1.6,
           label="recorded band width")
    c.set_xscale("log")
    c.set_yscale("log")
    c.set_xlabel(r"sink level  $\lambda$")
    c.set_ylabel("relative width")
    c.set_title("A stronger sink collapses the band")
    c.legend(loc="lower left", fontsize=6.8)
    c.text(0.97, 0.93, "%.2f at $\\lambda$=%.2f  ->  %.4f at $\\lambda$=%.0f"
           % (rw[pos][0], rl[pos][0], rw[-1], rl[-1]),
           transform=c.transAxes, ha="right", va="top", fontsize=6.8,
           color=INK2)
    tag(c, "c")

    # (d) prop:amplify as measured. The proposition's proof closes with
    # "taking lambda large or the analysis to deeper S drives the
    # fraction to 1", but the bound it derives, lambda/(lambda+C), is
    # CONSTANT in |S|. Both series here are exp5's own measurements: the
    # sink's share of each separation, and an ordinary vertex's. The
    # sink's floor is flat --- depth does not drive it anywhere --- while
    # the ordinary vertex's share falls away, which is the half of the
    # proposition that does hold.
    d = ax[3]
    am = R["amplify"]
    ac = R["amplify_control"]["ordinary_vertex"]
    ns = np.array([r["n"] for r in am], float)
    mz = np.array([r["mean_z_fraction"] for r in am], float)
    fz = np.array([r["min_z_fraction"] for r in am], float)
    oc = np.array([r["mean_ordinary_fraction"] for r in ac], float)
    ocn = np.array([r["n"] for r in ac], float)
    d.plot(ns, mz, "-o", color=C1, markersize=5, zorder=5,
           label="the sink's share of each separation")
    d.plot(ns, fz, "--", color=BAD, lw=1.8, zorder=4,
           label=r"the bound $\lambda/(\lambda + C)$")
    d.plot(ocn, oc, "-s", color=C2, markersize=5, zorder=5,
           label="an ordinary vertex's share")
    d.fill_between(ns, fz, mz, color=C1, alpha=0.10)
    d.set_ylim(0, 1.0)
    d.set_xticks(ns)
    d.set_xlabel("items  $n$")
    d.set_ylabel("share of the separation cost")
    d.set_title("The bound is flat: depth drives nothing")
    d.legend(loc="center right", fontsize=6.6)
    d.text(0.03, 0.06, "bound constant at %.5f for every $n$" % fz[0],
           transform=d.transAxes, fontsize=6.8, color=BAD)
    tag(d, "d")

    save(fig, PAPER, "panel1_collapse")


# =====================================================================
# Panel 2 --- degree cannot detect a sink; spread can
# =====================================================================
def panel2():
    fig, ax = panel(three_d=(2,))

    dg = SW["degree"]["rows"]
    ns = np.array([r["n"] for r in dg], float)

    # (a) thm:degree-fails swept. The record is one pair at n = 9: a
    # degree-9 vertex that moves nothing and a degree-3 vertex that
    # multiplies separations fourfold. The claim is that no degree
    # threshold works, and that is a claim about the whole family --- so
    # both degrees are plotted against n. They cross and stay crossed:
    # the harmless vertex has the LARGER degree at every size.
    a = ax[0]
    dh = np.array([r["deg_h"] for r in dg], float)
    dz = np.array([r["deg_z"] for r in dg], float)
    a.plot(ns, dh, "-o", color=GOOD, markersize=5,
           label="harmless vertex")
    a.plot(ns, dz, "-s", color=BAD, markersize=5, label="fatal vertex")
    a.fill_between(ns, dz, dh, color=MUTED, alpha=0.16)
    a.scatter([9], [R["degree_fails"]["harmless"]["degree"]], s=120,
              facecolor="white", edgecolor=INK, linewidth=1.7, zorder=7)
    a.scatter([9], [R["degree_fails"]["fatal"]["degree"]], s=120,
              facecolor="white", edgecolor=INK, linewidth=1.7, zorder=7)
    a.set_xticks(ns)
    a.set_xlabel("items  $n$")
    a.set_ylabel("degree of the vertex")
    a.set_ylim(0, dh.max() * 1.28)
    a.set_title("The harmless vertex always has more contacts")
    a.legend(loc="upper left", fontsize=6.8)
    a.text(0.97, 0.06, "circles: the recorded pair ($n$ = 9)",
           transform=a.transAxes, ha="right", fontsize=6.8, color=INK2)
    tag(a, "a")

    # (b) the damage each of them actually does, on the same axis: how
    # far deleting the vertex moves a separation. Degree said the green
    # one was the bigger threat; the measurement says it moves nothing
    # while the red one multiplies every separation it touches.
    b = ax[1]
    sh = np.array([r["shift_h"] for r in dg], float)
    mr = np.array([r["min_ratio"] for r in dg], float)
    b.plot(ns, 1.0 + sh, "-o", color=GOOD, markersize=5,
           label="harmless: separation after / before")
    b.plot(ns, mr, "-s", color=BAD, markersize=5,
           label="fatal: separation after / before")
    b.axhline(1.0, color=MUTED, lw=1.0, ls=":")
    b.set_yscale("log")
    b.set_xticks(ns)
    b.set_xlabel("items  $n$")
    b.set_ylabel("factor by which deletion moves a separation")
    b.set_title("More contacts, less damage")
    b.legend(loc="center right", fontsize=6.6)
    b.text(0.03, 0.20, "recorded at $n$ = 9:" + chr(10)
           + "shift %.5f vs ratio %.1fx"
           % (R["degree_fails"]["harmless"]["max_separation_shift"],
              min(R["degree_fails"]["fatal"]["separation_ratios"])),
           transform=b.transAxes, va="bottom", fontsize=6.8, color=INK2)
    tag(b, "b")

    # (c) 3-D: the two detectors side by side over the family. Degree
    # and weighted spread are drawn as two surfaces over (n, detector),
    # each vertex a ridge. Degree ranks the harmless vertex above the
    # fatal one at every n --- the wrong order --- while spread ranks
    # them the other way at every n. The two sheets cross exactly once,
    # between the detectors, which is rem:spread-vs-degree.
    c = ax[2]
    wh = np.array([r["ws_h"] for r in dg], float)
    wz = np.array([r["ws_z"] for r in dg], float)
    # normalise each detector to its own scale so the ORDER, which is
    # what the remark is about, is what the height shows.
    dn_h = dh / np.maximum(dh, dz)
    dn_z = dz / np.maximum(dh, dz)
    wn_h = wh / np.maximum(wh, wz)
    wn_z = wz / np.maximum(wh, wz)
    yy = np.array([0.0, 1.0])          # 0 = degree, 1 = weighted spread
    NN, YY = np.meshgrid(ns, yy, indexing="ij")
    Zh = np.stack([dn_h, wn_h], axis=1)
    Zz = np.stack([dn_z, wn_z], axis=1)
    c.view_init(elev=20, azim=-58)
    c.plot_surface(NN, YY, Zh, color=GOOD, alpha=0.60, linewidth=0,
                   shade=False)
    c.plot_surface(NN, YY, Zz, color=BAD, alpha=0.60, linewidth=0,
                   shade=False)
    for y, zh_, zz_ in ((0.0, dn_h, dn_z), (1.0, wn_h, wn_z)):
        c.plot(ns, np.full(len(ns), y), zh_, color=GOOD, lw=2.2, zorder=12)
        c.plot(ns, np.full(len(ns), y), zz_, color=BAD, lw=2.2, zorder=12)
    c.set_xlabel("items  $n$", labelpad=-2)
    c.set_ylabel("")
    c.set_yticks([0.0, 1.0])
    c.set_yticklabels(["degree", "spread"], fontsize=6.4)
    c.set_zlabel("")
    c.set_title("The order reverses between detectors", y=1.04)
    c.text2D(0.50, -0.12, "green: harmless    red: fatal    "
             + "height: rank within each detector",
             transform=c.transAxes, ha="center", fontsize=6.2, color=INK2)
    c.set_box_aspect(None, zoom=1.15)
    tag(c, "c", three_d=True)

    # (d) the margin each detector leaves, which is what makes one of
    # them usable. Plotted as the ratio fatal/harmless: below 1 the
    # detector ranks the wrong vertex first. Degree sits below 1 and
    # falls further as the graph grows; weighted spread sits above it by
    # more than a decade throughout.
    d = ax[3]
    rd = dz / dh
    rw = wz / wh
    d.plot(ns, rd, "-o", color=BAD, markersize=5,
           label="degree:  fatal / harmless")
    d.plot(ns, rw, "-s", color=GOOD, markersize=5,
           label="weighted spread:  fatal / harmless")
    d.axhline(1.0, color=INK, lw=1.2, ls="--", zorder=2)
    d.fill_between(ns, 1.0, rd, color=BAD, alpha=0.10)
    d.fill_between(ns, 1.0, rw, color=GOOD, alpha=0.10)
    d.set_yscale("log")
    d.set_xticks(ns)
    d.set_xlabel("items  $n$")
    d.set_ylabel("the detector's ratio, fatal to harmless")
    d.set_title("Only one detector puts the sink on top")
    d.legend(loc="lower left", fontsize=6.6)
    d.text(0.97, 0.52, "above the line: ranked correctly",
           transform=d.transAxes, ha="right", va="bottom", fontsize=6.8,
           color=INK2)
    tag(d, "d")

    save(fig, PAPER, "panel2_degree")


if __name__ == "__main__":
    panel1()
    panel2()
