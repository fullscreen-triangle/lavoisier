"""
Panels 1 and 2 for "Coordinates Without Provenance".

exp3's records are exhaustive counts and booleans: 0 auditable maps of
512, 0 restored of 256, 0 recoveries of 81. Four bar-per-zero charts
would be four charts of nothing. So, as in the observation-groups
panels, this file imports the experiment's own definitions and evaluates
them across the parameters the artefact fixed --- codomain size, context
count, pipeline depth --- and marks the recorded scalar as the point
that must land on the resulting curve.

The auditable-map count is computed in closed form rather than sampled.
Sampling 20000 maps at |Ctx| = 3, |Crd| = 4 returned zero auditable
maps, but the true rate there is 9.0e-4: a sampled zero is
indistinguishable from exp3's exhaustive zeros, which is exactly the
distinction the paper turns on. The closed form
    A(n, m, q) = sum over disjoint nonempty coordinate sets
               = sum_{s_1..s_m >= 1, sum s_i <= q} prod C(rem, s_i) Surj(n, s_i)
was checked against brute-force enumeration on every case small enough
to enumerate (see the assertion in `auditable_count`).
"""
from __future__ import annotations

import os
import sys
from math import comb

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..")))

from exp3_coordinate_provenance import (          # noqa: E402
    all_maps, is_auditable, rec_min, sufficient_for_comparison)
from panelkit import (C1, C2, C3, C4, GOOD, BAD, INK, INK2, MUTED,   # noqa: E402
                      panel, tag, load, save)

PAPER = "coordinate-provenance"
D = load("exp3_coordinate_provenance")
R = D["records"]

NX = 3          # |X|, the measurement set exp3 fixed


def _surj(n, k):
    """Maps [n] -> [k] that are onto."""
    return sum((-1) ** i * comb(k, i) * (k - i) ** n for i in range(k + 1))


def auditable_count(nx, nc, nq):
    """Exact number of auditable maps X x Ctx -> Crd.

    Auditable means no coordinate is produced under two contexts, i.e.
    the contexts use pairwise disjoint coordinate sets. Choose a
    nonempty set per context, then any onto map into it.
    """
    total = 0

    def rec(i, rem, acc):
        nonlocal total
        if i == nc:
            total += acc
            return
        for s in range(1, rem + 1):
            rec(i + 1, rem - s, acc * comb(rem, s) * _surj(nx, s))

    rec(0, nq, 1)
    return total


# The formula is only trustworthy if it reproduces the enumeration exp3
# actually ran. Check it against brute force wherever that is affordable,
# including exp3's own cell (|Ctx| = 3, |Crd| = 2, 512 maps, 0 auditable).
for _nc in (2, 3):
    for _nq in (1, 2, 3):
        _X = ["x%d" % i for i in range(NX)]
        _C = ["c%d" % i for i in range(_nc)]
        _dom = [(x, c) for x in _X for c in _C]
        _brute = sum(1 for p in all_maps(_dom, list(range(_nq)))
                     if is_auditable(p, _X, _C))
        assert auditable_count(NX, _nc, _nq) == _brute, (_nc, _nq)


# =====================================================================
# Panel 1 --- collapse is forced, and it is forced by cardinality
# =====================================================================
def panel1():
    fig, ax = panel(three_d=(1,))

    cs = R["collapse_small_codomain"]
    ctl = R["collapse_control"]

    # (a) auditable share against coordinate-space size, at the
    # experiment's own |Ctx| = 3. Exact counts, not a sample: the
    # recorded cell (|Crd| = 2) is the zero at the left, and the
    # recorded control (|Crd| = 9) is the point at the right.
    a = ax[0]
    qs = np.arange(1, 13)
    nc = cs["size_Ctx"]
    frac = np.array([auditable_count(NX, nc, q) / float(q ** (NX * nc))
                     for q in qs])
    a.plot(qs, frac, "-o", color=C1, zorder=3)
    a.axvline(nc, color=MUTED, lw=0.9, ls=":")
    jq = int(np.where(qs == cs["size_Crd"])[0][0])
    a.scatter([qs[jq]], [frac[jq]], s=110, facecolor="white", edgecolor=BAD,
              linewidth=1.8, zorder=6)
    a.annotate("recorded: 0 of %d" % cs["n_maps"], xy=(qs[jq], frac[jq]),
               xycoords="data", xytext=(0.30, 0.62),
               textcoords="axes fraction", fontsize=7, color=BAD,
               arrowprops=dict(arrowstyle="->", color=BAD, lw=0.9,
                               shrinkB=9))
    jc = int(np.where(qs == ctl["size_Crd"])[0][0])
    a.scatter([qs[jc]], [frac[jc]], s=90, color=GOOD, zorder=6,
              edgecolor="white", linewidth=1.1)
    a.annotate("control |Crd| = %d" % ctl["size_Crd"],
               xy=(qs[jc], frac[jc]), xytext=(-4, 12),
               textcoords="offset points", fontsize=7, color=GOOD,
               ha="right")
    a.text(nc + 0.15, 0.02, "|Crd| = |Ctx|", fontsize=6.8, color=INK2)
    a.set_xlabel("coordinate space size  |Crd|")
    a.set_ylabel("share of maps that are auditable")
    a.set_xticks(qs[::2])
    a.set_title("Auditability is exactly zero for |Crd| < %d" % nc)
    tag(a, "a")

    # (b) 3-D: the same share over both cardinalities. The zero wall to
    # the left of |Crd| = |Ctx| is the theorem; the recorded cell sits
    # on the floor of it.
    b = ax[1]
    ctxs = np.arange(2, 7)
    qq = np.arange(1, 13)
    Z = np.array([[auditable_count(NX, int(m), int(q)) / float(q ** (NX * m))
                   for q in qq] for m in ctxs])
    QQ, MM = np.meshgrid(qq, ctxs)
    b.view_init(elev=27, azim=-58)
    b.plot_surface(QQ, MM, Z, cmap="Blues", linewidth=0, alpha=0.80,
                   rstride=1, cstride=1)
    b.plot(qq, np.full(len(qq), nc),
           [auditable_count(NX, nc, int(q)) / float(q ** (NX * nc))
            for q in qq], color=BAD, lw=2.0, zorder=12)
    b.scatter([cs["size_Crd"]], [nc], [0.0], s=44, color=BAD,
              depthshade=False, zorder=14)
    b.set_xlabel("|Crd|", labelpad=-2)
    b.set_ylabel("|Ctx|", labelpad=-2)
    b.set_yticks(ctxs)
    b.set_zlabel("")
    b.set_title("A zero wall wherever |Crd| < |Ctx|", y=1.0)
    b.text2D(0.02, 0.80, "red: the recorded row" + chr(10) + "(|Ctx| = %d)" % nc,
             transform=b.transAxes, fontsize=6.6, color=BAD)
    b.set_box_aspect(None, zoom=1.15)
    tag(b, "b", three_d=True)

    # (c) pipeline loss: an upstream collapse survives every downstream
    # stage. Swept over stage-2 codomain size --- widening the second
    # stage never restores what the first discarded.
    c = ax[2]
    pl = R["pipeline_loss"]
    X = ["x%d" % i for i in range(NX)]
    Cx = ["c1", "c2", "c3"]
    phi1 = {}
    for x in X:
        phi1[(x, "c1")] = "a"
        phi1[(x, "c2")] = "a"          # exp3's deliberate collapse
        phi1[(x, "c3")] = "b"
    widths = list(range(2, 7))
    restored, totals = [], []
    for w in widths:
        C2s = ["d1", "d2"]
        dom2 = [(k, d) for k in ("a", "b") for d in C2s]
        CC = [(c1, c2) for c1 in Cx for c2 in C2s]
        n = 0
        rst = 0
        for phi2 in all_maps(dom2, ["p%d" % i for i in range(w)]):
            n += 1
            psi = {}
            for x in X:
                for (c1, c2) in CC:
                    psi[(x, (c1, c2))] = phi2[(phi1[(x, c1)], c2)]
            if is_auditable(psi, X, CC):
                rst += 1
        restored.append(rst)
        totals.append(n)
    c.plot(widths, totals, "-s", color=C1, label="stage-2 maps tried")
    c.plot(widths, restored, "-o", color=BAD,
           label="restore auditability")
    c.set_yscale("symlog", linthresh=1.0)
    c.set_xticks(widths)
    c.set_xlabel("stage-2 coordinate space size")
    c.set_ylabel("maps")
    c.set_title("No second stage restores the loss")
    c.legend(loc="center right")
    c.scatter([2], [pl["restored_auditability"]], s=95, facecolor="white",
              edgecolor=BAD, linewidth=1.7, zorder=6)
    c.text(2.06, 1.6, "recorded: %d of %d" % (pl["restored_auditability"],
                                              pl["n_stage2_maps"]),
           fontsize=7, color=BAD)
    tag(c, "c")

    # (d) no recovery: over the enumerated function space, how many
    # functions of the collapsed output reproduce the discarded context.
    d = ax[3]
    nr = R["no_recovery"]
    nrc = R["no_recovery_control"]
    labs = ["outputs\ncollapsed", "outputs\ndistinct"]
    vals = [nr["successful_recoveries"], nrc["recoveries_when_inputs_distinct"]]
    tot = nr["n_functions_enumerated"]
    idx = np.arange(2)
    d.bar(idx, [tot, tot], width=0.55, color=MUTED, alpha=0.32, zorder=2,
          label="functions enumerated")
    d.bar([idx[0]], [vals[0]], width=0.55, color=BAD, zorder=3,
          label="collapsed: no recovery")
    d.bar([idx[1]], [vals[1]], width=0.55, color=GOOD, zorder=3,
          label="distinct: recovery exists")
    for i, v in enumerate(vals):
        d.text(i, tot + 3.0, "%d of %d" % (v, tot), ha="center",
               fontsize=7.6, color=INK2)
    d.set_xticks(idx)
    d.set_xticklabels(labs, fontsize=7.5)
    d.set_ylabel("functions Crd -> Ctx")
    d.set_ylim(0, tot * 1.30)
    d.set_title("Nothing recovers a discarded context")
    d.legend(loc="upper center", ncol=1, fontsize=7)
    tag(d, "d")

    save(fig, PAPER, "panel1_collapse")


# =====================================================================
# Panel 2 --- what the record costs, and what it must hold
# =====================================================================
def panel2():
    fig, ax = panel(three_d=(2,))

    cost = R["cost"]
    rows = cost["rows"]
    Ns = np.array([r["N"] for r in rows], float)

    # (a) the two accounting schemes across five decades of run size.
    a = ax[0]
    pm = np.array([r["per_measurement_bits"] for r in rows], float)
    pr = np.array([r["per_run_bits"] for r in rows], float)
    a.loglog(Ns, pm, "-o", color=C2, label="per-measurement record")
    a.loglog(Ns, pr, "-s", color=C1, label="per-run record")
    a.set_xlabel("measurements  $N$")
    a.set_ylabel("record size (bits)")
    a.set_title("Both are linear; the constant differs")
    a.legend(loc="upper left")
    tag(a, "a")

    # (b) the ratio, which is what prop:cost actually claims: bounded,
    # and converging as N grows.
    b = ax[1]
    ratio = np.array([r["ratio"] for r in rows], float)
    lim = ratio[-1]
    b.semilogx(Ns, ratio, "-o", color=C1, zorder=3)
    b.axhline(lim, color=MUTED, lw=0.9, ls=":")
    b.text(0.03, 0.90, "limit %.4f" % lim, transform=b.transAxes,
           fontsize=7, color=INK2)
    for x, y in ((Ns[0], ratio[0]), (Ns[-1], ratio[-1])):
        b.annotate("%.4f" % y, xy=(x, y), xytext=(0, -13),
                   textcoords="offset points", ha="center", fontsize=6.6,
                   color=INK2)
    b.set_xlabel("measurements  $N$")
    b.set_ylabel("per-measurement / per-run")
    b.set_ylim(ratio.min() - 0.02, lim + 0.012)
    b.set_title("Saving is bounded over %d decades" % (len(Ns) - 1))
    tag(b, "b")

    # (c) 3-D: the saving over the two quantities prop:cost actually
    # trades --- how many stages there are, and how large each stage's
    # provenance record is. The first sweep put run size N on the second
    # axis, but the ratio is nearly flat in N (2.4529 to 2.5145 across
    # four decades, which is panel (b)), so that axis carried no
    # variation and its 2..6 range read as a duplicate of the z-axis.
    #
    # Both surfaces are exp3's own expressions:
    #   per_meas = N * sum(log2 |Prv_i|)
    #   per_run  = N * s * log2(s) + sum(log2 |Prv_i|)
    # evaluated at N = 10^6, where the ratio has converged.
    c = ax[2]
    sizes = cost["record_sizes"]
    ss = np.arange(2, 13)
    bits = np.arange(2, 13)          # log2 of a per-stage record size
    Nbig = float(rows[-1]["N"])
    SS, BB = np.meshgrid(ss, bits, indexing="ij")
    per_meas = Nbig * SS * BB
    per_run = Nbig * SS * np.log2(SS) + SS * BB
    Rz = per_meas / per_run
    c.view_init(elev=26, azim=-58)
    c.plot_surface(SS, BB, Rz, cmap="Blues", linewidth=0, alpha=0.80,
                   rstride=1, cstride=1)
    # The recorded pipeline: six stages, whose six record sizes are
    # 2^6 .. 2^9. Its mean log-size is the ridge coordinate, and the
    # recorded ratio at N = 10^6 is the height it must reach.
    lg = np.log2(np.array(sizes, float))
    c.plot(np.full(len(lg), len(sizes)), lg,
           np.full(len(lg), rows[-1]["ratio"]), "-o", color=BAD, lw=2.0,
           markersize=4, zorder=12)
    c.set_xlabel("stages  $s$", labelpad=-2)
    c.set_ylabel(r"$\log_2 |Prv_i|$", labelpad=-2)
    c.set_xticks(ss[::2])
    c.set_zlabel("")
    c.set_title("Saving is set by stages and record size", y=1.0)
    c.text2D(0.00, 0.80, "red: the recorded" + chr(10) + "%d-stage pipeline"
             % len(sizes), transform=c.transAxes, fontsize=6.6, color=BAD)
    c.set_box_aspect(None, zoom=1.15)
    tag(c, "c", three_d=True)

    # (d) the record floor: classes to be distinguished grow release by
    # release and cross a fixed provenance capacity.
    d = ax[3]
    rf = R["record_floor"]
    hist = rf["history"]
    rel = [h["release"] for h in hist]
    ncl = [h["n_classes"] for h in hist]
    nct = [h["n_contexts"] for h in hist]
    cap = rf["Prv_capacity"]
    d.plot(rel, nct, "-s", color=MUTED, lw=1.4, label="contexts")
    d.plot(rel, ncl, "-o", color=C1, label="classes to distinguish")
    d.axhline(cap, color=BAD, lw=1.3, ls="--")
    d.text(0.97, 0.06, "capacity of Prv = %d" % cap, transform=d.transAxes,
           ha="right", fontsize=7.4, color=BAD)
    fits = [h["fits_in_Prv"] for h in hist]
    j = fits.index(False)
    d.scatter([rel[j]], [ncl[j]], s=110, facecolor="white", edgecolor=BAD,
              linewidth=1.8, zorder=6)
    d.annotate("exhausted at release %d" % rf["exhausted_at_release"],
               xy=(rel[j], ncl[j]), xytext=(4.6, 5.4), fontsize=7,
               color=BAD,
               arrowprops=dict(arrowstyle="->", color=BAD, lw=0.9,
                               shrinkB=9))
    d.set_xlabel("release")
    d.set_ylabel("count")
    d.set_xticks(rel[::2])
    d.set_title("A fixed record runs out")
    d.legend(loc="upper left")
    tag(d, "d")

    save(fig, PAPER, "panel2_record")


if __name__ == "__main__":
    panel1()
    panel2()
