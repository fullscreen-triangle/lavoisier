"""Panels 3 and 4 for "Detecting a Sink" --- the two FAILED claims.

exp5 grades this paper at 13 of 15. Both failures are real defects in the
manuscript rather than defects in the experiment, and both are drawn as
measured rather than softened.

  thm:spread-sound   the UPPER bound holds everywhere; the LOWER bound is
                     false. Recorded: 0 upper, 29 lower breaches over 237
                     pairs. Swept here over 2403 pairs: 0 upper, 330 lower.

  thm:threshold      the antecedent is unsatisfiable. Recorded: 0 fires
                     over 4779 tests, best margin -0.06724. Swept over
                     4776: still 0, same best margin.

The panels do more than restate the counts, because a count is one number
and these are four charts. Sweeping exp5's own definitions produced a
mechanism for each failure, and the mechanism is what is drawn:

  * Every one of the 330 lower-bound breaches --- deletion leaving the
    separation SMALLER than the theorem predicts --- is a case where the
    MINIMISER MOVED. Over 1979 pairs whose separating set was stable,
    the lower bound breached exactly 0 times. The theorem's arithmetic
    is sound; its hypothesis silently assumes S* is invariant under
    deletion, and it is not. This is a repair, not just a refutation:
    the bound holds verbatim once "S*(v) unchanged" is added.

  * thm:threshold's antecedent needs wspread(z) to exceed
    1 - beta/W, and the slack identity (sep(v) - beta) >= z's weight
    across S*(v) is what forbids it. Measured over 9808 triples the
    slack is non-negative in every one, with minimum -4.44e-16 --- a
    floating-point zero. The antecedent is not merely rarely met; it is
    blocked by an identity the paper's own definitions imply.
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

SS = SW["spread_sound"]
TH = SW["threshold"]

# The sweeps must reproduce the artefact's VERDICT, or they are measuring
# a different object than the experiment ran. exp5 draws from a shared
# rng, so the counts differ; the direction of each failure does not.
assert SS["upper_breaches"] == 0 == R["spread_sound_a"]["upper_bound_breaches"]
assert SS["lower_breaches"] > 0
assert R["spread_sound_a"]["lower_bound_breaches"] > 0
# every recorded counterexample is itself a MOVED minimiser --- the
# mechanism this panel draws is already visible in exp5's own records.
assert all(sorted(c["old_minimiser"]) != sorted(c["new_minimiser"])
           for c in R["spread_sound_a"]["lower_bound_counterexamples"])
assert TH["fires"] == 0 == R["threshold_satisfiability"]["strict_fires"]
assert abs(TH["best_margin"]
           - R["threshold_satisfiability"]["best_margin"]) < 1e-9
assert TH["negative_slacks"] == 0
assert TH["min_slack"] >= -1e-9

# The mechanism, asserted rather than asserted-in-prose: a stable
# minimiser never breaches the lower bound.
_PTS = SS["points"]
_STABLE = [p for p in _PTS if not p["moved"]]
_MOVED = [p for p in _PTS if p["moved"]]
assert sum(1 for p in _STABLE if p["new"] < p["pred"] - 1e-9) == 0
assert sum(1 for p in _MOVED
           if p["new"] < p["pred"] - 1e-9) == SS["lower_breaches"]


# =====================================================================
# Panel 3 --- the upper bound holds; the lower one fails, and why
# =====================================================================
def panel3():
    fig, ax = panel(three_d=(2,))

    pred = np.array([p["pred"] for p in _PTS], float)
    new = np.array([p["new"] for p in _PTS], float)
    moved = np.array([p["moved"] for p in _PTS], bool)

    # (a) the theorem as a scatter against its own prediction. The
    # theorem claims equality, so it claims the diagonal: the upper
    # bound is the half saying nothing sits ABOVE it, the lower bound
    # the half saying nothing sits BELOW. Pairs whose separating set
    # survived deletion land on the diagonal exactly. Pairs whose set
    # moved fall away from it, always DOWNWARD --- the separation after
    # deletion is SMALLER than predicted, which is the lower bound
    # failing and the upper one holding.
    a = ax[0]
    a.plot([pred.min(), pred.max()], [pred.min(), pred.max()],
           color=INK, lw=1.3, ls="--", zorder=6,
           label="what the theorem predicts")
    a.scatter(pred[~moved], new[~moved], s=13, color=GOOD, alpha=0.55,
              zorder=3, edgecolor="none",
              label="$S^*$ unchanged (%d)" % int((~moved).sum()))
    a.scatter(pred[moved], new[moved], s=15, color=BAD, alpha=0.70,
              zorder=4, edgecolor="none",
              label="$S^*$ moved (%d)" % int(moved.sum()))
    a.set_xlabel("predicted separation after deletion")
    a.set_ylabel("separation measured")
    a.set_title("The upper bound holds; the lower one does not")
    a.legend(loc="upper left", fontsize=6.6)
    a.text(0.97, 0.06, "%d upper breaches, %d lower, of %d"
           % (SS["upper_breaches"], SS["lower_breaches"], SS["cases"]),
           transform=a.transAxes, ha="right", fontsize=6.8, color=INK2)
    tag(a, "a")

    # (b) how far the lower bound is missed, as a share of the
    # separation. A single count says the bound fails; this says by how
    # much, and the answer is not a rounding error --- the median breach
    # understates by 8.5% of the separation and the worst by 58%.
    b = ax[1]
    dev = (pred - new) / np.array([p["sep"] for p in _PTS], float)
    br = dev > 1e-9
    b.hist(dev[br], bins=34, color=BAD, alpha=0.85, zorder=3)
    b.axvline(float(np.median(dev[br])), color=INK, lw=1.4, ls="--",
              zorder=5)
    b.text(float(np.median(dev[br])), b.get_ylim()[1] * 0.92,
           "  median %.3f" % float(np.median(dev[br])), fontsize=7,
           color=INK, va="top")
    b.set_xlabel("how far below the bound, as a share of sep($v$)")
    b.set_ylabel("pairs")
    b.set_title("The miss is a magnitude, not a rounding error")
    b.text(0.97, 0.62, "worst %.3f" % float(dev.max()) + chr(10)
           + "%d breaching pairs" % int(br.sum()), transform=b.transAxes,
           ha="right", fontsize=6.8, color=INK2)
    tag(b, "b")

    # (c) 3-D: the mechanism. Height is the measured separation minus
    # the predicted one, over (graph size, tau = the deleted vertex's
    # share of the cut). The two clouds are the two populations of (a):
    # every stable pair sits exactly on the zero plane at every n and
    # every tau, and every departure from it belongs to a moved
    # minimiser. This is the repair the theorem needs, drawn: add
    # "S*(v) unchanged" to the hypothesis and the plane is the theorem.
    c = ax[2]
    tau = np.array([p["tau"] for p in _PTS], float)
    nn = np.array([p["n"] for p in _PTS], float)
    gap = new - pred
    jit = np.random.default_rng(0).uniform(-0.16, 0.16, len(nn))
    c.view_init(elev=20, azim=-58)
    c.scatter(nn[~moved] + jit[~moved], tau[~moved], gap[~moved], s=6,
              color=GOOD, alpha=0.45, depthshade=False, zorder=4)
    c.scatter(nn[moved] + jit[moved], tau[moved], gap[moved], s=9,
              color=BAD, alpha=0.80, depthshade=False, zorder=6)
    for n_ in sorted(set(int(v) for v in nn)):
        c.plot([n_ - 0.42, n_ + 0.42], [0.0, 0.0], [0.0, 0.0],
               color=INK, lw=1.4, zorder=20)
    c.set_xlabel("items  $n$", labelpad=-2)
    c.set_ylabel(r"$\tau$", labelpad=-2)
    c.set_xticks(sorted(set(int(v) for v in nn)))
    c.set_zlabel("")
    c.set_title("Only a moved minimiser leaves the plane", y=1.04)
    c.text2D(0.50, -0.10, "height: measured $-$ predicted    "
             + "green: $S^*$ unchanged    red: $S^*$ moved",
             transform=c.transAxes, ha="center", fontsize=6.2, color=INK2)
    c.set_box_aspect(None, zoom=1.15)
    tag(c, "c", three_d=True)

    # (d) the two claims separated by graph size, which is what says the
    # failure is structural rather than a small-graph artefact. The
    # upper bound holds at every size; the lower one fails at a rate
    # that does not decay.
    d = ax[3]
    bn = SS["by_n"]
    ns = np.array([r["n"] for r in bn], float)
    up = np.array([r["upper"] / float(r["cases"]) for r in bn])
    lo = np.array([r["lower"] / float(r["cases"]) for r in bn])
    mvr = np.array([r["moved"] / float(r["cases"]) for r in bn])
    d.plot(ns, lo, "-o", color=BAD, markersize=6, zorder=5,
           label="lower bound breached")
    d.plot(ns, up, "-s", color=GOOD, markersize=6, zorder=5,
           label="upper bound breached")
    d.plot(ns, mvr, "--^", color=C2, markersize=5, zorder=4,
           label=r"$S^*$ moved at all")
    d.fill_between(ns, 0, lo, color=BAD, alpha=0.10)
    d.set_xticks(ns)
    d.set_ylim(-0.02, max(mvr.max(), lo.max()) * 1.45)
    d.set_xlabel("items  $n$")
    d.set_ylabel("share of pairs")
    d.set_title("The failure does not thin out with size")
    d.legend(loc="upper right", fontsize=6.6)
    d.text(0.03, 0.20, "%d pairs" % SS["cases"] + chr(10)
           + "the lower rate is %.0f%% of the moved rate"
           % (100.0 * lo.sum() / mvr.sum()),
           transform=d.transAxes, va="bottom", fontsize=6.6, color=INK2)
    tag(d, "d")

    save(fig, PAPER, "panel3_spread_sound")


# =====================================================================
# Panel 4 --- a threshold that can never fire
# =====================================================================
def panel4():
    fig, ax = panel(three_d=(1,))

    mg = np.array([m["margin"] for m in TH["margins"]], float)
    ws = np.array([m["wspread"] for m in TH["margins"]], float)
    thr = np.array([m["thr"] for m in TH["margins"]], float)
    mn = np.array([m["n"] for m in TH["margins"]], float)

    # (a) the antecedent as the two quantities it compares. The theorem
    # fires when a vertex's weighted spread exceeds the forced
    # threshold; the diagonal is that condition. Every one of the 4776
    # tested vertices lies strictly below it, and the closest miss is
    # -0.067 --- not a near-miss at all, on a quantity bounded by 1.
    a = ax[0]
    lim = [0.0, 1.0]
    a.plot(lim, lim, color=INK, lw=1.4, ls="--", zorder=6,
           label="where the theorem would fire")
    for n_, col in zip(sorted(set(int(v) for v in mn)), [C1, C2, C4]):
        sel = mn == n_
        a.scatter(thr[sel], ws[sel], s=9, color=col, alpha=0.55,
                  edgecolor="none", zorder=3, label="$n$ = %d" % n_)
    jbest = int(np.argmax(mg))
    a.scatter([thr[jbest]], [ws[jbest]], s=120, facecolor="white",
              edgecolor=BAD, linewidth=1.8, zorder=8)
    a.annotate("closest: %.4f short" % (-mg[jbest]),
               xy=(thr[jbest], ws[jbest]), xytext=(0.30, 0.90),
               textcoords="axes fraction", fontsize=7, color=BAD,
               arrowprops=dict(arrowstyle="->", color=BAD, lw=0.9,
                               shrinkB=10))
    a.set_xlim(*lim)
    a.set_ylim(*lim)
    a.set_xlabel(r"forced threshold  $1 - \beta/W$")
    a.set_ylabel("weighted spread of the vertex")
    a.set_title("%d vertices tested, %d above the line"
                % (TH["tested"], TH["fires"]))
    a.legend(loc="upper left", fontsize=6.6)
    tag(a, "a")

    # (b) 3-D: the identity that forbids it. thm:threshold needs a
    # vertex z to carry more of the cut than the floor leaves free, and
    # the quantity that decides this is the slack
    # (sep(v) - beta) - w(z crossing S*(v)). Height is that slack over
    # (graph size, z's weight across the cut). It is non-negative in all
    # 9808 triples, with minimum -4.44e-16 --- a floating-point zero, and
    # the surface touches the plane rather than crossing it. The
    # antecedent is not rarely met; it is blocked.
    b = ax[1]
    sl = TH["slacks"]
    sn = np.array([s["n"] for s in sl], float)
    zw = np.array([s["zw"] for s in sl], float)
    sv = np.array([s["slack"] for s in sl], float)
    jit = np.random.default_rng(1).uniform(-0.20, 0.20, len(sn))
    b.view_init(elev=18, azim=-58)
    tight = sv < 1e-9
    b.scatter(sn[~tight] + jit[~tight], zw[~tight], sv[~tight], s=5,
              color=C1, alpha=0.35, depthshade=False, zorder=4)
    b.scatter(sn[tight] + jit[tight], zw[tight], sv[tight], s=16,
              color=BAD, alpha=0.95, depthshade=False, zorder=8)
    for n_ in sorted(set(int(v) for v in sn)):
        b.plot([n_ - 0.35, n_ + 0.35], [0.0, 0.0], [0.0, 0.0],
               color=INK, lw=1.2, zorder=9)
    b.set_xlabel("items  $n$", labelpad=-2)
    b.set_ylabel(r"$z$'s weight across $S^*$", labelpad=-2)
    b.set_xticks(sorted(set(int(v) for v in sn)))
    b.set_zlabel("")
    b.set_title("The slack never goes negative", y=1.04)
    b.text2D(0.50, -0.10, "height: $(\\mathrm{sep}(v)-\\beta) - w(z)$    "
             + "red: exactly tight (%d)    min %.2g"
             % (int(tight.sum()), TH["min_slack"]),
             transform=b.transAxes, ha="center", fontsize=6.2, color=INK2)
    b.set_box_aspect(None, zoom=1.12)
    tag(b, "b", three_d=True)

    # (c) the constructions where z demonstrably DOES supply the floor
    # --- exactly the situation thm:threshold is about --- swept over the
    # sink's weight. Even here the spread falls away from the threshold
    # rather than meeting it: as lambda shrinks the threshold rises and
    # the spread collapses, so the gap widens in the very regime the
    # theorem was written for. exp5's three recorded cells are marked.
    c = ax[2]
    fr = TH["floor_rows"]
    rec = {(x["n"], x["lambda"]): x["margin"]
           for x in R["threshold_satisfiability"][
               "floor_supplying_constructions"]}
    # The spread curve is drawn ONCE, not three times: wspread(z) is
    # identical at n = 4, 5 and 6 on this construction (0.005, 0.010,
    # 0.020, 0.048, 0.091, 0.200, 0.333 in all three), because z's share
    # of the cut does not depend on how many ordinary items surround it.
    # Three overlapping dashed curves would be three marks carrying one
    # measurement.
    ll = np.array([r["lam"] for r in fr if r["n"] == 4], float)
    for n_, col in zip((4, 5, 6), [C1, C2, C4]):
        rr = [r for r in fr if r["n"] == n_]
        c.semilogx(ll, [r["thr"] for r in rr], "-", color=col, lw=1.8,
                   label="threshold,  $n$ = %d" % n_)
    ws_ = np.array([r["wspread"] for r in fr if r["n"] == 4], float)
    thr4 = np.array([r["thr"] for r in fr if r["n"] == 4], float)
    c.fill_between(ll, ws_, thr4, color=BAD, alpha=0.09, zorder=1)
    c.semilogx(ll, ws_, "--o", color=INK, lw=1.6, markersize=4, zorder=4,
               label="spread of $z$ (same at every $n$)")
    for (n_, lam), m_ in sorted(rec.items()):
        rr = [r for r in fr if r["n"] == n_ and abs(r["lam"] - lam) < 1e-12][0]
        c.scatter([lam], [rr["wspread"]], s=110, facecolor="white",
                  edgecolor=BAD, linewidth=1.7, zorder=8)
    c.set_ylim(-0.03, 1.16)
    c.set_xlabel(r"the sink's weight  $\lambda$")
    c.set_ylabel("threshold and weighted spread")
    c.set_title("Even where $z$ supplies the floor, the gap is wide")
    c.legend(loc="center left", fontsize=6.4)
    c.text(0.97, 0.30, "circles: exp5's three recorded cells" + chr(10)
           + "margins %s" % ", ".join("%.3f" % v for _, v in sorted(
               rec.items())), transform=c.transAxes, ha="right",
           fontsize=6.6, color=BAD)
    tag(c, "c")

    # (d) the margin's distribution, which is the honest summary of the
    # claim's status. A theorem whose antecedent is merely rare would
    # show a distribution pressed against zero; this one is centred far
    # from it, and its best case over 4776 vertices is still -0.067.
    d = ax[3]
    for n_, col in zip(sorted(set(int(v) for v in mn)), [C1, C2, C4]):
        sel = mn == n_
        d.hist(mg[sel], bins=40, histtype="step", lw=1.8, color=col,
               label="$n$ = %d" % n_)
    d.axvline(0.0, color=INK, lw=1.5, ls="--", zorder=6)
    d.axvline(float(mg.max()), color=BAD, lw=1.3, zorder=5)
    d.set_xlabel("margin:  weighted spread $-$ threshold")
    d.set_ylabel("vertices")
    d.set_title("The best case over %d vertices is %.4f"
                % (TH["tested"], TH["best_margin"]))
    d.legend(loc="upper left", fontsize=6.6)
    d.text(0.98, 0.60, "dashed: where it would fire" + chr(10)
           + "red: the closest any vertex came",
           transform=d.transAxes, ha="right", fontsize=6.6, color=INK2)
    tag(d, "d")

    save(fig, PAPER, "panel4_threshold")


if __name__ == "__main__":
    panel3()
    panel4()
