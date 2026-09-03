"""
Panels 3 and 4 for "The Runtime Graph: Acquisition as a Queryable Object".

These spend the records panels 1 and 2 left: `medium` /
`medium_control`, `slack_bound`, `not_a_cut` / `not_a_cut_control`, and
`sensitivity_a`. Each is a scalar or a boolean at one fixed cell ---
`cut_without_medium` 0.0, `overstatements` 0 of 21 cases, `breaches` 0
of 400 trials --- so drawing the records alone would give four charts of
a single number.

As in panels 1 and 2, exp4's own definitions are imported and evaluated
over the parameters exp4 fixed, and the recorded scalar is marked as the
point the resulting curve must pass through. The sweeps are cached in
results/exp4_sweeps.json.

Three notes on getting the constructions right, all earned by getting
them wrong first:

  * thm:sensitivity(a) bounds the separation VALUE by eps*M --- not by
    eps*(m+M), which is the (b) guard on the separating SET. Using the
    (b) constant here reports a bound roughly four times looser than the
    one the theorem states.

  * exp4 runs sensitivity on `Gs2`, a FOUR-vertex graph, not on the
    six-vertex floor/monotone graph. Sweeping the wrong one gives M = 8,
    m = 3 against the recorded M = 5, m = 2. The assertion below pins
    both.

  * thm:not-a-cut is a claim about the key failing to be a COMPLETE
    invariant, so the measurement is how many distinct graphs share a
    key vector, not whether one witness pair exists. Enumerated
    exhaustively at n = 4 and n = 5 over the two-weight family: 57,888
    graphs at n = 5 realise only 162 distinct key vectors, and every one
    of those classes is shared.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..")))

from exp4_runtime_graph import (                    # noqa: E402
    MED, admissible_sets, crossing_count, cut, make_graph, separation)
from panelkit import (C1, C2, C3, C4, GOOD, BAD, INK, INK2, MUTED,   # noqa: E402
                      RESULTS, panel, tag, load, save)

PAPER = "runtime-graph"
D = load("exp4_runtime_graph")
R = D["records"]

with open(os.path.join(RESULTS, "exp4_sweeps.json"), encoding="utf8") as _fh:
    SW = json.load(_fh)

CONTACTS = {("u0", "u1"): 0.90, ("u1", "u2"): 0.70, ("u0", "u2"): 0.55,
            ("u3", "u4"): 0.80, ("u4", "u5"): 0.40}
MED_W = [0.30, 0.25, 0.35, 0.20, 0.28, 0.33]
G = make_graph(6, CONTACTS, MED_W)
U = ["u%d" % i for i in range(6)]

# exp4's sensitivity graph is a DIFFERENT, four-vertex object.
GS2 = make_graph(4, {("u0", "u1"): 0.50, ("u1", "u2"): 0.34,
                     ("u2", "u3"): 0.30}, [0.30, 0.32, 0.31, 0.33])

# The reconstructions must reproduce the artefact, or they are measuring
# a different object than the experiment ran.
assert SW["sens_a"]["M"] == R["sensitivity_a"]["M"]
assert SW["sens_a"]["m"] == R["sensitivity_a"]["m"]
assert all(r["breaches"] == 0 for r in SW["sens_a"]["rows"])
assert SW["slack_bound"]["overstatements"] == \
    R["slack_bound"]["overstatements"]
_m1 = [r for r in SW["medium"]["rows"] if r["mult"] == 1.0][0]
assert all(abs(v - R["medium"]["cut_without_medium"]) < 1e-12
           for v in _m1["without_medium"])
assert abs(min(_m1["with_medium"]) - R["floor"]["min_sep"]) < 1e-9


# =====================================================================
# Panel 3 --- without the medium there is nothing to separate
# =====================================================================
def panel3():
    fig, ax = panel(three_d=(1,))

    md = SW["medium"]
    mults = np.array(md["mults"], float)

    # (a) every vertex's separation with the medium present, against the
    # same vertex with the medium deleted, swept over a multiplier on
    # every medium weight. The recorded control is a single number
    # (cut_without_medium = 0.0); this is that number for all six
    # vertices at every medium strength --- deleting the medium does not
    # weaken the separation, it destroys it.
    a = ax[0]
    with_ = np.array([r["with_medium"] for r in md["rows"]])
    wo = np.array([r["without_medium"] for r in md["rows"]])
    for i, u in enumerate(U):
        a.plot(mults, with_[:, i], "-", color=C1, lw=1.4, alpha=0.75)
    a.plot(mults, wo.max(axis=1), "-o", color=BAD, lw=2.2, markersize=5,
           zorder=5, label="medium deleted (largest of the six)")
    a.plot([], [], color=C1, lw=1.4, label="medium present (six vertices)")
    j1 = int(np.argmin(np.abs(mults - 1.0)))
    a.axvline(1.0, color=MUTED, lw=0.9, ls=":")
    a.scatter([1.0], [R["medium"]["cut_without_medium"]], s=120,
              facecolor="white", edgecolor=BAD, linewidth=1.8, zorder=7)
    a.annotate("recorded: cut = %.1f"
               % R["medium"]["cut_without_medium"],
               xy=(1.0, R["medium"]["cut_without_medium"]),
               xytext=(0.40, 0.30), textcoords="axes fraction",
               fontsize=7, color=BAD,
               arrowprops=dict(arrowstyle="->", color=BAD, lw=0.9,
                               shrinkB=9))
    a.set_xlabel("multiplier on every medium weight")
    a.set_ylabel("separation")
    a.set_title("Delete the medium and separation is free")
    a.legend(loc="upper left", fontsize=6.8)
    tag(a, "a")

    # (b) 3-D: the same collapse per vertex. The upper sheet is the
    # measured separation with the medium, rising with the medium's
    # strength; the lower one is the zero floor the graph falls to
    # without it, flat across every vertex and every multiplier.
    b = ax[1]
    MM, VV = np.meshgrid(mults, np.arange(len(U)), indexing="ij")
    b.view_init(elev=18, azim=-52)
    b.plot_surface(MM, VV, with_, cmap="Blues", linewidth=0, alpha=0.82,
                   rstride=1, cstride=1)
    b.plot_surface(MM, VV, wo, color=BAD, alpha=0.55, linewidth=0,
                   shade=False)
    b.plot(np.full(len(U), 1.0), np.arange(len(U)), with_[j1], "-o",
           color=INK, lw=1.8, markersize=4, zorder=14)
    b.set_xlabel("medium multiplier", labelpad=-2)
    b.set_ylabel("observation", labelpad=-2)
    b.set_yticks(range(len(U)))
    b.set_yticklabels(U, fontsize=6)
    b.set_zlabel("")
    b.set_title("The floor is the medium, not the contacts", y=1.04)
    b.text2D(0.00, 0.80, "blue: medium present" + chr(10)
             + "red: medium deleted, flat at 0" + chr(10)
             + "black: as measured", transform=b.transAxes,
             fontsize=6.2, color=INK2)
    b.set_box_aspect(None, zoom=1.15)
    tag(b, "b", three_d=True)

    # (c) thm:not-a-cut as the invariant failure it is. The record is a
    # single witness pair; the claim is that the cut key does not
    # determine the graph. Enumerated exhaustively over the two-weight
    # family: how many graphs exist against how many distinct key
    # vectors they realise.
    c = ax[2]
    nc = SW["not_a_cut"]["rows"]
    ns = [r["n"] for r in nc]
    xi = np.arange(len(ns))
    gr = [r["graphs"] for r in nc]
    dk = [r["distinct_keys"] for r in nc]
    lc = [r["largest_class"] for r in nc]
    c.bar(xi - 0.19, gr, width=0.36, color=C1, zorder=3,
          label="graphs enumerated")
    c.bar(xi + 0.19, dk, width=0.36, color=BAD, zorder=3,
          label="distinct key vectors")
    c.plot(xi, lc, "-o", color=INK, markersize=5, zorder=5,
           label="largest class sharing one key")
    c.set_yscale("log")
    c.set_xticks(xi)
    c.set_xticklabels(["$n$ = %d" % n for n in ns])
    c.set_xlabel("observations in the graph")
    c.set_ylabel("count")
    c.set_ylim(0.6, max(gr) * 90.0)
    c.set_title("Every key at $n$ = %d is shared" % ns[-1])
    c.legend(loc="upper center", fontsize=6.6)
    for i, r in enumerate(nc):
        c.text(i, r["graphs"] * 2.2, "%d graphs" % r["graphs"] + chr(10)
               + "into %d keys" % r["distinct_keys"], ha="center",
               fontsize=6.6, color=INK2)
    tag(c, "c")

    # (d) how much the key resolves, swept over the richness of the
    # weight alphabet. The first version of this chart scattered exp4's
    # six vertices by their two key components, but that graph realises
    # only three distinct keys, so it drew three dots with six labels
    # overprinting them --- almost no measured content. The claim is
    # about resolution, and resolution is measurable: enumerate every
    # graph over a k-value weight alphabet and count the distinct key
    # vectors they realise. Enriching the weights makes the key WORSE,
    # not better, because graphs multiply faster than keys do.
    d = ax[3]
    kr = SW["key_resolution"]["rows"]
    al = np.array([r["alphabet"] for r in kr], float)
    grs = np.array([r["graphs"] for r in kr], float)
    dks = np.array([r["distinct_keys"] for r in kr], float)
    lcs = np.array([r["largest_class"] for r in kr], float)
    d.semilogy(al, grs, "-o", color=C1, markersize=5,
               label="graphs enumerated")
    d.semilogy(al, dks, "-s", color=BAD, markersize=5,
               label="distinct key vectors")
    d.fill_between(al, dks, grs, color=BAD, alpha=0.09, zorder=1)
    d.semilogy(al, lcs, "--^", color=C2, markersize=5, zorder=4,
               label="largest class sharing a key")
    d.set_xticks(al)
    d.set_xlabel("distinct contact weights available")
    d.set_ylabel("count  ($n$ = %d, exhaustive)" % SW["key_resolution"]["n"])
    d.set_ylim(0.6, grs.max() * 40.0)
    d.set_title("Richer weights resolve %.1f%% to %.1f%%"
                % (kr[0]["resolution"] * 100.0,
                   kr[-1]["resolution"] * 100.0))
    d.legend(loc="upper left", fontsize=6.6)
    d.text(0.97, 0.06, "shaded: graphs the key cannot tell apart",
           transform=d.transAxes, ha="right", fontsize=6.8, color=INK2)
    tag(d, "d")

    save(fig, PAPER, "panel3_medium")


# =====================================================================
# Panel 4 --- the answer's stability, bounded and measured
# =====================================================================
def panel4():
    fig, ax = panel(three_d=(2,))

    sa = SW["sens_a"]
    rows = sa["rows"]
    es = np.array([r["eps"] for r in rows], float)
    bd = np.array([r["bound"] for r in rows], float)
    wsh = np.array([r["worst_shift"] for r in rows], float)
    msh = np.array([r["mean_shift"] for r in rows], float)

    # (a) thm:sensitivity(a): the separation value moves by at most
    # eps*M. Both the bound and the measured worst move over 400 trials
    # per eps, on exp4's own four-vertex graph with its own measured
    # M = 5. The bound holds everywhere and is loose by a constant
    # factor --- it is a guarantee, not an estimate.
    a = ax[0]
    a.loglog(es, bd, "-", color=BAD, lw=2.0,
             label=r"the bound  $\epsilon M$,  $M$ = %d" % sa["M"])
    a.loglog(es, wsh, "-o", color=C1, markersize=5,
             label="worst move measured")
    a.loglog(es, msh, "--s", color=C2, markersize=4,
             label="mean move measured")
    a.fill_between(es, wsh, bd, color=GOOD, alpha=0.14)
    a.set_xlabel(r"perturbation  $\epsilon$")
    a.set_ylabel("move in the separation value")
    a.set_title("%d breaches in %d trials per $\\epsilon$"
                % (sum(r["breaches"] for r in rows), sa["trials"]))
    a.legend(loc="upper left", fontsize=6.6)
    a.text(0.97, 0.06, "green: unused headroom" + chr(10)
           + "recorded: %d of %d breaches"
           % (R["sensitivity_a"]["breaches"],
              R["sensitivity_a"]["trials"]), transform=a.transAxes,
           ha="right", fontsize=6.8, color=INK2)
    tag(a, "a")

    # (b) how loose the bound is, which is the quantity a user of
    # cor:no-rerun actually needs. Ratio of measured worst move to the
    # bound, over the same sweep: flat, so the bound tracks the right
    # scale but overstates by a fixed factor.
    b = ax[1]
    ratio = wsh / bd
    mratio = msh / bd
    b.semilogx(es, ratio, "-o", color=C1, markersize=5,
               label="worst / bound")
    b.semilogx(es, mratio, "--s", color=C2, markersize=4,
               label="mean / bound")
    b.axhline(1.0, color=BAD, lw=1.3, ls="--")
    b.text(es[0] * 1.1, 1.02, "the bound", fontsize=7, color=BAD,
           va="bottom")
    b.set_ylim(0, 1.14)
    b.set_xlabel(r"perturbation  $\epsilon$")
    b.set_ylabel("share of the bound used")
    b.set_title("Loose by %.1fx, and evenly so" % (1.0 / ratio.mean()))
    b.legend(loc="center right", fontsize=6.8)
    tag(b, "b")

    # (c) 3-D: def:slack's guard against the perturbation that actually
    # moved the separating set, per query. For each of exp4's queries
    # the guard says "no move below eps = slack/(m+M)"; the measured
    # value is the smallest eps at which a move was observed, found by
    # bisection. The guard surface lies below the measured one at every
    # query, which is the claim: it never overstates.
    c = ax[2]
    sb = SW["slack_bound"]["cases"]
    order = sorted(range(len(sb)), key=lambda i: sb[i]["guard_eps"])
    gi = np.array([sb[i]["guard_eps"] for i in order], float)
    ei = np.array([sb[i]["empirical_eps"] for i in order], float)
    qs = np.array([len(sb[i]["Q"]) for i in order], float)
    xi = np.arange(len(order), dtype=float)
    c.view_init(elev=22, azim=-62)
    for k in range(len(order)):
        c.plot([xi[k], xi[k]], [qs[k], qs[k]], [gi[k], ei[k]],
               color=MUTED, lw=0.8, alpha=0.6, zorder=3)
    c.scatter(xi, qs, ei, s=22, color=C1, depthshade=False, zorder=8,
              label="smallest move measured")
    c.scatter(xi, qs, gi, s=22, color=BAD, depthshade=False, zorder=9,
              label="the guard")
    c.set_xlabel("query", labelpad=-2)
    c.set_ylabel("|Q|", labelpad=-2)
    c.set_yticks(sorted(set(int(q) for q in qs)))
    c.set_zlabel("")
    c.set_title("The guard never overstates", y=1.04)
    c.text2D(0.00, 0.80, "red: guard $s/(m{+}M)$" + chr(10)
             + "blue: measured" + chr(10)
             + "%d queries, %d overstatements"
             % (len(sb), SW["slack_bound"]["overstatements"]),
             transform=c.transAxes, fontsize=6.2, color=INK2)
    c.set_box_aspect(None, zoom=1.15)
    tag(c, "c", three_d=True)

    # (d) the same 41 queries read as the headroom the guard leaves:
    # how much larger the perturbation could actually have been before
    # the set moved. Every point above the diagonal is a query where the
    # guard was conservative; nothing lies below it.
    d = ax[3]
    gi2 = np.array([c_["guard_eps"] for c_ in sb], float)
    ei2 = np.array([c_["empirical_eps"] for c_ in sb], float)
    qs2 = np.array([len(c_["Q"]) for c_ in sb], int)
    lim = [min(gi2.min(), ei2.min()) * 0.7, max(gi2.max(), ei2.max()) * 1.4]
    d.plot(lim, lim, color=BAD, lw=1.3, ls="--", zorder=2)
    for q, col in zip(sorted(set(qs2)), [C1, C2, C4]):
        sel = qs2 == q
        d.scatter(gi2[sel], ei2[sel], s=44, color=col, alpha=0.80,
                  zorder=4, edgecolor="white", linewidth=0.8,
                  label="|Q| = %d" % q)
    d.set_xscale("log")
    d.set_yscale("log")
    d.set_xlim(*lim)
    d.set_ylim(*lim)
    d.set_xlabel(r"the guard's $\epsilon$")
    d.set_ylabel(r"$\epsilon$ that actually moved $S^*$")
    d.set_title("Every query sits above the diagonal")
    d.legend(loc="lower right", fontsize=6.8)
    d.text(0.03, 0.04, "above the dashed line:" + chr(10)
           + "the guard was safe", transform=d.transAxes,
           va="bottom", fontsize=6.8, color=INK2)
    tag(d, "d")

    save(fig, PAPER, "panel4_stability")


if __name__ == "__main__":
    panel3()
    panel4()
