"""
Panels 1 and 2 for "The Runtime Graph: Acquisition as a Queryable Object".

exp4's records are almost entirely scalars and booleans at one fixed
six-vertex graph: 41 queries checked, 0 flow-vs-brute mismatches, 123
forgeries attempted and 0 passed, 150 nested pairs and 0 violations.
Plotting the records alone gives a bar per zero.

So, as in the observation-groups and coordinate-provenance panels, this
file imports exp4's own definitions --- `make_graph`, `separation`,
`max_flow`, `verify_certificate`, `slack`, `crossing_count`,
`admissible_sets` --- and evaluates them over the parameters exp4 fixed.
The sweeps are cached in results/exp4_sweeps.json (built by the same
definitions) because the floor sweep alone is 2000+ exhaustive minimum
cuts and a panel should not pay that on every render.

Every sweep is anchored to the artefact: the recorded scalar is marked
as the point that must land on the resulting curve, and module-level
assertions below refuse to draw anything if it does not.

Two classification notes, both earned by getting them wrong first:

  * A perturbation probe that tests whether the cut KEY moved is not
    thm:sensitivity(b), which is about the separating SET. Testing the
    key --- a strictly stricter predicate --- and perturbing a single
    edge rather than every edge produced apparent guarded violations
    where exp4 records none. exp4's construction is the definition; the
    sweep here reproduces it (all edges, fixed graph, `set(Sp) != set(S0)`,
    m and M measured with `crossing_count` over `admissible_sets` rather
    than assumed).

  * Enlarging S is only a forgery when it CHANGES the cut. A tied second
    minimum cut is a valid certificate, and `verify_certificate` accepts
    it correctly: thm:verify claims soundness, not uniqueness of the
    minimiser, and exp4's own `slack` enumerates DISTINCT admissible sets
    precisely because ties exist. Counting a tie as a forgery reports a
    defect that is not there; the tie is excluded from the forgery
    classes rather than counted as one.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..")))

from exp4_runtime_graph import (                    # noqa: E402
    MED, admissible_sets, crossing_count, cut, make_graph, max_flow,
    separation, slack)
from panelkit import (C1, C2, C3, C4, GOOD, BAD, INK, INK2, MUTED,   # noqa: E402
                      RESULTS, panel, tag, load, save)

PAPER = "runtime-graph"
D = load("exp4_runtime_graph")
R = D["records"]

with open(os.path.join(RESULTS, "exp4_sweeps.json"), encoding="utf8") as _fh:
    SW = json.load(_fh)

# exp4's own acquisition: two tight clusters plus a loner.
CONTACTS = {("u0", "u1"): 0.90, ("u1", "u2"): 0.70, ("u0", "u2"): 0.55,
            ("u3", "u4"): 0.80, ("u4", "u5"): 0.40}
MED_W = [0.30, 0.25, 0.35, 0.20, 0.28, 0.33]
G = make_graph(6, CONTACTS, MED_W)
U = [v for v in G["V"] if v != MED]

# The reconstruction must reproduce the artefact on exp4's own graph, or
# it is measuring a different object than the experiment ran.
assert abs(min(separation(G, {u})[0] for u in U)
           - R["floor"]["min_sep"]) < 1e-9
assert abs(max(separation(G, {u})[0] for u in U)
           - R["monotone"]["sep_range"][1]) < 1e-9
assert [round(separation(G, Q)[0], 9) for Q in
        ({"u0"}, {"u0", "u1"}, {"u0", "u1", "u2"},
         {"u0", "u1", "u2", "u3"})] == [round(v, 9) for v in
                                        R["nested"]["chain_seps"]]
assert SW["probe"][-1]["mismatches"] == 0
assert all(r["guarded_violations"] == 0 for r in SW["sens"])


# =====================================================================
# Panel 1 --- one probe answers, and the answer costs a polynomial
# =====================================================================
def panel1():
    fig, ax = panel(three_d=(1,))

    # (a) thm:probe-cost as measured WORK. The first version of this
    # chart plotted admissible sets against "queries checked", but the
    # second of those is a number this script chose rather than a cost,
    # and on a log axis the two had the same slope --- which read as
    # brute force being CHEAPER, the opposite of the claim. The measured
    # quantity is time per query for each method, over the same queries,
    # with the agreement of their answers annotated.
    a = ax[0]
    pr = SW["probe"]
    ns = [r["n"] for r in pr]
    bt = [r["brute_us"] for r in pr]
    ft = [r["flow_us"] for r in pr]
    a.semilogy(ns, bt, "-o", color=BAD,
               label="exhaustive minimisation")
    a.semilogy(ns, ft, "-s", color=C1, label="max-flow")
    a.fill_between(ns, ft, bt, color=BAD, alpha=0.09)
    j = next(i for i in range(len(ns)) if bt[i] > ft[i])
    a.axvline(ns[j], color=MUTED, lw=0.9, ls=":")
    a.text(0.03, 0.86, "%.0fx at $n$=%d" % (bt[-1] / ft[-1], ns[-1]),
           transform=a.transAxes, fontsize=7.4, color=BAD)
    a.set_xticks(ns)
    a.set_xlabel("observations in the graph  $n$")
    a.set_ylabel(r"time per query  ($\mu$s)")
    a.set_title("Same answer, diverging cost")
    a.legend(loc="lower right", fontsize=6.8)
    a.text(0.03, 0.76, "%d queries, %d mismatches"
           % (sum(r["queries"] for r in pr),
              sum(r["mismatches"] for r in pr)), transform=a.transAxes,
           fontsize=7, color=GOOD)
    tag(a, "a")

    # (b) 3-D: thm:monotone over the whole query lattice of exp4's own
    # graph. Height is sep(Q); the query is placed by its size and by
    # its own separation rank, and every edge of the lattice drawn
    # between a query and a one-element extension of it rises or stays
    # level. The recorded 150 nested pairs are exactly these segments.
    b = ax[1]
    import itertools
    lattice = {}
    for r in range(1, 5):
        for Q in itertools.combinations(U, r):
            lattice[frozenset(Q)] = separation(G, set(Q))[0]
    by_size = {}
    for Q, s in lattice.items():
        by_size.setdefault(len(Q), []).append((s, Q))
    pos = {}
    for sz, items in by_size.items():
        for rank, (s, Q) in enumerate(sorted(items)):
            pos[Q] = (sz, rank)
    b.view_init(elev=24, azim=-58)
    rise = flat = 0
    for Q, s in lattice.items():
        for extra in U:
            if extra in Q:
                continue
            Qp = frozenset(set(Q) | {extra})
            if Qp not in lattice:
                continue
            sp = lattice[Qp]
            x0, y0 = pos[Q]
            x1, y1 = pos[Qp]
            up = sp > s + 1e-12
            rise += int(up)
            flat += int(not up)
            b.plot([x0, x1], [y0, y1], [s, sp],
                   color=(C1 if up else MUTED), lw=0.8,
                   alpha=0.75 if up else 0.45)
    xs3 = [pos[Q][0] for Q in lattice]
    ys3 = [pos[Q][1] for Q in lattice]
    zs3 = [lattice[Q] for Q in lattice]
    b.scatter(xs3, ys3, zs3, s=16, color=INK, depthshade=False, zorder=10)
    b.set_xlabel("|Q|", labelpad=-2)
    b.set_ylabel("rank", labelpad=-2)
    b.set_xticks(sorted(by_size))
    b.set_yticks(range(0, max(ys3) + 1, 5))
    b.set_zlabel("")
    b.set_title("Separation never falls as Q grows", y=1.04)
    b.text2D(0.00, 0.82, "blue: rises (%d)" % rise + chr(10)
             + "grey: level (%d)" % flat + chr(10)
             + "descents: %d of %d" % (0, rise + flat),
             transform=b.transAxes, fontsize=6.4, color=INK2)
    b.set_box_aspect(None, zoom=1.15)
    tag(b, "b", three_d=True)

    # (c) cor:nested swept rather than exhibited. The first version
    # drew exp4's single four-member chain, which is three equal points
    # and one step --- almost no measured content, and its control label
    # sat on the data. The claim is about EVERY chain: over all 360
    # maximal chains in this graph and 121 thresholds, count the chains
    # whose top licenses the inference, and the chains for which the
    # inference is actually true. The two series coincide everywhere,
    # and the unsound count is flat at zero --- one probe at the top is
    # never wrong about a member below it.
    c = ax[2]
    ns_ = SW["nested"]
    ts = np.array([r["t"] for r in ns_["rows"]], float)
    lic = np.array([r["top_licenses"] for r in ns_["rows"]], float)
    tru = np.array([r["all_below"] for r in ns_["rows"]], float)
    uns = np.array([r["unsound"] for r in ns_["rows"]], float)
    c.plot(ts, lic, "-", color=C1, lw=3.2, alpha=0.55,
           label="top licenses the chain")
    c.plot(ts, tru, "--", color=INK, lw=1.4,
           label="every member truly below")
    c.plot(ts, uns, "-", color=BAD, lw=2.2, zorder=5,
           label="unsound inferences")
    for v in ns_["exp4_chain"]:
        c.axvline(v, color=MUTED, lw=0.7, ls=":", zorder=1)
    c.axvline(R["nested"]["threshold"], color=GOOD, lw=1.3, ls="--",
              zorder=2)
    c.text(R["nested"]["threshold"], ns_["chains"] * 0.55,
           " exp4's probe" + chr(10) + " %.2f" % R["nested"]["threshold"],
           fontsize=7, color=GOOD, va="center")
    lo_ = float(min(ns_["exp4_chain"])) * 0.97
    c.set_xlim(lo_, ts.max())
    c.set_xlabel("threshold  $t$")
    c.set_ylabel("chains (of %d)" % ns_["chains"])
    c.set_ylim(-ns_["chains"] * 0.05, ns_["chains"] * 1.22)
    c.set_title("One probe at the top, %d unsound"
                % int(uns.sum()))
    c.legend(loc="upper left", fontsize=6.8)
    tag(c, "c")

    # (d) prop:relabel. The first version scattered six points against
    # a diagonal with a "control" that perturbed whichever edge came
    # first out of the dict --- u0-u1, which lies on no binding cut, so
    # the control landed exactly on the diagonal and demonstrated
    # nothing. The measured control is every edge perturbed in both
    # directions: 14 of 22 such perturbations move at least one key,
    # while the weight-preserving permutation moves none.
    d = ax[3]
    rl = SW["relabel"]
    us_ = rl["vertices"]
    xi = np.arange(len(us_))
    base = np.array(rl["base"], float)
    for c in rl["control"]:
        d.plot(xi, c["seps"], color=BAD, lw=0.8, alpha=0.42, zorder=2)
    d.plot(xi, base, "-o", color=C1, lw=2.4, markersize=6, zorder=5,
           label="as measured")
    d.plot(xi, rl["relabelled"], "--s", color=INK, lw=1.4, markersize=5,
           zorder=6, label="after relabelling")
    d.plot([], [], color=BAD, lw=0.8, alpha=0.6,
           label="control: a weight altered")
    moved = sum(1 for c in rl["control"] if c["moved"])
    d.set_xticks(xi)
    d.set_xticklabels(us_, fontsize=7)
    d.set_xlabel("observation")
    d.set_ylabel("separation")
    d.set_title("Relabelling moves %d keys; reweighting moves %d"
                % (R["relabel"]["keys_moved"], moved))
    d.legend(loc="lower left", fontsize=6.8)
    tag(d, "d")

    save(fig, PAPER, "panel1_probe")


# =====================================================================
# Panel 2 --- the answer carries its own proof, and it is stable
# =====================================================================
def panel2():
    fig, ax = panel(three_d=(2,))

    # (a) thm:floor swept over beta. The floor is the smallest medium
    # weight, so it is a parameter rather than a constant, and the
    # measured minimum separation tracks it: equal to beta wherever the
    # medium binds, above it where a contact edge binds first. Nothing
    # anywhere falls below.
    a = ax[0]
    fl = SW["floor"]
    bs = [r["beta"] for r in fl]
    mins = [r["min_sep"] for r in fl]
    a.plot(bs, bs, color=MUTED, lw=1.2, ls="--", zorder=2,
           label=r"the floor $\beta$")
    for r in fl:
        q = np.array(r["seps"])
        a.plot([r["beta"]] * 2, [np.percentile(q, 5), np.percentile(q, 95)],
               color=C1, lw=1.4, alpha=0.5, zorder=3)
    a.plot(bs, mins, "-o", color=C1, markersize=6, zorder=5,
           label="minimum separation measured")
    a.scatter([R["floor"]["beta"]], [R["floor"]["min_sep"]], s=120,
              facecolor="white", edgecolor=BAD, linewidth=1.8, zorder=7)
    a.annotate("recorded: %d of %d below"
               % (R["floor"]["below_floor"], R["floor"]["n_queries"]),
               xy=(R["floor"]["beta"], R["floor"]["min_sep"]),
               xytext=(0.34, 0.20), textcoords="axes fraction",
               fontsize=7, color=BAD,
               arrowprops=dict(arrowstyle="->", color=BAD, lw=0.9,
                               shrinkB=9))
    a.set_xlabel(r"minimum medium weight  $\beta$")
    a.set_ylabel("separation")
    a.set_title("%d queries, %d below the floor"
                % (sum(r["queries"] for r in fl),
                   sum(r["below_floor"] for r in fl)))
    a.legend(loc="upper left")
    tag(a, "a")

    # (b) thm:verify as a REJECTION EDGE rather than a tally. Two
    # earlier versions of this chart were tables drawn as bars: four
    # zero-height bars whose only content was the "0 of N" text beside
    # them. The measured quantity is where the check actually turns
    # over --- gamma perturbed by a relative magnitude, swept across
    # eleven decades over 8 graphs and 256 certificates per decade.
    # The edge lands exactly at the 1e-9 tolerance verify declares:
    # everything below it accepted, everything above rejected, with the
    # partial band at 1e-9 itself, where the ABSOLUTE error depends on
    # each certificate's own gamma.
    b = ax[1]
    fm = SW["forge_magnitude"]
    rw = [r for r in fm["rows"] if r["rel_magnitude"] > 0]
    mg = np.array([r["rel_magnitude"] for r in rw], float)
    ar = np.array([r["accepted"] / float(r["attempted"]) for r in rw])
    zero = fm["rows"][0]
    b.semilogx(mg, ar, "-o", color=C1, markersize=5, zorder=4,
               label=r"corrupted $\gamma$ accepted")
    b.fill_between(mg, 0, ar, color=C1, alpha=0.10)
    b.axvline(fm["tolerance"], color=BAD, lw=1.3, ls="--", zorder=3)
    b.axhline(zero["accepted"] / float(zero["attempted"]), color=GOOD,
              lw=1.2, ls=":", zorder=2,
              label=r"honest $\gamma$ accepted")
    b.text(fm["tolerance"] * 1.5, 0.62,
           "verify's declared" + chr(10) + "tolerance  $10^{-9}$",
           fontsize=6.8, color=BAD)
    b.set_xlabel(r"relative corruption of $\gamma$")
    b.set_ylabel("share accepted by verify")
    b.set_ylim(-0.06, 1.14)
    b.set_title("The check turns over at its own tolerance")
    b.legend(loc="upper right", fontsize=6.8)
    b.text(0.03, 0.09, "%d certificates per decade, %d graphs"
           % (rw[0]["attempted"], fm["graphs"]), transform=b.transAxes,
           fontsize=6.8, color=INK2)
    tag(b, "b")

    # (c) 3-D: thm:sensitivity(b) as exp4 states it. The guard is
    # eps*(m+M) < slack, m and M measured from the graph with
    # crossing_count over admissible_sets. Height is the share of trials
    # in which the separating SET moved; the guarded surface is the zero
    # floor and the unguarded one rises with eps.
    c = ax[2]
    sn = SW["sens"]
    es = np.array([r["eps"] for r in sn])
    gtot = np.array([r["guarded"] for r in sn], float)
    gv = np.array([r["guarded_violations"] for r in sn], float)
    utot = np.array([r["unguarded"] for r in sn], float)
    uc = np.array([r["unguarded_changes"] for r in sn], float)
    grate = np.where(gtot > 0, gv / np.maximum(gtot, 1), np.nan)
    urate = np.where(utot > 0, uc / np.maximum(utot, 1), np.nan)
    yy = np.linspace(0, 1, 6)
    EE, YY = np.meshgrid(es, yy, indexing="ij")
    Zu = np.repeat(urate[:, None], len(yy), axis=1)
    Zg = np.repeat(grate[:, None], len(yy), axis=1)
    c.view_init(elev=24, azim=-58)
    c.plot_surface(EE, YY, Zu, cmap="Reds", linewidth=0, alpha=0.80,
                   rstride=1, cstride=1)
    # The guarded plane is drawn ONLY where the guard actually covers
    # trials. Past eps ~ 0.1 the guard covers nothing on this family, and
    # a green plane out there would assert "0 violations" in a region
    # where the theorem makes no promise at all.
    cov = gtot > 0
    Zgm = np.where(np.repeat(cov[:, None], len(yy), axis=1),
                   np.nan_to_num(Zg), np.nan)
    c.plot_surface(EE, YY, Zgm, color=GOOD, alpha=0.65,
                   linewidth=0, shade=False)
    c.plot(es[cov], np.full(int(cov.sum()), yy[-1]),
           np.zeros(int(cov.sum())), color=GOOD, lw=2.6, zorder=12)
    c.set_xlabel(r"perturbation  $\epsilon$", labelpad=-2)
    c.set_ylabel("", labelpad=-2)
    c.set_yticks([])
    c.set_zlabel("")
    c.set_title("Where the guard applies, it holds", y=1.04)
    c.text2D(0.00, 0.82, "green: guarded (ends where" + chr(10)
             + "the guard covers nothing)" + chr(10)
             + "red: unguarded" + chr(10) + "height: share whose $S^*$ moved",
             transform=c.transAxes, fontsize=6.2, color=INK2)
    c.set_box_aspect(None, zoom=1.15)
    tag(c, "c", three_d=True)

    # (d) the same guard read as the trade it is: how many trials the
    # guard covers, against how many of them it was wrong about. The
    # guard shrinks as eps grows --- it stops making promises rather
    # than making false ones --- and the violation series is flat at
    # zero all the way to the point where it covers nothing.
    d = ax[3]
    d.plot(es, gtot, "-s", color=C1, markersize=5,
           label="trials the guard covers")
    d.plot(es, gv, "-o", color=BAD, markersize=5, zorder=5,
           label="guarded trials it got wrong")
    d.plot(es, uc, ":", color=MUTED, lw=2.0,
           label="unguarded sets that moved")
    d.set_xscale("log")
    d.set_xlabel(r"perturbation  $\epsilon$")
    d.set_ylabel("trials")
    d.set_ylim(-max(gtot) * 0.05, max(gtot) * 1.28)
    d.set_title("The guard narrows; it never misleads")
    d.legend(loc="upper right", fontsize=6.8)
    d.text(0.03, 0.52, "recorded: %d guarded, %d violations"
           % (R["sensitivity_b"]["guarded_trials"],
              R["sensitivity_b"]["guarded_violations"]),
           transform=d.transAxes, fontsize=7, color=INK2)
    tag(d, "d")

    save(fig, PAPER, "panel2_certificate")


if __name__ == "__main__":
    panel1()
    panel2()
