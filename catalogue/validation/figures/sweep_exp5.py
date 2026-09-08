"""Sweeps for the sink-detection panels, built from exp5's OWN definitions.

exp5's records are scalars and booleans at fixed cells: 0 band breaches
over 6 items, 29 lower-bound breaches of 237, 0 threshold fires of 4779,
a min slack of -4.44e-16. Plotting those alone would give four charts of
a single number each. Every sweep below evaluates exp5's own functions
over the parameters exp5 fixed, and is checked against the recorded
scalar before it is written.

Written to results/exp5_sweeps.json.
"""
from __future__ import annotations

import itertools
import json
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..")))

from exp5_sink_detection import (                    # noqa: E402
    MED, all_separations, crossing_edges, cut, delete_vertex, floor_of,
    make_graph, separation, sink_graph, spread, wspread, forced_threshold)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))
LAM, CW, MW = 2.0, 0.30, 0.50

OUT = {}


# =====================================================================
# 1. thm:collapse: the two-sided band, swept over lambda and over n
# =====================================================================
def collapse_band():
    lams = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
    rows = []
    for lam in lams:
        G = sink_graph(6, lam, CW, MW)
        s = all_separations(G)
        beta = floor_of(G)
        items = [u for u in G["U"] if u != "z"]
        lo, hi, sep, dep = [], [], [], []
        for v in items:
            sv, dv, S = s[v]
            upper = (G["w"][frozenset((v, MED))]
                     + G["w"][frozenset((v, "z"))]
                     + sum(val for e, val in G["w"].items()
                           if v in e and MED not in e and "z" not in e))
            lo.append(beta)
            hi.append(upper)
            sep.append(sv)
            dep.append(dv)
        rows.append({"lam": lam, "beta": beta, "sep": sep, "upper": hi,
                     "dep": dep,
                     "breaches": sum(1 for i in range(len(sep))
                                     if sep[i] < lo[i] - 1e-9
                                     or sep[i] > hi[i] + 1e-9)})
    # the band's relative width over n, at exp5's own lambda
    over_n = []
    for n in range(3, 9):
        G = sink_graph(n, LAM, CW, MW)
        s = all_separations(G)
        beta = floor_of(G)
        items = [u for u in G["U"] if u != "z"]
        for v in items:
            sv, _, _ = s[v]
            upper = (G["w"][frozenset((v, MED))]
                     + G["w"][frozenset((v, "z"))]
                     + sum(val for e, val in G["w"].items()
                           if v in e and MED not in e and "z" not in e))
            over_n.append({"n": n, "v": v, "sep": sv, "beta": beta,
                           "upper": upper})
    # the 3-D surface: mean separation over (n, lambda)
    grid_lams = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
    grid_ns = list(range(3, 9))
    surf = []
    for n in grid_ns:
        row_sep, row_up = [], []
        for lam in grid_lams:
            G = sink_graph(n, lam, CW, MW)
            s = all_separations(G)
            items = [u for u in G["U"] if u != "z"]
            row_sep.append(sum(s[v][0] for v in items) / len(items))
            row_up.append(sum(
                G["w"][frozenset((v, MED))] + G["w"][frozenset((v, "z"))]
                + sum(val for e, val in G["w"].items()
                      if v in e and MED not in e and "z" not in e)
                for v in items) / len(items))
        row_ws = []
        for lam in grid_lams:
            G = sink_graph(n, lam, CW, MW)
            row_ws.append(wspread(G, "z", all_separations(G)))
        surf.append({"n": n, "sep": row_sep, "upper": row_up,
                     "wspread_z": row_ws,
                     "beta": floor_of(sink_graph(n, 1.0, CW, MW))})
    # thm:collapse's UPPER bound is not merely respected on this family
    # --- it is ATTAINED, in every cell, to full floating precision. The
    # minimiser for an ordinary item is always its own incident star, so
    # the cut equals the sum the theorem writes down. Recorded as a
    # finding rather than left implicit in a flat ratio.
    exact = sum(1 for r in surf for a, b_ in zip(r["sep"], r["upper"])
                if abs(a - b_) < 1e-12)
    OUT["collapse"] = {"lams": lams, "rows": rows, "over_n": over_n,
                       "ceiling_exact_cells": exact,
                       "ceiling_total_cells": len(surf) * len(grid_lams),
                       "grid_lams": grid_lams, "grid_ns": grid_ns,
                       "surface": surf}


# =====================================================================
# 2. thm:degree-fails / rem:spread-vs-degree, swept over graph size
# =====================================================================
def degree_vs_spread():
    EPS = 0.10
    rows = []
    for n in range(4, 13):
        h_items = ["u%d" % i for i in range(n)] + ["h"]
        h_edges = {("u%d" % i, "h"): EPS / n for i in range(n)}
        for i in range(n - 1):
            h_edges[("u%d" % i, "u%d" % (i + 1))] = 0.30
        Gh = make_graph(h_items, h_edges, 1.0)
        sh = all_separations(Gh)
        sh0 = all_separations(delete_vertex(Gh, "h"))
        shift_h = max(abs(sh[v][0] - sh0[v][0]) for v in sh0)
        deg_h = sum(1 for e in Gh["w"] if "h" in e and MED not in e)
        ws_h = wspread(Gh, "h", sh)
        sp_h = spread(Gh, "h", sh)

        z_items = ["v%d" % i for i in range(n)] + ["z"]
        z_edges = {("v%d" % i, "z"): 5.0 for i in range(3)}
        Gz = make_graph(z_items, z_edges, 1.0)
        sz = all_separations(Gz)
        sz0 = all_separations(delete_vertex(Gz, "z"))
        ratios = [sz["v%d" % i][0] / sz0["v%d" % i][0] for i in range(3)]
        deg_z = sum(1 for e in Gz["w"] if "z" in e and MED not in e)
        ws_z = wspread(Gz, "z", sz)
        sp_z = spread(Gz, "z", sz)

        rows.append({"n": n,
                     "deg_h": deg_h, "shift_h": shift_h,
                     "ws_h": ws_h, "sp_h": sp_h,
                     "deg_z": deg_z, "min_ratio": min(ratios),
                     "ws_z": ws_z, "sp_z": sp_z})
    OUT["degree"] = {"eps": EPS, "rows": rows}


# =====================================================================
# 3. thm:spread-sound: upper holds, lower breaches --- swept
# =====================================================================
def spread_sound():
    """exp5's own random family, over a range of trial budgets so the
    breach RATE is visible rather than a single count.  The recorded cell
    is 60 trials with rng seeded at 20260830 --- reproduced exactly by
    consuming the same stream from the same point is not possible here
    (exp5 draws from a shared rng), so the family is regenerated with its
    own seed and the recorded RATE is marked as the point the sweep must
    bracket."""
    pts = []
    per_n = {}
    rng = random.Random(4242)
    for trial in range(600):
        n = rng.randint(3, 5)
        items = ["a%d" % i for i in range(n)] + ["t"]
        edges = {}
        for i in range(n):
            if rng.random() < 0.8:
                edges[("a%d" % i, "t")] = rng.uniform(0.05, 2.0)
        for i in range(n - 1):
            if rng.random() < 0.6:
                edges[("a%d" % i, "a%d" % (i + 1))] = rng.uniform(0.1, 1.0)
        med = {u: rng.uniform(0.4, 1.2) for u in items}
        Gt = make_graph(items, edges, med)
        st = all_separations(Gt)
        Gt0 = delete_vertex(Gt, "t")
        st0 = all_separations(Gt0)
        for v in Gt0["U"]:
            sep_v, _, S = st[v]
            tw = sum(val for e, val in crossing_edges(Gt, S) if "t" in e)
            new = st0[v][0]
            pred = sep_v - tw
            pts.append({"n": n, "pred": pred, "new": new, "sep": sep_v,
                        "tau": tw / sep_v,
                        # S is computed on Gt and may contain the sink
                        # "t"; st0[v][2] is computed on Gt0 and cannot.
                        # Comparing them unstripped reports a move
                        # whenever "t" merely sat in the separating set,
                        # which inflates the moved population by 94 of
                        # 424 --- and every one of those 94 is
                        # moved-but-sound, a population exp5's register
                        # measures as empty.
                        "moved": sorted(x for x in S if x != "t")
                                 != sorted(st0[v][2])})
            d = per_n.setdefault(n, {"cases": 0, "upper": 0, "lower": 0,
                                     "moved": 0})
            d["cases"] += 1
            d["upper"] += int(new > pred + 1e-9)
            d["lower"] += int(new < pred - 1e-9)
            d["moved"] += int(sorted(x for x in S if x != "t")
                              != sorted(st0[v][2]))
    OUT["spread_sound"] = {
        "points": pts,
        "by_n": [dict(n=k, **v) for k, v in sorted(per_n.items())],
        "cases": len(pts),
        "upper_breaches": sum(1 for p in pts if p["new"] > p["pred"] + 1e-9),
        "lower_breaches": sum(1 for p in pts if p["new"] < p["pred"] - 1e-9)}


# =====================================================================
# 4. thm:threshold: the antecedent's margin, and the mechanism's slack
# =====================================================================
def threshold():
    rng = random.Random(7)
    margins = []
    by_n = {}
    for _ in range(1200):
        n = rng.randint(3, 5)
        items = ["u%d" % i for i in range(n)]
        edges = {}
        for a, b in itertools.combinations(items, 2):
            if rng.random() < 0.55:
                edges[(a, b)] = round(rng.uniform(0.05, 4.0), 3)
        med = {u: round(rng.uniform(0.1, 3.0), 3) for u in items}
        G = make_graph(items, edges, med)
        s = all_separations(G)
        thr = forced_threshold(G, s)
        for z in items:
            m = wspread(G, z, s) - thr
            margins.append({"n": n, "wspread": wspread(G, z, s),
                            "thr": thr, "margin": m})
            d = by_n.setdefault(n, {"tested": 0, "fires": 0,
                                    "best": -9e9})
            d["tested"] += 1
            d["fires"] += int(m > 1e-9)
            d["best"] = max(d["best"], m)

    # the floor-supplying constructions, swept over lambda: these are the
    # graphs where z demonstrably supplies the floor, i.e. exactly the
    # situation thm:threshold is about. They land ON the cutoff.
    floor_rows = []
    for n in (4, 5, 6):
        for lam in (0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5):
            items = ["u%d" % i for i in range(n)] + ["z"]
            edges = {("u%d" % i, "z"): lam for i in range(n)}
            med = {u: 1.0 for u in items}
            med["z"] = 1.0 * (n + 2)
            G = make_graph(items, edges, med)
            s = all_separations(G)
            thr = forced_threshold(G, s)
            ws = wspread(G, "z", s)
            floor_rows.append({"n": n, "lam": lam, "thr": thr,
                               "wspread": ws, "margin": ws - thr})

    # the mechanism: slack = (sep(v) - beta) - z_weight, which is why the
    # antecedent cannot be met. Swept the same way exp5 swept it.
    rngU = random.Random(11)
    slacks = []
    for _ in range(800):
        n = rngU.randint(3, 5)
        items = ["u%d" % i for i in range(n)]
        edges = {}
        for a, b in itertools.combinations(items, 2):
            if rngU.random() < 0.6:
                edges[(a, b)] = round(rngU.uniform(0.05, 4.0), 3)
        med = {u: round(rngU.uniform(0.1, 3.0), 3) for u in items}
        G = make_graph(items, edges, med)
        s = all_separations(G)
        b = floor_of(G)
        for z in items:
            for v in G["U"]:
                if v == z:
                    continue
                sv, _, S = s[v]
                zw = sum(val for e, val in crossing_edges(G, S) if z in e)
                slacks.append({"n": n, "slack": (sv - b) - zw,
                               "sep": sv, "beta": b, "zw": zw})
    OUT["threshold"] = {
        "margins": margins,
        "by_n": [dict(n=k, **v) for k, v in sorted(by_n.items())],
        "tested": len(margins),
        "fires": sum(1 for m in margins if m["margin"] > 1e-9),
        "best_margin": max(m["margin"] for m in margins),
        "floor_rows": floor_rows,
        "slacks": slacks,
        "min_slack": min(s_["slack"] for s_ in slacks),
        "negative_slacks": sum(1 for s_ in slacks if s_["slack"] < -1e-9)}


if __name__ == "__main__":
    collapse_band()
    print("collapse done")
    degree_vs_spread()
    print("degree done")
    spread_sound()
    print("spread_sound done")
    threshold()
    print("threshold done")

    with open(os.path.join(RESULTS, "exp5_sink_detection.json"),
              encoding="utf8") as fh:
        R = json.load(fh)["records"]

    # ---- the sweeps must reproduce the artefact, or they are measuring
    # a different object than the experiment ran.
    j2 = [r for r in OUT["collapse"]["rows"] if r["lam"] == LAM][0]
    assert j2["beta"] == R["collapse"]["beta"], (j2["beta"],)
    assert sorted(j2["sep"]) == sorted(
        v for k, v in R["collapse"]["seps"].items() if k != "z")
    assert sum(r["breaches"] for r in OUT["collapse"]["rows"]) == 0
    # the ceiling is attained everywhere, not merely respected
    assert (OUT["collapse"]["ceiling_exact_cells"]
            == OUT["collapse"]["ceiling_total_cells"]), (
        OUT["collapse"]["ceiling_exact_cells"],
        OUT["collapse"]["ceiling_total_cells"])

    d9 = [r for r in OUT["degree"]["rows"] if r["n"] == 9][0]
    assert d9["deg_h"] == R["degree_fails"]["harmless"]["degree"]
    assert d9["deg_z"] == R["degree_fails"]["fatal"]["degree"]
    assert abs(d9["shift_h"]
               - R["degree_fails"]["harmless"]["max_separation_shift"]) < 1e-12
    assert abs(d9["ws_h"]
               - R["spread_vs_degree"]["harmless"]["wspread"]) < 1e-12
    assert abs(d9["ws_z"]
               - R["spread_vs_degree"]["fatal"]["wspread"]) < 1e-12

    assert OUT["spread_sound"]["upper_breaches"] == 0
    assert OUT["spread_sound"]["lower_breaches"] > 0
    # exp5's register measures the stable/moved split as EXACT: every
    # moved pair breaches and every stable pair is sound, so the
    # moved-but-sound population is empty.  The sweep is a different
    # random family and must reproduce that structure, not merely a
    # correlation.  This assertion is what caught the "t"-stripping bug
    # above --- it failed at 94 moved-but-sound points.
    _ms = [q for q in OUT["spread_sound"]["points"]
           if q["moved"] and not (q["new"] < q["pred"] - 1e-9)]
    assert not _ms, "moved but sound: %d" % len(_ms)

    assert OUT["threshold"]["fires"] == 0
    assert OUT["threshold"]["negative_slacks"] == 0
    assert OUT["threshold"]["min_slack"] >= -1e-9
    # exp5's three floor-supplying constructions are (n, lambda) =
    # (4, 0.05), (5, 0.02), (6, 0.01). Their recorded margins are the
    # cells this sweep must pass through. (An earlier version of this
    # assertion demanded margin == 0, reading exp5's "land EXACTLY on
    # the cutoff" as a claim about the margin; it is a claim about
    # strict_fires --- the margins themselves are recorded as -0.791,
    # -0.840, -0.866 and the sweep reproduces all three.)
    _rec = {(c["n"], c["lambda"]): c["margin"]
            for c in R["threshold_satisfiability"][
                "floor_supplying_constructions"]}
    _hit = 0
    for r in OUT["threshold"]["floor_rows"]:
        k = (r["n"], r["lam"])
        if k in _rec:
            assert abs(r["margin"] - _rec[k]) < 1e-9, (k, r["margin"])
            _hit += 1
    assert _hit == len(_rec), (_hit, len(_rec))
    assert all(r["margin"] < 0 for r in OUT["threshold"]["floor_rows"])

    path = os.path.join(RESULTS, "exp5_sweeps.json")
    with open(path, "w", encoding="utf8") as fh:
        json.dump(OUT, fh)
    print("wrote", path)
    print("  collapse rows      ", len(OUT["collapse"]["rows"]))
    print("  degree rows        ", len(OUT["degree"]["rows"]))
    print("  spread_sound cases %d, upper %d, lower %d"
          % (OUT["spread_sound"]["cases"],
             OUT["spread_sound"]["upper_breaches"],
             OUT["spread_sound"]["lower_breaches"]))
    print("  threshold tested %d, fires %d, best margin %+.3e"
          % (OUT["threshold"]["tested"], OUT["threshold"]["fires"],
             OUT["threshold"]["best_margin"]))
    print("  min slack %+.3e over %d triples"
          % (OUT["threshold"]["min_slack"], len(OUT["threshold"]["slacks"])))
