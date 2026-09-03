"""Sweeps for the peptide-mass-invariance panels, from exp6's OWN code.

exp6's records are scalars at fixed cells: 0 key mismatches over 40
relabellings of one 8-compound subgraph, 17 key-sharing pairs of 5600,
44 mass-ambiguous compounds at 100 ppm, 2 plural closures of 210, a
recovery fraction per k. Drawing those alone gives four charts of one
number each.

Every sweep below imports exp6's own definitions and evaluates them over
the parameters exp6 fixed, and each is checked against the recorded
scalar before it is written. The NIST library is loaded once and shared.

Written to results/exp6_sweeps.json.
"""
from __future__ import annotations

import itertools
import json
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..")))

import nist_msp                                        # noqa: E402
from exp6_peptide_mass_invariance import (             # noqa: E402
    MED, all_separations, crossing_edges, cut, cut_key, induced)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))

OUT = {}

print("loading the library ...")
SPECS = nist_msp.usable(nist_msp.read_msp())
GROUPS = nist_msp.by_compound(SPECS)
SIG = {k: nist_msp.compound_signature(v) for k, v in GROUPS.items()}
MZ = {k: min(s["precursor_mz"] for s in v) for k, v in GROUPS.items()}
NAMES = sorted(SIG)
print("  %d spectra, %d compounds" % (len(SPECS), len(GROUPS)))


# =====================================================================
# 1. thm:invariance --- relabelling invariance over subgraph size
# =====================================================================
def invariance():
    """exp6 relabels ONE 8-compound subgraph 40 times. The claim is
    about every graph, so the same measurement is repeated over a range
    of sizes, and against the control exp6 uses: a genuine weight change
    must move a key, or invariance is vacuous."""
    rng = random.Random(90210)
    rows = []
    for n in range(4, 10):
        mism, trials, ctrl_moved, graphs = 0, 0, 0, 0
        keyspace = []
        for _ in range(6):
            sub = rng.sample(NAMES, n)
            G = induced(SIG, sub)
            seps = all_separations(G)
            keys = {v: cut_key(G, v, seps) for v in sub}
            keyspace.append(len(set(keys.values())))
            graphs += 1
            for _ in range(8):
                perm = list(sub)
                rng.shuffle(perm)
                ren = dict(zip(sub, perm))
                E2 = {}
                for edge, w in G["E"].items():
                    a, b = tuple(edge)
                    a2 = MED if a == MED else ren[a]
                    b2 = MED if b == MED else ren[b]
                    E2[frozenset((a2, b2))] = w
                G2 = {"U": [ren[v] for v in sub], "E": E2}
                s2 = all_separations(G2)
                trials += 1
                for v in sub:
                    if cut_key(G2, ren[v], s2) != keys[v]:
                        mism += 1
            # exp6's own control, at every size
            Ec = dict(G["E"])
            t = sub[0]
            Ec[frozenset((t, MED))] = Ec[frozenset((t, MED))] * 0.25
            Gc = {"U": list(G["U"]), "E": Ec}
            sc = all_separations(Gc)
            if any(cut_key(Gc, v, sc) != keys[v] for v in sub):
                ctrl_moved += 1
        rows.append({"n": n, "graphs": graphs, "relabellings": trials,
                     "mismatches": mism, "control_moved": ctrl_moved,
                     "mean_distinct_keys": sum(keyspace) / len(keyspace)})
    OUT["invariance"] = {"rows": rows}


# =====================================================================
# 2. thm:region --- the weight-to-cut map, over subgraph size
# =====================================================================
def region():
    """A scalar stands in for the region only if distinct minimum cuts
    carry distinct weights. exp6 measures 7 weights carrying several
    cuts over 420 cuts at one size; swept over size, this is the share
    of weights that are not injective."""
    rng = random.Random(555)
    rows = []
    for n in range(4, 10):
        checked, dup, weights = 0, 0, 0
        for _ in range(80):
            sub = rng.sample(NAMES, n)
            G = induced(SIG, sub)
            s = all_separations(G)
            byw = {}
            for v in sub:
                w = round(s[v][0], 9)
                eset = frozenset(crossing_edges(G, s[v][2]))
                byw.setdefault(w, set()).add(eset)
                checked += 1
            weights += len(byw)
            dup += sum(1 for w, sets in byw.items() if len(sets) > 1)
        rows.append({"n": n, "cuts": checked, "weights": weights,
                     "ambiguous_weights": dup,
                     "share": dup / float(weights)})
    OUT["region"] = {"rows": rows}


# =====================================================================
# 3. cor:mass --- mass against key, and the library's own resolution
# =====================================================================
def mass_vs_key():
    """Two measurements the corollary distinguishes and the paper must
    not conflate:

      (a) pairs sharing a CUT KEY while differing in MASS. Sharpening
          the mass cannot create key agreement, so such a pair survives
          every refinement --- this is cor:mass's actual content, and
          the ppm gaps are recorded so the survival is visible.

      (b) how well mass alone resolves THIS library. It resolves it
          completely at 10 ppm, which is the honest converse exp6
          records as T3b and which is NOT a refutation of (a)."""
    rng = random.Random(31337)
    pairs, deg = 0, []
    per_n = []
    for n in (6, 7, 8, 9):
        p_, d_ = 0, 0
        for _ in range(60):
            sub = rng.sample(NAMES, n)
            G = induced(SIG, sub)
            s = all_separations(G)
            k = {v: cut_key(G, v, s) for v in sub}
            for a, b in itertools.combinations(sub, 2):
                p_ += 1
                pairs += 1
                if k[a] == k[b]:
                    d_ += 1
                    deg.append(round(abs(MZ[a] - MZ[b]) / MZ[a] * 1e6, 3))
        per_n.append({"n": n, "pairs": p_, "sharing_key": d_,
                      "rate": d_ / float(p_)})
    # (b) the library's own mass resolution, swept finely
    ms = sorted({round(MZ[v], 6) for v in NAMES})
    tol = []
    t = 1000.0
    while t >= 0.5:
        amb = 0
        for m in ms:
            w = m * t * 1e-6
            if sum(1 for m2 in ms if abs(m2 - m) <= w) > 1:
                amb += 1
        tol.append({"tol_ppm": t, "ambiguous": amb,
                    "fraction": amb / float(len(ms))})
        t /= 1.4
    OUT["mass"] = {"per_n": per_n, "pairs": pairs,
                   "key_share_ppm_gaps": sorted(deg),
                   "masses": len(ms), "tolerance": tol}


# =====================================================================
# 4. thm:no-selector --- the medium as the only source of separation
# =====================================================================
def no_selector():
    """exp6 records one cell: 25 of 25 subgraphs positive with the
    medium, 25 of 25 admitting a zero-cost separation without it. The
    claim is that the medium is what makes separation cost anything, so
    the sweep varies the medium's weight and the contact scale and
    measures the separation in each regime.

    The contact scale matters: build_nist_graph normalises so the
    heaviest contact is max_contact x medium_w, which ax:noncomplete
    forces. Above 1.0 the graph enters the degenerate regime exp6
    records, where S* = U for well-connected items."""
    rng = random.Random(2718)
    subs = [rng.sample(NAMES, 6) for _ in range(12)]
    rows = []
    for mw in (0.25, 0.5, 1.0, 2.0, 4.0):
        pos, zero, mins, means = 0, 0, [], []
        for sub in subs:
            G = induced(SIG, sub, medium_w=mw)
            s = all_separations(G)
            if all(s[v][0] > 0 for v in sub):
                pos += 1
            Ei = {k: w for k, w in G["E"].items() if MED not in k}
            if abs(cut({"U": sub, "E": Ei}, set(sub))) < 1e-12:
                zero += 1
            mins.append(min(s[v][0] for v in sub))
            means.append(sum(s[v][0] for v in sub) / len(sub))
        rows.append({"medium_w": mw, "graphs": len(subs),
                     "positive_with_medium": pos,
                     "zero_without_medium": zero,
                     "min_sep": min(mins), "beta": mw,
                     "mean_sep": sum(means) / len(means)})
    # the contact-scale regimes: delta = |S*| as the contacts approach
    # and pass the medium
    scale = []
    for mc in (0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
        ds, sp = [], []
        for sub in subs[:8]:
            G = induced(SIG, sub, medium_w=1.0, max_contact=mc)
            s = all_separations(G)
            ds.append(sum(s[v][1] for v in sub) / len(sub))
            sp.append(sum(s[v][0] for v in sub) / len(sub))
        scale.append({"max_contact": mc,
                      "mean_delta": sum(ds) / len(ds),
                      "mean_sep": sum(sp) / len(sp),
                      "n": 6})
    # The regime transition over (contact scale, subgraph size). An
    # earlier panel drew this over (contact scale, medium weight) by
    # TILING one row across the weight axis, on the reasoning that delta
    # is scale-invariant in the medium because the contact scale is
    # defined relative to it. That reasoning is correct but it is not a
    # measurement, and a tiled axis carries no data --- so the second
    # axis is the size of the graph, where delta genuinely varies,
    # measured rather than asserted.
    regime_ns = list(range(4, 10))
    regime_mc = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    rng3 = random.Random(271828)
    regime = []
    for n in regime_ns:
        picks = [rng3.sample(NAMES, n) for _ in range(8)]
        row_d, row_s = [], []
        for mc in regime_mc:
            dd, ss = [], []
            for sub in picks:
                G = induced(SIG, sub, medium_w=1.0, max_contact=mc)
                sep = all_separations(G)
                dd.append(sum(sep[v][1] for v in sub) / float(n))
                ss.append(sum(sep[v][0] for v in sub) / float(n))
            row_d.append(sum(dd) / len(dd))
            row_s.append(sum(ss) / len(ss))
        regime.append({"n": n, "mean_delta": row_d, "mean_sep": row_s})

    # the 3-D surface: mean separation over (subgraph size, medium
    # weight), with the medium present and absent
    grid_ns = list(range(4, 10))
    grid_mw = [0.25, 0.5, 1.0, 2.0, 4.0]
    surf = []
    rng2 = random.Random(1618)
    for n in grid_ns:
        picks = [rng2.sample(NAMES, n) for _ in range(4)]
        row_w, row_o = [], []
        for mw in grid_mw:
            ww, oo = [], []
            for sub in picks:
                G = induced(SIG, sub, medium_w=mw)
                s = all_separations(G)
                ww.append(sum(s[v][0] for v in sub) / len(sub))
                Ei = {k: w for k, w in G["E"].items() if MED not in k}
                oo.append(cut({"U": sub, "E": Ei}, set(sub)))
            row_w.append(sum(ww) / len(ww))
            row_o.append(sum(oo) / len(oo))
        surf.append({"n": n, "with_medium": row_w, "without": row_o})
    OUT["no_selector"] = {"rows": rows, "scale": scale,
                          "grid_ns": grid_ns, "grid_mw": grid_mw,
                          "surface": surf, "regime_ns": regime_ns,
                          "regime_mc": regime_mc, "regime": regime}


# =====================================================================
# 5. thm:closure-stronger --- closed yet plural, over subgraph size
# =====================================================================
def closure():
    """A determination closes on a REGION. exp6 measures 2 plural of
    210 at n = 7. Plurality is a property of the key partition, so the
    sweep measures the partition itself over size: how many
    determinations close on more than one candidate."""
    rng = random.Random(8080)
    rows = []
    for n in range(4, 10):
        plural, single, runs, sizes = 0, 0, 0, []
        for _ in range(40):
            sub = rng.sample(NAMES, n)
            G = induced(SIG, sub)
            s = all_separations(G)
            k = {v: cut_key(G, v, s) for v in sub}
            for v in sub:
                amb = [u for u in sub if k[u] == k[v]]
                runs += 1
                sizes.append(len(amb))
                if len(amb) > 1:
                    plural += 1
                else:
                    single += 1
        rows.append({"n": n, "determinations": runs, "plural": plural,
                     "singleton": single, "rate": plural / float(runs),
                     "mean_region": sum(sizes) / float(len(sizes)),
                     "max_region": max(sizes)})
    OUT["closure"] = {"rows": rows}


# =====================================================================
# 6. prop:three --- the two readings, over the catalyst count
# =====================================================================
def three_catalysts():
    """exp6 grades prop:three FAILED, and the failure is in the
    STATEMENT rather than the proof. Both readings are swept over k and
    over the number of candidate regions, because the definite article
    in "THE erroneous catalyst" is what separates them:

      literal  the bad catalyst is identified and removed, then the
               survivors' majority is read. One sound survivor suffices,
               so k = 2 already recovers everything.

      unaided  the bad catalyst is not identified. k = 2 is a one-one
               split with no majority, so k = 3 is genuinely the first
               count that works --- which is the proof's own reason.
    """
    def region_of(vs, c):
        if not vs:
            return frozenset(range(c))
        best = max(vs.count(x) for x in set(vs))
        return frozenset(x for x in set(vs) if vs.count(x) == best)

    rows = []
    for cands in (3, 4, 5):
        rng = random.Random(4000 + cands)
        lit, una = [], []
        for k in range(1, 8):
            rl, ru, tot = 0, 0, 4000
            for _ in range(tot):
                truth = rng.randint(0, cands - 1)
                wrong = rng.choice([c for c in range(cands)
                                    if c != truth])
                votes = [truth] * (k - 1) + [wrong]
                rng.shuffle(votes)
                if region_of(votes, cands) == frozenset({truth}):
                    ru += 1
                err = votes.index(wrong)
                surv = votes[:err] + votes[err + 1:]
                if region_of(surv, cands) == frozenset({truth}):
                    rl += 1
            lit.append({"k": k, "fraction": rl / float(tot)})
            una.append({"k": k, "fraction": ru / float(tot)})
        rows.append({"candidates": cands, "literal": lit,
                     "unaided": una})
    OUT["three"] = {"rows": rows}


if __name__ == "__main__":
    invariance()
    print("invariance done")
    region()
    print("region done")
    mass_vs_key()
    print("mass done")
    no_selector()
    print("no_selector done")
    closure()
    print("closure done")
    three_catalysts()
    print("three done")

    with open(os.path.join(RESULTS, "exp6_peptide_mass_invariance.json"),
              encoding="utf8") as fh:
        R = json.load(fh)["records"]

    # ---- the sweeps must reproduce the artefact's verdict, or they are
    # measuring a different object than the experiment ran. exp6 draws
    # from a shared rng, so the subgraphs differ; the verdicts do not.

    # T1: invariance holds at every size, and the control is live at
    # every size --- a control that never fires makes the claim vacuous.
    assert all(r["mismatches"] == 0 for r in OUT["invariance"]["rows"])
    assert R["T1_invariance"]["key_mismatches"] == 0
    assert all(r["control_moved"] > 0 for r in OUT["invariance"]["rows"])
    assert R["T1_invariance"]["control_weight_change_moves_key"]

    # T2: the weight-to-cut map is not injective. Ambiguity is RARE ---
    # a fraction of a percent of weights --- so this asserts that it
    # occurs across the sweep and at the size exp6 measured, NOT that it
    # occurs in every sample at every size. An earlier version demanded
    # the latter and failed at n = 6 and 7 on 24 subgraphs, which was
    # the assertion being wrong about sampling, not the sweep.
    assert sum(r["ambiguous_weights"] for r in OUT["region"]["rows"]) > 0
    _n7 = [r for r in OUT["region"]["rows"] if r["n"] == 7][0]
    assert _n7["ambiguous_weights"] > 0, _n7
    assert R["T2_region"]["weights_carrying_several_distinct_cuts"] > 0

    # T3 / T3b: key-sharing pairs exist and are mass-distinct, while the
    # library itself resolves under refinement. Both, or the panel
    # would be asserting the corollary is about library ambiguity.
    assert len(OUT["mass"]["key_share_ppm_gaps"]) > 0
    assert min(OUT["mass"]["key_share_ppm_gaps"]) > 1e-6
    assert R["T3_mass_vs_key"]["pairs_sharing_a_cut_key"] > 0
    _res = [t for t in OUT["mass"]["tolerance"] if t["ambiguous"] == 0]
    assert _res, "library never resolves --- contradicts T3b"
    assert max(t["tol_ppm"] for t in _res) <= 100.0
    assert R["T3b_library_mass_degeneracy"]["resolved_at_ppm"] == 10.0

    # T4: the medium is the whole source of separation, at every medium
    # weight, and deleting it always leaves a zero-cost cut.
    for r in OUT["no_selector"]["rows"]:
        assert r["positive_with_medium"] == r["graphs"], r
        assert r["zero_without_medium"] == r["graphs"], r
    assert (R["T4_no_selector"]["all_positive_with_medium"]
            == R["T4_no_selector"]["graphs"])
    # ax:noncomplete's normalisation is load-bearing: delta must rise as
    # the contacts approach the medium. This is the degenerate regime
    # exp6 records, measured rather than asserted in prose.
    _sc = OUT["no_selector"]["scale"]
    assert _sc[0]["mean_delta"] < _sc[-1]["mean_delta"], _sc
    # the transition must be present at EVERY size, and delta must be
    # exactly 1 (a singleton minimiser) throughout the local regime ---
    # that flat floor is what ax:noncomplete's normalisation buys.
    _mc = OUT["no_selector"]["regime_mc"]
    _lo = [j for j, v in enumerate(_mc) if v <= 1.0]
    for _r in OUT["no_selector"]["regime"]:
        assert all(abs(_r["mean_delta"][j] - 1.0) < 1e-12 for j in _lo), _r
        assert _r["mean_delta"][-1] > 1.0, _r

    # T5: both outcomes occur, or closure and singleton-return were not
    # separated (which is exp6's own non-discriminating condition).
    assert any(r["plural"] > 0 for r in OUT["closure"]["rows"])
    assert all(r["singleton"] > 0 for r in OUT["closure"]["rows"])
    assert R["T5_closed_plural"]["closed_with_plural_region"] > 0

    # T7: the two readings of prop:three separate at k = 2, on every
    # candidate count. That separation IS the defect in the statement.
    for row in OUT["three"]["rows"]:
        lit = {d["k"]: d["fraction"] for d in row["literal"]}
        una = {d["k"]: d["fraction"] for d in row["unaided"]}
        assert lit[2] > 0.99, (row["candidates"], lit)
        assert una[2] == 0.0, (row["candidates"], una)
        assert una[3] > 0.99, (row["candidates"], una)
    assert R["T7a_three_catalysts_literal"]["by_k"]["2"]["fraction"] == 1.0
    assert R["T7b_three_catalysts_unaided"]["by_k"]["2"]["fraction"] == 0.0

    path = os.path.join(RESULTS, "exp6_sweeps.json")
    with open(path, "w", encoding="utf8") as fh:
        json.dump(OUT, fh)
    print("wrote", path)
    for r in OUT["invariance"]["rows"]:
        print("  n=%d  %d relabellings, %d mismatches, control %d/%d"
              % (r["n"], r["relabellings"], r["mismatches"],
                 r["control_moved"], r["graphs"]))
    print("  key-sharing pairs %d of %d"
          % (len(OUT["mass"]["key_share_ppm_gaps"]), OUT["mass"]["pairs"]))
    print("  library resolves at %.1f ppm"
          % max(t["tol_ppm"] for t in OUT["mass"]["tolerance"]
                if t["ambiguous"] == 0))
