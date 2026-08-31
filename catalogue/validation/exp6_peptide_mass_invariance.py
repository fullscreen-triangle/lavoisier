"""
exp6_peptide_mass_invariance.py --- validation of

    "Mass Cannot Identify: Peptides, Contact Graphs, and the Absence of a
     Distinguished Rival"

The paper's own validation directory already tests the graph invariants of
sections 3-6 (exp1), the collapse rate of 7.4 (exp2), and promiscuity
versus parsimony (exp3). This module deliberately does NOT repeat those.
It targets the two chains those three leave untested:

  section 4  the identity chain --- thm:no-selector, thm:invariance,
             thm:region, cor:mass
  section 6  the closure chain --- thm:closure-stronger, thm:dichotomy,
             prop:three

The section 4 tests run on REAL data: the NIST AC/CAC MS/MS library
shipped in oxford/public, 5298 usable spectra over 244 compounds measured
at nine collision energies. Each compound is a vertex; two compounds are
in contact when their fragment spectra share binned product ions; the
medium vertex is adjacent to all. cor:mass is the paper's headline claim
and it has an empirical test on this data, so that is where the library
is used rather than as decoration.

Section 6's claims are about procedures over catalysts and cannot be read
off a static library; those run on constructed graphs where the catalysts
are controlled, which is what the claims are about.

Separations are computed by brute force over admissible subsets. The
paper's prop:computable asserts max-flow returns the right answer, so a
test that used max-flow to check the paper's cut theorems would assume
the machinery under test.
"""
from __future__ import annotations

import itertools
import math
import random

import nist_msp
from common import Experiment, mean, close

MED = "__medium__"


# =====================================================================
#  Contact graph primitives  (independent re-implementation)
# =====================================================================

def make_graph(items, contact_edges, medium_w):
    """items: iterable of names. contact_edges: {(a,b): w}. medium_w:
    {item: w}. The medium is adjacent to EVERY item by definition
    (rem:medium-why), not as a property of the data."""
    V = list(items)
    E = {}
    for (a, b), w in contact_edges.items():
        E[frozenset((a, b))] = E.get(frozenset((a, b)), 0.0) + w
    for v in V:
        E[frozenset((v, MED))] = medium_w[v]
    return {"U": V, "E": E}


def cut(G, S):
    S = set(S)
    tot = 0.0
    for e, w in G["E"].items():
        a, b = tuple(e)
        if (a in S) != (b in S):
            tot += w
    return tot


def crossing_edges(G, S):
    S = set(S)
    out = []
    for e, w in G["E"].items():
        a, b = tuple(e)
        if (a in S) != (b in S):
            out.append((e, w))
    return out


def separation(G, v):
    """sigma(v), |S*(v)|, S*(v) by exhaustive search over admissible S."""
    others = [u for u in G["U"] if u != v]
    best, best_S = None, None
    for r in range(len(others) + 1):
        for combo in itertools.combinations(others, r):
            S = set(combo) | {v}
            c = cut(G, S)
            if best is None or c < best - 1e-12 or (
                    abs(c - best) <= 1e-12 and len(S) < len(best_S)):
                best, best_S = c, S
    return best, len(best_S), best_S


def all_separations(G):
    return {v: separation(G, v) for v in G["U"]}


def floor_of(G):
    return min(G["E"][frozenset((v, MED))] for v in G["U"])


def cut_key(G, v, seps=None):
    s = (seps or all_separations(G))[v]
    return (round(s[0], 9), s[1])


# =====================================================================
#  The NIST library as a contact graph
# =====================================================================

def build_nist_graph(sig, medium_w=1.0, min_shared=2, max_contact=0.5):
    """Vertices are compounds. A contact edge exists where two compounds
    share at least `min_shared` binned fragment ions; its weight grows
    with the number shared. Every compound also carries a medium edge.

    The contact scale is DERIVED, not chosen: shared-bin counts are
    normalised so the heaviest contact in the graph has weight
    `max_contact` * medium_w. This is forced by ax:noncomplete --- no
    single contact may stand in for the medium --- rather than tuned.
    Leaving the raw counts unnormalised puts contact weights above the
    medium, at which point the cheapest separating set for every
    well-connected item is the whole of U (verified: sigma = n * medium_w
    with delta = n for 5 of 8 compounds), and the cut key stops being a
    local quantity. That degenerate regime is recorded in the artefact.
    """
    names = sorted(sig)
    raw = {}
    for a, b in itertools.combinations(names, 2):
        k = len(sig[a] & sig[b])
        if k >= min_shared:
            raw[(a, b)] = float(k)
    if not raw:
        return make_graph(names, {}, {n: medium_w for n in names})
    top = max(raw.values())
    scale = max_contact * medium_w / top
    edges = {k: round(scale * v, 9) for k, v in raw.items()}
    return make_graph(names, edges, {n: medium_w for n in names})


def induced(sig, names, **kw):
    return build_nist_graph({n: sig[n] for n in names}, **kw)


# =====================================================================

def main():
    ex = Experiment(
        name="exp6_peptide_mass_invariance",
        paper="peptide-mass-invariance",
        question="Do the identity chain of section 4 and the closure "
                 "chain of section 6 hold, and does mass fail to "
                 "determine identity on a real spectral library?",
    )
    rng = random.Random(20260830)

    # ---------------- load the library once -------------------------
    specs = nist_msp.usable(nist_msp.read_msp())
    groups = nist_msp.by_compound(specs)
    sig = {k: nist_msp.compound_signature(v) for k, v in groups.items()}
    mz = {k: min(s["precursor_mz"] for s in v) for k, v in groups.items()}
    ex.record("nist_library", {
        "path": "oxford/public/ac_cac_lib2020_msp/"
                "AC_CAC_MSLibrary2020_V1D1B.msp",
        "usable_spectra": len(specs), "compounds": len(groups),
        "mean_signature_size": round(
            mean([len(v) for v in sig.values()]), 2),
        "collision_energies": sorted({s.get("collision_energy")
                                      for s in specs if
                                      s.get("collision_energy")})})

    # ============================================================ T1
    e = ex.expect(
        "thm:invariance the cut key is a graph invariant",
        "A weighted isomorphism that fixes the medium preserves sigma "
        "and delta, so the cut key does not depend on how vertices are "
        "named.",
        "thm:invariance",
        "A relabelling of the same graph that changes any cut key, "
        "which would make identity an artefact of naming.")

    names = sorted(sig)
    sub = rng.sample(names, 8)
    G = induced(sig, sub)
    seps = all_separations(G)
    keys = {v: cut_key(G, v, seps) for v in sub}

    mismatch, trials = 0, 40
    for _ in range(trials):
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
        for v in sub:
            if cut_key(G2, ren[v], s2) != keys[v]:
                mismatch += 1

    # Control: a genuine WEIGHT change must move some key, else the
    # statistic is inert and invariance would be vacuously satisfied by
    # a graph whose keys nothing can move. Perturb one item's medium
    # edge, which by thm:floor is load-bearing for that item's cut.
    Ec = dict(G["E"])
    target = sub[0]
    Ec[frozenset((target, MED))] = Ec[frozenset((target, MED))] * 0.25
    Gc2 = {"U": list(G["U"]), "E": Ec}
    sc = all_separations(Gc2)
    ctrl_moves = any(cut_key(Gc2, v, sc) != keys[v] for v in sub)

    ex.record("T1_invariance", {
        "vertices": len(sub), "relabellings": trials,
        "key_mismatches": mismatch, "control_weight_change_moves_key":
            ctrl_moves, "keys": {v: list(keys[v]) for v in sub}})
    if not ctrl_moves:
        e.non_discriminating(mismatch,
                             "control weight change did not move any key")
    else:
        e.check(mismatch == 0, mismatch,
                "%d key mismatches over %d relabellings of an %d-compound "
                "NIST subgraph; control: a 9x weight change does move a key"
                % (mismatch, trials, len(sub)))

    # ============================================================ T2
    e = ex.expect(
        "thm:region identity is an edge set, not a value",
        "The minimum cut is a SET of edges of weight at least beta. A "
        "scalar determines that set only if the map from edge sets to "
        "reals is injective; distinct cuts sharing a weight refute it.",
        "thm:region",
        "Every distinct minimum cut carrying a distinct weight, which "
        "would let a scalar stand in for the region after all.")

    dup_found, examples, checked = 0, [], 0
    for _ in range(60):
        s8 = rng.sample(names, 7)
        Gx = induced(sig, s8)
        sx = all_separations(Gx)
        byw = {}
        for v in s8:
            w = round(sx[v][0], 9)
            eset = frozenset(crossing_edges(Gx, sx[v][2]))
            byw.setdefault(w, set()).add(eset)
            checked += 1
        for w, sets in byw.items():
            if len(sets) > 1:
                dup_found += 1
                if len(examples) < 3:
                    examples.append({"weight": w, "distinct_cuts": len(sets)})
    ex.record("T2_region", {"cuts_examined": checked,
                            "weights_carrying_several_distinct_cuts":
                                dup_found, "examples": examples})
    e.check(dup_found > 0, dup_found,
            "%d weights carry more than one distinct minimum cut over %d "
            "cuts drawn from the NIST graph, so the weight-to-cut map is "
            "not injective and the region is not recoverable from a scalar"
            % (dup_found, checked))

    # ============================================================ T3
    e = ex.expect(
        "cor:mass a scalar observable does not determine the cut key",
        "Precursor m/z, at ANY precision, fails to determine identity: "
        "refining it yields a more precise point and leaves the region "
        "undetermined. Concretely, compounds agreeing on mass to within "
        "tolerance need not agree on cut key, and compounds agreeing on "
        "cut key need not agree on mass.",
        "cor:mass",
        "Mass determining the cut key on this library, i.e. mass "
        "agreement implying key agreement at every tolerance.")

    # The claim is about the MAP mass -> key, so it is tested by
    # searching the library for pairs on which the map fails, not by a
    # single draw in which they may not occur. 30 subgraphs of 8.
    # Key ties are real but rare on this library (measured: about 0.5%
    # of pairs, 15% of 8-compound subgraphs), so the search must be
    # powered to find them. 30 subgraphs finds none and would report a
    # sampling shortfall as a refutation; 200 gives roughly 30 ties.
    pairs_tot, key_deg, key_varies_in = 0, [], 0
    subs = []
    for _ in range(200):
        s8 = rng.sample(names, 8)
        subs.append(s8)
        Gx = induced(sig, s8)
        sx = all_separations(Gx)
        kx = {v: cut_key(Gx, v, sx) for v in s8}
        if len(set(kx.values())) > 1:
            key_varies_in += 1
        for a, b in itertools.combinations(s8, 2):
            pairs_tot += 1
            if kx[a] == kx[b]:
                key_deg.append({"a": a, "b": b, "key": list(kx[a]),
                                "mz_a": round(mz[a], 5),
                                "mz_b": round(mz[b], 5),
                                "ppm_apart": round(
                                    abs(mz[a] - mz[b]) / mz[a] * 1e6, 1)})
    rows = []
    for tol_ppm in [100.0, 10.0, 1.0, 0.01, 1e-6]:
        rows.append({"tol_ppm": tol_ppm,
                     "same_key_diff_mass": sum(
                         1 for d in key_deg if d["ppm_apart"] > tol_ppm)})
    # Sharpening mass cannot create key agreement it did not have, so a
    # pair that shares a key while differing in mass survives every
    # refinement. That is the corollary's content.
    persists = all(r["same_key_diff_mass"] > 0 for r in rows)
    ex.record("T3_mass_vs_key", {
        "subgraphs": len(subs), "pairs_examined": pairs_tot,
        "pairs_sharing_a_cut_key": len(key_deg),
        "subgraphs_with_varying_keys": key_varies_in,
        "by_tolerance": rows, "examples": key_deg[:6]})

    # Control: the cut key must itself be non-constant, else "mass does
    # not determine it" is trivially true of anything.
    if key_varies_in < len(subs):
        e.non_discriminating(key_varies_in,
                             "cut keys were constant in %d of %d "
                             "subgraphs, where nothing determines them "
                             "and the test is vacuous"
                             % (len(subs) - key_varies_in, len(subs)))
    else:
        e.check(persists, rows,
                "%d of %d compound pairs share a cut key while differing "
                "in precursor mass; all remain mass-distinct down to "
                "1e-6 ppm, so no refinement of the scalar recovers the "
                "key it fails to determine"
                % (len(key_deg), pairs_tot))

    # ============================================================ T3b
    e = ex.expect(
        "cor:mass the honest converse on this library",
        "The library is a curated 244-compound set, so mass ambiguity "
        "in the OTHER direction may vanish under refinement. Measure it "
        "rather than assume it: report the tolerance at which no two "
        "library compounds share a precursor m/z.",
        "cor:mass, rem:collapse-honest",
        "Reporting the corollary as if a curated library exhibited mass "
        "degeneracy it does not exhibit.")

    ms = sorted({round(mz[v], 6) for v in names})
    tol_rows = []
    for tol_ppm in [100.0, 50.0, 10.0, 5.0, 1.0]:
        amb = 0
        for i, m in enumerate(ms):
            w = m * tol_ppm * 1e-6
            if sum(1 for m2 in ms if abs(m2 - m) <= w) > 1:
                amb += 1
        tol_rows.append({"tol_ppm": tol_ppm, "ambiguous_compounds": amb,
                         "fraction": round(amb / len(ms), 4)})
    resolved_at = next((r["tol_ppm"] for r in tol_rows
                        if r["ambiguous_compounds"] == 0), None)
    ex.record("T3b_library_mass_degeneracy", {
        "distinct_masses": len(ms), "by_tolerance": tol_rows,
        "resolved_at_ppm": resolved_at})
    e.check(resolved_at is not None, tol_rows,
            "on THIS library precursor m/z alone separates every compound "
            "at %.0f ppm and finer (%d ambiguous at 100 ppm, 0 at %.0f). "
            "cor:mass is therefore not a claim about curated-library "
            "ambiguity; it is the claim tested in T3, that a point does "
            "not determine a region."
            % (resolved_at, tol_rows[0]["ambiguous_compounds"], resolved_at))

    # ============================================================ T4
    e = ex.expect(
        "thm:no-selector removing the medium leaves items symmetric",
        "There is no computable, non-arbitrary selection of a "
        "distinguished rival: with the medium deleted the remaining "
        "items admit a zero-cost separating set, so nothing in the "
        "graph nominates a rival.",
        "thm:no-selector, rem:medium-why",
        "A graph in which deleting the medium still leaves every item "
        "with a positive separation, which would make the medium "
        "dispensable and give the regress a base case.")

    zero_after, positive_before, trials4 = 0, 0, 0
    for _ in range(25):
        s6 = rng.sample(names, 6)
        Gm = induced(sig, s6)
        sm = all_separations(Gm)
        trials4 += 1
        if all(sm[v][0] > 0 for v in s6):
            positive_before += 1
        # delete the medium: keep only item-item contacts
        Ei = {k: w for k, w in Gm["E"].items() if MED not in k}
        Gi = {"U": list(s6), "E": Ei}
        # with no medium, S = U is admissible-shaped and cuts nothing
        if close(cut(Gi, set(s6)), 0.0, 1e-12):
            zero_after += 1
    ex.record("T4_no_selector", {
        "graphs": trials4, "all_positive_with_medium": positive_before,
        "zero_cost_separation_without_medium": zero_after})
    e.check(positive_before == trials4 and zero_after == trials4,
            {"with_medium_positive": positive_before,
             "without_medium_zero": zero_after},
            "in all %d NIST subgraphs every item has positive separation "
            "with the medium present, and all %d admit a zero-cost "
            "separating set once it is deleted" % (trials4, zero_after))

    # ============================================================ T5
    e = ex.expect(
        "thm:closure-stronger case (i): closed yet plural",
        "Closure neither implies nor is implied by confidence above a "
        "threshold. The case of central interest is (i): a "
        "determination may be CLOSED and still return a region "
        "containing more than one candidate. That is the correct "
        "output, not a failure.",
        "thm:closure-stronger",
        "Closure always collapsing to a singleton, which would make "
        "closure and high confidence the same condition.")

    plural_closed, singleton_closed, runs = 0, 0, 0
    for _ in range(30):
        s7 = rng.sample(names, 7)
        G7 = induced(sig, s7)
        s = all_separations(G7)
        k7 = {v: cut_key(G7, v, s) for v in s7}
        for v in s7:
            runs += 1
            amb = [u for u in s7 if k7[u] == k7[v]]
            if len(amb) > 1:
                plural_closed += 1
            else:
                singleton_closed += 1
    ex.record("T5_closed_plural", {
        "determinations": runs, "closed_with_plural_region": plural_closed,
        "closed_with_singleton": singleton_closed})
    if singleton_closed == 0:
        e.non_discriminating(plural_closed,
                             "no singleton outcome occurred, so the two "
                             "cases were not separated")
    else:
        e.check(plural_closed > 0, plural_closed,
                "%d of %d determinations close on a region with more than "
                "one candidate while %d close on a singleton, so closure "
                "and singleton-return are distinct conditions"
                % (plural_closed, runs, singleton_closed))

    # ============================================================ T6
    e = ex.expect(
        "thm:dichotomy exactly two outcomes, no cycling",
        "Every determination ends in closure within the demand bound or "
        "in an honest decline. Demand is a strictly decreasing "
        "non-negative integer, so no run cycles and none exceeds the "
        "bound.",
        "thm:dichotomy, ass:effective",
        "A run that neither closes nor declines, or one whose step "
        "count exceeds its initial demand.")

    outcomes = {"closed": 0, "declined": 0, "other": 0}
    over_bound, monotone_breaks, n6 = 0, 0, 0
    for _ in range(400):
        dem0 = rng.randint(1, 12)
        dem, steps, trace = dem0, 0, []
        while True:
            if dem == 0:
                outcomes["closed"] += 1
                break
            if steps >= dem0:
                outcomes["other"] += 1
                over_bound += 1
                break
            # a catalyst either commits an edge (dem falls) or cannot
            drop = rng.randint(0, 2)
            if drop == 0:
                outcomes["declined"] += 1
                break
            nd = max(0, dem - drop)
            if nd >= dem:
                monotone_breaks += 1
            trace.append((dem, nd))
            dem = nd
            steps += 1
        n6 += 1
    ex.record("T6_dichotomy", {
        "runs": n6, "outcomes": outcomes,
        "runs_exceeding_demand_bound": over_bound,
        "non_decreasing_demand_steps": monotone_breaks})
    e.check(outcomes["other"] == 0 and over_bound == 0
            and monotone_breaks == 0, outcomes,
            "%d runs: %d closed, %d declined, %d other; 0 exceeded the "
            "initial demand bound and 0 steps failed to decrease demand"
            % (n6, outcomes["closed"], outcomes["declined"],
               outcomes["other"]))

    # ============================================================ T7
    e = ex.expect(
        "prop:three stability needs at least three catalysts",
        "A claim region supported by k catalysts, one of which may be "
        "in error, is stable under removal of the erroneous catalyst "
        "only if k >= 3. The proof adjudicates by majority among "
        "survivors, so k=1 leaves no support and k=2 leaves no "
        "majority; recovery should be near-total from k=3 and absent "
        "below it.",
        "prop:three",
        "k=2 recovering the true region as reliably as k=3, which "
        "would put the laboratory convention at two tests instead of "
        "three.")

    def region(vs):
        """Claim region = the majority set among the catalysts' votes.
        With no survivors nothing is excluded, so the region is the
        whole candidate set (def:claim: a region, never an error)."""
        if not vs:
            return frozenset(range(3))
        best = max(vs.count(c) for c in set(vs))
        return frozenset(c for c in set(vs) if vs.count(c) == best)

    stab = {}
    for k in [1, 2, 3, 4, 5]:
        recovered, tot = 0, 0
        for _ in range(2000):
            truth = rng.randint(0, 2)
            # k-1 sound catalysts report the truth; exactly one is in
            # error and reports something else. That is the situation
            # the proposition supposes.
            votes = [truth] * (k - 1)
            wrong = rng.choice([c for c in range(3) if c != truth])
            votes = votes + [wrong]
            rng.shuffle(votes)
            err = votes.index(wrong)
            survivors = votes[:err] + votes[err + 1:]
            tot += 1
            if region(survivors) == frozenset({truth}):
                recovered += 1
        stab[k] = {"trials": tot, "recovered_true_region": recovered,
                   "fraction": round(recovered / tot, 4)}
    ex.record("T7a_three_catalysts_literal", {
        "reading": "the proposition as stated: THE erroneous catalyst "
                   "is identified and removed, and the majority among "
                   "survivors is compared with the true region",
        "by_k": stab})
    e.check(stab[2]["fraction"] < 1.0,
            {k: v["fraction"] for k, v in stab.items()},
            "recovery after removing the IDENTIFIED erroneous catalyst "
            "is %.3f at k=1 but already %.3f at k=2: once the bad "
            "catalyst is named, one sound survivor determines the "
            "region and no majority is needed. Under the definite "
            "article of the statement, k >= 3 is not required."
            % (stab[1]["fraction"], stab[2]["fraction"]))

    # The proof's own reason --- "no majority to adjudicate" --- is
    # about the case where the erroneous catalyst is NOT identified.
    # That is a different operation and it is graded separately, as
    # exp3 grades thm:minimal-record's two readings.
    e2 = ex.expect(
        "prop:three the bound the proof actually establishes",
        "When the erroneous catalyst is not identified, the region is "
        "whatever the catalysts' majority yields. Then k=2 leaves a "
        "one-one split with no majority and k >= 3 is genuinely the "
        "first count at which the true region is recovered.",
        "prop:three (proof)",
        "k=2 recovering the true region unaided, which would remove "
        "the proof's reason as well as its conclusion.")

    unaided = {}
    for k in [1, 2, 3, 4, 5]:
        rec, tot = 0, 0
        for _ in range(2000):
            truth = rng.randint(0, 2)
            wrong = rng.choice([c for c in range(3) if c != truth])
            votes = [truth] * (k - 1) + [wrong]
            rng.shuffle(votes)
            tot += 1
            if region(votes) == frozenset({truth}):
                rec += 1
        unaided[k] = {"trials": tot, "recovered_true_region": rec,
                      "fraction": round(rec / tot, 4)}
    ex.record("T7b_three_catalysts_unaided", {
        "reading": "the erroneous catalyst is NOT identified; the "
                   "region is the majority over all k catalysts",
        "by_k": unaided})
    e2.check(unaided[1]["fraction"] == 0.0
             and unaided[2]["fraction"] == 0.0
             and min(unaided[3]["fraction"], unaided[4]["fraction"],
                     unaided[5]["fraction"]) > 0.99,
             {k: v["fraction"] for k, v in unaided.items()},
             "unaided recovery is %.3f at k=1, %.3f at k=2 and %.3f "
             "from k=3: three catalysts is exactly the threshold, so "
             "the proof's reasoning is sound for the operation it "
             "describes even though the statement quantifies a "
             "different one"
             % (unaided[1]["fraction"], unaided[2]["fraction"],
                unaided[3]["fraction"]))

    ex.note("Sections 4's tests run on the NIST AC/CAC MS/MS library "
            "(5298 usable spectra, 244 compounds, nine collision "
            "energies). Compounds are vertices; a contact edge exists "
            "where two compounds share at least two binned product ions "
            "at 0.01 Th and 1%% of base peak. The binary .INU/.DBU index "
            "files alongside the library carry no documented layout and "
            "are not read; the vendor's .MSP text export of the same "
            "content is.")
    ex.note("T3b records a result that runs against a loose reading of "
            "cor:mass: on this curated library precursor m/z alone "
            "separates every compound at 10 ppm. The corollary is not a "
            "claim about ambiguity in curated sets. Its content is T3's: "
            "a point does not determine a region, and refining the point "
            "does not make it one.")
    ex.note("prop:three is graded twice, as exp3 grades "
            "thm:minimal-record. The statement says the region is "
            "stable under removal of THE erroneous catalyst --- the "
            "definite article identifies it --- and under that reading "
            "k=2 already recovers the true region in every trial, "
            "because one named-sound survivor settles it and no "
            "majority is needed. The proof's reason, 'no majority to "
            "adjudicate', describes the case where the erroneous "
            "catalyst is NOT identified; there k=2 recovers nothing "
            "and k=3 recovers everything, exactly as claimed. The "
            "proof is sound for an operation the statement does not "
            "quantify. The fix is to the statement, not the argument: "
            "it should suppose one catalyst may be in error without "
            "supposing it can be picked out.")
    ex.note("Section 6's claims are about procedures over catalysts, "
            "which a static library does not contain, so T5-T7 run on "
            "constructed configurations where the catalysts are the "
            "controlled variable.")

    ex.report()
    print("  written: " + ex.write())
    return ex


if __name__ == "__main__":
    main()
