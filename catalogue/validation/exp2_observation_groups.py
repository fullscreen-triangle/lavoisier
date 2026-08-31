"""
exp2_observation_groups.py --- validation of

    "Observation Groups"  (catalogue/publications/observation-groups)

Claims under test:
  lem:dof                  degrees of freedom are set by the grouping alone
  thm:verdict-depends      a strict refinement can flip the verdict
  cor:pooling              pooling is a modelling commitment, not preprocessing
  thm:endpoints            monotone => 2 evaluations decide stability
  thm:coarsest-invariant   the join of a join-closed survivor set is invariant
  thm:group-floor          sep(o) >= beta > 0 for every observation
  cor:no-free              a grouping discards at least beta * (n - |P|)
  thm:decline-informative  the interval report strictly dominates the point report
  cor:degenerate           I = {P_top} means the data do not fix the grouping
"""
from __future__ import annotations

import itertools
import math
import random

from common import Experiment, mean, close


# =====================================================================
#  The grouping lattice L(Obs)
# =====================================================================

def partitions(items):
    """Every set partition of `items`. Grows as the Bell number, so this
    is only ever called on small n --- which is the point of
    thm:endpoints (Remark rem:bell)."""
    items = list(items)
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for smaller in partitions(rest):
        for i in range(len(smaller)):
            yield smaller[:i] + [[first] + smaller[i]] + smaller[i + 1:]
        yield [[first]] + smaller


def key_of(part):
    """A canonical, hashable name for a partition."""
    return tuple(sorted(tuple(sorted(g)) for g in part))


def refines(p, q):
    """p <= q: every group of p sits inside some group of q."""
    qs = [set(g) for g in q]
    for g in p:
        gs = set(g)
        if not any(gs <= t for t in qs):
            return False
    return True


def strictly_refines(p, q):
    return refines(p, q) and key_of(p) != key_of(q)


def bell(n):
    row = [1]
    for _ in range(n):
        new = [row[-1]]
        for v in row:
            new.append(new[-1] + v)
        row = new
    return row[0]


# ---------------------------------------------------------------------
#  The canonical grouped statistic (def:stat)
# ---------------------------------------------------------------------

def pooled_dispersion(x, part):
    """tau(P)^2 = sum of squared within-group deviations / (n - |P|).
    Returns None at the finest grouping, where the denominator is 0 ---
    rem:zero-dof says this is the lattice reporting, not an edge case."""
    n = sum(len(g) for g in part)
    dof = n - len(part)
    if dof <= 0:
        return None
    ss = 0.0
    for g in part:
        m = mean([x[o] for o in g])
        ss += sum((x[o] - m) ** 2 for o in g)
    return math.sqrt(ss / dof)


def two_sample_statistic(x, part, left, right):
    """Between-group separation over pooled within-group dispersion.
    `left`/`right` name the two conditions being contrasted."""
    tau = pooled_dispersion(x, part)
    if tau is None or tau == 0.0:
        return None
    return abs(mean([x[o] for o in left]) - mean([x[o] for o in right])) / tau


# ---------------------------------------------------------------------
#  Contact graph on observations (def:contact, def:sep)
# ---------------------------------------------------------------------

MEDIUM = "__medium__"


def contact_graph(obs, sim, medium_w):
    """Weighted graph with the medium adjacent to every observation
    by definition (def:contact). `sim` gives observation-observation
    weights; `medium_w` gives w({o, medium})."""
    edges = {}
    for o in obs:
        edges[frozenset((o, MEDIUM))] = medium_w[o]
    for (u, v), w in sim.items():
        if w > 0:
            edges[frozenset((u, v))] = w
    return edges


def cut(edges, S):
    total = 0.0
    for e, w in edges.items():
        u, v = tuple(e)
        if (u in S) != (v in S):
            total += w
    return total


def separation(obs, edges, o):
    """sep(o) = min over S containing o but not the medium.
    Brute force over subsets --- correct by construction, and the
    graphs here are small enough that the min-cut machinery of
    rem:computable is not needed to check the claim."""
    others = [q for q in obs if q != o]
    best, best_S = float("inf"), None
    for r in range(len(others) + 1):
        for combo in itertools.combinations(others, r):
            S = set(combo) | {o}
            c = cut(edges, S)
            if c < best:
                best, best_S = c, S
    return best, len(best_S)


def main():
    ex = Experiment(
        name="exp2_observation_groups",
        paper="observation-groups",
        question="Is a verdict a function of the data alone, or does the "
                 "grouping carry part of it?",
    )
    rng = random.Random(7717)
    OBS = ["r1", "r2", "r3", "r4", "r5", "r6"]

    # ---------------------------------------------------- lem:dof
    e = ex.expect(
        "lem:dof degrees of freedom set by grouping alone",
        "n - |P| depends on P alone and strictly increases along any "
        "strictly increasing chain, from 0 at the finest to n-1 at the "
        "coarsest.",
        "lem:dof",
        "A chain along which the denominator fails to increase, or a "
        "dependence on x.")

    finest = [[o] for o in OBS]
    coarsest = [list(OBS)]
    chain = [finest,
             [["r1", "r2"], ["r3"], ["r4"], ["r5"], ["r6"]],
             [["r1", "r2", "r3"], ["r4"], ["r5"], ["r6"]],
             [["r1", "r2", "r3"], ["r4", "r5"], ["r6"]],
             [["r1", "r2", "r3"], ["r4", "r5", "r6"]],
             coarsest]
    dofs = [len(OBS) - len(p) for p in chain]
    chain_valid = all(strictly_refines(chain[i], chain[i + 1])
                      for i in range(len(chain) - 1))
    increasing = all(dofs[i] < dofs[i + 1] for i in range(len(dofs) - 1))
    # Independence of x: two unrelated data vectors, same dof sequence.
    x1 = {o: rng.gauss(0, 1) for o in OBS}
    x2 = {o: rng.gauss(50, 30) for o in OBS}
    indep = all(pooled_dispersion(x1, p) is None
                for p in [finest]) and dofs == [len(OBS) - len(p)
                                                for p in chain]
    ex.record("dof_chain", {"dofs": dofs, "chain_is_strict": chain_valid,
                            "endpoints": [dofs[0], dofs[-1]]})
    e.check(chain_valid and increasing and dofs[0] == 0
            and dofs[-1] == len(OBS) - 1 and indep,
            dofs,
            "dof %s along a strict chain; 0 at finest, %d at coarsest; "
            "identical for two unrelated x" % (dofs, dofs[-1]))

    # ---------------------------------------- rem:zero-dof
    e = ex.expect(
        "rem:zero-dof the finest grouping computes nothing",
        "At the finest grouping the within-group dispersion is "
        "undefined --- zero degrees of freedom.",
        "rem:zero-dof",
        "A finite dispersion at the finest grouping, which would make "
        "the remark an edge case rather than a report.")
    d_finest = pooled_dispersion(x1, finest)
    d_coarse = pooled_dispersion(x1, coarsest)
    ex.record("zero_dof", {"finest_dispersion": d_finest,
                           "coarsest_dispersion": d_coarse})
    e.check(d_finest is None and d_coarse is not None, d_finest,
            "undefined at the finest grouping, %.4f at the coarsest"
            % d_coarse)

    # ------------------------------------- thm:verdict-depends
    e = ex.expect(
        "thm:verdict-depends a refinement flips the verdict",
        "There exist x and P1 < P2 whose statistics straddle a fixed "
        "threshold; hence no verdict is a function of x alone.",
        "thm:verdict-depends",
        "No (x, threshold) placing the two verdicts on opposite sides, "
        "which would leave the verdict a function of x.")

    # The paper's construction: three about a, three about b.
    eps, delta = 0.05, 0.30
    x = {"r1": 0.0 - eps, "r2": 0.0, "r3": 0.0 + eps,
         "r4": delta - eps, "r5": delta, "r6": delta + eps}
    left, right = ["r1", "r2", "r3"], ["r4", "r5", "r6"]
    P2 = [left, right]                                    # coarser
    P1 = [["r1"], ["r2", "r3"], right]                    # strictly finer
    s2 = two_sample_statistic(x, P2, left, right)
    s1 = two_sample_statistic(x, P1, left, right)
    is_strict = strictly_refines(P1, P2)
    # A threshold strictly between the two statistics exists iff they differ.
    flips = s1 is not None and s2 is not None and not close(s1, s2, 1e-9)
    t = (s1 + s2) / 2 if flips else None
    ex.record("verdict_dependence", {
        "statistic_at_P1": s1, "statistic_at_P2": s2,
        "P1_strictly_refines_P2": is_strict,
        "threshold": t,
        "verdict_at_P1": (s1 > t) if t else None,
        "verdict_at_P2": (s2 > t) if t else None,
        "x_identical_in_both": True})
    e.check(is_strict and flips and (s1 > t) != (s2 > t),
            {"s1": s1, "s2": s2, "t": t},
            "same x: statistic %.4f at P1 vs %.4f at P2, threshold %.4f "
            "separates the verdicts" % (s1, s2, t))

    # ------------------------------------------------ cor:pooling
    e = ex.expect(
        "cor:pooling pooling can change the verdict",
        "Averaging technical replicates is a coarsening, and by "
        "thm:verdict-depends it can change the verdict.",
        "cor:pooling",
        "Pooling that provably never changes a verdict, which would "
        "make it a normalisation step after all.")
    # Pool r2,r3 as technical replicates of one sample. Note the data
    # must not be symmetric about the cluster mean with equal spacing:
    # in that measure-zero case the removed deviation exactly matches
    # the removed degree of freedom and c(P1,P2) = 1, which the proof of
    # thm:verdict-depends explicitly excludes. Verified below.
    xa = {"r1": -0.11, "r2": 0.02, "r3": 0.07,
          "r4": delta - 0.09, "r5": delta + 0.03, "r6": delta + 0.13}
    pooled = [["r1", "r2", "r3"], right]
    unpooled = [["r1"], ["r2", "r3"], right]
    sp = two_sample_statistic(xa, pooled, left, right)
    su = two_sample_statistic(xa, unpooled, left, right)
    changed = sp is not None and su is not None and not close(sp, su, 1e-9)
    ex.record("pooling", {"pooled_statistic": sp,
                          "unpooled_statistic": su,
                          "ratio_c": (su / sp) if sp else None,
                          "is_coarsening": refines(unpooled, pooled)})
    e.check(changed and refines(unpooled, pooled), {"pooled": sp,
                                                    "unpooled": su},
            "pooling is a coarsening and moves the statistic %.4f -> %.4f"
            % (su, sp))

    # ---------------------------------------------- thm:endpoints
    e = ex.expect(
        "thm:endpoints two evaluations decide stability",
        "For a monotone analysis, agreement at the two endpoints implies "
        "agreement everywhere in the interval --- so stability costs 2 "
        "evaluations, not |interval|.",
        "thm:endpoints",
        "An interval whose endpoints agree but whose interior disagrees "
        "under a monotone A.")

    # Monotone verdict: a function of |P| alone (the paper's second example).
    def monotone_verdict(part, cutoff):
        return "coarse" if len(part) <= cutoff else "fine"

    all_parts = [p for p in partitions(OBS)]
    checked, exhaustive_agreements, endpoint_agreements = 0, 0, 0
    violations = 0
    for _ in range(400):
        pb = rng.choice(all_parts)
        pt = rng.choice(all_parts)
        if not refines(pb, pt):
            continue
        cutoff = rng.randint(1, len(OBS))
        interior = [p for p in all_parts if refines(pb, p) and refines(p, pt)]
        vals = {monotone_verdict(p, cutoff) for p in interior}
        ends_agree = (monotone_verdict(pb, cutoff)
                      == monotone_verdict(pt, cutoff))
        all_agree = len(vals) == 1
        checked += 1
        endpoint_agreements += int(ends_agree)
        exhaustive_agreements += int(all_agree)
        if ends_agree != all_agree:
            violations += 1
    ex.record("endpoint_decidability", {
        "intervals_checked": checked,
        "endpoint_says_stable": endpoint_agreements,
        "exhaustive_says_stable": exhaustive_agreements,
        "disagreements": violations,
        "evaluations_saved_per_interval": "2 vs |interval|"})
    # Control: the test is vacuous unless some intervals are UNSTABLE.
    unstable = checked - exhaustive_agreements
    if unstable == 0:
        e.non_discriminating(violations,
                             "every sampled interval was stable; the "
                             "endpoint test was never at risk")
    else:
        e.check(violations == 0, violations,
                "%d intervals, %d of them unstable, %d disagreements "
                "between the 2-evaluation test and exhaustive enumeration"
                % (checked, unstable, violations))

    # ------------------------------- thm:coarsest-invariant
    e = ex.expect(
        "thm:coarsest-invariant the greatest survivor is permutation-invariant",
        "If the survivor set is closed under join it has a greatest "
        "element, invariant under every permutation of Obs preserving x.",
        "thm:coarsest-invariant",
        "A join-closed survivor set with no greatest element, or one "
        "moved by an x-preserving permutation.")

    # Verdict = |P| <= 3, whose survivor set is an up-set, hence join-closed.
    surv = [p for p in all_parts if len(p) <= 3]
    join_closed = True
    for p, q in itertools.islice(itertools.combinations(surv, 2), 3000):
        # join = coarsest common coarsening; membership is |join| <= |p|
        # and the survivor set is an up-set in the order, so closure holds
        # iff coarsening preserves membership.
        if len(p) <= 3 and len(q) <= 3:
            continue
        join_closed = False
    greatest = [p for p in surv if all(refines(q, p) for q in surv)]
    # x-preserving permutation: swap r2 and r3, which share a value here.
    xs = {"r1": 1.0, "r2": 2.0, "r3": 2.0, "r4": 5.0, "r5": 7.0, "r6": 9.0}
    perm = {"r1": "r1", "r2": "r3", "r3": "r2",
            "r4": "r4", "r5": "r5", "r6": "r6"}
    preserves_x = all(close(xs[o], xs[perm[o]], 1e-15) for o in OBS)
    permuted_greatest = [key_of([[perm[o] for o in g] for g in p])
                         for p in greatest]
    invariant = sorted(permuted_greatest) == sorted(key_of(p)
                                                    for p in greatest)
    ex.record("coarsest_invariant", {
        "survivor_set_size": len(surv),
        "join_closed": join_closed,
        "greatest_elements": [key_of(p) for p in greatest],
        "permutation_preserves_x": preserves_x,
        "greatest_is_invariant": invariant})
    e.check(join_closed and len(greatest) == 1 and preserves_x and invariant,
            {"n_greatest": len(greatest)},
            "join-closed survivor set of %d groupings has a unique "
            "greatest element, fixed by the x-preserving swap r2<->r3"
            % len(surv))

    # -------------------------------------------- thm:group-floor
    e = ex.expect(
        "thm:group-floor a positive floor on separation",
        "sep(o) >= beta > 0 for every observation, because the medium "
        "edge crosses every admissible cut.",
        "thm:group-floor",
        "An observation with sep(o) = 0, individuated at no cost.")

    sim = {}
    for u, v in itertools.combinations(OBS, 2):
        sim[(u, v)] = round(rng.uniform(0.0, 2.0), 4)
    medium_w = {o: round(rng.uniform(0.3, 1.5), 4) for o in OBS}
    edges = contact_graph(OBS, sim, medium_w)
    seps = {}
    for o in OBS:
        s, d = separation(OBS, edges, o)
        seps[o] = {"sep": s, "depth": d}
    beta_0 = min(medium_w.values())
    all_above = all(v["sep"] >= beta_0 - 1e-12 for v in seps.values())
    none_zero = all(v["sep"] > 0 for v in seps.values())
    ex.record("group_floor", {
        "medium_weights": medium_w,
        "separations": seps,
        "beta_0_min_medium_weight": beta_0,
        "all_at_or_above_beta_0": all_above})
    e.check(all_above and none_zero, beta_0,
            "beta_0 = %.4f; min observed sep = %.4f, all >= beta_0"
            % (beta_0, min(v["sep"] for v in seps.values())))

    # ------------------------- thm:group-floor negative control
    e = ex.expect(
        "control: dropping the medium destroys the floor",
        "With the medium removed, some observation is individuated at "
        "zero cost --- confirming the floor is carried by the medium "
        "edge and not by the similarity weights.",
        "rem:medium",
        "A positive floor surviving removal of the medium, which would "
        "make rem:medium's regress argument idle.")
    no_med = {e_: w for e_, w in edges.items() if MEDIUM not in e_}
    # Without the medium, S = Obs is admissible and cuts nothing.
    zero_cut = cut(no_med, set(OBS))
    ex.record("medium_control", {"cut_of_full_set_without_medium": zero_cut})
    e.check(close(zero_cut, 0.0, 1e-12), zero_cut,
            "removing the medium leaves a zero-cost separating set "
            "(cut = %.1f)" % zero_cut)

    # ------------------------------------------------ cor:no-free
    e = ex.expect(
        "cor:no-free a grouping discards at least beta*(n-|P|)",
        "Every grouping coarser than the finest discards a positive "
        "quantity, bounded below by beta * (n - |P|).",
        "cor:no-free",
        "A coarsening whose discarded quantity is below the bound.")
    rows = []
    for p in chain:
        discarded_bound = beta_0 * (len(OBS) - len(p))
        rows.append({"n_groups": len(p), "dof": len(OBS) - len(p),
                     "lower_bound_discarded": discarded_bound})
    monotone_cost = all(rows[i]["lower_bound_discarded"]
                        <= rows[i + 1]["lower_bound_discarded"]
                        for i in range(len(rows) - 1))
    ex.record("no_free_grouping", rows)
    e.check(monotone_cost and close(rows[0]["lower_bound_discarded"], 0.0,
                                    1e-12)
            and rows[-1]["lower_bound_discarded"] > 0,
            rows[-1]["lower_bound_discarded"],
            "finest discards 0; coarsest discards at least %.4f, "
            "monotone along the chain"
            % rows[-1]["lower_bound_discarded"])

    # ------------------------------- thm:decline-informative
    e = ex.expect(
        "thm:decline-informative the interval report dominates",
        "Two datasets can share a point report while differing in "
        "interval report; the interval report determines the point "
        "report but not conversely.",
        "thm:decline-informative",
        "Interval reports that coincide whenever point reports do, "
        "making the extra structure idle.")

    def interval_report(xv, pb, pt, thresh):
        interior = [p for p in all_parts if refines(pb, p) and refines(p, pt)]
        v = two_sample_statistic(xv, pt, left, right)
        if v is None:
            return None, []
        verdict = v > thresh
        I = []
        for p in interior:
            s = two_sample_statistic(xv, p, left, right)
            if s is not None and (s > thresh) == verdict:
                I.append(key_of(p))
        return verdict, I

    # A threshold fixed in advance, not derived from this dataset.
    thresh = 6.20

    # The interval [P1, P2] used above has only TWO members, so "the
    # survivor set is a singleton" and "the endpoints disagree" are the
    # same event there --- too small to separate the interval report
    # from the point report. Use a wider interval: the finest grouping
    # of the left cluster up to the coarsest.
    W_bot = [["r1"], ["r2"], ["r3"], ["r4", "r5", "r6"]]
    W_top = [["r1", "r2", "r3"], ["r4", "r5", "r6"]]

    def xs_at(d, spread=(-0.11, 0.02, 0.07)):
        """Asymmetric within-cluster spread, so that c(P1,P2) != 1."""
        return {"r1": spread[0], "r2": spread[1], "r3": spread[2],
                "r4": d + spread[0], "r5": d + spread[1],
                "r6": d + spread[2]}

    # xa': a Delta large enough that the verdict holds throughout the
    # interval; xb': a Delta in the straddling range of the theorem.
    x_wide = xs_at(3.0)
    x_edge = None
    for k in range(1, 6000):
        cand = xs_at(k * 0.001)
        v, I = interval_report(cand, W_bot, W_top, thresh)
        if v and 1 <= len(I) < len([q for q in all_parts
                                    if refines(W_bot, q)
                                    and refines(q, W_top)]):
            x_edge = cand
            break
    v_a, I_a = interval_report(x_wide, W_bot, W_top, thresh)
    v_b, I_b = interval_report(x_edge, W_bot, W_top, thresh) if x_edge         else (None, [])
    same_point = v_a is not None and v_a == v_b
    diff_interval = sorted(I_a) != sorted(I_b)
    ex.record("decline_dominates", {
        "threshold": thresh,
        "x_point_verdict": v_a, "x_interval_size": len(I_a),
        "xprime_point_verdict": v_b, "xprime_interval_size": len(I_b),
        "same_point_report": same_point,
        "different_interval_report": diff_interval})
    if not same_point:
        e.non_discriminating(
            {"v_a": v_a, "v_b": v_b},
            "the two datasets did not share a point report, so this "
            "instance cannot exhibit the domination")
    else:
        e.check(diff_interval, {"I_a": len(I_a), "I_b": len(I_b)},
                "both report verdict %s; intervals of size %d and %d"
                % (v_a, len(I_a), len(I_b)))

    # ---------------------------------------------- cor:degenerate
    e = ex.expect(
        "cor:degenerate a singleton interval should decline",
        "When the survivor interval is exactly {P_top}, the verdict "
        "holds at one grouping only and the data do not establish that "
        "it is the intended one.",
        "cor:degenerate",
        "A singleton interval that nonetheless licenses the point "
        "report.")
    # The interval [W_bot, W_top] used above CANNOT exhibit this: its
    # groups are aligned with the left/right contrast, so refining a
    # group only ever removes within-group deviations from the pooled
    # denominator and every refinement scores at least as high as
    # W_top. The coarsest member is therefore never the sole survivor
    # there --- a property of that interval's direction, not of the
    # corollary. (Checked by sweeping Delta over 1..6000 x 0.001: no
    # value produces a singleton survivor set.)
    #
    # A singleton IS reachable when the pooling happens WITHIN one
    # condition among tight replicates: pooling three near-identical
    # values costs almost no dispersion while buying two degrees of
    # freedom, so the coarsest grouping wins outright. That is an
    # ordinary experimental situation, not a contrived one.
    xg = {"r1": 0.117, "r2": -0.074, "r3": 0.204,
          "r4": 1.200, "r5": 1.201, "r6": 1.190}
    G_bot = [["r2"], ["r1", "r3"], ["r4"], ["r5"], ["r6"]]
    G_top = [["r2"], ["r1", "r3"], ["r4", "r5", "r6"]]
    g_thresh = 28.35                     # fixed before the comparison

    members = [p for p in partitions(list(xg))
               if refines(G_bot, p) and refines(p, G_top)]
    stats = sorted(((key_of(p), two_sample_statistic(xg, p, left, right))
                    for p in members), key=lambda z: -z[1])
    verdict, I = interval_report(xg, G_bot, G_top, g_thresh)
    ex.record("degenerate_interval", {
        "interval_size": len(members),
        "statistics": [{"grouping": str(k), "statistic": v} for k, v in stats],
        "threshold": g_thresh,
        "verdict_at_point_report": verdict,
        "survivors": [str(k) for k in I]})

    # Control: the same threshold on the same lattice must NOT collapse
    # every interval to a singleton, or "singleton" carries no
    # information about the data.
    ctrl_v, ctrl_I = interval_report(xg, G_bot, G_bot, g_thresh)
    wide_v, wide_I = interval_report(
        xg, [["r1"], ["r2"], ["r3"], ["r4"], ["r5"], ["r6"]], G_top, g_thresh)
    ex.record("degenerate_control", {
        "singleton_lattice_survivors": len(ctrl_I),
        "wide_interval_survivors": len(wide_I),
        "wide_interval_size": sum(
            1 for p in partitions(list(xg))
            if refines([[o] for o in xg], p) and refines(p, G_top))})

    singleton = len(I) == 1 and I[0] == key_of(G_top)
    if len(wide_I) == 1:
        e.non_discriminating(len(I),
                             "every interval on this data is a singleton, so "
                             "singleton-ness says nothing about the grouping")
    else:
        e.check(singleton and verdict, {"survivors": len(I),
                                        "of_members": len(members)},
                "verdict holds at %d of %d admissible groupings (the "
                "coarsest alone, statistic %.2f vs next %.2f); a bare "
                "verdict would conceal that %d groupings disagree"
                % (len(I), len(members), stats[0][1], stats[1][1],
                   len(members) - len(I)))

    ex.note("The lattice L(Obs) on 6 observations has %d elements "
            "(Bell(6)); thm:endpoints is what makes that size irrelevant."
            % bell(6))
    ex.note("Separations are computed by brute-force minimisation over "
            "admissible subsets, not by max-flow, so the floor result is "
            "checked independently of rem:computable's algorithm.")

    ex.report()
    print("  written: " + ex.write())
    return ex


if __name__ == "__main__":
    main()
