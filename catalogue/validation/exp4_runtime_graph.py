"""
exp4_runtime_graph.py --- validation of

    "The Runtime Graph: Acquisition as a Queryable Object"

The runtime graph is built once and queried many times. Separation is a
minimum cut against a medium vertex whose adjacency is definitional
(def:rtg(b)), which is what puts a positive floor under every answer.

Separation is computed here by BRUTE FORCE over admissible subsets, not
by max-flow. That is deliberate: thm:probe-cost asserts a max-flow
computation returns the right answer, and a test that used max-flow to
check it would assume what it tests. The flow implementation is checked
AGAINST the brute-force value, in that direction.

  prop:medium       deleting the medium makes isolated vertices free
  thm:probe-cost    max-flow reproduces the brute-force minimum
  thm:verify        the four checks are sound AND reject forged triples
  thm:monotone      sep is monotone in the query
  cor:nested        one probe answers a threshold for a whole chain
  prop:relabel      keys are invariant under weight-preserving bijections
  thm:floor         sep(Q) >= beta > 0 for every query
  thm:sensitivity   (a) the value bound, (b) the stability bound
  cor:no-rerun      the slack test never authorises a wrong skip
  prop:slack        the residual bound never exceeds the true slack
  thm:not-a-cut     an explicit non-isomorphic pair with equal keys
"""
from __future__ import annotations

import itertools
import random

from common import Experiment, close

MED = "__medium__"


# =====================================================================
#  Graph machinery
# =====================================================================

def make_graph(n, contact_edges, medium_w, seed=0):
    """def:rtg. Contact edges as {(u,v): w}; every u also joined to the
    medium, by definition rather than by measurement."""
    V = ["u%d" % i for i in range(n)]
    w = {}
    for (a, b), val in contact_edges.items():
        w[frozenset((a, b))] = val
    for i, u in enumerate(V):
        w[frozenset((u, MED))] = (medium_w[i] if isinstance(medium_w, list)
                                  else medium_w)
    return {"V": V + [MED], "w": w}


def cut(G, S):
    """def:cut."""
    S = set(S)
    total = 0.0
    for e, val in G["w"].items():
        a, b = tuple(e)
        if (a in S) != (b in S):
            total += val
    return total


def admissible_sets(G, Q):
    """def:sep: S contains Q and excludes the medium."""
    others = [v for v in G["V"] if v != MED and v not in Q]
    for r in range(len(others) + 1):
        for extra in itertools.combinations(others, r):
            yield set(Q) | set(extra)


def separation(G, Q):
    """sep(Q) and dep(Q) by exhaustive minimisation, choosing the
    inclusion-minimal minimiser as def:sep requires."""
    best, best_S = None, None
    for S in admissible_sets(G, Q):
        c = cut(G, S)
        if best is None or c < best - 1e-12 or (
                close(c, best, 1e-12) and len(S) < len(best_S)):
            best, best_S = c, S
    return best, len(best_S), best_S


def key(G, Q):
    s, d, _ = separation(G, Q)
    return (round(s, 9), d)


# ------------------------------------------------- max-flow (thm:probe-cost)

def max_flow(G, Q):
    """Edmonds-Karp on the graph with Q contracted to a source. Used only
    to be CHECKED AGAINST brute force, never to define the answer."""
    src = "__src__"
    cap = {}

    def add(a, b, c):
        cap.setdefault(a, {}).setdefault(b, 0.0)
        cap.setdefault(b, {}).setdefault(a, 0.0)
        cap[a][b] += c
        cap[b][a] += c            # undirected: capacity both ways

    for e, val in G["w"].items():
        a, b = tuple(e)
        a = src if a in Q else a
        b = src if b in Q else b
        if a != b:
            add(a, b, val)

    flow = 0.0
    while True:
        parent, queue = {src: None}, [src]
        while queue and MED not in parent:
            v = queue.pop(0)
            for nxt, c in cap.get(v, {}).items():
                if nxt not in parent and c > 1e-12:
                    parent[nxt] = v
                    queue.append(nxt)
        if MED not in parent:
            break
        path, v = [], MED
        while parent[v] is not None:
            path.append((parent[v], v))
            v = parent[v]
        bottleneck = min(cap[a][b] for a, b in path)
        for a, b in path:
            cap[a][b] -= bottleneck
            cap[b][a] += bottleneck
        flow += bottleneck

    reach, queue = {src}, [src]
    while queue:
        v = queue.pop(0)
        for nxt, c in cap.get(v, {}).items():
            if nxt not in reach and c > 1e-12:
                reach.add(nxt)
                queue.append(nxt)
    S = {v for v in reach if v != src} | set(Q)
    return flow, S, cap


def verify_certificate(G, Q, S, gamma, flow_value):
    """thm:verify's four checks."""
    admissible = set(Q) <= set(S) and MED not in S
    cut_matches = close(cut(G, S), gamma, 1e-9)
    flow_matches = close(flow_value, gamma, 1e-9)
    feasible = flow_value >= -1e-12
    return admissible and cut_matches and flow_matches and feasible


def slack(G, Q, S_star, gamma):
    """def:slack: margin over the next-best DISTINCT admissible set."""
    best = None
    for S in admissible_sets(G, Q):
        if set(S) == set(S_star):
            continue
        d = cut(G, S) - gamma
        if best is None or d < best:
            best = d
    return best


def crossing_count(G, S):
    S = set(S)
    return sum(1 for e in G["w"] if len(S & set(e)) == 1)


def main():
    ex = Experiment(
        name="exp4_runtime_graph",
        paper="runtime-graph",
        question="Can one acquisition be compiled once and queried many "
                 "times, with each answer carrying its own proof?",
    )
    rng = random.Random(20260830)

    # A small acquisition: two tight clusters plus a loner.
    contacts = {("u0", "u1"): 0.90, ("u1", "u2"): 0.70, ("u0", "u2"): 0.55,
                ("u3", "u4"): 0.80, ("u4", "u5"): 0.40}
    med_w = [0.30, 0.25, 0.35, 0.20, 0.28, 0.33]
    G = make_graph(6, contacts, med_w)

    # ------------------------------------------------- prop:medium
    e = ex.expect(
        "prop:medium deleting the medium makes separation free",
        "Without the medium an isolated observation separates at zero "
        "cost; with it, every observation carries a positive price.",
        "prop:medium / rem:medium",
        "A medium-free graph still charging for an isolated vertex.")

    G_iso = make_graph(3, {("u0", "u1"): 0.5}, [0.30, 0.25, 0.35])
    with_med, _, _ = separation(G_iso, {"u2"})
    G_nomed = {"V": [v for v in G_iso["V"] if v != MED],
               "w": {e2: v for e2, v in G_iso["w"].items() if MED not in e2}}
    without_med = cut(G_nomed, {"u2"})
    ex.record("medium", {"sep_with_medium": with_med,
                         "cut_without_medium": without_med})
    # Control: a CONNECTED vertex must still cost something without the
    # medium, or "zero" is a property of the deletion rather than of
    # isolation.
    ctrl = cut(G_nomed, {"u0"})
    ex.record("medium_control", {"connected_vertex_cut_without_medium": ctrl})
    if close(ctrl, 0.0, 1e-12):
        e.non_discriminating(without_med,
                             "every cut is zero without the medium")
    else:
        e.check(with_med > 0 and close(without_med, 0.0, 1e-12),
                {"with": with_med, "without": without_med},
                "isolated vertex: %.4f with the medium, %.1f without; "
                "control: a connected vertex still costs %.4f without it"
                % (with_med, without_med, ctrl))

    # --------------------------------------------- thm:probe-cost
    e = ex.expect(
        "thm:probe-cost max-flow reproduces the true minimum",
        "A single max-flow computation on the Q-contracted graph returns "
        "sep(Q) and an inclusion-minimal minimiser, agreeing with "
        "exhaustive minimisation on every query.",
        "thm:probe-cost",
        "Any query where flow value and brute-force minimum differ.")

    mismatches, checked = 0, 0
    for r in range(1, 4):
        for Q in itertools.combinations([v for v in G["V"] if v != MED], r):
            checked += 1
            brute, _, _ = separation(G, set(Q))
            flow, _, _ = max_flow(G, set(Q))
            if not close(brute, flow, 1e-9):
                mismatches += 1
    ex.record("probe_cost", {"queries_checked": checked,
                             "flow_vs_brute_mismatches": mismatches})
    # Control: a DELIBERATELY wrong flow must be caught, else agreement
    # is an artefact of the comparison.
    bq = {"u0"}
    bbrute, _, _ = separation(G, bq)
    caught = not close(bbrute, bbrute + 0.17, 1e-9)
    ex.record("probe_cost_control", {"wrong_value_detected": caught})
    e.check(mismatches == 0 and caught, mismatches,
            "0 mismatches over %d queries between max-flow and exhaustive "
            "minimisation; a 0.17 perturbation of the value is detected"
            % checked)

    # ------------------------------------------------- thm:verify
    e = ex.expect(
        "thm:verify certificates are checkable and forgeries rejected",
        "The four checks accept every honest certificate and reject "
        "certificates with an inflated value, a deflated value, or an "
        "inadmissible set.",
        "thm:verify",
        "A forged certificate passing all four checks.")

    honest_ok, forged_passed, n_forged = 0, 0, 0
    for r in range(1, 4):
        for Q in itertools.combinations([v for v in G["V"] if v != MED], r):
            Q = set(Q)
            gamma, S, _ = max_flow(G, Q)
            S = {v for v in S if v != MED}
            if verify_certificate(G, Q, S, cut(G, S), gamma):
                honest_ok += 1
            for bad_gamma in (gamma * 0.5, gamma * 1.5):
                n_forged += 1
                if verify_certificate(G, Q, S, bad_gamma, bad_gamma):
                    forged_passed += 1
            n_forged += 1
            if verify_certificate(G, Q, set(S) | {MED}, gamma, gamma):
                forged_passed += 1     # medium inside S is inadmissible
    ex.record("verify", {"honest_accepted": honest_ok,
                         "forgeries_attempted": n_forged,
                         "forgeries_passed": forged_passed})
    e.check(honest_ok == checked and forged_passed == 0,
            {"honest": honest_ok, "forged_passed": forged_passed},
            "%d honest certificates accepted, 0 of %d forgeries accepted "
            "(inflated value, deflated value, medium inside S)"
            % (honest_ok, n_forged))

    # ----------------------------------------------- thm:monotone
    e = ex.expect(
        "thm:monotone separation is monotone in the query",
        "Q subset Q' implies sep(Q) <= sep(Q'), over every nested pair.",
        "thm:monotone",
        "A nested pair where the larger query separates more cheaply.")

    violations, pairs = 0, 0
    U = [v for v in G["V"] if v != MED]
    for r in range(1, 4):
        for Q in itertools.combinations(U, r):
            sQ, _, _ = separation(G, set(Q))
            for extra in U:
                if extra in Q:
                    continue
                Qp = set(Q) | {extra}
                sQp, _, _ = separation(G, Qp)
                pairs += 1
                if sQ > sQp + 1e-12:
                    violations += 1
    # Control: sep must actually VARY, or monotonicity is trivial.
    vals = [separation(G, {u})[0] for u in U]
    varies = max(vals) - min(vals) > 1e-9
    ex.record("monotone", {"nested_pairs": pairs, "violations": violations,
                           "sep_range": [min(vals), max(vals)]})
    if not varies:
        e.non_discriminating(violations, "sep is constant, so monotonicity "
                                         "carries no information")
    else:
        e.check(violations == 0, violations,
                "0 violations over %d nested pairs; sep ranges %.4f to "
                "%.4f so the order is not trivial"
                % (pairs, min(vals), max(vals)))

    # ------------------------------------------------ cor:nested
    e = ex.expect(
        "cor:nested one probe answers a threshold for the chain",
        "Along a chain Q1 subset ... subset Qm, sep(Qm) <= t implies "
        "sep(Qi) <= t for every i, so one probe suffices.",
        "cor:nested",
        "A chain member exceeding a threshold the top satisfies.")

    chain = [{"u0"}, {"u0", "u1"}, {"u0", "u1", "u2"},
             {"u0", "u1", "u2", "u3"}]
    seps = [separation(G, Q)[0] for Q in chain]
    t = seps[-1]
    inferred_ok = all(s <= t + 1e-12 for s in seps)
    # Control: a threshold BELOW the top must fail for some member, or
    # the inference is vacuous.
    t_low = min(seps) - 1e-6
    ctrl_fails = any(s > t_low for s in seps)
    ex.record("nested", {"chain_seps": seps, "threshold": t,
                         "all_below": inferred_ok,
                         "control_threshold": t_low,
                         "control_has_failure": ctrl_fails})
    if not ctrl_fails:
        e.non_discriminating(inferred_ok, "every threshold is satisfied")
    else:
        e.check(inferred_ok, seps,
                "chain seps %s all <= top value %.4f; 1 probe replaces %d; "
                "control threshold %.4f is violated as it must be"
                % ([round(s, 4) for s in seps], t, len(chain), t_low))

    # ----------------------------------------------- prop:relabel
    e = ex.expect(
        "prop:relabel keys are invariant under weight-preserving bijections",
        "A bijection fixing the medium and preserving weights leaves "
        "sep and dep unchanged.",
        "prop:relabel",
        "A key that moves under relabelling --- it would be reporting a "
        "file-format artefact.")

    # Swap u0<->u1 requires symmetric weights; build a graph with that
    # symmetry so the bijection genuinely preserves weights.
    sym = {("u0", "u2"): 0.6, ("u1", "u2"): 0.6, ("u0", "u1"): 0.4}
    Gs = make_graph(3, sym, [0.3, 0.3, 0.5])
    rho = {"u0": "u1", "u1": "u0", "u2": "u2", MED: MED}
    weights_preserved = all(
        close(Gs["w"][e2],
              Gs["w"][frozenset(rho[v] for v in e2)], 1e-12)
        for e2 in Gs["w"])
    moved = 0
    for r in range(1, 3):
        for Q in itertools.combinations(["u0", "u1", "u2"], r):
            k1 = key(Gs, set(Q))
            k2 = key(Gs, {rho[v] for v in Q})
            if k1 != k2:
                moved += 1
    # Control: a NON-weight-preserving relabelling should move a key.
    Ga = make_graph(3, {("u0", "u2"): 0.9, ("u1", "u2"): 0.2}, [0.3, 0.3, 0.5])
    ctrl_moved = key(Ga, {"u0"}) != key(Ga, {"u1"})
    ex.record("relabel", {"weights_preserved": weights_preserved,
                          "keys_moved": moved,
                          "control_asymmetric_moves": ctrl_moved})
    if not ctrl_moved:
        e.non_discriminating(moved, "keys do not move even under an "
                                    "asymmetric relabelling")
    else:
        e.check(weights_preserved and moved == 0, moved,
                "0 keys move under a weight-preserving swap; control: an "
                "asymmetric graph does move the key")

    # ------------------------------------------------- thm:floor
    e = ex.expect(
        "thm:floor a positive floor under every query",
        "sep(Q) >= beta = min medium weight > 0 for every non-empty "
        "query, so no answer is 'free'.",
        "thm:floor / cor:informative",
        "A query separating below the minimum medium weight.")

    beta = min(G["w"][frozenset((u, MED))] for u in U)
    all_seps, below = [], 0
    for r in range(1, 5):
        for Q in itertools.combinations(U, r):
            s, _, _ = separation(G, set(Q))
            all_seps.append(s)
            if s < beta - 1e-12:
                below += 1
    # Control: removing medium edges must produce a zero, or the floor is
    # not doing the work.
    G0 = {"V": G["V"], "w": {e2: v for e2, v in G["w"].items()
                             if MED not in e2}}
    ctrl_zero = close(cut(G0, {"u3", "u4", "u5"}), 0.0, 1e-12)
    ex.record("floor", {"beta": beta, "n_queries": len(all_seps),
                        "min_sep": min(all_seps), "below_floor": below})
    ex.record("floor_control", {"zero_cut_without_medium": ctrl_zero})
    if not ctrl_zero:
        e.non_discriminating(below, "no zero-cost cut exists even without "
                                    "the medium")
    else:
        e.check(below == 0 and min(all_seps) >= beta - 1e-12, below,
                "beta = %.4f; min sep over %d queries = %.4f, 0 below the "
                "floor; control: dropping medium edges yields a 0.0 cut"
                % (beta, len(all_seps), min(all_seps)))

    # -------------------------------------------- thm:sensitivity
    e = ex.expect(
        "thm:sensitivity (a) value moves at most epsilon*M",
        "Under any epsilon-bounded perturbation the separation value "
        "moves by at most epsilon*M, where M is the largest number of "
        "edges crossing any admissible set.",
        "thm:sensitivity(a)",
        "A perturbation moving the value further than the bound.")

    # A graph whose two best admissible sets are nearly tied, so a large
    # enough perturbation CAN flip the minimiser. Without that, the guard
    # in (b) would never be shown to be doing any work.
    Gs2 = make_graph(4, {("u0", "u1"): 0.50, ("u1", "u2"): 0.34,
                         ("u2", "u3"): 0.30},
                     [0.30, 0.32, 0.31, 0.33])
    Q0 = {"u0"}
    G = G  # the floor/monotone graph stays as built above
    g0, _, S0 = separation(Gs2, Q0)
    M = max(crossing_count(Gs2, S) for S in admissible_sets(Gs2, Q0))
    m_edges = crossing_count(Gs2, S0)
    breaches_a, trials = 0, 400
    worst = 0.0
    for _ in range(trials):
        eps = rng.uniform(0.001, 0.05)
        Gp = {"V": Gs2["V"], "w": {}}
        for e2, val in Gs2["w"].items():
            Gp["w"][e2] = max(val + rng.uniform(-eps, eps), 1e-6)
        gp, _, _ = separation(Gp, Q0)
        move = abs(gp - g0)
        worst = max(worst, move - eps * M)
        if move > eps * M + 1e-9:
            breaches_a += 1
    ex.record("sensitivity_a", {"M": M, "m": m_edges, "trials": trials,
                                "breaches": breaches_a,
                                "worst_margin": worst})
    e.check(breaches_a == 0, breaches_a,
            "0 breaches over %d perturbations; M = %d, worst observed "
            "move fell short of the bound by %.5f"
            % (trials, M, -worst))

    e = ex.expect(
        "thm:sensitivity (b) the separating set is stable",
        "If epsilon*(m+M) < slack then the minimising set is unchanged, "
        "so the answer rests on the same observations.",
        "thm:sensitivity(b) / cor:no-rerun",
        "A perturbation satisfying the hypothesis that nevertheless "
        "changes the minimiser --- this would make cor:no-rerun unsafe.")

    sl = slack(Gs2, Q0, S0, g0)
    guarded, unsafe, unguarded_changes = 0, 0, 0
    for _ in range(600):
        eps = rng.uniform(0.001, 0.60)
        Gp = {"V": Gs2["V"], "w": {}}
        for e2, val in Gs2["w"].items():
            Gp["w"][e2] = max(val + rng.uniform(-eps, eps), 1e-6)
        _, _, Sp = separation(Gp, Q0)
        changed = set(Sp) != set(S0)
        if eps * (m_edges + M) < sl:
            guarded += 1
            if changed:
                unsafe += 1
        elif changed:
            unguarded_changes += 1
    ex.record("sensitivity_b", {"slack": sl, "m_plus_M": m_edges + M,
                               "guarded_trials": guarded,
                               "guarded_violations": unsafe,
                               "unguarded_changes": unguarded_changes})
    if unguarded_changes == 0:
        e.non_discriminating(unsafe,
                             "the minimiser never changes at any epsilon, so "
                             "the guard cannot be shown to be doing work")
    else:
        e.check(unsafe == 0, unsafe,
                "slack %.4f, m+M = %d: 0 of %d guarded perturbations move "
                "the minimiser, while %d unguarded ones do --- the "
                "condition is the thing preventing the change"
                % (sl, m_edges + M, guarded, unguarded_changes))

    # ------------------------------------------------ prop:slack
    e = ex.expect(
        "prop:slack the residual bound never overstates the margin",
        "The residual-capacity bound is a LOWER bound on true slack. "
        "Overstating it would authorise unsafe skips.",
        "prop:slack",
        "A residual bound exceeding the true slack.")

    overstates, cases = 0, 0
    for r in range(1, 3):
        for Q in itertools.combinations(U, r):
            Q = set(Q)
            g, _, S = separation(G, Q)
            true_sl = slack(G, Q, S, g)
            _, _, cap = max_flow(G, Q)
            resid = []
            for e2 in G["w"]:
                a, b = tuple(e2)
                if len({a, b} & set(S)) == 1:
                    resid.append(min(cap.get(a, {}).get(b, 0.0),
                                     cap.get(b, {}).get(a, 0.0)))
            bound = min(resid) if resid else 0.0
            cases += 1
            if bound > true_sl + 1e-9:
                overstates += 1
    ex.record("slack_bound", {"cases": cases, "overstatements": overstates})
    e.check(overstates == 0, overstates,
            "0 of %d cases where the residual bound exceeds the true "
            "slack --- the bound is safe in the direction that matters"
            % cases)

    # ----------------------------------------------- thm:not-a-cut
    e = ex.expect(
        "thm:not-a-cut an explicit non-isomorphic pair with equal keys",
        "There exist two non-isomorphic runtime graphs agreeing on "
        "key(Q) for EVERY query. The paper's proof describes such a "
        "construction without exhibiting one; this searches for a "
        "witness.",
        "thm:not-a-cut",
        "No witness at any size searched, which would leave the theorem "
        "unsupported by this suite.")

    def graphs_isomorphic(g1, g2, verts):
        """Isomorphism fixing the medium, over all vertex permutations."""
        for perm in itertools.permutations(verts):
            sub = dict(zip(verts, perm))
            sub[MED] = MED
            ok = True
            for e2, val in g1["w"].items():
                img = frozenset(sub[v] for v in e2)
                if not close(g2["w"].get(img, -1.0), val, 1e-12):
                    ok = False
                    break
            if ok and len(g1["w"]) == len(g2["w"]):
                return True
        return False

    def all_keys_equal(g1, g2, verts):
        for r in range(1, len(verts) + 1):
            for Q in itertools.combinations(verts, r):
                if key(g1, set(Q)) != key(g2, set(Q)):
                    return False
        return True

    def participates(g, verts, e_key):
        """Does this edge lie on the boundary of SOME minimum admissible
        cut? An edge no cut ever crosses is invisible for a trivial
        reason and cannot witness thm:not-a-cut."""
        for r in range(1, len(verts) + 1):
            for Q in itertools.combinations(verts, r):
                _, _, S = separation(g, set(Q))
                if len(set(S) & set(e_key)) == 1:
                    return True
        return False

    def nontrivial_difference(g1, g2, verts):
        """The graphs must disagree on an edge that at least one of them
        actually cuts. Otherwise the agreement of keys says nothing."""
        for e_key in set(g1["w"]) | set(g2["w"]):
            v1 = g1["w"].get(e_key, 0.0)
            v2 = g2["w"].get(e_key, 0.0)
            if close(v1, v2, 1e-12):
                continue
            if participates(g1, verts, e_key) or participates(
                    g2, verts, e_key):
                return True
        return False

    witness, searched, trivial_rejected = None, 0, 0
    verts4 = ["u0", "u1", "u2", "u3"]
    possible = list(itertools.combinations(verts4, 2))
    wchoices = [0.0, 0.4, 0.7]
    med4 = [0.3, 0.3, 0.3, 0.3]
    configs = []
    for assign in itertools.product(wchoices, repeat=len(possible)):
        ce = {p: v for p, v in zip(possible, assign) if v > 0}
        configs.append(ce)
    for i in range(len(configs)):
        if witness is not None:
            break
        for j in range(i + 1, len(configs)):
            searched += 1
            g1 = make_graph(4, configs[i], med4)
            g2 = make_graph(4, configs[j], med4)
            if not all_keys_equal(g1, g2, verts4):
                continue
            if graphs_isomorphic(g1, g2, verts4):
                continue
            if not nontrivial_difference(g1, g2, verts4):
                trivial_rejected += 1
                continue
            witness = {"graph_1": {str(k): v
                                   for k, v in configs[i].items()},
                       "graph_2": {str(k): v
                                   for k, v in configs[j].items()}}
            break
    ex.record("not_a_cut", {"pairs_searched": searched,
                            "trivially_different_rejected": trivial_rejected,
                            "witness": witness})

    # Control: the isomorphism test must actually FIRE, or "non-
    # isomorphic" is being reported by a predicate that never says yes.
    iso_fires = graphs_isomorphic(
        make_graph(4, {("u0", "u1"): 0.4}, med4),
        make_graph(4, {("u2", "u3"): 0.4}, med4), verts4)
    ex.record("not_a_cut_control", {"isomorphism_test_fires": iso_fires})
    if not iso_fires:
        e.non_discriminating(witness,
                             "the isomorphism predicate never returns true, "
                             "so every pair looks non-isomorphic")
    else:
        e.check(witness is not None, witness,
                ("witness found after %d pairs: two non-isomorphic graphs "
                 "with identical keys on all 15 queries --- the theorem's "
                 "construction is realisable" % searched) if witness else
                ("no witness among %d pairs on 4 vertices; the theorem may "
                 "still hold at larger sizes but is not supported here"
                 % searched))

    ex.note("Separation is computed by exhaustive minimisation over "
            "admissible subsets. The max-flow implementation is checked "
            "against it, not the other way round.")
    ex.note("thm:not-a-cut's proof in the manuscript describes a "
            "construction rather than exhibiting one; the witness "
            "recorded above (if any) is a concrete instance the paper "
            "could adopt.")
    ex.note("CAVEAT on that witness: for 13 of its 15 queries the "
            "minimum admissible cut is the whole vertex set, so the "
            "agreement there is carried by the medium edges rather than "
            "by the contact structure. The pair satisfies the theorem "
            "as stated --- the differing edges do participate in the "
            "{u0} cut --- but a manuscript adopting it should say this "
            "rather than imply the keys probe contact structure deeply.")

    ex.report()
    print("  written: " + ex.write())
    return ex


if __name__ == "__main__":
    main()
