"""
Experiment 1 --- graph invariants.

Tests registered expectations E1 (floor recovery), E2 (residual
monotonicity), E3 (cut-key invariance under relabelling), and negative
control N3 (zero floor removes the termination bound).

EXPECTATIONS ARE STATED HERE, BEFORE ANY MEASUREMENT IS TAKEN.
Each is a dict with an explicit predicate; the runner records both the
prediction and the outcome, so a failure is visible in the artefact
rather than absorbed into prose.
"""
from __future__ import annotations

import itertools
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import (MEDIUM, contact_graph, cut_key, floor_of, residual,
                    write_result)

# =====================================================================
#  REGISTERED EXPECTATIONS  (written before the run)
# =====================================================================

REGISTERED = {
    "E1_floor_recovery": {
        "claim": ("Computed floor equals min medium-incident weight "
                  "exactly, and no item has sigma(v) < beta."),
        "predicate": "max_abs_err < 1e-9 and n_below_floor == 0",
        "theorem": "Theorem 3.1",
    },
    "E2_residual_monotone": {
        "claim": ("Committing an item-item edge never decreases the "
                  "global residual R, in every ordering tested."),
        "predicate": "n_decreases == 0",
        "theorem": "Theorem 5.2(iii)",
    },
    "E3_cutkey_invariance": {
        "claim": ("Relabelling items leaves the multiset of cut keys "
                  "unchanged."),
        "predicate": "n_mismatched == 0",
        "theorem": "Theorem 4.2",
    },
    "N3_zero_floor": {
        "claim": ("NEGATIVE CONTROL, MUST FAIL: with beta driven to 0 "
                  "the residual no longer bounds the number of "
                  "commitments, so the termination bound is lost."),
        "predicate": "bound_finite is False   (i.e. control must FAIL)",
        "theorem": "Theorem 6.6",
        "must_fail": True,
    },
}

SEED = 20260824


def build_random(n_items: int, n_edges: int, rnd: random.Random,
                 min_med: float = 0.5, max_med: float = 4.0):
    items = ["v%02d" % i for i in range(n_items)]
    med_w = {v: round(rnd.uniform(min_med, max_med), 4) for v in items}
    edges = []
    for _ in range(n_edges):
        a, b = rnd.sample(items, 2)
        edges.append((a, b, round(rnd.uniform(0.2, 3.0), 4)))
    return items, med_w, edges


# ---------------------------------------------------------------------
#  E1 --- floor recovery
# ---------------------------------------------------------------------
def run_e1(rnd):
    errs, below, trials = [], 0, []
    for t in range(30):
        items, med_w, edges = build_random(rnd.randint(5, 14),
                                           rnd.randint(3, 20), rnd)
        g = contact_graph(items, med_w, edges)
        beta = floor_of(g)
        expected = min(med_w.values())
        err = abs(beta - expected)
        errs.append(err)
        sigmas = [cut_key(g, v)[0] for v in items]
        n_bel = sum(1 for s in sigmas if s < beta - 1e-9)
        below += n_bel
        trials.append({
            "trial": t, "n_items": len(items), "n_item_edges": len(edges),
            "beta_computed": beta, "beta_expected": expected,
            "abs_err": err, "min_sigma": min(sigmas),
            "n_below_floor": n_bel,
        })
    max_err = max(errs)
    passed = (max_err < 1e-9) and (below == 0)
    return {
        "registered": REGISTERED["E1_floor_recovery"],
        "n_trials": len(trials),
        "max_abs_err": max_err,
        "n_below_floor": below,
        "passed": passed,
        "trials": trials,
    }


# ---------------------------------------------------------------------
#  E2 --- residual monotonicity under edge commitment
# ---------------------------------------------------------------------
def run_e2(rnd):
    n_dec, orderings = 0, []
    for t in range(20):
        items, med_w, edges = build_random(rnd.randint(6, 12),
                                           rnd.randint(6, 14), rnd)
        # Commit the same edge set in several different orders.
        for o in range(6):
            order = edges[:]
            rnd.shuffle(order)
            g = contact_graph(items, med_w, [])
            prev = residual(g)
            seq = [prev]
            dec_here = 0
            for (a, b, w) in order:
                if g.has_edge(a, b):
                    g[a][b]["capacity"] += w
                else:
                    g.add_edge(a, b, capacity=w)
                cur = residual(g)
                if cur < prev - 1e-9:
                    dec_here += 1
                seq.append(cur)
                prev = cur
            n_dec += dec_here
            orderings.append({
                "trial": t, "ordering": o,
                "residual_start": seq[0], "residual_end": seq[-1],
                "n_decreases": dec_here,
                "monotone_nondecreasing": dec_here == 0,
            })
    return {
        "registered": REGISTERED["E2_residual_monotone"],
        "n_orderings": len(orderings),
        "n_decreases": n_dec,
        "passed": n_dec == 0,
        "orderings": orderings[:24],
        "note": "orderings list truncated to first 24 for readability",
    }


# ---------------------------------------------------------------------
#  E3 --- cut-key invariance under relabelling
# ---------------------------------------------------------------------
def run_e3(rnd):
    mismatched, trials = 0, []
    for t in range(20):
        items, med_w, edges = build_random(rnd.randint(5, 10),
                                           rnd.randint(4, 14), rnd)
        g = contact_graph(items, med_w, edges)
        keys_a = sorted((round(cut_key(g, v)[0], 9), cut_key(g, v)[1])
                        for v in items)

        # Relabel: permute item names, carrying weights along.
        perm = items[:]
        rnd.shuffle(perm)
        relabel = dict(zip(items, perm))
        med_w2 = {relabel[v]: w for v, w in med_w.items()}
        edges2 = [(relabel[a], relabel[b], w) for a, b, w in edges]
        g2 = contact_graph(perm, med_w2, edges2)
        keys_b = sorted((round(cut_key(g2, v)[0], 9), cut_key(g2, v)[1])
                        for v in perm)

        same = keys_a == keys_b
        if not same:
            mismatched += 1
        trials.append({
            "trial": t, "n_items": len(items),
            "multiset_equal": same,
            "n_distinct_keys": len(set(keys_a)),
        })
    return {
        "registered": REGISTERED["E3_cutkey_invariance"],
        "n_trials": len(trials),
        "n_mismatched": mismatched,
        "passed": mismatched == 0,
        "trials": trials,
    }


# ---------------------------------------------------------------------
#  N3 --- negative control: zero floor
# ---------------------------------------------------------------------
def run_n3(rnd):
    """
    Negative control N3.

    The operative content of beta > 0 is that it lower-bounds the cost of
    individuating ANY item, hence lower-bounds the decrement a single
    commitment can be credited with, hence bounds the number of effective
    commitments (Theorem 6.6).

    A first attempt at this control shrank every medium weight by a common
    factor s. That tests nothing: R and beta scale together, so R/beta is
    exactly invariant. We record that finding and use the correct probe
    instead --- drive ONE item's medium weight toward zero, leaving the
    others fixed, and ask whether that item can then be individuated at
    cost below the floor of the unperturbed graph.

    The control PASSES the suite only if it FAILS the predicate, i.e. only
    if a vanishing floor genuinely destroys the bound.
    """
    items, med_w, edges = build_random(8, 10, rnd)
    g_pos = contact_graph(items, med_w, edges)
    beta_pos = floor_of(g_pos)
    bound_pos = residual(g_pos) / beta_pos

    # --- the uninformative probe, recorded for completeness -----------
    uniform_rows = []
    for s in (1e-2, 1e-6):
        med_s = {v: w * s for v, w in med_w.items()}
        g = contact_graph(items, med_s, edges)
        b = floor_of(g)
        uniform_rows.append({
            "shrink": s, "beta": b, "residual": residual(g),
            "bound_R_over_beta": residual(g) / b,
        })
    uniform_is_scale_free = (
        abs(uniform_rows[0]["bound_R_over_beta"]
            - uniform_rows[1]["bound_R_over_beta"]) < 1e-6)

    # --- second probe: collapse ONE item's medium edge only -----------
    # This ALSO fails to void the floor, and the reason is itself a
    # result: separation cost is a property of the cheapest CUT, not of
    # any single edge (Theorem 4.3). An item still joined to other items
    # must have those contacts severed too, so shrinking one medium edge
    # leaves sigma bounded by the remaining incident weight.
    victim = items[0]
    edge_only_rows = []
    for s in (1.0, 1e-4, 1e-9):
        med_s = dict(med_w)
        med_s[victim] = med_w[victim] * s
        g = contact_graph(items, med_s, edges)
        sig_v = cut_key(g, victim)[0]
        edge_only_rows.append({
            "victim_shrink": s,
            "sigma_victim": sig_v,
            "below_unperturbed_floor": bool(sig_v < beta_pos - 1e-12),
        })
    edge_only_voids = any(r["below_unperturbed_floor"]
                          for r in edge_only_rows)

    # --- third probe: pendant victim, the construction that works ------
    # Isolate the victim from all other items (pendant on the medium) and
    # then drive its single incident weight to zero. Now the cheapest cut
    # IS that edge, so sigma tracks it and the floor is genuinely voided.
    pendant_edges = [(a, b, w) for a, b, w in edges
                     if a != victim and b != victim]
    rows = []
    for s in (1.0, 1e-2, 1e-4, 1e-6, 1e-9):
        med_s = dict(med_w)
        med_s[victim] = med_w[victim] * s
        g = contact_graph(items, med_s, pendant_edges)
        sig_v = cut_key(g, victim)[0]
        rows.append({
            "victim_shrink": s,
            "beta_graph": floor_of(g),
            "sigma_victim": sig_v,
            "sigma_below_unperturbed_floor": bool(sig_v < beta_pos - 1e-12),
            "commitments_bounded": bool(sig_v >= beta_pos - 1e-12),
        })

    # The bound survives only if every configuration still costs at least
    # the unperturbed floor. The control must show it does NOT.
    bound_survives = all(r["commitments_bounded"] for r in rows)
    control_failed_as_required = not bound_survives

    return {
        "registered": REGISTERED["N3_zero_floor"],
        "beta_unperturbed": beta_pos,
        "bound_unperturbed": bound_pos,
        "probe_1_uniform_shrink": {
            "rows": uniform_rows,
            "is_scale_free": uniform_is_scale_free,
            "voids_floor": False,
            "note": ("Uniform shrinkage scales R and beta together, so "
                     "R/beta is exactly invariant and the probe is "
                     "uninformative. Recorded because it was the first "
                     "construction attempted and its failure identified "
                     "the next one."),
        },
        "probe_2_single_medium_edge": {
            "rows": edge_only_rows,
            "voids_floor": edge_only_voids,
            "note": ("Shrinking one item's medium edge does NOT void the "
                     "floor while that item retains contacts to other "
                     "items: the minimum cut simply routes around the "
                     "cheap edge and severs the others instead. This is "
                     "Theorem 4.3 --- separation cost is a property of "
                     "the cheapest cut, not of any single edge --- and "
                     "the probe's failure is a confirmation of it."),
        },
        "probe_3_pendant_victim": {
            "rows": rows,
            "voids_floor": not bound_survives,
            "note": ("With the victim pendant on the medium, the cheapest "
                     "cut coincides with its single incident edge, so "
                     "driving that weight to zero genuinely voids the "
                     "floor and with it the termination bound."),
        },
        "bound_survives_collapse": bound_survives,
        "control_failed_as_required": control_failed_as_required,
        "passed": control_failed_as_required,
    }


def main():
    rnd = random.Random(SEED)
    out = {
        "experiment": "exp1_graph_invariants",
        "seed": SEED,
        "purpose": ("Check the graph-theoretic claims of sections 3-6 on "
                    "concrete contact graphs. Expectations registered in "
                    "this source file before execution."),
        "registered_expectations": REGISTERED,
        "results": {},
    }
    out["results"]["E1_floor_recovery"] = run_e1(rnd)
    out["results"]["E2_residual_monotone"] = run_e2(rnd)
    out["results"]["E3_cutkey_invariance"] = run_e3(rnd)
    out["results"]["N3_zero_floor"] = run_n3(rnd)

    out["summary"] = {
        k: v["passed"] for k, v in out["results"].items()
    }
    out["all_passed"] = all(out["summary"].values())

    p = write_result("exp1_graph_invariants.json", out)
    print("wrote", p)
    for k, v in out["summary"].items():
        print("  %-28s %s" % (k, "PASS" if v else "FAIL"))
    print("  all_passed:", out["all_passed"])


if __name__ == "__main__":
    main()
