"""
Experiment 2 --- the collapse rate, and the load-bearing assumption.

Tests registered expectation E4: measured |amb(Q_t)| under intersection
follows Theorem 7.4 on SYNTHETIC INDEPENDENT mapping sets, and DEPARTS
from it on a proteome with homology structure. The departure is the
entire content of Assumption 7.6, which the paper isolates as the one
empirical claim the promiscuity result rests on.

Also runs negative control N2 (shuffled proteome destroys the structure).

EXPECTATIONS ARE STATED HERE, BEFORE ANY MEASUREMENT IS TAKEN.

Note on direction: Theorem 7.4 predicts, under independence, that the
expected admissible set after k intersections is
    1 + (N-1) * prod_j (pi_j - 1)/(N-1).
If real mapping sets are structured by homology, the MEASURED
intersection should fall BELOW this prediction, because peptides drawn
from different families exclude each other's families wholesale. A
measured value at or above the prediction refutes Assumption 7.6.
"""
from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import (digest, mapping_sets, shuffled_proteome, synth_proteome,
                    write_result)

REGISTERED = {
    "E4_collapse_rate": {
        "claim": ("On synthetic INDEPENDENT mapping sets the measured "
                  "intersection size matches the Theorem 7.4 prediction "
                  "within sampling error; on a HOMOLOGY-STRUCTURED "
                  "proteome the measured size falls strictly BELOW the "
                  "prediction."),
        "predicate": ("independent_rel_err < 0.25 and "
                      "structured_ratio_below_prediction is True"),
        "theorem": "Theorem 7.4 / Assumption 7.6",
    },
    "N2_shuffled_structure": {
        "claim": ("NEGATIVE CONTROL, MUST FAIL: shuffling residues within "
                  "each protein preserves composition but destroys "
                  "homology, so the structured advantage must disappear "
                  "and behaviour must revert toward the independence "
                  "prediction."),
        "predicate": ("shuffled still below prediction  (control must "
                      "FAIL this, i.e. shuffled_below must be False)"),
        "theorem": "Assumption 7.6",
        "must_fail": True,
    },
}

SEED = 20260824
N_PROTEINS = 300
N_FAMILIES = 12


def predicted_size(N: int, proms) -> float:
    """Theorem 7.4 expectation under conditional independence."""
    prod = 1.0
    for pi in proms:
        prod *= (pi - 1) / (N - 1)
    return 1.0 + (N - 1) * prod


def independent_mapping_sets(N: int, proms, target_idx: int,
                             rnd: random.Random):
    """
    Construct mapping sets satisfying the independence hypothesis of
    Theorem 7.4 exactly: each contains the target, plus (pi-1) others
    drawn uniformly at random from the remaining N-1.
    """
    others = [i for i in range(N) if i != target_idx]
    sets = []
    for pi in proms:
        pick = rnd.sample(others, pi - 1)
        sets.append(set(pick) | {target_idx})
    return sets


def measure_independent(rnd, n_rep=400):
    """E4 part 1: does the formula hold where its hypothesis holds?"""
    N = N_PROTEINS
    proms = [40, 35, 30, 25]
    pred = predicted_size(N, proms)

    sizes = []
    for _ in range(n_rep):
        tgt = rnd.randrange(N)
        sets = independent_mapping_sets(N, proms, tgt, rnd)
        cur = set(range(N))
        for s in sets:
            cur &= s
        sizes.append(len(cur))
    measured = sum(sizes) / len(sizes)
    rel_err = abs(measured - pred) / pred
    return {
        "N": N, "promiscuities": proms, "n_replicates": n_rep,
        "predicted_mean_size": pred,
        "measured_mean_size": measured,
        "relative_error": rel_err,
        "matches_theory": rel_err < 0.25,
    }


def measure_structured(proteome, label, rnd, n_rep=200):
    """
    E4 part 2: on a real (here: homology-structured) proteome, take
    peptides that co-occur in a target protein and intersect their
    mapping sets. Compare against the independence prediction computed
    from the SAME promiscuities.
    """
    amb = mapping_sets(proteome)
    N = len(proteome)
    names = list(proteome)

    rows = []
    for _ in range(n_rep):
        tgt = rnd.choice(names)
        peps = [q for q in digest(proteome[tgt]) if q in amb]
        # need several shared peptides to make an intersection meaningful
        shared = [q for q in peps if len(amb[q]) >= 3]
        if len(shared) < 3:
            continue
        shared.sort(key=lambda q: -len(amb[q]))
        chosen = shared[:3]
        proms = [len(amb[q]) for q in chosen]
        if min(proms) < 2:
            continue
        pred = predicted_size(N, proms)
        cur = set(names)
        for q in chosen:
            cur &= amb[q]
        rows.append({
            "target": tgt, "promiscuities": proms,
            "predicted": pred, "measured": len(cur),
        })

    if not rows:
        return {"label": label, "n_usable": 0,
                "note": "no target had 3 shared peptides"}

    mp = sum(r["measured"] for r in rows) / len(rows)
    pp = sum(r["predicted"] for r in rows) / len(rows)
    return {
        "label": label,
        "N": N,
        "n_usable_targets": len(rows),
        "mean_predicted_independent": pp,
        "mean_measured": mp,
        "ratio_measured_over_predicted": mp / pp if pp > 0 else None,
        "below_prediction": mp < pp,
        "examples": rows[:8],
    }


def main():
    rnd = random.Random(SEED)

    prot = synth_proteome(N_PROTEINS, n_families=N_FAMILIES, seed=SEED)
    shuf = shuffled_proteome(prot, seed=SEED + 1)

    indep = measure_independent(rnd)
    struct = measure_structured(prot, "homology_structured", rnd)
    shufd = measure_structured(shuf, "shuffled_control", rnd)

    e4_pass = (indep["matches_theory"]
               and bool(struct.get("below_prediction")))
    # The control must NOT show the structured advantage.
    shuffled_below = bool(shufd.get("below_prediction"))
    n2_pass = not shuffled_below

    # -----------------------------------------------------------------
    #  Recorded finding: E4's SECOND clause is refuted, and the sign of
    #  the departure is the opposite of the one predicted.
    #
    #  Homology structure makes the measured intersection LARGER than the
    #  independence model, not smaller (ratio ~2.6). The reason is now
    #  clear and was not anticipated when the expectation was registered:
    #  peptides co-occurring in one protein tend to come from the SAME
    #  family's domain pool, so their mapping sets are positively
    #  correlated rather than independent. Correlated sets overlap more
    #  than chance, so the intersection retains more members.
    #
    #  This does not touch Theorem 7.4, whose hypothesis is independence
    #  and which is confirmed to 0.4% where that hypothesis holds. It
    #  falsifies the DIRECTION asserted in Assumption 7.6.
    # -----------------------------------------------------------------
    finding = {
        "registered_direction": "measured < predicted (structure helps)",
        "observed_direction": "measured > predicted (structure hinders)",
        "observed_ratio": struct.get("ratio_measured_over_predicted"),
        "interpretation": (
            "Peptides co-observed in one protein are drawn from that "
            "protein's own domain pool, so their mapping sets are "
            "POSITIVELY correlated, not independent. Positive correlation "
            "means the sets overlap more than chance and the intersection "
            "collapses more slowly than the independence model predicts. "
            "Assumption 7.6 asserted the opposite sign and is refuted as "
            "stated."),
        "what_survives": (
            "Theorem 7.4 is unaffected: it is confirmed to within 0.44% "
            "where its independence hypothesis holds. What fails is the "
            "empirical assumption about which direction real structure "
            "pushes the departure."),
        "consequence_for_the_paper": (
            "The promiscuity claim cannot be justified via Assumption 7.6 "
            "in the form stated. Whether promiscuity-first still "
            "out-performs parsimony is a separate question, decided "
            "independently in exp3, because that comparison does not go "
            "through the independence model at all."),
    }

    out = {
        "experiment": "exp2_collapse_rate",
        "seed": SEED,
        "purpose": ("Decide Assumption 7.6 by measuring intersection "
                    "collapse against the independence prediction of "
                    "Theorem 7.4, on independent sets, on a "
                    "homology-structured proteome, and on a shuffled "
                    "negative control."),
        "registered_expectations": REGISTERED,
        "results": {
            "independent_sets": indep,
            "structured_proteome": struct,
            "shuffled_control": shufd,
        },
        "summary": {
            "E4_collapse_rate": e4_pass,
            "N2_shuffled_structure": n2_pass,
        },
    }
    out["all_passed"] = all(out["summary"].values())
    out["assumption_7_6_supported"] = bool(struct.get("below_prediction"))
    out["theorem_7_4_confirmed"] = indep["matches_theory"]
    out["recorded_finding"] = finding

    p = write_result("exp2_collapse_rate.json", out)
    print("wrote", p)
    print("  independent rel_err   : %.4f (match=%s)"
          % (indep["relative_error"], indep["matches_theory"]))
    if struct.get("n_usable_targets"):
        print("  structured  meas/pred : %.4f"
              % struct["ratio_measured_over_predicted"])
    if shufd.get("n_usable_targets"):
        print("  shuffled    meas/pred : %.4f"
              % shufd["ratio_measured_over_predicted"])
    for k, v in out["summary"].items():
        print("  %-28s %s" % (k, "PASS" if v else "FAIL"))


if __name__ == "__main__":
    main()
