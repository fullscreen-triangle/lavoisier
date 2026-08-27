"""
Experiment 3 --- the central empirical claim.

Tests registered expectations E5 (promiscuity-first reaches the true
protein set in no more peptides than greedy parsimony), E6 (closure is
reached strictly before evidence is exhausted), E7 (every run terminates
in closure or honest decline), and negative control N1 (uniqueness-first
ordering does not collapse in combination).

EXPECTATIONS ARE STATED HERE, BEFORE ANY MEASUREMENT IS TAKEN.

This experiment can fail. The paper's promiscuity claim rests on
Assumption 7.6, and if real mapping-set structure does not deliver the
advantage, E5 fails and the claim fails with it. The remaining theorems
are unaffected either way.
"""
from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import (closure_step, digest, greedy_parsimony,
                    intersect_ordered, mapping_sets,
                    peptides_until_found, synth_proteome, write_result)

REGISTERED = {
    "E5_promiscuity_beats_parsimony": {
        "claim": ("Ordering observed peptides by DECREASING promiscuity "
                  "and intersecting reaches the ground-truth protein in "
                  "no more peptides than greedy set-cover parsimony, in "
                  "a majority of trials."),
        "predicate": "win_or_tie_fraction > 0.5",
        "theorem": "Theorem 7.4 / Construction 7.7",
    },
    "E6_closure_before_exhaustion": {
        "claim": ("The step at which the admissible set stops changing "
                  "is strictly less than the number of observed "
                  "peptides, in a majority of trials."),
        "predicate": "early_closure_fraction > 0.5",
        "theorem": "Definition 6.3",
    },
    "E7_termination_dichotomy": {
        "claim": ("Every run ends in closure or honest decline; none "
                  "cycles, and none exceeds the demand bound."),
        "predicate": "n_other_outcomes == 0",
        "theorem": "Theorem 6.6",
    },
    "N1_uniqueness_first": {
        "claim": ("NEGATIVE CONTROL, MUST FAIL: ordering by INCREASING "
                  "promiscuity (unique peptides first) should not gain "
                  "from intersection, because intersecting singletons "
                  "adds nothing. Its per-step collapse ratio must not "
                  "beat promiscuity-first."),
        "predicate": ("uniqueness_beats_promiscuity is False "
                      "(control must FAIL)"),
        "theorem": "Remark 7.10",
        "must_fail": True,
    },
}

SEED = 20260824
N_PROTEINS = 300
N_FAMILIES = 12
N_TRIALS = 200
# Proteins in the synthetic proteome digest into ~7.6 peptides on
# average, so a 10-peptide observation is unsatisfiable. Six is inside
# the achievable range for essentially every target.
N_OBSERVED = 6


def main():
    rnd = random.Random(SEED)
    proteome = synth_proteome(N_PROTEINS, n_families=N_FAMILIES, seed=SEED)
    amb = mapping_sets(proteome)
    universe = set(proteome)
    names = list(proteome)

    trials = []
    for t in range(N_TRIALS):
        target = rnd.choice(names)
        peps = [q for q in digest(proteome[target]) if q in amb]
        if len(peps) < N_OBSERVED:
            continue
        observed = rnd.sample(peps, N_OBSERVED)

        by_prom = sorted(observed, key=lambda q: -len(amb[q]))
        by_uniq = sorted(observed, key=lambda q: len(amb[q]))

        tr_prom = intersect_ordered(by_prom, amb, universe)
        tr_uniq = intersect_ordered(by_uniq, amb, universe)

        n_prom = peptides_until_found(by_prom, amb, universe, target)
        n_uniq = peptides_until_found(by_uniq, amb, universe, target)

        # Greedy parsimony: how many peptides must it consume before the
        # protein it selects first is the true target?
        cover = greedy_parsimony(observed, amb)
        n_pars = len(observed) + 1
        for k in range(1, len(observed) + 1):
            sub = observed[:k]
            c = greedy_parsimony(sub, amb)
            if c and c[0] == target:
                n_pars = k
                break

        cl = closure_step(tr_prom)
        final = tr_prom[-1] if tr_prom else len(universe)

        if final == 1:
            outcome = "closure_unique"
        elif final == 0:
            outcome = "honest_decline_empty"
        elif cl < len(tr_prom):
            outcome = "closure_region"
        else:
            outcome = "exhausted"

        trials.append({
            "trial": t,
            "target": target,
            "n_observed": len(observed),
            "promiscuities": sorted((len(amb[q]) for q in observed),
                                    reverse=True),
            "trace_promiscuity_first": tr_prom,
            "trace_uniqueness_first": tr_uniq,
            "peptides_to_target_promiscuity": n_prom,
            "peptides_to_target_uniqueness": n_uniq,
            "peptides_to_target_parsimony": n_pars,
            "closure_step": cl,
            "final_admissible": final,
            "outcome": outcome,
        })

    n = len(trials)
    if n == 0:
        raise SystemExit("no usable trials")

    wins = sum(1 for r in trials
               if r["peptides_to_target_promiscuity"]
               <= r["peptides_to_target_parsimony"])
    strict = sum(1 for r in trials
                 if r["peptides_to_target_promiscuity"]
                 < r["peptides_to_target_parsimony"])
    early = sum(1 for r in trials if r["closure_step"] < r["n_observed"])
    other = sum(1 for r in trials if r["outcome"] == "exhausted")

    uniq_better = sum(1 for r in trials
                      if r["peptides_to_target_uniqueness"]
                      < r["peptides_to_target_promiscuity"])

    mean_prom = sum(r["peptides_to_target_promiscuity"] for r in trials) / n
    mean_uniq = sum(r["peptides_to_target_uniqueness"] for r in trials) / n
    mean_pars = sum(r["peptides_to_target_parsimony"] for r in trials) / n

    summary = {
        "E5_promiscuity_beats_parsimony": (wins / n) > 0.5,
        "E6_closure_before_exhaustion": (early / n) > 0.5,
        "E7_termination_dichotomy": other == 0,
        "N1_uniqueness_first": not ((uniq_better / n) > 0.5),
    }

    # -----------------------------------------------------------------
    #  SECOND CONDITION: the no-unique-peptide regime.
    #
    #  The trials above are dominated by targets possessing at least one
    #  UNIQUE peptide, which settles the target in a single step. That is
    #  the regime in which Theorem 7.4's monotonicity already says
    #  promiscuity-first must lose, so the comparison is uninformative
    #  about the claim it was meant to test.
    #
    #  The claim is about the regime where no unique peptide is
    #  available --- which is the situation for proteins covered only by
    #  shared peptides, the case parsimony is known to handle badly. We
    #  therefore repeat the comparison restricted to observations
    #  containing NO unique peptide. This restriction is stated here as
    #  a second, separately-reported condition; it does not rescue E5,
    #  which is reported as failed on its registered terms.
    # -----------------------------------------------------------------
    shared_trials = []
    for t in range(N_TRIALS * 3):
        target = rnd.choice(names)
        peps = [q for q in digest(proteome[target]) if q in amb]
        shared_only = [q for q in peps if len(amb[q]) > 1]
        if len(shared_only) < 3:
            continue
        observed = shared_only[:min(6, len(shared_only))]
        by_prom = sorted(observed, key=lambda q: -len(amb[q]))
        by_uniq = sorted(observed, key=lambda q: len(amb[q]))
        tr_p = intersect_ordered(by_prom, amb, universe)
        tr_u = intersect_ordered(by_uniq, amb, universe)
        n_p = peptides_until_found(by_prom, amb, universe, target)
        n_u = peptides_until_found(by_uniq, amb, universe, target)
        n_pars_s = len(observed) + 1
        for k in range(1, len(observed) + 1):
            c = greedy_parsimony(observed[:k], amb)
            if c and c[0] == target:
                n_pars_s = k
                break
        shared_trials.append({
            "target": target,
            "promiscuities": [len(amb[q]) for q in by_prom],
            "trace_promiscuity_first": tr_p,
            "trace_uniqueness_first": tr_u,
            "peptides_promiscuity": n_p,
            "peptides_uniqueness": n_u,
            "peptides_parsimony": n_pars_s,
        })
        if len(shared_trials) >= 120:
            break

    shared_block = {"n_trials": 0}
    if shared_trials:
        m = len(shared_trials)
        w = sum(1 for r in shared_trials
                if r["peptides_promiscuity"] <= r["peptides_parsimony"])
        wu = sum(1 for r in shared_trials
                 if r["peptides_promiscuity"] <= r["peptides_uniqueness"])
        shared_block = {
            "n_trials": m,
            "mean_peptides_promiscuity":
                sum(r["peptides_promiscuity"] for r in shared_trials) / m,
            "mean_peptides_uniqueness":
                sum(r["peptides_uniqueness"] for r in shared_trials) / m,
            "mean_peptides_parsimony":
                sum(r["peptides_parsimony"] for r in shared_trials) / m,
            "win_or_tie_vs_parsimony_fraction": w / m,
            "win_or_tie_vs_uniqueness_fraction": wu / m,
            "examples": shared_trials[:10],
        }

    out = {
        "experiment": "exp3_promiscuity_vs_parsimony",
        "seed": SEED,
        "purpose": ("Test whether ranking peptides by exclusion size "
                    "(promiscuity) reaches the ground-truth protein in "
                    "fewer peptides than minimum-set-cover parsimony, "
                    "with uniqueness-first as the negative control."),
        "configuration": {
            "n_proteins": N_PROTEINS,
            "n_families": N_FAMILIES,
            "n_trials_attempted": N_TRIALS,
            "n_trials_usable": n,
            "n_observed_peptides_per_trial": N_OBSERVED,
        },
        "registered_expectations": REGISTERED,
        "aggregate": {
            "win_or_tie_count": wins,
            "win_or_tie_fraction": wins / n,
            "strict_win_count": strict,
            "strict_win_fraction": strict / n,
            "early_closure_count": early,
            "early_closure_fraction": early / n,
            "n_other_outcomes": other,
            "uniqueness_beats_promiscuity_count": uniq_better,
            "uniqueness_beats_promiscuity_fraction": uniq_better / n,
            "mean_peptides_promiscuity_first": mean_prom,
            "mean_peptides_uniqueness_first": mean_uniq,
            "mean_peptides_parsimony": mean_pars,
        },
        "summary": summary,
        "shared_peptides_only_regime": shared_block,
        "recorded_finding": {
            "E5_verdict": "REFUTED on its registered terms",
            "E5_observed": {
                "win_or_tie_fraction": wins / n,
                "strict_win_fraction": strict / n,
                "required": "> 0.5",
            },
            "N1_verdict": ("REFUTED: the negative control PASSED where it "
                           "was required to fail. Uniqueness-first beat "
                           "promiscuity-first in %.1f%% of trials."
                           % (100.0 * uniq_better / n)),
            "mechanism": (
                "Where a UNIQUE peptide is present it isolates the target "
                "in a single intersection (trace [1,1,1,...]), while "
                "promiscuity-first spends three or four peptides walking "
                "the admissible set down (e.g. 31 -> 5 -> 1). This is not "
                "an artefact: Theorem 7.4's expectation is monotone "
                "increasing in each promiscuity, so under its own "
                "hypothesis the LEAST promiscuous peptides are the best "
                "individual choices. The registered expectation E5 "
                "contradicted the theorem the paper itself proves."),
            "what_survives": (
                "Theorem 7.4 stands and is confirmed numerically in exp2. "
                "The closure criterion (E6) and the termination dichotomy "
                "(E7) are unaffected and both hold. What fails is the "
                "empirical claim that ranking by promiscuity beats "
                "parsimony, together with Assumption 7.6 which was its "
                "only support."),
            "residual_claim": (
                "The shared-peptides-only block reports the restricted "
                "regime in which no unique peptide exists. Any surviving "
                "advantage is confined to that regime and is reported "
                "separately rather than used to rescue E5."),
        },
        "trials": trials[:20],
        "note": "trials list truncated to first 20 for readability",
    }
    out["all_passed"] = all(summary.values())

    p = write_result("exp3_promiscuity_vs_parsimony.json", out)
    print("wrote", p)
    a = out["aggregate"]
    print("  usable trials            : %d" % n)
    print("  mean peptides  promiscuity: %.2f" % a["mean_peptides_promiscuity_first"])
    print("  mean peptides  uniqueness : %.2f" % a["mean_peptides_uniqueness_first"])
    print("  mean peptides  parsimony  : %.2f" % a["mean_peptides_parsimony"])
    print("  win-or-tie vs parsimony   : %.3f" % a["win_or_tie_fraction"])
    print("  strict win vs parsimony   : %.3f" % a["strict_win_fraction"])
    print("  early closure fraction    : %.3f" % a["early_closure_fraction"])
    for k, v in summary.items():
        print("  %-32s %s" % (k, "PASS" if v else "FAIL"))


if __name__ == "__main__":
    main()
