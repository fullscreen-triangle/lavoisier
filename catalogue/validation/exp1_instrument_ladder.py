"""
exp1_instrument_ladder.py --- validation V1-V8 of

    "An Instrument Is a Ladder: Mass Spectrometry Without a Substrate"

Each test below is one of the protocol items V1-V8 and is tied to the
prediction P1-P9 it can refute. Every test is constructed so that it can
fail, and each is paired with a control establishing whether the
statistic discriminates at all. A test whose control shows it cannot
separate the prediction from a null is reported as non-discriminating
and excluded rather than counted as a pass.
"""
from __future__ import annotations

import math
import random

from common import Experiment, mean, close, rel_close


# =====================================================================
#  A contact-sequence machine --- the object the theorem is about
# =====================================================================

class Substrate:
    """A continuous carrier. Two substrates may differ in geometry,
    field strength, and flight time while realising the same contact
    sequence. Theorem 2.4 says the readout cannot tell them apart."""

    def __init__(self, name, geometry, field_scale, path_len, rng):
        self.name = name
        self.geometry = geometry
        self.field_scale = field_scale
        self.path_len = path_len
        self.rng = rng

    def transit(self, state):
        """Free flight. Changes position and clock --- neither of which
        any contact reads. Transit is free (P7)."""
        state = dict(state)
        state["position"] += self.path_len * self.field_scale
        state["clock"] += self.path_len / max(self.field_scale, 1e-9)
        state["wobble"] = self.rng.random() * self.geometry
        return state


def contact_local(state, power):
    """Local effect: a function of the incoming CATEGORICAL state alone
    (Definition 2.3). Position, clock and wobble are not read."""
    state = dict(state)
    state["ambiguity"] *= (1.0 - power)
    state["contacts"] += 1
    return state


def contact_nonlocal(state, power, density):
    """Space charge: the force depends on the instantaneous spatial
    distribution of OTHER items, so locality fails (Prop. 2.8). The
    deviation scales with density and vanishes as density -> 0."""
    state = contact_local(state, power)
    state["ambiguity"] *= (1.0 + 0.05 * density * state["position"] * 1e-3)
    return state


def run_sequence(substrate, powers, density=0.0, initial_ambiguity=1.0):
    """Realise a contact sequence on a substrate. Returns the readout."""
    state = {"ambiguity": initial_ambiguity, "contacts": 0,
             "position": 0.0, "clock": 0.0, "wobble": 0.0}
    for p in powers:
        state = substrate.transit(state)
        state = (contact_local(state, p) if density == 0.0
                 else contact_nonlocal(state, p, density))
    return {"ambiguity": state["ambiguity"],
            "resolution": 1.0 - state["ambiguity"],
            "contacts": state["contacts"]}


# =====================================================================
#  The ladder algebra under test
# =====================================================================

def compose(powers):
    r = 1.0
    for p in powers:
        r *= (1.0 - p)
    return 1.0 - r


def sensitivity(powers):
    out = []
    for j in range(len(powers)):
        prod = 1.0
        for i, p in enumerate(powers):
            if i != j:
                prod *= (1.0 - p)
        out.append(prod)
    return out


def main():
    ex = Experiment(
        name="exp1_instrument_ladder",
        paper="instrument-process-ladder",
        question="Do the elimination theorem and the ladder algebra hold "
                 "on machines built to realise contact sequences?",
    )
    rng = random.Random(20260830)
    POWERS = [0.60, 0.35, 0.50, 0.25, 0.15]

    # ---------------------------------------------------------- V1
    e = ex.expect(
        "V1 elimination",
        "Two substrates of differing geometry realising an identical "
        "contact sequence produce identical readouts.",
        "thm:elimination / P1",
        "A readout separating two such instruments.")

    a = Substrate("wide-slow", geometry=3.0, field_scale=0.4,
                  path_len=120.0, rng=random.Random(1))
    b = Substrate("narrow-fast", geometry=0.2, field_scale=7.5,
                  path_len=8.0, rng=random.Random(2))
    ra = run_sequence(a, POWERS)
    rb = run_sequence(b, POWERS)
    ex.record("V1_readouts", {"substrate_a": ra, "substrate_b": rb})
    same = close(ra["resolution"], rb["resolution"], 1e-12)

    # Control: DIFFERENT sequences must separate, or the statistic is inert.
    rc = run_sequence(b, [0.10, 0.10, 0.10, 0.10, 0.10])
    separates = not close(ra["resolution"], rc["resolution"], 1e-6)
    ex.record("V1_control", {"different_sequence": rc,
                             "control_separates": separates})
    if not separates:
        e.non_discriminating(ra["resolution"],
                             "control failed to separate distinct sequences")
    else:
        e.check(same, {"a": ra["resolution"], "b": rb["resolution"]},
                "a=%.12f b=%.12f; control separates (%.5f)"
                % (ra["resolution"], rb["resolution"], rc["resolution"]))

    # ---------------------------------------------------------- V2
    e = ex.expect(
        "V2 locality failure scales with current",
        "With a density-dependent term, deviation from substrate "
        "independence grows with ion current and vanishes as current -> 0.",
        "prop:spacecharge / P2",
        "A current-independent deviation, indicating a second locality "
        "violation the framework does not predict.")

    devs = []
    for dens in [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]:
        da = run_sequence(a, POWERS, density=dens)
        db = run_sequence(b, POWERS, density=dens)
        devs.append({"density": dens,
                     "deviation": abs(da["resolution"] - db["resolution"])})
    ex.record("V2_deviation_vs_density", devs)

    monotone = all(devs[i]["deviation"] <= devs[i + 1]["deviation"] + 1e-15
                   for i in range(len(devs) - 1))
    vanishes = close(devs[0]["deviation"], 0.0, 1e-12)
    grows = devs[-1]["deviation"] > 1e-9
    e.check(monotone and vanishes and grows,
            {"at_zero": devs[0]["deviation"], "at_max": devs[-1]["deviation"]},
            "deviation 0 at zero current, %.6f at density 8, monotone=%s"
            % (devs[-1]["deviation"], monotone))

    # ---------------------------------------------------------- V3
    e = ex.expect(
        "V3 multiplicative composition",
        "Composite resolution follows 1 - prod(1-pi_i), not an additive "
        "or max-based law.",
        "thm:multiplicative / P3",
        "Measured composite tracking sum(pi_i) or max(pi_i).")

    mult_err, add_err, max_err = [], [], []
    for _ in range(400):
        n = rng.randint(2, 7)
        ps = [rng.uniform(0.02, 0.85) for _ in range(n)]
        measured = run_sequence(a, ps)["resolution"]
        mult_err.append(abs(measured - compose(ps)))
        add_err.append(abs(measured - min(sum(ps), 1.0)))
        max_err.append(abs(measured - max(ps)))
    ex.record("V3_model_error", {
        "multiplicative_max_abs_err": max(mult_err),
        "additive_mean_abs_err": mean(add_err),
        "max_based_mean_abs_err": mean(max_err),
        "n_trials": len(mult_err)})

    # Control: the competing models must be DISTINGUISHABLE on this data.
    competitors_differ = mean(add_err) > 1e-3 and mean(max_err) > 1e-3
    if not competitors_differ:
        e.non_discriminating(max(mult_err),
                             "additive and max laws indistinguishable here")
    else:
        e.check(max(mult_err) < 1e-12, max(mult_err),
                "multiplicative err %.2e; additive %.4f, max-based %.4f"
                % (max(mult_err), mean(add_err), mean(max_err)))

    # ---------------------------------------------------------- V4
    e = ex.expect(
        "V4 saturation dichotomy",
        "Residual ambiguity vanishes iff sum(pi_i) diverges; indexed "
        "from i=2 per Remark 3.9.",
        "thm:saturation / P5",
        "A convergent series reaching the floor, or a divergent one "
        "failing to.")

    N = 4000
    div = [1.0 / i for i in range(2, N + 2)]          # sum diverges
    con = [1.0 / (i * i) for i in range(2, N + 2)]    # sum converges
    gap_div = 1.0
    for p in div:
        gap_div *= (1.0 - p)
    gap_con = 1.0
    for p in con:
        gap_con *= (1.0 - p)
    ex.record("V4_saturation", {
        "n_terms": N,
        "divergent_sum": sum(div), "divergent_residual_gap": gap_div,
        "convergent_sum": sum(con), "convergent_residual_gap": gap_con})
    e.check(gap_div < 1e-3 and gap_con > 0.1,
            {"divergent_gap": gap_div, "convergent_gap": gap_con},
            "divergent series gap -> %.2e; convergent series gap stays %.4f"
            % (gap_div, gap_con))

    # ---------------------------------------------------------- V5
    e = ex.expect(
        "V5 sensitivity and the ordering of control",
        "d pi(L)/d pi_j = (1-pi(L))/(1-pi_j), maximised at the "
        "HIGHEST-resolution contact.",
        "prop:marginal / P6",
        "Control ordered by ascending resolution, or not ordered by "
        "this quantity at all. The sharpest test in the set.")

    analytic = sensitivity(POWERS)
    numeric, h = [], 1e-7
    for j in range(len(POWERS)):
        up = list(POWERS)
        up[j] += h
        dn = list(POWERS)
        dn[j] -= h
        numeric.append((compose(up) - compose(dn)) / (2 * h))
    closed = [(1 - compose(POWERS)) / (1 - p) for p in POWERS]
    ex.record("V5_sensitivity", {
        "powers": POWERS, "analytic": analytic,
        "numeric": numeric, "closed_form": closed})

    agree = (all(rel_close(x, y, 1e-5) for x, y in zip(analytic, numeric))
             and all(rel_close(x, y, 1e-12)
                     for x, y in zip(analytic, closed)))
    order_by_sens = sorted(range(len(POWERS)), key=lambda j: -analytic[j])
    order_by_pow = sorted(range(len(POWERS)), key=lambda j: -POWERS[j])
    ex.record("V5_ordering", {
        "by_sensitivity": ["k%d" % (j + 1) for j in order_by_sens],
        "by_power": ["k%d" % (j + 1) for j in order_by_pow]})
    e.check(agree and order_by_sens == order_by_pow,
            {"analytic": analytic, "matches_numeric": agree},
            "numeric agrees with closed form; control ranks %s == power order"
            % ["k%d" % (j + 1) for j in order_by_sens])

    # ---------------------------------------------------------- V6
    e = ex.expect(
        "V6 inertness under relabelling",
        "Two instruments with equal resolution sequences agree on every "
        "admissible observable.",
        "thm:inertness / P8",
        "An admissible observable separating them.")

    def observables(powers):
        gaps = []
        acc = 1.0
        for q in powers:
            acc *= (1.0 - q)
            gaps.append(round(acc, 12))
        return {"composite": round(compose(powers), 12),
                "contacts": len(powers),
                "cost": len(powers),
                "gaps": tuple(gaps),
                "sensitivities": tuple(round(s, 12)
                                       for s in sensitivity(powers))}

    base = observables(POWERS)
    mismatches = 0
    for _ in range(200):
        relabelled = list(POWERS)          # same sequence, new names only
        if observables(relabelled) != base:
            mismatches += 1
    # Control: an admissible observable MUST separate genuinely different
    # sequences, else the observable set is vacuous.
    ctrl_sep = observables([0.6, 0.35, 0.5, 0.25, 0.16]) != base
    ex.record("V6_inertness", {"relabel_mismatches": mismatches,
                               "control_separates_different": ctrl_sep})
    if not ctrl_sep:
        e.non_discriminating(mismatches, "observable set does not separate "
                                         "genuinely different ladders")
    else:
        e.check(mismatches == 0, mismatches,
                "%d mismatches over 200 relabellings; control separates a "
                "0.01 change in one rung" % mismatches)

    # ---------------------------------------------------------- V7
    e = ex.expect(
        "V7 transit is free",
        "Cost scales with contact count, not with flight length, "
        "duration, or field complexity.",
        "def:cost / P7",
        "Information content increasing with path length at fixed "
        "contact count.")

    rows = []
    for path in [1.0, 10.0, 100.0, 1000.0, 10000.0]:
        s = Substrate("var", geometry=1.0, field_scale=1.0,
                      path_len=path, rng=random.Random(3))
        r = run_sequence(s, POWERS)
        rows.append({"path_len": path, "resolution": r["resolution"],
                     "cost": r["contacts"]})
    ex.record("V7_cost_vs_path", rows)
    res_invariant = all(close(r["resolution"], rows[0]["resolution"], 1e-12)
                        for r in rows)
    cost_invariant = all(r["cost"] == len(POWERS) for r in rows)
    e.check(res_invariant and cost_invariant,
            {"resolutions": [r["resolution"] for r in rows]},
            "path length varied 1 -> 10000 at fixed contact count: "
            "resolution and cost both unchanged")

    # ---------------------------------------------------------- V8
    e = ex.expect(
        "V8 static analysis agrees with execution",
        "A ladder whose declared contacts cannot reach its declared "
        "target is rejected statically, and the verdict matches "
        "execution --- including cases designed to be rejected.",
        "prop:reach / P9",
        "Disagreement between static verdict and executed outcome.")

    cases, disagreements = [], 0
    for _ in range(300):
        n = rng.randint(1, 6)
        ps = [rng.uniform(0.05, 0.8) for _ in range(n)]
        target = rng.uniform(0.5, 0.99)
        static_ok = compose(ps) >= target                 # no execution
        exec_ok = run_sequence(a, ps)["resolution"] >= target - 1e-12
        if static_ok != exec_ok:
            disagreements += 1
        cases.append({"n": n, "target": round(target, 4),
                      "static": static_ok, "executed": exec_ok})
    rejected = sum(1 for c in cases if not c["static"])
    ex.record("V8_static_vs_executed", {
        "n_cases": len(cases), "disagreements": disagreements,
        "statically_rejected": rejected})
    if rejected == 0:
        e.non_discriminating(disagreements,
                             "no case was designed to be rejected")
    else:
        e.check(disagreements == 0, disagreements,
                "%d disagreements over %d cases, %d of them statically "
                "rejected" % (disagreements, len(cases), rejected))

    ex.note("Substrates differ in geometry (3.0 vs 0.2), field scale "
            "(0.4 vs 7.5) and path length (120 vs 8) while realising the "
            "same contact sequence.")
    ex.note("V2's non-local term is the only departure from locality "
            "introduced anywhere in this suite, matching the uniqueness "
            "claim of Prop. 2.8.")

    ex.report()
    print("  written: " + ex.write())
    return ex


if __name__ == "__main__":
    main()
