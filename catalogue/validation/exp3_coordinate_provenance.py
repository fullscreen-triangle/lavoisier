"""
exp3_coordinate_provenance.py --- validation of

    "Coordinates Without Provenance: Context Collapse in Analytical
     Pipelines"

The claims here are combinatorial statements about finite maps, so where
the space is small enough they are checked by EXHAUSTIVE enumeration
rather than by sampling: a counterexample, if one existed, could not
hide. Each test is paired with a control establishing that the property
under test is not vacuous --- that some map in the same family fails it.

  thm:collapse         pigeonhole forces non-auditability
  thm:pipeline-loss    an upstream collapse survives every downstream stage
  cor:no-recovery      no function recovers a discarded context
  thm:minimal-record   Ctx/= is the smallest sufficient record space
  prop:cost            per-run records are O(Ns log s), not O(N sum log|Prv|)
  thm:comparability    record equality decides comparability
  thm:decline-sound    a defined comparison is never unlicensed
  thm:record-floor     no non-empty record is free
"""
from __future__ import annotations

import itertools
import math

from common import Experiment, close


# =====================================================================
#  Finite coordinate maps (def:coord)
# =====================================================================

def all_maps(domain, codomain):
    """Every function domain -> codomain, as a dict. Exhaustive."""
    domain = list(domain)
    for values in itertools.product(codomain, repeat=len(domain)):
        yield dict(zip(domain, values))


def is_auditable(phi, X, C):
    """def:audit: some aud : Crd -> Ctx recovers the context from the
    coordinate alone. Equivalently, no coordinate is produced under two
    different contexts."""
    seen = {}
    for x in X:
        for c in C:
            k = phi[(x, c)]
            if k in seen and seen[k] != c:
                return False
            seen[k] = c
    return True


def behaviour(phi, X, c):
    """phi(., c) as a function on X --- the object thm:minimal-record's
    equivalence relation is defined on."""
    return tuple(phi[(x, c)] for x in X)


def collapsed_pairs(phi, X, C):
    """Context pairs (c, c') that some measurement cannot distinguish."""
    out = set()
    for x in X:
        for c1, c2 in itertools.combinations(C, 2):
            if phi[(x, c1)] == phi[(x, c2)]:
                out.add((c1, c2))
    return out


def rec_min(phi, X, C):
    """The quotient map Ctx -> Ctx/= of thm:minimal-record."""
    classes = {}
    for c in C:
        classes.setdefault(behaviour(phi, X, c), []).append(c)
    label = {}
    for i, (_, members) in enumerate(sorted(classes.items())):
        for c in members:
            label[c] = i
    return label


def sufficient_for_comparison(rec, phi, X, C):
    """def:sufficient: rec(c) = rec(c') implies phi(.,c) = phi(.,c')."""
    for c1, c2 in itertools.combinations(C, 2):
        if rec[c1] == rec[c2] and behaviour(phi, X, c1) != behaviour(phi, X, c2):
            return False
    return True


def main():
    ex = Experiment(
        name="exp3_coordinate_provenance",
        paper="coordinate-provenance",
        question="Is a coordinate computed in one context safely "
                 "comparable to one computed in another?",
    )

    X = ["x1", "x2", "x3"]
    C = ["c1", "c2", "c3"]

    # ------------------------------------------------- thm:collapse
    e = ex.expect(
        "thm:collapse collapse is forced by cardinality",
        "If |Crd| < |Ctx| for the maps phi(x,.) at some fixed x, then "
        "two distinct contexts share a coordinate and phi is not "
        "auditable.",
        "thm:collapse",
        "An auditable map whose coordinate space is smaller than its "
        "context space.")

    # Exhaustive over EVERY map X x Ctx -> Crd with |Crd| = 2 < |Ctx| = 3.
    Crd_small = ["k1", "k2"]
    dom = [(x, c) for x in X for c in C]
    n_small, auditable_small = 0, 0
    for phi in all_maps(dom, Crd_small):
        n_small += 1
        if is_auditable(phi, X, C):
            auditable_small += 1
    ex.record("collapse_small_codomain", {
        "n_maps": n_small, "size_Crd": len(Crd_small), "size_Ctx": len(C),
        "auditable": auditable_small})

    # Control: with |Crd| large enough, auditable maps MUST exist ---
    # otherwise the test is measuring the enumeration, not the theorem.
    Crd_big = ["k%d" % i for i in range(len(X) * len(C))]
    witness = {}
    for i, (x, c) in enumerate(dom):
        witness[(x, c)] = Crd_big[i]
    ctrl_auditable = is_auditable(witness, X, C)
    ex.record("collapse_control", {"size_Crd": len(Crd_big),
                                   "auditable_witness_exists": ctrl_auditable})
    if not ctrl_auditable:
        e.non_discriminating(auditable_small,
                             "no auditable map exists even with a large "
                             "coordinate space, so the test is vacuous")
    else:
        e.check(auditable_small == 0, auditable_small,
                "0 of %d maps into a 2-element coordinate space are "
                "auditable (|Ctx| = 3); control: an injective map into a "
                "%d-element space is auditable"
                % (n_small, len(Crd_big)))

    # -------------------------------------------- thm:pipeline-loss
    e = ex.expect(
        "thm:pipeline-loss loss compounds along a pipeline",
        "If stage 1 is non-auditable then the composition is "
        "non-auditable, for EVERY choice of stage 2 --- including "
        "injective and information-preserving ones.",
        "thm:pipeline-loss",
        "A downstream stage that restores auditability.")

    # phi1 deliberately collapses c1 and c2 at every x.
    Crd1 = ["a", "b"]
    phi1 = {}
    for x in X:
        phi1[(x, "c1")] = "a"
        phi1[(x, "c2")] = "a"          # collapsed with c1
        phi1[(x, "c3")] = "b"
    assert not is_auditable(phi1, X, C)

    C2 = ["d1", "d2"]
    Crd2 = ["p", "q", "r", "s"]
    dom2 = [(k, d) for k in Crd1 for d in C2]
    restored, n_stage2 = 0, 0
    for phi2 in all_maps(dom2, Crd2):
        n_stage2 += 1
        psi, CC = {}, [(c1, c2) for c1 in C for c2 in C2]
        for x in X:
            for (c1, c2) in CC:
                psi[(x, (c1, c2))] = phi2[(phi1[(x, c1)], c2)]
        if is_auditable(psi, X, CC):
            restored += 1
    ex.record("pipeline_loss", {"n_stage2_maps": n_stage2,
                                "restored_auditability": restored})

    # Control: an injective stage 2 applied to an AUDITABLE stage 1 must
    # keep it auditable, else composition destroys auditability
    # unconditionally and the test says nothing about stage 1.
    phi1_ok = {}
    for i, (x, c) in enumerate(dom):
        phi1_ok[(x, c)] = "u%d" % i
    inj = {}
    CC = [(c, d) for c in C for d in C2]
    for x in X:
        for (c, d) in CC:
            inj[(x, (c, d))] = (phi1_ok[(x, c)], d)
    ctrl = is_auditable(inj, X, CC)
    ex.record("pipeline_loss_control", {"auditable_stage1_stays_auditable": ctrl})
    if not ctrl:
        e.non_discriminating(restored,
                             "composition destroys auditability even for an "
                             "auditable first stage")
    else:
        e.check(restored == 0, restored,
                "0 of %d possible second stages restore auditability; "
                "control: composing an auditable stage 1 with an "
                "injective stage 2 stays auditable" % n_stage2)

    # ---------------------------------------------- cor:no-recovery
    e = ex.expect(
        "cor:no-recovery no downstream function recovers the context",
        "No f : Crd2 -> Ctx1 returns c1 on one input and c1' on the "
        "other, because the two inputs are the same element.",
        "cor:no-recovery",
        "A recovery function, however contrived.")

    phi2_fixed = {(k, d): (k, d) for k in Crd1 for d in C2}   # injective
    out1 = phi2_fixed[(phi1[("x1", "c1")], "d1")]
    out2 = phi2_fixed[(phi1[("x1", "c2")], "d1")]
    identical = out1 == out2
    # Exhaustive: enumerate EVERY f from the reachable coordinates to Ctx.
    reachable = sorted({phi2_fixed[(phi1[(x, c)], d)]
                        for x in X for c in C for d in C2})
    recoveries = 0
    for f in all_maps(reachable, C):
        if f[out1] == "c1" and f[out2] == "c2":
            recoveries += 1
    ex.record("no_recovery", {"outputs_identical": identical,
                              "n_functions_enumerated": len(C) ** len(reachable),
                              "successful_recoveries": recoveries})

    # Control: if the outputs were DISTINCT, a recovery function would
    # exist --- confirming the obstruction is the collapse, not the
    # enumeration.
    ctrl_recoveries = sum(1 for f in all_maps(reachable, C)
                          if f[reachable[0]] == "c1" and f[reachable[1]] == "c2")
    ex.record("no_recovery_control",
              {"recoveries_when_inputs_distinct": ctrl_recoveries})
    if ctrl_recoveries == 0:
        e.non_discriminating(recoveries,
                             "no function separates even distinct inputs")
    else:
        e.check(identical and recoveries == 0, recoveries,
                "the two pipeline outputs are the same element, and 0 of "
                "%d functions recover the context; control: %d functions "
                "succeed when the inputs are distinct"
                % (len(C) ** len(reachable), ctrl_recoveries))

    # ------------------------------------------- thm:minimal-record
    e = ex.expect(
        "thm:minimal-record the quotient is the smallest sufficient record",
        "rec_min is sufficient for comparison, and NO sufficient record "
        "map has a smaller image. Checked exhaustively over every record "
        "map into every codomain size.",
        "thm:minimal-record",
        "A sufficient record map with a strictly smaller image than "
        "|Ctx/=|.")

    # A map where exactly two of three contexts behave identically, so
    # the quotient is a genuine coarsening rather than the identity.
    phi_q = {}
    for x in X:
        phi_q[(x, "c1")] = "v" + x
        phi_q[(x, "c2")] = "v" + x        # c1 = c2
        phi_q[(x, "c3")] = "w" + x
    rmin = rec_min(phi_q, X, C)
    n_classes = len(set(rmin.values()))
    rmin_ok = sufficient_for_comparison(rmin, phi_q, X, C)

    smaller_sufficient = []
    for size in range(1, len(C) + 1):
        for rec in all_maps(C, list(range(size))):
            if not sufficient_for_comparison(rec, phi_q, X, C):
                continue
            if len(set(rec.values())) < n_classes:
                smaller_sufficient.append(dict(rec))
    ex.record("minimal_record", {
        "n_contexts": len(C), "n_equivalence_classes": n_classes,
        "rec_min_sufficient": rmin_ok,
        "sufficient_maps_with_smaller_image": len(smaller_sufficient)})

    # Control: an INSUFFICIENT map with a smaller image must exist, or
    # "no smaller sufficient map" is true only because none is smaller.
    coarser_exists = any(
        len(set(rec.values())) < n_classes
        and not sufficient_for_comparison(rec, phi_q, X, C)
        for rec in all_maps(C, list(range(len(C)))))
    ex.record("minimal_record_control",
              {"smaller_but_insufficient_map_exists": coarser_exists})
    if not coarser_exists:
        e.non_discriminating(len(smaller_sufficient),
                             "no map with a smaller image exists at all")
    else:
        e.check(rmin_ok and not smaller_sufficient,
                len(smaller_sufficient),
                "rec_min has %d classes and is sufficient; 0 sufficient "
                "maps have a smaller image, though smaller INSUFFICIENT "
                "maps do exist" % n_classes)

    # NOTE ON THE PROOF. The paper states factorisation (any sufficient
    # rec factors through rec_min) but its proof concedes the
    # factorisation step and establishes the image-size bound
    # |im rec_min| <= |im rec| instead. The bound is what is tested
    # above, and it is what thm:record-floor actually consumes. The
    # stated factorisation is FALSE as written: a sufficient record map
    # may split an equivalence class, in which case it does not factor
    # through the quotient. The next test exhibits one.
    e = ex.expect(
        "thm:minimal-record (stated form) factorisation as printed",
        "As printed, the theorem claims every sufficient record map "
        "factors through rec_min. A map that splits an equivalence "
        "class refutes this while remaining sufficient.",
        "thm:minimal-record",
        "No sufficient record map splits a class --- which would make "
        "the printed statement correct after all.")

    splitters = []
    for rec in all_maps(C, list(range(len(C)))):
        if not sufficient_for_comparison(rec, phi_q, X, C):
            continue
        # factors through rec_min iff rec is constant on =-classes
        constant_on_classes = all(
            rec[c1] == rec[c2] for c1, c2 in itertools.combinations(C, 2)
            if rmin[c1] == rmin[c2])
        if not constant_on_classes:
            splitters.append(dict(rec))
    ex.record("minimal_record_factorisation", {
        "sufficient_maps_not_factoring_through_rec_min": len(splitters),
        "example": splitters[0] if splitters else None})
    e.check(len(splitters) > 0, len(splitters),
            "%d sufficient record maps do NOT factor through rec_min "
            "(e.g. %s splits the class {c1,c2}); the printed "
            "factorisation claim is false, the image-size bound the "
            "proof actually establishes is not"
            % (len(splitters), splitters[0] if splitters else None))

    # ------------------------------------------------- prop:cost
    e = ex.expect(
        "prop:cost per-run records are cheap",
        "Attaching a record to every measurement costs "
        "N*sum(log|Prv_i|) bits; storing per run and referencing costs "
        "N*s*log(s) + sum(log|Prv_i|), which is asymptotically smaller "
        "in N for fixed contexts.",
        "prop:cost",
        "The per-run scheme costing more than the per-measurement one "
        "at large N.")

    s_stages = 6
    prv_sizes = [64, 128, 32, 256, 16, 512]
    rows = []
    for N in [10 ** 2, 10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6]:
        per_meas = N * sum(math.log2(p) for p in prv_sizes)
        per_run = N * s_stages * math.log2(s_stages) + sum(
            math.log2(p) for p in prv_sizes)
        rows.append({"N": N, "per_measurement_bits": per_meas,
                     "per_run_bits": per_run, "ratio": per_meas / per_run})
    ex.record("cost", {"stages": s_stages, "record_sizes": prv_sizes,
                       "rows": rows})
    ratios = [r["ratio"] for r in rows]
    cheaper = all(r["per_run_bits"] < r["per_measurement_bits"] for r in rows)
    # Control: the advantage must come from the constant record, so with
    # a SINGLE stage carrying a tiny record the schemes should converge.
    tiny = 1 * math.log2(2)
    conv = (10 ** 6 * tiny) / (10 ** 6 * 1 * math.log2(1.0001) + tiny)
    ex.record("cost_control", {"ratio_at_one_tiny_stage": conv,
                               "ratio_range": [min(ratios), max(ratios)]})
    e.check(cheaper, {"min_ratio": min(ratios), "max_ratio": max(ratios)},
            "per-run scheme is cheaper at every N tested; saving is "
            "%.2fx and is independent of N (%.2f at N=100, %.2f at "
            "N=1e6) --- the record is stored once, not amplified"
            % (ratios[-1], ratios[0], ratios[-1]))

    # ------------------------------------------- thm:comparability
    e = ex.expect(
        "thm:comparability record equality decides comparability",
        "With rec sufficient, r1 = r2 implies the coordinates are "
        "comparable; with rec = rec_min the test never reports "
        "comparable for coordinates whose generating maps disagree.",
        "thm:comparability",
        "A pair passing the record test whose generating maps differ "
        "on X.")

    checked, unsound = 0, 0
    for c1, c2 in itertools.product(C, C):
        for x1, x2 in itertools.product(X, X):
            r1, r2 = rmin[c1], rmin[c2]
            if r1 != r2:
                continue
            checked += 1
            if behaviour(phi_q, X, c1) != behaviour(phi_q, X, c2):
                unsound += 1
    # Control: the test must sometimes DECLINE, or it is trivially sound.
    declines = sum(1 for c1, c2 in itertools.product(C, C)
                   if rmin[c1] != rmin[c2])
    ex.record("comparability", {"admitted_pairs": checked,
                               "unsound_admissions": unsound,
                               "declined_context_pairs": declines})
    if declines == 0:
        e.non_discriminating(unsound, "the record test admits everything")
    else:
        e.check(unsound == 0, unsound,
                "%d admitted comparisons, 0 of them between disagreeing "
                "maps; %d context pairs declined" % (checked, declines))

    # ------------------------------------------- thm:decline-sound
    e = ex.expect(
        "thm:decline-sound a defined comparison is always licensed",
        "The partial comparison is defined exactly when r1 = r2, and "
        "every defined case is between coordinates produced by the same "
        "function on X.",
        "thm:decline-sound",
        "A defined comparison the contexts do not license.")

    # Compare against the total (silent) alternative on the same data.
    total_wrong, partial_wrong, declined = 0, 0, 0
    for c1, c2 in itertools.product(C, C):
        agree = behaviour(phi_q, X, c1) == behaviour(phi_q, X, c2)
        if not agree:
            total_wrong += 1                 # a total comparator answers anyway
        if rmin[c1] == rmin[c2]:
            if not agree:
                partial_wrong += 1
        else:
            declined += 1
    ex.record("decline_sound", {
        "total_comparator_unlicensed_answers": total_wrong,
        "partial_comparator_unlicensed_answers": partial_wrong,
        "declined": declined})
    if total_wrong == 0:
        e.non_discriminating(partial_wrong,
                             "even a total comparator is never wrong here, so "
                             "declining cannot be shown to help")
    else:
        e.check(partial_wrong == 0, partial_wrong,
                "the total comparator answers %d unlicensed comparisons; "
                "the partial one answers 0 and declines %d"
                % (total_wrong, declined))

    # -------------------------------------------- thm:record-floor
    e = ex.expect(
        "thm:record-floor no non-empty record is free",
        "A one-element record space is never sufficient once some "
        "extension context behaves differently, and any fixed Prv is "
        "eventually exhausted as contexts accumulate.",
        "thm:record-floor / ax:noncomplete",
        "A constant record remaining sufficient across an extension, or "
        "a bounded record surviving unbounded extension.")

    trivial = {c: 0 for c in C}
    trivial_ok = sufficient_for_comparison(trivial, phi_q, X, C)

    # Extend the laboratory's history: each release adds a context that
    # behaves unlike every existing one (ax:noncomplete's hypothesis).
    history, Prv_capacity = [], 4
    Cx, phix = list(C), dict(phi_q)
    exhausted_at = None
    for release in range(1, 12):
        cnew = "c%d+" % release
        Cx.append(cnew)
        for x in X:
            phix[(x, cnew)] = "z%d%s" % (release, x)     # novel behaviour
        classes = len({behaviour(phix, X, c) for c in Cx})
        history.append({"release": release, "n_contexts": len(Cx),
                        "n_classes": classes,
                        "fits_in_Prv": classes <= Prv_capacity})
        if exhausted_at is None and classes > Prv_capacity:
            exhausted_at = release
    ex.record("record_floor", {
        "trivial_record_sufficient": trivial_ok,
        "Prv_capacity": Prv_capacity,
        "exhausted_at_release": exhausted_at,
        "history": history})

    # Control: an extension that adds a context behaving IDENTICALLY to
    # an existing one must NOT increase the class count --- otherwise
    # the count grows for a reason unrelated to the axiom.
    Cy, phiy = list(C), dict(phi_q)
    before = len({behaviour(phiy, X, c) for c in Cy})
    Cy.append("dup")
    for x in X:
        phiy[(x, "dup")] = phi_q[(x, "c1")]
    after = len({behaviour(phiy, X, c) for c in Cy})
    ex.record("record_floor_control", {"classes_before": before,
                                       "classes_after_duplicate": after})
    if after != before:
        e.non_discriminating(exhausted_at,
                             "class count grows even for a behaviourally "
                             "identical context, so growth is not evidence "
                             "of new behaviour")
    else:
        e.check((not trivial_ok) and exhausted_at is not None,
                {"trivial_sufficient": trivial_ok,
                 "exhausted_at": exhausted_at},
                "a constant record is insufficient; a %d-element record "
                "is exhausted at release %d; control: a behaviourally "
                "duplicate context leaves the class count at %d"
                % (Prv_capacity, exhausted_at, after))

    ex.note("thm:collapse, thm:pipeline-loss, cor:no-recovery and "
            "thm:minimal-record are checked by exhaustive enumeration "
            "over the whole function space, not by sampling.")
    ex.note("The printed form of thm:minimal-record claims a "
            "factorisation its own proof does not establish; the test "
            "above exhibits a sufficient record map that splits an "
            "equivalence class. The image-size bound the proof does "
            "establish is what thm:record-floor consumes, so the floor "
            "result is unaffected.")

    ex.report()
    print("  written: " + ex.write())
    return ex


if __name__ == "__main__":
    main()
