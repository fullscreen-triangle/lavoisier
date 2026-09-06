"""Does the contact determination carry information, or is it scoring noise?

The CASMI answer key is not on disk, so we cannot grade against truth here.
But the central claim is testable without it, because the claim is about the
RELATIONSHIP between two determinations, not about either one's correctness:

  CLAIM.  Fragment contact is a determination independent of exact mass.

If that is false -- if contact is just a function of formula size, or if any
mass-correct formula explains the fragments about equally well -- then contact
cannot discriminate and the whole method reduces to picking by ppm.

Three controls:

  C1  DECOY.  For each challenge, compare the contact of the top-ranked
      candidate against the contact of the other mass-correct candidates.
      These decoys are all within the instrument's ppm window, so D1 cannot
      separate them.  If contact separates them, contact adds information.

  C2  SHUFFLE.  Score each candidate against another challenge's fragments.
      If contact is real, a candidate scores worse on fragments that did not
      come from it.  If contact scores just as well on foreign fragments, it is
      measuring formula size, not structure.

  C3  SIZE.  Correlate contact with the candidate's atom count.  A strong
      correlation means large formulas trivially explain everything -- the
      known failure mode of subformula assignment.
"""
import json
import os
import random

import identify as I

HERE = os.path.dirname(os.path.abspath(__file__))


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx) ** 2 for a in xs)
    syy = sum((b - my) ** 2 for b in ys)
    if sxx <= 0 or syy <= 0:
        return float("nan")
    return sxy / (sxx * syy) ** 0.5


def main():
    random.seed(20220601)
    data = json.load(open(os.path.join(HERE, "casmi_spectra.json")))
    prepared = []
    for ch in data:
        if not ch["scans"]:
            continue
        pol = ch["scans"][0]["polarity"]
        frags, _ = I.merge_ladder(ch["scans"])
        allc = I.d1_mass(ch["mz"], pol)
        if not allc:
            continue
        cands, n_all = I.prefilter(allc)
        used = I.d2_contact(cands, frags, ch["mz"], pol)
        if not used:
            continue
        for c in cands:
            c["score"] = c["contact"] - 0.02 * abs(c["ppm"]) / I.PPM - I._prior(c)
        prepared.append({"ch": ch, "pol": pol, "frags": frags,
                         "cands": cands, "n_frags": len(used)})
    print("challenges usable for validation: %d" % len(prepared))

    # ---- C1 decoy separation -------------------------------------------
    print()
    print("C1  DECOY SEPARATION (top candidate vs same-mass decoys)")
    gaps, tops, decs = [], [], []
    for p in prepared:
        ranked = sorted(p["cands"], key=lambda c: -c["score"])
        top = ranked[0]
        others = [c for c in ranked[1:] if c["fstr"] != top["fstr"]]
        if not others:
            continue
        mean_dec = sum(c["contact"] for c in others) / len(others)
        gaps.append(top["contact"] - mean_dec)
        tops.append(top["contact"])
        decs.append(mean_dec)
    n = len(gaps)
    pos = sum(1 for g in gaps if g > 0)
    print("  challenges compared:            %d" % n)
    print("  mean contact, top candidate:    %.4f" % (sum(tops) / n))
    print("  mean contact, same-mass decoys: %.4f" % (sum(decs) / n))
    print("  mean separation:                %+.4f" % (sum(gaps) / n))
    print("  top beats decoy mean in:        %d of %d (%.0f%%)"
          % (pos, n, 100.0 * pos / n))

    # ---- C2 shuffle control --------------------------------------------
    print()
    print("C2  SHUFFLE CONTROL (candidate scored on foreign fragments)")
    own, foreign = [], []
    for i, p in enumerate(prepared):
        j = random.choice([k for k in range(len(prepared)) if k != i])
        q = prepared[j]
        if q["pol"] != p["pol"]:
            continue
        ranked = sorted(p["cands"], key=lambda c: -c["score"])
        top = dict(ranked[0])
        own.append(ranked[0]["contact"])
        probe = [dict(top)]
        I.d2_contact(probe, q["frags"], q["ch"]["mz"], p["pol"])
        foreign.append(probe[0]["contact"])
    if own:
        print("  pairs tested:                   %d" % len(own))
        print("  mean contact on own fragments:  %.4f" % (sum(own) / len(own)))
        print("  mean contact on foreign:        %.4f" % (sum(foreign) / len(foreign)))
        drop = sum(o - f for o, f in zip(own, foreign)) / len(own)
        print("  mean drop:                      %+.4f" % drop)
        print("  own > foreign in:               %d of %d"
              % (sum(1 for o, f in zip(own, foreign) if o > f), len(own)))

    # ---- C3 size confound ----------------------------------------------
    print()
    print("C3  SIZE CONFOUND (contact vs heavy-atom count)")
    sz, ct = [], []
    for p in prepared:
        for c in p["cands"]:
            f = c["formula"]
            sz.append(sum(f.get(e, 0) for e in ("C", "N", "O", "P", "S")))
            ct.append(c["contact"])
    print("  candidate-fragment pairs:       %d" % len(sz))
    print("  Pearson r(size, contact):       %+.4f" % pearson(sz, ct))

    # ---- degeneracy of D1 alone ----------------------------------------
    print()
    print("D1 DEGENERACY (why mass alone cannot answer)")
    ns = sorted(len({c["fstr"] for c in p["cands"]}) for p in prepared)
    print("  distinct formulas within %.0f ppm, median: %d"
          % (I.PPM, ns[len(ns) // 2]))
    print("  min %d   max %d" % (ns[0], ns[-1]))


if __name__ == "__main__":
    main()
