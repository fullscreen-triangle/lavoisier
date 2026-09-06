"""Produce the measured records the six panels plot.

Everything written here is a measurement taken from the 58 CASMI challenges
resolvable in the 17 mzML files held locally.  Nothing is simulated.
"""
import json
import os
import random

import identify as I

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "panel_data.json")


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx) ** 2 for a in xs)
    syy = sum((b - my) ** 2 for b in ys)
    if sxx <= 0 or syy <= 0:
        return float("nan")
    return sxy / (sxx * syy) ** 0.5


def main():
    random.seed(20220601)
    data = json.load(open(os.path.join(HERE, "casmi_spectra.json")))
    ans = {o["id"]: o for o in json.load(open(os.path.join(HERE, "casmi_answers.json")))}

    rec = {"challenges": [], "ladder": [], "degeneracy_vs_ppm": [],
           "shuffle": [], "decoy": [], "size": [], "energy_surface": []}

    for ch in data:
        if not ch["scans"]:
            continue
        cid = ch["id"]
        pol = ch["scans"][0]["polarity"]
        frags, ces = I.merge_ladder(ch["scans"])
        allc = I.d1_mass(ch["mz"], pol)
        cands, n_all = I.prefilter(allc)
        used = I.d2_contact(cands, frags, ch["mz"], pol)
        for c in cands:
            c["score"] = c["contact"] - 0.02 * abs(c["ppm"]) / I.PPM - I._prior(c)
        ranked = sorted(cands, key=lambda c: -c["score"])
        a = ans.get(cid, {})

        rec["challenges"].append({
            "id": cid, "mz": ch["mz"], "rt": ch["rt"], "pol": pol,
            "n_scans": len(ch["scans"]), "n_ce": len(ces),
            "n_frags": len(used), "n_mass_candidates": n_all,
            "verdict": a.get("verdict", "?"),
            "contact": a.get("contact", 0.0),
            "margin": a.get("margin"),
            "ppm": a.get("ppm"),
            "top_contacts": [round(c["contact"], 4) for c in ranked[:20]],
            # Full same-mass rival set, de-duplicated by formula string, as in
            # validate.py's C1.  top_contacts is capped at 20 for the profile
            # plot; a decoy mean taken from it would compare the winner against
            # only its 19 strongest rivals, which is a different and much
            # harsher control than the one the paper states.
            "rival_mean": (
                round(sum(c["contact"] for c in ranked[1:]
                          if c["fstr"] != ranked[0]["fstr"])
                      / max(1, len([c for c in ranked[1:]
                                    if c["fstr"] != ranked[0]["fstr"]])), 4)
                if any(c["fstr"] != ranked[0]["fstr"] for c in ranked[1:])
                else None),
            "n_rivals": len([c for c in ranked[1:]
                             if c["fstr"] != ranked[0]["fstr"]]),
        })

        # per-candidate contact for the size confound / decoy clouds
        for c in cands:
            f = c["formula"]
            rec["size"].append({
                "id": cid,
                "heavy": sum(f.get(e, 0) for e in ("C", "N", "O", "P", "S")),
                "contact": round(c["contact"], 4),
                "ppm": round(c["ppm"], 3),
                "is_top": c is ranked[0],
            })

    # ---- energy ladder: peak survival across the three collision energies
    for ch in data:
        if not ch["scans"]:
            continue
        by_ce = {}
        for s in ch["scans"]:
            tot = max((i for _, i in s["peaks"]), default=1.0) or 1.0
            acc = by_ce.setdefault(s["ce"], {})
            for m, i in s["peaks"]:
                if i / tot >= 0.01:
                    acc[round(m, 3)] = max(acc.get(round(m, 3), 0.0), i / tot)
        if len(by_ce) < 3:
            continue
        ces = sorted(by_ce)
        # precursor survival and mean fragment mass at each energy
        row = {"id": ch["id"], "mz": ch["mz"], "ces": ces,
               "n_peaks": [len(by_ce[c]) for c in ces],
               "prec_survival": [], "mean_frag_mz": []}
        for c in ces:
            acc = by_ce[c]
            surv = max((v for m, v in acc.items()
                        if abs(m - ch["mz"]) < 0.02), default=0.0)
            row["prec_survival"].append(round(surv, 4))
            sub = [(m, v) for m, v in acc.items() if m < ch["mz"] - 0.5]
            if sub:
                wsum = sum(v for _, v in sub)
                row["mean_frag_mz"].append(
                    round(sum(m * v for m, v in sub) / wsum, 2) if wsum else 0.0)
            else:
                row["mean_frag_mz"].append(0.0)
        rec["ladder"].append(row)

    # ---- degeneracy as a function of tolerance, for a mass ladder
    probes = sorted({round(c["mz"]) for c in rec["challenges"]})
    sample = [c for c in rec["challenges"]]
    sample.sort(key=lambda c: c["mz"])
    picks = [sample[i] for i in range(0, len(sample), max(1, len(sample) // 12))]
    for ppm in (1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0):
        for c in picks:
            n = len(I.d1_mass(c["mz"], c["pol"], tol_ppm=ppm))
            rec["degeneracy_vs_ppm"].append(
                {"ppm": ppm, "mz": c["mz"], "id": c["id"], "n": n})

    # ---- shuffle control, recorded per pair rather than as a mean
    prepared = []
    for ch in data:
        if not ch["scans"]:
            continue
        pol = ch["scans"][0]["polarity"]
        frags, _ = I.merge_ladder(ch["scans"])
        allc = I.d1_mass(ch["mz"], pol)
        if not allc:
            continue
        cands, _n = I.prefilter(allc)
        used = I.d2_contact(cands, frags, ch["mz"], pol)
        if not used:
            continue
        for c in cands:
            c["score"] = c["contact"] - 0.02 * abs(c["ppm"]) / I.PPM - I._prior(c)
        prepared.append({"ch": ch, "pol": pol, "frags": frags,
                         "top": sorted(cands, key=lambda c: -c["score"])[0]})
    for i, p in enumerate(prepared):
        js = [k for k in range(len(prepared))
              if k != i and prepared[k]["pol"] == p["pol"]]
        if not js:
            continue
        for j in random.sample(js, min(3, len(js))):
            q = prepared[j]
            probe = [dict(p["top"])]
            I.d2_contact(probe, q["frags"], q["ch"]["mz"], p["pol"])
            rec["shuffle"].append({
                "id": p["ch"]["id"], "other": q["ch"]["id"],
                "own": round(p["top"]["contact"], 4),
                "foreign": round(probe[0]["contact"], 4),
            })

    # ---- decoy: top vs same-mass rivals, per challenge
    for p in prepared:
        cid = p["ch"]["id"]
        c = [x for x in rec["challenges"] if x["id"] == cid]
        if not c:
            continue
        tc = c[0]["top_contacts"]
        if len(tc) < 2 or c[0]["rival_mean"] is None:
            continue
        rec["decoy"].append({"id": cid, "top": tc[0],
                             "decoy_mean": c[0]["rival_mean"],
                             "n": c[0]["n_rivals"]})

    with open(OUT, "w") as f:
        json.dump(rec, f)

    print("challenges       :", len(rec["challenges"]))
    print("ladder rows      :", len(rec["ladder"]))
    print("degeneracy points:", len(rec["degeneracy_vs_ppm"]))
    print("shuffle pairs    :", len(rec["shuffle"]))
    print("decoy rows       :", len(rec["decoy"]))
    print("candidate rows   :", len(rec["size"]))
    xs = [r["heavy"] for r in rec["size"]]
    ys = [r["contact"] for r in rec["size"]]
    print("r(size,contact)  : %+.4f" % pearson(xs, ys))


if __name__ == "__main__":
    main()
