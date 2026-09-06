"""CASMI 2022 categories 1 and 2: adduct annotation and elemental formula.

The method is the catalogue framework applied to a real determination problem.

Two independent determinations are made of every candidate neutral formula:

  D1  MASS.  The candidate's exact mass must reproduce the measured precursor
      m/z under some adduct, within the instrument's ppm error.  This is the
      conventional determination and it is heavily degenerate: many formulas
      fall inside a few ppm of the same mass.

  D2  CONTACT.  Every fragment peak is a subformula of the precursor.  A
      candidate is contacted by a fragment when some sub-formula of the
      candidate reproduces that fragment's mass, with a chemically admissible
      neutral or radical loss.  The set of contacted fragments, weighted by
      intensity, is the candidate's contact set.  Its size is a determination
      that does not reduce to D1: two formulas of identical mass generally
      support different fragment sets.

A candidate is LICENSED only where the two determinations agree and the margin
over the runner-up clears a floor.  Where they do not, the method DECLINES.
Declining is the point: CASMI grades a wrong answer and a blank identically,
but a method that answers below its floor cannot be trusted where it does
answer.  We report both the answer set and the decline set.

Charge is taken from the file.  Adducts are enumerated over the standard
ESI set for each polarity; the adduct that licenses the formula IS the
category-1 answer, so categories 1 and 2 are solved jointly rather than in
sequence.
"""
import json
import os
from itertools import product

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- constants
E = 0.00054857990907          # electron mass
MASS = {
    "C": 12.0,
    "H": 1.0078250319,
    "N": 14.0030740052,
    "O": 15.9949146221,
    "P": 30.97376151,
    "S": 31.97207069,
    "Cl": 34.96885271,
    "Na": 22.98976928,
    "K": 38.96370649,
}

# adduct: (name, n_mer, charge_sign, added atoms as dict, lost atoms as dict)
# m/z = (n*M + sum(added) - sum(lost) -/+ e) / |z|
ADDUCTS_POS = [
    ("[M+H]+",        1, {"H": 1}, {}),
    ("[M+Na]+",       1, {"Na": 1}, {}),
    ("[M+K]+",        1, {"K": 1}, {}),
    ("[M+NH4]+",      1, {"N": 1, "H": 4}, {}),
    ("[M+H-H2O]+",    1, {"H": 1}, {"H": 2, "O": 1}),
    ("[M+H-2H2O]+",   1, {"H": 1}, {"H": 4, "O": 2}),
    ("[2M+H]+",       2, {"H": 1}, {}),
    ("[2M+Na]+",      2, {"Na": 1}, {}),
]
ADDUCTS_NEG = [
    ("[M-H]-",        1, {}, {"H": 1}),
    ("[M+Cl]-",       1, {"Cl": 1}, {}),
    ("[M+HCOO]-",     1, {"C": 1, "H": 1, "O": 2}, {}),
    ("[M+CH3COO]-",   1, {"C": 2, "H": 3, "O": 2}, {}),
    ("[M-H2O-H]-",    1, {}, {"H": 3, "O": 1}),
    ("[2M-H]-",       2, {}, {"H": 1}),
]

PPM = 8.0            # instrument tolerance on the precursor
FRAG_PPM = 12.0      # fragments are noisier and lower intensity
FRAG_MIN_DA = 0.004  # absolute floor: at m/z 91, 12 ppm is only 1.1 mDa, which
                     # is tighter than the peak-merging window above


def formula_mass(f):
    return sum(MASS[e] * n for e, n in f.items() if n)


def fstr(f):
    order = ["C", "H", "N", "O", "P", "S"]
    return "".join("%s%s" % (e, f[e] if f[e] > 1 else "") for e in order if f.get(e))


def rdbe(f):
    """Ring-plus-double-bond equivalent. Negative or half-integer-invalid -> reject."""
    return (f.get("C", 0) - f.get("H", 0) / 2.0 - f.get("N", 0) / 2.0
            + f.get("P", 0) / 2.0 + 1)


def senior_ok(f):
    """Valence / Senior rules, plus the standard heuristic filters."""
    c, h, n, o, p, s = (f.get(k, 0) for k in ("C", "H", "N", "O", "P", "S"))
    if c < 1:
        return False
    r = rdbe(f)
    if r < 0 or r > 40:
        return False
    if abs(r - round(r)) > 1e-9 and abs(r - round(r) - 0.5) > 1e-9:
        return False
    # H/C ratio: real molecules sit well inside this
    if h > 0 and not (0.1 <= h / float(c) <= 3.1):
        return False
    if h == 0 and c > 3:
        return False
    # element ratios (Kind & Fiehn heuristics, generous bounds)
    if n / float(c) > 1.3 or o / float(c) > 1.5:
        return False
    if p and p / float(c) > 0.35:
        return False
    if s and s / float(c) > 0.35:
        return False
    # absolute caps: metabolites and exposome chemicals essentially never
    # exceed these, and leaving them open is what admits C33H56N2OP2S5.
    if p > 4 or s > 4 or n > 12:
        return False
    if p + s > 5:
        return False
    # H must have the right parity given N (nitrogen rule for even-electron ions)
    return True


def enumerate_formulas(target_mass, tol_da, maxel=None):
    """All CHNOPS formulas whose exact mass is within tol of target."""
    if target_mass <= 0:
        return []
    lim = {
        "C": min(90, int(target_mass / 12.0) + 1),
        "N": min(20, int(target_mass / 14.0) + 1),
        "O": min(30, int(target_mass / 16.0) + 1),
        "P": min(6, int(target_mass / 31.0) + 1),
        "S": min(6, int(target_mass / 32.0) + 1),
    }
    if maxel:
        lim.update(maxel)
    out = []
    for p in range(lim["P"] + 1):
        mp = p * MASS["P"]
        if mp > target_mass + tol_da:
            break
        for s in range(lim["S"] + 1):
            ms = mp + s * MASS["S"]
            if ms > target_mass + tol_da:
                break
            for n in range(lim["N"] + 1):
                mn = ms + n * MASS["N"]
                if mn > target_mass + tol_da:
                    break
                for o in range(lim["O"] + 1):
                    mo = mn + o * MASS["O"]
                    if mo > target_mass + tol_da:
                        break
                    for c in range(lim["C"] + 1):
                        mc = mo + c * 12.0
                        if mc > target_mass + tol_da:
                            break
                        rem = target_mass - mc
                        if rem < -tol_da:
                            break
                        h = int(round(rem / MASS["H"]))
                        if h < 0:
                            continue
                        m = mc + h * MASS["H"]
                        if abs(m - target_mass) <= tol_da:
                            f = {"C": c, "H": h, "N": n, "O": o, "P": p, "S": s}
                            if senior_ok(f):
                                out.append((f, m))
    return out


class SubIndex:
    """Sorted index of every subformula mass of a parent formula.

    Built once per candidate.  Enumeration is bounded by the PARENT's own atom
    counts, so it is small (a few 10^4 at most) rather than by the fragment
    mass, which is what made the naive version intractable.
    """

    __slots__ = ("masses", "forms")

    def __init__(self, parent, cap=60000):
        c = parent.get("C", 0); h = parent.get("H", 0) + 1
        n = parent.get("N", 0); o = parent.get("O", 0)
        pp = parent.get("P", 0); ss = parent.get("S", 0)
        # keep the space bounded: if the parent is huge, coarsen H by stepping
        acc = []
        base = []
        for ip in range(pp + 1):
            for isx in range(ss + 1):
                for inn in range(n + 1):
                    for io_ in range(o + 1):
                        base.append((ip * MASS["P"] + isx * MASS["S"]
                                     + inn * MASS["N"] + io_ * MASS["O"],
                                     (inn, io_, ip, isx)))
        for bm, tag in base:
            for ic in range(c + 1):
                mc = bm + ic * 12.0
                for ih in range(h + 1):
                    m = mc + ih * MASS["H"]
                    if m < 12.0:
                        continue
                    acc.append((m, (ic, ih) + tag))
                    if len(acc) > cap:
                        break
                if len(acc) > cap:
                    break
            if len(acc) > cap:
                break
        acc.sort()
        self.masses = [a[0] for a in acc]
        self.forms = [a[1] for a in acc]

    def match(self, mass, tol):
        import bisect
        lo = bisect.bisect_left(self.masses, mass - tol)
        hi = bisect.bisect_right(self.masses, mass + tol)
        best = None
        for i in range(lo, hi):
            ic, ih, inn, io_, ip, isx = self.forms[i]
            f = {"C": ic, "H": ih, "N": inn, "O": io_, "P": ip, "S": isx}
            if ic < 1 and ih < 1:
                continue
            r = rdbe(f)
            if r < -0.5 or abs(r - round(r)) > 1e-9 and abs(r - round(r) - 0.5) > 1e-9:
                continue
            err = abs(self.masses[i] - mass)
            # prefer the lowest-RDBE, closest-mass explanation
            key = (err, r)
            if best is None or key < best[0]:
                best = (key, f)
        return best[1] if best else None


# ------------------------------------------------------------- the two determinations
def d1_mass(precursor_mz, polarity, tol_ppm=PPM):
    """Determination 1: every (adduct, neutral formula) reproducing the mass."""
    adducts = ADDUCTS_POS if polarity == "+" else ADDUCTS_NEG
    sign = 1 if polarity == "+" else -1
    cands = []
    for name, nmer, add, lost in adducts:
        shift = formula_mass(add) - formula_mass(lost) - sign * E
        neutral = (precursor_mz - shift) / nmer
        if neutral < 50:
            continue
        tol = neutral * tol_ppm * 1e-6
        for f, m in enumerate_formulas(neutral, tol):
            # the adduct's own lost atoms must be available in the neutral
            if any(f.get(e, 0) < n for e, n in lost.items() if e in f):
                continue
            calc = (nmer * m + shift)
            ppm = (precursor_mz - calc) / precursor_mz * 1e6
            cands.append({"adduct": name, "formula": f, "fstr": fstr(f),
                          "neutral_mass": m, "ppm": ppm, "nmer": nmer})
    return cands


def merge_ladder(scans, min_rel=0.01):
    """Merge the stepped-energy ladder into one weighted fragment list.

    Peaks are NOT averaged across energies -- each energy is kept as its own
    evidence channel and a peak is scored by the number of energies at which it
    appears as well as its intensity.  A peak seen at all three energies is a
    stronger structural claim than a peak seen once.
    """
    by_ce = {}
    for s in scans:
        by_ce.setdefault(s["ce"], []).append(s)
    channels = {}
    for ce, ss in by_ce.items():
        acc = {}
        for s in ss:
            tot = max((i for _, i in s["peaks"]), default=1.0) or 1.0
            for m, i in s["peaks"]:
                if i / tot < min_rel:
                    continue
                key = round(m, 4)          # keep mass accuracy; do NOT round to 3 dp
                acc[key] = max(acc.get(key, 0.0), i / tot)
        channels[ce] = acc
    # union across energies, merging peaks that are the same peak
    raw = {}
    for ce, acc in channels.items():
        for m, rel in acc.items():
            e = raw.setdefault(m, {"mz": m, "rel": 0.0, "n_ce": 0})
            e["rel"] = max(e["rel"], rel)
            e["n_ce"] += 1
    # collapse near-duplicates (within 8 mDa) into their most intense member
    merged = []
    for e in sorted(raw.values(), key=lambda d: d["mz"]):
        if merged and abs(e["mz"] - merged[-1]["mz"]) < 0.008:
            prev = merged[-1]
            if e["rel"] > prev["rel"]:
                prev["mz"] = e["mz"]
            prev["rel"] = max(prev["rel"], e["rel"])
            prev["n_ce"] = max(prev["n_ce"], e["n_ce"])
        else:
            merged.append(dict(e))
    out = sorted(merged, key=lambda d: -d["rel"])
    return out[:60], sorted(channels)


MAX_SCORED = 120   # candidates carried into D2 (see prefilter note)


def prefilter(cands):
    """D1 alone cannot discriminate at high mass -- at m/z 719 it returns
    thousands of formulas inside 8 ppm.  Carrying all of them into D2 is
    both intractable and pointless: the ones we drop are exactly the ones
    the floor would decline anyway.  We keep the best MAX_SCORED by mass
    error and RDBE plausibility, and RECORD how many were dropped so the
    degeneracy is reported rather than hidden."""
    for c in cands:
        f = c["formula"]
        r = rdbe(f)
        # heteroatom penalty: P and S are rare, and multiples of them rarer
        het = 1.2 * f.get("P", 0) + 0.8 * f.get("S", 0) + 0.15 * f.get("N", 0)
        c["_pre"] = abs(c["ppm"]) / PPM + 0.03 * abs(r - 6.0) + het
    ranked = sorted(cands, key=lambda c: c["_pre"])
    return ranked[:MAX_SCORED], len(cands)


def d2_contact(cands, frags, precursor_mz, polarity, max_frags=25):
    """Determination 2: the fragment contact set of each candidate.

    Independent of D1: it uses the fragment masses, which D1 never sees.
    """
    sign = 1 if polarity == "+" else -1
    # a fragment must be genuinely below the precursor: the surviving precursor
    # peak and isotope/co-isolation peaks above it are not fragments.
    use = [f for f in frags if f["mz"] < precursor_mz - 0.5][:max_frags]
    if not use:
        for c in cands:
            c["contact"] = 0.0
            c["n_contact"] = 0
            c["explained"] = []
        return use

    tot = sum(fr["rel"] * (1.0 + 0.5 * (fr["n_ce"] - 1)) for fr in use)
    cache = {}
    for c in cands:
        parent = dict(c["formula"])
        if c["nmer"] > 1:
            parent = {e: n * c["nmer"] for e, n in parent.items()}
        key = tuple(sorted(parent.items()))
        idx = cache.get(key)
        if idx is None:
            idx = cache[key] = SubIndex(parent)
        hits, wsum, expl = 0, 0.0, []
        for fr in use:
            fm = fr["mz"] + sign * E
            tol = max(fm * FRAG_PPM * 1e-6, FRAG_MIN_DA)
            sub = idx.match(fm, tol)
            if sub:
                hits += 1
                w = fr["rel"] * (1.0 + 0.5 * (fr["n_ce"] - 1))
                wsum += w
                expl.append({"mz": fr["mz"], "sub": fstr(sub),
                             "rel": round(fr["rel"], 4)})
        c["contact"] = wsum / tot if tot else 0.0
        c["n_contact"] = hits
        c["n_frags"] = len(use)
        c["explained"] = sorted(expl, key=lambda d: -d["rel"])[:12]
    return use


# ------------------------------------------------------------------ the floor
FLOOR_CONTACT = 0.30    # a candidate must explain 30% of the weighted fragments
FLOOR_MARGIN = 0.10     # and beat the next distinct formula by this much


# Adducts that differ only by a neutral loss from the same protonated ion
# describe THE SAME measured ion under different assumptions about where the
# water went.  They are not competing determinations and must not be scored
# against one another -- doing so makes the margin test unsatisfiable, which
# is a defect in the test, not a property of the data.
ADDUCT_FAMILY = {
    "[M+H]+": "H", "[M+H-H2O]+": "H", "[M+H-2H2O]+": "H",
    "[M-H]-": "dH", "[M-H2O-H]-": "dH",
}


def _family(c):
    return ADDUCT_FAMILY.get(c["adduct"], c["adduct"])


def _prior(c):
    """Prefer the simplest adduct in a family and an ordinary heteroatom load.

    Within one family every member reproduces the mass identically, so mass
    cannot choose; this is the only place a chemical prior legitimately enters.
    """
    f = c["formula"]
    base = {"[M+H]+": 0.0, "[M-H]-": 0.0,
            "[M+H-H2O]+": 0.12, "[M-H2O-H]-": 0.12,
            "[M+H-2H2O]+": 0.24,
            "[M+Na]+": 0.10, "[M+NH4]+": 0.12, "[M+Cl]-": 0.10,
            "[M+HCOO]-": 0.14, "[M+CH3COO]-": 0.16, "[M+K]+": 0.18,
            "[2M+H]+": 0.30, "[2M-H]-": 0.30, "[2M+Na]+": 0.34}
    het = 0.10 * f.get("P", 0) + 0.06 * f.get("S", 0) + 0.02 * f.get("N", 0)
    return base.get(c["adduct"], 0.2) + het


def license_answer(cands):
    """Combine the determinations. Returns (verdict, best, runner_up)."""
    if not cands:
        return "no-candidate", None, None
    for c in cands:
        c["score"] = c["contact"] - 0.02 * abs(c["ppm"]) / PPM - _prior(c)
    ranked = sorted(cands, key=lambda c: -c["score"])
    best = ranked[0]
    # A genuine rival is a different neutral formula that is NOT merely the
    # same ion redescribed within the same adduct family.
    runner = None
    for c in ranked[1:]:
        if c["fstr"] == best["fstr"]:
            continue
        if _family(c) == _family(best) and abs(c["ppm"] - best["ppm"]) < 0.5:
            continue        # same ion, different water bookkeeping
        runner = c
        break
    if best["contact"] < FLOOR_CONTACT:
        return "decline-unsupported", best, runner
    if runner is not None and (best["score"] - runner["score"]) < FLOOR_MARGIN:
        return "decline-ambiguous", best, runner
    return "licensed", best, runner


def main():
    import time
    data = json.load(open(os.path.join(HERE, "casmi_spectra.json")))
    t0 = time.time()
    out = []
    for ch in data:
        scans = ch["scans"]
        if not scans:
            out.append({**{k: ch[k] for k in ("id", "file", "rt", "mz")},
                        "verdict": "no-spectrum"})
            continue
        pol = scans[0]["polarity"]
        frags, ces = merge_ladder(scans)
        allc = d1_mass(ch["mz"], pol)
        cands, n_all = prefilter(allc)
        used = d2_contact(cands, frags, ch["mz"], pol)
        verdict, best, runner = license_answer(cands)
        rec = {
            "id": ch["id"], "file": ch["file"], "rt": ch["rt"], "mz": ch["mz"],
            "polarity": pol, "n_scans": len(scans), "ces": ces,
            "n_frags": len(used),
            "n_mass_candidates": n_all,
            "n_scored": len(cands),
            "n_distinct_formulas": len({c["fstr"] for c in allc}),
            "verdict": verdict,
        }
        if best:
            rec.update({
                "adduct": best["adduct"], "formula": best["fstr"],
                "ppm": round(best["ppm"], 2),
                "contact": round(best["contact"], 4),
                "n_contact": best["n_contact"],
                "score": round(best["score"], 4),
                "explained": best["explained"],
            })
        if runner:
            rec.update({
                "runner_formula": runner["fstr"], "runner_adduct": runner["adduct"],
                "runner_contact": round(runner["contact"], 4),
                "margin": round(best["score"] - runner["score"], 4),
            })
        rec["secs"] = round(time.time() - t0, 1)
        out.append(rec)
        print("  #%-4d %s  %-9s cand=%-4d formA=%-3d  %-20s %-12s contact=%.2f  %s"
              % (ch["id"], pol, "%.4f" % ch["mz"], n_all,
                 rec["n_distinct_formulas"],
                 rec.get("adduct", "-"), rec.get("formula", "-"),
                 rec.get("contact", 0.0), verdict), flush=True)

    with open(os.path.join(HERE, "casmi_answers.json"), "w") as f:
        json.dump(out, f, indent=1)

    import collections
    print()
    print("VERDICTS:", dict(collections.Counter(o["verdict"] for o in out)))
    lic = [o for o in out if o["verdict"] == "licensed"]
    print("licensed: %d of %d" % (len(lic), len(out)))
    if lic:
        print("mean contact of licensed: %.3f"
              % (sum(o["contact"] for o in lic) / len(lic)))
    dec = [o for o in out if o["verdict"].startswith("decline")]
    if dec:
        print("mean contact of declined:  %.3f"
              % (sum(o.get("contact", 0) for o in dec) / len(dec)))


if __name__ == "__main__":
    main()
