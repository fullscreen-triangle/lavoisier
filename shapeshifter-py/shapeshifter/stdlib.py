"""
Shapeshifter standard library: the acquisition half.

Operations are keyed by qualified name and return (value, kind).
Kind is producer-determined, never inferred from shape
[Thm. 6.3, kind preservation].

Implemented for Experiment 0 (NCE invariance):

    lavoisier.acquire.read_msp      -> scans
    lavoisier.acquire.filter_scans  -> scans
    lavoisier.transform.sentropy    -> coords
    lavoisier.analyse.group_by      -> object
    lavoisier.analyse.separation    -> object
    lavoisier.analyse.drift         -> object
    lavoisier.analyse.baseline      -> object

The S-entropy definitions follow eq. (7.1)-(7.3) of the specification.
"""

from __future__ import annotations

import math
import re
from typing import Any, Callable

EPSILON = 1e-10


class RefusalError(Exception):
    """Global failure: the operation has no meaningful result.
    [Prop. 6.9(ii)]"""


# ---------------------------------------------------------------- MSP reading

_COMPOUND_KEY_RE = re.compile(r"isomerNO=(\d+)")


def _parse_msp(path: str, log: Callable[[str, str], None]) -> list[dict]:
    """Parse a NIST-format .msp file into scan records.  [Def. 7.4]"""
    scans: list[dict] = []
    cur: dict[str, Any] | None = None
    peaks: list[tuple[float, float]] = []
    in_peaks = False

    def flush():
        nonlocal cur, peaks
        if cur is not None:
            cur["peaks"] = peaks
            cur["n_peaks"] = len(peaks)
            scans.append(cur)
        cur, peaks = None, []

    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n").rstrip("\r")
            if not line.strip():
                continue
            if line.startswith("Name:"):
                flush()
                cur = {"name": line[5:].strip()}
                in_peaks = False
                continue
            if cur is None:
                continue
            if line.startswith("Num peaks:"):
                in_peaks = True
                continue
            if in_peaks:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        peaks.append((float(parts[0]), float(parts[1])))
                    except ValueError:
                        pass
                continue
            if ":" in line:
                k, v = line.split(":", 1)
                cur[k.strip()] = v.strip()
    flush()

    # Derive the fields the experiment needs.
    out = []
    for s in scans:
        comment = s.get("Comment", "")
        iso = _COMPOUND_KEY_RE.search(comment)
        nce = None
        m = re.search(r"NCE=(\d+)", s.get("Collision_energy", ""))
        if m:
            nce = int(m.group(1))
        try:
            prec = float(s.get("PrecursorMZ", "nan"))
        except ValueError:
            prec = float("nan")
        out.append({
            "scan_id": s.get("ID"),
            "name": s.get("name"),
            "formula": s.get("Formula"),
            "precursor_mz": prec,
            "exact_mass": _safe_float(s.get("ExactMass")),
            "nce": nce,
            "polarity": s.get("Ion_mode"),
            "ms_level": 2,
            "isomer": iso.group(1) if iso else "?",
            "peaks": s["peaks"],
            "n_peaks": s["n_peaks"],
            # Compound identity: formula + precursor + isomer index.
            "compound": f"{s.get('Formula')}|{s.get('PrecursorMZ')}|"
                        f"{iso.group(1) if iso else '?'}",
        })
    log("info", f"  parsed {len(out)} spectra from {path}")
    return out


def _safe_float(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def op_read_msp(args, env, ast, log):
    name = args.get("dataset")
    block = ast.datasets.get(name) if name else None
    if block is None:
        raise RefusalError(f"no dataset block named {name!r}")
    files = block.get("files") or []
    if not files:
        raise RefusalError(f"dataset {name!r} declares no files")

    min_peaks = args.get("min_peaks", block.get("min_peaks", 1))
    min_intensity = args.get("min_intensity", block.get("min_intensity", 0))

    scans: list[dict] = []
    for path in files:
        try:
            scans.extend(_parse_msp(path, log))
        except FileNotFoundError:
            # Local failure: a missing file leaves the run meaningful.
            log("warn", f"  file not found, skipped: {path}")

    if not scans:
        raise RefusalError("no spectra read from any declared file")

    kept = []
    for s in scans:
        pk = [(m, i) for m, i in s["peaks"] if i >= min_intensity]
        if len(pk) < min_peaks:
            continue
        s = dict(s)
        s["peaks"] = pk
        s["n_peaks"] = len(pk)
        kept.append(s)
    log("info", f"  admitted {len(kept)}/{len(scans)} spectra "
                f"(min_peaks={min_peaks}, min_intensity={min_intensity})")
    return kept, "scans"


def op_filter_scans(args, env, ast, log):
    scans = args.get("scans") or []
    before = len(scans)
    nce_in = args.get("nce_in")
    polarity = args.get("polarity")
    min_peaks = args.get("min_peaks")
    out = []
    for s in scans:
        if nce_in is not None and s.get("nce") not in nce_in:
            continue
        if polarity is not None and s.get("polarity") != polarity:
            continue
        if min_peaks is not None and s.get("n_peaks", 0) < min_peaks:
            continue
        out.append(s)
    log("info", f"  filter: {before} -> {len(out)} scans")
    return out, "scans"


# ---------------------------------------------------------------- S-entropy

def _sentropy_one(peaks, precursor_mz, alpha, beta, k_neighbors):
    """Per-spectrum S-entropy triple.  [eq. 7.1-7.3, Prop. 7.9 totality]"""
    mz = [p[0] for p in peaks]
    inten = [p[1] for p in peaks]
    n = len(mz)
    imax = max(inten) if inten else 1.0
    if imax <= 0:
        imax = 1.0
    ihat = [i / imax for i in inten]

    m_star = precursor_mz if (precursor_mz and precursor_mz == precursor_mz
                              and precursor_mz > 0) else max(mz)

    # S_k : Shannon self-information + mass term            [eq. 7.1]
    sk = [-math.log2(h + EPSILON) + alpha * (m / m_star)
          for m, h in zip(mz, ihat)]

    # S_t : Gaussian weighting about the spectral centroid  [eq. 7.2]
    mean_m = sum(mz) / n
    var = sum((m - mean_m) ** 2 for m in mz) / n
    sd = math.sqrt(var)
    if sd > 0:
        st = [math.exp(-beta * abs(m - mean_m) / sd) for m in mz]
    else:
        st = [1.0] * n                       # degenerate: sigma_m = 0

    # S_e : local k-neighbourhood Shannon entropy           [eq. 7.3]
    k = min(k_neighbors, n - 1)
    se = []
    for i in range(n):
        if k < 1:
            se.append(0.0)                   # degenerate: single peak
            continue
        order = sorted(range(n), key=lambda j: abs(mz[j] - mz[i]))[:k]
        tot = sum(ihat[j] for j in order)
        if tot <= 0:
            se.append(0.0)
            continue
        h = 0.0
        for j in order:
            p = ihat[j] / tot
            if p > 0:
                h -= p * math.log2(p)
        se.append(h)

    return (sum(sk) / n, sum(st) / n, sum(se) / n)


def op_sentropy(args, env, ast, log):
    scans = args.get("scans") or []
    alpha = args.get("alpha", 1.0)
    beta = args.get("beta", 1.0)
    k = args.get("k_neighbors", 5)
    out = []
    for s in scans:
        if not s["peaks"]:
            continue
        sk, st, se = _sentropy_one(s["peaks"], s.get("precursor_mz"),
                                   alpha, beta, k)
        out.append({
            "scan_id": s.get("scan_id"),
            "compound": s.get("compound"),
            "formula": s.get("formula"),
            "nce": s.get("nce"),
            "precursor_mz": s.get("precursor_mz"),
            "n_peaks": s.get("n_peaks"),
            "s_k": sk, "s_t": st, "s_e": se,
        })
    log("info", f"  sentropy: {len(out)} coordinate triples "
                f"(alpha={alpha}, beta={beta}, k={k})")
    return out, "coords"


# ---------------------------------------------------------------- analysis

def _dist(a, b, axes):
    return math.sqrt(sum((a[x] - b[x]) ** 2 for x in axes))


def _group(coords, key):
    g: dict[Any, list] = {}
    for c in coords:
        g.setdefault(c.get(key), []).append(c)
    return g


def op_group_by(args, env, ast, log):
    coords = args.get("coords") or []
    key = args.get("key", "compound")
    g = _group(coords, key)
    sizes = sorted((len(v) for v in g.values()))
    summary = {
        "key": key,
        "n_groups": len(g),
        "n_items": len(coords),
        "min_group": sizes[0] if sizes else 0,
        "median_group": sizes[len(sizes) // 2] if sizes else 0,
        "max_group": sizes[-1] if sizes else 0,
    }
    log("info", f"  group_by {key}: {summary['n_groups']} groups")
    return summary, "object"


def op_separation(args, env, ast, log):
    """Within- vs between-compound spread. The experiment's criterion."""
    coords = args.get("coords") or []
    key = args.get("key", "compound")
    axes = args.get("axes", ["s_k", "s_t", "s_e"])
    min_group = args.get("min_group", 2)

    groups = {k: v for k, v in _group(coords, key).items()
              if len(v) >= min_group}
    if len(groups) < 2:
        raise RefusalError(
            f"separation needs >=2 groups of size >={min_group}; "
            f"got {len(groups)}")

    # Within: mean pairwise distance inside each group.
    within = []
    centroids = {}
    for gk, items in groups.items():
        cen = {a: sum(i[a] for i in items) / len(items) for a in axes}
        centroids[gk] = cen
        ds = [_dist(items[i], items[j], axes)
              for i in range(len(items)) for j in range(i + 1, len(items))]
        if ds:
            within.append(sum(ds) / len(ds))

    # Between: mean pairwise distance between group centroids.
    keys = list(centroids)
    between = [_dist(centroids[keys[i]], centroids[keys[j]], axes)
               for i in range(len(keys)) for j in range(i + 1, len(keys))]

    mw = sum(within) / len(within) if within else 0.0
    mb = sum(between) / len(between) if between else 0.0
    ratio = (mb / mw) if mw > 0 else float("inf")

    result = {
        "key": key,
        "axes": axes,
        "n_groups": len(groups),
        "mean_within": mw,
        "mean_between": mb,
        "separation_ratio": ratio,
    }
    log("info", f"  separation: within={mw:.4f} between={mb:.4f} "
                f"ratio={ratio:.3f}")
    return result, "object"


def op_drift(args, env, ast, log):
    """Per-axis trend against a covariate. Tests whether the address
    is a property of the compound or of the spectrum."""
    coords = args.get("coords") or []
    over = args.get("over", "nce")
    axes = args.get("axes", ["s_k", "s_t", "s_e"])

    pts = [c for c in coords if c.get(over) is not None]
    if len(pts) < 3:
        raise RefusalError(f"drift needs >=3 points with {over!r}")

    xs = [float(c[over]) for c in pts]
    mx = sum(xs) / len(xs)
    sxx = sum((x - mx) ** 2 for x in xs)

    out = {"over": over, "n": len(pts), "axes": {}}
    for a in axes:
        ys = [c[a] for c in pts]
        my = sum(ys) / len(ys)
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        syy = sum((y - my) ** 2 for y in ys)
        slope = sxy / sxx if sxx > 0 else 0.0
        r = (sxy / math.sqrt(sxx * syy)) if sxx > 0 and syy > 0 else 0.0
        out["axes"][a] = {"slope": slope, "pearson_r": r, "r_squared": r * r}
        log("info", f"  drift {a} vs {over}: slope={slope:+.5f} r={r:+.3f}")
    return out, "object"


def _cosine(a, b, tol=0.01):
    """Cosine similarity of two peak lists, matched within tol Da."""
    if not a or not b:
        return 0.0
    used = set()
    dot = 0.0
    for m1, i1 in a:
        best, bj = tol, None
        for j, (m2, i2) in enumerate(b):
            if j in used:
                continue
            d = abs(m1 - m2)
            if d <= best:
                best, bj = d, j
        if bj is not None:
            used.add(bj)
            dot += i1 * b[bj][1]
    na = math.sqrt(sum(i * i for _, i in a))
    nb = math.sqrt(sum(i * i for _, i in b))
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


def op_baseline(args, env, ast, log):
    """Comparison method: raw-spectrum cosine similarity across the
    covariate, within compound. Established practice; known to degrade."""
    scans = args.get("scans") or []
    key = args.get("key", "compound")
    over = args.get("over", "nce")
    tol = args.get("tolerance", 0.01)
    min_group = args.get("min_group", 2)

    groups = {k: v for k, v in _group(scans, key).items()
              if len(v) >= min_group}
    if not groups:
        raise RefusalError("baseline needs at least one multi-spectrum group")

    within_sims, adjacent_sims = [], []
    for items in groups.values():
        items = sorted(items, key=lambda s: (s.get(over) or 0))
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                s = _cosine(items[i]["peaks"], items[j]["peaks"], tol)
                within_sims.append(s)
                if j == i + 1:
                    adjacent_sims.append(s)

    mean_all = sum(within_sims) / len(within_sims) if within_sims else 0.0
    mean_adj = sum(adjacent_sims) / len(adjacent_sims) if adjacent_sims else 0.0
    result = {
        "metric": "cosine_similarity",
        "key": key, "over": over, "tolerance": tol,
        "n_groups": len(groups),
        "n_pairs": len(within_sims),
        "mean_within_compound": mean_all,
        "mean_adjacent_level": mean_adj,
    }
    log("info", f"  baseline cosine: all-pairs={mean_all:.4f} "
                f"adjacent={mean_adj:.4f} over {len(within_sims)} pairs")
    return result, "object"


def op_shuffle_control(args, env, ast, log):
    """Negative control: permute the grouping label. The separation
    ratio must collapse toward 1."""
    import random
    coords = args.get("coords") or []
    key = args.get("key", "compound")
    seed = args.get("seed", 0)
    rng = random.Random(seed)
    labels = [c.get(key) for c in coords]
    rng.shuffle(labels)
    shuffled = [dict(c, **{key: lab}) for c, lab in zip(coords, labels)]
    res, _ = op_separation({**args, "coords": shuffled}, env, ast, log)
    res["control"] = "label_shuffle"
    res["seed"] = seed
    return res, "object"


# ---------------------------------------------------------------- registry

REGISTRY: dict[str, Callable] = {
    "lavoisier.acquire.read_msp": op_read_msp,
    "lavoisier.acquire.filter_scans": op_filter_scans,
    "lavoisier.transform.sentropy": op_sentropy,
    "lavoisier.analyse.group_by": op_group_by,
    "lavoisier.analyse.separation": op_separation,
    "lavoisier.analyse.drift": op_drift,
    "lavoisier.analyse.baseline": op_baseline,
    "lavoisier.analyse.shuffle_control": op_shuffle_control,
}
