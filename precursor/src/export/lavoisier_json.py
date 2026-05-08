"""
Lavoisier web dashboard export.

Converts the output of the precursor analysis pipeline (SpectraReader +
SEntropyTransformer) into the PredictedRecord[] JSON schema consumed by
the web tool's crossfilter dashboard.

The mapping is:

  Pipeline concept            → Web schema field
  ──────────────────────────────────────────────────
  MS2_PR_mz                   → precursorMz
  apex intensity (MS1 xic)    → intensity
  SEntropyFeatures.mean_knowledge → sentropy.sk
  SEntropyFeatures.mean_time      → sentropy.st
  SEntropyFeatures.mean_entropy   → sentropy.se
  SEntropyFeatures.coordinate_entropy → partitionEntropy
  lipid DB match              → analyteClass, X, Y, neutralMass
  partition formulas          → n, l, m, s
  analyser observable         → observable
  MS2 fragment peaks          → ms2[]
"""

from __future__ import annotations

import json
import math
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Physical constants (same as lagrangian.js)
# ──────────────────────────────────────────────────────────────────────────────
AMU      = 1.66053907e-27
E_CHARGE = 1.602176634e-19

# ──────────────────────────────────────────────────────────────────────────────
# Lipid class database (ported from lipidomics.js)
# ──────────────────────────────────────────────────────────────────────────────
_C14 = 14.0156501
_C2  = 2.0156501

_LIPID_CLASSES: Dict[str, dict] = {
    "PC":  {"mass": lambda X, Y: _C14*X - _C2*Y + 285.0613929, "fa": 2,
            "X": (28, 44), "Y": (0, 6),  "pol": "+"},
    "PE":  {"mass": lambda X, Y: _C14*X - _C2*Y + 243.0144440, "fa": 2,
            "X": (28, 44), "Y": (0, 6),  "pol": "-"},
    "PS":  {"mass": lambda X, Y: _C14*X - _C2*Y + 287.0402737, "fa": 2,
            "X": (28, 44), "Y": (0, 6),  "pol": "-"},
    "PG":  {"mass": lambda X, Y: _C14*X - _C2*Y + 274.0399887, "fa": 2,
            "X": (28, 44), "Y": (0, 6),  "pol": "-"},
    "PI":  {"mass": lambda X, Y: _C14*X - _C2*Y + 362.0610177, "fa": 2,
            "X": (28, 44), "Y": (0, 6),  "pol": "-"},
    "SM":  {"mass": lambda X, Y: _C14*X - _C2*Y + 478.3909116, "fa": 1,
            "X": (14, 26), "Y": (0, 4),  "pol": "+"},
    "Cer": {"mass": lambda X, Y: _C14*X - _C2*Y + 313.2980,    "fa": 1,
            "X": (14, 26), "Y": (0, 4),  "pol": "-"},
    "TAG": {"mass": lambda X, Y: _C14*X - _C2*Y + 134.0942946, "fa": 3,
            "X": (42, 60), "Y": (0, 9),  "pol": "+"},
    "DAG": {"mass": lambda X, Y: _C14*X - _C2*Y + 92.0473470,  "fa": 2,
            "X": (28, 40), "Y": (0, 6),  "pol": "+"},
    "LPC": {"mass": lambda X, Y: _C14*X - _C2*Y + 271.0822178, "fa": 1,
            "X": (14, 24), "Y": (0, 4),  "pol": "+"},
    "CE":  {"mass": lambda X, Y: _C14*X - _C2*Y + 368.3443210, "fa": 1,
            "X": (14, 24), "Y": (0, 4),  "pol": "+"},
    "FA":  {"mass": lambda X, Y: _C14*X - _C2*Y - 0.000064,    "fa": 1,
            "X": (12, 24), "Y": (0, 6),  "pol": "-"},
}

_ADDUCTS: Dict[str, dict] = {
    "[M+H]+":    {"delta": +1.007276,  "z": +1, "pol": "+"},
    "[M+Na]+":   {"delta": +22.989218, "z": +1, "pol": "+"},
    "[M+NH4]+":  {"delta": +18.034374, "z": +1, "pol": "+"},
    "[M+K]+":    {"delta": +38.963158, "z": +1, "pol": "+"},
    "[M-H]-":    {"delta": -1.007276,  "z": -1, "pol": "-"},
    "[M+FA-H]-": {"delta": +44.997655, "z": -1, "pol": "-"},
    "[M+Cl]-":   {"delta": +34.969402, "z": -1, "pol": "-"},
}


def _build_lipid_db():
    entries = []
    for cls_key, cls in _LIPID_CLASSES.items():
        xlo, xhi = cls["X"]
        ylo, yhi = cls["Y"]
        for X in range(xlo, xhi + 1):
            max_y = min(yhi, (X - cls["fa"]) // 2)
            for Y in range(ylo, max_y + 1):
                entries.append({"class": cls_key, "X": X, "Y": Y,
                                 "mass": cls["mass"](X, Y)})
    entries.sort(key=lambda d: d["mass"])
    return entries


_DB = _build_lipid_db()
_DB_MASSES = np.array([d["mass"] for d in _DB])


def _annotate(neutral_mass: float, polarity: str, ppm: float = 5.0
              ) -> Optional[dict]:
    tol  = neutral_mass * ppm * 1e-6
    lo   = neutral_mass - tol
    hi   = neutral_mass + tol
    idx  = int(np.searchsorted(_DB_MASSES, lo))
    best, best_err = None, tol + 1
    while idx < len(_DB_MASSES) and _DB_MASSES[idx] <= hi:
        d   = _DB[idx]
        err = abs(_DB_MASSES[idx] - neutral_mass)
        if err < best_err and _LIPID_CLASSES[d["class"]]["pol"] == polarity:
            best_err, best = err, d
        idx += 1
    return best


def _neutral_from_mz(mz: float, adduct_key: str) -> float:
    a = _ADDUCTS[adduct_key]
    return mz * abs(a["z"]) - a["delta"] * abs(a["z"])


def _best_adduct_annotation(mz: float, polarity: str, ppm: float
                             ) -> Tuple[Optional[dict], str, float]:
    """Try all adducts for given polarity; return (annotation, adduct_key, neutral_mass)."""
    best_ann, best_key, best_nm = None, None, mz - _ADDUCTS[
        "[M+H]+" if polarity == "+" else "[M-H]-"]["delta"]
    for key, a in _ADDUCTS.items():
        if a["pol"] != polarity:
            continue
        nm  = _neutral_from_mz(mz, key)
        ann = _annotate(nm, polarity, ppm)
        if ann:
            best_ann, best_key, best_nm = ann, key, nm
            break  # first match wins (adducts ordered most → least common)
    return best_ann, (best_key or ("[M+H]+" if polarity == "+" else "[M-H]-")), best_nm


# ──────────────────────────────────────────────────────────────────────────────
# Partition coordinates (ported from virtualinstrument.js)
# ──────────────────────────────────────────────────────────────────────────────

def _principal_n(mass: float) -> int:
    return max(1, math.ceil(math.sqrt(mass / 162.0)))


def _angular_l(cls_key: str, X: int, Y: int, n: int) -> int:
    fa = _LIPID_CLASSES.get(cls_key, {}).get("fa", 1)
    return min(n - 1, Y + fa - 1)


def _magnetic_m(cls_key: str, X: int, Y: int, l: int) -> int:
    if l == 0:
        return 0
    h = (ord(cls_key[0]) * 31 + X * 7 + Y * 13) % (2 * l + 1)
    return h - l


def _spin_s(polarity: str) -> float:
    return 0.5 if polarity == "+" else -0.5


# ──────────────────────────────────────────────────────────────────────────────
# Analyser observables (ported from lagrangian.js)
# ──────────────────────────────────────────────────────────────────────────────

def _orbitrap_obs(mz: float, k_field: float = 1e12) -> dict:
    m_kg  = mz * AMU
    omega = math.sqrt((E_CHARGE * k_field) / m_kg)
    return {"omega": omega, "frequencyHz": omega / (2 * math.pi),
            "observable": "axialFrequency"}


def _tof_obs(mz: float, accel_v: float = 5000, L: float = 1.0) -> dict:
    m_kg = mz * AMU
    T = L * math.sqrt(m_kg / (2 * E_CHARGE * accel_v))
    return {"T": T, "observable": "flightTime", "unit": "s"}


def _fticr_obs(mz: float, B: float = 7.0) -> dict:
    m_kg    = mz * AMU
    omega_c = (E_CHARGE * B) / m_kg
    return {"omegaC": omega_c, "frequencyHz": omega_c / (2 * math.pi),
            "observable": "cyclotronFrequency"}


_OBS = {"orbitrap": _orbitrap_obs, "tof": _tof_obs, "fticr": _fticr_obs}


# ──────────────────────────────────────────────────────────────────────────────
# Apex intensity lookup
# ──────────────────────────────────────────────────────────────────────────────

def _apex_intensity(ms1_xic_df: pd.DataFrame, mz: float, rt: float,
                    ppm: float = 10.0, rt_window: float = 0.5) -> float:
    """Find highest MS1 intensity for `mz` within ±rt_window minutes."""
    if ms1_xic_df is None or ms1_xic_df.empty:
        return 1.0
    tol  = mz * ppm * 1e-6
    mask = (
        (ms1_xic_df["mz"].between(mz - tol, mz + tol)) &
        (ms1_xic_df["rt"].between(rt - rt_window, rt + rt_window))
    )
    sub  = ms1_xic_df.loc[mask, "i"]
    return float(sub.max()) if not sub.empty else 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Core translation
# ──────────────────────────────────────────────────────────────────────────────

def records_from_pipeline(
    scan_info_df: pd.DataFrame,
    spectra_dict: Dict[int, pd.DataFrame],
    ms1_xic_df: pd.DataFrame,
    polarity: str = "+",
    analyser: str = "orbitrap",
    ppm: float = 5.0,
    sentropy_transformer=None,
) -> List[dict]:
    """
    Convert the output of `extract_mzml()` + `SEntropyTransformer` into a
    list of PredictedRecord dicts that the web dashboard can consume.

    Args:
        scan_info_df:   from SpectraReader.extract_mzml()
        spectra_dict:   from SpectraReader.extract_mzml()
        ms1_xic_df:     from SpectraReader.extract_mzml()
        polarity:       "+" or "-"
        analyser:       "orbitrap" | "tof" | "fticr"
        ppm:            mass accuracy for lipid annotation
        sentropy_transformer: optional SEntropyTransformer instance; a default
                        one is created if None.

    Returns:
        List of PredictedRecord dicts.
    """
    # Lazy import so this module can be imported without sklearn if needed
    if sentropy_transformer is None:
        from ..core.EntropyTransformation import SEntropyTransformer
        sentropy_transformer = SEntropyTransformer()

    obs_fn = _OBS.get(analyser, _orbitrap_obs)

    # MS2 rows only
    ms2_rows = scan_info_df[scan_info_df["DDA_rank"] > 0]

    records = []
    for _, row in ms2_rows.iterrows():
        spec_idx  = int(row["spec_index"])
        pr_mz     = float(row["MS2_PR_mz"])
        rt        = float(row["scan_time"])   # minutes

        if pr_mz <= 0:
            continue

        # ── MS2 spectrum ───────────────────────────────────────────────────────
        spec_df   = spectra_dict.get(spec_idx)
        if spec_df is None or spec_df.empty:
            ms2_peaks = []
            mz_arr    = np.array([pr_mz])
            int_arr   = np.array([1.0])
        else:
            # column is 'i' from SpectraReader (renamed to 'intensity' in pipeline)
            int_col = "intensity" if "intensity" in spec_df.columns else "i"
            mz_arr  = spec_df["mz"].to_numpy(dtype=np.float64)
            int_arr = spec_df[int_col].to_numpy(dtype=np.float64)
            # top 50 fragments sorted by intensity
            order   = np.argsort(int_arr)[::-1][:50]
            ms2_peaks = [
                {"mz": float(mz_arr[k]), "intensity": float(int_arr[k]),
                 "label": f"f{j}", "type": "fragment"}
                for j, k in enumerate(order)
            ]

        # ── Apex intensity from MS1 ─────────────────────────────────────────
        intensity = _apex_intensity(ms1_xic_df, pr_mz, rt)

        # ── S-entropy via transformer ───────────────────────────────────────
        try:
            _, features = sentropy_transformer.transform_and_extract(
                mz_arr, int_arr, precursor_mz=pr_mz, rt=rt
            )
            sentropy = {
                "sk": float(np.clip(features.mean_knowledge,  0, 1)),
                "st": float(np.clip(features.mean_time,       0, 1)),
                "se": float(np.clip(features.mean_entropy,    0, 1)),
            }
            part_entropy = float(features.coordinate_entropy)
        except Exception:
            sentropy    = {"sk": 0.0, "st": 0.0, "se": 0.0}
            part_entropy = 0.0

        # ── Annotation ─────────────────────────────────────────────────────
        ann, adduct_key, neutral_mass = _best_adduct_annotation(pr_mz, polarity, ppm)
        cls_key   = ann["class"] if ann else "Unknown"
        X         = ann["X"]     if ann else 0
        Y         = ann["Y"]     if ann else 0
        n_mass    = ann["mass"]  if ann else neutral_mass

        # ── Partition coordinates ───────────────────────────────────────────
        n = _principal_n(n_mass)
        l = _angular_l(cls_key, X, Y, n)
        m = _magnetic_m(cls_key, X, Y, l)
        s = _spin_s(polarity)

        # ── Observable ─────────────────────────────────────────────────────
        obs = obs_fn(pr_mz)

        # ── Information bits ───────────────────────────────────────────────
        bits_precursor = math.log2(max(1.0, pr_mz)) + math.log2(max(1.0, intensity))
        bits_coord     = math.log2(max(1.0, 2 * n * n))
        bits_frags     = math.log2(max(1.0, len(ms2_peaks)))
        bits_total     = bits_precursor + bits_coord + bits_frags + 5

        records.append({
            "analyte":       f"{cls_key}({X}:{Y})" if ann else f"?({pr_mz:.3f})",
            "analyteClass":  cls_key,
            "X": X, "Y": Y,
            "composition":   {},
            "neutralMass":   n_mass,
            "adduct":        adduct_key,
            "adductAbbr":    adduct_key,
            "precursorMz":   pr_mz,
            "z":             1,
            "polarity":      polarity,
            "intensity":     intensity,
            "n": n, "l": l, "m": m, "s": s,
            "sentropy":      sentropy,
            "ternaryAddress": "",
            "analyserMode":  analyser,
            "observable":    obs,
            "shellDistribution": {},
            "partitionEntropy":  part_entropy,
            "ms1": [],
            "ms2": ms2_peaks,
            "peaksAll": ms2_peaks,
            "bitsTotal": bits_total,
            "sentropyVec": sentropy,
            # preserve pipeline metadata for downstream use
            "_rt_min":    rt,
            "_spec_index": spec_idx,
            "_dda_event": int(row["dda_event_idx"]),
        })

    return records


def export_lavoisier_json(
    scan_info_df: pd.DataFrame,
    spectra_dict: Dict[int, pd.DataFrame],
    ms1_xic_df: pd.DataFrame,
    output_path: Path,
    polarity: str = "+",
    analyser: str = "orbitrap",
    ppm: float = 5.0,
    sentropy_transformer=None,
    source_file: str = "",
) -> Path:
    """
    Full pipeline → .lavoisier.json export.

    Calls `records_from_pipeline()` then writes the JSON envelope.

    Returns:
        Path to the written file.
    """
    records = records_from_pipeline(
        scan_info_df, spectra_dict, ms1_xic_df,
        polarity=polarity,
        analyser=analyser,
        ppm=ppm,
        sentropy_transformer=sentropy_transformer,
    )

    envelope = {
        "version":    "1.0",
        "source":     "lavoisier-precursor",
        "sourceFile": str(source_file),
        "analyser":   analyser,
        "polarity":   polarity,
        "runDate":    datetime.datetime.utcnow().isoformat() + "Z",
        "records":    records,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(envelope, fh, separators=(",", ":"), allow_nan=False)

    return output_path
