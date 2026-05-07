"""
export-partition — process an mzML file through the partition Lagrangian and
export a .lavoisier.json file that the Lavoisier web dashboard can import.

Usage:
    lavoisier export-partition samples/plasma_pos.mzML
    lavoisier export-partition *.mzML --output results/ --analyser orbitrap --ppm 5

Algorithm
---------
1. Parse mzML (DDA or full-scan) with pymzml.
2. For each MS2 scan: use the precursor as the feature apex; link fragment ions.
   For high-intensity MS1-only features (above --ms1-threshold): include them too.
3. Annotate each feature against an in-silico lipid database (12 classes, ±ppm).
4. Compute partition coordinates (n, ℓ, m, s) from the partition Lagrangian.
5. Compute S-entropy (Sₖ, Sₜ, Sₑ) from the combined spectrum.
6. Write {version, source, analyser, records} JSON envelope.

The JSON schema exactly mirrors the web tool's PredictedRecord[] so the same
crossfilter dashboard visualises it without any conversion.
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

console = Console()

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
AMU      = 1.66053907e-27    # kg
E_CHARGE = 1.602176634e-19   # C

# ---------------------------------------------------------------------------
# Adduct table  (delta_mz = mz_observed − neutral_mass for z = 1)
# ---------------------------------------------------------------------------
ADDUCTS: Dict[str, dict] = {
    "[M+H]+":    {"delta": +1.007276,  "z": +1, "polarity": "+"},
    "[M+Na]+":   {"delta": +22.989218, "z": +1, "polarity": "+"},
    "[M+NH4]+":  {"delta": +18.034374, "z": +1, "polarity": "+"},
    "[M+K]+":    {"delta": +38.963158, "z": +1, "polarity": "+"},
    "[M-H]-":    {"delta": -1.007276,  "z": -1, "polarity": "-"},
    "[M+FA-H]-": {"delta": +44.997655, "z": -1, "polarity": "-"},
    "[M+Cl]-":   {"delta": +34.969402, "z": -1, "polarity": "-"},
    "[M+OAc]-":  {"delta": +59.013305, "z": -1, "polarity": "-"},
}


def mz_from_neutral(mass: float, adduct_key: str) -> float:
    a = ADDUCTS[adduct_key]
    return (mass + a["delta"] * abs(a["z"])) / abs(a["z"])


def neutral_from_mz(mz: float, adduct_key: str, z: int = 1) -> float:
    a = ADDUCTS[adduct_key]
    return mz * abs(a["z"]) - a["delta"] * abs(a["z"])


# ---------------------------------------------------------------------------
# Lipid class mass formulas (ported from web/src/lib/experiment/lipidomics.js)
# ---------------------------------------------------------------------------
_C14 = 14.0156501
_C2  = 2.0156501

LIPID_CLASSES: Dict[str, dict] = {
    "PC":  {"mass": lambda X, Y: _C14 * X - _C2 * Y + 285.0613929, "fa": 2,
            "X": (28, 44), "Y": (0, 6),  "pol": "+"},
    "PE":  {"mass": lambda X, Y: _C14 * X - _C2 * Y + 243.0144440, "fa": 2,
            "X": (28, 44), "Y": (0, 6),  "pol": "-"},
    "PS":  {"mass": lambda X, Y: _C14 * X - _C2 * Y + 287.0402737, "fa": 2,
            "X": (28, 44), "Y": (0, 6),  "pol": "-"},
    "PG":  {"mass": lambda X, Y: _C14 * X - _C2 * Y + 274.0399887, "fa": 2,
            "X": (28, 44), "Y": (0, 6),  "pol": "-"},
    "PI":  {"mass": lambda X, Y: _C14 * X - _C2 * Y + 362.0610177, "fa": 2,
            "X": (28, 44), "Y": (0, 6),  "pol": "-"},
    "SM":  {"mass": lambda X, Y: _C14 * X - _C2 * Y + 478.3909116, "fa": 1,
            "X": (14, 26), "Y": (0, 4),  "pol": "+"},
    "Cer": {"mass": lambda X, Y: _C14 * X - _C2 * Y + 313.2980,    "fa": 1,
            "X": (14, 26), "Y": (0, 4),  "pol": "-"},
    "TAG": {"mass": lambda X, Y: _C14 * X - _C2 * Y + 134.0942946, "fa": 3,
            "X": (42, 60), "Y": (0, 9),  "pol": "+"},
    "DAG": {"mass": lambda X, Y: _C14 * X - _C2 * Y + 92.0473470,  "fa": 2,
            "X": (28, 40), "Y": (0, 6),  "pol": "+"},
    "LPC": {"mass": lambda X, Y: _C14 * X - _C2 * Y + 271.0822178, "fa": 1,
            "X": (14, 24), "Y": (0, 4),  "pol": "+"},
    "CE":  {"mass": lambda X, Y: _C14 * X - _C2 * Y + 368.3443210, "fa": 1,
            "X": (14, 24), "Y": (0, 4),  "pol": "+"},
    "FA":  {"mass": lambda X, Y: _C14 * X - _C2 * Y - 0.000064,    "fa": 1,
            "X": (12, 24), "Y": (0, 6),  "pol": "-"},
}


def _build_lipid_db() -> List[dict]:
    """Pre-compute all (class, X, Y, mass) entries for fast annotation."""
    db = []
    for cls_key, cls in LIPID_CLASSES.items():
        xlo, xhi = cls["X"]
        ylo, yhi = cls["Y"]
        for X in range(xlo, xhi + 1):
            max_y = min(yhi, (X - cls["fa"]) // 2)
            for Y in range(ylo, max_y + 1):
                mass = cls["mass"](X, Y)
                db.append({"class": cls_key, "X": X, "Y": Y, "mass": mass})
    db.sort(key=lambda d: d["mass"])
    return db


_LIPID_DB: List[dict] = _build_lipid_db()
_DB_MASSES: np.ndarray = np.array([d["mass"] for d in _LIPID_DB])


def annotate(neutral_mass: float, polarity: str, ppm: float = 5.0
             ) -> Optional[dict]:
    """Return the best lipid annotation for a neutral mass within ppm tolerance."""
    tol = neutral_mass * ppm * 1e-6
    lo, hi = neutral_mass - tol, neutral_mass + tol
    idx = np.searchsorted(_DB_MASSES, lo)
    best_err = tol + 1
    best = None
    while idx < len(_DB_MASSES) and _DB_MASSES[idx] <= hi:
        d = _LIPID_DB[idx]
        err = abs(_DB_MASSES[idx] - neutral_mass)
        if err < best_err and LIPID_CLASSES[d["class"]]["pol"] == polarity:
            best_err = err
            best = d
        idx += 1
    return best


# ---------------------------------------------------------------------------
# Partition coordinates  (ported from web/src/lib/experiment/virtualinstrument.js)
# ---------------------------------------------------------------------------

def principal_coordinate(mass: float) -> int:
    return max(1, math.ceil(math.sqrt(mass / 162.0)))


def angular_coordinate(cls_key: str, X: int, Y: int, n: int) -> int:
    fa = LIPID_CLASSES.get(cls_key, {}).get("fa", 1)
    complexity = Y + fa - 1
    return min(n - 1, complexity)


def magnetic_coordinate(cls_key: str, X: int, Y: int, l: int) -> int:
    if l == 0:
        return 0
    h = (ord(cls_key[0]) * 31 + X * 7 + Y * 13) % (2 * l + 1)
    return h - l


def spin_coordinate(polarity: str) -> float:
    return 0.5 if polarity == "+" else -0.5


# ---------------------------------------------------------------------------
# Analyser observables  (ported from web/src/lib/partition/lagrangian.js)
# ---------------------------------------------------------------------------

def orbitrap_observable(mz: float, k_field: float = 1e12) -> dict:
    m_kg  = mz * AMU
    omega = math.sqrt((E_CHARGE * k_field) / m_kg)
    return {"omega": omega, "frequencyHz": omega / (2 * math.pi),
            "observable": "axialFrequency"}


def tof_observable(mz: float, accel_v: float = 5000,
                   flight_length: float = 1.0) -> dict:
    m_kg = mz * AMU
    T = flight_length * math.sqrt(m_kg / (2 * E_CHARGE * accel_v))
    return {"T": T, "observable": "flightTime", "unit": "s"}


def fticr_observable(mz: float, B: float = 7.0) -> dict:
    m_kg   = mz * AMU
    omega_c = (E_CHARGE * B) / m_kg
    return {"omegaC": omega_c, "frequencyHz": omega_c / (2 * math.pi),
            "observable": "cyclotronFrequency"}


def quadrupole_observable(mz: float, dc: float = 100, rf: float = 500,
                           omega: float = 1e6, r0: float = 5e-3) -> dict:
    m_kg  = mz * AMU
    denom = m_kg * r0 * r0 * omega * omega
    a = (8 * E_CHARGE * dc) / denom
    q = (4 * E_CHARGE * rf) / denom
    return {"a": a, "q": q, "stable": abs(a) < 0.237 and abs(q) < 0.908,
            "observable": "mathieu"}


_OBSERVABLE_FNS = {
    "orbitrap":   orbitrap_observable,
    "tof":        tof_observable,
    "fticr":      fticr_observable,
    "quadrupole": quadrupole_observable,
}


# ---------------------------------------------------------------------------
# S-entropy  (ported from web/src/lib/partition/sentropy.js)
# ---------------------------------------------------------------------------

OMEGA_REF_MAX  = 4401.0
OMEGA_REF_MIN  = 218.0
DELTA_HARMONIC = 0.05
P_MAX_HARMONIC = 8


def _sk(freqs: List[float]) -> float:
    n = len(freqs)
    if n == 0:
        return 0.0
    if n == 1:
        return freqs[0] / OMEGA_REF_MAX
    total = sum(freqs)
    if total == 0:
        return 0.0
    H = 0.0
    for f in freqs:
        p = f / total
        if p > 0:
            H -= p * math.log2(p)
    return H / math.log2(n)


def _st(freqs: List[float]) -> float:
    pos = [f for f in freqs if f > 0]
    if len(pos) < 2:
        return 0.0
    wmin, wmax = min(pos), max(pos)
    if wmin == wmax:
        return 0.0
    return math.log(wmax / wmin) / math.log(OMEGA_REF_MAX / OMEGA_REF_MIN)


def _se(freqs: List[float]) -> float:
    n = len(freqs)
    if n < 2:
        return 0.0
    n_pairs    = n * (n - 1) // 2
    n_harmonic = 0
    for i in range(n):
        for j in range(i + 1, n):
            a = max(freqs[i], freqs[j])
            b = min(freqs[i], freqs[j])
            if b <= 0:
                continue
            ratio   = a / b
            matched = False
            for p in range(1, P_MAX_HARMONIC + 1):
                if matched:
                    break
                for q in range(1, p + 1):
                    if abs(ratio - p / q) < DELTA_HARMONIC:
                        n_harmonic += 1
                        matched = True
                        break
    return n_harmonic / max(n_pairs, 1)


def sentropy_from_peaks(mz_arr, int_arr, top_n: int = 32) -> dict:
    if len(mz_arr) == 0:
        return {"sk": 0.0, "st": 0.0, "se": 0.0, "nPeaks": 0}
    pairs  = sorted(zip(int_arr, mz_arr), reverse=True)[:top_n]
    freqs  = [float(mz) for _, mz in pairs]
    return {
        "sk":     min(1.0, max(0.0, _sk(freqs))),
        "st":     min(1.0, max(0.0, _st(freqs))),
        "se":     min(1.0, max(0.0, _se(freqs))),
        "nPeaks": len(freqs),
    }


# ---------------------------------------------------------------------------
# Partition entropy  (Shannon entropy of the intensity distribution)
# ---------------------------------------------------------------------------

def partition_entropy_from_intensities(intensities) -> float:
    total = sum(intensities)
    if total == 0:
        return 0.0
    probs = [i / total for i in intensities if i > 0]
    return -sum(p * math.log(p) for p in probs)


# ---------------------------------------------------------------------------
# mzML parsing
# ---------------------------------------------------------------------------

def parse_mzml(path: str, ms1_min_intensity: float = 1000.0
               ) -> Tuple[List[dict], List[dict]]:
    """
    Return (ms1_scans, ms2_scans).

    ms1_scan = {rt, mz, intensity, polarity}   (arrays for mz/intensity)
    ms2_scan = {rt, precursor_mz, precursor_intensity, mz, intensity, polarity}
    """
    try:
        import pymzml
    except ImportError:
        console.print("[bold red]pymzml not installed.  pip install pymzml[/bold red]")
        raise typer.Exit(1)

    ms1, ms2 = [], []
    run = pymzml.run.Reader(path, MS1_Precision=5e-6, obo_version="4.1.33")

    for spec in run:
        try:
            level = spec.ms_level
            if level not in (1, 2):
                continue

            mz_raw, int_raw = spec.get_peaks()
            if len(mz_raw) == 0:
                continue
            mz_a  = np.asarray(mz_raw,  dtype=np.float64)
            int_a = np.asarray(int_raw, dtype=np.float64)

            # polarity
            pol = "+"
            if hasattr(spec, "scan_stats"):
                pid = spec.scan_stats.get("polarity", 1)
                pol = "-" if pid == -1 else "+"

            rt = float(spec.scan_time_in_minutes() or 0) * 60  # → seconds

            if level == 1:
                mask  = int_a >= ms1_min_intensity
                ms1.append({"rt": rt, "mz": mz_a[mask], "intensity": int_a[mask],
                             "polarity": pol})
            else:
                prec  = spec.selected_precursors
                if not prec:
                    continue
                pmz   = float(prec[0].get("mz", 0) or 0)
                pi    = float(prec[0].get("i",  0) or 0)
                if pmz <= 0:
                    continue
                ms2.append({"rt": rt, "precursor_mz": pmz, "precursor_intensity": pi,
                             "mz": mz_a, "intensity": int_a, "polarity": pol})
        except Exception:
            continue

    return ms1, ms2


def _apex_intensity_from_ms1(ms1_scans: List[dict], pmz: float, rt: float,
                              ppm: float, rt_window_s: float = 30.0) -> float:
    """Find the highest MS1 intensity for pmz within ±rt_window_s around rt."""
    tol  = pmz * ppm * 1e-6
    best = 0.0
    for sc in ms1_scans:
        if abs(sc["rt"] - rt) > rt_window_s:
            continue
        mz_a  = sc["mz"]
        int_a = sc["intensity"]
        mask  = np.abs(mz_a - pmz) <= tol
        if mask.any():
            best = max(best, float(int_a[mask].max()))
    return best


# ---------------------------------------------------------------------------
# Record assembly
# ---------------------------------------------------------------------------

def build_record(
    precursor_mz: float,
    intensity: float,
    ms2_peaks: List[dict],      # [{"mz": float, "intensity": float, "label": str, "type": str}]
    polarity: str,
    annotation: Optional[dict],
    analyser: str,
    analyser_cfg: dict,
) -> dict:
    cls_key  = annotation["class"] if annotation else "Unknown"
    X        = annotation["X"]     if annotation else 0
    Y        = annotation["Y"]     if annotation else 0
    n_mass   = annotation["mass"]  if annotation else precursor_mz
    adduct   = _guess_adduct(precursor_mz, n_mass, polarity)

    # Partition coordinates
    n = principal_coordinate(n_mass)
    l = angular_coordinate(cls_key, X, Y, n) if annotation else 0
    m = magnetic_coordinate(cls_key, X, Y, l) if annotation else 0
    s = spin_coordinate(polarity)

    # Observable
    obs_fn  = _OBSERVABLE_FNS.get(analyser, orbitrap_observable)
    obs     = obs_fn(precursor_mz, **analyser_cfg)

    # S-entropy from the MS2 spectrum (treat m/z as oscillatory proxy)
    ms2_mz   = [p["mz"]       for p in ms2_peaks]
    ms2_int  = [p["intensity"] for p in ms2_peaks]
    sentropy = sentropy_from_peaks(ms2_mz, ms2_int)

    # Information bits
    bits_precursor = math.log2(max(1.0, precursor_mz)) + math.log2(max(1.0, intensity))
    bits_coord     = math.log2(max(1.0, 2 * n * n))
    bits_frags     = math.log2(max(1.0, len(ms2_peaks)))
    bits_total     = bits_precursor + bits_coord + bits_frags + 5

    # Partition entropy from the fragment intensity distribution
    part_entropy = partition_entropy_from_intensities(ms2_int) if ms2_int else 0.0

    analyte_name = f"{cls_key}({X}:{Y})" if annotation else f"?({precursor_mz:.3f})"

    return {
        # identity
        "analyte":      analyte_name,
        "analyteClass": cls_key,
        "X":            X,
        "Y":            Y,
        "composition":  {},
        "neutralMass":  n_mass,
        "adduct":       adduct,
        "adductAbbr":   adduct,
        "precursorMz":  precursor_mz,
        "z":            1,
        "polarity":     polarity,
        "intensity":    intensity,
        # partition
        "n": n, "l": l, "m": m, "s": s,
        # S-entropy
        "sentropy":       sentropy,
        "ternaryAddress": "",
        # analyser
        "analyserMode": analyser,
        "observable":   obs,
        # spectra
        "ms1":      [],
        "ms2":      ms2_peaks,
        "peaksAll": ms2_peaks,
        # hierarchy / information
        "shellDistribution": {},
        "partitionEntropy":  part_entropy,
        "bitsTotal":         bits_total,
        "sentropyVec": sentropy,
    }


def _guess_adduct(mz: float, neutral_mass: float, polarity: str) -> str:
    """Pick the adduct whose delta best explains the mz/mass relationship."""
    best_key = "[M+H]+" if polarity == "+" else "[M-H]-"
    best_err = float("inf")
    for key, a in ADDUCTS.items():
        if a["polarity"] != polarity:
            continue
        predicted_mz = (neutral_mass + a["delta"] * abs(a["z"])) / abs(a["z"])
        err = abs(predicted_mz - mz)
        if err < best_err:
            best_err = err
            best_key = key
    return best_key


# ---------------------------------------------------------------------------
# Typer command
# ---------------------------------------------------------------------------

_standalone_app = typer.Typer(add_help_option=True)


@_standalone_app.command("export-partition")
def export_partition(
    input_files: List[Path] = typer.Argument(
        ..., help="mzML file(s) to process (glob-expanded by shell)"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o",
        help="Output path or directory.  Defaults to <input>.lavoisier.json"
    ),
    analyser: str = typer.Option(
        "orbitrap", "--analyser", "-a",
        help="Analyser type: orbitrap | tof | fticr | quadrupole"
    ),
    polarity_override: Optional[str] = typer.Option(
        None, "--polarity",
        help="Force polarity: + or -.  Auto-detected from mzML if omitted."
    ),
    ppm: float = typer.Option(
        5.0, "--ppm", help="Mass accuracy for lipid annotation (ppm)"
    ),
    ms1_threshold: float = typer.Option(
        1000.0, "--ms1-threshold",
        help="Minimum MS1 intensity to keep a feature"
    ),
    ms1_only: bool = typer.Option(
        False, "--ms1-only",
        help="Use MS1 features only (no MS2 required)"
    ),
    top_ms1: int = typer.Option(
        500, "--top-ms1",
        help="Maximum number of MS1-only features to include (by intensity)"
    ),
):
    """
    Process mzML file(s) → partition coordinates → .lavoisier.json

    The output can be drag-dropped onto the Lavoisier web dashboard
    (experiment page → Load .lavoisier.json) to visualise real MS data
    in the same crossfilter dashboard as the virtual instrument.
    """
    if analyser not in _OBSERVABLE_FNS:
        console.print(f"[red]Unknown analyser '{analyser}'.  "
                      f"Choose from: {', '.join(_OBSERVABLE_FNS)}[/red]")
        raise typer.Exit(1)

    analyser_cfg: dict = {}  # use defaults; could be exposed as options later

    all_records: List[dict] = []
    t_start = time.perf_counter()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as prog:
        task = prog.add_task("processing", total=len(input_files))

        for fpath in input_files:
            if not fpath.exists():
                console.print(f"[yellow]Warning: {fpath} not found, skipping.[/yellow]")
                prog.advance(task)
                continue

            prog.update(task, description=f"parsing {fpath.name}")
            ms1_scans, ms2_scans = parse_mzml(str(fpath), ms1_threshold)

            # ---- MS2-triggered features ----------------------------------------
            for sc in ms2_scans:
                pmz  = sc["precursor_mz"]
                pol  = polarity_override or sc["polarity"]
                pi   = sc["precursor_intensity"]
                # Prefer apex intensity from MS1 if available, else use precursor_intensity
                if pi <= 0 and ms1_scans:
                    pi = _apex_intensity_from_ms1(ms1_scans, pmz, sc["rt"], ppm)
                if pi <= 0:
                    pi = 1.0  # sentinel so the record is still included

                # MS2 peaks as fragment list (top 50 by intensity)
                mz_a, int_a = sc["mz"], sc["intensity"]
                if len(mz_a) > 0:
                    order = np.argsort(int_a)[::-1][:50]
                    ms2_peaks = [
                        {"mz": float(mz_a[i]), "intensity": float(int_a[i]),
                         "label": f"f{k}", "type": "fragment"}
                        for k, i in enumerate(order)
                    ]
                else:
                    ms2_peaks = []

                # Try multiple adducts for annotation
                ann = None
                for adt_key, adt in ADDUCTS.items():
                    if adt["polarity"] != pol:
                        continue
                    nm = neutral_from_mz(pmz, adt_key)
                    ann = annotate(nm, pol, ppm)
                    if ann:
                        break

                neutral_m = ann["mass"] if ann else pmz - ADDUCTS[
                    "[M+H]+" if pol == "+" else "[M-H]-"]["delta"]

                rec = build_record(
                    precursor_mz=pmz,
                    intensity=float(pi),
                    ms2_peaks=ms2_peaks,
                    polarity=pol,
                    annotation=ann,
                    analyser=analyser,
                    analyser_cfg=analyser_cfg,
                )
                all_records.append(rec)

            # ---- MS1-only features (if requested or no MS2 available) ----------
            if ms1_only or (not ms2_scans and ms1_scans):
                # Collect all peaks across MS1 scans, keep top-N unique m/z
                peak_map: Dict[int, Tuple[float, float, str]] = {}  # bin → (mz, int, pol)
                BIN_FACTOR = 1000  # 1 mDa bins
                for sc in ms1_scans:
                    pol = polarity_override or sc["polarity"]
                    for mz_val, int_val in zip(sc["mz"], sc["intensity"]):
                        b = int(mz_val * BIN_FACTOR)
                        if b not in peak_map or int_val > peak_map[b][1]:
                            peak_map[b] = (float(mz_val), float(int_val), pol)

                peaks_sorted = sorted(peak_map.values(), key=lambda x: -x[1])[:top_ms1]
                for pmz_val, pi_val, pol in peaks_sorted:
                    ann = None
                    for adt_key, adt in ADDUCTS.items():
                        if adt["polarity"] != pol:
                            continue
                        nm = neutral_from_mz(pmz_val, adt_key)
                        ann = annotate(nm, pol, ppm)
                        if ann:
                            break
                    rec = build_record(
                        precursor_mz=pmz_val,
                        intensity=pi_val,
                        ms2_peaks=[],
                        polarity=pol,
                        annotation=ann,
                        analyser=analyser,
                        analyser_cfg=analyser_cfg,
                    )
                    all_records.append(rec)

            prog.advance(task)

    if not all_records:
        console.print("[yellow]No records produced — check input files and thresholds.[/yellow]")
        raise typer.Exit(1)

    elapsed_ms = (time.perf_counter() - t_start) * 1000

    # ---- determine output path -----------------------------------------------
    if output is None:
        stem = input_files[0].stem if len(input_files) == 1 else "lavoisier_results"
        out_path = input_files[0].parent / f"{stem}.lavoisier.json"
    elif output.is_dir():
        stem = input_files[0].stem if len(input_files) == 1 else "lavoisier_results"
        out_path = output / f"{stem}.lavoisier.json"
    else:
        out_path = output

    envelope = {
        "version":   "1.0",
        "source":    "lavoisier-cli",
        "analyser":  analyser,
        "runDate":   _iso_now(),
        "elapsedMs": round(elapsed_ms, 1),
        "records":   all_records,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(envelope, fh, separators=(",", ":"), allow_nan=False)

    size_kb = out_path.stat().st_size / 1024
    console.print(
        f"[bold green]✓[/bold green]  {len(all_records)} records → "
        f"[cyan]{out_path}[/cyan]  "
        f"({size_kb:.0f} kB, {elapsed_ms:.0f} ms)"
    )


def _iso_now() -> str:
    import datetime
    return datetime.datetime.utcnow().isoformat() + "Z"


# ---------------------------------------------------------------------------
# Allow direct execution:  python -m lavoisier.cli.commands.export_partition
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    _standalone_app()
