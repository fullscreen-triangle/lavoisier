"""
nist_msp.py --- reader for the NIST-format MSP text libraries shipped in
oxford/public.

The binary .INU/.DBU forms of the same library are index files for the
NIST search executable and carry no documented layout; the .MSP export is
the same content in the vendor's text interchange format and is what the
suite reads. Nothing here is specific to a paper: this module only parses
and exposes spectra, so that an experiment can build its own graph from
them without also owning the file format.
"""
from __future__ import annotations

import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))

DEFAULT_MSP = os.path.join(
    REPO, "oxford", "public", "ac_cac_lib2020_msp",
    "AC_CAC_MSLibrary2020_V1D1B.msp")

_NUM = re.compile(r"^\s*[0-9]")


def read_msp(path=None, limit=None):
    """Parse an MSP file into a list of spectrum dicts.

    Keys: name, formula, precursor_mz, exact_mass, collision_energy,
    ion_mode, instrument, peaks [(mz, intensity, annotation)].
    Records missing a formula or peaks are returned as-is; callers filter.
    """
    path = path or DEFAULT_MSP
    out, cur = [], None
    with open(path, encoding="utf8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n").rstrip("\r")
            if line.startswith("Name:"):
                if cur is not None:
                    out.append(cur)
                    if limit and len(out) >= limit:
                        return out
                cur = {"name": line.split(":", 1)[1].strip(), "peaks": []}
                continue
            if cur is None:
                continue
            if not line.strip():
                continue
            if _NUM.match(line):
                tok = line.split(None, 2)
                try:
                    mz, inten = float(tok[0]), float(tok[1])
                except (ValueError, IndexError):
                    continue
                ann = tok[2].strip().strip('"') if len(tok) > 2 else ""
                cur["peaks"].append((mz, inten, ann))
                continue
            if ":" not in line:
                continue
            key, val = line.split(":", 1)
            key, val = key.strip().lower(), val.strip()
            if key == "formula":
                cur["formula"] = val
            elif key == "precursormz":
                cur["precursor_mz"] = _f(val)
            elif key == "exactmass":
                cur["exact_mass"] = _f(val)
            elif key == "collision_energy":
                cur["collision_energy"] = val
            elif key == "ion_mode":
                cur["ion_mode"] = val
            elif key == "instrument_type":
                cur["instrument"] = val
            elif key == "num peaks":
                cur["declared_peaks"] = _i(val)
    if cur is not None:
        out.append(cur)
    return out


def _f(s):
    try:
        return float(s)
    except ValueError:
        return None


def _i(s):
    try:
        return int(s)
    except ValueError:
        return None


def usable(specs, min_peaks=3):
    """Spectra carrying everything an experiment needs."""
    return [s for s in specs
            if s.get("formula") and s.get("precursor_mz")
            and len(s.get("peaks", [])) >= min_peaks]


def by_compound(specs):
    """Group spectra by formula. Each formula is one library compound
    measured at several collision energies; the group is the set of
    contacts that compound has been put through."""
    g = {}
    for s in specs:
        g.setdefault(s["formula"], []).append(s)
    return g


def fragment_signature(spec, tol=0.01, min_rel=0.01):
    """The set of fragment m/z a spectrum commits, binned at `tol` and
    thresholded at `min_rel` of base peak. Bins are absolute so that two
    spectra of the same compound at different collision energies share
    bin identifiers exactly."""
    if not spec["peaks"]:
        return frozenset()
    base = max(p[1] for p in spec["peaks"]) or 1.0
    q = 1.0 / tol
    return frozenset(int(round(p[0] * q)) for p in spec["peaks"]
                     if p[1] / base >= min_rel)


def compound_signature(group, **kw):
    """Union of fragment signatures across a compound's spectra."""
    out = set()
    for s in group:
        out |= fragment_signature(s, **kw)
    return frozenset(out)
