"""Extract the MS2 spectra matching each CASMI challenge from the mzML files we hold.

For each challenge (file, RT, precursor m/z) we collect every MS2 scan whose
isolation target is within a tolerance of the stated precursor and whose scan
start time is within an RT window.  Each precursor is fragmented at three
stepped collision energies (35/45/65 eV), so a challenge normally yields a
ladder of three or more scans.

Output: casmi_spectra.json  -- one record per challenge, with every matching
scan kept separately (never averaged: the energy ladder is the signal).
"""
import base64
import json
import os
import struct
import sys
import zlib
import xml.etree.ElementTree as ET

import xlsx

NS = "{http://psi.hupo.org/ms/mzml}"
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "ucdavis")

MZ_TOL = 0.01      # Da on the isolation target
RT_TOL = 0.35      # minutes


def cvparams(el):
    return {c.get("name"): c.get("value") for c in el.iter(NS + "cvParam")}


def decode(bda):
    p = cvparams(bda)
    raw = bda.find(NS + "binary").text or ""
    b = base64.b64decode(raw)
    if "zlib compression" in p:
        b = zlib.decompress(b)
    fmt = "d" if "64-bit float" in p else "f"
    n = len(b) // struct.calcsize(fmt)
    return struct.unpack("<" + fmt * n, b)


def load_challenges():
    rows = xlsx.read_sheet(
        os.path.join(DATA, "MetSoc2022_CASMI_Workshop_Challenges_Priority_1-250_040522_Share.xlsx"))
    ch = []
    for r in rows[2:]:
        if len(r) >= 4 and r[0].strip():
            ch.append({
                "id": int(r[0]),
                "file": r[1].strip(),
                "rt": float(r[2]),
                "mz": float(r[3]),
                "priority": r[4] if len(r) > 4 else "",
            })
    # errata: challenge 81 uses a corrected file/mz
    corr = xlsx.read_sheet(
        os.path.join(DATA, "MetSoc2022_CASMI_Workshop_Corrected Challenge_Priority_81_05192022_Share.xlsx"))
    for r in corr[2:]:
        if len(r) >= 4 and r[0].strip():
            cid = int(r[0])
            for c in ch:
                if c["id"] == cid:
                    c["file"], c["rt"], c["mz"] = r[1].strip(), float(r[2]), float(r[3])
    return ch


def scan_file(path, wants):
    """wants: list of challenge dicts for this file. Returns id -> [scans]."""
    out = {w["id"]: [] for w in wants}
    for _, el in ET.iterparse(path, events=("end",)):
        if el.tag != NS + "spectrum":
            continue
        p = cvparams(el)
        if int(p.get("ms level", 0)) != 2:
            el.clear()
            continue
        scan = el.find(NS + "scanList/" + NS + "scan")
        rt = float(cvparams(scan).get("scan start time", "nan"))
        prec = el.find(NS + "precursorList/" + NS + "precursor")
        if prec is None:
            el.clear()
            continue
        pi = prec.find(NS + "selectedIonList/" + NS + "selectedIon")
        pp = cvparams(pi)
        pmz = float(pp.get("selected ion m/z", "nan"))
        hits = [w for w in wants
                if abs(pmz - w["mz"]) <= MZ_TOL and abs(rt - w["rt"]) <= RT_TOL]
        if hits:
            act = cvparams(prec.find(NS + "activation"))
            ce = act.get("collision energy")
            arrs = el.findall(NS + "binaryDataArrayList/" + NS + "binaryDataArray")
            mz = decode(arrs[0])
            inten = decode(arrs[1])
            rec = {
                "rt": rt,
                "precursor_mz": pmz,
                "charge": pp.get("charge state"),
                "ce": float(ce) if ce else None,
                "polarity": "+" if "positive scan" in p else "-",
                "peaks": [[round(m, 5), round(i, 1)] for m, i in zip(mz, inten)],
            }
            for w in hits:
                out[w["id"]].append(rec)
        el.clear()
    return out


def main():
    ch = load_challenges()
    have = {f[:-5] for f in os.listdir(DATA) if f.endswith(".mzml")}
    todo = [c for c in ch if c["file"] in have]
    byfile = {}
    for c in todo:
        byfile.setdefault(c["file"], []).append(c)

    print("challenges answerable from files on disk: %d" % len(todo))
    results = {}
    for fn in sorted(byfile):
        path = os.path.join(DATA, fn + ".mzml")
        got = scan_file(path, byfile[fn])
        n = sum(1 for v in got.values() if v)
        print("  %-20s %2d wanted, %2d matched, %3d scans"
              % (fn, len(byfile[fn]), n, sum(len(v) for v in got.values())))
        for cid, scans in got.items():
            results[cid] = scans

    out = []
    for c in todo:
        out.append({**c, "scans": results.get(c["id"], [])})
    with open(os.path.join(HERE, "casmi_spectra.json"), "w") as f:
        json.dump(out, f)

    matched = [o for o in out if o["scans"]]
    print()
    print("MATCHED %d of %d challenges" % (len(matched), len(out)))
    print("unmatched ids:", [o["id"] for o in out if not o["scans"]])
    ces = sorted({s["ce"] for o in matched for s in o["scans"]})
    print("collision energies:", ces)


if __name__ == "__main__":
    main()
