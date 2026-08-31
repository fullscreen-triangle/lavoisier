"""
run_all.py --- run every validation experiment in the catalogue suite and
aggregate the verdicts.

Each experiment writes its own artefact to results/. This runner executes
them in order, collects the per-expectation verdicts, and prints one
table across all papers.

A suite-level FAIL is not a defect in the suite. Every FAIL is a claim
the experiments could not reproduce, and each is named here so that the
count is never reported without the reasons. Expectations marked
NON-DISCRIMINATING are excluded from the pass count rather than counted
as passes: a test whose control cannot separate the framework's
prediction from a null has measured nothing.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

# The suite spans two harnesses. The six in catalogue/validation are
# pre-registered with explicit controls; the three shipped inside the
# peptide paper's own directory predate that harness and record only a
# label -> bool summary with no control discipline. Both are run, and
# the aggregator reads both shapes, because retiring the older three
# would drop coverage the papers already claim.
NEW_DIR = HERE
OLD_DIR = os.path.abspath(os.path.join(
    HERE, "..", "publications", "peptide-mass-invariance", "validation"))

EXPERIMENTS = [
    (NEW_DIR, "exp1_instrument_ladder.py", "instrument-process-ladder"),
    (NEW_DIR, "exp2_observation_groups.py", "observation-groups"),
    (NEW_DIR, "exp3_coordinate_provenance.py", "coordinate-provenance"),
    (NEW_DIR, "exp4_runtime_graph.py", "runtime-graph"),
    (NEW_DIR, "exp5_sink_detection.py", "sink-detection"),
    (NEW_DIR, "exp6_peptide_mass_invariance.py", "peptide-mass-invariance"),
    (OLD_DIR, "exp1_graph_invariants.py", "peptide (legacy) graph"),
    (OLD_DIR, "exp2_collapse_rate.py", "peptide (legacy) collapse"),
    (OLD_DIR, "exp3_promiscuity_vs_parsimony.py", "peptide (legacy) promiscuity"),
]


def run(directory, script):
    r = subprocess.run([sys.executable, script], cwd=directory,
                       capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr


def artefact(directory, script):
    name = os.path.splitext(script)[0]
    p = os.path.join(directory, "results", name + ".json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf8") as fh:
        return json.load(fh)


def normalise(art):
    """Return (graded, non_discriminating) as lists of uniform dicts.

    Two artefact shapes exist. The new harness stores a list of
    expectation records under `expectations`, with `discriminating`
    set to False only where the experiment retired the test --- it is
    None on a graded one, so `is False` is the predicate. A truthiness
    test reads None as non-discriminating and silently retires the
    whole suite.

    The legacy harness stores `summary` as label -> bool with the claim
    text in `registered_expectations`. It has no notion of a retired
    test, so nothing there is ever reported as non-discriminating; that
    is a property of the older harness, not a finding about its claims.
    """
    if "expectations" in art:
        exps = art["expectations"]
        graded = [e for e in exps if e.get("discriminating") is not False]
        nd = [e for e in exps if e.get("discriminating") is False]
        return graded, nd
    reg = art.get("registered_expectations", {})
    graded = []
    for label, ok in art.get("summary", {}).items():
        meta = reg.get(label, {})
        graded.append({
            "label": label,
            "passed": bool(ok),
            "paper_ref": meta.get("theorem", ""),
            "detail": meta.get("claim", ""),
        })
    return graded, []


def main():
    only = sys.argv[1:]
    todo = [(d, s, p) for d, s, p in EXPERIMENTS if not only or
            any(o in s or o in p for o in only)]

    print("=" * 72)
    print("catalogue validation suite --- %d experiments" % len(todo))
    print("=" * 72)

    rows, failures, nondisc = [], [], []
    for directory, script, paper in todo:
        print("\n>>> %s  [%s]" % (script, paper))
        code, out, err = run(directory, script)
        if code != 0:
            print("    ERROR exit %d" % code)
            print((err or out).strip()[-1500:])
            rows.append((paper, script, None, None, None, "ERROR"))
            continue
        art = artefact(directory, script)
        if art is None:
            print("    no artefact written")
            rows.append((paper, script, None, None, None, "NO ARTEFACT"))
            continue
        graded, nd = normalise(art)
        passed = [e for e in graded if e["passed"]]
        failed = [e for e in graded if not e["passed"]]
        summ = art.get("summary", {})
        verdict = summ.get("verdict") if isinstance(summ, dict) else None
        if not isinstance(verdict, str):
            verdict = "PASS" if not failed else "FAIL"
        rows.append((paper, script, len(passed), len(graded), len(nd),
                     verdict))
        for e in failed:
            failures.append((paper, e["label"], e.get("paper_ref", ""),
                             e.get("detail", "")))
        for e in nd:
            nondisc.append((paper, e["label"], e.get("detail", "")))
        print("    %d/%d graded, %d non-discriminating :: %s"
              % (len(passed), len(graded), len(nd), verdict))

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print("%-28s %7s %7s %8s" % ("paper", "passed", "graded", "verdict"))
    print("-" * 72)
    tp = tg = tn = 0
    for paper, script, p, g, nd, v in rows:
        if p is None:
            print("%-28s %7s %7s %8s" % (paper[:28], "-", "-", v))
            continue
        tp += p
        tg += g
        tn += nd
        print("%-28s %7d %7d %8s" % (paper[:28], p, g, v))
    print("-" * 72)
    print("%-28s %7d %7d" % ("TOTAL", tp, tg))
    print("non-discriminating (excluded): %d" % tn)

    if failures:
        print("\n" + "=" * 72)
        print("CLAIMS NOT REPRODUCED (%d)" % len(failures))
        print("=" * 72)
        print("The new harness prints what was MEASURED. The legacy "
              "harness stores only the registered claim, so its lines "
              "below print the PREDICTION that failed, not the "
              "observation --- read its artefact for the numbers. In "
              "particular E4's text says the intersection falls below "
              "the prediction; the measurement is 2.56x ABOVE it.")
        for paper, label, ref, detail in failures:
            print("\n  %s :: %s" % (paper, label))
            if ref:
                print("    ref: %s" % ref)
            print("    %s" % _wrap(detail, 68, "    "))

    if nondisc:
        print("\n" + "=" * 72)
        print("NON-DISCRIMINATING (%d, excluded from the count)" % len(nondisc))
        print("=" * 72)
        for paper, label, detail in nondisc:
            print("\n  %s :: %s" % (paper, label))
            print("    %s" % _wrap(detail, 68, "    "))

    print("\nartefacts: %s" % RESULTS)
    return 0 if not failures else 1


def _wrap(text, width, indent):
    words, line, out = str(text).split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > width:
            out.append(line)
            line = w
        else:
            line = (line + " " + w).strip()
    if line:
        out.append(line)
    return ("\n" + indent).join(out)


if __name__ == "__main__":
    sys.exit(main())
