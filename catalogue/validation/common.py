"""
Shared harness for the catalogue validation suite.

One module per paper, named expN_<slug>.py, writing results/expN_<slug>.json.
The convention matches publications/peptide-mass-invariance/validation.

Design rule, inherited from that suite and enforced here by the
`Expectation` type: no experiment decides its expectation from the data.
Every expectation is registered BEFORE the measurement is taken, so the
recorded artefact carries the prediction and the outcome side by side.

Every test is paired with a control. A test whose control shows it cannot
separate the framework's prediction from a null is reported as
NON-DISCRIMINATING and excluded rather than counted as a pass. This rule
is stated in the protocol section of the instrument paper and applies to
the whole suite.
"""
from __future__ import annotations

import io
import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
MEDIUM = "__medium__"


# =====================================================================
#  Expectations registered before measurement
# =====================================================================

@dataclass
class Expectation:
    """A prediction, stated before the number it is compared against."""
    label: str
    prediction: str
    paper_ref: str = ""
    failure_mode: str = ""

    # filled at check time
    observed: Any = None
    passed: bool | None = None
    detail: str = ""
    discriminating: bool | None = None

    def check(self, ok: bool, observed: Any, detail: str = "") -> "Expectation":
        self.passed = bool(ok)
        self.observed = observed
        self.detail = detail
        return self

    def non_discriminating(self, observed: Any, detail: str) -> "Expectation":
        """The control shows the statistic cannot separate the prediction
        from a null. Excluded from the pass count rather than counted."""
        self.discriminating = False
        self.observed = observed
        self.detail = detail
        return self


class Experiment:
    """Collects expectations and writes one JSON artefact."""

    def __init__(self, name: str, paper: str, question: str):
        self.name = name
        self.paper = paper
        self.question = question
        self.expectations: list[Expectation] = []
        self.records: dict[str, Any] = {}
        self.notes: list[str] = []
        self._t0 = time.perf_counter()

    def expect(self, label: str, prediction: str, paper_ref: str = "",
               failure_mode: str = "") -> Expectation:
        e = Expectation(label, prediction, paper_ref, failure_mode)
        self.expectations.append(e)
        return e

    def record(self, key: str, value: Any) -> Any:
        self.records[key] = value
        return value

    def note(self, text: str) -> None:
        self.notes.append(text)

    # -------------------------------------------------------------- out

    def summary(self) -> dict:
        graded = [e for e in self.expectations if e.discriminating is not False]
        excluded = [e for e in self.expectations if e.discriminating is False]
        passed = [e for e in graded if e.passed]
        failed = [e for e in graded if e.passed is False]
        return {
            "graded": len(graded),
            "passed": len(passed),
            "failed": len(failed),
            "non_discriminating": len(excluded),
            "verdict": ("PASS" if graded and not failed
                        else "FAIL" if failed else "NO-VERDICT"),
        }

    def write(self) -> str:
        payload = {
            "experiment": self.name,
            "paper": self.paper,
            "question": self.question,
            "summary": self.summary(),
            "expectations": [asdict(e) for e in self.expectations],
            "records": self.records,
            "notes": self.notes,
            "environment": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "elapsed_s": round(time.perf_counter() - self._t0, 4),
            },
        }
        if not os.path.isdir(RESULTS):
            os.makedirs(RESULTS)
        path = os.path.join(RESULTS, f"{self.name}.json")
        with io.open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
        return path

    def report(self) -> None:
        s = self.summary()
        print("")
        print(f"=== {self.name} :: {self.paper} ===")
        print(f"    {self.question}")
        for e in self.expectations:
            if e.discriminating is False:
                mark = "SKIP"
            elif e.passed:
                mark = "PASS"
            elif e.passed is False:
                mark = "FAIL"
            else:
                mark = "----"
            print(f"  [{mark}] {e.label}: {e.detail}")
        print(f"  -> {s['passed']}/{s['graded']} graded, "
              f"{s['non_discriminating']} non-discriminating :: {s['verdict']}")


# =====================================================================
#  Numeric helpers
# =====================================================================

def mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def stdev(xs) -> float:
    xs = list(xs)
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def pearson(xs, ys) -> float:
    xs, ys = list(xs), list(ys)
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx > 0 and dy > 0 else 0.0


def close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


def rel_close(a: float, b: float, tol: float = 1e-6) -> bool:
    d = max(abs(a), abs(b), 1e-300)
    return abs(a - b) / d <= tol
