"""
Shapeshifter parser: .ss source -> AST.

Implements the lexical structure and grammar of the language
specification (sections 3 and 4):

  - vacuous lines (blank / comment-only) are discarded  [Def. 3.2]
  - comment stripping precedes vacuity testing          [Rem. 3.3]
  - bracket-balance continuation joins logical lines    [Def. 3.5]
  - six block forms, field blocks vs statement blocks   [Def. 4.1, 4.2]
  - all call arguments are named                        [Def. 4.3]
  - value language with depth-zero comma splitting      [Def. 4.4, Lem. 4.6]
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

OPEN, CLOSE = "[({", "])}"
QUOTES = ('"', "'")

FIELD_BLOCKS = ("objective", "instrument", "dataset", "target_list")
STMT_BLOCKS = ("phase", "validate")
LADDER_BLOCKS = ("ladder",)


class ParseError(Exception):
    """Raised when the source cannot be parsed; the compile stage reports it."""


# ---------------------------------------------------------------- values

def _split_depth0(raw: str) -> list[str]:
    """Split on commas at bracket depth zero.  [Lemma 4.6]"""
    parts, depth, start = [], 0, 0
    for i, c in enumerate(raw):
        if c in OPEN:
            depth += 1
        elif c in CLOSE:
            depth -= 1
        elif c == "," and depth == 0:
            parts.append(raw[start:i])
            start = i + 1
    parts.append(raw[start:])
    return [p for p in parts if p.strip()]


def _parse_object(raw: str) -> dict:
    inner = raw.strip()
    if inner.startswith("{"):
        inner = inner[1:]
    if inner.endswith("}"):
        inner = inner[:-1]
    obj: dict[str, Any] = {}
    for pair in _split_depth0(inner):
        ci = pair.find(":")
        if ci >= 0:
            obj[pair[:ci].strip()] = parse_value(pair[ci + 1:].strip())
    return obj


def _parse_object_array(raw: str) -> list[dict]:
    objs, depth, start = [], 0, 0
    for i, c in enumerate(raw):
        if c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                objs.append(_parse_object(raw[start:i + 1]))
    return objs


def parse_value(raw: Any) -> Any:
    """Value parsing function P.  [Def. 4.5, clauses V1-V7]"""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:                                                    # V1
        return None
    if len(s) >= 2 and s[0] == s[-1] and s[0] in QUOTES:         # V2
        return s[1:-1]
    if s == "true":                                              # V3
        return True
    if s == "false":
        return False
    if s.startswith("[") and s.endswith("]"):                    # V4
        inner = s[1:-1].strip()
        if not inner:
            return []
        if inner.startswith("{"):
            return _parse_object_array(inner)
        return [parse_value(v) for v in _split_depth0(inner)]
    if s.startswith("{") and s.endswith("}"):                    # V5
        return _parse_object(s)
    if re.fullmatch(r"[+-]?\d+", s):                             # V6
        return int(s)
    try:
        return float(s)
    except ValueError:
        pass
    return s                                                     # V7 bare word


# ---------------------------------------------------------------- AST

@dataclass
class Call:
    fn: str
    args: dict[str, Any]


@dataclass
class Assign:
    target: str
    value: Any            # Call when is_call, else a literal
    is_call: bool


@dataclass
class Rung:
    """One contact. `power` is a resolution increment in [0,1), not a
    displacement --- see Definition 3.2 of the instrument paper."""
    name: str
    power: float


@dataclass
class Ladder:
    """An ordered contact sequence with an optional declared target."""
    name: str
    toward: str | None
    rungs: list = field(default_factory=list)
    require: dict | None = None      # {"metric":..., "op":..., "value":...}


@dataclass
class AST:
    imports: list[str] = field(default_factory=list)
    objective: dict | None = None
    instruments: dict[str, dict] = field(default_factory=dict)
    datasets: dict[str, dict] = field(default_factory=dict)
    target_lists: dict[str, dict] = field(default_factory=dict)
    phases: dict[str, list] = field(default_factory=dict)
    validates: dict[str, list] = field(default_factory=dict)
    ladders: dict[str, Ladder] = field(default_factory=dict)

    def to_dict(self) -> dict:
        def stmt(s):
            if isinstance(s, Assign):
                v = ({"type": "call", "fn": s.value.fn, "args": s.value.args}
                     if s.is_call else {"type": "value", "value": s.value})
                return {"type": "assign", "target": s.target, "value": v}
            return {"type": "call", "fn": s.fn, "args": s.args}

        return {
            "imports": self.imports,
            "objective": self.objective,
            "instruments": self.instruments,
            "datasets": self.datasets,
            "target_lists": self.target_lists,
            "phases": {k: [stmt(s) for s in v] for k, v in self.phases.items()},
            "validates": {k: [stmt(s) for s in v]
                          for k, v in self.validates.items()},
            "ladders": {
                k: {"name": L.name, "toward": L.toward,
                    "rungs": [{"name": r.name, "power": r.power}
                              for r in L.rungs],
                    "require": L.require}
                for k, L in self.ladders.items()
            },
        }


# ---------------------------------------------------------------- lines

@dataclass
class Line:
    num: int
    indent: int
    body: str


def _logical_lines(source: str) -> list[Line]:
    """Strip comments, drop vacuous lines, join by bracket balance."""
    raw: list[Line] = []
    for i, ln in enumerate(source.split("\n")):
        body = re.sub(r"//.*$", "", ln).strip()
        if body:
            raw.append(Line(i + 1, len(ln) - len(ln.lstrip()), body))

    def bal(s: str) -> int:
        return sum(c in OPEN for c in s) - sum(c in CLOSE for c in s)

    out: list[Line] = []
    i = 0
    while i < len(raw):
        cur = raw[i]
        b = bal(cur.body)
        if b > 0:
            combined, j = cur.body, i + 1
            while j < len(raw) and b > 0:
                combined += " " + raw[j].body
                b += bal(raw[j].body)
                j += 1
            out.append(Line(cur.num, cur.indent, combined))
            i = j
        else:
            out.append(cur)
            i += 1
    return out


def _fields(lines: list[Line], i: int, base: int) -> tuple[dict, int]:
    out: dict[str, Any] = {}
    while i < len(lines) and lines[i].indent > base:
        m = re.match(r"^(\w+)\s*:\s*(.*)$", lines[i].body)
        if m:
            out[m.group(1)] = parse_value(m.group(2))
        i += 1
    return out, i


def _args(raw: str) -> dict:
    """All arguments are named.  [Def. 4.3, Rem. 4.7]"""
    args: dict[str, Any] = {}
    for part in _split_depth0(raw):
        ci = part.find(":")
        if ci >= 0:
            args[part[:ci].strip()] = parse_value(part[ci + 1:].strip())
    return args


RUNG_RE = re.compile(r"^rung\s+(\w+)\s+at\s+([-+0-9.eE]+)$")
REQUIRE_RE = re.compile(r"^require\s+(\w+)\s*(>=|<=|==|>|<)\s*([-+0-9.eE]+)$")


def _ladder_body(lines: list[Line], i: int, base: int,
                 name: str, toward: str | None) -> tuple[Ladder, int]:
    """Read a ladder block.

    Two statement forms only:
        rung <name> at <power>
        require <metric> <op> <value>

    A rung's power is a resolution increment; the parser enforces the
    half-open range [0,1) because a rung of resolution 1 would collapse
    the ambiguity to the floor in one contact, which Lemma 3.3 excludes.
    """
    lad = Ladder(name=name, toward=toward)
    while i < len(lines) and lines[i].indent > base:
        t = lines[i].body
        m = RUNG_RE.match(t)
        if m:
            power = float(m.group(2))
            if not (0.0 <= power < 1.0):
                raise ParseError(
                    f"line {lines[i].num}: rung {m.group(1)!r} has resolution "
                    f"{power}, outside [0,1)")
            lad.rungs.append(Rung(m.group(1), power))
            i += 1
            continue
        m = REQUIRE_RE.match(t)
        if m:
            lad.require = {"metric": m.group(1), "op": m.group(2),
                           "value": float(m.group(3))}
            i += 1
            continue
        raise ParseError(f"line {lines[i].num}: unrecognised ladder "
                         f"statement {t!r}")
    return lad, i


def _statements(lines: list[Line], i: int, base: int) -> tuple[list, int]:
    out: list[Any] = []
    while i < len(lines) and lines[i].indent > base:
        t = lines[i].body
        m = re.match(r"^(\w+)\s*=\s*(.+)$", t, re.S)
        if m:
            target, expr = m.group(1), m.group(2).strip()
            c = re.match(r"^([\w.]+)\s*\((.*)\)$", expr, re.S)
            if c:
                out.append(Assign(target, Call(c.group(1), _args(c.group(2))), True))
            else:
                out.append(Assign(target, parse_value(expr), False))
        else:
            c = re.match(r"^([\w.]+)\s*\((.*)\)$", t, re.S)
            if c:
                out.append(Call(c.group(1), _args(c.group(2))))
        i += 1
    return out, i


def parse(source: str) -> AST:
    """Parse .ss source into an AST."""
    lines = _logical_lines(source)
    ast, i = AST(), 0
    while i < len(lines):
        ln = lines[i]

        if ln.body.startswith("import "):
            ast.imports.append(ln.body[7:].strip())
            i += 1
            continue

        lm = re.match(r"^ladder\s+(\w+)(?:\s+toward\s+(\w+))?\s*:?$",
                      ln.body)
        if lm:
            lad, i = _ladder_body(lines, i + 1, ln.indent,
                                  lm.group(1), lm.group(2))
            ast.ladders[lad.name] = lad
            continue

        m = re.match(r"^(\w+)\s+(\w+)\s*:$", ln.body)
        if m and m.group(1) in FIELD_BLOCKS:
            kw, name = m.group(1), m.group(2)
            body, i = _fields(lines, i + 1, ln.indent)
            if kw == "objective":
                ast.objective = {"name": name, "fields": body}
            elif kw == "instrument":
                ast.instruments[name] = body
            elif kw == "dataset":
                ast.datasets[name] = body
            else:
                ast.target_lists[name] = {"name": name, **body}
            continue

        if m and m.group(1) in STMT_BLOCKS:
            kw, name = m.group(1), m.group(2)
            body, i = _statements(lines, i + 1, ln.indent)
            (ast.phases if kw == "phase" else ast.validates)[name] = body
            continue

        i += 1
    return ast
