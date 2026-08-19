"""
Shapeshifter compiler: two-stage model.

    compile_stage(source) -> parses, checks structure, executes nothing
    execute_stage(ast)    -> runs phases against a fixed AST

Implements the operational semantics of specification section 6:

  - configuration <env, kinds, order, log>            [Def. 6.5]
  - argument resolution is one level deep             [Rem. 6.8]
  - declaration order is execution order              [Thm. 6.10]
  - phases share one environment, no scope            [Thm. 6.11]
  - workspace is append-only                          [Thm. 6.12]
  - compile diagnostics are source-only               [Thm. 6.16]
  - kind is producer-determined                       [Thm. 6.3]
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from .parser import AST, Assign, Call, ParseError, parse
from .stdlib import REGISTRY, RefusalError

STRUCTURAL_KINDS = ("scalar", "list", "object")
PRIMARY_PREFERENCE = ("records", "cells", "addresses", "coords",
                      "coherence", "features", "scans", "linkage")


@dataclass
class Diagnostic:
    severity: str          # "error" | "warning"
    message: str


@dataclass
class CompileResult:
    ok: bool
    ast: AST | None
    ir: dict
    terminal: list[dict] = field(default_factory=list)
    diagnostics: list[Diagnostic] = field(default_factory=list)


@dataclass
class Binding:
    name: str
    kind: str
    value: Any


@dataclass
class ExecuteResult:
    result: dict
    workspace: list[Binding]
    log: list[dict]
    terminal: list[dict]
    elapsed_s: float


# ---------------------------------------------------------------- kinds

def classify(fn: str | None, value: Any) -> str:
    """Kind assignment. Depends on the producing operation, not on the
    shape of the value.  [Def. 6.2]"""
    if fn is not None:
        return "object"          # replaced by the operation's declared kind
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "object"
    return "scalar"


# ---------------------------------------------------------------- stage 1

def compile_stage(source: str) -> CompileResult:
    """Parse and check structure. Executes no phase, opens no file.
    [Def. 6.13, Thm. 6.16]"""
    term: list[dict] = [{"stream": "stage", "text": "shapeshifter compile"}]
    diags: list[Diagnostic] = []
    t0 = time.perf_counter()

    try:
        ast = parse(source)
    except (ParseError, Exception) as e:      # noqa: BLE001 - reported, not raised
        term.append({"stream": "stderr", "text": f"error: parse failed - {e}"})
        diags.append(Diagnostic("error", f"Parse error: {e}"))
        return CompileResult(False, None, {}, term, diags)

    blocks = []
    if ast.objective:
        blocks.append(f"objective {ast.objective['name']}")
    blocks += [f"instrument {k}" for k in ast.instruments]
    blocks += [f"dataset {k}" for k in ast.datasets]
    blocks += [f"target_list {k}" for k in ast.target_lists]
    blocks += [f"validate {k}" for k in ast.validates]
    blocks += [f"phase {k}" for k in ast.phases]

    term.append({"stream": "stdout",
                 "text": f"parsed {len(ast.imports)} import(s), "
                         f"{len(blocks)} block(s)"})
    for b in blocks:
        term.append({"stream": "stdout", "text": f"  . {b}"})

    # Structural checks  [Def. 6.14]
    if not ast.objective:
        diags.append(Diagnostic("warning", "no objective block declared"))
        term.append({"stream": "stderr",
                     "text": "warning: no objective block declared"})
    if not ast.phases:
        diags.append(Diagnostic("warning",
                                "no phase block - nothing will execute"))
        term.append({"stream": "stderr",
                     "text": "warning: no phase block - nothing will execute"})

    # Effect and input audit -- no file is opened  [Prop. 6.18, Cor. 6.19]
    effects, inputs = [], []
    for stmts in list(ast.phases.values()) + list(ast.validates.values()):
        for s in stmts:
            call = s.value if isinstance(s, Assign) and s.is_call else (
                s if isinstance(s, Call) else None)
            if call is None:
                continue
            if call.fn not in effects:
                effects.append(call.fn)
            if call.fn not in REGISTRY:
                diags.append(Diagnostic("warning",
                                        f"unknown operation {call.fn!r}"))
                term.append({"stream": "stderr",
                             "text": f"warning: unknown operation {call.fn}"})
            ds = call.args.get("dataset")
            if ds and ds in ast.datasets:
                inputs += list(ast.datasets[ds].get("files") or [])

    if effects:
        term.append({"stream": "stdout",
                     "text": f"effects: {', '.join(effects)}"})
    if inputs:
        term.append({"stream": "stdout",
                     "text": f"inputs ({len(inputs)}, not opened): "
                             + ", ".join(inputs)})

    dt = (time.perf_counter() - t0) * 1000
    term.append({"stream": "stdout", "text": f"compiled in {dt:.1f} ms"})

    ir = ast.to_dict()
    ir["_audit"] = {"effects": effects, "inputs": inputs}
    return CompileResult(True, ast, ir, term, diags)


# ---------------------------------------------------------------- stage 2

def _resolve(env: dict, v: Any) -> Any:
    """Argument resolution, one level deep.  [Rem. 6.8]"""
    if isinstance(v, str) and v in env:
        return env[v]
    return v


def execute_stage(ast: AST) -> ExecuteResult:
    """Run validates then phases, in source order."""
    env: dict[str, Any] = {}
    kinds: dict[str, str] = {}
    order: list[str] = []
    log: list[dict] = []
    term: list[dict] = [{"stream": "stage", "text": "shapeshifter run"}]
    t0 = time.perf_counter()

    def emit(level: str, message: str):
        log.append({"level": level, "message": message})

    if ast.objective:
        emit("info", f"Objective: {ast.objective['name']}")
        tgt = (ast.objective.get("fields") or {}).get("target")
        if tgt:
            emit("info", f"  {tgt}")

    def run_stmts(label: str, stmts: list):
        emit("info", f"Phase: {label}")
        for s in stmts:
            if not isinstance(s, Assign):
                continue
            if not s.is_call:
                env[s.target] = s.value
                if s.target not in order:
                    order.append(s.target)
                kinds[s.target] = classify(None, s.value)
                continue

            call: Call = s.value
            op = REGISTRY.get(call.fn)
            if op is None:
                emit("error", f"  unknown operation {call.fn}")
                raise RefusalError(f"unknown operation {call.fn}")

            resolved = {k: _resolve(env, v) for k, v in call.args.items()}
            emit("info", f"  {s.target} = {call.fn}(...)")
            value, kind = op(resolved, env, ast, emit)

            env[s.target] = value
            if s.target not in order:                # [Cor. 6.13 rebinding]
                order.append(s.target)
            kinds[s.target] = kind                   # producer-determined

    for name, stmts in ast.validates.items():
        run_stmts(f"validate {name}", stmts)
    for name, stmts in ast.phases.items():
        run_stmts(name, stmts)

    workspace = [Binding(n, kinds.get(n, "scalar"), env[n])
                 for n in order if env.get(n) is not None]

    # Primary result selection  [Def. 6.15]
    result = {"kind": "empty", "data": None}
    for pref in PRIMARY_PREFERENCE:
        hit = next((b for b in workspace if b.kind == pref), None)
        if hit:
            result = {"kind": hit.kind, "name": hit.name, "data": hit.value}
            break
    else:
        if workspace:
            last = workspace[-1]
            result = {"kind": last.kind, "name": last.name, "data": last.value}

    for entry in log:
        term.append({"stream": "stderr" if entry["level"] in ("error", "warn")
                     else "stdout", "text": entry["message"]})

    dt = time.perf_counter() - t0
    term.append({"stream": "stdout",
                 "text": f"workspace: "
                         + ", ".join(f"{b.name}:{b.kind}" for b in workspace)})
    term.append({"stream": "stdout", "text": f"finished in {dt*1000:.1f} ms"})

    return ExecuteResult(result, workspace, log, term, dt)


# ---------------------------------------------------------------- JSON out

def _jsonable(v: Any, max_items: int = 2000) -> Any:
    """Make a workspace value JSON-serialisable, truncating bulk arrays
    but recording that truncation happened."""
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, dict):
        return {k: _jsonable(x, max_items) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        if len(v) > max_items:
            return {"_truncated": True, "_n": len(v),
                    "_head": [_jsonable(x, max_items) for x in v[:max_items]]}
        return [_jsonable(x, max_items) for x in v]
    return str(v)


def run(source: str, out_dir: str | None = None,
        program_name: str = "program") -> dict:
    """Full pipeline: compile, execute, and write JSON artefacts."""
    import os

    comp = compile_stage(source)
    payload: dict[str, Any] = {
        "program": program_name,
        "compile": {
            "ok": comp.ok,
            "terminal": comp.terminal,
            "diagnostics": [{"severity": d.severity, "message": d.message}
                            for d in comp.diagnostics],
            "ir": comp.ir,
        },
    }

    if not comp.ok or comp.ast is None:
        payload["execute"] = None
    else:
        ex = execute_stage(comp.ast)
        payload["execute"] = {
            "elapsed_s": ex.elapsed_s,
            "terminal": ex.terminal,
            "log": ex.log,
            "result": {"kind": ex.result.get("kind"),
                       "name": ex.result.get("name")},
            "workspace": [
                {"name": b.name, "kind": b.kind, "value": _jsonable(b.value)}
                for b in ex.workspace
            ],
        }

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"{program_name}.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        if payload["execute"]:
            for b in payload["execute"]["workspace"]:
                p = os.path.join(out_dir, f"{program_name}.{b['name']}.json")
                with open(p, "w", encoding="utf-8") as fh:
                    json.dump({"name": b["name"], "kind": b["kind"],
                               "value": b["value"]}, fh, indent=2)
    return payload
