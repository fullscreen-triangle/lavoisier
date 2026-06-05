import React, { useState, useRef, useCallback, useEffect } from "react";
import {
  Files, Search, GitBranch, Play, Blocks, Settings,
  ChevronRight, ChevronDown, X, Circle, FileCode2, FileJson,
  FileText, Folder, FolderOpen, Terminal as TerminalIcon,
  AlertCircle, Bell, PanelBottomClose, Check, Code2,
  Trash2, RefreshCw, Cpu, Zap,
} from "lucide-react";
import { runExperiment, summariseRecords } from "@/lib/experiment/virtualinstrument";
import { LIPID_CLASSES } from "@/lib/experiment/lipidomics";
import { PROTEIN_CLASSES } from "@/lib/experiment/proteomics";

/* ─── Theme ──────────────────────────────────────────────────────────────── */
const T = {
  titlebar: "#3c3c3c", activitybar: "#333333", activitybarFg: "#858585",
  activitybarFgActive: "#ffffff", sidebar: "#252526", sidebarFg: "#cccccc",
  sidebarHeader: "#bbbbbb", editor: "#1e1e1e", editorFg: "#d4d4d4",
  tabBar: "#252526", tabActive: "#1e1e1e", tabInactive: "#2d2d2d",
  tabFg: "#969696", tabFgActive: "#ffffff", border: "#3c3c3c",
  accent: "#0e639c", accentBright: "#007acc", statusBar: "#007acc",
  statusFg: "#ffffff", panel: "#1e1e1e", gutter: "#858585",
  lineActive: "#2a2d2e",
};

const CLASS_COLORS = {
  PC: "#5fa8d3", PE: "#e07a7a", PS: "#b388eb", PG: "#e493b3",
  PI: "#5dc0d8", SM: "#7cc77c", Cer: "#e6a456", TAG: "#cdc15c",
  DAG: "#a07a5e", LPC: "#a8b2bd", CE: "#9cc4d8", FA: "#e8c598",
  HSA: "#60a5fa", HBB: "#f87171", ENO1: "#34d399", CYCS: "#a78bfa", CASE: "#fb923c",
};

/* ─── Example files ──────────────────────────────────────────────────────── */
const SS_HELLO_LIPID = `\
// Predict a lipidomics spectral library for plasma analysis.
// The virtual instrument simulates forward MS using the partition Lagrangian.
// Each analyte maps to partition coordinates (n, ℓ, m, s) and S-entropy (Sk, St, Se).

import lavoisier.instrument

objective PlasmaLipidomics:
    target: "predict lipidomics library for plasma LC-MS/MS"
    success_criteria: "records > 100"

instrument OrbitrapFusion:
    analyzer: "orbitrap"
    polarity: "+"
    collision_energy: 25
    mz_window: [200, 1500]

phase Design:
    classes = ["PC", "PE", "SM", "TAG", "Cer"]

phase VirtualRun:
    records = lavoisier.instrument.run_experiment(
        classes: classes,
        polarity: "+",
        analyser: "orbitrap",
        collision_energy: 25
    )
`;

const SS_PROTEOMICS = `\
// Proteomics virtual experiment.
// Predicts tryptic peptide library from common plasma protein standards.
// Multiply-charged ESI adducts are selected by peptide mass.

import lavoisier.instrument

objective PlasmaProteomics:
    target: "predict tryptic peptide library from plasma proteins"
    success_criteria: "records > 50"

instrument OrbitrapFusion:
    analyzer: "orbitrap"
    polarity: "+"
    collision_energy: 28

phase Design:
    proteins = ["HSA", "HBB", "ENO1"]

phase VirtualRun:
    records = lavoisier.instrument.run_proteomics(
        proteins: proteins,
        length_min: 7,
        length_max: 20,
        mc_max: 1,
        polarity: "+",
        analyser: "orbitrap",
        collision_energy: 28
    )
`;

const SS_TEMPORAL = `\
// Temporal programming targeted acquisition.
// Compiles an m/z target list to ΔP timing cells via the Orbitrap cell compiler.
// τ_min = ℏ / δM_i  — the exact minimum acquisition time
// derived from the Partition Uncertainty Relation.

import lavoisier.cells

objective TargetedAcquisition:
    target: "compile plasma lipid targets to ΔP timing cells"

instrument OrbitrapXL:
    analyzer: "orbitrap"
    kappa: 1.0e12
    ref_frequency: 1.0e7
    timing_jitter: 2.0e-9

target_list PlasmaPanel:
    instrument: OrbitrapXL
    window_ppm: 5.0
    targets: [
        { name: "PC(32:0)", mz: 734.5694 },
        { name: "PC(34:1)", mz: 760.5851 },
        { name: "PE(34:1)", mz: 716.5259 },
        { name: "SM(d18:1)", mz: 703.5749 },
        { name: "LPC(16:0)", mz: 496.3396 },
        { name: "TAG(52:2)", mz: 879.7716 }
    ]

phase CompileCells:
    registry = lavoisier.cells.compile(
        target_list: PlasmaPanel
    )

validate TimingFeasibility:
    check_resolution_time(
        instrument: OrbitrapXL,
        window_ppm: 5.0
    )
`;

const SS_PARTITION = `\
// Partition address computation.
// Demonstrates the bijection Φ: ℤ⁺ → 𝒫 from the Partition Bijection Theorem.
// Each ion mass maps to a unique partition state (n, ℓ, m, s) —
// the four quantum numbers of the bounded phase space formulation.

import lavoisier.partition

objective PartitionMapping:
    target: "map lipid panel to partition states (n, ℓ, m, s)"

phase Compute:
    lipids = [
        { name: "PC(32:0)", mass: 733.5622, adduct: "[M+H]+" },
        { name: "PC(34:1)", mass: 759.5779, adduct: "[M+H]+" },
        { name: "PE(34:1)", mass: 715.5153, adduct: "[M+H]+" },
        { name: "SM(d18:1/16:0)", mass: 702.5676, adduct: "[M+H]+" },
        { name: "TAG(52:2)", mass: 878.7638, adduct: "[M+NH4]+" },
        { name: "LPC(16:0)", mass: 495.3325, adduct: "[M+H]+" }
    ]

    addresses = lavoisier.partition.compute_addresses(lipids)
`;

const README_CONTENT = `\
# Shapeshifter Sandbox

A live compiler for the Shapeshifter mass spectrometry DSL.
Select an example from the explorer and hit **Run**.

## Language structure

\`\`\`
import lavoisier.instrument

objective Name:
    target: "what you want to achieve"
    success_criteria: "records > 100"

instrument Name:
    analyzer: "orbitrap"   // tof | orbitrap | fticr | quadrupole
    polarity: "+"
    collision_energy: 25

phase PhaseA:
    variable = value

phase PhaseB:
    result = lavoisier.module.function(key: value, ...)
\`\`\`

## Available functions

| Function | Output |
|---|---|
| \`lavoisier.instrument.run_experiment(...)\` | Lipidomics PredictedRecord[] |
| \`lavoisier.instrument.run_proteomics(...)\` | Proteomics PredictedRecord[] |
| \`lavoisier.partition.compute_addresses(lipids)\` | Partition states (n,ℓ,m,s) |
| \`lavoisier.cells.compile(target_list)\` | ΔP timing cell registry |

## Key concepts

- **Partition Lagrangian**: all four analyzer equations from one Lagrangian
- **ΔP timing cells**: the natural MS acquisition variable is timing deviation
- **Partition states (n,ℓ,m,s)**: ion masses map to quantum-number coordinates
- **τ_min = ℏ/δM_i**: exact resolution–speed tradeoff from Partition Uncertainty
`;

const initialFiles = {
  examples: {
    type: "folder",
    children: {
      "hello_lipid.ss":         { type: "file", lang: "ss", content: SS_HELLO_LIPID },
      "proteomics_experiment.ss": { type: "file", lang: "ss", content: SS_PROTEOMICS },
      "temporal_acquisition.ss": { type: "file", lang: "ss", content: SS_TEMPORAL },
      "partition_addresses.ss":  { type: "file", lang: "ss", content: SS_PARTITION },
    },
  },
  "README.md": { type: "file", lang: "md", content: README_CONTENT },
};

/* ─── Parser ─────────────────────────────────────────────────────────────── */

function splitCommas(raw) {
  const parts = [];
  let depth = 0, start = 0;
  for (let i = 0; i < raw.length; i++) {
    const c = raw[i];
    if (c === "[" || c === "{" || c === "(") depth++;
    else if (c === "]" || c === "}" || c === ")") depth--;
    else if (c === "," && depth === 0) { parts.push(raw.slice(start, i)); start = i + 1; }
  }
  parts.push(raw.slice(start));
  return parts.filter(p => p.trim());
}

function parseObjectArray(raw) {
  const objects = [];
  let depth = 0, start = 0;
  for (let i = 0; i < raw.length; i++) {
    if (raw[i] === "{") { if (depth === 0) start = i; depth++; }
    else if (raw[i] === "}") {
      depth--;
      if (depth === 0) {
        const obj = {};
        splitCommas(raw.slice(start + 1, i)).forEach(pair => {
          const ci = pair.indexOf(":");
          if (ci >= 0) obj[pair.slice(0, ci).trim()] = parseValue(pair.slice(ci + 1).trim());
        });
        objects.push(obj);
      }
    }
  }
  return objects;
}

function parseValue(raw) {
  if (!raw) return null;
  const s = raw.trim();
  if ((s.startsWith('"') && s.endsWith('"')) || (s.startsWith("'") && s.endsWith("'")))
    return s.slice(1, -1);
  if (s === "true") return true;
  if (s === "false") return false;
  if (s.startsWith("[") && s.endsWith("]")) {
    const inner = s.slice(1, -1).trim();
    if (!inner) return [];
    return inner.startsWith("{") ? parseObjectArray(inner)
      : splitCommas(inner).map(v => parseValue(v.trim()));
  }
  const n = Number(s);
  if (!isNaN(n) && s !== "") return n;
  return s;
}

function parseNamedArgs(raw) {
  const args = {};
  if (!raw.trim()) return args;
  splitCommas(raw).forEach(part => {
    const ci = part.indexOf(":");
    if (ci >= 0) args[part.slice(0, ci).trim()] = parseValue(part.slice(ci + 1).trim());
  });
  return args;
}

function parseExpr(raw) {
  const s = raw.trim();
  const callM = s.match(/^([\w.]+)\s*\(([\s\S]*)\)$/);
  if (callM) return { type: "call", fn: callM[1], args: parseNamedArgs(callM[2]) };
  return { type: "value", value: parseValue(s) };
}

function parseBlockFields(lines, startI, baseIndent) {
  const fields = {};
  let i = startI;
  while (i < lines.length && lines[i].indent > baseIndent) {
    const m = lines[i].trimmed.match(/^(\w+)\s*:\s*(.*)/);
    if (m) fields[m[1]] = parseValue(m[2]);
    i++;
  }
  return [fields, i];
}

function parseBlockStatements(lines, startI, baseIndent) {
  const stmts = [];
  let i = startI;
  while (i < lines.length && lines[i].indent > baseIndent) {
    const t = lines[i].trimmed;
    const assignM = t.match(/^(\w+)\s*=\s*([\s\S]+)/);
    if (assignM) {
      stmts.push({ type: "assign", target: assignM[1], value: parseExpr(assignM[2]) });
    } else if (/^[\w.]+\s*\(/.test(t)) {
      const callM = t.match(/^([\w.]+)\s*\(([\s\S]*)\)/);
      if (callM) stmts.push({ type: "call", fn: callM[1], args: parseNamedArgs(callM[2]) });
    }
    i++;
  }
  return [stmts, i];
}

function parseShapeshifter(source) {
  const ast = { imports: [], objective: null, instruments: {}, validates: {}, phases: {}, targetLists: {} };

  let lines = source.split("\n").map((raw, idx) => ({
    lineNum: idx + 1,
    raw,
    text: raw.replace(/\/\/.*$/, "").trimEnd(),
    indent: raw.match(/^(\s*)/)[1].length,
    trimmed: raw.replace(/\/\/.*$/, "").trim(),
  })).filter(l => l.trimmed.length > 0);

  // Join continuation lines (unclosed brackets span multiple lines)
  const joined = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    const open = (line.trimmed.match(/\[/g) || []).length;
    const close = (line.trimmed.match(/\]/g) || []).length;
    if (open > close) {
      let combined = line.trimmed, balance = open - close, j = i + 1;
      while (j < lines.length && balance > 0) {
        combined += " " + lines[j].trimmed;
        balance += (lines[j].trimmed.match(/\[/g) || []).length;
        balance -= (lines[j].trimmed.match(/\]/g) || []).length;
        j++;
      }
      joined.push({ ...line, trimmed: combined });
      i = j;
    } else {
      joined.push(line);
      i++;
    }
  }
  lines = joined;

  i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (line.trimmed.startsWith("import ")) {
      ast.imports.push(line.trimmed.slice(7).trim());
      i++;
    } else if (/^objective\s+\w+\s*:/.test(line.trimmed)) {
      const name = line.trimmed.match(/^objective\s+(\w+)/)[1];
      const [fields, ni] = parseBlockFields(lines, i + 1, line.indent);
      ast.objective = { name, fields };
      i = ni;
    } else if (/^instrument\s+\w+\s*:/.test(line.trimmed)) {
      const name = line.trimmed.match(/^instrument\s+(\w+)/)[1];
      const [fields, ni] = parseBlockFields(lines, i + 1, line.indent);
      ast.instruments[name] = fields;
      i = ni;
    } else if (/^validate\s+\w+\s*:/.test(line.trimmed)) {
      const name = line.trimmed.match(/^validate\s+(\w+)/)[1];
      const [stmts, ni] = parseBlockStatements(lines, i + 1, line.indent);
      ast.validates[name] = stmts;
      i = ni;
    } else if (/^phase\s+\w+\s*:/.test(line.trimmed)) {
      const name = line.trimmed.match(/^phase\s+(\w+)/)[1];
      const [stmts, ni] = parseBlockStatements(lines, i + 1, line.indent);
      ast.phases[name] = stmts;
      i = ni;
    } else if (/^target_list\s+\w+\s*:/.test(line.trimmed)) {
      const name = line.trimmed.match(/^target_list\s+(\w+)/)[1];
      const [fields, ni] = parseBlockFields(lines, i + 1, line.indent);
      ast.targetLists[name] = { name, ...fields };
      i = ni;
    } else {
      i++;
    }
  }
  return ast;
}

/* ─── Executor ───────────────────────────────────────────────────────────── */

function describeVal(v) {
  if (v == null) return "null";
  if (Array.isArray(v)) return `Array(${v.length})`;
  return String(v).slice(0, 40);
}

function executeCall(fn, args, env, ast, log) {
  // Resolve variable references
  const a = Object.fromEntries(
    Object.entries(args).map(([k, v]) =>
      [k, typeof v === "string" && env[v] !== undefined ? env[v] : v]
    )
  );

  if (fn === "lavoisier.instrument.run_experiment") {
    const classKeys = (a.classes || ["PC", "PE"]).filter(k => LIPID_CLASSES[k]);
    const classSpecs = classKeys.map(key => {
      const cls = LIPID_CLASSES[key];
      const Xmin = cls.defaults.Xrange[0];
      const Xmax = Math.min(cls.defaults.Xrange[1], Xmin + 6); // limit for speed
      return { classKey: key, Xmin, Xmax, Ymin: 0, Ymax: 4, enabled: true };
    });
    log(`  Computing ${classSpecs.length} lipid class(es)...`);
    const records = runExperiment({
      experimentType: "lipidomics", classSpecs, proteinSpecs: [],
      polarity: a.polarity || "+", analyser: a.analyser || "orbitrap",
      analyserCfg: { kField: 1e12, Rm: 1e-2 },
      collisionEnergy_eV: a.collision_energy || 25,
      mzWindow: [200, 1500],
    });
    log(`  → ${records.length} predicted ions`);
    return records;
  }

  if (fn === "lavoisier.instrument.run_proteomics") {
    const proteinKeys = (a.proteins || ["HSA"]).filter(k => PROTEIN_CLASSES[k]);
    const proteinSpecs = proteinKeys.map(key => ({
      classKey: key,
      lengthMin: a.length_min || 7, lengthMax: a.length_max || 20,
      mcMin: 0, mcMax: a.mc_max || 1, enabled: true,
    }));
    log(`  Computing ${proteinSpecs.length} protein standard(s)...`);
    const records = runExperiment({
      experimentType: "proteomics", classSpecs: [], proteinSpecs,
      polarity: a.polarity || "+", analyser: a.analyser || "orbitrap",
      analyserCfg: { kField: 1e12, Rm: 1e-2 },
      collisionEnergy_eV: a.collision_energy || 28,
      mzWindow: [200, 3000],
    });
    log(`  → ${records.length} predicted ions`);
    return records;
  }

  if (fn === "lavoisier.partition.compute_addresses") {
    const lipids = a.lipids || [];
    return lipids.map(lip => {
      const mass = lip.mass || 500;
      const n = Math.max(1, Math.ceil(Math.sqrt(mass / 162.0)));
      const l = Math.min(n - 1, Math.max(0, Math.floor(n / 2)));
      const hash = ((lip.name || "?").charCodeAt(0) * 31 + Math.round(mass * 7)) % (2 * l + 1);
      return { name: lip.name || "?", mass: +mass.toFixed(4), adduct: lip.adduct || "[M+H]+", n, l, m: hash - l, s: 0.5 };
    });
  }

  if (fn === "lavoisier.cells.compile") {
    const tlName = a.target_list;
    const tl = ast.targetLists[tlName] || {};
    const targets = tl.targets || [];
    const windowPpm = tl.window_ppm || 5.0;
    const instrName = tl.instrument;
    const instr = ast.instruments[instrName] || {};
    const kappa = instr.kappa || 1e12;
    const fRef = instr.ref_frequency || 10e6;

    return targets.map(t => {
      const mz = t.mz || 500;
      const dMz = mz * windowPpm * 1e-6;
      // ω_z = sqrt(e·κ / (mz·u))
      const e = 1.60218e-19, u = 1.66054e-27;
      const omega = Math.sqrt(e * kappa / (mz * u));
      const dOmega = omega * windowPpm * 0.5e-6;
      const hbar = 1.0546e-34;
      // τ_min = ℏ / δM_cell (δM_cell ~ e·κ·δ(mz)·u / mz^2, simplified)
      const dM = e * kappa * dMz * u / (mz * mz);
      const tauMs = Math.max(0.01, (hbar / (dM + 1e-60)) * 1e3);
      return {
        name: t.name || "?",
        mz: mz.toFixed(4),
        window_da: dMz.toFixed(4),
        omega_hz: (omega / (2 * Math.PI)).toExponential(3),
        dp_lo: (-(dOmega / (2 * Math.PI * fRef))).toExponential(3),
        dp_hi: (+(dOmega / (2 * Math.PI * fRef))).toExponential(3),
        tau_min_ms: tauMs.toFixed(3),
      };
    });
  }

  log(`  ⚠ Unknown function: ${fn}`, "warn");
  return null;
}

function executeShapeshifter(ast) {
  const env = {}, logs = [];
  const log = (msg, level = "info") => logs.push({ level, message: msg });

  if (ast.objective) {
    log(`🎯 Objective: ${ast.objective.name}`);
    if (ast.objective.fields?.target) log(`   Target: ${ast.objective.fields.target}`);
  }

  for (const [name, stmts] of Object.entries(ast.validates)) {
    log(`✓ Validate: ${name}`);
    for (const stmt of stmts) {
      if (stmt.type === "call" && stmt.fn === "check_resolution_time") {
        const ppm = stmt.args?.window_ppm || 5;
        const hbar = 1.0546e-34, kappa = ast.instruments[stmt.args?.instrument]?.kappa || 1e12;
        const dM = 1.60218e-19 * kappa * 500 * ppm * 1e-6 * 1.66054e-27 / (500 * 500);
        log(`   τ_min ≈ ${(hbar / dM * 1000).toFixed(1)} ms at ${ppm} ppm (500 Da reference)`);
      }
    }
  }

  for (const [phaseName, stmts] of Object.entries(ast.phases)) {
    log(`⚡ Phase: ${phaseName}`);
    for (const stmt of stmts) {
      if (stmt.type === "assign") {
        if (stmt.value.type === "call") {
          env[stmt.target] = executeCall(stmt.value.fn, stmt.value.args, env, ast, log);
          log(`  ${stmt.target} ← ${describeVal(env[stmt.target])}`);
        } else {
          env[stmt.target] = stmt.value.value;
        }
      }
    }
  }

  if (env.records && Array.isArray(env.records)) {
    const summary = summariseRecords(env.records);
    return { result: { type: "records", data: env.records, summary }, logs };
  }
  if (env.registry && Array.isArray(env.registry)) {
    return { result: { type: "cells", data: env.registry }, logs };
  }
  if (env.addresses && Array.isArray(env.addresses)) {
    return { result: { type: "addresses", data: env.addresses }, logs };
  }
  return { result: { type: "empty", data: null }, logs };
}

function compileShapeshifter(files, activePathArr) {
  try {
    const activeNode = activePathArr ? getNode(files, activePathArr) : null;
    if (!activeNode?.content) return { result: { type: "empty", data: null }, ir: "", logs: [] };
    const name = activePathArr[activePathArr.length - 1];
    if (!name.endsWith(".ss"))
      return { result: { type: "empty", data: null }, ir: "Select a .ss file", logs: [] };

    const ast = parseShapeshifter(activeNode.content);
    const { result, logs } = executeShapeshifter(ast);
    return { result, ir: JSON.stringify(ast, null, 2), logs };
  } catch (e) {
    return {
      result: { type: "empty", data: null }, ir: "",
      logs: [{ level: "error", message: `Compile error: ${e.message}` }],
    };
  }
}

/* ─── Visualization ──────────────────────────────────────────────────────── */

function PartitionScatter({ records }) {
  if (!records.length) return (
    <div className="flex items-center justify-center text-[11px]" style={{ height: 150, color: "#555" }}>
      no records
    </div>
  );
  const W = 380, H = 150, PL = 32, PR = 8, PT = 8, PB = 28;
  const pW = W - PL - PR, pH = H - PT - PB;
  const mzMin = Math.min(...records.map(r => r.precursorMz));
  const mzMax = Math.max(...records.map(r => r.precursorMz));
  const nMax  = Math.max(...records.map(r => r.n), 1);
  const toX   = mz => PL + ((mz - mzMin) / (mzMax - mzMin || 1)) * pW;
  const toY   = n  => H - PB - ((n - 1) / nMax) * pH;
  const sample = records.length > 400
    ? records.filter((_, i) => i % Math.ceil(records.length / 400) === 0)
    : records;
  const axes = [
    { x1: PL, y1: H - PB, x2: W - PR, y2: H - PB },
    { x1: PL, y1: PT,     x2: PL,     y2: H - PB },
  ];
  const nTicks = [1, 2, 3, 4, 5].filter(v => v <= nMax + 1);
  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ fontFamily: "monospace", display: "block" }}>
      {axes.map((a, i) => <line key={i} {...a} stroke="#3a3a3a" />)}
      {nTicks.map(v => (
        <g key={v}>
          <line x1={PL - 3} y1={toY(v)} x2={PL} y2={toY(v)} stroke="#3a3a3a" />
          <text x={PL - 5} y={toY(v) + 4} textAnchor="end" fontSize={8} fill="#666">{v}</text>
        </g>
      ))}
      <text x={PL - 14} y={H / 2 + 4} fontSize={8} fill="#555"
        transform={`rotate(-90,${PL - 14},${H / 2})`}>n</text>
      <text x={W / 2} y={H - 4} textAnchor="middle" fontSize={8} fill="#555">m/z</text>
      {sample.map((r, i) => (
        <circle key={i} cx={toX(r.precursorMz)} cy={toY(r.n)} r={1.8}
          fill={CLASS_COLORS[r.analyteClass] || "#555"} opacity={0.55} />
      ))}
    </svg>
  );
}

function RecordsPanel({ records, summary }) {
  const classes = Object.keys(summary.perClass || {});
  const top = records.slice(0, 20);
  return (
    <div className="flex flex-col gap-3 overflow-y-auto" style={{ padding: 12, height: "100%" }}>
      {/* Stats */}
      <div className="grid grid-cols-3 gap-2">
        {[
          ["Records", summary.count],
          ["Classes", classes.length],
          ["m/z range", `${summary.mzRange?.[0]?.toFixed(0)}–${summary.mzRange?.[1]?.toFixed(0)}`],
        ].map(([label, val]) => (
          <div key={label} className="rounded p-2 text-center"
            style={{ background: "#2a2d2e", border: `1px solid ${T.border}` }}>
            <div className="text-[9px] uppercase tracking-wider" style={{ color: "#666" }}>{label}</div>
            <div className="font-mono text-[13px]" style={{ color: T.editorFg }}>{val}</div>
          </div>
        ))}
      </div>
      {/* Class legend */}
      <div className="flex flex-wrap gap-1.5">
        {classes.map(cls => (
          <span key={cls} className="flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px]"
            style={{ background: "#2a2d2e", border: `1px solid ${T.border}` }}>
            <span className="h-2 w-2 rounded-full shrink-0"
              style={{ background: CLASS_COLORS[cls] || "#555" }} />
            <span style={{ color: T.sidebarFg }}>{cls}</span>
            <span style={{ color: "#666" }}>{summary.perClass[cls]}</span>
          </span>
        ))}
      </div>
      {/* Scatter */}
      <PartitionScatter records={records} />
      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full font-mono text-[10px]" style={{ borderCollapse: "collapse" }}>
          <thead>
            <tr style={{ color: "#666", borderBottom: `1px solid ${T.border}` }}>
              {["Analyte", "Class", "m/z", "Adduct", "n", "ℓ", "Sk"].map(h => (
                <th key={h} className="py-1 pr-3 text-left font-normal">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {top.map((r, i) => (
              <tr key={i} style={{ borderBottom: `1px solid #2a2a2a`, color: T.editorFg }}>
                <td className="py-0.5 pr-3 truncate" style={{ maxWidth: 120 }}>{r.analyte}</td>
                <td className="pr-3" style={{ color: CLASS_COLORS[r.analyteClass] || "#555" }}>{r.analyteClass}</td>
                <td className="pr-3">{r.precursorMz.toFixed(3)}</td>
                <td className="pr-3" style={{ color: "#9cdcfe" }}>{r.adduct}</td>
                <td className="pr-3">{r.n}</td>
                <td className="pr-3">{r.l}</td>
                <td className="pr-3">{r.sentropyVec?.sk?.toFixed(3) ?? "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {records.length > 20 && (
          <div className="pt-1 text-[10px]" style={{ color: "#555" }}>
            + {records.length - 20} more records
          </div>
        )}
      </div>
    </div>
  );
}

function CellsPanel({ cells }) {
  return (
    <div className="overflow-y-auto" style={{ padding: 12, height: "100%" }}>
      <div className="mb-3 text-[10px] uppercase tracking-wider" style={{ color: "#666" }}>
        ΔP Cell Registry — {cells.length} cells
      </div>
      <table className="w-full font-mono text-[10px]" style={{ borderCollapse: "collapse" }}>
        <thead>
          <tr style={{ color: "#666", borderBottom: `1px solid ${T.border}` }}>
            {["Target", "m/z", "±Da", "ω (Hz)", "ΔP lo", "ΔP hi", "τ_min ms"].map(h => (
              <th key={h} className="py-1 pr-3 text-left font-normal">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {cells.map((c, i) => (
            <tr key={i} style={{ borderBottom: `1px solid #2a2a2a`, color: T.editorFg }}>
              <td className="py-0.5 pr-3" style={{ color: "#9cdcfe" }}>{c.name}</td>
              <td className="pr-3">{c.mz}</td>
              <td className="pr-3" style={{ color: "#dcdcaa" }}>±{c.window_da}</td>
              <td className="pr-3" style={{ color: "#b388eb" }}>{c.omega_hz}</td>
              <td className="pr-3" style={{ color: "#e07a7a" }}>{c.dp_lo}</td>
              <td className="pr-3" style={{ color: "#7cc77c" }}>{c.dp_hi}</td>
              <td className="pr-3" style={{ color: "#e6a456" }}>{c.tau_min_ms}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function AddressesPanel({ addresses }) {
  return (
    <div className="overflow-y-auto" style={{ padding: 12, height: "100%" }}>
      <div className="mb-3 text-[10px] uppercase tracking-wider" style={{ color: "#666" }}>
        Partition Addresses — Φ: ℤ⁺ → 𝒫
      </div>
      <table className="w-full font-mono text-[10px]" style={{ borderCollapse: "collapse" }}>
        <thead>
          <tr style={{ color: "#666", borderBottom: `1px solid ${T.border}` }}>
            {["Name", "Mass (Da)", "Adduct", "n", "ℓ", "m", "s"].map(h => (
              <th key={h} className="py-1 pr-3 text-left font-normal">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {addresses.map((a, i) => (
            <tr key={i} style={{ borderBottom: `1px solid #2a2a2a`, color: T.editorFg }}>
              <td className="py-0.5 pr-3" style={{ color: "#9cdcfe" }}>{a.name}</td>
              <td className="pr-3">{a.mass}</td>
              <td className="pr-3" style={{ color: "#dcdcaa" }}>{a.adduct}</td>
              <td className="pr-3" style={{ color: "#5fa8d3" }}>{a.n}</td>
              <td className="pr-3" style={{ color: "#7cc77c" }}>{a.l}</td>
              <td className="pr-3" style={{ color: "#e07a7a" }}>{a.m}</td>
              <td className="pr-3" style={{ color: "#b388eb" }}>{a.s > 0 ? "+½" : "−½"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ResultsPanel({ result }) {
  if (!result || result.type === "empty") {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-2 text-[12px]"
        style={{ color: "#444" }}>
        <Zap size={24} opacity={0.3} />
        <span>Run a .ss script to see results</span>
      </div>
    );
  }
  if (result.type === "records")  return <RecordsPanel records={result.data} summary={result.summary} />;
  if (result.type === "cells")    return <CellsPanel cells={result.data} />;
  if (result.type === "addresses") return <AddressesPanel addresses={result.data} />;
  return (
    <pre className="overflow-auto p-3 font-mono text-[11px]" style={{ color: T.editorFg }}>
      {JSON.stringify(result.data, null, 2)}
    </pre>
  );
}

/* ─── Output column ──────────────────────────────────────────────────────── */
function OutputColumn({ result, ir, logs, onRun, onClear }) {
  const [tab, setTab] = useState("results");
  const tabs = [
    { id: "results",  label: "Results",  Icon: Cpu },
    { id: "console",  label: "Console",  Icon: TerminalIcon },
    { id: "ir",       label: "IR",       Icon: Code2 },
  ];
  const levelColor = { log: "#d4d4d4", info: "#9cdcfe", warn: "#dcdcaa", error: "#f48771" };
  return (
    <div className="flex min-w-0 flex-1 flex-col"
      style={{ background: T.editor, borderLeft: `1px solid ${T.border}` }}>
      <div className="flex h-9 shrink-0 items-center justify-between pr-2"
        style={{ background: T.tabInactive }}>
        <div className="flex h-full">
          {tabs.map(({ id, label, Icon }) => {
            const active = tab === id;
            return (
              <button key={id} onClick={() => setTab(id)}
                className="relative flex items-center gap-1.5 px-3 text-[12px] transition-colors"
                style={{ color: active ? T.tabFgActive : T.tabFg, background: active ? T.tabActive : "transparent" }}>
                <Icon size={13} />{label}
                {id === "console" && logs.length > 0 && (
                  <span className="rounded-full px-1.5 text-[10px]"
                    style={{ background: T.accent, color: "#fff" }}>{logs.length}</span>
                )}
                {active && <span className="absolute left-0 top-0 h-0.5 w-full"
                  style={{ background: T.accentBright }} />}
              </button>
            );
          })}
        </div>
        <div className="flex items-center gap-1">
          {tab === "console" && (
            <button onClick={onClear} title="Clear" className="flex h-6 w-6 items-center justify-center rounded"
              style={{ color: T.tabFg }}><Trash2 size={14} /></button>
          )}
          <button onClick={onRun}
            className="flex h-6 items-center gap-1 rounded px-2 text-[12px]"
            style={{ background: T.accent, color: "#fff" }}>
            <RefreshCw size={12} />Run
          </button>
        </div>
      </div>

      <div className="min-h-0 flex-1">
        {tab === "results" && <ResultsPanel result={result} />}
        {tab === "console" && (
          <div className="h-full overflow-y-auto p-2 font-mono text-[12px] leading-relaxed">
            {logs.length === 0
              ? <div className="px-1 pt-1" style={{ color: "#5a5a5a" }}>Execution log appears here.</div>
              : logs.map((l, i) => (
                  <div key={i} className="border-b px-1 py-0.5"
                    style={{ color: levelColor[l.level] || "#d4d4d4", borderColor: "#2a2a2a" }}>
                    <span className="mr-2 opacity-50">{l.level}</span>{l.message}
                  </div>
                ))}
          </div>
        )}
        {tab === "ir" && (
          <pre className="h-full overflow-auto p-3 font-mono text-[11px] leading-[1.5]"
            style={{ color: T.editorFg }}>{ir || "No IR — run a .ss file"}</pre>
        )}
      </div>
    </div>
  );
}

/* ─── Helpers ────────────────────────────────────────────────────────────── */
const getNode = (tree, path) => {
  let n = { children: tree };
  for (const p of path) { n = n.children?.[p]; if (!n) return null; }
  return n;
};

const fileIcon = (name) => {
  if (name.endsWith(".ss"))   return { Icon: FileCode2, color: "#c586c0" };
  if (name.endsWith(".json")) return { Icon: FileJson,  color: "#cbcb41" };
  if (name.endsWith(".md"))   return { Icon: FileText,  color: "#519aba" };
  return { Icon: FileText, color: "#858585" };
};

const langLabel = (lang) => ({
  ss: "Shapeshifter", json: "JSON", md: "Markdown",
}[lang] || "Plain Text");

/* ─── File tree ──────────────────────────────────────────────────────────── */
function Tree({ tree, path = [], depth = 0, expanded, toggle, activePath, openFile }) {
  const entries = Object.entries(tree).sort((a, b) =>
    a[1].type !== b[1].type ? (a[1].type === "folder" ? -1 : 1) : a[0].localeCompare(b[0]));
  return (
    <>
      {entries.map(([name, node]) => {
        const fullPath = [...path, name];
        const key = fullPath.join("/");
        const isFolder = node.type === "folder";
        const isOpen = expanded.has(key);
        const isActive = activePath === key;
        const { Icon, color } = isFolder
          ? { Icon: isOpen ? FolderOpen : Folder, color: "#90a4ae" }
          : fileIcon(name);
        return (
          <div key={key}>
            <button
              onClick={() => isFolder ? toggle(key) : openFile(fullPath)}
              className="flex w-full items-center gap-1 py-0.5 pr-2 text-left text-[13px] leading-relaxed"
              style={{
                paddingLeft: 8 + depth * 12,
                color: T.sidebarFg,
                background: isActive ? T.lineActive : "transparent",
              }}
              onMouseEnter={e => { if (!isActive) e.currentTarget.style.background = "#2a2d2e"; }}
              onMouseLeave={e => { if (!isActive) e.currentTarget.style.background = "transparent"; }}
            >
              {isFolder
                ? (isOpen ? <ChevronDown size={14} className="shrink-0 opacity-70" />
                          : <ChevronRight size={14} className="shrink-0 opacity-70" />)
                : <span className="w-[14px] shrink-0" />}
              <Icon size={15} className="shrink-0" style={{ color }} />
              <span className="truncate">{name}</span>
            </button>
            {isFolder && isOpen && (
              <Tree tree={node.children} path={fullPath} depth={depth + 1}
                expanded={expanded} toggle={toggle} activePath={activePath} openFile={openFile} />
            )}
          </div>
        );
      })}
    </>
  );
}

/* ─── Editor ─────────────────────────────────────────────────────────────── */
function Editor({ value, onChange, onCursor }) {
  const gutterRef = useRef(null);
  const lines = value.split("\n");
  const syncScroll = e => { if (gutterRef.current) gutterRef.current.scrollTop = e.target.scrollTop; };
  const handleCursor = e => {
    const upto = e.target.value.slice(0, e.target.selectionStart);
    onCursor({ ln: upto.split("\n").length, col: upto.length - upto.lastIndexOf("\n") });
  };
  return (
    <div className="flex min-h-0 flex-1" style={{ background: T.editor }}>
      <div ref={gutterRef}
        className="select-none overflow-hidden py-3 text-right font-mono text-[13px] leading-[1.5]"
        style={{ color: T.gutter, minWidth: 52, paddingRight: 16 }}>
        {lines.map((_, i) => <div key={i}>{i + 1}</div>)}
      </div>
      <textarea
        value={value} onChange={e => onChange(e.target.value)}
        onScroll={syncScroll} onKeyUp={handleCursor} onClick={handleCursor}
        spellCheck={false}
        className="min-h-0 flex-1 resize-none border-0 bg-transparent py-3 pr-4 font-mono text-[13px] leading-[1.5] outline-none"
        style={{ color: T.editorFg, tabSize: 2, caretColor: "#fff" }}
      />
    </div>
  );
}

/* ─── Main shell ─────────────────────────────────────────────────────────── */
export default function ShapeshifterSandbox() {
  const [files, setFiles]       = useState(initialFiles);
  const [expanded, setExpanded] = useState(new Set(["examples"]));
  const [openTabs, setOpenTabs] = useState([["examples", "hello_lipid.ss"]]);
  const [activeTab, setActiveTab] = useState("examples/hello_lipid.ss");
  const [dirty, setDirty]       = useState(new Set());
  const [sidebar, setSidebar]   = useState(true);
  const [panel, setPanel]       = useState(true);
  const [activity, setActivity] = useState("files");
  const [panelTab, setPanelTab] = useState("terminal");
  const [cursor, setCursor]     = useState({ ln: 1, col: 1 });

  const [result, setResult] = useState({ type: "empty", data: null });
  const [ir, setIr]         = useState("");
  const [logs, setLogs]     = useState([]);
  const [runKey, setRunKey] = useState(0);

  const [editorWidth, setEditorWidth] = useState(55);
  const splitRef  = useRef(null);
  const dragging  = useRef(false);

  const activePathArr = openTabs.find(t => t.join("/") === activeTab) || null;
  const activeNode    = activePathArr ? getNode(files, activePathArr) : null;

  const run = useCallback(() => {
    const { result: r, ir: i, logs: l } = compileShapeshifter(files, activePathArr);
    setResult(r);
    setIr(i);
    setLogs(l);
    setRunKey(k => k + 1);
  }, [files, activePathArr]);

  useEffect(() => { const t = setTimeout(run, 400); return () => clearTimeout(t); }, [files, activePathArr, run]);

  useEffect(() => {
    const move = e => {
      if (!dragging.current || !splitRef.current) return;
      const r = splitRef.current.getBoundingClientRect();
      setEditorWidth(Math.min(80, Math.max(25, ((e.clientX - r.left) / r.width) * 100)));
    };
    const up = () => { dragging.current = false; document.body.style.cursor = ""; };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
    return () => { window.removeEventListener("mousemove", move); window.removeEventListener("mouseup", up); };
  }, []);

  const toggleFolder = useCallback(key => {
    setExpanded(prev => { const n = new Set(prev); n.has(key) ? n.delete(key) : n.add(key); return n; });
  }, []);

  const openFile = useCallback(pathArr => {
    const key = pathArr.join("/");
    setOpenTabs(prev => prev.some(t => t.join("/") === key) ? prev : [...prev, pathArr]);
    setActiveTab(key);
  }, []);

  const closeTab = useCallback((key, e) => {
    e.stopPropagation();
    setOpenTabs(prev => {
      const next = prev.filter(t => t.join("/") !== key);
      if (activeTab === key) setActiveTab(next.length ? next[next.length - 1].join("/") : null);
      return next;
    });
    setDirty(prev => { const n = new Set(prev); n.delete(key); return n; });
  }, [activeTab]);

  const updateContent = useCallback(val => {
    if (!activePathArr) return;
    setFiles(prev => { const next = structuredClone(prev); getNode(next, activePathArr).content = val; return next; });
    setDirty(prev => new Set(prev).add(activeTab));
  }, [activePathArr, activeTab]);

  const activities = [
    { id: "files",  Icon: Files,    label: "Explorer" },
    { id: "search", Icon: Search,   label: "Search" },
    { id: "git",    Icon: GitBranch, label: "Source Control" },
    { id: "run",    Icon: Play,     label: "Run" },
    { id: "ext",    Icon: Blocks,   label: "Extensions" },
  ];

  return (
    <div className="flex h-[680px] w-full flex-col overflow-hidden rounded-lg text-sm shadow-2xl"
      style={{ background: T.editor, color: T.editorFg, border: `1px solid ${T.border}` }}>

      {/* Title bar */}
      <div className="flex h-9 shrink-0 items-center justify-between px-3"
        style={{ background: T.titlebar }}>
        <div className="flex items-center gap-2">
          <span className="h-3 w-3 rounded-full" style={{ background: "#ff5f56" }} />
          <span className="h-3 w-3 rounded-full" style={{ background: "#ffbd2e" }} />
          <span className="h-3 w-3 rounded-full" style={{ background: "#27c93f" }} />
        </div>
        <span className="text-xs" style={{ color: "#cccccc" }}>Shapeshifter Sandbox</span>
        <div className="w-12" />
      </div>

      <div className="flex min-h-0 flex-1">
        {/* Activity bar */}
        <div className="flex w-12 shrink-0 flex-col items-center justify-between py-2"
          style={{ background: T.activitybar }}>
          <div className="flex flex-col items-center gap-1">
            {activities.map(({ id, Icon, label }) => {
              const active = activity === id;
              return (
                <button key={id} title={label}
                  onClick={() => { if (active) setSidebar(s => !s); else { setActivity(id); setSidebar(true); } }}
                  className="relative flex h-11 w-12 items-center justify-center"
                  style={{ color: active ? T.activitybarFgActive : T.activitybarFg }}>
                  {active && <span className="absolute left-0 top-1/2 h-6 w-0.5 -translate-y-1/2"
                    style={{ background: "#fff" }} />}
                  <Icon size={24} strokeWidth={1.5} />
                </button>
              );
            })}
          </div>
          <button title="Settings" className="flex h-11 w-12 items-center justify-center"
            style={{ color: T.activitybarFg }}><Settings size={24} strokeWidth={1.5} /></button>
        </div>

        {/* Sidebar */}
        {sidebar && (
          <div className="flex w-60 shrink-0 flex-col overflow-hidden"
            style={{ background: T.sidebar, borderRight: `1px solid ${T.border}` }}>
            <div className="flex h-9 shrink-0 items-center px-4 text-[11px] font-medium uppercase tracking-wider"
              style={{ color: T.sidebarHeader }}>
              {activities.find(a => a.id === activity)?.label}
            </div>
            <div className="min-h-0 flex-1 overflow-y-auto pb-2">
              {activity === "files"
                ? <Tree tree={files} expanded={expanded} toggle={toggleFolder}
                    activePath={activeTab} openFile={openFile} />
                : <div className="px-4 py-6 text-[13px]" style={{ color: T.tabFg }}>
                    {activities.find(a => a.id === activity)?.label} panel
                  </div>
              }
            </div>
          </div>
        )}

        {/* Editor + Output split */}
        <div ref={splitRef} className="flex min-w-0 flex-1">
          {/* Editor column */}
          <div className="flex min-w-0 flex-col" style={{ width: `${editorWidth}%` }}>
            {/* Tab bar */}
            <div className="flex h-9 shrink-0 items-stretch overflow-x-auto"
              style={{ background: T.tabInactive }}>
              {openTabs.map(pathArr => {
                const key = pathArr.join("/");
                const name = pathArr[pathArr.length - 1];
                const active = key === activeTab;
                const isDirty = dirty.has(key);
                const { Icon, color } = fileIcon(name);
                return (
                  <div key={key} onClick={() => setActiveTab(key)}
                    className="group flex cursor-pointer items-center gap-2 border-r px-3 text-[13px]"
                    style={{
                      background: active ? T.tabActive : T.tabInactive,
                      color: active ? T.tabFgActive : T.tabFg,
                      borderColor: T.border,
                      borderTop: active ? `1px solid ${T.accentBright}` : "1px solid transparent",
                    }}>
                    <Icon size={15} style={{ color }} />
                    <span className="whitespace-nowrap">{name}</span>
                    <button onClick={e => closeTab(key, e)}
                      className="flex h-5 w-5 items-center justify-center rounded"
                      style={{ color: active ? T.tabFgActive : T.tabFg }}>
                      {isDirty ? <Circle size={9} fill="currentColor" className="group-hover:hidden" /> : null}
                      <X size={15} className={isDirty ? "hidden group-hover:block" : "opacity-0 group-hover:opacity-100"} />
                    </button>
                  </div>
                );
              })}
            </div>

            {/* Breadcrumb */}
            {activePathArr && (
              <div className="flex h-6 shrink-0 items-center gap-1 px-4 text-[12px]"
                style={{ background: T.editor, color: T.tabFg }}>
                {activePathArr.map((p, i) => (
                  <span key={i} className="flex items-center gap-1">
                    {i > 0 && <ChevronRight size={12} className="opacity-60" />}{p}
                  </span>
                ))}
              </div>
            )}

            {activeNode
              ? <Editor value={activeNode.content} onChange={updateContent} onCursor={setCursor} />
              : <div className="flex min-h-0 flex-1 items-center justify-center text-sm"
                  style={{ background: T.editor, color: "#5a5a5a" }}>
                  Select a file to start editing
                </div>
            }

            {/* Panel */}
            {panel && (
              <div className="flex h-40 shrink-0 flex-col"
                style={{ background: T.panel, borderTop: `1px solid ${T.border}` }}>
                <div className="flex h-9 items-center justify-between pr-2">
                  <div className="flex h-full items-center">
                    {[{ id: "problems", label: "Problems" }, { id: "terminal", label: "Terminal" }].map(({ id, label }) => {
                      const active = panelTab === id;
                      return (
                        <button key={id} onClick={() => setPanelTab(id)}
                          className="relative h-full px-3 text-[11px] font-medium uppercase tracking-wider"
                          style={{ color: active ? T.tabFgActive : T.tabFg }}>
                          {label}
                          {active && <span className="absolute bottom-0 left-0 h-0.5 w-full"
                            style={{ background: T.accentBright }} />}
                        </button>
                      );
                    })}
                  </div>
                  <button onClick={() => setPanel(false)} style={{ color: T.tabFg }}>
                    <PanelBottomClose size={16} />
                  </button>
                </div>
                <div className="min-h-0 flex-1 overflow-y-auto px-3 pb-3 font-mono text-[13px] leading-relaxed">
                  {panelTab === "terminal"
                    ? <div style={{ color: T.editorFg }}>
                        <div><span style={{ color: "#4ec9b0" }}>shapeshifter</span>
                          <span style={{ color: "#ce9178" }}> $</span> shapeshifter run experiment.ss</div>
                        <div className="mt-1" style={{ color: "#6a9955" }}>✓ Parsed successfully</div>
                        <div style={{ color: "#9cdcfe" }}>→ {result?.data?.length ?? 0} records produced</div>
                        <div className="mt-1 flex items-center gap-2">
                          <span style={{ color: "#ce9178" }}>$</span>
                          <span className="inline-block h-4 w-2 animate-pulse" style={{ background: "#d4d4d4" }} />
                        </div>
                      </div>
                    : <div className="flex items-center gap-2 pt-1" style={{ color: "#6a9955" }}>
                        <Check size={14} /> No problems detected.
                      </div>
                  }
                </div>
              </div>
            )}
          </div>

          {/* Splitter */}
          <div onMouseDown={() => { dragging.current = true; document.body.style.cursor = "col-resize"; }}
            className="w-1 shrink-0 cursor-col-resize"
            style={{ background: T.border }} />

          {/* Output column */}
          <OutputColumn
            result={result} ir={ir} logs={logs} runKey={runKey}
            onRun={run} onClear={() => setLogs([])}
          />
        </div>
      </div>

      {/* Status bar */}
      <div className="flex h-6 shrink-0 items-center justify-between px-3 text-[12px]"
        style={{ background: T.statusBar, color: T.statusFg }}>
        <div className="flex items-center gap-3">
          <button className="flex items-center gap-1" onClick={() => setPanel(p => !p)}>
            <GitBranch size={13} /> main
          </button>
          <span className="flex items-center gap-2">
            <span className="flex items-center gap-1"><X size={13} />0</span>
            <span className="flex items-center gap-1"><AlertCircle size={13} />0</span>
          </span>
        </div>
        <div className="flex items-center gap-3">
          <span>Ln {cursor.ln}, Col {cursor.col}</span>
          <span>Spaces: 4</span>
          <span>UTF-8</span>
          <span>{activeNode ? langLabel(activeNode.lang) : "—"}</span>
          <Bell size={13} />
        </div>
      </div>
    </div>
  );
}
