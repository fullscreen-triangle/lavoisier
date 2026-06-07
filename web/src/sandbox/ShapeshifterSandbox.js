import React, { useState, useRef, useCallback, useEffect } from "react";
import {
  Files, Search, GitBranch, Play, Blocks, Settings,
  ChevronRight, ChevronDown, X, Circle, FileCode2, FileJson,
  FileText, Folder, FolderOpen, Terminal as TerminalIcon,
  AlertCircle, Bell, PanelBottomClose, Check, Code2,
  Trash2, RefreshCw, Cpu, Zap, Hammer, Square,
} from "lucide-react";
import { compileStage, executeStage } from "@/lib/shapeshifter/compiler";

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

const SS_SENTROPY_OBS = `\
// S-entropy observation — compute partition addresses from vibrational frequencies.
// Implements the Context-Based Spectral Database (Paper 3) generative approach.
// The address IS the spectrum: no dynamics required (Partition Determinism Theorem).

import lavoisier.observe
import lavoisier.purpose

objective SEntropyObservation:
    target: "compute S-entropy coordinates and ternary addresses from vibrational modes"

phase ComputeSentropy:
    // H2O vibrational modes in cm⁻¹ (NIST CCCBDB)
    water_freqs = [1595.0, 3657.0, 3756.0]
    water_se    = lavoisier.observe.sentropy(frequencies: water_freqs)

    // Compute ternary address at depth 12
    water_addr  = lavoisier.observe.ternary_address(sentropy: water_se, depth: 12)

    // Ion-droplet bijection: dual-path validation
    validation  = lavoisier.observe.dual_path_validate(
        sentropy: water_se,
        ion: { mass: 18.01, kinetic_energy: 1.0 },
        depth: 10
    )

phase DomainContext:
    // Purpose function: restrict to metabolomics subspace
    domain_context = lavoisier.purpose.domain(domain: "metabolomics", depth: 4)

    // Find matching domains for the S-entropy point
    matches = lavoisier.purpose.match(sentropy: water_se)
`;

const SS_FORCE_FREE = `\
// Force-free mass spectrometry (Paper 1 — Shader Depth Minimisation).
// Demonstrates that all four analyser equations emerge from one partition Lagrangian.
// No force acts at any stage — ions follow −∇M through the partition landscape.

import lavoisier.instrument

objective ForceFreeMS:
    target: "demonstrate analyser universality from the partition Lagrangian"

// Four analysers — same Lagrangian, different M(x,t) topologies:
//   TOF:       M = -κz          (linear gradient)
//   Quadrupole: M = κ₀/2·(x²-y²)[U + V cos Ωt]  (saddle)
//   Orbitrap:  M = κ/2·(z²-r²/2) + κRm²/2·ln(r/Rm)  (well)
//   FT-ICR:    M = 0, A = B/2(-y,x,0)            (circular)

phase TOFExperiment:
    records_tof = lavoisier.instrument.run_experiment(
        classes: ["PC", "PE", "SM"],
        polarity: "+",
        analyser: "tof",
        collision_energy: 25
    )

phase OrbitrapExperiment:
    records_orbi = lavoisier.instrument.run_experiment(
        classes: ["PC", "PE", "SM"],
        polarity: "+",
        analyser: "orbitrap",
        collision_energy: 25
    )

phase FticExperiment:
    records_fticr = lavoisier.instrument.run_experiment(
        classes: ["PC", "PE"],
        polarity: "+",
        analyser: "fticr",
        collision_energy: 25
    )
`;

const SS_GENERATIVE_DB = `\
// Generative database demonstration (Paper 3).
// The database stores nothing — it IS the phase space structure.
// Entries materialise on demand and dissolve after use.
// Cost: O(k) per query, independent of database size N. Memory: O(1).

import lavoisier.observe
import lavoisier.purpose

objective GenerativeDatabase:
    target: "demonstrate O(1) generative database lookup from ternary prefix"

phase GenerateFromPrefix:
    // Given a ternary prefix, reconstruct the S-entropy coordinates
    // without any stored spectrum (Partition Determinism Theorem).
    entry_202  = lavoisier.db.generate(prefix: "202", depth: 12)
    entry_120  = lavoisier.db.generate(prefix: "120", depth: 12)
    entry_001  = lavoisier.db.generate(prefix: "001", depth: 12)

phase PurposeFiltering:
    // Proteomics domain: 98%+ search space reduction
    proteomics_ctx = lavoisier.purpose.domain(domain: "proteomics", depth: 4)

    // Lipidomics domain
    lipidomics_ctx = lavoisier.purpose.domain(domain: "lipidomics", depth: 4)

    // Intersect: proteomics ∩ lipidomics (Prompt Contraction Theorem)
    combined = lavoisier.purpose.combine(
        domains: ["proteomics", "lipidomics"],
        depth: 4
    )
`;

const SS_SEBD_SEARCH = `\
// Partition-State Graph Search — SEBD-MS algorithm.
// Each ion maps to node (n, ℓ, m, s). Searches forward from precursor,
// backward from each fragment via virtual predecessors Sv* = 2·Sv_f − Sv_2.
// Off-shell Sv* = chemical transition states (Theorem 5.2).
// Output: PredictedRecord[] → feeds all existing dashboard charts.

import lavoisier.msms

objective MSMSSearch:
    target: "identify fragmentation pathways in an HCD spectrum"

phase FragmentSearch:
    // Lysine [M+H]+, HCD fragments (NIST AC_CAC verified)
    precursor_mz = 147.1128
    fragments = [84.081, 101.107, 102.091, 130.087]

    records = lavoisier.msms.sebd_search(
        precursor_mz: precursor_mz,
        fragments: fragments,
        max_depth: 7,
        planck_depth: 56
    )

phase PhaseCoherence:
    // Verify: ω_f = ω_prec · sqrt(m_prec/m_f) — self-consistent <10⁻⁹ ppm
    subharmonics = lavoisier.msms.phase_coherence(
        precursor_mz: precursor_mz,
        fragments: fragments
    )
`;

const SS_VIRTUAL_TENSOR = `\
// Stacked Virtual Substates — V_{ijkl} partition tensor.
// Four stacking dimensions: instrument × charge × polarity × time.
// Off-shell fraction ~8.3% — the virtual transition states.
// One Orbitrap transient contains full MS/MS, all charge states, polarity complement.

import lavoisier.msms

objective VirtualSubstates:
    target: "decompose partition state across 4 virtual dimensions"

phase BuildTensor:
    mz = 162.1125  // leucine [M+H]+

    tensor = lavoisier.msms.virtual_tensor(
        mz: mz,
        charge: 1,
        time_steps: 10
    )

phase SingleTransient:
    // Theorem 11.1: all information in one transient
    transient = lavoisier.msms.transient_contents(
        precursor_mz: mz,
        fragments: [44.049, 86.096, 103.086, 118.110]
    )

phase CrossingSymmetry:
    // Impossible ions as crossing-symmetry probes
    probes = lavoisier.msms.impossible_ions(
        mz_list: [86.096, 103.086, 118.110]
    )
`;

const SS_DB_SEARCH = `\
// Online spectral database search.
// Search public MS/MS libraries via their REST APIs.
// Query cost: O(k) per trie lookup, independent of database size N.
// Databases: MassBank (EU), MoNA (Davis), GNPS (San Diego).

import lavoisier.db

objective SpectralSearch:
    target: "search public spectral databases for lysine HCD fragments"

phase SearchDatabases:
    precursor_mz = 147.1128  // lysine [M+H]+
    fragments = [84.081, 101.107, 102.091, 130.087]

    // Search MassBank and MoNA in parallel
    db_results = lavoisier.db.search(
        precursor_mz: precursor_mz,
        fragments: fragments,
        databases: ["massbank", "mona"]
    )
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

instrument Name:
    analyzer: "orbitrap"   // tof | orbitrap | fticr | quadrupole
    polarity: "+"

phase Design:
    variable = value

phase Execute:
    result = lavoisier.module.function(key: value, ...)
\`\`\`

## Available functions

### Virtual instrument
| Function | Output |
|---|---|
| \`lavoisier.instrument.run_experiment(...)\` | Lipidomics PredictedRecord[] |
| \`lavoisier.instrument.run_proteomics(...)\` | Proteomics PredictedRecord[] |

### Partition / observation (Papers 1–3)
| Function | Output |
|---|---|
| \`lavoisier.partition.compute_addresses(lipids)\` | Partition states (n,ℓ,m,s) |
| \`lavoisier.cells.compile(target_list)\` | ΔP timing cell registry |
| \`lavoisier.observe.sentropy(frequencies)\` | S-entropy (Sk,St,Se) |
| \`lavoisier.observe.ternary_address(sentropy, depth)\` | Ternary address string |
| \`lavoisier.observe.dual_path_validate(sentropy, ion)\` | Bijection validation |
| \`lavoisier.db.generate(prefix, depth)\` | Coords from prefix (generative DB) |
| \`lavoisier.purpose.domain(domain, depth)\` | Purpose prefixes + reduction ratio |
| \`lavoisier.purpose.match(sentropy)\` | Matching domain contexts |
| \`lavoisier.purpose.combine(domains, depth)\` | Prompt contraction |

## Key papers

- **Partition Lagrangian**: all four analyser equations from one Lagrangian (Papers 1–3)
- **ΔP timing cells**: MS is a timing instrument, not an m/z instrument (Paper 1)
- **Triple Equivalence**: oscillation ≡ counting ≡ partition (Paper 2)
- **Generative database**: O(1) memory, O(k) query, stores nothing (Paper 3)
- **Dual-path validation**: ion-droplet bijection, no ground truth needed (Papers 2–3)
`;

const initialFiles = {
  examples: {
    type: "folder",
    children: {
      "hello_lipid.ss":           { type: "file", lang: "ss", content: SS_HELLO_LIPID },
      "proteomics_experiment.ss": { type: "file", lang: "ss", content: SS_PROTEOMICS },
      "temporal_acquisition.ss":  { type: "file", lang: "ss", content: SS_TEMPORAL },
      "partition_addresses.ss":   { type: "file", lang: "ss", content: SS_PARTITION },
      "sentropy_observation.ss":  { type: "file", lang: "ss", content: SS_SENTROPY_OBS },
      "force_free_ms.ss":         { type: "file", lang: "ss", content: SS_FORCE_FREE },
      "generative_database.ss":   { type: "file", lang: "ss", content: SS_GENERATIVE_DB },
      "sebd_ms_search.ss":        { type: "file", lang: "ss", content: SS_SEBD_SEARCH },
      "virtual_tensor.ss":        { type: "file", lang: "ss", content: SS_VIRTUAL_TENSOR },
      "db_search.ss":             { type: "file", lang: "ss", content: SS_DB_SEARCH },
    },
  },
  "README.md": { type: "file", lang: "md", content: README_CONTENT },
};

/* ─── Sandbox staged-run helper (compile → execute) ──────────────────────── */

/**
 * Read the active .ss source. Returns null if no runnable file is open.
 */
function activeSource(files, activePathArr) {
  const node = activePathArr ? getNode(files, activePathArr) : null;
  if (!node?.content) return null;
  const name = activePathArr[activePathArr.length - 1];
  if (!name.endsWith(".ss")) return null;
  return node.content;
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

/* ─── Terminal stream renderer ───────────────────────────────────────────── */
function TerminalView({ term }) {
  const endRef = useRef(null);
  useEffect(() => { endRef.current?.scrollIntoView({ block: "end" }); }, [term]);

  const streamStyle = (stream) => {
    if (stream === "stderr") return { color: "#f48771" };
    if (stream === "stage")  return { color: "#4ec9b0", fontWeight: 600 };
    return { color: "#d4d4d4" };
  };

  return (
    <div className="h-full overflow-y-auto p-2 font-mono text-[12px] leading-[1.55]">
      {term.length === 0 ? (
        <div className="px-1 pt-1" style={{ color: "#5a5a5a" }}>
          Press <b>Compile</b> then <b>Run</b> (or ▶ Run) to execute the active script.
        </div>
      ) : (
        term.map((l, i) => {
          if (l.stream === "stage") {
            return (
              <div key={i} className="mt-1 flex items-center gap-2">
                <span style={{ color: "#ce9178" }}>$</span>
                <span style={streamStyle("stage")}>{l.text}</span>
              </div>
            );
          }
          return (
            <div key={i} className="px-1 whitespace-pre-wrap break-words" style={streamStyle(l.stream)}>
              {l.text}
            </div>
          );
        })
      )}
      <div ref={endRef} />
    </div>
  );
}

/* ─── Output column ──────────────────────────────────────────────────────── */
function OutputColumn({ result, ir, logs, term, running, onCompile, onRun, onClear }) {
  const [tab, setTab] = useState("results");
  const tabs = [
    { id: "results",  label: "Results",  Icon: Cpu },
    { id: "terminal", label: "Terminal", Icon: TerminalIcon },
    { id: "console",  label: "Console",  Icon: TerminalIcon },
    { id: "ir",       label: "IR",       Icon: Code2 },
  ];
  const levelColor = { log: "#d4d4d4", info: "#9cdcfe", warn: "#dcdcaa", error: "#f48771" };
  const errCount = (term || []).filter(l => l.stream === "stderr").length;

  return (
    <div className="flex min-w-0 flex-1 flex-col"
      style={{ background: T.editor, borderLeft: `1px solid ${T.border}` }}>
      <div className="flex h-9 shrink-0 items-center justify-between pr-2"
        style={{ background: T.tabInactive }}>
        <div className="flex h-full">
          {tabs.map(({ id, label, Icon }) => {
            const active = tab === id;
            const badge = id === "console" ? logs.length
                        : id === "terminal" ? errCount : 0;
            return (
              <button key={id} onClick={() => setTab(id)}
                className="relative flex items-center gap-1.5 px-3 text-[12px] transition-colors"
                style={{ color: active ? T.tabFgActive : T.tabFg, background: active ? T.tabActive : "transparent" }}>
                <Icon size={13} />{label}
                {badge > 0 && (
                  <span className="rounded-full px-1.5 text-[10px]"
                    style={{ background: id === "terminal" ? "#a82d2d" : T.accent, color: "#fff" }}>{badge}</span>
                )}
                {active && <span className="absolute left-0 top-0 h-0.5 w-full"
                  style={{ background: T.accentBright }} />}
              </button>
            );
          })}
        </div>
        <div className="flex items-center gap-1">
          {(tab === "console" || tab === "terminal") && (
            <button onClick={onClear} title="Clear" className="flex h-6 w-6 items-center justify-center rounded"
              style={{ color: T.tabFg }}><Trash2 size={14} /></button>
          )}
          <button onClick={onCompile} disabled={running} title="Parse + type-check only"
            className="flex h-6 items-center gap-1 rounded px-2 text-[12px] disabled:opacity-40"
            style={{ background: "transparent", color: T.tabFgActive, border: `1px solid ${T.border}` }}>
            <Hammer size={12} />Compile
          </button>
          <button onClick={onRun} disabled={running}
            className="flex h-6 items-center gap-1 rounded px-2 text-[12px] disabled:opacity-40"
            style={{ background: running ? "#555" : T.accent, color: "#fff" }}>
            {running ? <Square size={11} /> : <Play size={12} />}
            {running ? "Running…" : "Run"}
          </button>
        </div>
      </div>

      <div className="min-h-0 flex-1">
        {tab === "results" && <ResultsPanel result={result} />}
        {tab === "terminal" && <TerminalView term={term || []} />}
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
            style={{ color: T.editorFg }}>{ir || "No IR — Compile a .ss file first"}</pre>
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
  const [term, setTerm]     = useState([]);          // terminal stream lines
  const [running, setRunning] = useState(false);
  const [compiledAst, setCompiledAst] = useState(null);  // last successful compile

  const [editorWidth, setEditorWidth] = useState(55);
  const splitRef  = useRef(null);
  const dragging  = useRef(false);

  const activePathArr = openTabs.find(t => t.join("/") === activeTab) || null;
  const activeNode    = activePathArr ? getNode(files, activePathArr) : null;

  /* Stage 1 — Compile only (parse + type-check + IR). No execution. */
  const compile = useCallback(() => {
    const src = activeSource(files, activePathArr);
    if (src == null) {
      setTerm([{ stream: "stderr", text: "error: open a .ss file to compile" }]);
      setCompiledAst(null);
      return null;
    }
    const { ok, ast, ir: i, term: t, diagnostics } = compileStage(src);
    setIr(i);
    setTerm(t);
    setCompiledAst(ok ? ast : null);
    return ok ? { ast, diagnostics } : null;
  }, [files, activePathArr]);

  /* Stage 2 — Run: compile (fresh), then execute phases. */
  const run = useCallback(async () => {
    setRunning(true);
    // Always compile fresh so edits are picked up
    const compiled = compile();
    if (!compiled) { setRunning(false); return; }

    // Yield a frame so the "compile" terminal output paints before executing
    await new Promise(r => setTimeout(r, 16));

    const { result: r, logs: l, term: execTerm } = executeStage(compiled.ast);
    setResult(r);
    setLogs(l);
    setTerm(prev => [...prev, ...execTerm]);
    setRunning(false);
  }, [compile]);

  // Auto-compile (parse + type-check only) shortly after edits, for live
  // diagnostics. Execution is never automatic — it requires an explicit Run.
  useEffect(() => {
    const t = setTimeout(() => { compile(); }, 500);
    return () => clearTimeout(t);
  }, [files, activePathArr, compile]);

  // Keyboard: Ctrl/Cmd+Enter to Run, Ctrl/Cmd+B to Compile
  useEffect(() => {
    const onKey = (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { e.preventDefault(); run(); }
      else if ((e.ctrlKey || e.metaKey) && (e.key === "b" || e.key === "B")) { e.preventDefault(); compile(); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [run, compile]);

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
                <div className="min-h-0 flex-1 overflow-y-auto px-2 pb-2">
                  {panelTab === "terminal"
                    ? <TerminalView term={term} />
                    : (() => {
                        const problems = term.filter(l => l.stream === "stderr");
                        if (problems.length === 0) {
                          return (
                            <div className="flex items-center gap-2 pt-2 px-2 font-mono text-[13px]"
                              style={{ color: "#6a9955" }}>
                              <Check size={14} /> No problems detected.
                            </div>
                          );
                        }
                        return (
                          <div className="pt-1 font-mono text-[12px] leading-relaxed">
                            {problems.map((p, i) => (
                              <div key={i} className="flex items-start gap-2 px-2 py-0.5"
                                style={{ color: "#f48771" }}>
                                <AlertCircle size={13} className="mt-0.5 shrink-0" />
                                <span className="whitespace-pre-wrap break-words">
                                  {p.text.replace(/^(error|warning):\s*/, "")}
                                </span>
                              </div>
                            ))}
                          </div>
                        );
                      })()
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
            result={result} ir={ir} logs={logs} term={term} running={running}
            onCompile={compile} onRun={run}
            onClear={() => { setLogs([]); setTerm([]); }}
          />
        </div>
      </div>

      {/* Status bar */}
      <div className="flex h-6 shrink-0 items-center justify-between px-3 text-[12px]"
        style={{ background: running ? "#a05a00" : T.statusBar, color: T.statusFg }}>
        <div className="flex items-center gap-3">
          <button className="flex items-center gap-1" onClick={() => { setPanel(true); setPanelTab("problems"); }}>
            <AlertCircle size={13} />
            {term.filter(l => l.stream === "stderr").length} problem(s)
          </button>
          {running && (
            <span className="flex items-center gap-1">
              <RefreshCw size={12} className="animate-spin" /> running
            </span>
          )}
          {!running && result?.type === "records" && (
            <span>{result.data.length} records</span>
          )}
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
