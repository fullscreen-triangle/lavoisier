import React, { useState, useRef, useCallback, useEffect } from "react";
import { useStore } from "@/lib/state/store";
import { parseShapeshifter, executeShapeshifter } from "@/lib/shapeshifter/compiler";
import { PALETTE } from "./cf/chartUtils";

/* ── Starter templates ───────────────────────────────────────────────────── */
const TEMPLATES = {
  plasma_lipid: {
    label: "Plasma lipidomics",
    code: `\
// Plasma lipidomics — positive mode Orbitrap
// Forward-simulates from the partition Lagrangian.

import lavoisier.instrument

objective PlasmaLipidomics:
    target: "predict lipidomics library for plasma LC-MS/MS"

instrument OrbitrapFusion:
    analyzer: "orbitrap"
    polarity: "+"
    collision_energy: 25
    mz_window: [200, 1500]

phase Design:
    classes = ["PC", "PE", "SM", "Cer"]

phase VirtualRun:
    records = lavoisier.instrument.run_experiment(
        classes: classes,
        polarity: "+",
        analyser: "orbitrap",
        collision_energy: 25
    )
`,
  },

  extended_lipid: {
    label: "Extended lipidomics",
    code: `\
// Extended lipidomics — covers phospholipids, sphingolipids, neutral lipids

import lavoisier.instrument

objective ExtendedLipidomics:
    target: "predict extended lipidomics library"

phase Design:
    classes = ["PC", "PE", "PS", "PI", "PG", "SM", "Cer", "TAG", "DAG", "LPC"]

phase VirtualRun:
    records = lavoisier.instrument.run_experiment(
        classes: classes,
        polarity: "+",
        analyser: "orbitrap",
        collision_energy: 25
    )
`,
  },

  negative_lipid: {
    label: "Negative mode lipids",
    code: `\
// Negative mode — anionic phospholipids and free fatty acids

import lavoisier.instrument

objective NegativeLipidomics:
    target: "predict anionic phospholipid library"

phase Design:
    classes = ["PE", "PS", "PI", "PG", "FA"]

phase VirtualRun:
    records = lavoisier.instrument.run_experiment(
        classes: classes,
        polarity: "-",
        analyser: "orbitrap",
        collision_energy: 30
    )
`,
  },

  proteomics: {
    label: "Plasma proteomics",
    code: `\
// Proteomics — tryptic peptides from common plasma standards
// Multiply-charged adducts selected by peptide mass.

import lavoisier.instrument

objective PlasmaProteomics:
    target: "predict tryptic peptide library from plasma proteins"

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
`,
  },

  tof_lipid: {
    label: "TOF lipidomics",
    code: `\
// TOF acquisition — flight-time observables from the partition Lagrangian

import lavoisier.instrument

objective TOFLipidomics:
    target: "predict TOF lipidomics library"

phase Design:
    classes = ["PC", "PE", "SM", "TAG"]

phase VirtualRun:
    records = lavoisier.instrument.run_experiment(
        classes: classes,
        polarity: "+",
        analyser: "tof",
        collision_energy: 25
    )
`,
  },
};

const TEMPLATE_KEYS = Object.keys(TEMPLATES);

/* ── Syntax-aware line colouring (best-effort over a textarea) ───────────── */
// We render a div behind a transparent textarea.
// Lines are classified and given color classes.
function classify(line) {
  const t = line.trim();
  if (!t || t.startsWith("//")) return "comment";
  if (/^(import|objective|instrument|validate|phase|target_list)\b/.test(t)) return "keyword";
  if (/^\w+\s*=\s*lavoisier\./.test(t) || /^lavoisier\./.test(t)) return "call";
  if (/^\w+\s*=\s*\[/.test(t) || /^\w+\s*=\s*"/.test(t)) return "assign";
  if (/^\w+\s*:\s*/.test(t)) return "field";
  return "plain";
}

const LINE_COLORS = {
  comment: PALETTE.muted,
  keyword: "#569cd6",   // VS Code keyword blue
  call:    "#9cdcfe",   // parameter blue
  assign:  PALETTE.text,
  field:   "#ce9178",   // VS Code string orange
  plain:   PALETTE.text,
};

/* ── Code editor with line numbers + overlay highlight ───────────────────── */
function CodeEditor({ value, onChange, onCursor }) {
  const gutterRef   = useRef(null);
  const overlayRef  = useRef(null);
  const taRef       = useRef(null);
  const lines = value.split("\n");

  const syncScroll = useCallback(e => {
    const top = e.target.scrollTop;
    if (gutterRef.current) gutterRef.current.scrollTop = top;
    if (overlayRef.current) overlayRef.current.scrollTop = top;
  }, []);

  const handleCursor = useCallback(e => {
    if (!onCursor) return;
    const upto = e.target.value.slice(0, e.target.selectionStart);
    onCursor({ ln: upto.split("\n").length, col: upto.length - upto.lastIndexOf("\n") });
  }, [onCursor]);

  return (
    <div className="relative flex min-h-0 flex-1"
      style={{ background: "#0d0f12", fontFamily: "monospace" }}>

      {/* Gutter */}
      <div ref={gutterRef}
        className="select-none overflow-hidden py-2 text-right text-[12px] leading-[1.6]"
        style={{ color: PALETTE.muted, minWidth: 40, paddingRight: 10, paddingLeft: 4 }}>
        {lines.map((_, i) => <div key={i}>{i + 1}</div>)}
      </div>

      {/* Syntax-coloured overlay (non-interactive) */}
      <div ref={overlayRef}
        className="pointer-events-none absolute left-10 top-0 overflow-hidden py-2
          text-[12px] leading-[1.6] whitespace-pre-wrap break-all"
        style={{ right: 0, color: "transparent" }}
        aria-hidden>
        {lines.map((l, i) => (
          <div key={i} style={{ color: LINE_COLORS[classify(l)] }}>{l || " "}</div>
        ))}
      </div>

      {/* Actual textarea — transparent foreground text, caret visible */}
      <textarea
        ref={taRef}
        value={value}
        onChange={e => onChange(e.target.value)}
        onScroll={syncScroll}
        onKeyUp={handleCursor}
        onClick={handleCursor}
        spellCheck={false}
        className="min-h-0 flex-1 resize-none border-0 bg-transparent py-2 pr-3
          text-[12px] leading-[1.6] outline-none"
        style={{ color: PALETTE.text, tabSize: 4, caretColor: "#fff" }}
      />
    </div>
  );
}

/* ── Template selector ───────────────────────────────────────────────────── */
function TemplateBar({ current, onSelect }) {
  return (
    <div className="flex items-center gap-2 shrink-0">
      <span className="text-[9px] uppercase tracking-wider shrink-0"
        style={{ color: PALETTE.muted }}>template</span>
      <select
        value={current}
        onChange={e => onSelect(e.target.value)}
        className="flex-1 rounded border px-2 py-1 text-[11px]"
        style={{
          borderColor: PALETTE.grid, background: PALETTE.bg,
          color: PALETTE.text, outline: "none",
        }}>
        {TEMPLATE_KEYS.map(k => (
          <option key={k} value={k}>{TEMPLATES[k].label}</option>
        ))}
      </select>
    </div>
  );
}

/* ── Execution log strip ─────────────────────────────────────────────────── */
function LogStrip({ logs, error }) {
  if (!logs.length && !error) return null;
  const levelColor = { info: PALETTE.muted, warn: "#dcdcaa", error: "#f48771" };
  return (
    <div className="shrink-0 overflow-y-auto max-h-28 font-mono text-[10px] leading-relaxed"
      style={{ borderTop: `1px solid ${PALETTE.grid}`, padding: "6px 8px" }}>
      {error && (
        <div style={{ color: "#f48771" }}>✕ {error}</div>
      )}
      {logs.map((l, i) => (
        <div key={i} style={{ color: levelColor[l.level] || PALETTE.muted }}>{l.message}</div>
      ))}
    </div>
  );
}

/* ── Status line ─────────────────────────────────────────────────────────── */
function StatusLine({ cursor, lang, running, recordCount }) {
  return (
    <div className="flex items-center justify-between shrink-0 px-2 py-0.5 text-[10px]"
      style={{ borderTop: `1px solid ${PALETTE.grid}`, color: PALETTE.muted, background: PALETTE.bg }}>
      <span className="flex items-center gap-3">
        <span style={{ color: "#569cd6" }}>Shapeshifter</span>
        {running && <span style={{ color: "#dcdcaa" }}>running…</span>}
        {!running && recordCount > 0 &&
          <span style={{ color: PALETTE.muted }}>{recordCount.toLocaleString()} records</span>}
      </span>
      <span>Ln {cursor.ln}  Col {cursor.col}</span>
    </div>
  );
}

/* ── Main ExperimentEditor component ─────────────────────────────────────── */
export default function ExperimentEditor() {
  const setRecords  = useStore(s => s.setExperimentRecords);
  const setRunning  = useStore(s => s.setExperimentRunning);
  const recordCount = useStore(s => s.experimentRecords.length);
  const storeRunning = useStore(s => s.experimentRunning);

  const [template, setTemplate] = useState("plasma_lipid");
  const [code, setCode]         = useState(TEMPLATES.plasma_lipid.code);
  const [logs, setLogs]         = useState([]);
  const [error, setError]       = useState(null);
  const [cursor, setCursor]     = useState({ ln: 1, col: 1 });

  const handleTemplateSelect = useCallback(key => {
    setTemplate(key);
    setCode(TEMPLATES[key].code);
    setLogs([]);
    setError(null);
  }, []);

  const run = useCallback(async () => {
    setLogs([]);
    setError(null);
    setRunning(true);
    await new Promise(r => setTimeout(r, 16));
    try {
      const t0 = performance.now();
      const ast = parseShapeshifter(code);
      const { result, logs: execLogs } = executeShapeshifter(ast);
      setLogs(execLogs);
      if (result.type === "records") {
        const dt = performance.now() - t0;
        setRecords(result.data, result.summary, dt);
      } else if (result.type !== "empty") {
        // cells / addresses — still interesting but not dashboard-ready
        setError(
          `Script produced a "${result.type}" result, not experiment records. ` +
          `Use lavoisier.instrument.run_experiment() or run_proteomics() to generate records.`
        );
      } else {
        setError(
          "No records produced. Make sure a phase assigns to `records` via " +
          "lavoisier.instrument.run_experiment() or run_proteomics()."
        );
      }
    } catch (e) {
      setError(e.message);
    } finally {
      setRunning(false);
    }
  }, [code, setRecords, setRunning]);

  // Ctrl+Enter shortcut
  useEffect(() => {
    const handler = e => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { e.preventDefault(); run(); }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [run]);

  return (
    <div className="flex flex-col"
      style={{
        height: 480,
        border: `1px solid ${PALETTE.grid}`,
        borderRadius: 4,
        overflow: "hidden",
        background: "#0d0f12",
      }}>

      {/* Header: label + template selector */}
      <div className="flex items-center justify-between shrink-0 px-3 py-2"
        style={{ borderBottom: `1px solid ${PALETTE.grid}`, background: PALETTE.bg }}>
        <span className="text-[9px] uppercase tracking-[0.15em] font-medium"
          style={{ color: PALETTE.muted }}>
          shapeshifter
        </span>
        <div className="flex-1 ml-3 mr-2">
          <TemplateBar current={template} onSelect={handleTemplateSelect} />
        </div>
      </div>

      {/* Code editor */}
      <CodeEditor value={code} onChange={setCode} onCursor={setCursor} />

      {/* Log output */}
      <LogStrip logs={logs} error={error} />

      {/* Run button */}
      <div className="flex items-center gap-2 shrink-0 px-3 py-2"
        style={{ borderTop: `1px solid ${PALETTE.grid}`, background: PALETTE.bg }}>
        <button
          onClick={run}
          disabled={storeRunning}
          className="flex-1 py-2 rounded text-[11px] tracking-wide font-medium transition-opacity"
          style={{
            background: storeRunning ? "rgba(255,255,255,0.06)" : "#0e639c",
            color: storeRunning ? PALETTE.muted : "#fff",
            cursor: storeRunning ? "wait" : "pointer",
          }}>
          {storeRunning ? "running virtual instrument…" : "▶  Run  (Ctrl+Enter)"}
        </button>
      </div>

      {/* Status line */}
      <StatusLine cursor={cursor} running={storeRunning} recordCount={recordCount} />
    </div>
  );
}
