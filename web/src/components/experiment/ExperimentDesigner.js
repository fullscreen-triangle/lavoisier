import React, { useState } from "react";
import AnalyteBuilder from "./AnalyteBuilder";
import IonizationConfig from "./IonizationConfig";
import AcquisitionConfig from "./AcquisitionConfig";
import VirtualRun from "./VirtualRun";
import ResultsImport from "./ResultsImport";
import LibraryExport from "./LibraryExport";
import ExperimentEditor from "./ExperimentEditor";
import ResultsDashboard from "./ResultsDashboard";
import { PALETTE } from "./cf/chartUtils";

function Divider({ label }) {
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 border-t" style={{ borderColor: PALETTE.grid }} />
      <span className="text-[8px] uppercase tracking-[0.2em]"
        style={{ color: PALETTE.muted }}>
        {label}
      </span>
      <div className="flex-1 border-t" style={{ borderColor: PALETTE.grid }} />
    </div>
  );
}

function DownloadTool() {
  return (
    <div className="space-y-2">
      <p className="text-[10px] leading-relaxed" style={{ color: PALETTE.muted }}>
        Run the analysis tool locally on your own mzML files.
        No data leaves your machine.
      </p>

      <div className="rounded border p-2.5 space-y-2"
        style={{ borderColor: PALETTE.grid, background: "rgba(255,255,255,0.02)" }}>
        <div className="text-[9px] uppercase tracking-wider"
          style={{ color: PALETTE.muted }}>
          workflow
        </div>
        <ol className="text-[10px] space-y-1 list-none" style={{ color: PALETTE.text }}>
          {[
            "Download the analysis tool below",
            "Run it on your mzML files locally",
            "Load the resulting .lavoisier.json here",
          ].map((step, i) => (
            <li key={i} className="flex gap-2">
              <span className="font-mono text-[9px] mt-0.5 shrink-0"
                style={{ color: PALETTE.muted }}>
                {i + 1}.
              </span>
              <span>{step}</span>
            </li>
          ))}
        </ol>
      </div>

      <a
        href="https://github.com/fullscreen-triangle/lavoisier/releases/latest"
        target="_blank"
        rel="noopener noreferrer"
        className="flex items-center justify-between w-full py-2 px-3 rounded
          text-[11px] tracking-wide border transition-opacity hover:opacity-70"
        style={{
          borderColor: PALETTE.grid,
          color:       PALETTE.text,
          background:  "transparent",
          textDecoration: "none",
        }}
      >
        <span>↓  Download analysis tool</span>
        <span className="text-[9px]" style={{ color: PALETTE.muted }}>
          github releases
        </span>
      </a>

      <div className="rounded border px-2.5 py-1.5 font-mono text-[9px]"
        style={{ borderColor: PALETTE.grid, color: PALETTE.muted,
                 background: "rgba(255,255,255,0.015)" }}>
        lavoisier-export sample.mzML
      </div>
    </div>
  );
}

/** Toggle between the visual GUI and the Shapeshifter code editor. */
function ModeToggle({ mode, onChange }) {
  return (
    <div className="flex rounded overflow-hidden"
      style={{ border: `1px solid ${PALETTE.grid}` }}>
      {[
        { key: "gui",  label: "Visual" },
        { key: "code", label: "Shapeshifter" },
      ].map(({ key, label }) => (
        <button
          key={key}
          onClick={() => onChange(key)}
          className="flex-1 py-1 text-[10px] tracking-wide transition-colors"
          style={{
            background: mode === key ? "#0e639c" : "transparent",
            color:      mode === key ? "#ffffff" : PALETTE.muted,
            borderRight: key === "gui" ? `1px solid ${PALETTE.grid}` : "none",
          }}
        >
          {label}
        </button>
      ))}
    </div>
  );
}

/** Original visual controls — unchanged from before. */
function GuiControls() {
  return (
    <div className="space-y-5">
      <AnalyteBuilder />
      <IonizationConfig />
      <AcquisitionConfig />
      <VirtualRun />
      <LibraryExport />
    </div>
  );
}

export default function ExperimentDesigner() {
  const [mode, setMode] = useState("gui");   // "gui" | "code"

  return (
    <div
      className="flex flex-col w-full min-h-[calc(100vh-160px)]"
      style={{ background: "#070809", color: PALETTE.text }}
    >
      <div className="grid grid-cols-[360px_1fr] gap-0 flex-1 lg:grid-cols-1 min-h-0">
        <aside
          className="border-r p-4 overflow-y-auto space-y-5
            lg:border-r-0 lg:border-b"
          style={{ borderColor: PALETTE.grid, background: PALETTE.bg }}
        >
          {/* ── Virtual instrument header ───────────────────────────── */}
          <div>
            <h2 className="text-sm font-normal mb-1 tracking-wide"
              style={{ color: PALETTE.text }}>
              Virtual instrument
            </h2>
            <p className="text-[10px] leading-relaxed"
              style={{ color: PALETTE.muted }}>
              Design a mass spectrometry experiment. The forward simulation
              runs on this device and produces a predicted library — take it
              to your lab as a reference before acquisition.
            </p>
          </div>

          {/* ── Mode toggle ─────────────────────────────────────────── */}
          <ModeToggle mode={mode} onChange={setMode} />

          {mode === "gui" && (
            <p className="text-[9px] leading-relaxed"
              style={{ color: PALETTE.muted }}>
              Configure analytes, ionisation, and acquisition settings using
              the controls below, then click <b>Run virtual experiment</b>.
            </p>
          )}

          {mode === "code" && (
            <p className="text-[9px] leading-relaxed"
              style={{ color: PALETTE.muted }}>
              Write a <b>Shapeshifter</b> (.ss) script to define your
              experiment. Press <b>▶ Run</b> or Ctrl+Enter to execute.
              Switch to Visual mode at any time.
            </p>
          )}

          {/* ── Main controls ────────────────────────────────────────── */}
          {mode === "gui"  && <GuiControls />}
          {mode === "code" && <ExperimentEditor />}

          {/* ── Real data section (always visible) ───────────────────── */}
          <Divider label="real data" />

          <div>
            <h2 className="text-sm font-normal mb-1 tracking-wide"
              style={{ color: PALETTE.text }}>
              Local analysis
            </h2>
          </div>

          <DownloadTool />
          <ResultsImport />
        </aside>

        <main className="p-4 overflow-y-auto min-w-0 min-h-0">
          <ResultsDashboard />
        </main>
      </div>
    </div>
  );
}
