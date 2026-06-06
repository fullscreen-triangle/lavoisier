import React from "react";
import { useStore } from "@/lib/state/store";
import { CrossfilterProvider, useCrossfilter } from "./cf/CrossfilterContext";
import Row0Scatter from "./cf/Row0Scatter";
import Row1XIC from "./cf/Row1XIC";
import Row2ClassBubble from "./cf/Row2ClassBubble";
import Row3SEntropy from "./cf/Row3SEntropy";
import Row4Partition from "./cf/Row4Partition";
import Row5Categorical from "./cf/Row5Categorical";
import Row6Oscillatory from "./cf/Row6Oscillatory";
import Row7Statistics from "./cf/Row7Statistics";
import { Row8Droplets, Row9HeatmapAndPeak } from "./cf/Row8Special";
import RecordDetail from "./RecordDetail";
import LibraryExport from "./LibraryExport";
import GpuObservationPanel from "./GpuObservationPanel";
import { PALETTE } from "./cf/chartUtils";

/**
 * Crossfiltered dashboard.
 *   Row 0: 2D scatter (m/z × log I), 2D brush filter
 *   Row 1: Full-width XIC + brushable m/z bar histogram
 *   Row 2: Class bubble chart
 *   Row 3: S-entropy histograms (Sₖ, Sₜ, Sₑ, partition entropy)
 *   Row 4: Partition coordinates (n, ℓ, m, s)
 *   Row 5: Categorical coordinates (class, adduct, polarity, z)
 *   Row 6: Oscillatory coordinates (observable, bits, fragments, I)
 *   Row 7: Statistics (data count + filtered table + class breakdown)
 *   Row 8: Droplet bijection · heatmap · 3D peak surface
 */
export default function ResultsDashboard() {
  const records = useStore((s) => s.experimentRecords);
  const summary = useStore((s) => s.experimentSummary);
  const design = useStore((s) => s.experimentDesign);
  const lastRunMs = useStore((s) => s.experimentLastRunMs);

  if (!records || records.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh] text-center"
        style={{ color: PALETTE.muted }}>
        <div className="text-xl font-normal mb-2" style={{ color: PALETTE.text }}>
          virtual instrument idle
        </div>
        <div className="max-w-md text-sm">
          configure the experiment on the left and click <b>Run virtual experiment</b>
          — or switch to Shapeshifter mode to define the experiment as code.
          nothing is uploaded; everything is computed on this device.
        </div>
      </div>
    );
  }

  return (
    <CrossfilterProvider records={records}>
      <DashboardBody summary={summary} design={design} lastRunMs={lastRunMs} />
    </CrossfilterProvider>
  );
}

function DashboardBody({ summary, design, lastRunMs }) {
  return (
    <div className="space-y-4">
      <HeaderStrip summary={summary} design={design} lastRunMs={lastRunMs} />

      <RowSection title="0 · Scatter (m/z × log I)">
        <Row0Scatter height={220} />
      </RowSection>

      <RowSection title="1 · XIC + m/z range">
        <Row1XIC height={220} />
      </RowSection>

      <RowSection title="2 · Lipid class bubbles">
        <Row2ClassBubble height={280} />
      </RowSection>

      <RowSection title="3 · S-entropy">
        <Row3SEntropy />
      </RowSection>

      <RowSection title="4 · Partition coordinates (n, ℓ, m, s)">
        <Row4Partition />
      </RowSection>

      <RowSection title="5 · Categorical coordinates">
        <Row5Categorical />
      </RowSection>

      <RowSection title="6 · Oscillatory coordinates">
        <Row6Oscillatory />
      </RowSection>

      <RowSection title="7 · Statistics">
        <Row7Statistics />
      </RowSection>

      <RowSection title="8 · Droplet bijection (Ion → Drip)">
        <Row8Droplets height={360} />
      </RowSection>

      <RowSection title="9 · Heatmap · 3D peak surface">
        <Row9HeatmapAndPeak height={320} />
      </RowSection>

      <RecordDetail />

      <RowSection title="10 · GPU observation apparatus (partition depth field)">
        <GpuObservationPanel />
      </RowSection>

      <div className="grid grid-cols-[1fr_300px] gap-4 lg:grid-cols-1">
        <div />
        <LibraryExport />
      </div>
    </div>
  );
}

function HeaderStrip({ summary, design, lastRunMs }) {
  return (
    <header className="flex items-center justify-between flex-wrap gap-2">
      <div className="grid grid-cols-6 lg:grid-cols-3 gap-2 text-[10px] flex-1">
        <Stat label="records" value={summary?.count?.toLocaleString() ?? "0"} />
        <Stat label="classes" value={summary ? Object.keys(summary.perClass).length : 0} />
        <Stat label="adducts" value={summary ? Object.keys(summary.perAdduct).length : 0} />
        <Stat label="m/z range"
          value={summary
            ? `${summary.mzRange[0].toFixed(1)} – ${summary.mzRange[1].toFixed(1)}`
            : "-"}
        />
        <Stat label="analyser" value={design.analyser.toUpperCase()} />
        <Stat label="run time" value={`${lastRunMs.toFixed(0)} ms`} />
      </div>
      <ResetAllFilters />
    </header>
  );
}

function ResetAllFilters() {
  const { pack, redrawAll } = useCrossfilter();
  return (
    <button
      onClick={() => {
        for (const d of Object.values(pack.dims)) d.filterAll();
        redrawAll();
      }}
      className="text-[10px] px-3 py-1.5 rounded border font-normal tracking-wider
        uppercase transition-opacity hover:opacity-80"
      style={{
        background: PALETTE.bg, borderColor: PALETTE.grid, color: PALETTE.muted,
      }}
    >
      Reset filters
    </button>
  );
}

function Stat({ label, value }) {
  return (
    <div className="rounded px-3 py-1.5"
      style={{ background: "rgba(255,255,255,0.025)", color: PALETTE.text }}>
      <div className="text-[8px] uppercase tracking-wider"
        style={{ color: PALETTE.muted }}>
        {label}
      </div>
      <div className="font-mono text-[12px]">{value}</div>
    </div>
  );
}

function RowSection({ title, children }) {
  return (
    <section className="space-y-1.5">
      <h2 className="text-[9px] uppercase tracking-[0.2em] font-normal"
        style={{ color: PALETTE.muted }}>
        {title}
      </h2>
      <div className="rounded border p-3"
        style={{ background: PALETTE.bg, borderColor: PALETTE.grid }}>
        {children}
      </div>
    </section>
  );
}
