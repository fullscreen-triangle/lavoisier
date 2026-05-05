import React from "react";
import { useStore } from "@/lib/state/store";
import { CrossfilterProvider, useCrossfilter } from "./cf/CrossfilterContext";
import Row1XIC from "./cf/Row1XIC";
import Row2ClassBubble from "./cf/Row2ClassBubble";
import Row3SEntropy from "./cf/Row3SEntropy";
import Row4Partition from "./cf/Row4Partition";
import Row5Categorical from "./cf/Row5Categorical";
import Row6Oscillatory from "./cf/Row6Oscillatory";
import Row7Statistics from "./cf/Row7Statistics";
import Row8Special from "./cf/Row8Special";
import RecordDetail from "./RecordDetail";
import LibraryExport from "./LibraryExport";

/**
 * The full crossfiltered dashboard.
 * Layout (from the design spec):
 *   Row 1: Full-width XIC + brushable m/z bar histogram
 *   Row 2: Class bubble chart (filterable)
 *   Row 3: 4 charts — S-entropy
 *   Row 4: Partition coordinates (n, ℓ, m, s)
 *   Row 5: Categorical coordinates (class, adduct, polarity, z)
 *   Row 6: Oscillatory coordinates (observable, bits, fragments, I)
 *   Row 7: Statistics (data count + filtered table + class breakdown)
 *   Row 8: Droplet bijection + heatmap + 3D peak surface
 */
export default function ResultsDashboard() {
  const records = useStore((s) => s.experimentRecords);
  const summary = useStore((s) => s.experimentSummary);
  const design = useStore((s) => s.experimentDesign);
  const lastRunMs = useStore((s) => s.experimentLastRunMs);

  if (!records || records.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh] text-center
        text-dark/50 dark:text-light/50">
        <div className="text-2xl font-bold mb-2 text-dark/70 dark:text-light/70">
          virtual instrument idle
        </div>
        <div className="max-w-md text-sm">
          configure the experiment on the left and click <b>Run virtual experiment</b> to
          synthesise predictions. nothing is uploaded; everything is computed on this device.
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
    <div className="space-y-5">
      <HeaderStrip summary={summary} design={design} lastRunMs={lastRunMs} />

      <RowSection title="1 · XIC + m/z range">
        <Row1XIC height={260} />
      </RowSection>

      <RowSection title="2 · Lipid class bubbles">
        <Row2ClassBubble height={300} />
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

      <RowSection title="8 · Droplet bijection · heatmap · 3D peak surface">
        <Row8Special height={300} />
      </RowSection>

      <RecordDetail />

      <div className="grid grid-cols-[1fr_320px] gap-4 lg:grid-cols-1">
        <div />
        <LibraryExport />
      </div>
    </div>
  );
}

function HeaderStrip({ summary, design, lastRunMs }) {
  return (
    <header className="flex items-center justify-between flex-wrap gap-2">
      <div className="grid grid-cols-6 lg:grid-cols-3 gap-3 text-[11px] flex-1">
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
      className="text-[11px] px-3 py-1.5 rounded border border-dark/15 dark:border-light/15
        hover:bg-dark/5 dark:hover:bg-light/5"
    >
      Reset all filters
    </button>
  );
}

function Stat({ label, value }) {
  return (
    <div className="rounded bg-dark/5 dark:bg-light/5 px-3 py-2">
      <div className="text-[9px] uppercase tracking-wider text-dark/50 dark:text-light/50">
        {label}
      </div>
      <div className="font-mono text-[13px] font-bold">{value}</div>
    </div>
  );
}

function RowSection({ title, children }) {
  return (
    <section className="space-y-2">
      <h2 className="text-[11px] uppercase tracking-wider font-bold text-dark/70 dark:text-light/70">
        {title}
      </h2>
      <div className="rounded-md border border-dark/10 dark:border-light/10 p-3 bg-light dark:bg-dark">
        {children}
      </div>
    </section>
  );
}
