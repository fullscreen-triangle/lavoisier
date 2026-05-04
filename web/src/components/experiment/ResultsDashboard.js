import React from "react";
import { useStore } from "@/lib/state/store";
import D3PartitionScatter3D from "./charts/D3PartitionScatter3D";
import D3SEntropyCube from "./charts/D3SEntropyCube";
import D3CapacityFormula from "./charts/D3CapacityFormula";
import D3MassShellHistogram from "./charts/D3MassShellHistogram";
import D3ClassAbundance from "./charts/D3ClassAbundance";
import D3AdductDistribution from "./charts/D3AdductDistribution";
import D3ResolutionSurface from "./charts/D3ResolutionSurface";
import D3MultimodalReadout from "./charts/D3MultimodalReadout";
import D3ChainScatter from "./charts/D3ChainScatter";
import D3PartitionCellGrid from "./charts/D3PartitionCellGrid";
import D3MassRangeViolin from "./charts/D3MassRangeViolin";
import D3InfoBitsBar from "./charts/D3InfoBitsBar";
import D3CoverageMatrix from "./charts/D3CoverageMatrix";

import RecordsTable from "./RecordsTable";
import RecordDetail from "./RecordDetail";
import LibraryExport from "./LibraryExport";

/**
 * Six-panel results dashboard. Each panel groups related views, mirroring
 * the publication panels but rendered live with the user's design.
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
    <div className="space-y-6">
      <HeaderStrip summary={summary} design={design} lastRunMs={lastRunMs} />

      {/* Panel 1: design landscape */}
      <Panel title="Design landscape">
        <Card title="A. Chain composition">
          <D3ChainScatter records={records} width={460} height={280} />
        </Card>
        <Card title="B. m/z violin per class">
          <D3MassRangeViolin records={records} width={460} height={280} />
        </Card>
        <Card title="C. Coverage (X, Y) per class">
          <D3CoverageMatrix records={records} width={460} height={300} />
        </Card>
        <Card title="D. Class abundance">
          <D3ClassAbundance records={records} width={460} height={280} />
        </Card>
      </Panel>

      {/* Panel 2: partition coordinates */}
      <Panel title="Partition coordinates">
        <Card title="A. (n, ℓ, m) scatter (3D)">
          <D3PartitionScatter3D records={records} width={460} height={360} />
        </Card>
        <Card title="B. Capacity C(n)=2n²">
          <D3CapacityFormula records={records} width={360} height={220} />
        </Card>
        <Card title="C. Mass-shell map">
          <D3MassShellHistogram records={records} width={460} height={260} />
        </Card>
        <Card title="D. (ℓ, m) cell grid">
          <D3PartitionCellGrid records={records} width={320} height={280} />
        </Card>
      </Panel>

      {/* Panel 3: ionisation & adducts */}
      <Panel title="Ionisation & adducts">
        <Card title="A. S-entropy cube (3D)">
          <D3SEntropyCube records={records} width={460} height={360} />
        </Card>
        <Card title="B. Adduct × class">
          <D3AdductDistribution records={records} width={460} height={260} />
        </Card>
        <Card title="C. Multimodal radar per class">
          <D3MultimodalReadout records={records} width={360} height={320} />
        </Card>
        <Card title="D. Information bits per class">
          <D3InfoBitsBar records={records} width={460} height={220} />
        </Card>
      </Panel>

      {/* Panel 4: instrument scaling */}
      <Panel title="Instrument scaling">
        <Card title="A. R(ω, T) family">
          <D3ResolutionSurface width={460} height={280} />
        </Card>
      </Panel>

      {/* Detail of selected record */}
      <RecordDetail />

      {/* Records + export */}
      <div className="grid grid-cols-[1fr_320px] gap-4 lg:grid-cols-1">
        <RecordsTable />
        <LibraryExport />
      </div>
    </div>
  );
}

function HeaderStrip({ summary, design, lastRunMs }) {
  if (!summary) return null;
  const fmt = (n) => n.toLocaleString();
  return (
    <header className="grid grid-cols-6 lg:grid-cols-3 gap-3 text-[11px]">
      <Stat label="records" value={fmt(summary.count)} />
      <Stat label="classes" value={Object.keys(summary.perClass).length} />
      <Stat label="adducts" value={Object.keys(summary.perAdduct).length} />
      <Stat label="m/z range"
        value={`${summary.mzRange[0].toFixed(1)} – ${summary.mzRange[1].toFixed(1)}`} />
      <Stat label="analyser" value={design.analyser.toUpperCase()} />
      <Stat label="run time" value={`${lastRunMs.toFixed(0)} ms`} />
    </header>
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

function Panel({ title, children }) {
  return (
    <section className="space-y-2">
      <h2 className="text-sm uppercase tracking-wider font-bold text-dark/70 dark:text-light/70">
        {title}
      </h2>
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-1">
        {children}
      </div>
    </section>
  );
}

function Card({ title, children }) {
  return (
    <div className="rounded-md border border-dark/10 dark:border-light/10 p-3 bg-light dark:bg-dark">
      <div className="text-[11px] uppercase tracking-wider font-bold text-dark/60 dark:text-light/60 mb-2">
        {title}
      </div>
      <div className="flex items-center justify-center min-h-[200px]">
        {children}
      </div>
    </div>
  );
}
