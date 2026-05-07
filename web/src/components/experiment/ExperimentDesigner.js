import React from "react";
import AnalyteBuilder from "./AnalyteBuilder";
import IonizationConfig from "./IonizationConfig";
import AcquisitionConfig from "./AcquisitionConfig";
import VirtualRun from "./VirtualRun";
import ResultsImport from "./ResultsImport";
import LibraryExport from "./LibraryExport";
import ResultsDashboard from "./ResultsDashboard";
import { PALETTE } from "./cf/chartUtils";

export default function ExperimentDesigner() {
  return (
    <div
      className="flex flex-col w-full min-h-[calc(100vh-160px)]"
      style={{ background: "#070809", color: PALETTE.text }}
    >
      <div className="grid grid-cols-[320px_1fr] gap-0 flex-1 lg:grid-cols-1 min-h-0">
        <aside
          className="border-r p-4 overflow-y-auto space-y-5
            lg:border-r-0 lg:border-b"
          style={{ borderColor: PALETTE.grid, background: PALETTE.bg }}
        >
          <div>
            <h2 className="text-sm font-normal mb-1 tracking-wide"
              style={{ color: PALETTE.text }}>
              Virtual experiment
            </h2>
            <p className="text-[10px] leading-relaxed"
              style={{ color: PALETTE.muted }}>
              Design a lipidomics experiment. The instrument runs on this device and
              produces a synthetic library you take to your lab.
            </p>
          </div>

          <AnalyteBuilder />
          <IonizationConfig />
          <AcquisitionConfig />
          <VirtualRun />
          <ResultsImport />
          <LibraryExport />
        </aside>

        <main className="p-4 overflow-y-auto min-w-0 min-h-0">
          <ResultsDashboard />
        </main>
      </div>
    </div>
  );
}
