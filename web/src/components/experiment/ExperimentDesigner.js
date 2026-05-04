import React from "react";
import AnalyteBuilder from "./AnalyteBuilder";
import IonizationConfig from "./IonizationConfig";
import AcquisitionConfig from "./AcquisitionConfig";
import VirtualRun from "./VirtualRun";
import LibraryExport from "./LibraryExport";
import ResultsDashboard from "./ResultsDashboard";

/**
 * Top-level workspace: left pane = experiment designer, right pane =
 * results dashboard.
 */
export default function ExperimentDesigner() {
  return (
    <div className="flex flex-col w-full min-h-[calc(100vh-160px)] bg-light dark:bg-dark text-dark dark:text-light">
      <div className="grid grid-cols-[340px_1fr] gap-0 flex-1 lg:grid-cols-1 min-h-0">
        {/* Left: designer */}
        <aside className="border-r-2 border-dark/10 dark:border-light/10 p-4 overflow-y-auto
          space-y-5 lg:border-r-0 lg:border-b-2">
          <div>
            <h2 className="text-base font-bold mb-1">Virtual experiment</h2>
            <p className="text-[11px] text-dark/60 dark:text-light/60">
              Design a lipidomics experiment. The instrument runs on this device and
              produces a synthetic library you take to your lab.
            </p>
          </div>

          <AnalyteBuilder />
          <IonizationConfig />
          <AcquisitionConfig />
          <VirtualRun />
          <LibraryExport />
        </aside>

        {/* Right: results */}
        <main className="p-4 overflow-y-auto min-w-0 min-h-0">
          <ResultsDashboard />
        </main>
      </div>
    </div>
  );
}
