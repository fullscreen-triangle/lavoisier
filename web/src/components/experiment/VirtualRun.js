import React, { useState } from "react";
import { useStore } from "@/lib/state/store";
import { runExperiment, summariseRecords } from "@/lib/experiment/virtualinstrument";

/**
 * The "Run" button. Compiles the design into class specs, dispatches the
 * forward simulation on the main thread (it is fast enough for v1 — 5000
 * records in ~200 ms), and writes the result to the store.
 */
export default function VirtualRun() {
  const design = useStore((s) => s.experimentDesign);
  const setRecords = useStore((s) => s.setExperimentRecords);
  const setRunning = useStore((s) => s.setExperimentRunning);
  const running = useStore((s) => s.experimentRunning);
  const recordCount = useStore((s) => s.experimentRecords.length);
  const lastRunMs = useStore((s) => s.experimentLastRunMs);

  const [error, setError] = useState(null);

  const onRun = async () => {
    setError(null);
    setRunning(true);
    // Yield once so the UI can update before the synchronous CPU burn
    await new Promise((r) => setTimeout(r, 16));
    try {
      const t0 = performance.now();
      const classSpecs   = design.classSpecs.filter((cs) => cs.enabled);
      const proteinSpecs = (design.proteinSpecs || []).filter((ps) => ps.enabled);
      const records = runExperiment({
        experimentType: design.experimentType || "lipidomics",
        classSpecs,
        proteinSpecs,
        adductsAllowed: design.adductsAllowed,
        polarity: design.polarity,
        analyser: design.analyser,
        analyserCfg: design.analyserCfg,
        collisionEnergy_eV: design.collisionEnergy_eV,
        mzWindow: design.mzWindow,
      });
      const summary = summariseRecords(records);
      const dt = performance.now() - t0;
      setRecords(records, summary, dt);
    } catch (e) {
      setError(String(e?.message || e));
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="space-y-2">
      <button
        onClick={onRun}
        disabled={running}
        className={`w-full py-3 rounded-md font-bold tracking-wide transition
          ${running
            ? "bg-dark/10 dark:bg-light/10 text-dark/40 dark:text-light/40 cursor-wait"
            : "bg-dark text-light dark:bg-light dark:text-dark hover:opacity-90"
          }`}
      >
        {running ? "running virtual instrument…" : "Run virtual experiment"}
      </button>

      {error && (
        <div className="text-[11px] text-red-500">{error}</div>
      )}

      {recordCount > 0 && !running && (
        <div className="text-[11px] text-dark/60 dark:text-light/60">
          last run produced <b>{recordCount}</b> predicted ions in{" "}
          <b>{lastRunMs.toFixed(0)}</b> ms
        </div>
      )}
    </div>
  );
}
