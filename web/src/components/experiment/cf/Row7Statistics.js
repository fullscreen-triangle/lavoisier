import React, { useCallback, useState } from "react";
import { useCrossfilter, useChartRedraw } from "./CrossfilterContext";
import { classColor } from "./chartUtils";
import * as d3 from "d3";

/**
 * Row 7: data count widget + filtered records table (dc.js DataCount/DataTable
 * pattern). Both update under any active crossfilter.
 */
export default function Row7Statistics() {
  const { pack } = useCrossfilter();
  const [_, force] = useState(0);

  const refreshSnapshot = useCallback(() => force((x) => x + 1), []);
  useChartRedraw(refreshSnapshot);

  const total = pack.cf.size();
  const filtered = pack.all.value();
  const classBreak = pack.groups.class.all().filter((d) => d.value > 0)
    .sort((a, b) => b.value - a.value);

  // top-50 records under current filter
  const rows = pack.dims.intensity.top(50);

  // numeric summary
  const allRecords = pack.dims.mz.bottom(Infinity);
  const mzs = allRecords.map((r) => r.precursorMz);
  const intensities = allRecords.map((r) => r.intensity);
  const ns = allRecords.map((r) => r.n);
  const stats = {
    mzMean: d3.mean(mzs) || 0, mzMin: d3.min(mzs) || 0, mzMax: d3.max(mzs) || 0,
    intensityMean: d3.mean(intensities) || 0,
    nMode: d3.mode(ns),
  };

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-6 lg:grid-cols-3 gap-3">
        <Stat label="filtered" value={filtered.toLocaleString()} mono />
        <Stat label="total" value={total.toLocaleString()} mono />
        <Stat label="m/z mean" value={stats.mzMean.toFixed(2)} mono />
        <Stat label="m/z range"
          value={`${stats.mzMin.toFixed(0)}–${stats.mzMax.toFixed(0)}`} mono />
        <Stat label="ī predicted"
          value={stats.intensityMean.toExponential(1)} mono />
        <Stat label="modal n"
          value={stats.nMode ?? "-"} mono />
      </div>

      <div className="grid grid-cols-[1fr_360px] lg:grid-cols-1 gap-3">
        <div className="rounded-md border border-dark/10 dark:border-light/10 p-2 bg-light dark:bg-dark
          max-h-[260px] overflow-y-auto">
          <div className="text-[10px] uppercase tracking-wider font-bold text-dark/60 dark:text-light/60 mb-1">
            Filtered records (top 50 by intensity)
          </div>
          <table className="w-full text-[10px] font-mono">
            <thead>
              <tr className="text-dark/60 dark:text-light/60">
                <th className="text-left">cls</th>
                <th className="text-left">name·adduct</th>
                <th className="text-right">m/z</th>
                <th className="text-right">I</th>
                <th className="text-right">n,ℓ,m</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <tr key={i} className={i % 2 ? "bg-dark/[0.02] dark:bg-light/[0.02]" : ""}>
                  <td className="px-1">
                    <span style={{
                      display: "inline-block", width: 7, height: 7, borderRadius: 7,
                      background: classColor(r.analyteClass), marginRight: 3,
                    }} />
                    {r.analyteClass}
                  </td>
                  <td className="px-1">{r.analyte}{r.adductAbbr ? ` ${r.adductAbbr}` : ""}</td>
                  <td className="px-1 text-right">{r.precursorMz.toFixed(3)}</td>
                  <td className="px-1 text-right">{r.intensity.toExponential(1)}</td>
                  <td className="px-1 text-right">{r.n},{r.l},{r.m}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="rounded-md border border-dark/10 dark:border-light/10 p-2 bg-light dark:bg-dark">
          <div className="text-[10px] uppercase tracking-wider font-bold text-dark/60 dark:text-light/60 mb-1">
            Class breakdown (filtered)
          </div>
          <table className="w-full text-[11px]">
            <tbody>
              {classBreak.map((c) => (
                <tr key={c.key}>
                  <td className="px-1">
                    <span style={{
                      display: "inline-block", width: 9, height: 9, borderRadius: 9,
                      background: classColor(c.key), marginRight: 5,
                    }} />
                    <span className="font-bold">{c.key}</span>
                  </td>
                  <td className="px-1 font-mono text-right">{c.value}</td>
                  <td className="px-1">
                    <div style={{
                      background: classColor(c.key), opacity: 0.5,
                      height: 4, width: `${100 * c.value / Math.max(1, classBreak[0].value)}%`,
                    }} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function Stat({ label, value, mono }) {
  return (
    <div className="rounded-md bg-dark/5 dark:bg-light/5 px-3 py-2">
      <div className="text-[9px] uppercase tracking-wider text-dark/50 dark:text-light/50">
        {label}
      </div>
      <div className={`${mono ? "font-mono" : ""} text-[13px] font-bold`}>
        {value}
      </div>
    </div>
  );
}
