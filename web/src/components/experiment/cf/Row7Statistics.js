import React, { useCallback, useState } from "react";
import { useCrossfilter, useChartRedraw } from "./CrossfilterContext";
import { classColor, PALETTE } from "./chartUtils";
import * as d3 from "d3";

export default function Row7Statistics() {
  const { pack } = useCrossfilter();
  const [, force] = useState(0);

  const refreshSnapshot = useCallback(() => force((x) => x + 1), []);
  useChartRedraw(refreshSnapshot);

  const total = pack.cf.size();
  const filtered = pack.all.value();
  const classBreak = pack.groups.class.all().filter((d) => d.value > 0)
    .sort((a, b) => b.value - a.value);

  const rows = pack.dims.intensity.top(50);

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
      <div className="grid grid-cols-6 lg:grid-cols-3 gap-2 text-[10px]">
        <Stat label="filtered" value={filtered.toLocaleString()} />
        <Stat label="total" value={total.toLocaleString()} />
        <Stat label="m/z mean" value={stats.mzMean.toFixed(2)} />
        <Stat label="m/z range"
          value={`${stats.mzMin.toFixed(0)}–${stats.mzMax.toFixed(0)}`} />
        <Stat label="ī predicted" value={stats.intensityMean.toExponential(1)} />
        <Stat label="modal n" value={stats.nMode ?? "-"} />
      </div>

      <div className="grid grid-cols-[1fr_300px] lg:grid-cols-1 gap-3">
        <div
          className="rounded border p-2 max-h-[240px] overflow-y-auto"
          style={{ background: PALETTE.bg, borderColor: PALETTE.grid }}
        >
          <div className="text-[9px] uppercase tracking-wider mb-1 font-normal"
               style={{ color: PALETTE.muted }}>
            Filtered records (top 50)
          </div>
          <table className="w-full text-[10px] font-mono"
                 style={{ color: PALETTE.text }}>
            <thead>
              <tr style={{ color: PALETTE.muted }}>
                <th className="text-left font-normal">cls</th>
                <th className="text-left font-normal">name·adduct</th>
                <th className="text-right font-normal">m/z</th>
                <th className="text-right font-normal">I</th>
                <th className="text-right font-normal">n,ℓ,m</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <tr key={i}
                  style={i % 2
                    ? { background: "rgba(255,255,255,0.025)" }
                    : undefined}
                >
                  <td className="px-1">
                    <span style={{
                      display: "inline-block", width: 6, height: 6, borderRadius: 6,
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

        <div
          className="rounded border p-2"
          style={{ background: PALETTE.bg, borderColor: PALETTE.grid }}
        >
          <div className="text-[9px] uppercase tracking-wider mb-1 font-normal"
               style={{ color: PALETTE.muted }}>
            Class breakdown
          </div>
          <table className="w-full text-[10px]" style={{ color: PALETTE.text }}>
            <tbody>
              {classBreak.map((c) => (
                <tr key={c.key}>
                  <td className="px-1">
                    <span style={{
                      display: "inline-block", width: 7, height: 7, borderRadius: 7,
                      background: classColor(c.key), marginRight: 4,
                    }} />
                    <span>{c.key}</span>
                  </td>
                  <td className="px-1 font-mono text-right">{c.value}</td>
                  <td className="px-1">
                    <div style={{
                      background: classColor(c.key), opacity: 0.55,
                      height: 3, width: `${100 * c.value / Math.max(1, classBreak[0].value)}%`,
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

function Stat({ label, value }) {
  return (
    <div className="rounded px-3 py-2"
      style={{ background: "rgba(255,255,255,0.03)", color: PALETTE.text }}>
      <div className="text-[8px] uppercase tracking-wider"
        style={{ color: PALETTE.muted }}>
        {label}
      </div>
      <div className="font-mono text-[12px] font-normal">{value}</div>
    </div>
  );
}
