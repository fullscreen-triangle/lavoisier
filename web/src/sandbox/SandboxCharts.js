/**
 * SandboxCharts — a self-contained chart grid for Shapeshifter records.
 *
 * Takes a PredictedRecord[] directly (no store, no crossfilter context) and
 * renders the same families of charts the Experiment dashboard shows:
 *   1. m/z × intensity scatter (coloured by class)
 *   2. Class distribution bars
 *   3. m/z histogram
 *   4. S-entropy (Sk, St, Se) histograms
 *   5. Partition shell (n) distribution
 *   6. Adduct distribution
 *   7. Retention-time × m/z scatter (when RT present)
 *
 * Every chart is a plain SVG with a fixed viewBox, so it renders regardless of
 * container width — no clientWidth measurement, no D3 lifecycle.
 */

import React, { useMemo } from "react";

const CLASS_COLORS = {
  PC: "#5fa8d3", PE: "#e07a7a", PS: "#b388eb", PG: "#e493b3",
  PI: "#5dc0d8", SM: "#7cc77c", Cer: "#e6a456", TAG: "#cdc15c",
  DAG: "#a07a5e", LPC: "#a8b2bd", CE: "#9cc4d8", FA: "#e8c598",
  HSA: "#60a5fa", HBB: "#f87171", ENO1: "#34d399", CYCS: "#a78bfa", CASE: "#fb923c",
  SEBD: "#22d3ee",
};
const classColor = (c) => CLASS_COLORS[c] || "#7f8c9b";

const AXIS = "#5a6470";
const GRID = "#222831";
const TEXT = "#cdd5df";
const MUTED = "#6b7280";

/* ── Primitive chart helpers (fixed viewBox SVGs) ────────────────────────── */

function ChartFrame({ title, children, h = 200 }) {
  return (
    <div className="rounded border" style={{ borderColor: GRID, background: "#0d0f12" }}>
      <div className="px-3 pt-2 pb-1 text-[10px] uppercase tracking-[0.15em]" style={{ color: MUTED }}>
        {title}
      </div>
      <div style={{ height: h }}>{children}</div>
    </div>
  );
}

/** Scatter of (x, y) with per-point colour. */
function Scatter({ points, xLabel, yLabel, yLog = false }) {
  const W = 460, H = 200, PL = 46, PR = 14, PT = 12, PB = 30;
  const pw = W - PL - PR, ph = H - PT - PB;
  if (!points.length) return <Empty />;

  const xs = points.map(p => p.x), ys = points.map(p => p.y);
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  let yMin = Math.min(...ys), yMax = Math.max(...ys);
  if (yLog) { yMin = Math.max(yMin, 1e-6); }
  const xScale = (v) => PL + ((v - xMin) / (xMax - xMin || 1)) * pw;
  const yScale = (v) => {
    if (yLog) {
      const lv = Math.log10(Math.max(v, 1e-6)), lo = Math.log10(yMin), hi = Math.log10(yMax || 1);
      return H - PB - ((lv - lo) / (hi - lo || 1)) * ph;
    }
    return H - PB - ((v - yMin) / (yMax - yMin || 1)) * ph;
  };

  const xticks = ticks(xMin, xMax, 5);
  const yticks = yLog ? logTicks(yMin, yMax) : ticks(yMin, yMax, 4);

  return (
    <svg width="100%" height="100%" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet"
      style={{ fontFamily: "monospace", display: "block" }}>
      {/* grid + axes */}
      {xticks.map((t, i) => (
        <g key={`x${i}`}>
          <line x1={xScale(t)} y1={PT} x2={xScale(t)} y2={H - PB} stroke={GRID} strokeWidth={0.5} />
          <text x={xScale(t)} y={H - PB + 12} textAnchor="middle" fontSize={8} fill={MUTED}>{fmt(t)}</text>
        </g>
      ))}
      {yticks.map((t, i) => (
        <g key={`y${i}`}>
          <line x1={PL} y1={yScale(t)} x2={W - PR} y2={yScale(t)} stroke={GRID} strokeWidth={0.5} />
          <text x={PL - 5} y={yScale(t) + 3} textAnchor="end" fontSize={8} fill={MUTED}>{fmt(t)}</text>
        </g>
      ))}
      <line x1={PL} y1={H - PB} x2={W - PR} y2={H - PB} stroke={AXIS} />
      <line x1={PL} y1={PT} x2={PL} y2={H - PB} stroke={AXIS} />
      {/* points */}
      {points.map((p, i) => (
        <circle key={i} cx={xScale(p.x)} cy={yScale(p.y)} r={p.r ?? 2}
          fill={p.color} fillOpacity={0.6} />
      ))}
      <text x={(PL + W - PR) / 2} y={H - 4} textAnchor="middle" fontSize={8} fill={MUTED}>{xLabel}</text>
      <text x={12} y={(PT + H - PB) / 2} textAnchor="middle" fontSize={8} fill={MUTED}
        transform={`rotate(-90, 12, ${(PT + H - PB) / 2})`}>{yLabel}</text>
    </svg>
  );
}

/** Vertical bars from {label, value, color} rows. */
function Bars({ rows }) {
  const W = 460, H = 200, PL = 40, PR = 14, PT = 12, PB = 44;
  const pw = W - PL - PR, ph = H - PT - PB;
  if (!rows.length) return <Empty />;
  const max = Math.max(...rows.map(r => r.value), 1);
  const bw = pw / rows.length;
  const yticks = ticks(0, max, 4);

  return (
    <svg width="100%" height="100%" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet"
      style={{ fontFamily: "monospace", display: "block" }}>
      {yticks.map((t, i) => (
        <g key={i}>
          <line x1={PL} y1={H - PB - (t / max) * ph} x2={W - PR} y2={H - PB - (t / max) * ph}
            stroke={GRID} strokeWidth={0.5} />
          <text x={PL - 5} y={H - PB - (t / max) * ph + 3} textAnchor="end" fontSize={8} fill={MUTED}>{fmt(t)}</text>
        </g>
      ))}
      <line x1={PL} y1={H - PB} x2={W - PR} y2={H - PB} stroke={AXIS} />
      {rows.map((r, i) => {
        const bh = (r.value / max) * ph;
        return (
          <g key={i}>
            <rect x={PL + i * bw + bw * 0.12} y={H - PB - bh}
              width={bw * 0.76} height={Math.max(0, bh)}
              fill={r.color || "#5fa8d3"} fillOpacity={0.85} />
            <text x={PL + i * bw + bw / 2} y={H - PB + 11} textAnchor="end" fontSize={8} fill={MUTED}
              transform={`rotate(-35, ${PL + i * bw + bw / 2}, ${H - PB + 11})`}>{r.label}</text>
            <text x={PL + i * bw + bw / 2} y={H - PB - bh - 3} textAnchor="middle" fontSize={8} fill={TEXT}>
              {r.value}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

/** Histogram from a numeric array. */
function Histogram({ values, bins = 24, color = "#5fa8d3" }) {
  const data = useMemo(() => {
    if (!values.length) return [];
    const lo = Math.min(...values), hi = Math.max(...values);
    const w = (hi - lo) / bins || 1;
    const counts = new Array(bins).fill(0);
    for (const v of values) {
      let idx = Math.floor((v - lo) / w);
      if (idx >= bins) idx = bins - 1;
      if (idx < 0) idx = 0;
      counts[idx]++;
    }
    return counts.map((c, i) => ({ x0: lo + i * w, count: c }));
  }, [values, bins]);

  const W = 460, H = 200, PL = 40, PR = 14, PT = 12, PB = 30;
  const pw = W - PL - PR, ph = H - PT - PB;
  if (!data.length) return <Empty />;
  const max = Math.max(...data.map(d => d.count), 1);
  const bw = pw / data.length;

  return (
    <svg width="100%" height="100%" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet"
      style={{ fontFamily: "monospace", display: "block" }}>
      <line x1={PL} y1={H - PB} x2={W - PR} y2={H - PB} stroke={AXIS} />
      {data.map((d, i) => {
        const bh = (d.count / max) * ph;
        return (
          <rect key={i} x={PL + i * bw} y={H - PB - bh} width={Math.max(1, bw - 1)} height={Math.max(0, bh)}
            fill={color} fillOpacity={0.8} />
        );
      })}
      {[0, Math.floor(data.length / 2), data.length - 1].map(i => (
        <text key={i} x={PL + i * bw + bw / 2} y={H - PB + 12} textAnchor="middle" fontSize={8} fill={MUTED}>
          {fmt(data[i].x0)}
        </text>
      ))}
    </svg>
  );
}

function Empty() {
  return (
    <div className="flex h-full items-center justify-center text-[10px]" style={{ color: "#444" }}>
      no data
    </div>
  );
}

/* ── tick / format helpers ───────────────────────────────────────────────── */
function ticks(lo, hi, n) {
  if (!isFinite(lo) || !isFinite(hi) || lo === hi) return [lo];
  const step = (hi - lo) / n;
  return Array.from({ length: n + 1 }, (_, i) => lo + i * step);
}
function logTicks(lo, hi) {
  const a = Math.floor(Math.log10(Math.max(lo, 1e-9)));
  const b = Math.ceil(Math.log10(Math.max(hi, 1e-9)));
  const out = [];
  for (let e = a; e <= b; e++) out.push(Math.pow(10, e));
  return out;
}
function fmt(v) {
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 1000 || a < 0.01) return v.toExponential(1);
  return (Math.round(v * 100) / 100).toString();
}

/* ── Main grid ───────────────────────────────────────────────────────────── */
export default function SandboxCharts({ records }) {
  const summary = useMemo(() => {
    const perClass = {}, perAdduct = {}, perN = {};
    for (const r of records) {
      perClass[r.analyteClass]  = (perClass[r.analyteClass]  || 0) + 1;
      perAdduct[r.adduct]       = (perAdduct[r.adduct]       || 0) + 1;
      perN[r.n]                 = (perN[r.n]                 || 0) + 1;
    }
    return { perClass, perAdduct, perN };
  }, [records]);

  if (!records || records.length === 0) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-1 text-[12px]"
        style={{ color: "#444" }}>
        <span>No records to chart.</span>
        <span className="text-[10px]">Run a script that assigns to <code>records</code>.</span>
      </div>
    );
  }

  const hasRt = records.some(r => typeof r.retentionTime === "number");

  const scatterPts = records.map(r => ({
    x: r.precursorMz, y: Math.max(r.intensity, 1e-6),
    color: classColor(r.analyteClass), r: 1.6 + 1.8 * Math.sqrt(r.intensity || 0),
  }));

  const classRows = Object.entries(summary.perClass)
    .sort((a, b) => b[1] - a[1])
    .map(([k, v]) => ({ label: k, value: v, color: classColor(k) }));

  const adductRows = Object.entries(summary.perAdduct)
    .sort((a, b) => b[1] - a[1])
    .map(([k, v]) => ({ label: k, value: v, color: "#5fa8d3" }));

  const nRows = Object.entries(summary.perN)
    .sort((a, b) => +a[0] - +b[0])
    .map(([k, v]) => ({ label: `n=${k}`, value: v, color: "#a78bfa" }));

  const rtPts = hasRt ? records.map(r => ({
    x: r.retentionTime, y: r.precursorMz,
    color: classColor(r.analyteClass), r: 2,
  })) : [];

  return (
    <div className="h-full overflow-y-auto p-3 space-y-3" style={{ background: "#070809" }}>
      {/* headline stats */}
      <div className="grid grid-cols-4 gap-2">
        {[
          ["records", records.length],
          ["classes", Object.keys(summary.perClass).length],
          ["adducts", Object.keys(summary.perAdduct).length],
          ["m/z", `${Math.min(...records.map(r => r.precursorMz)).toFixed(0)}–${Math.max(...records.map(r => r.precursorMz)).toFixed(0)}`],
        ].map(([label, val]) => (
          <div key={label} className="rounded p-2 text-center"
            style={{ background: "#13161a", border: `1px solid ${GRID}` }}>
            <div className="text-[8px] uppercase tracking-wider" style={{ color: MUTED }}>{label}</div>
            <div className="font-mono text-[13px]" style={{ color: TEXT }}>{val}</div>
          </div>
        ))}
      </div>

      {/* class legend */}
      <div className="flex flex-wrap gap-1.5">
        {classRows.map(c => (
          <span key={c.label} className="flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px]"
            style={{ background: "#13161a", border: `1px solid ${GRID}` }}>
            <span className="h-2 w-2 rounded-full" style={{ background: c.color }} />
            <span style={{ color: TEXT }}>{c.label}</span>
            <span style={{ color: MUTED }}>{c.value}</span>
          </span>
        ))}
      </div>

      {/* chart grid — 2 columns */}
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-1">
        <ChartFrame title="m/z × intensity (log) — coloured by class">
          <Scatter points={scatterPts} xLabel="precursor m/z" yLabel="intensity (log)" yLog />
        </ChartFrame>

        <ChartFrame title="Class distribution">
          <Bars rows={classRows} />
        </ChartFrame>

        <ChartFrame title="m/z distribution">
          <Histogram values={records.map(r => r.precursorMz)} color="#5fa8d3" />
        </ChartFrame>

        <ChartFrame title="Partition shell n">
          <Bars rows={nRows} />
        </ChartFrame>

        <ChartFrame title="Sₖ knowledge entropy">
          <Histogram values={records.map(r => r.sentropyVec?.sk ?? 0)} color="#22d3ee" />
        </ChartFrame>

        <ChartFrame title="Sₜ temporal entropy">
          <Histogram values={records.map(r => r.sentropyVec?.st ?? 0)} color="#fbbf24" />
        </ChartFrame>

        <ChartFrame title="Sₑ evolution entropy">
          <Histogram values={records.map(r => r.sentropyVec?.se ?? 0)} color="#a78bfa" />
        </ChartFrame>

        <ChartFrame title="Adduct distribution">
          <Bars rows={adductRows} />
        </ChartFrame>

        {hasRt && (
          <ChartFrame title="Retention time × m/z — coloured by class">
            <Scatter points={rtPts} xLabel="RT (min)" yLabel="m/z" />
          </ChartFrame>
        )}
      </div>
    </div>
  );
}
