/**
 * ExperimentCharts — charts for the experiment kinds produced by the
 * acquisition half of the language (coords, separation, drift, criterion,
 * baselineStat, sweep, scans).
 *
 * These mirror the panels of the language specification: the coordinate
 * cloud, the collision-energy drift, the separation statistic against its
 * negative control, the comparison-method decay, and the parameter sweep
 * against the declared threshold.
 *
 * Every chart is a plain SVG with a fixed viewBox, matching SandboxCharts,
 * so it renders at any panel width without a charting dependency.
 */
"use client";
import React, { useMemo } from "react";

const GRID = "#1f2429";
const AXIS = "#39424b";
const MUTED = "#6b7885";
const TEXT = "#c9d4de";
const RED = "#f87171";
const GREEN = "#34d399";

const SK = "#22d3ee", ST = "#fbbf24", SE = "#a78bfa";
const AXIS_COLOR = { s_k: SK, s_t: ST, s_e: SE };
const AXIS_LABEL = { s_k: "Sk", s_t: "St", s_e: "Se" };
const AXES = ["s_k", "s_t", "s_e"];

function fmt(v) {
  if (typeof v !== "number" || !isFinite(v)) return "\u2014";
  const a = Math.abs(v);
  if (a === 0) return "0";
  if (a >= 1000 || a < 0.01) return v.toExponential(1);
  return a >= 10 ? v.toFixed(1) : v.toFixed(2);
}

function ticks(lo, hi, n = 4) {
  if (!(hi > lo)) return [lo];
  const out = [];
  for (let i = 0; i <= n; i++) out.push(lo + ((hi - lo) * i) / n);
  return out;
}

function Frame({ title, note, children, h = 190 }) {
  return (
    <div className="rounded border" style={{ borderColor: GRID, background: "#0d0f12" }}>
      <div className="flex items-baseline justify-between px-3 pt-2 pb-1">
        <span className="text-[10px] uppercase tracking-[0.15em]" style={{ color: MUTED }}>
          {title}
        </span>
        {note ? <span className="font-mono text-[9px]" style={{ color: MUTED }}>{note}</span> : null}
      </div>
      <div style={{ height: h }}>{children}</div>
    </div>
  );
}

function Empty({ msg = "no data" }) {
  return (
    <div className="flex h-full items-center justify-center text-[11px]" style={{ color: "#3f4650" }}>
      {msg}
    </div>
  );
}

function Stat({ label, value, color = TEXT, sub }) {
  return (
    <div className="rounded p-2 text-center" style={{ background: "#13161a", border: "1px solid " + GRID }}>
      <div className="text-[8px] uppercase tracking-wider" style={{ color: MUTED }}>{label}</div>
      <div className="font-mono text-[13px]" style={{ color }}>{value}</div>
      {sub ? <div className="font-mono text-[8px] mt-0.5" style={{ color: MUTED }}>{sub}</div> : null}
    </div>
  );
}

/* ── Coordinate cloud: isometric projection of the three axes ────────────── */

/**
 * The occupied region of coordinate space, coloured by the acquisition
 * setting so any systematic drift is visible before a statistic is taken.
 */
function CoordCloud({ coords, colorBy = "nce" }) {
  const geom = useMemo(() => {
    if (!coords || coords.length === 0) return null;
    const lo = {}, hi = {};
    for (const a of AXES) {
      const vs = coords.map(c => c[a]).filter(v => typeof v === "number");
      if (!vs.length) return null;
      lo[a] = Math.min.apply(null, vs);
      hi[a] = Math.max.apply(null, vs);
      if (hi[a] === lo[a]) hi[a] = lo[a] + 1;
    }
    const cv = coords.map(c => c[colorBy]).filter(v => typeof v === "number");
    const clo = cv.length ? Math.min.apply(null, cv) : 0;
    const chi = cv.length ? Math.max.apply(null, cv) : 1;
    const step = coords.length > 900 ? Math.ceil(coords.length / 900) : 1;
    return { lo, hi, clo, chi, pts: coords.filter((_, i) => i % step === 0) };
  }, [coords, colorBy]);

  if (!geom) return <Empty />;

  const W = 460, H = 190;
  const { lo, hi, clo, chi, pts } = geom;
  const nrm = (v, a) => (v - lo[a]) / (hi[a] - lo[a]);

  const OX = 150, OY = 140, S = 92;
  const proj = (u, v, w) => [
    OX + (u - v) * S * 0.82,
    OY - (u + v) * S * 0.30 - w * S * 0.84,
  ];

  const ramp = (t) => {
    const s = Math.max(0, Math.min(1, t));
    if (s < 0.5) {
      const k = s / 0.5;
      return "rgb(" + Math.round(34 + k * 133) + "," + Math.round(211 - k * 91) + "," + Math.round(238 - k * 3) + ")";
    }
    const k = (s - 0.5) / 0.5;
    return "rgb(" + Math.round(167 + k * 84) + "," + Math.round(120 + k * 71) + "," + Math.round(235 - k * 199) + ")";
  };

  const o = proj(0, 0, 0), xa = proj(1, 0, 0), ya = proj(0, 1, 0), za = proj(0, 0, 1);

  return (
    <svg width="100%" height="100%" viewBox={"0 0 " + W + " " + H} preserveAspectRatio="xMidYMid meet"
      style={{ fontFamily: "monospace", display: "block" }}>
      {[0.25, 0.5, 0.75, 1].map((t, i) => {
        const a1 = proj(t, 0, 0), a2 = proj(t, 1, 0);
        const b1 = proj(0, t, 0), b2 = proj(1, t, 0);
        return (
          <g key={i}>
            <line x1={a1[0]} y1={a1[1]} x2={a2[0]} y2={a2[1]} stroke={GRID} strokeWidth={0.5} />
            <line x1={b1[0]} y1={b1[1]} x2={b2[0]} y2={b2[1]} stroke={GRID} strokeWidth={0.5} />
          </g>
        );
      })}
      {pts.map((c, i) => {
        const p = proj(nrm(c.s_k, "s_k"), nrm(c.s_t, "s_t"), nrm(c.s_e, "s_e"));
        const t = chi > clo ? (c[colorBy] - clo) / (chi - clo) : 0.5;
        return <circle key={i} cx={p[0]} cy={p[1]} r={1.5} fill={ramp(t)} fillOpacity={0.6} />;
      })}
      <line x1={o[0]} y1={o[1]} x2={xa[0]} y2={xa[1]} stroke={SK} strokeWidth={1.2} />
      <line x1={o[0]} y1={o[1]} x2={ya[0]} y2={ya[1]} stroke={ST} strokeWidth={1.2} />
      <line x1={o[0]} y1={o[1]} x2={za[0]} y2={za[1]} stroke={SE} strokeWidth={1.2} />
      <text x={xa[0] + 4} y={xa[1] + 8} fontSize={9} fill={SK}>Sk</text>
      <text x={ya[0] - 15} y={ya[1] + 8} fontSize={9} fill={ST}>St</text>
      <text x={za[0] - 5} y={za[1] - 4} fontSize={9} fill={SE}>Se</text>
      <g transform={"translate(" + (W - 108) + ", 14)"}>
        <text x={0} y={0} fontSize={8} fill={MUTED}>{colorBy}</text>
        {Array.from({ length: 32 }, (_, i) => (
          <rect key={i} x={i * 3} y={5} width={3} height={6} fill={ramp(i / 31)} />
        ))}
        <text x={0} y={20} fontSize={7.5} fill={MUTED}>{fmt(clo)}</text>
        <text x={96} y={20} fontSize={7.5} fill={MUTED} textAnchor="end">{fmt(chi)}</text>
      </g>
    </svg>
  );
}

/* ── Marginal densities per axis ─────────────────────────────────────────── */

function AxisDensities({ coords }) {
  const series = useMemo(() => {
    if (!coords || !coords.length) return null;
    const out = [];
    for (const a of AXES) {
      const vs = coords.map(c => c[a]).filter(v => typeof v === "number");
      if (!vs.length) continue;
      const lo = Math.min.apply(null, vs), hi = Math.max.apply(null, vs);
      const B = 36, bins = new Array(B).fill(0);
      const span = hi - lo || 1;
      for (const v of vs) bins[Math.min(B - 1, Math.floor(((v - lo) / span) * B))]++;
      const mx = Math.max.apply(null, bins) || 1;
      const mean = vs.reduce((s, v) => s + v, 0) / vs.length;
      const sd = Math.sqrt(vs.reduce((s, v) => s + (v - mean) * (v - mean), 0) / vs.length);
      out.push({ a, lo, hi, bins: bins.map(b => b / mx), mean, sd });
    }
    return out.length ? out : null;
  }, [coords]);

  if (!series) return <Empty />;
  const W = 460, H = 190, PL = 34, PR = 12, PT = 12, PB = 30;
  const pw = W - PL - PR, ph = H - PT - PB;
  const gLo = Math.min.apply(null, series.map(s => s.lo));
  const gHi = Math.max.apply(null, series.map(s => s.hi));
  const X = v => PL + ((v - gLo) / (gHi - gLo || 1)) * pw;

  return (
    <svg width="100%" height="100%" viewBox={"0 0 " + W + " " + H} preserveAspectRatio="xMidYMid meet"
      style={{ fontFamily: "monospace", display: "block" }}>
      {ticks(gLo, gHi, 5).map((t, i) => (
        <g key={i}>
          <line x1={X(t)} y1={PT} x2={X(t)} y2={H - PB} stroke={GRID} strokeWidth={0.5} />
          <text x={X(t)} y={H - PB + 11} textAnchor="middle" fontSize={8} fill={MUTED}>{fmt(t)}</text>
        </g>
      ))}
      <line x1={PL} y1={H - PB} x2={W - PR} y2={H - PB} stroke={AXIS} />
      {series.map(s => {
        const bw = (s.hi - s.lo) / s.bins.length;
        const d = s.bins.map((b, i) => (i === 0 ? "M" : "L") + X(s.lo + i * bw) + "," + (H - PB - b * ph)).join(" ");
        return (
          <g key={s.a}>
            <path d={d + " L" + X(s.hi) + "," + (H - PB) + " L" + X(s.lo) + "," + (H - PB) + " Z"}
              fill={AXIS_COLOR[s.a]} fillOpacity={0.16} />
            <path d={d} stroke={AXIS_COLOR[s.a]} strokeWidth={1.3} fill="none" />
          </g>
        );
      })}
      <g transform={"translate(" + (PL + 4) + ", " + (PT + 8) + ")"}>
        {series.map((s, i) => (
          <text key={s.a} x={0} y={i * 11} fontSize={8} fill={AXIS_COLOR[s.a]}>
            {AXIS_LABEL[s.a] + " mean " + fmt(s.mean) + "  sd " + fmt(s.sd)}
          </text>
        ))}
      </g>
    </svg>
  );
}

/* ── Drift with the acquisition setting ──────────────────────────────────── */

function DriftChart({ drift, coords }) {
  const data = useMemo(() => {
    if (!drift || !coords || !coords.length) return null;
    const over = drift.over || "nce";
    const by = {};
    for (const c of coords) {
      const k = c[over];
      if (typeof k !== "number") continue;
      if (!by[k]) by[k] = { s_k: [], s_t: [], s_e: [] };
      for (const a of AXES) if (typeof c[a] === "number") by[k][a].push(c[a]);
    }
    const levels = Object.keys(by).map(Number).sort((a, b) => a - b);
    if (levels.length < 2) return null;
    const mean = arr => arr.reduce((s, v) => s + v, 0) / (arr.length || 1);
    const series = {};
    for (const a of AXES) {
      const ms = levels.map(l => mean(by[l][a]));
      const mu = mean(ms);
      const sd = Math.sqrt(ms.reduce((s, v) => s + (v - mu) * (v - mu), 0) / ms.length) || 1;
      series[a] = ms.map(v => (v - mu) / sd);
    }
    return { over, levels, series };
  }, [drift, coords]);

  if (!data) return <Empty msg="needs coords carrying the acquisition setting" />;
  const W = 460, H = 190, PL = 40, PR = 62, PT = 14, PB = 32;
  const pw = W - PL - PR, ph = H - PT - PB;
  const levels = data.levels, series = data.series;
  const X = i => PL + (levels.length === 1 ? pw / 2 : (i / (levels.length - 1)) * pw);
  const Y = z => PT + ph / 2 - (z / 2.4) * (ph / 2);

  return (
    <svg width="100%" height="100%" viewBox={"0 0 " + W + " " + H} preserveAspectRatio="xMidYMid meet"
      style={{ fontFamily: "monospace", display: "block" }}>
      <line x1={PL} y1={Y(0)} x2={W - PR} y2={Y(0)} stroke={GRID} strokeWidth={0.7} />
      {levels.map((l, i) => (
        <text key={i} x={X(i)} y={H - PB + 12} textAnchor="middle" fontSize={8} fill={MUTED}>{l}</text>
      ))}
      <text x={PL + pw / 2} y={H - 5} textAnchor="middle" fontSize={8} fill={MUTED}>{data.over}</text>
      <text x={10} y={PT + ph / 2} fontSize={8} fill={MUTED}
        transform={"rotate(-90, 10, " + (PT + ph / 2) + ")"} textAnchor="middle">z-scored mean</text>
      {AXES.map((a, ai) => {
        const d = series[a].map((z, i) => (i === 0 ? "M" : "L") + X(i) + "," + Y(z)).join(" ");
        const r = drift.axes && drift.axes[a] ? drift.axes[a].pearson_r : null;
        return (
          <g key={a}>
            <path d={d} stroke={AXIS_COLOR[a]} strokeWidth={1.6} fill="none" />
            {series[a].map((z, i) => (
              <circle key={i} cx={X(i)} cy={Y(z)} r={2.3} fill="#0d0f12"
                stroke={AXIS_COLOR[a]} strokeWidth={1.2} />
            ))}
            <text x={W - PR + 4} y={PT + 11 + ai * 11} fontSize={8} fill={AXIS_COLOR[a]}>
              {AXIS_LABEL[a] + " r=" + (typeof r === "number" ? r.toFixed(3) : "\u2014")}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

/** Correlation of each axis with the setting, against the declared bound. */
function DriftBars({ drift, maxAbsR }) {
  if (!drift || !drift.axes) return <Empty />;
  const bound = typeof maxAbsR === "number" ? maxAbsR : 0.3;
  const W = 460, H = 190, PL = 46, PR = 24, PT = 16, PB = 36;
  const pw = W - PL - PR, ph = H - PT - PB;
  const rows = AXES.map(a => ({ a, r: (drift.axes[a] && drift.axes[a].pearson_r) || 0 }));
  const lim = Math.max(0.5, bound * 1.4, Math.max.apply(null, rows.map(x => Math.abs(x.r) * 1.3)));
  const X = r => PL + ((r + lim) / (2 * lim)) * pw;
  const bh = ph / rows.length;

  return (
    <svg width="100%" height="100%" viewBox={"0 0 " + W + " " + H} preserveAspectRatio="xMidYMid meet"
      style={{ fontFamily: "monospace", display: "block" }}>
      {[-bound, bound].map((t, i) => (
        <line key={i} x1={X(t)} y1={PT} x2={X(t)} y2={H - PB}
          stroke={RED} strokeWidth={1} strokeDasharray="3 3" />
      ))}
      <line x1={X(0)} y1={PT} x2={X(0)} y2={H - PB} stroke={AXIS} strokeWidth={0.8} />
      {rows.map((row, i) => {
        const y = PT + i * bh + bh * 0.22;
        const x0 = X(0), x1 = X(row.r);
        const pass = Math.abs(row.r) < bound;
        return (
          <g key={row.a}>
            <rect x={Math.min(x0, x1)} y={y} width={Math.abs(x1 - x0)} height={bh * 0.56}
              fill={AXIS_COLOR[row.a]} fillOpacity={pass ? 0.9 : 0.4}
              stroke={pass ? "none" : RED} strokeWidth={pass ? 0 : 1} />
            <text x={PL - 6} y={y + bh * 0.4} textAnchor="end" fontSize={9} fill={AXIS_COLOR[row.a]}>
              {AXIS_LABEL[row.a]}
            </text>
            <text x={x1 + (row.r >= 0 ? 4 : -4)} y={y + bh * 0.4}
              textAnchor={row.r >= 0 ? "start" : "end"} fontSize={8} fill={pass ? TEXT : RED}>
              {row.r.toFixed(3) + (pass ? "" : "  fail")}
            </text>
          </g>
        );
      })}
      <text x={X(bound)} y={H - PB + 12} textAnchor="middle" fontSize={7.5} fill={RED}>
        {"|r| < " + bound}
      </text>
      <text x={PL + pw / 2} y={H - 4} textAnchor="middle" fontSize={8} fill={MUTED}>
        {"Pearson r with " + (drift.over || "setting")}
      </text>
    </svg>
  );
}

/* ── Separation against its control and the declared threshold ───────────── */

function SeparationChart({ separation, control, minRatio }) {
  if (!separation) return <Empty />;
  const thr = typeof minRatio === "number" ? minRatio : 2.0;
  const W = 460, H = 190, PL = 40, PR = 16, PT = 18, PB = 44;
  const pw = W - PL - PR, ph = H - PT - PB;
  const rows = [{ label: "true labels", v: separation.separation_ratio, color: "#5fa8d3" }];
  if (control) rows.push({ label: "shuffled labels", v: control.separation_ratio, color: "#7f8c9b" });
  const top = Math.max(thr * 1.18, 1.3, Math.max.apply(null, rows.map(r => r.v * 1.25)));
  const Y = v => H - PB - (v / top) * ph;
  const bw = pw / (rows.length * 2);

  return (
    <svg width="100%" height="100%" viewBox={"0 0 " + W + " " + H} preserveAspectRatio="xMidYMid meet"
      style={{ fontFamily: "monospace", display: "block" }}>
      {ticks(0, top, 4).map((t, i) => (
        <g key={i}>
          <line x1={PL} y1={Y(t)} x2={W - PR} y2={Y(t)} stroke={GRID} strokeWidth={0.5} />
          <text x={PL - 5} y={Y(t) + 3} textAnchor="end" fontSize={8} fill={MUTED}>{fmt(t)}</text>
        </g>
      ))}
      <line x1={PL} y1={Y(1)} x2={W - PR} y2={Y(1)} stroke={MUTED} strokeWidth={0.8} strokeDasharray="2 3" />
      <text x={W - PR - 2} y={Y(1) - 3} textAnchor="end" fontSize={7.5} fill={MUTED}>
        1.0 grouping carries no information
      </text>
      <line x1={PL} y1={Y(thr)} x2={W - PR} y2={Y(thr)} stroke={RED} strokeWidth={1.2} strokeDasharray="4 3" />
      <text x={W - PR - 2} y={Y(thr) - 3} textAnchor="end" fontSize={7.5} fill={RED}>
        {"declared threshold " + thr}
      </text>
      {rows.map((r, i) => {
        const x = PL + (i + 0.5) * (pw / rows.length) - bw / 2;
        return (
          <g key={i}>
            <rect x={x} y={Y(r.v)} width={bw} height={Math.max(0, H - PB - Y(r.v))}
              fill={r.color} fillOpacity={0.85} />
            <text x={x + bw / 2} y={Y(r.v) - 4} textAnchor="middle" fontSize={9} fill={TEXT}>
              {r.v.toFixed(3)}
            </text>
            <text x={x + bw / 2} y={H - PB + 12} textAnchor="middle" fontSize={8} fill={MUTED}>
              {r.label}
            </text>
          </g>
        );
      })}
      {rows.length === 2 ? (
        <text x={PL + pw / 2} y={H - 8} textAnchor="middle" fontSize={7.5} fill={MUTED}>
          {"apparatus discriminates real from random grouping by " +
            (rows[0].v / (rows[1].v || 1)).toFixed(2) + "x"}
        </text>
      ) : null}
      <line x1={PL} y1={H - PB} x2={W - PR} y2={H - PB} stroke={AXIS} />
    </svg>
  );
}

/** The statistic's two halves: within-group and between-group mean distance. */
function WithinBetween({ separation }) {
  if (!separation) return <Empty />;
  const W = 460, H = 190, PL = 82, PR = 54, PT = 26, PB = 34;
  const pw = W - PL - PR, ph = H - PT - PB;
  const rows = [
    { label: "within group", v: separation.mean_within, n: separation.n_within_pairs, color: "#e07a7a" },
    { label: "between groups", v: separation.mean_between, n: separation.n_between_pairs, color: "#5fa8d3" },
  ];
  const max = Math.max.apply(null, rows.map(r => r.v)) * 1.25 || 1;
  const bh = ph / rows.length;

  return (
    <svg width="100%" height="100%" viewBox={"0 0 " + W + " " + H} preserveAspectRatio="xMidYMid meet"
      style={{ fontFamily: "monospace", display: "block" }}>
      {ticks(0, max, 4).map((t, i) => (
        <g key={i}>
          <line x1={PL + (t / max) * pw} y1={PT} x2={PL + (t / max) * pw} y2={H - PB}
            stroke={GRID} strokeWidth={0.5} />
          <text x={PL + (t / max) * pw} y={H - PB + 11} textAnchor="middle" fontSize={8} fill={MUTED}>
            {fmt(t)}
          </text>
        </g>
      ))}
      {rows.map((r, i) => {
        const y = PT + i * bh + bh * 0.22;
        const w = (r.v / max) * pw;
        return (
          <g key={i}>
            <rect x={PL} y={y} width={w} height={bh * 0.5} fill={r.color} fillOpacity={0.85} />
            <text x={PL - 6} y={y + bh * 0.36} textAnchor="end" fontSize={8.5} fill={TEXT}>{r.label}</text>
            <text x={PL + w + 5} y={y + bh * 0.3} fontSize={8} fill={TEXT}>{r.v.toFixed(3)}</text>
            <text x={PL + w + 5} y={y + bh * 0.3 + 9} fontSize={7} fill={MUTED}>
              {(r.n != null ? r.n.toLocaleString() : "") + " pairs"}
            </text>
          </g>
        );
      })}
      <line x1={PL} y1={H - PB} x2={W - PR} y2={H - PB} stroke={AXIS} />
      <text x={PL + pw / 2} y={14} textAnchor="middle" fontSize={8} fill={MUTED}>
        mean coordinate distance, ratio = between / within
      </text>
    </svg>
  );
}

/* ── Comparison method: similarity decay with acquisition separation ─────── */

function BaselineDecay({ baseline }) {
  const rows = baseline && baseline.by_lag;
  if (!rows || !rows.length) return <Empty />;
  const W = 460, H = 190, PL = 40, PR = 16, PT = 16, PB = 36;
  const pw = W - PL - PR, ph = H - PT - PB;
  const maxLag = Math.max.apply(null, rows.map(r => r.lag)) || 1;
  const X = l => PL + (l / maxLag) * pw;
  const Y = v => H - PB - Math.max(0, Math.min(1, v)) * ph;
  const d = rows.map((r, i) => (i === 0 ? "M" : "L") + X(r.lag) + "," + Y(r.mean)).join(" ");

  return (
    <svg width="100%" height="100%" viewBox={"0 0 " + W + " " + H} preserveAspectRatio="xMidYMid meet"
      style={{ fontFamily: "monospace", display: "block" }}>
      {ticks(0, 1, 4).map((t, i) => (
        <g key={i}>
          <line x1={PL} y1={Y(t)} x2={W - PR} y2={Y(t)} stroke={GRID} strokeWidth={0.5} />
          <text x={PL - 5} y={Y(t) + 3} textAnchor="end" fontSize={8} fill={MUTED}>{t.toFixed(1)}</text>
        </g>
      ))}
      <path d={d + " L" + X(maxLag) + "," + (H - PB) + " L" + X(rows[0].lag) + "," + (H - PB) + " Z"}
        fill={GREEN} fillOpacity={0.14} />
      <path d={d} stroke={GREEN} strokeWidth={1.6} fill="none" />
      {rows.map((r, i) => (
        <g key={i}>
          <circle cx={X(r.lag)} cy={Y(r.mean)} r={2.5} fill="#0d0f12" stroke={GREEN} strokeWidth={1.2} />
          <text x={X(r.lag)} y={H - PB + 11} textAnchor="middle" fontSize={8} fill={MUTED}>{r.lag}</text>
        </g>
      ))}
      {rows[1] ? (
        <text x={X(rows[1].lag) + 6} y={Y(rows[1].mean) - 6} fontSize={8} fill={GREEN}>
          {"adjacent " + rows[1].mean.toFixed(3)}
        </text>
      ) : null}
      <line x1={PL} y1={H - PB} x2={W - PR} y2={H - PB} stroke={AXIS} />
      <text x={PL + pw / 2} y={H - 4} textAnchor="middle" fontSize={8} fill={MUTED}>
        separation in acquisition levels
      </text>
      <text x={10} y={PT + ph / 2} fontSize={8} fill={MUTED}
        transform={"rotate(-90, 10, " + (PT + ph / 2) + ")"} textAnchor="middle">
        {baseline.metric || "similarity"}
      </text>
    </svg>
  );
}

/* ── Parameter sweep against the declared threshold ──────────────────────── */

function SweepChart({ sweep, minRatio }) {
  const grid = sweep && sweep.grid;
  if (!grid || !grid.length) return <Empty />;
  const thr = typeof minRatio === "number" ? minRatio : 2.0;
  const W = 460, H = 190, PL = 40, PR = 58, PT = 16, PB = 36;
  const pw = W - PL - PR, ph = H - PT - PB;
  const byBeta = {};
  for (const g of grid) {
    if (typeof g.ratio !== "number") continue;
    if (!byBeta[g.beta]) byBeta[g.beta] = [];
    byBeta[g.beta].push(g);
  }
  const betas = Object.keys(byBeta).map(Number).sort((a, b) => a - b);
  if (!betas.length) return <Empty />;
  const alphas = Array.from(new Set(grid.map(g => g.alpha))).sort((a, b) => a - b);
  const ratios = grid.map(g => g.ratio).filter(r => typeof r === "number");
  const top = Math.max(thr * 1.15, 1.2, Math.max.apply(null, ratios) * 1.2);
  const X = a => PL + (alphas.length === 1 ? pw / 2 : (alphas.indexOf(a) / (alphas.length - 1)) * pw);
  const Y = v => H - PB - (v / top) * ph;
  const PALETTE = ["#22d3ee", "#5fa8d3", "#a78bfa", "#fbbf24", "#e07a7a"];

  return (
    <svg width="100%" height="100%" viewBox={"0 0 " + W + " " + H} preserveAspectRatio="xMidYMid meet"
      style={{ fontFamily: "monospace", display: "block" }}>
      {ticks(0, top, 4).map((t, i) => (
        <g key={i}>
          <line x1={PL} y1={Y(t)} x2={W - PR} y2={Y(t)} stroke={GRID} strokeWidth={0.5} />
          <text x={PL - 5} y={Y(t) + 3} textAnchor="end" fontSize={8} fill={MUTED}>{fmt(t)}</text>
        </g>
      ))}
      <line x1={PL} y1={Y(thr)} x2={W - PR} y2={Y(thr)} stroke={RED} strokeWidth={1.2} strokeDasharray="4 3" />
      <text x={W - PR + 3} y={Y(thr) + 3} fontSize={7.5} fill={RED}>{thr}</text>
      <line x1={PL} y1={Y(1)} x2={W - PR} y2={Y(1)} stroke={MUTED} strokeWidth={0.8} strokeDasharray="2 3" />
      <text x={W - PR + 3} y={Y(1) + 3} fontSize={7.5} fill={MUTED}>1.0</text>
      {betas.map((b, bi) => {
        const pts = byBeta[b].slice().sort((p, q) => p.alpha - q.alpha);
        const d = pts.map((p, i) => (i === 0 ? "M" : "L") + X(p.alpha) + "," + Y(p.ratio)).join(" ");
        const col = PALETTE[bi % PALETTE.length];
        return (
          <g key={b}>
            <path d={d} stroke={col} strokeWidth={1.4} fill="none" />
            {pts.map((p, i) => (
              <circle key={i} cx={X(p.alpha)} cy={Y(p.ratio)} r={2.1}
                fill="#0d0f12" stroke={col} strokeWidth={1.1} />
            ))}
            <text x={W - PR + 3} y={PT + 10 + bi * 10} fontSize={7.5} fill={col}>{"b=" + b}</text>
          </g>
        );
      })}
      {alphas.map((a, i) => (
        <text key={i} x={X(a)} y={H - PB + 12} textAnchor="middle" fontSize={8} fill={MUTED}>{a}</text>
      ))}
      <line x1={PL} y1={H - PB} x2={W - PR} y2={H - PB} stroke={AXIS} />
      <text x={PL + pw / 2} y={H - 4} textAnchor="middle" fontSize={8} fill={MUTED}>
        {"alpha (mass weight) \u2014 " + sweep.n_settings + " settings, best " + fmt(sweep.best_ratio)}
      </text>
    </svg>
  );
}

/* ── Criterion scorecard ─────────────────────────────────────────────────── */

function CriterionChart({ criterion }) {
  const conds = criterion && criterion.conditions;
  if (!conds || !conds.length) return <Empty />;
  const met = criterion.verdict === "MET";
  return (
    <div className="h-full overflow-y-auto p-2.5">
      <div className="mb-2 flex items-center justify-between rounded px-2.5 py-1.5"
        style={{ background: met ? "#0d2a1c" : "#2a1114",
                 border: "1px solid " + (met ? "#1c6b45" : "#7a2530") }}>
        <span className="font-mono text-[10px]" style={{ color: MUTED }}>{criterion.criterion}</span>
        <span className="font-mono text-[12px] font-bold" style={{ color: met ? GREEN : RED }}>
          {criterion.verdict + " " + criterion.n_passed + "/" + criterion.n_total}
        </span>
      </div>
      <div className="space-y-1">
        {conds.map((c, i) => (
          <div key={i} className="flex items-center gap-2 rounded px-2 py-1"
            style={{ background: "#13161a", border: "1px solid " + GRID }}>
            <span className="font-mono text-[11px] w-3" style={{ color: c.pass ? GREEN : RED }}>
              {c.pass ? "\u2713" : "\u2717"}
            </span>
            <span className="flex-1 truncate text-[10px]" style={{ color: TEXT }}>{c.name}</span>
            <span className="font-mono text-[9px]" style={{ color: MUTED }}>{"need " + c.required}</span>
            <span className="font-mono text-[10px] w-14 text-right" style={{ color: c.pass ? GREEN : RED }}>
              {typeof c.observed === "number" ? c.observed.toFixed(3) : String(c.observed)}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ── Acquisition profile from raw scans ──────────────────────────────────── */

function ScanProfile({ scans }) {
  const data = useMemo(() => {
    if (!scans || !scans.length) return null;
    const by = {};
    for (const s of scans) {
      const k = s.nce;
      if (typeof k !== "number") continue;
      if (!by[k]) by[k] = [];
      by[k].push(s.n_peaks != null ? s.n_peaks : (s.peaks ? s.peaks.length : 0));
    }
    const levels = Object.keys(by).map(Number).sort((a, b) => a - b);
    if (!levels.length) return null;
    return levels.map(l => ({
      l, n: by[l].length,
      mean: by[l].reduce((a, b) => a + b, 0) / by[l].length,
    }));
  }, [scans]);

  if (!data) return <Empty msg="scans carry no acquisition setting" />;
  const W = 460, H = 190, PL = 38, PR = 14, PT = 16, PB = 36;
  const pw = W - PL - PR, ph = H - PT - PB;
  const max = Math.max.apply(null, data.map(d => d.mean)) * 1.2 || 1;
  const bw = pw / data.length;

  return (
    <svg width="100%" height="100%" viewBox={"0 0 " + W + " " + H} preserveAspectRatio="xMidYMid meet"
      style={{ fontFamily: "monospace", display: "block" }}>
      {ticks(0, max, 4).map((t, i) => (
        <g key={i}>
          <line x1={PL} y1={H - PB - (t / max) * ph} x2={W - PR} y2={H - PB - (t / max) * ph}
            stroke={GRID} strokeWidth={0.5} />
          <text x={PL - 5} y={H - PB - (t / max) * ph + 3} textAnchor="end" fontSize={8} fill={MUTED}>
            {fmt(t)}
          </text>
        </g>
      ))}
      {data.map((d, i) => {
        const h = (d.mean / max) * ph;
        return (
          <g key={i}>
            <rect x={PL + i * bw + bw * 0.15} y={H - PB - h} width={bw * 0.7}
              height={Math.max(0, h)} fill="#5fa8d3" fillOpacity={0.82} />
            <text x={PL + i * bw + bw / 2} y={H - PB - h - 3} textAnchor="middle" fontSize={7.5} fill={TEXT}>
              {d.mean.toFixed(1)}
            </text>
            <text x={PL + i * bw + bw / 2} y={H - PB + 11} textAnchor="middle" fontSize={8} fill={MUTED}>
              {d.l}
            </text>
          </g>
        );
      })}
      <line x1={PL} y1={H - PB} x2={W - PR} y2={H - PB} stroke={AXIS} />
      <text x={PL + pw / 2} y={H - 4} textAnchor="middle" fontSize={8} fill={MUTED}>
        acquisition setting, mean peaks per spectrum
      </text>
    </svg>
  );
}

/* ── grid assembly ───────────────────────────────────────────────────────── */

const EXPERIMENT_KINDS = ["coords", "separation", "drift", "criterion",
                          "baselineStat", "sweep", "scans"];

/**
 * True when a workspace holds anything these charts can draw. The sandbox
 * uses this to decide whether the Charts tab has experiment content, since
 * experiment programs bind none of the `records` kind the dashboard wants.
 */
export function hasExperimentData(workspace) {
  const kinds = new Set((workspace || []).map(w => w.kind));
  return EXPERIMENT_KINDS.some(k => kinds.has(k));
}

/** Read a declared bound back out of the scored criterion. */
function boundFrom(criterion, match, fallback) {
  if (!criterion || !criterion.conditions) return fallback;
  const c = criterion.conditions.find(x => match.test(String(x.name)));
  if (!c) return fallback;
  const m = /([0-9]*\.?[0-9]+)/.exec(String(c.required));
  return m ? parseFloat(m[1]) : fallback;
}

export default function ExperimentCharts({ workspace }) {
  const ws = workspace || [];
  const pick = k => {
    const e = ws.find(w => w.kind === k);
    return e ? e.value : undefined;
  };

  const coords = pick("coords");
  const scans = pick("scans");
  const drift = pick("drift");
  const criterion = pick("criterion");
  const baseline = pick("baselineStat");
  const sweep = pick("sweep");

  // A shuffle control is a separation carrying a `control` tag; the true
  // statistic is the one without it.
  const seps = ws.filter(w => w.kind === "separation").map(w => w.value).filter(Boolean);
  const separation = seps.find(s => !s.control) || seps[0];
  const control = seps.find(s => s.control);

  // Draw the bounds the program actually declared, not defaults.
  const minRatio = useMemo(() => boundFrom(criterion, /separation/i, 2.0), [criterion]);
  const maxAbsR = useMemo(() => boundFrom(criterion, /\|r\|/, 0.3), [criterion]);

  if (!hasExperimentData(ws)) return null;

  const frames = [];
  const push = (key, title, note, node) =>
    frames.push(<Frame key={key} title={title} note={note}>{node}</Frame>);

  if (coords && coords.length) {
    push("cloud", "Coordinate cloud (Sk, St, Se)",
      coords.length.toLocaleString() + " spectra", <CoordCloud coords={coords} />);
    push("dens", "Marginal densities per axis", "scaled to unit maximum",
      <AxisDensities coords={coords} />);
  }
  if (drift && coords && coords.length) {
    push("drift", "Drift with " + (drift.over || "setting"), "z-scored per axis",
      <DriftChart drift={drift} coords={coords} />);
  }
  if (drift) {
    push("driftbar", "Correlation against the declared bound", "|r| < " + maxAbsR,
      <DriftBars drift={drift} maxAbsR={maxAbsR} />);
  }
  if (separation) {
    push("sep", "Separation ratio and control", separation.n_groups + " groups",
      <SeparationChart separation={separation} control={control} minRatio={minRatio} />);
    push("wb", "Within and between group distance", null,
      <WithinBetween separation={separation} />);
  }
  if (baseline) {
    push("base", "Comparison method decay",
      (baseline.n_pairs != null ? baseline.n_pairs.toLocaleString() + " pairs" : null),
      <BaselineDecay baseline={baseline} />);
  }
  if (sweep) {
    push("sweep", "Parameter sweep against threshold", sweep.n_settings + " settings",
      <SweepChart sweep={sweep} minRatio={minRatio} />);
  }
  if (scans && scans.length) {
    push("scan", "Acquisition profile", scans.length.toLocaleString() + " scans",
      <ScanProfile scans={scans} />);
  }
  if (criterion) {
    push("crit", "Declared criterion", "stated before the run",
      <CriterionChart criterion={criterion} />);
  }

  const verdict = criterion ? criterion.verdict : null;
  const nSpectra = (coords && coords.length) || (scans && scans.length) || 0;

  return (
    <div className="h-full overflow-y-auto p-3 space-y-3" style={{ background: "#070809" }}>
      <div className="grid grid-cols-4 gap-2">
        <Stat label="spectra" value={nSpectra.toLocaleString()} />
        <Stat label="groups" value={separation ? separation.n_groups : "\u2014"} />
        <Stat label="ratio"
          value={separation ? separation.separation_ratio.toFixed(3) : "\u2014"}
          color={separation ? (separation.separation_ratio > minRatio ? GREEN : RED) : TEXT}
          sub={separation ? "need > " + minRatio : null} />
        <Stat label="verdict" value={verdict || "\u2014"}
          color={verdict === "MET" ? GREEN : (verdict ? RED : TEXT)}
          sub={criterion ? criterion.n_passed + "/" + criterion.n_total + " conditions" : null} />
      </div>
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-1">{frames}</div>
    </div>
  );
}
