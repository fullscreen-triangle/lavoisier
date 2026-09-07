// Heat map: candidate degeneracy over (m/z, tolerance).
//
// Sequential magnitude, so one hue light to dark --- never a rainbow. The
// scale is explicitly log-capable because degeneracy here spans a median
// of 183 against a maximum of 4500, and a linear ramp would render every
// cell but one at the same step.

import { scaleSequential, scaleLog, scaleLinear, interpolateBlues } from 'd3'
import { PAD, AXIS, Svg, useTooltip } from './chart-kit.jsx'

export default function HeatChart({
  cells,
  xLabels,
  yLabels,
  w = 640,
  h = 320,
  xLabel,
  yLabel,
  vLabel = 'value',
  logV = false,
  vFmt,
}) {
  const { tip, show, hide } = useTooltip()

  const vals = cells.map((c) => c.v).filter((v) => v > 0)
  const lo = vals.length ? Math.min(...vals) : 0
  const hi = vals.length ? Math.max(...vals) : 1

  const base = logV
    ? scaleLog().domain([Math.max(lo, 1e-9), hi]).range([0.12, 1])
    : scaleLinear().domain([lo, hi]).range([0.12, 1])
  const colour = (v) => (v > 0 ? interpolateBlues(base(Math.max(v, lo))) : 'var(--panel-2)')

  const gw = (w - PAD.l - PAD.r - 70) / xLabels.length
  const gh = (h - PAD.t - PAD.b) / yLabels.length
  const fmt = vFmt || ((v) => String(v))

  // Legend: a vertical ramp of eleven steps on the right.
  const steps = Array.from({ length: 11 }, (_, i) => i / 10)

  return (
    <Svg w={w} h={h} tip={tip}>
      {cells.map((c, i) => {
        const cx = PAD.l + c.x * gw
        const cy = PAD.t + c.y * gh
        return (
          <rect
            key={i}
            x={cx + 1} y={cy + 1}
            width={Math.max(1, gw - 2)} height={Math.max(1, gh - 2)}
            rx="2"
            fill={colour(c.v)}
            onMouseEnter={() =>
              show(cx + gw / 2, cy + gh / 2, w, h, (
                <>
                  <b>{xLabels[c.x] + ' x ' + yLabels[c.y]}</b>
                  <br />
                  {vLabel + ' ' + fmt(c.v)}
                </>
              ))
            }
            onMouseLeave={hide}
          />
        )
      })}

      {xLabels.map((l, i) => (
        <text key={'x' + i} x={PAD.l + (i + 0.5) * gw} y={h - PAD.b + 14} textAnchor="middle" {...AXIS}>
          {l}
        </text>
      ))}
      {yLabels.map((l, i) => (
        <text key={'y' + i} x={PAD.l - 7} y={PAD.t + (i + 0.5) * gh + 3.2} textAnchor="end" {...AXIS}>
          {l}
        </text>
      ))}

      <g transform={'translate(' + (w - PAD.r - 52) + ',' + PAD.t + ')'}>
        {steps.map((s, i) => (
          <rect
            key={i}
            x="0"
            y={(10 - i) * 12}
            width="12" height="12"
            fill={interpolateBlues(0.12 + s * 0.88)}
          />
        ))}
        <text x="18" y="9" {...AXIS}>{fmt(hi)}</text>
        <text x="18" y="129" {...AXIS}>{fmt(lo)}</text>
        <text x="0" y="150" {...AXIS} fill="var(--ink-dim)">{vLabel}</text>
      </g>

      {xLabel ? (
        <text x={PAD.l + (w - PAD.l - PAD.r - 70) / 2} y={h - 3} textAnchor="middle" {...AXIS} fill="var(--ink-dim)">
          {xLabel}
        </text>
      ) : null}
      {yLabel ? (
        <text
          x={-(PAD.t + h - PAD.b) / 2} y={11}
          textAnchor="middle" transform="rotate(-90)" {...AXIS} fill="var(--ink-dim)"
        >
          {yLabel}
        </text>
      ) : null}
    </Svg>
  )
}
