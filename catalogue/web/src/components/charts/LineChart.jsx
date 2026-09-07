// Line chart: residual gap staircases, beta sweeps, tolerance sweeps.
//
// Domains are passed in by the caller and pinned. A slider that changes
// the data must not rescale the axes underneath the reader, or the shape
// of the change becomes invisible.

import { scaleLinear, scaleLog, line as d3line, curveLinear, curveStepAfter } from 'd3'
import { PAD, SERIES, Frame, Legend, Svg, useTooltip } from './chart-kit.jsx'

export default function LineChart({
  series,
  xDomain,
  yDomain,
  w = 640,
  h = 300,
  xLabel,
  yLabel,
  xFmt,
  yFmt,
  step = false,
  logY = false,
  rules = [],
}) {
  const { tip, show, hide } = useTooltip()
  const x = scaleLinear().domain(xDomain).range([PAD.l, w - PAD.r])
  const y = (logY ? scaleLog() : scaleLinear()).domain(yDomain).range([h - PAD.b, PAD.t])
  const path = d3line()
    .x((d) => x(d[0]))
    .y((d) => y(d[1]))
    .curve(step ? curveStepAfter : curveLinear)

  return (
    <Svg w={w} h={h} tip={tip}>
        <Frame
          w={w} h={h} x={x} y={y}
          xLabel={xLabel} yLabel={yLabel} xFmt={xFmt} yFmt={yFmt}
        >
          {rules.map((r, i) => (
            <g key={'r' + i}>
              <line
                x1={PAD.l} x2={w - PAD.r} y1={y(r.at)} y2={y(r.at)}
                stroke={r.color || 'var(--ink-faint)'} strokeWidth="1" strokeDasharray="5 4"
              />
              {r.label ? (
                <text x={w - PAD.r} y={y(r.at) - 4} textAnchor="end" fontSize="9.5"
                  fill={r.color || 'var(--ink-faint)'} fontFamily="var(--mono)">
                  {r.label}
                </text>
              ) : null}
            </g>
          ))}

          {series.map((s, si) => (
            <g key={si}>
              <path
                d={path(s.points)}
                fill="none"
                stroke={s.color || SERIES[si % SERIES.length]}
                strokeWidth="2"
                strokeLinejoin="round"
              />
              {s.points.map((p, pi) => (
                <circle
                  key={pi}
                  cx={x(p[0])}
                  cy={y(p[1])}
                  r="4.5"
                  fill={s.color || SERIES[si % SERIES.length]}
                  stroke="var(--panel)"
                  strokeWidth="2"
                  onMouseEnter={() =>
                    show(x(p[0]), y(p[1]), w, h, (
                      <>
                        <b>{s.label}</b>
                        <br />
                        {(xLabel || 'x') + ' ' + (xFmt ? xFmt(p[0]) : p[0])}
                        <br />
                        {(yLabel || 'y') + ' ' + (yFmt ? yFmt(p[1]) : p[1])}
                      </>
                    ))
                  }
                  onMouseLeave={hide}
                />
              ))}
            </g>
          ))}
        </Frame>
        {series.length > 1 ? (
          <Legend
            w={w}
            items={series.map((s, i) => ({ label: s.label, color: s.color || SERIES[i % SERIES.length] }))}
          />
        ) : null}
    </Svg>
  )
}
