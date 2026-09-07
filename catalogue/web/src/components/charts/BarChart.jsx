// Bar chart: rung sensitivities, ablation losses, verdict class means.
//
// Bars are drawn anchored to the baseline with rounded data-ends and a
// 2px surface gap between neighbours, so adjacent bars read as separate
// marks without a stroke.

import { scaleLinear, scaleBand } from 'd3'
import { PAD, SERIES, Svg, useTooltip, AXIS } from './chart-kit.jsx'

export default function BarChart({
  bars,
  yDomain,
  w = 640,
  h = 260,
  yLabel,
  yFmt,
  rules = [],
  horizontal = false,
}) {
  const { tip, show, hide } = useTooltip()
  const names = bars.map((b) => b.label)

  if (horizontal) {
    const y = scaleBand().domain(names).range([PAD.t, h - PAD.b]).padding(0.28)
    const x = scaleLinear().domain(yDomain).range([PAD.l, w - PAD.r])
    return (
      <Svg w={w} h={h} tip={tip}>
        {bars.map((b, i) => (
          <g key={i}>
            <rect
              x={PAD.l}
              y={y(b.label)}
              width={Math.max(0, x(b.value) - PAD.l)}
              height={y.bandwidth()}
              rx="4"
              fill={b.color || SERIES[i % SERIES.length]}
              onMouseEnter={() =>
                show(x(b.value), y(b.label), w, h, (
                  <>
                    <b>{b.label}</b>
                    <br />
                    {yFmt ? yFmt(b.value) : b.value}
                  </>
                ))
              }
              onMouseLeave={hide}
            />
            <text x={PAD.l - 7} y={y(b.label) + y.bandwidth() / 2 + 3.2} textAnchor="end" {...AXIS}>
              {b.label}
            </text>
            <text
              x={x(b.value) + 6}
              y={y(b.label) + y.bandwidth() / 2 + 3.2}
              {...AXIS}
              fill="var(--ink-dim)"
            >
              {yFmt ? yFmt(b.value) : b.value}
            </text>
          </g>
        ))}
      </Svg>
    )
  }

  const x = scaleBand().domain(names).range([PAD.l, w - PAD.r]).padding(0.26)
  const y = scaleLinear().domain(yDomain).range([h - PAD.b, PAD.t])
  const base = y(Math.max(yDomain[0], 0))

  return (
    <Svg w={w} h={h} tip={tip}>
      {y.ticks(5).map((t, i) => (
        <g key={'y' + i}>
          <line x1={PAD.l} x2={w - PAD.r} y1={y(t)} y2={y(t)} stroke="var(--line)" strokeWidth="1" />
          <text x={PAD.l - 7} y={y(t) + 3.2} textAnchor="end" {...AXIS}>
            {yFmt ? yFmt(t) : t}
          </text>
        </g>
      ))}
      {rules.map((r, i) => (
        <line
          key={'r' + i}
          x1={PAD.l} x2={w - PAD.r} y1={y(r.at)} y2={y(r.at)}
          stroke={r.color || 'var(--ink-faint)'} strokeWidth="1" strokeDasharray="5 4"
        />
      ))}
      {bars.map((b, i) => {
        const top = Math.min(y(b.value), base)
        const height = Math.abs(y(b.value) - base)
        return (
          <g key={i}>
            <rect
              x={x(b.label)}
              y={top}
              width={x.bandwidth()}
              height={Math.max(1, height)}
              rx="4"
              fill={b.color || SERIES[i % SERIES.length]}
              onMouseEnter={() =>
                show(x(b.label) + x.bandwidth() / 2, top, w, h, (
                  <>
                    <b>{b.label}</b>
                    <br />
                    {yFmt ? yFmt(b.value) : b.value}
                  </>
                ))
              }
              onMouseLeave={hide}
            />
            <text x={x(b.label) + x.bandwidth() / 2} y={h - PAD.b + 15} textAnchor="middle" {...AXIS}>
              {b.label}
            </text>
          </g>
        )
      })}
      {yLabel ? (
        <text
          x={-(PAD.t + h - PAD.b) / 2}
          y={11}
          textAnchor="middle"
          transform="rotate(-90)"
          {...AXIS}
          fill="var(--ink-dim)"
        >
          {yLabel}
        </text>
      ) : null}
    </Svg>
  )
}
