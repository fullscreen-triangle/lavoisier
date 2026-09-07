// =====================================================================
//  Shared chart plumbing.
//
//  D3 supplies scales, axis ticks, shape generators and interpolators.
//  React owns the DOM: no chart in this app calls d3.select on a node
//  React rendered. The one exception is brush and zoom, which cannot be
//  expressed declaratively and get an explicit useRef handoff.
//
//  Every SVG follows the reference convention --- a viewBox in design
//  units with width:100%, height:auto --- so charts scale without a
//  resize observer and without a media query.
// =====================================================================

import { useState, useCallback } from 'react'

export const PAD = { t: 14, r: 16, l: 52, b: 34 }

/* Series colours are literal hex, not CSS variables: variables are for
   chrome. Chosen so adjacent pairs stay separable under deuteranopia and
   protanopia, and so no two neighbours share a lightness band. */
export const SERIES = ['#58a6ff', '#eb8c34', '#7ee2c0', '#b98cf0', '#d9d36a', '#f07b7b']
export const OKC = '#3fb950'
export const BADC = '#f85149'
export const WARNC = '#d29922'

export const AXIS = {
  fontSize: 9.5,
  fill: 'var(--ink-faint)',
  fontFamily: 'var(--mono)',
}

// Nice-ish tick values without pulling in the axis component.
export function ticks(scale, n = 5) {
  return scale.ticks ? scale.ticks(n) : []
}

/* Tooltip state, shared by every chart.

   Position is in SVG design units and the tooltip is rendered as an HTML
   overlay positioned in percentages, so it tracks correctly at any
   rendered size without measuring the DOM. */
export function useTooltip() {
  const [tip, setTip] = useState(null)
  const show = useCallback((x, y, w, h, content) => {
    setTip({ left: (x / w) * 100, top: (y / h) * 100, content })
  }, [])
  const hide = useCallback(() => setTip(null), [])
  return { tip, show, hide }
}

export function Tooltip({ tip }) {
  if (!tip) return null
  return (
    <div
      className="tooltip"
      style={{
        left: tip.left + '%',
        top: tip.top + '%',
        transform:
          'translate(' + (tip.left > 60 ? '-108%' : '8%') + ', ' +
          (tip.top > 60 ? '-108%' : '8%') + ')',
      }}
    >
      {tip.content}
    </div>
  )
}

// A chart frame: gridlines, tick labels, axis titles. Marks go in as
// children, drawn on top.
export function Frame({ w, h, x, y, xLabel, yLabel, xTicks = 5, yTicks = 5, xFmt, yFmt, children }) {
  const xs = ticks(x, xTicks)
  const ys = ticks(y, yTicks)
  return (
    <>
      {ys.map((t, i) => (
        <g key={'y' + i}>
          <line x1={PAD.l} x2={w - PAD.r} y1={y(t)} y2={y(t)} stroke="var(--line)" strokeWidth="1" />
          <text x={PAD.l - 7} y={y(t) + 3.2} textAnchor="end" {...AXIS}>
            {yFmt ? yFmt(t) : t}
          </text>
        </g>
      ))}
      {xs.map((t, i) => (
        <text key={'x' + i} x={x(t)} y={h - PAD.b + 15} textAnchor="middle" {...AXIS}>
          {xFmt ? xFmt(t) : t}
        </text>
      ))}
      {children}
      {xLabel ? (
        <text x={(PAD.l + w - PAD.r) / 2} y={h - 3} textAnchor="middle" {...AXIS} fill="var(--ink-dim)">
          {xLabel}
        </text>
      ) : null}
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
    </>
  )
}

export function Legend({ items, w }) {
  return (
    <>
      {items.map((s, i) => (
        <g key={i} transform={'translate(' + (w - PAD.r - 118) + ',' + (PAD.t + 4 + i * 14) + ')'}>
          <rect width="9" height="9" rx="2" fill={s.color} />
          <text x="14" y="8.4" fontSize="10.5" fill="var(--ink-dim)" fontFamily="var(--sans)">
            {s.label}
          </text>
        </g>
      ))}
    </>
  )
}

// The tooltip is an HTML overlay positioned in percentages, so it must
// live inside .chart (which is position:relative) rather than beside it.
export function Svg({ w, h, tip, children }) {
  return (
    <div className="chart">
      <svg viewBox={'0 0 ' + w + ' ' + h} style={{ width: '100%', height: 'auto' }}>
        {children}
      </svg>
      <Tooltip tip={tip} />
    </div>
  )
}
