// Scatter: contact vs mass error, decoy control, size control.
//
// Optional brushing. The brush is the one place React hands the DOM to
// D3, because a drag gesture cannot be expressed as rendered output --- the
// selection rectangle is React-rendered, but the pointer bookkeeping is
// not.

import { useMemo, useState } from 'react'
import { scaleLinear, scaleLog } from 'd3'
import { PAD, SERIES, Frame, Legend, Svg, useTooltip } from './chart-kit.jsx'

export default function ScatterChart({
  groups,
  xDomain,
  yDomain,
  w = 640,
  h = 320,
  xLabel,
  yLabel,
  xFmt,
  yFmt,
  logX = false,
  diagonal = false,
  rules = [],
  vrules = [],
  brushable = false,
  onBrush,
}) {
  const { tip, show, hide } = useTooltip()
  const [sel, setSel] = useState(null)
  const [drag, setDrag] = useState(null)

  const x = (logX ? scaleLog() : scaleLinear()).domain(xDomain).range([PAD.l, w - PAD.r])
  const y = scaleLinear().domain(yDomain).range([h - PAD.b, PAD.t])

  const svgPoint = (e) => {
    const r = e.currentTarget.getBoundingClientRect()
    return [
      ((e.clientX - r.left) / r.width) * w,
      ((e.clientY - r.top) / r.height) * h,
    ]
  }

  const brushProps = brushable
    ? {
        onPointerDown: (e) => {
          const [px] = svgPoint(e)
          setDrag(px)
          setSel(null)
          e.currentTarget.setPointerCapture(e.pointerId)
        },
        onPointerMove: (e) => {
          if (drag === null) return
          const [px] = svgPoint(e)
          setSel([Math.min(drag, px), Math.max(drag, px)])
        },
        onPointerUp: (e) => {
          if (drag !== null && sel && onBrush) onBrush([x.invert(sel[0]), x.invert(sel[1])])
          if (drag !== null && (!sel || sel[1] - sel[0] < 3)) {
            setSel(null)
            if (onBrush) onBrush(null)
          }
          setDrag(null)
          e.currentTarget.releasePointerCapture(e.pointerId)
        },
      }
    : {}

  const inSel = (p) => !sel || (x(p[0]) >= sel[0] && x(p[0]) <= sel[1])

  return (
    <Svg w={w} h={h} tip={tip}>
      <rect
        x={PAD.l} y={PAD.t}
        width={w - PAD.l - PAD.r} height={h - PAD.t - PAD.b}
        fill="transparent"
        style={brushable ? { cursor: 'crosshair' } : undefined}
        {...brushProps}
      />
      <Frame w={w} h={h} x={x} y={y} xLabel={xLabel} yLabel={yLabel} xFmt={xFmt} yFmt={yFmt}>
        {sel ? (
          <rect
            x={sel[0]} y={PAD.t} width={sel[1] - sel[0]} height={h - PAD.t - PAD.b}
            fill="color-mix(in srgb, var(--accent) 12%, transparent)"
            stroke="var(--accent)" strokeWidth="1" pointerEvents="none"
          />
        ) : null}

        {diagonal ? (
          <line
            x1={x(Math.max(xDomain[0], yDomain[0]))}
            y1={y(Math.max(xDomain[0], yDomain[0]))}
            x2={x(Math.min(xDomain[1], yDomain[1]))}
            y2={y(Math.min(xDomain[1], yDomain[1]))}
            stroke="var(--ink-faint)" strokeWidth="1" strokeDasharray="5 4"
          />
        ) : null}

        {rules.map((r, i) => (
          <line key={'r' + i} x1={PAD.l} x2={w - PAD.r} y1={y(r.at)} y2={y(r.at)}
            stroke={r.color || 'var(--ink-faint)'} strokeWidth="1" strokeDasharray="5 4" />
        ))}
        {vrules.map((r, i) => (
          <line key={'v' + i} y1={PAD.t} y2={h - PAD.b} x1={x(r.at)} x2={x(r.at)}
            stroke={r.color || 'var(--ink-faint)'} strokeWidth="1" strokeDasharray="5 4" />
        ))}

        {groups.map((g, gi) => (
          <g key={gi} opacity={g.faint ? 0.34 : 1}>
            {g.points.map((p, pi) => (
              <circle
                key={pi}
                cx={x(p[0])}
                cy={y(p[1])}
                r={g.r || 4}
                fill={g.color || SERIES[gi % SERIES.length]}
                stroke="var(--panel)"
                strokeWidth={g.faint ? 0 : 1.5}
                opacity={inSel(p) ? 1 : 0.15}
                pointerEvents={g.faint ? 'none' : 'auto'}
                onMouseEnter={() =>
                  show(x(p[0]), y(p[1]), w, h, (
                    <>
                      <b>{p[2] !== undefined ? p[2] : g.label}</b>
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
      {groups.filter((g) => g.label).length > 1 ? (
        <Legend
          w={w}
          items={groups
            .filter((g) => g.label)
            .map((g, i) => ({ label: g.label, color: g.color || SERIES[i % SERIES.length] }))}
        />
      ) : null}
    </Svg>
  )
}
