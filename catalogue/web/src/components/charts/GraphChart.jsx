// Contact graph with the medium vertex, and the cut that separates two
// items.
//
// Layout is deterministic, not force-directed: a force simulation would
// move the same graph to a different place on every render, and these
// graphs are read for their cuts, not their aesthetics. Items sit on a
// circle; the medium sits at the centre, adjacent to everything, which
// is what makes it a medium.

import { SERIES, Svg, useTooltip } from './chart-kit.jsx'

export default function GraphChart({
  nodes,
  edges,
  cut = [],
  w = 560,
  h = 380,
  mediumRemoved = false,
}) {
  const { tip, show, hide } = useTooltip()
  const items = nodes.filter((n) => !n.medium)
  const medium = nodes.find((n) => n.medium)

  const cx = w / 2
  const cy = h / 2
  const R = Math.min(w, h) / 2 - 46

  const pos = {}
  items.forEach((n, i) => {
    const a = (i / items.length) * Math.PI * 2 - Math.PI / 2
    pos[n.id] = [cx + R * Math.cos(a), cy + R * Math.sin(a)]
  })
  if (medium) pos[medium.id] = [cx, cy]

  const isCut = (e) =>
    cut.some((c) => (c[0] === e[0] && c[1] === e[1]) || (c[0] === e[1] && c[1] === e[0]))
  const touchesMedium = (e) => medium && (e[0] === medium.id || e[1] === medium.id)
  const live = edges.filter((e) => !(mediumRemoved && touchesMedium(e)))

  return (
    <Svg w={w} h={h} tip={tip}>
      {live.map((e, i) => {
        const a = pos[e[0]]
        const b = pos[e[1]]
        if (!a || !b) return null
        const cutEdge = isCut(e)
        return (
          <line
            key={i}
            x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]}
            stroke={cutEdge ? '#f85149' : 'var(--line)'}
            strokeWidth={cutEdge ? 2.5 : 1.5}
            strokeDasharray={cutEdge ? '5 4' : undefined}
          />
        )
      })}

      {medium && !mediumRemoved ? (
        <g
          onMouseEnter={() =>
            show(pos[medium.id][0], pos[medium.id][1], w, h, (
              <>
                <b>{medium.label || medium.id}</b>
                <br />
                the medium: adjacent to every item by definition, not by measurement
              </>
            ))
          }
          onMouseLeave={hide}
        >
          <circle cx={cx} cy={cy} r="15" fill="#7ee2c0" stroke="var(--panel)" strokeWidth="2.5" />
          <text x={cx} y={cy + 4} textAnchor="middle" fontSize="11" fontFamily="var(--mono)" fill="#0e1116">
            {medium.label || 'm'}
          </text>
        </g>
      ) : null}
      {medium && mediumRemoved ? (
        <circle cx={cx} cy={cy} r="15" fill="none" stroke="var(--ink-faint)"
          strokeWidth="1.5" strokeDasharray="4 3" />
      ) : null}

      {items.map((n, i) => (
        <g
          key={n.id}
          onMouseEnter={() =>
            show(pos[n.id][0], pos[n.id][1], w, h, (
              <>
                <b>{n.label || n.id}</b>
                {n.note ? (<><br />{n.note}</>) : null}
              </>
            ))
          }
          onMouseLeave={hide}
        >
          <circle
            cx={pos[n.id][0]} cy={pos[n.id][1]} r="13"
            fill={n.color || SERIES[(n.side || 0) % SERIES.length]}
            stroke="var(--panel)" strokeWidth="2.5"
          />
          <text
            x={pos[n.id][0]} y={pos[n.id][1] + 4}
            textAnchor="middle" fontSize="10.5" fontFamily="var(--mono)" fill="#0e1116"
          >
            {n.label || n.id}
          </text>
        </g>
      ))}
    </Svg>
  )
}
