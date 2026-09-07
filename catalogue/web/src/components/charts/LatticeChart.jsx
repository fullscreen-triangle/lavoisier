// The refinement lattice of observation groupings.
//
// Nodes are partitions, ranked by block count; an edge means the lower
// partition refines the upper one by exactly one merge. The lattice is
// drawn by rank, not by force, because the rank IS the reading: a
// partition sits above everything it coarsens.

import { SERIES, Svg, useTooltip } from './chart-kit.jsx'

export default function LatticeChart({
  levels,
  edges = [],
  highlight = [],
  w = 640,
  h = 380,
  rankLabel = 'blocks',
}) {
  const { tip, show, hide } = useTooltip()

  // levels: [{ rank, nodes: [{id, label, note, value}] }], top rank first.
  const rowH = (h - 40) / Math.max(1, levels.length)
  const pos = {}
  levels.forEach((lv, li) => {
    const y = 24 + rowH * (li + 0.5)
    lv.nodes.forEach((n, ni) => {
      const x = 60 + ((ni + 1) / (lv.nodes.length + 1)) * (w - 120)
      pos[n.id] = [x, y]
    })
  })

  const isHot = (id) => highlight.indexOf(id) !== -1

  return (
    <Svg w={w} h={h} tip={tip}>
      {edges.map((e, i) => {
        const a = pos[e[0]]
        const b = pos[e[1]]
        if (!a || !b) return null
        const hot = isHot(e[0]) && isHot(e[1])
        return (
          <line
            key={i}
            x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]}
            stroke={hot ? SERIES[0] : 'var(--line)'}
            strokeWidth={hot ? 2 : 1}
          />
        )
      })}

      {levels.map((lv, li) => (
        <text
          key={'lb' + li}
          x="10"
          y={24 + rowH * (li + 0.5) + 3.5}
          fontSize="9.5"
          fill="var(--ink-faint)"
          fontFamily="var(--mono)"
        >
          {lv.rank + ' ' + rankLabel}
        </text>
      ))}

      {levels.map((lv) =>
        lv.nodes.map((n) => (
          <g
            key={n.id}
            onMouseEnter={() =>
              show(pos[n.id][0], pos[n.id][1], w, h, (
                <>
                  <b>{n.label || n.id}</b>
                  {n.note ? (<><br />{n.note}</>) : null}
                  {n.value !== undefined ? (<><br />{'statistic ' + n.value}</>) : null}
                </>
              ))
            }
            onMouseLeave={hide}
          >
            <circle
              cx={pos[n.id][0]} cy={pos[n.id][1]}
              r={isHot(n.id) ? 12 : 9}
              fill={isHot(n.id) ? SERIES[0] : 'var(--panel-2)'}
              stroke={isHot(n.id) ? SERIES[0] : 'var(--line)'}
              strokeWidth="1.5"
            />
            <text
              x={pos[n.id][0]} y={pos[n.id][1] - 15}
              textAnchor="middle" fontSize="9.5"
              fill={isHot(n.id) ? 'var(--ink)' : 'var(--ink-faint)'}
              fontFamily="var(--mono)"
            >
              {n.label || n.id}
            </text>
          </g>
        ))
      )}
    </Svg>
  )
}
