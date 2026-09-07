// An ordered list of cells.
//
// Cells are independent: each owns its source and its run, and there is
// no shared kernel state between them. That is a deliberate limit, not
// an omission --- a shared namespace would let one cell silently change
// what another one means, and every number on this site has to be
// traceable to the source shown directly above it.

import { useState } from 'react'
import Cell from './Cell.jsx'

export default function Notebook({ title, intro, cells }) {
  // Bumping the epoch remounts every cell, which is the reset.
  const [epoch, setEpoch] = useState(0)

  return (
    <div className="notebook">
      <div className="notebook-head">
        <b>{title || 'Run it yourself'}</b>
        <span style={{ marginLeft: 'auto' }} />
        <button className="ghost" onClick={() => setEpoch(epoch + 1)}>
          Reset all cells
        </button>
      </div>
      {intro ? <p className="sub">{intro}</p> : null}
      {cells.map((c, i) => (
        <Cell
          key={i + ':' + epoch}
          title={c.title}
          note={c.note}
          source={c.source}
          autorun={c.autorun !== false}
        />
      ))}
    </div>
  )
}
