// =====================================================================
//  One notebook cell.
//
//  A transparent <textarea> sits exactly on top of a highlighted <pre>.
//  The textarea holds the caret and the text; the pre holds the colour.
//  They must share font, size, line height and padding to the pixel or
//  the caret drifts away from the glyphs --- .cell-src in styles.css sets
//  those on both, and neither should be styled independently.
// =====================================================================

import { useState, useMemo, useRef } from 'react'
import { run } from '../../lib/ss/compiler.js'
import { highlight } from './highlight.js'
import { classify } from './outcome.js'
import { Pill } from '../Primitives.jsx'
import Outputs from './Outputs.jsx'

function Terminal({ lines }) {
  if (!lines || !lines.length) return null
  return (
    <div className="terminal">
      {lines.map((l, i) => (
        <div key={i} className={'stream-' + (l.stream === 'stderr' ? 'stderr' : 'stdout')}>
          {l.stream === 'stage' ? '$ ' + l.text : l.text}
        </div>
      ))}
    </div>
  )
}

function Workspace({ bindings }) {
  if (!bindings || !bindings.length) return null
  return (
    <table>
      <thead>
        <tr>
          <th>binding</th>
          <th>kind</th>
          <th>value</th>
        </tr>
      </thead>
      <tbody>
        {bindings.map((b) => (
          <tr key={b.name}>
            <td className="mono">{b.name}</td>
            <td>{b.kind}</td>
            <td className="mono" style={{ color: 'var(--ink-dim)' }}>
              {summarise(b.value)}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function summarise(v) {
  if (v === null || v === undefined) return '-'
  if (Array.isArray(v)) return '[' + v.length + ' item(s)]'
  if (typeof v === 'object') {
    const keys = Object.keys(v).filter((k) => !k.startsWith('_'))
    return '{' + keys.slice(0, 4).join(', ') + (keys.length > 4 ? ', ...' : '') + '}'
  }
  return String(v)
}

export default function Cell({ title, note, source, autorun = true }) {
  const [text, setText] = useState(source)
  const [ran, setRan] = useState(autorun ? source : null)
  const taRef = useRef(null)

  // Re-running is the only expensive thing a cell does, and it is keyed
  // on the exact source that was submitted, so editing does not run.
  const payload = useMemo(() => (ran === null ? null : run(ran)), [ran])
  const outcome = payload ? classify(payload) : null
  const dirty = ran !== null && text !== ran

  const doRun = () => setRan(text)
  const onKey = (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
      e.preventDefault()
      doRun()
    }
  }

  const lines = text.split('\n').length

  return (
    <div className="cell">
      <div className="cell-head">
        <b>{title}</b>
        {outcome ? <Pill verdict={outcome.kind === 'ok' ? 'ok' : outcome.kind} /> : null}
        <span style={{ marginLeft: 'auto' }} />
        {dirty ? <span className="note" style={{ margin: 0 }}>edited, not yet run</span> : null}
        <button className={dirty ? 'on' : ''} onClick={doRun}>
          Run
        </button>
        <button className="ghost" onClick={() => { setText(source); setRan(source) }}>
          Reset
        </button>
      </div>

      {note ? <p className="note">{note}</p> : null}

      <div className="cell-src" style={{ height: lines * 1.62 * 12.6 + 32 + 'px' }}>
        <pre aria-hidden="true" dangerouslySetInnerHTML={{ __html: highlight(text) + '\n' }} />
        <textarea
          ref={taRef}
          spellCheck="false"
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={onKey}
          aria-label={title + ' source'}
        />
      </div>
      <div className="note" style={{ marginTop: 6 }}>
        Edit the source and press Run, or Ctrl/Cmd-Enter.
      </div>

      {payload ? (
        <div className="cell-out">
          {outcome && outcome.blocker ? (
            <>
              <div className="blocker">
                <b>{outcome.label}</b> {outcome.blocker}
              </div>
              <div className="unblock">
                <b>what would satisfy it</b> {outcome.unblock}
              </div>
            </>
          ) : null}

          {outcome && outcome.note ? <div className="replayed">{outcome.note}</div> : null}

          <Terminal lines={payload.compile.terminal} />
          {payload.execute ? <Terminal lines={payload.execute.terminal} /> : null}
          {payload.execute ? <Workspace bindings={payload.execute.workspace} /> : null}
          {payload.execute ? <Outputs payload={payload} /> : null}
        </div>
      ) : null}
    </div>
  )
}
