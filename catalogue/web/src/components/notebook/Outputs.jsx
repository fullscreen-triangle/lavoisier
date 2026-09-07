// =====================================================================
//  Cell outputs.
//
//  Dispatches on the KIND of each workspace binding, not on which page
//  is rendering, so a cell the reader edits into a different shape still
//  draws the right thing. Everything here reads the values the runtime
//  just produced --- nothing on this page is a stored picture of a run.
//
//  The one exception is a binding carrying _replayed, which came from a
//  recorded run rather than from arithmetic performed in the browser.
//  Those are badged, every time, without exception.
// =====================================================================

import LineChart from '../charts/LineChart.jsx'
import BarChart from '../charts/BarChart.jsx'
import { SERIES, BADC } from '../charts/chart-kit.jsx'
import { Pill } from '../Primitives.jsx'

const f5 = (v) => (typeof v === 'number' ? v.toFixed(5) : String(v))
const f3 = (v) => (typeof v === 'number' ? v.toFixed(3) : String(v))

/* --------------------------------------------------------- ladder ---- */

// lavoisier.ladder.resolve: the residual gap staircase plus the per-rung
// sensitivities. The staircase is a step curve because the gap does not
// change BETWEEN contacts --- drawing it as a slope would claim
// intermediate states the model does not have.
function Resolve({ name, v }) {
  const n = v.rungs.length
  // rungs[i].gap_after is the residual once contact i has been applied,
  // so the curve starts at the whole gap and drops at each contact.
  const points = [[0, 1.0]].concat(v.rungs.map((r, i) => [i + 1, r.gap_after]))

  const sens = v.rungs.map((r) => r.sensitivity)
  const worst = Math.max(...sens)

  return (
    <div className="out-block">
      <div className="out-head">
        <b className="mono">{name}</b>
        <span className="note" style={{ margin: 0 }}>
          composite {f5(v.composite)} over {n} contact(s)
        </span>
        {v.requirement ? (
          <Pill verdict={v.requirement.satisfied ? 'PASS' : 'FAIL'} />
        ) : null}
      </div>

      <div className="grid g2">
        <LineChart
          series={[{ label: 'residual gap', points, color: SERIES[0] }]}
          xDomain={[0, n]}
          yDomain={[0, 1]}
          xLabel="contacts applied"
          yLabel="residual gap"
          xFmt={(t) => String(t)}
          yFmt={f3}
          step
          h={260}
          rules={
            v.requirement
              ? [{ at: 1 - v.requirement.value, color: BADC, label: 'requirement' }]
              : []
          }
        />
        <BarChart
          bars={v.rungs.map((r) => ({
            label: r.name,
            value: r.sensitivity,
            color: r.sensitivity === worst ? SERIES[1] : SERIES[0],
          }))}
          yDomain={[0, worst * 1.15]}
          yLabel="sensitivity"
          yFmt={f3}
          h={260}
        />
      </div>
      <p className="note">
        The staircase is the gap that remains after each contact; the bars are
        the derivative of the composite with respect to each contact power.
        Strongest contact: <b className="mono">{v.strongest_rung}</b>. Ranked{' '}
        <span className="mono">{v.sensitivity_rank.join(' > ')}</span>.
      </p>
    </div>
  )
}

// lavoisier.ladder.ablate: one row per removed contact.
function Ablate({ name, v }) {
  const rows = v.ablations || []
  const worst = rows.length ? Math.max(...rows.map((r) => r.loss)) : 1
  const graded = rows.some((r) => r.still_satisfied !== undefined)
  return (
    <div className="out-block">
      <div className="out-head">
        <b className="mono">{name}</b>
        <span className="note" style={{ margin: 0 }}>
          leave-one-out over {rows.length} contact(s), full composite {f5(v.full_composite)}
        </span>
      </div>
      <BarChart
        bars={rows.map((r) => ({
          label: r.dropped,
          value: r.loss,
          color: r.still_satisfied === false ? BADC : SERIES[0],
        }))}
        yDomain={[0, worst * 1.15]}
        yFmt={f5}
        horizontal
        h={40 + rows.length * 34}
      />
      <table>
        <thead>
          <tr>
            <th>removed</th>
            <th className="num">composite without it</th>
            <th className="num">loss</th>
            <th>requirement still met</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.dropped} className={r.still_satisfied === false ? 'hl' : ''}>
              <td className="mono">{r.dropped}</td>
              <td className="num">{f5(r.composite)}</td>
              <td className="num">{f5(r.loss)}</td>
              <td>
                {r.still_satisfied === undefined ? (
                  <span className="note" style={{ margin: 0 }}>no requirement stated</span>
                ) : (
                  <Pill verdict={r.still_satisfied ? 'PASS' : 'FAIL'} />
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="note">
        {graded
          ? 'A red bar is a contact the ladder cannot lose: removing it drops the composite below the stated requirement.'
          : 'This ladder states no requirement, so nothing here passes or fails --- the bars rank the contacts by what their removal costs, and no more than that.'}
      </p>
    </div>
  )
}

// lavoisier.ladder.minimum: the bound, and the curve it was read off.
function Minimum({ name, v }) {
  const p = v.pow_max
  const pts = []
  for (let n = 0; n <= Math.max(v.min_contacts + 3, 6); n++) {
    pts.push([n, 1 - Math.pow(1 - p, n)])
  }
  return (
    <div className="out-block">
      <div className="out-head">
        <b className="mono">{name}</b>
        <span className="note" style={{ margin: 0 }}>
          {v.min_contacts} identical contact(s) at power {v.pow_max}
        </span>
      </div>
      <LineChart
        series={[{ label: 'composite', points: pts, color: SERIES[2] }]}
        xDomain={[0, pts.length - 1]}
        yDomain={[0, 1]}
        xLabel="identical contacts"
        yLabel="composite resolution"
        xFmt={(t) => String(t)}
        yFmt={f3}
        h={260}
        rules={[{ at: v.target, color: BADC, label: 'target ' + v.target }]}
      />
      <p className="note">
        The bound is the first integer n whose composite clears the target. At
        n = {v.min_contacts} the composite is {f5(v.achieved_with_n)}. The curve
        approaches 1 and never reaches it, which is the floor stated as
        arithmetic rather than as an axiom.
      </p>
    </div>
  )
}

/* ------------------------------------------------- replayed values ---- */

// A recorded binding. Rendered as a key/value panel with the badge that
// says where the number came from.
function Recorded({ name, kind, v }) {
  const keys = Object.keys(v).filter((k) => !k.startsWith('_'))
  const scalars = keys.filter((k) => typeof v[k] !== 'object' || v[k] === null)
  const nested = keys.filter((k) => typeof v[k] === 'object' && v[k] !== null)

  return (
    <div className="out-block">
      <div className="out-head">
        <b className="mono">{name}</b>
        <span className="note" style={{ margin: 0 }}>{kind}</span>
        {v._replayed ? <Pill verdict="replayed" /> : <Pill verdict="computed" />}
      </div>
      {v._replayed ? (
        <div className="replayed">
          This value was <b>read from a recorded run</b>, not computed in your
          browser. The operation <span className="mono">{v._op}</span> needs the
          multi-megabyte spectral files, which are not shipped with this page.
          Change any argument it was recorded under and it will refuse rather
          than hand you a stale number.
        </div>
      ) : null}
      <table>
        <tbody>
          {scalars.map((k) => (
            <tr key={k}>
              <td className="mono">{k}</td>
              <td className="num">
                {typeof v[k] === 'number' ? String(v[k]) : String(v[k])}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {nested.map((k) => (
        <NestedTable key={k} label={k} v={v[k]} />
      ))}
    </div>
  )
}

function NestedTable({ label, v }) {
  if (Array.isArray(v)) {
    return (
      <p className="note">
        <b className="mono">{label}</b>{' '}
        <span className="mono">[{v.map((e) => String(e)).join(', ')}]</span>
      </p>
    )
  }
  const rows = Object.keys(v)
  const cols = rows.length && typeof v[rows[0]] === 'object' && v[rows[0]] !== null
    ? Object.keys(v[rows[0]])
    : null
  if (!cols) {
    return (
      <table>
        <thead>
          <tr>
            <th>{label}</th>
            <th className="num">value</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((k) => (
            <tr key={k}>
              <td className="mono">{k}</td>
              <td className="num">{String(v[k])}</td>
            </tr>
          ))}
        </tbody>
      </table>
    )
  }
  return (
    <table>
      <thead>
        <tr>
          <th>{label}</th>
          {cols.map((c) => (
            <th key={c} className="num">{c}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((k) => (
          <tr key={k}>
            <td className="mono">{k}</td>
            {cols.map((c) => (
              <td key={c} className="num">
                {typeof v[k][c] === 'number' ? v[k][c].toPrecision(6) : String(v[k][c])}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  )
}

/* ------------------------------------------------------- dispatch ---- */

function One({ b }) {
  const v = b.value
  if (v === null || typeof v !== 'object') {
    return (
      <div className="out-block">
        <div className="out-head">
          <b className="mono">{b.name}</b>
        </div>
        <p className="mono">{String(v)}</p>
      </div>
    )
  }
  if (b.kind === 'ladder' && !v._replayed) {
    if (Array.isArray(v.ablations)) return <Ablate name={b.name} v={v} />
    if (v.min_contacts !== undefined) return <Minimum name={b.name} v={v} />
    if (Array.isArray(v.rungs)) return <Resolve name={b.name} v={v} />
  }
  return <Recorded name={b.name} kind={b.kind} v={v} />
}

export default function Outputs({ payload }) {
  const ws = payload && payload.execute ? payload.execute.workspace : null
  if (!ws || !ws.length) return null
  return (
    <div className="outputs">
      {ws.map((b) => (
        <One key={b.name} b={b} />
      ))}
    </div>
  )
}
