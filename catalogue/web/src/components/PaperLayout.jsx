// =====================================================================
//  The shape every paper page shares.
//
//  Header, body, claims table, notebook, limitations. The order is
//  fixed on purpose: the claims table comes BEFORE the limitations and
//  after the argument, so a reader who scrolls meets what was predicted,
//  then what was observed, then what is wrong with it --- in that order,
//  without having to look for the last part.
// =====================================================================

import { Link } from 'react-router-dom'
import { Pill } from './Primitives.jsx'
import { EXPERIMENTS } from '../lib/data.js'

export function PaperHead({ paper }) {
  const exp = paper.exp ? EXPERIMENTS[paper.exp] : null
  const s = exp ? exp.summary : null
  return (
    <header className="paper-head">
      <div className="wrap">
        <div className="crumbs">
          <Link to="/">catalogue</Link> / <span className="mono">{paper.slug}</span>
        </div>
        <h1>{paper.title}</h1>
        <p className="claim">{paper.claim}</p>
        <div className="badges" style={{ marginTop: 16 }}>
          {s ? (
            <>
              <Pill verdict={s.verdict} />
              <span className="badge">
                <b>{s.passed}</b> of <b>{s.graded}</b> predictions reproduced
              </span>
              {s.failed ? (
                <span className="badge">
                  <b>{s.failed}</b> refuted
                </span>
              ) : null}
              {s.non_discriminating ? (
                <span className="badge">
                  <b>{s.non_discriminating}</b> non-discriminating
                </span>
              ) : null}
            </>
          ) : (
            <span className="badge">applied study, no expectation register</span>
          )}
          <span className="badge">{paper.role}</span>
        </div>
        {exp ? <p className="note">{exp.question}</p> : null}
      </div>
    </header>
  )
}

// The claims table. Every row is one registered prediction: what was
// expected, which theorem it came from, what would have counted as a
// failure, and what actually happened. A page without this is a
// brochure.
export function ClaimsTable({ expKey }) {
  const exp = EXPERIMENTS[expKey]
  if (!exp) return null
  return (
    <div className="panel">
      <table>
        <thead>
          <tr>
            <th>claim</th>
            <th className="hide-sm">stated in</th>
            <th>observed</th>
            <th>verdict</th>
          </tr>
        </thead>
        <tbody>
          {exp.expectations.map((e, i) => (
            <tr key={i} className={e.passed ? '' : 'hl'}>
              <td>
                <b>{e.label}</b>
                <div className="note" style={{ margin: '4px 0 0' }}>
                  {e.prediction}
                </div>
                <div className="note" style={{ margin: '4px 0 0' }}>
                  <b>counts as failure:</b> {e.failure_mode}
                </div>
              </td>
              <td className="mono hide-sm" style={{ fontSize: 12 }}>
                {e.paper_ref}
              </td>
              <td style={{ fontSize: 12.8, color: 'var(--ink-dim)' }}>
                {e.detail}
                {e.discriminating === false ? (
                  <div className="note" style={{ margin: '4px 0 0' }}>
                    control did not discriminate
                  </div>
                ) : null}
              </td>
              <td>
                <Pill verdict={e.passed ? 'PASS' : 'FAIL'} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// The notes the experiment itself recorded. These are where the suite
// wrote down what it found wrong with the paper, so they are rendered
// verbatim rather than paraphrased.
export function ExperimentNotes({ expKey }) {
  const exp = EXPERIMENTS[expKey]
  if (!exp || !exp.notes || !exp.notes.length) return null
  return (
    <>
      {exp.notes.map((n, i) => (
        <div className="callout warn" key={i}>
          {n}
        </div>
      ))}
    </>
  )
}

export function Environment({ expKey }) {
  const exp = EXPERIMENTS[expKey]
  if (!exp) return null
  const e = exp.environment
  return (
    <p className="note">
      Recorded on Python {e.python}, {e.platform}, in {e.elapsed_s}s. The numbers
      in the table above are read from{' '}
      <span className="mono">{exp.experiment}</span>&rsquo;s result file, not
      retyped.
    </p>
  )
}
