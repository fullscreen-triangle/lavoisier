// =====================================================================
//  The landing page: what the seven papers share, and where they
//  genuinely disagree.
//
//  The vocabulary section is the point of this page. Seven papers using
//  the same word is not the same as seven papers meaning the same
//  thing, and three of these terms carry more than one sense. Saying so
//  is more useful than a glossary that pretends otherwise.
// =====================================================================

import { useState } from 'react'
import { Link } from 'react-router-dom'
import { PAPERS, BY_SLUG, ORDER_ARGUMENT, ORDER_CONVINCED } from '../papers.js'
import { EXPERIMENTS, corpusTotals } from '../lib/data.js'
import { Pill, Stat, Section, Slider } from '../components/Primitives.jsx'
import LineChart from '../components/charts/LineChart.jsx'
import GraphChart from '../components/charts/GraphChart.jsx'
import { SERIES, BADC } from '../components/charts/chart-kit.jsx'

/* ------------------------------------------------------- the floor -- */

// The min-contacts bound, recomputed live. This is the same arithmetic
// lavoisier.ladder.minimum performs --- shown here because the floor is
// the one idea every paper in the catalogue uses.
function FloorDemo() {
  const [power, setPower] = useState(0.6)
  const [target, setTarget] = useState(0.9)

  const pts = []
  for (let n = 0; n <= 12; n++) pts.push([n, 1 - Math.pow(1 - power, n)])
  const need = Math.ceil(Math.log(1 - target) / Math.log(1 - power))
  const got = 1 - Math.pow(1 - power, need)

  return (
    <div className="panel">
      <div className="grid g2">
        <div>
          <LineChart
            series={[{ label: 'composite', points: pts, color: SERIES[0] }]}
            xDomain={[0, 12]}
            yDomain={[0, 1]}
            xLabel="identical contacts"
            yLabel="composite resolution"
            xFmt={(t) => String(t)}
            yFmt={(t) => t.toFixed(2)}
            h={280}
            rules={[{ at: target, color: BADC, label: 'target' }]}
          />
        </div>
        <div>
          <Slider
            label="contact power"
            value={power}
            min={0.05}
            max={0.95}
            step={0.01}
            onChange={setPower}
            fmt={(v) => v.toFixed(2)}
          />
          <Slider
            label="target resolution"
            value={target}
            min={0.5}
            max={0.999}
            step={0.001}
            onChange={setTarget}
            fmt={(v) => v.toFixed(3)}
          />
          <div className="grid g2" style={{ marginTop: 14 }}>
            <Stat value={need} label="contacts needed" />
            <Stat value={got.toFixed(5)} label="composite reached" />
          </div>
          <p className="note">
            The curve approaches 1 and never arrives. Push the target to 0.999
            and the bound grows; push it to 1 and there is no bound at all,
            which is what the floor means stated as arithmetic. Six of the seven
            papers <i>prove</i> a floor exists and then only ever use the fact
            that it is positive. The seventh picks a value.
          </p>
        </div>
      </div>
    </div>
  )
}

/* ------------------------------------------------------ the medium -- */

const MEDIUM_NODES = [
  { id: 'm', label: 'm', medium: true },
  { id: 'a', label: 'a', side: 0, note: 'source side' },
  { id: 'b', label: 'b', side: 0, note: 'source side' },
  { id: 'c', label: 'c', side: 0, note: 'source side' },
  { id: 'd', label: 'd', side: 1, note: 'sink side' },
  { id: 'e', label: 'e', side: 1, note: 'sink side' },
  { id: 'f', label: 'f', side: 1, note: 'sink side' },
]
const MEDIUM_EDGES = [
  ['a', 'b'], ['b', 'c'], ['a', 'c'],
  ['d', 'e'], ['e', 'f'], ['d', 'f'],
  ['c', 'd'],
  ['m', 'a'], ['m', 'b'], ['m', 'c'], ['m', 'd'], ['m', 'e'], ['m', 'f'],
]

function MediumDemo() {
  const [removed, setRemoved] = useState(false)
  return (
    <div className="panel">
      <div className="grid g2">
        <GraphChart
          nodes={MEDIUM_NODES}
          edges={MEDIUM_EDGES}
          cut={removed ? [['c', 'd']] : []}
          mediumRemoved={removed}
          h={340}
        />
        <div>
          <div className="seg">
            <button className={removed ? '' : 'on'} onClick={() => setRemoved(false)}>
              medium present
            </button>
            <button className={removed ? 'on' : ''} onClick={() => setRemoved(true)}>
              medium deleted
            </button>
          </div>
          <p style={{ marginTop: 14 }}>
            With <span className="mono">m</span> present, every pair of items has
            a two-step path through it, so no cut of the graph is small and the
            structure of the two clusters is invisible. Delete it and the single
            bridge <span className="mono">c &mdash; d</span> is the cut.
          </p>
          <div className="callout warn">
            The medium&rsquo;s adjacency is <b>definitional, not measured</b>.
            It is adjacent to everything because that is what makes it a medium;
            no experiment established the edges. Three of the four papers using
            the term measure deletion driving separation to exactly{' '}
            <span className="mono">0.0</span>, which is a consequence of the
            definition and should not be read as an empirical finding.
          </div>
        </div>
      </div>
    </div>
  )
}

/* --------------------------------------------------------- the page -- */

const SENSES = [
  {
    term: 'contact',
    n: 6,
    warn: true,
    senses: [
      ['graph-theoretic', 'an edge, or the adjacency two items have in a contact graph'],
      ['physical', 'one ion&ndash;substrate interaction; the atomic unit of a process ladder'],
      ['evidential', 'κ(F) &isin; [0,1], an intensity-weighted fraction of the evidence a candidate explains'],
    ],
    note:
      'Senses one and three are not the same notion, and no result in the ' +
      'catalogue converts between them. The shared name is a genuine hazard.',
  },
  {
    term: 'ladder',
    n: 2,
    warn: true,
    senses: [
      ['instrument', 'an ordered sequence of contacts, each with a resolving power'],
      ['fragment', 'a series of fragment masses differing by one residue'],
    ],
    note:
      'A real family resemblance --- both are ordered sequences whose members ' +
      'each contribute --- but no identity. Nothing proved of one transfers.',
  },
  {
    term: 'cut key',
    n: 2,
    warn: true,
    senses: [['κ = (σ, δ)', 'the pair of separation and depth that identifies a cut']],
    note:
      'Symbol clash: the CASMI paper writes κ for contact, in the evidential ' +
      'sense above. Same letter, unrelated quantity.',
  },
]

const AGREED = [
  {
    term: 'separation σ',
    n: 4,
    body:
      'The value of a minimum cut between two items. One definition, one ' +
      'algorithm (single max-flow, O(|V||E|)), no variation across the four ' +
      'graph papers.',
  },
  {
    term: 'depth δ',
    n: 4,
    body:
      'How far into the structure the minimising cut sits. Paired with σ to ' +
      'form the cut key.',
  },
  {
    term: 'invariance',
    n: 5,
    body:
      'Unchanged under relabelling or under a change of realisation. Stated ' +
      'identically in all five, and tested by relabelling controls in three.',
  },
  {
    term: 'non-completability',
    n: 6,
    body:
      'No finite procedure resolves an item completely. Stated near-verbatim ' +
      'in six papers. The instrument ladder uses the floor without needing ' +
      'the axiom; the CASMI study does not state it at all.',
  },
  {
    term: 'declining',
    n: 5,
    body:
      'Refusing to answer is a licensed outcome, not a failure. The CASMI ' +
      'study pays the price in public: 5 of 58 challenges licensed, 53 ' +
      'declined.',
  },
]

export default function Catalogue() {
  const [order, setOrder] = useState('argument')
  const t = corpusTotals()
  const list = (order === 'argument' ? ORDER_ARGUMENT : ORDER_CONVINCED).map((s) => BY_SLUG[s])

  return (
    <>
      <header className="hero">
        <div className="wrap">
          <div className="kicker">the catalogue</div>
          <h1>Seven papers about what a measurement leaves out</h1>
          <p className="lede">
            Each paper defines a floor below which an item cannot be resolved,
            and then asks what follows. This site is the arguments, the
            experiments that graded them, and a runtime that recomputes the
            ladder results in your browser from source you can edit.
          </p>
          <p className="pull">
            A number here is either computed in front of you or labelled with
            where it was read from. There is no third category.
          </p>
          <div className="badges">
            <span className="badge">
              <b>{PAPERS.length}</b> papers
            </span>
            <span className="badge">
              <b>{t.graded}</b> graded predictions
            </span>
            <span className="badge">
              <b>{t.passed}</b> reproduced
            </span>
            <span className="badge">
              <b>{t.failed}</b> refuted
            </span>
          </div>
        </div>
      </header>

      <Section
        id="vocabulary"
        kicker="shared vocabulary"
        title="The words, and how far they agree"
        sub={
          'These terms recur across the seven papers. Most are used identically. ' +
          'Three are not, and the differences matter more than the agreements.'
        }
      >
        <h3>floor (β) &mdash; all seven papers</h3>
        <p>
          The quantity that cannot be driven to zero. Every paper here needs
          one, and six of the seven prove that one exists without ever
          choosing a value &mdash; every result they derive uses only{' '}
          <span className="mono">β &gt; 0</span>. The CASMI study is the
          exception: it picks <span className="mono">β = 0.30</span>{' '}
          empirically, sweeps it, and reports that the chosen value sits at
          noise level. That is a different kind of claim and the page says so.
        </p>
        <FloorDemo />

        <h3>medium (𝔪) &mdash; four papers</h3>
        <p>
          A distinguished vertex adjacent to every item. Defined identically
          in observation groups, runtime graph, sink detection and peptide
          mass invariance.
        </p>
        <MediumDemo />

        <h3>terms that carry more than one sense</h3>
        <div className="grid g3">
          {SENSES.map((s) => (
            <div className="panel" key={s.term}>
              <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                <b className="mono">{s.term}</b>
                <span className="badge">
                  <b>{s.n}</b> papers
                </span>
              </div>
              <ul style={{ fontSize: 13.4, color: 'var(--ink-dim)', paddingLeft: 18 }}>
                {s.senses.map((x, i) => (
                  <li key={i} style={{ marginBottom: 6 }}>
                    <b style={{ color: 'var(--ink)' }}>{x[0]}</b> &mdash;{' '}
                    <span dangerouslySetInnerHTML={{ __html: x[1] }} />
                  </li>
                ))}
              </ul>
              <div className="note">{s.note}</div>
            </div>
          ))}
        </div>

        <h3>terms used identically</h3>
        <div className="grid g2">
          {AGREED.map((a) => (
            <div className="panel" key={a.term}>
              <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                <b className="mono">{a.term}</b>
                <span className="badge">
                  <b>{a.n}</b> papers
                </span>
              </div>
              <p style={{ fontSize: 13.6, marginTop: 8 }}>{a.body}</p>
            </div>
          ))}
        </div>

        <div className="callout">
          Two words readers expect and will not find: <b>category</b> has no
          technical sense anywhere in these seven, and <b>S-entropy</b> does
          not appear in them at all. Neither has an entry here because
          inventing one would be inventing agreement.
        </div>
      </Section>

      <Section
        id="papers"
        kicker="the seven"
        title="Where to start"
        sub={
          'Two orders, for two readers. The first builds the argument from its ' +
          'foundation; the second starts with the applied study and works back ' +
          'to why it is set up that way.'
        }
      >
        <div className="seg" style={{ marginBottom: 18 }}>
          <button
            className={order === 'argument' ? 'on' : ''}
            onClick={() => setOrder('argument')}
          >
            argument first
          </button>
          <button
            className={order === 'convinced' ? 'on' : ''}
            onClick={() => setOrder('convinced')}
          >
            convince me first
          </button>
        </div>

        <div className="paper-grid">
          {list.map((p, i) => {
            const s = p.exp ? EXPERIMENTS[p.exp].summary : null
            return (
              <Link className="paper-card" to={'/paper/' + p.slug} key={p.slug}>
                <div className="kicker">
                  {i + 1} &middot; {p.role}
                </div>
                <h3>{p.title}</h3>
                <p>{p.blurb}</p>
                <div className="row">
                  {s ? (
                    <>
                      <Pill verdict={s.verdict} />
                      <span className="badge">
                        <b>{s.passed}</b>/<b>{s.graded}</b>
                      </span>
                    </>
                  ) : (
                    <span className="badge">applied study</span>
                  )}
                </div>
              </Link>
            )
          })}
        </div>
      </Section>

      <Section
        id="validation"
        kicker="corpus-wide"
        title="What was graded, and what failed"
        sub={
          'Every prediction across the six experiment files, counted from those ' +
          'files rather than typed in here. Two experiments carry failures, and ' +
          'those are the interesting ones.'
        }
      >
        <div className="grid g4">
          <div className="panel stat">
            <div className="v">{t.graded}</div>
            <div className="k">predictions graded</div>
          </div>
          <div className="panel stat">
            <div className="v" style={{ color: 'var(--ok)' }}>
              {t.passed}
            </div>
            <div className="k">reproduced</div>
          </div>
          <div className="panel stat">
            <div className="v" style={{ color: 'var(--bad)' }}>
              {t.failed}
            </div>
            <div className="k">refuted</div>
          </div>
          <div className="panel stat">
            <div className="v">{t.nonDiscriminating}</div>
            <div className="k">non-discriminating</div>
          </div>
        </div>

        <table style={{ marginTop: 20 }}>
          <thead>
            <tr>
              <th>experiment</th>
              <th>paper</th>
              <th className="num">graded</th>
              <th className="num">passed</th>
              <th className="num">failed</th>
              <th>verdict</th>
            </tr>
          </thead>
          <tbody>
            {Object.keys(EXPERIMENTS).map((k) => {
              const e = EXPERIMENTS[k]
              return (
                <tr key={k} className={e.summary.failed ? 'hl' : ''}>
                  <td className="mono">{e.experiment}</td>
                  <td>
                    <Link to={'/paper/' + e.paper}>{e.paper}</Link>
                  </td>
                  <td className="num">{e.summary.graded}</td>
                  <td className="num">{e.summary.passed}</td>
                  <td className="num">{e.summary.failed}</td>
                  <td>
                    <Pill verdict={e.summary.verdict} />
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>

        <p className="note">
          A refuted prediction is not a bug that was fixed. Both failing
          experiments are reported as refutations in their own papers, and the
          paper pages state which theorem statements are defective as a
          result.
        </p>
      </Section>
    </>
  )
}
