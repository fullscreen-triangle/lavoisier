// =====================================================================
//  Observation groups.
//
//  The machinery from the peptide paper, turned on replicate structure.
//  The result that carries the page is thm:endpoints: the refinement
//  lattice on six observations has 203 elements, and you never have to
//  visit them --- checking the two ends of an interval decides the whole
//  interval. The numbers here come from exp2's records, not from a
//  sweeps file, so the charts are built from what the register recorded.
// =====================================================================

import { useState } from 'react'
import { PaperHead, ClaimsTable, ExperimentNotes, Environment } from '../../components/PaperLayout.jsx'
import { Section, Callout, Stat } from '../../components/Primitives.jsx'
import LineChart from '../../components/charts/LineChart.jsx'
import BarChart from '../../components/charts/BarChart.jsx'
import LatticeChart from '../../components/charts/LatticeChart.jsx'
import { SERIES, OKC, BADC } from '../../components/charts/chart-kit.jsx'
import { BY_SLUG } from '../../papers.js'
import { EXPERIMENTS } from '../../lib/data.js'

const PAPER = BY_SLUG['observation-groups']
const R = EXPERIMENTS.exp2.records

/* --------------------------------------------------- the lattice -- */

// The real refinement lattice --- but on FOUR observations, where it
// has 15 elements and can honestly be drawn. Six observations give 203,
// which is the number in the prose and is not a drawable chart. The
// structure is the same; only the size differs, and thm:endpoints is
// about not caring how big it gets.
function partitionsOf(items) {
  if (items.length === 0) return [[]]
  const [first, ...rest] = items
  const out = []
  for (const p of partitionsOf(rest)) {
    for (let i = 0; i < p.length; i++) {
      const q = p.map((b, j) => (j === i ? [first, ...b] : b))
      out.push(q)
    }
    out.push([[first], ...p])
  }
  return out
}

const ITEMS = ['a', 'b', 'c', 'd']
const PARTS = partitionsOf(ITEMS).map((p) => {
  const blocks = p.map((b) => b.slice().sort().join('')).sort()
  return { id: blocks.join('|'), label: blocks.join('·'), rank: blocks.length, blocks }
})

// One partition covers another when merging exactly two of its blocks
// gives the other --- so the coarser has one fewer block and every one
// of its blocks is a union of the finer's.
function covers(fine, coarse) {
  if (coarse.rank !== fine.rank - 1) return false
  return coarse.blocks.every((cb) =>
    fine.blocks
      .filter((fb) => fb.split('').every((ch) => cb.indexOf(ch) !== -1))
      .join('')
      .split('')
      .sort()
      .join('') === cb.split('').sort().join('')
  )
}

const LEVELS = [4, 3, 2, 1].map((rank) => ({
  rank,
  nodes: PARTS.filter((p) => p.rank === rank).map((p) => ({
    id: p.id,
    label: p.label,
    note: p.rank + ' block(s)',
  })),
}))

const EDGES = []
for (const f of PARTS)
  for (const c of PARTS) if (covers(f, c)) EDGES.push([f.id, c.id])

const TOP = PARTS.find((p) => p.rank === 4).id
const BOT = PARTS.find((p) => p.rank === 1).id

function LatticeDemo() {
  return (
    <div className="panel">
      <div className="grid g2">
        <LatticeChart
          levels={LEVELS}
          edges={EDGES}
          highlight={[TOP, BOT]}
          rankLabel="blocks"
          h={380}
        />
        <div>
          <div className="grid g2">
            <Stat value="203" label="groupings of 6 observations" />
            <Stat value="2" label="that must be evaluated" color="var(--ok)" />
          </div>
          <p>
            Every way of grouping replicates is one element of a lattice ordered
            by refinement. Drawn here for four observations, where it has{' '}
            {PARTS.length} elements and {EDGES.length} covering relations. On the
            six observations the experiment actually used there are <b>203</b>
            &nbsp;&mdash; the sixth Bell number &mdash; and a verdict that
            depended on which one you picked would be indefensible, because
            nobody can justify a choice among 203.
          </p>
          <p>
            <span className="mono">thm:endpoints</span> is what makes the size
            irrelevant. A verdict is monotone along refinement, so evaluating the
            two ends of an interval decides every element between them. The
            experiment checked {R.endpoint_decidability.intervals_checked}{' '}
            intervals both ways &mdash; two endpoints against exhaustive
            evaluation &mdash; and recorded{' '}
            <b>{R.endpoint_decidability.disagreements} disagreements</b>.
          </p>
          <p className="note">
            The highlighted nodes are the two ends: all singletons at the top,
            one block at the bottom. Everything between them is what you do not
            have to evaluate.
          </p>
        </div>
      </div>
    </div>
  )
}

/* ------------------------------------------- the verdict flip -- */

function VerdictDemo() {
  const v = R.verdict_dependence
  const bars = [
    { label: 'P₁ (finer)', value: v.statistic_at_P1, color: v.verdict_at_P1 ? OKC : BADC },
    { label: 'P₂ (coarser)', value: v.statistic_at_P2, color: v.verdict_at_P2 ? OKC : BADC },
  ]
  return (
    <div className="panel">
      <div className="grid g2">
        <BarChart
          bars={bars}
          yDomain={[0, v.statistic_at_P1 * 1.25]}
          yLabel="test statistic"
          yFmt={(t) => t.toFixed(2)}
          rules={[{ at: v.threshold, color: 'var(--warn)', label: 'threshold' }]}
          h={280}
        />
        <div>
          <p>
            The same data <span className="mono">x</span>, unchanged, evaluated
            under two groupings one of which strictly refines the other. The
            statistic reads {v.statistic_at_P1.toFixed(4)} under the finer and{' '}
            {v.statistic_at_P2.toFixed(4)} under the coarser, and the threshold
            of {v.threshold.toFixed(4)} sits <i>between them</i>. The verdict
            flips from significant to not.
          </p>
          <Callout tone="warn">
            Nothing about the measurement changed. The only thing that changed is
            a choice made after the data were collected, about which replicates
            count as the same observation. That is the problem the rest of the
            paper exists to solve, and it is a live problem rather than a
            hypothetical one.
          </Callout>
        </div>
      </div>
    </div>
  )
}

/* -------------------------------------------- the grouping floor -- */

function FloorDemo() {
  const rows = R.no_free_grouping
  const beta = R.group_floor.medium_weights.r3
  return (
    <div className="panel">
      <div className="grid g2">
        <LineChart
          series={[
            {
              label: 'information discarded (lower bound)',
              points: rows.map((r) => [r.dof, r.lower_bound_discarded]),
              color: SERIES[0],
            },
          ]}
          xDomain={[0, Math.max(...rows.map((r) => r.dof))]}
          yDomain={[0, Math.max(...rows.map((r) => r.lower_bound_discarded)) * 1.15]}
          xLabel="degrees of freedom given up"
          yLabel="lower bound on what is discarded"
          xFmt={(t) => String(t)}
          yFmt={(t) => t.toFixed(2)}
          h={290}
        />
        <div>
          <div className="grid g2">
            <Stat value={beta.toFixed(4)} label="floor β per degree of freedom" />
            <Stat
              value={R.medium_control.cut_of_full_set_without_medium.toFixed(1)}
              label="cut without the medium"
            />
          </div>
          <p>
            Pooling replicates is not free. Every degree of freedom surrendered
            discards at least β of the distinguishing information, and β here is
            the smallest medium weight in the graph &mdash; a positive number,
            which is the whole content of the claim. The line is straight because
            the bound is exactly linear in the degrees of freedom given up.
          </p>
          <p className="note">
            The second figure is the control, and it is zero by definition rather
            than by measurement: delete the medium and the observations are
            pairwise disconnected, so the cut costs nothing. It confirms the
            implementation is measuring what it claims to and nothing more.
          </p>
        </div>
      </div>
    </div>
  )
}

/* -------------------------------------------------------- the page -- */

export default function ObservationGroups() {
  const [showAll, setShowAll] = useState(false)
  const deg = R.degenerate_interval.statistics
  const shown = showAll ? deg : deg.slice(0, 3)

  return (
    <>
      <PaperHead paper={PAPER} />

      <Section
        id="argument"
        kicker="the claim"
        title="Which replicates count as one observation, and who decides"
        sub={
          'Grouping replicates is a choice made after the data are in, and it ' +
          'can change the verdict. This paper makes the choice into an object ' +
          'with a lattice structure, and then shows you only ever need its ends.'
        }
      >
        <h3>The same data, two verdicts</h3>
        <VerdictDemo />

        <h3>203 groupings, two evaluations</h3>
        <LatticeDemo />

        <h3>Pooling costs something, and the cost has a floor</h3>
        <FloorDemo />

        <h3>When the interval does not decide</h3>
        <p>
          The honest half of the endpoint result. An interval whose two ends
          agree is settled; an interval whose ends disagree is not, and the paper
          reports rather than resolves it. Here is one such interval &mdash;{' '}
          {R.degenerate_interval.interval_size} groupings whose statistics spread
          across a wide range:
        </p>
        <div className="panel">
          <table>
            <thead>
              <tr>
                <th>grouping</th>
                <th className="num">statistic</th>
              </tr>
            </thead>
            <tbody>
              {shown.map((s, i) => (
                <tr key={i}>
                  <td className="mono" style={{ fontSize: 12.4 }}>
                    {s.grouping}
                  </td>
                  <td className="num">{s.statistic.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {deg.length > 3 ? (
            <button
              className="ghost"
              style={{ marginTop: 12 }}
              onClick={() => setShowAll(!showAll)}
            >
              {showAll ? 'show fewer' : 'show all ' + deg.length}
            </button>
          ) : null}
          <p className="note">
            The correct report for this interval is not a number. It is the
            interval itself, plus the statement that its ends disagree &mdash;
            which is a decline, and a decline is a finding.
          </p>
        </div>
      </Section>

      <Section
        id="claims"
        kicker="registered predictions"
        title="What was predicted, and what happened"
        sub="Eleven registered, eleven reproduced."
      >
        <ClaimsTable expKey="exp2" />
        <Environment expKey="exp2" />
      </Section>

      <Section id="limits" kicker="limitations" title="Recorded by the experiment">
        <ExperimentNotes expKey="exp2" />
        <p className="note">
          A build note that belongs with the paper rather than the result: the
          manuscript&rsquo;s validation section writes{' '}
          <span className="mono">\Lat(\Obs)</span> while its preamble defines{' '}
          <span className="mono">L(\Obs)</span>. As it stands the file will not
          compile. Nothing about the experiment depends on it.
        </p>
      </Section>
    </>
  )
}
