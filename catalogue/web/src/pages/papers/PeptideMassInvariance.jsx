// =====================================================================
//  Peptide mass invariance --- the foundational paper.
//
//  Everything the other graph papers use is defined here: the medium,
//  separation, depth, the cut key. The page leads with the invariance
//  result and then spends most of its length on prop:three, which is
//  the one graded row that fails --- and fails on a definite article.
//  Both readings of the proposition are charted side by side, because
//  the difference between them is the whole defect.
// =====================================================================

import { PaperHead, ClaimsTable, ExperimentNotes, Environment } from '../../components/PaperLayout.jsx'
import { Section, Callout, Stat } from '../../components/Primitives.jsx'
import LineChart from '../../components/charts/LineChart.jsx'
import BarChart from '../../components/charts/BarChart.jsx'
import { SERIES, OKC, BADC } from '../../components/charts/chart-kit.jsx'
import { BY_SLUG } from '../../papers.js'
import SWEEPS from '../../data/exp6_sweeps.json'

const PAPER = BY_SLUG['peptide-mass-invariance']

/* ------------------------------------------------- key invariance -- */

function InvarianceDemo() {
  const rows = SWEEPS.invariance.rows
  const relab = rows.reduce((a, r) => a + r.relabellings, 0)
  const miss = rows.reduce((a, r) => a + r.mismatches, 0)
  const moved = rows.reduce((a, r) => a + r.control_moved, 0)

  return (
    <div className="panel">
      <div className="grid g2">
        <LineChart
          series={[
            {
              label: 'distinct keys per graph',
              points: rows.map((r) => [r.n, r.mean_distinct_keys]),
              color: SERIES[0],
            },
          ]}
          xDomain={[Math.min(...rows.map((r) => r.n)), Math.max(...rows.map((r) => r.n))]}
          yDomain={[0, Math.max(...rows.map((r) => r.mean_distinct_keys)) * 1.2]}
          xLabel="vertices"
          yLabel="mean distinct cut keys"
          xFmt={(v) => String(v)}
          yFmt={(v) => v.toFixed(1)}
          h={290}
        />
        <div>
          <div className="grid g3">
            <Stat value={relab} label="relabellings applied" />
            <Stat value={miss} label="keys moved" color="var(--ok)" />
            <Stat value={moved} label="control: weight changed" color="var(--warn)" />
          </div>
          <p>
            A relabelling renames the vertices and changes nothing else. The cut
            key <span className="mono">κ = (σ, δ)</span> is computed from the
            graph, so it should not notice &mdash; and across every relabelling
            in the sweep it did not.
          </p>
          <p className="note">
            The third figure is what makes the second meaningful. A test that
            cannot fail proves nothing, so the same machinery was run with the
            edge weights perturbed instead of the labels: there the key moved
            every time. The invariance is to relabelling specifically, not to
            everything.
          </p>
          <p className="note">
            The rising curve is a separate fact worth having: larger graphs carry
            more distinct keys, so the key is not collapsing everything onto one
            value as the graphs grow.
          </p>
        </div>
      </div>
    </div>
  )
}

/* --------------------------------------------- mass does not decide -- */

function MassDemo() {
  const per = SWEEPS.mass.per_n
  const tol = SWEEPS.mass.tolerance

  return (
    <div className="panel">
      <div className="grid g2">
        <BarChart
          bars={per.map((r) => ({
            label: 'n=' + r.n,
            value: r.rate,
            color: SERIES[0],
          }))}
          yDomain={[0, Math.max(...per.map((r) => r.rate)) * 1.3]}
          yLabel="pairs sharing a cut key"
          yFmt={(v) => (v * 100).toFixed(1) + '%'}
          h={270}
        />
        <LineChart
          series={[
            {
              label: 'mass-ambiguous compounds',
              points: tol.map((r) => [r.tol_ppm, r.fraction]),
              color: SERIES[1],
            },
          ]}
          xDomain={[
            Math.min(...tol.map((r) => r.tol_ppm)),
            Math.max(...tol.map((r) => r.tol_ppm)),
          ]}
          yDomain={[0, 1]}
          logX
          xLabel="mass tolerance (ppm)"
          yLabel="fraction ambiguous"
          xFmt={(v) => (v >= 1 ? String(v) : v.toExponential(0))}
          yFmt={(v) => v.toFixed(1)}
          h={270}
        />
      </div>
      <p className="note">
        Left: two graphs can share a cut key while differing in mass, and the
        rate does not fall away with size &mdash; so no amount of mass precision
        recovers the key. Right: the converse direction on this particular
        library. Tightening the tolerance <i>does</i> resolve mass ambiguity
        here, reaching zero ambiguous compounds by 10&nbsp;ppm &mdash; but the
        library is a curated set of 244 compounds, and the paper says explicitly
        that this direction may not survive on an uncurated one. The claim being
        made is the left panel; the right is reported so a reader is not left
        with a stronger impression than the data supports.
      </p>
    </div>
  )
}

/* ---------------------------------------------------- the medium -- */

function NoSelectorDemo() {
  const s = SWEEPS.no_selector.surface
  const mw = SWEEPS.no_selector.grid_mw
  const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length

  return (
    <div className="panel">
      <LineChart
        series={[
          {
            label: 'medium present',
            points: s.map((r) => [r.n, mean(r.with_medium)]),
            color: SERIES[0],
          },
          {
            label: 'medium deleted',
            points: s.map((r) => [r.n, mean(r.without)]),
            color: BADC,
          },
        ]}
        xDomain={[Math.min(...s.map((r) => r.n)), Math.max(...s.map((r) => r.n))]}
        yDomain={[0, Math.max(...s.map((r) => mean(r.with_medium))) * 1.2]}
        xLabel="vertices"
        yLabel="mean separation"
        xFmt={(v) => String(v)}
        yFmt={(v) => v.toFixed(1)}
        h={290}
      />
      <p className="note">
        Averaged over {mw.length} medium weights at each size. The lower line is
        exactly zero in all 30 cells, and that is arithmetic rather than
        evidence: the medium is <i>defined</i> to be adjacent to every item, so
        deleting it leaves the items pairwise disconnected and separation has
        nothing to measure. What the result establishes is the negative half of{' '}
        <span className="mono">thm:no-selector</span> &mdash; without the medium
        there is no non-arbitrary way to single out a rival, because every item
        looks like every other one.
      </p>
    </div>
  )
}

/* --------------------------------------- prop:three, both readings -- */

// The single failing row in the register. The proposition says the
// region is stable under removal of THE erroneous catalyst; if the
// erroneous one can be identified, two suffice, and the "at least
// three" claim is false as printed. Charting both readings is the
// cleanest way to show the definite article doing the work.
function ThreeDemo() {
  const row = SWEEPS.three.rows[0]
  const lit = row.literal
  const un = row.unaided

  return (
    <div className="panel">
      <LineChart
        series={[
          {
            label: 'literal reading --- the erroneous catalyst is identified',
            points: lit.map((p) => [p.k, p.fraction]),
            color: BADC,
          },
          {
            label: 'unaided --- it is not identified',
            points: un.map((p) => [p.k, p.fraction]),
            color: OKC,
          },
        ]}
        xDomain={[1, Math.max(...lit.map((p) => p.k))]}
        yDomain={[0, 1.05]}
        xLabel="catalysts k"
        yLabel="fraction recovering the true region"
        xFmt={(v) => String(v)}
        yFmt={(v) => v.toFixed(1)}
        step
        h={300}
      />
      <div className="grid g2" style={{ marginTop: 14 }}>
        <Stat value="k = 2" label="literal reading suffices at" color="var(--bad)" />
        <Stat value="k = 3" label="unaided reading suffices at" color="var(--ok)" />
      </div>
      <p className="note">
        Two curves, one proposition. Read as printed &mdash; remove{' '}
        <b>the</b> erroneous catalyst, meaning it has been identified &mdash; the
        red curve is already at 1 with two catalysts, so &ldquo;at least
        three&rdquo; is false. Read as the proof actually proceeds &mdash; the
        erroneous one is not known, so the region is the majority over all k
        &mdash; the green curve needs three, and the proposition is right. The
        difference between the two is a definite article.
      </p>
    </div>
  )
}

/* -------------------------------------------------------- the page -- */

export default function PeptideMassInvariance() {
  return (
    <>
      <PaperHead paper={PAPER} />

      <Section
        id="argument"
        kicker="the claim"
        title="Identity is an edge set, not a number"
        sub={
          'This is the paper the other graph papers stand on. It defines the ' +
          'medium, separation, depth and the cut key, and establishes that no ' +
          'scalar observable --- mass at any precision included --- determines ' +
          'the key.'
        }
      >
        <p>
          The claim region of a graph is{' '}
          <span className="mono">R(G) = Σ(σ(v&#7522;) &minus; β)</span>: the
          separations that clear the floor. Identity, in this framework, is that
          set of edges rather than any value computed from it &mdash; and the
          consequence is that refining a scalar measurement does not converge on
          identity, because the two are not the same kind of object.
        </p>

        <h3>The key survives relabelling</h3>
        <InvarianceDemo />

        <h3>Mass does not determine the key</h3>
        <MassDemo />

        <h3>Without the medium there is no rival to point at</h3>
        <NoSelectorDemo />
      </Section>

      <Section
        id="defect"
        kicker="what broke"
        title="prop:three, and the word that breaks it"
        sub={
          'Eight of the nine graded rows reproduced. The ninth is a statement ' +
          'defect rather than a wrong result, and it is worth the space because ' +
          'the fix is one word.'
        }
      >
        <ThreeDemo />
        <Callout tone="warn">
          The proposition says a region supported by k catalysts, one of which
          may be in error, is stable under removal of{' '}
          <b>the</b> erroneous catalyst. The definite article presupposes the
          erroneous one has been identified &mdash; and if it has, two catalysts
          are enough, so &ldquo;at least three&rdquo; is false as printed. The
          proof does not assume identification, and under that reading three is
          exactly right. Corrected statement:{' '}
          <b>&ldquo;stable under removal of <i>an</i> erroneous catalyst.&rdquo;</b>{' '}
          The proposition graded twice for the same reason coordinate
          provenance&rsquo;s <span className="mono">thm:minimal-record</span>{' '}
          does &mdash; the printed statement and the established statement are
          not the same statement.
        </Callout>
      </Section>

      <Section
        id="claims"
        kicker="registered predictions"
        title="What was predicted, and what happened"
        sub="Nine rows, eight reproduced, one refuted. The refutation is the definite article above."
      >
        <ClaimsTable expKey="exp6" />
        <Environment expKey="exp6" />
      </Section>

      <Section id="limits" kicker="limitations" title="Recorded by the experiment">
        <ExperimentNotes expKey="exp6" />
        <Callout tone="warn">
          <b>Two claims from the manuscript are refuted outright and reported as
          such.</b> <span className="mono">ass:structure</span> asserts a
          collapse direction the measurements contradict, and the promiscuity
          claim fails in both of its tests: the ratio came out 2.56 &mdash; the
          opposite sign to the one predicted &mdash; and the win-or-tie rate was
          0.074 against a required majority. Neither is softened in the paper and
          neither is softened here.
        </Callout>
        <p className="note">
          Vocabulary warning carried from the catalogue page: the
          &ldquo;ladder&rdquo; in this paper is a series of fragment masses
          differing by one residue. The instrument paper&rsquo;s ladder is a
          sequence of instrument contacts. The two share a shape and nothing
          else, and no result transfers between them.
        </p>
      </Section>
    </>
  )
}
