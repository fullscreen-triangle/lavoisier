// =====================================================================
//  Sink detection.
//
//  The page with the most to be honest about. Every graded cell hit its
//  bound to full precision, and three of the paper's own stated results
//  are defective --- one whose proof overreaches, one whose lower bound
//  is simply false, and one whose antecedent nothing satisfies. The
//  experiment found all three and the paper reports all three.
// =====================================================================

import { Link } from 'react-router-dom'
import { PaperHead, ClaimsTable, ExperimentNotes, Environment } from '../../components/PaperLayout.jsx'
import { Section, Callout, Stat } from '../../components/Primitives.jsx'
import LineChart from '../../components/charts/LineChart.jsx'
import ScatterChart from '../../components/charts/ScatterChart.jsx'
import BarChart from '../../components/charts/BarChart.jsx'
import { SERIES, OKC, BADC } from '../../components/charts/chart-kit.jsx'
import { BY_SLUG } from '../../papers.js'
import SWEEPS from '../../data/exp5_sweeps.json'
import { EXPERIMENTS } from '../../lib/data.js'

const PAPER = BY_SLUG['sink-detection']
const REC = EXPERIMENTS.exp5.records
const SA = REC.spread_sound_a
const CO = REC.collapse
const AM = REC.amplify
const AC = REC.amplify_control

/* --------------------------------------------------- the collapse -- */

// thm:collapse says separation sits inside a two-sided band whose width
// depends on lambda. The sweep walks lambda and reports breaches; the
// chart shows the band closing.
function CollapseDemo() {
  const rows = SWEEPS.collapse.rows
  const width = rows.map((r) => {
    const w = r.sep.map((s, i) => Math.abs(r.upper[i] - s))
    return [r.lam, Math.max(...w)]
  })
  const breaches = rows.reduce((a, r) => a + r.breaches, 0)

  return (
    <div className="panel">
      <div className="grid g2">
        <LineChart
          series={[{ label: 'worst gap to the bound', points: width, color: SERIES[0] }]}
          xDomain={[Math.min(...rows.map((r) => r.lam)), Math.max(...rows.map((r) => r.lam))]}
          yDomain={[1e-6, 1]}
          logY
          xLabel="λ"
          yLabel="|separation − bound|"
          xFmt={(v) => String(v)}
          yFmt={(v) => v.toExponential(0)}
          h={290}
        />
        <div>
          <div className="grid g2">
            <Stat value={rows.length} label="λ values swept" />
            <Stat
              value={breaches}
              label="band breaches"
              color={breaches ? 'var(--bad)' : 'var(--ok)'}
            />
          </div>
          <p>
            The upper bound is not approached &mdash; it is <i>attained</i>. Over
            every cell in the sweep the measured separation sits on the bound
            rather than under it, which is why the gap curve rides the floor of a
            log axis instead of decaying toward it.
          </p>
          <p className="note">
            This is the strongest result on the page. The register records{' '}
            {CO.band_breaches} band breaches and {CO.lower_bound_breaches}{' '}
            lower-bound breaches on the graded instance, and the sweep above adds{' '}
            {SWEEPS.collapse.rows.length} more &lambda; values with none either.
            The failures below are elsewhere.
          </p>
        </div>
      </div>
    </div>
  )
}

/* -------------------------------- weighted spread against degree -- */

// The operational claim: a universally-attached vertex is invisible to
// degree and visible to weighted spread. Both are plotted for the same
// graphs so the reader can see degree carry no signal.
function SpreadVsDegreeDemo() {
  const rows = SWEEPS.degree.rows
  return (
    <div className="panel">
      <div className="grid g2">
        <BarChart
          bars={rows.map((r) => ({
            label: 'n=' + r.n,
            value: r.ws_z,
            color: SERIES[0],
          }))}
          yDomain={[0, Math.max(...rows.map((r) => r.ws_z)) * 1.2]}
          yLabel="weighted spread at the sink"
          yFmt={(v) => v.toFixed(2)}
          h={260}
        />
        <BarChart
          bars={rows.map((r) => ({
            label: 'n=' + r.n,
            value: r.deg_z / r.n,
            color: SERIES[3],
          }))}
          yDomain={[0, 1.1]}
          yLabel="sink degree ÷ n"
          yFmt={(v) => v.toFixed(2)}
          h={260}
        />
      </div>
      <p className="note">
        Left: weighted spread separates the sink from everything else. Right: its
        degree, normalised by graph size, is flat &mdash; the sink looks like an
        ordinary well-connected vertex to a degree test, in every cell. That is
        the whole operational case for the more expensive statistic, and it is
        the reason the O(n&middot;|E|) cost is worth paying.
      </p>
    </div>
  )
}

/* --------------------------------------- the false lower bound -- */

// thm:spread-sound's lower bound. Plotting predicted against observed
// with the minimiser-moved split makes the failure legible: the breaches
// are not scattered, they are exactly the moved cases.
function SpreadSoundDemo() {
  const pts = SWEEPS.spread_sound.points
  const moved = pts.filter((p) => p.moved)
  const fixed = pts.filter((p) => !p.moved)
  const lo = Math.min(...pts.map((p) => Math.min(p.pred, p.new)))
  const hi = Math.max(...pts.map((p) => Math.max(p.pred, p.new)))

  return (
    <div className="panel">
      <ScatterChart
        groups={[
          {
            label: 'minimiser unchanged',
            color: OKC,
            points: fixed.map((p) => [p.pred, p.new, 'n = ' + p.n]),
          },
          {
            label: 'minimiser moved',
            color: BADC,
            points: moved.map((p) => [p.pred, p.new, 'n = ' + p.n]),
          },
        ]}
        xDomain={[lo, hi]}
        yDomain={[lo, hi]}
        xLabel="predicted spread"
        yLabel="observed after deletion"
        xFmt={(v) => v.toFixed(1)}
        yFmt={(v) => v.toFixed(1)}
        diagonal
        h={340}
      />
      <p className="note">
        Green points sit on the diagonal: where the minimising set is unchanged
        by the deletion, the theorem holds exactly. Red points are where deleting
        the vertex made a different admissible set cheaper. Over the register&rsquo;s{' '}
        {SA.cases} cases the upper bound held every time &mdash;{' '}
        <b>{SA.upper_bound_breaches} breaches</b>, worst overshoot{' '}
        {SA.worst_upper_overshoot.toExponential(1)}, which is floating-point
        noise &mdash; while the lower bound was breached{' '}
        <b>{SA.lower_bound_breaches} times</b>. Every one of those breaches has a
        moved minimiser, which is what the counterexamples in the register show:
        the predicted value assumes the old minimising set survives, and it does
        not.
      </p>
    </div>
  )
}

/* ---------------------------------- the unsatisfiable threshold -- */

function ThresholdDemo() {
  const m = SWEEPS.threshold.margins
  // The headline figures come from the RECORD, not from these points:
  // the chart is a downsample and its extremes are not the sweep's.
  const rec = EXPERIMENTS.exp5.records.threshold_satisfiability
  return (
    <div className="panel">
      <ScatterChart
        groups={[
          {
            label: 'vertices',
            color: SERIES[3],
            points: m.map((x) => [x.thr, x.wspread, 'n = ' + x.n]),
          },
        ]}
        xDomain={[0, Math.max(...m.map((x) => x.thr)) * 1.05]}
        yDomain={[0, Math.max(...m.map((x) => x.thr)) * 1.05]}
        xLabel="threshold required"
        yLabel="weighted spread achieved"
        xFmt={(v) => v.toFixed(1)}
        yFmt={(v) => v.toFixed(1)}
        diagonal
        h={320}
      />
      <p className="note">
        Every point lies <i>below</i> the diagonal, meaning no vertex reaches its
        own threshold. The theorem is not false &mdash; it is vacuous: its
        antecedent is satisfied by nothing. Over the full sweep of{' '}
        {rec.vertices_tested} vertices the antecedent fired{' '}
        {rec.strict_fires} times, and the closest any vertex came was{' '}
        {Math.abs(rec.best_margin).toFixed(4)} short. A conditional nothing
        satisfies proves nothing about anything. The {m.length} points plotted
        are a downsample of that sweep, so read the two figures above from the
        register rather than off the chart&rsquo;s extremes.
      </p>
    </div>
  )
}

/* -------------------------------------------------------- the page -- */

export default function SinkDetection() {
  return (
    <>
      <PaperHead paper={PAPER} />

      <Section
        id="argument"
        kicker="the claim"
        title="Finding the vertex a residual drains into"
        sub={
          'A sink is a vertex that quietly absorbs separation structure. It is ' +
          'invisible to the cheap test and visible to the expensive one, and the ' +
          'paper is about establishing that the expensive one is worth its cost.'
        }
      >
        <p>
          Attach one vertex to everything and the graph&rsquo;s separation
          structure collapses: every pair now has a short path, so cuts that were
          informative become uniform. A degree test cannot see this, because the
          offending vertex looks exactly like a hub. Weighted spread &mdash; the
          variance of a vertex&rsquo;s separations, weighted by cut cost &mdash;
          can.
        </p>

        <h3>The collapse bound is attained, not approached</h3>
        <CollapseDemo />

        <h3>Degree cannot see the sink</h3>
        <SpreadVsDegreeDemo />
      </Section>

      <Section
        id="defects"
        kicker="what broke"
        title="Three stated results that do not hold as printed"
        sub={
          'These were found by the experiment, not by a reviewer, and they are on ' +
          'this page rather than in an appendix. Two of the fifteen graded rows ' +
          'below fail, and this is why.'
        }
      >
        <h3>thm:spread-sound &mdash; the lower bound is false</h3>
        <SpreadSoundDemo />
        <Callout tone="warn">
          The proof pins the deleted separation to a single value by assuming the
          minimising set survives the deletion. It need not: removing a vertex
          removes edges elsewhere and can make a different admissible set
          cheaper. The upper bound is sound and unaffected. The paper replaces
          the statement with <span className="mono">thm:spread-sound-fixed</span>,
          conditioned on the minimiser being unchanged &mdash; which is exactly
          the green half of the chart above.
        </Callout>

        <h3>thm:threshold &mdash; the antecedent is unsatisfiable</h3>
        <ThresholdDemo />

        <h3>prop:amplify &mdash; the depth half of the proof is wrong</h3>
        <Callout tone="warn">
          The proof closes by claiming that taking λ large <i>or the analysis to
          deeper S</i> drives the fraction to 1. The bound it has just derived is{' '}
          <span className="mono">λ/(λ + C)</span>, which is <b>constant in
          |S|</b>. Depth does not drive it anywhere; only λ does. Measured over
          n = {AM[0].n}&hellip;{AM[AM.length - 1].n}, the <i>minimum</i> share is
          exactly {AM[0].min_z_fraction.toFixed(4)} at every n &mdash; flat, as
          the λ-only bound requires &mdash; while the mean drifts gently down
          from {AM[0].mean_z_fraction.toFixed(4)} to{' '}
          {AM[AM.length - 1].mean_z_fraction.toFixed(4)}. Neither rises with
          depth. The control is what makes this a result rather than an artefact:
          an ordinary vertex&rsquo;s share <i>dilutes</i> with n, from{' '}
          {AC.ordinary_vertex[0].mean_ordinary_fraction.toFixed(4)} to{' '}
          {AC.ordinary_vertex[AC.ordinary_vertex.length - 1].mean_ordinary_fraction.toFixed(4)},
          so the sink&rsquo;s refusal to dilute is the signal. The claim the proof
          actually establishes &mdash; that the fraction does not fall away with
          n &mdash; is what the experiment grades, and it passes.
        </Callout>
      </Section>

      <Section
        id="claims"
        kicker="registered predictions"
        title="What was predicted, and what happened"
        sub="Fifteen rows, thirteen reproduced, two refuted. The two failures are the defects above."
      >
        <ClaimsTable expKey="exp5" />
        <Environment expKey="exp5" />
      </Section>

      <Section id="limits" kicker="limitations" title="Recorded by the experiment">
        <ExperimentNotes expKey="exp5" />
        <p className="note">
          Read with <Link to="/paper/runtime-graph">runtime graph</Link>, and in that
          order: this paper is the second of an operational pair, and it assumes
          the cut and certificate machinery that one establishes.
        </p>
      </Section>
    </>
  )
}
