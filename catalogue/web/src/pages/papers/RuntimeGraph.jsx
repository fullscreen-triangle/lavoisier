// =====================================================================
//  Runtime graph certificates.
//
//  The first half of the operational pair. Everything on this page came
//  out as predicted --- eleven of eleven --- which makes the caveat in
//  the experiment notes the most interesting thing here: the witness the
//  paper could adopt for thm:not-a-cut is carried by medium edges for 13
//  of its 15 queries, and the notes say so.
// =====================================================================

import { Link } from 'react-router-dom'
import { PaperHead, ClaimsTable, ExperimentNotes, Environment } from '../../components/PaperLayout.jsx'
import { Section, Callout, Stat } from '../../components/Primitives.jsx'
import LineChart from '../../components/charts/LineChart.jsx'
import BarChart from '../../components/charts/BarChart.jsx'
import { SERIES, BADC } from '../../components/charts/chart-kit.jsx'
import { BY_SLUG } from '../../papers.js'
import SWEEPS from '../../data/exp4_sweeps.json'

const PAPER = BY_SLUG['runtime-graph']

/* ---------------------------------------------------- verify cost -- */

function CostDemo() {
  const rows = SWEEPS.probe
  return (
    <div className="panel">
      <div className="grid g2">
        <LineChart
          series={[
            {
              label: 'brute force',
              points: rows.map((r) => [r.n, r.brute_us]),
              color: SERIES[1],
            },
            {
              label: 'flow',
              points: rows.map((r) => [r.n, r.flow_us]),
              color: SERIES[0],
            },
          ]}
          xDomain={[Math.min(...rows.map((r) => r.n)), Math.max(...rows.map((r) => r.n))]}
          yDomain={[1, Math.max(...rows.map((r) => r.brute_us)) * 1.4]}
          logY
          xLabel="items"
          yLabel="microseconds per query"
          xFmt={(v) => String(v)}
          yFmt={(v) => v.toExponential(0)}
          h={300}
        />
        <div>
          <p>
            Both methods answer the same question and are checked against each
            other &mdash; the flow implementation is validated against exhaustive
            minimisation, not the other way round. What separates them is how the
            cost grows: exhaustive minimisation walks every admissible subset,
            the flow computation does not.
          </p>
          <div className="grid g2" style={{ marginTop: 12 }}>
            <Stat
              value={rows.reduce((a, r) => a + r.mismatches, 0)}
              label="mismatches between the two"
              color="var(--ok)"
            />
            <Stat value={rows.reduce((a, r) => a + r.queries, 0)} label="queries checked" />
          </div>
          <p className="note">
            At the small sizes on the left the brute-force method is genuinely
            faster: the flow setup costs more than enumerating a handful of
            subsets. The asymptotic claim is about the slope, and the chart shows
            the crossing rather than hiding it.
          </p>
        </div>
      </div>
    </div>
  )
}

/* ------------------------------------------------------- forgeries -- */

function ForgeryDemo() {
  const f = SWEEPS.forge
  const kinds = Object.keys(f)
  const attempted = kinds.reduce((a, k) => a + f[k].attempted, 0)
  const passed = kinds.reduce((a, k) => a + f[k].passed, 0)

  const mag = SWEEPS.forge_magnitude.rows

  return (
    <div className="panel">
      <div className="grid g2">
        <BarChart
          bars={kinds.map((k) => ({
            label: k,
            value: f[k].attempted,
            color: f[k].passed ? BADC : SERIES[0],
          }))}
          yDomain={[0, Math.max(...kinds.map((k) => f[k].attempted)) * 1.2]}
          yLabel="forgeries attempted"
          yFmt={(v) => String(Math.round(v))}
          horizontal
          w={560}
          h={240}
        />
        <div>
          <div className="grid g2">
            <Stat value={attempted} label="forgeries attempted" />
            <Stat
              value={passed}
              label="accepted"
              color={passed ? 'var(--bad)' : 'var(--ok)'}
            />
          </div>
          <p>
            A certificate is <span className="mono">z = (S, γ, f)</span>: the cut
            set, its claimed value, and the flow witnessing it. Verification is
            O(|E|) and consults only the certificate and the graph. Four families
            of tampering were tried &mdash; wrong value, a vertex added, a vertex
            dropped, and the medium smuggled into the cut set &mdash; and none
            verified.
          </p>
        </div>
      </div>

      <h3 style={{ marginTop: 24 }}>How small a lie gets through?</h3>
      <LineChart
        series={[
          {
            label: 'accepted',
            points: mag.map((r) => [Math.max(r.rel_magnitude, 1e-18), r.accepted]),
            color: SERIES[0],
          },
        ]}
        xDomain={[1e-18, 1]}
        yDomain={[0, 300]}
        xLabel="relative size of the tampering"
        yLabel="certificates accepted"
        xFmt={(v) => v.toExponential(0)}
        yFmt={(v) => String(v)}
        step
        h={260}
      />
      <p className="note">
        The honest reading of the forgery result. A tampering of relative
        magnitude 0 <i>is</i> the honest certificate, and one at 10&#8315;&sup1;&sup2;
        is below the precision anything here is computed to &mdash; both are
        accepted, and should be. The claim is about detectable forgeries, and the
        curve shows where the detection threshold actually lies rather than
        asserting it lies at zero.
      </p>
    </div>
  )
}

/* ---------------------------------------------------- the medium -- */

function MediumDemo() {
  const rows = SWEEPS.medium.rows
  const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length
  return (
    <div className="panel">
      <LineChart
        series={[
          {
            label: 'medium present',
            points: rows.map((r) => [r.mult, mean(r.with_medium)]),
            color: SERIES[0],
          },
          {
            label: 'medium deleted',
            points: rows.map((r) => [r.mult, mean(r.without_medium)]),
            color: BADC,
          },
        ]}
        xDomain={[0, Math.max(...rows.map((r) => r.mult))]}
        yDomain={[0, Math.max(...rows.map((r) => mean(r.with_medium))) * 1.2]}
        xLabel="medium edge weight multiplier"
        yLabel="mean separation"
        xFmt={(v) => v.toFixed(1)}
        yFmt={(v) => v.toFixed(2)}
        h={290}
      />
      <p className="note">
        The lower line is flat at exactly zero across the whole sweep. That is
        not a measurement of how graphs behave &mdash; the medium is{' '}
        <i>defined</i> as adjacent to every item, so deleting it disconnects
        everything and separation has nowhere to come from. Reported here because
        it is a useful control on the implementation, and because a reader should
        know which line is arithmetic and which is evidence.
      </p>
    </div>
  )
}

/* -------------------------------------------------------- the page -- */

export default function RuntimeGraph() {
  return (
    <>
      <PaperHead paper={PAPER} />

      <Section
        id="argument"
        kicker="the claim"
        title="Compile once, verify cheaply, and reject forgeries"
        sub={
          'A cut is expensive to find and should be cheap to check. This paper ' +
          'separates the two costs and shows the cheap half cannot be fooled.'
        }
      >
        <p>
          Compiling the graph costs{' '}
          <span className="mono">O(n log n + nd)</span>; verifying a certificate
          against it costs <span className="mono">O(|E|)</span>. The asymmetry is
          the point: an expensive claim can be handed to a party that will not
          pay to recompute it, provided the certificate cannot be forged.
        </p>

        <h3>The cost separation</h3>
        <CostDemo />

        <h3>Forgeries</h3>
        <ForgeryDemo />

        <h3>The medium, as a control</h3>
        <MediumDemo />
      </Section>

      <Section
        id="claims"
        kicker="registered predictions"
        title="What was predicted, and what happened"
        sub="Eleven registered, eleven reproduced. The interesting content is in the notes below."
      >
        <ClaimsTable expKey="exp4" />
        <Environment expKey="exp4" />
      </Section>

      <Section id="limits" kicker="limitations" title="Recorded by the experiment">
        <ExperimentNotes expKey="exp4" />
        <Callout tone="warn">
          <span className="mono">thm:not-a-cut</span> was located by examining
          31139 pairs, and the manuscript describes a construction rather than
          exhibiting one. The concrete witness the experiment found is available
          for adoption &mdash; but for 13 of its 15 queries the minimum
          admissible cut is the whole vertex set, so the agreement there is
          carried by the medium edges rather than by contact structure. It
          satisfies the theorem as stated; a manuscript adopting it should say
          this rather than imply the keys probe contact structure deeply.
        </Callout>
        <p className="note">
          Read this before <Link to="/paper/sink-detection">sink detection</Link>:
          the two are an operational pair, and the second assumes the cut and
          certificate machinery established here.
        </p>
      </Section>
    </>
  )
}
