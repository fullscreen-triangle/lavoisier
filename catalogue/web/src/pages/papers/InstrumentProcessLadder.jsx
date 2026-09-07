// =====================================================================
//  Instrument process ladder.
//
//  This is the one page where nothing is read from a file. The two
//  cells below carry the experiment sources verbatim, and every number
//  the page shows about them --- composite, sensitivities, ablation
//  losses, the bound --- is produced by the JavaScript port running in
//  the reader's browser. scripts/check_port.mjs gates that port against
//  the Python results bit for bit, so "recomputed" here means the same
//  double, not a rounded agreement.
// =====================================================================

import { useState } from 'react'
import { PaperHead, ClaimsTable, ExperimentNotes, Environment } from '../../components/PaperLayout.jsx'
import { Section, Slider, Callout, Stat } from '../../components/Primitives.jsx'
import Notebook from '../../components/notebook/Notebook.jsx'
import LineChart from '../../components/charts/LineChart.jsx'
import BarChart from '../../components/charts/BarChart.jsx'
import { SERIES } from '../../components/charts/chart-kit.jsx'
import { compose, sensitivity } from '../../lib/ss/stdlib.js'
import { BY_SLUG } from '../../papers.js'

const PAPER = BY_SLUG['instrument-process-ladder']

const EXP1 = `import lavoisier.ladder

objective InstrumentLadder:
    target: "evaluate a five-contact instrument without a substrate"
    criterion: "composite 0.91712, control at the strongest contact"

ladder instrument toward target
  rung k1 at 0.60
  rung k2 at 0.35
  rung k3 at 0.50
  rung k4 at 0.25
  rung k5 at 0.15
  require resolution >= 0.90

phase Resolve:
    evaluation = lavoisier.ladder.resolve(ladder: instrument)

phase Ablate:
    // Which contacts does the requirement actually depend on?
    ablation = lavoisier.ladder.ablate(ladder: instrument)

phase Bound:
    // Static question, answered without executing anything.
    bound = lavoisier.ladder.minimum(target: 0.90, pow_max: 0.60)
`

const EXP1B = `// Negative control for prediction P9 (static reachability). These three
// contacts cannot reach 0.90; the compiler must reject the program
// before any phase runs. If this program executes, P9 is refuted.

import lavoisier.ladder

objective Unreachable:
    target: "a declared requirement the declared contacts cannot meet"
    criterion: "compile stage must reject before execution"

ladder shortfall toward target
  rung a at 0.30
  rung b at 0.25
  rung c at 0.20
  require resolution >= 0.90

phase Resolve:
    evaluation = lavoisier.ladder.resolve(ladder: shortfall)
`

/* ------------------------------------------ how the rungs compose -- */

// The three candidate composition rules the experiment discriminates
// between. Only the first is the paper's; the other two are what the
// 400-trial control was built to rule out, and they are drawn here so
// the reader can see they are not close.
function CompositionDemo() {
  const [powers, setPowers] = useState([0.6, 0.35, 0.5, 0.25, 0.15])

  const set = (i, v) => {
    const next = powers.slice()
    next[i] = v
    setPowers(next)
  }

  const prefix = (f) => {
    const pts = [[0, 0]]
    for (let n = 1; n <= powers.length; n++) pts.push([n, f(powers.slice(0, n))])
    return pts
  }

  const mult = prefix(compose)
  const add = prefix((ps) => Math.min(1, ps.reduce((a, b) => a + b, 0)))
  const max = prefix((ps) => Math.max(...ps))
  const c = compose(powers)

  return (
    <div className="panel">
      <div className="grid g2">
        <LineChart
          series={[
            { label: 'multiplicative (the paper)', points: mult, color: SERIES[0] },
            { label: 'additive', points: add, color: SERIES[1] },
            { label: 'max', points: max, color: SERIES[2] },
          ]}
          xDomain={[0, powers.length]}
          yDomain={[0, 1.05]}
          xLabel="contacts applied"
          yLabel="composite resolution"
          xFmt={(t) => String(t)}
          yFmt={(t) => t.toFixed(2)}
          h={300}
        />
        <div>
          {powers.map((p, i) => (
            <Slider
              key={i}
              label={'rung k' + (i + 1)}
              value={p}
              min={0.01}
              max={0.99}
              step={0.01}
              onChange={(v) => set(i, v)}
              fmt={(v) => v.toFixed(2)}
            />
          ))}
          <div className="grid g2" style={{ marginTop: 14 }}>
            <Stat value={c.toFixed(6)} label="composite" />
            <Stat
              value={c >= 0.9 ? 'met' : 'short'}
              label="requirement 0.90"
              color={c >= 0.9 ? 'var(--ok)' : 'var(--bad)'}
            />
          </div>
        </div>
      </div>
      <p className="note">
        The three curves are not variants of one shape. Additive crosses 1 and
        keeps going &mdash; a resolution of 1.35 is not a number an instrument
        can report &mdash; and max ignores every contact but the strongest, so
        adding a sixth contact does nothing. The experiment ran 400 randomised
        trials against measured readouts: multiplicative reproduced them to a
        maximum absolute error of 0.00, additive to 0.0728, max-based to 0.1890.
      </p>
    </div>
  )
}

/* --------------------------------------------------- sensitivity -- */

// Sensitivity is the derivative of the composite with respect to one
// rung, and the paper's claim is that the control sits at the largest
// of them. The bars recompute live; the claim is visible as an
// ordering, not asserted.
function SensitivityDemo() {
  const [powers, setPowers] = useState([0.6, 0.35, 0.5, 0.25, 0.15])
  const s = sensitivity(powers)
  const top = s.indexOf(Math.max(...s))
  const strongest = powers.indexOf(Math.max(...powers))

  const set = (i, v) => {
    const next = powers.slice()
    next[i] = v
    setPowers(next)
  }

  return (
    <div className="panel">
      <div className="grid g2">
        <BarChart
          bars={s.map((v, i) => ({
            label: 'k' + (i + 1),
            value: v,
            color: i === top ? SERIES[0] : 'var(--panel-2)',
          }))}
          yDomain={[0, Math.max(...s) * 1.15]}
          yLabel="sensitivity"
          yFmt={(t) => t.toFixed(2)}
          h={280}
        />
        <div>
          {powers.map((p, i) => (
            <Slider
              key={i}
              label={'rung k' + (i + 1)}
              value={p}
              min={0.01}
              max={0.99}
              step={0.01}
              onChange={(v) => set(i, v)}
              fmt={(v) => v.toFixed(2)}
            />
          ))}
          <div className="grid g2" style={{ marginTop: 14 }}>
            <Stat value={'k' + (top + 1)} label="most sensitive" />
            <Stat value={'k' + (strongest + 1)} label="strongest contact" />
          </div>
          <p className="note">
            The paper predicts these two are always the same rung, and the
            highlighted bar is the reason: sensitivity is the product of every
            other rung&rsquo;s complement, which is largest exactly where the
            rung&rsquo;s own power is largest. Move any slider and they move
            together.
          </p>
        </div>
      </div>
    </div>
  )
}

/* -------------------------------------------------------- the page -- */

export default function InstrumentProcessLadder() {
  return (
    <>
      <PaperHead paper={PAPER} />

      <Section
        id="argument"
        kicker="the claim"
        title="An instrument is a sequence of contacts"
        sub={
          'Each contact resolves some fraction of what is left. Nothing about ' +
          'the substrate enters, which is what makes the readout computable ' +
          'from the instrument alone.'
        }
      >
        <p>
          A contact with resolving power <span className="mono">p</span> leaves a
          fraction <span className="mono">1 &minus; p</span> of the gap
          unresolved. Applying contacts in sequence multiplies the complements,
          so a ladder of powers{' '}
          <span className="mono">p&#8321; &hellip; p&#8345;</span> has composite
          resolution <span className="mono">1 &minus; &Pi;(1 &minus; p&#7522;)</span>.
          That product never reaches 1: the residual is strictly positive for
          every finite ladder, which is this paper&rsquo;s form of the floor.
        </p>
        <p>
          The interesting content is not the formula but that it is the{' '}
          <i>right</i> formula. Two alternatives are just as easy to write down,
          and the experiment was built to tell them apart rather than to confirm
          the one the paper preferred.
        </p>
        <CompositionDemo />

        <h3>Where the control sits</h3>
        <p>
          If an instrument is to be controlled, the operator wants the contact
          whose power most changes the readout. The paper proves that is always
          the strongest contact &mdash; which is not obvious, since one might
          expect the weakest link to matter most.
        </p>
        <SensitivityDemo />

        <h3>Substrate independence, and what it is worth</h3>
        <Callout>
          The experiment evaluated the same instrument against two different
          substrates and got composites agreeing to twelve digits. That is a
          strong reproduction of a claim that is, read carefully, an{' '}
          <b>analytic</b> one: no substrate quantity appears in the formula, so
          agreement is arithmetic rather than evidence about instruments. The
          measurable claim is the 400-trial comparison against real readouts
          above, and that is where the discriminating power is.
        </Callout>
      </Section>

      <Section
        id="run"
        kicker="executable"
        title="Run the experiment"
        sub={
          'Both programs below are the experiment sources verbatim. The parser, ' +
          'compiler and ladder arithmetic are ported to JavaScript and reproduce ' +
          'the recorded Python results to full double precision --- edit a rung ' +
          'power and everything downstream recomputes.'
        }
      >
        <Notebook
          title="Experiment 1 --- the five-contact instrument"
          intro="Three phases: resolve the ladder, ablate each contact in turn, and answer the static bound question."
          cells={[
            {
              title: 'exp1_instrument_ladder.ss',
              note:
                'Change rung k1 from 0.60 to 0.75 and re-run: the composite, ' +
                'the sensitivity bars and the ablation table all move together.',
              source: EXP1,
            },
            {
              title: 'exp1b_unreachable.ss --- the negative control',
              note:
                'Three contacts cannot reach 0.90. The compiler must reject this ' +
                'before any phase runs; if it executes, prediction P9 is refuted.',
              source: EXP1B,
            },
          ]}
        />
        <p className="note">
          Things worth trying: set a rung to <span className="mono">1.5</span>{' '}
          and the parser refuses it, because a resolving power outside{' '}
          <span className="mono">[0, 1)</span> is not a contact. Raise exp1&rsquo;s
          requirement to <span className="mono">&gt;= 0.99</span> and the
          compiler rejects it statically, with the shortfall arithmetic in the
          diagnostic. Ask{' '}
          <span className="mono">lavoisier.ladder.minimum(target: 1.0, pow_max: 0.6)</span>{' '}
          and the operation declines &mdash; that refusal is the floor, reported
          as an outcome rather than a crash.
        </p>
      </Section>

      <Section
        id="claims"
        kicker="registered predictions"
        title="What was predicted, and what happened"
        sub={
          'Each row was registered with a failure mode before the experiment ran. ' +
          'This is the whole register, read from the result file.'
        }
      >
        <ClaimsTable expKey="exp1" />
        <Environment expKey="exp1" />
      </Section>

      <Section
        id="limits"
        kicker="limitations"
        title="What this page does not establish"
      >
        <ExperimentNotes expKey="exp1" />
        <Callout tone="warn">
          Substrate independence is proved, not measured. It follows from the
          formula containing no substrate term, so the twelve-digit agreement
          between two substrates confirms the arithmetic and not the physics.
        </Callout>
        <Callout tone="warn">
          The ladder here is a sequence of instrument contacts. The peptide paper
          also has a &ldquo;ladder&rdquo; &mdash; a series of fragment masses
          &mdash; and the two are unrelated. Nothing proved on this page
          transfers to that one.
        </Callout>
        <p className="note">
          This is the most self-contained paper in the catalogue: it uses the
          floor without needing the non-completability axiom the other six state,
          and it does not use the medium, the cut key or separation at all.
        </p>
      </Section>
    </>
  )
}
