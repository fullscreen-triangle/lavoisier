// =====================================================================
//  Coordinate provenance --- the type-discipline paper.
//
//  Every other paper asks what a measurement leaves out. This one asks
//  what a PIPELINE leaves out: once a coordinate has been mapped into a
//  smaller space, no downstream stage recovers what it dropped. The
//  record-cost ratio is recomputed live here from the shipped stage
//  sizes, because it is arithmetic and there is no reason to quote it.
// =====================================================================

import { useState } from 'react'
import { PaperHead, ClaimsTable, ExperimentNotes, Environment } from '../../components/PaperLayout.jsx'
import { Section, Slider, Callout, Stat } from '../../components/Primitives.jsx'
import LineChart from '../../components/charts/LineChart.jsx'
import BarChart from '../../components/charts/BarChart.jsx'
import { SERIES, BADC } from '../../components/charts/chart-kit.jsx'
import { BY_SLUG } from '../../papers.js'
import { EXPERIMENTS } from '../../lib/data.js'

const PAPER = BY_SLUG['coordinate-provenance']
const R = EXPERIMENTS.exp3.records

/* ------------------------------------------------- the collapse -- */

function CollapseDemo() {
  const c = R.collapse_small_codomain
  const p = R.pipeline_loss
  const n = R.no_recovery

  return (
    <div className="panel">
      <div className="grid g2">
        <BarChart
          bars={[
            { label: 'stage-1 maps', value: c.n_maps, color: SERIES[0] },
            { label: 'auditable', value: c.auditable, color: BADC },
            { label: 'stage-2 maps', value: p.n_stage2_maps, color: SERIES[1] },
            { label: 'auditability restored', value: p.restored_auditability, color: BADC },
          ]}
          yDomain={[0, c.n_maps * 1.2]}
          yLabel="count"
          yFmt={(v) => String(Math.round(v))}
          horizontal
          w={560}
          h={260}
        />
        <div>
          <p>
            A coordinate map into a codomain smaller than its context space
            cannot be auditable: there are more contexts than slots, so two
            contexts must land on the same coordinate and nothing downstream can
            tell them apart. Enumerating all{' '}
            <b>{c.n_maps}</b> maps from a {c.size_Ctx}-context space into a{' '}
            {c.size_Crd}-element codomain found <b>{c.auditable}</b> auditable
            ones &mdash; not few, none.
          </p>
          <p>
            The second pair is the pipeline claim. Given a stage-1 map that has
            already collapsed, all {p.n_stage2_maps} possible stage-2 maps were
            tried, and <b>{p.restored_auditability}</b> restored auditability.
            Loss at any stage is terminal for the pipeline.
          </p>
          <p className="note">
            Both are paired with controls that came out the other way, which is
            what stops the zeros being vacuous: widening the codomain to{' '}
            {R.collapse_control.size_Crd} produces an auditable witness, an
            already-auditable stage 1 stays auditable, and with distinct inputs{' '}
            {R.no_recovery_control.recoveries_when_inputs_distinct} of{' '}
            {n.n_functions_enumerated} enumerated functions recover the
            distinction. The machinery can succeed; it just cannot succeed here.
          </p>
        </div>
      </div>
    </div>
  )
}

/* ------------------------------------------------ the record cost -- */

// The ratio is arithmetic over the shipped stage record sizes, so it is
// computed rather than quoted --- and the slider walks N across the
// range the paper reports, showing the ratio barely moving.
function CostDemo() {
  const sizes = R.cost.record_sizes
  const rows = R.cost.rows
  const [logN, setLogN] = useState(2)
  const N = Math.pow(10, logN)

  // prop:cost verbatim, transcribed from exp3_coordinate_provenance.py:
  // attaching a record to every measurement costs N*sum(log2|Prv_i|);
  // storing per run and referencing costs N*s*log2(s) + sum(log2|Prv_i|),
  // where s is the stage count. Reproduces the shipped rows exactly.
  const stages = R.cost.stages
  const logSum = sizes.reduce((a, b) => a + Math.log2(b), 0)
  const perMeasurement = N * logSum
  const perRun = N * stages * Math.log2(stages) + logSum
  const ratio = perMeasurement / perRun

  return (
    <div className="panel">
      <div className="grid g2">
        <LineChart
          series={[
            {
              label: 'recorded ratio',
              points: rows.map((r) => [Math.log10(r.N), r.ratio]),
              color: SERIES[0],
            },
            {
              label: 'control --- one tiny stage',
              points: rows.map((r) => [
                Math.log10(r.N),
                R.cost_control.ratio_at_one_tiny_stage,
              ]),
              color: BADC,
            },
          ]}
          xDomain={[2, 6]}
          yDomain={[1, R.cost_control.ratio_at_one_tiny_stage * 1.3]}
          logY
          xLabel="log₁₀ N (measurements)"
          yLabel="cost ratio"
          xFmt={(v) => String(v)}
          yFmt={(v) => v.toFixed(0)}
          h={290}
        />
        <div>
          <Slider
            label="measurements N"
            value={logN}
            min={2}
            max={6}
            step={1}
            onChange={setLogN}
            fmt={(v) => '10^' + v}
          />
          <div className="grid g3" style={{ marginTop: 14 }}>
            <Stat value={ratio.toFixed(4)} label="ratio, recomputed here" />
            <Stat
              value={Math.abs(ratio - rows[logN - 2].ratio) < 1e-12 ? 'exact' : 'DRIFT'}
              label="against the shipped row"
              color="var(--ok)"
            />
            <Stat value={N.toExponential(0)} label="measurements" />
          </div>
          <p>
            Provenance is cheap when it is recorded per run rather than per
            measurement. Attaching a record to every measurement costs{' '}
            <span className="mono">N&middot;&Sigma;log&#8322;|Prv&#7522;|</span>;
            storing it once per run and referencing it costs{' '}
            <span className="mono">N&middot;s&middot;log&#8322;s + &Sigma;log&#8322;|Prv&#7522;|</span>.
            Over four decades the ratio barely moves &mdash;{' '}
            {rows[0].ratio.toFixed(4)} at N = 10&sup2; against{' '}
            {rows[rows.length - 1].ratio.toFixed(4)} at 10&#8310; &mdash; because
            both schemes are linear in N and only the constant differs. The{' '}
            {stages} stage records are {sizes.join(', ')} bits.
          </p>
          <p className="note">
            The red line is the control, and it is the reason the blue one means
            anything: make one stage tiny and the ratio jumps to{' '}
            {R.cost_control.ratio_at_one_tiny_stage.toFixed(1)}. The measurement
            is sensitive to stage structure &mdash; the flatness is a property of
            these stages, not of the formula.
          </p>
        </div>
      </div>
    </div>
  )
}

/* ------------------------------------------------ declining well -- */

function ComparabilityDemo() {
  const c = R.comparability
  const d = R.decline_sound
  return (
    <div className="panel">
      <div className="grid g3">
        <Stat value={c.admitted_pairs} label="pairs admitted for comparison" />
        <Stat value={c.unsound_admissions} label="unsound admissions" color="var(--ok)" />
        <Stat value={c.declined_context_pairs} label="declined" color="var(--warn)" />
      </div>
      <p style={{ marginTop: 14 }}>
        The comparator is partial by construction: it compares two coordinates
        only when their contexts license the comparison, and declines otherwise.
        Over {c.admitted_pairs} admitted pairs it made{' '}
        <b>{c.unsound_admissions}</b> unsound admissions, and it declined{' '}
        {c.declined_context_pairs} pairs.
      </p>
      <Callout>
        Those {d.total_comparator_unlicensed_answers} declines are exactly the
        pairs on which a <i>total</i> comparator &mdash; one obliged to return an
        answer &mdash; gives an unlicensed one. The partial comparator gives{' '}
        {d.partial_comparator_unlicensed_answers}. The declines are not the price
        of the discipline; they are the discipline working, on precisely the
        cases where answering would be wrong.
      </Callout>
    </div>
  )
}

/* -------------------------------------------------------- the page -- */

export default function CoordinateProvenance() {
  const mr = R.minimal_record
  const mf = R.minimal_record_factorisation
  const rf = R.record_floor

  return (
    <>
      <PaperHead paper={PAPER} />

      <Section
        id="argument"
        kicker="the claim"
        title="What a pipeline drops, it does not get back"
        sub={
          'The type-discipline paper. A coordinate carries the context it was ' +
          'computed in; once a stage discards that context, no later stage ' +
          'recovers it, and the paper turns that into a condition on maps.'
        }
      >
        <h3>Collapse is terminal</h3>
        <CollapseDemo />

        <h3>Provenance costs almost nothing, recorded properly</h3>
        <CostDemo />

        <h3>Comparing coordinates, or declining to</h3>
        <ComparabilityDemo />

        <h3>The record has a floor, and the floor is reached</h3>
        <p>
          <span className="mono">thm:record-floor</span> says the minimum record
          cannot be trivial: with {rf.Prv_capacity} slots of provenance capacity,
          the number of equivalence classes grows with each release and exhausts
          the capacity at release {rf.exhausted_at_release}. Its control confirms
          the count is measuring structure rather than volume &mdash; adding a
          duplicate context leaves the class count at{' '}
          {R.record_floor_control.classes_after_duplicate}, unchanged.
        </p>
        <div className="panel">
          <LineChart
            series={[
              {
                label: 'equivalence classes',
                points: rf.history.map((h) => [h.release, h.n_classes]),
                color: SERIES[0],
              },
            ]}
            xDomain={[
              Math.min(...rf.history.map((h) => h.release)),
              Math.max(...rf.history.map((h) => h.release)),
            ]}
            yDomain={[0, Math.max(...rf.history.map((h) => h.n_classes)) * 1.2]}
            xLabel="release"
            yLabel="classes to record"
            xFmt={(v) => String(v)}
            yFmt={(v) => String(v)}
            rules={[
              { at: rf.Prv_capacity, color: BADC, label: 'capacity' },
            ]}
            h={280}
          />
          <p className="note">
            The line crosses the capacity rule and never comes back. A provenance
            budget fixed at design time is a budget that runs out, and the paper
            reports the release at which it does rather than choosing a capacity
            large enough to hide it.
          </p>
        </div>
      </Section>

      <Section
        id="defect"
        kicker="what broke"
        title="thm:minimal-record claims more than its proof establishes"
        sub={
          'All nine graded rows reproduced, so this is not a failed prediction ' +
          '--- it is a statement defect the experiment went looking for and ' +
          'found, and it is reported here rather than left in a footnote.'
        }
      >
        <Callout tone="warn">
          The theorem has two halves. The half its proof establishes &mdash; that{' '}
          <span className="mono">rec_min</span> is sufficient and has the
          smallest image among sufficient maps, confirmed here with{' '}
          <b>{mr.sufficient_maps_with_smaller_image}</b> sufficient maps having a
          smaller image over {mr.n_contexts} contexts and{' '}
          {mr.n_equivalence_classes} classes &mdash; is sound and is what{' '}
          <span className="mono">thm:record-floor</span> consumes. The half the
          printed statement <i>also</i> claims is a universal property: that
          every sufficient record map factors through{' '}
          <span className="mono">rec_min</span>. That is false. The experiment
          exhibits <b>{mf.sufficient_maps_not_factoring_through_rec_min}</b>{' '}
          sufficient maps that do not factor, one of them as explicit as{' '}
          <span className="mono">{JSON.stringify(mf.example)}</span> &mdash; a
          map that splits an equivalence class and is still sufficient.
        </Callout>
        <p className="note">
          The consequence is bounded and worth stating precisely: the floor
          result is unaffected, because it consumes only the image-size bound.
          What must go is the word &ldquo;factors&rdquo; in the statement of{' '}
          <span className="mono">thm:minimal-record</span>. The control confirms
          the test can fail in the other direction too &mdash; a smaller but
          insufficient map exists, so sufficiency is doing real work.
        </p>
      </Section>

      <Section
        id="claims"
        kicker="registered predictions"
        title="What was predicted, and what happened"
        sub="Nine registered, nine reproduced. Every claim is paired with a control that came out the other way."
      >
        <ClaimsTable expKey="exp3" />
        <Environment expKey="exp3" />
      </Section>

      <Section id="limits" kicker="limitations" title="Recorded by the experiment">
        <ExperimentNotes expKey="exp3" />
        <p className="note">
          This paper uses none of the graph machinery &mdash; no medium, no
          separation, no cut key. It shares only the floor and the stance on
          declining, and it is the one place in the catalogue where the floor is
          a bound on a <i>record</i> rather than on a measurement.
        </p>
      </Section>
    </>
  )
}
