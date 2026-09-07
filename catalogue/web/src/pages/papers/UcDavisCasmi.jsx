// =====================================================================
//  UC Davis CASMI catalogue --- the applied end.
//
//  This is the only paper with no expectation register, because it is
//  not testing a theorem: it runs the method against 58 real
//  identification challenges and reports what came out. The licensing
//  decision on this page is recomputed from the shipped per-challenge
//  measurements every time a slider moves, and at the paper's own
//  settings it reproduces the recorded 5 / 35 / 18 exactly.
// =====================================================================

import { useMemo, useState } from 'react'
import { PaperHead } from '../../components/PaperLayout.jsx'
import { Section, Slider, Callout, Stat } from '../../components/Primitives.jsx'
import LineChart from '../../components/charts/LineChart.jsx'
import BarChart from '../../components/charts/BarChart.jsx'
import ScatterChart from '../../components/charts/ScatterChart.jsx'
import { SERIES, OKC, BADC, WARNC } from '../../components/charts/chart-kit.jsx'
import { BY_SLUG } from '../../papers.js'
import PANEL from '../../data/panel_data.json'

const PAPER = BY_SLUG['uc-davis-casmi-catalogue']
const CH = PANEL.challenges

// The paper's settings. Everything else on this page moves; these two
// are what it actually reported against.
const BETA0 = 0.3
const MU0 = 0.1

function verdictOf(c, beta, mu) {
  if (c.contact < beta) return 'unsupported'
  if (c.margin < mu) return 'ambiguous'
  return 'licensed'
}

function tally(beta, mu) {
  const t = { licensed: 0, ambiguous: 0, unsupported: 0 }
  for (const c of CH) t[verdictOf(c, beta, mu)]++
  return t
}

/* ------------------------------------------------- live licensing -- */

function LicensingDemo() {
  const [beta, setBeta] = useState(BETA0)
  const [mu, setMu] = useState(MU0)

  const t = tally(beta, mu)
  const atPaper = Math.abs(beta - BETA0) < 1e-9 && Math.abs(mu - MU0) < 1e-9

  // The floor sweep: how the licensed count moves as beta walks the
  // whole range, holding the margin at its current value.
  const sweep = useMemo(() => {
    const pts = []
    for (let b = 0; b <= 1.0001; b += 0.02) pts.push([b, tally(b, mu).licensed])
    return pts
  }, [mu])

  const groups = [
    {
      label: 'licensed',
      color: OKC,
      points: CH.filter((c) => verdictOf(c, beta, mu) === 'licensed').map((c) => [
        c.contact,
        c.margin,
        'challenge ' + c.id,
      ]),
    },
    {
      label: 'decline --- ambiguous',
      color: WARNC,
      points: CH.filter((c) => verdictOf(c, beta, mu) === 'ambiguous').map((c) => [
        c.contact,
        c.margin,
        'challenge ' + c.id,
      ]),
    },
    {
      label: 'decline --- unsupported',
      color: BADC,
      points: CH.filter((c) => verdictOf(c, beta, mu) === 'unsupported').map((c) => [
        c.contact,
        c.margin,
        'challenge ' + c.id,
      ]),
    },
  ]

  return (
    <div className="panel">
      <div className="grid g2">
        <div>
          <ScatterChart
            groups={groups}
            xDomain={[0, 1]}
            yDomain={[0, 0.28]}
            xLabel="contact κ"
            yLabel="margin over runner-up"
            xFmt={(v) => v.toFixed(1)}
            yFmt={(v) => v.toFixed(2)}
            vrules={[{ at: beta, color: 'var(--ink-faint)', label: 'β' }]}
            rules={[{ at: mu, color: 'var(--ink-faint)', label: 'μ' }]}
            h={330}
          />
          <p className="note">
            Each dot is one of the 58 challenges. Only the top-right quadrant is
            licensed &mdash; enough evidence explained, and enough distance to
            the runner-up.
          </p>
        </div>
        <div>
          <Slider
            label="floor β --- minimum contact"
            value={beta}
            min={0}
            max={1}
            step={0.01}
            onChange={setBeta}
            fmt={(v) => v.toFixed(2)}
          />
          <Slider
            label="margin μ --- minimum lead over the runner-up"
            value={mu}
            min={0}
            max={0.28}
            step={0.005}
            onChange={setMu}
            fmt={(v) => v.toFixed(3)}
          />
          <div className="grid g3" style={{ marginTop: 14 }}>
            <Stat value={t.licensed} label="licensed" color="var(--ok)" />
            <Stat value={t.ambiguous} label="ambiguous" color="var(--warn)" />
            <Stat value={t.unsupported} label="unsupported" color="var(--bad)" />
          </div>
          {atPaper ? (
            <Callout>
              These are the paper&rsquo;s own settings, and this is the
              paper&rsquo;s own result: <b>5 licensed, 35 ambiguous, 18
              unsupported</b>. The counts are recomputed here from the
              per-challenge contact and margin, not looked up.
            </Callout>
          ) : (
            <p className="note">
              Moved off the paper&rsquo;s settings (β = 0.30, μ = 0.10, which
              give 5 / 35 / 18). Everything shown is still computed from the
              same 58 measurements.
            </p>
          )}
          <LineChart
            series={[{ label: 'licensed', points: sweep, color: SERIES[0] }]}
            xDomain={[0, 1]}
            yDomain={[0, 20]}
            xLabel="floor β"
            yLabel="challenges licensed"
            xFmt={(v) => v.toFixed(1)}
            yFmt={(v) => String(v)}
            step
            h={220}
            rules={[]}
          />
          <p className="note">
            The sweep is nearly flat across the middle of the range. That is the
            paper&rsquo;s own uncomfortable finding: β = 0.30 was not read off a
            structure in the data, and the yield barely notices where it is put.
          </p>
        </div>
      </div>
    </div>
  )
}

/* ------------------------------------------------------- degeneracy -- */

function DegeneracyDemo() {
  const rows = PANEL.degeneracy_vs_ppm
  const byPpm = {}
  for (const r of rows) {
    if (!byPpm[r.ppm]) byPpm[r.ppm] = []
    byPpm[r.ppm].push(r.n)
  }
  const ppms = Object.keys(byPpm)
    .map(Number)
    .sort((a, b) => a - b)
  const bars = ppms.map((p) => {
    const v = byPpm[p].slice().sort((a, b) => a - b)
    return {
      label: p + ' ppm',
      value: v[Math.floor(v.length / 2)],
      note: 'max ' + v[v.length - 1] + ' over ' + v.length + ' challenges',
    }
  })

  return (
    <div className="panel">
      <BarChart
        bars={bars}
        yDomain={[0, Math.max(...bars.map((b) => b.value)) * 1.2]}
        yLabel="median formula candidates"
        yFmt={(v) => String(Math.round(v))}
        h={260}
      />
      <p className="note">
        Median candidate count at each mass tolerance, over the challenges the
        grid covers. Across the full 58 at the working tolerance the median is
        183 and the maximum 4500 &mdash; and the true formula was uniquely
        determined by mass in <b>0 of 58</b>. Mass alone never identifies
        anything here, which is the premise the rest of the paper is built on.
      </p>
    </div>
  )
}

/* -------------------------------------------------------- controls -- */

function ControlsDemo() {
  const sh = PANEL.shuffle
  const own = sh.map((r) => r.own)
  const foreign = sh.map((r) => r.foreign)
  const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length
  const wins = sh.filter((r) => r.own > r.foreign).length

  const dec = PANEL.decoy
  const decWins = dec.filter((r) => r.top > r.decoy_mean).length

  return (
    <div className="grid g2">
      <div className="panel">
        <b>C2 &mdash; spectra swapped between challenges</b>
        <ScatterChart
          groups={[
            {
              label: 'challenge pairs',
              color: SERIES[0],
              points: sh.map((r) => [r.foreign, r.own, 'own vs foreign spectrum']),
            },
          ]}
          xDomain={[0, 1]}
          yDomain={[0, 1]}
          xLabel="contact with a foreign spectrum"
          yLabel="contact with its own"
          xFmt={(v) => v.toFixed(1)}
          yFmt={(v) => v.toFixed(1)}
          diagonal
          h={300}
        />
        <p className="note">
          Mean contact falls from {mean(own).toFixed(3)} on the right spectrum to{' '}
          {mean(foreign).toFixed(3)} on a foreign one, and the own spectrum wins
          in {wins} of {sh.length} pairs. Points above the diagonal are the wins.
          A method that scored the same either way would be measuring nothing.
        </p>
      </div>
      <div className="panel">
        <b>C1 &mdash; the true candidate against its rivals</b>
        <ScatterChart
          groups={[
            {
              label: 'challenges',
              color: SERIES[2],
              points: dec.map((r) => [r.decoy_mean, r.top, 'n = ' + r.n + ' rivals']),
            },
          ]}
          xDomain={[0, 1]}
          yDomain={[0, 1]}
          xLabel="mean rival contact"
          yLabel="top candidate contact"
          xFmt={(v) => v.toFixed(1)}
          yFmt={(v) => v.toFixed(1)}
          diagonal
          h={300}
        />
        <p className="note">
          The top candidate leads its rivals by +0.1056 on average, in {decWins}{' '}
          of {dec.length} challenges. An earlier version of this control drew its
          rivals by rejection sampling and reported +0.2316; that run is rejected
          in the paper as biased, and the honest figure is the smaller one.
        </p>
      </div>
    </div>
  )
}

/* -------------------------------------------------------- the page -- */

export default function UcDavisCasmi() {
  return (
    <>
      <PaperHead paper={PAPER} />

      <Section
        id="argument"
        kicker="the setting"
        title="58 real challenges, and a method allowed to decline"
        sub={
          'Raw LC-MS/MS from the CASMI challenge set. The question is not how ' +
          'often the method is right --- it is how often it is willing to answer, ' +
          'and whether it declines for the right reasons.'
        }
      >
        <p>
          Every other paper in the catalogue proves that a floor exists. This one
          has to <i>pick</i> one, because it is being run against data rather
          than derived. That difference is the most important thing on this page:
          β = 0.30 here is an empirical choice, not a theorem, and the paper
          reports that the choice sits at noise level rather than at a structure
          in the data.
        </p>

        <h3>Mass alone determines nothing</h3>
        <DegeneracyDemo />

        <h3>The licensing decision, recomputed</h3>
        <p>
          A candidate is licensed when it explains enough of the observed
          evidence (contact <span className="mono">κ &ge; β</span>) and leads the
          runner-up by enough (<span className="mono">margin &ge; μ</span>).
          Anything else is declined, and the two ways of declining are kept
          apart: <i>unsupported</i> means no candidate explained the evidence,{' '}
          <i>ambiguous</i> means more than one did.
        </p>
        <LicensingDemo />

        <h3>Does contact measure anything?</h3>
        <p>
          Three controls, all of which could have come out flat. They did not,
          but the effects are modest and the paper reports the modest numbers
          rather than the flattering ones.
        </p>
        <ControlsDemo />
      </Section>

      <Section
        id="results"
        kicker="the result"
        title="Five answers and fifty-three declines"
        sub="Stated plainly, because the declining is the result rather than the shortfall."
      >
        <div className="grid g4">
          <Stat value="5" label="licensed" color="var(--ok)" />
          <Stat value="35" label="declined --- ambiguous" color="var(--warn)" />
          <Stat value="18" label="declined --- unsupported" color="var(--bad)" />
          <Stat value="0 of 58" label="uniquely determined by mass" />
        </div>
        <Callout>
          A method that answered all 58 would be reporting 53 answers it has no
          evidence for. The point of the floor is that those 53 are not failures
          of the method &mdash; they are the method working. What the yield does
          establish is the price: this is what honest licensing costs on real
          data.
        </Callout>
      </Section>

      <Section
        id="limits"
        kicker="limitations"
        title="What this study does not establish"
      >
        <Callout tone="warn">
          <b>Nothing here is graded against truth.</b> There is no expectation
          register on this page because the study registers no falsifiable
          predictions: it reports verdicts, not accuracy. Whether the five
          licensed answers are <i>correct</i> is a question this study does not
          ask and cannot answer from what it measured.
        </Callout>
        <Callout tone="warn">
          <b>One of the five is carried by a prior, not by contact.</b> In
          challenge 26 the runner-up has contact 0.748 against the winner&rsquo;s
          0.690 &mdash; the margin that licensed it comes from a chemical prior
          applied on top. That is one of five, and it is reported rather than
          quietly dropped.
        </Callout>
        <Callout tone="warn">
          <b>β = 0.30 is chosen, not derived.</b> Sweep it above and the licensed
          count barely responds across most of the range. The paper says so; a
          floor at noise level is a weaker object than the floors the other six
          papers prove.
        </Callout>
        <p className="note">
          A note on notation: this paper writes <span className="mono">κ</span>{' '}
          for contact, an intensity-weighted fraction in{' '}
          <span className="mono">[0, 1]</span>. Two other papers write{' '}
          <span className="mono">κ</span> for the cut key{' '}
          <span className="mono">(σ, δ)</span>. Same letter, unrelated quantity,
          and no result converts between them.
        </p>
      </Section>
    </>
  )
}
