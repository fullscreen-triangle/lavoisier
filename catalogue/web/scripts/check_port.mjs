// =====================================================================
//  The gate.
//
//  Runs the two ladder experiments through the JS runtime and diffs the
//  result against what shapeshifter-py wrote. Doubles are compared with
//  Object.is, not a tolerance: both languages are IEEE-754 binary64 and
//  a literal transcription reproduces every bit, so any difference is a
//  port bug and not a rounding preference.
//
//    node scripts/check_port.mjs
// =====================================================================

import { readFileSync } from 'node:fs'
import { resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { run } from '../src/lib/ss/compiler.js'

const HERE = dirname(fileURLToPath(import.meta.url))
const PY = resolve(HERE, '../../../shapeshifter-py')

let failures = 0
const fail = (what, want, got) => {
  failures += 1
  console.log('  FAIL  ' + what)
  console.log('        want ' + JSON.stringify(want))
  console.log('        got  ' + JSON.stringify(got))
}
const eq = (what, want, got) => {
  if (Object.is(want, got)) console.log('  ok    ' + what + ' = ' + JSON.stringify(got))
  else fail(what, want, got)
}
const eqArr = (what, want, got) => {
  const same =
    Array.isArray(got) && want.length === got.length && want.every((w, i) => Object.is(w, got[i]))
  if (same) console.log('  ok    ' + what + ' = [' + got.join(', ') + ']')
  else fail(what, want, got)
}

// ------------------------------------------------- exp1: executes fully

console.log('exp1_instrument_ladder.ss')
const src1 = readFileSync(resolve(PY, 'experiments/exp1_instrument_ladder.ss'), 'utf8')
const oracle1 = JSON.parse(
  readFileSync(resolve(PY, 'results/exp1_instrument_ladder.json'), 'utf8')
)
const got1 = run(src1)

eq('compile.ok', oracle1.compile.ok, got1.compile.ok)
if (!got1.execute) {
  fail('execute present', 'an execute payload', null)
} else {
  const ws = Object.fromEntries(got1.execute.workspace.map((b) => [b.name, b.value]))

  const ev = ws.evaluation || {}
  eq('evaluation.composite', 0.917125, ev.composite)
  eqArr(
    'evaluation sensitivities',
    [0.20718750000000002, 0.1275, 0.16575, 0.1105, 0.0975],
    (ev.rungs || []).map((r) => r.sensitivity)
  )
  eqArr(
    'evaluation gap_after',
    [0.4, 0.26, 0.13, 0.0975, 0.082875],
    (ev.rungs || []).map((r) => r.gap_after)
  )
  eqArr('sensitivity_rank', ['k1', 'k3', 'k2', 'k4', 'k5'], ev.sensitivity_rank)
  eq('strongest_rung', 'k1', ev.strongest_rung)
  eq('requirement.satisfied', true, ev.requirement && ev.requirement.satisfied)

  const ab = ws.ablation || {}
  eqArr(
    'ablation composites',
    [0.7928124999999999, 0.8725, 0.8342499999999999, 0.8895, 0.9025],
    (ab.ablations || []).map((a) => a.composite)
  )
  eq('ablation k1 loss', 0.12431250000000005, (ab.ablations || [])[0] &&
    ab.ablations[0].loss)
  eq('ablation k1 still_satisfied', false, (ab.ablations || [])[0] &&
    ab.ablations[0].still_satisfied)
  eq('ablation k5 still_satisfied', true, (ab.ablations || [])[4] &&
    ab.ablations[4].still_satisfied)

  const bd = ws.bound || {}
  eq('bound.min_contacts', 3, bd.min_contacts)
  eq('bound.achieved_with_n', 0.9359999999999999, bd.achieved_with_n)

  eq('result.name', oracle1.execute.result.name, got1.execute.result.name)
  eq('result.kind', oracle1.execute.result.kind, got1.execute.result.kind)
  eq(
    'workspace line',
    'evaluation:ladder, ablation:ladder, bound:ladder',
    got1.execute.workspace.map((b) => b.name + ':' + b.kind).join(', ')
  )
  eq('log record count', oracle1.execute.log.length, got1.execute.log.length)

  // The log messages are format-string sensitive; diff them line by line.
  const wantLog = oracle1.execute.log.map((r) => r.level + '|' + r.message)
  const gotLog = got1.execute.log.map((r) => r.level + '|' + r.message)
  for (let i = 0; i < Math.max(wantLog.length, gotLog.length); i++)
    if (wantLog[i] !== gotLog[i]) fail('log[' + i + ']', wantLog[i], gotLog[i])
  if (wantLog.join('\n') === gotLog.join('\n')) console.log('  ok    log matches line for line')
}

// ------------------------- exp1b: rejected statically, never executes

console.log('\nexp1b_unreachable.ss')
const src1b = readFileSync(resolve(PY, 'experiments/exp1b_unreachable.ss'), 'utf8')
const oracle1b = JSON.parse(readFileSync(resolve(PY, 'results/exp1b_unreachable.json'), 'utf8'))
const got1b = run(src1b)

eq('compile.ok', false, got1b.compile.ok)
eq('execute', null, got1b.execute)
const wantErr = (oracle1b.compile.diagnostics || []).filter((d) => d.severity === 'error')
const gotErr = got1b.compile.diagnostics.filter((d) => d.severity === 'error')
eq('error diagnostic count', wantErr.length, gotErr.length)
if (wantErr.length && gotErr.length) eq('error message', wantErr[0].message, gotErr[0].message)

// --------------------------------------------------------------- verdict

console.log('')
if (failures) {
  console.log(failures + ' check(s) FAILED - the port does not reproduce the Python.')
  process.exit(1)
}
console.log('all checks passed - the JS runtime reproduces the Python bit for bit.')
