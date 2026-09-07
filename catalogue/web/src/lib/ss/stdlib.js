// =====================================================================
//  Shapeshifter stdlib --- the ladder algebra, ported from
//  shapeshifter-py/shapeshifter/stdlib.py (lines 435-608).
//
//  Both languages are IEEE-754 binary64, so a literal transcription
//  reproduces every digit of the Python output, including
//  0.20718750000000002 and 0.9359999999999999. Do not round, do not
//  refactor into closed forms: see the comment on sensitivity().
// =====================================================================

export class RefusalError extends Error {
  constructor(message) {
    super(message)
    this.name = 'RefusalError'
  }
}

// Composite resolution of a ladder: what is resolved is the complement
// of the product of what each contact leaves unresolved.
export function compose(powers) {
  let residual = 1.0
  for (const p of powers) residual *= 1.0 - p
  return 1.0 - residual
}

// The ambiguity remaining after each successive contact.
export function residualGap(powers, gap0 = 1.0) {
  const out = []
  let g = Number(gap0)
  for (const p of powers) {
    g *= 1.0 - p
    out.push(g)
  }
  return out
}

/* Sensitivity of the composite to each rung: the residual the ladder
   would carry if that one contact were absent.

   Computed as an explicit skip-one product, NOT as
   (1 - composite) / (1 - p_j). The two agree in exact arithmetic and
   differ in the last ULP in binary64, and the Python takes the loop --- so
   this does too, in the same left-to-right order. */
export function sensitivity(powers) {
  const out = []
  for (let j = 0; j < powers.length; j++) {
    let prod = 1.0
    for (let i = 0; i < powers.length; i++) if (i !== j) prod *= 1.0 - powers[i]
    out.push(prod)
  }
  return out
}

// Fewest equal contacts of strength pow_max that reach the target.
export function minRungs(target, powMax) {
  if (!(powMax > 0.0 && powMax < 1.0)) throw new RefusalError('pow_max must lie in (0,1)')
  if (target >= 1.0)
    throw new RefusalError(
      'composite resolution 1 is unreachable: the floor is strictly positive'
    )
  return Math.ceil(Math.log(1.0 - target) / Math.log(1.0 - powMax))
}

// ------------------------------------------------------- argument forms

/* Three ways a caller can name a ladder, in the Python order:

     1. the name of a `ladder` block --- the ONLY form carrying a require
     2. a bare list of numbers, named "0", "1", ...
     3. a list of {name, power} objects

   Argument resolution in the compiler is one level deep, so `ladder:
   instrument` arrives here as the string "instrument" and hits form 1. */
function powersOf(args, env, ast) {
  const lad = args.has('ladder') ? args.get('ladder') : null
  if (typeof lad === 'string' && ast.ladders.has(lad)) {
    const L = ast.ladders.get(lad)
    return [L.rungs.map((r) => r.power), L.rungs.map((r) => r.name), L]
  }
  const raw = args.has('powers') ? args.get('powers') : lad
  if (Array.isArray(raw) && raw.length) {
    if (typeof raw[0] === 'number')
      return [raw.map(Number), raw.map((_, i) => String(i)), null]
    if (raw[0] && typeof raw[0] === 'object')
      return [
        raw.map((d) => Number(d.power)),
        raw.map((d, i) => String(d.name === undefined ? i : d.name)),
        null,
      ]
  }
  throw new RefusalError('no ladder or powers given')
}

const fmt5 = (x) => x.toFixed(5)

// -------------------------------------------------------------- the ops

export function opLadderResolve(args, env, ast, emit) {
  const [powers, names, L] = powersOf(args, env, ast)
  const comp = compose(powers)
  const sens = sensitivity(powers)
  const gaps = residualGap(powers)

  const rungs = powers.map((p, i) => ({
    name: names[i],
    power: p,
    sensitivity: sens[i],
    gap_after: gaps[i],
  }))

  const idx = powers.map((_, i) => i)
  const rank = idx.slice().sort((a, b) => -sens[a] - -sens[b]).map((j) => names[j])
  let strongest = 0
  for (let j = 1; j < powers.length; j++) if (powers[j] > powers[strongest]) strongest = j

  const req = L && L.require ? L.require : args.has('require') ? args.get('require') : null
  let requirement = null
  if (req) {
    const metric = req.metric
    const value = Number(req.value)
    const achievedMap = { resolution: comp, contacts: powers.length * 1.0 }
    if (!(metric in achievedMap))
      throw new RefusalError('unknown requirement metric ' + JSON.stringify(metric))
    const achieved = achievedMap[metric]
    const ok = compare(achieved, req.op, value)
    requirement = { metric, op: req.op, value, achieved, satisfied: ok }
    emit(
      ok ? 'info' : 'warn',
      '  require ' + metric + ' ' + req.op + ' ' + value + ': achieved ' +
        fmt5(achieved) + ' -> ' + (ok ? 'PASS' : 'FAIL')
    )
  }

  emit('info', '  composite resolution ' + fmt5(comp) + ' over ' + powers.length + ' contact(s)')

  return [
    {
      rungs,
      composite: comp,
      contacts: powers.length,
      cost: powers.length,
      sensitivity_rank: rank,
      strongest_rung: names[strongest],
      requirement,
    },
    'ladder',
  ]
}

export function opLadderAblate(args, env, ast, emit) {
  const [powers, names, L] = powersOf(args, env, ast)
  const full = compose(powers)
  const req = L && L.require ? L.require : args.has('require') ? args.get('require') : null

  const ablations = []
  for (let j = 0; j < powers.length; j++) {
    const kept = powers.filter((_, i) => i !== j)
    const c = compose(kept)
    const row = { dropped: names[j], power: powers[j], composite: c, loss: full - c }
    let tail = ''
    if (req) {
      // The Python compares against req.value without consulting
      // req.metric --- preserved.
      const ok = compare(c, req.op, Number(req.value))
      row.still_satisfied = ok
      tail = ok ? ' (still meets requirement)' : ' (fails requirement)'
    }
    emit('info', '  without ' + names[j] + ': ' + fmt5(c) + tail)
    ablations.push(row)
  }

  return [{ full_composite: full, ablations }, 'ladder']
}

export function opLadderMinimum(args, env, ast, emit) {
  const target = args.has('target') ? Number(args.get('target')) : 0.9
  const powMax = args.has('pow_max') ? Number(args.get('pow_max')) : 0.6
  const n = minRungs(target, powMax)
  emit(
    'info',
    '  target ' + target + ' at pow_max ' + powMax + ': minimum ' + n + ' contact(s)'
  )
  return [
    {
      target,
      pow_max: powMax,
      min_contacts: n,
      achieved_with_n: compose(new Array(n).fill(powMax)),
    },
    'ladder',
  ]
}

export function compare(a, op, b) {
  if (op === '>=') return a >= b
  if (op === '<=') return a <= b
  if (op === '==') return a === b
  if (op === '>') return a > b
  if (op === '<') return a < b
  throw new RefusalError('unknown comparison ' + JSON.stringify(op))
}
