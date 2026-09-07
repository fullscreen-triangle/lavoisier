// =====================================================================
//  Shapeshifter compiler and interpreter --- ported from
//  shapeshifter-py/shapeshifter/compiler.py
//
//  Two stages. compileStage audits the program without opening a single
//  file and can REJECT it; executeStage then runs what survived. The
//  static reachability check is the interesting half: a ladder whose
//  rungs cannot reach its own requirement is refused before anything
//  runs, which is what makes exp1b fail with execute === null.
// =====================================================================

import { parse, astToDict, ParseError } from './parser.js'
import {
  RefusalError,
  opLadderResolve,
  opLadderAblate,
  opLadderMinimum,
  compare,
} from './stdlib.js'
import { REPLAY_OPS } from './stdlib-replay.js'

export { ParseError, RefusalError }

const PRIMARY_PREFERENCE = [
  'records', 'cells', 'addresses', 'coords',
  'coherence', 'features', 'scans', 'linkage',
]

// Operations that genuinely run in the browser. Everything else lives in
// REPLAY_OPS, reads a shipped fixture, and says so in the cell.
export const REGISTRY = {
  'lavoisier.ladder.resolve': { fn: opLadderResolve, effects: ['pure'], inputs: [] },
  'lavoisier.ladder.ablate': { fn: opLadderAblate, effects: ['pure'], inputs: [] },
  'lavoisier.ladder.minimum': { fn: opLadderMinimum, effects: ['pure'], inputs: [] },
  ...REPLAY_OPS,
}

const diag = (severity, message) => ({ severity, message })
const out = (stream, text) => ({ stream, text })

// Structural kind of a binding, used for the workspace listing.
function classify(fn, value) {
  if (fn !== null && fn !== undefined) return 'object'
  if (Array.isArray(value)) return 'list'
  if (value && typeof value === 'object') return 'object'
  return 'scalar'
}

// ------------------------------------------------------------ compile

export function compileStage(program) {
  const t0 = performance.now()
  const term = [out('stage', 'shapeshifter compile')]
  const diags = []

  let ast
  try {
    ast = parse(program)
  } catch (e) {
    if (!(e instanceof ParseError)) throw e
    term.push(out('stderr', 'error: parse failed - ' + e.message))
    diags.push(diag('error', e.message))
    return { ok: false, ast: null, ir: {}, terminal: term, diagnostics: diags }
  }

  // The block listing order is fixed here and is NOT source order.
  const blocks = []
  if (ast.objective) blocks.push('objective ' + ast.objective.name)
  for (const n of ast.instruments.keys()) blocks.push('instrument ' + n)
  for (const n of ast.datasets.keys()) blocks.push('dataset ' + n)
  for (const n of ast.target_lists.keys()) blocks.push('target_list ' + n)
  for (const n of ast.ladders.keys()) blocks.push('ladder ' + n)
  for (const n of ast.validates.keys()) blocks.push('validate ' + n)
  for (const n of ast.phases.keys()) blocks.push('phase ' + n)

  term.push(
    out('stdout', 'parsed ' + ast.imports.length + ' import(s), ' + blocks.length + ' block(s)')
  )
  for (const b of blocks) term.push(out('stdout', '  . ' + b))

  if (!ast.objective) diags.push(diag('warning', 'no objective block'))
  if (ast.phases.size === 0) {
    const m = 'no phase block - nothing will execute'
    diags.push(diag('warning', m))
    term.push(out('stderr', 'warning: ' + m))
  }

  // Effect and input audit. No file is opened here, by design: the point
  // is to state what the program WOULD touch before it touches anything.
  const effects = new Set()
  const inputs = new Set()
  for (const stmts of [...ast.validates.values(), ...ast.phases.values()]) {
    for (const s of stmts) {
      const call = s.type === 'call' ? s : s.isCall ? s.call : null
      if (!call) continue
      const spec = REGISTRY[call.fn]
      if (!spec) {
        const m = 'unknown operation ' + JSON.stringify(call.fn)
        diags.push(diag('warning', m))
        term.push(out('stderr', 'warning: ' + m))
        continue
      }
      for (const e of spec.effects) effects.add(e)
      const ds = call.args.get('dataset')
      if (typeof ds === 'string' && ast.datasets.has(ds)) {
        const files = ast.datasets.get(ds).get('files')
        if (Array.isArray(files)) for (const f of files) inputs.add(String(f))
      }
    }
  }

  /* Static reachability. A ladder states a requirement; its rungs fix
     the best composite it can ever reach. If the requirement is out of
     reach the program is rejected HERE, before execution, because no run
     could satisfy it. Only the resolution metric is decidable this way. */
  for (const [name, L] of ast.ladders) {
    const req = L.require
    if (!req || req.metric !== 'resolution') continue
    let residual = 1.0
    for (const r of L.rungs) residual *= 1.0 - r.power
    const achieved = 1.0 - residual
    const msg =
      'ladder ' + name + ': ' + L.rungs.length + ' contact(s) give resolution ' +
      achieved.toFixed(5) + ', requirement is ' + req.op + ' ' + req.value
    if (compare(achieved, req.op, Number(req.value))) {
      term.push(out('stdout', '  . ' + msg + ' -> reachable'))
    } else {
      diags.push(diag('error', msg + ' -> UNREACHABLE'))
      term.push(out('stderr', 'error: ' + msg + ' -> UNREACHABLE'))
    }
  }

  term.push(out('stdout', 'effects: ' + [...effects].join(', ')))
  term.push(out('stdout', 'inputs (' + inputs.size + ', not opened): ' + [...inputs].join(', ')))
  term.push(out('stdout', 'compiled in ' + (performance.now() - t0).toFixed(1) + ' ms'))

  const ir = astToDict(ast)
  ir._audit = { effects: [...effects], inputs: [...inputs] }

  const ok = !diags.some((d) => d.severity === 'error')
  return { ok, ast, ir, terminal: term, diagnostics: diags }
}

// ------------------------------------------------------------ execute

/* Argument resolution is ONE LEVEL DEEP: a bare string naming a binding
   is replaced by that binding, and nothing else is. List elements are
   not resolved. This is how `ladder: instrument` survives as the string
   "instrument" when no binding of that name exists, and so reaches the
   stdlib as a reference to the ladder block of that name. */
function resolveArg(env, v) {
  if (typeof v === 'string' && Object.prototype.hasOwnProperty.call(env, v)) return env[v]
  return v
}

export function executeStage(ast) {
  const t0 = performance.now()
  const env = {}
  const kinds = {}
  const order = []
  const log = []
  const emit = (level, message) => log.push({ level, message })

  if (ast.objective) {
    emit('info', 'Objective: ' + ast.objective.name)
    const t = ast.objective.fields.get('target')
    if (t !== undefined && t !== null) emit('info', '  ' + t)
  }

  // One flat namespace, no scoping: a later phase sees everything an
  // earlier one bound.
  const runStmts = (label, stmts) => {
    emit('info', 'Phase: ' + label)
    for (const s of stmts) {
      // Bare calls parse and are audited, but never execute.
      if (s.type !== 'assign') continue
      if (!s.isCall) {
        env[s.target] = s.value
        kinds[s.target] = classify(null, s.value)
        if (!order.includes(s.target)) order.push(s.target)
        continue
      }
      const call = s.call
      const spec = REGISTRY[call.fn]
      if (!spec) {
        emit('error', '  unknown operation ' + JSON.stringify(call.fn))
        throw new RefusalError('unknown operation ' + JSON.stringify(call.fn))
      }
      const args = new Map()
      for (const [k, v] of call.args) args.set(k, resolveArg(env, v))
      // The announcement fires BEFORE the call, so a refusal is preceded
      // by the line naming what refused.
      emit('info', '  ' + s.target + ' = ' + call.fn + '(...)')
      const [value, kind] = spec.fn(args, env, ast, emit)
      env[s.target] = value
      kinds[s.target] = kind
      if (!order.includes(s.target)) order.push(s.target)
    }
  }

  // All validates first, then all phases, each in insertion order.
  for (const [name, stmts] of ast.validates) runStmts(name, stmts)
  for (const [name, stmts] of ast.phases) runStmts(name, stmts)

  // Not-null rather than truthiness: 0, "" and [] are real bindings.
  const workspace = order
    .filter((n) => env[n] !== undefined && env[n] !== null)
    .map((n) => ({ name: n, kind: kinds[n] || 'scalar', value: env[n] }))

  // The first binding whose kind is preferred; failing that, the LAST
  // binding. "ladder" is not in the preference list, which is why exp1
  // reports `bound` rather than `evaluation`.
  let primary = null
  for (const pref of PRIMARY_PREFERENCE) {
    const hit = workspace.find((b) => b.kind === pref)
    if (hit) {
      primary = hit
      break
    }
  }
  if (!primary && workspace.length) primary = workspace[workspace.length - 1]

  const term = log.map((r) =>
    out(r.level === 'error' || r.level === 'warn' ? 'stderr' : 'stdout', r.message)
  )
  term.push(out('stdout', 'workspace: ' + workspace.map((b) => b.name + ':' + b.kind).join(', ')))
  const dt = (performance.now() - t0) / 1000
  term.push(out('stdout', 'finished in ' + (dt * 1000).toFixed(1) + ' ms'))

  return {
    elapsed_s: dt,
    terminal: term,
    log,
    result: primary ? { kind: primary.kind, name: primary.name, data: primary.value } : null,
    workspace,
  }
}

function jsonable(v, maxItems = 2000) {
  if (Array.isArray(v)) {
    if (v.length > maxItems)
      return {
        _truncated: true,
        _n: v.length,
        _head: v.slice(0, maxItems).map((x) => jsonable(x, maxItems)),
      }
    return v.map((x) => jsonable(x, maxItems))
  }
  if (v && typeof v === 'object')
    return Object.fromEntries(Object.entries(v).map(([k, x]) => [k, jsonable(x, maxItems)]))
  return v
}

/* One deliberate divergence from the Python: there a RefusalError
   escapes executeStage uncaught and kills the process. In a notebook
   cell a refusal is a RESULT, not a crash --- that is the stance the
   papers themselves take about declining --- so it is caught here and
   rendered as a terminal error line plus a flag the UI turns into a
   pill. */
export function run(program) {
  const c = compileStage(program)
  const payload = {
    program,
    compile: { ok: c.ok, terminal: c.terminal, diagnostics: c.diagnostics, ir: c.ir },
    execute: null,
    refused: null,
  }
  if (!c.ok || !c.ast) return payload

  let ex
  try {
    ex = executeStage(c.ast)
  } catch (e) {
    if (!(e instanceof RefusalError)) throw e
    payload.refused = e.message
    payload.execute = {
      elapsed_s: 0,
      terminal: [out('stderr', 'error: refused - ' + e.message)],
      log: [{ level: 'error', message: e.message }],
      result: null,
      workspace: [],
    }
    return payload
  }

  payload.execute = {
    elapsed_s: ex.elapsed_s,
    terminal: ex.terminal,
    log: ex.log,
    // Python discards `data` at this boundary; the browser keeps it,
    // because the charts drawn below the cell read from it.
    result: ex.result
      ? { kind: ex.result.kind, name: ex.result.name, data: jsonable(ex.result.data) }
      : null,
    workspace: ex.workspace.map((b) => ({ name: b.name, kind: b.kind, value: jsonable(b.value) })),
  }
  return payload
}
