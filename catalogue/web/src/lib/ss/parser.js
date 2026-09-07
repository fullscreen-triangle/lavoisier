// =====================================================================
//  Shapeshifter parser --- a literal transcription of
//  shapeshifter-py/shapeshifter/parser.py
//
//  This is a PORT, not a reimplementation. Where the Python does
//  something surprising, so does this file, and a comment says why.
//  The oracle is shapeshifter-py/results/*.json; scripts/check_port.mjs
//  diffs against it.
// =====================================================================

const OPEN = '[({'
const CLOSE = '])}'
const QUOTES = ['"', "'"]

export const FIELD_BLOCKS = ['objective', 'instrument', 'dataset', 'target_list']
export const STMT_BLOCKS = ['phase', 'validate']

export class ParseError extends Error {
  constructor(message) {
    super(message)
    this.name = 'ParseError'
  }
}

// ------------------------------------------------------------- values

/* Split on commas at bracket depth zero.

   The Python counts brackets and ignores quotes entirely, so a comma
   inside a string literal at depth zero splits the value. That is a bug,
   and it is preserved here: a port that fixed it would accept programs
   the reference implementation rejects. */
export function splitDepth0(raw) {
  const parts = []
  let depth = 0
  let start = 0
  for (let i = 0; i < raw.length; i++) {
    const c = raw[i]
    if (OPEN.includes(c)) depth += 1
    else if (CLOSE.includes(c)) depth -= 1
    else if (c === ',' && depth === 0) {
      parts.push(raw.slice(start, i))
      start = i + 1
    }
  }
  parts.push(raw.slice(start))
  return parts.filter((p) => p.trim())
}

function parseObject(raw) {
  let inner = raw.trim()
  if (inner.startsWith('{')) inner = inner.slice(1)
  if (inner.endsWith('}')) inner = inner.slice(0, -1)
  const obj = {}
  for (const pair of splitDepth0(inner)) {
    const ci = pair.indexOf(':')
    if (ci >= 0) obj[pair.slice(0, ci).trim()] = parseValue(pair.slice(ci + 1).trim())
  }
  return obj
}

function parseObjectArray(raw) {
  const objs = []
  let depth = 0
  let start = 0
  for (let i = 0; i < raw.length; i++) {
    const c = raw[i]
    if (c === '{') {
      if (depth === 0) start = i
      depth += 1
    } else if (c === '}') {
      depth -= 1
      if (depth === 0) objs.push(parseObject(raw.slice(start, i + 1)))
    }
  }
  return objs
}

/* Python float() and JS parseFloat() disagree at both ends: float() takes
   "nan" / "inf" / "1_000", parseFloat takes "1abc". Neither loose form is
   admitted, so this regex is the intersection --- what a Python float
   literal looks like and nothing else. Anything outside it falls through
   to V7 and stays a bare word, exactly as the Python does. */
const FLOAT_RE = /^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$/
const INT_RE = /^[+-]?\d+$/

// Value parsing, clauses V1-V7 in the Python order.
export function parseValue(raw) {
  if (raw === null || raw === undefined) return null
  const s = String(raw).trim()
  if (!s) return null                                              // V1
  if (s.length >= 2 && s[0] === s[s.length - 1] && QUOTES.includes(s[0]))
    return s.slice(1, -1)                                          // V2 no escapes
  if (s === 'true') return true                                    // V3
  if (s === 'false') return false
  if (s.startsWith('[') && s.endsWith(']')) {                      // V4
    const inner = s.slice(1, -1).trim()
    if (!inner) return []
    if (inner.startsWith('{')) return parseObjectArray(inner)
    return splitDepth0(inner).map(parseValue)
  }
  if (s.startsWith('{') && s.endsWith('}')) return parseObject(s)   // V5
  if (INT_RE.test(s)) return parseInt(s, 10)                       // V6
  if (FLOAT_RE.test(s)) {
    const f = Number(s)
    if (Number.isFinite(f)) return f
  }
  return s                                                         // V7 bare word
}

// ------------------------------------------------------------- lines

function balance(s) {
  let b = 0
  for (const c of s) {
    if (OPEN.includes(c)) b += 1
    else if (CLOSE.includes(c)) b -= 1
  }
  return b
}

/* Strip comments, drop vacuous lines, join by bracket balance.

   Order matters: the comment is removed BEFORE the vacuity test, so a
   comment-only line disappears entirely. The indent is measured on the
   ORIGINAL raw line, not on the stripped body. A continued line keeps the
   first line number and the first indent, and the pieces join with a
   single space. */
function logicalLines(source) {
  const raw = []
  const src = source.split('\n')
  for (let i = 0; i < src.length; i++) {
    const ln = src[i].replace(/\r$/, '')
    const body = ln.replace(/\/\/.*$/, '').trim()
    if (body) raw.push({ num: i + 1, indent: ln.length - ln.replace(/^\s+/, '').length, body })
  }

  const out = []
  let i = 0
  while (i < raw.length) {
    const cur = raw[i]
    let b = balance(cur.body)
    if (b > 0) {
      let combined = cur.body
      let j = i + 1
      while (j < raw.length && b > 0) {
        combined += ' ' + raw[j].body
        b += balance(raw[j].body)
        j += 1
      }
      out.push({ num: cur.num, indent: cur.indent, body: combined })
      i = j
    } else {
      out.push(cur)
      i += 1
    }
  }
  return out
}

function readFields(lines, i, base) {
  const out = new Map()
  while (i < lines.length && lines[i].indent > base) {
    const m = /^(\w+)\s*:\s*([\s\S]*)$/.exec(lines[i].body)
    if (m) out.set(m[1], parseValue(m[2]))
    i += 1
  }
  return [out, i]
}

// Every argument is named; positional arguments cannot be expressed.
function readArgs(raw) {
  const args = new Map()
  for (const part of splitDepth0(raw)) {
    const ci = part.indexOf(':')
    if (ci >= 0) args.set(part.slice(0, ci).trim(), parseValue(part.slice(ci + 1).trim()))
  }
  return args
}

const RUNG_RE = /^rung\s+(\w+)\s+at\s+([-+0-9.eE]+)$/
const REQUIRE_RE = /^require\s+(\w+)\s*(>=|<=|==|>|<)\s*([-+0-9.eE]+)$/

/* Read a ladder block. Two statement forms only:

       rung <name> at <power>
       require <metric> <op> <value>

   A rung power is a resolution increment and is hard-validated into
   [0,1): a rung of resolution 1 would collapse the ambiguity to the floor
   in a single contact, which the non-completability axiom excludes.

   Ladders are the ONLY strict block --- an unrecognised statement raises.
   Every other block silently drops what it cannot match. */
function readLadderBody(lines, i, base, name, toward) {
  const lad = { name, toward: toward === undefined ? null : toward, rungs: [], require: null }
  while (i < lines.length && lines[i].indent > base) {
    const t = lines[i].body
    let m = RUNG_RE.exec(t)
    if (m) {
      const power = Number(m[2])
      if (!(power >= 0.0 && power < 1.0))
        throw new ParseError(
          'line ' + lines[i].num + ': rung ' + JSON.stringify(m[1]) +
            ' has resolution ' + power + ', outside [0,1)'
        )
      lad.rungs.push({ name: m[1], power })
      i += 1
      continue
    }
    m = REQUIRE_RE.exec(t)
    if (m) {
      // A second require overwrites the first; only one survives.
      lad.require = { metric: m[1], op: m[2], value: Number(m[3]) }
      i += 1
      continue
    }
    throw new ParseError(
      'line ' + lines[i].num + ': unrecognised ladder statement ' + JSON.stringify(t)
    )
  }
  return [lad, i]
}

function readStatements(lines, i, base) {
  const out = []
  while (i < lines.length && lines[i].indent > base) {
    const t = lines[i].body
    const m = /^(\w+)\s*=\s*([\s\S]+)$/.exec(t)
    if (m) {
      const target = m[1]
      const expr = m[2].trim()
      const c = /^([\w.]+)\s*\(([\s\S]*)\)$/.exec(expr)
      if (c)
        out.push({
          type: 'assign', target, isCall: true,
          call: { fn: c[1], args: readArgs(c[2]) },
        })
      else out.push({ type: 'assign', target, isCall: false, value: parseValue(expr) })
    } else {
      const c = /^([\w.]+)\s*\(([\s\S]*)\)$/.exec(t)
      // A bare call parses and is audited at compile time, but
      // execute_stage skips it: only assignments ever run. Preserved.
      if (c) out.push({ type: 'call', fn: c[1], args: readArgs(c[2]) })
    }
    i += 1
  }
  return [out, i]
}

// ------------------------------------------------------------- parse

/* Blocks live in Maps keyed by name: a redeclaration overwrites, and
   insertion order is execution order. */
export function parse(source) {
  const lines = logicalLines(source)
  const ast = {
    imports: [],
    objective: null,
    instruments: new Map(),
    datasets: new Map(),
    target_lists: new Map(),
    phases: new Map(),
    validates: new Map(),
    ladders: new Map(),
  }
  let i = 0
  while (i < lines.length) {
    const ln = lines[i]

    if (ln.body.startsWith('import ')) {
      ast.imports.push(ln.body.slice(7).trim())
      i += 1
      continue
    }

    // The ladder form is tested BEFORE the generic block form, and it is
    // the only block whose trailing colon is optional.
    const lm = /^ladder\s+(\w+)(?:\s+toward\s+(\w+))?\s*:?$/.exec(ln.body)
    if (lm) {
      const [lad, ni] = readLadderBody(lines, i + 1, ln.indent, lm[1], lm[2])
      ast.ladders.set(lad.name, lad)
      i = ni
      continue
    }

    const m = /^(\w+)\s+(\w+)\s*:$/.exec(ln.body)
    if (m && FIELD_BLOCKS.includes(m[1])) {
      const kw = m[1]
      const name = m[2]
      const [body, ni] = readFields(lines, i + 1, ln.indent)
      i = ni
      if (kw === 'objective') ast.objective = { name, fields: body }
      else if (kw === 'instrument') ast.instruments.set(name, body)
      else if (kw === 'dataset') ast.datasets.set(name, body)
      else {
        // target_list merges its own name INTO the field map; objective
        // keeps the name outside, under a separate key.
        const merged = new Map([['name', name]])
        for (const [k, v] of body) merged.set(k, v)
        ast.target_lists.set(name, merged)
      }
      continue
    }

    if (m && STMT_BLOCKS.includes(m[1])) {
      const kw = m[1]
      const name = m[2]
      const [body, ni] = readStatements(lines, i + 1, ln.indent)
      i = ni
      ;(kw === 'phase' ? ast.phases : ast.validates).set(name, body)
      continue
    }

    i += 1 // anything unmatched is silently skipped
  }
  return ast
}

const mapToObj = (m) => Object.fromEntries(m)

// Mirrors AST.to_dict(); this shape is what becomes compile.ir.
export function astToDict(ast) {
  const stmt = (s) => {
    if (s.type === 'assign') {
      const v = s.isCall
        ? { type: 'call', fn: s.call.fn, args: mapToObj(s.call.args) }
        : { type: 'value', value: s.value }
      return { type: 'assign', target: s.target, value: v }
    }
    return { type: 'call', fn: s.fn, args: mapToObj(s.args) }
  }
  const stmtMap = (m) => Object.fromEntries([...m].map(([k, v]) => [k, v.map(stmt)]))

  return {
    imports: ast.imports,
    objective: ast.objective
      ? { name: ast.objective.name, fields: mapToObj(ast.objective.fields) }
      : null,
    instruments: Object.fromEntries([...ast.instruments].map(([k, v]) => [k, mapToObj(v)])),
    datasets: Object.fromEntries([...ast.datasets].map(([k, v]) => [k, mapToObj(v)])),
    target_lists: Object.fromEntries([...ast.target_lists].map(([k, v]) => [k, mapToObj(v)])),
    phases: stmtMap(ast.phases),
    validates: stmtMap(ast.validates),
    ladders: Object.fromEntries(
      [...ast.ladders].map(([k, L]) => [
        k,
        {
          name: L.name,
          toward: L.toward,
          rungs: L.rungs.map((r) => ({ name: r.name, power: r.power })),
          require: L.require,
        },
      ])
    ),
  }
}
