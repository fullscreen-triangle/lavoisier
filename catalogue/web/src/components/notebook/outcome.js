// =====================================================================
//  What happened when the cell ran.
//
//  Five outcomes, and every one of them is a RESULT --- including the
//  three that produced no value. That is the corpus stance: a program
//  refused before it ran has told you something, and a cell that renders
//  it as a blank pane would be throwing that finding away.
//
//  Following the reference implementation, the constructor THROWS on a
//  degenerate outcome rather than trusting convention: a blocked outcome
//  with no unblock line is a bug in the caller, not a display state.
// =====================================================================

export const OUTCOMES = {
  ok: {
    tone: 'ok',
    label: 'ran',
    gloss: 'The program compiled and executed; every number below was computed here.',
  },
  replayed: {
    tone: 'warn',
    label: 'replayed',
    gloss:
      'The program ran, but at least one operation read a recorded result instead of computing it.',
  },
  'parse-error': {
    tone: 'bad',
    label: 'parse error',
    gloss: 'The source could not be read. Nothing was compiled and nothing ran.',
  },
  'compile-rejected': {
    tone: 'bad',
    label: 'rejected',
    gloss:
      'The program was rejected by the compiler before execution, because no run could satisfy what it asks for.',
  },
  refused: {
    tone: 'warn',
    label: 'refused',
    gloss:
      'An operation declined to answer. A refusal is a finding about the question, not a failure of the machine.',
  },
}

function makeOutcome(kind, extra) {
  const spec = OUTCOMES[kind]
  if (!spec) throw new Error('unknown outcome kind: ' + kind)
  const o = { kind, ...spec, ...extra }
  // A blocked outcome must say both what stopped it and what would
  // unblock it. Half of that pair is not an explanation.
  if (kind !== 'ok' && kind !== 'replayed') {
    if (!o.blocker) throw new Error(kind + ' outcome carries no blocker')
    if (!o.unblock) throw new Error(kind + ' outcome carries no unblock')
  }
  return o
}

const REPLAY_HINT =
  'Recomputing it needs the multi-megabyte source library and the Python ' +
  'implementation; run the experiment under shapeshifter-py to get a fresh number.'

// Classify a payload from ss/compiler.run().
export function classify(payload) {
  const diags = payload.compile.diagnostics || []
  const errs = diags.filter((d) => d.severity === 'error')

  if (!payload.compile.ok) {
    const msg = errs.length ? errs[0].message : 'compilation failed'
    if (/^line \d+:/.test(msg))
      return makeOutcome('parse-error', {
        blocker: msg,
        unblock:
          'Fix the line the message names. The parser is strict about ladder ' +
          'statements only: a rung takes a resolution in [0,1), and the only ' +
          'other statement a ladder admits is a require clause.',
      })
    return makeOutcome('compile-rejected', {
      blocker: msg,
      unblock:
        'Either lower the requirement, or add contacts. Composite resolution ' +
        'is 1 - the product of the residuals, so it approaches 1 from below ' +
        'and never reaches it: no finite ladder satisfies a requirement of 1.',
    })
  }

  if (payload.refused)
    return makeOutcome('refused', {
      blocker: payload.refused,
      unblock:
        'Ask a question the operation can answer. If the message names an ' +
        'argument and the value a fixture holds, setting it back replays; ' +
        REPLAY_HINT,
    })

  const ws = payload.execute ? payload.execute.workspace : []
  const replayedOps = ws
    .filter((b) => b.value && typeof b.value === 'object' && b.value._replayed)
    .map((b) => b.value._op)
  if (replayedOps.length)
    return makeOutcome('replayed', {
      replayedOps: [...new Set(replayedOps)],
      note:
        'Read from a recorded run, not computed in your browser: ' +
        [...new Set(replayedOps)].join(', ') + '. ' + REPLAY_HINT,
    })

  return makeOutcome('ok', {})
}
