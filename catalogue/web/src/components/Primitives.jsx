// =====================================================================
//  Small shared pieces, following the reference implementation.
//
//  These are the only components in the app that take props for
//  presentation. Everything with behaviour owns its own state.
// =====================================================================

export const VERDICTS = {
  PASS: { label: 'pass', tone: 'ok', gloss: 'Every graded expectation was met.' },
  FAIL: {
    label: 'fail',
    tone: 'bad',
    gloss: 'At least one registered prediction was not met, and is reported as such.',
  },
  ok: { label: 'ok', tone: 'ok' },
  computed: {
    label: 'computed here',
    tone: 'ok',
    gloss: 'Recomputed in your browser from the definitions on this page.',
  },
  measured: {
    label: 'measured',
    tone: 'neutral',
    gloss: 'Read from a shipped result file; it cannot be recomputed client-side.',
  },
  replayed: {
    label: 'replayed',
    tone: 'warn',
    gloss: 'Read from a recorded run rather than computed here.',
  },
  refuted: {
    label: 'refuted',
    tone: 'bad',
    gloss: 'The paper states this claim and its own experiment contradicts it.',
  },
  defect: {
    label: 'defect',
    tone: 'warn',
    gloss: 'A statement in the paper that is wrong as written, reported here unsoftened.',
  },
}

export function Pill({ verdict }) {
  const v = VERDICTS[verdict] || { label: String(verdict), tone: 'neutral' }
  return (
    <span className={'pill ' + v.tone} title={v.gloss}>
      {v.label}
    </span>
  )
}

export function Stat({ value, label, color }) {
  return (
    <div className="stat">
      <div className="v" style={color ? { color } : undefined}>
        {value}
      </div>
      <div className="k">{label}</div>
    </div>
  )
}

export function Section({ id, kicker, title, sub, children }) {
  return (
    <section id={id}>
      <div className="wrap">
        {kicker ? <div className="kicker">{kicker}</div> : null}
        <h2>{title}</h2>
        {sub ? <p className="sub">{sub}</p> : null}
        {children}
      </div>
    </section>
  )
}

// Returns a fragment, and hands onChange a NUMBER, never an event.
export function Slider({ label, value, min, max, step, onChange, fmt }) {
  return (
    <>
      <label className="ctl">
        {label} <b>{fmt ? fmt(value) : value}</b>
      </label>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
      />
    </>
  )
}

export function Callout({ tone, children }) {
  return <div className={tone === 'warn' ? 'callout warn' : 'callout'}>{children}</div>
}
