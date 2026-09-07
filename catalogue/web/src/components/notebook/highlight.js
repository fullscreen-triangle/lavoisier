// =====================================================================
//  Shapeshifter syntax highlighting.
//
//  The reference stylesheet already declares `pre .kw`, `pre .cm` and
//  `pre .st` and never uses them. They are used here. The grammar is
//  small enough that a regex pass is the right size of tool --- CodeMirror
//  would be a megabyte to colour four token classes.
// =====================================================================

const KEYWORDS = [
  'import', 'objective', 'instrument', 'dataset', 'target_list',
  'phase', 'validate', 'ladder', 'toward', 'rung', 'at', 'require',
  'true', 'false',
]

const KW_RE = new RegExp('\\b(' + KEYWORDS.join('|') + ')\\b', 'g')

const escapeHtml = (s) =>
  s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')

/* Highlight one line. Comments win outright: a `//` swallows the rest of
   the line, so it is split off first and the head alone is tokenised.
   Strings are matched before keywords so a keyword inside a quoted value
   stays a string. */
function highlightLine(line) {
  const ci = line.indexOf('//')
  const head = ci >= 0 ? line.slice(0, ci) : line
  const tail = ci >= 0 ? line.slice(ci) : ''

  const parts = []
  let last = 0
  const strRe = /"[^"]*"|'[^']*'/g
  let m
  while ((m = strRe.exec(head)) !== null) {
    parts.push({ text: head.slice(last, m.index), str: false })
    parts.push({ text: m[0], str: true })
    last = m.index + m[0].length
  }
  parts.push({ text: head.slice(last), str: false })

  let outHtml = ''
  for (const p of parts) {
    if (p.str) outHtml += '<span class="st">' + escapeHtml(p.text) + '</span>'
    else
      outHtml += escapeHtml(p.text).replace(KW_RE, '<span class="kw">$1</span>')
  }
  if (tail) outHtml += '<span class="cm">' + escapeHtml(tail) + '</span>'
  return outHtml
}

export function highlight(source) {
  return source.split('\n').map(highlightLine).join('\n')
}
