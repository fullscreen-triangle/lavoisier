// =====================================================================
//  Copy the shipped measurements into src/data and record what they
//  are.
//
//  Two rules this script exists to enforce:
//
//    1. Nothing is transcribed. Every number the site shows that it did
//       not compute is copied byte-for-byte from the file the validation
//       suite wrote, and the manifest records that file's path and its
//       SHA-256 so the copy can be checked against the original.
//
//    2. One file is too large to ship whole. It is downsampled, and the
//       manifest says so, with the original and reduced record counts,
//       rather than quietly shipping a smaller file that looks complete.
// =====================================================================

import { createHash } from 'node:crypto'
import { mkdirSync, readFileSync, writeFileSync, existsSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const HERE = dirname(fileURLToPath(import.meta.url))
const WEB = resolve(HERE, '..')
const CATALOGUE = resolve(WEB, '..')
/* Output goes to src/data, not public/: these files are IMPORTED by the
   pages, so Vite must see them as modules to bundle and hash. Anything
   in public/ is copied verbatim and would have to be fetched at
   runtime, which would make every page async for no gain. */
const OUT = join(WEB, 'src', 'data')

const sha = (buf) => createHash('sha256').update(buf).digest('hex')

/* Files copied verbatim, byte for byte. */
const VERBATIM = [
  ['validation/results/exp1_instrument_ladder.json', 'exp1_instrument_ladder.json'],
  ['validation/results/exp2_observation_groups.json', 'exp2_observation_groups.json'],
  ['validation/results/exp3_coordinate_provenance.json', 'exp3_coordinate_provenance.json'],
  ['validation/results/exp4_runtime_graph.json', 'exp4_runtime_graph.json'],
  ['validation/results/exp5_sink_detection.json', 'exp5_sink_detection.json'],
  ['validation/results/exp6_peptide_mass_invariance.json', 'exp6_peptide_mass_invariance.json'],
  ['validation/results/exp4_sweeps.json', 'exp4_sweeps.json'],
  ['validation/results/exp6_sweeps.json', 'exp6_sweeps.json'],
  ['casmi/panel_data.json', 'panel_data.json'],
]

// exp5_sweeps.json is ~1.5 MB. Its long arrays are downsampled by
// stride; every other key is copied untouched.
const DOWNSAMPLE = ['validation/results/exp5_sweeps.json', 'exp5_sweeps.json']
const KEEP_AT_MOST = 400

function stride(arr, n) {
  if (arr.length <= n) return arr
  const step = arr.length / n
  const out = []
  for (let i = 0; i < n; i++) out.push(arr[Math.floor(i * step)])
  // Always keep the true last element: an extremum at the end of a sweep
  // is exactly the point a stride is most likely to drop.
  if (out[out.length - 1] !== arr[arr.length - 1]) out.push(arr[arr.length - 1])
  return out
}

function reduce(v, trail, log) {
  if (Array.isArray(v)) {
    if (v.length > KEEP_AT_MOST) {
      const cut = stride(v, KEEP_AT_MOST)
      log.push({ path: trail, from: v.length, to: cut.length })
      return cut.map((e) => reduce(e, trail + '[]', log))
    }
    return v.map((e) => reduce(e, trail + '[]', log))
  }
  if (v && typeof v === 'object') {
    const o = {}
    for (const k of Object.keys(v)) o[k] = reduce(v[k], trail ? trail + '.' + k : k, log)
    return o
  }
  return v
}

mkdirSync(OUT, { recursive: true })

const manifest = { generated: new Date().toISOString(), files: [] }
const missing = []

for (const [src, dst] of VERBATIM) {
  const from = join(CATALOGUE, src)
  if (!existsSync(from)) {
    missing.push(src)
    continue
  }
  const buf = readFileSync(from)
  writeFileSync(join(OUT, dst), buf)
  manifest.files.push({
    file: dst,
    source: src,
    bytes: buf.length,
    sha256: sha(buf),
    downsampled: false,
  })
}

{
  const [src, dst] = DOWNSAMPLE
  const from = join(CATALOGUE, src)
  if (!existsSync(from)) {
    missing.push(src)
  } else {
    const raw = readFileSync(from)
    const log = []
    const cut = reduce(JSON.parse(raw), '', log)
    const buf = Buffer.from(JSON.stringify(cut))
    writeFileSync(join(OUT, dst), buf)
    manifest.files.push({
      file: dst,
      source: src,
      bytes: buf.length,
      sha256: sha(buf),
      downsampled: true,
      source_bytes: raw.length,
      source_sha256: sha(raw),
      reductions: log,
      note:
        'Arrays longer than ' + KEEP_AT_MOST + ' entries were reduced by even ' +
        'stride, keeping the final entry. Aggregate statistics quoted on the ' +
        'site come from the full file via the experiment results, not from this ' +
        'reduced copy.',
    })
  }
}

writeFileSync(join(OUT, 'manifest.json'), JSON.stringify(manifest, null, 2))

const lines = [
  '# Shipped measurements',
  '',
  'These files are copies of what the validation suite wrote. The site does',
  'not transcribe numbers out of them by hand, and it does not round them.',
  '',
  '`manifest.json` records each file with its source path inside `catalogue/`,',
  'its byte size and its SHA-256, so any copy here can be checked against the',
  'original.',
  '',
  '## What is recomputed and what is read',
  '',
  'Recomputed in your browser, from source you can edit:',
  '',
  '- everything the shapeshifter cells produce (`lavoisier.ladder.*`) --- the',
  '  parser, compiler and ladder arithmetic are ported to JavaScript and',
  '  reproduce the Python outputs to full double precision;',
  '- the CASMI licensing decision, recomputed from `panel_data.json` whenever',
  '  you move the floor or margin sliders.',
  '',
  'Read from these files, not computed here:',
  '',
  '- every claim in a paper page’s claims table (`exp*.json`);',
  '- the sweep curves (`exp4_sweeps.json`, `exp5_sweeps.json`, `exp6_sweeps.json`);',
  '- the spectral bindings replayed inside notebook cells --- these are marked',
  '  in the cell itself, every time, and the operation refuses rather than',
  '  returning a stale value if you change an argument the recording did not',
  '  cover.',
  '',
  '## Downsampling',
  '',
  '`exp5_sweeps.json` is ~1.5 MB at source and is reduced here by even stride,',
  'keeping the final entry of every array. The manifest lists each reduction',
  'with its before and after lengths. No statistic on the site is computed from',
  'the reduced copy; the reduced copy is used for drawing curves only.',
  '',
]
writeFileSync(join(OUT, 'README.md'), lines.join('\n'))

for (const f of manifest.files) {
  console.log(
    (f.downsampled ? 'reduced ' : 'copied  ') +
      f.file.padEnd(38) +
      String(f.bytes).padStart(9) +
      ' bytes' +
      (f.downsampled ? '  (from ' + f.source_bytes + ')' : '')
  )
}
if (missing.length) {
  console.error('\nmissing source files:')
  for (const m of missing) console.error('  ' + m)
  process.exitCode = 1
}
