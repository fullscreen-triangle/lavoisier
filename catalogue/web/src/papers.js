// =====================================================================
//  The seven papers, in one place.
//
//  `verdict` and the graded counts are NOT typed in here --- they come
//  from the validation JSON at render time, so this file cannot drift
//  out of agreement with the experiments. What is typed here is only
//  what the JSON does not carry: the slug, the title, the one-sentence
//  claim, and where each paper sits in the two reading orders.
// =====================================================================

export const PAPERS = [
  {
    slug: 'peptide-mass-invariance',
    exp: 'exp6',
    title: 'Peptide mass invariance',
    short: 'Mass invariance',
    claim:
      'A cut key computed on a contact graph is unchanged by relabelling the items, ' +
      'and collapses to nothing when the medium is deleted.',
    role: 'foundational',
    blurb:
      'The paper the others stand on. Defines the floor, the medium and the ' +
      'residual, and reports two of its own predictions refuted.',
  },
  {
    slug: 'runtime-graph',
    exp: 'exp4',
    title: 'Runtime graph certificates',
    short: 'Runtime graph',
    claim:
      'A cut can be certified in time linear in the edges, and a forged ' +
      'certificate cannot be made to verify.',
    role: 'operational pair (first)',
    blurb:
      'Compile once in O(n log n + nd), verify in O(|E|). 123 forgeries, none accepted.',
  },
  {
    slug: 'sink-detection',
    exp: 'exp5',
    title: 'Sink detection',
    short: 'Sink detection',
    claim:
      'Weighted spread locates the vertex a residual drains into, and hits its ' +
      'bound in every cell tested.',
    role: 'operational pair (second)',
    blurb:
      'All 54 cells hit the bound to full precision. Three of its own stated ' +
      'results are shown to be defective, and are replaced rather than hidden.',
  },
  {
    slug: 'observation-groups',
    exp: 'exp2',
    title: 'Observation groups',
    short: 'Observation groups',
    claim:
      'Replicate structure forms a refinement lattice, and the interval that ' +
      'separates two groupings is computable.',
    role: 'machinery reused',
    blurb:
      'The same cut machinery applied to replicate structure: a lattice of 203 ' +
      'partitions with a threshold that lands between the two candidates.',
  },
  {
    slug: 'coordinate-provenance',
    exp: 'exp3',
    title: 'Coordinate provenance',
    short: 'Coordinate provenance',
    claim:
      'Of 512 coordinate maps, none is auditable without a record; the minimal ' +
      'record costs a constant factor, not a growing one.',
    role: 'type discipline',
    blurb:
      'The type-discipline paper. Also the one whose stated universal property ' +
      'is only half true, and says so.',
  },
  {
    slug: 'instrument-process-ladder',
    exp: 'exp1',
    title: 'Instrument process ladder',
    short: 'Process ladder',
    claim:
      'An instrument is a sequence of contacts whose resolutions compose ' +
      'multiplicatively, and the composition is independent of the substrate.',
    role: 'most self-contained',
    blurb:
      'The paper the shapeshifter language expresses directly. Every number on ' +
      'its page is recomputed in your browser from the source shown.',
  },
  {
    slug: 'uc-davis-casmi-catalogue',
    exp: null,
    title: 'UC Davis CASMI catalogue',
    short: 'CASMI',
    claim:
      'Applied to 58 real identification challenges, the method licenses 5 and ' +
      'declines 53 --- and the declining is the result, not the failure.',
    role: 'empirical demonstration',
    blurb:
      'The empirical end of the catalogue. Chooses its floor empirically rather ' +
      'than proving one, and reports that the chosen value sits at noise level.',
  },
]

export const BY_SLUG = Object.fromEntries(PAPERS.map((p) => [p.slug, p]))

// Two orders, because they suit two different readers. Neither is the
// order the papers were written in.
export const ORDER_ARGUMENT = [
  'peptide-mass-invariance',
  'runtime-graph',
  'sink-detection',
  'observation-groups',
  'coordinate-provenance',
  'instrument-process-ladder',
  'uc-davis-casmi-catalogue',
]

export const ORDER_CONVINCED = [
  'uc-davis-casmi-catalogue',
  'peptide-mass-invariance',
  'runtime-graph',
  'sink-detection',
  'observation-groups',
  'coordinate-provenance',
  'instrument-process-ladder',
]
