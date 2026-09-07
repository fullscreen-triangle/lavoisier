// =====================================================================
//  Loading the shipped measurements.
//
//  These files are static and small enough to import, so they are
//  imported --- no fetch, no loading state, no async anywhere in the
//  app. Vite inlines the JSON at build time and the pages render in one
//  pass, which is what keeps every component here synchronous.
// =====================================================================

import exp1 from '../data/exp1_instrument_ladder.json'
import exp2 from '../data/exp2_observation_groups.json'
import exp3 from '../data/exp3_coordinate_provenance.json'
import exp4 from '../data/exp4_runtime_graph.json'
import exp5 from '../data/exp5_sink_detection.json'
import exp6 from '../data/exp6_peptide_mass_invariance.json'

export const EXPERIMENTS = { exp1, exp2, exp3, exp4, exp5, exp6 }

// Every graded claim across the six experiment files, in one list, so
// the landing page can report the corpus total without any number
// being typed in by hand.
export function corpusTotals() {
  let graded = 0
  let passed = 0
  let failed = 0
  let nonDiscriminating = 0
  for (const k of Object.keys(EXPERIMENTS)) {
    const s = EXPERIMENTS[k].summary
    graded += s.graded
    passed += s.passed
    failed += s.failed
    nonDiscriminating += s.non_discriminating
  }
  return { graded, passed, failed, nonDiscriminating, files: Object.keys(EXPERIMENTS).length }
}
