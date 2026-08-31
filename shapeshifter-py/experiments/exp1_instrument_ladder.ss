// =====================================================================
// Experiment 1 — The instrument of Listing 1
//
// The five-contact instrument of the instrument-process-ladder paper,
// expressed in shapeshifter and evaluated from its numbers alone. No
// substrate is simulated: by Theorem 2.4 no readout consults one.
//
// Values to be reproduced, computed by hand in the paper before this
// program was written:
//     composite resolution  0.91712
//     sensitivities         0.2072 0.1275 0.1658 0.1105 0.0975
//     drop k5  -> 0.90250   (still meets the requirement)
//     drop k1  -> 0.79280   (fails it)
//     minimum contacts at pow_max 0.60 for target 0.90  ->  3
//
// The experiment can fail: if composite resolution tracked the sum or
// the maximum of the rung resolutions rather than eq. (compose), or if
// control were ordered by ASCENDING resolution, the ladder algebra of
// Part II would be refuted.
// =====================================================================

import lavoisier.ladder

objective InstrumentLadder:
    target: "evaluate a five-contact instrument without a substrate"
    criterion: "composite 0.91712, control at the strongest contact"

ladder instrument toward target
  rung k1 at 0.60
  rung k2 at 0.35
  rung k3 at 0.50
  rung k4 at 0.25
  rung k5 at 0.15
  require resolution >= 0.90

phase Resolve:
    evaluation = lavoisier.ladder.resolve(ladder: instrument)

phase Ablate:
    // Which contacts does the requirement actually depend on?
    ablation = lavoisier.ladder.ablate(ladder: instrument)

phase Bound:
    // Static question, answered without executing anything.
    bound = lavoisier.ladder.minimum(target: 0.90, pow_max: 0.60)
