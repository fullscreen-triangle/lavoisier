// Negative control for prediction P9 (static reachability). These three
// contacts cannot reach 0.90; the compiler must reject the program
// before any phase runs. If this program executes, P9 is refuted.

import lavoisier.ladder

objective Unreachable:
    target: "a declared requirement the declared contacts cannot meet"
    criterion: "compile stage must reject before execution"

ladder shortfall toward target
  rung a at 0.30
  rung b at 0.25
  rung c at 0.20
  require resolution >= 0.90

phase Resolve:
    evaluation = lavoisier.ladder.resolve(ladder: shortfall)
