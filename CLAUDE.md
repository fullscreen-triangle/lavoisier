# Lavoisier

## Code search routing

Three `purpose` subcommands answer three different questions. Pick by question,
never by cost — measured both ways, neither reliably undercuts the other.

| You want | Use |
|---|---|
| Where is `X` defined? | `purpose ask "X"` — one **stem**, never a question |
| Which modules is this task actually about? | `purpose ckg ask "<goal>"` |
| What breaks if I change this file? | `purpose ckg why <module>` |
| Which turns of this conversation still matter? | `purpose ledger` |

`purpose ask` guidance (stems not questions, filenames not indexed, a miss proves
nothing) is in the global CLAUDE.md and applies here unchanged.

### The ckg graph in this repo

`.purpose/lens.toml` is committed and carries the reasoning inline — read it
before re-tuning. `.purpose/index.json` and `.purpose/ckg.json` are gitignored
caches; rebuild with `purpose index && purpose ckg build`.

Two facts about this repo that drove the lens, both non-obvious:

- **It is manuscript-heavy, not code-heavy.** 35,874 indexed symbols, but 69% are
  prose (`section` 16,125 + `heading` 8,606). Code is `def` 8,257 / `class` 1,392
  / `fn` 755 / `func` 474 / `struct` 221.
- **File granularity does not terminate.** 1,662 files is too many modules;
  `purpose ckg lens` hung past 300s. The lens uses `granularity = "dir"` (145
  modules). If you switch back to `"file"`, expect it to hang, not to be slow.

`def` is excluded from `include.kinds` deliberately — it is the difference between
a graph where `__init__` touches 51% of modules and one whose top terms are
`SEntropyCoordinates` / `Spectrum` / `DDAEvent`. The trade is recorded in the lens
file with both sets of numbers.

### Known tool bug: the `m` phantom node

`purpose ckg build` writes a node `m` into `.purpose/ckg.json` that is absent from
the 145-entry `items` array yet appears in **145 of 463 edges** — exactly one edge
to every real module, each at weight 1.0. No indexed path maps to it.

Scope, verified rather than assumed:

- **Sound anyway** — `COMPONENTS`, `DENSITY`, `TERM SPREAD` (the tool's own
  diagnostics already exclude it; recomputing without it reproduces 80/55%/0.030
  exactly), and `ckg ask` verdicts and σ (σ is a min-cut, so `m` shifts it by at
  most 1.0).
- **Affected** — `ckg why` resting cuts, which gain one spurious edge per module.
  Ignore any `m — <module>` line in that output.

### Reading diagnostics

There is deliberately **no score to maximise**, and β* is a monotonicity signal,
not a quality measure — a lens where every module draws identical distinctions
induces among the *highest* floors while discriminating *worst*. Never report a
rising β* as an improvement. Judge by component sizes, term spread, and goal
saturation (want <50%).

When tuning, reach for `include.kinds` first. Stopwords cannot touch symbol names,
and `[terms.weight]` moves σ and Ω but **cannot split components** — verified here:
down-weighting the dunders took Ω 20,472 → 13,351 and left `e = 4535` and the
127-module giant component completely unchanged.

Full documentation: [mushina/sources/purpose-ckg.md](mushina/sources/purpose-ckg.md)
