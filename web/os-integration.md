# Using Shape Shifter in Buhera OS

This is the practical guide for running the lavoisier Shape Shifter interpreter
(`@lavoisier/shapeshifter`) inside the Buhera OS terminal. It covers first-time
setup, everyday use, and what to do when the link breaks.

The interpreter lives in **this repo** at `web/src/lib` (published as the npm
package `@lavoisier/shapeshifter`). Buhera OS lives in a separate repo at
`architecture/buhera/long-grass`. Buhera consumes the interpreter as a normal
package dependency; the React chart panels that render its output live in Buhera.

---

## TL;DR — it already works

If both repos are checked out side by side and Buhera's dependencies are
installed, you can already run scripts. In the Buhera terminal:

```
dispatch("shapeshifter", "demo")
```

You should see a compile/run log followed by an inline chart (135 PC-lipid
records). If that works, skip to [Everyday use](#everyday-use). If it does not,
do the [First-time setup](#first-time-setup) below.

---

## How it fits together

```
  @lavoisier/shapeshifter          long-grass (Buhera OS)
  (this repo, web/src/lib)         (architecture/buhera/long-grass)
  ───────────────────────          ─────────────────────────────────
  parser + executor        ──▶     src/lib/modules/shapeshifter-module.js
  + lavoisier.* science            (the module trait: describe/execute/…)
  (pure JS, no deps)                        │
                                            ▼
                                   src/components/shapeshifter/
                                     WorkspaceValue.js   (chart panels)
                                     SandboxCharts.js    (records charts)
                                            │
                                            ▼
                                   src/components/BuheraTerminal.js
                                     registers the module + renders charts
```

- **The interpreter is pure JavaScript with zero runtime dependencies.** It
  compiles a `.ss` script, runs its `phase` blocks, and returns a `workspace`
  (a list of produced values, each tagged with a visualisation `kind`).
- **Buhera renders one panel per workspace value, inline** — REPL/notebook
  style, one result per `dispatch`, not a whole dashboard.

---

## First-time setup

You only do this once per machine (or after moving/cloning the repos).

### Prerequisite: both repos side by side

Buhera's `package.json` refers to this package by a relative path:

```json
"@lavoisier/shapeshifter": "file:../../../bioinformatics/lavoisier/web/src/lib"
```

That path assumes the standard layout:

```
Documents/
  bioinformatics/lavoisier/web/src/lib      ← this package
  architecture/buhera/long-grass            ← Buhera OS
```

If your repos sit somewhere else, the relative path in Buhera's `package.json`
must match — that is the only path you would ever adjust.

### Install

From the Buhera repo:

```bash
cd architecture/buhera/long-grass
npm install
```

`npm install` reads the `file:` dependency and wires `@lavoisier/shapeshifter`
into `node_modules`. Nothing else is required for the normal (side-by-side)
case.

### Verify

```bash
npx next build
```

A clean `Compiled successfully` means the module, the package, and the chart
renderers all resolve. To confirm the package points at real interpreter code:

```bash
node -e "const p=require('fs').realpathSync('node_modules/@lavoisier/shapeshifter'); console.log(p, require('fs').existsSync(p+'/shapeshifter/compiler.js'))"
```

Then start the dev server and try the demo:

```bash
npm run dev
# open the terminal, type:  dispatch("shapeshifter", "demo")
```

---

## Everyday use

Once set up, using the framework is just typing scripts into the Buhera
terminal. Every dispatch is one notebook cell: a small script, one result,
rendered inline.

### The demo

```
dispatch("shapeshifter", "demo")
```

Runs a canned PC-lipid experiment and charts the records.

### A single-step cell (the common case)

A bare string is treated as `.ss` source. You do not need a full
`objective`/`instrument` program — a single `phase` block is enough:

```
dispatch("shapeshifter", "phase p:\n  se = lavoisier.observe.sentropy(frequencies: [1200, 1550, 2900])")
```

Produces an `se:sentropy` value → rendered as the S-entropy panel.

### A records cell (produces the chart grid)

```
dispatch("shapeshifter", "phase p:\n  r = lavoisier.instrument.run_experiment(classes: [\"PC\"], polarity: \"+\", analyser: \"orbitrap\")")
```

Produces `r:records` → the m/z × intensity scatter, class/adduct bars, and
S-entropy histograms.

### Explicit form

```
dispatch("shapeshifter", { kind: "run", source: "objective o:\n  target: \"...\"\nphase p:\n  ..." })
```

### What renders as a chart

Each workspace value's `kind` selects a panel:

| kind           | rendered as                              |
| -------------- | ---------------------------------------- |
| `records`      | full record chart grid (SandboxCharts)   |
| `cells`        | ΔP cell registry table                   |
| `addresses`    | partition address table                  |
| `sentropy`     | S-entropy coordinate panel               |
| `subharmonics` | fragment subharmonic bars                |
| `tensorReport` | virtual tensor summary                   |
| `impossible`   | crossing-symmetry probe table            |
| `transient`    | single-transient contents                |
| `complement`   | SWIFT antistate                          |
| `domain`       | purpose-domain reduction                 |
| `validation`   | dual-path validation                     |
| `scalar`/`string`/`number` | value card                   |
| anything else  | JSON dump                                |

### Capabilities you can call in a `phase`

`lavoisier.instrument.run_experiment`, `lavoisier.instrument.run_proteomics`,
`lavoisier.partition.compute_addresses`, `lavoisier.cells.compile`,
`lavoisier.msms.sebd_search`, `lavoisier.msms.phase_coherence`,
`lavoisier.msms.virtual_tensor`, `lavoisier.msms.impossible_ions`,
`lavoisier.msms.partition_complement`, `lavoisier.msms.transient_contents`,
`lavoisier.observe.sentropy`, `lavoisier.observe.ternary_address`,
`lavoisier.observe.partition_field`, `lavoisier.observe.dual_path_validate`,
`lavoisier.db.generate`, `lavoisier.purpose.domain`, `lavoisier.purpose.match`,
`lavoisier.purpose.combine`.

> **Online DB search** (`lavoisier.db.search*`) currently renders as a
> "pending" placeholder in Buhera — the interpreter returns it as an
> unresolved async sentinel and the network resolver is not wired in this build.
> Every other (pure-compute) capability runs fully.

---

## Picking up interpreter changes

Because the dependency is a live link to `web/src/lib`, edits you make to the
interpreter here are seen by Buhera immediately — just restart the Buhera dev
server (`npm run dev`). No reinstall, no version bump.

If you ever change the package's **`exports` map or `package.json`** (not just
its code), re-run `npm install` in Buhera so npm re-reads the manifest.

---

## When it breaks

### "Module not found: @lavoisier/shapeshifter"

The link is missing or points nowhere. Fixes, in order:

1. Confirm the two repos are side by side and the `file:` path in Buhera's
   `package.json` matches their actual relative location.
2. Re-run `npm install` in `long-grass`.
3. If still failing, relink explicitly:
   ```bash
   cd bioinformatics/lavoisier/web/src/lib && npm link
   cd architecture/buhera/long-grass && npm link @lavoisier/shapeshifter
   ```

### Scripts run but show no chart, only text

The workspace value has a `kind` with no panel — it falls back to a JSON dump.
Add a panel for that kind in
`long-grass/src/components/shapeshifter/WorkspaceValue.js`.

### A capability call logs "Unknown function"

The `fn` name in the script is not one the interpreter dispatches. Check spelling
against [the capability list above](#capabilities-you-can-call-in-a-phase);
adding a new capability means adding a branch in this package's
`shapeshifter/compiler.js` (`executeCall`).

---

## Limits of the current setup (and the proper fix later)

The `file:` link works on any machine where **both repos are checked out side by
side**. It does **not** work where Buhera is cloned alone (a fresh CI runner, a
teammate who only has Buhera) — there the relative path resolves to nothing.

The bulletproof fix for that case is to **publish `@lavoisier/shapeshifter` to a
registry** (GitHub Packages for private, or public npm) and depend on it by
version:

```json
"@lavoisier/shapeshifter": "^0.1.0"
```

Then there are no local paths at all, and `npm install` behaves identically
everywhere. That step needs a registry login and is worth doing the first time
you need Buhera to build somewhere lavoisier is not checked out beside it — not
before. It can be automated with a GitHub Action that publishes on each push to
this package.

---

## File map (for when you need to change something)

| To change…                        | Edit…                                                              |
| --------------------------------- | ------------------------------------------------------------------ |
| interpreter / a capability        | `web/src/lib/shapeshifter/compiler.js` (this repo)                 |
| the module's demo / instruction   | `long-grass/src/lib/modules/shapeshifter-module.js`                |
| a chart panel                     | `long-grass/src/components/shapeshifter/WorkspaceValue.js`         |
| the records chart grid            | `long-grass/src/components/shapeshifter/SandboxCharts.js`          |
| registration / terminal wiring    | `long-grass/src/components/BuheraTerminal.js`                      |
| how Buhera finds the package      | `long-grass/package.json` (the `@lavoisier/shapeshifter` line)     |
