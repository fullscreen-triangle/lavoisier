import React, { useState, useRef, useCallback, useEffect } from "react";
import {
  Files, Search, GitBranch, Play, Blocks, Settings,
  ChevronRight, ChevronDown, X, Circle, FileCode2, FileJson,
  FileText, Folder, FolderOpen, Terminal as TerminalIcon,
  AlertCircle, Bell, PanelBottomClose, Check, Code2,
  Trash2, RefreshCw, Cpu, Zap, Hammer, Square,
} from "lucide-react";
import { compileStage, executeStage } from "@/lib/shapeshifter/compiler";
import { summariseRecords } from "@/lib/experiment/virtualinstrument";
import { searchAll, searchMassBank, searchGNPS, searchMoNA } from "@/lib/spectral/dbSearch";
import { useStore } from "@/lib/state/store";
import ResultsDashboard from "@/components/experiment/ResultsDashboard";
import SandboxCharts from "@/sandbox/SandboxCharts";

const summariseForStore = (recs) => summariseRecords(recs);

/** Resolve a pending __async DB-search placeholder into real results. */
async function resolvePending(p) {
  try {
    switch (p.__fn) {
      case "db.search":          return await searchAll(p.precMz, p.frags || [], p.dbs || ["massbank", "mona"]);
      case "db.search_massbank": return await searchMassBank(p.precMz);
      case "db.search_gnps":     return await searchGNPS(p.precMz, p.frags || []);
      case "db.search_mona":     return await searchMoNA(p.precMz);
      default:                   return { hits: [], error: `unknown ${p.__fn}` };
    }
  } catch (e) {
    return { hits: [], error: e.message };
  }
}

/* ─── Theme ──────────────────────────────────────────────────────────────── */
const T = {
  titlebar: "#3c3c3c", activitybar: "#333333", activitybarFg: "#858585",
  activitybarFgActive: "#ffffff", sidebar: "#252526", sidebarFg: "#cccccc",
  sidebarHeader: "#bbbbbb", editor: "#1e1e1e", editorFg: "#d4d4d4",
  tabBar: "#252526", tabActive: "#1e1e1e", tabInactive: "#2d2d2d",
  tabFg: "#969696", tabFgActive: "#ffffff", border: "#3c3c3c",
  accent: "#0e639c", accentBright: "#007acc", statusBar: "#007acc",
  statusFg: "#ffffff", panel: "#1e1e1e", gutter: "#858585",
  lineActive: "#2a2d2e",
};

const CLASS_COLORS = {
  PC: "#5fa8d3", PE: "#e07a7a", PS: "#b388eb", PG: "#e493b3",
  PI: "#5dc0d8", SM: "#7cc77c", Cer: "#e6a456", TAG: "#cdc15c",
  DAG: "#a07a5e", LPC: "#a8b2bd", CE: "#9cc4d8", FA: "#e8c598",
  HSA: "#60a5fa", HBB: "#f87171", ENO1: "#34d399", CYCS: "#a78bfa", CASE: "#fb923c",
};

/* ─── Example files ──────────────────────────────────────────────────────── */
const SS_HELLO_LIPID = `\
// Predict a lipidomics spectral library for plasma analysis.
// The virtual instrument simulates forward MS using the partition Lagrangian.
// Each analyte maps to partition coordinates (n, ℓ, m, s) and S-entropy (Sk, St, Se).

import lavoisier.instrument

objective PlasmaLipidomics:
    target: "predict lipidomics library for plasma LC-MS/MS"
    success_criteria: "records > 100"

instrument OrbitrapFusion:
    analyzer: "orbitrap"
    polarity: "+"
    collision_energy: 25
    mz_window: [200, 1500]

phase Design:
    classes = ["PC", "PE", "SM", "TAG", "Cer"]

phase VirtualRun:
    records = lavoisier.instrument.run_experiment(
        classes: classes,
        polarity: "+",
        analyser: "orbitrap",
        collision_energy: 25
    )
`;

const SS_TARGETED_LIPID = `\
// Targeted lipidomics with per-class ranges and acquisition rules.
// This is the experiment designer expressed as code — instead of clicking
// through analyte / ionisation / acquisition tabs, every range and rule is
// declared explicitly. Far more precise than UI sliders.

import lavoisier.instrument

objective TargetedPlasmaLipids:
    target: "PC and PE in a defined RT window, plus long-chain triacylglycerols"

instrument OrbitrapFusion:
    analyzer: "orbitrap"
    polarity: "+"
    collision_energy: 27
    mz_window: [400, 1000]

phase Design:
    // Each class carries its OWN acyl-carbon and double-bond ranges.
    panel = [
        { class: "PC",  carbons: [30, 40], db: [0, 4] },
        { class: "PE",  carbons: [32, 38], db: [0, 3] },
        { class: "SM",  carbons: [16, 24], db: [0, 2] },
        { class: "TAG", carbons: [48, 56], db: [1, 6] }
    ]

phase VirtualRun:
    records = lavoisier.instrument.run_experiment(
        classes: panel,
        polarity: "+",
        adducts: ["[M+H]+", "[M+Na]+", "[M+NH4]+"],
        analyser: "orbitrap",
        collision_energy: 27,
        gradient_min: 30,
        // Post-generation rules: keep only PC/PE eluting mid-gradient (8–18 min),
        // m/z 600–900, with at least one double bond. TAGs pass the m/z/rt net.
        filters: {
            classes: ["PC", "PE"],
            rt: [8.0, 18.0],
            mz: [600, 900],
            db: [1, 4],
            min_intensity: 0.02
        }
    )
`;

const SS_PROTEOMICS = `\
// Proteomics virtual experiment.
// Predicts tryptic peptide library from common plasma protein standards.
// Multiply-charged ESI adducts are selected by peptide mass.

import lavoisier.instrument

objective PlasmaProteomics:
    target: "predict tryptic peptide library from plasma proteins"
    success_criteria: "records > 50"

instrument OrbitrapFusion:
    analyzer: "orbitrap"
    polarity: "+"
    collision_energy: 28

phase Design:
    proteins = ["HSA", "HBB", "ENO1"]

phase VirtualRun:
    records = lavoisier.instrument.run_proteomics(
        proteins: proteins,
        length_min: 7,
        length_max: 20,
        mc_max: 1,
        polarity: "+",
        analyser: "orbitrap",
        collision_energy: 28
    )
`;

const SS_TEMPORAL = `\
// Temporal programming targeted acquisition.
// Compiles an m/z target list to ΔP timing cells via the Orbitrap cell compiler.
// τ_min = ℏ / δM_i  — the exact minimum acquisition time
// derived from the Partition Uncertainty Relation.

import lavoisier.cells

objective TargetedAcquisition:
    target: "compile plasma lipid targets to ΔP timing cells"

instrument OrbitrapXL:
    analyzer: "orbitrap"
    kappa: 1.0e12
    ref_frequency: 1.0e7
    timing_jitter: 2.0e-9

target_list PlasmaPanel:
    instrument: OrbitrapXL
    window_ppm: 5.0
    targets: [
        { name: "PC(32:0)", mz: 734.5694 },
        { name: "PC(34:1)", mz: 760.5851 },
        { name: "PE(34:1)", mz: 716.5259 },
        { name: "SM(d18:1)", mz: 703.5749 },
        { name: "LPC(16:0)", mz: 496.3396 },
        { name: "TAG(52:2)", mz: 879.7716 }
    ]

phase CompileCells:
    registry = lavoisier.cells.compile(
        target_list: PlasmaPanel
    )

validate TimingFeasibility:
    check_resolution_time(
        instrument: OrbitrapXL,
        window_ppm: 5.0
    )
`;

const SS_PARTITION = `\
// Partition address computation.
// Demonstrates the bijection Φ: ℤ⁺ → 𝒫 from the Partition Bijection Theorem.
// Each ion mass maps to a unique partition state (n, ℓ, m, s) —
// the four quantum numbers of the bounded phase space formulation.

import lavoisier.partition

objective PartitionMapping:
    target: "map lipid panel to partition states (n, ℓ, m, s)"

phase Compute:
    lipids = [
        { name: "PC(32:0)", mass: 733.5622, adduct: "[M+H]+" },
        { name: "PC(34:1)", mass: 759.5779, adduct: "[M+H]+" },
        { name: "PE(34:1)", mass: 715.5153, adduct: "[M+H]+" },
        { name: "SM(d18:1/16:0)", mass: 702.5676, adduct: "[M+H]+" },
        { name: "TAG(52:2)", mass: 878.7638, adduct: "[M+NH4]+" },
        { name: "LPC(16:0)", mass: 495.3325, adduct: "[M+H]+" }
    ]

    addresses = lavoisier.partition.compute_addresses(lipids)
`;

const SS_SENTROPY_OBS = `\
// S-entropy observation — compute partition addresses from vibrational frequencies.
// Implements the Context-Based Spectral Database (Paper 3) generative approach.
// The address IS the spectrum: no dynamics required (Partition Determinism Theorem).

import lavoisier.observe
import lavoisier.purpose

objective SEntropyObservation:
    target: "compute S-entropy coordinates and ternary addresses from vibrational modes"

phase ComputeSentropy:
    // H2O vibrational modes in cm⁻¹ (NIST CCCBDB)
    water_freqs = [1595.0, 3657.0, 3756.0]
    water_se    = lavoisier.observe.sentropy(frequencies: water_freqs)

    // Compute ternary address at depth 12
    water_addr  = lavoisier.observe.ternary_address(sentropy: water_se, depth: 12)

    // Ion-droplet bijection: dual-path validation
    validation  = lavoisier.observe.dual_path_validate(
        sentropy: water_se,
        ion: { mass: 18.01, kinetic_energy: 1.0 },
        depth: 10
    )

phase DomainContext:
    // Purpose function: restrict to metabolomics subspace
    domain_context = lavoisier.purpose.domain(domain: "metabolomics", depth: 4)

    // Find matching domains for the S-entropy point
    matches = lavoisier.purpose.match(sentropy: water_se)
`;

const SS_FORCE_FREE = `\
// Force-free mass spectrometry (Paper 1 — Shader Depth Minimisation).
// Demonstrates that all four analyser equations emerge from one partition Lagrangian.
// No force acts at any stage — ions follow −∇M through the partition landscape.

import lavoisier.instrument

objective ForceFreeMS:
    target: "demonstrate analyser universality from the partition Lagrangian"

// Four analysers — same Lagrangian, different M(x,t) topologies:
//   TOF:       M = -κz          (linear gradient)
//   Quadrupole: M = κ₀/2·(x²-y²)[U + V cos Ωt]  (saddle)
//   Orbitrap:  M = κ/2·(z²-r²/2) + κRm²/2·ln(r/Rm)  (well)
//   FT-ICR:    M = 0, A = B/2(-y,x,0)            (circular)

phase TOFExperiment:
    records_tof = lavoisier.instrument.run_experiment(
        classes: ["PC", "PE", "SM"],
        polarity: "+",
        analyser: "tof",
        collision_energy: 25
    )

phase OrbitrapExperiment:
    records_orbi = lavoisier.instrument.run_experiment(
        classes: ["PC", "PE", "SM"],
        polarity: "+",
        analyser: "orbitrap",
        collision_energy: 25
    )

phase FticExperiment:
    records_fticr = lavoisier.instrument.run_experiment(
        classes: ["PC", "PE"],
        polarity: "+",
        analyser: "fticr",
        collision_energy: 25
    )
`;

const SS_GENERATIVE_DB = `\
// Generative database demonstration (Paper 3).
// The database stores nothing — it IS the phase space structure.
// Entries materialise on demand and dissolve after use.
// Cost: O(k) per query, independent of database size N. Memory: O(1).

import lavoisier.observe
import lavoisier.purpose

objective GenerativeDatabase:
    target: "demonstrate O(1) generative database lookup from ternary prefix"

phase GenerateFromPrefix:
    // Given a ternary prefix, reconstruct the S-entropy coordinates
    // without any stored spectrum (Partition Determinism Theorem).
    entry_202  = lavoisier.db.generate(prefix: "202", depth: 12)
    entry_120  = lavoisier.db.generate(prefix: "120", depth: 12)
    entry_001  = lavoisier.db.generate(prefix: "001", depth: 12)

phase PurposeFiltering:
    // Proteomics domain: 98%+ search space reduction
    proteomics_ctx = lavoisier.purpose.domain(domain: "proteomics", depth: 4)

    // Lipidomics domain
    lipidomics_ctx = lavoisier.purpose.domain(domain: "lipidomics", depth: 4)

    // Intersect: proteomics ∩ lipidomics (Prompt Contraction Theorem)
    combined = lavoisier.purpose.combine(
        domains: ["proteomics", "lipidomics"],
        depth: 4
    )
`;

const SS_SEBD_SEARCH = `\
// Partition-State Graph Search — SEBD-MS algorithm.
// Each ion maps to node (n, ℓ, m, s). Searches forward from precursor,
// backward from each fragment via virtual predecessors Sv* = 2·Sv_f − Sv_2.
// Off-shell Sv* = chemical transition states (Theorem 5.2).
// Output: PredictedRecord[] → feeds all existing dashboard charts.

import lavoisier.msms

objective MSMSSearch:
    target: "identify fragmentation pathways in an HCD spectrum"

phase FragmentSearch:
    // Lysine [M+H]+, HCD fragments (NIST AC_CAC verified)
    precursor_mz = 147.1128
    fragments = [84.081, 101.107, 102.091, 130.087]

    records = lavoisier.msms.sebd_search(
        precursor_mz: precursor_mz,
        fragments: fragments,
        max_depth: 7,
        planck_depth: 56
    )

phase PhaseCoherence:
    // Verify: ω_f = ω_prec · sqrt(m_prec/m_f) — self-consistent <10⁻⁹ ppm
    subharmonics = lavoisier.msms.phase_coherence(
        precursor_mz: precursor_mz,
        fragments: fragments
    )
`;

const SS_VIRTUAL_TENSOR = `\
// Stacked Virtual Substates — V_{ijkl} partition tensor.
// Four stacking dimensions: instrument × charge × polarity × time.
// Off-shell fraction ~8.3% — the virtual transition states.
// One Orbitrap transient contains full MS/MS, all charge states, polarity complement.

import lavoisier.msms

objective VirtualSubstates:
    target: "decompose partition state across 4 virtual dimensions"

phase BuildTensor:
    mz = 162.1125  // leucine [M+H]+

    tensor = lavoisier.msms.virtual_tensor(
        mz: mz,
        charge: 1,
        time_steps: 10
    )

phase SingleTransient:
    // Theorem 11.1: all information in one transient
    transient = lavoisier.msms.transient_contents(
        precursor_mz: mz,
        fragments: [44.049, 86.096, 103.086, 118.110]
    )

phase CrossingSymmetry:
    // Impossible ions as crossing-symmetry probes
    probes = lavoisier.msms.impossible_ions(
        mz_list: [86.096, 103.086, 118.110]
    )
`;

const SS_DB_SEARCH = `\
// Online spectral database search.
// Search public MS/MS libraries via their REST APIs.
// Query cost: O(k) per trie lookup, independent of database size N.
// Databases: MassBank (EU), MoNA (Davis), GNPS (San Diego).

import lavoisier.db

objective SpectralSearch:
    target: "search public spectral databases for lysine HCD fragments"

phase SearchDatabases:
    precursor_mz = 147.1128  // lysine [M+H]+
    fragments = [84.081, 101.107, 102.091, 130.087]

    // Search MassBank and MoNA in parallel
    db_results = lavoisier.db.search(
        precursor_mz: precursor_mz,
        fragments: fragments,
        databases: ["massbank", "mona"]
    )
`;

const README_CONTENT = `\
# Shapeshifter Sandbox

A live compiler for the Shapeshifter mass spectrometry DSL.
Select an example from the explorer and hit **Run**.

## Language structure

\`\`\`
import lavoisier.instrument

objective Name:
    target: "what you want to achieve"

instrument Name:
    analyzer: "orbitrap"   // tof | orbitrap | fticr | quadrupole
    polarity: "+"

phase Design:
    variable = value

phase Execute:
    result = lavoisier.module.function(key: value, ...)
\`\`\`

## Available functions

### Virtual instrument
| Function | Output |
|---|---|
| \`lavoisier.instrument.run_experiment(...)\` | Lipidomics PredictedRecord[] |
| \`lavoisier.instrument.run_proteomics(...)\` | Proteomics PredictedRecord[] |

### Partition / observation (Papers 1–3)
| Function | Output |
|---|---|
| \`lavoisier.partition.compute_addresses(lipids)\` | Partition states (n,ℓ,m,s) |
| \`lavoisier.cells.compile(target_list)\` | ΔP timing cell registry |
| \`lavoisier.observe.sentropy(frequencies)\` | S-entropy (Sk,St,Se) |
| \`lavoisier.observe.ternary_address(sentropy, depth)\` | Ternary address string |
| \`lavoisier.observe.dual_path_validate(sentropy, ion)\` | Bijection validation |
| \`lavoisier.db.generate(prefix, depth)\` | Coords from prefix (generative DB) |
| \`lavoisier.purpose.domain(domain, depth)\` | Purpose prefixes + reduction ratio |
| \`lavoisier.purpose.match(sentropy)\` | Matching domain contexts |
| \`lavoisier.purpose.combine(domains, depth)\` | Prompt contraction |

### Experiment operations (acquisition half)
| Function | Output |
|---|---|
| \`lavoisier.acquire.library(analytes, seed)\` | Spectra: N analytes x 9 collision energies |
| \`lavoisier.acquire.filter_scans(scans, nce_in)\` | Filtered scan set |
| \`lavoisier.transform.sentropy(scans, alpha, beta, k_neighbors)\` | (Sk,St,Se) per spectrum |
| \`lavoisier.analyse.separation(coords, key, min_group)\` | Within vs between spread + ratio |
| \`lavoisier.analyse.drift(coords, over)\` | Per-axis slope and correlation |
| \`lavoisier.analyse.baseline(scans, tolerance)\` | Cosine similarity (comparison method) |
| \`lavoisier.analyse.shuffle_control(coords, seed)\` | Label-permutation negative control |
| \`lavoisier.analyse.score(separation, drift, min_ratio, max_abs_r)\` | Verdict against a stated criterion |
| \`lavoisier.analyse.sweep(scans, alphas, betas)\` | Ratio over a parameter grid |

See \`experiments/TUTORIAL.md\` for a walkthrough of a program that states
a criterion in advance and can return a verdict against its author.

## Key papers

- **Partition Lagrangian**: all four analyser equations from one Lagrangian (Papers 1–3)
- **ΔP timing cells**: MS is a timing instrument, not an m/z instrument (Paper 1)
- **Triple Equivalence**: oscillation ≡ counting ≡ partition (Paper 2)
- **Generative database**: O(1) memory, O(k) query, stores nothing (Paper 3)
- **Dual-path validation**: ion-droplet bijection, no ground truth needed (Papers 2–3)
`;


const TUTORIAL_CONTENT = "# Running an experiment that can fail\n\nMost of the scripts in \\`examples/\\` **generate** data: you declare\nparameters and the runtime synthesises records. Nothing they produce can\ncontradict you.\n\nThe scripts in this folder do something different. They state a\n**criterion before the run**, then score the result against it. That is\nthe difference between a demonstration and an experiment.\n\n---\n\n## 1. The shape of a testable program\n\nFour blocks, in this order:\n\n| Block | Role |\n|---|---|\n| \\`objective\\` | states the claim **and the criterion**, in the source |\n| \\`phase Read\\` | acquires spectra |\n| \\`phase Transform\\` | computes coordinates |\n| \\`phase Score\\` | scores the criterion, returns a verdict |\n\nThe criterion lives in the document. A reviewer can see it was chosen\n*before* the numbers, not after.\n\n---\n\n## 2. Experiment 0 \u2014 the address and collision energy\n\nOpen \\`exp0_invariance.ss\\` and run it.\n\n**The claim.** The S-entropy address \\`(Sk, St, Se)\\` is a property of the\ncompound, not of the spectrum.\n\n**The test.** A reference library measures each analyte at nine collision\nenergies. If the claim holds, moving the energy should shift the address\n*less* than changing the analyte does.\n\n**The criterion**, declared in the \\`objective\\` block:\n\n    separation_ratio > 2.0    and    |r| < 0.3  for each axis\n\nwhere \\`separation_ratio = mean_between / mean_within\\`.\n\n**What you will see.** The ratio comes out near **0.93** \u2014 below 1.0,\nmeaning changing the collision energy moves a spectrum as far through\nthe coordinate space as changing the compound. \\`Sk\\` and \\`Se\\` both breach\nthe correlation bound. The verdict is \\`NOT MET\\`.\n\nThat is the point. The program was capable of returning a different\nanswer, and did not.\n\n---\n\n## 3. Why a failed criterion needs controls\n\nA negative result is uninterpretable on its own: it could mean the claim\nis wrong, or that the pipeline is broken. Run \\`exp0_controls.ss\\`.\n\n**Negative control** \u2014 shuffle the compound labels. The ratio drops to\nabout **0.35**, well below the measured 0.93. So the statistic *does*\ndistinguish real grouping from random grouping, by roughly 2.7\u00d7. The\napparatus works; the transformation is what falls short.\n\n**Comparison method** \u2014 cosine similarity of the raw peak lists. Between\n*adjacent* energy levels it stays near **0.93**; across the full range it\nfalls to near zero. The established method is stable exactly where the\ncriterion asked the coordinates to be stable.\n\nA result reported without both of these is a claim, not evidence.\n\n---\n\n## 4. Was it just a bad parameter choice?\n\n\\`alpha\\`, \\`beta\\` and \\`k\\` are analyst choices. The language makes you\nstate them; it does not tell you what to pick. So the honest follow-up is\nwhether some *other* setting would have passed.\n\nRun \\`exp0_sweep.ss\\`. Fifteen settings, scored identically. None reaches\nthe threshold, and none reaches even 1.0.\n\nNote the shape of the dependence: the ratio is flat in \\`beta\\` and falls\nin \\`alpha\\`. That is diagnostic \u2014 the mass term separates *spectra*\nwithout separating *compounds*, so weighting it harder makes things\nworse.\n\nA grid search proves no impossibility, and this one does not claim to.\nIt shows the declared setting was not unlucky.\n\n---\n\n## 5. The mechanism\n\nRun \\`exp0_mechanism.ss\\`. It splits the library into low and high\ncollision energy and transforms each separately.\n\n\\`Se\\` is the Shannon entropy of the local intensity neighbourhood.\nRaising collision energy produces more fragments of comparable\nintensity \u2014 which raises that entropy directly. The correlation is not\nan implementation defect; it is what the definition computes.\n\nFragment count rises from about 7 peaks at NCE 10 to 34 at NCE 40, and\n\\`Se\\` tracks it.\n\n---\n\n## 6. What the language contributed\n\nEverything above could be done in a script. What the document adds:\n\n- **The criterion is in the source**, not in a notebook cell or a memory.\n- **Compile before execute.** Compiling lists the operations and inputs\n  a program will touch, having touched none of them.\n- **The workspace is append-only.** Every intermediate stays inspectable,\n  so a surprising verdict can be traced back through the bindings that\n  produced it.\n- **Failure is named.** An operation that cannot proceed says which\n  condition it could not meet.\n\n---\n\n## 7. Try to break the result\n\nThe useful exercise is to attempt a pass:\n\n1. In \\`exp0_invariance.ss\\`, change \\`k_neighbors\\` to 2, then 20.\n2. Restrict the axes: \\`axes: [\"s_k\"]\\` in the separation call.\n3. Narrow the energy range with \\`filter_scans(nce_in: [20, 25, 30])\\`\n   \u2014 a narrower range is an easier test, and an honest report says so.\n4. Lower \\`min_ratio\\` to 1.0 and ask whether the weaker criterion is one\n   you would have declared beforehand.\n\nIf you find a configuration that passes, the next question is whether you\nwould have chosen it before seeing the data.\n\\";

/* ── Experiment 0: does the address survive collision energy? ───────────── */

const SS_EXP0_INVARIANCE = `\
// Experiment 0 — NCE invariance of the S-entropy address.
//
// CLAIM UNDER TEST: the address (Sk, St, Se) is a property of the
// COMPOUND, not of the spectrum. If true, changing collision energy at
// fixed compound should move the address LESS than changing compound.
//
// The criterion is declared here, before the run. The program can fail.

import lavoisier.acquire
import lavoisier.transform
import lavoisier.analyse

objective NCEInvariance:
    target: "does the S-entropy address survive collision-energy variation"
    criterion: "separation_ratio > 2.0 and |r| < 0.3 for each axis"

phase Read:
    // 60 analytes x 9 collision energies, HCD-like fragment ladders
    scans = lavoisier.acquire.library(analytes: 60, min_peaks: 3)

phase Transform:
    coords = lavoisier.transform.sentropy(
        scans: scans,
        alpha: 1.0,
        beta: 1.0,
        k_neighbors: 5
    )

phase Test:
    // Primary: within-compound spread vs between-compound spread
    separation = lavoisier.analyse.separation(
        coords: coords,
        key: "compound",
        min_group: 9
    )

    // Secondary: does any axis track collision energy?
    drift = lavoisier.analyse.drift(coords: coords, over: "nce")

phase Score:
    // The verdict, computed against the criterion declared above.
    verdict = lavoisier.analyse.score(
        separation: separation,
        drift: drift,
        min_ratio: 2.0,
        max_abs_r: 0.3
    )
`;

const SS_EXP0_CONTROLS = `\
// Controls — what makes a negative result interpretable.
//
// A criterion that fails tells you nothing unless you can show the
// apparatus WOULD have detected the effect had it been there.
//   Negative control : shuffle the labels; the ratio must collapse.
//   Comparison method: raw cosine similarity, the established practice.

import lavoisier.acquire
import lavoisier.transform
import lavoisier.analyse

objective ExperimentControls:
    target: "show the pipeline discriminates, and score it against practice"

phase Read:
    scans  = lavoisier.acquire.library(analytes: 40, min_peaks: 3)
    coords = lavoisier.transform.sentropy(scans: scans, alpha: 1.0, beta: 1.0)

phase Measured:
    measured = lavoisier.analyse.separation(coords: coords, min_group: 9)

phase NegativeControl:
    // Permute compound labels. If the measured ratio is real, this must
    // fall well below it. If it does not, the statistic is measuring noise.
    shuffled = lavoisier.analyse.shuffle_control(
        coords: coords,
        min_group: 9,
        seed: 20260819
    )

phase ComparisonMethod:
    // Cosine similarity of raw peak lists across collision energy.
    // Known to be stable between adjacent levels and to degrade across
    // distant ones — the behaviour the coordinates were meant to improve on.
    baseline = lavoisier.analyse.baseline(
        scans: scans,
        tolerance: 0.01,
        min_group: 9,
        max_groups: 30
    )
`;

const SS_EXP0_SWEEP = `\
// Parameter sweep — is the outcome just an unlucky parameter choice?
//
// alpha, beta and k are analyst choices. The language requires them to
// be stated in the source but offers no procedure for selecting them.
// So the honest question after a failed criterion is whether some other
// setting would have passed.

import lavoisier.acquire
import lavoisier.transform
import lavoisier.analyse

objective ParameterDependence:
    target: "does any (alpha, beta, k) setting meet the criterion"

phase Read:
    scans = lavoisier.acquire.library(analytes: 40, min_peaks: 3)

phase Sweep:
    // Every combination is scored by the same separation statistic.
    sweep = lavoisier.analyse.sweep(
        scans: scans,
        alphas: [0.0, 0.5, 1.0, 2.0, 4.0],
        betas:  [0.5, 1.0, 2.0],
        k_neighbors: 5,
        min_group: 9
    )

phase Compare:
    // The declared setting, for reference against the sweep's best.
    coords     = lavoisier.transform.sentropy(scans: scans, alpha: 1.0, beta: 1.0)
    declared   = lavoisier.analyse.separation(coords: coords, min_group: 9)
`;

const SS_EXP0_MECHANISM = `\
// Mechanism — WHY the evolution coordinate tracks collision energy.
//
// Se is the Shannon entropy of the local intensity neighbourhood.
// Raising collision energy adds fragments of comparable intensity,
// which raises that entropy directly. The correlation is not a bug in
// the implementation; it is what the definition computes.
//
// Run this at low and high energy separately and compare.

import lavoisier.acquire
import lavoisier.transform
import lavoisier.analyse

objective FragmentationMechanism:
    target: "trace Se to fragment count, not to compound identity"

phase Read:
    scans = lavoisier.acquire.library(analytes: 50, min_peaks: 3)

phase LowEnergy:
    low        = lavoisier.acquire.filter_scans(scans: scans, nce_in: [10, 15, 20])
    low_coords = lavoisier.transform.sentropy(scans: low, alpha: 1.0, beta: 1.0)

phase HighEnergy:
    high        = lavoisier.acquire.filter_scans(scans: scans, nce_in: [50, 60, 80])
    high_coords = lavoisier.transform.sentropy(scans: high, alpha: 1.0, beta: 1.0)

phase Drift:
    // Across the full range, the trend is monotone in Se and opposite in Sk.
    all_coords = lavoisier.transform.sentropy(scans: scans, alpha: 1.0, beta: 1.0)
    trend      = lavoisier.analyse.drift(coords: all_coords, over: "nce")
`;

const initialFiles = {
  examples: {
    type: "folder",
    children: {
      "hello_lipid.ss":           { type: "file", lang: "ss", content: SS_HELLO_LIPID },
      "targeted_lipidomics.ss":   { type: "file", lang: "ss", content: SS_TARGETED_LIPID },
      "proteomics_experiment.ss": { type: "file", lang: "ss", content: SS_PROTEOMICS },
      "temporal_acquisition.ss":  { type: "file", lang: "ss", content: SS_TEMPORAL },
      "partition_addresses.ss":   { type: "file", lang: "ss", content: SS_PARTITION },
      "sentropy_observation.ss":  { type: "file", lang: "ss", content: SS_SENTROPY_OBS },
      "force_free_ms.ss":         { type: "file", lang: "ss", content: SS_FORCE_FREE },
      "generative_database.ss":   { type: "file", lang: "ss", content: SS_GENERATIVE_DB },
      "sebd_ms_search.ss":        { type: "file", lang: "ss", content: SS_SEBD_SEARCH },
      "virtual_tensor.ss":        { type: "file", lang: "ss", content: SS_VIRTUAL_TENSOR },
      "db_search.ss":             { type: "file", lang: "ss", content: SS_DB_SEARCH },
    },
  },
  experiments: {
    type: "folder",
    children: {
      "exp0_invariance.ss": { type: "file", lang: "ss", content: SS_EXP0_INVARIANCE },
      "exp0_controls.ss":   { type: "file", lang: "ss", content: SS_EXP0_CONTROLS },
      "exp0_sweep.ss":      { type: "file", lang: "ss", content: SS_EXP0_SWEEP },
      "exp0_mechanism.ss":  { type: "file", lang: "ss", content: SS_EXP0_MECHANISM },
      "TUTORIAL.md":        { type: "file", lang: "md", content: TUTORIAL_CONTENT },
    },
  },
  "README.md": { type: "file", lang: "md", content: README_CONTENT },
};

/* ─── Sandbox staged-run helper (compile → execute) ──────────────────────── */

/**
 * Read the active .ss source. Returns null if no runnable file is open.
 */
function activeSource(files, activePathArr) {
  const node = activePathArr ? getNode(files, activePathArr) : null;
  if (!node?.content) return null;
  const name = activePathArr[activePathArr.length - 1];
  if (!name.endsWith(".ss")) return null;
  return node.content;
}

/* ─── Visualization ──────────────────────────────────────────────────────── */

function PartitionScatter({ records }) {
  if (!records.length) return (
    <div className="flex items-center justify-center text-[11px]" style={{ height: 150, color: "#555" }}>
      no records
    </div>
  );
  const W = 380, H = 150, PL = 32, PR = 8, PT = 8, PB = 28;
  const pW = W - PL - PR, pH = H - PT - PB;
  const mzMin = Math.min(...records.map(r => r.precursorMz));
  const mzMax = Math.max(...records.map(r => r.precursorMz));
  const nMax  = Math.max(...records.map(r => r.n), 1);
  const toX   = mz => PL + ((mz - mzMin) / (mzMax - mzMin || 1)) * pW;
  const toY   = n  => H - PB - ((n - 1) / nMax) * pH;
  const sample = records.length > 400
    ? records.filter((_, i) => i % Math.ceil(records.length / 400) === 0)
    : records;
  const axes = [
    { x1: PL, y1: H - PB, x2: W - PR, y2: H - PB },
    { x1: PL, y1: PT,     x2: PL,     y2: H - PB },
  ];
  const nTicks = [1, 2, 3, 4, 5].filter(v => v <= nMax + 1);
  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ fontFamily: "monospace", display: "block" }}>
      {axes.map((a, i) => <line key={i} {...a} stroke="#3a3a3a" />)}
      {nTicks.map(v => (
        <g key={v}>
          <line x1={PL - 3} y1={toY(v)} x2={PL} y2={toY(v)} stroke="#3a3a3a" />
          <text x={PL - 5} y={toY(v) + 4} textAnchor="end" fontSize={8} fill="#666">{v}</text>
        </g>
      ))}
      <text x={PL - 14} y={H / 2 + 4} fontSize={8} fill="#555"
        transform={`rotate(-90,${PL - 14},${H / 2})`}>n</text>
      <text x={W / 2} y={H - 4} textAnchor="middle" fontSize={8} fill="#555">m/z</text>
      {sample.map((r, i) => (
        <circle key={i} cx={toX(r.precursorMz)} cy={toY(r.n)} r={1.8}
          fill={CLASS_COLORS[r.analyteClass] || "#555"} opacity={0.55} />
      ))}
    </svg>
  );
}

function RecordsPanel({ records, summary }) {
  const classes = Object.keys(summary.perClass || {});
  const top = records.slice(0, 20);
  return (
    <div className="flex flex-col gap-3 overflow-y-auto" style={{ padding: 12, height: "100%" }}>
      {/* Stats */}
      <div className="grid grid-cols-3 gap-2">
        {[
          ["Records", summary.count],
          ["Classes", classes.length],
          ["m/z range", `${summary.mzRange?.[0]?.toFixed(0)}–${summary.mzRange?.[1]?.toFixed(0)}`],
        ].map(([label, val]) => (
          <div key={label} className="rounded p-2 text-center"
            style={{ background: "#2a2d2e", border: `1px solid ${T.border}` }}>
            <div className="text-[9px] uppercase tracking-wider" style={{ color: "#666" }}>{label}</div>
            <div className="font-mono text-[13px]" style={{ color: T.editorFg }}>{val}</div>
          </div>
        ))}
      </div>
      {/* Class legend */}
      <div className="flex flex-wrap gap-1.5">
        {classes.map(cls => (
          <span key={cls} className="flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px]"
            style={{ background: "#2a2d2e", border: `1px solid ${T.border}` }}>
            <span className="h-2 w-2 rounded-full shrink-0"
              style={{ background: CLASS_COLORS[cls] || "#555" }} />
            <span style={{ color: T.sidebarFg }}>{cls}</span>
            <span style={{ color: "#666" }}>{summary.perClass[cls]}</span>
          </span>
        ))}
      </div>
      {/* Scatter */}
      <PartitionScatter records={records} />
      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full font-mono text-[10px]" style={{ borderCollapse: "collapse" }}>
          <thead>
            <tr style={{ color: "#666", borderBottom: `1px solid ${T.border}` }}>
              {["Analyte", "Class", "m/z", "Adduct", "n", "ℓ", "Sk"].map(h => (
                <th key={h} className="py-1 pr-3 text-left font-normal">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {top.map((r, i) => (
              <tr key={i} style={{ borderBottom: `1px solid #2a2a2a`, color: T.editorFg }}>
                <td className="py-0.5 pr-3 truncate" style={{ maxWidth: 120 }}>{r.analyte}</td>
                <td className="pr-3" style={{ color: CLASS_COLORS[r.analyteClass] || "#555" }}>{r.analyteClass}</td>
                <td className="pr-3">{r.precursorMz.toFixed(3)}</td>
                <td className="pr-3" style={{ color: "#9cdcfe" }}>{r.adduct}</td>
                <td className="pr-3">{r.n}</td>
                <td className="pr-3">{r.l}</td>
                <td className="pr-3">{r.sentropyVec?.sk?.toFixed(3) ?? "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {records.length > 20 && (
          <div className="pt-1 text-[10px]" style={{ color: "#555" }}>
            + {records.length - 20} more records
          </div>
        )}
      </div>
    </div>
  );
}

function CellsPanel({ cells }) {
  return (
    <div className="overflow-y-auto" style={{ padding: 12, height: "100%" }}>
      <div className="mb-3 text-[10px] uppercase tracking-wider" style={{ color: "#666" }}>
        ΔP Cell Registry — {cells.length} cells
      </div>
      <table className="w-full font-mono text-[10px]" style={{ borderCollapse: "collapse" }}>
        <thead>
          <tr style={{ color: "#666", borderBottom: `1px solid ${T.border}` }}>
            {["Target", "m/z", "±Da", "ω (Hz)", "ΔP lo", "ΔP hi", "τ_min ms"].map(h => (
              <th key={h} className="py-1 pr-3 text-left font-normal">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {cells.map((c, i) => (
            <tr key={i} style={{ borderBottom: `1px solid #2a2a2a`, color: T.editorFg }}>
              <td className="py-0.5 pr-3" style={{ color: "#9cdcfe" }}>{c.name}</td>
              <td className="pr-3">{c.mz}</td>
              <td className="pr-3" style={{ color: "#dcdcaa" }}>±{c.window_da}</td>
              <td className="pr-3" style={{ color: "#b388eb" }}>{c.omega_hz}</td>
              <td className="pr-3" style={{ color: "#e07a7a" }}>{c.dp_lo}</td>
              <td className="pr-3" style={{ color: "#7cc77c" }}>{c.dp_hi}</td>
              <td className="pr-3" style={{ color: "#e6a456" }}>{c.tau_min_ms}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function AddressesPanel({ addresses }) {
  return (
    <div className="overflow-y-auto" style={{ padding: 12, height: "100%" }}>
      <div className="mb-3 text-[10px] uppercase tracking-wider" style={{ color: "#666" }}>
        Partition Addresses — Φ: ℤ⁺ → 𝒫
      </div>
      <table className="w-full font-mono text-[10px]" style={{ borderCollapse: "collapse" }}>
        <thead>
          <tr style={{ color: "#666", borderBottom: `1px solid ${T.border}` }}>
            {["Name", "Mass (Da)", "Adduct", "n", "ℓ", "m", "s"].map(h => (
              <th key={h} className="py-1 pr-3 text-left font-normal">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {addresses.map((a, i) => (
            <tr key={i} style={{ borderBottom: `1px solid #2a2a2a`, color: T.editorFg }}>
              <td className="py-0.5 pr-3" style={{ color: "#9cdcfe" }}>{a.name}</td>
              <td className="pr-3">{a.mass}</td>
              <td className="pr-3" style={{ color: "#dcdcaa" }}>{a.adduct}</td>
              <td className="pr-3" style={{ color: "#5fa8d3" }}>{a.n}</td>
              <td className="pr-3" style={{ color: "#7cc77c" }}>{a.l}</td>
              <td className="pr-3" style={{ color: "#e07a7a" }}>{a.m}</td>
              <td className="pr-3" style={{ color: "#b388eb" }}>{a.s > 0 ? "+½" : "−½"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ─── Generic value visualisations for the workspace ─────────────────────── */

function Card({ label, value, color = T.editorFg }) {
  return (
    <div className="rounded p-2"
      style={{ background: "#2a2d2e", border: `1px solid ${T.border}` }}>
      <div className="text-[9px] uppercase tracking-wider mb-0.5" style={{ color: "#666" }}>{label}</div>
      <div className="font-mono text-[12px] break-words" style={{ color }}>{value}</div>
    </div>
  );
}

/** Horizontal mini-bar chart from {label, value} rows in [0,1] or normalised. */
function MiniBars({ rows, max }) {
  const m = max ?? Math.max(...rows.map(r => Math.abs(r.value)), 1e-9);
  return (
    <div className="space-y-1">
      {rows.map((r, i) => (
        <div key={i} className="flex items-center gap-2 text-[10px]">
          <span className="w-24 shrink-0 truncate font-mono" style={{ color: T.sidebarFg }}>{r.label}</span>
          <div className="flex-1 h-2 rounded-full overflow-hidden" style={{ background: "#2a2d2e" }}>
            <div className="h-full rounded-full"
              style={{ width: `${Math.min(100, (Math.abs(r.value) / m) * 100)}%`, background: r.color || "#5fa8d3" }} />
          </div>
          <span className="w-16 text-right font-mono" style={{ color: "#888" }}>
            {typeof r.value === "number" ? r.value.toFixed(3) : r.value}
          </span>
        </div>
      ))}
    </div>
  );
}

function Section({ title, children }) {
  return (
    <div className="space-y-1.5">
      <div className="text-[9px] uppercase tracking-[0.15em]" style={{ color: "#666" }}>{title}</div>
      {children}
    </div>
  );
}

/** S-entropy coordinate as a triangular ternary plot + cards. */
function SentropyView({ value }) {
  const { sk = 0, st = 0, se = 0 } = value;
  return (
    <Section title="S-entropy coordinates (Sₖ, Sₜ, Sₑ)">
      <div className="grid grid-cols-3 gap-2">
        <Card label="Sₖ knowledge" value={sk.toFixed(4)} color="#22d3ee" />
        <Card label="Sₜ temporal"  value={st.toFixed(4)} color="#fbbf24" />
        <Card label="Sₑ evolution" value={se.toFixed(4)} color="#a78bfa" />
      </div>
      <MiniBars max={1} rows={[
        { label: "Sₖ", value: sk, color: "#22d3ee" },
        { label: "Sₜ", value: st, color: "#fbbf24" },
        { label: "Sₑ", value: se, color: "#a78bfa" },
      ]} />
    </Section>
  );
}

/** Dual-path validation: convergence + false-positive probability. */
function ValidationView({ value }) {
  const cp = value.commonPrefixLen ?? 0;
  const conv = value.convergenceScore ?? 0;
  return (
    <Section title="Ion-droplet dual-path validation">
      <div className="grid grid-cols-3 gap-2">
        <Card label="Common prefix" value={`${cp} trits`} color="#9cdcfe" />
        <Card label="Convergence" value={`${(conv * 100).toFixed(1)}%`}
          color={conv > 0.7 ? "#34d399" : conv > 0.4 ? "#dcdcaa" : "#f48771"} />
        <Card label="False-pos ≤" value={(value.falsePosProb ?? 1).toExponential(2)} color="#fb923c" />
      </div>
      {value.ionAddress && (
        <div className="font-mono text-[9px]" style={{ color: "#666" }}>
          ion:&nbsp;&nbsp;&nbsp;<span style={{ color: "#5fa8d3" }}>{value.ionAddress}</span><br />
          drip:&nbsp;&nbsp;<span style={{ color: "#e07a7a" }}>{value.dropletAddress}</span>
        </div>
      )}
    </Section>
  );
}

/** Purpose domain: reduction ratio + prefix count. */
function DomainView({ value }) {
  const rho = value.reductionPct ?? (value.reductionRatio != null ? value.reductionRatio * 100 : null);
  const prefixes = value.prefixes?.length ?? 0;
  return (
    <Section title={`Purpose domain — ${value.label || value.name || "context"}`}>
      <div className="grid grid-cols-2 gap-2">
        <Card label="Prefixes" value={prefixes} color="#9cdcfe" />
        {rho != null && <Card label="Reduction" value={`${rho.toFixed(1)}%`} color="#34d399" />}
      </div>
      {value.bounds && (
        <MiniBars max={1} rows={[
          { label: "Sₖ range", value: value.bounds.sk[1] - value.bounds.sk[0], color: "#22d3ee" },
          { label: "Sₜ range", value: value.bounds.st[1] - value.bounds.st[0], color: "#fbbf24" },
          { label: "Sₑ range", value: value.bounds.se[1] - value.bounds.se[0], color: "#a78bfa" },
        ]} />
      )}
    </Section>
  );
}

/** Subharmonic frequency ratios as a bar chart. */
function SubharmonicsView({ value }) {
  const rows = value.slice(0, 16).map(s => ({
    label: `${s.fragmentMz?.toFixed(1)}`,
    value: s.frequencyRatio ?? 0,
    color: s.selfConsistent ? "#34d399" : "#f48771",
  }));
  return (
    <Section title={`Fragment subharmonics — ω_f / ω_prec (${value.length})`}>
      <MiniBars rows={rows} />
      <div className="text-[9px]" style={{ color: "#666" }}>
        Self-consistent: {value.filter(s => s.selfConsistent).length}/{value.length} (&lt;10⁻⁶ ppm)
      </div>
    </Section>
  );
}

/** Virtual tensor report: off-shell fraction + mean recovery. */
function TensorReportView({ value }) {
  const v = value.verified || {};
  return (
    <Section title="Virtual partition tensor V_{ijkl}">
      <div className="grid grid-cols-3 gap-2">
        <Card label="Components" value={(value.tensor?.length ?? 0).toLocaleString()} color="#9cdcfe" />
        <Card label="Off-shell" value={`${((v.offShellFraction ?? 0) * 100).toFixed(1)}%`} color="#fb923c" />
        <Card label="Planck depth" value={value.planckDepth ?? "—"} color="#a78bfa" />
      </div>
      <MiniBars max={1} rows={[
        { label: "mean (recov.)", value: v.mean ?? 0, color: "#34d399" },
        { label: "v_phys", value: v.vPhys ?? 0, color: "#5fa8d3" },
      ]} />
      <div className="text-[9px]" style={{ color: v.meanRecoveryHolds ? "#6a9955" : "#f48771" }}>
        mean-recovery {v.meanRecoveryHolds ? "✓ holds" : "✗ violated"} · d_eff = {(value.dEff ?? 0).toLocaleString()}
      </div>
    </Section>
  );
}

function ImpossibleView({ value }) {
  return (
    <Section title={`Impossible ions — crossing-symmetry probes (${value.length})`}>
      <table className="w-full font-mono text-[10px]" style={{ borderCollapse: "collapse" }}>
        <thead><tr style={{ color: "#666", borderBottom: `1px solid ${T.border}` }}>
          {["ion 1", "ion 2", "impossible m/z"].map(h => <th key={h} className="py-1 pr-3 text-left font-normal">{h}</th>)}
        </tr></thead>
        <tbody>
          {value.slice(0, 12).map((p, i) => (
            <tr key={i} style={{ borderBottom: "1px solid #2a2a2a", color: T.editorFg }}>
              <td className="pr-3">{p.ion1_mz?.toFixed(3)}</td>
              <td className="pr-3">{p.ion2_mz?.toFixed(3)}</td>
              <td className="pr-3" style={{ color: "#fb923c" }}>{p.impossibleMz?.toFixed(3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Section>
  );
}

function TransientView({ value }) {
  return (
    <Section title="Single-transient contents (Theorem 11.1)">
      <div className="grid grid-cols-2 gap-2">
        <Card label="Precursor freq" value={`${value.precursor?.freq_Hz?.toExponential(2)} Hz`} color="#5fa8d3" />
        <Card label="Fragment subharmonics" value={value.fragments?.length ?? 0} color="#7cc77c" />
        <Card label="Charge states" value={value.chargeStates?.length ?? 0} color="#a78bfa" />
        <Card label="Polarity Δφ" value="π" color="#e07a7a" />
      </div>
    </Section>
  );
}

function ComplementView({ value }) {
  return (
    <Section title="Partition complement (SWIFT antistate)">
      <div className="grid grid-cols-2 gap-2">
        <Card label="Original m/z" value={value.originalMz?.toFixed(3)} color="#5fa8d3" />
        <Card label="Complement m/z" value={value.complementMz?.toFixed(3)} color="#e07a7a" />
        <Card label="M_ion" value={value.M_ion} />
        <Card label="C_max" value={value.Cmax} />
      </div>
    </Section>
  );
}

function ScalarView({ name, value }) {
  return (
    <Section title={name}>
      <Card label="value"
        value={typeof value === "string" ? value : JSON.stringify(value)} />
    </Section>
  );
}

/** Online DB search results (resolved asynchronously by the sandbox). */
function DbSearchView({ value }) {
  if (value.__async && !value.resolved) {
    return (
      <Section title={`Database search — ${value.__fn}`}>
        <div className="flex items-center gap-2 text-[11px]" style={{ color: "#dcdcaa" }}>
          <RefreshCw size={13} className="animate-spin" />
          querying {(value.dbs || ["massbank"]).join(", ")} for m/z {value.precMz?.toFixed(4)}…
        </div>
      </Section>
    );
  }
  const hits = value.hits || [];
  return (
    <Section title={`Database search — ${hits.length} hit(s)`}>
      {value.summary && (
        <div className="flex flex-wrap gap-1.5 mb-1">
          {Object.entries(value.summary).map(([db, s]) => (
            <span key={db} className="rounded px-1.5 py-0.5 text-[9px]"
              style={{ background: "#2a2d2e", border: `1px solid ${T.border}`,
                       color: s.error ? "#f48771" : "#9cdcfe" }}>
              {db}: {s.error ? "error" : `${s.count} hit(s)`}
            </span>
          ))}
        </div>
      )}
      {hits.length === 0
        ? <div className="text-[10px]" style={{ color: "#666" }}>
            No hits (public APIs may be unavailable from the browser due to CORS).
          </div>
        : (
          <table className="w-full font-mono text-[10px]" style={{ borderCollapse: "collapse" }}>
            <thead><tr style={{ color: "#666", borderBottom: `1px solid ${T.border}` }}>
              {["Compound", "m/z", "Formula", "Score", "DB"].map(h =>
                <th key={h} className="py-1 pr-3 text-left font-normal">{h}</th>)}
            </tr></thead>
            <tbody>
              {hits.slice(0, 20).map((h, i) => (
                <tr key={i} style={{ borderBottom: "1px solid #2a2a2a", color: T.editorFg }}>
                  <td className="py-0.5 pr-3 truncate" style={{ maxWidth: 160, color: "#9cdcfe" }}>{h.name}</td>
                  <td className="pr-3">{Number(h.precursorMz)?.toFixed?.(3) ?? h.precursorMz}</td>
                  <td className="pr-3" style={{ color: "#dcdcaa" }}>{h.formula}</td>
                  <td className="pr-3" style={{ color: "#34d399" }}>{Number(h.score)?.toFixed?.(2) ?? h.score}</td>
                  <td className="pr-3" style={{ color: "#666" }}>{h.database}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
    </Section>
  );
}

/** Render a single workspace variable by its classified kind. */
function WorkspaceValue({ entry }) {
  const { name, kind, value } = entry;
  switch (kind) {
    case "cells":        return <CellsPanel cells={value} />;
    case "addresses":    return <AddressesPanel addresses={value} />;
    case "sentropy":     return <SentropyView value={value} />;
    case "validation":   return <ValidationView value={value} />;
    case "domain":       return <DomainView value={value} />;
    case "subharmonics": return <SubharmonicsView value={value} />;
    case "tensorReport": return <TensorReportView value={value} />;
    case "impossible":   return <ImpossibleView value={value} />;
    case "transient":    return <TransientView value={value} />;
    case "complement":   return <ComplementView value={value} />;
    case "pending":      return <DbSearchView value={value} />;
    case "string":
    case "number":
    case "scalar":       return <ScalarView name={name} value={value} />;
    default:
      return (
        <Section title={`${name} (${kind})`}>
          <pre className="overflow-auto rounded p-2 font-mono text-[10px]"
            style={{ background: "#1a1c1e", color: T.editorFg, maxHeight: 220 }}>
            {JSON.stringify(value, null, 2)}
          </pre>
        </Section>
      );
  }
}

/* ─── Results panel: dashboard for records, workspace grid otherwise ──────── */
function ResultsPanel({ result, workspace }) {
  const hasRecords = (workspace || []).some(w => w.kind === "records");

  if ((!result || result.type === "empty") && (!workspace || workspace.length === 0)) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-2 text-[12px]"
        style={{ color: "#444" }}>
        <Zap size={24} opacity={0.3} />
        <span>Run a .ss script to see results</span>
      </div>
    );
  }

  // Records → the FULL crossfilter dashboard (same as Experiment page),
  // populated from the store. Any non-record workspace vars render below it.
  if (hasRecords) {
    const extras = (workspace || []).filter(w => w.kind !== "records");
    return (
      <div className="h-full overflow-y-auto" style={{ background: "#070809", padding: 12 }}>
        <ResultsDashboard />
        {extras.length > 0 && (
          <div className="mt-4 space-y-4">
            {extras.map((w, i) => <WorkspaceValue key={i} entry={w} />)}
          </div>
        )}
      </div>
    );
  }

  // No records: render every workspace variable as its own comprehensible panel.
  return (
    <div className="h-full overflow-y-auto space-y-5" style={{ padding: 14 }}>
      {(workspace || []).map((w, i) => (
        <WorkspaceValue key={i} entry={w} />
      ))}
      {(!workspace || workspace.length === 0) && result?.data != null && (
        <pre className="overflow-auto p-3 font-mono text-[11px]" style={{ color: T.editorFg }}>
          {JSON.stringify(result.data, null, 2)}
        </pre>
      )}
    </div>
  );
}

/* ─── Terminal stream renderer ───────────────────────────────────────────── */
function TerminalView({ term }) {
  const endRef = useRef(null);
  useEffect(() => { endRef.current?.scrollIntoView({ block: "end" }); }, [term]);

  const streamStyle = (stream) => {
    if (stream === "stderr") return { color: "#f48771" };
    if (stream === "stage")  return { color: "#4ec9b0", fontWeight: 600 };
    return { color: "#d4d4d4" };
  };

  return (
    <div className="h-full overflow-y-auto p-2 font-mono text-[12px] leading-[1.55]">
      {term.length === 0 ? (
        <div className="px-1 pt-1" style={{ color: "#5a5a5a" }}>
          Press <b>Compile</b> then <b>Run</b> (or ▶ Run) to execute the active script.
        </div>
      ) : (
        term.map((l, i) => {
          if (l.stream === "stage") {
            return (
              <div key={i} className="mt-1 flex items-center gap-2">
                <span style={{ color: "#ce9178" }}>$</span>
                <span style={streamStyle("stage")}>{l.text}</span>
              </div>
            );
          }
          return (
            <div key={i} className="px-1 whitespace-pre-wrap break-words" style={streamStyle(l.stream)}>
              {l.text}
            </div>
          );
        })
      )}
      <div ref={endRef} />
    </div>
  );
}

/* ─── Output column ──────────────────────────────────────────────────────── */
function OutputColumn({ result, workspace, ir, logs, term, running, onCompile, onRun, onClear }) {
  const [tab, setTab] = useState("charts");
  // Records from the workspace drive the Charts tab directly.
  const recordEntry = (workspace || []).find(w => w.kind === "records");
  const records = recordEntry ? recordEntry.value : [];

  const tabs = [
    { id: "charts",   label: "Charts",   Icon: Cpu },
    { id: "results",  label: "Results",  Icon: Cpu },
    { id: "terminal", label: "Terminal", Icon: TerminalIcon },
    { id: "console",  label: "Console",  Icon: TerminalIcon },
    { id: "ir",       label: "IR",       Icon: Code2 },
  ];
  const levelColor = { log: "#d4d4d4", info: "#9cdcfe", warn: "#dcdcaa", error: "#f48771" };
  const errCount = (term || []).filter(l => l.stream === "stderr").length;

  return (
    <div className="flex min-w-0 flex-1 flex-col"
      style={{ background: T.editor, borderLeft: `1px solid ${T.border}` }}>
      <div className="flex h-9 shrink-0 items-center justify-between pr-2"
        style={{ background: T.tabInactive }}>
        <div className="flex h-full">
          {tabs.map(({ id, label, Icon }) => {
            const active = tab === id;
            const badge = id === "console" ? logs.length
                        : id === "terminal" ? errCount : 0;
            return (
              <button key={id} onClick={() => setTab(id)}
                className="relative flex items-center gap-1.5 px-3 text-[12px] transition-colors"
                style={{ color: active ? T.tabFgActive : T.tabFg, background: active ? T.tabActive : "transparent" }}>
                <Icon size={13} />{label}
                {badge > 0 && (
                  <span className="rounded-full px-1.5 text-[10px]"
                    style={{ background: id === "terminal" ? "#a82d2d" : T.accent, color: "#fff" }}>{badge}</span>
                )}
                {active && <span className="absolute left-0 top-0 h-0.5 w-full"
                  style={{ background: T.accentBright }} />}
              </button>
            );
          })}
        </div>
        <div className="flex items-center gap-1">
          {(tab === "console" || tab === "terminal") && (
            <button onClick={onClear} title="Clear" className="flex h-6 w-6 items-center justify-center rounded"
              style={{ color: T.tabFg }}><Trash2 size={14} /></button>
          )}
          <button onClick={onCompile} disabled={running} title="Parse + type-check only"
            className="flex h-6 items-center gap-1 rounded px-2 text-[12px] disabled:opacity-40"
            style={{ background: "transparent", color: T.tabFgActive, border: `1px solid ${T.border}` }}>
            <Hammer size={12} />Compile
          </button>
          <button onClick={onRun} disabled={running}
            className="flex h-6 items-center gap-1 rounded px-2 text-[12px] disabled:opacity-40"
            style={{ background: running ? "#555" : T.accent, color: "#fff" }}>
            {running ? <Square size={11} /> : <Play size={12} />}
            {running ? "Running…" : "Run"}
          </button>
        </div>
      </div>

      <div className="min-h-0 flex-1">
        {tab === "charts" && <SandboxCharts records={records} />}
        {tab === "results" && <ResultsPanel result={result} workspace={workspace} />}
        {tab === "terminal" && <TerminalView term={term || []} />}
        {tab === "console" && (
          <div className="h-full overflow-y-auto p-2 font-mono text-[12px] leading-relaxed">
            {logs.length === 0
              ? <div className="px-1 pt-1" style={{ color: "#5a5a5a" }}>Execution log appears here.</div>
              : logs.map((l, i) => (
                  <div key={i} className="border-b px-1 py-0.5"
                    style={{ color: levelColor[l.level] || "#d4d4d4", borderColor: "#2a2a2a" }}>
                    <span className="mr-2 opacity-50">{l.level}</span>{l.message}
                  </div>
                ))}
          </div>
        )}
        {tab === "ir" && (
          <pre className="h-full overflow-auto p-3 font-mono text-[11px] leading-[1.5]"
            style={{ color: T.editorFg }}>{ir || "No IR — Compile a .ss file first"}</pre>
        )}
      </div>
    </div>
  );
}

/* ─── Helpers ────────────────────────────────────────────────────────────── */
const getNode = (tree, path) => {
  let n = { children: tree };
  for (const p of path) { n = n.children?.[p]; if (!n) return null; }
  return n;
};

const fileIcon = (name) => {
  if (name.endsWith(".ss"))   return { Icon: FileCode2, color: "#c586c0" };
  if (name.endsWith(".json")) return { Icon: FileJson,  color: "#cbcb41" };
  if (name.endsWith(".md"))   return { Icon: FileText,  color: "#519aba" };
  return { Icon: FileText, color: "#858585" };
};

const langLabel = (lang) => ({
  ss: "Shapeshifter", json: "JSON", md: "Markdown",
}[lang] || "Plain Text");

/* ─── File tree ──────────────────────────────────────────────────────────── */
function Tree({ tree, path = [], depth = 0, expanded, toggle, activePath, openFile }) {
  const entries = Object.entries(tree).sort((a, b) =>
    a[1].type !== b[1].type ? (a[1].type === "folder" ? -1 : 1) : a[0].localeCompare(b[0]));
  return (
    <>
      {entries.map(([name, node]) => {
        const fullPath = [...path, name];
        const key = fullPath.join("/");
        const isFolder = node.type === "folder";
        const isOpen = expanded.has(key);
        const isActive = activePath === key;
        const { Icon, color } = isFolder
          ? { Icon: isOpen ? FolderOpen : Folder, color: "#90a4ae" }
          : fileIcon(name);
        return (
          <div key={key}>
            <button
              onClick={() => isFolder ? toggle(key) : openFile(fullPath)}
              className="flex w-full items-center gap-1 py-0.5 pr-2 text-left text-[13px] leading-relaxed"
              style={{
                paddingLeft: 8 + depth * 12,
                color: T.sidebarFg,
                background: isActive ? T.lineActive : "transparent",
              }}
              onMouseEnter={e => { if (!isActive) e.currentTarget.style.background = "#2a2d2e"; }}
              onMouseLeave={e => { if (!isActive) e.currentTarget.style.background = "transparent"; }}
            >
              {isFolder
                ? (isOpen ? <ChevronDown size={14} className="shrink-0 opacity-70" />
                          : <ChevronRight size={14} className="shrink-0 opacity-70" />)
                : <span className="w-[14px] shrink-0" />}
              <Icon size={15} className="shrink-0" style={{ color }} />
              <span className="truncate">{name}</span>
            </button>
            {isFolder && isOpen && (
              <Tree tree={node.children} path={fullPath} depth={depth + 1}
                expanded={expanded} toggle={toggle} activePath={activePath} openFile={openFile} />
            )}
          </div>
        );
      })}
    </>
  );
}

/* ─── Editor ─────────────────────────────────────────────────────────────── */
function Editor({ value, onChange, onCursor }) {
  const gutterRef = useRef(null);
  const lines = value.split("\n");
  const syncScroll = e => { if (gutterRef.current) gutterRef.current.scrollTop = e.target.scrollTop; };
  const handleCursor = e => {
    const upto = e.target.value.slice(0, e.target.selectionStart);
    onCursor({ ln: upto.split("\n").length, col: upto.length - upto.lastIndexOf("\n") });
  };
  return (
    <div className="flex min-h-0 flex-1" style={{ background: T.editor }}>
      <div ref={gutterRef}
        className="select-none overflow-hidden py-3 text-right font-mono text-[13px] leading-[1.5]"
        style={{ color: T.gutter, minWidth: 52, paddingRight: 16 }}>
        {lines.map((_, i) => <div key={i}>{i + 1}</div>)}
      </div>
      <textarea
        value={value} onChange={e => onChange(e.target.value)}
        onScroll={syncScroll} onKeyUp={handleCursor} onClick={handleCursor}
        spellCheck={false}
        className="min-h-0 flex-1 resize-none border-0 bg-transparent py-3 pr-4 font-mono text-[13px] leading-[1.5] outline-none"
        style={{ color: T.editorFg, tabSize: 2, caretColor: "#fff" }}
      />
    </div>
  );
}

/* ─── Main shell ─────────────────────────────────────────────────────────── */
export default function ShapeshifterSandbox() {
  const [files, setFiles]       = useState(initialFiles);
  const [expanded, setExpanded] = useState(new Set(["examples"]));
  const [openTabs, setOpenTabs] = useState([["examples", "hello_lipid.ss"]]);
  const [activeTab, setActiveTab] = useState("examples/hello_lipid.ss");
  const [dirty, setDirty]       = useState(new Set());
  const [sidebar, setSidebar]   = useState(true);
  const [panel, setPanel]       = useState(true);
  const [activity, setActivity] = useState("files");
  const [panelTab, setPanelTab] = useState("terminal");
  const [cursor, setCursor]     = useState({ ln: 1, col: 1 });

  const [result, setResult] = useState({ type: "empty", data: null });
  const [workspace, setWorkspace] = useState([]);    // [{ name, kind, value }]
  const [ir, setIr]         = useState("");
  const [logs, setLogs]     = useState([]);
  const [term, setTerm]     = useState([]);          // terminal stream lines
  const [running, setRunning] = useState(false);
  const [compiledAst, setCompiledAst] = useState(null);  // last successful compile

  const [editorWidth, setEditorWidth] = useState(55);
  const splitRef  = useRef(null);
  const dragging  = useRef(false);

  const activePathArr = openTabs.find(t => t.join("/") === activeTab) || null;
  const activeNode    = activePathArr ? getNode(files, activePathArr) : null;

  /* Stage 1 — Compile only (parse + type-check + IR). No execution. */
  const compile = useCallback(() => {
    const src = activeSource(files, activePathArr);
    if (src == null) {
      setTerm([{ stream: "stderr", text: "error: open a .ss file to compile" }]);
      setCompiledAst(null);
      return null;
    }
    const { ok, ast, ir: i, term: t, diagnostics } = compileStage(src);
    setIr(i);
    setTerm(t);
    setCompiledAst(ok ? ast : null);
    return ok ? { ast, diagnostics } : null;
  }, [files, activePathArr]);

  /* Stage 2 — Run: compile (fresh), then execute phases. */
  const setExperimentRecords = useStore(s => s.setExperimentRecords);

  const run = useCallback(async () => {
    setRunning(true);
    // Always compile fresh so edits are picked up
    const compiled = compile();
    if (!compiled) { setRunning(false); return; }

    // Yield a frame so the "compile" terminal output paints before executing
    await new Promise(r => setTimeout(r, 16));

    const t0 = performance.now();
    const { result: r, logs: l, term: execTerm, workspace: ws } = executeStage(compiled.ast);
    const dt = performance.now() - t0;
    setResult(r);
    setLogs(l);
    setWorkspace(ws || []);
    setTerm(prev => [...prev, ...execTerm]);

    // If the workspace contains records, push them into the shared store so the
    // full ResultsDashboard (the 10-row crossfilter dashboard from the
    // Experiment page) renders. Otherwise clear any stale dashboard records.
    const recEntry = (ws || []).find(w => w.kind === "records");
    if (recEntry) {
      const recs = recEntry.value;
      setExperimentRecords(recs, summariseForStore(recs), dt);
      setEditorWidth(w => Math.min(w, 32));   // give the dashboard room
    }

    // Resolve any pending async DB searches and patch them into the workspace.
    const pending = (ws || []).filter(w => w.kind === "pending");
    if (pending.length > 0) {
      setTerm(prev => [...prev, { stream: "stdout", text: `resolving ${pending.length} database query(ies)…` }]);
      await Promise.all(pending.map(async (w) => {
        const resolved = await resolvePending(w.value);
        setWorkspace(cur => cur.map(e =>
          e.name === w.name ? { ...e, value: { ...resolved, resolved: true } } : e
        ));
      }));
      setTerm(prev => [...prev, { stream: "stdout", text: "✓ database queries complete" }]);
    }

    setRunning(false);
  }, [compile, setExperimentRecords]);

  // Auto-compile (parse + type-check only) shortly after edits, for live
  // diagnostics. Execution is never automatic — it requires an explicit Run.
  useEffect(() => {
    const t = setTimeout(() => { compile(); }, 500);
    return () => clearTimeout(t);
  }, [files, activePathArr, compile]);

  // Keyboard: Ctrl/Cmd+Enter to Run, Ctrl/Cmd+B to Compile
  useEffect(() => {
    const onKey = (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { e.preventDefault(); run(); }
      else if ((e.ctrlKey || e.metaKey) && (e.key === "b" || e.key === "B")) { e.preventDefault(); compile(); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [run, compile]);

  useEffect(() => {
    const move = e => {
      if (!dragging.current || !splitRef.current) return;
      const r = splitRef.current.getBoundingClientRect();
      setEditorWidth(Math.min(80, Math.max(25, ((e.clientX - r.left) / r.width) * 100)));
    };
    const up = () => { dragging.current = false; document.body.style.cursor = ""; };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
    return () => { window.removeEventListener("mousemove", move); window.removeEventListener("mouseup", up); };
  }, []);

  const toggleFolder = useCallback(key => {
    setExpanded(prev => { const n = new Set(prev); n.has(key) ? n.delete(key) : n.add(key); return n; });
  }, []);

  const openFile = useCallback(pathArr => {
    const key = pathArr.join("/");
    setOpenTabs(prev => prev.some(t => t.join("/") === key) ? prev : [...prev, pathArr]);
    setActiveTab(key);
  }, []);

  const closeTab = useCallback((key, e) => {
    e.stopPropagation();
    setOpenTabs(prev => {
      const next = prev.filter(t => t.join("/") !== key);
      if (activeTab === key) setActiveTab(next.length ? next[next.length - 1].join("/") : null);
      return next;
    });
    setDirty(prev => { const n = new Set(prev); n.delete(key); return n; });
  }, [activeTab]);

  const updateContent = useCallback(val => {
    if (!activePathArr) return;
    setFiles(prev => { const next = structuredClone(prev); getNode(next, activePathArr).content = val; return next; });
    setDirty(prev => new Set(prev).add(activeTab));
  }, [activePathArr, activeTab]);

  const activities = [
    { id: "files",  Icon: Files,    label: "Explorer" },
    { id: "search", Icon: Search,   label: "Search" },
    { id: "git",    Icon: GitBranch, label: "Source Control" },
    { id: "run",    Icon: Play,     label: "Run" },
    { id: "ext",    Icon: Blocks,   label: "Extensions" },
  ];

  return (
    <div className="flex h-[85vh] min-h-[680px] w-full flex-col overflow-hidden rounded-lg text-sm shadow-2xl"
      style={{ background: T.editor, color: T.editorFg, border: `1px solid ${T.border}` }}>

      {/* Title bar */}
      <div className="flex h-9 shrink-0 items-center justify-between px-3"
        style={{ background: T.titlebar }}>
        <div className="flex items-center gap-2">
          <span className="h-3 w-3 rounded-full" style={{ background: "#ff5f56" }} />
          <span className="h-3 w-3 rounded-full" style={{ background: "#ffbd2e" }} />
          <span className="h-3 w-3 rounded-full" style={{ background: "#27c93f" }} />
        </div>
        <span className="text-xs" style={{ color: "#cccccc" }}>Shapeshifter Sandbox</span>
        <div className="w-12" />
      </div>

      <div className="flex min-h-0 flex-1">
        {/* Activity bar */}
        <div className="flex w-12 shrink-0 flex-col items-center justify-between py-2"
          style={{ background: T.activitybar }}>
          <div className="flex flex-col items-center gap-1">
            {activities.map(({ id, Icon, label }) => {
              const active = activity === id;
              return (
                <button key={id} title={label}
                  onClick={() => { if (active) setSidebar(s => !s); else { setActivity(id); setSidebar(true); } }}
                  className="relative flex h-11 w-12 items-center justify-center"
                  style={{ color: active ? T.activitybarFgActive : T.activitybarFg }}>
                  {active && <span className="absolute left-0 top-1/2 h-6 w-0.5 -translate-y-1/2"
                    style={{ background: "#fff" }} />}
                  <Icon size={24} strokeWidth={1.5} />
                </button>
              );
            })}
          </div>
          <button title="Settings" className="flex h-11 w-12 items-center justify-center"
            style={{ color: T.activitybarFg }}><Settings size={24} strokeWidth={1.5} /></button>
        </div>

        {/* Sidebar */}
        {sidebar && (
          <div className="flex w-60 shrink-0 flex-col overflow-hidden"
            style={{ background: T.sidebar, borderRight: `1px solid ${T.border}` }}>
            <div className="flex h-9 shrink-0 items-center px-4 text-[11px] font-medium uppercase tracking-wider"
              style={{ color: T.sidebarHeader }}>
              {activities.find(a => a.id === activity)?.label}
            </div>
            <div className="min-h-0 flex-1 overflow-y-auto pb-2">
              {activity === "files"
                ? <Tree tree={files} expanded={expanded} toggle={toggleFolder}
                    activePath={activeTab} openFile={openFile} />
                : <div className="px-4 py-6 text-[13px]" style={{ color: T.tabFg }}>
                    {activities.find(a => a.id === activity)?.label} panel
                  </div>
              }
            </div>
          </div>
        )}

        {/* Editor + Output split */}
        <div ref={splitRef} className="flex min-w-0 flex-1">
          {/* Editor column */}
          <div className="flex min-w-0 flex-col" style={{ width: `${editorWidth}%` }}>
            {/* Tab bar */}
            <div className="flex h-9 shrink-0 items-stretch overflow-x-auto"
              style={{ background: T.tabInactive }}>
              {openTabs.map(pathArr => {
                const key = pathArr.join("/");
                const name = pathArr[pathArr.length - 1];
                const active = key === activeTab;
                const isDirty = dirty.has(key);
                const { Icon, color } = fileIcon(name);
                return (
                  <div key={key} onClick={() => setActiveTab(key)}
                    className="group flex cursor-pointer items-center gap-2 border-r px-3 text-[13px]"
                    style={{
                      background: active ? T.tabActive : T.tabInactive,
                      color: active ? T.tabFgActive : T.tabFg,
                      borderColor: T.border,
                      borderTop: active ? `1px solid ${T.accentBright}` : "1px solid transparent",
                    }}>
                    <Icon size={15} style={{ color }} />
                    <span className="whitespace-nowrap">{name}</span>
                    <button onClick={e => closeTab(key, e)}
                      className="flex h-5 w-5 items-center justify-center rounded"
                      style={{ color: active ? T.tabFgActive : T.tabFg }}>
                      {isDirty ? <Circle size={9} fill="currentColor" className="group-hover:hidden" /> : null}
                      <X size={15} className={isDirty ? "hidden group-hover:block" : "opacity-0 group-hover:opacity-100"} />
                    </button>
                  </div>
                );
              })}
            </div>

            {/* Breadcrumb */}
            {activePathArr && (
              <div className="flex h-6 shrink-0 items-center gap-1 px-4 text-[12px]"
                style={{ background: T.editor, color: T.tabFg }}>
                {activePathArr.map((p, i) => (
                  <span key={i} className="flex items-center gap-1">
                    {i > 0 && <ChevronRight size={12} className="opacity-60" />}{p}
                  </span>
                ))}
              </div>
            )}

            {activeNode
              ? <Editor value={activeNode.content} onChange={updateContent} onCursor={setCursor} />
              : <div className="flex min-h-0 flex-1 items-center justify-center text-sm"
                  style={{ background: T.editor, color: "#5a5a5a" }}>
                  Select a file to start editing
                </div>
            }

            {/* Panel */}
            {panel && (
              <div className="flex h-40 shrink-0 flex-col"
                style={{ background: T.panel, borderTop: `1px solid ${T.border}` }}>
                <div className="flex h-9 items-center justify-between pr-2">
                  <div className="flex h-full items-center">
                    {[{ id: "problems", label: "Problems" }, { id: "terminal", label: "Terminal" }].map(({ id, label }) => {
                      const active = panelTab === id;
                      return (
                        <button key={id} onClick={() => setPanelTab(id)}
                          className="relative h-full px-3 text-[11px] font-medium uppercase tracking-wider"
                          style={{ color: active ? T.tabFgActive : T.tabFg }}>
                          {label}
                          {active && <span className="absolute bottom-0 left-0 h-0.5 w-full"
                            style={{ background: T.accentBright }} />}
                        </button>
                      );
                    })}
                  </div>
                  <button onClick={() => setPanel(false)} style={{ color: T.tabFg }}>
                    <PanelBottomClose size={16} />
                  </button>
                </div>
                <div className="min-h-0 flex-1 overflow-y-auto px-2 pb-2">
                  {panelTab === "terminal"
                    ? <TerminalView term={term} />
                    : (() => {
                        const problems = term.filter(l => l.stream === "stderr");
                        if (problems.length === 0) {
                          return (
                            <div className="flex items-center gap-2 pt-2 px-2 font-mono text-[13px]"
                              style={{ color: "#6a9955" }}>
                              <Check size={14} /> No problems detected.
                            </div>
                          );
                        }
                        return (
                          <div className="pt-1 font-mono text-[12px] leading-relaxed">
                            {problems.map((p, i) => (
                              <div key={i} className="flex items-start gap-2 px-2 py-0.5"
                                style={{ color: "#f48771" }}>
                                <AlertCircle size={13} className="mt-0.5 shrink-0" />
                                <span className="whitespace-pre-wrap break-words">
                                  {p.text.replace(/^(error|warning):\s*/, "")}
                                </span>
                              </div>
                            ))}
                          </div>
                        );
                      })()
                  }
                </div>
              </div>
            )}
          </div>

          {/* Splitter */}
          <div onMouseDown={() => { dragging.current = true; document.body.style.cursor = "col-resize"; }}
            className="w-1 shrink-0 cursor-col-resize"
            style={{ background: T.border }} />

          {/* Output column */}
          <OutputColumn
            result={result} workspace={workspace} ir={ir} logs={logs} term={term} running={running}
            onCompile={compile} onRun={run}
            onClear={() => { setLogs([]); setTerm([]); }}
          />
        </div>
      </div>

      {/* Status bar */}
      <div className="flex h-6 shrink-0 items-center justify-between px-3 text-[12px]"
        style={{ background: running ? "#a05a00" : T.statusBar, color: T.statusFg }}>
        <div className="flex items-center gap-3">
          <button className="flex items-center gap-1" onClick={() => { setPanel(true); setPanelTab("problems"); }}>
            <AlertCircle size={13} />
            {term.filter(l => l.stream === "stderr").length} problem(s)
          </button>
          {running && (
            <span className="flex items-center gap-1">
              <RefreshCw size={12} className="animate-spin" /> running
            </span>
          )}
          {!running && result?.type === "records" && (
            <span>{result.data.length} records</span>
          )}
        </div>
        <div className="flex items-center gap-3">
          <span>Ln {cursor.ln}, Col {cursor.col}</span>
          <span>Spaces: 4</span>
          <span>UTF-8</span>
          <span>{activeNode ? langLabel(activeNode.lang) : "—"}</span>
          <Bell size={13} />
        </div>
      </div>
    </div>
  );
}
