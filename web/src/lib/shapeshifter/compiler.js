/**
 * Shapeshifter compiler — shared between the Sandbox and the Experiment page.
 *
 * parse    : .ss source  →  AST
 * execute  : AST         →  { result, logs }
 * compile  : source      →  { result, ir, logs }  (full pipeline)
 *
 * The result object carries a `type` tag:
 *   "records"   — PredictedRecord[], feeds directly into the store
 *   "cells"     — ΔP timing cell registry (from lavoisier.cells.compile)
 *   "addresses" — partition state table (from lavoisier.partition.compute_addresses)
 *   "empty"     — nothing produced
 */

import { runExperiment, summariseRecords } from "../experiment/virtualinstrument.js";
import { LIPID_CLASSES } from "../experiment/lipidomics.js";
import { PROTEIN_CLASSES } from "../experiment/proteomics.js";
import { computeSEntropyFromFrequencies, dualPathValidate, ternaryAddress } from "../partition/ionDroplet.js";
import { GenerativeDb, addressToSentropy, commonPrefixScore } from "../partition/GenerativeDb.js";
import { DOMAINS, getPurposePrefixes, matchingDomains, combineDomains } from "./purpose.js";
import { sebdMs, sebdMsToRecords, reconstructPrecursor } from "../partition/partitionStateGraph.js";
import { fragmentSubharmonics, virtualTensorComponents, verifyMeanRecovery, planckDepth, effectiveDimensionality, impossibleIons, partitionComplement, transientContents } from "../partition/virtualTensor.js";
import { searchAll } from "../spectral/dbSearch.js";

/* ── Parser helpers ──────────────────────────────────────────────────────── */

export function splitCommas(raw) {
  const parts = [];
  let depth = 0, start = 0;
  for (let i = 0; i < raw.length; i++) {
    const c = raw[i];
    if ("([{".includes(c)) depth++;
    else if (")]}".includes(c)) depth--;
    else if (c === "," && depth === 0) { parts.push(raw.slice(start, i)); start = i + 1; }
  }
  parts.push(raw.slice(start));
  return parts.filter(p => p.trim());
}

function parseObjectArray(raw) {
  const objects = [];
  let depth = 0, start = 0;
  for (let i = 0; i < raw.length; i++) {
    if (raw[i] === "{") { if (depth === 0) start = i; depth++; }
    else if (raw[i] === "}") {
      depth--;
      if (depth === 0) {
        const obj = {};
        splitCommas(raw.slice(start + 1, i)).forEach(pair => {
          const ci = pair.indexOf(":");
          if (ci >= 0) obj[pair.slice(0, ci).trim()] = parseValue(pair.slice(ci + 1).trim());
        });
        objects.push(obj);
      }
    }
  }
  return objects;
}

/** Parse a single `{ key: value, ... }` object literal into an object. */
function parseObjectLiteral(raw) {
  const obj = {};
  const inner = raw.trim().replace(/^\{/, "").replace(/\}$/, "");
  splitCommas(inner).forEach(pair => {
    const ci = pair.indexOf(":");
    if (ci >= 0) obj[pair.slice(0, ci).trim()] = parseValue(pair.slice(ci + 1).trim());
  });
  return obj;
}

export function parseValue(raw) {
  if (!raw) return null;
  const s = raw.trim();
  if ((s.startsWith('"') && s.endsWith('"')) || (s.startsWith("'") && s.endsWith("'")))
    return s.slice(1, -1);
  if (s === "true") return true;
  if (s === "false") return false;
  if (s.startsWith("[") && s.endsWith("]")) {
    const inner = s.slice(1, -1).trim();
    if (!inner) return [];
    return inner.startsWith("{") ? parseObjectArray(inner)
      : splitCommas(inner).map(v => parseValue(v.trim()));
  }
  if (s.startsWith("{") && s.endsWith("}")) {
    return parseObjectLiteral(s);
  }
  const n = Number(s);
  if (!isNaN(n) && s !== "") return n;
  return s;
}

function parseNamedArgs(raw) {
  const args = {};
  if (!raw.trim()) return args;
  splitCommas(raw).forEach(part => {
    const ci = part.indexOf(":");
    if (ci >= 0) args[part.slice(0, ci).trim()] = parseValue(part.slice(ci + 1).trim());
  });
  return args;
}

function parseExpr(raw) {
  const s = raw.trim();
  const m = s.match(/^([\w.]+)\s*\(([\s\S]*)\)$/);
  if (m) return { type: "call", fn: m[1], args: parseNamedArgs(m[2]) };
  return { type: "value", value: parseValue(s) };
}

function parseBlockFields(lines, startI, baseIndent) {
  const fields = {};
  let i = startI;
  while (i < lines.length && lines[i].indent > baseIndent) {
    const m = lines[i].trimmed.match(/^(\w+)\s*:\s*(.*)/);
    if (m) fields[m[1]] = parseValue(m[2]);
    i++;
  }
  return [fields, i];
}

function parseBlockStatements(lines, startI, baseIndent) {
  const stmts = [];
  let i = startI;
  while (i < lines.length && lines[i].indent > baseIndent) {
    const t = lines[i].trimmed;
    const assignM = t.match(/^(\w+)\s*=\s*([\s\S]+)/);
    if (assignM) {
      stmts.push({ type: "assign", target: assignM[1], value: parseExpr(assignM[2]) });
    } else if (/^[\w.]+\s*\(/.test(t)) {
      const callM = t.match(/^([\w.]+)\s*\(([\s\S]*)\)/);
      if (callM) stmts.push({ type: "call", fn: callM[1], args: parseNamedArgs(callM[2]) });
    }
    i++;
  }
  return [stmts, i];
}

/** Parse a .ss source string into an AST. */
export function parseShapeshifter(source) {
  const ast = {
    imports: [], objective: null,
    instruments: {}, validates: {}, phases: {}, targetLists: {},
  };

  let lines = source.split("\n").map((raw, idx) => ({
    lineNum: idx + 1,
    raw,
    indent: raw.match(/^(\s*)/)[1].length,
    trimmed: raw.replace(/\/\/.*$/, "").trim(),
  })).filter(l => l.trimmed.length > 0);

  // Join continuation lines: any unclosed bracket — [, ( or { — extends the
  // statement onto following lines until the balance closes. This lets a
  // multi-line classes: [ {...}, {...} ] or filters: { ... } span lines.
  const opens  = (s) => ((s.match(/[\[({]/g) || []).length);
  const closes = (s) => ((s.match(/[\])}]/g) || []).length);
  const joined = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    let balance = opens(line.trimmed) - closes(line.trimmed);
    if (balance > 0) {
      let combined = line.trimmed, j = i + 1;
      while (j < lines.length && balance > 0) {
        combined += " " + lines[j].trimmed;
        balance += opens(lines[j].trimmed) - closes(lines[j].trimmed);
        j++;
      }
      joined.push({ ...line, trimmed: combined });
      i = j;
    } else {
      joined.push(line);
      i++;
    }
  }
  lines = joined;

  i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (line.trimmed.startsWith("import ")) {
      ast.imports.push(line.trimmed.slice(7).trim());
      i++;
    } else if (/^objective\s+\w+\s*:/.test(line.trimmed)) {
      const name = line.trimmed.match(/^objective\s+(\w+)/)[1];
      const [fields, ni] = parseBlockFields(lines, i + 1, line.indent);
      ast.objective = { name, fields };
      i = ni;
    } else if (/^instrument\s+\w+\s*:/.test(line.trimmed)) {
      const name = line.trimmed.match(/^instrument\s+(\w+)/)[1];
      const [fields, ni] = parseBlockFields(lines, i + 1, line.indent);
      ast.instruments[name] = fields;
      i = ni;
    } else if (/^validate\s+\w+\s*:/.test(line.trimmed)) {
      const name = line.trimmed.match(/^validate\s+(\w+)/)[1];
      const [stmts, ni] = parseBlockStatements(lines, i + 1, line.indent);
      ast.validates[name] = stmts;
      i = ni;
    } else if (/^phase\s+\w+\s*:/.test(line.trimmed)) {
      const name = line.trimmed.match(/^phase\s+(\w+)/)[1];
      const [stmts, ni] = parseBlockStatements(lines, i + 1, line.indent);
      ast.phases[name] = stmts;
      i = ni;
    } else if (/^target_list\s+\w+\s*:/.test(line.trimmed)) {
      const name = line.trimmed.match(/^target_list\s+(\w+)/)[1];
      const [fields, ni] = parseBlockFields(lines, i + 1, line.indent);
      ast.targetLists[name] = { name, ...fields };
      i = ni;
    } else {
      i++;
    }
  }
  return ast;
}

/* ── Executor ────────────────────────────────────────────────────────────── */

function describeVal(v) {
  if (v == null) return "null";
  if (Array.isArray(v)) return `Array(${v.length})`;
  return String(v).slice(0, 50);
}

/**
 * Predict reverse-phase (C18) retention time for a lipid record.
 * Hydrophobicity ≈ total acyl carbons; each double bond reduces retention.
 * Returns minutes within a [0.5, gradient] window.
 */
function predictRtRPLC(record, gradientMin = 30) {
  const X = record.X ?? 16;             // acyl carbons (or peptide length)
  const Y = record.Y ?? 0;              // double bonds (or missed cleavages)
  // Carbon span maps ~roughly 8..70 → 0..1; double bonds pull retention earlier.
  const hydro = (X - 8) / 62 - Y * 0.035;
  const frac  = Math.max(0, Math.min(1, hydro));
  const rt    = 0.5 + frac * (gradientMin - 1.0);
  return Math.round(rt * 100) / 100;
}

/**
 * Apply post-generation filter rules to the record set.
 * Supported filter keys (all optional):
 *   mz:       [lo, hi]   precursor m/z window
 *   rt:       [lo, hi]   retention-time window (min)
 *   classes:  ["PC"]     keep only these analyte classes
 *   db:       [lo, hi]   double-bond range (Y)
 *   carbons:  [lo, hi]   acyl-carbon range (X)
 *   adducts:  ["[M+H]+"] keep only these adducts
 *   min_intensity: 0.05  drop low-intensity ions
 */
function applyExperimentFilters(records, filters, log) {
  if (!filters || typeof filters !== "object") return records;
  const inRange = (v, r) => !r || (v >= r[0] && v <= r[1]);
  return records.filter(rec => {
    if (filters.mz       && !inRange(rec.precursorMz, filters.mz))    return false;
    if (filters.rt       && !inRange(rec.retentionTime, filters.rt))  return false;
    if (filters.db       && !inRange(rec.Y, filters.db))              return false;
    if (filters.carbons  && !inRange(rec.X, filters.carbons))         return false;
    if (filters.min_intensity != null && rec.intensity < filters.min_intensity) return false;
    if (filters.classes  && Array.isArray(filters.classes) &&
        !filters.classes.includes(rec.analyteClass))                 return false;
    if (filters.adducts  && Array.isArray(filters.adducts) &&
        !filters.adducts.includes(rec.adduct))                       return false;
    return true;
  });
}

function executeCall(fn, args, env, ast, log) {
  const a = Object.fromEntries(
    Object.entries(args).map(([k, v]) =>
      [k, typeof v === "string" && env[v] !== undefined ? env[v] : v]
    )
  );

  if (fn === "lavoisier.instrument.run_experiment") {
    // `classes` may be either:
    //   ["PC", "PE"]                                  — defaults per class, or
    //   [{ class:"PC", carbons:[30,40], db:[0,4] }]   — explicit per-class ranges
    const rawClasses = a.classes || ["PC", "PE"];
    const classSpecs = [];
    for (const entry of rawClasses) {
      const key = typeof entry === "string" ? entry : entry.class;
      const cls = LIPID_CLASSES[key];
      if (!cls) { log(`  ⚠ unknown lipid class "${key}" — skipped`, "warn"); continue; }
      // Range resolution: explicit object > script default > class default (capped)
      const carbons = (typeof entry === "object" && entry.carbons) || null;
      const db      = (typeof entry === "object" && entry.db)      || null;
      const Xmin = carbons ? carbons[0] : cls.defaults.Xrange[0];
      const Xmax = carbons ? carbons[1] : Math.min(cls.defaults.Xrange[1], cls.defaults.Xrange[0] + 8);
      const Ymin = db ? db[0] : 0;
      const Ymax = db ? db[1] : 4;
      classSpecs.push({ classKey: key, Xmin, Xmax, Ymin, Ymax, enabled: true });
    }
    if (!classSpecs.length) throw new Error("No valid lipid classes. Use keys like PC, PE, SM, TAG, Cer.");

    log(`  Computing ${classSpecs.map(c => `${c.classKey}(${c.Xmin}-${c.Xmax}:${c.Ymin}-${c.Ymax})`).join(", ")}`);
    let records = runExperiment({
      experimentType: "lipidomics", classSpecs, proteinSpecs: [],
      polarity:     a.polarity     || "+",
      adductsAllowed: a.adducts    || null,
      analyser:     a.analyser     || "orbitrap",
      analyserCfg:  { kField: 1e12, Rm: 1e-2 },
      collisionEnergy_eV: a.collision_energy || 25,
      mzWindow: a.mz_window || [200, 1500],
    });

    // Predicted retention time (RPLC C18 model): RT grows with hydrophobicity,
    // ≈ carbon number, decreasing with each double bond. Needed for rt filters.
    const gradient = a.gradient_min || 30;  // total gradient length (min)
    records = records.map(r => ({ ...r, retentionTime: predictRtRPLC(r, gradient) }));

    // Post-generation filter rules (the "specific classes in an RT window" case)
    const before = records.length;
    records = applyExperimentFilters(records, a.filters, log);
    if (a.filters) log(`  Filters: ${before} → ${records.length} records`);

    log(`  → ${records.length} predicted ions`);
    return records;
  }

  if (fn === "lavoisier.instrument.run_proteomics") {
    const proteinKeys = (a.proteins || ["HSA"]).filter(k => PROTEIN_CLASSES[k]);
    if (!proteinKeys.length) throw new Error("No valid protein classes. Use HSA, HBB, ENO1, CYCS, or CASE.");
    const proteinSpecs = proteinKeys.map(key => ({
      classKey: key,
      lengthMin: a.length_min ?? 7, lengthMax: a.length_max ?? 20,
      mcMin: 0, mcMax: a.mc_max ?? 1, enabled: true,
    }));
    log(`  Computing ${proteinKeys.join(", ")} — ${proteinSpecs.length} protein standard(s)`);
    const records = runExperiment({
      experimentType: "proteomics", classSpecs: [], proteinSpecs,
      polarity:     a.polarity    || "+",
      analyser:     a.analyser    || "orbitrap",
      analyserCfg:  { kField: 1e12, Rm: 1e-2 },
      collisionEnergy_eV: a.collision_energy || 28,
      mzWindow: a.mz_window || [200, 3000],
    });
    log(`  → ${records.length} predicted ions`);
    return records;
  }

  if (fn === "lavoisier.partition.compute_addresses") {
    const lipids = a.lipids || [];
    return lipids.map(lip => {
      const mass = lip.mass || 500;
      const n = Math.max(1, Math.ceil(Math.sqrt(mass / 162.0)));
      const l = Math.min(n - 1, Math.max(0, Math.floor(n / 2)));
      const hash = ((lip.name || "?").charCodeAt(0) * 31 + Math.round(mass * 7)) % (2 * l + 1);
      return {
        name: lip.name || "?", mass: +mass.toFixed(4),
        adduct: lip.adduct || "[M+H]+",
        n, l, m: hash - l, s: 0.5,
      };
    });
  }

  if (fn === "lavoisier.cells.compile") {
    const tlName = a.target_list;
    const tl = ast.targetLists[tlName] || {};
    const targets = tl.targets || [];
    const windowPpm = tl.window_ppm ?? a.window_ppm ?? 5.0;
    const instr = ast.instruments[tl.instrument] || {};
    const kappa = instr.kappa ?? 1e12;
    const fRef  = instr.ref_frequency ?? 10e6;
    const e = 1.60218e-19, u = 1.66054e-27, hbar = 1.0546e-34;
    return targets.map(t => {
      const mz = t.mz || 500;
      const dMz = mz * windowPpm * 1e-6;
      const omega   = Math.sqrt(e * kappa / (mz * u));
      const dOmega  = omega * windowPpm * 0.5e-6;
      const dM      = e * kappa * dMz * u / (mz * mz);
      const tauMs   = Math.max(0.01, (hbar / (dM + 1e-60)) * 1e3);
      return {
        name: t.name || "?",
        mz: mz.toFixed(4),
        window_da: dMz.toFixed(4),
        omega_hz: (omega / (2 * Math.PI)).toExponential(3),
        dp_lo: (-(dOmega / (2 * Math.PI * fRef))).toExponential(3),
        dp_hi: (+(dOmega / (2 * Math.PI * fRef))).toExponential(3),
        tau_min_ms: tauMs.toFixed(3),
      };
    });
  }

  /* ── SEBD-MS: Partition-State Graph Search ──────────────────────────────
   * lavoisier.msms.sebd_search(precursor_mz, fragments, opts)
   *   Runs SEBD-MS. Returns records[] compatible with ResultsDashboard.
   * lavoisier.msms.phase_coherence(precursor_mz, fragments)
   *   Phase Coherence Theorem: subharmonic frequencies & self-consistency.
   * lavoisier.msms.virtual_tensor(mz, charge, time_steps)
   *   Stacked virtual partition tensor V_{ijkl} across 4 dimensions.
   * lavoisier.msms.impossible_ions(mz_list)
   *   Impossible ions as crossing-symmetry probes.
   * lavoisier.msms.partition_complement(mz, planck_depth)
   *   Ion removal via virtual antistate (SWIFT derivation).
   * lavoisier.msms.transient_contents(precursor_mz, fragments)
   *   Single-measurement completeness: all information in one transient.
   * lavoisier.db.search(precursor_mz, fragments, databases)
   *   Search public spectral databases (MassBank, GNPS, MoNA).
   ─────────────────────────────────────────────────────────────────────── */

  if (fn === "lavoisier.msms.sebd_search") {
    const precMz   = a.precursor_mz ?? 500;
    const frags    = a.fragments ?? [];
    const maxDepth = a.max_depth ?? 7;
    const nP       = a.planck_depth ?? 56;
    if (!Array.isArray(frags) || frags.length === 0) {
      log("  ⚠ sebd_search: fragments must be a non-empty array of m/z values", "warn");
      return null;
    }
    log(`  SEBD-MS: precursor ${precMz.toFixed(4)} Da, ${frags.length} fragments, maxDepth ${maxDepth}`);
    const result  = sebdMs(precMz, frags, { maxDepth, planckDepth: nP });
    const records = sebdMsToRecords(result);
    log(`  → ${records.length} reachable fragment nodes`);
    log(`  → reachable fraction: ${(result.reachableFraction * 100).toFixed(1)}%`);
    log(`  → off-shell (transition state) fraction: ${(result.offShellFraction * 100).toFixed(1)}%`);
    log(`  → mean |Δn|: ${result.meanDeltaN.toFixed(1)}`);
    log(`  → mean S-entropy path length: ${result.meanPathLength.toFixed(3)}`);
    return records;  // PredictedRecord[] → feeds directly into setExperimentRecords
  }

  if (fn === "lavoisier.msms.phase_coherence") {
    const precMz = a.precursor_mz ?? 500;
    const frags  = a.fragments ?? [];
    if (!Array.isArray(frags)) return null;
    const result = fragmentSubharmonics(precMz, frags);
    log(`  Phase coherence: ${result.length} fragment subharmonics`);
    const allSelfConsistent = result.every(r => r.selfConsistent);
    log(`  Self-consistency: ${allSelfConsistent ? "✓ 1.0000 (<10⁻⁶ ppm)" : "✗ inconsistent"}`);
    result.forEach(r => log(`  ω_f/ω_p = ${r.frequencyRatio.toFixed(6)}, Δθ = ${r.phaseDiff.toFixed(2)} rad, err = ${r.backConversionError_ppm.toExponential(2)} ppm`));
    return result;
  }

  if (fn === "lavoisier.msms.virtual_tensor") {
    const mz        = a.mz ?? 500;
    const charge    = a.charge ?? 1;
    const nT        = a.time_steps ?? 10;
    const tensor    = virtualTensorComponents(mz, charge, nT);
    const verified  = verifyMeanRecovery(tensor);
    const dEff      = effectiveDimensionality(Math.round(mz / 10), nT);
    const tauOsc    = 1 / (Math.sqrt(1.60218e-19 * 1e12 / (mz * 1.66054e-27)) / (2 * Math.PI));
    const nP        = planckDepth(dEff, tauOsc);
    log(`  Virtual tensor: ${tensor.N} components, d_eff = ${dEff.toLocaleString()}`);
    log(`  Off-shell fraction: ${(verified.offShellFraction * 100).toFixed(1)}%`);
    log(`  Mean-recovery: mean = ${verified.mean.toFixed(4)}, holds = ${verified.meanRecoveryHolds}`);
    log(`  Planck depth (stacked): n_P = ${nP}`);
    return { tensor: tensor.components, verified, dEff, planckDepth: nP };
  }

  if (fn === "lavoisier.msms.impossible_ions") {
    const mzList = a.mz_list ?? a.ions ?? [];
    const result = impossibleIons(mzList);
    log(`  Impossible ions: ${result.length} crossing-symmetry probes`);
    result.forEach(r => log(`  (${r.ion1_mz.toFixed(3)} + ${r.ion2_mz.toFixed(3)}) / 2 → ${r.impossibleMz.toFixed(3)}`));
    return result;
  }

  if (fn === "lavoisier.msms.partition_complement") {
    const mz = a.mz ?? 500;
    const nP = a.planck_depth ?? 56;
    const result = partitionComplement(mz, nP);
    log(`  Partition complement (SWIFT antistate):`);
    log(`  M_ion = ${result.M_ion}, M_comp = ${result.M_comp}, C_max = ${result.Cmax}`);
    log(`  Complement m/z = ${result.complementMz.toFixed(4)}`);
    return result;
  }

  if (fn === "lavoisier.msms.transient_contents") {
    const precMz = a.precursor_mz ?? 500;
    const frags  = a.fragments ?? [];
    const result = transientContents(precMz, frags);
    log(`  Single-transient completeness (Theorem 11.1):`);
    log(`  Precursor: ${result.precursor.freq_Hz.toExponential(3)} Hz`);
    log(`  Fragment subharmonics: ${result.fragments.length}`);
    log(`  Charge states encoded: ${result.chargeStates.length}`);
    log(`  Polarity complement: Δφ = π`);
    return result;
  }

  /* ── Online spectral database search ──────────────────────────────────── */

  if (fn === "lavoisier.db.search") {
    const precMz = a.precursor_mz ?? 500;
    const frags  = a.fragments ?? [];
    const dbs    = a.databases ?? ["massbank", "mona"];
    log(`  Searching ${dbs.join(", ")} for m/z ${precMz.toFixed(4)}...`);
    // Return a promise-based result; executor handles async
    return { __async: true, __fn: "db.search", precMz, frags, dbs };
  }

  if (fn === "lavoisier.db.search_massbank") {
    const precMz = a.precursor_mz ?? 500;
    log(`  Searching MassBank for m/z ${precMz.toFixed(4)}...`);
    return { __async: true, __fn: "db.search_massbank", precMz };
  }

  if (fn === "lavoisier.db.search_gnps") {
    const precMz = a.precursor_mz ?? 500;
    const frags  = a.fragments ?? [];
    log(`  Searching GNPS for m/z ${precMz.toFixed(4)}...`);
    return { __async: true, __fn: "db.search_gnps", precMz, frags };
  }

  if (fn === "lavoisier.db.search_mona") {
    const precMz = a.precursor_mz ?? 500;
    log(`  Searching MoNA for m/z ${precMz.toFixed(4)}...`);
    return { __async: true, __fn: "db.search_mona", precMz };
  }

  /* ── MassScript vocabulary (Paper 2, §8.2) ────────────────────────────── */

  // lavoisier.observe.partition_field(records)
  // Computes wave-field parameters for each record — used by GpuObserver Pass 1
  if (fn === "lavoisier.observe.partition_field") {
    const records = a.records ?? env[a.records] ?? [];
    if (!Array.isArray(records)) { log("  ⚠ observe.partition_field: records must be an array", "warn"); return null; }
    log(`  Mapping ${records.length} records to partition wave-field ions`);
    return records.map(r => ({
      center:     [r.sentropyVec?.sk ?? 0.5, r.sentropyVec?.st ?? 0.5],
      amplitude:  Math.max(0, Math.min(1, r.intensity ?? 0.5)),
      wavelength: Math.max(0.01, 1 / ((r.n ?? 1) + 1)),
      decay:      Math.max(0.01, r.sentropyVec?.se ?? 0.3),
      angle:      ((r.l ?? 0) / Math.max(1, (r.n ?? 1))) * Math.PI,
      phase:      ((r.m ?? 0) / Math.max(1, (r.l ?? 1) + 1)) * Math.PI,
      sk: r.sentropyVec?.sk ?? 0,
      st: r.sentropyVec?.st ?? 0,
      se: r.sentropyVec?.se ?? 0,
    }));
  }

  // lavoisier.observe.sentropy(frequencies)
  // Compute S-entropy coordinates from a list of vibrational frequencies
  if (fn === "lavoisier.observe.sentropy") {
    const freqs = a.frequencies ?? a.freqs ?? [];
    if (!Array.isArray(freqs)) { log("  ⚠ observe.sentropy: frequencies must be an array", "warn"); return null; }
    const result = computeSEntropyFromFrequencies(freqs);
    log(`  Sk=${result.sk.toFixed(3)} St=${result.st.toFixed(3)} Se=${result.se.toFixed(3)}`);
    return result;
  }

  // lavoisier.observe.ternary_address(sentropy, depth)
  // Compute the ternary address for S-entropy coordinates
  if (fn === "lavoisier.observe.ternary_address") {
    const se = a.sentropy ?? {};
    const depth = a.depth ?? 12;
    const addr = ternaryAddress(se.sk ?? 0, se.st ?? 0, se.se ?? 0, depth);
    log(`  Ternary address (depth ${depth}): ${addr}`);
    return addr;
  }

  // lavoisier.observe.dual_path_validate(sentropy, ion_params)
  // Ion-droplet bijection validation: dual oscillatory path cross-check
  if (fn === "lavoisier.observe.dual_path_validate") {
    const ionSE     = a.sentropy  ?? {};
    const ionParams = a.ion       ?? {};
    const depth     = a.depth     ?? 12;
    const result    = dualPathValidate(ionSE, ionParams, depth);
    log(`  Common prefix: ${result.commonPrefixLen} / ${depth}`);
    log(`  Convergence score: ${result.convergenceScore.toFixed(4)}`);
    log(`  False-positive prob: ${result.falsePosProb.toExponential(3)}`);
    return result;
  }

  // lavoisier.db.generate(prefix, depth)
  // Generate S-entropy coordinates from a ternary prefix (Partition Determinism)
  if (fn === "lavoisier.db.generate") {
    const prefix = String(a.prefix ?? "");
    const depth  = a.depth ?? 12;
    const coords = addressToSentropy(prefix.padEnd(depth, "1"));
    log(`  Generated coords from prefix "${prefix}": Sk=${coords.sk.toFixed(3)} St=${coords.st.toFixed(3)} Se=${coords.se.toFixed(3)}`);
    return { prefix, coords, depth };
  }

  // lavoisier.purpose.domain(name)
  // Get S-entropy region and ternary prefixes for a domain context
  if (fn === "lavoisier.purpose.domain") {
    const name  = a.domain ?? a.name ?? "all";
    const depth = a.depth  ?? 4;
    const def   = DOMAINS[name] ?? DOMAINS.all;
    const prefs = getPurposePrefixes(name, depth);
    const total = Math.pow(3, depth);
    const rho   = (1 - prefs.length / total) * 100;
    log(`  Domain: ${def.label}`);
    log(`  Prefixes: ${prefs.length} / ${total} (${rho.toFixed(1)}% reduction)`);
    log(`  S-entropy bounds: Sk ${def.bounds.sk} St ${def.bounds.st} Se ${def.bounds.se}`);
    return { name, label: def.label, bounds: def.bounds, prefixes: prefs, reductionPct: rho };
  }

  // lavoisier.purpose.match(sentropy)
  // Find all matching domains for given S-entropy coordinates
  if (fn === "lavoisier.purpose.match") {
    const se = a.sentropy ?? {};
    const matches = matchingDomains(se);
    log(`  Matching domains: ${matches.map(m => m.label).join(", ") || "none"}`);
    return matches;
  }

  // lavoisier.purpose.combine(domains)
  // Intersect multiple domain constraints (Prompt Contraction Theorem)
  if (fn === "lavoisier.purpose.combine") {
    const domains = a.domains ?? [];
    const depth   = a.depth ?? 4;
    const result  = combineDomains(domains, depth);
    log(`  Combined ${domains.join(" ∩ ")}: ${result.prefixes.length} prefixes, ${(result.reductionRatio*100).toFixed(1)}% reduction`);
    return result;
  }

  log(`  ⚠ Unknown function: ${fn}`, "warn");
  return null;
}

/**
 * Classify a (functionName, value) pair into a visualisation kind so the
 * sandbox can pick the right panel. Returns a string tag.
 */
function classifyValue(fn, value) {
  if (Array.isArray(value)) {
    if (value.length && value[0] && typeof value[0] === "object") {
      if ("precursorMz" in value[0] && "sentropyVec" in value[0]) return "records";
      if ("dp_lo" in value[0] || "tau_min_ms" in value[0])         return "cells";
      if ("n" in value[0] && "l" in value[0] && "m" in value[0])   return "addresses";
      if ("frequencyRatio" in value[0])                            return "subharmonics";
      if ("impossibleMz" in value[0])                              return "impossible";
      if ("instrument" in value[0] && "value" in value[0])         return "tensor";
    }
    return "array";
  }
  if (value && typeof value === "object") {
    if (value.__async)                    return "pending";
    if (fn && fn.includes("dual_path"))   return "validation";
    if (fn && fn.includes("domain"))      return "domain";
    if (fn && fn.includes("combine"))     return "domain";
    if ("sk" in value && "st" in value && "se" in value) return "sentropy";
    if ("tensor" in value && "verified" in value) return "tensorReport";
    if ("complementMz" in value)          return "complement";
    if ("precursor" in value && "fragments" in value) return "transient";
    return "object";
  }
  if (typeof value === "string") return "string";
  if (typeof value === "number") return "number";
  return "scalar";
}

/** Execute a parsed AST. Returns { result, logs, workspace }. */
export function executeShapeshifter(ast) {
  const env = {}, kinds = {}, order = [], logs = [];
  const log = (msg, level = "info") => logs.push({ level, message: msg });

  if (ast.objective) {
    log(`🎯 Objective: ${ast.objective.name}`);
    if (ast.objective.fields?.target) log(`   ${ast.objective.fields.target}`);
  }

  for (const [name, stmts] of Object.entries(ast.validates)) {
    log(`✓ Validate: ${name}`);
    for (const stmt of stmts) {
      if (stmt.type === "call" && stmt.fn === "check_resolution_time") {
        const ppm = stmt.args?.window_ppm ?? 5;
        const kappa = ast.instruments[stmt.args?.instrument]?.kappa ?? 1e12;
        const hbar = 1.0546e-34, e = 1.60218e-19, u = 1.66054e-27;
        const dM = e * kappa * (500 * ppm * 1e-6) * u / (500 * 500);
        log(`   τ_min ≈ ${((hbar / dM) * 1e3).toFixed(1)} ms at ${ppm} ppm (500 Da ref)`);
      }
    }
  }

  for (const [phaseName, stmts] of Object.entries(ast.phases)) {
    log(`⚡ Phase: ${phaseName}`);
    for (const stmt of stmts) {
      if (stmt.type === "assign") {
        let fn = null;
        if (stmt.value.type === "call") {
          fn = stmt.value.fn;
          env[stmt.target] = executeCall(stmt.value.fn, stmt.value.args, env, ast, log);
          log(`  ${stmt.target} ← ${describeVal(env[stmt.target])}`);
        } else {
          env[stmt.target] = stmt.value.value;
        }
        if (!order.includes(stmt.target)) order.push(stmt.target);
        kinds[stmt.target] = classifyValue(fn, env[stmt.target]);
      }
    }
  }

  // Build the workspace: every assigned variable, in declaration order,
  // tagged with a visualisation kind. The sandbox renders ALL of them.
  const workspace = order
    .filter(name => env[name] !== undefined && env[name] !== null)
    .map(name => ({ name, kind: kinds[name], value: env[name] }));

  // Primary result (kept for back-compat + store integration).
  let result;
  if (env.records && Array.isArray(env.records) && env.records.length) {
    result = { type: "records", data: env.records, summary: summariseRecords(env.records) };
  } else if (env.registry && Array.isArray(env.registry)) {
    result = { type: "cells", data: env.registry };
  } else if (env.addresses && Array.isArray(env.addresses)) {
    result = { type: "addresses", data: env.addresses };
  } else {
    // Pick the most "chart-worthy" workspace entry as the primary.
    const pref = ["records", "cells", "addresses", "subharmonics", "tensor", "impossible"];
    const chosen = workspace.find(w => pref.includes(w.kind)) || workspace[workspace.length - 1];
    if (chosen) {
      if (chosen.kind === "records") result = { type: "records", data: chosen.value, summary: summariseRecords(chosen.value) };
      else if (chosen.kind === "cells") result = { type: "cells", data: chosen.value };
      else if (chosen.kind === "addresses") result = { type: "addresses", data: chosen.value };
      else result = { type: "workspace", data: chosen.value };
    } else {
      result = { type: "empty", data: null };
    }
  }

  return { result, logs, workspace };
}

/** Full pipeline: source → { result, ir, logs }. */
export function compileShapeshifter(source) {
  try {
    const ast = parseShapeshifter(source);
    const { result, logs } = executeShapeshifter(ast);
    return { result, ir: JSON.stringify(ast, null, 2), logs };
  } catch (e) {
    return {
      result: { type: "empty", data: null },
      ir: "",
      logs: [{ level: "error", message: `Compile error: ${e.message}` }],
    };
  }
}

/* ── Staged compile + execute (for the Sandbox terminal) ─────────────────── */

/**
 * Summarise an AST into a one-line-per-block structure list.
 */
function astSummary(ast) {
  const lines = [];
  if (ast.objective) lines.push(`objective ${ast.objective.name}`);
  for (const k of Object.keys(ast.instruments))  lines.push(`instrument ${k}`);
  for (const k of Object.keys(ast.targetLists))  lines.push(`target_list ${k}`);
  for (const k of Object.keys(ast.validates))    lines.push(`validate ${k}`);
  for (const k of Object.keys(ast.phases))        lines.push(`phase ${k}`);
  return lines;
}

/**
 * Stage 1 — COMPILE.
 * Parses, type-checks block structure, and produces the IR + terminal output.
 * Does NOT execute any phase.
 *
 * @returns {{ ok, ast, ir, term: TermLine[], diagnostics: Diag[] }}
 *   TermLine = { stream: "stdout"|"stderr"|"stage", text: string }
 *   Diag     = { severity: "error"|"warning", message: string }
 */
export function compileStage(source) {
  const term = [];
  const diagnostics = [];
  const t0 = (typeof performance !== "undefined" ? performance.now() : Date.now());

  term.push({ stream: "stage", text: "shapeshifter compile" });

  let ast;
  try {
    ast = parseShapeshifter(source);
  } catch (e) {
    term.push({ stream: "stderr", text: `error: parse failed — ${e.message}` });
    diagnostics.push({ severity: "error", message: `Parse error: ${e.message}` });
    return { ok: false, ast: null, ir: "", term, diagnostics };
  }

  // Basic structural checks (the "type checker")
  const blocks = astSummary(ast);
  term.push({ stream: "stdout", text: `parsed ${ast.imports.length} import(s), ${blocks.length} block(s)` });
  blocks.forEach(b => term.push({ stream: "stdout", text: `  · ${b}` }));

  if (!ast.objective) {
    diagnostics.push({ severity: "warning", message: "no objective block declared" });
    term.push({ stream: "stderr", text: "warning: no objective block declared" });
  }
  const phaseCount = Object.keys(ast.phases).length;
  if (phaseCount === 0) {
    diagnostics.push({ severity: "warning", message: "no phase block — nothing will execute" });
    term.push({ stream: "stderr", text: "warning: no phase block — nothing will execute" });
  }

  const dt = ((typeof performance !== "undefined" ? performance.now() : Date.now()) - t0);
  term.push({ stream: "stdout", text: `✓ compiled in ${dt.toFixed(1)} ms` });

  return { ok: true, ast, ir: JSON.stringify(ast, null, 2), term, diagnostics };
}

/**
 * Stage 2 — EXECUTE.
 * Runs phases on a previously-compiled AST and produces results + terminal output.
 *
 * @param {object} ast  from compileStage
 * @returns {{ result, term: TermLine[], logs }}
 */
export function executeStage(ast) {
  const term = [];
  const t0 = (typeof performance !== "undefined" ? performance.now() : Date.now());
  term.push({ stream: "stage", text: "shapeshifter run" });

  let result, logs, workspace;
  try {
    ({ result, logs, workspace } = executeShapeshifter(ast));
  } catch (e) {
    term.push({ stream: "stderr", text: `error: runtime — ${e.message}` });
    return {
      result: { type: "empty", data: null },
      logs: [{ level: "error", message: e.message }],
      workspace: [],
      term,
    };
  }

  // Mirror execution logs into the terminal stream
  for (const l of logs) {
    term.push({ stream: l.level === "error" ? "stderr" : "stdout", text: l.message });
  }

  // Result summary line
  const dt = ((typeof performance !== "undefined" ? performance.now() : Date.now()) - t0);
  let summary = "";
  switch (result.type) {
    case "records":
      summary = `→ ${result.data.length} record(s) produced`;
      break;
    case "cells":
      summary = `→ ${result.data.length} ΔP timing cell(s)`;
      break;
    case "addresses":
      summary = `→ ${result.data.length} partition address(es)`;
      break;
    case "empty":
      summary = "→ no output value";
      break;
    default:
      summary = `→ produced a "${result.type}" result`;
  }
  term.push({ stream: "stdout", text: summary });
  if (workspace && workspace.length) {
    term.push({ stream: "stdout", text: `workspace: ${workspace.map(w => `${w.name}:${w.kind}`).join(", ")}` });
  }
  term.push({ stream: "stdout", text: `✓ finished in ${dt.toFixed(1)} ms` });

  return { result, logs, workspace: workspace || [], term };
}
