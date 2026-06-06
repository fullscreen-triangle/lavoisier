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

import { runExperiment, summariseRecords } from "@/lib/experiment/virtualinstrument";
import { LIPID_CLASSES } from "@/lib/experiment/lipidomics";
import { PROTEIN_CLASSES } from "@/lib/experiment/proteomics";
import { computeSEntropyFromFrequencies, dualPathValidate, ternaryAddress } from "@/lib/partition/ionDroplet";
import { GenerativeDb, addressToSentropy, commonPrefixScore } from "@/lib/partition/GenerativeDb";
import { DOMAINS, getPurposePrefixes, matchingDomains, combineDomains } from "@/lib/shapeshifter/purpose";

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

  // Join continuation lines (unclosed brackets span multiple source lines)
  const joined = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    const open  = (line.trimmed.match(/\[/g) || []).length;
    const close = (line.trimmed.match(/\]/g) || []).length;
    if (open > close) {
      let combined = line.trimmed, balance = open - close, j = i + 1;
      while (j < lines.length && balance > 0) {
        combined += " " + lines[j].trimmed;
        balance += (lines[j].trimmed.match(/\[/g) || []).length;
        balance -= (lines[j].trimmed.match(/\]/g) || []).length;
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

function executeCall(fn, args, env, ast, log) {
  const a = Object.fromEntries(
    Object.entries(args).map(([k, v]) =>
      [k, typeof v === "string" && env[v] !== undefined ? env[v] : v]
    )
  );

  if (fn === "lavoisier.instrument.run_experiment") {
    const classKeys = (a.classes || ["PC", "PE"]).filter(k => LIPID_CLASSES[k]);
    if (!classKeys.length) throw new Error("No valid lipid classes. Use keys like PC, PE, SM, TAG, Cer.");
    const classSpecs = classKeys.map(key => {
      const cls = LIPID_CLASSES[key];
      const Xmin = cls.defaults.Xrange[0];
      // Cap range for browser performance (full range can be resumed in the Rust binary)
      const Xmax = Math.min(cls.defaults.Xrange[1], Xmin + 8);
      return { classKey: key, Xmin, Xmax, Ymin: 0, Ymax: 4, enabled: true };
    });
    log(`  Computing ${classKeys.join(", ")} — ${classSpecs.length} class(es)`);
    const records = runExperiment({
      experimentType: "lipidomics", classSpecs, proteinSpecs: [],
      polarity:     a.polarity     || "+",
      analyser:     a.analyser     || "orbitrap",
      analyserCfg:  { kField: 1e12, Rm: 1e-2 },
      collisionEnergy_eV: a.collision_energy || 25,
      mzWindow: a.mz_window || [200, 1500],
    });
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

/** Execute a parsed AST. Returns { result, logs }. */
export function executeShapeshifter(ast) {
  const env = {}, logs = [];
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
        if (stmt.value.type === "call") {
          env[stmt.target] = executeCall(stmt.value.fn, stmt.value.args, env, ast, log);
          log(`  ${stmt.target} ← ${describeVal(env[stmt.target])}`);
        } else {
          env[stmt.target] = stmt.value.value;
        }
      }
    }
  }

  if (env.records && Array.isArray(env.records)) {
    const summary = summariseRecords(env.records);
    return { result: { type: "records", data: env.records, summary }, logs };
  }
  if (env.registry && Array.isArray(env.registry)) {
    return { result: { type: "cells", data: env.registry }, logs };
  }
  if (env.addresses && Array.isArray(env.addresses)) {
    return { result: { type: "addresses", data: env.addresses }, logs };
  }
  // MassScript observation results
  if (env.observation && typeof env.observation === "object" && env.observation.type === "observation") {
    return { result: env.observation, logs };
  }
  if (env.domain_context && typeof env.domain_context === "object") {
    return { result: { type: "domain", data: env.domain_context }, logs };
  }
  if (env.validation && typeof env.validation === "object") {
    return { result: { type: "validation", data: env.validation }, logs };
  }
  return { result: { type: "empty", data: null }, logs };
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
