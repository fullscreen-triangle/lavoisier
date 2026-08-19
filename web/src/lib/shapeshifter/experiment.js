/**
 * Experiment operations for the Shapeshifter web runtime.
 *
 * These implement the acquisition half of the language: reading a
 * reference library, transforming spectra to S-entropy coordinates, and
 * computing the separation / drift / baseline statistics that a stated
 * criterion is scored against.
 *
 * The browser cannot open a local .msp, so `lavoisier.acquire.library`
 * synthesises a library with the same designed structure as the NIST
 * AC_CAC reference set: N analytes, each measured at nine normalised
 * collision energies, with fragment count and intensity distribution
 * varying with collision energy the way HCD spectra actually do.
 *
 * The point of the exercise is not the data. It is that the criterion,
 * the control and the comparison are declared in the source and scored
 * by the runtime, so the program can return a verdict against its author.
 */

const NCE_LEVELS = [10, 15, 20, 25, 30, 40, 50, 60, 80];
const EPS = 1e-10;

/* ── deterministic PRNG so a run is reproducible ─────────────────────────── */

function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/* ── library synthesis ───────────────────────────────────────────────────── */

/**
 * Mean fragment count as a function of collision energy.
 * Rises steeply, peaks near NCE 40, then falls as fragments are driven
 * below the detection threshold. Matches the measured AC_CAC profile
 * (6.9 -> 34.0 -> 28.1 peaks per spectrum).
 */
function peakCountAt(nce) {
  const p = [6.9, 11.6, 18.0, 25.2, 31.1, 34.0, 32.4, 31.3, 28.1];
  const i = NCE_LEVELS.indexOf(nce);
  return i >= 0 ? p[i] : 20;
}

export function synthLibrary({ analytes = 60, seed = 20260819 } = {}) {
  const rnd = mulberry32(seed);
  const scans = [];
  for (let c = 0; c < analytes; c++) {
    // Each analyte gets a precursor mass and a characteristic fragment ladder.
    const precursor = 150 + rnd() * 550;
    // A large, compound-specific ladder: real analytes differ mainly in
    // WHICH fragments they give, which is the signal the coordinates
    // would have to capture to identify a compound.
    const nLadder = 20 + Math.floor(rnd() * 25);
    // Each analyte responds to energy at its own rate; without this the
    // whole library drifts in lock-step and the correlation is inflated.
    const respond = 0.55 + rnd() * 0.9;
    const offset = -12 + rnd() * 24;
    const ladder = [];
    for (let f = 0; f < nLadder; f++) {
      ladder.push({
        mz: 40 + rnd() * (precursor - 45),
        // Lability: how readily this fragment appears as energy rises.
        lability: rnd(),
        // Compound-specific intensity weight, stable across energies.
        weight: 0.2 + rnd() * 0.8,
      });
    }
    for (const nce of NCE_LEVELS) {
      const eff = Math.max(5, offset + nce * respond);   // effective energy
      const target = Math.max(3, peakCountAt(nce) * (0.55 + 0.9 * rnd()));
      const frac = target / Math.max(nLadder, 1);
      const peaks = [];
      // Precursor survives at low energy, is consumed at high energy.
      const survive = Math.max(0, 1 - eff / 55);
      if (survive > 0.02) peaks.push([precursor, 10000 * survive]);
      for (const f of ladder) {
        // A fragment appears once the energy exceeds its lability threshold.
        const onset = 8 + f.lability * 55;
        if (eff < onset) continue;
        const decay = Math.exp(-(eff - onset) / 90);
        const amp = 10000 * decay * f.weight * (0.7 + 0.6 * rnd());
        if (amp > 25) peaks.push([f.mz, amp]);
      }
      // Trim or pad toward the measured peak-count profile, keeping the
      // strongest compound-specific fragments.
      peaks.sort((x, y) => y[1] - x[1]);
      if (peaks.length > target) peaks.length = Math.round(target);
      const nSecondary = Math.max(0, Math.round(target - peaks.length));
      for (let s = 0; s < nSecondary; s++) {
        peaks.push([40 + rnd() * (precursor - 45), 150 + rnd() * 900]);
      }
      if (peaks.length < 3) continue;
      peaks.sort((a, b) => a[0] - b[0]);
      scans.push({
        scan_id: `${c}_${nce}`,
        compound: `A${String(c).padStart(3, "0")}`,
        precursor_mz: precursor,
        nce,
        n_peaks: peaks.length,
        peaks,
      });
    }
  }
  return scans;
}

/* ── S-entropy transformation (specification eq. 7.1-7.3) ────────────────── */

export function sentropyOne(peaks, precursorMz, alpha, beta, k) {
  const n = peaks.length;
  const mz = peaks.map(p => p[0]);
  const inten = peaks.map(p => p[1]);
  const imax = Math.max(...inten, 1);
  const ihat = inten.map(i => i / imax);
  const mStar = precursorMz > 0 ? precursorMz : Math.max(...mz);

  // S_k : Shannon self-information + mass term
  let sk = 0;
  for (let i = 0; i < n; i++) {
    sk += -Math.log2(ihat[i] + EPS) + alpha * (mz[i] / mStar);
  }

  // S_t : Gaussian weighting about the spectral centroid
  const mean = mz.reduce((a, b) => a + b, 0) / n;
  const sd = Math.sqrt(mz.reduce((a, m) => a + (m - mean) ** 2, 0) / n);
  let st = 0;
  for (let i = 0; i < n; i++) {
    st += sd > 0 ? Math.exp(-beta * Math.abs(mz[i] - mean) / sd) : 1.0;
  }

  // S_e : local k-neighbourhood Shannon entropy
  const kk = Math.min(k, n - 1);
  let se = 0;
  for (let i = 0; i < n; i++) {
    if (kk < 1) continue;                       // degenerate: single peak
    const idx = mz
      .map((m, j) => [Math.abs(m - mz[i]), j])
      .sort((a, b) => a[0] - b[0])
      .slice(0, kk)
      .map(p => p[1]);
    const tot = idx.reduce((a, j) => a + ihat[j], 0);
    if (tot <= 0) continue;
    let h = 0;
    for (const j of idx) {
      const p = ihat[j] / tot;
      if (p > 0) h -= p * Math.log2(p);
    }
    se += h;
  }
  return { s_k: sk / n, s_t: st / n, s_e: se / n };
}

/* ── statistics ──────────────────────────────────────────────────────────── */

const AXES = ["s_k", "s_t", "s_e"];

function dist(a, b, axes) {
  let s = 0;
  for (const x of axes) s += (a[x] - b[x]) ** 2;
  return Math.sqrt(s);
}

function groupBy(rows, key) {
  const g = {};
  for (const r of rows) (g[r[key]] ||= []).push(r);
  return g;
}

const mean = a => a.reduce((x, y) => x + y, 0) / (a.length || 1);

function sd(a) {
  const m = mean(a);
  return Math.sqrt(mean(a.map(v => (v - m) ** 2)));
}

export function separation(coords, { key = "compound", axes = AXES, minGroup = 9 } = {}) {
  const g = groupBy(coords, key);
  const keys = Object.keys(g).filter(k => g[k].length >= minGroup);
  if (keys.length < 2) return null;

  const within = [];
  const cents = {};
  for (const k of keys) {
    const items = g[k];
    const c = {};
    for (const a of axes) c[a] = mean(items.map(i => i[a]));
    cents[k] = c;
    for (let i = 0; i < items.length; i++)
      for (let j = i + 1; j < items.length; j++)
        within.push(dist(items[i], items[j], axes));
  }
  const between = [];
  for (let i = 0; i < keys.length; i++)
    for (let j = i + 1; j < keys.length; j++)
      between.push(dist(cents[keys[i]], cents[keys[j]], axes));

  const mw = mean(within), mb = mean(between);
  return {
    key, axes,
    n_groups: keys.length,
    n_within_pairs: within.length,
    n_between_pairs: between.length,
    mean_within: mw,
    mean_between: mb,
    separation_ratio: mw > 0 ? mb / mw : Infinity,
  };
}

export function drift(coords, { over = "nce", axes = AXES } = {}) {
  const pts = coords.filter(c => c[over] != null);
  const xs = pts.map(c => c[over]);
  const mx = mean(xs);
  const sxx = xs.reduce((a, x) => a + (x - mx) ** 2, 0);
  const out = { over, n: pts.length, axes: {} };
  for (const a of axes) {
    const ys = pts.map(c => c[a]);
    const my = mean(ys);
    let sxy = 0, syy = 0;
    for (let i = 0; i < pts.length; i++) {
      sxy += (xs[i] - mx) * (ys[i] - my);
      syy += (ys[i] - my) ** 2;
    }
    const slope = sxx > 0 ? sxy / sxx : 0;
    const r = sxx > 0 && syy > 0 ? sxy / Math.sqrt(sxx * syy) : 0;
    out.axes[a] = { slope, pearson_r: r, r_squared: r * r };
  }
  return out;
}

function cosine(a, b, tol) {
  const used = new Set();
  let dot = 0;
  for (const [m1, i1] of a) {
    let best = tol, bj = -1;
    for (let j = 0; j < b.length; j++) {
      if (used.has(j)) continue;
      const d = Math.abs(m1 - b[j][0]);
      if (d <= best) { best = d; bj = j; }
    }
    if (bj >= 0) { used.add(bj); dot += i1 * b[bj][1]; }
  }
  const na = Math.sqrt(a.reduce((s, p) => s + p[1] * p[1], 0));
  const nb = Math.sqrt(b.reduce((s, p) => s + p[1] * p[1], 0));
  return na > 0 && nb > 0 ? dot / (na * nb) : 0;
}

export function baseline(scans, { key = "compound", tolerance = 0.01, minGroup = 9, maxGroups = 40 } = {}) {
  const g = groupBy(scans, key);
  const keys = Object.keys(g).filter(k => g[k].length >= minGroup).slice(0, maxGroups);
  const all = [], adjacent = [];
  const lag = {};
  for (const k of keys) {
    const items = g[k].slice().sort((a, b) => a.nce - b.nce);
    for (let i = 0; i < items.length; i++)
      for (let j = i + 1; j < items.length; j++) {
        const s = cosine(items[i].peaks, items[j].peaks, tolerance);
        all.push(s);
        if (j === i + 1) adjacent.push(s);
        (lag[j - i] ||= []).push(s);
      }
  }
  return {
    metric: "cosine_similarity",
    key, tolerance,
    n_groups: keys.length,
    n_pairs: all.length,
    mean_within_compound: mean(all),
    mean_adjacent_level: mean(adjacent),
    by_lag: Object.keys(lag).sort((a, b) => a - b)
      .map(l => ({ lag: +l, mean: mean(lag[l]) })),
  };
}

export function shuffleControl(coords, opts = {}) {
  const seed = opts.seed ?? 20260819;
  const rnd = mulberry32(seed);
  const labels = coords.map(c => c.compound);
  for (let i = labels.length - 1; i > 0; i--) {
    const j = Math.floor(rnd() * (i + 1));
    [labels[i], labels[j]] = [labels[j], labels[i]];
  }
  const shuffled = coords.map((c, i) => ({ ...c, compound: labels[i] }));
  const res = separation(shuffled, opts);
  if (res) { res.control = "label_shuffle"; res.seed = seed; }
  return res;
}

/**
 * Score a stated criterion. Returns pass/fail per condition plus the
 * overall verdict. A criterion the program declares BEFORE running is
 * what makes an execution a test rather than a description.
 */
export function scoreCriterion({ separation: sep, drift: dr, minRatio = 2.0, maxAbsR = 0.3 }) {
  const conds = [];
  if (sep) {
    conds.push({
      name: "separation ratio",
      required: `> ${minRatio}`,
      observed: sep.separation_ratio,
      pass: sep.separation_ratio > minRatio,
    });
  }
  if (dr) {
    for (const a of AXES) {
      const r = dr.axes[a]?.pearson_r ?? 0;
      conds.push({
        name: `|r| ${a} vs ${dr.over}`,
        required: `< ${maxAbsR}`,
        observed: r,
        pass: Math.abs(r) < maxAbsR,
      });
    }
  }
  const passed = conds.filter(c => c.pass).length;
  return {
    criterion: `ratio > ${minRatio} and |r| < ${maxAbsR} per axis`,
    conditions: conds,
    n_passed: passed,
    n_total: conds.length,
    verdict: passed === conds.length ? "MET" : "NOT MET",
  };
}

export function parameterSweep(scans, { alphas, betas, k = 5, minGroup = 9 } = {}) {
  const A = alphas || [0, 0.5, 1, 2, 4];
  const B = betas || [0.5, 1, 2];
  const grid = [];
  for (const alpha of A) {
    for (const beta of B) {
      const coords = scans.map(s => ({
        compound: s.compound, nce: s.nce,
        ...sentropyOne(s.peaks, s.precursor_mz, alpha, beta, k),
      }));
      const sep = separation(coords, { minGroup });
      grid.push({ alpha, beta, k, ratio: sep ? sep.separation_ratio : null });
    }
  }
  const best = grid.reduce((a, b) => (b.ratio > (a?.ratio ?? -1) ? b : a), null);
  return {
    n_settings: grid.length,
    grid,
    best_ratio: best?.ratio ?? null,
    best_at: best ? { alpha: best.alpha, beta: best.beta, k: best.k } : null,
  };
}
