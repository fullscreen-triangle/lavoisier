/**
 * Ion-Droplet Bijection (Papers 2 & 3).
 *
 * Every ion has a corresponding thermodynamic droplet with Rayleigh surface
 * modes. Both representations converge to the same S-entropy coordinates,
 * providing dual-path interference validation without requiring external
 * ground truth.
 *
 * Ion path:     vibrational frequencies {ω_i} → (Sk, St, Se)
 * Droplet path: Rayleigh surface modes {κ_j} → (Sk', St', Se')
 *
 * Bijection theorem: Sk ≈ Sk', St ≈ St', Se ≈ Se' for the same compound.
 * Validation: common prefix length between the two ternary addresses.
 */

// Note: computeFromSpectrum (from ./sentropy) works on spectrum peaks.
// computeSEntropyFromFrequencies below works on raw vibrational frequencies
// as per the three-paper framework (Papers 2 & 3).

/* ── Physical constants ──────────────────────────────────────────────────── */
const kB    = 1.38064852e-23;   // Boltzmann, J/K
const u     = 1.66053906660e-27; // atomic mass unit, kg
const sigma = 72.0e-3;          // default water surface tension, N/m
const rho   = 997.0;            // default liquid density, kg/m³

/**
 * Compute the Rayleigh surface-wave modes of a liquid droplet of radius R.
 *
 * The l-th mode frequency (l = 2, 3, ..., Nmodes):
 *   ω_l = sqrt( l(l-1)(l+2) * σ / (ρ * R³) )
 *
 * Physical origin: surface tension restores deformations of the droplet
 * surface, which has spherical harmonic normal modes indexed by l.
 * This is the Rayleigh oscillation of a liquid sphere (Rayleigh 1879, Lamb 1881).
 */
export function rayleighModes(R, Nmodes = 8, surfaceTension = sigma, density = rho) {
  const modes = [];
  for (let l = 2; l <= Nmodes + 1; l++) {
    // ω_l = sqrt( l(l-1)(l+2) * σ / (ρ R³) )   [rad/s]
    const omega = Math.sqrt(l * (l - 1) * (l + 2) * surfaceTension / (density * Math.pow(R, 3)));
    modes.push(omega);
  }
  return modes;
}

/**
 * Map an ion to its corresponding droplet.
 *
 * @param {{
 *   mass: number,           // neutral molecular mass in Da
 *   kineticEnergy: number,  // approximate ion kinetic energy in eV
 *   composition?: Object,   // atom counts {C, H, N, O, ...}
 * }} ionParams
 * @returns {{
 *   radius:    number,  // droplet radius in nm
 *   velocity:  number,  // droplet velocity in m/s
 *   modes:     number[], // Rayleigh surface-mode frequencies in rad/s
 *   sentropy:  {sk, st, se},
 * }}
 */
export function ionToDroplet(ionParams) {
  const { mass, kineticEnergy = 1.0, composition = {} } = ionParams;

  const massKg = mass * u;

  // Radius: from molecular volume (spherical approximation)
  // V ≈ M / (N_A * ρ_mol) with effective molecular density 1200 kg/m³
  const rho_mol = 1200;   // kg/m³ effective
  const V = massKg / rho_mol;
  const R = Math.cbrt(3 * V / (4 * Math.PI));  // m

  // Velocity: from kinetic energy E_k = 0.5 * M * v²
  const E_k = kineticEnergy * 1.60218e-19;  // eV → J
  const v   = Math.sqrt(2 * E_k / massKg); // m/s

  // Surface tension modulated by molecular polarity
  // Estimate polarity from composition: more O/N → more polar → higher σ
  const nO = (composition.O || 0);
  const nN = (composition.N || 0);
  const nC = (composition.C || 1);
  const polarityFactor = 1.0 + 0.1 * (nO + nN) / (nC + 1);
  const sigma_eff = sigma * polarityFactor;

  // Surface modes
  const Nmodes = Math.min(12, Math.max(2, Math.floor(mass / 50)));
  const modes = rayleighModes(R, Nmodes, sigma_eff, rho);

  // Compute S-entropy from surface modes (same formulas as vibrational)
  // Convert modes from rad/s to cm⁻¹ for dimensional consistency
  const c = 2.99792458e10;  // cm/s
  const modesInCmInv = modes.map(w => w / (2 * Math.PI * c));

  const sentropy = computeSEntropyFromFrequencies(modesInCmInv);

  return { radius: R * 1e9, velocity: v, modes, sentropy };  // R in nm
}

/**
 * Compute S-entropy coordinates from a list of frequencies.
 * Mirrors the S-entropy definitions in Paper 3, §3.
 */
export function computeSEntropyFromFrequencies(freqs) {
  if (!freqs || freqs.length === 0) return { sk: 0, st: 0, se: 0 };

  const sorted   = [...freqs].filter(f => f > 0).sort((a, b) => a - b);
  const N        = sorted.length;
  if (N === 0) return { sk: 0, st: 0, se: 0 };

  // Knowledge entropy Sk: normalised Shannon entropy of frequency distribution
  const total = sorted.reduce((s, f) => s + f, 0);
  let sk = 0;
  if (N >= 2) {
    let H = 0;
    for (const f of sorted) {
      const p = f / total;
      if (p > 0) H -= p * Math.log2(p);
    }
    sk = H / Math.log2(N);
  } else {
    sk = sorted[0] / 4401;  // diatomic: normalise by H₂ frequency
  }

  // Temporal entropy St: dynamic range ratio
  const ST_REF_MAX = 4401, ST_REF_MIN = 67;  // H₂, I₂ in cm⁻¹
  let st = 0;
  if (N >= 2) {
    const ratio = sorted[N - 1] / sorted[0];
    st = Math.log(ratio) / Math.log(ST_REF_MAX / ST_REF_MIN);
  } else if (N === 1) {
    st = Math.log(sorted[0] / ST_REF_MIN) / Math.log(ST_REF_MAX / ST_REF_MIN);
  }

  // Evolution entropy Se: fraction of harmonic-proximity pairs
  let se = 0;
  if (N >= 2) {
    const delta    = 0.05;
    let  nHarmonic = 0;
    const nPairs   = N * (N - 1) / 2;
    for (let i = 0; i < N; i++) {
      for (let j = i + 1; j < N; j++) {
        const ratio = sorted[j] / sorted[i];
        let isHarmonic = false;
        for (let p = 1; p <= 8 && !isHarmonic; p++) {
          for (let q = 1; q <= 8 && !isHarmonic; q++) {
            if (Math.abs(ratio - p / q) < delta) isHarmonic = true;
          }
        }
        if (isHarmonic) nHarmonic++;
      }
    }
    se = nHarmonic / nPairs;
  }

  return {
    sk: Math.max(0, Math.min(1, sk)),
    st: Math.max(0, Math.min(1, st)),
    se: Math.max(0, Math.min(1, se)),
  };
}

/**
 * Validate a compound identification via dual-path interference (Paper 2, §9).
 *
 * Computes S-entropy from both the ion vibrational path and the droplet
 * Rayleigh path, then measures their common ternary prefix length.
 * Longer common prefix = higher confidence identification.
 *
 * @param {{sk, st, se}} ionSentropy
 * @param {{mass, kineticEnergy, composition}} ionParams  (for droplet path)
 * @param {number} depth   ternary address depth
 * @returns {{
 *   ionAddress:     string,
 *   dropletAddress: string,
 *   commonPrefixLen: number,
 *   convergenceScore: number,   // 0–1
 *   falsePosProb:    number,    // 3^(-commonPrefixLen)
 * }}
 */
export function dualPathValidate(ionSentropy, ionParams, depth = 12) {
  const droplet        = ionToDroplet(ionParams);
  const dropletSentropy = droplet.sentropy;

  const ionAddr     = ternaryAddress(ionSentropy.sk,      ionSentropy.st,      ionSentropy.se,      depth);
  const dropletAddr = ternaryAddress(dropletSentropy.sk, dropletSentropy.st, dropletSentropy.se, depth);

  let commonLen = 0;
  for (let i = 0; i < Math.min(ionAddr.length, dropletAddr.length); i++) {
    if (ionAddr[i] === dropletAddr[i]) commonLen++;
    else break;
  }

  const convergenceScore = commonLen / depth;
  const falsePosProb     = Math.pow(3, -commonLen);

  return {
    ionAddress: ionAddr,
    dropletAddress: dropletAddr,
    commonPrefixLen: commonLen,
    convergenceScore,
    falsePosProb,
    droplet: { radius: droplet.radius, velocity: droplet.velocity },
  };
}

/**
 * Compute an interleaved ternary address for S-entropy coordinates.
 * Implements Algorithm 1 from Paper 3, §4.
 */
export function ternaryAddress(sk, st, se, depth = 12) {
  let lo = [0, 0, 0], hi = [1, 1, 1];
  const coords = [sk, st, se];
  let addr = "";
  for (let j = 0; j < depth; j++) {
    const dim  = j % 3;
    const delta = (hi[dim] - lo[dim]) / 3;
    const v    = coords[dim];
    let trit;
    if      (v < lo[dim] + delta)     { trit = 0; hi[dim]  = lo[dim] + delta; }
    else if (v < lo[dim] + 2 * delta) { trit = 1; lo[dim] += delta; hi[dim]  = lo[dim] + delta; }
    else                              { trit = 2; lo[dim] += 2 * delta; }
    addr += trit;
  }
  return addr;
}
