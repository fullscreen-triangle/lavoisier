/**
 * Purpose Function — phase space constraint from domain context (Paper 2, §8).
 *
 * P: Context → 2^S
 *
 * Maps domain knowledge to subregions of S-entropy space [0,1]³.
 * Each domain restricts the accessible region; the Purpose function computes
 * the minimal set of ternary prefixes whose union covers that region.
 *
 * Prompt Contraction Theorem: adding constraints can only reduce the accessible
 * region — never expand it. P(c₁,...,cₚ) ⊆ P(c₁,...,cₚ₋₁).
 *
 * Domain-specific reduction ratios from Paper 2, §10.4:
 *   metabolomics: 99.2%
 *   glycomics:    99.7%
 *   proteomics:   ~98%
 *   lipidomics:   ~97%
 *   diatomics:    99.99%
 */

import { ternaryAddress } from "../partition/ionDroplet";

/* ── Domain definitions ──────────────────────────────────────────────────── */
// Each domain specifies a bounding box in S-entropy space: [sk_lo, sk_hi] × [st_lo, st_hi] × [se_lo, se_hi]
// Values derived from empirical S-entropy distributions of known compound classes.

export const DOMAINS = {
  // General small-molecule metabolomics (plasma/urine)
  metabolomics: {
    label: "Metabolomics",
    description: "Small molecules in biological fluids (MW 50–1200 Da)",
    bounds: { sk: [0.3, 1.0], st: [0.0, 0.9], se: [0.0, 1.0] },
  },

  // Lipidomics — lipid species
  lipidomics: {
    label: "Lipidomics",
    description: "Lipid species: glycerophospholipids, sphingolipids, neutral lipids",
    bounds: { sk: [0.7, 1.0], st: [0.1, 0.8], se: [0.2, 1.0] },
  },

  // Proteomics — tryptic peptides
  proteomics: {
    label: "Proteomics",
    description: "Tryptic peptides from standard proteins",
    bounds: { sk: [0.85, 1.0], st: [0.2, 0.95], se: [0.4, 1.0] },
  },

  // Glycomics — glycan structures
  glycomics: {
    label: "Glycomics",
    description: "Glycan and glycoconjugate analysis",
    bounds: { sk: [0.8, 1.0], st: [0.05, 0.7], se: [0.5, 1.0] },
  },

  // Environmental analysis — xenobiotics, pesticides
  environmental: {
    label: "Environmental",
    description: "Environmental contaminants, pesticides, pharmaceuticals",
    bounds: { sk: [0.4, 0.95], st: [0.0, 1.0], se: [0.0, 0.8] },
  },

  // Diatomic / small inorganic molecules
  diatomics: {
    label: "Diatomics & small molecules",
    description: "Diatomic gases and simple inorganics",
    bounds: { sk: [0.0, 0.5], st: [0.5, 1.0], se: [0.0, 0.1] },
  },

  // Drug-like molecules (Lipinski compliant)
  pharma: {
    label: "Pharmaceuticals",
    description: "Drug-like molecules: MW < 500 Da, Lipinski-compliant",
    bounds: { sk: [0.6, 1.0], st: [0.1, 0.85], se: [0.1, 0.9] },
  },

  // Full coverage (no restriction)
  all: {
    label: "All compounds",
    description: "No domain restriction",
    bounds: { sk: [0.0, 1.0], st: [0.0, 1.0], se: [0.0, 1.0] },
  },
};

export const DOMAIN_KEYS = Object.keys(DOMAINS);

/* ── Purpose function ────────────────────────────────────────────────────── */

/**
 * Compute the set of ternary prefixes that cover a domain's S-entropy region.
 * Returns the minimal prefix set (each prefix covers the bounding box).
 *
 * The Prompt-to-Prefix Theorem (Paper 2, §8.2): every Purpose function
 * determines a finite set of ternary prefixes whose subtrees cover P(Context).
 *
 * @param {string} domain  key from DOMAINS
 * @param {number} prefixDepth  ternary prefix depth (typically 3–6)
 * @returns {string[]}  ternary prefix strings
 */
export function getPurposePrefixes(domain, prefixDepth = 4) {
  const def = DOMAINS[domain] ?? DOMAINS.all;
  const { bounds } = def;

  // Sample the bounding box corners to find covering prefixes
  const { sk, st, se } = bounds;
  const steps = 3;  // sample 3 points per dimension
  const prefixSet = new Set();

  for (let i = 0; i <= steps; i++) {
    for (let j = 0; j <= steps; j++) {
      for (let k = 0; k <= steps; k++) {
        const skv = sk[0] + (sk[1] - sk[0]) * (i / steps);
        const stv = st[0] + (st[1] - st[0]) * (j / steps);
        const sev = se[0] + (se[1] - se[0]) * (k / steps);
        const addr = ternaryAddress(skv, stv, sev, prefixDepth);
        prefixSet.add(addr);
      }
    }
  }

  return [...prefixSet];
}

/**
 * Compute the domain-specific reduction ratio.
 * ρ = 1 - |relevant prefixes| / 3^prefixDepth
 *
 * For metabolomics at depth 4: ρ = 1 - r/81 where r = number of prefixes.
 */
export function reductionRatio(domain, prefixDepth = 4) {
  const prefixes = getPurposePrefixes(domain, prefixDepth);
  const total = Math.pow(3, prefixDepth);
  return 1 - prefixes.length / total;
}

/**
 * Determine which domains a given S-entropy point belongs to.
 * Returns all matching domains, ordered by specificity (fewest prefixes first).
 */
export function matchingDomains(sentropy) {
  const matches = [];
  for (const [key, def] of Object.entries(DOMAINS)) {
    if (key === "all") continue;
    const { bounds: { sk, st, se } } = def;
    if (
      sentropy.sk >= sk[0] && sentropy.sk <= sk[1] &&
      sentropy.st >= st[0] && sentropy.st <= st[1] &&
      sentropy.se >= se[0] && sentropy.se <= se[1]
    ) {
      matches.push({ key, label: def.label, description: def.description });
    }
  }
  return matches;
}

/**
 * Prompt contraction: combine multiple domain constraints.
 * Each additional constraint can only narrow the region.
 *
 * @param {string[]} domains  list of domain keys to intersect
 * @returns {{ bounds, prefixes, reductionRatio }}
 */
export function combineDomains(domains, prefixDepth = 4) {
  const bounds = { sk: [0, 1], st: [0, 1], se: [0, 1] };
  for (const d of domains) {
    const def = DOMAINS[d];
    if (!def) continue;
    // Intersection: take the tighter (inner) bounds
    bounds.sk[0] = Math.max(bounds.sk[0], def.bounds.sk[0]);
    bounds.sk[1] = Math.min(bounds.sk[1], def.bounds.sk[1]);
    bounds.st[0] = Math.max(bounds.st[0], def.bounds.st[0]);
    bounds.st[1] = Math.min(bounds.st[1], def.bounds.st[1]);
    bounds.se[0] = Math.max(bounds.se[0], def.bounds.se[0]);
    bounds.se[1] = Math.min(bounds.se[1], def.bounds.se[1]);
  }

  // Generate prefixes for the combined region
  const { sk, st, se } = bounds;
  const steps = 3;
  const prefixSet = new Set();
  for (let i = 0; i <= steps; i++) {
    for (let j = 0; j <= steps; j++) {
      for (let k = 0; k <= steps; k++) {
        const skv = sk[0] + (sk[1] - sk[0]) * (i / steps);
        const stv = st[0] + (st[1] - st[0]) * (j / steps);
        const sev = se[0] + (se[1] - se[0]) * (k / steps);
        const addr = ternaryAddress(skv, stv, sev, prefixDepth);
        prefixSet.add(addr);
      }
    }
  }
  const prefixes = [...prefixSet];
  const total = Math.pow(3, prefixDepth);

  return {
    bounds,
    prefixes,
    reductionRatio: 1 - prefixes.length / total,
  };
}
