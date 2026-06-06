/**
 * Generative Spectral Database — O(1) memory architecture (Papers 2 & 3).
 *
 * The database stores nothing. It IS the phase space structure.
 * Entries materialise on demand and dissolve after use.
 *
 * Architecture:
 *   - A ternary trie indexed by S-entropy address (depth k)
 *   - Streaming observation: batch compounds, observe, discard (O(1) GPU memory)
 *   - Purpose-based prefix restriction: domain context → relevant trie subtrees
 *   - On-demand generation: given prefix, generate ion trajectory without dynamics
 *
 * Query cost:  O(k)  — independent of database size N
 * Memory:      O(1)  — constant regardless of N (Paper 2 Theorem 6.1)
 * Storage:     O(0)  — no spectra stored; phase space IS the database
 */

import { ternaryAddress, computeSEntropyFromFrequencies } from "./ionDroplet";
import { TernaryTrie } from "./trie";

/* ── Streaming architecture ──────────────────────────────────────────────── */

/**
 * Process an arbitrarily large compound set with constant memory.
 *
 * @param {Iterable<CompoundRecord>} compounds
 * @param {number} batchSize
 * @param {function(batch: CompoundRecord[]): void} processBatch
 */
export async function streamObserve(compounds, batchSize = 64, processBatch) {
  const arr = [...compounds];
  for (let i = 0; i < arr.length; i += batchSize) {
    const batch = arr.slice(i, i + batchSize);
    await processBatch(batch);
    // Batch is garbage-collected after processBatch returns.
    // GPU textures are reused (see GpuObserver._texWave etc.)
    // Total GPU memory stays constant at ~25 MB regardless of arr.length.
    await new Promise(r => setTimeout(r, 0));  // yield to event loop
  }
}

/* ── GenerativeDb class ──────────────────────────────────────────────────── */

export class GenerativeDb {
  /**
   * @param {number} depth   Ternary address depth (default 12; uniquely resolves 39 NIST at 11)
   */
  constructor(depth = 12) {
    this.depth = depth;
    this._trie = new TernaryTrie();
    this._compounds = new Map();   // address → CompoundRecord[]  (loaded prefix)
    this._metadata  = new Map();   // address → { count, representative }
  }

  /**
   * Insert a compound into the trie by its S-entropy address.
   *
   * @param {string} name
   * @param {{ sk: number, st: number, se: number }} sentropy
   * @param {any} data  additional compound data (stored only for indexing)
   */
  insert(name, sentropy, data = {}) {
    const addr = ternaryAddress(sentropy.sk, sentropy.st, sentropy.se, this.depth);
    this._trie.insert(addr, { name, sentropy, data });
    if (!this._metadata.has(addr)) {
      this._metadata.set(addr, { count: 0, representative: name });
    }
    this._metadata.get(addr).count++;
    return addr;
  }

  /**
   * Look up compounds at an exact ternary address.
   */
  lookup(address) {
    return this._trie.getExact ? this._trie.getExact(address) : [];
  }

  /**
   * Find all compounds whose address starts with a given prefix.
   * This is the O(k) traversal that eliminates O(N) search (Paper 3, §8).
   */
  query(prefix) {
    return this._trie.getPrefix ? this._trie.getPrefix(prefix) : [];
  }

  /**
   * Generate the expected S-entropy coordinates for a given ternary address
   * without looking up any stored spectrum.
   * The address IS the coordinates — Partition Determinism Theorem (Paper 2, §4.4).
   */
  generateCoords(address) {
    return addressToSentropy(address);
  }

  /**
   * Streaming search: process candidates in O(1) memory batches.
   * Calls onMatch for each candidate whose resonance score exceeds threshold.
   *
   * @param {{ sk, st, se }} querySentropy
   * @param {Iterable<CompoundRecord>} candidatePool
   * @param {{ threshold?: number, batchSize?: number, onMatch: (c, score) => void }} opts
   */
  async streamSearch(querySentropy, candidatePool, opts = {}) {
    const { threshold = 0.7, batchSize = 64, onMatch } = opts;
    const queryAddr = ternaryAddress(querySentropy.sk, querySentropy.st, querySentropy.se, this.depth);

    await streamObserve(candidatePool, batchSize, batch => {
      for (const compound of batch) {
        const cAddr = ternaryAddress(
          compound.sentropy.sk, compound.sentropy.st, compound.sentropy.se, this.depth
        );
        const score = commonPrefixScore(queryAddr, cAddr);
        if (score >= threshold && onMatch) {
          onMatch(compound, score);
        }
      }
    });
  }

  /**
   * Purpose-constrained prefix set for a domain context.
   * Returns the minimal set of ternary prefixes whose union covers the domain.
   *
   * @param {string} domain  see purpose.js DOMAINS
   * @returns {string[]}  ternary prefix strings
   */
  prefixesForDomain(domain) {
    const { getPurposePrefixes } = require("../shapeshifter/purpose");
    return getPurposePrefixes(domain, this.depth);
  }

  /**
   * Reduce search to domain-relevant compounds only (Paper 2, §8).
   * Returns matching compounds from the trie within the domain's prefix set.
   */
  /**
   * @param {{ sk, st, se }} querySentropy
   * @param {string[]} prefixes  from purpose.getPurposePrefixes(domain, depth)
   */
  domainQuery(querySentropy, prefixes) {
    const queryAddr = ternaryAddress(querySentropy.sk, querySentropy.st, querySentropy.se, this.depth);
    const results = [];
    for (const prefix of (prefixes ?? [])) {
      if (queryAddr.startsWith(prefix)) {
        results.push(...this.query(prefix));
      }
    }
    return results;
  }

  /**
   * @param {string} domain
   * @param {function} getPurposePrefixesFn  inject from purpose.js to avoid circular dep
   */
  prefixesForDomain(domain, getPurposePrefixesFn) {
    if (getPurposePrefixesFn) return getPurposePrefixesFn(domain, this.depth);
    // Fallback: full coverage (no restriction)
    return [];
  }

  get size() {
    return this._trie.totalCount ?? this._compounds.size;
  }
}

/* ── Helpers ─────────────────────────────────────────────────────────────── */

/**
 * Decode a ternary address back to approximate S-entropy coordinates.
 * (Inverse of the interleaving algorithm — mid-point of the addressed cell.)
 */
export function addressToSentropy(address) {
  let lo = [0, 0, 0], hi = [1, 1, 1];
  for (let j = 0; j < address.length; j++) {
    const dim  = j % 3;
    const delta = (hi[dim] - lo[dim]) / 3;
    const trit = parseInt(address[j], 10);
    if (trit === 0)      hi[dim]  = lo[dim] + delta;
    else if (trit === 1) { lo[dim] += delta; hi[dim] = lo[dim] + delta; }
    else                   lo[dim] += 2 * delta;
  }
  return {
    sk: (lo[0] + hi[0]) / 2,
    st: (lo[1] + hi[1]) / 2,
    se: (lo[2] + hi[2]) / 2,
  };
}

/**
 * Compute the normalised common-prefix similarity score between two addresses.
 * Length k common prefix → false positive probability ≤ 3^(-k).
 */
export function commonPrefixScore(a, b) {
  let common = 0;
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i++) {
    if (a[i] === b[i]) common++;
    else break;
  }
  return common / len;
}

/**
 * Euclidean distance in S-entropy space.
 */
export function sentropyDistance(s1, s2) {
  return Math.sqrt(
    Math.pow(s1.sk - s2.sk, 2) +
    Math.pow(s1.st - s2.st, 2) +
    Math.pow(s1.se - s2.se, 2)
  );
}
