/**
 * Ternary trie for O(k) search in S-entropy space.
 *
 * The trie itself is sparse — we only allocate nodes along observed paths.
 * Insertion is O(k); prefix search returns all addresses sharing a prefix
 * and runs in O(|matches|).
 *
 * Because the address IS the molecular identity, the trie is deterministic:
 * two identical compounds map to the same address, collisions = genuine
 * identity (at the given resolution).
 */

/**
 * @typedef {Object} TrieEntry
 * @property {string} address
 * @property {*} payload
 */

export class TernaryTrie {
  constructor() {
    this.root = makeNode();
    this.size = 0;
  }

  /**
   * Insert an address with arbitrary payload.
   * @param {string} address
   * @param {*} payload
   */
  insert(address, payload) {
    let node = this.root;
    for (let i = 0; i < address.length; i++) {
      const t = address.charCodeAt(i) - 48; // '0' = 48
      if (t < 0 || t > 2) continue;
      if (!node.children[t]) node.children[t] = makeNode();
      node = node.children[t];
    }
    const isNew = node.entries.length === 0;
    node.entries.push({ address, payload });
    if (isNew) this.size++;
  }

  /**
   * Exact lookup — returns all entries stored at this exact address.
   * @param {string} address
   * @returns {TrieEntry[]}
   */
  lookup(address) {
    const node = this._descend(address);
    return node ? node.entries.slice() : [];
  }

  /**
   * Prefix search — returns every entry whose address begins with `prefix`.
   * Traversal cost is O(|matches|).
   * @param {string} prefix
   * @returns {TrieEntry[]}
   */
  prefixSearch(prefix) {
    const node = this._descend(prefix);
    if (!node) return [];
    const out = [];
    collectEntries(node, out);
    return out;
  }

  /**
   * Nearest neighbours: for a query address, find the entries with the
   * longest shared prefix. Returns at most `limit` results ranked by
   * prefix length (descending).
   *
   * Strategy: try progressively shorter prefixes until we accumulate
   * enough results.
   *
   * @param {string} address
   * @param {number} [limit=10]
   * @returns {Array<{entry: TrieEntry, prefixLength: number}>}
   */
  nearest(address, limit = 10) {
    const hits = [];
    const seen = new Set();

    for (let k = address.length; k > 0 && hits.length < limit; k--) {
      const prefix = address.substring(0, k);
      const matches = this.prefixSearch(prefix);
      for (const m of matches) {
        if (seen.has(m.address)) continue;
        seen.add(m.address);
        hits.push({ entry: m, prefixLength: k });
        if (hits.length >= limit) break;
      }
    }

    return hits;
  }

  /**
   * Remove all entries at an exact address.
   * @param {string} address
   * @returns {boolean} true if anything removed
   */
  remove(address) {
    const node = this._descend(address);
    if (!node || node.entries.length === 0) return false;
    this.size -= node.entries.length;
    node.entries = [];
    return true;
  }

  /**
   * Iterate every entry in the trie (order: depth-first).
   * @returns {Generator<TrieEntry>}
   */
  *entries() {
    const stack = [this.root];
    while (stack.length) {
      const node = stack.pop();
      for (const entry of node.entries) yield entry;
      for (let i = 2; i >= 0; i--) {
        if (node.children[i]) stack.push(node.children[i]);
      }
    }
  }

  _descend(str) {
    let node = this.root;
    for (let i = 0; i < str.length; i++) {
      const t = str.charCodeAt(i) - 48;
      if (t < 0 || t > 2) return null;
      const child = node.children[t];
      if (!child) return null;
      node = child;
    }
    return node;
  }
}

function makeNode() {
  return {
    children: [null, null, null],
    entries: [],
  };
}

function collectEntries(node, out) {
  for (const entry of node.entries) out.push(entry);
  for (let i = 0; i < 3; i++) {
    if (node.children[i]) collectEntries(node.children[i], out);
  }
}
