"""
Shared machinery for the peptide-mass-invariance validation suite.

Contains the contact-graph primitives (cut key, floor, residual), the
in-silico digestion used to build mapping sets, and the JSON writer.

Nothing here decides an expectation. Every registered expectation lives
in the experiment script that tests it, stated before the measurement is
taken, so that the recorded artefact carries both the prediction and the
outcome.
"""
from __future__ import annotations

import io
import json
import os
import random
from typing import Dict, List, Sequence, Set, Tuple

import networkx as nx

# =====================================================================
#  Paths
# =====================================================================

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
MEDIUM = "__medium__"


def ensure_results() -> str:
    if not os.path.isdir(RESULTS):
        os.makedirs(RESULTS)
    return RESULTS


def write_result(name: str, payload: dict) -> str:
    """Write one experiment artefact. Returns the path written."""
    ensure_results()
    path = os.path.join(RESULTS, name)
    with io.open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False)
    return path


# =====================================================================
#  Contact graph primitives  (paper sections 2-5)
# =====================================================================

def contact_graph(items: Sequence[str],
                  medium_weights: Dict[str, float],
                  item_edges: Sequence[Tuple[str, str, float]] = ()) -> nx.Graph:
    """
    Build a contact graph: every item adjacent to the medium, plus any
    committed item-item edges. Definition 2.1 of the paper.
    """
    g = nx.Graph()
    g.add_node(MEDIUM)
    for v in items:
        g.add_node(v)
        w = medium_weights[v]
        if w <= 0:
            raise ValueError("medium weights must be strictly positive")
        g.add_edge(v, MEDIUM, capacity=float(w))
    for u, v, w in item_edges:
        if u == MEDIUM or v == MEDIUM:
            raise ValueError("item_edges must join two items")
        # Committing the same contact twice adds capacity rather than
        # replacing it; a contact is one-use per commitment event.
        if g.has_edge(u, v):
            g[u][v]["capacity"] += float(w)
        else:
            g.add_edge(u, v, capacity=float(w))
    return g


def cut_key(g: nx.Graph, v: str) -> Tuple[float, int]:
    """
    The cut key kappa(v) = (sigma(v), delta(v)) of Definition 2.2:
    minimum weight of a cut separating v from the medium, and the size
    of the minimising side. Computed by max-flow (Remark 2.3).
    """
    sigma, (side_v, _side_m) = nx.minimum_cut(g, v, MEDIUM, capacity="capacity")
    return float(sigma), int(len(side_v))


def floor_of(g: nx.Graph) -> float:
    """
    beta = min over items of the medium-incident weight. This is the
    quantity the proof of Theorem 3.1 identifies as the floor.
    """
    return min(d["capacity"] for u, v, d in g.edges(data=True)
               if u == MEDIUM or v == MEDIUM)


def residual(g: nx.Graph) -> float:
    """Global residual R = sum_v (sigma(v) - beta), Definition 5.1."""
    beta = floor_of(g)
    items = [n for n in g.nodes if n != MEDIUM]
    return float(sum(cut_key(g, v)[0] - beta for v in items))


# =====================================================================
#  Proteome and in-silico digestion  (paper section 7)
# =====================================================================

AA = "ACDEFGHIKLMNPQRSTVWY"


def synth_proteome(n_proteins: int,
                   mean_len: int = 320,
                   n_families: int = 12,
                   n_domains: int = 40,
                   domain_len: int = 14,
                   domains_per_protein: int = 4,
                   seed: int = 20260824) -> Dict[str, str]:
    """
    Synthesise a proteome whose peptide sharing arises from CONSERVED
    DOMAINS, which is how sharing actually arises in real proteomes
    (gene families, isoforms, conserved motifs).

    A first generator mutated a family ancestor residue-by-residue at 72%
    identity. That produced almost no shared tryptic peptides --- 5068 of
    5120 were unique --- because a peptide of length >= 6 survives point
    mutation only rarely. The failure is recorded in the experiment
    artefact because it identifies what the model needs: sharing comes
    from exactly-conserved blocks, not from average sequence identity.

    Here each protein is assembled from a family-specific backbone plus
    `domains_per_protein` domains drawn from a shared pool of `n_domains`
    exactly-conserved blocks. Proteins in the same family draw from an
    overlapping domain subset, so mapping sets are organised by family
    --- the structure Assumption 7.6 asserts and the independence model
    of Theorem 7.4 lacks.
    """
    rnd = random.Random(seed)

    # Pool of exactly-conserved domains. Each ends in K or R so that
    # tryptic digestion yields whole domains as peptides.
    domains = []
    for _ in range(n_domains):
        core = "".join(rnd.choice("ACDEFGHILMNPQSTVWY")
                       for _ in range(domain_len - 1))
        domains.append(core + rnd.choice("KR"))

    # Each family favours a contiguous slice of the domain pool.
    per_fam = max(2, n_domains // n_families)

    proteome: Dict[str, str] = {}
    for i in range(n_proteins):
        fam = i % n_families
        lo = (fam * per_fam) % n_domains
        fam_pool = [domains[(lo + j) % n_domains]
                    for j in range(per_fam * 2)]

        # Filler blocks are themselves tryptic-sized so that a protein
        # digests into peptides inside the [6, 30] window.
        filler_len = 18
        parts = []
        for d in range(domains_per_protein):
            filler = "".join(rnd.choice("ACDEFGHILMNPQSTVWY")
                             for _ in range(filler_len - 1))
            parts.append(filler + rnd.choice("KR"))
            parts.append(rnd.choice(fam_pool))
        tail = "".join(rnd.choice("ACDEFGHILMNPQSTVWY")
                       for _ in range(filler_len))
        parts.append(tail)
        proteome["P%04d_fam%02d" % (i, fam)] = "".join(parts)
    return proteome


def shuffled_proteome(proteome: Dict[str, str], seed: int = 7) -> Dict[str, str]:
    """
    Negative control N2: destroy homology structure by shuffling residues
    within each protein. Composition and length are preserved exactly, so
    any change in behaviour is attributable to structure alone.
    """
    rnd = random.Random(seed)
    out = {}
    for name, seq in proteome.items():
        chars = list(seq)
        rnd.shuffle(chars)
        out[name] = "".join(chars)
    return out


def digest(seq: str, min_len: int = 6, max_len: int = 30,
           missed: int = 0) -> Set[str]:
    """
    Tryptic in-silico digestion: cleave C-terminal to K or R, not before
    P (Olsen et al. 2004). Returns the set of peptides within the length
    window, including `missed` cleavages.
    """
    sites = [0]
    for i, c in enumerate(seq):
        if c in "KR" and not (i + 1 < len(seq) and seq[i + 1] == "P"):
            sites.append(i + 1)
    if sites[-1] != len(seq):
        sites.append(len(seq))

    peps: Set[str] = set()
    for a in range(len(sites) - 1):
        for extra in range(missed + 1):
            b = a + 1 + extra
            if b >= len(sites):
                break
            p = seq[sites[a]:sites[b]]
            if min_len <= len(p) <= max_len:
                peps.add(p)
    return peps


def mapping_sets(proteome: Dict[str, str], **kw) -> Dict[str, Set[str]]:
    """
    amb(q) = { P : q occurs in P }, Definition 7.1. Built by digesting
    every protein and inverting the peptide->protein relation.
    """
    inv: Dict[str, Set[str]] = {}
    for name, seq in proteome.items():
        for p in digest(seq, **kw):
            inv.setdefault(p, set()).add(name)
    return inv


# =====================================================================
#  Inference procedures  (paper section 7.3)
# =====================================================================

def intersect_ordered(observed: List[str],
                      amb: Dict[str, Set[str]],
                      universe: Set[str]) -> List[int]:
    """
    Intersect mapping sets in the given order, recording |amb(Q_t)| after
    each step. Construction 7.7 steps 1-3.
    """
    cur = set(universe)
    trace = []
    for q in observed:
        cur = cur & amb[q]
        trace.append(len(cur))
        if not cur:
            break
    return trace


def closure_step(trace: List[int]) -> int:
    """
    The first index at which the admissible set stops changing --- the
    closure point of Definition 6.3. Returns a 1-based step count.
    """
    for i in range(1, len(trace)):
        if trace[i] == trace[i - 1]:
            return i
    return len(trace)


def greedy_parsimony(observed: List[str],
                     amb: Dict[str, Set[str]]) -> List[str]:
    """
    Greedy minimum set cover over proteins: repeatedly take the protein
    explaining the most yet-unexplained peptides. This is the standard
    parsimony heuristic (Nesvizhskii et al. 2003); exact minimum set
    cover is NP-hard (Proposition 7.9).
    """
    uncovered = set(observed)
    prot_to_peps: Dict[str, Set[str]] = {}
    for q in observed:
        for p in amb[q]:
            prot_to_peps.setdefault(p, set()).add(q)

    chosen: List[str] = []
    while uncovered:
        best, best_n = None, 0
        for p, peps in prot_to_peps.items():
            n = len(peps & uncovered)
            if n > best_n:
                best, best_n = p, n
        if best is None:
            break
        chosen.append(best)
        uncovered -= prot_to_peps[best]
    return chosen


def peptides_until_found(observed: List[str],
                         amb: Dict[str, Set[str]],
                         universe: Set[str],
                         target: str) -> int:
    """
    Number of peptides consumed, in the given order, before the
    admissible set is reduced to {target} (or as far as it will go).
    Returns len(observed)+1 if the target is never isolated.
    """
    cur = set(universe)
    for i, q in enumerate(observed, start=1):
        cur = cur & amb[q]
        if cur == {target}:
            return i
    return len(observed) + 1
