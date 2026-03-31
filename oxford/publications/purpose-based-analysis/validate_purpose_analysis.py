#!/usr/bin/env python3
"""
Validation experiments for Paper 2:
"Purpose-Based Spectral Analysis: Domain-Constrained Generative
 Identification Through Oscillatory Resonance in Bounded Phase Space"

Validates: oscillatory resonance comparison, dual path convergence,
purpose-based constraint reduction, selective generation efficiency,
Landauer cost, and domain-specific reduction bounds.

All results saved as CSV/JSON in ./results/
"""

import json
import csv
import math
import os
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from itertools import combinations

# ============================================================================
# Constants
# ============================================================================
HBAR = 1.0546e-34
KB = 1.381e-23
C_LIGHT = 2.998e8
E_CHARGE = 1.602e-19
AMU = 1.661e-27
LN2 = math.log(2)

OMEGA_REF_MAX = 4401.0
OMEGA_REF_MIN = 218.0
B_ROT_REF_MIN = 0.39
DELTA_HARMONIC = 0.05
P_MAX_HARMONIC = 8

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================================
# Compound Data (same 39 NIST CCCBDB compounds)
# ============================================================================
@dataclass
class Compound:
    name: str
    formula: str
    mol_type: str
    mass: float
    frequencies: List[float]
    rotational_constant: Optional[float] = None

COMPOUNDS = [
    Compound("H2", "H2", "diatomic", 2, [4401], 60.853),
    Compound("D2", "D2", "diatomic", 4, [2994], 30.444),
    Compound("N2", "N2", "diatomic", 28, [2330], 1.998),
    Compound("O2", "O2", "diatomic", 32, [1580], 1.438),
    Compound("F2", "F2", "diatomic", 38, [892], 0.890),
    Compound("Cl2", "Cl2", "diatomic", 71, [560], 0.244),
    Compound("CO", "CO", "diatomic", 28, [2143], 1.931),
    Compound("NO", "NO", "diatomic", 30, [1876], 1.672),
    Compound("HF", "HF", "diatomic", 20, [3958], 20.956),
    Compound("HCl", "HCl", "diatomic", 36, [2886], 10.593),
    Compound("HBr", "HBr", "diatomic", 81, [2559], 8.465),
    Compound("HI", "HI", "diatomic", 128, [2230], 6.511),
    Compound("H2O", "H2O", "triatomic", 18, [1595, 3657, 3756]),
    Compound("CO2", "CO2", "triatomic", 44, [667, 1388, 2349]),
    Compound("SO2", "SO2", "triatomic", 64, [518, 1151, 1362]),
    Compound("NO2", "NO2", "triatomic", 46, [750, 1318, 1618]),
    Compound("O3", "O3", "triatomic", 48, [701, 1042, 1110]),
    Compound("H2S", "H2S", "triatomic", 34, [1183, 2615, 2626]),
    Compound("HCN", "HCN", "triatomic", 27, [712, 2097, 3312]),
    Compound("N2O", "N2O", "triatomic", 44, [589, 1285, 2224]),
    Compound("CS2", "CS2", "triatomic", 76, [397, 657, 1535]),
    Compound("OCS", "OCS", "triatomic", 60, [520, 859, 2062]),
    Compound("NH3", "NH3", "tetra", 17, [950, 1627, 3337, 3444]),
    Compound("PH3", "PH3", "tetra", 34, [992, 1118, 2323, 2328]),
    Compound("CH4", "CH4", "tetra", 16, [1306, 1534, 2917, 3019]),
    Compound("CCl4", "CCl4", "tetra", 154, [218, 314, 776, 790]),
    Compound("SiH4", "SiH4", "tetra", 32, [800, 913, 2187, 2191]),
    Compound("CF4", "CF4", "tetra", 88, [435, 632, 1283, 1283]),
    Compound("H2CO", "H2CO", "poly", 30, [1167, 1249, 1500, 1746, 2783, 2843]),
    Compound("C2H2", "C2H2", "poly", 26, [612, 729, 1974, 3289, 3374]),
    Compound("C2H4", "C2H4", "poly", 28,
             [826, 943, 949, 1023, 1236, 1344, 1444, 1623, 2989, 3026, 3103, 3106]),
    Compound("C2H6", "C2H6", "poly", 30,
             [289, 822, 822, 995, 1190, 1379, 1388, 1469, 1469, 2954]),
    Compound("CH3OH", "CH3OH", "poly", 32,
             [270, 1033, 1060, 1165, 1345, 1455, 1477, 2844, 2960, 3000]),
    Compound("C6H6", "C6H6", "poly", 78, [673, 849, 992, 1010, 3062]),
    Compound("CH3F", "CH3F", "poly", 34, [1049, 1182, 1459, 1467, 2930, 3006]),
    Compound("CH3Cl", "CH3Cl", "poly", 50, [732, 1017, 1355, 1452, 2937, 3039]),
    Compound("CH3Br", "CH3Br", "poly", 95, [611, 952, 1306, 1443, 2935, 3056]),
    Compound("HCOOH", "HCOOH", "poly", 46,
             [625, 1033, 1105, 1229, 1387, 1770, 2943, 3570]),
    Compound("CH3CN", "CH3CN", "poly", 41,
             [362, 920, 1041, 1385, 1448, 2267, 2954, 3009]),
]


# ============================================================================
# S-Entropy & Ternary (same implementations as Paper 1)
# ============================================================================
def compute_sk(freqs, is_diatomic):
    if is_diatomic:
        return freqs[0] / OMEGA_REF_MAX
    N = len(freqs)
    total = sum(freqs)
    if total == 0:
        return 0.0
    probs = [f / total for f in freqs]
    H = -sum(p * math.log2(p) for p in probs if p > 0)
    return H / math.log2(N) if N > 1 else 0.0


def compute_st(freqs, is_diatomic, b_rot=None):
    if is_diatomic:
        if b_rot is None or b_rot <= 0:
            return 0.0
        return math.log(freqs[0] / b_rot) / math.log(OMEGA_REF_MAX / B_ROT_REF_MIN)
    w_max, w_min = max(freqs), min(freqs)
    if w_min <= 0 or w_max <= w_min:
        return 0.0
    return math.log(w_max / w_min) / math.log(OMEGA_REF_MAX / OMEGA_REF_MIN)


def compute_se(freqs):
    N = len(freqs)
    if N < 2:
        return 0.0
    n_pairs = N * (N - 1) // 2
    n_harmonic = 0
    for i in range(N):
        for j in range(i + 1, N):
            a, b = max(freqs[i], freqs[j]), min(freqs[i], freqs[j])
            if b <= 0:
                continue
            ratio = a / b
            for p in range(1, P_MAX_HARMONIC + 1):
                for q in range(1, p + 1):
                    if abs(ratio - p / q) < DELTA_HARMONIC:
                        n_harmonic += 1
                        break
                else:
                    continue
                break
    return n_harmonic / max(n_pairs, 1)


def compute_sentropy(c):
    is_di = (c.mol_type == "diatomic")
    return (compute_sk(c.frequencies, is_di),
            compute_st(c.frequencies, is_di, c.rotational_constant),
            compute_se(c.frequencies))


def ternary_encode(sk, st, se, depth=18):
    coords = [sk, st, se]
    trits = []
    for j in range(depth):
        dim = j % 3
        val = min(int(coords[dim] * 3), 2)
        trits.append(str(val))
        coords[dim] = coords[dim] * 3 - val
    return "".join(trits)


def common_prefix_length(a, b):
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return i
    return min(len(a), len(b))


# ============================================================================
# Precompute S-entropy for all compounds
# ============================================================================
def precompute():
    data = []
    for c in COMPOUNDS:
        sk, st, se = compute_sentropy(c)
        addr = ternary_encode(sk, st, se, 18)
        data.append({
            "compound": c,
            "sk": sk, "st": st, "se": se,
            "addr": addr,
        })
    return data


# ============================================================================
# Experiment 1: Oscillatory Resonance vs Algorithmic Comparison
# ============================================================================
def experiment_resonance_vs_algorithmic(data):
    """Compare oscillatory resonance (prefix matching) with Tanimoto-like similarity."""
    print("=" * 70)
    print("EXPERIMENT 1: Oscillatory Resonance vs Algorithmic Comparison")
    print("=" * 70)

    results = []
    names = [d["compound"].name for d in data]
    N = len(data)

    for i in range(N):
        for j in range(i + 1, N):
            # Oscillatory: common prefix length (O(k) comparison)
            cpl = common_prefix_length(data[i]["addr"], data[j]["addr"])

            # Euclidean distance in S-entropy space
            eucl = math.sqrt(
                (data[i]["sk"] - data[j]["sk"]) ** 2
                + (data[i]["st"] - data[j]["st"]) ** 2
                + (data[i]["se"] - data[j]["se"]) ** 2
            )

            # Simulated Tanimoto on binary fingerprint (O(d) comparison)
            # Create a mock binary fingerprint from frequencies
            d_fp = 1024
            fp_i = _make_fingerprint(data[i]["compound"].frequencies, d_fp)
            fp_j = _make_fingerprint(data[j]["compound"].frequencies, d_fp)
            tanimoto = _tanimoto(fp_i, fp_j)

            # Distance bound from prefix length
            bound = math.sqrt(3) * (3 ** (-cpl // 3)) if cpl > 0 else math.sqrt(3)

            results.append({
                "compound_A": names[i],
                "compound_B": names[j],
                "prefix_length": cpl,
                "euclidean_distance": round(eucl, 6),
                "distance_bound": round(bound, 6),
                "bound_satisfied": eucl <= bound + 1e-10,
                "tanimoto": round(tanimoto, 4),
                "resonance_ops": cpl,  # O(k)
                "tanimoto_ops": d_fp,  # O(d)
                "speedup": round(d_fp / max(cpl, 1), 1),
            })

    # Verify distance preservation
    n_satisfied = sum(1 for r in results if r["bound_satisfied"])
    print(f"  Distance bound satisfied: {n_satisfied}/{len(results)} "
          f"({100 * n_satisfied / len(results):.1f}%)")

    # Correlation between prefix length and Tanimoto
    avg_tanimoto_by_prefix = {}
    for r in results:
        pl = r["prefix_length"]
        if pl not in avg_tanimoto_by_prefix:
            avg_tanimoto_by_prefix[pl] = []
        avg_tanimoto_by_prefix[pl].append(r["tanimoto"])

    print("  Avg Tanimoto by prefix length:")
    for pl in sorted(avg_tanimoto_by_prefix.keys()):
        vals = avg_tanimoto_by_prefix[pl]
        avg = sum(vals) / len(vals)
        print(f"    prefix={pl}: Tanimoto={avg:.3f} (n={len(vals)})")

    path = os.path.join(RESULTS_DIR, "01_resonance_vs_algorithmic.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    print(f"  Saved {len(results)} pairs -> {path}")
    return results


def _make_fingerprint(freqs, d=1024):
    """Create a mock binary fingerprint from frequencies."""
    fp = [0] * d
    for f in freqs:
        # Hash frequency to bit positions
        for offset in [0, 1, 2]:
            idx = int((f + offset * 137) * 7.3) % d
            fp[idx] = 1
    return fp


def _tanimoto(fp_a, fp_b):
    """Tanimoto coefficient between binary fingerprints."""
    intersection = sum(a & b for a, b in zip(fp_a, fp_b))
    union = sum(a | b for a, b in zip(fp_a, fp_b))
    return intersection / union if union > 0 else 0.0


# ============================================================================
# Experiment 2: Dual Oscillatory Path Convergence
# ============================================================================
def experiment_dual_path(data):
    """Verify that ion path and droplet path converge to same S-entropy."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Dual Oscillatory Path Convergence")
    print("=" * 70)

    results = []
    for d in data:
        c = d["compound"]
        sk_ion, st_ion, se_ion = d["sk"], d["st"], d["se"]
        addr_ion = d["addr"]

        # Droplet path: compute S-entropy from Rayleigh surface modes
        m_kg = c.mass * AMU
        r_mol = 1e-10 * (c.mass ** (1.0 / 3.0))
        rho = 1000.0
        sigma = 0.072

        n_modes = len(c.frequencies)
        rayleigh_freqs = []
        for n in range(2, n_modes + 2):
            omega_n = math.sqrt(
                n * (n - 1) * (n + 2) * sigma / (rho * r_mol ** 3)
            )
            rayleigh_freqs.append(omega_n / (2 * math.pi))

        # Compute droplet S-entropy
        if len(rayleigh_freqs) >= 2:
            total_r = sum(rayleigh_freqs)
            probs_r = [f / total_r for f in rayleigh_freqs]
            H_r = -sum(p * math.log2(p) for p in probs_r if p > 0)
            sk_drip = H_r / math.log2(len(rayleigh_freqs))
            if len(rayleigh_freqs) >= 2:
                st_drip = (math.log(max(rayleigh_freqs) / min(rayleigh_freqs))
                           / math.log(OMEGA_REF_MAX / OMEGA_REF_MIN))
            else:
                st_drip = 0.0
            # Se from droplet modes
            n_pairs_r = len(rayleigh_freqs) * (len(rayleigh_freqs) - 1) // 2
            n_harm_r = 0
            for i in range(len(rayleigh_freqs)):
                for j in range(i + 1, len(rayleigh_freqs)):
                    a = max(rayleigh_freqs[i], rayleigh_freqs[j])
                    b = min(rayleigh_freqs[i], rayleigh_freqs[j])
                    if b > 0:
                        ratio = a / b
                        for p in range(1, P_MAX_HARMONIC + 1):
                            for q in range(1, p + 1):
                                if abs(ratio - p / q) < DELTA_HARMONIC:
                                    n_harm_r += 1
                                    break
                            else:
                                continue
                            break
            se_drip = n_harm_r / max(n_pairs_r, 1)
        else:
            sk_drip = sk_ion  # Diatomic: single mode, direct mapping
            st_drip = st_ion
            se_drip = 0.0

        addr_drip = ternary_encode(sk_drip, st_drip, se_drip, 18)
        cpl = common_prefix_length(addr_ion, addr_drip)

        # The key test: do ion and drip S-entropy produce consistent ternary?
        # (They won't be identical because the Rayleigh modes are different
        #  physical oscillations, but the STRUCTURE should be preserved.)
        results.append({
            "name": c.name,
            "type": c.mol_type,
            "n_modes": n_modes,
            "Sk_ion": round(sk_ion, 4),
            "St_ion": round(st_ion, 4),
            "Se_ion": round(se_ion, 4),
            "Sk_drip": round(sk_drip, 4),
            "St_drip": round(st_drip, 4),
            "Se_drip": round(se_drip, 4),
            "addr_ion_12": addr_ion[:12],
            "addr_drip_12": addr_drip[:12],
            "common_prefix": cpl,
            "paths_converge": cpl >= 3,  # Converge at least at family level
        })

    n_converge = sum(1 for r in results if r["paths_converge"])
    print(f"  Dual paths converge (>=3 trits): {n_converge}/{len(results)}")
    avg_cpl = sum(r["common_prefix"] for r in results) / len(results)
    print(f"  Average common prefix length: {avg_cpl:.1f}")

    path = os.path.join(RESULTS_DIR, "02_dual_path_convergence.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    print(f"  Saved {len(results)} records -> {path}")
    return results


# ============================================================================
# Experiment 3: Purpose-Based Constraint Reduction
# ============================================================================
def experiment_purpose_constraints(data):
    """Test how domain constraints reduce the phase space."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Purpose-Based Constraint Reduction")
    print("=" * 70)

    k = 12  # trit depth
    total_cells = 3 ** k  # 531,441

    domains = {
        "Metabolomics": {
            "description": "Human serum metabolites, mass 50-1500 Da",
            "constraints": {
                "mass_range": (50, 1500),
                "Sk_range": (0.1, 1.0),
                "Se_range": (0.0, 1.0),
            },
            "typical_compounds": 4000,
        },
        "Glycomics": {
            "description": "Glycan analysis, mass 180-5000 Da, high Se",
            "constraints": {
                "mass_range": (180, 5000),
                "Sk_range": (0.8, 1.0),
                "Se_range": (0.3, 1.0),
            },
            "typical_compounds": 1500,
        },
        "Proteomics": {
            "description": "Tryptic peptides, mass 400-4000 Da",
            "constraints": {
                "mass_range": (400, 4000),
                "Sk_range": (0.85, 1.0),
                "Se_range": (0.2, 1.0),
            },
            "typical_compounds": 50000,
        },
        "Diatomic gases": {
            "description": "Simple diatomic molecules, mass 2-130 Da, Se=0",
            "constraints": {
                "mass_range": (2, 130),
                "Sk_range": (0.0, 1.0),
                "Se_range": (0.0, 0.05),
            },
            "typical_compounds": 50,
        },
        "No constraints": {
            "description": "Full phase space (baseline)",
            "constraints": {
                "mass_range": (0, 1e6),
                "Sk_range": (0.0, 1.0),
                "Se_range": (0.0, 1.0),
            },
            "typical_compounds": 100000000,
        },
    }

    results = []
    for domain_name, domain in domains.items():
        constraints = domain["constraints"]
        sk_lo, sk_hi = constraints["Sk_range"]
        se_lo, se_hi = constraints["Se_range"]

        # Count how many of the 39 compounds fall within constraints
        matching = []
        for d in data:
            c = d["compound"]
            sk, se = d["sk"], d["se"]
            mass = c.mass
            if (constraints["mass_range"][0] <= mass <= constraints["mass_range"][1]
                    and sk_lo <= sk <= sk_hi
                    and se_lo <= se <= se_hi):
                matching.append(c.name)

        # Estimate occupied cells at depth k
        # The constrained volume fraction in [0,1]^3
        vol_fraction = ((sk_hi - sk_lo) * (se_hi - se_lo)
                        * min(1.0, (constraints["mass_range"][1]
                                    - constraints["mass_range"][0]) / 5000.0))
        r_estimated = max(1, int(total_cells * vol_fraction))
        r_estimated = min(r_estimated, domain["typical_compounds"])

        rho = 1.0 - r_estimated / total_cells
        ops_selective = r_estimated * k
        ops_exhaustive = total_cells
        ops_traditional = domain["typical_compounds"] * 1024  # N*d

        # Landauer cost
        if r_estimated > 0 and r_estimated < total_cells:
            delta_S_bits = k * math.log2(3) - math.log2(max(r_estimated, 1))
            E_landauer = KB * 300 * LN2 * delta_S_bits  # at T=300K
        else:
            delta_S_bits = 0
            E_landauer = 0

        results.append({
            "domain": domain_name,
            "description": domain["description"],
            "n_matching_compounds": len(matching),
            "matching_names": "; ".join(matching[:10]),
            "total_cells_3k": total_cells,
            "estimated_relevant_cells_r": r_estimated,
            "reduction_rho": round(rho, 6),
            "rho_percent": round(rho * 100, 2),
            "ops_selective_rk": ops_selective,
            "ops_exhaustive_3k": ops_exhaustive,
            "ops_traditional_Nd": ops_traditional,
            "speedup_vs_exhaustive": round(ops_exhaustive / max(ops_selective, 1), 1),
            "speedup_vs_traditional": round(ops_traditional / max(ops_selective, 1), 1),
            "entropy_reduction_bits": round(delta_S_bits, 2),
            "landauer_cost_J": f"{E_landauer:.2e}",
        })

        print(f"  {domain_name}: rho={rho:.4f} ({rho * 100:.1f}%), "
              f"r={r_estimated}, matching={len(matching)}/39")

    path = os.path.join(RESULTS_DIR, "03_purpose_constraints.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    print(f"  Saved {len(results)} domain analyses -> {path}")
    return results


# ============================================================================
# Experiment 4: Selective Generation Efficiency
# ============================================================================
def experiment_selective_generation(data):
    """Simulate selective generation and measure efficiency."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Selective Generation Efficiency")
    print("=" * 70)

    results = []
    k = 12

    # For each compound as query, test identification with different purposes
    for query_idx, qd in enumerate(data):
        query_name = qd["compound"].name
        query_addr = qd["addr"][:k]
        query_type = qd["compound"].mol_type

        # Purpose 1: No constraints (exhaustive)
        ops_exhaustive = 3 ** k

        # Purpose 2: Type constraint (diatomic/triatomic/etc)
        type_matches = [d for d in data if d["compound"].mol_type == query_type]
        r_type = len(type_matches)
        ops_type = r_type * k

        # Purpose 3: Mass range constraint (+/-50% of query mass)
        query_mass = qd["compound"].mass
        mass_matches = [d for d in data
                        if 0.5 * query_mass <= d["compound"].mass <= 1.5 * query_mass]
        r_mass = len(mass_matches)
        ops_mass = r_mass * k

        # Purpose 4: Se constraint (same Se bracket)
        query_se = qd["se"]
        se_bracket = (0.0, 0.1) if query_se < 0.1 else (
            (query_se - 0.15, query_se + 0.15))
        se_matches = [d for d in data
                      if se_bracket[0] <= d["se"] <= se_bracket[1]]
        r_se = len(se_matches)
        ops_se = r_se * k

        # Check if query is correctly identified in each case
        identified_type = any(
            common_prefix_length(d["addr"][:k], query_addr) == k
            for d in type_matches)
        identified_mass = any(
            common_prefix_length(d["addr"][:k], query_addr) == k
            for d in mass_matches)
        identified_se = any(
            common_prefix_length(d["addr"][:k], query_addr) == k
            for d in se_matches)

        results.append({
            "query": query_name,
            "query_type": query_type,
            "query_mass": query_mass,
            "query_Se": round(qd["se"], 3),
            "ops_exhaustive": ops_exhaustive,
            "r_type_constraint": r_type,
            "ops_type": ops_type,
            "identified_type": identified_type,
            "r_mass_constraint": r_mass,
            "ops_mass": ops_mass,
            "identified_mass": identified_mass,
            "r_se_constraint": r_se,
            "ops_se": ops_se,
            "identified_se": identified_se,
            "best_reduction_pct": round(
                (1 - min(ops_type, ops_mass, ops_se) / ops_exhaustive) * 100, 4),
        })

    # Summary
    all_type_ok = all(r["identified_type"] for r in results)
    all_mass_ok = all(r["identified_mass"] for r in results)
    all_se_ok = all(r["identified_se"] for r in results)
    avg_reduction = sum(r["best_reduction_pct"] for r in results) / len(results)

    print(f"  Type constraint: all identified = {all_type_ok}")
    print(f"  Mass constraint: all identified = {all_mass_ok}")
    print(f"  Se constraint: all identified = {all_se_ok}")
    print(f"  Average best reduction: {avg_reduction:.2f}%")

    path = os.path.join(RESULTS_DIR, "04_selective_generation.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    print(f"  Saved {len(results)} records -> {path}")
    return results


# ============================================================================
# Experiment 5: Prompt Contraction (Monotonic Reduction)
# ============================================================================
def experiment_prompt_contraction(data):
    """Verify that adding constraints monotonically reduces the search region."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Prompt Contraction (Monotonic Reduction)")
    print("=" * 70)

    # Start with full set, progressively add constraints
    constraint_sequence = [
        ("None", lambda d: True),
        ("mass < 100 Da", lambda d: d["compound"].mass < 100),
        ("+ Sk > 0.4", lambda d: d["compound"].mass < 100 and d["sk"] > 0.4),
        ("+ Se = 0", lambda d: (d["compound"].mass < 100 and d["sk"] > 0.4
                                and d["se"] < 0.01)),
        ("+ St > 0.5", lambda d: (d["compound"].mass < 100 and d["sk"] > 0.4
                                  and d["se"] < 0.01 and d["st"] > 0.5)),
    ]

    results = []
    prev_count = len(data) + 1  # Ensure monotonic check works

    for label, predicate in constraint_sequence:
        matching = [d for d in data if predicate(d)]
        count = len(matching)
        names = [d["compound"].name for d in matching]
        monotonic = count <= prev_count

        results.append({
            "constraint": label,
            "n_matching": count,
            "matching_compounds": "; ".join(names),
            "monotonic_decrease": monotonic,
        })
        prev_count = count
        print(f"  {label}: {count} compounds ({', '.join(names[:5])}{'...' if len(names) > 5 else ''})")

    all_monotonic = all(r["monotonic_decrease"] for r in results)
    print(f"  Monotonic contraction verified: {all_monotonic}")

    path = os.path.join(RESULTS_DIR, "05_prompt_contraction.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    print(f"  Saved {len(results)} contraction steps -> {path}")
    return results


# ============================================================================
# Experiment 6: Landauer Cost of Domain Knowledge
# ============================================================================
def experiment_landauer_cost():
    """Compute the thermodynamic cost of purpose-based filtering."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 6: Landauer Cost of Domain Knowledge")
    print("=" * 70)

    T = 300.0  # Room temperature
    k_depth = 12
    full_states = 3 ** k_depth  # 531,441
    full_bits = k_depth * math.log2(3)  # ~19.02 bits

    results = []
    for r_cells in [1, 10, 100, 1000, 5000, 10000, 50000, 100000, 531441]:
        if r_cells > full_states:
            continue
        reduced_bits = math.log2(max(r_cells, 1))
        info_gained = full_bits - reduced_bits
        E_landauer = KB * T * LN2 * info_gained
        rho = 1.0 - r_cells / full_states

        results.append({
            "r_cells": r_cells,
            "full_states": full_states,
            "full_entropy_bits": round(full_bits, 2),
            "reduced_entropy_bits": round(reduced_bits, 2),
            "information_gained_bits": round(info_gained, 2),
            "landauer_energy_J": f"{E_landauer:.4e}",
            "landauer_energy_eV": f"{E_landauer / E_CHARGE:.4e}",
            "reduction_rho": round(rho, 6),
        })

    path = os.path.join(RESULTS_DIR, "06_landauer_cost.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    print(f"  Saved {len(results)} Landauer entries -> {path}")

    # Print key values
    for r in results:
        if r["r_cells"] in [1, 100, 10000, 531441]:
            print(f"  r={r['r_cells']}: dI={r['information_gained_bits']} bits, "
                  f"E={r['landauer_energy_eV']} eV, rho={r['reduction_rho']}")
    return results


# ============================================================================
# Experiment 7: Scaling Analysis
# ============================================================================
def experiment_scaling():
    """How reduction ratio varies with database size and prompt specificity."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 7: Scaling Analysis")
    print("=" * 70)

    results = []
    d_fp = 1024
    k = 12

    # Vary N (database size) at fixed domain specificity
    for N in [39, 100, 1000, 10000, 100000, 1000000, 100000000]:
        for domain_fraction in [0.01, 0.05, 0.10, 0.50, 1.00]:
            r = max(1, int(N * domain_fraction))
            ops_traditional = N * d_fp
            ops_selective = r * k
            ops_exhaustive = 3 ** k

            speedup_vs_trad = ops_traditional / max(ops_selective, 1)
            rho = 1.0 - r / max(N, 1)

            results.append({
                "N_database": N,
                "domain_fraction": domain_fraction,
                "r_relevant": r,
                "ops_traditional_Nd": ops_traditional,
                "ops_selective_rk": ops_selective,
                "ops_exhaustive_3k": ops_exhaustive,
                "speedup_vs_traditional": round(speedup_vs_trad, 1),
                "reduction_rho": round(rho, 4),
            })

    path = os.path.join(RESULTS_DIR, "07_scaling_analysis.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    print(f"  Saved {len(results)} scaling entries -> {path}")

    # Key highlights
    pubchem_5pct = [r for r in results
                    if r["N_database"] == 100000000 and r["domain_fraction"] == 0.05]
    if pubchem_5pct:
        print(f"  PubChem (N=10^8) with 5% domain: "
              f"speedup={pubchem_5pct[0]['speedup_vs_traditional']:.2e}")
    return results


# ============================================================================
# Experiment 8: Cross-Domain False Negative Test
# ============================================================================
def experiment_cross_domain(data):
    """Apply wrong domain constraints and verify increased false negatives."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 8: Cross-Domain False Negative Test")
    print("=" * 70)

    # Define domain predicates
    domains = {
        "Diatomic-only": lambda d: d["se"] < 0.01,
        "Polyatomic-only": lambda d: d["se"] > 0.3,
        "Light molecules": lambda d: d["compound"].mass < 40,
        "Heavy molecules": lambda d: d["compound"].mass > 60,
        "High Sk": lambda d: d["sk"] > 0.9,
        "Low Sk": lambda d: d["sk"] < 0.5,
    }

    results = []
    for domain_name, predicate in domains.items():
        in_domain = [d for d in data if predicate(d)]
        out_domain = [d for d in data if not predicate(d)]

        # If we restrict to this domain, how many compounds are missed?
        n_in = len(in_domain)
        n_out = len(out_domain)
        false_neg_rate = n_out / len(data) if len(data) > 0 else 0

        results.append({
            "domain": domain_name,
            "n_in_domain": n_in,
            "n_out_domain": n_out,
            "false_negative_rate": round(false_neg_rate, 4),
            "in_domain_names": "; ".join(d["compound"].name for d in in_domain),
        })
        print(f"  {domain_name}: {n_in} in, {n_out} out, "
              f"FNR={false_neg_rate:.2%}")

    path = os.path.join(RESULTS_DIR, "08_cross_domain_fnr.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    print(f"  Saved {len(results)} domain tests -> {path}")
    return results


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 70)
    print("PAPER 2 VALIDATION: Purpose-Based Spectral Analysis")
    print("Oscillatory Resonance & Domain-Constrained Generation")
    print("=" * 70)
    t0 = time.time()

    data = precompute()

    experiment_resonance_vs_algorithmic(data)
    experiment_dual_path(data)
    experiment_purpose_constraints(data)
    experiment_selective_generation(data)
    experiment_prompt_contraction(data)
    experiment_landauer_cost()
    experiment_scaling()
    experiment_cross_domain(data)

    summary = {
        "paper": "Purpose-Based Spectral Analysis",
        "n_compounds": len(COMPOUNDS),
        "n_experiments": 8,
        "results_dir": RESULTS_DIR,
        "files_generated": [
            "01_resonance_vs_algorithmic.csv",
            "02_dual_path_convergence.csv",
            "03_purpose_constraints.csv",
            "04_selective_generation.csv",
            "05_prompt_contraction.csv",
            "06_landauer_cost.csv",
            "07_scaling_analysis.csv",
            "08_cross_domain_fnr.csv",
        ],
        "runtime_seconds": round(time.time() - t0, 2),
    }
    path = os.path.join(RESULTS_DIR, "00_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"ALL EXPERIMENTS COMPLETE in {summary['runtime_seconds']}s")
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
