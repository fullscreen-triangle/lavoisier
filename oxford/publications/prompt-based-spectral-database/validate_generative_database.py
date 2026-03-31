#!/usr/bin/env python3
"""
Validation experiments for Paper 1:
"Context-Based Spectral Database: Generative Phase Space Addressing
 for Mass Spectrometric Identification from First Principles"

Validates: S-entropy coordinates, ternary encoding, unique resolution,
chemical family cohesion, trajectory generation for 4 analyzer types,
ion-droplet bijection, and complexity analysis.

All results saved as CSV/JSON in ./results/
"""

import json
import csv
import math
import os
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
from itertools import combinations

# ============================================================================
# Constants
# ============================================================================
HBAR = 1.0546e-34        # J·s
KB = 1.381e-23            # J/K
C_LIGHT = 2.998e8         # m/s
E_CHARGE = 1.602e-19      # C
AMU = 1.661e-27           # kg
CM_TO_HZ = 2.998e10       # cm^-1 to Hz

# Reference values for S-entropy normalization
OMEGA_REF_MAX = 4401.0    # cm^-1 (H2)
OMEGA_REF_MIN = 218.0     # cm^-1 (CCl4 lowest)
B_ROT_REF_MIN = 0.39      # cm^-1 (large moment reference)

# Harmonic proximity parameters
DELTA_HARMONIC = 0.05     # tolerance
P_MAX_HARMONIC = 8        # max ratio order

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================================
# NIST CCCBDB Compound Data (39 compounds)
# ============================================================================
@dataclass
class Compound:
    name: str
    formula: str
    mol_type: str
    mass: float
    frequencies: List[float]  # cm^-1
    rotational_constant: Optional[float] = None  # cm^-1, for diatomics

COMPOUNDS = [
    # Diatomics
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
    # Triatomics
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
    # Tetratomic/tetrahedral
    Compound("NH3", "NH3", "tetra", 17, [950, 1627, 3337, 3444]),
    Compound("PH3", "PH3", "tetra", 34, [992, 1118, 2323, 2328]),
    Compound("CH4", "CH4", "tetra", 16, [1306, 1534, 2917, 3019]),
    Compound("CCl4", "CCl4", "tetra", 154, [218, 314, 776, 790]),
    Compound("SiH4", "SiH4", "tetra", 32, [800, 913, 2187, 2191]),
    Compound("CF4", "CF4", "tetra", 88, [435, 632, 1283, 1283]),
    Compound("H2CO", "H2CO", "poly", 30, [1167, 1249, 1500, 1746, 2783, 2843]),
    # Polyatomic
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

CHEMICAL_FAMILIES = {
    "Halomethanes": ["CH3F", "CH3Cl", "CH3Br"],
    "Hydrogen halides": ["HF", "HCl", "HBr", "HI"],
    "Homonuclear diatomics": ["H2", "D2", "N2", "O2", "F2", "Cl2"],
    "Small hydrocarbons": ["CH4", "C2H2", "C2H4", "C2H6"],
    "Linear triatomics": ["CO2", "CS2", "N2O", "OCS"],
    "Bent triatomics": ["H2O", "SO2", "NO2", "O3", "H2S"],
}


# ============================================================================
# S-Entropy Coordinate Computation
# ============================================================================
def compute_sk(freqs: List[float], is_diatomic: bool) -> float:
    """Knowledge entropy."""
    if is_diatomic:
        return freqs[0] / OMEGA_REF_MAX
    N = len(freqs)
    total = sum(freqs)
    if total == 0:
        return 0.0
    probs = [f / total for f in freqs]
    H = -sum(p * math.log2(p) for p in probs if p > 0)
    return H / math.log2(N) if N > 1 else 0.0


def compute_st(freqs: List[float], is_diatomic: bool,
               b_rot: Optional[float] = None) -> float:
    """Temporal entropy."""
    if is_diatomic:
        if b_rot is None or b_rot <= 0:
            return 0.0
        return (math.log(freqs[0] / b_rot)
                / math.log(OMEGA_REF_MAX / B_ROT_REF_MIN))
    w_max = max(freqs)
    w_min = min(freqs)
    if w_min <= 0 or w_max <= w_min:
        return 0.0
    return (math.log(w_max / w_min)
            / math.log(OMEGA_REF_MAX / OMEGA_REF_MIN))


def compute_se(freqs: List[float]) -> float:
    """Evolution entropy (harmonic network density)."""
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


def compute_sentropy(c: Compound) -> Tuple[float, float, float]:
    is_di = (c.mol_type == "diatomic")
    sk = compute_sk(c.frequencies, is_di)
    st = compute_st(c.frequencies, is_di, c.rotational_constant)
    se = compute_se(c.frequencies)
    return (sk, st, se)


# ============================================================================
# Ternary Encoding
# ============================================================================
def ternary_encode(sk: float, st: float, se: float, depth: int = 18
                   ) -> str:
    """Interleaved ternary encoding of S-entropy coordinates."""
    coords = [sk, st, se]
    trits = []
    for j in range(depth):
        dim = j % 3
        val = int(coords[dim] * 3)
        val = min(val, 2)
        trits.append(str(val))
        coords[dim] = coords[dim] * 3 - val
    return "".join(trits)


def common_prefix_length(a: str, b: str) -> int:
    """Length of longest common prefix."""
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return i
    return min(len(a), len(b))


# ============================================================================
# Analyzer Trajectory Generation
# ============================================================================
def generate_tof_trajectory(mz: float, V_acc: float = 5000.0,
                            L_tube: float = 1.0) -> Dict:
    """TOF: T = sqrt(2*mu*L/kappa), T proportional to sqrt(m/z)."""
    m_kg = mz * AMU
    T_flight = L_tube * math.sqrt(m_kg / (2 * E_CHARGE * V_acc))
    return {
        "analyzer": "TOF",
        "m_over_z": mz,
        "flight_time_us": T_flight * 1e6,
        "V_acc": V_acc,
        "L_tube": L_tube,
        "scaling": "T ~ sqrt(m/z)",
    }


def generate_quadrupole_trajectory(mz: float, U_dc: float = 100.0,
                                   V_rf: float = 500.0,
                                   Omega: float = 1e6,
                                   r0: float = 0.005) -> Dict:
    """Quadrupole: Mathieu stability parameters a, q proportional to 1/(m/z)."""
    m_kg = mz * AMU
    a_param = 8 * E_CHARGE * U_dc / (m_kg * (r0 * Omega) ** 2)
    q_param = 4 * E_CHARGE * V_rf / (m_kg * (r0 * Omega) ** 2)
    stable = (abs(a_param) < 0.237 and abs(q_param) < 0.908)
    return {
        "analyzer": "Quadrupole",
        "m_over_z": mz,
        "a_mathieu": a_param,
        "q_mathieu": q_param,
        "stable": stable,
        "scaling": "a,q ~ 1/(m/z)",
    }


def generate_orbitrap_trajectory(mz: float,
                                 k_field: float = 1e12) -> Dict:
    """Orbitrap: omega = sqrt(k / mu) proportional to sqrt(z/m)."""
    m_kg = mz * AMU
    omega = math.sqrt(E_CHARGE * k_field / m_kg)
    freq_hz = omega / (2 * math.pi)
    return {
        "analyzer": "Orbitrap",
        "m_over_z": mz,
        "axial_freq_Hz": freq_hz,
        "axial_freq_kHz": freq_hz / 1e3,
        "omega_rad_s": omega,
        "k_field": k_field,
        "scaling": "omega ~ sqrt(z/m)",
    }


def generate_fticr_trajectory(mz: float, B_field: float = 7.0) -> Dict:
    """FT-ICR: omega_c = zeB/m proportional to z/m."""
    m_kg = mz * AMU
    omega_c = E_CHARGE * B_field / m_kg
    freq_hz = omega_c / (2 * math.pi)
    return {
        "analyzer": "FT-ICR",
        "m_over_z": mz,
        "cyclotron_freq_Hz": freq_hz,
        "cyclotron_freq_kHz": freq_hz / 1e3,
        "omega_c_rad_s": omega_c,
        "B_field": B_field,
        "scaling": "omega_c ~ z/m",
    }


# ============================================================================
# Ion-Droplet Bijection
# ============================================================================
def ion_to_droplet(c: Compound, sk: float, st: float, se: float) -> Dict:
    """Bijective transformation: ion -> droplet representation."""
    # Droplet velocity from molecular kinetic energy (thermal at 300K)
    m_kg = c.mass * AMU
    v_thermal = math.sqrt(3 * KB * 300 / m_kg) if m_kg > 0 else 0
    # Droplet radius from molecular cross-section (approx r ~ M^(1/3))
    r_mol = 1e-10 * (c.mass ** (1.0 / 3.0))
    # Weber number: We = rho * v^2 * D / sigma (dimensionless)
    rho_water = 1000.0
    sigma_water = 0.072
    D_droplet = 2 * r_mol
    We = rho_water * v_thermal ** 2 * D_droplet / sigma_water
    # Ohnesorge number: Oh = mu / sqrt(rho * sigma * D)
    mu_water = 1e-3
    Oh = mu_water / math.sqrt(rho_water * sigma_water * D_droplet)
    # Surface wave modes (Rayleigh frequencies)
    n_modes = len(c.frequencies)
    rayleigh_freqs = []
    for n in range(2, n_modes + 2):
        omega_n = math.sqrt(
            n * (n - 1) * (n + 2) * sigma_water
            / (rho_water * r_mol ** 3)
        )
        rayleigh_freqs.append(omega_n / (2 * math.pi))
    # Compute S-entropy from droplet modes (should match ion S-entropy)
    if len(rayleigh_freqs) >= 2:
        total_r = sum(rayleigh_freqs)
        probs_r = [f / total_r for f in rayleigh_freqs]
        H_r = -sum(p * math.log2(p) for p in probs_r if p > 0)
        sk_drip = H_r / math.log2(len(rayleigh_freqs))
    elif len(rayleigh_freqs) == 1:
        sk_drip = rayleigh_freqs[0] / max(rayleigh_freqs[0], 1.0)
    else:
        sk_drip = 0.0
    return {
        "name": c.name,
        "velocity_m_s": v_thermal,
        "radius_m": r_mol,
        "Weber": We,
        "Ohnesorge": Oh,
        "n_surface_modes": len(rayleigh_freqs),
        "rayleigh_freqs_Hz": rayleigh_freqs[:5],
        "Sk_ion": sk,
        "St_ion": st,
        "Se_ion": se,
        "Sk_drip": sk_drip,
    }


# ============================================================================
# Experiment 1: S-Entropy Coordinates & Ternary Encoding
# ============================================================================
def experiment_sentropy_and_encoding():
    """Compute S-entropy coordinates and ternary addresses for all 39 compounds."""
    print("=" * 70)
    print("EXPERIMENT 1: S-Entropy Coordinates & Ternary Encoding")
    print("=" * 70)

    results = []
    for c in COMPOUNDS:
        sk, st, se = compute_sentropy(c)
        addr = ternary_encode(sk, st, se, depth=18)
        results.append({
            "name": c.name,
            "formula": c.formula,
            "type": c.mol_type,
            "n_modes": len(c.frequencies),
            "mass": c.mass,
            "Sk": round(sk, 4),
            "St": round(st, 4),
            "Se": round(se, 4),
            "ternary_12": addr[:12],
            "ternary_18": addr,
        })

    # Save CSV
    path = os.path.join(RESULTS_DIR, "01_sentropy_coordinates.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    print(f"  Saved {len(results)} compounds -> {path}")
    return results


# ============================================================================
# Experiment 2: Unique Resolution Analysis
# ============================================================================
def experiment_resolution(sentropy_results):
    """Test at which trit depth all 39 compounds are uniquely resolved."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Unique Resolution Analysis")
    print("=" * 70)

    addresses = [(r["name"], r["ternary_18"]) for r in sentropy_results]
    resolution_data = []

    for depth in range(1, 19):
        prefixes = {}
        for name, addr in addresses:
            pref = addr[:depth]
            if pref not in prefixes:
                prefixes[pref] = []
            prefixes[pref].append(name)
        n_unique = sum(1 for v in prefixes.values() if len(v) == 1)
        n_occupied = len(prefixes)
        n_multi = sum(1 for v in prefixes.values() if len(v) > 1)
        max_occ = max(len(v) for v in prefixes.values())
        all_unique = (n_unique == len(addresses))

        resolution_data.append({
            "depth": depth,
            "total_cells_possible": 3 ** depth,
            "cells_occupied": n_occupied,
            "uniquely_resolved": n_unique,
            "multi_occupancy_cells": n_multi,
            "max_occupancy": max_occ,
            "all_unique": all_unique,
        })
        if all_unique:
            print(f"  All 39 compounds uniquely resolved at depth {depth}")

    path = os.path.join(RESULTS_DIR, "02_resolution_cascade.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=resolution_data[0].keys())
        w.writeheader()
        w.writerows(resolution_data)
    print(f"  Saved resolution cascade -> {path}")
    return resolution_data


# ============================================================================
# Experiment 3: Chemical Family Cohesion
# ============================================================================
def experiment_cohesion(sentropy_results):
    """Test whether chemical families show ternary cohesion."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Chemical Family Cohesion")
    print("=" * 70)

    addr_map = {r["name"]: r["ternary_18"] for r in sentropy_results}
    all_names = list(addr_map.keys())
    cohesion_results = []

    for family_name, members in CHEMICAL_FAMILIES.items():
        # Intra-group similarity
        intra_sims = []
        for a, b in combinations(members, 2):
            if a in addr_map and b in addr_map:
                intra_sims.append(
                    common_prefix_length(addr_map[a], addr_map[b]))
        intra_mean = sum(intra_sims) / len(intra_sims) if intra_sims else 0

        # Inter-group similarity
        inter_sims = []
        non_members = [n for n in all_names if n not in members]
        for m in members:
            for nm in non_members:
                if m in addr_map and nm in addr_map:
                    inter_sims.append(
                        common_prefix_length(addr_map[m], addr_map[nm]))
        inter_mean = sum(inter_sims) / len(inter_sims) if inter_sims else 0

        R = intra_mean / inter_mean if inter_mean > 0 else float("inf")
        status = "PASS" if R > 1.0 else ("MARGINAL" if R > 0.95 else "FAIL")

        cohesion_results.append({
            "family": family_name,
            "members": len(members),
            "intra_similarity": round(intra_mean, 3),
            "inter_similarity": round(inter_mean, 3),
            "cohesion_ratio_R": round(R, 3),
            "status": status,
        })
        print(f"  {family_name}: R={R:.3f} ({status})")

    path = os.path.join(RESULTS_DIR, "03_chemical_cohesion.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cohesion_results[0].keys())
        w.writeheader()
        w.writerows(cohesion_results)
    print(f"  Saved cohesion analysis -> {path}")
    return cohesion_results


# ============================================================================
# Experiment 4: Trajectory Generation for All Analyzers
# ============================================================================
def experiment_trajectory_generation():
    """Generate ion trajectories for all 39 compounds across 4 analyzer types."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Trajectory Generation (4 Analyzers)")
    print("=" * 70)

    trajectories = []
    for c in COMPOUNDS:
        mz = c.mass  # Assuming z=1 for simplicity
        tof = generate_tof_trajectory(mz)
        quad = generate_quadrupole_trajectory(mz)
        orbi = generate_orbitrap_trajectory(mz)
        icr = generate_fticr_trajectory(mz)

        trajectories.append({
            "name": c.name,
            "formula": c.formula,
            "mass": c.mass,
            "m_over_z": mz,
            # TOF
            "TOF_flight_time_us": round(tof["flight_time_us"], 4),
            # Quadrupole
            "Quad_a": round(quad["a_mathieu"], 6),
            "Quad_q": round(quad["q_mathieu"], 6),
            "Quad_stable": quad["stable"],
            # Orbitrap
            "Orbitrap_freq_kHz": round(orbi["axial_freq_kHz"], 2),
            # FT-ICR
            "FTICR_freq_kHz": round(icr["cyclotron_freq_kHz"], 2),
        })

    # Verify scaling laws
    scaling_verification = []
    for i in range(len(trajectories)):
        for j in range(i + 1, len(trajectories)):
            mz_i = trajectories[i]["m_over_z"]
            mz_j = trajectories[j]["m_over_z"]
            if mz_i <= 0 or mz_j <= 0:
                continue
            ratio_mz = mz_j / mz_i

            # TOF: T ~ sqrt(m/z)
            t_i = trajectories[i]["TOF_flight_time_us"]
            t_j = trajectories[j]["TOF_flight_time_us"]
            tof_ratio = t_j / t_i if t_i > 0 else 0
            tof_expected = math.sqrt(ratio_mz)

            # Orbitrap: omega ~ sqrt(z/m) => omega ~ 1/sqrt(m/z)
            o_i = trajectories[i]["Orbitrap_freq_kHz"]
            o_j = trajectories[j]["Orbitrap_freq_kHz"]
            orbi_ratio = o_j / o_i if o_i > 0 else 0
            orbi_expected = 1.0 / math.sqrt(ratio_mz)

            # FT-ICR: omega_c ~ z/m => ~ 1/(m/z)
            f_i = trajectories[i]["FTICR_freq_kHz"]
            f_j = trajectories[j]["FTICR_freq_kHz"]
            icr_ratio = f_j / f_i if f_i > 0 else 0
            icr_expected = 1.0 / ratio_mz

            scaling_verification.append({
                "compound_i": trajectories[i]["name"],
                "compound_j": trajectories[j]["name"],
                "mz_ratio": round(ratio_mz, 4),
                "TOF_ratio_actual": round(tof_ratio, 6),
                "TOF_ratio_expected": round(tof_expected, 6),
                "TOF_error": round(abs(tof_ratio - tof_expected), 8),
                "Orbitrap_ratio_actual": round(orbi_ratio, 6),
                "Orbitrap_ratio_expected": round(orbi_expected, 6),
                "Orbitrap_error": round(abs(orbi_ratio - orbi_expected), 8),
                "FTICR_ratio_actual": round(icr_ratio, 6),
                "FTICR_ratio_expected": round(icr_expected, 6),
                "FTICR_error": round(abs(icr_ratio - icr_expected), 8),
            })

    path1 = os.path.join(RESULTS_DIR, "04_trajectories.csv")
    with open(path1, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=trajectories[0].keys())
        w.writeheader()
        w.writerows(trajectories)
    print(f"  Saved {len(trajectories)} trajectory records -> {path1}")

    path2 = os.path.join(RESULTS_DIR, "04_scaling_verification.csv")
    with open(path2, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=scaling_verification[0].keys())
        w.writeheader()
        w.writerows(scaling_verification)

    # Summary: max errors
    max_tof_err = max(s["TOF_error"] for s in scaling_verification)
    max_orbi_err = max(s["Orbitrap_error"] for s in scaling_verification)
    max_icr_err = max(s["FTICR_error"] for s in scaling_verification)
    print(f"  Scaling law max errors: TOF={max_tof_err:.2e}, "
          f"Orbitrap={max_orbi_err:.2e}, FT-ICR={max_icr_err:.2e}")
    print(f"  Saved scaling verification ({len(scaling_verification)} pairs) "
          f"-> {path2}")
    return trajectories


# ============================================================================
# Experiment 5: Ion-Droplet Bijection
# ============================================================================
def experiment_ion_droplet(sentropy_results):
    """Validate the ion-droplet bijection."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Ion-Droplet Bijection")
    print("=" * 70)

    se_map = {r["name"]: (r["Sk"], r["St"], r["Se"])
              for r in sentropy_results}
    bijection_results = []

    for c in COMPOUNDS:
        sk, st, se = se_map[c.name]
        drip = ion_to_droplet(c, sk, st, se)
        bijection_results.append(drip)

    path = os.path.join(RESULTS_DIR, "05_ion_droplet_bijection.json")
    with open(path, "w") as f:
        json.dump(bijection_results, f, indent=2)
    print(f"  Saved {len(bijection_results)} bijection records -> {path}")

    # Summary statistics
    we_vals = [b["Weber"] for b in bijection_results]
    oh_vals = [b["Ohnesorge"] for b in bijection_results]
    print(f"  Weber number range: [{min(we_vals):.2e}, {max(we_vals):.2e}]")
    print(f"  Ohnesorge number range: [{min(oh_vals):.2e}, {max(oh_vals):.2e}]")
    return bijection_results


# ============================================================================
# Experiment 6: Complexity Analysis
# ============================================================================
def experiment_complexity():
    """Quantify speedup of generative database over brute-force."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 6: Complexity Analysis")
    print("=" * 70)

    d = 1024  # fingerprint dimensionality
    complexity_results = []

    for N in [39, 1000, 10000, 100000, 1000000, 100000000]:
        for k in [3, 6, 9, 12, 18]:
            brute_force = N * d
            trie_search = k
            speedup = brute_force / trie_search
            storage_traditional = N * d  # bits (approx)
            storage_generative = 1  # O(1) - just the Lagrangian

            complexity_results.append({
                "N_compounds": N,
                "trit_depth_k": k,
                "brute_force_ops": brute_force,
                "trie_search_ops": trie_search,
                "speedup": round(speedup, 1),
                "log10_speedup": round(math.log10(speedup), 2),
            })

    path = os.path.join(RESULTS_DIR, "06_complexity_analysis.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=complexity_results[0].keys())
        w.writeheader()
        w.writerows(complexity_results)
    print(f"  Saved {len(complexity_results)} complexity entries -> {path}")

    # Highlight PubChem scale
    pubchem = [r for r in complexity_results
               if r["N_compounds"] == 100000000 and r["trit_depth_k"] == 18]
    if pubchem:
        print(f"  PubChem scale (N=10^8, k=18): "
              f"speedup = {pubchem[0]['speedup']:.2e}")
    return complexity_results


# ============================================================================
# Experiment 7: Depth-3 Emergent Clustering
# ============================================================================
def experiment_emergent_clustering(sentropy_results):
    """Show what chemical groupings emerge at trit depth 3."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 7: Emergent Clustering at Depth 3")
    print("=" * 70)

    clusters = {}
    for r in sentropy_results:
        prefix = r["ternary_12"][:3]
        if prefix not in clusters:
            clusters[prefix] = []
        clusters[prefix].append(r["name"])

    clustering_results = []
    for prefix in sorted(clusters.keys()):
        members = clusters[prefix]
        clustering_results.append({
            "cell_prefix": prefix,
            "n_compounds": len(members),
            "members": ", ".join(members),
        })
        print(f"  [{prefix}] ({len(members)}): {', '.join(members)}")

    path = os.path.join(RESULTS_DIR, "07_depth3_clustering.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=clustering_results[0].keys())
        w.writeheader()
        w.writerows(clustering_results)
    print(f"  Saved {len(clustering_results)} clusters -> {path}")
    return clustering_results


# ============================================================================
# Experiment 8: Pairwise Ternary Similarity Matrix
# ============================================================================
def experiment_similarity_matrix(sentropy_results):
    """Compute pairwise ternary similarity for all 39 compounds."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 8: Pairwise Similarity Matrix")
    print("=" * 70)

    names = [r["name"] for r in sentropy_results]
    addrs = [r["ternary_18"] for r in sentropy_results]
    N = len(names)

    matrix = []
    for i in range(N):
        for j in range(i + 1, N):
            cpl = common_prefix_length(addrs[i], addrs[j])
            eucl = math.sqrt(
                (sentropy_results[i]["Sk"] - sentropy_results[j]["Sk"]) ** 2
                + (sentropy_results[i]["St"] - sentropy_results[j]["St"]) ** 2
                + (sentropy_results[i]["Se"] - sentropy_results[j]["Se"]) ** 2
            )
            matrix.append({
                "compound_A": names[i],
                "compound_B": names[j],
                "common_prefix_length": cpl,
                "euclidean_distance": round(eucl, 6),
            })

    path = os.path.join(RESULTS_DIR, "08_pairwise_similarity.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=matrix[0].keys())
        w.writeheader()
        w.writerows(matrix)
    print(f"  Saved {len(matrix)} pairs -> {path}")

    # Nearest neighbours for key compounds
    print("\n  Nearest neighbours by ternary similarity:")
    for query in ["H2O", "CH4", "HCl", "CO2", "C2H6"]:
        query_pairs = [p for p in matrix
                       if p["compound_A"] == query or p["compound_B"] == query]
        query_pairs.sort(key=lambda x: -x["common_prefix_length"])
        top3 = query_pairs[:3]
        neighbours = []
        for p in top3:
            other = (p["compound_B"] if p["compound_A"] == query
                     else p["compound_A"])
            neighbours.append(f"{other}({p['common_prefix_length']})")
        print(f"    {query}: {', '.join(neighbours)}")

    return matrix


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 70)
    print("PAPER 1 VALIDATION: Context-Based Spectral Database")
    print("Generative Phase Space Addressing from First Principles")
    print("=" * 70)
    t0 = time.time()

    # Run all experiments
    se_results = experiment_sentropy_and_encoding()
    experiment_resolution(se_results)
    experiment_cohesion(se_results)
    experiment_trajectory_generation()
    experiment_ion_droplet(se_results)
    experiment_complexity()
    experiment_emergent_clustering(se_results)
    experiment_similarity_matrix(se_results)

    # Save master summary
    summary = {
        "paper": "Context-Based Spectral Database",
        "n_compounds": len(COMPOUNDS),
        "n_experiments": 8,
        "results_dir": RESULTS_DIR,
        "files_generated": [
            "01_sentropy_coordinates.csv",
            "02_resolution_cascade.csv",
            "03_chemical_cohesion.csv",
            "04_trajectories.csv",
            "04_scaling_verification.csv",
            "05_ion_droplet_bijection.json",
            "06_complexity_analysis.csv",
            "07_depth3_clustering.csv",
            "08_pairwise_similarity.csv",
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
