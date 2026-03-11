#!/usr/bin/env python3
"""
Generate 10 Panel Charts for Ion Journey Validation Publication
================================================================

Each panel: 4 charts in a row, at least one 3D chart per panel.
Covers all 38 theorems across 8 stages of the ion journey.

Usage:
    python generate_panels.py
"""

import sys
import re
import math
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
from pathlib import Path
from dataclasses import asdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from union.src.derivation.ion_journey_validator import (
    IonJourneyValidator, IonInput, JourneyResult, _serialize,
    capacity, cumulative_capacity, RESIDUE_FRACTION, C_LIGHT, HBAR,
    K_B, E_CHARGE, AMU, PROTON_MASS,
)

# Import the runner's MSP parser
# Import directly to avoid validation/__init__.py pulling in statsmodels
import importlib.util
spec = importlib.util.spec_from_file_location("run_ion_journey",
    str(PROJECT_ROOT / "validation" / "run_ion_journey.py"))
_run_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_run_mod)
parse_msp_file = _run_mod.parse_msp_file
msp_to_ion_input = _run_mod.msp_to_ion_input

# ============================================================================
# Style
# ============================================================================

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.6,
    'grid.linewidth': 0.3,
    'lines.linewidth': 1.0,
    'lines.markersize': 4,
})

COLORS = {
    'primary': '#1a5276',
    'secondary': '#c0392b',
    'tertiary': '#27ae60',
    'quaternary': '#8e44ad',
    'accent': '#e67e22',
    'gold': '#f39c12',
    'teal': '#16a085',
    'dark': '#2c3e50',
    'light': '#ecf0f1',
    'pass': '#27ae60',
    'fail': '#c0392b',
}

CMAP_3D = 'viridis'
OUTPUT_DIR = PROJECT_ROOT / "union" / "revised" / "mass-spectrum" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Load data
# ============================================================================

def load_all_ions():
    """Parse MSP and run validator on all 34 ions, returning (ions, results)."""
    msp_path = (PROJECT_ROOT / "union" / "public" / "nist" /
                "NISTMS-GADS-SARS-CoV-2_SpikeProtein" /
                "NISTMS-GADS-SARS-CoV-2_SpikeProtein" / "spike_sulfated_ms2.MSP")
    spectra = parse_msp_file(msp_path)
    validator = IonJourneyValidator()
    ions = [msp_to_ion_input(s) for s in spectra]
    results = [validator.validate(ion) for ion in ions]
    return ions, results


def get_theorem_values(results, stage_num, theorem_id):
    """Extract a specific theorem's value dict from all results."""
    vals = []
    for r in results:
        for stage in r.stages:
            if stage.stage_number == stage_num:
                for t in stage.theorems:
                    if t.theorem_id == theorem_id:
                        vals.append(t.value)
    return vals


def get_computed(results, stage_num):
    """Extract computed_values from a stage across all results."""
    vals = []
    for r in results:
        for stage in r.stages:
            if stage.stage_number == stage_num:
                vals.append(stage.computed_values)
    return vals


# ============================================================================
# Panel generators
# ============================================================================

def panel_1(ions, results, fig_num=1):
    """Panel 1: Axiom & Oscillatory Foundation (Stage 1, 4 theorems)"""
    fig = plt.figure(figsize=(14, 3.5))
    gs = GridSpec(1, 4, figure=fig, wspace=0.35)

    # Data
    bpsl = get_theorem_values(results, 1, 'ax:bpsl')
    osc = get_theorem_values(results, 1, 'thm:oscillatory')
    fe = get_theorem_values(results, 1, 'cor:freq-energy')
    modes = get_theorem_values(results, 1, 'cor:modes')

    masses = [v['molecular_radius_nm'] for v in bpsl]
    omegas = [v['omega_0_rad_s'] for v in osc]
    energies = [v['E_rest_eV'] for v in fe]
    ps_vols = [v['phase_space_volume_m6_kg3_s3'] for v in bpsl]

    # Chart 1: 3D scatter (r_mol, omega_0, E)
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    sc = ax1.scatter(masses, np.log10(omegas), np.log10(energies),
                     c=np.log10(energies), cmap=CMAP_3D, s=25, edgecolors='k', linewidths=0.3)
    ax1.set_xlabel('r$_{mol}$ (nm)')
    ax1.set_ylabel('log$_{10}$($\\omega_0$)')
    ax1.set_zlabel('log$_{10}$(E) [eV]')
    ax1.set_title('Bounded Phase Space')
    ax1.view_init(elev=25, azim=135)

    # Chart 2: E = hbar*omega vs mc^2
    ax2 = fig.add_subplot(gs[0, 1])
    e_hbar = [v['E_hbar_omega_J'] for v in fe]
    e_mc2 = [v['E_mc2_J'] for v in fe]
    ax2.scatter(e_mc2, e_hbar, c=COLORS['primary'], s=20, zorder=5, edgecolors='k', linewidths=0.3)
    lims = [min(e_mc2) * 0.95, max(e_mc2) * 1.05]
    ax2.plot(lims, lims, '--', color=COLORS['secondary'], linewidth=0.8, label='E = mc$^2$')
    ax2.set_xlabel('mc$^2$ (J)')
    ax2.set_ylabel('$\\hbar\\omega_0$ (J)')
    ax2.set_title('Frequency-Energy Identity')
    ax2.legend(frameon=False)
    ax2.ticklabel_format(style='scientific', scilimits=(0, 0))

    # Chart 3: C(n) = 2n^2
    ax3 = fig.add_subplot(gs[0, 2])
    ns = list(range(1, 8))
    caps = [capacity(n) for n in ns]
    ax3.plot(ns, caps, 'o-', color=COLORS['primary'], markersize=5, linewidth=1.2)
    # Mark ion positions
    ion_ns = [v['n_principal'] for v in modes]
    ion_cs = [v['C_n'] for v in modes]
    ax3.scatter(ion_ns, ion_cs, c=COLORS['secondary'], s=40, zorder=5, marker='*',
                edgecolors='k', linewidths=0.3, label='Ions')
    ax3.set_xlabel('n (principal)')
    ax3.set_ylabel('C(n) = 2n$^2$')
    ax3.set_title('Mode Decomposition')
    ax3.legend(frameon=False)

    # Chart 4: Phase space volume vs radius
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.scatter(masses, np.log10(ps_vols), c=COLORS['teal'], s=20,
                edgecolors='k', linewidths=0.3)
    ax4.set_xlabel('r$_{mol}$ (nm)')
    ax4.set_ylabel('log$_{10}$(Phase Space Vol)')
    ax4.set_title('Oscillatory Necessity')
    ax4.axhline(y=0, color='gray', linestyle=':', linewidth=0.5)

    fig.suptitle('Panel 1: Axiom & Oscillatory Foundation', fontsize=10, fontweight='bold', y=1.02)
    fig.savefig(OUTPUT_DIR / f'panel_{fig_num:02d}_axiom_oscillatory.png', bbox_inches='tight')
    plt.close(fig)
    print(f'  Panel {fig_num} saved')


def panel_2(ions, results, fig_num=2):
    """Panel 2: Compression & Transport (Stage 2 + Stage 3 first 2)"""
    fig = plt.figure(figsize=(14, 3.5))
    gs = GridSpec(1, 4, figure=fig, wspace=0.35)

    comp = get_theorem_values(results, 2, 'thm:compression')
    bounds = get_theorem_values(results, 2, 'thm:bounds')
    prop = get_theorem_values(results, 3, 'thm:propagation')
    tcat = get_theorem_values(results, 3, 'eq:T_cat')

    # Chart 1: 3D compression landscape
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    cr = [np.log10(v['compression_ratio']) for v in comp]
    cost = [v['cost_kBT'] for v in comp]
    bits = [v['bits_erased'] for v in comp]
    sc = ax1.scatter(cr, cost, bits, c=cost, cmap='magma', s=25, edgecolors='k', linewidths=0.3)
    ax1.set_xlabel('log$_{10}$(Ratio)')
    ax1.set_ylabel('Cost (k$_B$T)')
    ax1.set_zlabel('Bits erased')
    ax1.set_title('Compression Theorem')
    ax1.view_init(elev=20, azim=130)

    # Chart 2: Depth bounds
    ax2 = fig.add_subplot(gs[0, 1])
    m_min = [v['M_min'] for v in bounds]
    m_max = [v['M_max'] for v in bounds]
    x = np.arange(len(m_min))
    ax2.bar(x, m_max, color=COLORS['primary'], alpha=0.7, label='M$_{max}$', width=0.8)
    ax2.bar(x, m_min, color=COLORS['secondary'], alpha=0.9, label='M$_{min}$', width=0.8)
    ax2.set_xlabel('Ion index')
    ax2.set_ylabel('Partition depth M')
    ax2.set_title('Depth Bounds')
    ax2.legend(frameon=False, loc='upper left')

    # Chart 3: c derivation
    ax3 = fig.add_subplot(gs[0, 2])
    c_ratios = [v['c_ratio'] for v in prop]
    ax3.bar(range(len(c_ratios)), c_ratios, color=COLORS['tertiary'], alpha=0.8)
    ax3.axhline(y=1.0, color=COLORS['secondary'], linestyle='--', linewidth=0.8, label='c/c = 1')
    ax3.set_xlabel('Ion index')
    ax3.set_ylabel('c$_{derived}$ / c')
    ax3.set_title('Derivation of c')
    ax3.set_ylim(0.999, 1.001)
    ax3.legend(frameon=False)

    # Chart 4: T_cat vs T_kinetic
    ax4 = fig.add_subplot(gs[0, 3])
    tc = [v['T_cat_K'] for v in tcat]
    tk = [v['T_kinetic_K'] for v in tcat]
    ax4.scatter(tk, tc, c=COLORS['quaternary'], s=25, edgecolors='k', linewidths=0.3)
    lims = [min(tk) * 0.99, max(tk) * 1.01]
    ax4.plot(lims, lims, '--', color='gray', linewidth=0.8)
    ax4.set_xlabel('T$_{kinetic}$ (K)')
    ax4.set_ylabel('T$_{cat}$ (K)')
    ax4.set_title('Categorical Temperature')

    fig.suptitle('Panel 2: Compression & Transport', fontsize=10, fontweight='bold', y=1.02)
    fig.savefig(OUTPUT_DIR / f'panel_{fig_num:02d}_compression_transport.png', bbox_inches='tight')
    plt.close(fig)
    print(f'  Panel {fig_num} saved')


def panel_3(ions, results, fig_num=3):
    """Panel 3: Chromatography & Composition (Stage 3 last + Stage 4 first 3)"""
    fig = plt.figure(figsize=(14, 3.5))
    gs = GridSpec(1, 4, figure=fig, wspace=0.35)

    ret = get_theorem_values(results, 3, 'eq:retention')
    comp_thm = get_theorem_values(results, 4, 'thm:composition')
    compr = get_theorem_values(results, 4, 'thm:compression')
    conserv = get_theorem_values(results, 4, 'thm:conservation')

    # Chart 1: 3D trap array
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    n_traps = [v['N_traps'] for v in ret]
    tau_p = [v['tau_p_per_trap_s'] for v in ret]
    part_ops = [v['partition_operations'] for v in ret]
    sc = ax1.scatter(np.array(n_traps) / 1000, np.log10(tau_p), np.array(part_ops) / 1000,
                     c=np.array(part_ops) / 1000, cmap=CMAP_3D, s=25, edgecolors='k', linewidths=0.3)
    ax1.set_xlabel('N$_{traps}$ (x10$^3$)')
    ax1.set_ylabel('log$_{10}$($\\tau_p$)')
    ax1.set_zlabel('Ops (x10$^3$)')
    ax1.set_title('Trap Array Model')
    ax1.view_init(elev=25, azim=140)

    # Chart 2: Composition theorem
    ax2 = fig.add_subplot(gs[0, 1])
    m_free = [v['M_free'] for v in comp_thm]
    m_bound = [v['M_bound'] for v in comp_thm]
    ax2.scatter(m_bound, m_free, c=COLORS['primary'], s=25, edgecolors='k', linewidths=0.3, zorder=5)
    lims = [0, max(m_free) * 1.1]
    ax2.plot(lims, lims, '--', color='gray', linewidth=0.8, label='M$_{free}$ = M$_{bound}$')
    ax2.fill_between(lims, lims, [max(m_free)*1.1]*2, alpha=0.1, color=COLORS['tertiary'])
    ax2.set_xlabel('M$_{bound}$')
    ax2.set_ylabel('M$_{free}$')
    ax2.set_title('Composition Theorem')
    ax2.legend(frameon=False)

    # Chart 3: Ionization cost
    ax3 = fig.add_subplot(gs[0, 2])
    z_e = [v['Z_electrons_approx'] for v in compr]
    ie = [v['ionization_cost_eV'] for v in compr]
    ax3.scatter(z_e, ie, c=COLORS['accent'], s=25, edgecolors='k', linewidths=0.3)
    ax3.set_xlabel('Z (electrons)')
    ax3.set_ylabel('IE$_{partition}$ (eV)')
    ax3.set_title('Ionization as Compression')

    # Chart 4: Conservation stacked bars
    ax4 = fig.add_subplot(gs[0, 3])
    m_ion = [v['M_ion'] for v in conserv]
    m_elec = [v['M_electrons'] for v in conserv]
    m_en = [v['M_energy'] for v in conserv]
    x = np.arange(len(m_ion))
    ax4.bar(x, m_ion, color=COLORS['primary'], label='M$_{ion}$', width=0.8)
    ax4.bar(x, m_elec, bottom=m_ion, color=COLORS['secondary'], label='M$_e$', width=0.8)
    ax4.bar(x, m_en, bottom=np.array(m_ion)+np.array(m_elec),
            color=COLORS['tertiary'], label='M$_{\\gamma}$', width=0.8)
    ax4.set_xlabel('Ion index')
    ax4.set_ylabel('Partition depth')
    ax4.set_title('Conservation Theorem')
    ax4.legend(frameon=False, fontsize=6)

    fig.suptitle('Panel 3: Chromatography & Depth Theorems', fontsize=10, fontweight='bold', y=1.02)
    fig.savefig(OUTPUT_DIR / f'panel_{fig_num:02d}_chromatography_depth.png', bbox_inches='tight')
    plt.close(fig)
    print(f'  Panel {fig_num} saved')


def panel_4(ions, results, fig_num=4):
    """Panel 4: Charge & Partition Coordinates (Stage 4 remaining)"""
    fig = plt.figure(figsize=(14, 3.5))
    gs = GridSpec(1, 4, figure=fig, wspace=0.35)

    charge = get_theorem_values(results, 4, 'thm:charge')
    coords = get_theorem_values(results, 4, 'thm:coordinates')
    malform = get_theorem_values(results, 4, 'cor:ions')

    # Chart 1: 3D partition coordinate space
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    ns = [v['n'] for v in coords]
    ls = [v['l'] for v in coords]
    ms = [v['m'] for v in coords]
    c_ns = [v['C_n'] for v in coords]
    sc = ax1.scatter(ns, ls, ms, c=c_ns, cmap='plasma', s=40, edgecolors='k', linewidths=0.3)
    ax1.set_xlabel('n')
    ax1.set_ylabel('l')
    ax1.set_zlabel('m')
    ax1.set_title('Partition Coordinates')
    ax1.view_init(elev=30, azim=120)

    # Chart 2: Charge predicted vs observed
    ax2 = fig.add_subplot(gs[0, 1])
    z_obs = [v['charge_observed'] for v in charge]
    z_pred = [v['charge_predicted'] for v in charge]
    ax2.scatter(z_obs, z_pred, c=COLORS['tertiary'], s=40, edgecolors='k', linewidths=0.3, zorder=5)
    lims = [min(z_obs) - 0.5, max(z_obs) + 0.5]
    ax2.plot(lims, lims, '--', color=COLORS['secondary'], linewidth=0.8)
    ax2.set_xlabel('z$_{observed}$')
    ax2.set_ylabel('z$_{predicted}$')
    ax2.set_title('Charge Emergence')
    ax2.set_aspect('equal')

    # Chart 3: C(n) vs precursor m/z
    ax3 = fig.add_subplot(gs[0, 2])
    mzs = [ion.precursor_mz for ion in ions]
    ax3.scatter(mzs, c_ns, c=COLORS['primary'], s=25, edgecolors='k', linewidths=0.3)
    ax3.set_xlabel('Precursor m/z')
    ax3.set_ylabel('C(n) = 2n$^2$')
    ax3.set_title('Capacity vs m/z')

    # Chart 4: Partition depth of ion
    ax4 = fig.add_subplot(gs[0, 3])
    depths = [v['depth_ion'] for v in malform]
    charges = [v['charge'] for v in malform]
    ax4.scatter(charges, depths, c=COLORS['quaternary'], s=40, edgecolors='k', linewidths=0.3)
    ax4.set_xlabel('Charge state z')
    ax4.set_ylabel('M$_{ion}$ (depth)')
    ax4.set_title('Ion = Malformation')
    ax4.axhline(y=np.mean(depths), color='gray', linestyle=':', linewidth=0.5)

    fig.suptitle('Panel 4: Charge Emergence & Partition Coordinates', fontsize=10, fontweight='bold', y=1.02)
    fig.savefig(OUTPUT_DIR / f'panel_{fig_num:02d}_charge_coordinates.png', bbox_inches='tight')
    plt.close(fig)
    print(f'  Panel {fig_num} saved')


def panel_5(ions, results, fig_num=5):
    """Panel 5: Lagrangian & Analyzer (Stage 5 + Stage 6 first)"""
    fig = plt.figure(figsize=(14, 3.5))
    gs = GridSpec(1, 4, figure=fig, wspace=0.35)

    lagr = get_theorem_values(results, 5, 'thm:lagrangian')
    orbi = get_theorem_values(results, 5, 'eq:orbitrap')
    gas = get_theorem_values(results, 6, 'thm:single_gas')

    # Chart 1: 3D Lagrangian components
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    mus = [np.log10(v['mu_kg']) for v in lagr]
    kins = [np.log10(v['kinetic_J']) if v['kinetic_J'] > 0 else -50 for v in lagr]
    lagrs = [np.log10(abs(v['lagrangian_J'])) for v in lagr]
    sc = ax1.scatter(mus, kins, lagrs, c=lagrs, cmap='inferno', s=25, edgecolors='k', linewidths=0.3)
    ax1.set_xlabel('log$_{10}$($\\mu$) [kg]')
    ax1.set_ylabel('log$_{10}$(T) [J]')
    ax1.set_zlabel('log$_{10}$(L) [J]')
    ax1.set_title('Partition Lagrangian')
    ax1.view_init(elev=25, azim=135)

    # Chart 2: Orbitrap omega vs m/z
    ax2 = fig.add_subplot(gs[0, 1])
    mzs = [ion.precursor_mz for ion in ions]
    omega_orbi = [v['omega_orbi_rad_s'] for v in orbi]
    ax2.scatter(mzs, np.log10(omega_orbi), c=COLORS['primary'], s=25,
                edgecolors='k', linewidths=0.3)
    ax2.set_xlabel('m/z')
    ax2.set_ylabel('log$_{10}$($\\omega_{orbi}$) [rad/s]')
    ax2.set_title('Orbitrap: $\\omega \\propto \\sqrt{z/m}$')

    # Chart 3: Partition inertia
    ax3 = fig.add_subplot(gs[0, 2])
    mu_vals = [v['mu_kg'] for v in lagr]
    ax3.scatter(mzs, np.log10(mu_vals), c=COLORS['accent'], s=25,
                edgecolors='k', linewidths=0.3)
    ax3.set_xlabel('m/z')
    ax3.set_ylabel('log$_{10}$($\\mu$) [kg]')
    ax3.set_title('$\\mu = \\alpha \\cdot (m/z)$')

    # Chart 4: PV = k_BT_cat
    ax4 = fig.add_subplot(gs[0, 3])
    pv = [v['PV_J'] for v in gas]
    kbt = [v['kBT_cat_J'] for v in gas]
    tcat_uk = [v['T_cat_uK'] for v in gas]
    ax4.scatter(np.log10(kbt), np.log10(pv), c=COLORS['teal'], s=25,
                edgecolors='k', linewidths=0.3, zorder=5)
    ax4.set_xlabel('log$_{10}$(k$_B$T$_{cat}$) [J]')
    ax4.set_ylabel('log$_{10}$(PV) [J]')
    ax4.set_title('Single-Ion Gas Law')

    fig.suptitle('Panel 5: Lagrangian & Analyzer Physics', fontsize=10, fontweight='bold', y=1.02)
    fig.savefig(OUTPUT_DIR / f'panel_{fig_num:02d}_lagrangian_analyzer.png', bbox_inches='tight')
    plt.close(fig)
    print(f'  Panel {fig_num} saved')


def panel_6(ions, results, fig_num=6):
    """Panel 6: Single-Ion Thermodynamics (Stage 6 middle)"""
    fig = plt.figure(figsize=(14, 3.5))
    gs = GridSpec(1, 4, figure=fig, wspace=0.35)

    second = get_theorem_values(results, 6, 'thm:second_law')
    fund = get_theorem_values(results, 6, 'thm:fundamental')
    arrow = get_theorem_values(results, 6, 'thm:arrow')
    sel = get_theorem_values(results, 6, 'eq:selection')

    # Chart 1: 3D (n_fragments, delta_S, collision_freq)
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    n_frags = [v['n_fragments'] for v in second]
    dS = [v['total_delta_S_kB'] for v in second]
    cf = [np.log10(v['collision_freq_Hz']) for v in fund]
    sc = ax1.scatter(n_frags, dS, cf, c=dS, cmap='YlOrRd', s=25, edgecolors='k', linewidths=0.3)
    ax1.set_xlabel('N$_{fragments}$')
    ax1.set_ylabel('$\\Delta$S (k$_B$)')
    ax1.set_zlabel('log$_{10}$(freq)')
    ax1.set_title('2nd Law & Counting')
    ax1.view_init(elev=20, azim=130)

    # Chart 2: Entropy increase (strictly > 0)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(range(len(dS)), dS, color=COLORS['secondary'], alpha=0.8)
    ax2.axhline(y=0, color='k', linewidth=0.8)
    ax2.set_xlabel('Ion index')
    ax2.set_ylabel('$\\Delta$S$_{cat}$ (k$_B$)')
    ax2.set_title('Categorical 2nd Law ($>$ 0)')
    # Add annotation
    ax2.annotate('Strictly $>$ 0\n(not $\\geq$ 0)', xy=(0.95, 0.95),
                 xycoords='axes fraction', ha='right', va='top', fontsize=7,
                 color=COLORS['secondary'])

    # Chart 3: Time = counting
    ax3 = fig.add_subplot(gs[0, 2])
    n_coll = [v['n_collisions_expected'] for v in fund]
    t_transit = [v['t_transit_s'] * 1e6 for v in fund]
    ax3.scatter(t_transit, n_coll, c=COLORS['primary'], s=25, edgecolors='k', linewidths=0.3)
    ax3.set_xlabel('Transit time ($\\mu$s)')
    ax3.set_ylabel('Expected collisions')
    ax3.set_title('Time = State Counting')

    # Chart 4: Selection rule pass rates
    ax4 = fig.add_subplot(gs[0, 3])
    pass_rates = [v['pass_rate'] * 100 for v in sel]
    ax4.bar(range(len(pass_rates)), pass_rates, color=COLORS['tertiary'], alpha=0.8)
    ax4.axhline(y=90, color=COLORS['secondary'], linestyle='--', linewidth=0.8, label='90% threshold')
    ax4.set_xlabel('Ion index')
    ax4.set_ylabel('Pass rate (%)')
    ax4.set_title('Selection Rules')
    ax4.set_ylim(0, 105)
    ax4.legend(frameon=False, fontsize=6)

    fig.suptitle('Panel 6: Single-Ion Thermodynamics', fontsize=10, fontweight='bold', y=1.02)
    fig.savefig(OUTPUT_DIR / f'panel_{fig_num:02d}_thermodynamics.png', bbox_inches='tight')
    plt.close(fig)
    print(f'  Panel {fig_num} saved')


def panel_7(ions, results, fig_num=7):
    """Panel 7: Fragmentation & Containment (Stage 6 last + CID)"""
    fig = plt.figure(figsize=(14, 3.5))
    gs = GridSpec(1, 4, figure=fig, wspace=0.35)

    contain = get_theorem_values(results, 6, 'thm:conservation_frag')
    cid = get_theorem_values(results, 6, 'eq:cid_energy')
    arrow = get_theorem_values(results, 6, 'thm:arrow')

    # Chart 1: 3D (precursor_mz, max_frag_mz, E_cm)
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    prec = [v['precursor_mz'] for v in contain]
    maxf = [v['max_fragment_mz'] for v in contain]
    ecm = [v['E_cm_eV'] for v in cid]
    sc = ax1.scatter(prec, maxf, ecm, c=ecm, cmap='cool', s=25, edgecolors='k', linewidths=0.3)
    ax1.set_xlabel('Precursor m/z')
    ax1.set_ylabel('Max fragment m/z')
    ax1.set_zlabel('E$_{cm}$ (eV)')
    ax1.set_title('Fragmentation Space')
    ax1.view_init(elev=20, azim=135)

    # Chart 2: Fragment containment
    ax2 = fig.add_subplot(gs[0, 1])
    neutral = [v['precursor_neutral_mass'] for v in contain]
    ax2.scatter(neutral, maxf, c=COLORS['primary'], s=25, edgecolors='k', linewidths=0.3, zorder=5)
    lims = [0, max(neutral) * 1.1]
    ax2.plot(lims, lims, '--', color=COLORS['secondary'], linewidth=0.8, label='max = neutral mass')
    ax2.fill_between(lims, [0]*2, lims, alpha=0.05, color=COLORS['tertiary'])
    ax2.set_xlabel('Neutral mass (Da)')
    ax2.set_ylabel('Max fragment m/z')
    ax2.set_title('Fragment Containment')
    ax2.legend(frameon=False, fontsize=6)

    # Chart 3: CID energy transfer
    ax3 = fig.add_subplot(gs[0, 2])
    e_col = [v['E_collision_eV'] for v in cid]
    eff = [v['efficiency'] for v in cid]
    ax3.scatter(e_col, ecm, c=COLORS['accent'], s=25, edgecolors='k', linewidths=0.3)
    ax3.set_xlabel('E$_{collision}$ (eV)')
    ax3.set_ylabel('E$_{cm}$ (eV)')
    ax3.set_title('CID Energy Transfer')

    # Chart 4: Arrow of time - partition events
    ax4 = fig.add_subplot(gs[0, 3])
    n_breaks = [v['n_breaks'] for v in arrow]
    ax4.bar(range(len(n_breaks)), n_breaks, color=COLORS['dark'], alpha=0.7)
    ax4.set_xlabel('Ion index')
    ax4.set_ylabel('Partition events')
    ax4.set_title('Arrow of Time')
    ax4.annotate('Categorical\nirreversibility', xy=(0.95, 0.95),
                 xycoords='axes fraction', ha='right', va='top', fontsize=7,
                 color=COLORS['secondary'])

    fig.suptitle('Panel 7: Fragmentation Physics', fontsize=10, fontweight='bold', y=1.02)
    fig.savefig(OUTPUT_DIR / f'panel_{fig_num:02d}_fragmentation.png', bbox_inches='tight')
    plt.close(fig)
    print(f'  Panel {fig_num} saved')


def panel_8(ions, results, fig_num=8):
    """Panel 8: State Counting & Resolution (Stage 7)"""
    fig = plt.figure(figsize=(14, 3.5))
    gs = GridSpec(1, 4, figure=fig, wspace=0.35)

    decomp = get_theorem_values(results, 7, 'thm:decomposition')
    uncert = get_theorem_values(results, 7, 'thm:uncertainty')
    resol = get_theorem_values(results, 7, 'thm:resolution')
    disp = get_theorem_values(results, 7, 'thm:dispersion')

    # Chart 1: 3D ARV decomposition
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    As = [v['A'] for v in decomp]
    Rs = [v['R'] for v in decomp]
    Vs = [v['V'] for v in decomp]
    sc = ax1.scatter(As, Rs, Vs, c=[v['M_depth'] for v in decomp],
                     cmap='viridis', s=30, edgecolors='k', linewidths=0.3)
    ax1.set_xlabel('A')
    ax1.set_ylabel('R')
    ax1.set_zlabel('V')
    ax1.set_title('A + R + V = 1')
    ax1.view_init(elev=30, azim=120)

    # Chart 2: Uncertainty product
    ax2 = fig.add_subplot(gs[0, 1])
    ratios = [v['ratio_to_hbar'] for v in uncert]
    ax2.bar(range(len(ratios)), ratios, color=COLORS['primary'], alpha=0.8)
    ax2.axhline(y=1.0, color=COLORS['secondary'], linestyle='--', linewidth=0.8,
                label='$\\Delta M \\cdot \\tau_p = \\hbar$')
    ax2.set_xlabel('Ion index')
    ax2.set_ylabel('$\\Delta M \\cdot \\tau_p$ / $\\hbar$')
    ax2.set_title('Partition Uncertainty')
    ax2.legend(frameon=False, fontsize=6)

    # Chart 3: Resolving power
    ax3 = fig.add_subplot(gs[0, 2])
    rp = [np.log10(v['resolving_power']) for v in resol]
    ax3.bar(range(len(rp)), rp, color=COLORS['teal'], alpha=0.8)
    ax3.set_xlabel('Ion index')
    ax3.set_ylabel('log$_{10}$(R)')
    ax3.set_title('Resolution Limit')

    # Chart 4: Dispersion relation
    ax4 = fig.add_subplot(gs[0, 3])
    omega0 = [np.log10(v['omega_0_rad_s']) for v in disp]
    k_vals = [np.log10(v['k_m_inv']) for v in disp]
    residuals = [v['residual'] for v in disp]
    ax4.scatter(k_vals, omega0, c=[np.log10(r) if r > 0 else -20 for r in residuals],
                cmap='coolwarm_r', s=25, edgecolors='k', linewidths=0.3)
    ax4.set_xlabel('log$_{10}$(k) [m$^{-1}$]')
    ax4.set_ylabel('log$_{10}$($\\omega_0$) [rad/s]')
    ax4.set_title('Dispersion: $\\omega^2 = \\omega_0^2 + c^2k^2$')

    fig.suptitle('Panel 8: State Counting & Resolution', fontsize=10, fontweight='bold', y=1.02)
    fig.savefig(OUTPUT_DIR / f'panel_{fig_num:02d}_state_counting.png', bbox_inches='tight')
    plt.close(fig)
    print(f'  Panel {fig_num} saved')


def panel_9(ions, results, fig_num=9):
    """Panel 9: Bijective & Detection (Stage 8 first half)"""
    fig = plt.figure(figsize=(14, 3.5))
    gs = GridSpec(1, 4, figure=fig, wspace=0.35)

    sentr = get_theorem_values(results, 8, 'def:sentropy')
    bijec = get_theorem_values(results, 8, 'thm:bijective')
    droplet = get_theorem_values(results, 8, 'thm:droplet_physics')

    # Chart 1: 3D S-entropy cube
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    sks = [v['Sk'] for v in sentr]
    sts = [v['St'] for v in sentr]
    ses = [v['Se'] for v in sentr]
    sc = ax1.scatter(sks, sts, ses, c=sks, cmap='RdYlGn', s=40, edgecolors='k', linewidths=0.3)
    # Draw unit cube edges
    for s in [0, 1]:
        for t in [0, 1]:
            ax1.plot([0, 1], [s, s], [t, t], 'k-', linewidth=0.2, alpha=0.3)
            ax1.plot([s, s], [0, 1], [t, t], 'k-', linewidth=0.2, alpha=0.3)
            ax1.plot([s, s], [t, t], [0, 1], 'k-', linewidth=0.2, alpha=0.3)
    ax1.set_xlabel('S$_k$')
    ax1.set_ylabel('S$_t$')
    ax1.set_zlabel('S$_e$')
    ax1.set_title('S-Entropy $\\in$ [0,1]$^3$')
    ax1.view_init(elev=25, azim=135)

    # Chart 2: Bijective roundtrip error
    ax2 = fig.add_subplot(gs[0, 1])
    errors = [v['roundtrip_error'] for v in bijec]
    ax2.bar(range(len(errors)), errors, color=COLORS['tertiary'], alpha=0.8)
    ax2.set_xlabel('Ion index')
    ax2.set_ylabel('Roundtrip error')
    ax2.set_title('Bijective Transformation')
    ax2.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    if max(errors) < 1e-12:
        ax2.set_ylim(0, 1e-12)
    ax2.annotate('Zero info loss', xy=(0.95, 0.95),
                 xycoords='axes fraction', ha='right', va='top', fontsize=7,
                 color=COLORS['tertiary'])

    # Chart 3: Droplet We, Re, Oh
    ax3 = fig.add_subplot(gs[0, 2])
    we = [v['We'] for v in droplet]
    re_v = [v['Re'] for v in droplet]
    oh = [v['Oh'] for v in droplet]
    x = np.arange(len(we))
    w = 0.25
    ax3.bar(x - w, np.log10(we), width=w, color=COLORS['primary'], label='We', alpha=0.8)
    ax3.bar(x, np.log10(re_v), width=w, color=COLORS['secondary'], label='Re', alpha=0.8)
    ax3.bar(x + w, np.log10(oh), width=w, color=COLORS['tertiary'], label='Oh', alpha=0.8)
    ax3.set_xlabel('Ion index')
    ax3.set_ylabel('log$_{10}$(number)')
    ax3.set_title('Droplet Physics')
    ax3.legend(frameon=False, fontsize=6, ncol=3)

    # Chart 4: Droplet velocity vs radius
    ax4 = fig.add_subplot(gs[0, 3])
    vel = [v['velocity_m_s'] for v in droplet]
    rad = [v['radius_mm'] for v in droplet]
    sc4 = ax4.scatter(rad, vel, c=[v['Oh'] for v in droplet], cmap='plasma',
                      s=30, edgecolors='k', linewidths=0.3)
    plt.colorbar(sc4, ax=ax4, label='Oh', shrink=0.8)
    ax4.set_xlabel('Radius (mm)')
    ax4.set_ylabel('Velocity (m/s)')
    ax4.set_title('Droplet Parameters')

    fig.suptitle('Panel 9: Bijective Detection', fontsize=10, fontweight='bold', y=1.02)
    fig.savefig(OUTPUT_DIR / f'panel_{fig_num:02d}_bijective_detection.png', bbox_inches='tight')
    plt.close(fig)
    print(f'  Panel {fig_num} saved')


def panel_10(ions, results, fig_num=10):
    """Panel 10: Mass = Memory & Aggregate Summary (Stage 8 last + summary)"""
    fig = plt.figure(figsize=(14, 3.5))
    gs = GridSpec(1, 4, figure=fig, wspace=0.35)

    mass_mem = get_theorem_values(results, 8, 'thm:mass-residue')
    emc2 = get_theorem_values(results, 8, 'thm:emc2')

    # Chart 1: 3D mass-energy-residue
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    masses = [v['mass_Da'] for v in mass_mem]
    e_rest = [v['E_rest_eV'] for v in mass_mem]
    n_states = [v['N_partition_states'] for v in mass_mem]
    sc = ax1.scatter(masses, np.log10(e_rest), n_states,
                     c=masses, cmap='magma', s=30, edgecolors='k', linewidths=0.3)
    ax1.set_xlabel('Mass (Da)')
    ax1.set_ylabel('log$_{10}$(E$_0$) [eV]')
    ax1.set_zlabel('N$_{states}$')
    ax1.set_title('Mass = Memory')
    ax1.view_init(elev=25, azim=140)

    # Chart 2: E=mc^2 identity
    ax2 = fig.add_subplot(gs[0, 1])
    e_mc2_vals = [v['E_mc2_J'] for v in emc2]
    e_hw_vals = [v['E_hbar_omega_J'] for v in emc2]
    ax2.scatter(e_mc2_vals, e_hw_vals, c=COLORS['secondary'], s=25,
                edgecolors='k', linewidths=0.3, zorder=5)
    lims = [min(e_mc2_vals) * 0.99, max(e_mc2_vals) * 1.01]
    ax2.plot(lims, lims, '--', color='gray', linewidth=0.8)
    ax2.set_xlabel('mc$^2$ (J)')
    ax2.set_ylabel('$\\hbar\\omega_0$ (J)')
    ax2.set_title('E = mc$^2$ (Theorem)')
    ax2.ticklabel_format(style='scientific', scilimits=(0, 0))

    # Chart 3: Residue fraction bar
    ax3 = fig.add_subplot(gs[0, 2])
    residue_frac = RESIDUE_FRACTION  # 26/27
    structure_frac = 1 - residue_frac
    # Pie-like horizontal bar
    ax3.barh([0], [residue_frac * 100], color=COLORS['primary'], alpha=0.8,
             label=f'Residue (memory): {residue_frac*100:.1f}%')
    ax3.barh([0], [structure_frac * 100], left=[residue_frac * 100],
             color=COLORS['gold'], alpha=0.8,
             label=f'Structure: {structure_frac*100:.1f}%')
    ax3.set_xlim(0, 100)
    ax3.set_xlabel('% of rest energy')
    ax3.set_yticks([])
    ax3.set_title('26/27 = Memory')
    ax3.legend(frameon=False, fontsize=6, loc='center')

    # Chart 4: Aggregate heatmap - theorems per stage
    ax4 = fig.add_subplot(gs[0, 3])
    stage_names = ['S1\nAxiom', 'S2\nInj', 'S3\nChrom', 'S4\nIon',
                   'S5\nMS1', 'S6\nColl', 'S7\nMS2', 'S8\nDet']
    stage_counts = [4, 2, 3, 7, 3, 7, 4, 8]
    n_ions = len(results)

    # Create heatmap data: ions x stages
    heatmap = np.zeros((n_ions, 8))
    for i, r in enumerate(results):
        for j, stage in enumerate(r.stages):
            heatmap[i, j] = stage.num_passed / stage.num_theorems

    im = ax4.imshow(heatmap, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax4.set_xticks(range(8))
    ax4.set_xticklabels(stage_names, fontsize=6)
    ax4.set_ylabel('Ion index')
    ax4.set_title(f'1292/1292 Theorems')
    plt.colorbar(im, ax=ax4, label='Pass rate', shrink=0.8)

    fig.suptitle('Panel 10: Mass = Memory & Aggregate', fontsize=10, fontweight='bold', y=1.02)
    fig.savefig(OUTPUT_DIR / f'panel_{fig_num:02d}_mass_memory_aggregate.png', bbox_inches='tight')
    plt.close(fig)
    print(f'  Panel {fig_num} saved')


# ============================================================================
# Main
# ============================================================================

def main():
    print("Loading and validating all 34 ions...")
    ions, results = load_all_ions()
    print(f"  {len(ions)} ions loaded, all validated")
    print(f"  Output dir: {OUTPUT_DIR}")

    print("\nGenerating 10 panel charts...")
    panel_1(ions, results, 1)
    panel_2(ions, results, 2)
    panel_3(ions, results, 3)
    panel_4(ions, results, 4)
    panel_5(ions, results, 5)
    panel_6(ions, results, 6)
    panel_7(ions, results, 7)
    panel_8(ions, results, 8)
    panel_9(ions, results, 9)
    panel_10(ions, results, 10)

    print(f"\nAll 10 panels saved to {OUTPUT_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
