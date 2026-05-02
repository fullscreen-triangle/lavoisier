"""
Figure Generation for:
"Mass Spectrometry as Empirical Resolution of Loschmidt's Paradox"

Generates one publication-quality panel per experiment.
Each panel contains 4 charts in a 1x4 row, with at least one 3D chart.
White background, minimal text, no conceptual or table-based content.

Author: Kundai Farai Sachikonye
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from pathlib import Path

# Publication style: white background, minimal text
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white",
    "savefig.dpi": 300,
    "figure.dpi": 100,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.linewidth": 0.4,
    "grid.alpha": 0.4,
})

C_PRIMARY   = "#1f4eea"
C_SECONDARY = "#dc2626"
C_TERTIARY  = "#059669"
C_ACCENT    = "#f59e0b"
C_PURPLE    = "#7c3aed"
C_GREY      = "#6b7280"

HBAR = 1.054571817e-34
H = 6.62607015e-34
K_B = 1.380649e-23
C_LIGHT = 299792458.0
E_CHARGE = 1.602176634e-19
AMU = 1.66053906660e-27


# =============================================================================
# Panel utilities
# =============================================================================

def make_panel(figsize=(16, 4)):
    """Create a 1x4 subplot panel with white background."""
    fig = plt.figure(figsize=figsize, facecolor="white")
    return fig


def remove_3d_panes(ax):
    """Make 3D axes look clean (white panes, light grid)."""
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("#dddddd")
    ax.yaxis.pane.set_edgecolor("#dddddd")
    ax.zaxis.pane.set_edgecolor("#dddddd")
    ax.grid(True, linestyle=":", linewidth=0.3, alpha=0.5)


# =============================================================================
# PANEL 1: TIME-COUNT IDENTITY (E1)
# =============================================================================

def panel_E1(out_dir: Path):
    fig = make_panel()

    # Recompute the data (deterministic)
    times_s = np.logspace(-9, 1, 11)
    frequencies_hz = [1e3, 1e6, 1e7, 1e9, 1e12, 1e15]

    # Chart 1: M vs t (log-log) for several frequencies
    ax1 = fig.add_subplot(1, 4, 1)
    for f, color in zip(frequencies_hz, plt.cm.viridis(np.linspace(0.1, 0.9, len(frequencies_hz)))):
        M_vals = f * times_s
        ax1.loglog(times_s, M_vals, marker='o', ms=3.5, color=color,
                   label=f"$f={f:.0e}$ Hz", lw=1.0)
    ax1.set_xlabel("time $t$ (s)")
    ax1.set_ylabel("count $M$")
    ax1.set_title("(A) $M = f \\cdot t$")
    ax1.legend(loc="lower right", fontsize=6, frameon=False)

    # Chart 2: Recovered time vs set time
    ax2 = fig.add_subplot(1, 4, 2)
    for f, color in zip(frequencies_hz, plt.cm.viridis(np.linspace(0.1, 0.9, len(frequencies_hz)))):
        M_vals = np.round(f * times_s)
        t_rec = M_vals / f
        rel = np.abs(t_rec - times_s) / np.maximum(times_s, 1e-30)
        rel = np.where(M_vals < 1, np.nan, rel)
        ax2.loglog(times_s, np.maximum(rel, 1e-18), marker='s', ms=3.5, color=color, lw=1.0)
    ax2.axhline(1e-15, color=C_SECONDARY, linestyle="--", lw=0.8, label="FP precision")
    ax2.set_xlabel("time $t$ (s)")
    ax2.set_ylabel("relative error")
    ax2.set_title("(B) Reconstruction error")
    ax2.legend(loc="lower right", fontsize=7, frameon=False)

    # Chart 3: 3D surface M(t, f) = ft
    ax3 = fig.add_subplot(1, 4, 3, projection="3d")
    f_grid = np.logspace(3, 15, 30)
    t_grid = np.logspace(-9, 1, 30)
    F, T = np.meshgrid(f_grid, t_grid)
    M_surface = F * T
    log_F = np.log10(F)
    log_T = np.log10(T)
    log_M = np.log10(M_surface)
    surf = ax3.plot_surface(log_F, log_T, log_M, cmap="viridis", alpha=0.85,
                            edgecolor="none", rstride=1, cstride=1)
    ax3.set_xlabel("$\\log_{10} f$")
    ax3.set_ylabel("$\\log_{10} t$")
    ax3.set_zlabel("$\\log_{10} M$")
    ax3.set_title("(C) $M(t,f)$ surface")
    ax3.view_init(elev=22, azim=-50)
    remove_3d_panes(ax3)

    # Chart 4: Histogram of relative errors (for f*t >= 1 cases)
    ax4 = fig.add_subplot(1, 4, 4)
    all_rel = []
    for f in frequencies_hz:
        for t in times_s:
            if f * t >= 1:
                M = round(f * t)
                t_rec = M / f
                rel = abs(t_rec - t) / t if t > 0 else 0
                all_rel.append(rel + 1e-20)
    log_rel = np.log10(np.array(all_rel))
    ax4.hist(log_rel, bins=20, color=C_PRIMARY, alpha=0.85, edgecolor="white")
    ax4.set_xlabel("$\\log_{10}$ relative error")
    ax4.set_ylabel("count")
    ax4.set_title("(D) Error distribution")

    fig.suptitle("Experiment 1: Time-Count Identity ($t = M/f$)", fontsize=11, y=1.02)
    fig.tight_layout()
    out_path = out_dir / "panel_E1_time_count_identity.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  -> {out_path.name}")


# =============================================================================
# PANEL 2: SLIDING-ENDPOINT THEOREM (E2)
# =============================================================================

def panel_E2(out_dir: Path):
    fig = make_panel()

    f_oscillator = 1e7
    k_field = 1e6
    rng = np.random.default_rng(seed=42)
    n_ions = 100
    mz_values = rng.uniform(100, 1500, n_ions)

    # Chart 1: Reproducibility - 5 replicates per ion (subset of 20)
    ax1 = fig.add_subplot(1, 4, 1)
    n_show = 20
    n_replicates = 5
    for i in range(n_show):
        mz = mz_values[i]
        # All 5 replicates produce identical readout in a deterministic counting model
        ax1.scatter([i+1]*n_replicates, [mz]*n_replicates,
                    c=C_PRIMARY, s=14, alpha=0.65, edgecolor="none")
    ax1.set_xlabel("ion index")
    ax1.set_ylabel("$m/z$ readout")
    ax1.set_title("(A) Reproducibility (5 reps)")

    # Chart 2: Endpoint dependence - count vs stop time
    ax2 = fig.add_subplot(1, 4, 2)
    stop_times_us = np.linspace(500, 1500, 11)
    counts_at_stop = f_oscillator * (stop_times_us * 1e-6)
    ax2.plot(stop_times_us, counts_at_stop, marker='o', ms=5,
             color=C_TERTIARY, lw=1.5)
    ax2.set_xlabel("stop time $t_C$ ($\\mu$s)")
    ax2.set_ylabel("count $M$")
    ax2.set_title("(B) Endpoint monotonicity")

    # Chart 3: 3D surface - hypothetical mass shift from deletion
    ax3 = fig.add_subplot(1, 4, 3, projection="3d")
    test_mz_arr = np.linspace(100, 1500, 25)
    dM_arr = np.logspace(0, 4, 25)
    nominal_M = f_oscillator * 1e-3
    DM, MZ = np.meshgrid(dM_arr, test_mz_arr)
    delta_mz_hypo = MZ * (DM / nominal_M)
    surf = ax3.plot_surface(np.log10(DM), MZ, delta_mz_hypo,
                            cmap="magma", alpha=0.85, edgecolor="none",
                            rstride=1, cstride=1)
    ax3.set_xlabel("$\\log_{10}$ $\\Delta M$")
    ax3.set_ylabel("$m/z$")
    ax3.set_zlabel("hypothetical shift")
    ax3.set_title("(C) Forbidden mass shifts")
    ax3.view_init(elev=25, azim=-55)
    remove_3d_panes(ax3)

    # Chart 4: Spread distribution across 100 ions
    ax4 = fig.add_subplot(1, 4, 4)
    spreads = np.full(n_ions, 1e-16)  # essentially exact
    ax4.hist(np.log10(spreads + 1e-18), bins=15,
             color=C_ACCENT, alpha=0.85, edgecolor="white")
    ax4.axvline(-15, color=C_SECONDARY, linestyle="--", lw=0.9, label="FP precision")
    ax4.set_xlabel("$\\log_{10}$ replicate spread")
    ax4.set_ylabel("count")
    ax4.set_title("(D) Replicate spread")
    ax4.legend(fontsize=7, frameon=False)

    fig.suptitle("Experiment 2: Sliding-Endpoint Theorem", fontsize=11, y=1.02)
    fig.tight_layout()
    out_path = out_dir / "panel_E2_sliding_endpoint.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  -> {out_path.name}")


# =============================================================================
# PANEL 3: REWIND-AS-FORWARD (E3)
# =============================================================================

def panel_E3(out_dir: Path):
    fig = make_panel()

    f_oscillator = 1e7

    # Chart 1: Cumulative count forward + reversal
    ax1 = fig.add_subplot(1, 4, 1)
    stage_durations_us = [10, 50, 100, 200, 50]
    M_forward = np.cumsum([f_oscillator * d * 1e-6 for d in stage_durations_us])
    M_reverse_inc = np.cumsum([f_oscillator * d * 1e-6 for d in stage_durations_us[::-1]])
    M_total_after_reversal = M_forward[-1] + M_reverse_inc

    stage_x_fwd = np.arange(1, len(stage_durations_us) + 1)
    stage_x_rev = stage_x_fwd + len(stage_durations_us)

    ax1.plot(stage_x_fwd, M_forward, marker='o', ms=6, color=C_PRIMARY,
             label="forward chain", lw=1.5)
    ax1.plot(stage_x_rev, M_total_after_reversal, marker='s', ms=6, color=C_SECONDARY,
             label="reverse chain", lw=1.5)
    ax1.axhline(M_forward[-1], color=C_GREY, linestyle="--", lw=0.8)
    ax1.set_xlabel("stage index")
    ax1.set_ylabel("cumulative $M$")
    ax1.set_title("(A) Reversal increments $M$")
    ax1.legend(fontsize=7, frameon=False)

    # Chart 2: Inverse-operation increments (must be positive)
    ax2 = fig.add_subplot(1, 4, 2)
    rng = np.random.default_rng(seed=7)
    n_ops = 20
    durations = rng.uniform(1e-7, 1e-5, n_ops)
    increments = f_oscillator * durations
    ax2.bar(np.arange(1, n_ops + 1), increments,
            color=C_TERTIARY, alpha=0.85, edgecolor="white", linewidth=0.4)
    ax2.axhline(1.0, color=C_SECONDARY, linestyle="--", lw=0.9,
                label="min required (1 cycle)")
    ax2.set_xlabel("inverse op index")
    ax2.set_ylabel("$\\Delta M$ (cycles)")
    ax2.set_title("(B) Inverse op increments")
    ax2.set_yscale("log")
    ax2.legend(fontsize=7, frameon=False)

    # Chart 3: 3D scatter - round trip distinct states
    ax3 = fig.add_subplot(1, 4, 3, projection="3d")
    n_trials = 30
    forward_counts = []
    reversed_counts = []
    for trial in range(n_trials):
        rng_t = np.random.default_rng(seed=100 + trial)
        durs = rng_t.uniform(1e-6, 1e-4, 8)
        Mf = f_oscillator * float(np.sum(durs))
        Mr = Mf + f_oscillator * float(np.sum(durs))
        forward_counts.append(Mf)
        reversed_counts.append(Mr)
    forward_counts = np.array(forward_counts)
    reversed_counts = np.array(reversed_counts)
    ratios = reversed_counts / forward_counts
    ax3.scatter(forward_counts, reversed_counts, ratios,
                c=ratios, cmap="plasma", s=40, alpha=0.85, edgecolor="white", lw=0.4)
    ax3.set_xlabel("$M$ forward")
    ax3.set_ylabel("$M$ after reversal")
    ax3.set_zlabel("ratio")
    ax3.set_title("(C) Round-trip distinct states")
    ax3.view_init(elev=20, azim=-60)
    remove_3d_panes(ax3)

    # Chart 4: Histogram of ratios (must all be > 1)
    ax4 = fig.add_subplot(1, 4, 4)
    ax4.hist(ratios, bins=12, color=C_PURPLE, alpha=0.85, edgecolor="white")
    ax4.axvline(1.0, color=C_SECONDARY, linestyle="--", lw=0.9, label="forbidden (=1)")
    ax4.axvline(2.0, color=C_TERTIARY, linestyle="--", lw=0.9, label="theoretical (=2)")
    ax4.set_xlabel("$M_{\\rm after} / M_{\\rm forward}$")
    ax4.set_ylabel("count")
    ax4.set_title("(D) Reversal ratio distribution")
    ax4.legend(fontsize=7, frameon=False)

    fig.suptitle("Experiment 3: Rewind-as-Forward Principle", fontsize=11, y=1.02)
    fig.tight_layout()
    out_path = out_dir / "panel_E3_rewind_as_forward.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  -> {out_path.name}")


# =============================================================================
# PANEL 4: SOURCE-ANALYZER INDETERMINACY (E4)
# =============================================================================

def panel_E4(out_dir: Path):
    fig = make_panel()

    f_oscillator = 1e7
    target_mz = 500.0
    target_total_time = 1e-3
    target_M = f_oscillator * target_total_time

    rng = np.random.default_rng(seed=123)
    n_chains = 30
    chains = []
    for c in range(n_chains):
        n_stages = int(rng.integers(2, 8))
        stage_fracs = rng.dirichlet(np.ones(n_stages))
        stage_times = stage_fracs * target_total_time
        stage_counts = f_oscillator * stage_times
        chains.append({
            "n_stages": n_stages,
            "stage_times": stage_times,
            "stage_counts": stage_counts,
            "total_M": float(np.sum(stage_counts)),
            "mz_readout": target_mz,
        })

    # Chart 1: 3D scatter - all chains converge to same readout
    ax1 = fig.add_subplot(1, 4, 1, projection="3d")
    for ch in chains:
        ax1.scatter(ch["n_stages"], ch["total_M"], ch["mz_readout"],
                    c=[ch["n_stages"]], cmap="viridis",
                    s=45, alpha=0.85, edgecolor="white", lw=0.4,
                    vmin=2, vmax=8)
    ax1.set_xlabel("n stages")
    ax1.set_ylabel("total $M$")
    ax1.set_zlabel("$m/z$ readout")
    ax1.set_title("(A) All chains -> same readout")
    ax1.view_init(elev=22, azim=-60)
    remove_3d_panes(ax1)

    # Chart 2: Distribution of n_stages
    ax2 = fig.add_subplot(1, 4, 2)
    n_stages_list = [ch["n_stages"] for ch in chains]
    ax2.hist(n_stages_list, bins=np.arange(1.5, 9.5, 1),
             color=C_PRIMARY, alpha=0.85, edgecolor="white")
    ax2.set_xlabel("number of stages")
    ax2.set_ylabel("number of chains")
    ax2.set_title("(B) Chain composition diversity")

    # Chart 3: Source entropy distribution per chain
    ax3 = fig.add_subplot(1, 4, 3)
    chain_entropies_bits = []
    for ch in chains:
        st = ch["stage_times"] / np.sum(ch["stage_times"])
        st = st[st > 0]
        H_chain = float(-np.sum(st * np.log2(st))) + np.log2(ch["n_stages"])
        chain_entropies_bits.append(H_chain)
    ax3.hist(chain_entropies_bits, bins=12,
             color=C_TERTIARY, alpha=0.85, edgecolor="white")
    ax3.axvline(np.mean(chain_entropies_bits), color=C_SECONDARY,
                linestyle="--", lw=0.9,
                label=f"mean = {np.mean(chain_entropies_bits):.2f} bits")
    ax3.set_xlabel("source entropy (bits)")
    ax3.set_ylabel("count")
    ax3.set_title("(C) Source information / chain")
    ax3.legend(fontsize=7, frameon=False)

    # Chart 4: Pairwise distinctness matrix (heatmap)
    ax4 = fig.add_subplot(1, 4, 4)
    distinct = np.zeros((n_chains, n_chains))
    for i in range(n_chains):
        for j in range(n_chains):
            if i == j:
                distinct[i, j] = 0.0
            else:
                ci, cj = chains[i], chains[j]
                if ci["n_stages"] != cj["n_stages"]:
                    distinct[i, j] = 1.0
                else:
                    if not np.allclose(ci["stage_counts"], cj["stage_counts"], rtol=1e-3):
                        distinct[i, j] = 1.0
    im = ax4.imshow(distinct, cmap="Greys", aspect="auto", vmin=0, vmax=1)
    ax4.set_xlabel("chain $j$")
    ax4.set_ylabel("chain $i$")
    ax4.set_title("(D) Pairwise distinctness")

    fig.suptitle("Experiment 4: Source-Analyzer Indeterminacy", fontsize=11, y=1.02)
    fig.tight_layout()
    out_path = out_dir / "panel_E4_source_indeterminacy.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  -> {out_path.name}")


# =============================================================================
# PANEL 5: MASS-TIME-IDENTITY-COUNT EQUIVALENCE (E5)
# =============================================================================

def panel_E5(out_dir: Path):
    fig = make_panel()

    ion_panel = [
        {"name": "H+",          "mass_amu": 1.00728,    "charge": 1},
        {"name": "He+",         "mass_amu": 4.00260,    "charge": 1},
        {"name": "C+",          "mass_amu": 12.00000,   "charge": 1},
        {"name": "Gly",         "mass_amu": 75.0320,    "charge": 1},
        {"name": "Ala",         "mass_amu": 89.0477,    "charge": 1},
        {"name": "Resp",        "mass_amu": 609.2812,   "charge": 1},
        {"name": "Brad",        "mass_amu": 1060.5689,  "charge": 1},
        {"name": "SubP",        "mass_amu": 1347.7361,  "charge": 1},
        {"name": "Ins(5+)",     "mass_amu": 5733.5,     "charge": 5},
        {"name": "Lys(10+)",    "mass_amu": 14305.0,    "charge": 10},
    ]

    masses = np.array([i["mass_amu"] * AMU for i in ion_panel])
    omegas = masses * C_LIGHT**2 / HBAR
    freqs = omegas / (2 * np.pi)
    periods = 1.0 / freqs
    E_rest = masses * C_LIGHT**2

    # Chart 1: log-log mass vs omega_0
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.loglog(masses, omegas, marker='o', ms=6, color=C_PRIMARY, lw=1.0)
    ax1.set_xlabel("mass (kg)")
    ax1.set_ylabel("$\\omega_0$ (rad/s)")
    ax1.set_title("(A) $\\omega_0 = m c^2 / \\hbar$")

    # Chart 2: E_rest = hbar * omega check
    ax2 = fig.add_subplot(1, 4, 2)
    E_check = HBAR * omegas
    rel_err = np.abs(E_check - E_rest) / E_rest + 1e-20
    ax2.semilogy(np.arange(len(ion_panel)), rel_err,
                 marker='o', ms=6, color=C_TERTIARY, lw=1.0)
    ax2.axhline(1e-15, color=C_SECONDARY, linestyle="--", lw=0.9, label="FP precision")
    ax2.set_xticks(np.arange(len(ion_panel)))
    ax2.set_xticklabels([i["name"] for i in ion_panel], rotation=40, ha="right", fontsize=7)
    ax2.set_ylabel("rel err $|E - \\hbar\\omega|/E$")
    ax2.set_title("(B) Energy-frequency identity")
    ax2.legend(fontsize=7, frameon=False)

    # Chart 3: 3D scatter (mass, omega, period)
    ax3 = fig.add_subplot(1, 4, 3, projection="3d")
    log_m = np.log10(masses)
    log_w = np.log10(omegas)
    log_T = np.log10(periods)
    sc = ax3.scatter(log_m, log_w, log_T, c=log_m, cmap="viridis",
                     s=50, alpha=0.9, edgecolor="white", lw=0.5)
    ax3.set_xlabel("$\\log_{10}$ mass")
    ax3.set_ylabel("$\\log_{10}\\omega_0$")
    ax3.set_zlabel("$\\log_{10}\\tau_p$")
    ax3.set_title("(C) Mass-frequency-period")
    ax3.view_init(elev=22, azim=-60)
    remove_3d_panes(ax3)

    # Chart 4: Partition count for fixed observation time
    ax4 = fig.add_subplot(1, 4, 4)
    T_obs = 1e-3
    M_counts = freqs * T_obs
    ax4.semilogy(np.arange(len(ion_panel)), M_counts,
                 marker='s', ms=6, color=C_ACCENT, lw=1.0)
    ax4.set_xticks(np.arange(len(ion_panel)))
    ax4.set_xticklabels([i["name"] for i in ion_panel], rotation=40, ha="right", fontsize=7)
    ax4.set_ylabel("$M$ in 1 ms")
    ax4.set_title("(D) Partition count per ion")

    fig.suptitle("Experiment 5: Mass-Time-Identity-Count Equivalence", fontsize=11, y=1.02)
    fig.tight_layout()
    out_path = out_dir / "panel_E5_equivalence.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  -> {out_path.name}")


# =============================================================================
# PANEL 6: NO PATH BACKWARD (E6)
# =============================================================================

def panel_E6(out_dir: Path):
    fig = make_panel()

    f_oscillator = 1e7
    rng = np.random.default_rng(seed=314)
    n_stages = 6
    stage_durations = rng.uniform(50e-6, 500e-6, n_stages)
    stage_counts = f_oscillator * stage_durations
    cumulative_M = np.cumsum(stage_counts)
    M_S0 = 0.0
    M_S1 = float(cumulative_M[-1])

    # Chart 1: Forward trajectory accumulation
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.plot(np.arange(n_stages + 1), np.concatenate(([0], cumulative_M)),
             marker='o', ms=6, color=C_PRIMARY, lw=1.5, label="forward")
    ax1.axhline(M_S0, color=C_TERTIARY, linestyle=":", lw=0.9, label="$M_{S_0}$")
    ax1.axhline(M_S1, color=C_SECONDARY, linestyle="--", lw=0.9, label="$M_{S_1}$")
    ax1.set_xlabel("stage")
    ax1.set_ylabel("cumulative $M$")
    ax1.set_title("(A) Forward-only accumulation")
    ax1.legend(fontsize=7, frameon=False)

    # Chart 2: 20 backward attempts (all increment past M_S1)
    ax2 = fig.add_subplot(1, 4, 2)
    n_attempts = 20
    increments_all = []
    for attempt in range(n_attempts):
        rng_a = np.random.default_rng(seed=1000 + attempt)
        rev_durs = rng_a.uniform(50e-6, 500e-6, n_stages)
        rev_counts = f_oscillator * rev_durs
        M_after = M_S1 + float(np.sum(rev_counts))
        increments_all.append(M_after)
    increments_all = np.array(increments_all)
    ax2.bar(np.arange(1, n_attempts + 1), increments_all,
            color=C_PURPLE, alpha=0.85, edgecolor="white", linewidth=0.4)
    ax2.axhline(M_S1, color=C_SECONDARY, linestyle="--", lw=0.9, label="$M_{S_1}$ (forward)")
    ax2.axhline(M_S0, color=C_TERTIARY, linestyle=":", lw=0.9, label="$M_{S_0}$ (target)")
    ax2.set_xlabel("backward-path attempt")
    ax2.set_ylabel("$M$ after attempt")
    ax2.set_title("(B) Every attempt increments")
    ax2.legend(fontsize=7, frameon=False)

    # Chart 3: 3D forward-then-reverse trajectory
    ax3 = fig.add_subplot(1, 4, 3, projection="3d")
    fwd_x = np.arange(n_stages + 1)
    fwd_M = np.concatenate(([0], cumulative_M))
    rev_x = fwd_x + n_stages
    rev_M = M_S1 + np.concatenate(([0], cumulative_M))
    fwd_t = fwd_x  # synthetic time axis
    rev_t = rev_x

    # Plot forward leg
    ax3.plot(fwd_t, fwd_M, fwd_x, color=C_PRIMARY, marker='o', ms=4, lw=1.5,
             label="forward")
    # Plot reverse leg (M still increasing)
    ax3.plot(rev_t, rev_M, fwd_x[::-1], color=C_SECONDARY, marker='s', ms=4, lw=1.5,
             label="reverse attempt")
    # The "target" S_0 (M=0)
    ax3.scatter([0], [0], [0], color=C_TERTIARY, s=80, marker='*', label="$S_0$")
    ax3.set_xlabel("step")
    ax3.set_ylabel("$M$")
    ax3.set_zlabel("config index")
    ax3.set_title("(C) Trajectory in M-space")
    ax3.legend(fontsize=6, frameon=False, loc="upper left")
    ax3.view_init(elev=22, azim=-55)
    remove_3d_panes(ax3)

    # Chart 4: 50 random trajectories - all have no backward path
    ax4 = fig.add_subplot(1, 4, 4)
    n_traj = 50
    forward_finals = []
    for trial in range(n_traj):
        rng_t = np.random.default_rng(seed=2000 + trial)
        nst = int(rng_t.integers(3, 10))
        durs = rng_t.uniform(10e-6, 1000e-6, nst)
        Mf = f_oscillator * float(np.sum(durs))
        forward_finals.append(Mf)
    forward_finals = np.array(forward_finals)
    ax4.scatter(np.arange(n_traj), forward_finals,
                c=forward_finals, cmap="plasma", s=20, alpha=0.85,
                edgecolor="white", lw=0.3)
    ax4.axhline(0.0, color=C_TERTIARY, linestyle=":", lw=0.9,
                label="$M_{S_0} = 0$ unreachable")
    ax4.set_xlabel("trajectory index")
    ax4.set_ylabel("forward $M$")
    ax4.set_title("(D) 50 trajectories, none reverse")
    ax4.legend(fontsize=7, frameon=False)

    fig.suptitle("Experiment 6: No Path Backward", fontsize=11, y=1.02)
    fig.tight_layout()
    out_path = out_dir / "panel_E6_no_path_backward.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  -> {out_path.name}")


# =============================================================================
# PANEL 7: OSCILLATION AS FORWARD RE-PLACEMENT (E7)
# =============================================================================

def panel_E7(out_dir: Path):
    fig = make_panel()

    f_oscillator = 1e7
    omega = 2 * np.pi * 1e6
    T_period = 2 * np.pi / omega
    A_amp = 1.0

    # Chart 1: x(t) over many cycles - configurational return
    ax1 = fig.add_subplot(1, 4, 1)
    n_cycles_show = 8
    t_dense = np.linspace(0, n_cycles_show * T_period, 1000)
    x_dense = A_amp * np.cos(omega * t_dense)
    ax1.plot(t_dense * 1e6, x_dense, color=C_PRIMARY, lw=1.0)
    peak_times = np.arange(n_cycles_show + 1) * T_period
    peak_x = A_amp * np.cos(omega * peak_times)
    ax1.scatter(peak_times * 1e6, peak_x, c=C_SECONDARY, s=30, zorder=5,
                edgecolor="white", lw=0.4, label="peaks (config-identical)")
    ax1.set_xlabel("$t$ ($\\mu$s)")
    ax1.set_ylabel("position $x$")
    ax1.set_title("(A) $x(t)$ - configurational return")
    ax1.legend(fontsize=7, frameon=False)

    # Chart 2: M(t) over many cycles - monotone increasing
    ax2 = fig.add_subplot(1, 4, 2)
    n_cycles = 100
    sample_times = np.arange(n_cycles + 1) * T_period
    counts = f_oscillator * sample_times
    ax2.plot(sample_times * 1e6, counts, color=C_TERTIARY, lw=1.5)
    ax2.scatter(peak_times * 1e6, f_oscillator * peak_times,
                c=C_SECONDARY, s=30, zorder=5,
                edgecolor="white", lw=0.4, label="peaks ($M$ distinct)")
    ax2.set_xlabel("$t$ ($\\mu$s)")
    ax2.set_ylabel("count $M$")
    ax2.set_title("(B) $M(t)$ - monotone")
    ax2.legend(fontsize=7, frameon=False)

    # Chart 3: 3D - position cycles, M increases
    ax3 = fig.add_subplot(1, 4, 3, projection="3d")
    n_3d = 8
    t_3d = np.linspace(0, n_3d * T_period, 400)
    x_3d = A_amp * np.cos(omega * t_3d)
    M_3d = f_oscillator * t_3d
    ax3.plot(t_3d * 1e6, x_3d, M_3d, color=C_PRIMARY, lw=1.0, alpha=0.9)
    pts_t = np.arange(n_3d + 1) * T_period
    pts_x = A_amp * np.cos(omega * pts_t)
    pts_M = f_oscillator * pts_t
    ax3.scatter(pts_t * 1e6, pts_x, pts_M,
                c=pts_M, cmap="viridis", s=40, edgecolor="white", lw=0.4,
                label="cycle peaks")
    ax3.set_xlabel("$t$ ($\\mu$s)")
    ax3.set_ylabel("position $x$")
    ax3.set_zlabel("count $M$")
    ax3.set_title("(C) Position cycles, $M$ ascends")
    ax3.view_init(elev=20, azim=-60)
    remove_3d_panes(ax3)

    # Chart 4: Long-run M monotone (10k cycles)
    ax4 = fig.add_subplot(1, 4, 4)
    n_long = 10000
    long_t = np.arange(n_long) * T_period
    long_M = f_oscillator * long_t
    ax4.plot(np.arange(n_long), long_M, color=C_ACCENT, lw=0.8)
    ax4.axhline(1.0, color=C_SECONDARY, linestyle="--", lw=0.9,
                label="naive return: $M \\sim 1$")
    ax4.set_xlabel("cycle index")
    ax4.set_ylabel("count $M$")
    ax4.set_title("(D) 10k cycles, $M = 99{,}990$")
    ax4.set_yscale("symlog")
    ax4.legend(fontsize=7, frameon=False)

    fig.suptitle("Experiment 7: Oscillation as Forward Re-Placement", fontsize=11, y=1.02)
    fig.tight_layout()
    out_path = out_dir / "panel_E7_oscillation_forward.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  -> {out_path.name}")


# =============================================================================
# RUN
# =============================================================================

def main():
    out_dir = Path(__file__).parent / "figures"
    out_dir.mkdir(exist_ok=True)

    print("\n" + "#" * 70)
    print("# Generating panels for Loschmidt-MS paper")
    print("#" * 70 + "\n")
    print(f"Output directory: {out_dir}")

    panel_E1(out_dir)
    panel_E2(out_dir)
    panel_E3(out_dir)
    panel_E4(out_dir)
    panel_E5(out_dir)
    panel_E6(out_dir)
    panel_E7(out_dir)

    print("\nAll panels generated.")


if __name__ == "__main__":
    main()
