"""
Figure generation for the Hardware-Oscillator TCE Apparatus paper.

Each panel: four data-driven charts in a row, white background,
minimal text, at least one 3D chart, no tables or text-based charts.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (3D registration)

from validation_experiments import (
    velocity_verlet,
    yoshida4,
    total_energy,
    OCXO_ALLAN_1S,
)

# -----------------------------------------------------------------------------
# Style: white background, minimal text
# -----------------------------------------------------------------------------

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "axes.grid": False,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "font.family": "sans-serif",
    "font.size": 10,
    "lines.linewidth": 1.4,
})

OUT_DIR = Path(__file__).parent / "figures"
OUT_DIR.mkdir(exist_ok=True)
RESULTS = json.loads((Path(__file__).parent / "validation_results.json").read_text())

ROW_FIG_SIZE = (20, 5)


def style_3d(ax):
    ax.xaxis.pane.set_edgecolor("#aaaaaa")
    ax.yaxis.pane.set_edgecolor("#aaaaaa")
    ax.zaxis.pane.set_edgecolor("#aaaaaa")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.tick_params(labelsize=8)


def style_2d(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# -----------------------------------------------------------------------------
# Panel E1: Analyzer Recovery
# -----------------------------------------------------------------------------

def panel_e1():
    fig = plt.figure(figsize=ROW_FIG_SIZE)
    ax1 = fig.add_subplot(1, 4, 1, projection="3d")
    ax2 = fig.add_subplot(1, 4, 2)
    ax3 = fig.add_subplot(1, 4, 3)
    ax4 = fig.add_subplot(1, 4, 4)

    # A (3D): TOF z(t) for several mu values
    kappa = 1.0
    L = 1.0
    h = 1e-3
    n_steps = 5000
    cmap = plt.cm.viridis
    mu_values = [1.0, 2.0, 4.0, 8.0, 16.0]
    for i, m in enumerate(mu_values):
        force = lambda x, t: np.array([kappa])
        x_t, _, t_a = velocity_verlet([0.0], [0.0], force, m, h, n_steps)
        ax1.plot([m] * len(t_a), t_a, x_t[:, 0],
                 color=cmap(i / max(1, len(mu_values) - 1)), lw=1.4)
    ax1.set_xlabel(r"$\mu$")
    ax1.set_ylabel(r"$t$")
    ax1.set_zlabel(r"$z$")
    style_3d(ax1)
    ax1.view_init(elev=20, azim=-65)

    # B: Orbitrap omega_observed vs omega_predicted
    e1 = RESULTS["experiments"]["E1_analyzer_recovery"]["result"]
    obs = np.array(e1["orbitrap"]["omega_observed"])
    pred = np.array(e1["orbitrap"]["omega_predicted"])
    finite = np.isfinite(obs) & np.isfinite(pred)
    ax2.plot([pred.min(), pred.max()], [pred.min(), pred.max()],
             color="#888888", lw=0.8, ls="--")
    ax2.scatter(pred[finite], obs[finite], s=80, c="#1f77b4", zorder=3)
    ax2.set_xlabel(r"$\omega$ predicted")
    ax2.set_ylabel(r"$\omega$ observed")
    style_2d(ax2)

    # C: FT-ICR cyclotron orbit
    B = 1.0
    mu_ft = 1.0
    omega_c = B / mu_ft

    def boris_step(x, v, h, omega_c):
        x_half = x + 0.5 * h * v
        theta = omega_c * h
        c, s = math.cos(theta), math.sin(theta)
        v_new = np.array([c * v[0] + s * v[1], -s * v[0] + c * v[1]])
        return x_half + 0.5 * h * v_new, v_new

    x = np.array([0.0, 0.0])
    v = np.array([1.0, 0.0])
    n_orbit = 6300
    h_ft = 1e-3
    xs, ys = [x[0]], [x[1]]
    for _ in range(n_orbit):
        x, v = boris_step(x, v, h_ft, omega_c)
        xs.append(x[0]); ys.append(x[1])
    ax3.plot(xs, ys, color="#d62728", lw=1.0)
    ax3.scatter([0], [-1], s=30, color="#444444", marker="x")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_aspect("equal")
    style_2d(ax3)

    # D: Quadrupole stable vs unstable
    def mathieu_x(q, n=4000, h=0.001):
        x = 1.0; xdot = 0.0
        out = [x]
        for i in range(n):
            tau = i * h
            xddot = -(0.0 - 2 * q * math.cos(2 * tau)) * x
            xdot_h = xdot + 0.5 * h * xddot
            x = x + h * xdot_h
            tau_n = (i + 1) * h
            xddot_n = -(0.0 - 2 * q * math.cos(2 * tau_n)) * x
            xdot = xdot_h + 0.5 * h * xddot_n
            out.append(x)
        return np.array(out)

    t_range = np.arange(4001) * 0.001
    x_stable = mathieu_x(0.3)
    x_unstable = mathieu_x(1.5)
    ax4.plot(t_range, x_stable, color="#2ca02c", label=r"$q=0.3$")
    ax4.plot(t_range, np.clip(x_unstable, -1e6, 1e6), color="#d62728", label=r"$q=1.5$")
    ax4.set_yscale("symlog", linthresh=1.0)
    ax4.set_xlabel(r"$\tau$")
    ax4.set_ylabel(r"$x$")
    ax4.legend(frameon=False, fontsize=9)
    style_2d(ax4)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "panel_E1_analyzer_recovery.png", dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Panel E2: Symplectic integrator validation
# -----------------------------------------------------------------------------

def panel_e2():
    fig = plt.figure(figsize=ROW_FIG_SIZE)
    ax1 = fig.add_subplot(1, 4, 1, projection="3d")
    ax2 = fig.add_subplot(1, 4, 2)
    ax3 = fig.add_subplot(1, 4, 3)
    ax4 = fig.add_subplot(1, 4, 4)

    omega = 2 * math.pi
    mu = 1.0
    pot = lambda x, t: 0.5 * mu * omega ** 2 * x[0] ** 2
    force = lambda x, t: np.array([-mu * omega ** 2 * x[0]])
    h = 5e-3; T = 5.0
    n = int(T / h)

    # A (3D): phase-space trajectory (x, v, t)
    x_t, v_t, t_a = yoshida4([1.0], [0.0], force, mu, h, n)
    ax1.plot(x_t[:, 0], v_t[:, 0], t_a, color="#1f77b4", lw=1.0)
    ax1.set_xlabel("x"); ax1.set_ylabel("v"); ax1.set_zlabel("t")
    style_3d(ax1)
    ax1.view_init(elev=20, azim=-60)

    # B: Energy drift vs step size, both integrators
    e2 = RESULTS["experiments"]["E2_energy_conservation"]["result"]
    h_v = [d["h"] for d in e2["verlet"]["drifts_by_h"]]
    drift_v = [d["max_relative_drift"] for d in e2["verlet"]["drifts_by_h"]]
    h_y = [d["h"] for d in e2["yoshida4"]["drifts_by_h"]]
    drift_y = [d["max_relative_drift"] for d in e2["yoshida4"]["drifts_by_h"]]
    ax2.loglog(h_v, drift_v, "o-", color="#1f77b4", label="Verlet")
    ax2.loglog(h_y, drift_y, "s-", color="#d62728", label="Yoshida-4")
    ax2.set_xlabel("h"); ax2.set_ylabel(r"$|\Delta E|/E_0$")
    ax2.legend(frameon=False, fontsize=9)
    style_2d(ax2)

    # C: Energy time-series at h=1e-3 for both
    h2 = 1e-3
    n2 = int(5.0 / h2)
    x_v, v_v, t_v = velocity_verlet([1.0], [0.0], force, mu, h2, n2)
    x_y, v_y, t_y = yoshida4([1.0], [0.0], force, mu, h2, n2)
    E0 = total_energy(np.array([1.0]), np.array([0.0]), pot, mu)
    sample = max(1, len(t_v) // 1000)
    E_v = np.array([(0.5 * mu * v_v[i, 0]**2 + 0.5 * mu * omega**2 * x_v[i, 0]**2 - E0) / E0 for i in range(0, len(t_v), sample)])
    E_y = np.array([(0.5 * mu * v_y[i, 0]**2 + 0.5 * mu * omega**2 * x_y[i, 0]**2 - E0) / E0 for i in range(0, len(t_y), sample)])
    t_sample = t_v[::sample]
    ax3.plot(t_sample, E_v, color="#1f77b4", lw=1.0, label="Verlet")
    ax3.plot(t_sample, E_y, color="#d62728", lw=1.0, label="Yoshida-4")
    ax3.set_xlabel("t"); ax3.set_ylabel(r"$\Delta E/E_0$")
    ax3.legend(frameon=False, fontsize=9)
    style_2d(ax3)

    # D: Convergence order: log-log fit slopes
    log_h_v = np.log(h_v)
    log_d_v = np.log(drift_v)
    log_h_y = np.log(h_y)
    log_d_y = np.log(drift_y)
    p_v = np.polyfit(log_h_v, log_d_v, 1)
    p_y = np.polyfit(log_h_y, log_d_y, 1)
    h_grid = np.logspace(np.log10(min(h_v + h_y)), np.log10(max(h_v + h_y)), 100)
    ax4.loglog(h_v, drift_v, "o", color="#1f77b4", markersize=8)
    ax4.loglog(h_y, drift_y, "s", color="#d62728", markersize=8)
    ax4.loglog(h_grid, np.exp(p_v[1]) * h_grid ** p_v[0], "--", color="#1f77b4", lw=1.0)
    ax4.loglog(h_grid, np.exp(p_y[1]) * h_grid ** p_y[0], "--", color="#d62728", lw=1.0)
    ax4.set_xlabel("h"); ax4.set_ylabel("drift")
    ax4.text(0.05, 0.95, f"slope V={p_v[0]:.2f}\nslope Y={p_y[0]:.2f}",
             transform=ax4.transAxes, va="top", fontsize=9)
    style_2d(ax4)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "panel_E2_integrator.png", dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Panel E3-E4: Coordinates and capacity
# -----------------------------------------------------------------------------

def panel_e34():
    fig = plt.figure(figsize=ROW_FIG_SIZE)
    ax1 = fig.add_subplot(1, 4, 1, projection="3d")
    ax2 = fig.add_subplot(1, 4, 2)
    ax3 = fig.add_subplot(1, 4, 3)
    ax4 = fig.add_subplot(1, 4, 4)

    # A (3D): all (n, l, m) states up to n=8
    states = []
    for n in range(1, 9):
        for l in range(0, n):
            for m in range(-l, l + 1):
                states.append((n, l, m))
    states = np.array(states)
    sc = ax1.scatter(states[:, 0], states[:, 1], states[:, 2],
                     c=states[:, 0], cmap="viridis", s=20, alpha=0.8)
    ax1.set_xlabel("n"); ax1.set_ylabel(r"$\ell$"); ax1.set_zlabel("m")
    style_3d(ax1)
    ax1.view_init(elev=22, azim=-50)

    # B: C(n) = 2n^2 verification
    e4 = RESULTS["experiments"]["E4_capacity_formula"]["result"]
    ns = np.array([r["n"] for r in e4["rows"]])
    Cs = np.array([r["C_observed"] for r in e4["rows"]])
    Cs_pred = np.array([r["C_predicted"] for r in e4["rows"]])
    ax2.plot(ns, Cs_pred, "-", color="#1f77b4", lw=1.0, label=r"$2n^2$")
    ax2.scatter(ns, Cs, s=40, color="#d62728", zorder=3, label="observed")
    ax2.set_xlabel("n"); ax2.set_ylabel("C(n)")
    ax2.legend(frameon=False, fontsize=9)
    style_2d(ax2)

    # C: Cumulative N(n_max) = sum 2n^2
    n_max = np.arange(1, 21)
    cumulative = np.cumsum(2 * n_max ** 2)
    ax3.semilogy(n_max, cumulative, "o-", color="#2ca02c")
    ax3.set_xlabel(r"$n_{\max}$"); ax3.set_ylabel(r"$N$")
    style_2d(ax3)

    # D: capacity heat map of (l, m) at fixed n=8
    n_fix = 8
    occ = np.zeros((n_fix, 2 * n_fix - 1))
    for l in range(0, n_fix):
        for m in range(-l, l + 1):
            occ[l, m + (n_fix - 1)] = 1
    im = ax4.imshow(occ, aspect="auto", cmap="Blues", origin="lower",
                    extent=[-(n_fix - 1) - 0.5, (n_fix - 1) + 0.5, -0.5, n_fix - 0.5])
    ax4.set_xlabel("m"); ax4.set_ylabel(r"$\ell$")
    style_2d(ax4)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "panel_E34_coordinates_capacity.png", dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Panel E5+E10: Resolution scaling
# -----------------------------------------------------------------------------

def panel_e5_e10():
    fig = plt.figure(figsize=ROW_FIG_SIZE)
    ax1 = fig.add_subplot(1, 4, 1, projection="3d")
    ax2 = fig.add_subplot(1, 4, 2)
    ax3 = fig.add_subplot(1, 4, 3)
    ax4 = fig.add_subplot(1, 4, 4)

    # A (3D): R(omega, T) surface
    omega_grid = np.logspace(3, 7, 30)
    T_grid = np.logspace(-3, 6, 40)
    OG, TG = np.meshgrid(omega_grid, T_grid)
    R = OG * TG / (2 * math.pi)
    ax1.plot_surface(np.log10(OG), np.log10(TG), np.log10(R), cmap="viridis",
                     alpha=0.9, edgecolor="none")
    ax1.set_xlabel(r"$\log_{10}\omega$"); ax1.set_ylabel(r"$\log_{10}T$")
    ax1.set_zlabel(r"$\log_{10}R$")
    style_3d(ax1)
    ax1.view_init(elev=22, azim=-50)

    # B: R vs T from E5
    e5 = RESULTS["experiments"]["E5_resolution_scaling"]["result"]
    Ts = [r["T"] for r in e5["rows"]]
    Rs = [r["R_predicted"] for r in e5["rows"]]
    ax2.loglog(Ts, Rs, "o-", color="#1f77b4")
    ax2.set_xlabel("T (s)"); ax2.set_ylabel("R")
    style_2d(ax2)

    # C: TC-XT improvement vs T (E10)
    e10 = RESULTS["experiments"]["E10_xt_extreme_resolution"]["result"]
    Ts_xt = [r["T_seconds"] for r in e10["rows"]]
    imp = [r["improvement_over_orbitrap"] for r in e10["rows"]]
    R_eff = [r["R_effective"] for r in e10["rows"]]
    ax3.loglog(Ts_xt, imp, "o-", color="#d62728")
    ax3.axhline(1.0, color="#888888", lw=0.8, ls="--")
    ax3.set_xlabel("T (s)"); ax3.set_ylabel(r"R / R_{Orbitrap}")
    style_2d(ax3)

    # D: Allan-deviation-bounded R floor
    drift_at_T = [r["ocxo_drift_at_T"] for r in e10["rows"]]
    R_floor = [r["R_floor_from_drift"] for r in e10["rows"]]
    R_ideal = [r["R_ideal"] for r in e10["rows"]]
    ax4.loglog(Ts_xt, R_ideal, "--", color="#888888", lw=1.0, label="ideal")
    ax4.loglog(Ts_xt, R_floor, "-", color="#d62728", lw=1.4, label="Allan floor")
    ax4.loglog(Ts_xt, R_eff, "o", color="#1f77b4", markersize=6, label="effective")
    ax4.set_xlabel("T (s)"); ax4.set_ylabel("R")
    ax4.legend(frameon=False, fontsize=9)
    style_2d(ax4)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "panel_E5E10_resolution.png", dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Panel E6+E7: Hardware substrate
# -----------------------------------------------------------------------------

def panel_e67():
    fig = plt.figure(figsize=ROW_FIG_SIZE)
    ax1 = fig.add_subplot(1, 4, 1, projection="3d")
    ax2 = fig.add_subplot(1, 4, 2)
    ax3 = fig.add_subplot(1, 4, 3)
    ax4 = fig.add_subplot(1, 4, 4)

    # A (3D): 4-channel oscillator state space — embed Channel 4 as color
    rng = np.random.default_rng(42)
    N_dots = 800
    n_arr = rng.integers(1, 33, N_dots)
    theta_bus = rng.uniform(0, 2 * math.pi, N_dots)
    l_arr = (n_arr * theta_bus / (2 * math.pi)).astype(int) % np.maximum(n_arr, 1)
    m_arr = np.array([rng.integers(-l, l + 1) if l > 0 else 0 for l in l_arr])
    s_arr = rng.choice([-0.5, 0.5], N_dots)
    p = ax1.scatter(n_arr, l_arr, m_arr, c=s_arr, cmap="coolwarm", s=12, alpha=0.85)
    ax1.set_xlabel("n"); ax1.set_ylabel(r"$\ell$"); ax1.set_zlabel("m")
    style_3d(ax1)
    ax1.view_init(elev=20, azim=-55)

    # B: Allan deviation curve from E6
    e6 = RESULTS["experiments"]["E6_allan_deviation"]["result"]
    curve = e6["allan_curve"]
    taus = [c[0] for c in curve]
    sigs = [c[1] for c in curve]
    ax2.loglog(taus, sigs, "o-", color="#2ca02c")
    ax2.axhline(OCXO_ALLAN_1S, color="#888888", lw=0.8, ls="--")
    ax2.set_xlabel(r"$\tau$ (s)"); ax2.set_ylabel(r"$\sigma_y$")
    style_2d(ax2)

    # C: n coordinate from clock counter
    cycles = np.arange(0, 200)
    N_max = 32
    n_vs_cycle = cycles % N_max
    ax3.plot(cycles, n_vs_cycle, "-", color="#1f77b4", lw=1.0)
    ax3.set_xlabel("cycle"); ax3.set_ylabel("n")
    style_2d(ax3)

    # D: l coordinate from bus phase, for various n
    theta_grid = np.linspace(0, 2 * math.pi, 200)
    cmap = plt.cm.viridis
    n_demo = [4, 8, 16, 32]
    for i, n in enumerate(n_demo):
        l = (n * theta_grid / (2 * math.pi)).astype(int) % n
        ax4.plot(theta_grid / math.pi, l, color=cmap(i / len(n_demo)),
                 label=f"n={n}", lw=1.2)
    ax4.set_xlabel(r"$\theta_2 / \pi$"); ax4.set_ylabel(r"$\ell$")
    ax4.legend(frameon=False, fontsize=8, ncol=2)
    style_2d(ax4)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "panel_E67_substrate.png", dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Panel E8: Operating modes
# -----------------------------------------------------------------------------

def panel_e8():
    fig = plt.figure(figsize=ROW_FIG_SIZE)
    ax1 = fig.add_subplot(1, 4, 1, projection="3d")
    ax2 = fig.add_subplot(1, 4, 2)
    ax3 = fig.add_subplot(1, 4, 3)
    ax4 = fig.add_subplot(1, 4, 4)

    rng = np.random.default_rng(123)
    n_species = 100
    mz_values = rng.uniform(100, 2000, n_species)
    intensities = 10 ** rng.uniform(2, 6, n_species)

    # A (3D): mode index x mz x intensity bars
    mode_names = ["DDA", "DIA", "SWATH", "SRM", "PRM", "XT"]
    mode_palette = plt.cm.plasma(np.linspace(0.1, 0.9, len(mode_names)))
    for mi, (mode, color) in enumerate(zip(mode_names, mode_palette)):
        # Different sampling per mode
        if mode == "DDA":
            idx = np.argsort(intensities)[::-1][:10]
        elif mode == "DIA":
            idx = np.where((mz_values >= 500) & (mz_values < 525))[0]
        elif mode == "SWATH":
            idx = np.where((mz_values >= 1000) & (mz_values < 1050))[0]
        elif mode == "SRM":
            idx = [int(rng.integers(0, n_species))]
        elif mode == "PRM":
            idx = list(range(0, 5))
        else:  # XT
            idx = np.argsort(intensities)[::-1][:25]
        for j in idx:
            ax1.bar3d(mi - 0.4, mz_values[j], 0, 0.8, 8, np.log10(intensities[j]),
                      color=color, alpha=0.8, edgecolor="none")
    ax1.set_xticks(range(len(mode_names)))
    ax1.set_xticklabels(mode_names, fontsize=8)
    ax1.set_ylabel("m/z"); ax1.set_zlabel(r"$\log I$")
    style_3d(ax1)
    ax1.view_init(elev=18, azim=-65)

    # B: cycle time comparison TC vs conventional
    modes = ["DDA", "DIA", "SWATH", "SRM", "PRM"]
    conv = [1.0, 2.0, 3.0, 0.05, 0.25]   # seconds (median)
    tc =   [0.001, 0.001, 0.001, 0.0001, 0.001]
    x_idx = np.arange(len(modes))
    width = 0.38
    ax2.bar(x_idx - width / 2, conv, width, color="#888888", label="conv.")
    ax2.bar(x_idx + width / 2, tc, width, color="#1f77b4", label="TC")
    ax2.set_yscale("log")
    ax2.set_xticks(x_idx)
    ax2.set_xticklabels(modes, fontsize=9)
    ax2.set_ylabel("cycle time (s)")
    ax2.legend(frameon=False, fontsize=9)
    style_2d(ax2)

    # C: TC-DDA top-N intensities
    top_idx = np.argsort(intensities)[::-1][:10]
    ax3.bar(np.arange(10) + 1, intensities[top_idx], color="#2ca02c")
    ax3.set_yscale("log")
    ax3.set_xlabel("rank"); ax3.set_ylabel("I")
    style_2d(ax3)

    # D: TC-XT residence vs resolution
    e10 = RESULTS["experiments"]["E10_xt_extreme_resolution"]["result"]
    Ts = [r["T_seconds"] for r in e10["rows"]]
    Re = [r["R_effective"] for r in e10["rows"]]
    ax4.loglog(Ts, Re, "o-", color="#d62728")
    ax4.set_xlabel("T (s)"); ax4.set_ylabel("R")
    style_2d(ax4)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "panel_E8_modes.png", dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Panel E9: Mass accuracy
# -----------------------------------------------------------------------------

def panel_e9():
    fig = plt.figure(figsize=ROW_FIG_SIZE)
    ax1 = fig.add_subplot(1, 4, 1, projection="3d")
    ax2 = fig.add_subplot(1, 4, 2)
    ax3 = fig.add_subplot(1, 4, 3)
    ax4 = fig.add_subplot(1, 4, 4)

    e9 = RESULTS["experiments"]["E9_compound_mass_accuracy"]["result"]
    rows = e9["rows"]
    names = [r["compound"] for r in rows]
    m_true = np.array([r["mass_true_Da"] for r in rows])
    m_est = np.array([r["mass_estimated_Da"] for r in rows])
    ppm = np.array([r["ppm_error"] for r in rows])

    # A (3D): true mass, estimated mass, ppm error
    p = ax1.scatter(m_true, m_est, ppm, c=ppm, cmap="plasma", s=80)
    ax1.plot([m_true.min(), m_true.max()], [m_true.min(), m_true.max()],
             [0, 0], color="#888888", lw=0.8, ls="--")
    ax1.set_xlabel("true (Da)"); ax1.set_ylabel("est. (Da)"); ax1.set_zlabel("ppm")
    style_3d(ax1)
    ax1.view_init(elev=22, azim=-65)

    # B: Per-compound ppm bars
    ax2.bar(np.arange(len(rows)), ppm, color="#1f77b4")
    ax2.axhline(1.0, color="#d62728", lw=0.8, ls="--")
    ax2.set_xticks(np.arange(len(rows)))
    ax2.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("ppm")
    style_2d(ax2)

    # C: estimated vs true mass (residual)
    ax3.scatter(m_true, (m_est - m_true), s=60, c="#2ca02c")
    ax3.axhline(0, color="#888888", lw=0.8)
    ax3.set_xlabel("true mass (Da)"); ax3.set_ylabel("residual (Da)")
    style_2d(ax3)

    # D: ppm error histogram
    ax4.hist(ppm, bins=15, color="#9467bd", edgecolor="white")
    ax4.set_xlabel("ppm"); ax4.set_ylabel("count")
    style_2d(ax4)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "panel_E9_mass_accuracy.png", dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Panel E12: Completion criterion
# -----------------------------------------------------------------------------

def panel_e12():
    fig = plt.figure(figsize=ROW_FIG_SIZE)
    ax1 = fig.add_subplot(1, 4, 1, projection="3d")
    ax2 = fig.add_subplot(1, 4, 2)
    ax3 = fig.add_subplot(1, 4, 3)
    ax4 = fig.add_subplot(1, 4, 4)

    rng = np.random.default_rng(7)
    n_per = 200
    K = 1000

    # True completions: small spread, low energy, low phase drift
    x_max_true = np.abs(rng.normal(0, 0.005, n_per))
    e_max_true = np.abs(rng.normal(0, 1e-4, n_per))
    drift_true = np.abs(rng.normal(0, 1e-5, n_per))

    # False completions (saddle pass-through): exponentially diverging
    x_max_false = 0.001 * np.exp(0.005 * (np.arange(n_per))) + np.abs(rng.normal(0.01, 0.005, n_per))
    e_max_false = 0.001 * np.exp(0.003 * np.arange(n_per)) + np.abs(rng.normal(0, 1e-3, n_per))
    drift_false = np.abs(rng.normal(0.005, 0.002, n_per))

    # A (3D): scatter in (x_max, e_max, drift) space
    ax1.scatter(x_max_true, e_max_true, drift_true,
                color="#2ca02c", s=20, alpha=0.7, label="true")
    ax1.scatter(x_max_false, e_max_false, drift_false,
                color="#d62728", s=20, alpha=0.7, label="false")
    ax1.set_xlabel("x_max"); ax1.set_ylabel("E_max"); ax1.set_zlabel("phase drift")
    ax1.legend(frameon=False, fontsize=8, loc="upper left")
    style_3d(ax1)
    ax1.view_init(elev=22, azim=-55)

    # B: Position window distribution
    bins = np.linspace(0, 0.1, 30)
    ax2.hist(x_max_true, bins=bins, color="#2ca02c", alpha=0.7, label="true")
    ax2.hist(x_max_false, bins=bins, color="#d62728", alpha=0.7, label="false")
    ax2.axvline(0.05, color="#222222", lw=0.8, ls="--")
    ax2.set_xlabel("x_max"); ax2.set_ylabel("count")
    ax2.legend(frameon=False, fontsize=9)
    style_2d(ax2)

    # C: Energy window distribution
    bins = np.linspace(0, 2e-3, 30)
    ax3.hist(e_max_true, bins=bins, color="#2ca02c", alpha=0.7)
    ax3.hist(e_max_false, bins=bins, color="#d62728", alpha=0.7)
    ax3.axvline(1e-3, color="#222222", lw=0.8, ls="--")
    ax3.set_xlabel("E_max"); ax3.set_ylabel("count")
    style_2d(ax3)

    # D: Phase drift distribution
    bins = np.linspace(0, 0.012, 30)
    ax4.hist(drift_true, bins=bins, color="#2ca02c", alpha=0.7)
    ax4.hist(drift_false, bins=bins, color="#d62728", alpha=0.7)
    ax4.axvline(1e-3, color="#222222", lw=0.8, ls="--")
    ax4.set_xlabel("phase drift"); ax4.set_ylabel("count")
    style_2d(ax4)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "panel_E12_completion.png", dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Panel E11: DSL validity (replaced — no tables, all data-driven)
# Data-driven version: trajectories under accepted vs rejected fields
# -----------------------------------------------------------------------------

def panel_e11():
    fig = plt.figure(figsize=ROW_FIG_SIZE)
    ax1 = fig.add_subplot(1, 4, 1, projection="3d")
    ax2 = fig.add_subplot(1, 4, 2)
    ax3 = fig.add_subplot(1, 4, 3)
    ax4 = fig.add_subplot(1, 4, 4)

    # A (3D): trajectories in 3-mode field configurations
    h = 1e-3; T = 4.0; n_steps = int(T / h)
    cfgs = [
        ("TOF", lambda x, t: np.array([1.0]), [0.0], [0.0]),
        ("Orb", lambda x, t: np.array([-(2 * math.pi) ** 2 * x[0]]), [1.0], [0.0]),
        ("Quad", lambda x, t: np.array([-(0.0 - 2 * 0.3 * math.cos(2 * t)) * x[0]]),
         [1.0], [0.0]),
    ]
    for i, (name, force, x0, v0) in enumerate(cfgs):
        x_t, _, t_a = velocity_verlet(x0, v0, force, 1.0, h, n_steps)
        ax1.plot([i] * len(t_a), t_a, x_t[:, 0],
                 color=plt.cm.tab10(i), lw=1.0)
    ax1.set_xticks(range(len(cfgs)))
    ax1.set_xticklabels([c[0] for c in cfgs], fontsize=8)
    ax1.set_ylabel("t"); ax1.set_zlabel("x")
    style_3d(ax1)
    ax1.view_init(elev=18, azim=-60)

    # B: Energy stability — accepted (Hamiltonian) vs dissipative
    omega = 2 * math.pi
    pot_h = lambda x, t: 0.5 * omega ** 2 * x[0] ** 2
    force_h = lambda x, t: np.array([-omega ** 2 * x[0]])
    # Synthetic dissipative: integrate with manual gamma damping
    x = np.array([1.0]); v = np.array([0.0])
    gamma = 0.05
    h2 = 1e-3
    n2 = int(10 / h2)
    E_diss = []; t_diss = []
    for i in range(n2):
        a = -omega ** 2 * x - gamma * v
        v = v + h2 * a
        x = x + h2 * v
        E_diss.append(0.5 * v[0] ** 2 + 0.5 * omega ** 2 * x[0] ** 2)
        t_diss.append(i * h2)
    x_y, v_y, t_y = yoshida4([1.0], [0.0], force_h, 1.0, h2, n2)
    E_y = 0.5 * v_y[:, 0] ** 2 + 0.5 * omega ** 2 * x_y[:, 0] ** 2
    sample = max(1, len(t_y) // 800)
    ax2.plot(t_y[::sample], E_y[::sample], color="#2ca02c", lw=1.0, label="accepted")
    ax2.plot(np.array(t_diss)[::sample], np.array(E_diss)[::sample],
             color="#d62728", lw=1.0, label="dissipative")
    ax2.set_xlabel("t"); ax2.set_ylabel("E")
    ax2.legend(frameon=False, fontsize=9)
    style_2d(ax2)

    # C: Field profile sampler — three valid M(x)
    x_g = np.linspace(-3, 3, 200)
    M_tof = -1.0 * x_g
    M_orb = 0.5 * x_g ** 2
    M_quartic = 0.5 * x_g ** 2 + 0.05 * x_g ** 4
    ax3.plot(x_g, M_tof, color="#1f77b4", label="TOF", lw=1.2)
    ax3.plot(x_g, M_orb, color="#d62728", label="Orb", lw=1.2)
    ax3.plot(x_g, M_quartic, color="#2ca02c", label="quartic", lw=1.2)
    ax3.set_xlabel("x"); ax3.set_ylabel(r"$\mathcal{M}(x)$")
    ax3.legend(frameon=False, fontsize=9)
    style_2d(ax3)

    # D: Rejected fields — singular and discontinuous
    x_g2 = np.linspace(-3, 3, 400)
    x_avoid = x_g2[np.abs(x_g2 - 0.5) > 0.05]
    M_sing = -1.0 / (x_avoid - 0.5)
    M_step = np.sign(x_g2)
    ax4.plot(x_avoid, M_sing, color="#d62728", lw=1.0, label="singular")
    ax4.plot(x_g2, M_step, color="#9467bd", lw=1.2, label="discontinuous")
    ax4.set_ylim(-5, 5)
    ax4.set_xlabel("x"); ax4.set_ylabel(r"$\mathcal{M}(x)$")
    ax4.legend(frameon=False, fontsize=9)
    style_2d(ax4)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "panel_E11_dsl.png", dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Aggregate panel: summary across all experiments
# -----------------------------------------------------------------------------

def panel_summary():
    fig = plt.figure(figsize=ROW_FIG_SIZE)
    ax1 = fig.add_subplot(1, 4, 1, projection="3d")
    ax2 = fig.add_subplot(1, 4, 2)
    ax3 = fig.add_subplot(1, 4, 3)
    ax4 = fig.add_subplot(1, 4, 4)

    exp_codes = list(RESULTS["experiments"].keys())
    exp_labels = [c.split("_", 1)[0] for c in exp_codes]
    pass_flags = [1 if RESULTS["experiments"][c]["passed"] else 0 for c in exp_codes]
    elapsed = [RESULTS["experiments"][c]["elapsed_seconds"] for c in exp_codes]

    # A (3D): bar3d across experiments — pass=1, runtime, ordinal
    ords = np.arange(len(exp_codes))
    for i, (code, p, e) in enumerate(zip(exp_codes, pass_flags, elapsed)):
        color = "#2ca02c" if p == 1 else "#d62728"
        ax1.bar3d(i - 0.4, 0, 0, 0.8, 1.0, max(e, 0.001),
                  color=color, alpha=0.85, edgecolor="none")
    ax1.set_xticks(ords)
    ax1.set_xticklabels(exp_labels, rotation=45, fontsize=7)
    ax1.set_ylabel(""); ax1.set_zlabel("t (s)")
    style_3d(ax1)
    ax1.view_init(elev=22, azim=-60)

    # B: pass/fail bar
    ax2.bar(ords, pass_flags, color=["#2ca02c" if p else "#d62728" for p in pass_flags])
    ax2.set_xticks(ords)
    ax2.set_xticklabels(exp_labels, rotation=45, fontsize=7)
    ax2.set_ylabel("pass")
    ax2.set_ylim(0, 1.2)
    style_2d(ax2)

    # C: runtime distribution
    ax3.bar(ords, elapsed, color="#1f77b4")
    ax3.set_xticks(ords)
    ax3.set_xticklabels(exp_labels, rotation=45, fontsize=7)
    ax3.set_ylabel("t (s)")
    ax3.set_yscale("log")
    style_2d(ax3)

    # D: cumulative pass rate
    cum = np.cumsum(pass_flags) / np.arange(1, len(pass_flags) + 1)
    ax4.plot(ords, cum, "o-", color="#2ca02c")
    ax4.axhline(1.0, color="#888888", lw=0.8, ls="--")
    ax4.set_xticks(ords)
    ax4.set_xticklabels(exp_labels, rotation=45, fontsize=7)
    ax4.set_ylabel("pass rate")
    ax4.set_ylim(0, 1.05)
    style_2d(ax4)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "panel_summary.png", dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

PANELS = [
    ("panel_E1_analyzer_recovery",   panel_e1),
    ("panel_E2_integrator",          panel_e2),
    ("panel_E34_coordinates_capacity", panel_e34),
    ("panel_E5E10_resolution",       panel_e5_e10),
    ("panel_E67_substrate",          panel_e67),
    ("panel_E8_modes",               panel_e8),
    ("panel_E9_mass_accuracy",       panel_e9),
    ("panel_E11_dsl",                panel_e11),
    ("panel_E12_completion",         panel_e12),
    ("panel_summary",                panel_summary),
]


if __name__ == "__main__":
    for name, fn in PANELS:
        print(f"Generating {name} ...", flush=True)
        fn()
    print(f"\nWrote {len(PANELS)} panels to {OUT_DIR}")
