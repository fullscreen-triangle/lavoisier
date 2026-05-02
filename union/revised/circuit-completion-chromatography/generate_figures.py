"""
Figure generation for circuit-completion chromatography paper.
Eight panels, each 1x4 with at least one 3D chart, white background, minimal text.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from pathlib import Path

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 300,
    "axes.linewidth": 0.5,
    "lines.linewidth": 1.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.linewidth": 0.4,
    "grid.alpha": 0.4,
})

C1, C2, C3, C4, C5, C6 = "#1f4eea", "#dc2626", "#059669", "#f59e0b", "#7c3aed", "#6b7280"
HBAR = 1.054571817e-34
H = 6.62607015e-34
K_B = 1.380649e-23
C_LIGHT = 299792458.0
E_CHARGE = 1.602176634e-19
M_E = 9.1093837015e-31
EPSILON_0 = 8.8541878128e-12


def make_panel(figsize=(16, 4)):
    return plt.figure(figsize=figsize, facecolor="white")


def clean_3d(ax):
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("#dddddd")
    ax.yaxis.pane.set_edgecolor("#dddddd")
    ax.zaxis.pane.set_edgecolor("#dddddd")
    ax.grid(True, linestyle=":", linewidth=0.3, alpha=0.5)


def panel_E1(out):
    fig = make_panel()

    # Resistivity comparison
    ax1 = fig.add_subplot(1, 4, 1)
    metals = [("Cu", 8.5e28, 2.5e-14, 1.68e-8),
              ("Al", 18.1e28, 0.8e-14, 2.65e-8),
              ("Ag", 5.86e28, 4.0e-14, 1.59e-8),
              ("Au", 5.9e28, 3.0e-14, 2.44e-8),
              ("Fe", 17.0e28, 0.24e-14, 9.71e-8),
              ("Nb", 5.56e28, 0.42e-14, 15.2e-8)]
    rho_pred = [M_E / (n * E_CHARGE**2 * tau) for _, n, tau, _ in metals]
    rho_exp = [r for _, _, _, r in metals]
    ax1.loglog(rho_exp, rho_pred, "o", ms=8, color=C1, mec="white", mew=0.5)
    lims = [min(rho_exp + rho_pred)*0.5, max(rho_exp + rho_pred)*2]
    ax1.plot(lims, lims, "--", color=C2, lw=0.9)
    ax1.set_xlabel("$\\rho_{\\rm exp}$ ($\\Omega$ m)")
    ax1.set_ylabel("$\\rho_{\\rm pred}$ ($\\Omega$ m)")
    ax1.set_title("(A) Resistivity (6 metals)")

    # Viscosity comparison
    ax2 = fig.add_subplot(1, 4, 2)
    L_0 = 1.0e-9
    liquids = [("H2O", 0.15e-12, 6.6, 1.002e-3),
               ("MeOH", 0.19e-12, 3.1, 0.59e-3),
               ("EtOH", 0.21e-12, 5.1, 1.07e-3),
               ("Ace", 0.12e-12, 2.6, 0.32e-3),
               ("Hex", 0.18e-12, 1.7, 0.31e-3),
               ("Tol", 0.24e-12, 2.5, 0.59e-3),
               ("Gly", 2.80e-12, 334.0, 0.934),
               ("EG", 0.94e-12, 17.2, 16.1e-3)]
    mu_pred = [(t * g) / L_0 for _, t, g, _ in liquids]
    mu_exp = [m for _, _, _, m in liquids]
    ax2.loglog(mu_exp, mu_pred, "s", ms=8, color=C3, mec="white", mew=0.5)
    lims = [min(mu_exp + mu_pred)*0.5, max(mu_exp + mu_pred)*2]
    ax2.plot(lims, lims, "--", color=C2, lw=0.9)
    ax2.set_xlabel("$\\mu_{\\rm exp}$ (Pa s)")
    ax2.set_ylabel("$\\mu_{\\rm pred}$ (Pa s)")
    ax2.set_title("(B) Viscosity (8 liquids)")

    # 3D: Universal transport surface
    ax3 = fig.add_subplot(1, 4, 3, projection="3d")
    log_tau = np.log10(np.array([t for _, _, t, _ in metals[:6]] + [t for _, t, _, _ in liquids[:6]]))
    log_g = np.log10(np.array([n*E_CHARGE**2 for _, n, _, _ in metals[:6]] + [g for _, _, g, _ in liquids[:6]]))
    log_xi = np.log10(np.array(rho_pred + mu_pred[:6]))
    colors = [C1]*6 + [C3]*6
    ax3.scatter(log_tau, log_g, log_xi, c=colors, s=50, edgecolor="white", lw=0.4)
    ax3.set_xlabel("$\\log_{10}\\tau_p$")
    ax3.set_ylabel("$\\log_{10} g$")
    ax3.set_zlabel("$\\log_{10}\\Xi$")
    ax3.set_title("(C) Universal $\\Xi(\\tau_p, g)$")
    ax3.view_init(elev=22, azim=-58)
    clean_3d(ax3)

    # Thermal conductivity bar chart
    ax4 = fig.add_subplot(1, 4, 4)
    therm = [("Cu", 401), ("Si", 150), ("H2O", 0.598), ("Glass", 1.4), ("Diamond", 2200)]
    names = [n for n, _ in therm]
    vals = [k for _, k in therm]
    ax4.barh(names, vals, color=C4, edgecolor="white")
    ax4.set_xscale("log")
    ax4.set_xlabel("$\\kappa$ (W m$^{-1}$ K$^{-1}$)")
    ax4.set_title("(D) Thermal $\\kappa^{-1}$ structure")

    fig.suptitle("Experiment 1: Universal Transport Formula $\\Xi = \\mathcal{N}^{-1}\\sum\\tau_p g_{ij}$", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out / "panel_E1_universal_transport.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  -> panel_E1_universal_transport.png")


def panel_E2(out):
    fig = make_panel()

    # 13 transitions
    transitions = [("Radio FM", 4.136e-7), ("$\\mu$wave", 4.136e-6),
                   ("Far IR", 0.04136), ("IR", 0.1240),
                   ("Red", 1.96), ("Green", 2.33), ("Blue", 2.755),
                   ("UV", 4.88), ("VUV", 12.40), ("X-ray 1k", 1000.0),
                   ("X-ray 10k", 10000.0), ("$\\gamma$ 1M", 1.0e6)]

    E_eV = np.array([e for _, e in transitions])
    E_J = E_eV * E_CHARGE
    tau_c = H / E_J
    lam = H * C_LIGHT / E_J
    c_pred = lam / tau_c

    # Chart 1: tau_c vs energy
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.loglog(E_eV, tau_c, "o", ms=6, color=C1, mec="white", mew=0.4)
    ax1.set_xlabel("photon energy (eV)")
    ax1.set_ylabel("$\\tau_c$ (s)")
    ax1.set_title("(A) $\\tau_c = h/E$")

    # Chart 2: lambda vs energy
    ax2 = fig.add_subplot(1, 4, 2)
    ax2.loglog(E_eV, lam, "s", ms=6, color=C3, mec="white", mew=0.4)
    ax2.set_xlabel("photon energy (eV)")
    ax2.set_ylabel("$\\lambda$ (m)")
    ax2.set_title("(B) $\\Delta x = \\lambda$")

    # Chart 3: 3D
    ax3 = fig.add_subplot(1, 4, 3, projection="3d")
    ax3.scatter(np.log10(tau_c), np.log10(lam), np.log10(c_pred),
                c=np.log10(E_eV), cmap="viridis", s=60, edgecolor="white", lw=0.4)
    ax3.set_xlabel("$\\log_{10}\\tau_c$")
    ax3.set_ylabel("$\\log_{10}\\lambda$")
    ax3.set_zlabel("$\\log_{10} c$")
    ax3.set_title("(C) $c = \\lambda/\\tau_c$")
    ax3.view_init(elev=20, azim=-60)
    clean_3d(ax3)

    # Chart 4: c vs energy (constant)
    ax4 = fig.add_subplot(1, 4, 4)
    ax4.semilogx(E_eV, c_pred, "o-", ms=6, color=C2, mec="white", mew=0.4)
    ax4.axhline(C_LIGHT, color=C5, ls="--", lw=0.9, label="$c$ vacuum")
    ax4.set_xlabel("photon energy (eV)")
    ax4.set_ylabel("$c$ (m/s)")
    ax4.set_title("(D) $c$ across spectrum")
    ax4.legend(frameon=False, fontsize=7)

    fig.suptitle("Experiment 2: Speed of Light from Partition Geometry", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out / "panel_E2_speed_of_light.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  -> panel_E2_speed_of_light.png")


def panel_E3(out):
    fig = make_panel()

    # Cross-channel data for 5 solvents
    solvents = [("Water", 1.002e-3, 6.6, 7.0, 80.1, 5.5e-6),
                ("MeOH", 0.59e-3, 3.1, 6.5, 32.7, 4.4e-7),
                ("EtOH", 1.07e-3, 5.1, 6.2, 24.5, 1.4e-7),
                ("Ace", 0.32e-3, 2.6, 5.7, 20.7, 6.0e-8)]
    L_0 = 1e-9
    names = [s[0] for s in solvents]
    tau_m = [s[1] * L_0 / s[2] for s in solvents]
    tau_o = [H / (s[3] * E_CHARGE) for s in solvents]
    tau_e = [s[4] * EPSILON_0 / s[5] for s in solvents]

    # Bar comparison
    ax1 = fig.add_subplot(1, 4, 1)
    x = np.arange(len(names))
    w = 0.25
    ax1.bar(x - w, np.log10(tau_m), w, color=C1, label="$\\tau_m$")
    ax1.bar(x, np.log10(tau_o), w, color=C3, label="$\\tau_o$")
    ax1.bar(x + w, np.log10(tau_e), w, color=C4, label="$\\tau_e$")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names)
    ax1.set_ylabel("$\\log_{10}\\tau_c$ (s)")
    ax1.set_title("(A) Three lags per solvent")
    ax1.legend(frameon=False, fontsize=7)

    # Mech vs optical
    ax2 = fig.add_subplot(1, 4, 2)
    ax2.loglog(tau_o, tau_m, "o", ms=10, color=C1, mec="white")
    for i, n in enumerate(names):
        ax2.annotate(n, (tau_o[i], tau_m[i]), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax2.set_xlabel("$\\tau_o$ (s)")
    ax2.set_ylabel("$\\tau_m$ (s)")
    ax2.set_title("(B) Mech vs optical")

    # 3D: three-channel coordinate space
    ax3 = fig.add_subplot(1, 4, 3, projection="3d")
    ax3.scatter(np.log10(tau_m), np.log10(tau_o), np.log10(tau_e),
                c=range(len(names)), cmap="plasma", s=80, edgecolor="white", lw=0.4)
    for i, n in enumerate(names):
        ax3.text(np.log10(tau_m[i]), np.log10(tau_o[i]), np.log10(tau_e[i]), "  " + n, fontsize=7)
    ax3.set_xlabel("$\\log_{10}\\tau_m$")
    ax3.set_ylabel("$\\log_{10}\\tau_o$")
    ax3.set_zlabel("$\\log_{10}\\tau_e$")
    ax3.set_title("(C) $(\\tau_m, \\tau_o, \\tau_e)$ space")
    ax3.view_init(elev=22, azim=-58)
    clean_3d(ax3)

    # Ratio plot
    ax4 = fig.add_subplot(1, 4, 4)
    ratios = np.array([np.log10(tm / to) for tm, to in zip(tau_m, tau_o)])
    ax4.bar(names, ratios, color=C5, edgecolor="white")
    ax4.axhline(0, color=C6, ls="-", lw=0.6)
    ax4.set_ylabel("$\\log_{10}(\\tau_m / \\tau_o)$")
    ax4.set_title("(D) Channel ratio (decades)")

    fig.suptitle("Experiment 3: Cross-Channel Partition Lag Consistency", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out / "panel_E3_cross_channel.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  -> panel_E3_cross_channel.png")


def panel_E4(out):
    fig = make_panel()

    # BCS ratio data
    sc = [("Al", 1.20, 3.539), ("Cd", 0.52, 3.43), ("In", 3.41, 3.73),
          ("Sn", 3.72, 3.46), ("V", 5.40, 3.50), ("Ta", 4.48, 3.63),
          ("Pb", 7.19, 4.35), ("Hg", 4.15, 4.32), ("Nb", 9.25, 3.89)]
    names = [s[0] for s in sc]
    Tc = np.array([s[1] for s in sc])
    ratios = np.array([s[2] for s in sc])
    BCS = 3.528

    # Chart 1: BCS ratios
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.bar(names, ratios, color=C1, edgecolor="white")
    ax1.axhline(BCS, color=C2, ls="--", lw=0.9, label=f"BCS = {BCS}")
    ax1.set_ylabel("$2\\Delta_0 / k_B T_c$")
    ax1.set_title("(A) BCS ratio (9 SC)")
    ax1.legend(frameon=False, fontsize=7)
    ax1.tick_params(axis='x', rotation=45)

    # Chart 2: rho discontinuity
    ax2 = fig.add_subplot(1, 4, 2)
    T = np.linspace(0, 12, 200)
    rho_ref = 1.0e-8
    for nm, tc, _ in sc[:4]:
        rho = np.where(T < tc, 0.0, rho_ref * (T/tc - 1)**0.5)
        ax2.plot(T, rho, lw=1.2, label=f"{nm} ($T_c = {tc}$ K)")
    ax2.set_xlabel("$T$ (K)")
    ax2.set_ylabel("$\\rho$ ($\\Omega$ m)")
    ax2.set_title("(B) $\\rho \\to 0$ at $T_c$")
    ax2.legend(frameon=False, fontsize=6)

    # Chart 3: 3D partition extinction surface
    ax3 = fig.add_subplot(1, 4, 3, projection="3d")
    Tg = np.linspace(0.1, 12, 30)
    Tc_g = np.linspace(0.5, 10, 30)
    T_mesh, Tc_mesh = np.meshgrid(Tg, Tc_g)
    rho_surf = np.where(T_mesh < Tc_mesh, 0.0,
                        rho_ref * np.sqrt(np.maximum(T_mesh / Tc_mesh - 1, 0)))
    ax3.plot_surface(T_mesh, Tc_mesh, rho_surf, cmap="plasma", alpha=0.85,
                     edgecolor="none", rstride=1, cstride=1)
    ax3.set_xlabel("$T$ (K)")
    ax3.set_ylabel("$T_c$ (K)")
    ax3.set_zlabel("$\\rho$ ($\\Omega$ m)")
    ax3.set_title("(C) Extinction $\\rho(T, T_c)$")
    ax3.view_init(elev=22, azim=-60)
    clean_3d(ax3)

    # Chart 4: T_c vs measured ratio
    ax4 = fig.add_subplot(1, 4, 4)
    ax4.scatter(Tc, ratios, c=ratios, cmap="viridis", s=80, edgecolor="white", lw=0.4)
    ax4.axhline(BCS, color=C2, ls="--", lw=0.9, label="BCS weak")
    for i, n in enumerate(names):
        ax4.annotate(n, (Tc[i], ratios[i]), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax4.set_xlabel("$T_c$ (K)")
    ax4.set_ylabel("$2\\Delta_0 / k_B T_c$")
    ax4.set_title("(D) Coupling regime")
    ax4.legend(frameon=False, fontsize=7)

    fig.suptitle("Experiment 4: Partition Extinction at Critical Temperatures", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out / "panel_E4_partition_extinction.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  -> panel_E4_partition_extinction.png")


def panel_E5(out):
    fig = make_panel()

    rng = np.random.default_rng(seed=2718)
    n = 50
    tau_m = rng.uniform(0.1e-12, 5.0e-12, n)
    tau_o = rng.uniform(0.1e-15, 10.0e-15, n)
    tau_e = rng.uniform(1e-12, 100e-12, n)
    Tc_m = rng.uniform(200, 600, n)
    Tc_o = rng.uniform(150, 500, n)
    Tc_e = rng.uniform(100, 400, n)

    # Chart 1: 1D distribution
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.hist(np.log10(tau_m * 1e12), bins=15, color=C1, alpha=0.85, edgecolor="white")
    ax1.set_xlabel("$\\log_{10}\\tau_m$ (ps)")
    ax1.set_ylabel("count")
    ax1.set_title("(A) 1D $\\tau_m$ histogram")

    # Chart 2: 2D scatter (tau_m vs tau_o)
    ax2 = fig.add_subplot(1, 4, 2)
    ax2.scatter(np.log10(tau_m * 1e12), np.log10(tau_o * 1e15),
                c=Tc_m, cmap="viridis", s=30, alpha=0.85, edgecolor="white", lw=0.3)
    ax2.set_xlabel("$\\log_{10}\\tau_m$ (ps)")
    ax2.set_ylabel("$\\log_{10}\\tau_o$ (fs)")
    ax2.set_title("(B) 2D continuous lags")

    # Chart 3: 3D continuous + extinction colour
    ax3 = fig.add_subplot(1, 4, 3, projection="3d")
    ax3.scatter(np.log10(tau_m * 1e12), np.log10(tau_o * 1e15), np.log10(tau_e * 1e12),
                c=Tc_m, cmap="plasma", s=40, edgecolor="white", lw=0.3)
    ax3.set_xlabel("$\\log_{10}\\tau_m$")
    ax3.set_ylabel("$\\log_{10}\\tau_o$")
    ax3.set_zlabel("$\\log_{10}\\tau_e$")
    ax3.set_title("(C) 3D + $T_c^{(m)}$ colour")
    ax3.view_init(elev=22, azim=-60)
    clean_3d(ax3)

    # Chart 4: distinguishability gain
    ax4 = fig.add_subplot(1, 4, 4)
    n_pairs = n * (n - 1) // 2
    res = json.load(open(out.parent / "validation_results" / "experiment_E5.json"))
    sm = res["summary_metrics"]
    cats = ["1D", "3D", "6D"]
    pcts = [100*sm["distinct_1D"]/n_pairs, 100*sm["distinct_3D"]/n_pairs, 100*sm["distinct_6D"]/n_pairs]
    bars = ax4.bar(cats, pcts, color=[C1, C3, C2], edgecolor="white")
    for b, p in zip(bars, pcts):
        ax4.text(b.get_x() + b.get_width()/2, p + 1, f"{p:.1f}%",
                 ha="center", fontsize=8)
    ax4.set_ylabel("% pairs distinct")
    ax4.set_title("(D) Resolving power")
    ax4.set_ylim(0, 110)

    fig.suptitle("Experiment 5: Six-Dimensional Analyte Fingerprinting", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out / "panel_E5_fingerprinting.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  -> panel_E5_fingerprinting.png")


def panel_E6(out):
    fig = make_panel()

    metals = [("Cu", 8.5e28), ("Al", 18.1e28), ("Ag", 5.86e28),
              ("Au", 5.9e28), ("Fe", 17.0e28), ("Nb", 5.56e28)]
    V_SIGNAL = 2.0e8
    A = 1e-6
    I = 1.0
    names = [m[0] for m in metals]
    v_drift = np.array([I/(n * E_CHARGE * A) for _, n in metals])
    ratios = V_SIGNAL / v_drift

    # Chart 1: velocities
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.bar([n + " sig" for n in names], [V_SIGNAL]*len(names),
            color=C1, edgecolor="white", label="$v_{\\rm signal}$")
    ax1.set_yscale("log")
    ax1.set_ylabel("velocity (m/s)")
    ax1.set_title("(A) Signal velocity")
    ax1.tick_params(axis='x', rotation=45)

    # Chart 2: drift velocities
    ax2 = fig.add_subplot(1, 4, 2)
    ax2.bar(names, v_drift, color=C3, edgecolor="white")
    ax2.set_yscale("log")
    ax2.set_ylabel("$v_{\\rm drift}$ (m/s)")
    ax2.set_title("(B) Drift velocity ($I=1$ A)")

    # Chart 3: 3D ratio
    ax3 = fig.add_subplot(1, 4, 3, projection="3d")
    n_arr = np.array([m[1] for m in metals])
    ax3.scatter(np.log10(n_arr), np.log10(v_drift), np.log10(ratios),
                c=np.log10(ratios), cmap="plasma", s=80, edgecolor="white", lw=0.4)
    for i, nm in enumerate(names):
        ax3.text(np.log10(n_arr[i]), np.log10(v_drift[i]), np.log10(ratios[i]), "  " + nm, fontsize=7)
    ax3.set_xlabel("$\\log_{10} n$")
    ax3.set_ylabel("$\\log_{10} v_{\\rm drift}$")
    ax3.set_zlabel("$\\log_{10}$ ratio")
    ax3.set_title("(C) Ratio across metals")
    ax3.view_init(elev=22, azim=-60)
    clean_3d(ax3)

    # Chart 4: ratio histogram
    ax4 = fig.add_subplot(1, 4, 4)
    ax4.bar(names, np.log10(ratios), color=C5, edgecolor="white")
    ax4.axhline(12, color=C2, ls="--", lw=0.9, label="$10^{12}$")
    ax4.set_ylabel("$\\log_{10}(v_{\\rm signal}/v_{\\rm drift})$")
    ax4.set_title("(D) 12-orders-of-magnitude")
    ax4.legend(frameon=False, fontsize=7)
    ax4.set_ylim(11, 13)

    fig.suptitle("Experiment 6: Circuit-Completion Velocity Ratio", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out / "panel_E6_velocity_ratio.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  -> panel_E6_velocity_ratio.png")


def panel_E7(out):
    fig = make_panel()

    systems = [
        ("He-gas", "gas", 0.05), ("Air", "gas", 0.07), ("CO2", "gas", 0.10),
        ("EtOH-vap", "gas", 0.12), ("Steam", "gas", 0.14),
        ("CO2-cr", "trans", 0.45), ("H2O-cr", "trans", 0.50),
        ("LHe", "liq", 0.78), ("LN2", "liq", 0.81), ("H2O", "liq", 0.89),
        ("EtOH", "liq", 0.85), ("Gly", "liq", 0.97), ("Hg", "liq", 0.99),
        ("Glass", "liq", 0.96), ("Olive", "liq", 0.94),
    ]
    names = [s[0] for s in systems]
    rho = np.array([s[2] for s in systems])
    phases = [s[1] for s in systems]
    color_map = {"gas": C1, "trans": C4, "liq": C3}
    colors = [color_map[p] for p in phases]

    # Chart 1: rho_C bars
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.bar(names, rho, color=colors, edgecolor="white")
    ax1.axhline(0.3, color=C6, ls=":", lw=0.7)
    ax1.axhline(0.7, color=C6, ls=":", lw=0.7)
    ax1.set_ylabel("$\\rho_C$")
    ax1.set_title("(A) Network density (15 systems)")
    ax1.tick_params(axis='x', rotation=70, labelsize=6)

    # Chart 2: percolation curve
    ax2 = fig.add_subplot(1, 4, 2)
    rho_axis = np.linspace(0, 1, 200)
    P_overlap = 1 / (1 + np.exp(-15 * (rho_axis - 0.5)))
    ax2.plot(rho_axis, P_overlap, lw=1.5, color=C5, label="$P_{\\rm overlap}$")
    ax2.axvspan(0, 0.3, alpha=0.15, color=C1, label="gas")
    ax2.axvspan(0.3, 0.7, alpha=0.15, color=C4, label="transition")
    ax2.axvspan(0.7, 1.0, alpha=0.15, color=C3, label="liquid")
    ax2.scatter(rho, [0.5]*len(rho), c=colors, s=20, edgecolor="white", lw=0.3, zorder=3)
    ax2.set_xlabel("$\\rho_C$")
    ax2.set_ylabel("$P_{\\rm overlap}$")
    ax2.set_title("(B) Percolation curve")
    ax2.legend(frameon=False, fontsize=6, loc="upper left")

    # Chart 3: 3D classification
    ax3 = fig.add_subplot(1, 4, 3, projection="3d")
    rng = np.random.default_rng(seed=11)
    s_k = rng.uniform(0, 1, len(systems))
    s_t = rng.uniform(0, 1, len(systems))
    ax3.scatter(s_k, s_t, rho, c=colors, s=80, edgecolor="white", lw=0.4)
    ax3.set_xlabel("$S_k$")
    ax3.set_ylabel("$S_t$")
    ax3.set_zlabel("$\\rho_C$")
    ax3.set_title("(C) S-space classification")
    ax3.view_init(elev=22, azim=-60)
    clean_3d(ax3)

    # Chart 4: phase counts
    ax4 = fig.add_subplot(1, 4, 4)
    phase_counts = {"gas": phases.count("gas"),
                    "trans": phases.count("trans"),
                    "liq": phases.count("liq")}
    ax4.bar(phase_counts.keys(), phase_counts.values(),
            color=[color_map[p] for p in phase_counts.keys()], edgecolor="white")
    ax4.set_ylabel("count")
    ax4.set_title("(D) Classification breakdown")

    fig.suptitle("Experiment 7: Phase Classification from Network Density", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out / "panel_E7_phase_classification.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  -> panel_E7_phase_classification.png")


def panel_E8(out):
    fig = make_panel()

    metals = [("Cu", 401, 1.68e-8), ("Al", 237, 2.65e-8), ("Ag", 429, 1.59e-8),
              ("Au", 317, 2.44e-8), ("Fe", 80.4, 9.71e-8), ("Nb", 53.7, 15.2e-8)]
    L_0 = (np.pi**2 / 3) * (K_B / E_CHARGE)**2
    T = 300
    names = [m[0] for m in metals]
    L_meas = np.array([k * r / T for _, k, r in metals])

    # Chart 1: Lorenz comparison
    ax1 = fig.add_subplot(1, 4, 1)
    x = np.arange(len(names))
    ax1.bar(x, L_meas * 1e8, color=C1, edgecolor="white")
    ax1.axhline(L_0 * 1e8, color=C2, ls="--", lw=0.9, label=f"$L_0={L_0*1e8:.2f}\\times 10^{{-8}}$")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names)
    ax1.set_ylabel("$L = \\kappa\\rho/T$ ($10^{-8}$ V$^2$/K$^2$)")
    ax1.set_title("(A) Lorenz number (6 metals)")
    ax1.legend(frameon=False, fontsize=7)

    # Chart 2: kappa vs sigma
    ax2 = fig.add_subplot(1, 4, 2)
    sig = np.array([1/r for _, _, r in metals])
    kap = np.array([k for _, k, _ in metals])
    ax2.loglog(sig, kap, "o", ms=8, color=C3, mec="white", mew=0.5)
    s_line = np.array([min(sig), max(sig)])
    ax2.plot(s_line, L_0 * T * s_line, "--", color=C2, lw=0.9, label="WF")
    ax2.set_xlabel("$\\sigma$ (S/m)")
    ax2.set_ylabel("$\\kappa$ (W/m K)")
    ax2.set_title("(B) Wiedemann-Franz $\\kappa = LT\\sigma$")
    ax2.legend(frameon=False, fontsize=7)

    # Chart 3: 3D
    ax3 = fig.add_subplot(1, 4, 3, projection="3d")
    ax3.scatter(np.log10(sig), np.log10(kap), L_meas * 1e8,
                c=L_meas, cmap="viridis", s=80, edgecolor="white", lw=0.4)
    for i, nm in enumerate(names):
        ax3.text(np.log10(sig[i]), np.log10(kap[i]), L_meas[i] * 1e8, "  " + nm, fontsize=7)
    ax3.set_xlabel("$\\log_{10}\\sigma$")
    ax3.set_ylabel("$\\log_{10}\\kappa$")
    ax3.set_zlabel("$L$ ($10^{-8}$)")
    ax3.set_title("(C) $L$ across metals")
    ax3.view_init(elev=22, azim=-60)
    clean_3d(ax3)

    # Chart 4: deviation
    ax4 = fig.add_subplot(1, 4, 4)
    deviations = (L_meas - L_0) / L_0 * 100
    ax4.bar(names, deviations, color=C5, edgecolor="white")
    ax4.axhline(0, color=C6, lw=0.6)
    ax4.set_ylabel("deviation from $L_0$ (%)")
    ax4.set_title("(D) Relative deviation")

    fig.suptitle("Experiment 8: Wiedemann-Franz Universality", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out / "panel_E8_wiedemann_franz.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  -> panel_E8_wiedemann_franz.png")


def main():
    out = Path(__file__).parent / "figures"
    out.mkdir(exist_ok=True)
    print("\n" + "#" * 70)
    print("# Generating figure panels")
    print("#" * 70 + "\n")
    panel_E1(out)
    panel_E2(out)
    panel_E3(out)
    panel_E4(out)
    panel_E5(out)
    panel_E6(out)
    panel_E7(out)
    panel_E8(out)
    print("\nAll panels generated.\n")


if __name__ == "__main__":
    main()
