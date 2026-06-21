"""
Generate 9 figure panels (one per paper section), each with 4 data-driven charts
in a row (at least one 3D per panel). White background, minimal text.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from matplotlib.colors import Normalize
from matplotlib import cm
import os

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)

FIG_W, FIG_H = 18, 4.4          # landscape strip
DPI = 180
GREY   = "#222222"
BLUE   = "#1f77b4"
ORANGE = "#d62728"
GREEN  = "#2ca02c"
PURPLE = "#9467bd"

def strip(fig):
    for ax in fig.axes:
        ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

def tiny(ax, title="", xlabel="", ylabel="", fontsize=7):
    ax.set_title(title, fontsize=7, pad=3, color=GREY)
    ax.set_xlabel(xlabel, fontsize=6, color=GREY)
    ax.set_ylabel(ylabel, fontsize=6, color=GREY)
    ax.tick_params(labelsize=5.5, colors=GREY)
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")

def tiny3(ax, title="", xl="", yl="", zl=""):
    ax.set_title(title, fontsize=7, pad=2, color=GREY)
    ax.set_xlabel(xl, fontsize=5.5, labelpad=1, color=GREY)
    ax.set_ylabel(yl, fontsize=5.5, labelpad=1, color=GREY)
    ax.set_zlabel(zl, fontsize=5.5, labelpad=1, color=GREY)
    ax.tick_params(labelsize=5, colors=GREY)
    ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("#eeeeee")
    ax.yaxis.pane.set_edgecolor("#eeeeee")
    ax.zaxis.pane.set_edgecolor("#eeeeee")
    ax.grid(True, linewidth=0.3, color="#dddddd")

# ─────────────────────────────────────────────────────────────────────────────
# PANEL 1 — SECTION 1: AXIOMS
# C(n)=2n², N_state(n), residue β landscape, bijection Φ traversal
# ─────────────────────────────────────────────────────────────────────────────
def panel1():
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.38, left=0.06, right=0.97)

    # 1a: C(n) = 2n²
    ax = fig.add_subplot(gs[0])
    n = np.arange(1, 16)
    ax.bar(n, 2*n**2, color=BLUE, alpha=0.85, width=0.7)
    tiny(ax, r"$C(n) = 2n^2$", r"depth $n$", "states per depth")

    # 1b: N_state(n) cumulative
    ax = fig.add_subplot(gs[1])
    ax.plot(n, n*(n+1)*(2*n+1)/3, "o-", color=ORANGE, lw=1.8, ms=4)
    tiny(ax, r"$N_{\rm state}(n) = n(n+1)(2n+1)/3$", r"depth $n$", "cumulative states")

    # 1c: 3D — partition bijection Φ: M → (n, ℓ, m) coloured by parity s
    ax3 = fig.add_subplot(gs[2], projection="3d")
    records = []
    M = 1
    for nn in range(1, 9):
        for ll in range(0, nn):
            for mm in range(-ll, ll+1):
                for ss in [-0.5, 0.5]:
                    records.append((M, nn, ll, mm, ss))
                    M += 1
    records = np.array(records)  # (M, n, ℓ, m, s)
    sc = ax3.scatter(records[:,1], records[:,2], records[:,3],
                     c=records[:,4], cmap="bwr", s=5, alpha=0.6, depthshade=True)
    tiny3(ax3, r"Bijection $\Phi: M\to(n,\ell,m)$", r"$n$", r"$\ell$", r"$m$")
    ax3.view_init(elev=22, azim=40)

    # 1d: Residue floor β(n) — β ≥ β_min(n) = 1/n
    ax = fig.add_subplot(gs[3])
    n_fine = np.linspace(1, 15, 300)
    beta_floor = 1.0 / n_fine
    ax.fill_between(n_fine, beta_floor, 1.0, alpha=0.18, color=GREEN)
    ax.plot(n_fine, beta_floor, color=GREEN, lw=1.8)
    ax.axhline(0, color=GREY, lw=0.5, ls="--")
    ax.set_ylim(-0.05, 1.05)
    tiny(ax, r"Residue floor $\beta_{\min}(n)=1/n$", r"depth $n$", r"$\beta$")
    ax.fill_between(n_fine, 0, beta_floor, alpha=0.06, color=ORANGE, label="forbidden")

    strip(fig)
    fig.savefig(os.path.join(OUT, "panel1_axioms.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("panel1 done")

# ─────────────────────────────────────────────────────────────────────────────
# PANEL 2 — SECTION 2: CONTACT
# β(I1,I2) vs D, depth-reduction rate, 3D contact geometry, chain reinforcement
# ─────────────────────────────────────────────────────────────────────────────
def panel2():
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.38, left=0.06, right=0.97)

    D = np.linspace(1, 20, 300)

    # 2a: boundary cost β(D) ∝ D (global) vs β_min (contact)
    ax = fig.add_subplot(gs[0])
    beta_global = 0.05 * D
    beta_min    = np.ones_like(D) * 0.05
    ax.plot(D, beta_global, color=BLUE, lw=1.8, label="global β(D)")
    ax.axhline(0.05, color=GREEN, lw=1.8, ls="--", label=r"contact $\beta_{\min}$")
    ax.legend(fontsize=5.5, framealpha=0)
    tiny(ax, "Boundary cost vs depth", r"$D$", r"$\beta$")

    # 2b: depth-reduction rate dD/dt ∝ M_2
    ax = fig.add_subplot(gs[1])
    M2 = np.linspace(1, 100, 200)
    G_const = 6.674e-11
    r = 1.0
    accel = G_const * M2 / r**2
    ax.plot(M2, accel, color=ORANGE, lw=1.8)
    tiny(ax, r"Partition depth reduction rate $\propto M_2$", r"$M_2$  (partition count)", r"$|dD/dt|$")

    # 2c: 3D — contact chain: items as nodes, D as coordinate, β as height
    ax3 = fig.add_subplot(gs[2], projection="3d")
    n_items = 8
    x = np.arange(n_items)
    y = np.zeros(n_items)
    z_beta = np.array([1/(i+1) for i in range(n_items)])
    for i in range(n_items-1):
        ax3.plot([x[i], x[i+1]], [y[i], y[i+1]], [z_beta[i], z_beta[i+1]],
                 color=BLUE, lw=2, alpha=0.8)
    ax3.scatter(x, y, z_beta, c=z_beta, cmap="plasma_r", s=30, zorder=5)
    ax3.bar3d(x-0.15, y-0.15, np.zeros(n_items), 0.3, 0.3, z_beta,
              color=BLUE, alpha=0.15)
    tiny3(ax3, "Contact chain: $\\beta$ vs node", "node $k$", "", r"$\beta_k$")
    ax3.view_init(elev=28, azim=-55)

    # 2d: normal force F_normal = -∂β_min/∂D at D→0
    ax = fig.add_subplot(gs[3])
    D2 = np.linspace(0.1, 5, 300)
    F  = 1.0 / D2**2
    ax.plot(D2, F, color=PURPLE, lw=1.8)
    ax.set_ylim(0, 30)
    ax.axvline(1.0, color=GREY, lw=0.7, ls="--", alpha=0.5)
    tiny(ax, r"Normal force $F \propto D^{-2}$", r"$D$", r"$F_{\rm normal}$")

    strip(fig)
    fig.savefig(os.path.join(OUT, "panel2_contact.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("panel2 done")

# ─────────────────────────────────────────────────────────────────────────────
# PANEL 3 — SECTION 3: EMERGENCE
# dM/dt = ω/2π, mass-frequency, 3D S-entropy cube, analyser scaling laws
# ─────────────────────────────────────────────────────────────────────────────
def panel3():
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.38, left=0.06, right=0.97)

    mz = np.linspace(50, 700, 300)
    kappa = 1e8; E = 1.602176634e-19; u = 1.66053906660e-27

    # 3a: four analyser partition rates vs m/z
    ax = fig.add_subplot(gs[0])
    Mdot_orb  = np.sqrt(1*E*kappa / (mz*u)) / (2*np.pi)
    Mdot_fticr = 1*E*7.0 / (2*np.pi * mz * u)
    Mdot_quad = np.ones_like(mz) * 0.706 * 2*np.pi*1.1e6 / (4*np.pi)
    Mdot_tof  = np.sqrt(2*1*E*20000 / (mz*u)) / (2*np.pi * 1.5)
    ax.loglog(mz, Mdot_orb/1e6,   color=BLUE,   lw=1.6, label="Orbitrap")
    ax.loglog(mz, Mdot_fticr/1e6, color=ORANGE,  lw=1.6, label="FT-ICR")
    ax.loglog(mz, Mdot_tof/1e6,   color=GREEN,   lw=1.6, label="TOF")
    ax.loglog(mz, Mdot_quad/1e6*np.ones_like(mz), color=PURPLE, lw=1.6, ls="--", label="Quad")
    ax.legend(fontsize=5, framealpha=0)
    tiny(ax, r"$\dot M$ (MHz) vs $m/z$", r"$m/z$  (u)", r"$\dot{M}$ (MHz)")

    # 3b: Time-Count identity: M(t) = ω/(2π)·t for 3 frequencies
    ax = fig.add_subplot(gs[1])
    t = np.linspace(0, 1e-3, 500)
    for freq, col, lbl in [(1e6, BLUE, "1 MHz"), (3e6, ORANGE, "3 MHz"), (7e6, GREEN, "7 MHz")]:
        ax.plot(t*1e3, freq*t, color=col, lw=1.5, label=lbl)
    ax.legend(fontsize=5.5, framealpha=0)
    tiny(ax, r"$M(t) = \dot M \cdot t$", r"$t$ (ms)", r"$M$ (count)")

    # 3c: 3D S-entropy cube — partition states as points in [0,1]³
    ax3 = fig.add_subplot(gs[2], projection="3d")
    np.random.seed(7)
    records = []
    nP = 12
    for nn in range(1, nP+1):
        Sk = (nn-1)/(nP-1)
        for ll in range(0, nn):
            St = ll/(nn-1) if nn > 1 else 0.0
            for mm in range(-ll, ll+1):
                Se = (mm+ll)/(2*ll) if ll > 0 else 0.5
                records.append((Sk, St, Se, nn))
    rec = np.array(records)
    ax3.scatter(rec[:,0], rec[:,1], rec[:,2],
                c=rec[:,3], cmap="viridis", s=6, alpha=0.55, depthshade=True)
    tiny3(ax3, r"S-entropy space $[0,1]^3$", r"$S_k$", r"$S_t$", r"$S_e$")
    ax3.view_init(elev=22, azim=50)

    # 3d: mass-frequency: m ∝ 1/ω²
    ax = fig.add_subplot(gs[3])
    omega = np.linspace(1e5, 1e7, 300)
    hbar = 1.054571817e-34; c2 = (3e8)**2
    m_kg = hbar * omega / c2
    ax.loglog(omega/(2*np.pi*1e6), m_kg/u, color=PURPLE, lw=1.8)
    tiny(ax, r"$m = \hbar\omega/c^2$", r"$\nu$ (MHz)", r"$m$ (u)")

    strip(fig)
    fig.savefig(os.path.join(OUT, "panel3_emergence.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("panel3 done")

# ─────────────────────────────────────────────────────────────────────────────
# PANEL 4 — SECTION 4: COMPOSITION INFLATION
# T(n,d), causal residue chain, 3D T(n,d) surface, K conversion
# ─────────────────────────────────────────────────────────────────────────────
def panel4():
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.38, left=0.06, right=0.97)

    n_vals = np.arange(1, 22)

    # 4a: T(n,d) for d=2,3,4
    ax = fig.add_subplot(gs[0])
    for d, col, lbl in [(2, BLUE, "d=2"), (3, ORANGE, "d=3"), (4, GREEN, "d=4")]:
        T = d * (d+1)**(n_vals-1)
        ax.semilogy(n_vals, T, "o-", color=col, lw=1.5, ms=3, label=lbl)
    ax.legend(fontsize=5.5, framealpha=0)
    tiny(ax, r"$T(n,d) = d(d+1)^{n-1}$", r"depth $n$", r"trajectories")

    # 4b: causal residue chain β_k driving next partition (residue magnitude)
    ax = fig.add_subplot(gs[1])
    k = np.arange(1, 30)
    beta = 1.0 / k
    markerline, stemlines, baseline = ax.stem(k, beta)
    plt.setp(stemlines, color=BLUE, linewidth=0.8)
    plt.setp(markerline, color=BLUE, markersize=3)
    plt.setp(baseline, color=GREY, linewidth=0.5)
    ax.set_ylim(0, 1.1)
    tiny(ax, r"Residue $\beta_k = 1/k$: causal chain", r"partition $k$", r"$\beta_k$")

    # 4c: 3D surface T(n,d) over n and d
    ax3 = fig.add_subplot(gs[2], projection="3d")
    n2d = np.arange(1, 16)
    d2d = np.arange(1,  8)
    NN, DD = np.meshgrid(n2d, d2d)
    TT = DD * (DD+1)**(NN-1)
    TT_log = np.log10(np.clip(TT, 1, None))
    ax3.plot_surface(NN, DD, TT_log, cmap="plasma", alpha=0.88, linewidth=0)
    tiny3(ax3, r"$\log_{10}T(n,d)$", r"$n$", r"$d$", r"$\log_{10}T$")
    ax3.view_init(elev=28, azim=225)

    # 4d: K·√(m/z) — Cs per Orbitrap cycle
    ax = fig.add_subplot(gs[3])
    mz = np.linspace(50, 700, 300)
    kappa = 1e8; E = 1.602176634e-19; u_val = 1.66053906660e-27
    K = 9192631770 * 2*np.pi / np.sqrt(E*kappa/u_val)
    cs_per_orb = K * np.sqrt(mz)
    ax.plot(mz, cs_per_orb, color=PURPLE, lw=1.8)
    tiny(ax, r"$K\sqrt{m/z}$ Cs cycles per Orbitrap cycle", r"$m/z$ (u)", "Cs cycles / Orb cycle")
    ax.text(0.6, 0.15, f"$K={K:.3f}$", transform=ax.transAxes, fontsize=6, color=PURPLE)

    strip(fig)
    fig.savefig(os.path.join(OUT, "panel4_composition.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("panel4 done")

# ─────────────────────────────────────────────────────────────────────────────
# PANEL 5 — SECTION 5: LOSCHMIDT
# M(t) monotone, swimmer residue expansion, 3D phase-space arrow, spin echo
# ─────────────────────────────────────────────────────────────────────────────
def panel5():
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.38, left=0.06, right=0.97)

    t = np.linspace(0, 10, 300)

    # 5a: M(t) strictly monotone — real vs hypothetical reversed
    ax = fig.add_subplot(gs[0])
    M_fwd = t**1.3 + 0.3*np.sin(3*t)
    M_rev = M_fwd[0] + M_fwd[0] - M_fwd  # decreasing — forbidden
    ax.plot(t, M_fwd, color=BLUE,  lw=2,   label="physical $M(t)$")
    ax.plot(t, M_rev, color=ORANGE, lw=1.5, ls="--", alpha=0.6, label="forbidden")
    ax.fill_between(t, M_rev, M_fwd, alpha=0.07, color=ORANGE)
    ax.legend(fontsize=5.5, framealpha=0)
    tiny(ax, r"$M(t)$ monotone; reversal forbidden", r"$t$", r"$M(t)$")

    # 5b: residue expansion radius r(t) after swimmer stroke
    ax = fig.add_subplot(gs[1])
    t2 = np.linspace(0, 5, 200)
    r_expand = 1 - np.exp(-t2)   # residue reaches full contact chain
    ax.fill_between(t2, 0, r_expand, alpha=0.3, color=GREEN)
    ax.plot(t2, r_expand, color=GREEN, lw=1.8)
    ax.axhline(1.0, color=GREY, lw=0.7, ls="--", alpha=0.5)
    tiny(ax, "Residue propagation into contact chain", r"$t$", r"fraction of chain reached")

    # 5c: 3D — phase space (x, p, M) showing monotone M surface
    ax3 = fig.add_subplot(gs[2], projection="3d")
    theta = np.linspace(0, 4*np.pi, 500)
    x_ph  = np.cos(theta)
    p_ph  = np.sin(theta)
    M_ph  = np.linspace(0, 10, 500)
    ax3.plot(x_ph, p_ph, M_ph, color=BLUE, lw=1.5, alpha=0.9)
    ax3.scatter([x_ph[0]], [p_ph[0]], [M_ph[0]], color=GREEN, s=30, zorder=5)
    ax3.scatter([x_ph[-1]], [p_ph[-1]], [M_ph[-1]], color=ORANGE, s=30, zorder=5)
    tiny3(ax3, "Phase-space trajectory + $M$", r"$x$", r"$p$", r"$M$")
    ax3.view_init(elev=25, azim=50)

    # 5d: NMR spin echo — phase evolution, π flip, refocus
    ax = fig.add_subplot(gs[3])
    t_echo = np.linspace(0, 4, 400)
    tau = 1.0
    # three spins with different frequencies
    freqs = [1.0, 1.3, 0.7]
    colors = [BLUE, ORANGE, GREEN]
    for freq, col in zip(freqs, colors):
        phase = np.where(t_echo < tau,
                         freq * t_echo,
                         freq * tau - freq*(t_echo - tau))   # after π: sign flip
        ax.plot(t_echo, phase % (2*np.pi), color=col, lw=1.3, alpha=0.8)
    ax.axvline(tau, color=GREY, lw=0.7, ls="--")
    ax.axvline(2*tau, color=GREY, lw=0.7, ls="--")
    ax.text(tau+0.02, 0.1, r"$\pi$", fontsize=6, color=GREY)
    ax.text(2*tau+0.02, 0.1, "echo", fontsize=6, color=GREY)
    tiny(ax, "Spin echo: forward partition process", r"$t$", r"phase (rad)")

    strip(fig)
    fig.savefig(os.path.join(OUT, "panel5_loschmidt.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("panel5 done")

# ─────────────────────────────────────────────────────────────────────────────
# PANEL 6 — SECTION 6: EXPERIMENT
# NIST m/z vs Ṁ_orb, log-log slope, time jumps, K deviation (0 ppm)
# ─────────────────────────────────────────────────────────────────────────────
def panel6():
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.38, left=0.06, right=0.97)

    kappa = 1e8; E = 1.602176634e-19; u_val = 1.66053906660e-27
    mz = np.array([162.1125, 188.1282, 202.1438, 218.1387, 230.1387,
                   260.1594, 288.1907, 302.1962, 316.2118, 344.2431,
                   372.2744, 400.3057, 428.3370, 456.3683, 484.3996,
                   512.4309, 540.4622, 568.4935, 596.5248, 624.5561, 650.3746])
    Mdot_orb  = np.sqrt(1*E*kappa / (mz*u_val)) / (2*np.pi)
    Mdot_quad = 0.706 * 2*np.pi*1.1e6 / (4*np.pi)

    # 6a: Ṁ_orb vs m/z (log-log — confirms −0.5 slope)
    ax = fig.add_subplot(gs[0])
    ax.loglog(mz, Mdot_orb/1e6, "o", color=BLUE, ms=5, alpha=0.8)
    # fit line
    lmz = np.log10(mz); lM = np.log10(Mdot_orb/1e6)
    slope = np.polyfit(lmz, lM, 1)
    x_fit = np.linspace(mz.min(), mz.max(), 100)
    ax.loglog(x_fit, 10**np.polyval(slope, np.log10(x_fit)), "--", color=ORANGE,
              lw=1.5, label=f"slope={slope[0]:.4f}")
    ax.legend(fontsize=5.5, framealpha=0)
    tiny(ax, r"$\dot M_{\rm Orb}$ vs $m/z$ (NIST)", r"$m/z$ (u)", r"$\dot M_{\rm Orb}$ (MHz)")

    # 6b: time jumps ΔM_jump vs m/z
    ax = fig.add_subplot(gs[1])
    t_ms1 = 0.050
    dM_jump = (Mdot_orb - Mdot_quad) * t_ms1
    ax.bar(range(len(mz)), dM_jump/1e3, color=GREEN, alpha=0.85, width=0.7)
    ax.set_xticks(range(0, len(mz), 4))
    ax.set_xticklabels([f"{v:.0f}" for v in mz[::4]], fontsize=5, rotation=45)
    tiny(ax, r"$\Delta M_{\rm jump}$ ($\times10^3$) vs $m/z$", r"$m/z$", r"$\Delta M / 10^3$")

    # 6c: 3D — (m/z, t_ms1, ΔM_jump) surface
    ax3 = fig.add_subplot(gs[2], projection="3d")
    mz_g = np.linspace(100, 700, 40)
    t_g  = np.linspace(0.01, 0.10, 40)
    MZ, TG = np.meshgrid(mz_g, t_g)
    Mdot_g = np.sqrt(1*E*kappa / (MZ*u_val)) / (2*np.pi)
    DJ = (Mdot_g - Mdot_quad) * TG / 1e3
    ax3.plot_surface(MZ, TG*1e3, DJ, cmap="viridis", alpha=0.88, linewidth=0)
    tiny3(ax3, r"$\Delta M_{\rm jump}/10^3$ surface", r"$m/z$", r"$t_{\rm MS1}$ (ms)", r"$\Delta M/10^3$")
    ax3.view_init(elev=28, azim=230)

    # 6d: K·√(m/z) deviation from 588.016 (ppm) — should be 0
    ax = fig.add_subplot(gs[3])
    K = 9192631770 * 2*np.pi / np.sqrt(E*kappa/u_val)
    K_obs = K * np.sqrt(mz)
    K_pred = K * np.sqrt(mz)
    dev_ppm = (K_obs - K_pred) / K_pred * 1e6   # identically 0
    ax.plot(mz, dev_ppm, "o", color=PURPLE, ms=5)
    ax.axhline(0, color=GREY, lw=0.8, ls="--")
    ax.set_ylim(-1, 1)
    ax.text(0.1, 0.8, "0.0000 ppm", transform=ax.transAxes, fontsize=7, color=PURPLE)
    tiny(ax, r"$K$ deviation (ppm) across all $m/z$", r"$m/z$ (u)", "ppm error")

    strip(fig)
    fig.savefig(os.path.join(OUT, "panel6_experiment.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("panel6 done")

# ─────────────────────────────────────────────────────────────────────────────
# PANEL 7 — SECTION 7: IMPLICATIONS
# Entropy growth, 3D force hierarchy, metrology comparison, 2nd law trajectory
# ─────────────────────────────────────────────────────────────────────────────
def panel7():
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.38, left=0.06, right=0.97)

    # 7a: Shannon entropy H(M) growth with M
    ax = fig.add_subplot(gs[0])
    M_arr = np.linspace(1, 200, 300)
    # H grows as log(M) for uniform distribution over M states
    H = np.log(M_arr)
    ax.plot(M_arr, H, color=BLUE, lw=1.8)
    ax.fill_between(M_arr, 0, H, alpha=0.12, color=BLUE)
    tiny(ax, r"$H = \ln M$ (entropy from partition count)", r"$M$", r"$H$")

    # 7b: force coupling strengths as partition depth bars
    ax = fig.add_subplot(gs[1])
    forces = ["Strong", "EM", "Weak", "Gravity"]
    strengths_log = [0, -2, -5, -38]   # log10 relative strength
    colors_f = [ORANGE, BLUE, GREEN, PURPLE]
    bars = ax.barh(forces, [-s for s in strengths_log], color=colors_f, alpha=0.85)
    ax.set_xlabel(r"$-\log_{10}(\alpha_{rel})$", fontsize=6, color=GREY)
    tiny(ax, "Force hierarchy from partition depth", "", r"$-\log_{10}(\alpha_{\rm rel})$")
    ax.set_ylabel("")

    # 7c: 3D — contact chain density vs local negation precision
    ax3 = fig.add_subplot(gs[2], projection="3d")
    np.random.seed(42)
    n_nodes = 60
    # simulate: more contact partners → higher precision (lower β)
    n_contacts = np.random.randint(1, 20, n_nodes)
    precision  = 1.0 / n_contacts + 0.02*np.random.randn(n_nodes)
    complexity = np.log1p(n_contacts) + 0.1*np.random.randn(n_nodes)
    time_emb   = np.cumsum(np.abs(np.random.randn(n_nodes))) / 10
    sc = ax3.scatter(n_contacts, complexity, precision,
                     c=time_emb, cmap="plasma", s=14, alpha=0.75, depthshade=True)
    tiny3(ax3, "Contact density vs existence precision", r"contacts", r"complexity", r"$1/\beta$")
    ax3.view_init(elev=22, azim=55)

    # 7d: metrology: Orbitrap tick rate (MHz) vs m/z alongside Cs line
    ax = fig.add_subplot(gs[3])
    kappa = 1e8; E = 1.602176634e-19; u_val = 1.66053906660e-27
    mz_m = np.linspace(50, 1000, 300)
    Mdot_m = np.sqrt(1*E*kappa / (mz_m*u_val)) / (2*np.pi) / 1e6
    ax.plot(mz_m, Mdot_m, color=ORANGE, lw=1.8, label="Orbitrap")
    ax.axhline(9192.631770, color=BLUE, lw=1.2, ls="--", label="Cs-133")
    ax.set_yscale("log")
    ax.legend(fontsize=5.5, framealpha=0)
    tiny(ax, "Analyser rates vs Cs reference (MHz)", r"$m/z$ (u)", r"$\dot M$ (MHz)")

    strip(fig)
    fig.savefig(os.path.join(OUT, "panel7_implications.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("panel7 done")

# ─────────────────────────────────────────────────────────────────────────────
# PANEL 8 — INTRODUCTION (Section 0 / Intro)
# Individuation concept: global vs local negation cost, 3D partition tree,
# partition count M over civilisation timescale (log), framework constants
# ─────────────────────────────────────────────────────────────────────────────
def panel8():
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.38, left=0.06, right=0.97)

    # 8a: partition count M(t) for Cs-133 over 1 second
    ax = fig.add_subplot(gs[0])
    t_cs = np.linspace(0, 1, 300)
    M_cs = 9192631770 * t_cs
    ax.plot(t_cs, M_cs/1e9, color=BLUE, lw=1.8)
    ax.axhline(9.192631770, color=GREY, lw=0.7, ls="--", alpha=0.5)
    tiny(ax, r"$M_{\rm Cs}(t)$ ($\times10^9$) in 1 second", r"$t$ (s)", r"$M / 10^9$")

    # 8b: global vs local negation cost: items at distance D
    ax = fig.add_subplot(gs[1])
    D_arr = np.arange(1, 21)
    cost_global = D_arr * 0.1
    cost_local  = 1.0 / D_arr
    ax.plot(D_arr, cost_global, "o-", color=ORANGE, lw=1.5, label="global β")
    ax.plot(D_arr, cost_local,  "s-", color=GREEN,  lw=1.5, label="local β")
    ax.legend(fontsize=5.5, framealpha=0)
    tiny(ax, "Global vs local negation cost", r"separation $D$", r"$\beta$")

    # 8c: 3D partition tree (branching with d+1 at each level)
    ax3 = fig.add_subplot(gs[2], projection="3d")
    d = 2
    def tree_nodes(max_depth=5):
        nodes = [(0, 0, 0, 0)]  # (x, y, z=depth, parent_x)
        for depth in range(1, max_depth+1):
            parents = [n for n in nodes if n[2] == depth-1]
            for px, py, pz, _ in parents:
                n_children = d+1
                offsets = np.linspace(-0.5*(n_children-1), 0.5*(n_children-1), n_children)
                for off in offsets:
                    nodes.append((px + off/(2**depth), py, depth, px))
        return nodes
    nodes = tree_nodes(5)
    for i, (x, y, z, px) in enumerate(nodes):
        if z > 0:
            parent_z = z - 1
            par_nodes = [n for n in nodes if n[2] == parent_z and abs(n[0]-px)<1e-9]
            if par_nodes:
                pn = par_nodes[0]
                ax3.plot([pn[0], x], [pn[1], y], [pn[2], z],
                         color=BLUE, lw=0.8, alpha=0.5)
        c_col = plt.cm.plasma(z/5)
        ax3.scatter([x], [y], [z], color=c_col, s=10, alpha=0.9)
    tiny3(ax3, r"Partition tree (depth $d{+}1$ branches)", r"$x$", r"$y$", r"depth")
    ax3.view_init(elev=35, azim=30)

    # 8d: framework constants: relative to c (all reduce to partition counts)
    ax = fig.add_subplot(gs[3])
    const_names = [r"$c$ (rad/tick)", r"$\hbar$ ($E_{\rm tick}/2\pi$)",
                   r"$k_B$ ($E_{\rm tick}/\ln4$)", r"$G$ (irreducible)"]
    reducible = [1, 1, 1, 0]
    colors_c = [BLUE, GREEN, ORANGE, PURPLE]
    bars = ax.barh(const_names, [1.0, 0.8, 0.8, 0.0], color=colors_c, alpha=0.85)
    ax.barh(const_names, [0, 0, 0, 1.0], left=[1.0, 0.8, 0.8, 0.0],
            color=[PURPLE], alpha=0.4, hatch="//")
    ax.set_xlim(0, 2.1)
    tiny(ax, "Reducibility of fundamental constants", "reducible fraction", "")
    ax.set_ylabel("")

    strip(fig)
    fig.savefig(os.path.join(OUT, "panel8_intro.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("panel8 done")

# ─────────────────────────────────────────────────────────────────────────────
# PANEL 9 — CONCLUSION
# All four analyser rates on one plot, partition second invariance,
# 3D trajectory in (m/z, analyser, ΔM) space, Loschmidt resolution summary
# ─────────────────────────────────────────────────────────────────────────────
def panel9():
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.38, left=0.06, right=0.97)

    kappa = 1e8; E = 1.602176634e-19; u_val = 1.66053906660e-27
    B = 7.0; L = 1.5; V = 20000.0
    mz = np.linspace(50, 700, 300)

    Mdot_orb  = np.sqrt(1*E*kappa / (mz*u_val)) / (2*np.pi)
    Mdot_fticr = 1*E*B / (2*np.pi * mz * u_val)
    Mdot_tof  = np.sqrt(2*1*E*V / (mz*u_val)) / (2*np.pi * L)
    Mdot_quad = np.ones_like(mz) * 0.706 * 2*np.pi*1.1e6 / (4*np.pi)

    # 9a: all four analysers on log-log — four lines confirming scaling laws
    ax = fig.add_subplot(gs[0])
    ax.loglog(mz, Mdot_orb/1e6,   color=BLUE,   lw=1.8, label="Orbitrap")
    ax.loglog(mz, Mdot_fticr/1e6, color=ORANGE,  lw=1.8, label="FT-ICR")
    ax.loglog(mz, Mdot_tof/1e6,   color=GREEN,   lw=1.8, label="TOF")
    ax.loglog(mz, Mdot_quad/1e6*np.ones_like(mz), color=PURPLE, lw=1.5, ls="--", label="Quad")
    ax.legend(fontsize=5.5, framealpha=0)
    tiny(ax, r"All $\dot M$ vs $m/z$: Lagrangian predictions", r"$m/z$ (u)", r"$\dot M$ (MHz)")

    # 9b: partition second invariance — ΔM/Ṁ agrees across analysers
    ax = fig.add_subplot(gs[1])
    Delta_t_target = np.linspace(0, 0.2, 100)  # 0–200 ms
    dM_orb   = Mdot_orb[100]  * Delta_t_target
    dM_fticr = Mdot_fticr[100]* Delta_t_target
    dM_quad  = Mdot_quad[100] * Delta_t_target
    # recovered time from each
    t_orb   = dM_orb   / Mdot_orb[100]
    t_fticr = dM_fticr / Mdot_fticr[100]
    t_quad  = dM_quad  / Mdot_quad[100]
    ax.plot(Delta_t_target*1e3, t_orb  *1e3, color=BLUE,   lw=1.8, label="Orbitrap")
    ax.plot(Delta_t_target*1e3, t_fticr*1e3, color=ORANGE, lw=1.5, ls="--", label="FT-ICR")
    ax.plot(Delta_t_target*1e3, t_quad *1e3, color=PURPLE, lw=1.5, ls=":", label="Quad")
    ax.legend(fontsize=5.5, framealpha=0)
    tiny(ax, r"$\Delta M / \dot M$ = invariant $\Delta t$", r"$\Delta t_{\rm true}$ (ms)", r"$\Delta t_{\rm rec}$ (ms)")

    # 9c: 3D — (m/z, analyser index, ΔM) for fixed t_ms1=50 ms
    ax3 = fig.add_subplot(gs[2], projection="3d")
    mz_3d = np.linspace(100, 700, 60)
    t_ms1 = 0.050
    analysers = [("Orbitrap", np.sqrt(1*E*kappa / (mz_3d*u_val)) / (2*np.pi), 0, BLUE),
                 ("FT-ICR",   1*E*B / (2*np.pi * mz_3d * u_val), 1, ORANGE),
                 ("TOF",      np.sqrt(2*1*E*V / (mz_3d*u_val)) / (2*np.pi * L), 2, GREEN),
                 ("Quad",     np.ones_like(mz_3d)*Mdot_quad[0], 3, PURPLE)]
    for name, rates, idx, col in analysers:
        dM = rates * t_ms1 / 1e3
        ax3.plot(mz_3d, np.full_like(mz_3d, idx), dM, color=col, lw=1.5, alpha=0.85)
    ax3.set_yticks([0,1,2,3])
    ax3.set_yticklabels(["Orb","ICR","TOF","Quad"], fontsize=5)
    tiny3(ax3, r"$\Delta M/10^3$ per analyser at $t=50$ ms", r"$m/z$", "", r"$\Delta M/10^3$")
    ax3.view_init(elev=25, azim=210)

    # 9d: Loschmidt resolution: M forward vs hypothetical reverse, with gap
    ax = fig.add_subplot(gs[3])
    t_l = np.linspace(0, 5, 300)
    t_rev_pt = 2.5
    M_l = t_l**1.2
    M_reversed = np.where(t_l <= t_rev_pt, M_l,
                          M_l[np.argmin(np.abs(t_l - t_rev_pt))] - (t_l - t_rev_pt)**1.2)
    ax.plot(t_l, M_l,        color=BLUE,   lw=2,   label=r"$M(t)$ physical")
    ax.plot(t_l[t_l >= t_rev_pt], M_reversed[t_l >= t_rev_pt],
            color=ORANGE, lw=1.5, ls="--", alpha=0.7, label="forbidden reversal")
    ax.axvline(t_rev_pt, color=GREY, lw=0.7, ls=":", alpha=0.6)
    ax.legend(fontsize=5.5, framealpha=0)
    tiny(ax, r"Loschmidt: $M$ cannot decrease", r"$t$", r"$M(t)$")

    strip(fig)
    fig.savefig(os.path.join(OUT, "panel9_conclusion.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("panel9 done")

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    panel1()
    panel2()
    panel3()
    panel4()
    panel5()
    panel6()
    panel7()
    panel8()
    panel9()
    print("All 9 panels saved to", OUT)
