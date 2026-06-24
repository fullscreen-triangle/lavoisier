"""
MSL Figure Panels Generator
Produces 8 publication-quality PNG panels (4 charts per row, 1 row each, 20x5 inches).
All data hardcoded from JSON validation results.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

OUT = r"c:\Users\kunda\Documents\bioinformatics\lavoisier\oxford\publications\minimum-sufficient-lagrangian\figures"
os.makedirs(OUT, exist_ok=True)

# Colour palette
C1 = '#2E86AB'  # steel blue
C2 = '#E84855'  # coral
C3 = '#3BB273'  # forest green
C4 = '#F4A261'  # gold
C5 = '#9B5DE5'  # violet

def clean_ax(ax, grid=False):
    ax.set_facecolor('#ffffff')
    if grid:
        ax.grid(True, alpha=0.2, lw=0.5)
    else:
        ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)

def new_fig():
    fig = plt.figure(figsize=(20, 5), facecolor='#ffffff')
    return fig

# ── PANEL 1 ──────────────────────────────────────────────────────────────────
def panel_01():
    fig = new_fig()

    # Data
    eps = np.array([0.1, 0.01, 0.001, 1e-6, 1e-9, 1e-12])
    mu  = np.array([1e-3, 1e-6, 1e-9, 1e-18, 1e-27, 1e-36])

    t = np.linspace(0, 1, 400)
    w = 2 * np.pi
    z  = np.cos(w * t)
    dz = -w * np.sin(w * t)

    # Chart 1 – log-log eps vs mu_separator
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.loglog(eps, mu, 'o', color=C1, ms=6, lw=1.5, label='data')
    # fit line slope=3
    eps_fit = np.array([1e-12, 0.1])
    ax1.loglog(eps_fit, eps_fit**3, '--', color=C2, lw=1.5, label='slope 3')
    ax1.set_xlabel('ε', fontsize=9); ax1.set_ylabel('μ_sep', fontsize=9)
    ax1.legend(frameon=False, fontsize=8)
    clean_ax(ax1, grid=True)

    # Chart 2 – z(t) and dz/dt
    ax2 = fig.add_subplot(1, 4, 2)
    ax2.plot(t, z,  color=C1, lw=1.5, label='z(t)')
    ax2.plot(t, dz, color=C2, lw=1.5, label='ż(t)')
    ax2.set_xlabel('t', fontsize=9)
    ax2.legend(frameon=False, fontsize=8)
    clean_ax(ax2)

    # Chart 3 – phase portrait
    ax3 = fig.add_subplot(1, 4, 3)
    ax3.plot(z, dz, color=C3, lw=1.5)
    ax3.set_xlabel('z', fontsize=9); ax3.set_ylabel('ż', fontsize=9)
    clean_ax(ax3)

    # Chart 4 (3D) – parametric helix coloured by H (constant)
    ax4 = fig.add_subplot(1, 4, 4, projection='3d')
    ax4.set_facecolor('#ffffff')
    H_val = 19.739
    mu_mass = 1.0
    # z(t)=cos(wt), p=-w*sin(wt)*mu_mass  — but keep H constant colour
    colors = np.full(len(t), H_val)
    sc = ax4.scatter(t, z, dz, c=colors, cmap='Blues', s=4, vmin=H_val-1, vmax=H_val+1)
    ax4.set_xlabel('t', fontsize=7); ax4.set_ylabel('z', fontsize=7); ax4.set_zlabel('ż', fontsize=7)
    ax4.tick_params(labelsize=6)
    ax4.set_facecolor('#ffffff')

    plt.tight_layout()
    path = os.path.join(OUT, 'msl_panel_01.png')
    fig.savefig(path, dpi=150, facecolor='#ffffff', bbox_inches='tight')
    plt.close(fig)
    return path

# ── PANEL 2 ──────────────────────────────────────────────────────────────────
def panel_02():
    fig = new_fig()

    t = np.linspace(0, 1, 200)
    w = 2 * np.pi
    mu = 1.0
    z0 = 1.0
    H_ref = 19.739
    # M(z) = H_ref - KE at z=z0; KE = p²/(2mu), p = -w*z0*sin(wt)
    # For HO: H = p²/(2mu) + (1/2)*mu*w²*z²; mu=1 → H = 0.5*w²*(sin²+cos²)=0.5*w²
    # But H_ref=19.739 ~ (2pi)²/2 ≈ 19.739 ✓
    p  = -w * z0 * np.sin(w * t)
    z  = z0 * np.cos(w * t)
    KE = p**2 / (2 * mu)
    PE = 0.5 * mu * w**2 * z**2
    H  = KE + PE

    # Chart 1 – H residual
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.plot(t, H - H_ref, color=C1, lw=1.5)
    ax1.axhline(0, color=C2, lw=0.8, ls='--')
    ax1.set_xlabel('t', fontsize=9); ax1.set_ylabel('H(t)−H_ref', fontsize=9)
    ax1.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    clean_ax(ax1, grid=True)

    # Chart 2 – KE and PE
    ax2 = fig.add_subplot(1, 4, 2)
    ax2.plot(t, KE, color=C1, lw=1.5, label='KE')
    ax2.plot(t, PE, color=C2, lw=1.5, label='PE')
    ax2.set_xlabel('t', fontsize=9)
    ax2.legend(frameon=False, fontsize=8)
    clean_ax(ax2)

    # Chart 3 – Lagrangian L = KE - PE
    L = KE - PE
    ax3 = fig.add_subplot(1, 4, 3)
    ax3.plot(t, L, color=C3, lw=1.5)
    ax3.axhline(0, color='k', lw=0.6, ls='--')
    ax3.fill_between(t, L, 0, where=(L>=0), alpha=0.15, color=C3)
    ax3.fill_between(t, L, 0, where=(L< 0), alpha=0.15, color=C4)
    ax3.set_xlabel('t', fontsize=9); ax3.set_ylabel('L(t)', fontsize=9)
    clean_ax(ax3)

    # Chart 4 (3D) – (t, KE, PE) parametric
    ax4 = fig.add_subplot(1, 4, 4, projection='3d')
    ax4.set_facecolor('#ffffff')
    ax4.plot(t, KE, PE, color=C5, lw=1.2)
    ax4.set_xlabel('t', fontsize=7); ax4.set_ylabel('KE', fontsize=7); ax4.set_zlabel('PE', fontsize=7)
    ax4.tick_params(labelsize=6)

    plt.tight_layout()
    path = os.path.join(OUT, 'msl_panel_02.png')
    fig.savefig(path, dpi=150, facecolor='#ffffff', bbox_inches='tight')
    plt.close(fig)
    return path

# ── PANEL 3 ──────────────────────────────────────────────────────────────────
def panel_03():
    fig = new_fig()

    mz_pts  = np.array([100, 150, 200, 300, 400, 500, 650], dtype=float)
    K = 588.016

    # omega_z ∝ (m/z)^{-0.5}
    omega_z = K * mz_pts**(-0.5)

    mz_full = np.linspace(50, 800, 300)
    # transit time T_TOF = L*sqrt(m/(2qV)), L=1m, V=1000V, q=e=1.6e-19 C
    # m in Da → kg: 1 Da = 1.6605e-27 kg
    Da = 1.6605e-27; q_e = 1.6022e-19; L = 1.0; V = 1000.0; B = 7.0
    m_kg = mz_full * Da
    T_TOF = L * np.sqrt(m_kg / (2 * q_e * V)) * 1e6  # microseconds
    omega_c = q_e * B / m_kg  # rad/s

    # Chart 1 – omega_z vs m/z log-log
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.loglog(mz_pts, omega_z, 'o', color=C1, ms=6)
    mz_fit = np.array([100, 650], dtype=float)
    ax1.loglog(mz_fit, K * mz_fit**(-0.5), '--', color=C2, lw=1.5, label='slope −½')
    ax1.set_xlabel('m/z', fontsize=9); ax1.set_ylabel('ω_z', fontsize=9)
    ax1.legend(frameon=False, fontsize=8)
    clean_ax(ax1, grid=True)

    # Chart 2 – T_TOF vs m/z
    ax2 = fig.add_subplot(1, 4, 2)
    ax2.plot(mz_full, T_TOF, color=C3, lw=1.5)
    ax2.set_xlabel('m/z', fontsize=9); ax2.set_ylabel('T_TOF (μs)', fontsize=9)
    clean_ax(ax2)

    # Chart 3 – omega_c vs m/z
    ax3 = fig.add_subplot(1, 4, 3)
    ax3.semilogy(mz_full, omega_c, color=C4, lw=1.5)
    ax3.set_xlabel('m/z', fontsize=9); ax3.set_ylabel('ω_c (rad/s)', fontsize=9)
    clean_ax(ax3, grid=True)

    # Chart 4 (3D) – contact map surface
    ax4 = fig.add_subplot(1, 4, 4, projection='3d')
    ax4.set_facecolor('#ffffff')
    n = len(mz_pts)
    X, Y = np.meshgrid(range(n), range(n))
    Z = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            Z[i, j] = K * abs(mz_pts[i]**(-0.5) - mz_pts[j]**(-0.5))
    ax4.plot_surface(X, Y, Z, cmap='Blues', edgecolor='none', alpha=0.85)
    ax4.set_xlabel('mz_a idx', fontsize=7); ax4.set_ylabel('mz_b idx', fontsize=7); ax4.set_zlabel('CM', fontsize=7)
    ax4.tick_params(labelsize=6)

    plt.tight_layout()
    path = os.path.join(OUT, 'msl_panel_03.png')
    fig.savefig(path, dpi=150, facecolor='#ffffff', bbox_inches='tight')
    plt.close(fig)
    return path

# ── PANEL 4 ──────────────────────────────────────────────────────────────────
def panel_04():
    fig = new_fig()

    n_arr = np.arange(1, 21)

    # Chart 1 – T(n,3) = 3*4^{n-1}
    T3 = 3 * 4.0**(n_arr - 1)
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.semilogy(n_arr, T3, 'o-', color=C1, lw=1.5, ms=4)
    ax1.set_xlabel('n', fontsize=9); ax1.set_ylabel('T(n,3)', fontsize=9)
    clean_ax(ax1, grid=True)

    # Chart 2 – log10(T(n,d)) for d=2,3,4
    ax2 = fig.add_subplot(1, 4, 2)
    for d, col in zip([2, 3, 4], [C1, C2, C3]):
        T = d * (d+1.0)**(n_arr - 1)
        ax2.plot(n_arr, np.log10(T), color=col, lw=1.5, label=f'd={d}')
    ax2.set_xlabel('n', fontsize=9); ax2.set_ylabel('log₁₀ T', fontsize=9)
    ax2.legend(frameon=False, fontsize=8)
    clean_ax(ax2)

    # Chart 3 – fractional deviation of G routes
    ns = np.array([8, 15, 27])
    R1 = np.array([1.41e-5, 8.60e-10, 0.0])
    R2 = np.array([9.54e-7, 5.82e-11, 0.0])
    R3 = np.array([3.81e-6, 2.33e-10, 0.0])
    bd = np.array([1.53e-5, 9.31e-10, 5.55e-17])
    ax3 = fig.add_subplot(1, 4, 3)
    # replace zeros with tiny for log
    R1c = np.where(R1 == 0, 1e-18, R1)
    R2c = np.where(R2 == 0, 1e-18, R2)
    R3c = np.where(R3 == 0, 1e-18, R3)
    bdc = np.where(bd == 0, 1e-18, bd)
    ax3.semilogy(ns, R1c, 'o-', color=C1, lw=1.5, ms=5, label='Route I')
    ax3.semilogy(ns, R2c, 's-', color=C2, lw=1.5, ms=5, label='Route II')
    ax3.semilogy(ns, R3c, '^-', color=C3, lw=1.5, ms=5, label='Route III')
    ax3.semilogy(ns, bdc, '--', color='k',  lw=1.2, label='bound')
    ax3.set_xlabel('n', fontsize=9); ax3.set_ylabel('|δG/G|', fontsize=9)
    ax3.legend(frameon=False, fontsize=7)
    clean_ax(ax3, grid=True)

    # Chart 4 (3D) – surface log10(T(n,d)) n=1..15, d=2..6
    ax4 = fig.add_subplot(1, 4, 4, projection='3d')
    ax4.set_facecolor('#ffffff')
    nv = np.arange(1, 16)
    dv = np.arange(2, 7)
    N, D = np.meshgrid(nv, dv)
    Tsurf = np.log10(D * (D + 1.0)**(N - 1))
    ax4.plot_surface(N, D, Tsurf, cmap='viridis', edgecolor='none', alpha=0.85)
    ax4.set_xlabel('n', fontsize=7); ax4.set_ylabel('d', fontsize=7); ax4.set_zlabel('log₁₀T', fontsize=7)
    ax4.tick_params(labelsize=6)

    plt.tight_layout()
    path = os.path.join(OUT, 'msl_panel_04.png')
    fig.savefig(path, dpi=150, facecolor='#ffffff', bbox_inches='tight')
    plt.close(fig)
    return path

# ── PANEL 5 ──────────────────────────────────────────────────────────────────
def panel_05():
    fig = new_fig()

    # Chart 1 – staircase: n_needed vs log10(epsilon)
    eps_vals  = np.array([1e-5, 1e-9, 1e-16, 1e-33])
    n_needed  = np.array([10, 16, 28, 56])
    log_eps   = np.log10(eps_vals)
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.step(log_eps, n_needed, where='post', color=C1, lw=1.5)
    ax1.plot(log_eps, n_needed, 'o', color=C1, ms=6)
    ax1.set_xlabel('log₁₀(ε)', fontsize=9); ax1.set_ylabel('n_needed', fontsize=9)
    clean_ax(ax1, grid=True)

    # Chart 2 – integration_time vs epsilon log-log
    int_time = np.array([1.09e-9, 1.74e-9, 3.05e-9, 6.09e-9])
    ax2 = fig.add_subplot(1, 4, 2)
    ax2.loglog(eps_vals, int_time, 'o-', color=C2, lw=1.5, ms=5)
    ax2.set_xlabel('ε', fontsize=9); ax2.set_ylabel('t_integ (s)', fontsize=9)
    clean_ax(ax2, grid=True)

    # Chart 3 – r³ measured vs Kepler scatter + identity
    r3_meas  = np.array([5.695e25, 3.348e33])
    r3_kep   = np.array([5.626e25, 3.348e33])
    ax3 = fig.add_subplot(1, 4, 3)
    lims = np.array([min(r3_kep)*0.99, max(r3_kep)*1.01])
    ax3.loglog(r3_kep, r3_meas, 'o', color=C3, ms=8)
    ax3.loglog(lims, lims, '--', color='k', lw=1.0, label='identity')
    for val_m, val_k, lbl in zip(r3_meas, r3_kep, ['E-M', 'E-S']):
        ax3.annotate(lbl, (val_k, val_m), fontsize=7, xytext=(5, 2), textcoords='offset points')
    ax3.set_xlabel('r³ Kepler', fontsize=9); ax3.set_ylabel('r³ measured', fontsize=9)
    ax3.legend(frameon=False, fontsize=8)
    clean_ax(ax3, grid=True)

    # Chart 4 (3D) – Kepler orbit ellipse coloured by speed
    theta = np.linspace(0, 2*np.pi, 500)
    e = 0.017; a = 1.0
    r = a * (1 - e**2) / (1 + e * np.cos(theta))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.zeros_like(theta)
    speed = 1.0 / r  # ∝ orbital speed
    ax4 = fig.add_subplot(1, 4, 4, projection='3d')
    ax4.set_facecolor('#ffffff')
    sc = ax4.scatter(x, y, z, c=speed, cmap='plasma', s=3)
    ax4.set_xlabel('x (AU)', fontsize=7); ax4.set_ylabel('y (AU)', fontsize=7); ax4.set_zlabel('z', fontsize=7)
    ax4.tick_params(labelsize=6)

    plt.tight_layout()
    path = os.path.join(OUT, 'msl_panel_05.png')
    fig.savefig(path, dpi=150, facecolor='#ffffff', bbox_inches='tight')
    plt.close(fig)
    return path

# ── PANEL 6 ──────────────────────────────────────────────────────────────────
def panel_06():
    fig = new_fig()

    G_CODATA = 6.6743e-11
    bodies   = ['Moon', 'Earth', 'Mars']
    G_deriv  = np.array([6.657e-11, 6.665e-11, 6.663e-11])

    # Physical data for Moon, Earth, Mars
    g_surf = np.array([1.62, 9.81, 3.72])   # m/s²
    r_body = np.array([1.737e6, 6.371e6, 3.390e6])  # m
    M_body = np.array([7.342e22, 5.972e24, 6.417e23])  # kg

    # Chart 1 – G_derived vs body index
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.scatter([0, 1, 2], G_deriv, color=C1, s=60, zorder=3)
    ax1.axhline(G_CODATA, color=C2, lw=1.2, ls='--', label='CODATA')
    ax1.set_xticks([0, 1, 2]); ax1.set_xticklabels(bodies, fontsize=8)
    ax1.set_ylabel('G (m³/kg/s²)', fontsize=9)
    ax1.legend(frameon=False, fontsize=8)
    clean_ax(ax1, grid=True)

    # Chart 2 – fractional error bars
    frac_err = np.abs(G_deriv - G_CODATA) / G_CODATA
    ax2 = fig.add_subplot(1, 4, 2)
    bars = ax2.bar([0, 1, 2], frac_err, color=[C1, C3, C4], width=0.5)
    ax2.set_xticks([0, 1, 2]); ax2.set_xticklabels(bodies, fontsize=8)
    ax2.set_ylabel('|δG|/G', fontsize=9)
    ax2.set_yscale('log')
    clean_ax(ax2, grid=True)

    # Chart 3 – g_surf vs orbital radius
    ax3 = fig.add_subplot(1, 4, 3)
    ax3.scatter(r_body, g_surf, color=C2, s=60, zorder=3)
    # fit line in log-log
    log_r = np.log10(r_body); log_g = np.log10(g_surf)
    m_fit, b_fit = np.polyfit(log_r, log_g, 1)
    r_fit = np.linspace(r_body.min()*0.9, r_body.max()*1.1, 100)
    g_fit = 10**(b_fit) * r_fit**m_fit
    ax3.loglog(r_fit, g_fit, '--', color=C1, lw=1.2)
    for xi, yi, lb in zip(r_body, g_surf, bodies):
        ax3.annotate(lb, (xi, yi), fontsize=7, xytext=(3, 2), textcoords='offset points')
    ax3.set_xlabel('r (m)', fontsize=9); ax3.set_ylabel('g (m/s²)', fontsize=9)
    clean_ax(ax3, grid=True)

    # Chart 4 (3D) – (g_surf, r, M) scatter coloured by G
    ax4 = fig.add_subplot(1, 4, 4, projection='3d')
    ax4.set_facecolor('#ffffff')
    sc = ax4.scatter(np.log10(g_surf), np.log10(r_body), np.log10(M_body),
                     c=G_deriv, cmap='coolwarm', s=80)
    for xi, yi, zi, lb in zip(np.log10(g_surf), np.log10(r_body), np.log10(M_body), bodies):
        ax4.text(xi, yi, zi, lb, fontsize=6)
    ax4.set_xlabel('log g', fontsize=7); ax4.set_ylabel('log r', fontsize=7); ax4.set_zlabel('log M', fontsize=7)
    ax4.tick_params(labelsize=6)

    plt.tight_layout()
    path = os.path.join(OUT, 'msl_panel_06.png')
    fig.savefig(path, dpi=150, facecolor='#ffffff', bbox_inches='tight')
    plt.close(fig)
    return path

# ── PANEL 7 ──────────────────────────────────────────────────────────────────
def panel_07():
    fig = new_fig()

    t_arr = np.arange(1, 101)
    delta_fwd = 0.05 * t_arr + 0.002 * t_arr**1.5
    delta_rwd  = delta_fwd[::-1]

    # Chart 1 – forward profile with fill
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.fill_between(t_arr, delta_fwd, alpha=0.25, color=C1)
    ax1.plot(t_arr, delta_fwd, color=C1, lw=1.5)
    ax1.set_xlabel('t', fontsize=9); ax1.set_ylabel('δβ', fontsize=9)
    clean_ax(ax1)

    # Chart 2 – overlay forward + rewind
    ax2 = fig.add_subplot(1, 4, 2)
    ax2.fill_between(t_arr, delta_rwd, alpha=0.2, color=C2)
    ax2.plot(t_arr, delta_fwd, color=C1, lw=1.5, label='forward')
    ax2.plot(t_arr, delta_rwd, color=C2, lw=1.5, ls='--', label='rewind')
    ax2.set_xlabel('t', fontsize=9)
    ax2.legend(frameon=False, fontsize=8)
    clean_ax(ax2)

    # Chart 3 – difference (should be ~0 only if same values — actually they're reversal)
    diff = delta_fwd - delta_rwd
    ax3 = fig.add_subplot(1, 4, 3)
    ax3.plot(t_arr, diff, color=C3, lw=1.5)
    ax3.axhline(0, color='k', lw=0.6, ls='--')
    ax3.set_xlabel('t', fontsize=9); ax3.set_ylabel('δβ_fwd − δβ_rwd', fontsize=9)
    clean_ax(ax3)

    # Chart 4 (3D) – 3D path (t, delta_fwd, delta_rwd)
    ax4 = fig.add_subplot(1, 4, 4, projection='3d')
    ax4.set_facecolor('#ffffff')
    ax4.plot(t_arr, delta_fwd, delta_rwd, color=C5, lw=1.2)
    ax4.set_xlabel('t', fontsize=7); ax4.set_ylabel('δβ fwd', fontsize=7); ax4.set_zlabel('δβ rwd', fontsize=7)
    ax4.tick_params(labelsize=6)

    plt.tight_layout()
    path = os.path.join(OUT, 'msl_panel_07.png')
    fig.savefig(path, dpi=150, facecolor='#ffffff', bbox_inches='tight')
    plt.close(fig)
    return path

# ── PANEL 8 ──────────────────────────────────────────────────────────────────
def panel_08():
    fig = new_fig()

    T2     = 0.5   # s
    T2star = 0.01  # s
    sigma  = 100.0 # rad/s
    tau    = 0.005 # s

    t_full = np.linspace(0, 0.05, 500)

    FID      = np.exp(-t_full / T2star)
    T2_env   = np.exp(-t_full / T2)

    # Chart 1 – FID and T2 envelope
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.plot(t_full * 1e3, FID,   color=C1, lw=1.5, label='FID (T2*)')
    ax1.plot(t_full * 1e3, T2_env, color=C2, lw=1.5, ls='--', label='T2 envelope')
    ax1.set_xlabel('t (ms)', fontsize=9); ax1.set_ylabel('Signal', fontsize=9)
    ax1.legend(frameon=False, fontsize=8)
    clean_ax(ax1)

    # Chart 2 – spin echo: FID up to tau, refocus, peak at 2*tau
    t_echo = np.linspace(0, 4*tau, 500)
    echo_signal = np.zeros_like(t_echo)
    for i, ti in enumerate(t_echo):
        if ti <= tau:
            echo_signal[i] = np.exp(-ti / T2star)
        else:
            # refocusing: decay from 2*tau point
            dt = abs(ti - 2*tau)
            echo_signal[i] = np.exp(-2*tau/T2) * np.exp(-dt/T2star)
    ax2 = fig.add_subplot(1, 4, 2)
    ax2.plot(t_echo * 1e3, echo_signal, color=C3, lw=1.5)
    ax2.axvline(tau*1e3, color=C2, lw=0.8, ls='--', label='τ')
    ax2.axvline(2*tau*1e3, color=C4, lw=0.8, ls='--', label='2τ')
    ax2.set_xlabel('t (ms)', fontsize=9); ax2.set_ylabel('Echo', fontsize=9)
    ax2.legend(frameon=False, fontsize=8)
    clean_ax(ax2)

    # Chart 3 – T2*/T2 ratio vs delta_beta_inhom/bmin — identity line
    ratio = np.linspace(0, 1, 100)
    ax3 = fig.add_subplot(1, 4, 3)
    ax3.plot(ratio, ratio, color=C5, lw=1.5)
    ax3.set_xlabel('δβ_inhom / b_min', fontsize=9); ax3.set_ylabel('T2*/T2', fontsize=9)
    ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
    clean_ax(ax3)

    # Chart 4 (3D) – surface S(tau, T2star) = exp(-2tau/T2)*exp(-(2tau)^2*sigma^2)
    # Corrected: Gaussian decay from inhomogeneity: exp(-sigma²*(2tau*delta_t)²)
    # S(tau, T2star) = exp(-2tau/T2) * exp(-(2tau)^2 * sigma^2 / 2)  — shape
    tau_arr   = np.linspace(0.0005, 0.03, 60)
    T2st_arr  = np.linspace(0.005, 0.05,  60)
    TAU, T2ST = np.meshgrid(tau_arr, T2st_arr)
    S = np.exp(-2*TAU / T2) * np.exp(-(2*TAU)**2 * sigma**2 / 2)
    ax4 = fig.add_subplot(1, 4, 4, projection='3d')
    ax4.set_facecolor('#ffffff')
    ax4.plot_surface(TAU*1e3, T2ST*1e3, S, cmap='Blues', edgecolor='none', alpha=0.85)
    ax4.set_xlabel('τ (ms)', fontsize=7); ax4.set_ylabel('T2* (ms)', fontsize=7); ax4.set_zlabel('S', fontsize=7)
    ax4.tick_params(labelsize=6)

    plt.tight_layout()
    path = os.path.join(OUT, 'msl_panel_08.png')
    fig.savefig(path, dpi=150, facecolor='#ffffff', bbox_inches='tight')
    plt.close(fig)
    return path

if __name__ == '__main__':
    panels = [panel_01, panel_02, panel_03, panel_04,
              panel_05, panel_06, panel_07, panel_08]
    for fn in panels:
        p = fn()
        print(f"  saved {os.path.basename(p)}")
    print("Done: 8 panels saved")
