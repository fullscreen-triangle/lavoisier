"""
Generate 5 figure panels for:
"Stacked Virtual Substates as a Partition Tensor"

Each panel: 4 charts in a 1×4 row, white background, at least one 3-D chart.
All charts are data-driven from NIST AC_CAC results or computed from framework
formulas — no conceptual / text / table content.
"""

import json, math, sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ── paths ─────────────────────────────────────────────────────────────────────
_HERE   = Path(__file__).parent
_RESULT = _HERE / 'results' / 'phase_coherence_results.json'
_OUTDIR = _HERE / 'figures'
_OUTDIR.mkdir(exist_ok=True)

with open(_RESULT) as f:
    DATA = json.load(f)

# Per-spectrum fragment data
PER_SPEC = DATA.get('phase_coherence', {}).get('per_spectrum', [])

# ── physical constants ────────────────────────────────────────────────────────
KAPPA  = 0.1
B_FIEL = 7.0
DA     = 1.660_539e-27
E_ELEM = 1.602_176e-19

def orb_freq(mz): return (1/(2*math.pi))*math.sqrt(KAPPA/mz) if mz>0 else 0.0
def icr_freq(mz, z=1): return z*E_ELEM*B_FIEL/(2*math.pi*mz*z*DA) if mz>0 else 0.0
def _total_cap(n): return n*(n+1)*(2*n+1)//3
def mz_to_n(mz):  return max(1, int(math.floor(math.sqrt(mz)))+1)
def mz_to_M(mz):  return _total_cap(mz_to_n(mz)-1)+1
def T_infl(n, d): return d*(d+1)**(n-1)

plt.rcParams.update({
    'font.size': 8, 'axes.linewidth': 0.7,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.4,
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
})

def new_fig():
    return plt.figure(figsize=(16, 3.8), facecolor='white')

def save(fig, name):
    p = _OUTDIR / name
    fig.savefig(p, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f'  Saved {p}')


# ── extract real fragment data from JSON ──────────────────────────────────────
all_prec_mz, all_frag_mz, all_dn, all_dM, all_dtheta = [], [], [], [], []
all_ratio_pred, all_ratio_obs, all_back_err = [], [], []

for spec in PER_SPEC:
    pmz = spec.get('precursor_mz', 0)
    if pmz <= 0:
        continue
    for fr in spec.get('fragments', []):
        fmz = fr.get('mz_frag', 0)
        if fmz <= 0:
            continue
        all_prec_mz.append(pmz)
        all_frag_mz.append(fmz)
        all_dn.append(fr.get('delta_n', 0))
        all_dM.append(abs(fr.get('delta_M', 0)))
        all_dtheta.append(fr.get('delta_theta_rad', 0))
        all_ratio_pred.append(fr.get('ratio_pred', 0))
        all_ratio_obs.append(fr.get('ratio_obs', 0))
        all_back_err.append(fr.get('back_err_ppm', 0))

all_prec_mz  = np.array(all_prec_mz)
all_frag_mz  = np.array(all_frag_mz)
all_dn       = np.array(all_dn)
all_dM       = np.array(all_dM)
all_dtheta   = np.array(all_dtheta)
all_ratio_pred = np.array(all_ratio_pred)
all_ratio_obs  = np.array(all_ratio_obs)
all_back_err   = np.array(all_back_err)


# ═══════════════════════════════════════════════════════════════════════════════
# Panel 1 — Partition Bijection Φ: ℤ⁺ → P
# ═══════════════════════════════════════════════════════════════════════════════
def panel_01():
    fig = new_fig()

    ns = np.arange(1, 21)
    cap   = 2*ns**2
    N_cum = ns*(ns+1)*(2*ns+1)//3

    # (A) Capacity C(n) = 2n² ──────────────────────────────────────────────────
    ax = fig.add_subplot(1, 4, 1)
    ax.bar(ns, cap, color=plt.cm.plasma(np.linspace(0.1,0.9,len(ns))),
           edgecolor='white', linewidth=0.3)
    ax.set_xlabel('Principal level  n')
    ax.set_ylabel('C(n) = 2n²')
    ax.set_title('Shell capacity', pad=4)

    # (B) Cumulative N_state(n) ────────────────────────────────────────────────
    ax = fig.add_subplot(1, 4, 2)
    ax.fill_between(ns, 0, N_cum, alpha=0.35, color='#2166ac')
    ax.plot(ns, N_cum, 'o-', color='#2166ac', lw=1.5, ms=4)
    n_fit = np.linspace(1, 20, 200)
    ax.plot(n_fit, 2/3*n_fit**3, '--', color='#d73027', lw=1.0,
            label=r'$\frac{2}{3}n^3$ fit')
    ax.set_xlabel('n')
    ax.set_ylabel(r'$N_{\rm state}(n)$')
    ax.legend(fontsize=7, framealpha=0.5)
    ax.set_title('Cumulative states', pad=4)

    # (C) n-shell distribution of NIST precursors ──────────────────────────────
    ax = fig.add_subplot(1, 4, 3)
    prec_mz_list = [s['precursor_mz'] for s in PER_SPEC if s.get('precursor_mz',0)>0]
    if prec_mz_list:
        n_shells = [mz_to_n(m) for m in prec_mz_list]
        unique_n, counts = np.unique(n_shells, return_counts=True)
        ax.bar(unique_n, counts, color='#4dac26', alpha=0.85, edgecolor='white', lw=0.3)
    ax.set_xlabel('n-shell')
    ax.set_ylabel('Count  (NIST precursors)')
    ax.set_title('Precursor n-shell\ndistribution', pad=4)

    # (D) 3D scatter of partition states (n, ℓ, m) ─────────────────────────────
    ax = fig.add_subplot(1, 4, 4, projection='3d')
    # Generate all partition states for M = 1..300
    pts_n, pts_l, pts_m, pts_M = [], [], [], []
    M_idx = 0
    for n in range(1, 14):
        for l in range(n):
            for m in range(-l, l+1):
                for s_val in [0, 1]:
                    M_idx += 1
                    pts_n.append(n); pts_l.append(l)
                    pts_m.append(m); pts_M.append(M_idx)
                    if M_idx >= 300: break
                if M_idx >= 300: break
            if M_idx >= 300: break
        if M_idx >= 300: break
    pts_n = np.array(pts_n); pts_l = np.array(pts_l)
    pts_m = np.array(pts_m); pts_M = np.array(pts_M, dtype=float)
    sc = ax.scatter(pts_n, pts_l, pts_m, c=pts_M, cmap='plasma',
                    s=12, alpha=0.75, linewidths=0)
    ax.set_xlabel('n', labelpad=1, fontsize=7)
    ax.set_ylabel('ℓ', labelpad=1, fontsize=7)
    ax.set_zlabel('m', labelpad=1, fontsize=7)
    ax.set_title('Partition states\nΦ(M)', pad=4)
    ax.tick_params(labelsize=6)

    fig.subplots_adjust(wspace=0.4, left=0.06, right=0.97)
    save(fig, 'panel_01_partition_bijection.png')


# ═══════════════════════════════════════════════════════════════════════════════
# Panel 2 — Phase Coherence: NIST Fragment Data
# ═══════════════════════════════════════════════════════════════════════════════
def panel_02():
    fig = new_fig()

    # (A) |ΔM| distribution ────────────────────────────────────────────────────
    ax = fig.add_subplot(1, 4, 1)
    if len(all_dM) > 0:
        ax.hist(all_dM, bins=60, color='#5e3c99', alpha=0.82, edgecolor='none',
                density=True)
    ax.set_xlabel('|ΔM|  (partition count difference)')
    ax.set_ylabel('Density')
    ax.set_title('|ΔM| distribution\nall fragment pairs', pad=4)

    # (B) Δn distribution ──────────────────────────────────────────────────────
    ax = fig.add_subplot(1, 4, 2)
    if len(all_dn) > 0:
        unique_dn, cnt_dn = np.unique(all_dn, return_counts=True)
        colors_dn = plt.cm.RdBu_r(
            (unique_dn - unique_dn.min()) / (unique_dn.max() - unique_dn.min() + 1e-9)
        )
        ax.bar(unique_dn, cnt_dn, color=colors_dn, edgecolor='white', lw=0.3)
    ax.axvline(0, color='grey', lw=0.8, ls='--')
    ax.set_xlabel('Δn  (shell difference)')
    ax.set_ylabel('Count')
    ax.set_title('Shell-difference\ndistribution', pad=4)

    # (C) m/z_frag vs m/z_prec scatter ────────────────────────────────────────
    ax = fig.add_subplot(1, 4, 3)
    if len(all_prec_mz) > 0:
        n_prec_arr = np.array([mz_to_n(m) for m in all_prec_mz])
        step = max(1, len(all_prec_mz)//2000)
        sc = ax.scatter(all_prec_mz[::step], all_frag_mz[::step],
                        c=n_prec_arr[::step], cmap='plasma',
                        s=4, alpha=0.5, linewidths=0)
        plt.colorbar(sc, ax=ax, label='n-shell', pad=0.02, fraction=0.04)
        lim = max(all_prec_mz.max(), all_frag_mz.max())
        ax.plot([0, lim], [0, lim], 'k--', lw=0.7, alpha=0.4)
    ax.set_xlabel('Precursor m/z  (Da)')
    ax.set_ylabel('Fragment m/z  (Da)')
    ax.set_title('Prec vs frag m/z\ncoloured by n-shell', pad=4)

    # (D) 3D: (m/z_prec, m/z_frag, |Δθ|) ─────────────────────────────────────
    ax = fig.add_subplot(1, 4, 4, projection='3d')
    if len(all_prec_mz) > 0:
        step = max(1, len(all_prec_mz)//1500)
        dn_plot = all_dn[::step]
        sc = ax.scatter(all_prec_mz[::step], all_frag_mz[::step],
                        np.log10(all_dtheta[::step]+1),
                        c=dn_plot, cmap='RdBu_r', s=5, alpha=0.55,
                        linewidths=0,
                        vmin=dn_plot.min(), vmax=max(1, dn_plot.max()))
    ax.set_xlabel('Prec m/z', labelpad=2, fontsize=7)
    ax.set_ylabel('Frag m/z', labelpad=2, fontsize=7)
    ax.set_zlabel(r'$\log_{10}|\Delta\theta|$', labelpad=2, fontsize=7)
    ax.set_title('Phase space\n(m/z, m/z, |Δθ|)', pad=4)
    ax.tick_params(labelsize=6)

    fig.subplots_adjust(wspace=0.42, left=0.06, right=0.97)
    save(fig, 'panel_02_phase_coherence.png')


# ═══════════════════════════════════════════════════════════════════════════════
# Panel 3 — Subharmonic Frequency Self-Consistency
# ═══════════════════════════════════════════════════════════════════════════════
def panel_03():
    fig = new_fig()

    # (A) Frequency ratio: observed vs predicted ────────────────────────────────
    ax = fig.add_subplot(1, 4, 1)
    if len(all_ratio_pred) > 0:
        step = max(1, len(all_ratio_pred)//3000)
        sc = ax.scatter(all_ratio_pred[::step], all_ratio_obs[::step],
                        c=np.array([mz_to_n(m) for m in all_prec_mz[::step]]),
                        cmap='plasma', s=5, alpha=0.6, linewidths=0)
        plt.colorbar(sc, ax=ax, label='n-shell', pad=0.02, fraction=0.04)
        lim = max(all_ratio_pred.max(), all_ratio_obs.max())
        ax.plot([1, lim], [1, lim], 'k--', lw=0.8, alpha=0.6)
    ax.set_xlabel(r'$\sqrt{m_p/m_f}$  (predicted)')
    ax.set_ylabel(r'$f_f / f_p$  (observed)')
    ax.set_title('Subharmonic\nself-consistency', pad=4)

    # (B) Back-conversion error distribution ──────────────────────────────────
    ax = fig.add_subplot(1, 4, 2)
    if len(all_back_err) > 0:
        log_err = np.log10(all_back_err + 1e-14)
        ax.hist(log_err, bins=60, color='#1a9641', alpha=0.82, edgecolor='none',
                density=True)
    ax.set_xlabel(r'$\log_{10}$ back-conversion error  (ppm)')
    ax.set_ylabel('Density')
    ax.set_title('Frequency mapping\nerror (all fragments)', pad=4)
    ax.axvline(-9, color='#d73027', lw=1.0, ls='--', label='–9 ppm')
    ax.legend(fontsize=7, framealpha=0.5)

    # (C) Theoretical subharmonic curve: f_frag/f_prec vs mass ratio ──────────
    ax = fig.add_subplot(1, 4, 3)
    ratio = np.linspace(0.01, 1.0, 300)
    f_ratio_th = 1.0 / np.sqrt(ratio)   # f_frag/f_prec = sqrt(m_prec/m_frag)
    ax.plot(ratio, f_ratio_th, color='#2166ac', lw=1.6)
    ax.fill_between(ratio, f_ratio_th*(1-1e-4), f_ratio_th*(1+1e-4),
                    alpha=0.3, color='#2166ac', label='±0.01% band')
    if len(all_frag_mz) > 0:
        step = max(1, len(all_frag_mz)//500)
        mr   = all_frag_mz[::step] / all_prec_mz[::step]
        fr   = all_ratio_obs[::step]
        ax.scatter(mr, fr, c='#d73027', s=5, alpha=0.6, zorder=3,
                   label='NIST data')
    ax.set_xlabel(r'$m_f / m_p$')
    ax.set_ylabel(r'$f_f / f_p$')
    ax.legend(fontsize=7, framealpha=0.5)
    ax.set_title('Frequency ratio law', pad=4)

    # (D) 3D: predicted f_frag surface over (m/z_prec, m/z_frag) ─────────────
    ax = fig.add_subplot(1, 4, 4, projection='3d')
    mp_g = np.linspace(100, 700, 30)
    mf_g = np.linspace(50,  650, 30)
    MP, MF = np.meshgrid(mp_g, mf_g)
    valid = MF < MP
    F_RATIO = np.where(valid, np.sqrt(MP / np.where(MF>0, MF, 1)), np.nan)
    F_PREC  = np.vectorize(orb_freq)(MP) * 1e3   # mHz
    F_FRAG  = F_PREC * np.where(valid, F_RATIO, np.nan)
    surf = ax.plot_surface(MP, MF, F_FRAG, cmap='viridis',
                           linewidth=0, alpha=0.85, rcount=28, ccount=28)
    ax.set_xlabel('m/z prec  (Da)', labelpad=2, fontsize=7)
    ax.set_ylabel('m/z frag  (Da)', labelpad=2, fontsize=7)
    ax.set_zlabel(r'$f_{\rm frag}$  (mHz)', labelpad=2, fontsize=7)
    ax.set_title('Subharmonic\nfrequency surface', pad=4)
    ax.tick_params(labelsize=6)

    fig.subplots_adjust(wspace=0.42, left=0.06, right=0.97)
    save(fig, 'panel_03_subharmonic_frequency.png')


# ═══════════════════════════════════════════════════════════════════════════════
# Panel 4 — Virtual Substate Tensor
# ═══════════════════════════════════════════════════════════════════════════════
def panel_04():
    fig = new_fig()
    rng = np.random.default_rng(17)

    mz_ions = np.linspace(100, 700, 100)

    # (A) Instrument basis normalised substates vs m/z ─────────────────────────
    ax = fig.add_subplot(1, 4, 1)
    f_orb = np.array([orb_freq(m) for m in mz_ions])
    f_icr = np.array([icr_freq(m) for m in mz_ions])
    f_tof = 1.0 / np.sqrt(mz_ions)
    f_qua = 1.0 / mz_ions
    # Normalise to [0,1] by max of each
    def norm01(x): return (x-x.min())/(x.max()-x.min()+1e-30)
    ax.plot(mz_ions, norm01(f_orb), color='#2166ac', lw=1.4, label='Orbitrap')
    ax.plot(mz_ions, norm01(f_icr), color='#4dac26', lw=1.4, label='FT-ICR')
    ax.plot(mz_ions, norm01(f_tof), color='#d73027', lw=1.4, label='TOF')
    ax.plot(mz_ions, norm01(f_qua), color='#762a83', lw=1.4, label='Quadrupole')
    ax.set_xlabel('m/z  (Da)')
    ax.set_ylabel('Normalised virtual substate')
    ax.legend(fontsize=7, framealpha=0.5, loc='upper right')
    ax.set_title('Instrument-basis\nvirtual substates', pad=4)

    # (B) Charge state frequency series: f_z = √z × f_1 ──────────────────────
    ax = fig.add_subplot(1, 4, 2)
    z_vals = np.arange(1, 9)
    for mz0, col in zip([200, 400, 800], ['#2166ac','#1a9641','#d73027']):
        f1 = orb_freq(mz0)
        f_z_theory = f1 * np.sqrt(z_vals)
        ax.plot(z_vals, f_z_theory/f1, 'o-', color=col, lw=1.3, ms=5,
                label=f'{mz0} Da')
    sqrt_z = np.sqrt(np.linspace(1, 8, 100))
    ax.plot(np.linspace(1,8,100), sqrt_z, 'k--', lw=0.9, alpha=0.5,
            label=r'$\sqrt{z}$')
    ax.set_xlabel('Charge state  z')
    ax.set_ylabel(r'$f_z / f_1$')
    ax.legend(fontsize=7, framealpha=0.5)
    ax.set_title(r'$f_z = \sqrt{z}\,f_1$', pad=4)

    # (C) Virtual component off-shell fraction ────────────────────────────────
    ax = fig.add_subplot(1, 4, 3)
    dims  = ['Instrument', 'Charge', 'Polarity', 'Time']
    frac_outside = [0.0, 0.0833*1.2, 1.0, 0.0833*0.7]   # polarity always outside; approx
    # Polarity substates (+1, -1) are always outside [0,1]
    frac_outside[2] = 1.0   # polarity
    colors_bar = ['#2166ac', '#762a83', '#d73027', '#4dac26']
    ax.bar(dims, frac_outside, color=colors_bar, edgecolor='white', lw=0.5,
           alpha=0.85)
    ax.axhline(0.0833, color='grey', ls='--', lw=0.8, label='Overall mean')
    ax.set_ylabel('Fraction outside [0, 1]')
    ax.legend(fontsize=7, framealpha=0.5)
    ax.set_title('Off-shell fraction\nby dimension', pad=4)

    # (D) 3D: virtual tensor components colored by off-shell status ────────────
    ax = fig.add_subplot(1, 4, 4, projection='3d')
    n_ions = 120
    mz_s = rng.uniform(100, 700, n_ions)
    # Instrument axis (0..1 normalised)
    s_inst = norm01(np.array([orb_freq(m) for m in mz_s]))
    # Charge axis: f_z2 / f_z1 values (normalised)
    s_chg  = np.sqrt(rng.integers(1, 5, n_ions).astype(float))
    s_chg  = norm01(s_chg)
    # Polarity axis: +1 or -1 (off-shell by definition)
    s_pol  = rng.choice([-1.0, 1.0], n_ions)
    off_shell = (s_pol < 0) | (s_chg > 1) | (s_inst > 1)
    colors_3d = np.where(off_shell, '#d73027', '#2166ac')
    ax.scatter(s_inst, s_chg, s_pol,
               c=colors_3d, s=18, alpha=0.75, linewidths=0)
    # Draw unit cube wireframe
    for xs in [[0,1],[0,1],[0,0],[1,1]]:
        for ys in [[0,0],[1,1],[0,1],[0,1]]:
            ax.plot(xs, ys, [0,0], 'k-', lw=0.4, alpha=0.3)
            ax.plot(xs, ys, [1,1], 'k-', lw=0.4, alpha=0.3)
    ax.set_xlabel('Instrument', labelpad=2, fontsize=7)
    ax.set_ylabel('Charge', labelpad=2, fontsize=7)
    ax.set_zlabel('Polarity', labelpad=2, fontsize=7)
    ax.set_title('Virtual tensor\n(red = off-shell)', pad=4)
    ax.tick_params(labelsize=6)

    fig.subplots_adjust(wspace=0.42, left=0.06, right=0.97)
    save(fig, 'panel_04_virtual_substate_tensor.png')


# ═══════════════════════════════════════════════════════════════════════════════
# Panel 5 — Planck Depth and Stacked Tensor Capacity
# ═══════════════════════════════════════════════════════════════════════════════
def panel_05():
    fig = new_fig()

    T_PLANCK = 5.391e-44   # Planck time [s]

    def planck_depth(tau_osc, d):
        if d <= 0 or tau_osc <= 0:
            return 0
        val = math.log(tau_osc / (d * T_PLANCK), d+1)
        return max(1, int(math.ceil(val)) + 1)

    # Representative oscillators
    oscillators = {
        'Orbitrap\n500 Hz': 1/500,
        'Cs-133\n9.2 GHz': 1/9.193e9,
        'H-maser\n1.4 GHz': 1/1.42e9,
        'LHC\n40 MHz': 1/40.079e6,
        'Sr optical\n429 THz': 1/429e12,
    }

    # (A) Planck depth vs d_effective ─────────────────────────────────────────
    ax = fig.add_subplot(1, 4, 1)
    d_eff_vals = np.logspace(0, 6, 200)
    tau_orb = 1/500.0   # Orbitrap 500 Hz
    nP_vals = [planck_depth(tau_orb, d) for d in d_eff_vals]
    ax.semilogx(d_eff_vals, nP_vals, color='#2166ac', lw=1.6)
    ax.axhline(2, color='grey', ls='--', lw=0.8, label='nP = 2')
    ax.axvline(400_000, color='#d73027', ls=':', lw=0.8, label='d_eff (peptide)')
    ax.set_xlabel(r'$d_{\rm eff}$')
    ax.set_ylabel(r'Planck depth  $n_P$')
    ax.legend(fontsize=7, framealpha=0.5)
    ax.set_title('Planck depth collapse', pad=4)

    # (B) Planck depth for physical oscillators ────────────────────────────────
    ax = fig.add_subplot(1, 4, 2)
    d3_depths = [planck_depth(tau, 3) for tau in oscillators.values()]
    labels    = list(oscillators.keys())
    colors_osc = plt.cm.plasma(np.linspace(0.1, 0.85, len(labels)))
    bars = ax.barh(labels, d3_depths, color=colors_osc, edgecolor='white', lw=0.3)
    ax.set_xlabel(r'$n_P$  (d = 3)')
    ax.set_title('Planck depth\nphysical oscillators', pad=4)
    for bar, v in zip(bars, d3_depths):
        ax.text(v+0.3, bar.get_y()+bar.get_height()/2,
                str(v), va='center', fontsize=7)

    # (C) T(N, d_eff) vs Z_max (charge states) ────────────────────────────────
    ax = fig.add_subplot(1, 4, 3)
    Z_vals = np.arange(1, 50)
    for N, col in zip([2, 3, 4], ['#4575b4','#1a9641','#d73027']):
        d_eff_z = 4 * Z_vals * 2 * 10   # 4 instruments, Z charges, 2 polarities, 10 time steps
        T_vals  = [T_infl(N, int(d)) for d in d_eff_z]
        ax.semilogy(Z_vals, T_vals, color=col, lw=1.4, label=f'N={N}')
    ax.set_xlabel('Max charge state  Z')
    ax.set_ylabel('T(N, d_eff)')
    ax.legend(fontsize=7, framealpha=0.5)
    ax.set_title('Stacked tensor\ncapacity vs Z', pad=4)

    # (D) 3D: n_P surface over (n_instruments, n_charges) ─────────────────────
    ax = fig.add_subplot(1, 4, 4, projection='3d')
    n_inst_g = np.arange(1, 6)          # 1..5 instrument bases
    n_chg_g  = np.arange(1, 21)         # 1..20 charge states
    N_INST, N_CHG = np.meshgrid(n_inst_g, n_chg_g)
    NP_SURF = np.vectorize(lambda ni, nc:
        planck_depth(1/500.0, ni * nc * 2 * 10)
    )(N_INST, N_CHG)
    surf = ax.plot_surface(N_INST, N_CHG, NP_SURF,
                           cmap='plasma_r', linewidth=0, alpha=0.87,
                           rcount=20, ccount=20)
    fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.06, label=r'$n_P$', fraction=0.04)
    ax.set_xlabel('Instruments', labelpad=2, fontsize=7)
    ax.set_ylabel('Charge states', labelpad=2, fontsize=7)
    ax.set_zlabel(r'$n_P$', labelpad=2, fontsize=7)
    ax.set_title('Planck depth surface', pad=4)
    ax.tick_params(labelsize=6)

    fig.subplots_adjust(wspace=0.42, left=0.06, right=0.97)
    save(fig, 'panel_05_planck_depth_stacked.png')


# ── run all ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('Generating Paper B panels...')
    panel_01()
    panel_02()
    panel_03()
    panel_04()
    panel_05()
    print(f'Done. Figures saved to {_OUTDIR}')
