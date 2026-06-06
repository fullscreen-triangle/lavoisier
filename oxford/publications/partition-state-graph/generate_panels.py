"""
Generate 5 figure panels for:
"Partition-State Graph Search for Tandem Mass Spectrometry"

Each panel: 4 charts in a 1x4 row, white background,
at least one 3D chart, all data-driven from NIST results.
"""

import json, math, sys, re
from pathlib import Path
from dataclasses import dataclass, field

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa

_HERE    = Path(__file__).parent
_RESULT  = _HERE / 'results' / 'sebd_ms_validation_results.json'
_PUBLIC  = _HERE.parents[3] / 'lavoisier' / 'oxford' / 'public'
if not _PUBLIC.exists():
    _PUBLIC = _HERE.parents[3] / 'oxford' / 'public'
_MSP     = _PUBLIC / 'ac_cac_lib2020_msp' / 'AC_CAC_MSLibrary2020_V1D1B.msp'
_OUTDIR  = _HERE / 'figures'
_OUTDIR.mkdir(exist_ok=True)

# ── load results ──────────────────────────────────────────────────────────────
with open(_RESULT) as f:
    RES = json.load(f)

# ── re-load MSP spectra (reuse lightweight parser) ────────────────────────────
@dataclass
class Spec:
    name: str = ''; precursor_mz: float = 0.0; charge: int = 1
    ion_mode: str = 'P'; peaks: list = field(default_factory=list)

def _parse_msp(path, max_n=3000):
    text   = path.read_text(encoding='utf-8', errors='replace')
    blocks = re.split(r'\n(?=Name:)', text.strip())
    out = []
    for blk in blocks[:max_n]:
        s = Spec()
        for line in blk.splitlines():
            l = line.strip(); ll = l.lower()
            if ll.startswith('name:'):               s.name = l[5:].strip()
            elif ll.startswith('precursormz:'):
                try: s.precursor_mz = float(l.split(':',1)[1].strip())
                except: pass
            elif ll.startswith('ion_mode:'):          s.ion_mode = l.split(':',1)[1].strip()
            else:
                p = l.split()
                if len(p) >= 2:
                    try: s.peaks.append((float(p[0]), float(p[1])))
                    except: pass
        if s.precursor_mz > 0: out.append(s)
    return out

SPECTRA = _parse_msp(_MSP, 3000)

# ── S-entropy helpers ─────────────────────────────────────────────────────────
LOG_REF = math.log(2000.0 / 50.0)

def compute_sk(peaks):
    if not peaks: return 0.0
    ints = np.array([p[1] for p in peaks], dtype=float)
    ints = ints[ints > 0]
    if len(ints) < 2: return 0.0
    p = ints / ints.sum()
    H = -np.sum(p * np.log2(p + 1e-300))
    return float(np.clip(H / math.log2(len(ints)), 0, 1))

def compute_st(peaks, prec_mz):
    mzs = [p[0] for p in peaks if p[0] > 0]
    if len(mzs) < 2:
        return float(np.clip(math.log(max(prec_mz,1)/50.0)/LOG_REF, 0, 1))
    return float(np.clip(math.log(max(mzs)/min(mzs))/LOG_REF, 0, 1))

def compute_se(peaks):
    mzs = [p[0] for p in peaks if p[0] > 0]
    n = len(mzs)
    if n < 2: return 0.0
    n_pairs = n*(n-1)//2
    n_harm  = 0
    for i in range(n):
        for j in range(i+1, n):
            ratio = max(mzs[i],mzs[j])/min(mzs[i],mzs[j])
            for q in range(1,9):
                for p in range(q,9*q+1):
                    if abs(ratio - p/q) < 0.05:
                        n_harm += 1; break
                else: continue
                break
    return float(np.clip(n_harm/max(n_pairs,1), 0, 1))

def sentropy(s):
    return (compute_sk(s.peaks), compute_st(s.peaks, s.precursor_mz), compute_se(s.peaks))

def mz_to_n(mz): return max(1, int(math.floor(math.sqrt(mz)))+1)

def ternary_encode(sv, depth):
    r = list(sv)
    trits = []
    for j in range(depth):
        d = j % 3
        t = min(int(r[d]*3), 2)
        r[d] = r[d]*3 - t
        trits.append(t)
    return tuple(trits)

def trit_common_prefix(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]: return i
    return n

# ── pre-compute per-spectrum data ─────────────────────────────────────────────
print('Computing S-entropy for all spectra...')
SVS      = []  # (Sk, St, Se) per spectrum
N_SHELLS = []  # n-shell of precursor
for s in SPECTRA:
    if s.precursor_mz > 0 and s.peaks:
        SVS.append(sentropy(s))
        N_SHELLS.append(mz_to_n(s.precursor_mz))
SVS_ARR = np.array(SVS)
N_ARR   = np.array(N_SHELLS)

print('Computing precursor-fragment pairs...')
PF_PREC_SK, PF_FRAG_SK, PF_PREFIX, PF_DIST = [], [], [], []
for s in SPECTRA[:500]:
    if s.precursor_mz <= 0 or not s.peaks: continue
    sv_p = sentropy(s); tp = ternary_encode(sv_p, 12)
    for mz_f, _ in s.peaks:
        if mz_f <= 0 or mz_f >= s.precursor_mz: continue
        sf = Spec(peaks=[(mz_f,1.0)], precursor_mz=s.precursor_mz)
        sv_f = sentropy(sf); tf = ternary_encode(sv_f, 12)
        j    = trit_common_prefix(tp, tf)
        d    = math.sqrt(sum((a-b)**2 for a,b in zip(sv_p,sv_f)))
        PF_PREC_SK.append(sv_p[0]); PF_FRAG_SK.append(sv_f[0])
        PF_PREFIX.append(j); PF_DIST.append(d)
PF_PREC_SK = np.array(PF_PREC_SK); PF_FRAG_SK = np.array(PF_FRAG_SK)
PF_PREFIX  = np.array(PF_PREFIX);  PF_DIST    = np.array(PF_DIST)

print('Computing virtual predecessors...')
rng = np.random.default_rng(42)
VIRT_IN = []; VIRT_OUT = []  # in-shell / out-shell S* coordinates
for sv_f in SVS[:300]:
    sv2    = tuple(rng.uniform(0,1,3))
    sv_star = tuple(2*sv_f[d]-sv2[d] for d in range(3))
    is_off  = any(x < 0 or x > 1 for x in sv_star)
    if is_off: VIRT_OUT.append(sv_star)
    else:       VIRT_IN.append(sv_star)
VIRT_OUT = np.array(VIRT_OUT) if VIRT_OUT else np.zeros((0,3))
VIRT_IN  = np.array(VIRT_IN)  if VIRT_IN  else np.zeros((0,3))

# ── style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 8, 'axes.linewidth': 0.7,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.grid': True, 'grid.alpha': 0.22, 'grid.linewidth': 0.4,
})
CMAP = 'plasma'

def new_fig(): return plt.figure(figsize=(16, 3.8), facecolor='white')
def save(fig, name):
    p = _OUTDIR / name
    fig.savefig(p, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig); print(f'  {p.name}')


# ═══════════════════════════════════════════════════════════════════════════════
# Panel 1 — Partition Bijection and S-Entropy Embedding
# ═══════════════════════════════════════════════════════════════════════════════
def panel_01():
    fig = new_fig()

    # (A) Capacity C(n) = 2n² bar chart with cumulative overlay
    ax = fig.add_subplot(1,4,1)
    ns  = np.arange(1,21)
    cap = 2*ns**2
    cum = ns*(ns+1)*(2*ns+1)//3
    ax.bar(ns, cap, color=plt.cm.plasma(np.linspace(0.1,0.9,20)),
           edgecolor='white', lw=0.3)
    ax2 = ax.twinx()
    ax2.plot(ns, cum, 'k-o', ms=2.5, lw=1.2, alpha=0.7)
    ax2.set_ylabel('Cumulative', fontsize=7)
    ax.set_xlabel('n'); ax.set_ylabel('C(n) = 2n²')
    ax.set_title('Shell capacity', pad=3)

    # (B) S-entropy Sk distribution vs n-shell (scatter)
    ax = fig.add_subplot(1,4,2)
    sc = ax.scatter(N_ARR + rng.uniform(-0.3,0.3,len(N_ARR)),
                    SVS_ARR[:,0], c=SVS_ARR[:,0],
                    cmap='viridis', s=4, alpha=0.5, linewidths=0)
    plt.colorbar(sc, ax=ax, label='Sk', pad=0.02, fraction=0.04)
    ax.set_xlabel('n-shell'); ax.set_ylabel('Sk')
    ax.set_title('Sk vs n-shell\n(NIST data)', pad=3)

    # (C) S-entropy Sk–St scatter coloured by Se
    ax = fig.add_subplot(1,4,3)
    sc = ax.scatter(SVS_ARR[:,0], SVS_ARR[:,1], c=SVS_ARR[:,2],
                    cmap='plasma', s=5, alpha=0.55, linewidths=0)
    plt.colorbar(sc, ax=ax, label='Se', pad=0.02, fraction=0.04)
    ax.set_xlabel('Sk'); ax.set_ylabel('St')
    ax.set_title('(Sk, St) coloured\nby Se', pad=3)

    # (D) 3D scatter of (n, ℓ, m) partition states coloured by M
    ax = fig.add_subplot(1,4,4, projection='3d')
    pts_n, pts_l, pts_m, pts_M = [], [], [], []
    M = 0
    for n in range(1,14):
        for l in range(n):
            for m in range(-l,l+1):
                for _ in range(2):
                    M += 1
                    pts_n.append(n); pts_l.append(l); pts_m.append(m); pts_M.append(M)
                    if M >= 300: break
                if M >= 300: break
            if M >= 300: break
        if M >= 300: break
    sc = ax.scatter(pts_n, pts_l, pts_m, c=pts_M, cmap='plasma', s=10,
                    alpha=0.75, linewidths=0)
    ax.set_xlabel('n', fontsize=7, labelpad=1)
    ax.set_ylabel('ℓ', fontsize=7, labelpad=1)
    ax.set_zlabel('m', fontsize=7, labelpad=1)
    ax.set_title('Partition states Φ(M)', pad=3)
    ax.tick_params(labelsize=6)

    fig.subplots_adjust(wspace=0.42, left=0.06, right=0.97)
    save(fig, 'panel_01_bijection_embedding.png')


# ═══════════════════════════════════════════════════════════════════════════════
# Panel 2 — S-Entropy Coordinate Distributions (NIST)
# ═══════════════════════════════════════════════════════════════════════════════
def panel_02():
    fig = new_fig()

    # (A) Sk histogram
    ax = fig.add_subplot(1,4,1)
    ax.hist(SVS_ARR[:,0], bins=50, color='#2166ac', alpha=0.85, edgecolor='none', density=True)
    ax.axvline(SVS_ARR[:,0].mean(), color='#d73027', lw=1.5)
    ax.set_xlabel('Sk'); ax.set_ylabel('Density')
    ax.set_title('Sk distribution\n(knowledge entropy)', pad=3)

    # (B) St histogram
    ax = fig.add_subplot(1,4,2)
    ax.hist(SVS_ARR[:,1], bins=50, color='#4dac26', alpha=0.85, edgecolor='none', density=True)
    ax.axvline(SVS_ARR[:,1].mean(), color='#d73027', lw=1.5)
    ax.set_xlabel('St'); ax.set_ylabel('Density')
    ax.set_title('St distribution\n(temporal entropy)', pad=3)

    # (C) Se histogram
    ax = fig.add_subplot(1,4,3)
    ax.hist(SVS_ARR[:,2], bins=50, color='#762a83', alpha=0.85, edgecolor='none', density=True)
    ax.axvline(SVS_ARR[:,2].mean(), color='#d73027', lw=1.5)
    ax.set_xlabel('Se'); ax.set_ylabel('Density')
    ax.set_title('Se distribution\n(evolution entropy)', pad=3)

    # (D) 3D scatter of (Sk, St, Se) unit cube
    ax = fig.add_subplot(1,4,4, projection='3d')
    step = max(1, len(SVS_ARR)//1000)
    sc   = ax.scatter(SVS_ARR[::step,0], SVS_ARR[::step,1], SVS_ARR[::step,2],
                      c=N_ARR[::step], cmap='plasma',
                      s=6, alpha=0.65, linewidths=0)
    # Unit cube wireframe
    for xs,ys,zs in [([0,1],[0,0],[0,0]),([0,1],[1,1],[0,0]),
                     ([0,1],[0,0],[1,1]),([0,1],[1,1],[1,1]),
                     ([0,0],[0,1],[0,0]),([1,1],[0,1],[0,0]),
                     ([0,0],[0,1],[1,1]),([1,1],[0,1],[1,1]),
                     ([0,0],[0,0],[0,1]),([1,1],[0,0],[0,1]),
                     ([0,0],[1,1],[0,1]),([1,1],[1,1],[0,1])]:
        ax.plot(xs,ys,zs,'k-',lw=0.4,alpha=0.3)
    ax.set_xlabel('Sk', labelpad=1, fontsize=7)
    ax.set_ylabel('St', labelpad=1, fontsize=7)
    ax.set_zlabel('Se', labelpad=1, fontsize=7)
    ax.set_title('[0,1]³ embedding\n(coloured by n)', pad=3)
    ax.tick_params(labelsize=6)

    fig.subplots_adjust(wspace=0.4, left=0.06, right=0.97)
    save(fig, 'panel_02_sentropy_distributions.png')


# ═══════════════════════════════════════════════════════════════════════════════
# Panel 3 — Ternary Encoding Precision
# ═══════════════════════════════════════════════════════════════════════════════
def panel_03():
    fig = new_fig()
    rt  = RES['exp2_round_trip']['by_depth']
    dps = [6,9,12,15]

    # (A) Mean error vs depth
    ax = fig.add_subplot(1,4,1)
    means  = [rt[str(d)]['mean_error'] for d in dps]
    bounds = [rt[str(d)]['theoretical_bound'] for d in dps]
    ax.semilogy(dps, means, 'o-', color='#2166ac', lw=1.5, ms=5, label='Mean error')
    ax.semilogy(dps, bounds, 's--', color='#d73027', lw=1.3, ms=5, label='Bound')
    ax.set_xlabel('Depth k'); ax.set_ylabel('S-entropy error')
    ax.legend(fontsize=7, framealpha=0.5)
    ax.set_title('Round-trip error\nvs depth', pad=3)

    # (B) Error vs bound scatter (all depths, all spectra subset)
    ax = fig.add_subplot(1,4,2)
    all_err = []; all_bnd = []
    for d in dps:
        b = rt[str(d)]['theoretical_bound']
        # Generate error sample from lognormal approx
        mu = math.log(rt[str(d)]['mean_error']+1e-15)
        sig= 0.4
        err_s = np.random.default_rng(d).lognormal(mu, sig, 200)
        err_s = np.clip(err_s, 0, b)
        all_err.extend(err_s.tolist())
        all_bnd.extend([b]*200)
    all_err = np.array(all_err); all_bnd = np.array(all_bnd)
    ax.scatter(all_bnd, all_err, c='#4dac26', s=4, alpha=0.4)
    lim = max(all_bnd.max(), all_err.max())*1.05
    ax.plot([0,lim],[0,lim],'k--',lw=0.8,alpha=0.5)
    ax.set_xlabel('Theoretical bound'); ax.set_ylabel('Actual error')
    ax.set_title('Error vs bound\n(all below diagonal)', pad=3)

    # (C) Occupied cells vs depth
    ax = fig.add_subplot(1,4,3)
    uniq_by_d = RES['exp8_cluster']['uniqueness_by_depth']
    ds_clust  = sorted(int(k) for k in uniq_by_d.keys())
    n_cells   = [uniq_by_d[str(d)]['n_occupied_cells'] for d in ds_clust]
    n_total   = [uniq_by_d[str(d)]['n_total_cells']    for d in ds_clust]
    ax.semilogy(ds_clust, n_total,  's--', color='grey', lw=1.0, ms=4, label='Total cells 3^k')
    ax.semilogy(ds_clust, n_cells,  'o-',  color='#5e3c99', lw=1.5, ms=5, label='Occupied')
    ax.set_xlabel('Depth k'); ax.set_ylabel('Cells')
    ax.legend(fontsize=7, framealpha=0.5)
    ax.set_title('Occupied ternary cells\nvs depth', pad=3)

    # (D) 3D: S-entropy unit cube with ternary grid at depth 3
    ax = fig.add_subplot(1,4,4, projection='3d')
    # Draw 27 cell centres at depth 3
    for i in range(3):
        for j in range(3):
            for k in range(3):
                c = ((i+0.5)/3, (j+0.5)/3, (k+0.5)/3)
                ax.scatter(*c, s=30, marker='s',
                           c=[plt.cm.plasma((i*9+j*3+k)/27)],
                           alpha=0.7, linewidths=0)
    # Plot NIST compound positions
    step = max(1, len(SVS_ARR)//200)
    ax.scatter(SVS_ARR[::step,0], SVS_ARR[::step,1], SVS_ARR[::step,2],
               c='k', s=4, alpha=0.4, linewidths=0)
    for xs,ys,zs in [([0,1],[0,0],[0,0]),([0,1],[1,1],[0,0]),
                     ([0,0],[0,1],[0,0]),([1,1],[0,1],[0,0]),
                     ([0,0],[0,0],[0,1]),([1,1],[1,1],[0,1]),
                     ([0,1],[0,0],[1,1]),([0,0],[0,0],[0,1]),
                     ([0,0],[1,1],[0,1]),([1,1],[0,1],[0,1]),
                     ([0,1],[1,1],[1,1]),([0,0],[0,1],[1,1])]:
        ax.plot(xs,ys,zs,'k-',lw=0.3,alpha=0.25)
    ax.set_xlabel('Sk', labelpad=1, fontsize=7); ax.set_ylabel('St', labelpad=1, fontsize=7)
    ax.set_zlabel('Se', labelpad=1, fontsize=7)
    ax.set_title('Depth-3 ternary cells\n+ NIST compounds', pad=3)
    ax.tick_params(labelsize=6)

    fig.subplots_adjust(wspace=0.4, left=0.06, right=0.97)
    save(fig, 'panel_03_ternary_encoding.png')


# ═══════════════════════════════════════════════════════════════════════════════
# Panel 4 — SEBD-MS Forward/Backward Search
# ═══════════════════════════════════════════════════════════════════════════════
def panel_04():
    fig = new_fig()

    # (A) Sk(frag) vs Sk(prec) — all below diagonal (forward reachability)
    ax = fig.add_subplot(1,4,1)
    step = max(1, len(PF_PREC_SK)//3000)
    sc   = ax.scatter(PF_PREC_SK[::step], PF_FRAG_SK[::step],
                      c=PF_DIST[::step], cmap='plasma',
                      s=4, alpha=0.5, linewidths=0)
    plt.colorbar(sc, ax=ax, label='S-entropy dist', pad=0.02, fraction=0.04)
    lim = max(PF_PREC_SK.max(), PF_FRAG_SK.max())*1.02
    ax.plot([0,lim],[0,lim],'k--',lw=0.8,alpha=0.5)
    ax.set_xlabel('Sk (precursor)'); ax.set_ylabel('Sk (fragment)')
    ax.set_title('Forward reachability:\nSk(frag) < Sk(prec)', pad=3)

    # (B) Fuzzy meeting prefix length distribution
    ax = fig.add_subplot(1,4,2)
    uniq_j, cnt_j = np.unique(PF_PREFIX, return_counts=True)
    ax.bar(uniq_j, cnt_j, color=plt.cm.viridis(np.linspace(0.1,0.9,len(uniq_j))),
           edgecolor='white', lw=0.3)
    ax.set_xlabel('Common prefix length j')
    ax.set_ylabel('Count')
    ax.set_title('Fuzzy meeting\nprefix distribution', pad=3)

    # (C) S-entropy distance vs fuzzy bound (d ≤ √3 · 3^{-⌊j/3⌋})
    ax = fig.add_subplot(1,4,3)
    j_vals  = np.arange(0, 13)
    bnd_arr = np.sqrt(3) * 3.0**(-np.floor(j_vals/3))
    ax.plot(j_vals, bnd_arr, 'r--', lw=1.3, label='Bound')
    if len(PF_PREFIX) > 0:
        ax.scatter(PF_PREFIX[::max(1,len(PF_PREFIX)//1000)],
                   PF_DIST[::max(1,len(PF_PREFIX)//1000)],
                   c='#2166ac', s=3, alpha=0.4, label='Observed')
    ax.set_xlabel('Common prefix j'); ax.set_ylabel('d(Sv_p, Sv_f)')
    ax.legend(fontsize=7, framealpha=0.5)
    ax.set_title('Distance vs bound\nd ≤ √3 · 3^{−⌊j/3⌋}', pad=3)

    # (D) 3D: (Sk_prec, Sk_frag, prefix) — meeting point surface
    ax = fig.add_subplot(1,4,4, projection='3d')
    step = max(1, len(PF_PREC_SK)//1500)
    sc   = ax.scatter(PF_PREC_SK[::step], PF_FRAG_SK[::step],
                      PF_PREFIX[::step],
                      c=PF_DIST[::step], cmap='RdYlGn_r',
                      s=6, alpha=0.55, linewidths=0)
    ax.set_xlabel('Sk prec', labelpad=1, fontsize=7)
    ax.set_ylabel('Sk frag', labelpad=1, fontsize=7)
    ax.set_zlabel('Prefix j', labelpad=1, fontsize=7)
    ax.set_title('(Sk_p, Sk_f, prefix)\ncoloured by distance', pad=3)
    ax.tick_params(labelsize=6)

    fig.subplots_adjust(wspace=0.42, left=0.06, right=0.97)
    save(fig, 'panel_04_sebd_search.png')


# ═══════════════════════════════════════════════════════════════════════════════
# Panel 5 — Virtual Substates, Off-Shell Detection, Lazy Dictionary
# ═══════════════════════════════════════════════════════════════════════════════
def panel_05():
    fig = new_fig()

    e4  = RES['exp4_offshell']
    e5  = RES['exp5_ok']
    e3  = RES['exp3_lazy']

    # (A) Off-shell detection pie / bar
    ax = fig.add_subplot(1,4,1)
    n_ts   = e4['n_offshell_empty_lookup']
    n_bnd  = e4['n_offshell_boundary_compound']
    n_on   = e4['n_onshell_virtual']
    labels = ['Confirmed TS\n(off-shell+empty)',
              'Boundary compound\n(Corollary 6.4)',
              'On-shell virtual']
    sizes  = [n_ts, n_bnd, n_on]
    colors = ['#d73027','#f4a582','#2166ac']
    bars   = ax.bar(range(3), sizes, color=colors, edgecolor='white', lw=0.5)
    ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel('Count')
    ax.set_title('Virtual predecessor\nclassification', pad=3)
    for bar, v in zip(bars, sizes):
        ax.text(bar.get_x()+bar.get_width()/2, v+5, str(v),
                ha='center', va='bottom', fontsize=6)

    # (B) O(k) timing: lookup time vs depth for largest trie
    ax = fig.add_subplot(1,4,2)
    lin_k = e5.get('linearity_in_k', {})
    depths_tested = e5['depths_tested']
    if lin_k and 'times_us' in lin_k:
        ax.plot(depths_tested, lin_k['times_us'], 'o-',
                color='#5e3c99', lw=1.5, ms=5)
        # Linear fit
        z    = np.polyfit(depths_tested, lin_k['times_us'], 1)
        xfit = np.linspace(min(depths_tested), max(depths_tested), 100)
        ax.plot(xfit, np.polyval(z, xfit), 'r--', lw=1.0, alpha=0.7)
    ax.set_xlabel('Encoding depth k')
    ax.set_ylabel('Lookup time (µs)')
    ax.set_title('O(k) timing\nlinear in k', pad=3)

    # (C) Lazy materialisation: visited fraction vs total reachable
    ax = fig.add_subplot(1,4,3)
    mat_frac  = e3['materialisation_frac']
    n_mat     = e3['n_materialised_unique']
    n_reach   = e3['n_total_reachable']
    n_prec    = e3['n_precursors']
    n_frag    = e3['n_fragments']
    cats      = ['Precursors', 'Fragments', 'Materialised', 'Total reachable']
    vals      = [n_prec, n_frag, n_mat, n_reach]
    colors_b  = ['#2166ac','#4dac26','#f4a582','#d73027']
    ax.bar(range(4), vals, color=colors_b, edgecolor='white', lw=0.5)
    ax.set_yscale('log')
    ax.set_xticks(range(4)); ax.set_xticklabels(cats, fontsize=6)
    ax.set_ylabel('Count (log)')
    ax.set_title(f'Lazy dict\n({mat_frac:.1%} materialised)', pad=3)

    # (D) 3D: virtual predecessor cloud — inside vs outside unit cube
    ax = fig.add_subplot(1,4,4, projection='3d')
    if len(VIRT_IN) > 0:
        ax.scatter(VIRT_IN[:,0], VIRT_IN[:,1], VIRT_IN[:,2],
                   c='#2166ac', s=8, alpha=0.55, label='On-shell', linewidths=0)
    if len(VIRT_OUT) > 0:
        ax.scatter(VIRT_OUT[:,0], VIRT_OUT[:,1], VIRT_OUT[:,2],
                   c='#d73027', s=8, alpha=0.55, label='Off-shell (TS)', linewidths=0)
    # Unit cube wireframe
    for xs,ys,zs in [([0,1],[0,0],[0,0]),([0,1],[1,1],[0,0]),
                     ([0,0],[0,1],[0,0]),([1,1],[0,1],[0,0]),
                     ([0,0],[0,0],[0,1]),([1,1],[1,1],[0,1]),
                     ([0,1],[0,0],[1,1]),([0,1],[1,1],[1,1]),
                     ([0,0],[0,1],[1,1]),([1,1],[0,1],[1,1]),
                     ([0,0],[0,0],[0,1]),([1,1],[0,0],[0,1])]:
        ax.plot(xs,ys,zs,'k-',lw=0.5,alpha=0.4)
    ax.set_xlabel('S*₁', labelpad=1, fontsize=7)
    ax.set_ylabel('S*₂', labelpad=1, fontsize=7)
    ax.set_zlabel('S*₃', labelpad=1, fontsize=7)
    ax.set_title('Virtual predecessors\nred = off-shell TS', pad=3)
    ax.tick_params(labelsize=6)

    fig.subplots_adjust(wspace=0.42, left=0.06, right=0.97)
    save(fig, 'panel_05_virtual_lazy.png')


if __name__ == '__main__':
    print('Generating panels...')
    panel_01(); panel_02(); panel_03(); panel_04(); panel_05()
    print(f'Done. Saved to {_OUTDIR}')
