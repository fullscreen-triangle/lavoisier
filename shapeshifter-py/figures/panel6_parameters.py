"""Panel 6 - Parameter dependence: does any (alpha, beta, k) meet the criterion?"""
import numpy as np
import statistics as st
from collections import defaultdict
from panelstyle import (new_panel, tag, save, BLUE, RED, GREEN, ORANGE,
                        PURPLE, GREY, LIGHT, AXLAB)
import sys
sys.path.insert(0, "..")
from shapeshifter.stdlib import op_read_msp, op_sentropy
from shapeshifter.parser import AST

log = lambda a, b: None
ast = AST()
import os
_MSP = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..",
    "oxford", "public", "ac_cac_lib2020_msp",
    "AC_CAC_MSLibrary2020_V1D1B.msp"))
ast.datasets = {"D": {"files": [_MSP]}}
scans, _ = op_read_msp({"dataset": "D", "min_peaks": 3}, {}, ast, log)
AX = ("s_k", "s_t", "s_e")


def ratio_for(alpha, beta, k, axes=AX):
    co, _ = op_sentropy({"scans": scans, "alpha": alpha, "beta": beta,
                         "k_neighbors": k}, {}, ast, log)
    g = defaultdict(list)
    for c in co:
        g[c["compound"]].append(c)
    full = {a: b for a, b in g.items() if len(b) >= 9}
    wit, cen = [], []
    for v in full.values():
        c0 = {a: st.mean([i[a] for i in v]) for a in axes}
        cen.append(c0)
        ds = [np.sqrt(sum((v[i][a] - v[j][a]) ** 2 for a in axes))
              for i in range(len(v)) for j in range(i + 1, len(v))]
        wit.append(np.mean(ds))
    bt = [np.sqrt(sum((cen[i][a] - cen[j][a]) ** 2 for a in axes))
          for i in range(len(cen)) for j in range(i + 1, len(cen))]
    return float(np.mean(bt) / np.mean(wit))


# --- sweeps --------------------------------------------------------------
alphas = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
betas = [0.25, 0.5, 1.0, 2.0, 4.0]
ks = [2, 3, 5, 8, 12, 20]

r_alpha = [ratio_for(a, 1.0, 5) for a in alphas]
r_beta = [ratio_for(1.0, b, 5) for b in betas]
r_k = [ratio_for(1.0, 1.0, k) for k in ks]

# 3-D surface over (alpha, beta)
A, B = np.meshgrid(alphas, betas)
Z = np.zeros_like(A, dtype=float)
for i, b in enumerate(betas):
    for j, a in enumerate(alphas):
        Z[i, j] = ratio_for(a, b, 5)

# axis-subset comparison
subsets = [("s_k",), ("s_t",), ("s_e",), ("s_k", "s_t"),
           ("s_k", "s_e"), ("s_t", "s_e"), AX]
r_sub = [ratio_for(1.0, 1.0, 5, axes=s) for s in subsets]

fig, ax = new_panel(4, w=16.0, d3=(2,))

# --- A: alpha and k sweeps ----------------------------------------------
ax[0].plot(alphas, r_alpha, "o-", color=BLUE, ms=5, mfc="white", mew=1.4,
           label=r"$\alpha$ (with $\beta$=1, k=5)")
ax0b = ax[0].twiny()
ax0b.plot(ks, r_k, "s--", color=ORANGE, ms=5, mfc="white", mew=1.4,
          label="k")
ax0b.set_xlabel("k neighbours", color=ORANGE)
ax0b.tick_params(axis="x", colors=ORANGE)
ax[0].axhline(2.0, color=RED, ls="--", lw=1.4)
ax[0].axhline(1.0, color="black", ls=":", lw=0.9)
ax[0].set_xlabel(r"$\alpha$", color=BLUE)
ax[0].tick_params(axis="x", colors=BLUE)
ax[0].set_ylabel("separation ratio")
ax[0].set_ylim(0, 2.3)
tag(ax[0], "A")

# --- B: beta sweep -------------------------------------------------------
ax[1].plot(betas, r_beta, "o-", color=GREEN, ms=5.5, mfc="white", mew=1.4)
ax[1].fill_between(betas, 0, r_beta, color=GREEN, alpha=0.14)
ax[1].axhline(2.0, color=RED, ls="--", lw=1.4)
ax[1].axhline(1.0, color="black", ls=":", lw=0.9)
ax[1].set_xscale("log")
ax[1].set_xlabel(r"$\beta$ (log)")
ax[1].set_ylabel("separation ratio")
ax[1].set_ylim(0, 2.3)
tag(ax[1], "B")

# --- C (3-D): ratio surface over (alpha, beta), with the threshold plane -
ax[2].plot_surface(A, B, Z, cmap="coolwarm", vmin=0.5, vmax=2.0,
                   edgecolor="none", alpha=0.95, rstride=1, cstride=1)
thr = np.full_like(Z, 2.0)
ax[2].plot_surface(A, B, thr, color=RED, alpha=0.13, edgecolor="none")
ax[2].set_xlabel(r"$\alpha$", labelpad=-3)
ax[2].set_ylabel(r"$\beta$", labelpad=-3)
ax[2].set_zlabel("ratio", labelpad=-5)
ax[2].set_zlim(0, 2.3)
ax[2].view_init(elev=24, azim=-58)
ax[2].tick_params(pad=-1)
tag(ax[2], "C", d3=True)

# --- D: which axis subset does best -------------------------------------
lab = ["+".join(AXLAB[a] for a in s) for s in subsets]
order = np.argsort(r_sub)
cols = [PURPLE if len(subsets[i]) == 3 else
        (BLUE if len(subsets[i]) == 1 else GREEN) for i in order]
ax[3].barh(range(len(order)), [r_sub[i] for i in order], color=cols,
           height=0.6, edgecolor="white")
ax[3].axvline(2.0, color=RED, ls="--", lw=1.4)
ax[3].axvline(1.0, color="black", ls=":", lw=0.9)
ax[3].set_yticks(range(len(order)))
ax[3].set_yticklabels([lab[i] for i in order], fontsize=7.5)
ax[3].set_xlabel("separation ratio")
ax[3].set_xlim(0, 2.3)
tag(ax[3], "D")

save(fig, "panel6_parameters.png")
print("alpha:", [round(v, 3) for v in r_alpha])
print("beta :", [round(v, 3) for v in r_beta])
print("k    :", [round(v, 3) for v in r_k])
print("subs :", [(l, round(v, 3)) for l, v in zip(lab, r_sub)])
