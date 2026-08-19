"""Panel 5 - Comparison method: raw-spectrum cosine similarity across NCE."""
import json
import numpy as np
import statistics as st
from collections import defaultdict
from panelstyle import (load, groups, new_panel, tag, save, _res,
                        BLUE, RED, GREEN, ORANGE, GREY, LIGHT, NCE)

coords, meta = load()
peaks = json.load(open(_res("_peaks.json")))
full = groups(coords, 9)


def cosine(a, b, tol=0.01):
    if not a or not b:
        return 0.0
    used, dot = set(), 0.0
    for m1, i1 in a:
        best, bj = tol, None
        for j, (m2, i2) in enumerate(b):
            if j in used:
                continue
            d = abs(m1 - m2)
            if d <= best:
                best, bj = d, j
        if bj is not None:
            used.add(bj)
            dot += i1 * b[bj][1]
    na = np.sqrt(sum(i * i for _, i in a))
    nb = np.sqrt(sum(i * i for _, i in b))
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


# Similarity matrix over NCE levels, averaged across compounds.
M = np.zeros((len(NCE), len(NCE)))
C = np.zeros((len(NCE), len(NCE)))
lag_sims = defaultdict(list)
sel = list(full)[:120]                     # bound the O(n^2) cost
for k in sel:
    v = {c["nce"]: c["scan_id"] for c in full[k]}
    for i, a in enumerate(NCE):
        for j, b in enumerate(NCE):
            if j < i or a not in v or b not in v:
                continue
            pa = peaks.get(str(v[a])) or peaks.get(v[a])
            pb = peaks.get(str(v[b])) or peaks.get(v[b])
            if not pa or not pb:
                continue
            s = cosine(pa, pb)
            M[i, j] += s
            M[j, i] += s
            C[i, j] += 1
            C[j, i] += 1
            lag_sims[abs(i - j)].append(s)
M = np.divide(M, np.maximum(C, 1))

fig, ax = new_panel(4, w=16.4, d3=(1,))

# --- A: similarity heat map over NCE pairs -----------------------------
im = ax[0].imshow(M, cmap="viridis", vmin=0, vmax=1, origin="lower")
ax[0].set_xticks(range(len(NCE)))
ax[0].set_xticklabels(NCE, fontsize=7)
ax[0].set_yticks(range(len(NCE)))
ax[0].set_yticklabels(NCE, fontsize=7)
ax[0].set_xlabel("NCE (%)")
ax[0].set_ylabel("NCE (%)")
ax[0].grid(False)
cb = fig.colorbar(im, ax=ax[0], pad=0.02, shrink=0.85, aspect=18)
cb.set_label("cosine similarity", fontsize=8)
cb.ax.tick_params(labelsize=7)
tag(ax[0], "A")

# --- B (3-D): the same surface, showing the ridge ----------------------
X, Y = np.meshgrid(np.arange(len(NCE)), np.arange(len(NCE)))
ax[1].plot_surface(X, Y, M, cmap="viridis", vmin=0, vmax=1,
                   edgecolor="none", alpha=0.95, rstride=1, cstride=1)
ax[1].set_xticks(range(0, len(NCE), 2))
ax[1].set_xticklabels([NCE[i] for i in range(0, len(NCE), 2)], fontsize=7)
ax[1].set_yticks(range(0, len(NCE), 2))
ax[1].set_yticklabels([NCE[i] for i in range(0, len(NCE), 2)], fontsize=7)
ax[1].set_xlabel("NCE (%)", labelpad=-3)
ax[1].set_ylabel("NCE (%)", labelpad=-3)
ax[1].set_zlabel("cosine", labelpad=-6)
ax[1].set_zlim(0, 1)
ax[1].view_init(elev=26, azim=-56)
ax[1].tick_params(pad=-1)
tag(ax[1], "B", d3=True)

# --- C: similarity decay with NCE separation ---------------------------
lags = sorted(lag_sims)
mu = [np.mean(lag_sims[l]) for l in lags]
sd = [np.std(lag_sims[l]) for l in lags]
ax[2].plot(lags, mu, "o-", color=BLUE, ms=5, mfc="white", mew=1.4)
ax[2].fill_between(lags, np.array(mu) - np.array(sd),
                   np.array(mu) + np.array(sd), color=BLUE, alpha=0.16)
ax[2].set_xlabel("separation in NCE levels")
ax[2].set_ylabel("cosine similarity")
ax[2].set_ylim(0, 1.05)
tag(ax[2], "C")

# --- D: coordinate distance vs spectral dissimilarity ------------------
xs, ys = [], []
for k in sel:
    v = {c["nce"]: c for c in full[k]}
    for i, a in enumerate(NCE):
        for b in NCE[i + 1:]:
            if a not in v or b not in v:
                continue
            pa = peaks.get(str(v[a]["scan_id"])) or peaks.get(v[a]["scan_id"])
            pb = peaks.get(str(v[b]["scan_id"])) or peaks.get(v[b]["scan_id"])
            if not pa or not pb:
                continue
            d = np.sqrt(sum((v[a][x] - v[b][x]) ** 2
                            for x in ("s_k", "s_t", "s_e")))
            xs.append(1 - cosine(pa, pb))
            ys.append(d)
xs, ys = np.array(xs), np.array(ys)
ax[3].scatter(xs, ys, s=4, color=ORANGE, alpha=0.28, linewidths=0)
m, b = np.polyfit(xs, ys, 1)
gx = np.linspace(0, xs.max(), 40)
ax[3].plot(gx, m * gx + b, "--", color=GREY, lw=1.4)
ax[3].set_xlabel("1 - cosine similarity")
ax[3].set_ylabel("coordinate distance")
tag(ax[3], "D")

save(fig, "panel5_baseline.png")
