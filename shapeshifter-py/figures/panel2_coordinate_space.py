"""Panel 2 - The coordinate space: occupancy, marginals, and mass relation."""
import numpy as np
import statistics as st
from panelstyle import (load, groups, new_panel, tag, save, AXCOL, AXLAB,
                        BLUE, RED, GREEN, GREY, LIGHT, NCE)

coords, meta = load()

sk = np.array([c["s_k"] for c in coords])
stt = np.array([c["s_t"] for c in coords])
se = np.array([c["s_e"] for c in coords])
nce = np.array([c["nce"] if c["nce"] else 0 for c in coords])
mz = np.array([c["precursor_mz"] for c in coords])
npk = np.array([c["n_peaks"] for c in coords])

fig, ax = new_panel(4, d3=(0,))

# --- A (3-D): occupancy of the coordinate cube, coloured by NCE ---------
s = ax[0].scatter(sk, stt, se, c=nce, cmap="plasma", s=3.5, alpha=0.55,
                  linewidths=0, depthshade=False)
ax[0].set_xlabel(AXLAB["s_k"], labelpad=-4)
ax[0].set_ylabel(AXLAB["s_t"], labelpad=-4)
ax[0].set_zlabel(AXLAB["s_e"], labelpad=-4)
ax[0].view_init(elev=20, azim=-62)
ax[0].tick_params(pad=-1)
cb = fig.colorbar(s, ax=ax[0], pad=0.10, shrink=0.62, aspect=14)
cb.set_label("NCE (%)", fontsize=8)
cb.ax.tick_params(labelsize=7)
tag(ax[0], "A", d3=True)

# --- B: marginal densities of the three axes ----------------------------
for a, v in (("s_k", sk), ("s_t", stt), ("s_e", se)):
    lo, hi = np.percentile(v, [0.2, 99.8])
    grid = np.linspace(lo, hi, 220)
    # Gaussian KDE by hand (no scipy dependency needed for one array)
    h = 1.06 * v.std() * len(v) ** (-1 / 5)
    d = np.exp(-0.5 * ((grid[:, None] - v[None, :]) / h) ** 2).sum(1)
    d /= d.max()
    ax[1].plot(grid, d, color=AXCOL[a], label=AXLAB[a])
    ax[1].fill_between(grid, 0, d, color=AXCOL[a], alpha=0.13)
ax[1].set_xlabel("coordinate value")
ax[1].set_ylabel("density (scaled)")
ax[1].set_xlim(-0.4, 10.2)
ax[1].legend(loc="upper right")
tag(ax[1], "B")

# --- C: S_k against precursor m/z, coloured by peak count ---------------
sc = ax[2].scatter(mz, sk, c=npk, cmap="viridis", s=5, alpha=0.6,
                   linewidths=0)
m, b = np.polyfit(mz, sk, 1)
xs = np.linspace(mz.min(), mz.max(), 50)
ax[2].plot(xs, m * xs + b, "--", color=GREY, lw=1.2)
ax[2].set_xlabel("precursor $m/z$ (Da)")
ax[2].set_ylabel(AXLAB["s_k"])
cb2 = fig.colorbar(sc, ax=ax[2], pad=0.02, shrink=0.85, aspect=18)
cb2.set_label("peaks per spectrum", fontsize=8)
cb2.ax.tick_params(labelsize=7)
tag(ax[2], "C")

# --- D: pairwise axis correlation as a scatter of |r| -------------------
pairs = [("s_k", "s_t"), ("s_k", "s_e"), ("s_t", "s_e")]
vals = {"s_k": sk, "s_t": stt, "s_e": se}
rs = [abs(np.corrcoef(vals[a], vals[b])[0, 1]) for a, b in pairs]
x = np.arange(3)
ax[3].vlines(x, 0, rs, color=[BLUE, RED, GREEN], lw=7, alpha=0.85)
ax[3].plot(x, rs, "o", color="white", ms=7, mec=GREY, mew=1.4, zorder=3)
ax[3].set_xticks(x)
ax[3].set_xticklabels([f"{AXLAB[a]}–{AXLAB[b]}" for a, b in pairs])
ax[3].set_ylabel(r"$|r|$ between axes")
ax[3].set_ylim(0, max(rs) * 1.25)
tag(ax[3], "D")

save(fig, "panel2_coordinate_space.png")
