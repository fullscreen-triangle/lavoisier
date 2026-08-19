"""Panel 3 - Collision-energy dependence: the coordinates track NCE."""
import numpy as np
import statistics as st
from panelstyle import (load, new_panel, tag, save, by_nce, sd_by_nce, fit,
                        AXCOL, AXLAB, BLUE, RED, GREEN, ORANGE, GREY, NCE)

coords, meta = load()
nce = np.array([c["nce"] for c in coords if c["nce"]], float)

# --- A: mean coordinate against NCE, all three axes ---------------------
fig, ax = new_panel(4, d3=(2,))

for a in ("s_k", "s_t", "s_e"):
    mu = np.array(by_nce(coords, a))
    sd = np.array(sd_by_nce(coords, a))
    z = (mu - mu.mean()) / mu.std()          # standardised so axes overlay
    zs = sd / mu.std()
    ax[0].plot(NCE, z, "o-", color=AXCOL[a], ms=4.5, mfc="white", mew=1.3,
               label=AXLAB[a])
    ax[0].fill_between(NCE, z - 0.28 * zs, z + 0.28 * zs,
                       color=AXCOL[a], alpha=0.13)
ax[0].axhline(0, color=GREY, lw=0.8, ls=":")
ax[0].set_xlabel("normalised collision energy (%)")
ax[0].set_ylabel("standardised mean")
ax[0].legend(loc="upper left", ncol=3, columnspacing=0.9, handlelength=1.3)
tag(ax[0], "A")

# --- B: peak count vs NCE, with S_e overlaid (the mechanism) ------------
npk = [st.mean([m["n_peaks"] for m in meta if m["nce"] == n]) for n in NCE]
ax[1].bar(range(len(NCE)), npk, color=BLUE, alpha=0.55, width=0.62,
          edgecolor="white")
ax[1].set_xticks(range(len(NCE)))
ax[1].set_xticklabels(NCE)
ax[1].set_xlabel("NCE (%)")
ax[1].set_ylabel("mean peaks per spectrum", color=BLUE)
ax[1].tick_params(axis="y", colors=BLUE)
axb = ax[1].twinx()
axb.plot(range(len(NCE)), by_nce(coords, "s_e"), "o-", color=RED, ms=5,
         mfc="white", mew=1.4)
axb.set_ylabel(AXLAB["s_e"], color=RED)
axb.tick_params(axis="y", colors=RED)
axb.spines["right"].set_visible(True)
axb.spines["right"].set_color(RED)
tag(ax[1], "B")

# --- C (3-D): NCE-binned centroid surface with a few tracked compounds -
from collections import defaultdict
g = defaultdict(list)
for c in coords:
    g[c["compound"]].append(c)
full = {k: v for k, v in g.items() if len(v) >= 9}

# Global centroid path: mean coordinate at each NCE level.
cx = [st.mean([c["s_k"] for c in coords if c["nce"] == n]) for n in NCE]
cy = [st.mean([c["s_t"] for c in coords if c["nce"] == n]) for n in NCE]
cz = [st.mean([c["s_e"] for c in coords if c["nce"] == n]) for n in NCE]

# Faint cloud for context.
ax[2].scatter([c["s_k"] for c in coords], [c["s_t"] for c in coords],
              [c["s_e"] for c in coords], c="#D9D9D9", s=1.6, alpha=0.30,
              linewidths=0, depthshade=False)

# Four representative compounds, each a clean path low->high NCE.
import matplotlib as _mpl
sel = list(full)[:4]
cmap = _mpl.colormaps["turbo"].resampled(len(sel))
for i, k in enumerate(sel):
    tr = sorted(full[k], key=lambda c: c["nce"])
    ax[2].plot([t["s_k"] for t in tr], [t["s_t"] for t in tr],
               [t["s_e"] for t in tr], "-o", color=cmap(i), lw=1.4, ms=3.2,
               alpha=0.95, mfc="white", mew=0.9)

# Global centroid trajectory, thick and dark: the systematic drift.
ax[2].plot(cx, cy, cz, "-", color="black", lw=3.0, zorder=6)
ax[2].scatter(cx, cy, cz, c=NCE, cmap="plasma", s=46, edgecolors="black",
              linewidths=0.7, depthshade=False, zorder=7)

ax[2].set_xlabel(AXLAB["s_k"], labelpad=-4)
ax[2].set_ylabel(AXLAB["s_t"], labelpad=-4)
ax[2].set_zlabel(AXLAB["s_e"], labelpad=-4)
ax[2].view_init(elev=20, azim=-60)
ax[2].tick_params(pad=-1)
tag(ax[2], "C", d3=True)

# --- D: correlation of each axis with NCE, against the threshold -------
rs = []
for a in ("s_k", "s_t", "s_e"):
    v = np.array([c[a] for c in coords if c["nce"]], float)
    rs.append(np.corrcoef(nce, v)[0, 1])
x = np.arange(3)
cols = [AXCOL[a] for a in ("s_k", "s_t", "s_e")]
ax[3].bar(x, rs, color=cols, width=0.55, edgecolor="white")
ax[3].axhline(0.3, color=GREY, ls="--", lw=1.2)
ax[3].axhline(-0.3, color=GREY, ls="--", lw=1.2)
ax[3].axhline(0, color="black", lw=0.8)
ax[3].set_xticks(x)
ax[3].set_xticklabels([AXLAB[a] for a in ("s_k", "s_t", "s_e")])
ax[3].set_ylabel(r"Pearson $r$ vs NCE")
ax[3].set_ylim(-0.55, 0.55)
tag(ax[3], "D")

save(fig, "panel3_nce_dependence.png")
