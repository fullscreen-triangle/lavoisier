"""Panel 4 - Separation: within- vs between-compound spread, and the control."""
import numpy as np
import random
import statistics as st
from collections import defaultdict
from panelstyle import (load, groups, axis_stats, new_panel, tag, save,
                        AXCOL, AXLAB, BLUE, RED, GREEN, ORANGE, PURPLE,
                        GREY, LIGHT)

coords, meta = load()
full = groups(coords, 9)
AX = ("s_k", "s_t", "s_e")


def dist(a, b):
    return np.sqrt(sum((a[x] - b[x]) ** 2 for x in AX))


# within/between distance populations
within, between = [], []
cents = {}
for k, v in full.items():
    cents[k] = {a: st.mean([i[a] for i in v]) for a in AX}
    for i in range(len(v)):
        for j in range(i + 1, len(v)):
            within.append(dist(v[i], v[j]))
ks = list(cents)
for i in range(len(ks)):
    for j in range(i + 1, len(ks)):
        between.append(dist(cents[ks[i]], cents[ks[j]]))
within = np.array(within)
between = np.array(between)

# shuffled control
rng = random.Random(20260818)
labels = [c["compound"] for c in coords]
rng.shuffle(labels)
sh = [dict(c, compound=l) for c, l in zip(coords, labels)]
shg = groups(sh, 9)
sw, sb = [], []
sc = {}
for k, v in shg.items():
    sc[k] = {a: st.mean([i[a] for i in v]) for a in AX}
    for i in range(len(v)):
        for j in range(i + 1, len(v)):
            sw.append(dist(v[i], v[j]))
sks = list(sc)
for i in range(len(sks)):
    for j in range(i + 1, len(sks)):
        sb.append(dist(sc[sks[i]], sc[sks[j]]))

fig, ax = new_panel(4, d3=(2,))

# --- A: overlapping distance distributions -----------------------------
bins = np.linspace(0, np.percentile(np.r_[within, between], 99.5), 70)
ax[0].hist(within, bins=bins, color=RED, alpha=0.55, density=True,
           label="within compound", edgecolor="none")
ax[0].hist(between, bins=bins, color=BLUE, alpha=0.55, density=True,
           label="between compounds", edgecolor="none")
ax[0].axvline(within.mean(), color=RED, lw=1.8)
ax[0].axvline(between.mean(), color=BLUE, lw=1.8)
ax[0].set_xlabel("coordinate distance")
ax[0].set_ylabel("density")
ax[0].legend(loc="upper right")
tag(ax[0], "A")

# --- B: per-axis within vs between spread ------------------------------
stats = axis_stats(full)
x = np.arange(3)
w = 0.36
wv = [stats[a][0] for a in AX]
bv = [stats[a][1] for a in AX]
ax[1].bar(x - w / 2, wv, w, color=RED, alpha=0.8, edgecolor="white",
          label="within (sd)")
ax[1].bar(x + w / 2, bv, w, color=BLUE, alpha=0.8, edgecolor="white",
          label="between (sd)")
ax[1].set_yscale("log")
ax[1].set_xticks(x)
ax[1].set_xticklabels([AXLAB[a] for a in AX])
ax[1].set_ylabel("standard deviation (log)")
ax[1].legend(loc="lower right")
tag(ax[1], "B")

# --- C (3-D): compound centroids, sized by within-compound scatter -----
cx = np.array([cents[k]["s_k"] for k in ks])
cy = np.array([cents[k]["s_t"] for k in ks])
cz = np.array([cents[k]["s_e"] for k in ks])
spread = np.array([st.mean([dist(i, cents[k]) for i in full[k]]) for k in ks])
p = ax[2].scatter(cx, cy, cz, c=spread, cmap="magma_r", s=14 + 26 * spread,
                  alpha=0.9, linewidths=0.25, edgecolors="white",
                  depthshade=False)
ax[2].set_xlabel(AXLAB["s_k"], labelpad=-4)
ax[2].set_ylabel(AXLAB["s_t"], labelpad=-4)
ax[2].set_zlabel(AXLAB["s_e"], labelpad=-4)
ax[2].view_init(elev=20, azim=-60)
ax[2].tick_params(pad=-1)
cb = fig.colorbar(p, ax=ax[2], pad=0.10, shrink=0.62, aspect=14)
cb.set_label("within-compound spread", fontsize=8)
cb.ax.tick_params(labelsize=7)
tag(ax[2], "C", d3=True)

# --- D: separation ratio, true vs shuffled, against the threshold ------
true_ratio = np.mean(between) / np.mean(within)
sh_ratio = np.mean(sb) / np.mean(sw)
xs = np.arange(2)
ax[3].bar(xs, [true_ratio, sh_ratio], color=[GREEN, GREY], width=0.5,
          edgecolor="white")
ax[3].axhline(2.0, color=RED, ls="--", lw=1.6)
ax[3].axhline(1.0, color="black", lw=0.9, ls=":")
ax[3].set_xticks(xs)
ax[3].set_xticklabels(["true labels", "shuffled"])
ax[3].set_ylabel("separation ratio")
ax[3].set_ylim(0, 2.35)
tag(ax[3], "D")

save(fig, "panel4_separation.png")
