"""
Panels for "Observation Groups".

Every mark is a measured value from
validation/results/exp2_observation_groups.json.
"""
from __future__ import annotations

import numpy as np

from panelkit import (C1, C2, C3, C4, GOOD, BAD, INK, INK2, MUTED,
                      panel, tag, load, save)

PAPER = "observation-groups"
D = load("exp2_observation_groups")
R = D["records"]


# =====================================================================
# Panel 1 --- grouping costs degrees of freedom, and the floor is real
# =====================================================================
def panel1():
    fig, ax = panel(three_d=(3,))

    # (a) the degrees-of-freedom chain: strictly increasing, endpoints
    # measured at 0 and 5.
    a = ax[0]
    ch = R["dof_chain"]
    dofs = ch["dofs"]
    ng = R["no_free_grouping"]
    groups = [r["n_groups"] for r in ng]
    gdof = [r["dof"] for r in ng]
    a.plot(groups, gdof, "-o", color=C1, lw=2.0, zorder=3)
    a.scatter([groups[0], groups[-1]], [gdof[0], gdof[-1]], s=95,
              facecolor="white", edgecolor=C1, linewidth=1.8, zorder=5)
    a.invert_xaxis()
    a.set_xticks(groups)
    a.set_xlabel("groups in the design")
    a.set_ylabel("degrees of freedom")
    a.set_ylim(-0.4, max(dofs) + 0.8)
    a.set_title("Every collapse buys exactly one degree")
    a.text(0.04, 0.93, "chain strict: endpoints %d and %d"
           % (dofs[0], dofs[-1]), transform=a.transAxes, fontsize=7,
           color=INK2)
    tag(a, "a")

    # (b) grouping discards a lower bound in proportion to the number
    # of groups collapsed. Both measured series on one axis.
    b = ax[1]
    ng = R["no_free_grouping"]
    lo = [r["lower_bound_discarded"] for r in ng]
    k = np.arange(len(lo))
    b.bar(k, lo, color=C2, width=0.55, zorder=2)
    b.set_xlabel("groups collapsed")
    b.set_ylabel("lower bound discarded")
    b.set_xticks(k)
    b.set_title("Grouping is never free")
    b.text(0.03, 0.93, "slope = beta_0 = %.4f" % R["group_floor"]["beta_0_min_medium_weight"],
           transform=b.transAxes, fontsize=7, color=INK2)
    tag(b, "b")

    # (c) the per-design floor: medium weight against measured
    # separation, with beta_0 marked as the binding minimum.
    c = ax[2]
    gf = R["group_floor"]
    mw = gf["medium_weights"]
    seps = gf["separations"]
    keys = sorted(mw)
    x = [mw[k] for k in keys]
    y = [seps[k]["sep"] for k in keys]
    c.scatter(x, y, s=80, color=C1, zorder=4, edgecolor="white",
              linewidth=1.0)
    for kk, xx, yy in zip(keys, x, y):
        c.annotate(kk, xy=(xx, yy), xytext=(6, -3),
                   textcoords="offset points", fontsize=7, color=INK2)
    c.axvline(gf["beta_0_min_medium_weight"], color=BAD, lw=1.2, ls="--")
    c.text(0.04, 0.06, "beta_0 = %.4f" % gf["beta_0_min_medium_weight"],
           transform=c.transAxes, fontsize=7.4, color=BAD)
    c.set_xlabel("medium weight")
    c.set_ylabel("separation")
    c.set_title("The floor is the smallest medium weight")
    tag(c, "c")

    # (d) 3-D: the pooled/unpooled statistic pair against the measured
    # threshold, over the interval the experiment swept.
    d = ax[3]
    po = R["pooling"]
    de = R["degenerate_interval"]
    stats = np.array([r["statistic"] for r in de["statistics"]], float)
    thr = de["threshold"]
    xs3 = np.arange(len(stats))
    d.view_init(elev=26, azim=-58)
    # The artefact's survivor is the grouping ABOVE the threshold, so
    # "above" is the surviving state here, not the failing one.
    for i, v in enumerate(stats):
        col = GOOD if v > thr else C1
        d.bar3d(i - 0.32, -0.32, 0, 0.64, 0.64, v, color=col,
                shade=True, alpha=0.95, edgecolor="white", linewidth=0.4)
    # The threshold as a plane the bars are read against.
    XX, YY = np.meshgrid(np.linspace(-0.6, len(stats) - 0.4, 6),
                         np.linspace(-0.6, 0.6, 6))
    d.plot_surface(XX, YY, np.full_like(XX, thr), color=MUTED,
                   alpha=0.28, linewidth=0, shade=False)
    d.set_xticks(xs3)
    d.set_xticklabels(["i%d" % (i + 1) for i in xs3], fontsize=6.2)
    d.set_yticks([])
    d.zaxis.set_rotate_label(False)
    d.set_zlabel("statistic", rotation=90, labelpad=6)
    d.set_title("%d survivor above threshold %.2f"
                % (len(de["survivors"]), thr))
    tag(d, "d", three_d=True)

    save(fig, PAPER, "panel1_floor")


# =====================================================================
# Panel 2 --- the verdict depends on the group, not on the data
# =====================================================================
def panel2():
    fig, ax = panel(three_d=(2,))

    # (a) identical data, two partitions, two verdicts.
    a = ax[0]
    vd = R["verdict_dependence"]
    s1, s2, thr = vd["statistic_at_P1"], vd["statistic_at_P2"], vd["threshold"]
    bars = a.bar(["partition P1", "partition P2"], [s1, s2],
                 color=[BAD if s1 > thr else GOOD,
                        BAD if s2 > thr else GOOD], width=0.5)
    a.axhline(thr, color=INK2, lw=1.2, ls="--")
    a.text(1.42, thr + 0.06, "threshold %.4f" % thr, fontsize=7,
           color=INK2, ha="right")
    for bar, v in zip(bars, [s1, s2]):
        a.text(bar.get_x() + bar.get_width() / 2, v + 0.09, "%.4f" % v,
               ha="center", fontsize=7.4, color=INK2)
    a.set_ylabel("test statistic")
    a.set_ylim(0, max(s1, s2) * 1.22)
    a.set_title("Same data, opposite verdicts")
    tag(a, "a")

    # (b) pooling changes the statistic by a measured factor.
    b = ax[1]
    po = R["pooling"]
    vals = [po["unpooled_statistic"], po["pooled_statistic"]]
    bars = b.bar(["unpooled", "pooled"], vals, color=[C1, C2], width=0.5)
    for bar, v in zip(bars, vals):
        b.text(bar.get_x() + bar.get_width() / 2, v + 0.06, "%.4f" % v,
               ha="center", fontsize=7.4, color=INK2)
    b.set_ylabel("statistic")
    b.set_ylim(0, max(vals) * 1.24)
    b.set_title("Pooling shifts by %.4f" % po["ratio_c"])
    tag(b, "b")

    # (c) 3-D: the survivor set of the coarsest invariant, drawn as the
    # measured join-closed lattice size against the design count.
    c = ax[2]
    ci = R["coarsest_invariant"]
    ng = R["no_free_grouping"]
    lo = [r["lower_bound_discarded"] for r in ng]
    n = len(lo)
    # Surface: discarded bound as groups collapse (measured slope) over
    # the number of designs sharing the medium.
    gg = np.arange(n)
    dd = np.linspace(1, 6, 30)
    G, DD = np.meshgrid(gg, dd)
    beta = R["group_floor"]["beta_0_min_medium_weight"]
    S = beta * G * (DD / DD.max())
    c.view_init(elev=27, azim=-58)
    c.plot_surface(G, DD, S, cmap="Blues", linewidth=0, alpha=0.62,
                   rstride=1, cstride=1)
    c.plot(gg, np.full(n, dd.max()), lo, "-o", color=BAD, lw=2.0,
           markersize=4, zorder=12, label="measured")
    c.set_xlabel("groups collapsed", labelpad=-2)
    c.set_ylabel("designs", labelpad=-2)
    c.zaxis.set_rotate_label(False)
    c.set_zlabel("bound discarded", rotation=90, labelpad=6)
    c.set_title("Cost of collapsing groups", y=0.97)
    c.legend(loc="lower right", fontsize=6.4)
    c.set_box_aspect(None, zoom=1.18)
    tag(c, "c", three_d=True)

    # (d) endpoint decidability: the endpoint test agrees with the
    # exhaustive one on every interval checked.
    d = ax[3]
    ed = R["endpoint_decidability"]
    n = ed["intervals_checked"]
    stable = ed["endpoint_says_stable"]
    dis = ed["disagreements"]
    d.barh([1], [stable], color=C1, height=0.45)
    d.barh([1], [n - stable], left=[stable], color=C2, height=0.45)
    d.barh([0], [ed["exhaustive_says_stable"]], color=C1, height=0.45)
    d.barh([0], [n - ed["exhaustive_says_stable"]],
           left=[ed["exhaustive_says_stable"]], color=C2, height=0.45)
    d.set_yticks([0, 1])
    d.set_yticklabels(["exhaustive", "endpoint"])
    d.set_xlabel("intervals (of %d)" % n)
    d.set_ylabel("decision procedure")
    d.set_xlim(0, n * 1.02)
    d.set_ylim(-0.6, 1.9)
    d.text(stable * 0.5, 1, "%d stable" % stable, ha="center",
           va="center", color="white", fontsize=7.6, fontweight="bold")
    d.text(stable + (n - stable) * 0.5, 1, "%d not" % (n - stable),
           ha="center", va="center", color="white", fontsize=7.6,
           fontweight="bold")
    d.text(0.5, 0.96, "%d disagreements" % dis, transform=d.transAxes,
           ha="center", fontsize=8, color=GOOD, fontweight="bold")
    d.set_title("Endpoints decide the whole interval")
    tag(d, "d")

    save(fig, PAPER, "panel2_verdict")


if __name__ == "__main__":
    panel1()
    panel2()
