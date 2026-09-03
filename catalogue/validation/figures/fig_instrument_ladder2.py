"""
Panels 3 and 4 for "The Instrument as a Process Ladder".

Same rule as panels 1 and 2: every mark is a measured value from
validation/results/exp1_instrument_ladder.json, or the paper's own
algebra evaluated on a grid with the measured points overplotted.
"""
from __future__ import annotations

import numpy as np

from panelkit import (C1, C2, C3, C4, GOOD, BAD, INK, INK2, MUTED,
                      panel, tag, load, save)

PAPER = "instrument-process-ladder"
D = load("exp1_instrument_ladder")
R = D["records"]

POWERS = R["V5_sensitivity"]["powers"]
NRUNG = len(POWERS)


def composite(ps):
    out = 1.0
    for p in ps:
        out *= (1.0 - p)
    return 1.0 - out


# =====================================================================
# Panel 3 --- the readout is a property of the process, not the label
# =====================================================================
def panel3():
    fig, ax = panel(three_d=(1,))

    # (a) two differently-labelled substrates run through the same
    # ladder give the identical readout; a different SEQUENCE does not.
    a = ax[0]
    v1 = R["V1_readouts"]
    ctl = R["V1_control"]["different_sequence"]
    names = ["substrate A", "substrate B", "different" + chr(10) + "sequence"]
    resol = [v1["substrate_a"]["resolution"], v1["substrate_b"]["resolution"],
             ctl["resolution"]]
    ambig = [v1["substrate_a"]["ambiguity"], v1["substrate_b"]["ambiguity"],
             ctl["ambiguity"]]
    idx = np.arange(3)
    w = 0.36
    a.bar(idx - w / 2, resol, w, color=C1, label="resolution")
    a.bar(idx + w / 2, ambig, w, color=C2, label="ambiguity")
    for i, (r, m) in enumerate(zip(resol, ambig)):
        a.text(i - w / 2, r + 0.02, "%.4f" % r, ha="center",
               fontsize=6.2, color=INK2)
        a.text(i + w / 2, m + 0.02, "%.4f" % m, ha="center",
               fontsize=6.2, color=INK2)
    a.set_xticks(idx)
    a.set_xticklabels(names, fontsize=7)
    a.set_ylim(0, 1.18)
    a.set_ylabel("readout")
    a.set_title("Relabelling changes nothing; reordering does")
    a.legend(loc="upper center", ncol=2)
    tag(a, "a")

    # (b) 3-D: the two-rung resolution sheet with the measured pair of
    # operating points on it --- the substrates coincide, the control
    # sits elsewhere.
    b = ax[1]
    g = np.linspace(0.05, 0.9, 40)
    X, Y = np.meshgrid(g, g)
    rest = composite([POWERS[2], POWERS[3], POWERS[4]])
    Z = 1.0 - (1.0 - X) * (1.0 - Y) * (1.0 - rest)
    b.view_init(elev=30, azim=-58)
    b.plot_wireframe(X, Y, Z, color=C1, linewidth=0.4, alpha=0.55,
                     rstride=3, cstride=3)
    zt = v1["substrate_a"]["resolution"]
    zc = ctl["resolution"]
    b.scatter([POWERS[0]], [POWERS[1]], [zt], s=54, color=GOOD,
              edgecolor="white", linewidth=1.0, depthshade=False,
              zorder=12, label="both substrates %.4f" % zt)
    # The control's measured readout is 0.4095, which is BELOW the
    # sheet everywhere: solving the ladder for the pi1 that would put
    # it on the sheet gives -1.85. That is the finding, not a placement
    # problem --- the remaining three rungs alone floor this surface at
    # `rest`, so no choice of the first two reaches the control. Draw
    # it as the plane it is, rather than inventing coordinates for a
    # point that does not lie on the sheet.
    b.plot_surface(X, Y, np.full_like(Z, zc), color=BAD, alpha=0.16,
                   linewidth=0, shade=False)
    b.text(0.05, 0.9, zc + 0.012, "control %.4f (off the sheet)" % zc,
           fontsize=6.2, color=BAD)
    b.set_zlim(min(zc, float(Z.min())) - 0.03, 1.02)
    b.set_xlabel(r"$\pi_1$", labelpad=-2)
    b.set_ylabel(r"$\pi_2$", labelpad=-2)
    b.zaxis.set_rotate_label(False)
    b.set_zlabel(r"$\pi(L)$", rotation=90, labelpad=-1)
    b.set_title("No rung pair reaches the control")
    b.legend(loc="upper left", fontsize=6.2)
    tag(b, "b", three_d=True)

    # (c) sensitivity ordering agrees with power ordering, rung by rung.
    c = ax[2]
    o = R["V5_ordering"]
    rank_s = {k: i for i, k in enumerate(o["by_sensitivity"])}
    rank_p = {k: i for i, k in enumerate(o["by_power"])}
    keys = sorted(rank_s, key=lambda k: rank_p[k])
    xs = [rank_p[k] for k in keys]
    ys = [rank_s[k] for k in keys]
    c.plot([0, 4], [0, 4], color=MUTED, lw=0.9, ls="--")
    c.scatter(xs, ys, s=95, color=C1, zorder=4, edgecolor="white",
              linewidth=1.0)
    for k, x, y in zip(keys, xs, ys):
        c.annotate(k, xy=(x, y), xytext=(7, -3),
                   textcoords="offset points", fontsize=7, color=INK2)
    c.set_xticks(range(5))
    c.set_yticks(range(5))
    c.set_xlabel("rank by rung power")
    c.set_ylabel("rank by sensitivity")
    c.set_title("Sensitivity ranks with power")
    c.set_xlim(-0.5, 4.7)
    c.set_ylim(-0.5, 4.7)
    tag(c, "c")

    # (d) relabelling mismatches: measured zero across the trial set.
    d = ax[3]
    v6 = R["V6_inertness"]
    n = R["V3_model_error"]["n_trials"]
    mism = v6["relabel_mismatches"]
    d.bar([0], [n - mism], color=C1, width=0.55)
    d.bar([0], [mism], bottom=[n - mism], color=BAD, width=0.55)
    d.bar([1], [n], color=C3, width=0.55)
    d.set_xticks([0, 1])
    d.set_xticklabels(["relabelled" + chr(10) + "substrate",
                       "different" + chr(10) + "sequence"], fontsize=7)
    d.set_ylabel("trials")
    d.set_ylim(0, n * 1.26)
    d.set_title("Inertness holds on every trial")
    d.text(0, n * 1.06, "%d / %d identical" % (n - mism, n), ha="center",
           fontsize=7.4, color=GOOD, fontweight="bold")
    d.text(1, n * 1.06, "separated by control", ha="center",
           fontsize=7.4, color=INK2)
    tag(d, "d")

    fig.subplots_adjust(wspace=0.34, left=0.045, right=0.985)
    save(fig, PAPER, "panel3_inertness")


# =====================================================================
# Panel 4 --- what the ladder costs: contacts, density, decidability
# =====================================================================
def panel4():
    fig, ax = panel(three_d=(0,))

    # (a) 3-D: deviation over contact density and rung power. The
    # measured density sweep is drawn on the surface as a ridge; the
    # surface itself is the ladder's own form extended in rung power.
    a = ax[0]
    rows = R["V2_deviation_vs_density"]
    dens = np.array([r["density"] for r in rows], float)
    dev = np.array([r["deviation"] for r in rows], float)
    slope = dev[-1] / dens[-1]
    dg = np.linspace(0, 8, 40)
    pg = np.linspace(0.05, 0.9, 40)
    DG, PG = np.meshgrid(dg, pg)
    S = slope * DG * (PG / POWERS[0])
    a.view_init(elev=28, azim=-56)
    a.plot_surface(DG, PG, S, cmap="Blues", linewidth=0, alpha=0.60,
                   rstride=1, cstride=1)
    a.plot(dens, np.full_like(dens, POWERS[0]), dev, "-o", color=BAD,
           lw=2.0, markersize=4, zorder=12, label="measured sweep")
    a.set_xlabel("contact density", labelpad=-2)
    a.set_ylabel("rung power", labelpad=-2)
    a.zaxis.set_rotate_label(False)
    a.set_zlabel("deviation", rotation=90, labelpad=6)
    a.set_title("Deviation over density and power")
    a.legend(loc="upper left", fontsize=6.4)
    tag(a, "a", three_d=True)

    # (b) the three composition rules evaluated on the ladder's own
    # rung powers: the additive rule leaves the unit interval.
    b = ax[1]
    ks = np.arange(1, NRUNG + 1)
    prod = [composite(POWERS[:k]) for k in ks]
    add = [sum(POWERS[:k]) for k in ks]
    mx = [max(POWERS[:k]) for k in ks]
    b.plot(ks, prod, "-o", color=C1, label="product (measured)")
    b.plot(ks, add, "-s", color=C2, label="additive")
    b.plot(ks, mx, "-^", color=C4, label="max-based")
    b.axhline(1.0, color=MUTED, lw=0.9, ls=":")
    b.text(1.05, 1.03, "unit interval", fontsize=6.8, color=INK2)
    b.set_xticks(ks)
    b.set_xlabel("rungs composed")
    b.set_ylabel("resolving power")
    b.set_ylim(0, 1.9)
    b.set_title("Additive rule leaves the unit interval")
    b.legend(loc="lower right")
    tag(b, "b")

    # (c) model error per rule on the measured 400-trial set.
    c = ax[2]
    v3 = R["V3_model_error"]
    labs = ["product", "additive", "max-based"]
    vals = [v3["multiplicative_max_abs_err"], v3["additive_mean_abs_err"],
            v3["max_based_mean_abs_err"]]
    bars = c.bar(labs, vals, color=[GOOD, C2, C4], width=0.55)
    for bar, v in zip(bars, vals):
        c.text(bar.get_x() + bar.get_width() / 2, v + 0.007, "%.4f" % v,
               ha="center", fontsize=7, color=INK2)
    c.set_ylabel("mean abs. error")
    c.set_ylim(0, max(vals) * 1.28)
    c.set_title("Only the product rule is exact (n=%d)" % v3["n_trials"])
    tag(c, "c")

    # (d) how much the static analysis settles before anything runs.
    d = ax[3]
    v8 = R["V8_static_vs_executed"]
    n, rej = v8["n_cases"], v8["statically_rejected"]
    frac = rej / float(n)
    d.barh([0], [frac], color=C2, height=0.42, label="rejected statically")
    d.barh([0], [1 - frac], left=[frac], color=C1, height=0.42,
           label="run to completion")
    d.text(frac / 2, 0, "%.0f%%" % (100 * frac), ha="center", va="center",
           color="white", fontsize=9.5, fontweight="bold")
    d.text(frac + (1 - frac) / 2, 0, "%.0f%%" % (100 * (1 - frac)),
           ha="center", va="center", color="white", fontsize=9.5,
           fontweight="bold")
    d.set_yticks([])
    d.set_xlim(0, 1)
    d.set_ylim(-0.62, 0.95)
    d.set_xlabel("share of %d cases" % n)
    d.set_title("Static analysis settles %d of %d" % (rej, n))
    d.legend(loc="upper center", ncol=2)
    tag(d, "d")

    fig.subplots_adjust(wspace=0.42, left=0.03, right=0.985)
    save(fig, PAPER, "panel4_cost")


if __name__ == "__main__":
    panel3()
    panel4()
