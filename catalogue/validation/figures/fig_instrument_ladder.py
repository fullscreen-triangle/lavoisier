"""
Panels for "The Instrument as a Process Ladder".

Every mark is a measured value from
validation/results/exp1_instrument_ladder.json. Where a surface is drawn
over a continuous parameter the ladder algebra is evaluated on a grid ---
that is the paper's own model being displayed as a surface, with the
measured points overplotted on it, not a schematic.
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
# Panel 1 --- the ladder resolves: composition, sensitivity, saturation
# =====================================================================
def panel1():
    fig, ax = panel(three_d=(3,))

    # (a) cumulative resolution as rungs are added, measured composite.
    a = ax[0]
    cum = [composite(POWERS[:k + 1]) for k in range(NRUNG)]
    xs = np.arange(1, NRUNG + 1)
    a.plot(xs, cum, "-o", color=C1, zorder=3)
    a.bar(xs, POWERS, width=0.5, color=C2, alpha=0.85, zorder=2)
    a.set_xticks(xs)
    a.set_xticklabels(["k%d" % i for i in xs])
    a.set_ylim(0, 1.0)
    a.set_xlabel("rung")
    a.set_ylabel("resolving power")
    a.set_title("Composition reaches %.4f" % cum[-1])
    a.annotate("composite", xy=(NRUNG, cum[-1]), xytext=(2.4, 0.965),
               color=C1, fontsize=7.5)
    a.annotate("per rung", xy=(4, POWERS[3]), xytext=(3.30, 0.30),
               color=C2, fontsize=7.5)
    tag(a, "a")

    # (b) sensitivity: analytic vs numeric vs closed form, three ways of
    # computing the same five derivatives.
    b = ax[1]
    s = R["V5_sensitivity"]
    pw = np.asarray(s["powers"], float)
    an = np.asarray(s["analytic"], float)
    nu = np.asarray(s["numeric"], float)
    # The three methods agree to ~1e-9, so plotting them side by side
    # wastes the slot. Plot sensitivity against rung power instead ---
    # that is the monotone relation the theorem asserts --- and carry
    # the agreement as the residual on the twin axis.
    order = np.argsort(pw)
    b.plot(pw[order], an[order], "-o", color=C1, zorder=3,
           label="analytic")
    b.scatter(pw, nu, s=70, facecolor="none", edgecolor=C2,
              linewidth=1.6, zorder=4, label="numeric")
    for x, y, k in zip(pw, an, range(1, NRUNG + 1)):
        b.annotate("k%d" % k, xy=(x, y), xytext=(3, 5),
                   textcoords="offset points", fontsize=6.6, color=INK2)
    b.set_xlabel(r"rung power $\pi_j$")
    b.set_ylabel(r"$\partial\pi(L)/\partial\pi_j$")
    b.set_title("Control sits at the strongest rung")
    b.legend(loc="upper left")
    resid = float(np.max(np.abs(an - nu)))
    b.text(0.97, 0.05, "max |analytic - numeric| = %.1e" % resid,
           transform=b.transAxes, ha="right", fontsize=6.8, color=INK2)
    tag(b, "b")

    # (c) saturation: residual gap for a divergent vs convergent series.
    c = ax[2]
    v4 = R["V4_saturation"]
    # Reproduce the experiment's own recurrence exactly: terms indexed
    # from i=2 (Remark 3.9), residual the running PRODUCT of (1-pi_i).
    # exp(-cumsum) is a different curve and misses the measured
    # endpoint, which is the whole claim.
    N = v4["n_terms"]
    i = np.arange(2, N + 2)
    rdiv = np.cumprod(1.0 - 1.0 / i)
    rcon = np.cumprod(1.0 - 1.0 / (i * i))
    n = np.arange(1, N + 1)
    c.loglog(n, rdiv, color=C1, label="divergent $\\sum\\pi_i$")
    c.loglog(n, rcon, color=C2, label="convergent $\\sum\\pi_i$")
    c.scatter([N], [v4["divergent_residual_gap"]], s=42, color=C1,
              zorder=5, edgecolor="white", linewidth=0.9)
    c.scatter([N], [v4["convergent_residual_gap"]], s=42, color=C2,
              zorder=5, edgecolor="white", linewidth=0.9)
    c.annotate("%.4f" % v4["convergent_residual_gap"],
               xy=(N, v4["convergent_residual_gap"]),
               xytext=(45, v4["convergent_residual_gap"] * 1.9),
               fontsize=7, color=C2)
    c.annotate("%.1e" % v4["divergent_residual_gap"],
               xy=(N, v4["divergent_residual_gap"]),
               xytext=(18, v4["divergent_residual_gap"] * 2.6),
               fontsize=7, color=C1)
    c.set_xlabel("rungs $n$")
    c.set_ylabel("residual $r_n$")
    c.set_title("Saturation needs a divergent sum")
    c.legend(loc="lower left")
    tag(c, "c")

    # (d) 3-D: composite resolution over two rung powers, with the
    # measured operating point on the surface.
    d = ax[3]
    g = np.linspace(0.0, 0.9, 44)
    X, Y = np.meshgrid(g, g)
    rest = composite([POWERS[2], POWERS[3], POWERS[4]])
    Z = 1.0 - (1.0 - X) * (1.0 - Y) * (1.0 - rest)
    # View chosen so the operating corner faces the camera: at
    # azim=-132 the marker rendered behind the sheet and read as a dull
    # violet rather than as the status red.
    d.view_init(elev=30, azim=-58)
    d.plot_surface(X, Y, Z, cmap="Blues", vmin=0.3, vmax=1.05,
                   linewidth=0, antialiased=True, alpha=0.78,
                   rstride=1, cstride=1)
    zop = composite(POWERS)
    d.plot([POWERS[0], POWERS[0]], [POWERS[1], POWERS[1]],
           [float(Z.min()), zop], color=BAD, lw=1.1, ls="--", zorder=9)
    d.scatter([POWERS[0]], [POWERS[1]], [zop], s=58, color=BAD,
              edgecolor="white", linewidth=1.0, depthshade=False,
              zorder=12)
    d.text(POWERS[0] - 0.02, POWERS[1] + 0.05, zop + 0.05,
           "operating point %.4f" % zop, fontsize=6.6, color=INK)
    d.set_xlabel("$\\pi_1$", labelpad=-1)
    d.set_ylabel("$\\pi_2$", labelpad=-1)
    d.set_zlabel("$\\pi(L)$", labelpad=-1)
    d.zaxis.set_rotate_label(False)
    d.set_zlabel("$\\pi(L)$", rotation=90, labelpad=-1)
    d.set_title("Resolution over two rungs")
    tag(d, "d", three_d=True)

    fig.subplots_adjust(wspace=0.32, left=0.04, right=0.985)
    save(fig, PAPER, "panel1_composition")


# =====================================================================
# Panel 2 --- contact is counted, transit is free
# =====================================================================
def panel2():
    fig, ax = panel(three_d=(2,))

    # (a) deviation grows with contact density; zero at zero density.
    a = ax[0]
    rows = R["V2_deviation_vs_density"]
    dens = [r["density"] for r in rows]
    dev = [r["deviation"] for r in rows]
    a.plot(dens, dev, "-o", color=C1, zorder=3)
    a.fill_between(dens, 0, dev, color=C1, alpha=0.12)
    a.set_xlabel("contact density")
    a.set_ylabel("readout deviation")
    a.set_title("Deviation tracks contact")
    a.scatter([0], [0], s=64, facecolor="white", edgecolor=C1,
              linewidth=1.6, zorder=5)
    a.annotate("deviation 0 at zero contact density",
               xy=(0.0, 0.0), xytext=(2.6, 0.0009),
               fontsize=7, color=INK2, va="center",
               arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8,
                               shrinkB=7))
    tag(a, "a")

    # (b) cost is flat in path length: transit is free.
    b = ax[1]
    rows = R["V7_cost_vs_path"]
    pl = [r["path_len"] for r in rows]
    cost = [r["cost"] for r in rows]
    res = [r["resolution"] for r in rows]
    # Both measured quantities are flat across four decades of path
    # length; the point of the chart is the pair of flat lines, not one
    # of them. Normalise each to its own value at the shortest path so
    # a single axis carries both without a second scale.
    b.semilogx(pl, [c / cost[0] + 0.02 for c in cost], "-o", color=C1,
               label="cost (contacts), %d" % cost[0])
    b.semilogx(pl, [r / res[0] - 0.02 for r in res], "-s", color=C2,
               markerfacecolor="none", markeredgewidth=1.4,
               label="resolution, %.4f" % res[0])
    b.axhline(1.0, color=MUTED, lw=0.8, ls=":")
    b.set_ylim(0.90, 1.12)
    b.set_xlabel("path length (arb. units)")
    b.set_ylabel("value / value at shortest path")
    b.set_title("Transit adds no cost")
    b.legend(loc="upper left")
    b.text(0.97, 0.06, "unchanged over $10^4$ in path length",
           transform=b.transAxes, ha="right", fontsize=6.8, color=INK2)
    tag(b, "b")

    # (c) 3-D: model error for the three composition rules.
    c = ax[2]
    v3 = R["V3_model_error"]
    names = ["multiplicative", "additive", "max-based"]
    vals = [v3["multiplicative_max_abs_err"],
            v3["additive_mean_abs_err"],
            v3["max_based_mean_abs_err"]]
    cols = [GOOD, C2, C4]
    for i, (nm, v, col) in enumerate(zip(names, vals, cols)):
        c.bar3d(i - 0.3, -0.3, 0, 0.6, 0.6, max(v, 1e-4),
                color=col, shade=True, alpha=0.95, edgecolor="white",
                linewidth=0.4)
    c.set_zlim(0, max(vals) * 1.18)
    c.set_xticks(range(3))
    # Values ride on the tick labels: 3-D text placement collides with
    # the bars at every offset that keeps it inside the axes.
    c.set_xticklabels([nm + chr(10) + "%.3f" % v for nm, v in zip(names, vals)],
                      fontsize=6.2)
    c.set_yticks([])
    c.set_zlabel("mean abs. error", labelpad=-4)
    c.set_title("Only the product rule is exact")
    c.view_init(elev=20, azim=-72)
    tag(c, "c", three_d=True)

    # (d) static analysis agrees with execution on every case.
    d = ax[3]
    v8 = R["V8_static_vs_executed"]
    n = v8["n_cases"]
    rej = v8["statically_rejected"]
    acc = n - rej
    d.barh([1], [acc], color=C1, height=0.5, label="accepted")
    d.barh([0], [rej], color=C2, height=0.5, label="rejected")
    d.set_yticks([0, 1])
    d.set_yticklabels(["statically\nrejected", "accepted"])
    d.set_xlabel("cases (of %d)" % n)
    d.set_title("Static verdict matches execution")
    d.text(acc * 0.5, 1, "%d" % acc, ha="center", va="center",
           color="white", fontsize=8, fontweight="bold")
    d.text(rej * 0.5, 0, "%d" % rej, ha="center", va="center",
           color="white", fontsize=8, fontweight="bold")
    d.set_xlim(0, n * 0.78)
    d.set_ylim(-0.55, 1.55)
    d.text(0.97, 0.94, "%d disagreements between static verdict "
           "and execution" % v8["disagreements"],
           transform=d.transAxes, ha="right", fontsize=7.4, color=GOOD,
           fontweight="bold")
    tag(d, "d")

    fig.subplots_adjust(wspace=0.36, left=0.045, right=0.985)
    save(fig, PAPER, "panel2_contact")


if __name__ == "__main__":
    panel1()
    panel2()
