"""Six panels for the CASMI catalogue paper.

Every mark is a number measured from the 58 CASMI 2022 priority
challenges resolvable in the seventeen local Q Exactive HF files, read
from ../panel_data.json.  Nothing is simulated, no chart is conceptual,
and no chart is a table or text.

Each panel is four charts in a row on a white ground with at least one
three-dimensional axis.
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import casmikit as K
from casmikit import C1, C2, C3, C4, GOOD, BAD, INK, INK2, MUTED, GRID
from casmikit import VCOL, VLAB, VORDER


# ---------------------------------------------------------------- helpers
def by_verdict(rows):
    out = {v: [] for v in VORDER}
    for r in rows:
        if r.get("verdict") in out:
            out[r["verdict"]].append(r)
    return out


def linfit(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return my, 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return my - b * mx, b


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx) ** 2 for a in xs)
    syy = sum((b - my) ** 2 for b in ys)
    if sxx <= 0 or syy <= 0:
        return float("nan")
    return sxy / (sxx * syy) ** 0.5


def vlegend(ax, present, loc="best", ncol=1):
    """Legend keyed to verdict class.  Identity is never colour alone."""
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", linestyle="none", markersize=5,
                color=VCOL[v], label=VLAB[v])
         for v in VORDER if v in present]
    ax.legend(handles=h, loc=loc, ncol=ncol)


# ================================================================ panel 1
def panel1(D):
    """Degeneracy of the mass determination."""
    ch = D["challenges"]
    fig, ax = K.panel(three_d=(2,))

    # (a) uncapped candidate count vs precursor m/z, by polarity
    a = ax[0]
    for pol, col, lab, mk in (("+", C1, "positive ESI", "o"),
                              ("-", C2, "negative ESI", "s")):
        sub = [c for c in ch if c["pol"] == pol]
        a.scatter([c["mz"] for c in sub], [c["n_mass_candidates"] for c in sub],
                  s=26, c=col, marker=mk, edgecolor="white", linewidth=0.6,
                  label=lab, zorder=3)
    a.axhline(1, color=BAD, linewidth=1.4, linestyle="--", zorder=2)
    a.text(a.get_xlim()[1], 1.25, "unique formula", ha="right", va="bottom",
           fontsize=6.8, color=BAD)
    a.set_yscale("log")
    a.set_xlabel("precursor $m/z$")
    a.set_ylabel("formulas within 8 ppm")
    a.set_title("Mass candidates per challenge")
    a.legend(loc="upper left")
    K.tag(a, "a")

    # (b) distribution of the same counts
    b = ax[1]
    ns = sorted(c["n_mass_candidates"] for c in ch)
    edges = np.logspace(math.log10(max(1, ns[0])), math.log10(ns[-1] * 1.05), 16)
    b.hist(ns, bins=edges, color=C1, edgecolor="white", linewidth=0.8, zorder=3)
    med = ns[len(ns) // 2]
    mean = sum(ns) / len(ns)
    b.axvline(med, color=C2, linewidth=2.0, zorder=4)
    b.axvline(mean, color=C4, linewidth=2.0, linestyle="--", zorder=4)
    top = b.get_ylim()[1]
    b.text(med, top * 0.96, " median %d" % med, color=C2, fontsize=7,
           va="top", ha="left")
    b.text(mean, top * 0.80, " mean %d" % mean, color=C4, fontsize=7,
           va="top", ha="left")
    b.set_xscale("log")
    b.set_xlabel("formulas within 8 ppm")
    b.set_ylabel("challenges")
    b.set_title("Degeneracy distribution (n = %d)" % len(ns))
    K.tag(b, "b")

    # (c) 3-D surface: candidates over (m/z, tolerance)
    c = ax[2]
    dv = D["degeneracy_vs_ppm"]
    ppms = sorted({r["ppm"] for r in dv})
    ids = sorted({r["id"] for r in dv})
    grid = {}
    for r in dv:
        grid[(r["ppm"], r["id"])] = r
    mzs = []
    for i in ids:
        row = [grid[(p, i)] for p in ppms if (p, i) in grid]
        if row:
            mzs.append(row[0]["mz"])
    order = sorted(range(len(ids)), key=lambda k: mzs[k])
    ids = [ids[k] for k in order]
    mzs = [mzs[k] for k in order]
    X, Y = np.meshgrid(np.array(mzs), np.array(ppms))
    Z = np.zeros_like(X, dtype=float)
    for pi, p in enumerate(ppms):
        for mi, i in enumerate(ids):
            Z[pi, mi] = grid[(p, i)]["n"] if (p, i) in grid else np.nan
    Zl = np.log10(np.maximum(Z, 1.0))
    c.plot_surface(X, Y, Zl, cmap="Blues", linewidth=0.3,
                   edgecolor="white", rstride=1, cstride=1,
                   antialiased=True, alpha=0.95, vmin=0)
    c.set_xlabel("$m/z$", labelpad=-2)
    c.set_ylabel("tolerance (ppm)", labelpad=-2)
    c.set_zlabel("$\\log_{10}$ candidates", labelpad=-3)
    c.set_title("Candidates over mass and tolerance")
    c.view_init(elev=24, azim=-128)
    K.tag(c, "c", three_d=True)

    # (d) candidate count vs tolerance, per challenge
    d = ax[3]
    picks = ids[:: max(1, len(ids) // 6)][:6]
    for k, i in enumerate(picks):
        ys = [grid[(p, i)]["n"] for p in ppms if (p, i) in grid]
        xs = [p for p in ppms if (p, i) in grid]
        col = SERIES_CYCLE[k % len(SERIES_CYCLE)]
        d.plot(xs, ys, marker="o", color=col, linewidth=1.8, markersize=4,
               zorder=3)
        d.annotate(" %d" % round(grid[(ppms[-1], i)]["mz"]),
                   (xs[-1], ys[-1]), fontsize=6.6, color=col,
                   va="center", ha="left")
    d.set_yscale("log")
    d.set_xlabel("mass tolerance (ppm)")
    d.set_ylabel("candidate formulas")
    d.set_title("Attrition with tolerance (labels: $m/z$)")
    K.tag(d, "d")

    K.save(fig, "panel1_degeneracy")


SERIES_CYCLE = [C1, C2, C4, "#1baf7a", "#8a5ad6", "#c2185b"]


# ================================================================ panel 2
def panel2(D):
    """Mass error, candidate density, and the geometry of the window."""
    ch = D["challenges"]
    sz = D["size"]
    fig, ax = K.panel(three_d=(2,))

    # (a) mass error of top candidate vs m/z, with the 8 ppm envelope
    a = ax[0]
    grp = by_verdict(ch)
    for v in VORDER:
        sub = [c for c in grp[v] if c.get("ppm") is not None]
        if not sub:
            continue
        a.scatter([c["mz"] for c in sub], [c["ppm"] for c in sub],
                  s=30, c=VCOL[v], edgecolor="white", linewidth=0.6,
                  zorder=3 if v == "licensed" else 2)
    a.axhline(8, color=MUTED, linewidth=1.2, linestyle="--")
    a.axhline(-8, color=MUTED, linewidth=1.2, linestyle="--")
    a.axhline(0, color=GRID, linewidth=1.0)
    a.text(a.get_xlim()[1], 8.2, "8 ppm window", ha="right", va="bottom",
           fontsize=6.8, color=MUTED)
    a.set_xlabel("precursor $m/z$")
    a.set_ylabel("mass error of top candidate (ppm)")
    a.set_title("Where the top candidate sits in the window")
    # Headroom below the -8 ppm envelope so the legend does not sit on it.
    a.set_ylim(-12.6, 9.6)
    vlegend(a, grp, loc="lower right")
    K.tag(a, "a")

    # (b) candidate count vs mass error, all scored candidates
    b = ax[1]
    b.scatter([r["ppm"] for r in sz], [r["contact"] for r in sz],
              s=5, c=C1, alpha=0.22, linewidth=0, zorder=2)
    tops = [r for r in sz if r["is_top"]]
    b.scatter([r["ppm"] for r in tops], [r["contact"] for r in tops],
              s=26, c=C2, edgecolor="white", linewidth=0.6, zorder=4,
              label="top-ranked")
    b.set_xlabel("mass error (ppm)")
    b.set_ylabel("contact $\\kappa$")
    b.set_title("Mass error does not order contact")
    b.legend(loc="upper right")
    K.tag(b, "b")

    # (c) 3-D: m/z, mass error, contact -- coloured by verdict
    c = ax[2]
    vmap = {r["id"]: r["verdict"] for r in ch}
    mzmap = {r["id"]: r["mz"] for r in ch}
    for v in VORDER:
        xs = [mzmap[r["id"]] for r in sz if vmap.get(r["id"]) == v]
        ys = [r["ppm"] for r in sz if vmap.get(r["id"]) == v]
        zs = [r["contact"] for r in sz if vmap.get(r["id"]) == v]
        if not xs:
            continue
        c.scatter(xs, ys, zs, s=6, c=VCOL[v], alpha=0.45, linewidth=0,
                  depthshade=False, label=VLAB[v])
    c.set_xlabel("$m/z$", labelpad=-2)
    c.set_ylabel("ppm", labelpad=-2)
    c.set_zlabel("contact $\\kappa$", labelpad=-3)
    c.set_title("Mass and contact are not aligned")
    c.view_init(elev=20, azim=-58)
    c.legend(loc="upper left", bbox_to_anchor=(-0.02, 1.02))
    K.tag(c, "c", three_d=True)

    # (d) cumulative retention as tolerance tightens
    d = ax[3]
    dv = D["degeneracy_vs_ppm"]
    ppms = sorted({r["ppm"] for r in dv})
    ids = sorted({r["id"] for r in dv})
    grid = {(r["ppm"], r["id"]): r["n"] for r in dv}
    for i in ids:
        base = grid.get((ppms[-1], i))
        if not base:
            continue
        xs = [p for p in ppms if (p, i) in grid]
        ys = [100.0 * grid[(p, i)] / base for p in xs]
        d.plot(xs, ys, color=C1, alpha=0.45, linewidth=1.4, zorder=2)
    # the mean curve, drawn on top and directly labelled
    mean = []
    for p in ppms:
        vals = [100.0 * grid[(p, i)] / grid[(ppms[-1], i)]
                for i in ids if (p, i) in grid and grid.get((ppms[-1], i))]
        mean.append(sum(vals) / len(vals) if vals else 0.0)
    d.plot(ppms, mean, color=C2, linewidth=2.6, marker="o", markersize=5,
           zorder=4)
    d.annotate(" mean", (ppms[-1], mean[-1]), color=C2, fontsize=7.2,
               va="center", ha="left")
    d.set_xlabel("mass tolerance (ppm)")
    d.set_ylabel("candidates retained (% of 20 ppm set)")
    d.set_title("Slow attrition: no window isolates one")
    K.tag(d, "d")

    K.save(fig, "panel2_tolerance")


# ================================================================ panel 3
def panel3(D):
    """The three collision energies are not redundant."""
    lad = D["ladder"]
    fig, ax = K.panel(three_d=(2,))
    ces = [35.0, 45.0, 65.0]

    # (a) peak count per challenge at each energy
    a = ax[0]
    for k, e in enumerate(ces):
        ys = [r["n_peaks"][k] for r in lad if len(r["n_peaks"]) == 3]
        xs = np.full(len(ys), k, dtype=float)
        xs = xs + np.linspace(-0.16, 0.16, len(ys))
        a.scatter(xs, ys, s=18, c=SERIES_CYCLE[k], edgecolor="white",
                  linewidth=0.5, zorder=3)
        m = sum(ys) / len(ys)
        a.plot([k - 0.30, k + 0.30], [m, m], color=INK, linewidth=2.0,
               zorder=5)
        a.annotate("%.0f" % m, (k + 0.32, m), fontsize=7, color=INK,
                   va="center", ha="left")
    a.set_xticks(range(3))
    a.set_xticklabels(["35 eV", "45 eV", "65 eV"])
    a.set_xlabel("collision energy")
    a.set_ylabel("peaks above 1% base")
    a.set_title("Ladder depth by energy (bars: mean)")
    K.tag(a, "a")

    # (b) precursor survival at each energy, paired by challenge
    b = ax[1]
    keep = [r for r in lad if len(r["prec_survival"]) == 3]
    for r in keep:
        b.plot(range(3), r["prec_survival"], color=C1, alpha=0.28,
               linewidth=1.1, zorder=2)
    means = [sum(r["prec_survival"][k] for r in keep) / len(keep)
             for k in range(3)]
    b.plot(range(3), means, color=C2, linewidth=2.8, marker="o",
           markersize=7, zorder=5)
    for k, m in enumerate(means):
        b.annotate("%.2f" % m, (k, m), textcoords="offset points",
                   xytext=(6, 7), fontsize=7, color=C2)
    b.set_xticks(range(3))
    b.set_xticklabels(["35 eV", "45 eV", "65 eV"])
    b.set_xlabel("collision energy")
    b.set_ylabel("surviving precursor (rel. to base)")
    b.set_title("Precursor survival falls monotonically")
    K.tag(b, "b")

    # (c) 3-D trajectories through (energy, survival, mean fragment m/z)
    c = ax[2]
    keep2 = [r for r in lad
             if len(r["prec_survival"]) == 3 and len(r["mean_frag_mz"]) == 3
             and any(v > 0 for v in r["mean_frag_mz"])]
    for r in keep2:
        c.plot(ces, r["prec_survival"], r["mean_frag_mz"],
               color=C1, alpha=0.30, linewidth=1.0)
    mz_means = [sum(r["mean_frag_mz"][k] for r in keep2) / len(keep2)
                for k in range(3)]
    sv_means = [sum(r["prec_survival"][k] for r in keep2) / len(keep2)
                for k in range(3)]
    c.plot(ces, sv_means, mz_means, color=C2, linewidth=3.0,
           marker="o", markersize=6, zorder=10)
    c.set_xlabel("energy (eV)", labelpad=-2)
    c.set_ylabel("survival", labelpad=-2)
    c.set_zlabel("mean frag. $m/z$", labelpad=-3)
    c.set_title("Trajectories fan, they do not collapse")
    c.set_xticks(ces)
    c.view_init(elev=20, azim=-62)
    K.tag(c, "c", three_d=True)

    # (d) mean fragment m/z vs precursor m/z at each energy
    d = ax[3]
    for k, e in enumerate(ces):
        xs = [r["mz"] for r in keep2]
        ys = [r["mean_frag_mz"][k] for r in keep2]
        d.scatter(xs, ys, s=18, c=SERIES_CYCLE[k], edgecolor="white",
                  linewidth=0.5, zorder=3, label="%.0f eV" % e)
        a0, b0 = linfit(xs, ys)
        xr = [min(xs), max(xs)]
        d.plot(xr, [a0 + b0 * x for x in xr], color=SERIES_CYCLE[k],
               linewidth=1.6, linestyle="--", zorder=4)
    lim = max(max(r["mz"] for r in keep2), 1)
    d.plot([0, lim], [0, lim], color=MUTED, linewidth=1.0, linestyle=":",
           zorder=1)
    d.set_xlabel("precursor $m/z$")
    d.set_ylabel("intensity-weighted mean fragment $m/z$")
    d.set_title("Higher energy samples deeper")
    d.legend(loc="upper left")
    K.tag(d, "d")

    K.save(fig, "panel3_energy")


# ================================================================ panel 4
def panel4(D):
    """Contact separates candidates that mass cannot."""
    ch = D["challenges"]
    dec = D["decoy"]
    fig, ax = K.panel(three_d=(2,))
    vmap = {r["id"]: r["verdict"] for r in ch}

    # (a) top vs same-mass decoy mean, one point per challenge
    a = ax[0]
    for v in VORDER:
        sub = [r for r in dec if vmap.get(r["id"]) == v]
        if not sub:
            continue
        a.scatter([r["decoy_mean"] for r in sub], [r["top"] for r in sub],
                  s=32, c=VCOL[v], edgecolor="white", linewidth=0.6,
                  zorder=4 if v == "licensed" else 3)
    lo = 0.0
    hi = max(max(r["top"] for r in dec), max(r["decoy_mean"] for r in dec)) * 1.06
    a.plot([lo, hi], [lo, hi], color=MUTED, linewidth=1.3, linestyle="--",
           zorder=2)
    wins = sum(1 for r in dec if r["top"] > r["decoy_mean"])
    a.text(0.04, 0.95, "top above line in %d of %d" % (wins, len(dec)),
           transform=a.transAxes, fontsize=7.2, color=INK, va="top")
    a.set_xlabel("mean contact of same-mass decoys")
    a.set_ylabel("contact of top candidate")
    a.set_title("Decoy separation (C1)")
    vlegend(a, by_verdict(ch), loc="lower right")
    K.tag(a, "a")

    # (b) ordered contact profile within each challenge
    b = ax[1]
    for v in ("decline-unsupported", "decline-ambiguous", "licensed"):
        for r in ch:
            if r["verdict"] != v:
                continue
            tc = r["top_contacts"]
            if len(tc) < 2:
                continue
            b.plot(range(1, len(tc) + 1), tc, color=VCOL[v],
                   alpha=0.85 if v == "licensed" else 0.30,
                   linewidth=2.0 if v == "licensed" else 1.0,
                   zorder=5 if v == "licensed" else 2)
    b.set_xlabel("candidate rank within challenge")
    b.set_ylabel("contact $\\kappa$")
    b.set_title("Licensed profiles drop after first place")
    vlegend(b, by_verdict(ch), loc="upper right")
    K.tag(b, "b")

    # (c) 3-D: degeneracy, contact, margin -- by verdict
    c = ax[2]
    for v in VORDER:
        sub = [r for r in ch if r["verdict"] == v]
        if not sub:
            continue
        c.scatter([math.log10(max(1, r["n_mass_candidates"])) for r in sub],
                  [r["contact"] for r in sub],
                  [r["margin"] if r["margin"] is not None else 0.0 for r in sub],
                  s=34, c=VCOL[v], edgecolor="white", linewidth=0.5,
                  depthshade=False, label=VLAB[v])
    c.set_xlabel("$\\log_{10}$ candidates", labelpad=-2)
    c.set_ylabel("contact $\\kappa$", labelpad=-2)
    c.set_zlabel("margin", labelpad=-3)
    c.set_title("The licensed set is a corner, not a slice")
    c.view_init(elev=20, azim=-62)
    c.legend(loc="upper left", bbox_to_anchor=(-0.04, 1.03))
    K.tag(c, "c", three_d=True)

    # (d) contact vs uncapped degeneracy
    d = ax[3]
    for v in VORDER:
        sub = [r for r in ch if r["verdict"] == v]
        if not sub:
            continue
        d.scatter([r["n_mass_candidates"] for r in sub],
                  [r["contact"] for r in sub],
                  s=32, c=VCOL[v], edgecolor="white", linewidth=0.6,
                  zorder=4 if v == "licensed" else 3)
    d.axhline(0.30, color=MUTED, linewidth=1.3, linestyle="--", zorder=2)
    d.text(d.get_xlim()[1], 0.315, "floor $\\beta$ = 0.30", ha="right",
           va="bottom", fontsize=6.8, color=MUTED)
    d.set_xscale("log")
    d.set_xlabel("formulas within 8 ppm (uncapped)")
    d.set_ylabel("contact $\\kappa$ of top candidate")
    d.set_title("Licensing needs a small candidate set")
    K.tag(d, "d")

    K.save(fig, "panel4_contact")


# ================================================================ panel 5
def panel5(D):
    """Three controls: structural, not size, not coincidence."""
    sh = D["shuffle"]
    sz = D["size"]
    fig, ax = K.panel(three_d=(2,))

    # (a) own vs foreign contact
    a = ax[0]
    a.scatter([r["foreign"] for r in sh], [r["own"] for r in sh],
              s=26, c=C1, edgecolor="white", linewidth=0.5, zorder=3)
    hi = max(max(r["own"] for r in sh), max(r["foreign"] for r in sh)) * 1.06
    a.plot([0, hi], [0, hi], color=MUTED, linewidth=1.3, linestyle="--",
           zorder=2)
    mo = sum(r["own"] for r in sh) / len(sh)
    mf = sum(r["foreign"] for r in sh) / len(sh)
    a.scatter([mf], [mo], s=110, c=C2, marker="D", edgecolor="white",
              linewidth=1.0, zorder=6)
    a.annotate("mean %.3f / %.3f" % (mo, mf), (mf, mo),
               textcoords="offset points", xytext=(11, -9),
               fontsize=7.2, color=C2, va="center", ha="left")
    wins = sum(1 for r in sh if r["own"] > r["foreign"])
    a.text(0.04, 0.95, "own > foreign in %d of %d" % (wins, len(sh)),
           transform=a.transAxes, fontsize=7.2, color=INK, va="top")
    a.set_xlabel("contact on a foreign fragment ladder")
    a.set_ylabel("contact on its own ladder")
    a.set_title("Shuffle control (C2)")
    K.tag(a, "a")

    # (b) sorted paired drop
    b = ax[1]
    drops = sorted(r["own"] - r["foreign"] for r in sh)
    cols = [GOOD if d > 0 else BAD for d in drops]
    b.bar(range(len(drops)), drops, color=cols, width=0.82,
          edgecolor="white", linewidth=0.6, zorder=3)
    b.axhline(0, color=INK2, linewidth=1.0, zorder=4)
    md = sum(drops) / len(drops)
    b.axhline(md, color=C2, linewidth=2.0, linestyle="--", zorder=5)
    b.annotate(" mean drop %+.3f" % md, (0, md), fontsize=7.2, color=C2,
               va="bottom", ha="left")
    b.set_xlabel("shuffle pair (sorted)")
    b.set_ylabel("contact drop, own $-$ foreign")
    b.set_title("The collapse is general, not an outlier")
    K.tag(b, "b")

    # (c) 3-D: heavy atoms, mass error, contact
    c = ax[2]
    xs = [r["heavy"] for r in sz]
    ys = [r["ppm"] for r in sz]
    zs = [r["contact"] for r in sz]
    c.scatter(xs, ys, zs, s=4, c=C1, alpha=0.20, linewidth=0,
              depthshade=False)
    tops = [r for r in sz if r["is_top"]]
    c.scatter([r["heavy"] for r in tops], [r["ppm"] for r in tops],
              [r["contact"] for r in tops], s=22, c=C2, linewidth=0.4,
              edgecolor="white", depthshade=False, label="top-ranked")
    c.set_xlabel("heavy atoms", labelpad=-2)
    c.set_ylabel("ppm", labelpad=-2)
    c.set_zlabel("contact $\\kappa$", labelpad=-3)
    c.set_title("Flat in the size direction (n = %d)" % len(sz))
    c.view_init(elev=18, azim=-62)
    c.legend(loc="upper left", bbox_to_anchor=(-0.04, 1.03))
    K.tag(c, "c", three_d=True)

    # (d) contact vs heavy atoms with fit
    d = ax[3]
    d.scatter(xs, zs, s=5, c=C1, alpha=0.20, linewidth=0, zorder=2)
    a0, b0 = linfit(xs, zs)
    xr = [min(xs), max(xs)]
    d.plot(xr, [a0 + b0 * x for x in xr], color=C2, linewidth=2.4, zorder=5)
    r = pearson(xs, zs)
    d.text(0.04, 0.95, "Pearson $r$ = %+.4f" % r, transform=d.transAxes,
           fontsize=8, color=INK, va="top")
    d.text(0.04, 0.87, "%.1f%% of variance" % (100 * r * r),
           transform=d.transAxes, fontsize=7.2, color=INK2, va="top")
    d.set_xlabel("heavy-atom count of candidate")
    d.set_ylabel("contact $\\kappa$")
    d.set_title("Size confound (C3)")
    K.tag(d, "d")

    K.save(fig, "panel5_controls")


# ================================================================ panel 6
def panel6(D):
    """The floor, the decline, and what declining buys."""
    ch = D["challenges"]
    fig, ax = K.panel(three_d=(2,))
    BETA, MU = 0.30, 0.10

    def marg(r):
        return r["margin"] if r["margin"] is not None else 0.0

    # (a) contact vs margin with the licensing region
    a = ax[0]
    for v in VORDER:
        sub = [r for r in ch if r["verdict"] == v]
        if not sub:
            continue
        a.scatter([marg(r) for r in sub], [r["contact"] for r in sub],
                  s=34, c=VCOL[v], edgecolor="white", linewidth=0.6,
                  zorder=4 if v == "licensed" else 3)
    xmax = max(marg(r) for r in ch) * 1.08
    a.axvspan(MU, xmax, ymin=0, ymax=1, color=C1, alpha=0.05, zorder=1)
    a.axhline(BETA, color=MUTED, linewidth=1.3, linestyle="--", zorder=2)
    a.axvline(MU, color=MUTED, linewidth=1.3, linestyle="--", zorder=2)
    a.text(MU, 0.02, " $\\mu$ = 0.10", fontsize=6.8, color=MUTED,
           va="bottom", ha="left")
    a.text(xmax, BETA + 0.015, "$\\beta$ = 0.30 ", fontsize=6.8,
           color=MUTED, va="bottom", ha="right")
    a.set_xlim(0, xmax)
    a.set_xlabel("margin over nearest genuine rival")
    a.set_ylabel("contact $\\kappa$")
    a.set_title("The licensing region")
    vlegend(a, by_verdict(ch), loc="lower right")
    K.tag(a, "a")

    # (b) licensed count as beta is swept
    b = ax[1]
    betas = [i / 100.0 for i in range(0, 91, 2)]
    n_lic = [sum(1 for r in ch if r["contact"] >= t and marg(r) >= MU)
             for t in betas]
    n_sup = [sum(1 for r in ch if r["contact"] >= t) for t in betas]
    b.plot(betas, n_sup, color=C2, linewidth=2.2, zorder=3)
    b.annotate(" above floor only", (betas[-1], n_sup[-1]), fontsize=7,
               color=C2, va="center", ha="left")
    b.plot(betas, n_lic, color=C1, linewidth=2.6, zorder=4)
    b.annotate(" licensed", (betas[-1], n_lic[-1]), fontsize=7, color=C1,
               va="center", ha="left")
    b.axvline(BETA, color=MUTED, linewidth=1.3, linestyle="--", zorder=2)
    k = min(range(len(betas)), key=lambda i: abs(betas[i] - BETA))
    b.scatter([BETA], [n_lic[k]], s=70, c=C1, edgecolor="white",
              linewidth=1.0, zorder=6)
    b.annotate("  %d at $\\beta$ = 0.30" % n_lic[k], (BETA, n_lic[k]),
               fontsize=7.2, color=INK, va="bottom", ha="left")
    b.set_xlabel("contact floor $\\beta$")
    b.set_ylabel("challenges (of %d)" % len(ch))
    b.set_title("Yield as the floor is swept")
    K.tag(b, "b")

    # (c) 3-D: contact, margin, degeneracy
    c = ax[2]
    for v in VORDER:
        sub = [r for r in ch if r["verdict"] == v]
        if not sub:
            continue
        c.scatter([r["contact"] for r in sub], [marg(r) for r in sub],
                  [math.log10(max(1, r["n_mass_candidates"])) for r in sub],
                  s=34, c=VCOL[v], edgecolor="white", linewidth=0.5,
                  depthshade=False, label=VLAB[v])
    zlo, zhi = c.get_zlim()
    xs2 = np.array([[BETA, BETA], [1.0, 1.0]])
    ys2 = np.array([[MU, max(marg(r) for r in ch) * 1.05]] * 2)
    zs2 = np.full_like(xs2, zlo, dtype=float)
    c.plot_surface(xs2, ys2, zs2, color=C1, alpha=0.10, linewidth=0)
    c.set_xlabel("contact $\\kappa$", labelpad=-2)
    c.set_ylabel("margin", labelpad=-2)
    c.set_zlabel("$\\log_{10}$ candidates", labelpad=-3)
    c.set_title("Two floors cut along different axes")
    c.view_init(elev=20, azim=-60)
    c.legend(loc="upper left", bbox_to_anchor=(-0.04, 1.03))
    K.tag(c, "c", three_d=True)

    # (d) contact distribution by verdict class
    d = ax[3]
    for k, v in enumerate(VORDER):
        vals = [r["contact"] for r in ch if r["verdict"] == v]
        if not vals:
            continue
        xs = np.full(len(vals), k, dtype=float)
        xs = xs + np.linspace(-0.17, 0.17, len(vals))
        d.scatter(xs, vals, s=24, c=VCOL[v], edgecolor="white",
                  linewidth=0.5, zorder=3)
        m = sum(vals) / len(vals)
        d.plot([k - 0.30, k + 0.30], [m, m], color=INK, linewidth=2.2,
               zorder=5)
        d.annotate("%.3f\n(n=%d)" % (m, len(vals)), (k + 0.33, m),
                   fontsize=7, color=INK, va="center", ha="left")
    d.axhline(BETA, color=MUTED, linewidth=1.3, linestyle="--", zorder=2)
    d.set_xticks(range(3))
    d.set_xticklabels(["licensed", "ambiguous", "unsupported"], fontsize=7)
    d.set_xlim(-0.55, 2.95)
    d.set_ylabel("contact $\\kappa$")
    d.set_title("Ambiguous cases fail on margin, not evidence")
    K.tag(d, "d")

    K.save(fig, "panel6_licensing")


def main():
    D = K.load()
    print("challenges  :", len(D["challenges"]))
    print("ladder rows :", len(D["ladder"]))
    print("shuffle     :", len(D["shuffle"]))
    print("decoy       :", len(D["decoy"]))
    print("candidates  :", len(D["size"]))
    print("degeneracy  :", len(D["degeneracy_vs_ppm"]))
    print()
    for fn in (panel1, panel2, panel3, panel4, panel5, panel6):
        fn(D)


if __name__ == "__main__":
    main()
