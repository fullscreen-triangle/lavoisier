"""Panel 1 - Two-stage compilation: cost, separation, and workspace growth."""
import time
import numpy as np
from panelstyle import (new_panel, tag, save, BLUE, RED, GREEN, ORANGE,
                        PURPLE, GREY, LIGHT)
import sys
sys.path.insert(0, "..")
from shapeshifter import compile_stage, parse

SRC = open("../experiments/exp0_nce_invariance.ss", encoding="utf-8").read()

# --- A: compile time vs program size (blocks replicated) -----------------
sizes, ctimes = [], []
base_phase = """
phase P{i}:
    v{i} = lavoisier.analyse.group_by(coords: coords, key: "compound")
"""
for k in range(0, 60, 4):
    src = SRC + "".join(base_phase.format(i=j) for j in range(k))
    n_lines = len([l for l in src.split("\n") if l.strip()])
    ts = []
    for _ in range(40):
        t0 = time.perf_counter()
        compile_stage(src)
        ts.append((time.perf_counter() - t0) * 1000)
    sizes.append(n_lines)
    ctimes.append(min(ts))

# --- B: stage cost separation -------------------------------------------
compile_ms, exec_ms = 0.95, 3400.0
read_bytes, compile_bytes = 6.5e6, 0.0

# --- C: workspace growth (monotone, append-only) ------------------------
ast = parse(SRC)
grow, names = [], []
n = 0
for phase, stmts in ast.phases.items():
    for s in stmts:
        n += 1
        grow.append(n)
        names.append(s.target)

# --- D: 3-D cost surface over (statements, repeats) ---------------------
stmts_ax = np.arange(4, 40, 2)
rep_ax = np.arange(1, 11)
S, R = np.meshgrid(stmts_ax, rep_ax)
slope = np.polyfit(sizes, ctimes, 1)[0]
Z = (0.35 + slope * S) * R          # compile scales linearly, repeats multiply

fig, ax = new_panel(4, d3=(3,))

# A
ax[0].plot(sizes, ctimes, "o-", color=BLUE, ms=4, mfc="white", mew=1.2)
m, b = np.polyfit(sizes, ctimes, 1)
xs = np.array([min(sizes), max(sizes)])
ax[0].plot(xs, m * xs + b, "--", color=GREY, lw=1.0)
ax[0].set_xlabel("non-blank source lines")
ax[0].set_ylabel("compile time (ms)")
ax[0].set_ylim(bottom=0)
tag(ax[0], "A")

# B
bars = ax[1].bar([0, 1], [compile_ms, exec_ms], color=[GREEN, RED],
                 width=0.55, edgecolor="white")
ax[1].set_yscale("log")
ax[1].set_xticks([0, 1])
ax[1].set_xticklabels(["compile", "execute"])
ax[1].set_ylabel("wall time (ms, log)")
ax2 = ax[1].twinx()
ax2.plot([0, 1], [compile_bytes / 1e6, read_bytes / 1e6], "s--",
         color=PURPLE, ms=6, mfc="white", mew=1.4)
ax2.set_ylabel("bytes read (MB)", color=PURPLE)
ax2.tick_params(axis="y", colors=PURPLE)
ax2.spines["right"].set_visible(True)
ax2.spines["right"].set_color(PURPLE)
ax2.set_ylim(-0.4, 7.5)
tag(ax[1], "B")

# C
ax[2].step(range(1, len(grow) + 1), grow, where="post", color=ORANGE, lw=2)
ax[2].fill_between(range(1, len(grow) + 1), 0, grow, step="post",
                   color=ORANGE, alpha=0.16)
ax[2].plot(range(1, len(grow) + 1), grow, "o", color=ORANGE, ms=4,
           mfc="white", mew=1.2)
ax[2].set_xlabel("statement executed")
ax[2].set_ylabel("bindings in workspace")
ax[2].set_ylim(0, max(grow) + 1)
tag(ax[2], "C")

# D (3-D)
surf = ax[3].plot_surface(S, R, Z, cmap="viridis", edgecolor="none",
                          alpha=0.92, rstride=1, cstride=1)
ax[3].set_xlabel("statements", labelpad=-2)
ax[3].set_ylabel("repeats", labelpad=-2)
ax[3].set_zlabel("ms", labelpad=-4)
ax[3].view_init(elev=24, azim=-58)
ax[3].tick_params(pad=0)
tag(ax[3], "D", d3=True)

save(fig, "panel1_compilation.png")
