#!/usr/bin/env python3
"""
Generate 4 publication panels for Paper 3: Observation-Based Mass Computing.
Each panel: white background, 4 charts in a row, at least one 3D chart.
"""

import json, csv, os, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.facecolor': 'white', 'font.size': 8,
    'axes.titlesize': 9, 'axes.labelsize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 6, 'figure.dpi': 300,
})

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, "results")
FIG = os.path.join(BASE, "figures")
os.makedirs(FIG, exist_ok=True)

def load_csv(name):
    with open(os.path.join(RES, name), newline='') as f:
        return list(csv.DictReader(f))

def load_json(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)

TYPE_COLORS = {'diatomic':'#2196F3','triatomic':'#4CAF50','tetra':'#FF9800','poly':'#F44336'}

# ============================================================================
# Panel 1: Triple Equivalence & Observation-Computation
# ============================================================================
def panel1():
    te = load_csv("01_triple_equivalence.csv")
    oc = load_json("02_observation_computation.json")
    pp = load_json("03_four_pass_pipeline.json")

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    fig.subplots_adjust(left=0.04, right=0.97, top=0.88, bottom=0.14, wspace=0.35)

    # A: 3D — M, Omega, S for triple equivalence
    ax = fig.add_subplot(141, projection='3d')
    axes[0].set_visible(False)
    M_vals = [int(t['M_degrees']) for t in te]
    Omega_vals = [math.log10(int(t['Omega_osc'])) for t in te]
    S_vals = [float(t['S_entropy_kB']) for t in te]
    colors = [TYPE_COLORS.get(t['type'], '#999') for t in te]
    ax.scatter(M_vals, Omega_vals, S_vals, c=colors, s=30, alpha=0.8,
               edgecolors='k', linewidth=0.3)
    ax.set_xlabel('$M$ (modes)')
    ax.set_ylabel('$\\log_{10}\\Omega$')
    ax.set_zlabel('$S/k_B$')
    ax.set_title('(A) Triple Equivalence')
    ax.view_init(elev=20, azim=45)

    # B: Observation-Computation — vectorised vs pixel-by-pixel time
    ax2 = axes[1]
    methods = ['Vectorised\n(numpy)', 'Pixel-by-pixel\n(shader ref)']
    times = [oc['vectorised_time_ms'], oc['pixel_time_ms']]
    bars = ax2.bar(methods, times, color=['#2196F3', '#FF9800'], edgecolor='k', linewidth=0.3)
    ax2.set_ylabel('Time (ms)')
    ax2.set_title('(B) Obs=Comp Equivalence')
    ax2.text(0.5, 0.92, f"Max diff: {oc['max_absolute_diff']:.0e}",
             transform=ax2.transAxes, ha='center', fontsize=7, color='green')

    # C: Four-pass pipeline timing
    ax3 = axes[2]
    passes = ['Pass 1\nWave', 'Pass 2\nCoord', 'Pass 3\nBijective', 'Pass 4\nResonance']
    pass_times = [pp['pass1_wave_time_ms'], pp['pass2_coord_time_ms'],
                  pp['pass3_bijective_time_ms'], pp['pass4_resonance_time_ms']]
    colors_pass = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']
    ax3.bar(passes, pass_times, color=colors_pass, edgecolor='k', linewidth=0.3)
    ax3.set_ylabel('Time (ms)')
    ax3.set_title('(C) Four-Pass Pipeline')

    # D: Bijective & resonance scores
    ax4 = axes[3]
    labels = ['Bij.\nself', 'Bij.\nnoisy', 'Res.\nself', 'Res.\nrandom']
    scores = [pp['bijective_score_self'], pp['bijective_score_noisy'],
              pp['resonance_self_shifted'], pp['resonance_random']]
    bar_cols = ['#4CAF50', '#FF9800', '#4CAF50', '#F44336']
    ax4.bar(labels, scores, color=bar_cols, edgecolor='k', linewidth=0.3)
    ax4.set_ylabel('Score')
    ax4.set_ylim(0, 1.1)
    ax4.axhline(1.0, color='gray', linestyle=':', alpha=0.4)
    ax4.set_title('(D) Validation Scores')

    fig.savefig(os.path.join(FIG, "panel_1_triple_observation.png"), dpi=300)
    plt.close()
    print("  Panel 1 saved")


# ============================================================================
# Panel 2: Quality Metrics & Memory
# ============================================================================
def panel2():
    qm = load_json("04_quality_metrics.json")
    mem = load_csv("05_memory_scaling.csv")

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    fig.subplots_adjust(left=0.05, right=0.97, top=0.88, bottom=0.14, wspace=0.38)

    # A: 3D quality metric space (PS, NL, PC for signal vs noise)
    ax = fig.add_subplot(141, projection='3d')
    axes[0].set_visible(False)
    sig = qm['signal']
    noi = qm['pure_noise']
    ax.scatter([sig['partition_sharpness']], [sig['noise_level']],
               [sig['phase_coherence']], c='#4CAF50', s=100, marker='o',
               edgecolors='k', linewidth=0.5, label='Signal', zorder=5)
    ax.scatter([noi['partition_sharpness']], [noi['noise_level']],
               [noi['phase_coherence']], c='#F44336', s=100, marker='^',
               edgecolors='k', linewidth=0.5, label='Noise', zorder=5)
    ax.set_xlabel('Sharpness')
    ax.set_ylabel('Noise Level')
    ax.set_zlabel('Coherence')
    ax.set_title('(A) Quality Metric Space')
    ax.legend(fontsize=6)
    ax.view_init(elev=25, azim=135)

    # B: Quality metric comparison bars (signal vs noise)
    ax2 = axes[1]
    metrics = ['PS\n(norm)', 'NL', 'PC', 'IV', 'MRC']
    sig_vals = [sig['partition_sharpness']/(sig['partition_sharpness']+1),
                sig['noise_level'], sig['phase_coherence'],
                sig['interference_visibility'], sig['multiresolution_consistency']]
    noi_vals = [noi['partition_sharpness']/(noi['partition_sharpness']+1),
                noi['noise_level'], noi['phase_coherence'],
                noi['interference_visibility'], noi['multiresolution_consistency']]
    x = np.arange(len(metrics))
    w = 0.35
    ax2.bar(x - w/2, sig_vals, w, color='#4CAF50', label='Signal', edgecolor='k', linewidth=0.3)
    ax2.bar(x + w/2, noi_vals, w, color='#F44336', label='Noise', edgecolor='k', linewidth=0.3)
    ax2.set_xticks(x); ax2.set_xticklabels(metrics)
    ax2.set_ylabel('Value')
    ax2.set_title('(B) 5 Physical Observables')
    ax2.legend(fontsize=6)

    # C: Composite quality comparison
    ax3 = axes[2]
    ax3.bar(['Signal', 'Noise'],
            [sig['composite_quality'], noi['composite_quality']],
            color=['#4CAF50', '#F44336'], edgecolor='k', linewidth=0.3)
    ax3.set_ylabel('Composite Quality $Q$')
    ax3.set_title('(C) Composite Quality')
    ax3.set_ylim(0, 1.1)

    # D: Memory scaling (log-log)
    ax4 = axes[3]
    Ns = [int(m['N_database']) for m in mem]
    trad = [float(m['traditional_memory_MB']) for m in mem]
    obs = [float(m['observation_memory_MB']) for m in mem]
    ax4.plot(Ns, trad, 'o-', color='#F44336', markersize=4, label='Traditional $O(N)$')
    ax4.plot(Ns, obs, 's-', color='#4CAF50', markersize=4, label='Observation $O(1)$')
    ax4.set_xscale('log'); ax4.set_yscale('log')
    ax4.axhline(2000, color='gray', linestyle=':', alpha=0.5, label='2 GB GPU')
    ax4.set_xlabel('Database Size $N$')
    ax4.set_ylabel('Memory (MB)')
    ax4.set_title('(D) Memory Scaling')
    ax4.legend(fontsize=6)

    fig.savefig(os.path.join(FIG, "panel_2_quality_memory.png"), dpi=300)
    plt.close()
    print("  Panel 2 saved")


# ============================================================================
# Panel 3: Training Signal & Dual-Path Interference
# ============================================================================
def panel3():
    ts = load_csv("06_training_signal.csv")
    dp = load_csv("07_dual_path_interference.csv")

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    fig.subplots_adjust(left=0.05, right=0.97, top=0.88, bottom=0.14, wspace=0.35)

    # A: 3D training trajectory (epoch, quality, loss)
    ax = fig.add_subplot(141, projection='3d')
    axes[0].set_visible(False)
    epochs = [int(t['epoch']) for t in ts]
    qualities = [float(t['composite_quality']) for t in ts]
    losses = [float(t['loss']) for t in ts]
    noises = [float(t['noise_level']) for t in ts]
    ax.scatter(epochs, qualities, losses, c=epochs, cmap='viridis', s=30, alpha=0.8,
               edgecolors='k', linewidth=0.3)
    for i in range(len(epochs)-1):
        ax.plot([epochs[i], epochs[i+1]], [qualities[i], qualities[i+1]],
                [losses[i], losses[i+1]], color='gray', alpha=0.3, linewidth=0.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Quality $Q$')
    ax.set_zlabel('Loss')
    ax.set_title('(A) Training Trajectory')
    ax.view_init(elev=25, azim=45)

    # B: Training curve — quality and loss vs epoch
    ax2 = axes[1]
    ax2.plot(epochs, qualities, 'o-', color='#4CAF50', markersize=3, label='Quality $Q$')
    ax2_twin = ax2.twinx()
    ax2_twin.plot(epochs, losses, 's-', color='#F44336', markersize=3, label='Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Quality $Q$', color='#4CAF50')
    ax2_twin.set_ylabel('Loss', color='#F44336')
    ax2.set_title('(B) GPU-Supervised Training')
    ax2.legend(loc='center left', fontsize=6)
    ax2_twin.legend(loc='center right', fontsize=6)

    # C: Dual-path Sk comparison
    ax3 = axes[2]
    sk_ion = [float(d['Sk_ion']) for d in dp]
    sk_drip = [float(d['Sk_drip']) for d in dp]
    types = [d['type'] for d in dp]
    colors = [TYPE_COLORS.get(t, '#999') for t in types]
    ax3.scatter(sk_ion, sk_drip, c=colors, s=30, alpha=0.8, edgecolors='k', linewidth=0.3)
    ax3.plot([0, 1], [0, 1], 'r--', linewidth=0.8, alpha=0.5)
    ax3.set_xlabel('$S_k$ (Ion Path)')
    ax3.set_ylabel('$S_k$ (Droplet Path)')
    ax3.set_title('(C) Dual-Path Convergence')
    ax3.set_xlim(0, 1.05); ax3.set_ylim(0, 1.05)

    # D: False positive bound vs common prefix
    ax4 = axes[3]
    cpls = [int(d['common_prefix']) for d in dp]
    fp_bounds = [float(d['false_positive_bound']) for d in dp]
    ax4.scatter(cpls, fp_bounds, c=colors, s=30, alpha=0.8, edgecolors='k', linewidth=0.3)
    ax4.set_yscale('log')
    ax4.set_xlabel('Common Prefix Length')
    ax4.set_ylabel('False Positive Bound $3^{-k}$')
    ax4.set_title('(D) Interference Validation')

    fig.savefig(os.path.join(FIG, "panel_3_training_dualpath.png"), dpi=300)
    plt.close()
    print("  Panel 3 saved")


# ============================================================================
# Panel 4: Throughput, Scaling, Integration
# ============================================================================
def panel4():
    tp = load_csv("08_throughput_scaling.csv")
    mem = load_csv("05_memory_scaling.csv")
    qm = load_json("04_quality_metrics.json")
    pp = load_json("03_four_pass_pipeline.json")

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    fig.subplots_adjust(left=0.05, right=0.97, top=0.88, bottom=0.14, wspace=0.38)

    # A: 3D — N, throughput, memory
    ax = fig.add_subplot(141, projection='3d')
    axes[0].set_visible(False)
    Ns = [int(t['N']) for t in tp]
    throughputs = [float(t['throughput_batch_per_s']) for t in tp]
    batch_times = [float(t['batched_time_s']) for t in tp]
    ax.scatter(np.log10(Ns), np.log10(np.array(throughputs)+1),
               batch_times, c=np.log10(Ns), cmap='plasma', s=40,
               alpha=0.8, edgecolors='k', linewidth=0.3)
    ax.set_xlabel('$\\log_{10} N$')
    ax.set_ylabel('$\\log_{10}$ Throughput')
    ax.set_zlabel('Batch Time (s)')
    ax.set_title('(A) Performance Space')
    ax.view_init(elev=25, azim=45)

    # B: Throughput vs N
    ax2 = axes[1]
    ax2.plot(Ns, throughputs, 'o-', color='#2196F3', markersize=4)
    ax2.set_xscale('log')
    ax2.set_xlabel('Database Size $N$')
    ax2.set_ylabel('Throughput (examples/s)')
    ax2.set_title('(B) Observation Throughput')
    ax2.axhline(20e6, color='gray', linestyle=':', alpha=0.4, label='20M/s plateau')
    ax2.legend(fontsize=6)

    # C: Batch time vs N (showing O(N/batch) linear scaling)
    ax3 = axes[2]
    ax3.plot(Ns, batch_times, 'o-', color='#FF9800', markersize=4)
    ax3.set_xscale('log'); ax3.set_yscale('log')
    ax3.set_xlabel('Database Size $N$')
    ax3.set_ylabel('Batch Time (s)')
    ax3.set_title('(C) Search Time')
    ax3.axhline(10, color='gray', linestyle=':', alpha=0.4, label='10s')
    ax3.legend(fontsize=6)

    # D: Memory ratio (traditional/observation) vs N
    ax4 = axes[3]
    Ns_mem = [int(m['N_database']) for m in mem]
    ratios = [float(m['ratio']) for m in mem]
    ax4.plot(Ns_mem, ratios, 'o-', color='#9C27B0', markersize=4)
    ax4.set_xscale('log'); ax4.set_yscale('log')
    ax4.set_xlabel('Database Size $N$')
    ax4.set_ylabel('Memory Ratio (Trad/Obs)')
    ax4.set_title('(D) Memory Advantage')
    ax4.axhline(1, color='gray', linestyle=':', alpha=0.4)

    fig.savefig(os.path.join(FIG, "panel_4_throughput_scaling.png"), dpi=300)
    plt.close()
    print("  Panel 4 saved")


# ============================================================================
if __name__ == "__main__":
    print("Generating 4 panels for Paper 3...")
    panel1()
    panel2()
    panel3()
    panel4()
    print("All 4 panels generated.")
