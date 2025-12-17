#!/usr/bin/env python3
"""
UC DAVIS METABOLOMICS VISUALIZATION
====================================

Comprehensive visualization of S-Entropy analysis results across all 10 UC Davis files.

Generates:
- Figure 1: S-Entropy 3D Space (all samples)
- Figure 2: Sample Comparison (M3, M4, M5)
- Figure 3: Ionization Mode Comparison (positive vs negative)
- Figure 4: Coherence Distribution
- Figure 5: Completion Confidence Analysis
- Figure 6: Cross-Sample Trajectory Analysis
- Figure 7: Master Summary Panel

Author: Kundai Farai Sachikonye
Date: December 2025
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results" / "ucdavis_fast_analysis"
OUTPUT_DIR = SCRIPT_DIR / "results" / "visualizations" / "ucdavis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Publication-quality settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2

# Color palette for samples
SAMPLE_COLORS = {
    'M3': '#E64B35',  # Red
    'M4': '#4DBBD5',  # Cyan
    'M5': '#00A087',  # Teal
}

MODE_COLORS = {
    'pos': '#3C5488',  # Blue (positive)
    'neg': '#F39B7F',  # Coral (negative)
}

# All sample info
SAMPLES = [
    {'name': 'A_M3_negPFP_03', 'mouse': 'M3', 'mode': 'neg'},
    {'name': 'A_M3_negPFP_04', 'mouse': 'M3', 'mode': 'neg'},
    {'name': 'A_M3_posPFP_01', 'mouse': 'M3', 'mode': 'pos'},
    {'name': 'A_M3_posPFP_02', 'mouse': 'M3', 'mode': 'pos'},
    {'name': 'A_M4_negPFP_03', 'mouse': 'M4', 'mode': 'neg'},
    {'name': 'A_M4_posPFP_01', 'mouse': 'M4', 'mode': 'pos'},
    {'name': 'A_M4_posPFP_02', 'mouse': 'M4', 'mode': 'pos'},
    {'name': 'A_M5_negPFP_03', 'mouse': 'M5', 'mode': 'neg'},
    {'name': 'A_M5_negPFP_04', 'mouse': 'M5', 'mode': 'neg'},
    {'name': 'A_M5_posPFP_01', 'mouse': 'M5', 'mode': 'pos'},
]


def load_all_data():
    """Load data from all 10 samples."""
    print("=" * 60)
    print("LOADING UC DAVIS ANALYSIS RESULTS")
    print("=" * 60)

    all_data = []

    for sample in SAMPLES:
        sample_dir = RESULTS_DIR / sample['name']

        if not sample_dir.exists():
            print(f"⚠ Missing: {sample['name']}")
            continue

        # Load sentropy features
        sentropy_path = sample_dir / 'stage_02_sentropy' / 'sentropy_features.csv'
        if sentropy_path.exists():
            df = pd.read_csv(sentropy_path)
            df['sample'] = sample['name']
            df['mouse'] = sample['mouse']
            df['mode'] = sample['mode']
            all_data.append(df)
            print(f"✓ {sample['name']}: {len(df)} spectra")
        else:
            print(f"⚠ No sentropy data: {sample['name']}")

    if not all_data:
        print("ERROR: No data loaded!")
        return None

    combined = pd.concat(all_data, ignore_index=True)
    print(f"\n✓ Total: {len(combined)} spectra across {len(all_data)} samples")

    return combined


def load_coherence_data():
    """Load coherence data from all samples."""
    all_coherence = []

    for sample in SAMPLES:
        coherence_path = RESULTS_DIR / sample['name'] / 'stage_03_bmd' / 'coherence_results.csv'
        if coherence_path.exists():
            df = pd.read_csv(coherence_path)
            df['sample'] = sample['name']
            df['mouse'] = sample['mouse']
            df['mode'] = sample['mode']
            all_coherence.append(df)

    if all_coherence:
        return pd.concat(all_coherence, ignore_index=True)
    return None


def load_completion_data():
    """Load completion data from all samples."""
    all_completion = []

    for sample in SAMPLES:
        completion_path = RESULTS_DIR / sample['name'] / 'stage_04_completion' / 'completion_results.csv'
        if completion_path.exists():
            df = pd.read_csv(completion_path)
            df['sample'] = sample['name']
            df['mouse'] = sample['mouse']
            df['mode'] = sample['mode']
            all_completion.append(df)

    if all_completion:
        return pd.concat(all_completion, ignore_index=True)
    return None


def load_metrics():
    """Load pipeline metrics from all samples."""
    metrics = []

    for sample in SAMPLES:
        results_path = RESULTS_DIR / sample['name'] / 'pipeline_results.json'
        if results_path.exists():
            with open(results_path) as f:
                data = json.load(f)
                data['sample'] = sample['name']
                data['mouse'] = sample['mouse']
                data['mode'] = sample['mode']
                metrics.append(data)

    return metrics


def create_figure1_sentropy_3d(data):
    """Figure 1: 3D S-Entropy Space with all samples."""
    print("\nCreating Figure 1: S-Entropy 3D Space...")

    fig = plt.figure(figsize=(14, 10))

    # Main 3D plot
    ax = fig.add_subplot(111, projection='3d')

    # Sample subset for clarity (max 500 per sample)
    for mouse in ['M3', 'M4', 'M5']:
        subset = data[data['mouse'] == mouse].sample(n=min(500, len(data[data['mouse'] == mouse])), random_state=42)
        ax.scatter(
            subset['s_k_mean'],
            subset['s_t_mean'],
            subset['s_e_mean'],
            c=SAMPLE_COLORS[mouse],
            alpha=0.5,
            s=10,
            label=f'Mouse {mouse} (n={len(data[data["mouse"] == mouse])})'
        )

    ax.set_xlabel('S_knowledge', fontsize=12, labelpad=10)
    ax.set_ylabel('S_time', fontsize=12, labelpad=10)
    ax.set_zlabel('S_entropy', fontsize=12, labelpad=10)
    ax.set_title('S-Entropy Coordinate Space\nUC Davis Metabolomics Dataset', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)

    # Add text annotation
    total = len(data)
    ax.text2D(0.02, 0.02, f'Total spectra: {total:,}', transform=ax.transAxes, fontsize=10)

    plt.tight_layout()

    output_path = OUTPUT_DIR / 'figure1_sentropy_3d.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✓ Saved: {output_path}")


def create_figure2_sample_comparison(data):
    """Figure 2: Sample comparison across mice."""
    print("\nCreating Figure 2: Sample Comparison...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # S_knowledge distribution by mouse
    ax = axes[0, 0]
    for mouse in ['M3', 'M4', 'M5']:
        subset = data[data['mouse'] == mouse]['s_k_mean']
        ax.hist(subset, bins=50, alpha=0.6, color=SAMPLE_COLORS[mouse], label=f'Mouse {mouse}', density=True)
    ax.set_xlabel('S_knowledge (mean)')
    ax.set_ylabel('Density')
    ax.set_title('S_knowledge Distribution')
    ax.legend()

    # S_time distribution by mouse
    ax = axes[0, 1]
    for mouse in ['M3', 'M4', 'M5']:
        subset = data[data['mouse'] == mouse]['s_t_mean']
        ax.hist(subset, bins=50, alpha=0.6, color=SAMPLE_COLORS[mouse], label=f'Mouse {mouse}', density=True)
    ax.set_xlabel('S_time (mean)')
    ax.set_ylabel('Density')
    ax.set_title('S_time Distribution')
    ax.legend()

    # S_entropy distribution by mouse
    ax = axes[0, 2]
    for mouse in ['M3', 'M4', 'M5']:
        subset = data[data['mouse'] == mouse]['s_e_mean']
        ax.hist(subset, bins=50, alpha=0.6, color=SAMPLE_COLORS[mouse], label=f'Mouse {mouse}', density=True)
    ax.set_xlabel('S_entropy (mean)')
    ax.set_ylabel('Density')
    ax.set_title('S_entropy Distribution')
    ax.legend()

    # Peak count by mouse
    ax = axes[1, 0]
    mouse_stats = data.groupby('mouse')['n_peaks'].agg(['mean', 'std'])
    bars = ax.bar(mouse_stats.index, mouse_stats['mean'],
                  yerr=mouse_stats['std'], capsize=5,
                  color=[SAMPLE_COLORS[m] for m in mouse_stats.index])
    ax.set_xlabel('Mouse')
    ax.set_ylabel('Mean Peak Count')
    ax.set_title('Spectral Complexity by Sample')

    # Boxplot of S_k by sample
    ax = axes[1, 1]
    sample_order = sorted(data['sample'].unique())
    colors = [SAMPLE_COLORS[s.split('_')[1]] for s in sample_order]
    bp = ax.boxplot([data[data['sample'] == s]['s_k_mean'].values for s in sample_order],
                    labels=[s.split('_')[1] + '\n' + s.split('_')[2][:3] for s in sample_order],
                    patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_xlabel('Sample')
    ax.set_ylabel('S_knowledge')
    ax.set_title('S_knowledge Variability')
    ax.tick_params(axis='x', rotation=45)

    # Spectra count by sample
    ax = axes[1, 2]
    counts = data.groupby('sample').size()
    colors = [SAMPLE_COLORS[s.split('_')[1]] for s in counts.index]
    bars = ax.bar(range(len(counts)), counts.values, color=colors)
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels([s.split('_')[2][:3] + '\n' + s.split('_')[1] for s in counts.index], rotation=45)
    ax.set_xlabel('Sample')
    ax.set_ylabel('Spectra Count')
    ax.set_title('Spectra per Sample')

    plt.suptitle('UC Davis Metabolomics: Sample Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path = OUTPUT_DIR / 'figure2_sample_comparison.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✓ Saved: {output_path}")


def create_figure3_ionization_mode(data):
    """Figure 3: Positive vs Negative ionization mode comparison."""
    print("\nCreating Figure 3: Ionization Mode Comparison...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 2D scatter: S_k vs S_t by mode
    ax = axes[0, 0]
    for mode, color in MODE_COLORS.items():
        subset = data[data['mode'] == mode].sample(n=min(1000, len(data[data['mode'] == mode])), random_state=42)
        label = 'Positive ESI' if mode == 'pos' else 'Negative ESI'
        ax.scatter(subset['s_k_mean'], subset['s_t_mean'], c=color, alpha=0.3, s=5, label=label)
    ax.set_xlabel('S_knowledge')
    ax.set_ylabel('S_time')
    ax.set_title('S-Entropy: S_k vs S_t')
    ax.legend()

    # 2D scatter: S_k vs S_e by mode
    ax = axes[0, 1]
    for mode, color in MODE_COLORS.items():
        subset = data[data['mode'] == mode].sample(n=min(1000, len(data[data['mode'] == mode])), random_state=42)
        label = 'Positive ESI' if mode == 'pos' else 'Negative ESI'
        ax.scatter(subset['s_k_mean'], subset['s_e_mean'], c=color, alpha=0.3, s=5, label=label)
    ax.set_xlabel('S_knowledge')
    ax.set_ylabel('S_entropy')
    ax.set_title('S-Entropy: S_k vs S_e')
    ax.legend()

    # Distribution comparison
    ax = axes[1, 0]
    for mode, color in MODE_COLORS.items():
        subset = data[data['mode'] == mode]
        label = f"Positive (n={len(subset)})" if mode == 'pos' else f"Negative (n={len(subset)})"
        ax.hist(subset['s_k_mean'], bins=50, alpha=0.6, color=color, label=label, density=True)
    ax.set_xlabel('S_knowledge')
    ax.set_ylabel('Density')
    ax.set_title('S_knowledge by Ionization Mode')
    ax.legend()

    # Stats summary
    ax = axes[1, 1]
    stats_data = []
    for mode in ['pos', 'neg']:
        subset = data[data['mode'] == mode]
        stats_data.append({
            'Mode': 'Positive' if mode == 'pos' else 'Negative',
            'Count': len(subset),
            'S_k mean': subset['s_k_mean'].mean(),
            'S_t mean': subset['s_t_mean'].mean(),
            'S_e mean': subset['s_e_mean'].mean(),
            'Peaks mean': subset['n_peaks'].mean()
        })

    stats_df = pd.DataFrame(stats_data)
    ax.axis('off')
    table = ax.table(
        cellText=[[f"{v:.3f}" if isinstance(v, float) else str(v) for v in row]
                  for row in stats_df.values],
        colLabels=stats_df.columns,
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax.set_title('Ionization Mode Statistics', fontsize=12, fontweight='bold', pad=20)

    plt.suptitle('Ionization Mode Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path = OUTPUT_DIR / 'figure3_ionization_mode.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✓ Saved: {output_path}")


def create_figure4_coherence(coherence_data):
    """Figure 4: Coherence distribution analysis."""
    print("\nCreating Figure 4: Coherence Analysis...")

    if coherence_data is None or len(coherence_data) == 0:
        print("⚠ No coherence data available")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Overall coherence distribution
    ax = axes[0, 0]
    ax.hist(coherence_data['coherence'], bins=50, color='#3C5488', alpha=0.7, edgecolor='white')
    ax.axvline(coherence_data['coherence'].mean(), color='red', linestyle='--',
               label=f"Mean: {coherence_data['coherence'].mean():.4f}")
    ax.set_xlabel('Coherence Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Coherence Distribution (All Samples)')
    ax.legend()

    # Coherence by mouse
    ax = axes[0, 1]
    for mouse in ['M3', 'M4', 'M5']:
        subset = coherence_data[coherence_data['mouse'] == mouse]['coherence']
        ax.hist(subset, bins=30, alpha=0.5, color=SAMPLE_COLORS[mouse], label=f'Mouse {mouse}', density=True)
    ax.set_xlabel('Coherence Score')
    ax.set_ylabel('Density')
    ax.set_title('Coherence by Sample')
    ax.legend()

    # Coherence by mode
    ax = axes[1, 0]
    for mode, color in MODE_COLORS.items():
        subset = coherence_data[coherence_data['mode'] == mode]['coherence']
        label = 'Positive ESI' if mode == 'pos' else 'Negative ESI'
        ax.hist(subset, bins=30, alpha=0.6, color=color, label=label, density=True)
    ax.set_xlabel('Coherence Score')
    ax.set_ylabel('Density')
    ax.set_title('Coherence by Ionization Mode')
    ax.legend()

    # Coherence vs divergence
    ax = axes[1, 1]
    sample_coherence = coherence_data.sample(n=min(2000, len(coherence_data)), random_state=42)
    colors = [SAMPLE_COLORS[m] for m in sample_coherence['mouse']]
    ax.scatter(sample_coherence['coherence'], sample_coherence['divergence'],
               c=colors, alpha=0.3, s=5)
    ax.plot([0, 1], [1, 0], 'k--', alpha=0.5, label='coherence + divergence = 1')
    ax.set_xlabel('Coherence')
    ax.set_ylabel('Divergence')
    ax.set_title('Coherence vs Divergence')
    ax.legend()

    plt.suptitle('BMD Hardware Grounding: Coherence Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path = OUTPUT_DIR / 'figure4_coherence.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✓ Saved: {output_path}")


def create_figure5_completion(completion_data):
    """Figure 5: Completion confidence analysis."""
    print("\nCreating Figure 5: Completion Analysis...")

    if completion_data is None or len(completion_data) == 0:
        print("⚠ No completion data available")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Completion confidence distribution
    ax = axes[0, 0]
    ax.hist(completion_data['completion_confidence'], bins=50, color='#009E73', alpha=0.7, edgecolor='white')
    mean_conf = completion_data['completion_confidence'].mean()
    ax.axvline(mean_conf, color='red', linestyle='--', label=f"Mean: {mean_conf:.4f}")
    ax.set_xlabel('Completion Confidence')
    ax.set_ylabel('Frequency')
    ax.set_title('Completion Confidence Distribution')
    ax.legend()

    # Confidence by sample
    ax = axes[0, 1]
    sample_means = completion_data.groupby('sample')['completion_confidence'].mean().sort_values()
    colors = [SAMPLE_COLORS[s.split('_')[1]] for s in sample_means.index]
    bars = ax.barh(range(len(sample_means)), sample_means.values, color=colors)
    ax.set_yticks(range(len(sample_means)))
    ax.set_yticklabels([s.replace('A_', '').replace('PFP_', '') for s in sample_means.index])
    ax.set_xlabel('Mean Completion Confidence')
    ax.set_title('Completion Confidence by Sample')

    # S_k vs confidence
    ax = axes[1, 0]
    sample = completion_data.sample(n=min(2000, len(completion_data)), random_state=42)
    colors = [SAMPLE_COLORS[m] for m in sample['mouse']]
    ax.scatter(sample['s_k'], sample['completion_confidence'], c=colors, alpha=0.3, s=5)
    ax.set_xlabel('S_knowledge')
    ax.set_ylabel('Completion Confidence')
    ax.set_title('S_knowledge vs Confidence')

    # Coherence vs confidence
    ax = axes[1, 1]
    ax.scatter(sample['coherence'], sample['completion_confidence'], c=colors, alpha=0.3, s=5)
    ax.set_xlabel('Coherence')
    ax.set_ylabel('Completion Confidence')
    ax.set_title('Coherence vs Confidence')

    # Add legend
    legend_elements = [mpatches.Patch(facecolor=SAMPLE_COLORS[m], label=f'Mouse {m}') for m in ['M3', 'M4', 'M5']]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))

    plt.suptitle('Categorical Completion Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path = OUTPUT_DIR / 'figure5_completion.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✓ Saved: {output_path}")


def create_figure6_trajectories(data):
    """Figure 6: S-Entropy trajectory analysis."""
    print("\nCreating Figure 6: Trajectory Analysis...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # S_k trajectory (sorted by scan)
    ax = axes[0]
    for mouse, color in SAMPLE_COLORS.items():
        sample_data = data[data['mouse'] == mouse].sort_values('scan_id').reset_index()
        # Take first sample of this mouse
        first_sample = sample_data['sample'].iloc[0] if len(sample_data) > 0 else None
        if first_sample:
            subset = sample_data[sample_data['sample'] == first_sample].iloc[:200]
            ax.plot(range(len(subset)), subset['s_k_mean'], color=color, alpha=0.7, label=f'{mouse}')
    ax.set_xlabel('Scan Index')
    ax.set_ylabel('S_knowledge')
    ax.set_title('S_knowledge Trajectory')
    ax.legend()

    # S_t trajectory
    ax = axes[1]
    for mouse, color in SAMPLE_COLORS.items():
        sample_data = data[data['mouse'] == mouse].sort_values('scan_id').reset_index()
        first_sample = sample_data['sample'].iloc[0] if len(sample_data) > 0 else None
        if first_sample:
            subset = sample_data[sample_data['sample'] == first_sample].iloc[:200]
            ax.plot(range(len(subset)), subset['s_t_mean'], color=color, alpha=0.7, label=f'{mouse}')
    ax.set_xlabel('Scan Index')
    ax.set_ylabel('S_time')
    ax.set_title('S_time Trajectory')
    ax.legend()

    # S_e trajectory
    ax = axes[2]
    for mouse, color in SAMPLE_COLORS.items():
        sample_data = data[data['mouse'] == mouse].sort_values('scan_id').reset_index()
        first_sample = sample_data['sample'].iloc[0] if len(sample_data) > 0 else None
        if first_sample:
            subset = sample_data[sample_data['sample'] == first_sample].iloc[:200]
            ax.plot(range(len(subset)), subset['s_e_mean'], color=color, alpha=0.7, label=f'{mouse}')
    ax.set_xlabel('Scan Index')
    ax.set_ylabel('S_entropy')
    ax.set_title('S_entropy Trajectory')
    ax.legend()

    plt.suptitle('S-Entropy Coordinate Trajectories (First 200 Scans)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path = OUTPUT_DIR / 'figure6_trajectories.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✓ Saved: {output_path}")


def create_figure7_master_summary(data, coherence_data, completion_data, metrics):
    """Figure 7: Master summary panel."""
    print("\nCreating Figure 7: Master Summary...")

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Panel A: 3D S-Entropy space
    ax = fig.add_subplot(gs[0, 0], projection='3d')
    for mouse in ['M3', 'M4', 'M5']:
        subset = data[data['mouse'] == mouse].sample(n=min(300, len(data[data['mouse'] == mouse])), random_state=42)
        ax.scatter(subset['s_k_mean'], subset['s_t_mean'], subset['s_e_mean'],
                   c=SAMPLE_COLORS[mouse], alpha=0.5, s=5, label=mouse)
    ax.set_xlabel('S_k', fontsize=8)
    ax.set_ylabel('S_t', fontsize=8)
    ax.set_zlabel('S_e', fontsize=8)
    ax.set_title('A. S-Entropy Space', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)

    # Panel B: S_k distribution
    ax = fig.add_subplot(gs[0, 1])
    for mouse in ['M3', 'M4', 'M5']:
        subset = data[data['mouse'] == mouse]['s_k_mean']
        ax.hist(subset, bins=40, alpha=0.5, color=SAMPLE_COLORS[mouse], label=mouse, density=True)
    ax.set_xlabel('S_knowledge')
    ax.set_ylabel('Density')
    ax.set_title('B. S_knowledge Distribution', fontsize=10, fontweight='bold')
    ax.legend()

    # Panel C: Mode comparison
    ax = fig.add_subplot(gs[0, 2])
    mode_counts = data.groupby('mode').size()
    colors = [MODE_COLORS[m] for m in mode_counts.index]
    bars = ax.bar(['Negative', 'Positive'], [mode_counts.get('neg', 0), mode_counts.get('pos', 0)], color=colors)
    ax.set_ylabel('Spectra Count')
    ax.set_title('C. Ionization Mode', fontsize=10, fontweight='bold')
    for bar, count in zip(bars, [mode_counts.get('neg', 0), mode_counts.get('pos', 0)]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                f'{count:,}', ha='center', fontsize=9)

    # Panel D: Coherence distribution
    ax = fig.add_subplot(gs[1, 0])
    if coherence_data is not None:
        ax.hist(coherence_data['coherence'], bins=40, color='#4DBBD5', alpha=0.7, edgecolor='white')
        ax.axvline(coherence_data['coherence'].mean(), color='red', linestyle='--')
        ax.set_xlabel('Coherence')
        ax.set_ylabel('Frequency')
    ax.set_title('D. BMD Coherence', fontsize=10, fontweight='bold')

    # Panel E: Completion confidence
    ax = fig.add_subplot(gs[1, 1])
    if completion_data is not None:
        ax.hist(completion_data['completion_confidence'], bins=40, color='#00A087', alpha=0.7, edgecolor='white')
        ax.axvline(completion_data['completion_confidence'].mean(), color='red', linestyle='--')
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Frequency')
    ax.set_title('E. Completion Confidence', fontsize=10, fontweight='bold')

    # Panel F: Peak count by sample
    ax = fig.add_subplot(gs[1, 2])
    sample_peaks = data.groupby('mouse')['n_peaks'].mean()
    bars = ax.bar(sample_peaks.index, sample_peaks.values,
                  color=[SAMPLE_COLORS[m] for m in sample_peaks.index])
    ax.set_xlabel('Mouse')
    ax.set_ylabel('Mean Peaks/Spectrum')
    ax.set_title('F. Spectral Complexity', fontsize=10, fontweight='bold')

    # Panel G: Summary statistics table
    ax = fig.add_subplot(gs[2, :])
    ax.axis('off')

    # Create summary table
    summary_stats = []
    for mouse in ['M3', 'M4', 'M5']:
        subset = data[data['mouse'] == mouse]
        summary_stats.append({
            'Sample': f'Mouse {mouse}',
            'Spectra': f'{len(subset):,}',
            'S_k (mean±std)': f'{subset["s_k_mean"].mean():.3f}±{subset["s_k_mean"].std():.3f}',
            'S_t (mean±std)': f'{subset["s_t_mean"].mean():.3f}±{subset["s_t_mean"].std():.3f}',
            'S_e (mean±std)': f'{subset["s_e_mean"].mean():.3f}±{subset["s_e_mean"].std():.3f}',
            'Peaks (mean)': f'{subset["n_peaks"].mean():.0f}'
        })

    # Add total row
    summary_stats.append({
        'Sample': 'TOTAL',
        'Spectra': f'{len(data):,}',
        'S_k (mean±std)': f'{data["s_k_mean"].mean():.3f}±{data["s_k_mean"].std():.3f}',
        'S_t (mean±std)': f'{data["s_t_mean"].mean():.3f}±{data["s_t_mean"].std():.3f}',
        'S_e (mean±std)': f'{data["s_e_mean"].mean():.3f}±{data["s_e_mean"].std():.3f}',
        'Peaks (mean)': f'{data["n_peaks"].mean():.0f}'
    })

    summary_df = pd.DataFrame(summary_stats)
    table = ax.table(
        cellText=summary_df.values,
        colLabels=summary_df.columns,
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    ax.set_title('G. Summary Statistics', fontsize=10, fontweight='bold', y=0.9)

    plt.suptitle('UC Davis Metabolomics: S-Entropy Analysis Summary',
                 fontsize=14, fontweight='bold', y=0.98)

    output_path = OUTPUT_DIR / 'figure7_master_summary.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✓ Saved: {output_path}")


def main():
    """Generate all visualizations."""
    print("=" * 70)
    print("UC DAVIS METABOLOMICS VISUALIZATION SUITE")
    print("=" * 70)
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")

    # Load all data
    data = load_all_data()
    if data is None:
        print("ERROR: Failed to load data")
        return 1

    coherence_data = load_coherence_data()
    completion_data = load_completion_data()
    metrics = load_metrics()

    print(f"\nLoaded coherence data: {len(coherence_data) if coherence_data is not None else 0} rows")
    print(f"Loaded completion data: {len(completion_data) if completion_data is not None else 0} rows")
    print(f"Loaded metrics: {len(metrics)} samples")

    # Generate all figures
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    create_figure1_sentropy_3d(data)
    create_figure2_sample_comparison(data)
    create_figure3_ionization_mode(data)
    create_figure4_coherence(coherence_data)
    create_figure5_completion(completion_data)
    create_figure6_trajectories(data)
    create_figure7_master_summary(data, coherence_data, completion_data, metrics)

    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)
    print(f"All figures saved to: {OUTPUT_DIR}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
