#!/usr/bin/env python3
"""
Generate ion journey visualizations for all completed pipeline results.
"""

import sys
from pathlib import Path
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# Add parent directories to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))
sys.path.insert(0, str(script_dir.parent.parent.parent))

from ion_journey import (
    load_pipeline_results, extract_sample_ion,
    draw_stage_1_injection, draw_stage_2_chromatography,
    draw_stage_3_ionization, draw_stage_4_ms1,
    draw_stage_5_sentropy, draw_stage_6_partition,
    draw_stage_7_thermodynamics, draw_stage_8_droplet,
    COLORS, IonState
)

# Dataset metadata
DATASETS = {
    'H11_BD_A_neg_hilic': {
        'platform': 'HILIC',
        'mode': 'Negative ESI',
        'type': 'Metabolomics',
        'lab': 'Lab A'
    },
    'PL_Neg_Waters_qTOF': {
        'platform': 'Waters qTOF',
        'mode': 'Negative ESI',
        'type': 'Phospholipids',
        'lab': 'Lab B'
    },
    'TG_Pos_Thermo_Orbi': {
        'platform': 'Thermo Orbitrap',
        'mode': 'Positive ESI',
        'type': 'Triglycerides',
        'lab': 'Lab C'
    },
    'BSA1': {
        'platform': 'Unknown',
        'mode': 'Unknown',
        'type': 'Proteomics (BSA)',
        'lab': 'Lab D'
    },
}


def find_results_dir(base_dir: Path, pattern: str) -> Path:
    """Find the most recent results directory matching pattern."""
    matches = sorted(base_dir.glob(f'{pattern}_*'))
    if matches:
        return matches[-1]
    return None


def generate_ion_journey(results_dir: Path, name: str, metadata: dict, output_dir: Path) -> bool:
    """Generate ion journey visualization for one dataset."""
    try:
        stages = load_pipeline_results(results_dir)
        if not stages:
            print(f"  No stages found for {name}")
            return False

        ion = extract_sample_ion(stages)

        # Create figure
        fig = plt.figure(figsize=(12, 7))

        # Header
        fig.suptitle(f'Ion Journey: {name.replace("_", " ")}', fontsize=12, fontweight='bold', y=0.98)
        subtitle = f"{metadata['platform']} | {metadata['mode']} | {metadata['type']} | {metadata['lab']}"
        fig.text(0.5, 0.94, subtitle, ha='center', fontsize=10, style='italic', color='#457B9D')
        ion_info = f"m/z = {ion.mz:.2f}, RT = {ion.rt:.1f} min, State = {ion.categorical_state}"
        fig.text(0.5, 0.91, ion_info, ha='center', fontsize=9)

        # Draw all 8 stages
        ax1 = fig.add_subplot(2, 4, 1)
        draw_stage_1_injection(ax1, ion)

        ax2 = fig.add_subplot(2, 4, 2)
        draw_stage_2_chromatography(ax2, ion)

        ax3 = fig.add_subplot(2, 4, 3)
        draw_stage_3_ionization(ax3, ion)

        ax4 = fig.add_subplot(2, 4, 4)
        draw_stage_4_ms1(ax4, ion)

        ax5 = fig.add_subplot(2, 4, 5)
        draw_stage_5_sentropy(ax5, ion)

        ax6 = fig.add_subplot(2, 4, 6)
        draw_stage_6_partition(ax6, ion)

        ax7 = fig.add_subplot(2, 4, 7)
        draw_stage_7_thermodynamics(ax7, ion)

        ax8 = fig.add_subplot(2, 4, 8)
        draw_stage_8_droplet(ax8, ion)

        # Stage numbers
        for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8], 1):
            ax.text(0.02, 0.98, f'{i}', transform=ax.transAxes,
                    fontsize=10, fontweight='bold', va='top',
                    bbox=dict(boxstyle='circle', facecolor='#F4A261',
                             edgecolor='none', alpha=0.8))

        plt.tight_layout(rect=[0, 0, 1, 0.89])

        # Save
        fig.savefig(output_dir / f'ion_journey_{name}.png', dpi=300)
        fig.savefig(output_dir / f'ion_journey_{name}.pdf')
        plt.close(fig)

        print(f"  Saved: ion_journey_{name}.png")
        return True

    except Exception as e:
        print(f"  Error for {name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_comparison_panel(all_ions: dict, output_dir: Path) -> None:
    """Create multi-dataset comparison panel."""
    n = len(all_ions)
    if n == 0:
        return

    fig = plt.figure(figsize=(16, 4 * n))

    fig.suptitle('Multi-Platform Ion Journey Comparison', fontsize=14, fontweight='bold', y=0.98)

    for row, (name, data) in enumerate(all_ions.items()):
        ion = data['ion']
        meta = data['metadata']

        # Create subplots for this row
        for col, (draw_func, title) in enumerate([
            (draw_stage_1_injection, 'Sample'),
            (draw_stage_2_chromatography, 'Chrom'),
            (draw_stage_3_ionization, 'Ion'),
            (draw_stage_4_ms1, 'MS1'),
            (draw_stage_5_sentropy, 'S-Ent'),
            (draw_stage_6_partition, 'Part'),
            (draw_stage_7_thermodynamics, 'Thermo'),
            (draw_stage_8_droplet, 'Drop'),
        ]):
            ax = fig.add_subplot(n, 8, row * 8 + col + 1)
            try:
                draw_func(ax, ion)
            except:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center')
                ax.axis('off')

            if row == 0:
                ax.set_title(title, fontsize=9)

            if col == 0:
                ax.text(-0.4, 0.5, f"{name[:15]}\n{meta['platform'][:10]}",
                       transform=ax.transAxes, fontsize=7, va='center', ha='right')

    plt.tight_layout(rect=[0.08, 0, 1, 0.96])
    fig.savefig(output_dir / 'ion_journey_comparison.png', dpi=300)
    fig.savefig(output_dir / 'ion_journey_comparison.pdf')
    plt.close(fig)
    print(f"\nSaved: ion_journey_comparison.png")


def main():
    base_dir = Path(__file__).parent.parent.parent.parent
    results_base = base_dir / 'pipeline_results'
    output_dir = Path(__file__).parent / 'output'
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("GENERATING ION JOURNEY VISUALIZATIONS")
    print("=" * 60)

    all_ions = {}

    for name, metadata in DATASETS.items():
        print(f"\nProcessing: {name}")

        results_dir = find_results_dir(results_base, name)
        if results_dir is None:
            print(f"  No results found")
            continue

        print(f"  Using: {results_dir.name}")

        if generate_ion_journey(results_dir, name, metadata, output_dir):
            # Store ion for comparison
            stages = load_pipeline_results(results_dir)
            ion = extract_sample_ion(stages)
            all_ions[name] = {'ion': ion, 'metadata': metadata}

    # Generate comparison panel
    if len(all_ions) > 1:
        print("\nGenerating comparison panel...")
        create_comparison_panel(all_ions, output_dir)

    print("\n" + "=" * 60)
    print(f"Generated {len(all_ions)} ion journeys")
    print(f"Output: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
