#!/usr/bin/env python3
"""
Batch Ion Journey Generator
============================

Runs the pipeline on multiple public mzML files and generates
ion journey visualizations for each, covering:
- Multiple instruments (Waters qTOF, Thermo Orbitrap)
- Multiple labs
- Multiple ionization modes (positive, negative)
- Multiple data types (metabolomics, proteomics)
"""

import sys
import os
from pathlib import Path
import json
import subprocess
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import after path setup
from union.src.pipeline_runner import PipelineRunner
from union.src.figures.ion_journey import (
    load_pipeline_results, extract_sample_ion, create_ion_journey_panel,
    create_compact_flow_diagram, IonState
)

# Files to process - diverse selection
PUBLIC_FILES = {
    'Waters_qTOF_Neg': {
        'path': 'union/public/PL_Neg_Waters_qTOF.mzML',
        'platform': 'Waters qTOF',
        'mode': 'Negative',
        'type': 'Metabolomics (Phospholipids)',
    },
    'Thermo_Orbi_Pos': {
        'path': 'union/public/TG_Pos_Thermo_Orbi.mzML',
        'platform': 'Thermo Orbitrap',
        'mode': 'Positive',
        'type': 'Metabolomics (Triglycerides)',
    },
    'HILIC_Neg': {
        'path': 'union/public/H11_BD_A_neg_hilic.mzML',
        'platform': 'Unknown',
        'mode': 'Negative',
        'type': 'Metabolomics (HILIC)',
    },
    'Proteomics_BSA': {
        'path': 'precursor/public/proteomics/BSA1.mzML',
        'platform': 'Unknown',
        'mode': 'Unknown',
        'type': 'Proteomics (BSA)',
    },
}


def run_pipeline_for_file(file_info: dict, name: str, base_dir: Path) -> Path:
    """Run pipeline on a single file and return results directory."""
    file_path = base_dir / file_info['path']

    if not file_path.exists():
        print(f"  File not found: {file_path}")
        return None

    print(f"\n{'='*60}")
    print(f"Processing: {name}")
    print(f"  Platform: {file_info['platform']}")
    print(f"  Mode: {file_info['mode']}")
    print(f"  Type: {file_info['type']}")
    print(f"{'='*60}")

    # Create output directory
    output_dir = base_dir / 'pipeline_results' / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    try:
        # Run pipeline
        runner = PipelineRunner(output_base_dir=str(output_dir.parent))
        runner.results_dir = output_dir
        runner.results_dir.mkdir(parents=True, exist_ok=True)
        (runner.results_dir / "stages").mkdir(exist_ok=True)
        (runner.results_dir / "figures").mkdir(exist_ok=True)

        # Determine ionization method from mode
        ionization = 'negative_esi' if 'Neg' in file_info['mode'] or 'neg' in name else 'positive_esi'

        # Run the pipeline
        results = runner.run(
            str(file_path),
            ionization_method=ionization,
            ms_platform=file_info['platform'].lower().replace(' ', '_') if file_info['platform'] != 'Unknown' else 'generic'
        )

        print(f"  Pipeline completed: {results.get('summary', {}).get('status', 'unknown')}")
        return output_dir

    except Exception as e:
        print(f"  Pipeline failed: {str(e)}")
        return None


def generate_ion_journey_for_results(results_dir: Path, name: str, file_info: dict, output_dir: Path) -> bool:
    """Generate ion journey visualization for pipeline results."""
    try:
        # Load results
        stages = load_pipeline_results(results_dir)

        if not stages:
            print(f"  No stage results found in {results_dir}")
            return False

        # Extract sample ion
        ion = extract_sample_ion(stages)

        # Create output with descriptive name
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, Rectangle
        from matplotlib.colors import LinearSegmentedColormap
        import numpy as np

        # Import drawing functions
        from union.src.figures.ion_journey import (
            draw_stage_1_injection, draw_stage_2_chromatography,
            draw_stage_3_ionization, draw_stage_4_ms1,
            draw_stage_5_sentropy, draw_stage_6_partition,
            draw_stage_7_thermodynamics, draw_stage_8_droplet,
            COLORS
        )

        # Create figure with 2 rows, 4 columns
        fig = plt.figure(figsize=(12, 7))

        # Add header with file info
        fig.suptitle(f'Ion Journey: {name}', fontsize=12, fontweight='bold', y=0.98)
        subtitle = f"{file_info['platform']} | {file_info['mode']} | {file_info['type']}"
        fig.text(0.5, 0.94, subtitle, ha='center', fontsize=10, style='italic', color='#457B9D')
        ion_info = f"m/z = {ion.mz:.2f}, RT = {ion.rt:.1f} min, State = {ion.categorical_state}"
        fig.text(0.5, 0.91, ion_info, ha='center', fontsize=9)

        # Top row: physical stages
        ax1 = fig.add_subplot(2, 4, 1)
        draw_stage_1_injection(ax1, ion)

        ax2 = fig.add_subplot(2, 4, 2)
        draw_stage_2_chromatography(ax2, ion)

        ax3 = fig.add_subplot(2, 4, 3)
        draw_stage_3_ionization(ax3, ion)

        ax4 = fig.add_subplot(2, 4, 4)
        draw_stage_4_ms1(ax4, ion)

        # Bottom row: computational/encoding stages
        ax5 = fig.add_subplot(2, 4, 5)
        draw_stage_5_sentropy(ax5, ion)

        ax6 = fig.add_subplot(2, 4, 6)
        draw_stage_6_partition(ax6, ion)

        ax7 = fig.add_subplot(2, 4, 7)
        draw_stage_7_thermodynamics(ax7, ion)

        ax8 = fig.add_subplot(2, 4, 8)
        draw_stage_8_droplet(ax8, ion)

        # Add stage numbers
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

        print(f"  Saved: ion_journey_{name}.png/pdf")
        return True

    except Exception as e:
        print(f"  Ion journey generation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def create_summary_panel(all_results: dict, output_dir: Path) -> None:
    """Create a summary panel showing all ion journeys side by side."""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap

    n_files = len(all_results)
    if n_files == 0:
        return

    # Create summary figure
    fig, axes = plt.subplots(n_files, 8, figsize=(16, 3 * n_files))

    if n_files == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle('Multi-Platform Ion Journey Comparison', fontsize=14, fontweight='bold', y=0.98)

    # Import drawing functions
    from union.src.figures.ion_journey import (
        draw_stage_1_injection, draw_stage_2_chromatography,
        draw_stage_3_ionization, draw_stage_4_ms1,
        draw_stage_5_sentropy, draw_stage_6_partition,
        draw_stage_7_thermodynamics, draw_stage_8_droplet,
        load_pipeline_results, extract_sample_ion
    )

    draw_funcs = [
        draw_stage_1_injection, draw_stage_2_chromatography,
        draw_stage_3_ionization, draw_stage_4_ms1,
        draw_stage_5_sentropy, draw_stage_6_partition,
        draw_stage_7_thermodynamics, draw_stage_8_droplet
    ]

    stage_names = ['Sample', 'Chrom', 'Ion', 'MS1', 'S-Ent', 'Part', 'Thermo', 'Drop']

    for row, (name, info) in enumerate(all_results.items()):
        results_dir = info.get('results_dir')
        if results_dir is None:
            continue

        try:
            stages = load_pipeline_results(results_dir)
            ion = extract_sample_ion(stages)

            # Row label
            axes[row, 0].text(-0.3, 0.5, name.replace('_', '\n'), transform=axes[row, 0].transAxes,
                            fontsize=8, fontweight='bold', va='center', ha='right')

            for col, (draw_func, stage_name) in enumerate(zip(draw_funcs, stage_names)):
                ax = axes[row, col]
                try:
                    draw_func(ax, ion)
                except:
                    ax.text(0.5, 0.5, 'N/A', ha='center', va='center')
                    ax.axis('off')

                if row == 0:
                    ax.set_title(stage_name, fontsize=8)

        except Exception as e:
            print(f"  Could not draw row for {name}: {e}")

    plt.tight_layout(rect=[0.05, 0, 1, 0.95])
    fig.savefig(output_dir / 'ion_journey_comparison.png', dpi=300)
    fig.savefig(output_dir / 'ion_journey_comparison.pdf')
    plt.close(fig)
    print(f"\nSaved: ion_journey_comparison.png/pdf")


def main():
    """Run batch ion journey generation."""
    base_dir = Path(__file__).parent.parent.parent.parent
    output_dir = Path(__file__).parent / 'output'
    output_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("BATCH ION JOURNEY GENERATOR")
    print("=" * 70)
    print(f"Base directory: {base_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Files to process: {len(PUBLIC_FILES)}")

    all_results = {}

    for name, file_info in PUBLIC_FILES.items():
        # Check if we already have recent results
        results_pattern = f"{name}_*"
        existing = list((base_dir / 'pipeline_results').glob(results_pattern))

        if existing:
            # Use most recent existing results
            results_dir = sorted(existing)[-1]
            print(f"\nUsing existing results for {name}: {results_dir.name}")
        else:
            # Run pipeline
            results_dir = run_pipeline_for_file(file_info, name, base_dir)

        if results_dir and results_dir.exists():
            all_results[name] = {
                'file_info': file_info,
                'results_dir': results_dir
            }

            # Generate ion journey
            print(f"\nGenerating ion journey for {name}...")
            generate_ion_journey_for_results(results_dir, name, file_info, output_dir)

    # Create comparison panel
    if len(all_results) > 1:
        print("\nGenerating comparison panel...")
        create_summary_panel(all_results, output_dir)

    print("\n" + "=" * 70)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 70)
    print(f"Processed {len(all_results)} files")
    print(f"Output saved to: {output_dir}")


if __name__ == '__main__':
    main()
