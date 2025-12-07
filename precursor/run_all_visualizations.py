#!/usr/bin/env python3
"""
run_all_visualizations.py

Master script to run all visualization modules.
Generates all publication-quality figures from test results.

Author: Kundai Farai Sachikonye (with AI assistance)
Date: December 2025
"""

import sys
from pathlib import Path

# Add src to path
PRECURSOR_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PRECURSOR_ROOT / 'src'))


def run_module(module_name, main_func_name='main'):
    """Run a visualization module."""
    print(f"\n{'='*60}")
    print(f"Running: {module_name}")
    print('='*60)
    try:
        # Import and run
        module = __import__(module_name, fromlist=[main_func_name])
        main_func = getattr(module, main_func_name)
        main_func()
        print(f"✓ {module_name} completed successfully")
        return True
    except Exception as e:
        print(f"✗ {module_name} failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all visualization modules."""
    print("="*60)
    print("LAVOISIER VISUALIZATION SUITE")
    print("="*60)
    print(f"Results directory: {PRECURSOR_ROOT / 'results' / 'tests'}")
    print(f"Output directory: {PRECURSOR_ROOT / 'results' / 'visualizations'}")

    # Create output directory
    output_dir = PRECURSOR_ROOT / 'results' / 'visualizations'
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    # 1. Entropy Space Visualizations (Figure 1, 4, 5)
    results.append(('dictionary.entropy_space', run_module('dictionary.entropy_space')))

    # 2. Trajectory Visualizations (Figure 1, 2)
    results.append(('dictionary.trajectories', run_module('dictionary.trajectories')))

    # 3. Zero-Shot Visualizations
    results.append(('dictionary.zero_shot_visualisation', run_module('dictionary.zero_shot_visualisation')))

    # 4. MMD Validation Visualizations
    results.append(('mmdsystem.molecular_maxwell_demon', run_module('mmdsystem.molecular_maxwell_demon')))

    # 5. Molecular Language Atlas
    results.append(('molecular_language.molecular_language_atlas', run_module('molecular_language.molecular_language_atlas')))

    # 6. Fragment Graph Visualizations
    results.append(('sequence.graph_completion', run_module('sequence.graph_completion')))

    # 7. Sequence Trajectory Visualizations
    results.append(('sequence.sequence_visualisation', run_module('sequence.sequence_visualisation')))

    # Summary
    print("\n" + "="*60)
    print("VISUALIZATION SUMMARY")
    print("="*60)

    success_count = sum(1 for _, success in results if success)
    total_count = len(results)

    for module_name, success in results:
        status = "✓" if success else "✗"
        print(f"  {status} {module_name}")

    print(f"\nCompleted: {success_count}/{total_count} modules")
    print(f"Output saved to: {output_dir}")
    print("="*60)

    return success_count == total_count


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
