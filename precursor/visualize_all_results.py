"""
Master Visualization Script - REAL DATA

Runs ALL visualization scripts to generate comprehensive analysis
of REAL experimental data from the fragmentation pipeline.
"""
import subprocess
import sys
from pathlib import Path

def run_script(script_path, script_name):
    """Run a visualization script and report results"""
    print(f"\n{'='*80}")
    print(f"Running: {script_name}")
    print('='*80)

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=script_path.parent.parent,  # Run from precursor root
            capture_output=False,
            text=True
        )

        if result.returncode == 0:
            print(f"✓ {script_name} completed successfully")
            return True
        else:
            print(f"✗ {script_name} failed with code {result.returncode}")
            return False

    except Exception as e:
        print(f"✗ {script_name} error: {e}")
        return False


def main():
    """Run all visualization scripts"""
    print("="*80)
    print("MASTER VISUALIZATION - ALL SCRIPTS - REAL DATA")
    print("="*80)
    print("\nThis will run ALL 7 visualization scripts using REAL experimental data")
    print("from results/fragmentation_comparison/\n")

    # Get script directory
    script_dir = Path(__file__).parent
    virtual_dir = script_dir / "src" / "virtual"

    # List of scripts to run (in order)
    scripts = [
        ("entropy_transformation.py", "S-Entropy 3D Visualization"),
        ("fragmentation_landscape.py", "Fragmentation Landscape"),
        ("phase_lock_networks.py", "Phase-Lock Networks"),
        ("phase_diagrams.py", "Phase Diagrams (Polar Histograms)"),
        ("fragment_trajectories.py", "Fragment Trajectories"),
        ("detector_visualisation.py", "Virtual Detector Comparison"),
        ("virtual_spectra_comparison.py", "Virtual vs Original qTOF"),
        ("vector_transformation.py", "Vector Transformation Analysis"),
        ("validation_charts.py", "Validation Charts"),
        ("fragmentation_spectra.py", "Individual Spectra"),
        ("computer_vision_validation.py", "CV Droplet Analysis"),
        ("virtual_stages.py", "Pipeline Stage Visualization"),
        ("molecular_maxwell_demon.py", "Molecular Maxwell Demon Framework"),
        ("experimental_validation.py", "Proteomics Experimental Validation"),
    ]

    results = {}

    for script_file, script_name in scripts:
        script_path = virtual_dir / script_file

        if not script_path.exists():
            print(f"✗ Script not found: {script_path}")
            results[script_name] = False
            continue

        success = run_script(script_path, script_name)
        results[script_name] = success

    # Summary
    print("\n" + "="*80)
    print("VISUALIZATION SUMMARY")
    print("="*80)

    for script_name, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{status}: {script_name}")

    total = len(results)
    successful = sum(results.values())

    print(f"\nTotal: {successful}/{total} scripts completed successfully")

    if successful == total:
        print("\n✓ ALL VISUALIZATIONS COMPLETE")
        print(f"  Check visualizations/ directory for outputs")
    else:
        print(f"\n⚠ {total - successful} script(s) failed")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
