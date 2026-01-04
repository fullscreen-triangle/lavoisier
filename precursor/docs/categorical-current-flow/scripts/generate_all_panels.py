#!/usr/bin/env python3
"""
Master script to generate all visualization panels for the Categorical Current Flow paper.
"""

import subprocess
import sys
from pathlib import Path

def run_script(script_name):
    """Run a panel generation script."""
    script_path = Path(__file__).parent / script_name
    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print('='*60)
    
    result = subprocess.run([sys.executable, str(script_path)], capture_output=False)
    
    if result.returncode != 0:
        print(f"Warning: {script_name} returned non-zero exit code")
    
    return result.returncode == 0

def main():
    scripts = [
        "generate_newton_cradle_panel.py",
        "generate_dimensional_reduction_panel.py",
        "generate_ohm_kirchhoff_panel.py",
        "generate_maxwell_panel.py",
        "generate_scattering_aperture_panel.py",
        "generate_temperature_superconductivity_panel.py",
        "validate_current_propagation.py",
    ]
    
    print("="*60)
    print("CATEGORICAL CURRENT FLOW - PANEL GENERATION")
    print("="*60)
    
    success_count = 0
    for script in scripts:
        if run_script(script):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"Completed: {success_count}/{len(scripts)} scripts successful")
    print("="*60)

if __name__ == "__main__":
    main()

