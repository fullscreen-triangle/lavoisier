"""
Master script to generate all visualization panels for the
Derivation of Physics from First Principles paper.

Run this script from the scripts directory:
    python generate_all_panels.py
"""

import os
import sys

# Ensure figures directory exists
os.makedirs('../figures', exist_ok=True)

print("=" * 60)
print("Generating Visualization Panels for Derivation of Physics")
print("=" * 60)

# Import and run each panel generator
print("\n[1/4] Generating Oscillatory Dynamics panels (Sections 2-3)...")
try:
    from generate_oscillatory_panels import generate_bounded_phase_space_panel
    generate_bounded_phase_space_panel()
except Exception as e:
    print(f"Error: {e}")

print("\n[2/4] Generating Categorical/Partition panels (Sections 4-5)...")
try:
    from generate_categorical_panels import generate_categorical_structure_panel
    generate_categorical_structure_panel()
except Exception as e:
    print(f"Error: {e}")

print("\n[3/4] Generating Spatial/Matter panels (Sections 6-7)...")
try:
    from generate_spatial_matter_panels import generate_spatial_matter_panel
    generate_spatial_matter_panel()
except Exception as e:
    print(f"Error: {e}")

print("\n[4/4] Generating Forces/Cosmology panels (Sections 8-9)...")
try:
    from generate_forces_cosmology_panels import generate_forces_cosmology_panel
    generate_forces_cosmology_panel()
except Exception as e:
    print(f"Error: {e}")

print("\n[5/5] Generating Atomic Structure panels (Section 10)...")
try:
    from generate_atomic_panels import generate_atomic_structure_panel
    generate_atomic_structure_panel()
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 60)
print("Panel generation complete!")
print("Output files in: ../figures/")
print("=" * 60)

# List generated files
print("\nGenerated files:")
for f in sorted(os.listdir('../figures')):
    if f.endswith('.png') or f.endswith('.pdf'):
        size = os.path.getsize(f'../figures/{f}') / 1024
        print(f"  - {f} ({size:.1f} KB)")

