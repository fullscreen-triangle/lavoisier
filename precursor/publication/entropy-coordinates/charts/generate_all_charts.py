#!/usr/bin/env python3
"""
Generate All Publication Charts
================================

Master script to generate all validation result visualizations for publication.

Generates 5 comprehensive figures:
1. Quality Control Validation (4 panels)
2. Database Search Performance (4 panels)
3. Spectrum Embedding Analysis (4 panels)
4. Feature Clustering Results (4 panels)
5. Comparative Analysis: Numerical vs Visual (4 panels)

Total: 20 publication-quality charts

Author: Lavoisier Team
Date: 2025-10-27
"""

import sys
from pathlib import Path

print("="*80)
print("LAVOISIER PUBLICATION CHARTS - COMPLETE GENERATION")
print("="*80)
print("\nGenerating all validation result visualizations...")
print("Total: 5 figures × 4 panels = 20 charts\n")

# Import and run each visualization module
scripts = [
    ('quality_control', 'Quality Control Validation'),
    ('database_search', 'Database Search Performance'),
    ('spectrum_embedding', 'Spectrum Embedding Analysis'),
    ('feature_clustering_numerical', 'Feature Clustering Results'),
    ('comparative_analysis', 'Comparative Analysis: Numerical vs Visual'),
]

success_count = 0
failed = []

for i, (script_name, description) in enumerate(scripts, 1):
    print(f"[{i}/5] {description}")
    print("-" * 80)

    try:
        # Import the module
        module = __import__(script_name)

        # Run the main function
        if script_name == 'quality_control':
            module.create_quality_control_figure()
        elif script_name == 'database_search':
            module.create_database_search_figure()
        elif script_name == 'spectrum_embedding':
            module.create_spectrum_embedding_figure()
        elif script_name == 'feature_clustering_numerical':
            module.create_feature_clustering_figure()
        elif script_name == 'comparative_analysis':
            module.create_comparative_analysis_figure()

        success_count += 1
        print(f"✓ Success\n")

    except Exception as e:
        print(f"✗ Failed: {e}\n")
        failed.append((script_name, str(e)))

print("="*80)
print("GENERATION COMPLETE")
print("="*80)
print(f"\nSuccessfully generated: {success_count}/5 figures")

if failed:
    print(f"\nFailed figures:")
    for script, error in failed:
        print(f"  ✗ {script}: {error}")
else:
    print("\n✓ All figures generated successfully!")
    print("\nOutput files:")
    output_dir = Path(__file__).parent
    print(f"  - {output_dir / 'quality_control_validation.png'}")
    print(f"  - {output_dir / 'database_search_validation.png'}")
    print(f"  - {output_dir / 'spectrum_embedding_validation.png'}")
    print(f"  - {output_dir / 'feature_clustering_validation.png'}")
    print(f"  - {output_dir / 'comparative_analysis.png'}")
    print("\n📊 Total: 20 publication-quality charts ready for manuscript!")

sys.exit(0 if not failed else 1)
