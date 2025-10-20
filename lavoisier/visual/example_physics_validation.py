"""
Example: Physics Validation for Ion-to-Droplet Conversion
=========================================================

Demonstrates physics-based validation inspired by high-speed movement detection
to ensure only physically plausible ion-to-droplet transformations are accepted.

Key Validations:
- Ion flight time consistency (TOF principles)
- Energy conservation
- Thermodynamic parameter bounds (Weber, Reynolds numbers)
- Signal detection plausibility

Inspired by: github.com/fullscreen-triangle/vibrio

Author: Kundai Chinyamakobvu
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List

# Import converters
from IonToDropletConverter import IonToDropletConverter
from PhysicsValidator import PhysicsValidator, PhysicsConstraints


def example_physics_validation_comparison():
    """Compare conversion with and without physics validation."""
    print("="*70)
    print("PHYSICS VALIDATION COMPARISON")
    print("="*70)

    # Generate example spectrum with some problematic ions
    np.random.seed(42)

    # Mix of good and bad ions
    mzs = np.array([
        100.0, 150.0, 250.0,  # Low m/z (might be noise)
        500.0, 524.3, 600.0, 750.0,  # Good ions
        9500.0,  # Very high m/z (unusual)
        1000.0, 1100.0, 1200.0  # More good ions
    ])

    intensities = np.array([
        1e2,  # Very low (below detection limit)
        1e11,  # Saturated detector
        1e5,   # Good
        1e6, 5e6, 3e6, 2e6,  # Good
        1e4,   # Good but unusual m/z
        1e6, 8e5, 7e5  # Good
    ])

    print(f"\nTest spectrum:")
    print(f"  Number of ions: {len(mzs)}")
    print(f"  m/z range: {mzs.min():.1f} - {mzs.max():.1f}")
    print(f"  Intensity range: {intensities.min():.2e} - {intensities.max():.2e}")

    # Convert WITHOUT validation
    print("\n" + "-"*50)
    print("Converting WITHOUT physics validation...")
    print("-"*50)

    converter_no_validation = IonToDropletConverter(
        resolution=(512, 512),
        enable_physics_validation=False
    )

    image_no_val, droplets_no_val = converter_no_validation.convert_spectrum_to_image(
        mzs=mzs,
        intensities=intensities
    )

    print(f"  Ions converted: {len(droplets_no_val)}/{len(mzs)}")
    summary_no_val = converter_no_validation.get_droplet_summary(droplets_no_val)
    print(f"  Image pixels non-zero: {np.count_nonzero(image_no_val)}")

    # Convert WITH validation
    print("\n" + "-"*50)
    print("Converting WITH physics validation...")
    print("-"*50)

    converter_with_validation = IonToDropletConverter(
        resolution=(512, 512),
        enable_physics_validation=True,
        validation_threshold=0.5  # Moderate threshold
    )

    image_with_val, droplets_with_val = converter_with_validation.convert_spectrum_to_image(
        mzs=mzs,
        intensities=intensities
    )

    print(f"  Ions converted: {len(droplets_with_val)}/{len(mzs)}")
    summary_with_val = converter_with_validation.get_droplet_summary(droplets_with_val)
    print(f"  Image pixels non-zero: {np.count_nonzero(image_with_val)}")

    # Show validation report
    print("\n" + converter_with_validation.get_validation_report())

    # Show which ions were filtered
    print("\nFiltered ions:")
    converted_mzs_with_val = set(d.mz for d in droplets_with_val)
    for i, mz in enumerate(mzs):
        if mz not in converted_mzs_with_val:
            print(f"  m/z {mz:.1f} (intensity {intensities[i]:.2e}) - FILTERED")

    # Compare quality scores
    if droplets_with_val:
        print("\nQuality scores of accepted ions:")
        for d in sorted(droplets_with_val, key=lambda x: x.physics_quality):
            warnings_str = f" [{len(d.validation_warnings)} warnings]" if d.validation_warnings else ""
            print(f"  m/z {d.mz:.1f}: quality={d.physics_quality:.3f}{warnings_str}")

    # Visualize comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Images
    axes[0, 0].imshow(image_no_val, cmap='viridis', aspect='auto')
    axes[0, 0].set_title(f'WITHOUT Validation ({len(droplets_no_val)} ions)')
    axes[0, 0].set_xlabel('m/z direction')
    axes[0, 0].set_ylabel('S_time direction')

    axes[0, 1].imshow(image_with_val, cmap='viridis', aspect='auto')
    axes[0, 1].set_title(f'WITH Validation ({len(droplets_with_val)} ions)')
    axes[0, 1].set_xlabel('m/z direction')
    axes[0, 1].set_ylabel('S_time direction')

    # Stick plots
    axes[1, 0].vlines(mzs, 0, intensities, color='blue', alpha=0.7, label='All ions')
    filtered_mzs = [mz for mz in mzs if mz not in converted_mzs_with_val]
    filtered_intensities = [intensities[i] for i, mz in enumerate(mzs) if mz not in converted_mzs_with_val]
    if filtered_mzs:
        axes[1, 0].vlines(filtered_mzs, 0, filtered_intensities, color='red', alpha=0.7,
                         linewidth=3, label='Filtered by validation')
    axes[1, 0].set_xlabel('m/z')
    axes[1, 0].set_ylabel('Intensity')
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_title('Original Spectrum (red = filtered)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Quality scores
    if droplets_with_val:
        quality_mzs = [d.mz for d in droplets_with_val]
        quality_scores = [d.physics_quality for d in droplets_with_val]
        colors = ['green' if q >= 0.7 else 'orange' if q >= 0.5 else 'red' for q in quality_scores]

        axes[1, 1].bar(range(len(quality_scores)), quality_scores, color=colors, alpha=0.7, edgecolor='black')
        axes[1, 1].axhline(y=converter_with_validation.validation_threshold, color='red',
                          linestyle='--', label=f'Threshold ({converter_with_validation.validation_threshold})')
        axes[1, 1].set_xlabel('Ion Index')
        axes[1, 1].set_ylabel('Physics Quality Score')
        axes[1, 1].set_title('Physics Validation Quality Scores')
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, axis='y')

        # Add m/z labels
        for i, (mz, score) in enumerate(zip(quality_mzs, quality_scores)):
            axes[1, 1].text(i, score + 0.02, f'{mz:.0f}', ha='center', va='bottom',
                           fontsize=8, rotation=45)

    plt.tight_layout()
    plt.savefig('physics_validation_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved: physics_validation_comparison.png")

    return converter_with_validation, droplets_with_val


def example_detailed_validation():
    """Show detailed physics validation metrics for a single ion."""
    print("\n" + "="*70)
    print("DETAILED PHYSICS VALIDATION")
    print("="*70)

    # Example ion
    mz = 524.372
    intensity = 1.5e6
    rt = 12.5

    print(f"\nValidating ion:")
    print(f"  m/z: {mz:.3f}")
    print(f"  Intensity: {intensity:.2e}")
    print(f"  RT: {rt:.2f} min")

    # Create validator
    validator = PhysicsValidator()

    # First convert to get droplet parameters
    converter = IonToDropletConverter(enable_physics_validation=False)
    droplet = converter.convert_ion_to_droplet(mz=mz, intensity=intensity, rt=rt)

    print(f"\nDroplet parameters:")
    print(f"  Velocity: {droplet.droplet_params.velocity:.2f} m/s")
    print(f"  Radius: {droplet.droplet_params.radius:.2f} mm")
    print(f"  Surface tension: {droplet.droplet_params.surface_tension:.4f} N/m")
    print(f"  Temperature: {droplet.droplet_params.temperature:.2f} K")
    print(f"  Phase coherence: {droplet.droplet_params.phase_coherence:.4f}")

    # Comprehensive validation
    print("\n" + "-"*50)
    print("Running comprehensive physics validation...")
    print("-"*50)

    validation_results = validator.comprehensive_validation(
        mz=mz,
        intensity=intensity,
        velocity=droplet.droplet_params.velocity,
        radius=droplet.droplet_params.radius,
        surface_tension=droplet.droplet_params.surface_tension,
        temperature=droplet.droplet_params.temperature,
        phase_coherence=droplet.droplet_params.phase_coherence,
        rt=rt,
        charge=1
    )

    # Show results for each category
    for category, result in validation_results.items():
        print(f"\n{category.upper()} Validation:")
        print(f"  Valid: {result.is_valid}")
        print(f"  Quality: {result.quality_score:.3f}")

        if result.metrics:
            print(f"  Key Metrics:")
            for key, value in list(result.metrics.items())[:5]:  # Show first 5
                if isinstance(value, float):
                    print(f"    {key}: {value:.4f}")
                else:
                    print(f"    {key}: {value}")

        if result.warnings:
            print(f"  Warnings:")
            for warning in result.warnings:
                print(f"    - {warning}")

        if result.violations:
            print(f"  Violations:")
            for violation in result.violations:
                print(f"    - {violation}")

    # Overall quality
    overall_quality, is_valid = validator.get_overall_quality(validation_results)
    print("\n" + "="*50)
    print(f"OVERALL QUALITY: {overall_quality:.3f}")
    print(f"IS VALID: {is_valid}")
    print("="*50)

    return validation_results


def example_problematic_ions():
    """Test validation with deliberately problematic ions."""
    print("\n" + "="*70)
    print("TESTING PROBLEMATIC IONS")
    print("="*70)

    validator = PhysicsValidator()
    converter = IonToDropletConverter(enable_physics_validation=False)

    # Define problematic test cases
    test_cases = [
        {"name": "Very low m/z", "mz": 5.0, "intensity": 1e5, "should_fail": True},
        {"name": "Very high m/z", "mz": 15000.0, "intensity": 1e5, "should_fail": True},
        {"name": "Below detection limit", "mz": 500.0, "intensity": 10.0, "should_fail": True},
        {"name": "Saturated detector", "mz": 500.0, "intensity": 1e12, "should_fail": False},  # Warning only
        {"name": "Normal ion", "mz": 524.3, "intensity": 1e6, "should_fail": False},
        {"name": "Edge case low intensity", "mz": 500.0, "intensity": 150.0, "should_fail": False},
    ]

    results = []

    for test in test_cases:
        print(f"\n{'-'*50}")
        print(f"Test: {test['name']}")
        print(f"  m/z: {test['mz']:.1f}, Intensity: {test['intensity']:.2e}")

        # Basic ion validation
        ion_result = validator.validate_ion_properties(
            mz=test['mz'],
            intensity=test['intensity']
        )

        print(f"  Valid: {ion_result.is_valid}")
        print(f"  Quality: {ion_result.quality_score:.3f}")
        print(f"  Violations: {len(ion_result.violations)}")
        print(f"  Warnings: {len(ion_result.warnings)}")

        if ion_result.violations:
            print(f"  First violation: {ion_result.violations[0]}")

        # Check if result matches expectation
        if test['should_fail']:
            status = "✓ CORRECTLY REJECTED" if not ion_result.is_valid else "✗ FALSE PASS"
        else:
            status = "✓ CORRECTLY ACCEPTED" if ion_result.is_valid else "✗ FALSE REJECT"

        print(f"  Status: {status}")

        results.append({
            'test': test['name'],
            'is_valid': ion_result.is_valid,
            'quality': ion_result.quality_score,
            'expected': not test['should_fail'],
            'correct': ion_result.is_valid == (not test['should_fail'])
        })

    # Summary
    print("\n" + "="*50)
    print("VALIDATION SUMMARY")
    print("="*50)

    correct_count = sum(1 for r in results if r['correct'])
    total_count = len(results)

    print(f"\nCorrect validations: {correct_count}/{total_count} ({100*correct_count/total_count:.1f}%)")

    print("\nResults table:")
    print(f"{'Test':<30} {'Valid':<8} {'Quality':<10} {'Expected':<10} {'Result':<10}")
    print("-"*70)
    for r in results:
        result_str = "✓ PASS" if r['correct'] else "✗ FAIL"
        print(f"{r['test']:<30} {str(r['is_valid']):<8} {r['quality']:<10.3f} {str(r['expected']):<10} {result_str:<10}")

    return results


def main():
    """Run all physics validation examples."""
    print("\n" + "="*70)
    print("PHYSICS VALIDATION FOR ION-TO-DROPLET CONVERSION")
    print("="*70)
    print("\nInspired by high-speed movement detection")
    print("Ensures only physically plausible transformations are accepted")
    print("="*70)

    # Example 1: Comparison with/without validation
    converter, droplets = example_physics_validation_comparison()

    # Example 2: Detailed validation metrics
    validation_results = example_detailed_validation()

    # Example 3: Problematic ions
    test_results = example_problematic_ions()

    print("\n" + "="*70)
    print("ALL PHYSICS VALIDATION EXAMPLES COMPLETED")
    print("="*70)

    print("\nKey Validations Implemented:")
    print("  ✓ Ion flight time consistency (TOF principles)")
    print("  ✓ Energy conservation in droplet formation")
    print("  ✓ Thermodynamic parameter bounds (Weber, Reynolds, Capillary numbers)")
    print("  ✓ Signal detection plausibility (intensity limits)")
    print("  ✓ Mass-to-charge ratio validation")
    print("  ✓ Phase coherence consistency checks")

    print("\nBenefits:")
    print("  • Filters spurious/artifact peaks")
    print("  • Ensures physical plausibility")
    print("  • Provides quality scores for ranking")
    print("  • Detects instrument saturation/limits")
    print("  • Validates energy conservation")

    print("\nInspired by: github.com/fullscreen-triangle/vibrio")
    print("  (High-speed movement detection with trajectory validation)")
    print("="*70)


if __name__ == "__main__":
    main()
