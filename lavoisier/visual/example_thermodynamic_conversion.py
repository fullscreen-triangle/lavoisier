"""
Example: Ion-to-Droplet Thermodynamic Conversion
================================================

Demonstrates the complete thermodynamic visual modality pipeline:
1. Convert ions to droplets with S-Entropy encoding
2. Generate thermodynamic wave patterns
3. Extract phase-lock signatures
4. Compare with database using dual-modality features

This is the visual component of the dual-graph intersection system.

Author: Kundai Chinyamakobvu
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2

# Import the new converters
from IonToDropletConverter import (
    IonToDropletConverter,
    SEntropyCalculator,
    DropletMapper,
    ThermodynamicWaveGenerator
)
from MSImageDatabase_Enhanced import MSImageDatabase


def example_single_ion_conversion():
    """Example: Convert a single ion to droplet."""
    print("="*70)
    print("EXAMPLE 1: Single Ion to Droplet Conversion")
    print("="*70)

    # Initialize calculator
    s_calculator = SEntropyCalculator()
    d_mapper = DropletMapper()

    # Example ion
    mz = 524.372
    intensity = 1.5e6
    rt = 12.5  # minutes

    print(f"\nInput Ion:")
    print(f"  m/z: {mz:.3f}")
    print(f"  Intensity: {intensity:.2e}")
    print(f"  RT: {rt:.2f} min")

    # Calculate S-Entropy coordinates
    s_coords = s_calculator.calculate_s_entropy(
        mz=mz,
        intensity=intensity,
        rt=rt
    )

    print(f"\nS-Entropy Coordinates:")
    print(f"  S_knowledge: {s_coords.s_knowledge:.4f}")
    print(f"  S_time: {s_coords.s_time:.4f}")
    print(f"  S_entropy: {s_coords.s_entropy:.4f}")

    # Map to droplet parameters
    droplet_params = d_mapper.map_to_droplet(s_coords, intensity)

    print(f"\nDroplet Parameters:")
    print(f"  Velocity: {droplet_params.velocity:.2f} m/s")
    print(f"  Radius: {droplet_params.radius:.2f} mm")
    print(f"  Surface Tension: {droplet_params.surface_tension:.4f} N/m")
    print(f"  Impact Angle: {droplet_params.impact_angle:.1f}°")
    print(f"  Temperature: {droplet_params.temperature:.2f} K")
    print(f"  Phase Coherence: {droplet_params.phase_coherence:.4f}")

    print("\n✓ Single ion successfully converted to thermodynamic droplet")


def example_spectrum_conversion():
    """Example: Convert a complete spectrum to thermodynamic image."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Spectrum to Thermodynamic Image")
    print("="*70)

    # Generate example spectrum (simulating a peptide fragment spectrum)
    np.random.seed(42)
    n_peaks = 50
    mzs = np.sort(np.random.uniform(100, 1000, n_peaks))
    intensities = np.random.lognormal(10, 2, n_peaks)
    rt = 15.3

    print(f"\nInput Spectrum:")
    print(f"  Number of peaks: {n_peaks}")
    print(f"  m/z range: {mzs.min():.1f} - {mzs.max():.1f}")
    print(f"  Intensity range: {intensities.min():.2e} - {intensities.max():.2e}")
    print(f"  RT: {rt:.2f} min")

    # Initialize converter
    converter = IonToDropletConverter(resolution=(512, 512))

    # Convert spectrum
    print("\nConverting spectrum to thermodynamic image...")
    image, ion_droplets = converter.convert_spectrum_to_image(
        mzs=mzs,
        intensities=intensities,
        rt=rt
    )

    print(f"\nConversion Results:")
    print(f"  Image shape: {image.shape}")
    print(f"  Ion droplets created: {len(ion_droplets)}")
    print(f"  Image intensity range: {image.min()} - {image.max()}")

    # Get droplet summary
    summary = converter.get_droplet_summary(ion_droplets)
    print(f"\nDroplet Summary:")
    print(f"  S_knowledge (mean): {summary['s_entropy_coords']['s_knowledge_mean']:.4f}")
    print(f"  S_time (mean): {summary['s_entropy_coords']['s_time_mean']:.4f}")
    print(f"  S_entropy (mean): {summary['s_entropy_coords']['s_entropy_mean']:.4f}")
    print(f"  Velocity (mean): {summary['droplet_params']['velocity_mean']:.2f} m/s")
    print(f"  Phase coherence (mean): {summary['droplet_params']['phase_coherence_mean']:.4f}")

    # Extract phase-lock features
    features = converter.extract_phase_lock_features(image, ion_droplets)
    print(f"\nPhase-Lock Features:")
    print(f"  Feature vector length: {len(features)}")
    print(f"  Feature range: {features.min():.2f} - {features.max():.2f}")

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Original spectrum (stick plot)
    axes[0].vlines(mzs, 0, intensities, color='blue', alpha=0.7)
    axes[0].set_xlabel('m/z')
    axes[0].set_ylabel('Intensity')
    axes[0].set_title('Original Spectrum')
    axes[0].grid(True, alpha=0.3)

    # Thermodynamic droplet image
    axes[1].imshow(image, cmap='viridis', aspect='auto')
    axes[1].set_xlabel('m/z direction')
    axes[1].set_ylabel('S_time direction')
    axes[1].set_title('Thermodynamic Droplet Image')
    axes[1].colorbar()

    # S-Entropy coordinates in 3D
    ax3d = fig.add_subplot(1, 3, 3, projection='3d')
    s_k = [d.s_entropy_coords.s_knowledge for d in ion_droplets]
    s_t = [d.s_entropy_coords.s_time for d in ion_droplets]
    s_e = [d.s_entropy_coords.s_entropy for d in ion_droplets]
    scatter = ax3d.scatter(s_k, s_t, s_e, c=intensities, cmap='hot', s=50, alpha=0.7)
    ax3d.set_xlabel('S_knowledge')
    ax3d.set_ylabel('S_time')
    ax3d.set_zlabel('S_entropy')
    ax3d.set_title('S-Entropy Coordinate Space')
    plt.colorbar(scatter, ax=ax3d, label='Intensity')

    plt.tight_layout()
    plt.savefig('thermodynamic_conversion_example.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved: thermodynamic_conversion_example.png")

    return image, ion_droplets


def example_database_search():
    """Example: Search database using thermodynamic features."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Database Search with Thermodynamic Features")
    print("="*70)

    # Initialize database with thermodynamic mode
    print("\nInitializing thermodynamic database...")
    db = MSImageDatabase(resolution=(512, 512), use_thermodynamic=True)

    # Add example spectra to database
    np.random.seed(123)
    print("\nAdding spectra to database...")

    for i in range(5):
        # Generate example spectrum
        n_peaks = np.random.randint(30, 70)
        mzs = np.sort(np.random.uniform(100, 1000, n_peaks))
        intensities = np.random.lognormal(10, 2, n_peaks)
        rt = 10.0 + i * 2.0

        spectrum_id = db.add_spectrum(
            mzs=mzs,
            intensities=intensities,
            rt=rt,
            metadata={'sample': f'sample_{i}', 'experiment': 'demo'}
        )
        print(f"  Added spectrum {i+1}/5: {spectrum_id[:8]}... (RT: {rt:.1f} min)")

    # Query spectrum
    print("\nGenerating query spectrum...")
    query_mzs = np.sort(np.random.uniform(100, 1000, 50))
    query_intensities = np.random.lognormal(10, 2, 50)
    query_rt = 14.5

    # Search
    print("\nSearching database...")
    matches = db.search(
        query_mzs=query_mzs,
        query_intensities=query_intensities,
        query_rt=query_rt,
        k=3
    )

    print(f"\nTop {len(matches)} matches:")
    for i, match in enumerate(matches, 1):
        print(f"\nMatch {i}:")
        print(f"  Database ID: {match.database_id[:8]}...")
        print(f"  Overall Similarity: {match.similarity:.4f}")
        print(f"  Structural Similarity (SSIM): {match.structural_similarity:.4f}")
        print(f"  Phase-Lock Similarity: {match.phase_lock_similarity:.4f}")
        print(f"  Categorical Match: {match.categorical_state_match:.4f}")
        print(f"  S-Entropy Distance: {match.s_entropy_distance:.4f}")
        print(f"  Matched Features: {len(match.matched_features)}")

    print("\n✓ Database search completed successfully")

    # Save database
    print("\nSaving database...")
    db.save_database('thermodynamic_database')
    print("✓ Database saved: thermodynamic_database/")

    return db, matches


def example_phase_lock_comparison():
    """Example: Compare phase-lock signatures between spectra."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Phase-Lock Signature Comparison")
    print("="*70)

    # Create two similar spectra (e.g., same peptide, different charge states)
    np.random.seed(456)

    # Spectrum 1
    base_mzs = np.array([100, 200, 300, 400, 500, 600, 700, 800])
    spectrum1_mzs = base_mzs + np.random.normal(0, 1, len(base_mzs))
    spectrum1_intensities = np.array([1e5, 5e5, 8e5, 1e6, 6e5, 3e5, 1e5, 5e4])

    # Spectrum 2 (shifted, similar pattern)
    spectrum2_mzs = (base_mzs + 10) + np.random.normal(0, 1, len(base_mzs))
    spectrum2_intensities = spectrum1_intensities * (1 + np.random.normal(0, 0.1, len(base_mzs)))

    print("\nSpectrum 1:")
    print(f"  Peaks: {len(spectrum1_mzs)}")
    print(f"  m/z range: {spectrum1_mzs.min():.1f} - {spectrum1_mzs.max():.1f}")

    print("\nSpectrum 2:")
    print(f"  Peaks: {len(spectrum2_mzs)}")
    print(f"  m/z range: {spectrum2_mzs.min():.1f} - {spectrum2_mzs.max():.1f}")

    # Convert both to thermodynamic images
    converter = IonToDropletConverter(resolution=(512, 512))

    image1, droplets1 = converter.convert_spectrum_to_image(
        spectrum1_mzs, spectrum1_intensities
    )

    # Reset categorical state counter for second spectrum
    converter.categorical_state_counter = 0
    image2, droplets2 = converter.convert_spectrum_to_image(
        spectrum2_mzs, spectrum2_intensities
    )

    # Extract features
    features1 = converter.extract_phase_lock_features(image1, droplets1)
    features2 = converter.extract_phase_lock_features(image2, droplets2)

    # Compare features
    feature_similarity = 1.0 - np.linalg.norm(features1 - features2) / np.sqrt(len(features1))

    print(f"\nPhase-Lock Feature Comparison:")
    print(f"  Feature vector length: {len(features1)}")
    print(f"  Cosine similarity: {feature_similarity:.4f}")

    # Compare S-Entropy coordinates
    def avg_s_entropy(droplets):
        return {
            's_knowledge': np.mean([d.s_entropy_coords.s_knowledge for d in droplets]),
            's_time': np.mean([d.s_entropy_coords.s_time for d in droplets]),
            's_entropy': np.mean([d.s_entropy_coords.s_entropy for d in droplets])
        }

    s1 = avg_s_entropy(droplets1)
    s2 = avg_s_entropy(droplets2)

    print(f"\nAverage S-Entropy Coordinates:")
    print(f"  Spectrum 1: S_k={s1['s_knowledge']:.4f}, S_t={s1['s_time']:.4f}, S_e={s1['s_entropy']:.4f}")
    print(f"  Spectrum 2: S_k={s2['s_knowledge']:.4f}, S_t={s2['s_time']:.4f}, S_e={s2['s_entropy']:.4f}")

    s_entropy_distance = np.linalg.norm([
        s1['s_knowledge'] - s2['s_knowledge'],
        s1['s_time'] - s2['s_time'],
        s1['s_entropy'] - s2['s_entropy']
    ])
    print(f"  S-Entropy distance: {s_entropy_distance:.4f}")

    # Visualize comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Thermodynamic images
    axes[0, 0].imshow(image1, cmap='viridis')
    axes[0, 0].set_title('Spectrum 1: Thermodynamic Image')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(image2, cmap='viridis')
    axes[0, 1].set_title('Spectrum 2: Thermodynamic Image')
    axes[0, 1].axis('off')

    # Feature comparison
    axes[1, 0].plot(features1, label='Spectrum 1', alpha=0.7)
    axes[1, 0].plot(features2, label='Spectrum 2', alpha=0.7)
    axes[1, 0].set_xlabel('Feature Index')
    axes[1, 0].set_ylabel('Feature Value')
    axes[1, 0].set_title('Phase-Lock Feature Vectors')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # S-Entropy comparison
    categories = ['S_knowledge', 'S_time', 'S_entropy']
    values1 = [s1['s_knowledge'], s1['s_time'], s1['s_entropy']]
    values2 = [s2['s_knowledge'], s2['s_time'], s2['s_entropy']]

    x = np.arange(len(categories))
    width = 0.35
    axes[1, 1].bar(x - width/2, values1, width, label='Spectrum 1', alpha=0.7)
    axes[1, 1].bar(x + width/2, values2, width, label='Spectrum 2', alpha=0.7)
    axes[1, 1].set_xlabel('S-Entropy Dimension')
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].set_title('S-Entropy Coordinate Comparison')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(categories)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('phase_lock_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved: phase_lock_comparison.png")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("ION-TO-DROPLET THERMODYNAMIC CONVERSION EXAMPLES")
    print("="*70)
    print("\nDemonstrating the visual modality of the dual-graph system")
    print("for phase-lock-based molecular annotation.")
    print("="*70)

    # Example 1: Single ion
    example_single_ion_conversion()

    # Example 2: Spectrum conversion
    example_spectrum_conversion()

    # Example 3: Database search
    example_database_search()

    # Example 4: Phase-lock comparison
    example_phase_lock_comparison()

    print("\n" + "="*70)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
    print("="*70)
    print("\nKey Achievements:")
    print("  ✓ Ion-to-droplet thermodynamic conversion")
    print("  ✓ S-Entropy coordinate calculation")
    print("  ✓ Thermodynamic wave pattern generation")
    print("  ✓ Phase-lock signature extraction")
    print("  ✓ Dual-modality database search")
    print("\nThis visual modality now integrates with the numerical modality")
    print("to form the complete dual-graph intersection system for")
    print("categorical-state-based molecular annotation.")
    print("="*70)


if __name__ == "__main__":
    main()
