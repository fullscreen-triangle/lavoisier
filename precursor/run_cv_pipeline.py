"""
CV Pipeline - Ion to Droplet Conversion
=========================================

Reads existing spectra and applies computer vision module.
No metabolomics pipeline, JUST CV.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import cv2

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from core.IonToDropletConverter import IonToDropletConverter

def main():
    print("="*80)
    print("CV PIPELINE - ION TO DROPLET CONVERSION")
    print("="*80)

    # Input: existing spectra from metabolomics results
    spectra_dir = Path(__file__).parent / 'results' / 'metabolomics_analysis' / 'PL_Neg_Waters_qTOF' / 'stage_01_preprocessing' / 'spectra'

    if not spectra_dir.exists():
        print(f"ERROR: Spectra directory not found: {spectra_dir}")
        return

    # Output directory
    output_dir = Path(__file__).parent / 'results' / 'cv_output'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nInput: {spectra_dir}")
    print(f"Output: {output_dir}")

    # Initialize ion-to-droplet converter
    print("\nInitializing IonToDropletConverter...")
    converter = IonToDropletConverter(
        resolution=(512, 512),
        enable_physics_validation=True,
        validation_threshold=0.3
    )
    print("✓ Converter ready")

    # Get spectra files
    spectra_files = sorted(list(spectra_dir.glob('spectrum_*.tsv')))
    print(f"\nFound {len(spectra_files)} spectra files")

    # Process each spectrum
    for i, spec_file in enumerate(spectra_files[:10]):  # First 10
        print(f"\n{'='*80}")
        print(f"Processing {spec_file.name} ({i+1}/10)")
        print(f"{'='*80}")

        # Read spectrum
        df = pd.read_csv(spec_file, sep='\t')

        # Get columns
        if 'mz' not in df.columns:
            print(f"  ERROR: No 'mz' column")
            continue

        intensity_col = 'intensity' if 'intensity' in df.columns else 'i'
        if intensity_col not in df.columns:
            print(f"  ERROR: No intensity column")
            continue

        mzs = df['mz'].values
        intensities = df[intensity_col].values

        print(f"  {len(mzs)} ions, m/z range: {mzs.min():.2f}-{mzs.max():.2f}")

        # Convert to thermodynamic droplets
        print("  Converting ions to droplets...")
        image, ion_droplets = converter.convert_spectrum_to_image(
            mzs=mzs,
            intensities=intensities,
            rt=None,
            normalize=True
        )

        print(f"  Generated {len(ion_droplets)} droplets")

        if len(ion_droplets) > 0:
            # Show droplet stats
            avg_quality = np.mean([d.physics_quality for d in ion_droplets])
            avg_coherence = np.mean([d.droplet_params.phase_coherence for d in ion_droplets])
            print(f"  Avg physics quality: {avg_quality:.3f}")
            print(f"  Avg phase coherence: {avg_coherence:.3f}")

            # Save image
            image_path = output_dir / f"{spec_file.stem}_droplet.png"
            cv2.imwrite(str(image_path), image)
            print(f"  ✓ Saved image: {image_path.name}")

            # Save droplet data
            droplet_data = []
            for idx, droplet in enumerate(ion_droplets):
                droplet_data.append({
                    'droplet_idx': idx,
                    'mz': droplet.mz,
                    'intensity': droplet.intensity,
                    's_knowledge': droplet.s_entropy_coords.s_knowledge,
                    's_time': droplet.s_entropy_coords.s_time,
                    's_entropy': droplet.s_entropy_coords.s_entropy,
                    'velocity': droplet.droplet_params.velocity,
                    'radius': droplet.droplet_params.radius,
                    'phase_coherence': droplet.droplet_params.phase_coherence,
                    'categorical_state': droplet.categorical_state,
                    'physics_quality': droplet.physics_quality
                })

            droplet_df = pd.DataFrame(droplet_data)
            droplet_path = output_dir / f"{spec_file.stem}_droplets.tsv"
            droplet_df.to_csv(droplet_path, sep='\t', index=False)
            print(f"  ✓ Saved droplet data: {droplet_path.name}")

    # Print validation report
    print("\n" + "="*80)
    print("VALIDATION REPORT")
    print("="*80)
    print(converter.get_validation_report())

    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print(f"  - Droplet images: spectrum_*_droplet.png")
    print(f"  - Droplet data: spectrum_*_droplets.tsv")

if __name__ == '__main__':
    main()
