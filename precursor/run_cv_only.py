"""
Simple CV-Only Pipeline
========================

Just uses the ion-to-droplet algorithm and saves images.
No complex pipeline, no stages, no excuses.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add lavoisier/visual to path
visual_path = Path(__file__).parent.parent / 'visual'
sys.path.insert(0, str(visual_path))

# Import the CV modules
from IonToDropletConverter import IonToDropletConverter
from MSImageDatabase_Enhanced import MSImageDatabase

def main():
    # Input: Read a spectrum from stage_02 (has S-Entropy coords)
    spectra_dir = Path(__file__).parent / 'results' / 'metabolomics_analysis' / 'PL_Neg_Waters_qTOF' / 'stage_02_sentropy' / 'spectra'

    if not spectra_dir.exists():
        # Try stage_01
        spectra_dir = Path(__file__).parent / 'results' / 'metabolomics_analysis' / 'PL_Neg_Waters_qTOF' / 'stage_01_preprocessing' / 'spectra'

    if not spectra_dir.exists():
        print(f"ERROR: Spectra directory not found: {spectra_dir}")
        return

    # Output directory
    output_dir = Path(__file__).parent / 'results' / 'cv_only_output'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading spectra from: {spectra_dir}")
    print(f"Output directory: {output_dir}")

    # Initialize the ion-to-droplet converter
    print("\n" + "="*80)
    print("Initializing Ion-to-Droplet Converter")
    print("="*80)
    converter = IonToDropletConverter(
        resolution=(512, 512),
        enable_physics_validation=True,
        validation_threshold=0.3
    )

    # Initialize MS Image Database
    print("\n" + "="*80)
    print("Initializing MS Image Database")
    print("="*80)
    ms_db = MSImageDatabase(
        resolution=(512, 512),
        use_thermodynamic=True
    )

    # Process each spectrum
    spectra_files = sorted(list(spectra_dir.glob('spectrum_*.tsv')))
    print(f"\nFound {len(spectra_files)} spectra files")

    # Process first 10 spectra
    for i, spec_file in enumerate(spectra_files[:10]):
        print(f"\n{'='*80}")
        print(f"Processing: {spec_file.name}")
        print(f"{'='*80}")

        # Read spectrum
        df = pd.read_csv(spec_file, sep='\t')

        # Check columns
        if 'mz' not in df.columns:
            print(f"ERROR: No 'mz' column in {spec_file.name}")
            continue

        # Get intensity column (could be 'i' or 'intensity')
        intensity_col = None
        if 'intensity' in df.columns:
            intensity_col = 'intensity'
        elif 'i' in df.columns:
            intensity_col = 'i'
        else:
            print(f"ERROR: No intensity column in {spec_file.name}")
            continue

        mzs = df['mz'].values
        intensities = df[intensity_col].values

        print(f"Spectrum has {len(mzs)} ions")
        print(f"m/z range: {mzs.min():.2f} - {mzs.max():.2f}")
        print(f"Intensity range: {intensities.min():.2f} - {intensities.max():.2f}")

        # Convert to thermodynamic droplet image
        print("\nConverting ions to thermodynamic droplets...")
        image, ion_droplets = converter.convert_spectrum_to_image(
            mzs=mzs,
            intensities=intensities,
            rt=None,
            normalize=True
        )

        print(f"Generated {len(ion_droplets)} droplets (after physics filtering)")

        if len(ion_droplets) > 0:
            # Show some droplet details
            print("\nFirst 5 droplets:")
            for idx, droplet in enumerate(ion_droplets[:5]):
                print(f"  {idx+1}. m/z={droplet.mz:.4f}")
                print(f"     S-Entropy: knowledge={droplet.s_entropy_coords.s_knowledge:.3f}, "
                      f"time={droplet.s_entropy_coords.s_time:.3f}, "
                      f"entropy={droplet.s_entropy_coords.s_entropy:.3f}")
                print(f"     Droplet: velocity={droplet.droplet_params.velocity:.3f} m/s, "
                      f"radius={droplet.droplet_params.radius:.3f} mm, "
                      f"phase_coherence={droplet.droplet_params.phase_coherence:.3f}")
                print(f"     Categorical state: {droplet.categorical_state}")
                print(f"     Physics quality: {droplet.physics_quality:.3f}")

        # Save image
        import cv2
        image_path = output_dir / f"{spec_file.stem}_droplet.png"
        cv2.imwrite(str(image_path), image)
        print(f"\n✅ Saved droplet image: {image_path}")

        # Save droplet data
        if len(ion_droplets) > 0:
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
                    'surface_tension': droplet.droplet_params.surface_tension,
                    'temperature': droplet.droplet_params.temperature,
                    'phase_coherence': droplet.droplet_params.phase_coherence,
                    'categorical_state': droplet.categorical_state,
                    'physics_quality': droplet.physics_quality,
                    'is_valid': droplet.is_physically_valid
                })

            droplet_df = pd.DataFrame(droplet_data)
            droplet_path = output_dir / f"{spec_file.stem}_droplets.tsv"
            droplet_df.to_csv(droplet_path, sep='\t', index=False)
            print(f"✅ Saved droplet data: {droplet_path}")

        # Add to database
        spectrum_id = ms_db.add_spectrum(
            mzs=mzs,
            intensities=intensities,
            rt=None,
            metadata={'file': spec_file.name}
        )
        print(f"✅ Added to MS database with ID: {spectrum_id[:16]}...")

    # Print validation report
    print("\n" + "="*80)
    print("Physics Validation Report")
    print("="*80)
    print(converter.get_validation_report())

    # Save database
    db_path = output_dir / 'ms_image_database'
    ms_db.save_database(str(db_path))
    print(f"\n✅ Saved MS Image Database: {db_path}")

    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print(f"- Droplet images: spectrum_*_droplet.png")
    print(f"- Droplet data: spectrum_*_droplets.tsv")
    print(f"- MS database: ms_image_database/")

if __name__ == '__main__':
    main()
