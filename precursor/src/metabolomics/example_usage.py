"""
Example: Using SpectraReader with MSDataContainer
=================================================

Demonstrates the complete workflow:
1. Extract mzML data using SpectraReader
2. Organize into MSDataContainer
3. Access organized data structures
4. Compute S-Entropy coordinates
5. Prepare for downstream analysis
"""

from SpectraReader import extract_spectra
from DataStructure import MSDataContainer

def main():
    # ========================================================================
    # Step 1: Extract mzML data using existing SpectraReader
    # ========================================================================
    
    mzml_file = "path/to/sample_pos.mzml"  # Replace with actual path
    
    # Extraction parameters
    rt_range = [0, 30]  # RT range in minutes
    dda_top = 6
    ms1_threshold = 1000
    ms2_threshold = 10
    ms1_precision = 50e-6  # 50 ppm
    ms2_precision = 500e-6  # 500 ppm
    vendor = "thermo"  # or "waters", "agilent", "bruker", "sciex"
    ms1_max = 0  # Set to non-zero to limit MS1 max intensity
    
    print("="*70)
    print("EXTRACTING MZML DATA")
    print("="*70)
    
    # Extract using existing SpectraReader (UNCHANGED)
    scan_info_df, spectra_dict, ms1_xic_df = extract_spectra(
        usr_mzml=mzml_file,
        usr_rt_range=rt_range,
        usr_dda_top=dda_top,
        usr_ms1_threshold=ms1_threshold,
        usr_ms2_threshold=ms2_threshold,
        usr_ms1_precision=ms1_precision,
        usr_ms2_precision=ms2_precision,
        usr_vendor=vendor,
        usr_ms1_max=ms1_max
    )
    
    # ========================================================================
    # Step 2: Create MSDataContainer (NEW - ORGANIZES EXTRACTED DATA)
    # ========================================================================
    
    print("\n" + "="*70)
    print("CREATING DATA CONTAINER")
    print("="*70)
    
    # Store extraction parameters for reference
    extraction_params = {
        'rt_range': rt_range,
        'dda_top': dda_top,
        'ms1_threshold': ms1_threshold,
        'ms2_threshold': ms2_threshold,
        'ms1_precision': ms1_precision,
        'ms2_precision': ms2_precision,
        'vendor': vendor,
        'ms1_max': ms1_max,
    }
    
    # Create container
    container = MSDataContainer(
        mzml_filepath=mzml_file,
        scan_info_df=scan_info_df,
        spectra_dict=spectra_dict,
        ms1_xic_df=ms1_xic_df,
        extraction_params=extraction_params
    )
    
    print(container)
    
    # ========================================================================
    # Step 3: Access organized data (EXAMPLES)
    # ========================================================================
    
    print("\n" + "="*70)
    print("ACCESSING ORGANIZED DATA")
    print("="*70)
    
    # 3a. Get summary
    summary = container.summary()
    print(f"\nSample: {summary['file_info']['sample_name']}")
    print(f"Polarity: {summary['file_info']['polarity']}")
    print(f"Total spectra: {summary['statistics']['total_spectra']}")
    print(f"DDA events: {summary['statistics']['dda_events']}")
    
    # 3b. Get all MS1 spectra
    ms1_spectra = container.get_ms1_spectra()
    print(f"\nMS1 spectra: {len(ms1_spectra)}")
    
    # 3c. Get all MS2 spectra
    ms2_spectra = container.get_ms2_spectra()
    print(f"MS2 spectra: {len(ms2_spectra)}")
    
    # 3d. Get spectra in RT range
    rt_window_spectra = container.get_spectra_in_rt_range(10.0, 15.0, ms_level=2)
    print(f"\nMS2 spectra in RT 10-15 min: {len(rt_window_spectra)}")
    
    # 3e. Get precursor-fragment pairs
    all_pairs = container.get_precursor_fragment_pairs()
    print(f"\nTotal precursor-fragment pairs: {len(all_pairs)}")
    
    if all_pairs:
        example_pair = all_pairs[0]
        print(f"\nExample pair:")
        print(f"  Precursor m/z: {example_pair.precursor_mz:.4f}")
        print(f"  Precursor RT: {example_pair.precursor_rt:.2f} min")
        print(f"  Precursor intensity: {example_pair.precursor_intensity:.2e}")
        print(f"  Fragment spectrum peaks: {len(example_pair.ms2_spectrum)}")
    
    # 3f. Get DDA events
    dda_events = container._dda_events
    print(f"\nDDA events: {len(dda_events)}")
    
    if dda_events:
        first_event_idx = min(dda_events.keys())
        first_event = container.get_dda_event(first_event_idx)
        print(f"\nFirst DDA event:")
        print(f"  MS1 spectrum index: {first_event.ms1_spec_index}")
        print(f"  MS2 spectra: {len(first_event.ms2_spec_indices)}")
        print(f"  Precursor m/z values: {[f'{mz:.4f}' for mz in first_event.ms2_precursor_mzs[:3]]}")
    
    # 3g. Search by precursor m/z
    target_mz = 500.0  # Example m/z
    matching_spectra = container.get_spectra_by_precursor_mz(target_mz, tolerance_ppm=10.0)
    print(f"\nMS2 spectra with precursor ~{target_mz:.2f}: {len(matching_spectra)}")
    
    # 3h. Extract XIC
    xic_data = container.extract_xic(target_mz=500.0, tolerance_ppm=10.0)
    print(f"\nXIC data points: {len(xic_data)}")
    if not xic_data.empty:
        print(f"  RT range: {xic_data['rt'].min():.2f} - {xic_data['rt'].max():.2f} min")
        print(f"  Intensity range: {xic_data['i'].min():.2e} - {xic_data['i'].max():.2e}")
    
    # ========================================================================
    # Step 4: Compute S-Entropy coordinates (INTEGRATION POINT)
    # ========================================================================
    
    print("\n" + "="*70)
    print("COMPUTING S-ENTROPY COORDINATES")
    print("="*70)
    
    # This will be integrated with your actual S-Entropy framework
    # For now, it creates placeholder coordinates
    container.compute_s_entropy_coordinates()
    
    # Access computed coordinates
    first_spec_idx = list(container._spectrum_metadata.keys())[0]
    first_metadata = container.get_spectrum_metadata(first_spec_idx)
    print(f"\nExample S-Entropy coordinates (spec {first_spec_idx}):")
    print(f"  S_knowledge: {first_metadata.s_entropy_coords[0]:.4f}")
    print(f"  S_time: {first_metadata.s_entropy_coords[1]:.4f}")
    print(f"  S_entropy: {first_metadata.s_entropy_coords[2]:.4f}")
    
    # ========================================================================
    # Step 5: Export to DataFrame for analysis
    # ========================================================================
    
    print("\n" + "="*70)
    print("EXPORTING TO DATAFRAME")
    print("="*70)
    
    df = container.to_dataframe()
    print(f"\nExported DataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    # Can save to CSV for further analysis
    # df.to_csv('spectrum_metadata.csv', index=False)
    
    # ========================================================================
    # Step 6: Demonstrate precursor-fragment linking
    # ========================================================================
    
    print("\n" + "="*70)
    print("PRECURSOR-FRAGMENT LINKING")
    print("="*70)
    
    # Get a specific DDA event
    if dda_events:
        example_event_idx = list(dda_events.keys())[0]
        example_event = container.get_dda_event(example_event_idx)
        
        print(f"\nDDA Event {example_event_idx}:")
        print(f"  MS1 RT: {example_event.ms1_scan_time:.2f} min")
        print(f"  MS1 peaks: {len(example_event.ms1_spectrum)}")
        print(f"  MS2 scans: {len(example_event.ms2_spectra)}")
        
        # Show each precursor-fragment pair
        for i, pair in enumerate(example_event.precursor_fragment_pairs[:3]):  # First 3
            print(f"\n  Pair {i+1}:")
            print(f"    Precursor m/z: {pair.precursor_mz:.4f}")
            print(f"    Precursor intensity: {pair.precursor_intensity:.2e}")
            print(f"    PPM error: {pair.precursor_ppm_error:.2f}")
            print(f"    Fragment RT: {pair.fragment_rt:.2f} min")
            print(f"    Fragment peaks: {len(pair.ms2_spectrum)}")
    
    print("\n" + "="*70)
    print("WORKFLOW COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("  1. Integrate actual S-Entropy coordinate calculation")
    print("  2. Compute phase-lock signatures")
    print("  3. Apply dual-modality analysis")
    print("  4. Perform categorical completion")
    print("  5. Generate annotation using Empty Dictionary")

if __name__ == "__main__":
    main()

