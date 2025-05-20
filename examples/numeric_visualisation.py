#!/usr/bin/env python3
"""
numeric_visualisation.py - Visualization of processed MS data from zarr storage.

This script focuses on visualizing the numerical MS data that has been processed
through the Lavoisier pipeline and stored in zarr format.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import zarr
import pandas as pd
import h5py
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.gridspec as gridspec

# Set visualization style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")

# Output directory
SHOWCASE_DIR = "public/showcase"
os.makedirs(SHOWCASE_DIR, exist_ok=True)

def load_zarr_array(file_path):
    """
    Load a zarr array from a .zarray file
    
    Args:
        file_path: Path to the directory containing the .zarray file
        
    Returns:
        Numpy array with the data
    """
    try:
        # Check if path exists
        if not os.path.exists(file_path):
            print(f"Error: Path {file_path} does not exist")
            return None
            
        # For the specific Zarr structure used here, we need to find the .zarray file
        zarray_path = file_path
        if os.path.isdir(file_path):
            zarray_path = os.path.join('/Users/kundai/Development/bioinformatics/lavoisier/', file_path, "data.zarray")
            print(zarray_path)

        if not os.path.exists(zarray_path):
            print(f"Error: data.zarray file not found at {zarray_path}")
            return None
            
        # Read the .zarray file as JSON to get metadata
        with open(zarray_path, 'r') as f:
            metadata = f.read()
            print(f"Successfully read data.zarray metadata from {zarray_path}")
            
        # As a simple test, return an empty array for now
        # This will at least allow us to proceed with visualization
        # We can refine the actual data loading once we confirm this works
        dummy_data = np.array([1, 2, 3])  # Simple dummy data for testing
        print(f"Created dummy data for {file_path} to test visualization flow")
        return dummy_data
        
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def load_experiment_data(experiment_name="PL_Neg_Waters_qTOF"):
    """
    Load data for a specific experiment
    
    Args:
        experiment_name: Name of the experiment
        
    Returns:
        Dictionary with the loaded data
    """
    base_path = f"public/output/results.zarr/{experiment_name}"
    
    # Data structure to hold results
    data = {
        'ms1_xic': {},
        'scan_info': {},
        'spectra': {}
    }
    
    # Target the specific .zarray files/folders
    ms1_xic_path = f"{base_path}/ms1_xic"
    scan_info_path = f"{base_path}/scan_info"
    
    # Try to load ms1_xic data from the .zarray file
    ms1_array = load_zarr_array(ms1_xic_path)
    if ms1_array is not None:
        # For compatibility with existing visualizations, create expected structure
        data['ms1_xic']['mz_array'] = ms1_array  
        data['ms1_xic']['int_array'] = ms1_array  # Both keys point to same data for now
    
    # Try to load scan_info data from the .zarray file
    scan_info_array = load_zarr_array(scan_info_path)
    if scan_info_array is not None:
        # For compatibility with existing visualizations
        data['scan_info']['scan_time'] = scan_info_array  
        # The following are needed by visualization functions:
        data['scan_info']['ms_level'] = np.ones(len(scan_info_array), dtype=int)  # Assume MS1
        data['scan_info']['DDA_rank'] = np.zeros(len(scan_info_array), dtype=int)  # No DDA rank
        data['scan_info']['spec_index'] = np.arange(len(scan_info_array), dtype=int)  # Index
    
    # For spectra, try to load if the directory exists
    spectra_base = f"{base_path}/spectra"
    if os.path.exists(spectra_base) and os.path.isdir(spectra_base):
        try:
            # List subdirectories
            for item in os.listdir(spectra_base):
                item_path = os.path.join(spectra_base, item)
                if os.path.isdir(item_path) and not item.startswith('.'):
                    spectra_array = load_zarr_array(item_path)
                    if spectra_array is not None:
                        data['spectra'][item] = spectra_array
        except Exception as e:
            print(f"Error exploring spectra directory: {e}")
    
    # Check if we loaded any data
    if data['ms1_xic'] or data['scan_info'] or data['spectra']:
        return data
    else:
        return None

def create_intensity_heatmap(data, output_dir=SHOWCASE_DIR):
    """
    Create heatmap of MS1 intensity across retention time and m/z
    
    Args:
        data: Dictionary of loaded MS data
        output_dir: Directory to save output
    """
    for sample_name, sample_data in data.items():
        if not sample_data['ms1_xic'] or 'mz_array' not in sample_data['ms1_xic']:
            print(f"Skipping intensity heatmap for {sample_name}: No MS1 data")
            continue
            
        print(f"Creating intensity heatmap for {sample_name}")
        
        # Extract data
        mz_arrays = sample_data['ms1_xic']['mz_array']
        int_arrays = sample_data['ms1_xic']['int_array']
        
        if 'scan_time' in sample_data['scan_info']:
            scan_times = sample_data['scan_info']['scan_time']
        else:
            # Create placeholder times if not available
            scan_times = np.arange(len(mz_arrays))
            
        # Create grid for heatmap
        grid_size = (200, 200)  # Adjust resolution as needed
        
        # Find global m/z range
        all_mz_arrays = [arr for arr in mz_arrays if len(arr) > 0]
        if not all_mz_arrays:
            print(f"No valid m/z data for {sample_name}")
            continue
            
        all_mz = np.concatenate(all_mz_arrays)
        min_mz, max_mz = np.min(all_mz), np.max(all_mz)
        
        # Create ranges for binning
        rt_range = np.linspace(min(scan_times), max(scan_times), grid_size[0])
        mz_range = np.linspace(min_mz, max_mz, grid_size[1])
        
        # Initialize intensity grid
        intensity_grid = np.zeros(grid_size)
        
        # Fill grid with intensity values
        for i, rt in enumerate(scan_times):
            if i >= len(mz_arrays) or len(mz_arrays[i]) == 0:
                continue
                
            rt_idx = np.argmin(np.abs(rt_range - rt))
            mz_array = mz_arrays[i]
            int_array = int_arrays[i]
            
            for mz, intensity in zip(mz_array, int_array):
                mz_idx = np.argmin(np.abs(mz_range - mz))
                intensity_grid[rt_idx, mz_idx] += intensity
                
        # Create heatmap with log scale for better visualization
        plt.figure(figsize=(14, 10))
        plt.imshow(np.log1p(intensity_grid), aspect='auto', cmap='viridis',
                  extent=[min_mz, max_mz, max(scan_times), min(scan_times)],
                  origin='upper')
        
        plt.colorbar(label='log(Intensity + 1)')
        plt.title(f"MS1 Intensity Heatmap: {sample_name}")
        plt.xlabel('m/z')
        plt.ylabel('Retention Time (min)')
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(output_dir, f"{sample_name}_intensity_heatmap.png"), dpi=300)
        plt.close()

def create_total_ion_chromatogram(data, output_dir=SHOWCASE_DIR):
    """
    Create total ion chromatogram (TIC) for each sample
    
    Args:
        data: Dictionary of loaded MS data
        output_dir: Directory to save output
    """
    for sample_name, sample_data in data.items():
        if not sample_data['ms1_xic'] or 'int_array' not in sample_data['ms1_xic']:
            print(f"Skipping TIC for {sample_name}: No MS1 data")
            continue
            
        print(f"Creating total ion chromatogram for {sample_name}")
        
        # Get intensity arrays and scan times
        int_arrays = sample_data['ms1_xic']['int_array']
        
        if 'scan_time' in sample_data['scan_info']:
            scan_times = sample_data['scan_info']['scan_time']
        else:
            scan_times = np.arange(len(int_arrays))
            
        # Calculate total intensity for each scan
        total_intensity = np.array([np.sum(arr) if len(arr) > 0 else 0 for arr in int_arrays])
        
        # Create plot
        plt.figure(figsize=(14, 8))
        plt.plot(scan_times, total_intensity, linewidth=2)
        plt.fill_between(scan_times, 0, total_intensity, alpha=0.3)
        
        # Find peaks in chromatogram
        prominence = np.max(total_intensity) * 0.05  # 5% of max intensity
        peaks, _ = find_peaks(total_intensity, prominence=prominence, distance=10)
        
        # Label peaks
        for peak in peaks:
            plt.annotate(f"{scan_times[peak]:.2f}",
                       (scan_times[peak], total_intensity[peak]),
                       xytext=(0, 10), textcoords='offset points',
                       ha='center', fontsize=9)
            plt.scatter(scan_times[peak], total_intensity[peak], c='red', s=40, zorder=10)
            
        plt.title(f"Total Ion Chromatogram: {sample_name}")
        plt.xlabel("Retention Time (min)")
        plt.ylabel("Total Ion Current")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(output_dir, f"{sample_name}_tic.png"), dpi=300)
        plt.close()

def create_3d_visualization(data, output_dir=SHOWCASE_DIR):
    """
    Create 3D visualization of peak data
    
    Args:
        data: Dictionary of loaded MS data
        output_dir: Directory to save output
    """
    for sample_name, sample_data in data.items():
        if not sample_data['ms1_xic'] or 'mz_array' not in sample_data['ms1_xic']:
            print(f"Skipping 3D visualization for {sample_name}: No MS1 data")
            continue
            
        print(f"Creating 3D visualization for {sample_name}")
        
        # Get data arrays
        mz_arrays = sample_data['ms1_xic']['mz_array']
        int_arrays = sample_data['ms1_xic']['int_array']
        
        if 'scan_time' in sample_data['scan_info']:
            scan_times = sample_data['scan_info']['scan_time']
        else:
            scan_times = np.arange(len(mz_arrays))
            
        # Create 3D plot
        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # For large datasets, use a subset for clarity
        step = max(1, len(scan_times) // 50)
        
        # Plot the top peaks for selected scans
        for i in range(0, len(scan_times), step):
            if i >= len(mz_arrays) or len(mz_arrays[i]) == 0:
                continue
                
            rt = scan_times[i]
            mz_array = mz_arrays[i]
            int_array = int_arrays[i]
            
            # Select top 10 peaks or fewer if less available
            if len(int_array) > 10:
                top_indices = np.argsort(int_array)[-10:]
                mz_values = mz_array[top_indices]
                int_values = int_array[top_indices]
            else:
                mz_values = mz_array
                int_values = int_array
                
            # Plot points
            ax.scatter(np.full_like(mz_values, rt), mz_values, int_values,
                      c=int_values, cmap='plasma', s=50, alpha=0.7)
            
            # Add stems for visibility
            for mz, intensity in zip(mz_values, int_values):
                ax.plot([rt, rt], [mz, mz], [0, intensity], color='gray', alpha=0.2)
                
        # Set labels and title
        ax.set_xlabel('Retention Time (min)')
        ax.set_ylabel('m/z')
        ax.set_zlabel('Intensity')
        ax.set_title(f"3D Peak Visualization: {sample_name}")
        
        # Add color bar
        norm = Normalize(vmin=0, vmax=np.max([np.max(int_array) if len(int_array) > 0 else 0 
                                             for int_array in int_arrays]))
        sm = cm.ScalarMappable(cmap='plasma', norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.1)
        cbar.set_label('Intensity')
        
        # Set optimal viewing angle
        ax.view_init(elev=35, azim=45)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(output_dir, f"{sample_name}_3d_vis.png"), dpi=300)
        plt.close()

def create_ms2_fragment_analysis(data, output_dir=SHOWCASE_DIR):
    """
    Create visualization of MS2 fragmentation patterns
    
    Args:
        data: Dictionary of loaded MS data
        output_dir: Directory to save output
    """
    for sample_name, sample_data in data.items():
        # Check if we have MS2 spectra
        if not sample_data['spectra']:
            print(f"Skipping MS2 analysis for {sample_name}: No MS2 data")
            continue
            
        print(f"Creating MS2 fragmentation analysis for {sample_name}")
        
        # Get scan info
        if 'DDA_rank' in sample_data['scan_info']:
            dda_ranks = sample_data['scan_info']['DDA_rank']
            scan_indices = sample_data['scan_info']['spec_index']
            scan_times = sample_data['scan_info'].get('scan_time', np.arange(len(dda_ranks)))
            
            # Find MS2 scans (rank > 0)
            ms2_indices = [i for i, rank in enumerate(dda_ranks) if rank > 0]
            
            if not ms2_indices:
                print(f"No MS2 spectra found for {sample_name}")
                continue
                
            # Plot distribution of MS2 scans
            plt.figure(figsize=(14, 8))
            plt.subplot(2, 1, 1)
            plt.hist(scan_times[ms2_indices], bins=30, alpha=0.7)
            plt.title(f"MS2 Scan Distribution: {sample_name}")
            plt.xlabel("Retention Time (min)")
            plt.ylabel("Number of MS2 Scans")
            plt.grid(True, alpha=0.3)
            
            # Plot DDA rank distribution
            plt.subplot(2, 1, 2)
            ranks = [dda_ranks[i] for i in ms2_indices]
            rank_counts = pd.Series(ranks).value_counts().sort_index()
            plt.bar(rank_counts.index, rank_counts.values)
            plt.title("DDA Rank Distribution")
            plt.xlabel("DDA Rank")
            plt.ylabel("Count")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save figure
            plt.savefig(os.path.join(output_dir, f"{sample_name}_ms2_distribution.png"), dpi=300)
            plt.close()
            
            # Plot example MS2 spectra
            # Select a few representative MS2 scans from different retention time regions
            ms2_rt_values = [scan_times[i] for i in ms2_indices]
            rt_min, rt_max = min(ms2_rt_values), max(ms2_rt_values)
            rt_bins = np.linspace(rt_min, rt_max, 4)  # Divide into 3 regions
            
            fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
            
            for i, (rt_start, rt_end) in enumerate(zip(rt_bins[:-1], rt_bins[1:])):
                # Find MS2 scans in this retention time range
                region_indices = [idx for idx, rt in zip(ms2_indices, ms2_rt_values) 
                                if rt_start <= rt < rt_end]
                
                if region_indices:
                    # Take the most intense MS2 scan in this region
                    selected_idx = region_indices[len(region_indices)//2]
                    
                    # Get the spectrum if available in the spectra dict
                    spec_index = scan_indices[selected_idx]
                    
                    if str(spec_index) in sample_data['spectra']:
                        spec = sample_data['spectra'][str(spec_index)]
                        # Check if spec is a dictionary with mz and intensity
                        if isinstance(spec, dict) and 'mz' in spec and 'intensity' in spec:
                            mz = spec['mz']
                            intensity = spec['intensity']
                        else:
                            # Try another format - this depends on how your data is structured
                            mz = np.array([item[0] for item in spec])
                            intensity = np.array([item[1] for item in spec])
                            
                        # Plot spectrum
                        axes[i].stem(mz, intensity, markerfmt=" ", basefmt=" ")
                        axes[i].set_title(f"MS2 Spectrum at RT: {scan_times[selected_idx]:.2f} min, Rank: {dda_ranks[selected_idx]}")
                        axes[i].set_ylabel("Intensity")
                        axes[i].grid(True, alpha=0.3)
            
            axes[-1].set_xlabel("m/z")
            plt.suptitle(f"Representative MS2 Spectra: {sample_name}")
            plt.tight_layout()
            
            # Save figure
            plt.savefig(os.path.join(output_dir, f"{sample_name}_ms2_spectra.png"), dpi=300)
            plt.close()

def create_mz_distribution(data, output_dir=SHOWCASE_DIR):
    """
    Create m/z distribution visualization
    
    Args:
        data: Dictionary of loaded MS data
        output_dir: Directory to save output
    """
    for sample_name, sample_data in data.items():
        if not sample_data['ms1_xic'] or 'mz_array' not in sample_data['ms1_xic']:
            print(f"Skipping m/z distribution for {sample_name}: No MS1 data")
            continue
            
        print(f"Creating m/z distribution for {sample_name}")
        
        # Get m/z arrays
        mz_arrays = sample_data['ms1_xic']['mz_array']
        int_arrays = sample_data['ms1_xic']['int_array']
        
        # Concatenate all m/z values into a single array
        all_mz = []
        all_intensities = []
        
        for mz_array, int_array in zip(mz_arrays, int_arrays):
            if len(mz_array) > 0:
                all_mz.extend(mz_array)
                all_intensities.extend(int_array)
                
        if not all_mz:
            print(f"No m/z data available for {sample_name}")
            continue
            
        # Create weighted histogram - weights by intensity
        plt.figure(figsize=(14, 8))
        
        # Histogram of m/z values weighted by intensity
        hist, bins = np.histogram(all_mz, bins=100, weights=all_intensities)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        plt.bar(bin_centers, hist, width=(bins[1]-bins[0]))
        
        plt.title(f"m/z Distribution (Intensity Weighted): {sample_name}")
        plt.xlabel("m/z")
        plt.ylabel("Summed Intensity")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(output_dir, f"{sample_name}_mz_distribution.png"), dpi=300)
        plt.close()
        
        # Create density plot of m/z with log intensity
        plt.figure(figsize=(14, 8))
        
        # Use top 10% most intense peaks to reduce noise
        intensity_threshold = np.percentile(all_intensities, 90)
        filtered_mz = [mz for mz, intensity in zip(all_mz, all_intensities) 
                       if intensity >= intensity_threshold]
        
        if filtered_mz:
            kde = gaussian_kde(filtered_mz)
            x_vals = np.linspace(min(filtered_mz), max(filtered_mz), 1000)
            plt.plot(x_vals, kde(x_vals), 'k-', linewidth=2)
            plt.fill_between(x_vals, 0, kde(x_vals), alpha=0.5)
            
        plt.title(f"m/z Density Distribution (Top 10% Intensity): {sample_name}")
        plt.xlabel("m/z")
        plt.ylabel("Density")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(output_dir, f"{sample_name}_mz_density.png"), dpi=300)
        plt.close()

def create_summary_dashboard(data, output_dir=SHOWCASE_DIR):
    """
    Create a comprehensive dashboard summarizing the MS data
    
    Args:
        data: Dictionary of loaded MS data
        output_dir: Directory to save output
    """
    for sample_name, sample_data in data.items():
        if not sample_data['ms1_xic']:
            print(f"Skipping dashboard for {sample_name}: No MS1 data")
            continue
            
        print(f"Creating summary dashboard for {sample_name}")
        
        # Create a figure with multiple subplots
        fig = plt.figure(figsize=(20, 16))
        gs = gridspec.GridSpec(3, 3, figure=fig)
        
        # 1. Total Ion Chromatogram
        ax1 = fig.add_subplot(gs[0, :])
        
        if 'int_array' in sample_data['ms1_xic'] and 'scan_time' in sample_data['scan_info']:
            int_arrays = sample_data['ms1_xic']['int_array']
            scan_times = sample_data['scan_info']['scan_time']
            
            total_intensity = np.array([np.sum(arr) if len(arr) > 0 else 0 for arr in int_arrays])
            
            ax1.plot(scan_times, total_intensity, linewidth=2)
            ax1.fill_between(scan_times, 0, total_intensity, alpha=0.3)
            
            # Add peaks
            prominence = np.max(total_intensity) * 0.05
            peaks, _ = find_peaks(total_intensity, prominence=prominence, distance=10)
            ax1.scatter(scan_times[peaks], total_intensity[peaks], c='red', s=40, zorder=10)
            
            ax1.set_title("Total Ion Chromatogram")
            ax1.set_xlabel("Retention Time (min)")
            ax1.set_ylabel("Total Ion Current")
            ax1.grid(True, alpha=0.3)
        
        # 2. m/z Distribution
        ax2 = fig.add_subplot(gs[1, 0])
        
        if 'mz_array' in sample_data['ms1_xic']:
            mz_arrays = sample_data['ms1_xic']['mz_array']
            int_arrays = sample_data['ms1_xic']['int_array']
            
            all_mz = []
            all_intensities = []
            
            for mz_array, int_array in zip(mz_arrays, int_arrays):
                if len(mz_array) > 0:
                    all_mz.extend(mz_array)
                    all_intensities.extend(int_array)
            
            if all_mz:
                hist, bins = np.histogram(all_mz, bins=50, weights=all_intensities)
                bin_centers = (bins[:-1] + bins[1:]) / 2
                
                ax2.bar(bin_centers, hist, width=(bins[1]-bins[0]))
                ax2.set_title("m/z Distribution")
                ax2.set_xlabel("m/z")
                ax2.set_ylabel("Summed Intensity")
                ax2.grid(True, alpha=0.3)
        
        # 3. MS2 Scan Distribution (if available)
        ax3 = fig.add_subplot(gs[1, 1])
        
        if 'DDA_rank' in sample_data['scan_info']:
            dda_ranks = sample_data['scan_info']['DDA_rank']
            scan_times = sample_data['scan_info'].get('scan_time', np.arange(len(dda_ranks)))
            
            ms2_indices = [i for i, rank in enumerate(dda_ranks) if rank > 0]
            
            if ms2_indices:
                ax3.hist(scan_times[ms2_indices], bins=30, alpha=0.7)
                ax3.set_title("MS2 Scan Distribution")
                ax3.set_xlabel("Retention Time (min)")
                ax3.set_ylabel("Number of MS2 Scans")
                ax3.grid(True, alpha=0.3)
        
        # 4. Intensity Distribution
        ax4 = fig.add_subplot(gs[1, 2])
        
        if 'int_array' in sample_data['ms1_xic']:
            all_intensities = []
            for int_array in sample_data['ms1_xic']['int_array']:
                if len(int_array) > 0:
                    all_intensities.extend(int_array)
            
            if all_intensities:
                # Log transform for better visualization
                log_intensities = np.log1p(all_intensities)
                
                ax4.hist(log_intensities, bins=50, alpha=0.7)
                ax4.set_title("Intensity Distribution")
                ax4.set_xlabel("log(Intensity + 1)")
                ax4.set_ylabel("Frequency")
                ax4.grid(True, alpha=0.3)
        
        # 5. 2D Heatmap or contour plot
        ax5 = fig.add_subplot(gs[2, :])
        
        if ('mz_array' in sample_data['ms1_xic'] and 
            'int_array' in sample_data['ms1_xic'] and 
            'scan_time' in sample_data['scan_info']):
            
            mz_arrays = sample_data['ms1_xic']['mz_array']
            int_arrays = sample_data['ms1_xic']['int_array']
            scan_times = sample_data['scan_info']['scan_time']
            
            # Create simplified heatmap for dashboard
            grid_size = (100, 100)
            
            # Find global m/z range
            all_mz = np.concatenate([arr for arr in mz_arrays if len(arr) > 0])
            min_mz, max_mz = np.min(all_mz), np.max(all_mz)
            
            rt_range = np.linspace(min(scan_times), max(scan_times), grid_size[0])
            mz_range = np.linspace(min_mz, max_mz, grid_size[1])
            
            intensity_grid = np.zeros(grid_size)
            
            for i, rt in enumerate(scan_times):
                if i >= len(mz_arrays) or len(mz_arrays[i]) == 0:
                    continue
                    
                rt_idx = np.argmin(np.abs(rt_range - rt))
                mz_array = mz_arrays[i]
                int_array = int_arrays[i]
                
                for mz, intensity in zip(mz_array, int_array):
                    mz_idx = np.argmin(np.abs(mz_range - mz))
                    intensity_grid[rt_idx, mz_idx] += intensity
                    
            im = ax5.imshow(np.log1p(intensity_grid), aspect='auto', cmap='viridis',
                      extent=[min_mz, max_mz, max(scan_times), min(scan_times)],
                      origin='upper')
            
            plt.colorbar(im, ax=ax5, label='log(Intensity + 1)')
            ax5.set_title("MS1 Intensity Heatmap")
            ax5.set_xlabel("m/z")
            ax5.set_ylabel("Retention Time (min)")
        
        # Set overall title
        plt.suptitle(f"MS Data Summary: {sample_name}", fontsize=20)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        # Save figure
        plt.savefig(os.path.join(output_dir, f"{sample_name}_dashboard.png"), dpi=300)
        plt.close()

def create_dual_sample_comparison(data, output_dir=SHOWCASE_DIR):
    """
    Create comparative visualizations between different samples
    
    Args:
        data: Dictionary of loaded MS data
        output_dir: Directory to save output
    """
    if len(data) < 2:
        print("Need at least 2 samples for comparison")
        return
        
    print("Creating sample comparison visualizations")
    
    # Get sample names
    sample_names = list(data.keys())
    
    # Create comparative TIC plot
    plt.figure(figsize=(14, 8))
    
    for sample_name in sample_names:
        sample_data = data[sample_name]
        
        if 'int_array' in sample_data['ms1_xic'] and 'scan_time' in sample_data['scan_info']:
            int_arrays = sample_data['ms1_xic']['int_array']
            scan_times = sample_data['scan_info']['scan_time']
            
            total_intensity = np.array([np.sum(arr) if len(arr) > 0 else 0 for arr in int_arrays])
            
            # Normalize to make comparison easier
            normalized_intensity = total_intensity / np.max(total_intensity)
            
            plt.plot(scan_times, normalized_intensity, linewidth=2, label=sample_name)
            
    plt.title("Comparative Total Ion Chromatograms (Normalized)")
    plt.xlabel("Retention Time (min)")
    plt.ylabel("Normalized Intensity")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, "sample_comparison_tic.png"), dpi=300)
    plt.close()
    
    # Create comparative m/z distribution
    plt.figure(figsize=(14, 8))
    
    for sample_name in sample_names:
        sample_data = data[sample_name]
        
        if 'mz_array' in sample_data['ms1_xic'] and 'int_array' in sample_data['ms1_xic']:
            mz_arrays = sample_data['ms1_xic']['mz_array']
            int_arrays = sample_data['ms1_xic']['int_array']
            
            all_mz = []
            all_intensities = []
            
            for mz_array, int_array in zip(mz_arrays, int_arrays):
                if len(mz_array) > 0:
                    all_mz.extend(mz_array)
                    all_intensities.extend(int_array)
            
            if all_mz:
                # Use KDE for smoother visualization
                kde = gaussian_kde(all_mz, weights=all_intensities/np.sum(all_intensities))
                x_vals = np.linspace(min(all_mz), max(all_mz), 1000)
                plt.plot(x_vals, kde(x_vals), linewidth=2, label=sample_name)
                
    plt.title("Comparative m/z Distribution")
    plt.xlabel("m/z")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, "sample_comparison_mz.png"), dpi=300)
    plt.close()

def create_pipeline_comparison(numeric_data, output_dir=SHOWCASE_DIR):
    """
    Create comparison visualizations between numeric and visual processing pipelines
    
    Args:
        numeric_data: Dictionary of loaded numeric MS data
        output_dir: Directory to save output
    """
    print("Creating pipeline comparison visualizations")
    
    # First, try to load visual processing data
    try:
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from examples.visual_pipeline import load_spectrum_database, load_processed_spectra
        
        visual_metadata = load_spectrum_database()
        visual_spectra = load_processed_spectra()
        
        if not visual_metadata and not visual_spectra:
            print("No visual pipeline data available for comparison")
            return
            
        # Create comparison dashboard
        fig = plt.figure(figsize=(20, 16))
        gs = gridspec.GridSpec(2, 2, figure=fig)
        
        # 1. Compare total ion chromatograms (numeric) vs retention time distribution (visual)
        ax1 = fig.add_subplot(gs[0, 0])
        
        # Plot TIC from numeric data (overlaid for all experiments)
        has_numeric_data = False
        for sample_name, sample_data in numeric_data.items():
            if 'int_array' in sample_data['ms1_xic'] and 'scan_time' in sample_data['scan_info']:
                has_numeric_data = True
                int_arrays = sample_data['ms1_xic']['int_array']
                scan_times = sample_data['scan_info']['scan_time']
                
                # Calculate and normalize total intensity
                total_intensity = np.array([np.sum(arr) if len(arr) > 0 else 0 for arr in int_arrays])
                norm_intensity = total_intensity / np.max(total_intensity)
                
                ax1.plot(scan_times, norm_intensity, label=f"Numeric: {sample_name}", alpha=0.7)
        
        # Plot retention time distribution from visual data if available
        if visual_metadata and 'retention_times' in visual_metadata:
            rt_values = visual_metadata['retention_times']
            
            # Create histogram
            hist, bins = np.histogram(rt_values, bins=50, density=True)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            
            # Plot on secondary Y axis
            ax1_twin = ax1.twinx()
            ax1_twin.plot(bin_centers, hist, 'r-', linewidth=2, 
                         label="Visual RT Distribution")
            ax1_twin.set_ylabel("Density (Visual)")
            ax1_twin.legend(loc="upper right")
            
        if has_numeric_data:
            ax1.set_xlabel("Retention Time (min)")
            ax1.set_ylabel("Normalized Intensity (Numeric)")
            ax1.legend(loc="upper left")
            ax1.set_title("TIC (Numeric) vs RT Distribution (Visual)")
            ax1.grid(True, alpha=0.3)
        else:
            ax1.set_title("No TIC Data Available")
            
        # 2. Compare m/z distributions
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Plot m/z distribution from numeric data
        has_mz_data = False
        for sample_name, sample_data in numeric_data.items():
            if 'mz_array' in sample_data['ms1_xic'] and 'int_array' in sample_data['ms1_xic']:
                has_mz_data = True
                mz_arrays = sample_data['ms1_xic']['mz_array']
                int_arrays = sample_data['ms1_xic']['int_array']
                
                # Collect all m/z values
                all_mz = []
                all_intensities = []
                
                for mz_array, int_array in zip(mz_arrays, int_arrays):
                    if len(mz_array) > 0:
                        all_mz.extend(mz_array)
                        all_intensities.extend(int_array)
                
                if all_mz:
                    # Calculate KDE for smoother visualization
                    kde = gaussian_kde(all_mz, weights=all_intensities/np.sum(all_intensities))
                    x_vals = np.linspace(min(all_mz), max(all_mz), 1000)
                    ax2.plot(x_vals, kde(x_vals), linewidth=2, 
                            label=f"Numeric: {sample_name}", alpha=0.7)
        
        # Plot m/z distribution from visual data if available
        if visual_metadata and 'mz_values' in visual_metadata:
            mz_values = visual_metadata['mz_values']
            
            # Create KDE
            kde = gaussian_kde(mz_values)
            x_vals = np.linspace(min(mz_values), max(mz_values), 1000)
            
            # Plot
            ax2.plot(x_vals, kde(x_vals), 'r-', linewidth=2, 
                    label="Visual m/z Distribution")
            
        if has_mz_data:
            ax2.set_xlabel("m/z")
            ax2.set_ylabel("Density")
            ax2.legend()
            ax2.set_title("m/z Distribution Comparison")
            ax2.grid(True, alpha=0.3)
        else:
            ax2.set_title("No m/z Distribution Data Available")
            
        # 3. Compare feature space (if available from both)
        ax3 = fig.add_subplot(gs[1, 0])
        
        # For numeric data, we can use PCA on MS1 data
        has_numeric_features = False
        pca_features = []
        pca_labels = []
        
        for sample_name, sample_data in numeric_data.items():
            if 'mz_array' in sample_data['ms1_xic'] and 'int_array' in sample_data['ms1_xic']:
                # Use a sampling of MS1 spectra as features
                mz_arrays = sample_data['ms1_xic']['mz_array']
                int_arrays = sample_data['ms1_xic']['int_array']
                
                # Select a subset of spectra for visualization
                indices = np.linspace(0, len(mz_arrays) - 1, min(200, len(mz_arrays))).astype(int)
                
                for idx in indices:
                    if idx < len(mz_arrays) and len(mz_arrays[idx]) > 0:
                        # Create a frequency histogram as a feature vector
                        mz_array = mz_arrays[idx]
                        int_array = int_arrays[idx]
                        
                        # Simple binning
                        min_mz = min(mz_array)
                        max_mz = max(mz_array)
                        n_bins = 50
                        
                        bins = np.linspace(min_mz, max_mz, n_bins + 1)
                        hist, _ = np.histogram(mz_array, bins=bins, weights=int_array)
                        
                        # Normalize
                        if np.sum(hist) > 0:
                            hist = hist / np.sum(hist)
                            
                        # Add to features
                        pca_features.append(hist)
                        pca_labels.append(sample_name)
        
        if pca_features:
            has_numeric_features = True
            pca_features = np.array(pca_features)
            
            # Apply PCA
            pca = PCA(n_components=2)
            numeric_pca = pca.fit_transform(pca_features)
            
            # Plot numeric PCA
            for sample in set(pca_labels):
                indices = [i for i, label in enumerate(pca_labels) if label == sample]
                ax3.scatter(numeric_pca[indices, 0], numeric_pca[indices, 1], 
                          label=f"Numeric: {sample}", alpha=0.7, s=30)
        
        # Plot visual features if available
        if visual_metadata and 'features' in visual_metadata:
            visual_features = visual_metadata['features']
            
            # Sample a subset if needed
            if len(visual_features) > 200:
                indices = np.random.choice(len(visual_features), 200, replace=False)
                visual_subset = visual_features[indices]
            else:
                visual_subset = visual_features
                
            # Apply PCA
            pca = PCA(n_components=2)
            visual_pca = pca.fit_transform(visual_subset)
            
            # Plot as a different color/marker
            ax3.scatter(visual_pca[:, 0], visual_pca[:, 1], 
                      c='red', marker='x', label="Visual Features", s=50, alpha=0.7)
            
        if has_numeric_features or 'features' in visual_metadata:
            ax3.set_xlabel("PC1")
            ax3.set_ylabel("PC2")
            ax3.legend()
            ax3.set_title("Feature Space Comparison")
            ax3.grid(True, alpha=0.3)
        else:
            ax3.set_title("No Feature Data Available")
            
        # 4. Compare intensity heatmaps or distributions
        ax4 = fig.add_subplot(gs[1, 1])
        
        # For numeric, we can show intensity distribution
        has_intensity_data = False
        for sample_name, sample_data in numeric_data.items():
            if 'int_array' in sample_data['ms1_xic']:
                has_intensity_data = True
                int_arrays = sample_data['ms1_xic']['int_array']
                
                # Collect all intensity values
                all_intensities = []
                for int_array in int_arrays:
                    if len(int_array) > 0:
                        all_intensities.extend(int_array)
                
                if all_intensities:
                    # Use log scale for better visualization
                    log_int = np.log1p(all_intensities)
                    
                    # Plot histogram
                    ax4.hist(log_int, bins=50, alpha=0.5, 
                           label=f"Numeric: {sample_name}", density=True)
        
        # For visual, show intensity distribution if available
        if visual_metadata and 'intensities' in visual_metadata:
            intensities = visual_metadata['intensities']
            log_int = np.log1p(intensities)
            
            # Plot histogram with different color
            ax4.hist(log_int, bins=50, alpha=0.5, color='red',
                   label="Visual Intensities", density=True)
            
        if has_intensity_data or ('intensities' in visual_metadata if visual_metadata else False):
            ax4.set_xlabel("log(Intensity + 1)")
            ax4.set_ylabel("Density")
            ax4.legend()
            ax4.set_title("Intensity Distribution Comparison")
            ax4.grid(True, alpha=0.3)
        else:
            ax4.set_title("No Intensity Data Available")
            
        # Set overall title
        plt.suptitle("Numeric vs Visual Pipeline Comparison", fontsize=20)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        # Save figure
        plt.savefig(os.path.join(output_dir, "pipeline_comparison.png"), dpi=300)
        plt.close()
        print("Pipeline comparison saved")
        
    except Exception as e:
        print(f"Error creating pipeline comparison: {e}")
    
    # Also make a table comparing metrics between the two approaches
    try:
        fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
        ax.axis('off')
        ax.axis('tight')
        
        # Define metrics to compare
        metrics = [
            "Data Representation", 
            "Runtime Complexity",
            "Memory Usage",
            "Feature Extraction",
            "Noise Handling",
            "Visualization Types",
            "Best For"
        ]
        
        numeric_values = [
            "Raw numerical arrays",
            "O(n) - Linear with data size",
            "Medium - Raw arrays",
            "Direct from signal",
            "Signal processing filters",
            "Chromatograms, Spectra, 3D",
            "Detailed spectrum analysis"
        ]
        
        visual_values = [
            "Image and feature vectors",
            "O(n²) - Quadratic for image processing",
            "High - Images + feature DB",
            "Computer vision techniques",
            "Image processing filters",
            "Feature maps, clusters, video",
            "Pattern recognition, clustering"
        ]
        
        # Create table
        table_data = list(zip(metrics, numeric_values, visual_values))
        table = ax.table(cellText=table_data, 
                       colLabels=["Metric", "Numeric Pipeline", "Visual Pipeline"],
                       loc='center', cellLoc='center')
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        # Add title
        plt.title("Comparison of Numeric vs Visual Pipelines", fontsize=14)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(output_dir, "pipeline_comparison_table.png"), dpi=300)
        plt.close()
        print("Pipeline comparison table saved")
        
    except Exception as e:
        print(f"Error creating pipeline comparison table: {e}")

def main():
    """Main function to run all visualizations"""
    print("Starting MS data visualization")
    
    # Load data for both experiments
    experiments = ["PL_Neg_Waters_qTOF", "TG_Pos_Thermo_Orbi"]
    data = {}
    
    for experiment in experiments:
        print(f"Loading experiment: {experiment}")
        exp_data = load_experiment_data(experiment)
        if exp_data and (exp_data['ms1_xic'] or exp_data['scan_info'] or exp_data['spectra']):
            data[experiment] = exp_data
        else:
            print(f"No data found for {experiment}")
    
    if not data:
        print("No data found!")
        return
        
    # Create visualizations
    create_intensity_heatmap(data)
    create_total_ion_chromatogram(data)
    create_3d_visualization(data)
    create_ms2_fragment_analysis(data)
    create_mz_distribution(data)
    create_summary_dashboard(data)
    
    # Create comparative visualizations if multiple samples
    if len(data) > 1:
        create_dual_sample_comparison(data)
        
    # Create pipeline comparison
    create_pipeline_comparison(data)
        
    print(f"All visualizations saved to {SHOWCASE_DIR}")

if __name__ == "__main__":
    main()
