#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import zarr
from pathlib import Path
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")

# Create output directory for visualizations
os.makedirs("public/showcase", exist_ok=True)

def generate_mock_ms_data(n_points=1000):
    """Generate mock mass spectrometry data if real data cannot be loaded"""
    mz = np.linspace(100, 1000, n_points)
    # Create peaks at random locations
    intensities = np.zeros_like(mz)
    peak_positions = np.random.choice(range(n_points), size=15)
    for pos in peak_positions:
        # Create a Gaussian peak
        peak_height = np.random.uniform(0.3, 1.0)
        peak_width = np.random.uniform(1, 5)
        intensities += peak_height * np.exp(-(mz - mz[pos])**2 / (2 * peak_width**2))
    
    # Add noise
    intensities += np.random.normal(0, 0.01, size=n_points)
    intensities = np.maximum(intensities, 0)  # No negative intensities
    return mz, intensities

def generate_chromatogram_data(n_points=500):
    """Generate mock chromatogram data"""
    time = np.linspace(0, 20, n_points)  # 0-20 minutes
    intensity = np.zeros_like(time)
    
    # Create several peaks
    peak_times = [3, 7, 9, 12, 15, 18]
    peak_heights = [0.8, 1.0, 0.6, 0.9, 0.5, 0.7]
    peak_widths = [0.3, 0.4, 0.2, 0.5, 0.3, 0.4]
    
    for t, h, w in zip(peak_times, peak_heights, peak_widths):
        intensity += h * np.exp(-(time - t)**2 / (2 * w**2))
    
    # Add noise
    intensity += np.random.normal(0, 0.01, size=n_points)
    intensity = np.maximum(intensity, 0)
    
    return time, intensity

def plot_mass_spectrum():
    """Plot a single mass spectrum"""
    mz, intensity = generate_mock_ms_data()
    
    plt.figure(figsize=(12, 6))
    plt.plot(mz, intensity, 'k-', linewidth=1)
    
    # Add peak annotations
    peak_indices = np.where((intensity > 0.2) & (np.r_[True, intensity[1:] > intensity[:-1]] & 
                                               np.r_[intensity[:-1] > intensity[1:], True]))[0]
    
    for idx in peak_indices:
        if intensity[idx] > 0.3:  # Only label more significant peaks
            plt.annotate(f'{mz[idx]:.1f}', 
                        xy=(mz[idx], intensity[idx]),
                        xytext=(0, 10),
                        textcoords='offset points',
                        ha='center', va='bottom',
                        fontsize=8)
    
    plt.xlabel('m/z')
    plt.ylabel('Relative Intensity')
    plt.title('MS1 Spectrum for TG Lipid Analysis')
    plt.xlim(min(mz), max(mz))
    plt.ylim(0, max(intensity) * 1.1)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('public/showcase/ms1_spectrum.png', dpi=300)
    plt.close()

def plot_chromatogram():
    """Plot a TIC chromatogram"""
    time, intensity = generate_chromatogram_data()
    
    plt.figure(figsize=(12, 6))
    plt.plot(time, intensity, 'k-', linewidth=1.5)
    
    # Annotate major peaks
    peak_indices = np.where((intensity > 0.3) & (np.r_[True, intensity[1:] > intensity[:-1]] & 
                                               np.r_[intensity[:-1] > intensity[1:], True]))[0]
    
    for idx in peak_indices:
        plt.annotate(f'{time[idx]:.1f} min', 
                    xy=(time[idx], intensity[idx]),
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=9)
    
    plt.xlabel('Retention Time (min)')
    plt.ylabel('Intensity')
    plt.title('Total Ion Chromatogram')
    plt.xlim(0, max(time))
    plt.ylim(0, max(intensity) * 1.1)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('public/showcase/chromatogram.png', dpi=300)
    plt.close()

def plot_3d_surface():
    """Create a 3D visualization of MS data (retention time vs m/z vs intensity)"""
    # Create a grid of data
    rt = np.linspace(0, 15, 100)  # Retention time from 0 to 15 min
    mz = np.linspace(400, 900, 100)  # m/z range from 400 to 900
    
    # Create meshgrid
    RT, MZ = np.meshgrid(rt, mz)
    
    # Create intensity distribution - multiple Gaussian peaks
    intensity = np.zeros_like(RT)
    
    # Add several peaks at different RT and m/z values
    peaks = [
        (5, 600, 1.0, 0.5, 10),    # RT, m/z, height, RT width, m/z width
        (8, 700, 0.8, 0.3, 15),
        (10, 550, 0.9, 0.6, 8),
        (7, 820, 0.7, 0.4, 12),
        (12, 650, 0.6, 0.7, 9)
    ]
    
    for rt_pos, mz_pos, height, rt_width, mz_width in peaks:
        rt_idx = np.argmin(np.abs(rt - rt_pos))
        mz_idx = np.argmin(np.abs(mz - mz_pos))
        
        for i in range(len(rt)):
            for j in range(len(mz)):
                # 2D Gaussian
                intensity[j, i] += height * np.exp(
                    -((rt[i] - rt_pos) ** 2) / (2 * rt_width ** 2) 
                    -((mz[j] - mz_pos) ** 2) / (2 * mz_width ** 2)
                )
    
    # Add noise
    intensity += np.random.normal(0, 0.01, intensity.shape)
    intensity = np.maximum(intensity, 0)  # No negative intensities
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create the surface plot
    surf = ax.plot_surface(RT, MZ, intensity, cmap='viridis', 
                          linewidth=0, antialiased=True, alpha=0.8)
    
    # Add colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label('Intensity')
    
    # Set labels and title
    ax.set_xlabel('Retention Time (min)')
    ax.set_ylabel('m/z')
    ax.set_zlabel('Intensity')
    ax.set_title('3D LC-MS Data Visualization')
    
    # Adjust the viewing angle for better perspective
    ax.view_init(elev=30, azim=-60)
    
    plt.tight_layout()
    plt.savefig('public/showcase/3d_lcms_visualization.png', dpi=300)
    plt.close()

def plot_heatmap():
    """Create a heatmap visualization of LC-MS data"""
    # Create a grid of data (similar to 3D plot but presented as heatmap)
    rt = np.linspace(0, 15, 200)  # Retention time from 0 to 15 min
    mz = np.linspace(400, 900, 200)  # m/z range from 400 to 900
    
    # Create intensity distribution - multiple Gaussian peaks
    intensity = np.zeros((len(mz), len(rt)))
    
    # Add several peaks at different RT and m/z values
    peaks = [
        (5, 600, 1.0, 0.5, 10),    # RT, m/z, height, RT width, m/z width
        (8, 700, 0.8, 0.3, 15),
        (10, 550, 0.9, 0.6, 8),
        (7, 820, 0.7, 0.4, 12),
        (12, 650, 0.6, 0.7, 9),
        (4, 500, 0.5, 0.3, 7),
        (9, 780, 0.7, 0.5, 11)
    ]
    
    for rt_pos, mz_pos, height, rt_width, mz_width in peaks:
        for i in range(len(rt)):
            for j in range(len(mz)):
                # 2D Gaussian
                intensity[j, i] += height * np.exp(
                    -((rt[i] - rt_pos) ** 2) / (2 * rt_width ** 2) 
                    -((mz[j] - mz_pos) ** 2) / (2 * mz_width ** 2)
                )
    
    # Create the heatmap plot
    plt.figure(figsize=(14, 8))
    
    # Create the heatmap
    plt.pcolormesh(rt, mz, intensity, cmap='viridis', shading='auto')
    
    # Add colorbar
    cbar = plt.colorbar()
    cbar.set_label('Intensity')
    
    # Set labels and title
    plt.xlabel('Retention Time (min)')
    plt.ylabel('m/z')
    plt.title('LC-MS Data Heatmap Visualization')
    
    # Annotate major peaks
    for rt_pos, mz_pos, height, _, _ in peaks:
        if height > 0.6:  # Only annotate more significant peaks
            plt.annotate(f'm/z: {mz_pos}',
                       xy=(rt_pos, mz_pos),
                       xytext=(10, 10),
                       textcoords='offset points',
                       color='white',
                       backgroundcolor='black',
                       alpha=0.7,
                       fontsize=9)
    
    plt.tight_layout()
    plt.savefig('public/showcase/lcms_heatmap.png', dpi=300)
    plt.close()

def plot_xic_comparison():
    """Plot extracted ion chromatograms for comparing compounds"""
    # Create retention time array
    rt = np.linspace(0, 15, 500)
    
    # Generate chromatograms for three different compounds
    compounds = {
        'TG(50:1)': {
            'mz': 834.7,
            'rt_peak': 8.5,
            'intensity': 0.9,
            'width': 0.4,
            'color': 'blue'
        },
        'TG(52:2)': {
            'mz': 860.8,
            'rt_peak': 10.2,
            'intensity': 1.0,
            'width': 0.35,
            'color': 'red'
        },
        'TG(48:0)': {
            'mz': 808.7,
            'rt_peak': 7.3,
            'intensity': 0.65,
            'width': 0.45,
            'color': 'green'
        }
    }
    
    plt.figure(figsize=(12, 7))
    
    for name, params in compounds.items():
        # Generate the XIC
        intensity = params['intensity'] * np.exp(-(rt - params['rt_peak'])**2 / (2 * params['width']**2))
        intensity += np.random.normal(0, 0.005, size=len(rt))  # Add noise
        intensity = np.maximum(intensity, 0)  # No negative values
        
        # Plot
        plt.plot(rt, intensity, color=params['color'], label=f"{name} (m/z {params['mz']})")
        
        # Add peak annotation
        max_idx = np.argmax(intensity)
        plt.annotate(f"{params['rt_peak']} min",
                   xy=(rt[max_idx], intensity[max_idx]),
                   xytext=(0, 10),
                   textcoords='offset points',
                   ha='center',
                   fontsize=9)
    
    plt.xlabel('Retention Time (min)')
    plt.ylabel('Intensity')
    plt.title('Extracted Ion Chromatograms')
    plt.legend(loc='upper right')
    plt.xlim(5, 12)  # Focus on the region with peaks
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('public/showcase/xic_comparison.png', dpi=300)
    plt.close()

def plot_ms_ms_spectrum():
    """Plot MS/MS spectrum with fragment annotations"""
    mz, intensities = generate_mock_ms_data(n_points=500)
    
    # Set the precursor m/z
    precursor_mz = 760.5
    
    # Create fragment peaks
    fragments = [
        {'mz': 184.1, 'intensity': 0.95, 'annotation': 'Phosphocholine'},
        {'mz': 500.3, 'intensity': 0.78, 'annotation': 'LPC fragment'},
        {'mz': 577.5, 'intensity': 0.65, 'annotation': 'FA loss'},
        {'mz': 476.2, 'intensity': 0.45, 'annotation': 'Neutral loss (DG)'},
    ]
    
    plt.figure(figsize=(12, 6))
    
    # Plot the base spectrum
    plt.stem(mz, intensities, 'k-', basefmt=' ', markerfmt=' ', use_line_collection=True)
    
    # Highlight and annotate key fragments
    for fragment in fragments:
        idx = np.argmin(np.abs(mz - fragment['mz']))
        intensities[idx] = fragment['intensity']  # Override with fragment intensity
        
        plt.annotate(f"{fragment['mz']:.1f}\n{fragment['annotation']}",
                   xy=(mz[idx], intensities[idx]),
                   xytext=(0, 10),
                   textcoords='offset points',
                   ha='center',
                   fontsize=8,
                   bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.5))
    
    # Add title and precursor info
    plt.title(f'MS/MS Spectrum (Precursor m/z: {precursor_mz})')
    plt.xlabel('m/z')
    plt.ylabel('Relative Intensity')
    
    # Add precursor info box
    plt.figtext(0.15, 0.85, f'Precursor: m/z {precursor_mz}\nCollision Energy: 30 eV',
               bbox=dict(boxstyle="round,pad=0.5", fc="lightblue", alpha=0.8))
    
    plt.xlim(min(mz), max(mz))
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('public/showcase/msms_spectrum.png', dpi=300)
    plt.close()

def plot_multiclass_pca():
    """Create a PCA plot of samples from different classes"""
    # Generate random data for PCA visualization
    np.random.seed(42)
    
    # Create three sample groups
    n_samples = 20
    
    # Generate random data for each group with different centers
    group1 = np.random.randn(n_samples, 2) * 0.5 + np.array([2, 2])
    group2 = np.random.randn(n_samples, 2) * 0.4 + np.array([-2, 2])
    group3 = np.random.randn(n_samples, 2) * 0.6 + np.array([0, -2])
    
    # Combine data
    X = np.vstack([group1, group2, group3])
    
    # Create labels
    y = np.hstack([np.zeros(n_samples), np.ones(n_samples), np.ones(n_samples) * 2])
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    # Define group information
    groups = {
        0: {'name': 'Control', 'color': 'blue', 'marker': 'o'},
        1: {'name': 'Diseased', 'color': 'red', 'marker': 's'},
        2: {'name': 'Treated', 'color': 'green', 'marker': '^'}
    }
    
    # Plot each group
    for group_id, group_info in groups.items():
        mask = y == group_id
        plt.scatter(X[mask, 0], X[mask, 1], 
                   c=group_info['color'], 
                   marker=group_info['marker'], 
                   label=group_info['name'],
                   s=80, alpha=0.7, edgecolors='k')
    
    # Add confidence ellipses
    from matplotlib.patches import Ellipse
    
    for group_id, group_info in groups.items():
        mask = y == group_id
        x = X[mask, 0]
        y_vals = X[mask, 1]
        
        # Calculate the eigenvectors and eigenvalues
        cov = np.cov(x, y_vals)
        evals, evecs = np.linalg.eigh(cov)
        
        # Sort eigenvalues in decreasing order
        idx = np.argsort(evals)[::-1]
        evals = evals[idx]
        evecs = evecs[:, idx]
        
        # Compute angle and width/height
        angle = np.arctan2(evecs[1, 0], evecs[0, 0]) * 180 / np.pi
        width, height = 2 * np.sqrt(evals) * 2  # 2 standard deviations
        
        # Draw the ellipse
        ellipse = Ellipse(xy=(np.mean(x), np.mean(y_vals)),
                         width=width, height=height,
                         angle=angle, 
                         color=group_info['color'], alpha=0.2)
        plt.gca().add_patch(ellipse)
    
    plt.xlabel('Principal Component 1 (43%)')
    plt.ylabel('Principal Component 2 (28%)')
    plt.title('PCA of LC-MS Metabolomic Profiles')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add annotations
    for group_id, group_info in groups.items():
        mask = y == group_id
        center_x = np.mean(X[mask, 0])
        center_y = np.mean(X[mask, 1])
        
        plt.annotate(group_info['name'],
                   xy=(center_x, center_y),
                   xytext=(20, 20),
                   textcoords='offset points',
                   ha='center',
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
                   arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    
    plt.tight_layout()
    plt.savefig('public/showcase/metabolomics_pca.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    # Generate all visualizations
    print("Generating mass spectrum plot...")
    plot_mass_spectrum()
    
    print("Generating chromatogram plot...")
    plot_chromatogram()
    
    print("Generating 3D surface plot...")
    plot_3d_surface()
    
    print("Generating LC-MS heatmap...")
    plot_heatmap()
    
    print("Generating XIC comparison...")
    plot_xic_comparison()
    
    print("Generating MS/MS spectrum...")
    plot_ms_ms_spectrum()
    
    print("Generating PCA plot...")
    plot_multiclass_pca()
    
    print("All visualizations generated in public/showcase/") 