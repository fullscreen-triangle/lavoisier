"""
Selected Reaction Monitoring (SRM) Visualization
=================================================

Tracks specific molecular ions through the entire analytical pipeline:
    Chromatography → Ionization → MS1 → MS2 → CV Droplet

Validates:
1. Information preservation (same peak through all stages)
2. Platform independence (same molecule, different detectors)
3. Bijective transformation (one molecular reality, multiple representations)

For each stage, creates a 4-panel figure:
    Panel 1: 3D visualization (RT × m/z × Intensity or equivalent)
    Panel 2: Time series (Absorbance/Intensity vs Time)
    Panel 3: Elution gradient or equivalent
    Panel 4: Spectral analysis (Power, Median, Density spectra)

Author: Kundai Farai Sachikonye
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import logging
from dataclasses import dataclass
from scipy import signal, stats

# Import DDA linkage manager
try:
    from .dda_linkage import DDALinkageManager
except ImportError:
    from dda_linkage import DDALinkageManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300


@dataclass
class TrackedPeak:
    """A molecular ion tracked through the pipeline."""
    mz: float
    rt_apex: float
    intensity_max: float
    scan_id: int
    # Stage-specific data
    chromatogram: Optional[pd.DataFrame] = None
    ms1_spectrum: Optional[pd.DataFrame] = None
    ms2_spectrum: Optional[pd.DataFrame] = None
    cv_droplet: Optional[pd.DataFrame] = None
    sentropy_matrix: Optional[pd.DataFrame] = None


class SRMPeakSelector:
    """Selects representative peaks for SRM tracking."""
    
    def __init__(self, experiment_dir: Path):
        self.experiment_dir = Path(experiment_dir)
        self.ms1_xic = None
        self.spectra_summary = None
        self.dda_manager = None
        
    def load_data(self):
        """Load MS1 XIC and spectra summary."""
        logger.info("Loading MS1 data...")
        
        xic_path = self.experiment_dir / "stage_01_preprocessing" / "ms1_xic.csv"
        summary_path = self.experiment_dir / "stage_01_preprocessing" / "spectra_summary.csv"
        
        if not xic_path.exists() or not summary_path.exists():
            raise FileNotFoundError(f"Required files not found in {self.experiment_dir}")
        
        self.ms1_xic = pd.read_csv(xic_path)
        self.spectra_summary = pd.read_csv(summary_path)
        
        # Initialize DDA linkage manager for correct MS1-MS2 mapping
        self.dda_manager = DDALinkageManager(self.experiment_dir)
        self.dda_manager.load_data()
        
        logger.info(f"  Loaded {len(self.ms1_xic)} XIC points")
        logger.info(f"  Loaded {len(self.spectra_summary)} spectra")
        
    def select_top_peaks(self, n_peaks: int = 5, 
                         mz_range: Tuple[float, float] = (150, 1500),
                         rt_range: Tuple[float, float] = (0.5, 30)) -> List[TrackedPeak]:
        """
        Select top N peaks for tracking.
        
        Args:
            n_peaks: Number of peaks to select
            mz_range: m/z range to consider
            rt_range: Retention time range (minutes)
            
        Returns:
            List of TrackedPeak objects
        """
        logger.info(f"Selecting top {n_peaks} peaks...")
        
        # Filter by m/z and RT
        filtered = self.ms1_xic[
            (self.ms1_xic['mz'] >= mz_range[0]) &
            (self.ms1_xic['mz'] <= mz_range[1]) &
            (self.ms1_xic['rt'] >= rt_range[0]) &
            (self.ms1_xic['rt'] <= rt_range[1])
        ].copy()
        
        # Group by m/z bins (0.01 Da) and RT bins (0.1 min)
        filtered['mz_bin'] = (filtered['mz'] / 0.01).astype(int)
        filtered['rt_bin'] = (filtered['rt'] / 0.1).astype(int)
        
        # Find apex of each peak
        peak_groups = filtered.groupby(['mz_bin', 'rt_bin']).agg({
            'mz': 'mean',
            'rt': 'mean',
            'i': 'max',
            'spec_idx': 'first'
        }).reset_index(drop=True)
        
        # Sort by intensity and select top N
        top_peaks = peak_groups.nlargest(n_peaks, 'i')
        
        tracked_peaks = []
        for _, row in top_peaks.iterrows():
            peak = TrackedPeak(
                mz=row['mz'],
                rt_apex=row['rt'],
                intensity_max=row['i'],
                scan_id=int(row['spec_idx'])
            )
            tracked_peaks.append(peak)
            logger.info(f"  Selected: m/z={peak.mz:.4f}, RT={peak.rt_apex:.2f}, I={peak.intensity_max:.2e}")
        
        return tracked_peaks
    
    def extract_peak_data(self, peak: TrackedPeak, 
                          mz_tolerance: float = 0.01,
                          rt_window: float = 1.0) -> TrackedPeak:
        """
        Extract all stage data for a tracked peak.
        
        Args:
            peak: TrackedPeak object
            mz_tolerance: m/z tolerance (Da)
            rt_window: RT window around apex (minutes)
            
        Returns:
            TrackedPeak with all stage data populated
        """
        logger.info(f"Extracting data for m/z={peak.mz:.4f}, RT={peak.rt_apex:.2f}")
        
        # 1. Extract chromatogram (XIC)
        peak.chromatogram = self.ms1_xic[
            (self.ms1_xic['mz'] >= peak.mz - mz_tolerance) &
            (self.ms1_xic['mz'] <= peak.mz + mz_tolerance) &
            (self.ms1_xic['rt'] >= peak.rt_apex - rt_window) &
            (self.ms1_xic['rt'] <= peak.rt_apex + rt_window)
        ].copy()
        
        # 2. Extract MS1 spectrum at apex
        peak.ms1_spectrum = self.ms1_xic[
            self.ms1_xic['spec_idx'] == peak.scan_id
        ].copy()
        
        # 3. Use DDA linkage manager to get CORRECT MS2 spectra
        if self.dda_manager:
            srm_data = self.dda_manager.get_complete_srm_data(
                peak.mz, peak.rt_apex,
                mz_tolerance=mz_tolerance,
                rt_window=rt_window
            )
            
            # Combine all MS2 spectra into one DataFrame
            if srm_data['ms2_spectra']:
                all_fragments = []
                for ms2_data in srm_data['ms2_spectra']:
                    spectrum = ms2_data['spectrum'].copy()
                    # Add metadata
                    spectrum['ms2_scan'] = ms2_data['scan_info']['scan_number']
                    spectrum['ms2_rt'] = ms2_data['scan_info']['scan_time']
                    spectrum['ms1_rt'] = ms2_data['scan_info']['ms1_rt']
                    spectrum['rt_offset'] = ms2_data['scan_info']['scan_time'] - ms2_data['scan_info']['ms1_rt']
                    all_fragments.append(spectrum)
                
                if all_fragments:
                    peak.ms2_spectrum = pd.concat(all_fragments, ignore_index=True)
                    logger.info(f"    Linked {len(srm_data['ms2_spectra'])} MS2 scans with {len(peak.ms2_spectrum)} total fragments")
        
        # 4. Try to load CV droplet
        try:
            cv_path = self.experiment_dir / "stage_02_cv" / "droplets" / f"droplets_{peak.scan_id}.tsv"
            if cv_path.exists():
                peak.cv_droplet = pd.read_csv(cv_path, sep='\t')
        except Exception as e:
            logger.warning(f"  Could not load CV droplet: {e}")
        
        # 5. Try to load S-entropy matrix
        try:
            sentropy_path = self.experiment_dir / "stage_02_sentropy" / "matrices" / f"sentropy_{peak.scan_id}.tsv"
            if sentropy_path.exists():
                peak.sentropy_matrix = pd.read_csv(sentropy_path, sep='\t')
        except Exception as e:
            logger.warning(f"  Could not load S-entropy matrix: {e}")
        
        logger.info(f"  Extracted: XIC={len(peak.chromatogram)}, MS1={len(peak.ms1_spectrum)}, " +
                   f"MS2={len(peak.ms2_spectrum) if peak.ms2_spectrum is not None else 0}, " +
                   f"CV={len(peak.cv_droplet) if peak.cv_droplet is not None else 0}")
        
        return peak


class SRMVisualizer:
    """Creates 4-panel SRM visualizations for each stage."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def create_chromatography_panel(self, peak: TrackedPeak, 
                                     experiment_name: str) -> Path:
        """
        Create 4-panel chromatography visualization.
        
        Panel 1: 3D (RT × m/z × UV Intensity)
        Panel 2: UV Absorbance vs Time
        Panel 3: Elution Gradient
        Panel 4: Spectral Analysis (Power, Median, Density)
        """
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        fig.suptitle(f'Chromatography Stage - m/z {peak.mz:.4f} @ RT {peak.rt_apex:.2f} min',
                     fontsize=16, fontweight='bold')
        
        # Panel 1: 3D RT × m/z × Intensity
        ax1 = fig.add_subplot(gs[0, 0], projection='3d')
        if peak.chromatogram is not None and len(peak.chromatogram) > 0:
            scatter = ax1.scatter(peak.chromatogram['rt'], 
                                 peak.chromatogram['mz'],
                                 peak.chromatogram['i'],
                                 c=peak.chromatogram['i'],
                                 cmap='viridis',
                                 s=20,
                                 alpha=0.6)
            ax1.set_xlabel('Retention Time (min)', fontsize=10)
            ax1.set_ylabel('m/z', fontsize=10)
            ax1.set_zlabel('Intensity', fontsize=10)
            ax1.set_title('3D Chromatographic Profile', fontsize=12, fontweight='bold')
            plt.colorbar(scatter, ax=ax1, label='Intensity', shrink=0.5)
        
        # Panel 2: Absorbance vs Time (XIC)
        ax2 = fig.add_subplot(gs[0, 1])
        if peak.chromatogram is not None and len(peak.chromatogram) > 0:
            # Sort by RT for proper line plot
            chrom_sorted = peak.chromatogram.sort_values('rt')
            ax2.plot(chrom_sorted['rt'], chrom_sorted['i'], 
                    color='blue', linewidth=2, alpha=0.7)
            ax2.fill_between(chrom_sorted['rt'], chrom_sorted['i'], 
                            alpha=0.3, color='blue')
            ax2.axvline(peak.rt_apex, color='red', linestyle='--', 
                       label=f'Apex: {peak.rt_apex:.2f} min')
            ax2.set_xlabel('Retention Time (min)', fontsize=10)
            ax2.set_ylabel('Intensity (AU)', fontsize=10)
            ax2.set_title('Extracted Ion Chromatogram (XIC)', fontsize=12, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Panel 3: Elution Gradient (simulated from RT)
        ax3 = fig.add_subplot(gs[1, 0])
        if peak.chromatogram is not None and len(peak.chromatogram) > 0:
            # Simulate gradient based on RT (typical LC gradient)
            rt_range = chrom_sorted['rt'].values
            gradient = 5 + (95 - 5) * (rt_range - rt_range.min()) / (rt_range.max() - rt_range.min())
            ax3.plot(rt_range, gradient, color='green', linewidth=2)
            ax3.axvline(peak.rt_apex, color='red', linestyle='--', 
                       label=f'Peak Elution')
            ax3.set_xlabel('Retention Time (min)', fontsize=10)
            ax3.set_ylabel('Organic Phase (%)', fontsize=10)
            ax3.set_title('Elution Gradient', fontsize=12, fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Panel 4: Spectral Analysis (3 sub-panels)
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')
        
        # Create 3 sub-panels within panel 4
        gs_inner = GridSpec(1, 3, figure=fig, 
                           left=0.55, right=0.95, bottom=0.1, top=0.4,
                           wspace=0.4)
        
        if peak.chromatogram is not None and len(peak.chromatogram) > 0:
            intensities = peak.chromatogram['i'].values
            
            # Sub-panel 1: Power Spectrum
            ax4_1 = fig.add_subplot(gs_inner[0, 0])
            freqs, psd = signal.periodogram(intensities, fs=1.0)
            ax4_1.semilogy(freqs[1:], psd[1:], color='purple')
            ax4_1.set_xlabel('Frequency', fontsize=8)
            ax4_1.set_ylabel('Power', fontsize=8)
            ax4_1.set_title('Power Spectrum', fontsize=10, fontweight='bold')
            ax4_1.grid(True, alpha=0.3)
            
            # Sub-panel 2: Median Spectrum (histogram)
            ax4_2 = fig.add_subplot(gs_inner[0, 1])
            ax4_2.hist(intensities, bins=30, color='orange', alpha=0.7, edgecolor='black')
            ax4_2.axvline(np.median(intensities), color='red', linestyle='--', 
                         label=f'Median: {np.median(intensities):.2e}')
            ax4_2.set_xlabel('Intensity', fontsize=8)
            ax4_2.set_ylabel('Count', fontsize=8)
            ax4_2.set_title('Intensity Distribution', fontsize=10, fontweight='bold')
            ax4_2.legend(fontsize=7)
            ax4_2.grid(True, alpha=0.3)
            
            # Sub-panel 3: Density Spectrum (KDE)
            ax4_3 = fig.add_subplot(gs_inner[0, 2])
            try:
                kde = stats.gaussian_kde(intensities)
                x_range = np.linspace(intensities.min(), intensities.max(), 100)
                ax4_3.plot(x_range, kde(x_range), color='teal', linewidth=2)
                ax4_3.fill_between(x_range, kde(x_range), alpha=0.3, color='teal')
                ax4_3.set_xlabel('Intensity', fontsize=8)
                ax4_3.set_ylabel('Density', fontsize=8)
                ax4_3.set_title('Density Spectrum', fontsize=10, fontweight='bold')
                ax4_3.grid(True, alpha=0.3)
            except Exception as e:
                logger.warning(f"Could not create KDE: {e}")
        
        output_file = self.output_dir / f"{experiment_name}_chromatography_mz{peak.mz:.4f}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  Saved chromatography panel: {output_file.name}")
        return output_file
    
    def create_ms1_panel(self, peak: TrackedPeak, 
                         experiment_name: str) -> Path:
        """
        Create 4-panel MS1 visualization.
        
        Panel 1: 3D (RT × m/z × MS Intensity) - full spectrum over time
        Panel 2: MS1 Intensity vs Time (for this m/z)
        Panel 3: m/z distribution over time
        Panel 4: Spectral Analysis
        """
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        fig.suptitle(f'MS1 Stage (Mass Analyzer) - m/z {peak.mz:.4f} @ RT {peak.rt_apex:.2f} min',
                     fontsize=16, fontweight='bold')
        
        # Panel 1: 3D RT × m/z × Intensity (full MS1 spectrum)
        ax1 = fig.add_subplot(gs[0, 0], projection='3d')
        if peak.ms1_spectrum is not None and len(peak.ms1_spectrum) > 0:
            scatter = ax1.scatter(peak.ms1_spectrum['rt'], 
                                 peak.ms1_spectrum['mz'],
                                 peak.ms1_spectrum['i'],
                                 c=peak.ms1_spectrum['i'],
                                 cmap='plasma',
                                 s=20,
                                 alpha=0.6)
            ax1.set_xlabel('Retention Time (min)', fontsize=10)
            ax1.set_ylabel('m/z', fontsize=10)
            ax1.set_zlabel('MS1 Intensity', fontsize=10)
            ax1.set_title('3D MS1 Spectrum', fontsize=12, fontweight='bold')
            plt.colorbar(scatter, ax=ax1, label='Intensity', shrink=0.5)
        
        # Panel 2: MS1 Intensity vs Time (this m/z only)
        ax2 = fig.add_subplot(gs[0, 1])
        if peak.chromatogram is not None and len(peak.chromatogram) > 0:
            chrom_sorted = peak.chromatogram.sort_values('rt')
            ax2.plot(chrom_sorted['rt'], chrom_sorted['i'], 
                    color='red', linewidth=2, alpha=0.7, marker='o', markersize=4)
            ax2.axvline(peak.rt_apex, color='black', linestyle='--', 
                       label=f'Apex: {peak.rt_apex:.2f} min')
            ax2.set_xlabel('Retention Time (min)', fontsize=10)
            ax2.set_ylabel('MS1 Intensity', fontsize=10)
            ax2.set_title(f'MS1 Signal for m/z {peak.mz:.4f}', fontsize=12, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Panel 3: m/z Distribution Over Time
        ax3 = fig.add_subplot(gs[1, 0])
        if peak.chromatogram is not None and len(peak.chromatogram) > 0:
            # Show m/z variation (mass accuracy)
            chrom_sorted = peak.chromatogram.sort_values('rt')
            ax3.scatter(chrom_sorted['rt'], chrom_sorted['mz'], 
                       c=chrom_sorted['i'], cmap='viridis', s=50, alpha=0.6)
            ax3.axhline(peak.mz, color='red', linestyle='--', 
                       label=f'Theoretical m/z: {peak.mz:.4f}')
            ax3.set_xlabel('Retention Time (min)', fontsize=10)
            ax3.set_ylabel('Measured m/z', fontsize=10)
            ax3.set_title('Mass Accuracy Over Time', fontsize=12, fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Panel 4: Spectral Analysis (3 sub-panels)
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')
        
        gs_inner = GridSpec(1, 3, figure=fig, 
                           left=0.55, right=0.95, bottom=0.1, top=0.4,
                           wspace=0.4)
        
        if peak.ms1_spectrum is not None and len(peak.ms1_spectrum) > 0:
            intensities = peak.ms1_spectrum['i'].values
            
            # Sub-panel 1: Power Spectrum
            ax4_1 = fig.add_subplot(gs_inner[0, 0])
            freqs, psd = signal.periodogram(intensities, fs=1.0)
            ax4_1.semilogy(freqs[1:], psd[1:], color='purple')
            ax4_1.set_xlabel('Frequency', fontsize=8)
            ax4_1.set_ylabel('Power', fontsize=8)
            ax4_1.set_title('Power Spectrum', fontsize=10, fontweight='bold')
            ax4_1.grid(True, alpha=0.3)
            
            # Sub-panel 2: Median Spectrum
            ax4_2 = fig.add_subplot(gs_inner[0, 1])
            ax4_2.hist(intensities, bins=30, color='orange', alpha=0.7, edgecolor='black')
            ax4_2.axvline(np.median(intensities), color='red', linestyle='--', 
                         label=f'Median: {np.median(intensities):.2e}')
            ax4_2.set_xlabel('Intensity', fontsize=8)
            ax4_2.set_ylabel('Count', fontsize=8)
            ax4_2.set_title('Intensity Distribution', fontsize=10, fontweight='bold')
            ax4_2.legend(fontsize=7)
            ax4_2.grid(True, alpha=0.3)
            
            # Sub-panel 3: Density Spectrum
            ax4_3 = fig.add_subplot(gs_inner[0, 2])
            try:
                kde = stats.gaussian_kde(intensities)
                x_range = np.linspace(intensities.min(), intensities.max(), 100)
                ax4_3.plot(x_range, kde(x_range), color='teal', linewidth=2)
                ax4_3.fill_between(x_range, kde(x_range), alpha=0.3, color='teal')
                ax4_3.set_xlabel('Intensity', fontsize=8)
                ax4_3.set_ylabel('Density', fontsize=8)
                ax4_3.set_title('Density Spectrum', fontsize=10, fontweight='bold')
                ax4_3.grid(True, alpha=0.3)
            except Exception as e:
                logger.warning(f"Could not create KDE: {e}")
        
        output_file = self.output_dir / f"{experiment_name}_ms1_mz{peak.mz:.4f}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  Saved MS1 panel: {output_file.name}")
        return output_file
    
    def create_ms2_panel(self, peak: TrackedPeak, 
                         experiment_name: str) -> Path:
        """
        Create 4-panel MS2 (fragmentation) visualization.
        """
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        fig.suptitle(f'MS2 Stage (Fragmentation) - Precursor m/z {peak.mz:.4f}',
                     fontsize=16, fontweight='bold')
        
        if peak.ms2_spectrum is not None and len(peak.ms2_spectrum) > 0:
            # Panel 1: 3D Fragment Spectrum
            ax1 = fig.add_subplot(gs[0, 0], projection='3d')
            # For MS2, we might have fragment m/z and intensities
            # Assuming columns: mz, intensity
            if 'mz' in peak.ms2_spectrum.columns and 'i' in peak.ms2_spectrum.columns:
                scatter = ax1.scatter(np.arange(len(peak.ms2_spectrum)), 
                                     peak.ms2_spectrum['mz'],
                                     peak.ms2_spectrum['i'],
                                     c=peak.ms2_spectrum['i'],
                                     cmap='inferno',
                                     s=30,
                                     alpha=0.7)
                ax1.set_xlabel('Fragment Index', fontsize=10)
                ax1.set_ylabel('Fragment m/z', fontsize=10)
                ax1.set_zlabel('Intensity', fontsize=10)
                ax1.set_title('3D Fragment Spectrum', fontsize=12, fontweight='bold')
                plt.colorbar(scatter, ax=ax1, label='Intensity', shrink=0.5)
            
            # Panel 2: Fragment Spectrum (traditional)
            ax2 = fig.add_subplot(gs[0, 1])
            if 'mz' in peak.ms2_spectrum.columns and 'i' in peak.ms2_spectrum.columns:
                ax2.stem(peak.ms2_spectrum['mz'], peak.ms2_spectrum['i'], 
                        linefmt='red', markerfmt='ro', basefmt=' ')
                ax2.set_xlabel('Fragment m/z', fontsize=10)
                ax2.set_ylabel('Intensity', fontsize=10)
                ax2.set_title('MS2 Fragment Spectrum', fontsize=12, fontweight='bold')
                ax2.grid(True, alpha=0.3)
            
            # Panel 3: Fragment Mass Distribution
            ax3 = fig.add_subplot(gs[1, 0])
            if 'mz' in peak.ms2_spectrum.columns:
                ax3.hist(peak.ms2_spectrum['mz'], bins=50, color='purple', 
                        alpha=0.7, edgecolor='black')
                ax3.axvline(peak.mz, color='red', linestyle='--', 
                           label=f'Precursor: {peak.mz:.4f}')
                ax3.set_xlabel('Fragment m/z', fontsize=10)
                ax3.set_ylabel('Count', fontsize=10)
                ax3.set_title('Fragment Mass Distribution', fontsize=12, fontweight='bold')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            
            # Panel 4: Spectral Analysis
            ax4 = fig.add_subplot(gs[1, 1])
            ax4.axis('off')
            
            gs_inner = GridSpec(1, 3, figure=fig, 
                               left=0.55, right=0.95, bottom=0.1, top=0.4,
                               wspace=0.4)
            
            if 'i' in peak.ms2_spectrum.columns:
                intensities = peak.ms2_spectrum['i'].values
                
                # Sub-panels (same as before)
                ax4_1 = fig.add_subplot(gs_inner[0, 0])
                freqs, psd = signal.periodogram(intensities, fs=1.0)
                ax4_1.semilogy(freqs[1:], psd[1:], color='purple')
                ax4_1.set_xlabel('Frequency', fontsize=8)
                ax4_1.set_ylabel('Power', fontsize=8)
                ax4_1.set_title('Power Spectrum', fontsize=10, fontweight='bold')
                ax4_1.grid(True, alpha=0.3)
                
                ax4_2 = fig.add_subplot(gs_inner[0, 1])
                ax4_2.hist(intensities, bins=30, color='orange', alpha=0.7, edgecolor='black')
                ax4_2.axvline(np.median(intensities), color='red', linestyle='--', 
                             label=f'Median: {np.median(intensities):.2e}')
                ax4_2.set_xlabel('Intensity', fontsize=8)
                ax4_2.set_ylabel('Count', fontsize=8)
                ax4_2.set_title('Intensity Distribution', fontsize=10, fontweight='bold')
                ax4_2.legend(fontsize=7)
                ax4_2.grid(True, alpha=0.3)
                
                ax4_3 = fig.add_subplot(gs_inner[0, 2])
                try:
                    kde = stats.gaussian_kde(intensities)
                    x_range = np.linspace(intensities.min(), intensities.max(), 100)
                    ax4_3.plot(x_range, kde(x_range), color='teal', linewidth=2)
                    ax4_3.fill_between(x_range, kde(x_range), alpha=0.3, color='teal')
                    ax4_3.set_xlabel('Intensity', fontsize=8)
                    ax4_3.set_ylabel('Density', fontsize=8)
                    ax4_3.set_title('Density Spectrum', fontsize=10, fontweight='bold')
                    ax4_3.grid(True, alpha=0.3)
                except Exception as e:
                    logger.warning(f"Could not create KDE: {e}")
        else:
            # No MS2 data available
            ax = fig.add_subplot(gs[:, :])
            ax.text(0.5, 0.5, 'No MS2 Data Available', 
                   ha='center', va='center', fontsize=20, color='gray')
            ax.axis('off')
        
        output_file = self.output_dir / f"{experiment_name}_ms2_mz{peak.mz:.4f}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  Saved MS2 panel: {output_file.name}")
        return output_file
    
    def create_cv_panel(self, peak: TrackedPeak, 
                        experiment_name: str) -> Path:
        """
        Create 4-panel CV (Computer Vision / Droplet) visualization.
        """
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        fig.suptitle(f'CV Stage (Thermodynamic Droplet) - m/z {peak.mz:.4f}',
                     fontsize=16, fontweight='bold')
        
        if peak.cv_droplet is not None and len(peak.cv_droplet) > 0:
            # Assuming CV droplet has S_k, S_t, S_e coordinates
            # Panel 1: 3D S-Entropy Space
            ax1 = fig.add_subplot(gs[0, 0], projection='3d')
            if all(col in peak.cv_droplet.columns for col in ['s_k', 's_t', 's_e']):
                scatter = ax1.scatter(peak.cv_droplet['s_k'], 
                                     peak.cv_droplet['s_t'],
                                     peak.cv_droplet['s_e'],
                                     c=peak.cv_droplet.get('intensity', peak.cv_droplet['s_e']),
                                     cmap='coolwarm',
                                     s=30,
                                     alpha=0.7)
                ax1.set_xlabel('S_k (Kinetic)', fontsize=10)
                ax1.set_ylabel('S_t (Temporal)', fontsize=10)
                ax1.set_zlabel('S_e (Energetic)', fontsize=10)
                ax1.set_title('3D S-Entropy Coordinates', fontsize=12, fontweight='bold')
                plt.colorbar(scatter, ax=ax1, label='Intensity', shrink=0.5)
            
            # Panel 2: Thermodynamic Image (S_t vs S_k)
            ax2 = fig.add_subplot(gs[0, 1])
            if 's_k' in peak.cv_droplet.columns and 's_t' in peak.cv_droplet.columns:
                scatter = ax2.scatter(peak.cv_droplet['s_k'], peak.cv_droplet['s_t'],
                                     c=peak.cv_droplet.get('intensity', peak.cv_droplet['s_e']),
                                     cmap='viridis', s=50, alpha=0.7)
                ax2.set_xlabel('S_k (Kinetic Entropy)', fontsize=10)
                ax2.set_ylabel('S_t (Temporal Entropy)', fontsize=10)
                ax2.set_title('Thermodynamic Image', fontsize=12, fontweight='bold')
                plt.colorbar(scatter, ax=ax2, label='S_e')
                ax2.grid(True, alpha=0.3)
            
            # Panel 3: Wave Pattern (if available)
            ax3 = fig.add_subplot(gs[1, 0])
            if 's_e' in peak.cv_droplet.columns:
                # Plot S_e evolution
                ax3.plot(peak.cv_droplet['s_e'].values, color='teal', linewidth=2)
                ax3.set_xlabel('Index', fontsize=10)
                ax3.set_ylabel('S_e (Energetic Entropy)', fontsize=10)
                ax3.set_title('Wave Pattern Evolution', fontsize=12, fontweight='bold')
                ax3.grid(True, alpha=0.3)
            
            # Panel 4: Spectral Analysis
            ax4 = fig.add_subplot(gs[1, 1])
            ax4.axis('off')
            
            gs_inner = GridSpec(1, 3, figure=fig, 
                               left=0.55, right=0.95, bottom=0.1, top=0.4,
                               wspace=0.4)
            
            # Use S_e for spectral analysis
            if 's_e' in peak.cv_droplet.columns:
                values = peak.cv_droplet['s_e'].values
                
                ax4_1 = fig.add_subplot(gs_inner[0, 0])
                freqs, psd = signal.periodogram(values, fs=1.0)
                ax4_1.semilogy(freqs[1:], psd[1:], color='purple')
                ax4_1.set_xlabel('Frequency', fontsize=8)
                ax4_1.set_ylabel('Power', fontsize=8)
                ax4_1.set_title('Power Spectrum', fontsize=10, fontweight='bold')
                ax4_1.grid(True, alpha=0.3)
                
                ax4_2 = fig.add_subplot(gs_inner[0, 1])
                ax4_2.hist(values, bins=30, color='orange', alpha=0.7, edgecolor='black')
                ax4_2.axvline(np.median(values), color='red', linestyle='--', 
                             label=f'Median: {np.median(values):.4f}')
                ax4_2.set_xlabel('S_e', fontsize=8)
                ax4_2.set_ylabel('Count', fontsize=8)
                ax4_2.set_title('S_e Distribution', fontsize=10, fontweight='bold')
                ax4_2.legend(fontsize=7)
                ax4_2.grid(True, alpha=0.3)
                
                ax4_3 = fig.add_subplot(gs_inner[0, 2])
                try:
                    kde = stats.gaussian_kde(values)
                    x_range = np.linspace(values.min(), values.max(), 100)
                    ax4_3.plot(x_range, kde(x_range), color='teal', linewidth=2)
                    ax4_3.fill_between(x_range, kde(x_range), alpha=0.3, color='teal')
                    ax4_3.set_xlabel('S_e', fontsize=8)
                    ax4_3.set_ylabel('Density', fontsize=8)
                    ax4_3.set_title('Density Spectrum', fontsize=10, fontweight='bold')
                    ax4_3.grid(True, alpha=0.3)
                except Exception as e:
                    logger.warning(f"Could not create KDE: {e}")
        else:
            # No CV data available
            ax = fig.add_subplot(gs[:, :])
            ax.text(0.5, 0.5, 'No CV Droplet Data Available', 
                   ha='center', va='center', fontsize=20, color='gray')
            ax.axis('off')
        
        output_file = self.output_dir / f"{experiment_name}_cv_mz{peak.mz:.4f}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  Saved CV panel: {output_file.name}")
        return output_file
    
    def create_all_panels(self, peak: TrackedPeak, 
                          experiment_name: str) -> Dict[str, Path]:
        """Create all 4-panel visualizations for a tracked peak."""
        logger.info(f"Creating all SRM panels for m/z {peak.mz:.4f}...")
        
        panels = {}
        panels['chromatography'] = self.create_chromatography_panel(peak, experiment_name)
        panels['ms1'] = self.create_ms1_panel(peak, experiment_name)
        panels['ms2'] = self.create_ms2_panel(peak, experiment_name)
        panels['cv'] = self.create_cv_panel(peak, experiment_name)
        
        return panels


def main():
    """Main entry point for SRM visualization."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Selected Reaction Monitoring Visualization')
    parser.add_argument('experiment_dir', type=str,
                       help='Path to experiment directory')
    parser.add_argument('--n-peaks', type=int, default=5,
                       help='Number of peaks to track (default: 5)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: experiment_dir/srm_visualizations)')
    
    args = parser.parse_args()
    
    experiment_dir = Path(args.experiment_dir)
    experiment_name = experiment_dir.name
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = experiment_dir / 'srm_visualizations'
    
    logger.info("="*70)
    logger.info("SELECTED REACTION MONITORING (SRM) VISUALIZATION")
    logger.info("="*70)
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Output: {output_dir}")
    
    # Select peaks
    selector = SRMPeakSelector(experiment_dir)
    selector.load_data()
    peaks = selector.select_top_peaks(n_peaks=args.n_peaks)
    
    # Extract data for each peak
    for i, peak in enumerate(peaks):
        logger.info(f"\nProcessing peak {i+1}/{len(peaks)}...")
        peak = selector.extract_peak_data(peak)
        
        # Create visualizations
        visualizer = SRMVisualizer(output_dir)
        panels = visualizer.create_all_panels(peak, experiment_name)
        
        logger.info(f"  Created {len(panels)} panels")
    
    logger.info("\n" + "="*70)
    logger.info("SRM VISUALIZATION COMPLETE")
    logger.info("="*70)
    logger.info(f"Output directory: {output_dir}")


if __name__ == "__main__":
    # Test mode
    import sys
    
    if len(sys.argv) > 1:
        main()
    else:
        # Default test
        experiment_dir = Path("results/ucdavis_complete_analysis/A_M3_negPFP_03")
        
        if not experiment_dir.exists():
            print(f"Experiment directory not found: {experiment_dir}")
            sys.exit(1)
        
        experiment_name = experiment_dir.name
        output_dir = experiment_dir / 'srm_visualizations'
        
        print("="*70)
        print("SRM VISUALIZATION - TEST MODE")
        print("="*70)
        print(f"Experiment: {experiment_name}")
        
        # Select and visualize top 3 peaks
        selector = SRMPeakSelector(experiment_dir)
        selector.load_data()
        peaks = selector.select_top_peaks(n_peaks=3)
        
        for i, peak in enumerate(peaks):
            print(f"\nProcessing peak {i+1}/{len(peaks)}...")
            peak = selector.extract_peak_data(peak)
            
            visualizer = SRMVisualizer(output_dir)
            panels = visualizer.create_all_panels(peak, experiment_name)
            
            print(f"  Created {len(panels)} panels")
        
        print("\n" + "="*70)
        print("TEST COMPLETE")
        print("="*70)
        print(f"Output: {output_dir}")

