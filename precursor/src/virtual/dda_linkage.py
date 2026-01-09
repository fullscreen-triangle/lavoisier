"""
DDA (Data-Dependent Acquisition) Linkage Module
================================================

Solves the fundamental problem of linking MS1 scans to their MS2 fragments.

In DDA mass spectrometry:
- MS1 scan at time T identifies precursors
- MS2 scans at time T+Δt fragment those precursors
- The linkage is through dda_event_idx, NOT retention time or scan number

This module provides the correct MS1 ↔ MS2 mapping.

Author: Kundai Farai Sachikonye
Date: 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DDAEvent:
    """A complete DDA event: one MS1 scan + its MS2 children."""
    dda_event_idx: int
    ms1_scan: Dict
    ms2_scans: List[Dict]
    
    @property
    def n_ms2(self) -> int:
        """Number of MS2 scans in this event."""
        return len(self.ms2_scans)
    
    @property
    def ms1_rt(self) -> float:
        """MS1 retention time."""
        return self.ms1_scan['scan_time']
    
    @property
    def ms2_precursors(self) -> List[float]:
        """List of precursor m/z values that were fragmented."""
        return [scan['MS2_PR_mz'] for scan in self.ms2_scans]


class DDALinkageManager:
    """
    Manages the linkage between MS1 and MS2 scans in DDA experiments.
    
    The key insight: dda_event_idx links MS1 to MS2, not retention time!
    """
    
    def __init__(self, experiment_dir: Path):
        self.experiment_dir = Path(experiment_dir)
        self.scan_info = None
        self.ms1_xic = None
        self.dda_events: Dict[int, DDAEvent] = {}
        
    def load_data(self):
        """Load scan info and MS1 XIC data."""
        logger.info("Loading DDA linkage data...")
        
        scan_info_path = self.experiment_dir / "stage_01_preprocessing" / "scan_info.csv"
        xic_path = self.experiment_dir / "stage_01_preprocessing" / "ms1_xic.csv"
        
        if not scan_info_path.exists():
            raise FileNotFoundError(f"scan_info.csv not found: {scan_info_path}")
        
        self.scan_info = pd.read_csv(scan_info_path)
        
        if xic_path.exists():
            self.ms1_xic = pd.read_csv(xic_path)
            logger.info(f"  Loaded {len(self.ms1_xic)} XIC points")
        
        logger.info(f"  Loaded {len(self.scan_info)} scans")
        
        # Build DDA events
        self._build_dda_events()
        
    def _build_dda_events(self):
        """Build the DDA event structure."""
        logger.info("Building DDA event structure...")
        
        # Group by dda_event_idx
        for event_idx, group in self.scan_info.groupby('dda_event_idx'):
            # MS1 scan (DDA_rank == 0)
            ms1_scans = group[group['DDA_rank'] == 0]
            
            if len(ms1_scans) == 0:
                logger.warning(f"  No MS1 scan for event {event_idx}")
                continue
            
            ms1_scan = ms1_scans.iloc[0].to_dict()
            
            # MS2 scans (DDA_rank > 0)
            ms2_scans = group[group['DDA_rank'] > 0]
            ms2_scan_list = [row.to_dict() for _, row in ms2_scans.iterrows()]
            
            # Create DDA event
            event = DDAEvent(
                dda_event_idx=int(event_idx),
                ms1_scan=ms1_scan,
                ms2_scans=ms2_scan_list
            )
            
            self.dda_events[int(event_idx)] = event
        
        n_with_ms2 = sum(1 for e in self.dda_events.values() if e.n_ms2 > 0)
        logger.info(f"  Built {len(self.dda_events)} DDA events")
        logger.info(f"  {n_with_ms2} events have MS2 scans")
        
    def get_ms2_for_precursor(self, precursor_mz: float, 
                              rt: float,
                              mz_tolerance: float = 0.01,
                              rt_tolerance: float = 0.5) -> List[Dict]:
        """
        Find all MS2 scans for a given precursor m/z and RT.
        
        Args:
            precursor_mz: Precursor m/z to search for
            rt: Retention time (minutes)
            mz_tolerance: m/z tolerance (Da)
            rt_tolerance: RT tolerance (minutes)
            
        Returns:
            List of MS2 scan dictionaries
        """
        matching_ms2 = []
        
        for event in self.dda_events.values():
            # Check if MS1 RT is close
            if abs(event.ms1_rt - rt) > rt_tolerance:
                continue
            
            # Check each MS2 scan
            for ms2_scan in event.ms2_scans:
                if abs(ms2_scan['MS2_PR_mz'] - precursor_mz) <= mz_tolerance:
                    # Add event info
                    ms2_scan['ms1_rt'] = event.ms1_rt
                    ms2_scan['ms1_spec_index'] = event.ms1_scan['spec_index']
                    matching_ms2.append(ms2_scan)
        
        return matching_ms2
    
    def get_ms2_spectrum(self, ms2_spec_index: int) -> Optional[pd.DataFrame]:
        """
        Load the actual MS2 spectrum data.
        
        Args:
            ms2_spec_index: The spec_index of the MS2 scan
            
        Returns:
            DataFrame with fragment m/z and intensities, or None
        """
        spectrum_path = self.experiment_dir / "stage_01_preprocessing" / "spectra" / f"spectrum_{ms2_spec_index}.csv"
        
        if not spectrum_path.exists():
            return None
        
        try:
            spectrum = pd.read_csv(spectrum_path)
            return spectrum
        except Exception as e:
            logger.warning(f"Could not load spectrum {ms2_spec_index}: {e}")
            return None
    
    def get_complete_srm_data(self, precursor_mz: float, rt: float,
                              mz_tolerance: float = 0.01,
                              rt_window: float = 1.0) -> Dict:
        """
        Get complete SRM data: MS1 XIC + all MS2 spectra for a precursor.
        
        This is the CORRECT way to link MS1 and MS2!
        
        Args:
            precursor_mz: Precursor m/z
            rt: Retention time apex
            mz_tolerance: m/z tolerance
            rt_window: RT window around apex
            
        Returns:
            Dictionary with 'xic', 'ms2_scans', 'ms2_spectra'
        """
        logger.info(f"Getting complete SRM data for m/z={precursor_mz:.4f}, RT={rt:.2f}")
        
        # 1. Get XIC (MS1 chromatogram)
        xic = None
        if self.ms1_xic is not None:
            xic = self.ms1_xic[
                (self.ms1_xic['mz'] >= precursor_mz - mz_tolerance) &
                (self.ms1_xic['mz'] <= precursor_mz + mz_tolerance) &
                (self.ms1_xic['rt'] >= rt - rt_window) &
                (self.ms1_xic['rt'] <= rt + rt_window)
            ].copy()
        
        # 2. Get MS2 scans (metadata)
        ms2_scans = self.get_ms2_for_precursor(precursor_mz, rt, 
                                                mz_tolerance, rt_window)
        
        # 3. Load actual MS2 spectra
        ms2_spectra = []
        for ms2_scan in ms2_scans:
            spectrum = self.get_ms2_spectrum(int(ms2_scan['spec_index']))
            if spectrum is not None:
                ms2_spectra.append({
                    'scan_info': ms2_scan,
                    'spectrum': spectrum
                })
        
        logger.info(f"  Found: XIC={len(xic) if xic is not None else 0} points, " +
                   f"MS2 scans={len(ms2_scans)}, MS2 spectra={len(ms2_spectra)}")
        
        return {
            'xic': xic,
            'ms2_scans': ms2_scans,
            'ms2_spectra': ms2_spectra,
            'precursor_mz': precursor_mz,
            'rt_apex': rt
        }
    
    def get_dda_event_by_ms1_scan(self, ms1_spec_index: int) -> Optional[DDAEvent]:
        """
        Get the complete DDA event for a given MS1 scan index.
        
        Args:
            ms1_spec_index: The spec_index of the MS1 scan
            
        Returns:
            DDAEvent or None
        """
        for event in self.dda_events.values():
            if event.ms1_scan['spec_index'] == ms1_spec_index:
                return event
        return None
    
    def export_linkage_table(self, output_path: Path):
        """
        Export a complete MS1-MS2 linkage table.
        
        This table explicitly shows which MS2 scans came from which MS1 scan.
        """
        logger.info("Exporting MS1-MS2 linkage table...")
        
        rows = []
        for event in self.dda_events.values():
            ms1_info = {
                'dda_event_idx': event.dda_event_idx,
                'ms1_spec_index': event.ms1_scan['spec_index'],
                'ms1_scan_number': event.ms1_scan['scan_number'],
                'ms1_rt': event.ms1_rt,
                'n_ms2_scans': event.n_ms2
            }
            
            if event.n_ms2 == 0:
                # MS1 with no MS2
                rows.append(ms1_info)
            else:
                # One row per MS2
                for ms2_scan in event.ms2_scans:
                    row = ms1_info.copy()
                    row.update({
                        'ms2_spec_index': ms2_scan['spec_index'],
                        'ms2_scan_number': ms2_scan['scan_number'],
                        'ms2_rt': ms2_scan['scan_time'],
                        'ms2_dda_rank': ms2_scan['DDA_rank'],
                        'precursor_mz': ms2_scan['MS2_PR_mz'],
                        'rt_offset': ms2_scan['scan_time'] - event.ms1_rt
                    })
                    rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        
        logger.info(f"  Exported {len(df)} linkage rows to {output_path}")
        return df
    
    def get_statistics(self) -> Dict:
        """Get DDA acquisition statistics."""
        n_events = len(self.dda_events)
        n_with_ms2 = sum(1 for e in self.dda_events.values() if e.n_ms2 > 0)
        total_ms2 = sum(e.n_ms2 for e in self.dda_events.values())
        
        ms2_per_event = [e.n_ms2 for e in self.dda_events.values() if e.n_ms2 > 0]
        
        return {
            'n_dda_events': n_events,
            'n_events_with_ms2': n_with_ms2,
            'total_ms2_scans': total_ms2,
            'avg_ms2_per_event': np.mean(ms2_per_event) if ms2_per_event else 0,
            'max_ms2_per_event': max(ms2_per_event) if ms2_per_event else 0,
            'min_ms2_per_event': min(ms2_per_event) if ms2_per_event else 0
        }


def main():
    """Test the DDA linkage manager."""
    import sys
    
    if len(sys.argv) > 1:
        experiment_dir = Path(sys.argv[1])
    else:
        experiment_dir = Path("results/ucdavis_complete_analysis/A_M3_negPFP_03")
    
    if not experiment_dir.exists():
        print(f"Experiment directory not found: {experiment_dir}")
        sys.exit(1)
    
    print("="*70)
    print("DDA LINKAGE ANALYSIS")
    print("="*70)
    print(f"Experiment: {experiment_dir.name}\n")
    
    # Create manager
    manager = DDALinkageManager(experiment_dir)
    manager.load_data()
    
    # Get statistics
    stats = manager.get_statistics()
    print("\nDDA Acquisition Statistics:")
    print(f"  Total DDA events:        {stats['n_dda_events']}")
    print(f"  Events with MS2:         {stats['n_events_with_ms2']}")
    print(f"  Total MS2 scans:         {stats['total_ms2_scans']}")
    print(f"  Avg MS2 per event:       {stats['avg_ms2_per_event']:.2f}")
    print(f"  Max MS2 per event:       {stats['max_ms2_per_event']}")
    print(f"  Min MS2 per event:       {stats['min_ms2_per_event']}")
    
    # Export linkage table
    output_path = experiment_dir / "ms1_ms2_linkage.csv"
    linkage_df = manager.export_linkage_table(output_path)
    
    print(f"\nLinkage table exported: {output_path}")
    print("\nFirst few linkages:")
    print(linkage_df[linkage_df['n_ms2_scans'] > 0].head(10))
    
    # Test: Get complete SRM data for a specific precursor
    print("\n" + "="*70)
    print("TEST: Get complete SRM data for m/z=293.124, RT=0.54")
    print("="*70)
    
    srm_data = manager.get_complete_srm_data(293.124, 0.54, 
                                             mz_tolerance=0.01, 
                                             rt_window=0.5)
    
    print(f"\nXIC points: {len(srm_data['xic']) if srm_data['xic'] is not None else 0}")
    print(f"MS2 scans: {len(srm_data['ms2_scans'])}")
    print(f"MS2 spectra loaded: {len(srm_data['ms2_spectra'])}")
    
    if srm_data['ms2_spectra']:
        print("\nFirst MS2 spectrum:")
        first_ms2 = srm_data['ms2_spectra'][0]
        print(f"  Scan: {first_ms2['scan_info']['scan_number']}")
        print(f"  RT: {first_ms2['scan_info']['scan_time']:.4f}")
        print(f"  MS1 RT: {first_ms2['scan_info']['ms1_rt']:.4f}")
        print(f"  RT offset: {first_ms2['scan_info']['scan_time'] - first_ms2['scan_info']['ms1_rt']:.4f}")
        print(f"  Fragments: {len(first_ms2['spectrum'])}")
        print(f"\n  Top 5 fragments:")
        top_frags = first_ms2['spectrum'].nlargest(5, 'i')
        for _, frag in top_frags.iterrows():
            print(f"    m/z {frag['mz']:.4f}: {frag['i']:.2e}")


if __name__ == "__main__":
    main()

