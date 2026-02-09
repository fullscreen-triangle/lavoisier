"""
Comprehensive Proteomics Analysis Pipeline
==========================================

Integrates:
- St. Stella's Sequence transformation
- S-Entropy coordinate encoding
- Fragment graph construction
- Categorical completion
- Validation against ground truth identifications
- PRIDE database mzML download support

Uses the existing SpectraReader from precursor/src/core for mzML parsing.

Author: Kundai Sachikonye
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
import json
import time
import re
import requests
import ftplib
from collections import defaultdict
from io import BytesIO
import gzip

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

# Import existing SpectraReader from precursor module
try:
    import sys
    precursor_path = Path(__file__).parent.parent.parent.parent / 'precursor' / 'src'
    if str(precursor_path) not in sys.path:
        sys.path.insert(0, str(precursor_path))

    from core.SpectraReader import extract_mzml, extract_spectra, get_spectra
    from core.DataStructure import MSDataContainer, SpectrumMetadata, DDAEvent
    SPECTRA_READER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import SpectraReader: {e}")
    SPECTRA_READER_AVAILABLE = False

# Import our modules
try:
    from st_stellas_sequence import (
        StStellasSequenceTransformer,
        PeptideCoordinatePath,
        AMINO_ACID_MASSES
    )
    from proteomics_core import SEntropyProteomicsEngine, SEntropySpectrum
    from state_counting import (
        StateCountingReconstructor,
        CircularValidationReconstructor,
        validate_fragment_hierarchy,
        mz_to_partition_depth,
        capacity,
        FragmentParentValidation,
        reconstruct_sequence_circular_validation
    )
    STATE_COUNTING_AVAILABLE = True
    CIRCULAR_VALIDATION_AVAILABLE = True
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from st_stellas_sequence import (
        StStellasSequenceTransformer,
        PeptideCoordinatePath,
        AMINO_ACID_MASSES
    )
    from proteomics_core import SEntropyProteomicsEngine, SEntropySpectrum
    try:
        from state_counting import (
            StateCountingReconstructor,
            CircularValidationReconstructor,
            validate_fragment_hierarchy,
            mz_to_partition_depth,
            capacity,
            FragmentParentValidation,
            validate_charge_redistribution,
            ChargeRedistributionValidation,
            reconstruct_sequence_circular_validation
        )
        STATE_COUNTING_AVAILABLE = True
        CIRCULAR_VALIDATION_AVAILABLE = True
    except ImportError:
        STATE_COUNTING_AVAILABLE = False
        CIRCULAR_VALIDATION_AVAILABLE = False


# ============================================================================
# PRIDE DATABASE DOWNLOAD
# ============================================================================

class PRIDEDownloader:
    """
    Download mzML files from PRIDE database.

    Uses the PRIDE Archive API and FTP to retrieve proteomics data.
    """

    PRIDE_API_URL = "https://www.ebi.ac.uk/pride/ws/archive/v2"
    PRIDE_FTP_HOST = "ftp.pride.ebi.ac.uk"

    def __init__(self, output_dir: str):
        """
        Initialize PRIDE downloader.

        Args:
            output_dir: Directory to save downloaded files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_project_info(self, project_accession: str) -> Dict:
        """
        Get project metadata from PRIDE API.

        Args:
            project_accession: PRIDE project accession (e.g., "PXD000001")

        Returns:
            Project metadata dictionary
        """
        url = f"{self.PRIDE_API_URL}/projects/{project_accession}"
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error fetching project info: {e}")
            return {}

    def list_project_files(self, project_accession: str) -> List[Dict]:
        """
        List all files in a PRIDE project.

        Args:
            project_accession: PRIDE project accession

        Returns:
            List of file metadata dictionaries
        """
        url = f"{self.PRIDE_API_URL}/files/byProject"
        params = {"accession": project_accession}
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error listing project files: {e}")
            return []

    def download_file_ftp(self, ftp_path: str, local_filename: str) -> Optional[Path]:
        """
        Download a file from PRIDE FTP.

        Args:
            ftp_path: FTP path after pride.ebi.ac.uk
            local_filename: Local filename to save as

        Returns:
            Path to downloaded file or None if failed
        """
        local_path = self.output_dir / local_filename

        if local_path.exists():
            print(f"  File already exists: {local_path}")
            return local_path

        try:
            print(f"  Connecting to PRIDE FTP...")
            ftp = ftplib.FTP(self.PRIDE_FTP_HOST)
            ftp.login()

            # Navigate to directory
            dir_path = '/'.join(ftp_path.split('/')[:-1])
            filename = ftp_path.split('/')[-1]

            ftp.cwd(dir_path)

            # Download file
            print(f"  Downloading: {filename}")
            with open(local_path, 'wb') as f:
                ftp.retrbinary(f'RETR {filename}', f.write)

            ftp.quit()

            # Decompress if gzipped
            if local_filename.endswith('.gz'):
                print(f"  Decompressing: {local_filename}")
                decompressed_path = local_path.with_suffix('')
                with gzip.open(local_path, 'rb') as f_in:
                    with open(decompressed_path, 'wb') as f_out:
                        f_out.write(f_in.read())
                local_path.unlink()  # Remove gzipped file
                return decompressed_path

            return local_path

        except Exception as e:
            print(f"  FTP download failed: {e}")
            return None

    def download_project_mzml(
        self,
        project_accession: str,
        max_files: int = 1,
        file_pattern: str = ".mzML"
    ) -> List[Path]:
        """
        Download mzML files from a PRIDE project.

        Args:
            project_accession: PRIDE project accession (e.g., "PXD000001")
            max_files: Maximum number of files to download
            file_pattern: File extension pattern to match

        Returns:
            List of paths to downloaded files
        """
        print(f"\n{'='*60}")
        print(f"DOWNLOADING FROM PRIDE: {project_accession}")
        print(f"{'='*60}")

        # Get project info
        project_info = self.get_project_info(project_accession)
        if project_info:
            print(f"  Project: {project_info.get('title', 'Unknown')}")

        # List files
        files = self.list_project_files(project_accession)
        print(f"  Total files in project: {len(files)}")

        # Filter for mzML files
        mzml_files = [
            f for f in files
            if file_pattern.lower() in f.get('fileName', '').lower()
        ]
        print(f"  mzML files found: {len(mzml_files)}")

        if not mzml_files:
            print("  No mzML files found, checking for raw/peak files...")
            # Try alternative patterns
            for pattern in ['.raw', '.peak', '.mgf']:
                alt_files = [f for f in files if pattern in f.get('fileName', '').lower()]
                if alt_files:
                    print(f"  Found {len(alt_files)} {pattern} files")
                    mzml_files = alt_files[:max_files]
                    break

        # Download files
        downloaded = []
        for file_info in mzml_files[:max_files]:
            filename = file_info.get('fileName', '')
            ftp_link = file_info.get('publicFileLocation', '')

            if not ftp_link:
                # Construct FTP path
                ftp_link = f"/pride/data/archive/{project_accession}/{filename}"
            else:
                # Extract path from full URL
                ftp_link = ftp_link.replace(f"ftp://{self.PRIDE_FTP_HOST}", "")

            print(f"\n  File: {filename}")

            local_file = self.download_file_ftp(ftp_link, filename)
            if local_file:
                downloaded.append(local_file)
                print(f"  -> Saved: {local_file}")

        print(f"\n  Downloaded {len(downloaded)} files")
        return downloaded

    def download_example_dataset(self) -> List[Path]:
        """
        Download a small example dataset for testing.

        Uses PXD000001 which is a small, well-annotated dataset.

        Returns:
            List of paths to downloaded files
        """
        # PXD000001 is a classic small proteomics dataset
        return self.download_project_mzml("PXD000001", max_files=1)


@dataclass
class MGFSpectrum:
    """Parsed MGF spectrum."""
    title: str
    precursor_mz: float
    charge: int
    mz_array: np.ndarray
    intensity_array: np.ndarray
    retention_time: float = 0.0


@dataclass
class PSMRecord:
    """Peptide-Spectrum Match from mzTab."""
    sequence: str
    spectrum_ref: str
    accession: str
    charge: int
    experimental_mz: float
    theoretical_mz: float
    score: float
    modifications: str


@dataclass
class ValidationResult:
    """Result of sequence validation."""
    spectrum_id: str
    true_sequence: str
    predicted_sequence: str
    sequence_match: bool
    partial_match_score: float
    mass_error: float
    stellas_path_length: float
    stellas_tortuosity: float
    mean_s_entropy: float
    n_fragments: int
    processing_time: float


class ProteomicsPipelineRunner:
    """
    Comprehensive proteomics analysis pipeline.

    Integrates St. Stella's Sequence transformation with S-Entropy
    coordinate encoding for database-free peptide analysis.
    """

    def __init__(self, data_dir: str, output_dir: str):
        """
        Initialize pipeline.

        Args:
            data_dir: Directory containing proteomics data files
            output_dir: Directory for output results
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize transformers
        self.stellas = StStellasSequenceTransformer()
        self.sentropy_engine = SEntropyProteomicsEngine()

        # Results storage
        self.spectra: List[MGFSpectrum] = []
        self.psms: Dict[str, PSMRecord] = {}
        self.validation_results: List[ValidationResult] = []
        self.statistics = {}

        # Data containers from YOUR SpectraReader
        self.data_container: Optional['MSDataContainer'] = None
        self.scan_info_df: Optional[pd.DataFrame] = None
        self.spectra_dict: Optional[Dict] = None
        self.ms1_xic_df: Optional[pd.DataFrame] = None

    # ========================================================================
    # DATA LOADING
    # ========================================================================

    def parse_mgf(self, mgf_path: str, max_spectra: Optional[int] = None) -> List[MGFSpectrum]:
        """
        Parse MGF file into list of spectra.

        Args:
            mgf_path: Path to MGF file
            max_spectra: Maximum number of spectra to load (None = all)

        Returns:
            List of MGFSpectrum objects
        """
        print(f"Parsing MGF file: {mgf_path}")
        spectra = []

        with open(mgf_path, 'r') as f:
            current_spectrum = None
            mz_list = []
            intensity_list = []

            for line in f:
                line = line.strip()

                if line == 'BEGIN IONS':
                    current_spectrum = {
                        'title': '',
                        'pepmass': 0.0,
                        'charge': 2,
                        'rt': 0.0
                    }
                    mz_list = []
                    intensity_list = []

                elif line == 'END IONS':
                    if current_spectrum and len(mz_list) > 0:
                        spectrum = MGFSpectrum(
                            title=current_spectrum['title'],
                            precursor_mz=current_spectrum['pepmass'],
                            charge=current_spectrum['charge'],
                            mz_array=np.array(mz_list),
                            intensity_array=np.array(intensity_list),
                            retention_time=current_spectrum['rt']
                        )
                        spectra.append(spectrum)

                        if max_spectra and len(spectra) >= max_spectra:
                            break

                    current_spectrum = None

                elif current_spectrum is not None:
                    if line.startswith('TITLE='):
                        current_spectrum['title'] = line[6:]

                    elif line.startswith('PEPMASS='):
                        parts = line[8:].split()
                        current_spectrum['pepmass'] = float(parts[0])

                    elif line.startswith('CHARGE='):
                        charge_str = line[7:].replace('+', '').replace('-', '')
                        try:
                            current_spectrum['charge'] = int(charge_str)
                        except ValueError:
                            current_spectrum['charge'] = 2

                    elif line.startswith('RTINSECONDS='):
                        current_spectrum['rt'] = float(line[12:]) / 60.0  # Convert to minutes

                    elif line and not line.startswith('#') and '\t' in line or ' ' in line:
                        # Fragment ion line
                        parts = line.replace('\t', ' ').split()
                        if len(parts) >= 2:
                            try:
                                mz = float(parts[0])
                                intensity = float(parts[1])
                                mz_list.append(mz)
                                intensity_list.append(intensity)
                            except ValueError:
                                pass

        print(f"  Loaded {len(spectra)} spectra")
        return spectra

    def parse_mztab(self, mztab_path: str) -> Dict[str, PSMRecord]:
        """
        Parse mzTab file for PSMs (Peptide-Spectrum Matches).

        Args:
            mztab_path: Path to mzTab file

        Returns:
            Dictionary mapping spectrum reference to PSMRecord
        """
        print(f"Parsing mzTab file: {mztab_path}")
        psms = {}

        with open(mztab_path, 'r') as f:
            for line in f:
                if line.startswith('PSM\t'):
                    parts = line.strip().split('\t')

                    if len(parts) >= 15:
                        try:
                            # mzTab PSM columns:
                            # 0: PSM, 1: sequence, 2: PSM_ID, 3: accession
                            # 8: search_engine_score, 9: modifications
                            # 11: charge, 12: exp_mz, 13: calc_mz
                            # 14: spectra_ref (ms_run[1]:spectrum=N)
                            sequence = parts[1]
                            spectrum_ref = parts[14]  # Correct column for spectrum reference
                            accession = parts[3]

                            # Parse charge
                            try:
                                charge = int(parts[11])
                            except (ValueError, IndexError):
                                charge = 2

                            # Parse m/z values
                            try:
                                exp_mz = float(parts[12])
                            except (ValueError, IndexError):
                                exp_mz = 0.0

                            try:
                                theo_mz = float(parts[13])
                            except (ValueError, IndexError):
                                theo_mz = 0.0

                            # Parse score
                            try:
                                score = float(parts[8]) if parts[8] != 'null' else 0.0
                            except (ValueError, IndexError):
                                score = 0.0

                            # Modifications
                            mods = parts[9] if len(parts) > 9 else ''

                            psm = PSMRecord(
                                sequence=sequence,
                                spectrum_ref=spectrum_ref,
                                accession=accession,
                                charge=charge,
                                experimental_mz=exp_mz,
                                theoretical_mz=theo_mz,
                                score=score,
                                modifications=mods
                            )

                            # Extract spectrum number from reference
                            match = re.search(r'spectrum=(\d+)', spectrum_ref)
                            if match:
                                spectrum_key = match.group(1)
                                psms[spectrum_key] = psm

                        except Exception as e:
                            continue

        print(f"  Loaded {len(psms)} PSMs")
        return psms

    def load_mzml(
        self,
        mzml_path: str,
        rt_range: List[float] = [0.0, 100.0],
        max_spectra: Optional[int] = 2000,
        vendor: str = "thermo"
    ) -> Tuple[pd.DataFrame, Dict, pd.DataFrame]:
        """
        Load mzML file using the existing SpectraReader.

        This uses YOUR existing extract_mzml function from precursor/src/core/SpectraReader.py

        Args:
            mzml_path: Path to mzML file
            rt_range: Retention time range [start, end] in minutes
            max_spectra: Maximum spectra to load (not used directly, but limits processing)
            vendor: Instrument vendor (thermo, waters, agilent, etc.)

        Returns:
            Tuple of (scan_info_df, spectra_dict, ms1_xic_df)
        """
        if not SPECTRA_READER_AVAILABLE:
            raise ImportError(
                "SpectraReader not available. Please ensure precursor/src/core/SpectraReader.py exists."
            )

        print(f"\nUsing YOUR SpectraReader.extract_mzml() function")
        print(f"  File: {mzml_path}")
        print(f"  RT range: {rt_range[0]:.2f} - {rt_range[1]:.2f} min")
        print(f"  Vendor: {vendor}")

        # Use YOUR existing extract_mzml function!
        scan_info_df, spectra_dict, ms1_xic_df = extract_mzml(
            mzml=mzml_path,
            rt_range=rt_range,
            dda_top=6,
            ms1_threshold=1000,
            ms2_threshold=10,
            ms1_precision=50e-6,
            ms2_precision=500e-6,
            vendor=vendor
        )

        print(f"  Loaded {len(spectra_dict)} spectra")
        print(f"  Scan info shape: {scan_info_df.shape}")

        return scan_info_df, spectra_dict, ms1_xic_df

    def load_mzml_as_spectra(
        self,
        mzml_path: str,
        rt_range: List[float] = [0.0, 100.0],
        max_spectra: Optional[int] = 2000,
        vendor: str = "thermo"
    ) -> List[MGFSpectrum]:
        """
        Load mzML file and convert to MGFSpectrum format for compatibility.

        Uses YOUR existing SpectraReader, then converts to internal format.

        Args:
            mzml_path: Path to mzML file
            rt_range: Retention time range in minutes
            max_spectra: Maximum spectra to return
            vendor: Instrument vendor

        Returns:
            List of MGFSpectrum objects
        """
        scan_info_df, spectra_dict, ms1_xic_df = self.load_mzml(
            mzml_path, rt_range, max_spectra, vendor
        )

        # Store raw data for DDA linkage
        self.scan_info_df = scan_info_df
        self.spectra_dict = spectra_dict
        self.ms1_xic_df = ms1_xic_df

        # Convert to MGFSpectrum format for compatibility with existing pipeline
        spectra = []
        ms2_scans = scan_info_df[scan_info_df['DDA_rank'] > 0]

        for idx, row in ms2_scans.iterrows():
            spec_index = row['spec_index']
            if spec_index not in spectra_dict:
                continue

            spec_df = spectra_dict[spec_index]

            # Create MGFSpectrum
            spectrum = MGFSpectrum(
                title=f"spectrum={row['scan_number']};dda_event={row['dda_event_idx']}",
                precursor_mz=row['MS2_PR_mz'],
                charge=2,  # Default, could be extracted from mzML
                mz_array=spec_df['mz'].values,
                intensity_array=spec_df['i'].values,
                retention_time=row['scan_time']
            )
            spectra.append(spectrum)

            if max_spectra and len(spectra) >= max_spectra:
                break

        print(f"  Converted {len(spectra)} MS2 spectra to internal format")
        return spectra

    def load_data(
        self,
        mgf_file: str = "PXD000001.mgf",
        mztab_file: str = "PXD000001.mztab",
        max_spectra: Optional[int] = 2000
    ):
        """
        Load proteomics data files.

        Args:
            mgf_file: MGF filename in data_dir
            mztab_file: mzTab filename in data_dir
            max_spectra: Maximum spectra to load
        """
        print("\n" + "=" * 80)
        print("LOADING PROTEOMICS DATA")
        print("=" * 80)

        mgf_path = self.data_dir / mgf_file
        mztab_path = self.data_dir / mztab_file

        if not mgf_path.exists():
            raise FileNotFoundError(f"MGF file not found: {mgf_path}")

        self.spectra = self.parse_mgf(str(mgf_path), max_spectra)

        if mztab_path.exists():
            self.psms = self.parse_mztab(str(mztab_path))
        else:
            print(f"  Warning: mzTab file not found, skipping PSM loading")
            self.psms = {}

        # Match spectra with PSMs
        n_matched = 0
        for spectrum in self.spectra:
            # Extract spectrum number from title
            match = re.search(r'spectrum=(\d+)', spectrum.title)
            if match:
                spectrum_key = match.group(1)
                if spectrum_key in self.psms:
                    n_matched += 1

        print(f"\n  Matched {n_matched} spectra with ground truth PSMs")

    def load_data_mzml(
        self,
        mzml_file: str,
        mztab_file: Optional[str] = None,
        rt_range: List[float] = [0.0, 100.0],
        max_spectra: Optional[int] = 2000,
        vendor: str = "thermo"
    ):
        """
        Load mzML data using YOUR existing SpectraReader.

        This is the PREFERRED method - uses your own infrastructure!

        Args:
            mzml_file: mzML filename in data_dir
            mztab_file: Optional mzTab filename for ground truth
            rt_range: Retention time range in minutes
            max_spectra: Maximum spectra to load
            vendor: Instrument vendor
        """
        print("\n" + "=" * 80)
        print("LOADING mzML DATA (using YOUR SpectraReader)")
        print("=" * 80)

        mzml_path = self.data_dir / mzml_file

        if not mzml_path.exists():
            raise FileNotFoundError(f"mzML file not found: {mzml_path}")

        # Use YOUR SpectraReader!
        self.spectra = self.load_mzml_as_spectra(
            str(mzml_path), rt_range, max_spectra, vendor
        )

        # Load ground truth if available
        if mztab_file:
            mztab_path = self.data_dir / mztab_file
            if mztab_path.exists():
                self.psms = self.parse_mztab(str(mztab_path))
            else:
                print(f"  Warning: mzTab file not found")
                self.psms = {}
        else:
            self.psms = {}

        # Match spectra with PSMs
        n_matched = 0
        for spectrum in self.spectra:
            match = re.search(r'spectrum=(\d+)', spectrum.title)
            if match:
                spectrum_key = match.group(1)
                if spectrum_key in self.psms:
                    n_matched += 1

        print(f"\n  Matched {n_matched} spectra with ground truth PSMs")

    def download_and_load_from_pride(
        self,
        project_accession: str = "PXD000001",
        max_files: int = 1,
        rt_range: List[float] = [0.0, 100.0],
        max_spectra: Optional[int] = 2000,
        vendor: str = "thermo"
    ):
        """
        Download mzML from PRIDE and load using YOUR SpectraReader.

        This is what you asked for - download from PRIDE and use your reader!

        Args:
            project_accession: PRIDE project accession (e.g., "PXD000001")
            max_files: Maximum files to download
            rt_range: Retention time range
            max_spectra: Maximum spectra to load
            vendor: Instrument vendor
        """
        print("\n" + "=" * 80)
        print(f"DOWNLOADING FROM PRIDE: {project_accession}")
        print("=" * 80)

        # Download from PRIDE
        downloader = PRIDEDownloader(str(self.data_dir))
        downloaded_files = downloader.download_project_mzml(
            project_accession, max_files
        )

        if not downloaded_files:
            raise FileNotFoundError(f"No files downloaded from {project_accession}")

        # Load the first mzML file using YOUR SpectraReader
        mzml_file = downloaded_files[0]
        print(f"\n  Loading with YOUR SpectraReader: {mzml_file.name}")

        self.spectra = self.load_mzml_as_spectra(
            str(mzml_file), rt_range, max_spectra, vendor
        )

        # Try to find mzTab for ground truth
        mztab_candidates = list(self.data_dir.glob("*.mztab")) + \
                          list(self.data_dir.glob("*.mzTab"))
        if mztab_candidates:
            self.psms = self.parse_mztab(str(mztab_candidates[0]))
        else:
            self.psms = {}

        print(f"\n  Loaded {len(self.spectra)} spectra from PRIDE")

    def create_data_container(self, mzml_path: str) -> Optional['MSDataContainer']:
        """
        Create MSDataContainer from loaded mzML data.

        Uses YOUR MSDataContainer from precursor/src/core/DataStructure.py
        for organized data handling with DDA linkage and S-Entropy support.

        Args:
            mzml_path: Path to the mzML file

        Returns:
            MSDataContainer instance or None if not available
        """
        if not SPECTRA_READER_AVAILABLE:
            print("  Warning: MSDataContainer not available")
            return None

        if self.scan_info_df is None or self.spectra_dict is None:
            print("  Warning: No mzML data loaded. Call load_mzml first.")
            return None

        try:
            container = MSDataContainer(
                mzml_filepath=mzml_path,
                scan_info_df=self.scan_info_df,
                spectra_dict=self.spectra_dict,
                ms1_xic_df=self.ms1_xic_df,
                extraction_params={
                    'ms1_threshold': 1000,
                    'ms2_threshold': 10,
                    'ms1_precision': 50e-6,
                    'ms2_precision': 500e-6
                }
            )
            self.data_container = container
            print(f"\n  Created MSDataContainer:")
            print(container)
            return container
        except Exception as e:
            print(f"  Error creating MSDataContainer: {e}")
            return None

    def get_dda_linked_pairs(self) -> List[Dict]:
        """
        Get DDA-linked precursor-fragment pairs using YOUR infrastructure.

        This uses the DDA linkage from your SpectraReader and MSDataContainer.
        Each pair contains MS1 precursor and its linked MS2 fragments.

        Returns:
            List of dictionaries with linked MS1-MS2 data
        """
        if self.data_container is None:
            print("  Warning: No MSDataContainer. Using scan_info_df for linkage.")

            if self.scan_info_df is None:
                return []

            # Manual linkage using dda_event_idx
            pairs = []
            dda_events = self.scan_info_df.groupby('dda_event_idx')

            for dda_idx, event_df in dda_events:
                ms1_rows = event_df[event_df['DDA_rank'] == 0]
                ms2_rows = event_df[event_df['DDA_rank'] > 0]

                if ms1_rows.empty:
                    continue

                ms1_row = ms1_rows.iloc[0]
                ms1_spec_idx = int(ms1_row['spec_index'])
                ms1_rt = float(ms1_row['scan_time'])

                for _, ms2_row in ms2_rows.iterrows():
                    ms2_spec_idx = int(ms2_row['spec_index'])

                    pair = {
                        'dda_event_idx': int(dda_idx),
                        'ms1_spec_index': ms1_spec_idx,
                        'ms1_rt': ms1_rt,
                        'ms2_spec_index': ms2_spec_idx,
                        'ms2_rt': float(ms2_row['scan_time']),
                        'precursor_mz': float(ms2_row['MS2_PR_mz']),
                        'dda_rank': int(ms2_row['DDA_rank']),
                        'ms1_spectrum': self.spectra_dict.get(ms1_spec_idx) if self.spectra_dict else None,
                        'ms2_spectrum': self.spectra_dict.get(ms2_spec_idx) if self.spectra_dict else None,
                    }
                    pairs.append(pair)

            return pairs

        # Use MSDataContainer's precursor-fragment pairs
        pairs = []
        for pf_pair in self.data_container.get_precursor_fragment_pairs():
            pair = {
                'dda_event_idx': pf_pair.dda_event_idx,
                'ms1_spec_index': pf_pair.precursor_spec_index,
                'ms1_rt': pf_pair.precursor_rt,
                'ms2_spec_index': pf_pair.fragment_spec_index,
                'ms2_rt': pf_pair.fragment_rt,
                'precursor_mz': pf_pair.precursor_mz,
                'dda_rank': pf_pair.dda_rank,
                'precursor_intensity': pf_pair.precursor_intensity,
                'precursor_ppm_error': pf_pair.precursor_ppm_error,
                'ms1_spectrum': pf_pair.ms1_spectrum,
                'ms2_spectrum': pf_pair.ms2_spectrum,
            }
            pairs.append(pair)

        return pairs

    # ========================================================================
    # ANALYSIS PIPELINE
    # ========================================================================

    def run_stellas_transformation(self) -> Dict:
        """
        Run St. Stella's Sequence transformation on all identified peptides.

        Returns:
            Statistics dictionary
        """
        print("\n" + "=" * 80)
        print("ST. STELLA'S SEQUENCE TRANSFORMATION")
        print("=" * 80)

        stellas_results = []
        unique_sequences = set()

        for psm in self.psms.values():
            sequence = psm.sequence

            # Skip if already processed
            if sequence in unique_sequences:
                continue
            unique_sequences.add(sequence)

            # Transform peptide
            path = self.stellas.transform_peptide(sequence)

            stellas_results.append({
                'sequence': sequence,
                'length': len(sequence),
                'path_length': path.path_length,
                'endpoint_distance': path.endpoint_distance,
                'tortuosity': path.tortuosity,
                'mean_s_knowledge': path.mean_s_knowledge,
                'mean_s_time': path.mean_s_time,
                'mean_s_entropy': path.mean_s_entropy
            })

        print(f"\n  Transformed {len(stellas_results)} unique peptide sequences")

        # Statistics
        if stellas_results:
            df = pd.DataFrame(stellas_results)

            stats = {
                'n_sequences': len(df),
                'mean_length': float(df['length'].mean()),
                'mean_path_length': float(df['path_length'].mean()),
                'mean_tortuosity': float(df['tortuosity'].mean()),
                'mean_s_entropy': float(df['mean_s_entropy'].mean()),
                'correlation_length_path': float(pearsonr(df['length'], df['path_length'])[0])
            }

            print(f"\n  Statistics:")
            print(f"    Mean peptide length: {stats['mean_length']:.2f}")
            print(f"    Mean path length: {stats['mean_path_length']:.4f}")
            print(f"    Mean tortuosity: {stats['mean_tortuosity']:.4f}")
            print(f"    Length-path correlation: {stats['correlation_length_path']:.4f}")

            # Save to CSV
            df.to_csv(self.output_dir / 'stellas_transformations.csv', index=False)
            print(f"\n  Saved: stellas_transformations.csv")

            self.statistics['stellas'] = stats

            return stats

        return {}

    def run_fragment_analysis(self, max_spectra: int = 500) -> Dict:
        """
        Run fragment graph analysis on spectra.

        Args:
            max_spectra: Maximum spectra to analyze

        Returns:
            Statistics dictionary
        """
        print("\n" + "=" * 80)
        print("FRAGMENT GRAPH ANALYSIS")
        print("=" * 80)

        fragment_results = []
        spectra_to_analyze = self.spectra[:max_spectra]

        start_time = time.time()

        for i, spectrum in enumerate(spectra_to_analyze):
            if i % 100 == 0:
                print(f"  Processing spectrum {i+1}/{len(spectra_to_analyze)}...")

            # Build fragment graph
            G = self.stellas.build_fragment_graph(
                spectrum.mz_array,
                spectrum.intensity_array,
                spectrum.precursor_mz,
                spectrum.charge
            )

            fragment_results.append({
                'spectrum_id': spectrum.title,
                'n_fragments': len(spectrum.mz_array),
                'n_nodes': len(G.nodes()),
                'n_edges': len(G.edges()),
                'precursor_mz': spectrum.precursor_mz,
                'charge': spectrum.charge
            })

        elapsed = time.time() - start_time

        print(f"\n  Analyzed {len(fragment_results)} spectra in {elapsed:.2f}s")
        print(f"  Average: {elapsed/len(fragment_results)*1000:.2f} ms/spectrum")

        if fragment_results:
            df = pd.DataFrame(fragment_results)

            stats = {
                'n_spectra': len(df),
                'mean_fragments': float(df['n_fragments'].mean()),
                'mean_nodes': float(df['n_nodes'].mean()),
                'mean_edges': float(df['n_edges'].mean()),
                'processing_time_ms': elapsed / len(fragment_results) * 1000
            }

            print(f"\n  Statistics:")
            print(f"    Mean fragments per spectrum: {stats['mean_fragments']:.1f}")
            print(f"    Mean graph nodes: {stats['mean_nodes']:.1f}")
            print(f"    Mean graph edges: {stats['mean_edges']:.1f}")

            df.to_csv(self.output_dir / 'fragment_analysis.csv', index=False)
            print(f"\n  Saved: fragment_analysis.csv")

            self.statistics['fragments'] = stats

            return stats

        return {}

    def run_sequence_reconstruction(self, max_spectra: int = 100, use_circular_validation: bool = True) -> Dict:
        """
        Run de novo sequence reconstruction with validation.

        Uses CIRCULAR VALIDATION for improved accuracy:
        A→B→C→A: spectrum → candidate → predicted_spectrum → compare_to_original

        The candidate whose predicted spectrum best matches the original IS correct.
        This leverages the bijective property of the ion-to-droplet transformation.

        Args:
            max_spectra: Maximum spectra to reconstruct
            use_circular_validation: Use circular validation (recommended)

        Returns:
            Validation statistics
        """
        print("\n" + "=" * 80)
        print("SEQUENCE RECONSTRUCTION & VALIDATION (CIRCULAR VALIDATION)")
        print("=" * 80)

        if use_circular_validation and CIRCULAR_VALIDATION_AVAILABLE:
            print("  Using CIRCULAR VALIDATION reconstruction method")
            print("  Algorithm: spectrum -> candidate -> predicted_spectrum -> compare -> validate")
            print("  The bijective property ensures correct sequence completes the cycle")
            reconstructor = CircularValidationReconstructor()
        elif STATE_COUNTING_AVAILABLE:
            print("  Using state counting reconstruction method")
            reconstructor = StateCountingReconstructor()
        else:
            print("  Using graph-based reconstruction method (state counting not available)")
            reconstructor = None

        validation_results = []
        correct_count = 0
        partial_match_scores = []
        hierarchy_validations = []
        circular_validation_scores = []

        # Filter spectra with known sequences
        spectra_with_psms = []
        for spectrum in self.spectra[:max_spectra * 5]:  # Look at more to find matches
            match = re.search(r'spectrum=(\d+)', spectrum.title)
            if match:
                spectrum_key = match.group(1)
                if spectrum_key in self.psms:
                    spectra_with_psms.append((spectrum, self.psms[spectrum_key]))

                    if len(spectra_with_psms) >= max_spectra:
                        break

        print(f"\n  Found {len(spectra_with_psms)} spectra with ground truth sequences")

        for i, (spectrum, psm) in enumerate(spectra_with_psms):
            if i % 25 == 0:
                print(f"  Processing spectrum {i+1}/{len(spectra_with_psms)}...")

            start_time = time.time()

            true_seq = psm.sequence

            # Reconstruct sequence using circular validation
            if use_circular_validation and CIRCULAR_VALIDATION_AVAILABLE:
                result = reconstructor.reconstruct(
                    spectrum.mz_array,
                    spectrum.intensity_array,
                    spectrum.precursor_mz,
                    spectrum.charge,
                    known_sequence=true_seq
                )
                # Track circular validation score
                cv_score = result.get('circular_validation_score', 0.0)
                circular_validation_scores.append(cv_score)
            elif reconstructor is not None:
                result = reconstructor.reconstruct(
                    spectrum.mz_array,
                    spectrum.intensity_array,
                    spectrum.precursor_mz,
                    spectrum.charge,
                    known_sequence=true_seq
                )
                circular_validation_scores.append(0.0)
            else:
                # Fallback to stellas
                result = self.stellas.reconstruct_sequence(
                    spectrum.mz_array,
                    spectrum.intensity_array,
                    spectrum.precursor_mz,
                    spectrum.charge,
                    known_sequence=true_seq,
                    use_state_counting=False
                )
                circular_validation_scores.append(0.0)

            processing_time = time.time() - start_time

            predicted_seq = result['best_sequence']

            # Get validation metrics from result if available
            if 'validation' in result and result['validation']:
                val_metrics = result['validation']
                sequence_match = val_metrics.get('exact_match', False)
                partial_score = val_metrics.get('partial_score', 0.0)
            else:
                # Compute manually
                sequence_match = predicted_seq.upper() == true_seq.upper()
                partial_score = self._compute_partial_match(predicted_seq, true_seq)

            if sequence_match:
                correct_count += 1

            partial_match_scores.append(partial_score)

            # Fragment-parent hierarchical validation
            if STATE_COUNTING_AVAILABLE:
                hierarchy_val = validate_fragment_hierarchy(
                    spectrum.mz_array,
                    spectrum.intensity_array,
                    spectrum.precursor_mz
                )
                hierarchy_validations.append({
                    'overlap_score': hierarchy_val.overlap_score,
                    'wavelength_ratio': hierarchy_val.wavelength_ratio,
                    'energy_ratio': hierarchy_val.energy_ratio,
                    'phase_coherence': hierarchy_val.phase_coherence,
                    'is_valid': hierarchy_val.is_valid,
                    'overall_score': hierarchy_val.overall_score
                })
            else:
                hierarchy_validations.append({
                    'overlap_score': 0.0,
                    'wavelength_ratio': 0.0,
                    'energy_ratio': 0.0,
                    'phase_coherence': 0.0,
                    'is_valid': False,
                    'overall_score': 0.0
                })

            # St. Stella's path for true sequence
            true_path = self.stellas.transform_peptide(true_seq)

            # Mass error
            predicted_mass = sum(AMINO_ACID_MASSES.get(aa, 110) for aa in predicted_seq) if predicted_seq else 0
            true_mass = sum(AMINO_ACID_MASSES.get(aa, 110) for aa in true_seq)
            mass_error = abs(predicted_mass - true_mass)

            validation = ValidationResult(
                spectrum_id=spectrum.title,
                true_sequence=true_seq,
                predicted_sequence=predicted_seq,
                sequence_match=sequence_match,
                partial_match_score=partial_score,
                mass_error=mass_error,
                stellas_path_length=true_path.path_length,
                stellas_tortuosity=true_path.tortuosity,
                mean_s_entropy=true_path.mean_s_entropy,
                n_fragments=len(spectrum.mz_array),
                processing_time=processing_time
            )

            validation_results.append(validation)

        self.validation_results = validation_results

        # Compute statistics
        if validation_results:
            n_total = len(validation_results)
            accuracy = correct_count / n_total * 100
            mean_partial = np.mean(partial_match_scores)

            # Hierarchy validation stats
            if hierarchy_validations:
                n_hierarchy_valid = sum(1 for h in hierarchy_validations if h['is_valid'])
                mean_hierarchy_score = np.mean([h['overall_score'] for h in hierarchy_validations])
            else:
                n_hierarchy_valid = 0
                mean_hierarchy_score = 0.0

            # Circular validation stats
            mean_cv_score = np.mean(circular_validation_scores) if circular_validation_scores else 0.0

            # Determine method name
            if use_circular_validation and CIRCULAR_VALIDATION_AVAILABLE:
                method_name = 'circular_validation'
            elif STATE_COUNTING_AVAILABLE:
                method_name = 'state_counting'
            else:
                method_name = 'graph_based'

            stats = {
                'method': method_name,
                'n_validated': n_total,
                'n_correct': correct_count,
                'accuracy_percent': accuracy,
                'mean_partial_match': float(mean_partial),
                'mean_circular_validation_score': float(mean_cv_score),
                'mean_processing_time_ms': float(np.mean([v.processing_time for v in validation_results]) * 1000),
                'n_hierarchy_valid': n_hierarchy_valid,
                'hierarchy_valid_percent': n_hierarchy_valid / n_total * 100,
                'mean_hierarchy_score': float(mean_hierarchy_score)
            }

            print(f"\n  Validation Results:")
            print(f"    Method: {stats['method']}")
            print(f"    Total spectra: {n_total}")
            print(f"    Exact matches: {correct_count} ({accuracy:.1f}%)")
            print(f"    Mean partial match score: {mean_partial:.3f}")
            print(f"    Mean circular validation score: {mean_cv_score:.3f}")
            print(f"    Hierarchy valid: {n_hierarchy_valid} ({stats['hierarchy_valid_percent']:.1f}%)")
            print(f"    Mean hierarchy score: {mean_hierarchy_score:.3f}")
            print(f"    Mean processing time: {stats['mean_processing_time_ms']:.2f} ms")

            # Save results with hierarchy validation and circular validation scores
            rows = []
            for idx, (v, h) in enumerate(zip(validation_results, hierarchy_validations)):
                cv_score = circular_validation_scores[idx] if idx < len(circular_validation_scores) else 0.0
                rows.append({
                    'spectrum_id': v.spectrum_id,
                    'true_sequence': v.true_sequence,
                    'predicted_sequence': v.predicted_sequence,
                    'match': v.sequence_match,
                    'partial_score': v.partial_match_score,
                    'circular_validation_score': cv_score,
                    'mass_error': v.mass_error,
                    'path_length': v.stellas_path_length,
                    'tortuosity': v.stellas_tortuosity,
                    'mean_s_entropy': v.mean_s_entropy,
                    'n_fragments': v.n_fragments,
                    'processing_time_ms': v.processing_time * 1000,
                    'hierarchy_overlap': h['overlap_score'],
                    'hierarchy_wavelength': h['wavelength_ratio'],
                    'hierarchy_energy': h['energy_ratio'],
                    'hierarchy_phase': h['phase_coherence'],
                    'hierarchy_valid': h['is_valid'],
                    'hierarchy_score': h['overall_score']
                })

            df = pd.DataFrame(rows)
            df.to_csv(self.output_dir / 'sequence_reconstruction.csv', index=False)
            print(f"\n  Saved: sequence_reconstruction.csv")

            self.statistics['reconstruction'] = stats

            return stats

        return {}

    def _compute_partial_match(self, predicted: str, true: str) -> float:
        """Compute partial match score using longest common subsequence."""
        if not predicted or not true:
            return 0.0

        m, n = len(predicted), len(true)

        # LCS dynamic programming
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if predicted[i-1].upper() == true[j-1].upper():
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        lcs_length = dp[m][n]

        # Normalize by true sequence length
        return lcs_length / len(true) if len(true) > 0 else 0.0

    def run_bijective_validation(self, max_spectra: int = 200) -> Dict:
        """
        Run bijective validation using ion-to-droplet transformation.

        This is CRITICAL for grounding the S-Entropy framework in physical
        reality through computer vision validation.

        Args:
            max_spectra: Maximum spectra to validate

        Returns:
            Statistics dictionary
        """
        print("\n" + "=" * 80)
        print("BIJECTIVE ION-TO-DROPLET VALIDATION")
        print("=" * 80)

        try:
            from bijective_validation import BijectiveProteomicsValidator
        except ImportError:
            print("  Warning: Bijective validation not available")
            return {}

        # Initialize validator
        validator = BijectiveProteomicsValidator(
            resolution=(256, 256),  # Smaller for speed
            enable_physics_validation=True,
            physics_threshold=0.3,
            enable_reconstruction_check=True
        )

        # Prepare spectra for validation
        spectra_to_validate = self.spectra[:max_spectra]
        validation_results = []

        start_time = time.time()

        for i, spectrum in enumerate(spectra_to_validate):
            if i % 50 == 0:
                print(f"  Validating spectrum {i+1}/{len(spectra_to_validate)}...")

            # Get ground truth sequence if available
            match = re.search(r'spectrum=(\d+)', spectrum.title)
            sequence = None
            if match:
                spectrum_key = match.group(1)
                if spectrum_key in self.psms:
                    sequence = self.psms[spectrum_key].sequence

            result = validator.validate_spectrum(
                spectrum_id=spectrum.title,
                mzs=spectrum.mz_array,
                intensities=spectrum.intensity_array,
                rt=spectrum.retention_time,
                peptide_sequence=sequence
            )
            validation_results.append(result)

            # Add to reference library if bijective
            if result.is_bijective and result.physics_quality_mean >= 0.3:
                validator.add_reference_spectrum(
                    spectrum_id=spectrum.title,
                    mzs=spectrum.mz_array,
                    intensities=spectrum.intensity_array,
                    metadata={'sequence': sequence}
                )

        elapsed = time.time() - start_time

        # Statistics
        n_valid = sum(1 for r in validation_results if r.n_valid_droplets > 0)
        n_bijective = sum(1 for r in validation_results if r.is_bijective)
        mean_physics = np.mean([r.physics_quality_mean for r in validation_results])
        mean_recon_error = np.mean([r.reconstruction_error for r in validation_results])

        stats = {
            'n_validated': len(validation_results),
            'n_valid_droplets': n_valid,
            'n_bijective': n_bijective,
            'bijective_rate': n_bijective / len(validation_results) * 100,
            'mean_physics_quality': float(mean_physics),
            'mean_reconstruction_error': float(mean_recon_error),
            'reference_library_size': len(validator.reference_library),
            'processing_time_s': elapsed
        }

        print(f"\n  Validation Results:")
        print(f"    Total validated: {stats['n_validated']}")
        print(f"    Bijective: {stats['n_bijective']} ({stats['bijective_rate']:.1f}%)")
        print(f"    Mean physics quality: {stats['mean_physics_quality']:.4f}")
        print(f"    Mean reconstruction error: {stats['mean_reconstruction_error']:.6f}")
        print(f"    Reference library size: {stats['reference_library_size']}")
        print(f"    Processing time: {elapsed:.2f}s")

        # Save detailed results
        rows = []
        for r in validation_results:
            rows.append({
                'spectrum_id': r.spectrum_id,
                'n_ions': r.n_ions,
                'n_valid_droplets': r.n_valid_droplets,
                'physics_quality_mean': r.physics_quality_mean,
                's_knowledge_mean': r.s_knowledge_mean,
                's_time_mean': r.s_time_mean,
                's_entropy_mean': r.s_entropy_mean,
                'velocity_mean': r.velocity_mean,
                'radius_mean': r.radius_mean,
                'phase_coherence_mean': r.phase_coherence_mean,
                'reconstruction_error': r.reconstruction_error,
                'is_bijective': r.is_bijective,
                'ion_validation_score': r.ion_validation_score,
                'droplet_validation_score': r.droplet_validation_score,
                'energy_conservation_score': r.energy_conservation_score
            })

        df = pd.DataFrame(rows)
        df.to_csv(self.output_dir / 'bijective_validation.csv', index=False)
        print(f"\n  Saved: bijective_validation.csv")

        # Save report
        report = validator.get_validation_report()
        with open(self.output_dir / 'bijective_validation_report.txt', 'w') as f:
            f.write(report)
        print(f"  Saved: bijective_validation_report.txt")

        self.statistics['bijective'] = stats

        return stats

    def run_charge_validation(self, max_spectra: int = 200) -> Dict:
        """
        Run charge localization and redistribution validation.

        Validates the mobile proton model for peptide fragmentation:
        - Charges localize on basic residues (K, R, H) and termini
        - Fragment charge density affects S-entropy
        - Total fragment charge ≤ parent charge (conservation)

        Args:
            max_spectra: Maximum spectra to validate

        Returns:
            Statistics dictionary
        """
        print("\n" + "=" * 80)
        print("CHARGE LOCALIZATION & REDISTRIBUTION VALIDATION")
        print("=" * 80)

        if not STATE_COUNTING_AVAILABLE:
            print("  Warning: State counting module not available")
            return {}

        # Import locally to avoid issues
        from state_counting import validate_charge_redistribution

        validation_results = []
        spectra_with_psms = []

        # Filter spectra with known sequences (need sequence for charge analysis)
        for spectrum in self.spectra[:max_spectra * 3]:
            match = re.search(r'spectrum=(\d+)', spectrum.title)
            if match:
                spectrum_key = match.group(1)
                if spectrum_key in self.psms:
                    spectra_with_psms.append((spectrum, self.psms[spectrum_key]))
                    if len(spectra_with_psms) >= max_spectra:
                        break

        print(f"\n  Found {len(spectra_with_psms)} spectra with sequences for charge validation")

        for i, (spectrum, psm) in enumerate(spectra_with_psms):
            if i % 50 == 0:
                print(f"  Validating spectrum {i+1}/{len(spectra_with_psms)}...")

            # Compute parent S-entropy
            parent_sentropy = self.sentropy_engine.process_spectrum(
                mz_array=spectrum.mz_array,
                intensity_array=spectrum.intensity_array,
                precursor_mz=spectrum.precursor_mz,
                precursor_charge=spectrum.charge,
                spectrum_id=spectrum.title,
                peptide_sequence=psm.sequence
            )

            # Prepare parent spectrum info
            parent_mass = (spectrum.precursor_mz * spectrum.charge) - (spectrum.charge * 1.007276)
            parent_info = {
                'charge': spectrum.charge,
                'mass': parent_mass,
                's_entropy': parent_sentropy.s_value,
                'sequence': psm.sequence
            }

            # Prepare fragment info (each peak as a potential fragment)
            fragments = []
            for j, (mz, intensity) in enumerate(zip(spectrum.mz_array, spectrum.intensity_array)):
                # Simple S-entropy estimate for fragment
                s_entropy = np.log1p(intensity) / 10.0 * (mz / parent_mass)
                fragments.append({
                    'id': f'frag_{j}',
                    'mz': mz,
                    'mass': mz,  # Assume singly charged for fragments
                    's_entropy': s_entropy,
                    'sequence': ''  # Fragment sequences not known
                })

            # Validate charge redistribution
            charge_result = validate_charge_redistribution(parent_info, fragments)

            validation_results.append({
                'spectrum_id': spectrum.title,
                'parent_charge': charge_result.parent_charge,
                'parent_mass': charge_result.parent_mass,
                'parent_charge_density': charge_result.parent_charge_density,
                'n_fragments': len(fragments),
                'total_fragment_charge': charge_result.total_fragment_charge,
                'charge_conserved': charge_result.charge_conserved,
                'charge_balance': charge_result.charge_balance,
                'overall_valid': charge_result.overall_valid,
                'mean_redistribution_factor': np.mean([f.redistribution_factor for f in charge_result.fragments]) if charge_result.fragments else 0.0,
                'mean_s_entropy_error': np.mean([f.s_entropy_error for f in charge_result.fragments]) if charge_result.fragments else 0.0
            })

        # Compute statistics
        if validation_results:
            df = pd.DataFrame(validation_results)

            n_charge_conserved = df['charge_conserved'].sum()
            n_overall_valid = df['overall_valid'].sum()

            stats = {
                'n_validated': len(df),
                'n_charge_conserved': int(n_charge_conserved),
                'charge_conserved_percent': n_charge_conserved / len(df) * 100,
                'n_overall_valid': int(n_overall_valid),
                'overall_valid_percent': n_overall_valid / len(df) * 100,
                'mean_charge_balance': float(df['charge_balance'].mean()),
                'mean_redistribution_factor': float(df['mean_redistribution_factor'].mean()),
                'mean_s_entropy_error': float(df['mean_s_entropy_error'].mean())
            }

            print(f"\n  Charge Validation Results:")
            print(f"    Total validated: {stats['n_validated']}")
            print(f"    Charge conserved: {stats['n_charge_conserved']} ({stats['charge_conserved_percent']:.1f}%)")
            print(f"    Overall valid: {stats['n_overall_valid']} ({stats['overall_valid_percent']:.1f}%)")
            print(f"    Mean charge balance: {stats['mean_charge_balance']:.3f}")
            print(f"    Mean redistribution factor: {stats['mean_redistribution_factor']:.3f}")
            print(f"    Mean S-entropy error: {stats['mean_s_entropy_error']:.4f}")

            df.to_csv(self.output_dir / 'charge_validation.csv', index=False)
            print(f"\n  Saved: charge_validation.csv")

            self.statistics['charge'] = stats
            return stats

        return {}

    def run_sentropy_encoding(self, max_spectra: int = 500) -> Dict:
        """
        Run S-Entropy coordinate encoding on spectra.

        Args:
            max_spectra: Maximum spectra to encode

        Returns:
            Statistics dictionary
        """
        print("\n" + "=" * 80)
        print("S-ENTROPY COORDINATE ENCODING")
        print("=" * 80)

        encoded_spectra = []
        spectra_to_encode = self.spectra[:max_spectra]

        for i, spectrum in enumerate(spectra_to_encode):
            if i % 100 == 0:
                print(f"  Encoding spectrum {i+1}/{len(spectra_to_encode)}...")

            # Get sequence if available
            match = re.search(r'spectrum=(\d+)', spectrum.title)
            peptide_seq = None
            if match:
                spectrum_key = match.group(1)
                if spectrum_key in self.psms:
                    peptide_seq = self.psms[spectrum_key].sequence

            # Encode with S-Entropy engine
            sentropy_spectrum = self.sentropy_engine.process_spectrum(
                mz_array=spectrum.mz_array,
                intensity_array=spectrum.intensity_array,
                precursor_mz=spectrum.precursor_mz,
                precursor_charge=spectrum.charge,
                spectrum_id=spectrum.title,
                peptide_sequence=peptide_seq,
                retention_time=spectrum.retention_time
            )

            encoded_spectra.append(sentropy_spectrum)

        print(f"\n  Encoded {len(encoded_spectra)} spectra")

        # Compute statistics
        if encoded_spectra:
            s_values = [s.s_value for s in encoded_spectra]

            stats = {
                'n_encoded': len(encoded_spectra),
                'mean_s_value': float(np.mean(s_values)),
                'std_s_value': float(np.std(s_values)),
                'mean_processing_time_ms': float(np.mean([s.processing_time for s in encoded_spectra]) * 1000)
            }

            print(f"\n  Statistics:")
            print(f"    Mean S-value: {stats['mean_s_value']:.4f}")
            print(f"    Std S-value: {stats['std_s_value']:.4f}")
            print(f"    Mean encoding time: {stats['mean_processing_time_ms']:.4f} ms")

            # Save 14D features
            feature_matrix = np.vstack([s.sentropy_features_14d for s in encoded_spectra])
            feature_df = pd.DataFrame(
                feature_matrix,
                columns=[f'feature_{i}' for i in range(14)]
            )
            feature_df['spectrum_id'] = [s.spectrum_id for s in encoded_spectra]
            feature_df['s_value'] = s_values

            feature_df.to_csv(self.output_dir / 'sentropy_features.csv', index=False)
            print(f"\n  Saved: sentropy_features.csv")

            self.statistics['sentropy'] = stats

            return stats

        return {}

    # ========================================================================
    # VISUALIZATION
    # ========================================================================

    def generate_figures(self):
        """Generate validation and analysis figures."""
        print("\n" + "=" * 80)
        print("GENERATING FIGURES")
        print("=" * 80)

        figures_dir = self.output_dir / 'figures'
        figures_dir.mkdir(exist_ok=True)

        # Figure 1: St. Stella's transformation statistics
        self._plot_stellas_statistics(figures_dir)

        # Figure 2: Fragment graph statistics
        self._plot_fragment_statistics(figures_dir)

        # Figure 3: Validation results
        self._plot_validation_results(figures_dir)

        # Figure 4: S-Entropy feature distribution
        self._plot_sentropy_distribution(figures_dir)

        # Figure 5: Bijective validation (CRITICAL)
        self._plot_bijective_validation(figures_dir)

        # Figure 6: Charge validation
        self._plot_charge_validation(figures_dir)

        print(f"\n  Saved figures to: {figures_dir}")

    def _plot_stellas_statistics(self, figures_dir: Path):
        """Plot St. Stella's transformation statistics."""
        csv_path = self.output_dir / 'stellas_transformations.csv'
        if not csv_path.exists():
            return

        df = pd.read_csv(csv_path)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Path length vs sequence length
        axes[0, 0].scatter(df['length'], df['path_length'], alpha=0.6)
        axes[0, 0].set_xlabel('Peptide Length')
        axes[0, 0].set_ylabel('St. Stella\'s Path Length')
        axes[0, 0].set_title('Path Length vs Sequence Length')
        axes[0, 0].grid(True, alpha=0.3)

        # Tortuosity distribution
        axes[0, 1].hist(df['tortuosity'], bins=30, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Tortuosity')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Tortuosity Distribution')
        axes[0, 1].grid(True, alpha=0.3)

        # S-Entropy components
        axes[1, 0].scatter(df['mean_s_knowledge'], df['mean_s_entropy'], alpha=0.6)
        axes[1, 0].set_xlabel('Mean S_knowledge')
        axes[1, 0].set_ylabel('Mean S_entropy')
        axes[1, 0].set_title('S-Entropy Components')
        axes[1, 0].grid(True, alpha=0.3)

        # Length distribution
        axes[1, 1].hist(df['length'], bins=20, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Peptide Length')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Peptide Length Distribution')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(figures_dir / 'stellas_statistics.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Saved: stellas_statistics.png")

    def _plot_fragment_statistics(self, figures_dir: Path):
        """Plot fragment graph statistics."""
        csv_path = self.output_dir / 'fragment_analysis.csv'
        if not csv_path.exists():
            return

        df = pd.read_csv(csv_path)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Fragments distribution
        axes[0, 0].hist(df['n_fragments'], bins=30, alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Number of Fragments')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Fragment Count Distribution')
        axes[0, 0].grid(True, alpha=0.3)

        # Nodes vs Edges
        axes[0, 1].scatter(df['n_nodes'], df['n_edges'], alpha=0.6)
        axes[0, 1].set_xlabel('Graph Nodes')
        axes[0, 1].set_ylabel('Graph Edges')
        axes[0, 1].set_title('Graph Connectivity')
        axes[0, 1].grid(True, alpha=0.3)

        # Precursor m/z distribution
        axes[1, 0].hist(df['precursor_mz'], bins=30, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Precursor m/z')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Precursor m/z Distribution')
        axes[1, 0].grid(True, alpha=0.3)

        # Charge distribution
        charge_counts = df['charge'].value_counts()
        axes[1, 1].bar(charge_counts.index.astype(str), charge_counts.values, alpha=0.7)
        axes[1, 1].set_xlabel('Charge State')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Charge State Distribution')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(figures_dir / 'fragment_statistics.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Saved: fragment_statistics.png")

    def _plot_validation_results(self, figures_dir: Path):
        """Plot sequence reconstruction validation with fragment-parent hierarchy."""
        csv_path = self.output_dir / 'sequence_reconstruction.csv'
        if not csv_path.exists():
            return

        df = pd.read_csv(csv_path)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Partial match score distribution
        axes[0, 0].hist(df['partial_score'], bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Partial Match Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Reconstruction Accuracy Distribution')
        axes[0, 0].axvline(df['partial_score'].mean(), color='r', linestyle='--',
                          label=f'Mean: {df["partial_score"].mean():.2f}')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Mass error distribution
        axes[0, 1].hist(df['mass_error'], bins=30, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Mass Error (Da)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Mass Error Distribution')
        axes[0, 1].grid(True, alpha=0.3)

        # Path length vs partial score
        axes[0, 2].scatter(df['path_length'], df['partial_score'], alpha=0.6)
        axes[0, 2].set_xlabel('St. Stella\'s Path Length')
        axes[0, 2].set_ylabel('Partial Match Score')
        axes[0, 2].set_title('Path Length vs Reconstruction Accuracy')
        axes[0, 2].grid(True, alpha=0.3)

        # Fragment-Parent Hierarchy Validation (if available)
        if 'hierarchy_score' in df.columns:
            # Hierarchy score distribution
            axes[1, 0].hist(df['hierarchy_score'], bins=20, alpha=0.7, edgecolor='black', color='green')
            axes[1, 0].axvline(0.7, color='r', linestyle='--', label='Valid threshold (0.7)')
            axes[1, 0].set_xlabel('Hierarchy Score')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Fragment-Parent Hierarchy Validation')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            # Hierarchy components
            hierarchy_cols = ['hierarchy_overlap', 'hierarchy_wavelength', 'hierarchy_energy', 'hierarchy_phase']
            if all(col in df.columns for col in hierarchy_cols):
                means = [df[col].mean() for col in hierarchy_cols]
                labels = ['Overlap', 'Wavelength', 'Energy', 'Phase']
                colors = ['blue', 'green', 'orange', 'purple']
                axes[1, 1].bar(labels, means, alpha=0.7, color=colors)
                axes[1, 1].axhline(0.7, color='r', linestyle='--', label='Threshold')
                axes[1, 1].set_xlabel('Hierarchy Constraint')
                axes[1, 1].set_ylabel('Mean Score')
                axes[1, 1].set_title('Hierarchy Constraint Scores')
                axes[1, 1].set_ylim(0, 1.2)
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)

            # Hierarchy vs partial match correlation
            axes[1, 2].scatter(df['hierarchy_score'], df['partial_score'], alpha=0.6, c='green')
            axes[1, 2].set_xlabel('Hierarchy Score')
            axes[1, 2].set_ylabel('Partial Match Score')
            axes[1, 2].set_title('Hierarchy vs Reconstruction Accuracy')
            axes[1, 2].grid(True, alpha=0.3)

            # Add correlation text
            if df['hierarchy_score'].std() > 0 and df['partial_score'].std() > 0:
                corr = df['hierarchy_score'].corr(df['partial_score'])
                axes[1, 2].text(0.05, 0.95, f'r = {corr:.3f}', transform=axes[1, 2].transAxes,
                              fontsize=12, verticalalignment='top')
        else:
            # Fallback: Processing time distribution
            axes[1, 0].hist(df['processing_time_ms'], bins=30, alpha=0.7, edgecolor='black')
            axes[1, 0].set_xlabel('Processing Time (ms)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Processing Time Distribution')
            axes[1, 0].grid(True, alpha=0.3)

            # Tortuosity distribution
            axes[1, 1].hist(df['tortuosity'], bins=30, alpha=0.7, edgecolor='black')
            axes[1, 1].set_xlabel('Tortuosity')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Path Tortuosity Distribution')
            axes[1, 1].grid(True, alpha=0.3)

            # S-entropy distribution
            axes[1, 2].hist(df['mean_s_entropy'], bins=30, alpha=0.7, edgecolor='black')
            axes[1, 2].set_xlabel('Mean S-Entropy')
            axes[1, 2].set_ylabel('Frequency')
            axes[1, 2].set_title('S-Entropy Distribution')
            axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(figures_dir / 'validation_results.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Saved: validation_results.png")

    def _plot_sentropy_distribution(self, figures_dir: Path):
        """Plot S-Entropy feature distribution."""
        csv_path = self.output_dir / 'sentropy_features.csv'
        if not csv_path.exists():
            return

        df = pd.read_csv(csv_path)

        feature_cols = [c for c in df.columns if c.startswith('feature_')]

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # S-value distribution
        axes[0, 0].hist(df['s_value'], bins=30, alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('S-value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('S-value Distribution')
        axes[0, 0].grid(True, alpha=0.3)

        # First 3 features scatter
        axes[0, 1].scatter(df['feature_0'], df['feature_1'], c=df['feature_2'],
                          alpha=0.6, cmap='viridis')
        axes[0, 1].set_xlabel('Feature 0 (S_knowledge)')
        axes[0, 1].set_ylabel('Feature 1 (S_time)')
        axes[0, 1].set_title('S-Entropy Feature Space')
        axes[0, 1].grid(True, alpha=0.3)

        # Feature correlation heatmap
        if len(feature_cols) > 0:
            corr = df[feature_cols[:8]].corr()  # First 8 features
            sns.heatmap(corr, annot=False, cmap='coolwarm', center=0,
                       ax=axes[1, 0], square=True)
            axes[1, 0].set_title('Feature Correlation Matrix')

        # Feature variance
        variances = df[feature_cols].var()
        axes[1, 1].bar(range(len(variances)), variances, alpha=0.7)
        axes[1, 1].set_xlabel('Feature Index')
        axes[1, 1].set_ylabel('Variance')
        axes[1, 1].set_title('Feature Variance')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(figures_dir / 'sentropy_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Saved: sentropy_distribution.png")

    def _plot_bijective_validation(self, figures_dir: Path):
        """Plot bijective validation results (CRITICAL for grounding theory)."""
        csv_path = self.output_dir / 'bijective_validation.csv'
        if not csv_path.exists():
            return

        df = pd.read_csv(csv_path)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Physics quality distribution
        axes[0, 0].hist(df['physics_quality_mean'], bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(0.3, color='r', linestyle='--', label='Threshold (0.3)')
        axes[0, 0].set_xlabel('Physics Quality Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Physics Validation Quality')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Reconstruction error distribution
        axes[0, 1].hist(df['reconstruction_error'], bins=20, alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(0.01, color='r', linestyle='--', label='Bijective threshold (1%)')
        axes[0, 1].set_xlabel('Reconstruction Error')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Bijective Reconstruction Error')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # S-Entropy coordinates 3D visualization (as 2D scatter with color)
        scatter = axes[0, 2].scatter(
            df['s_knowledge_mean'], df['s_time_mean'],
            c=df['s_entropy_mean'], cmap='viridis', alpha=0.6
        )
        axes[0, 2].set_xlabel('S_knowledge')
        axes[0, 2].set_ylabel('S_time')
        axes[0, 2].set_title('S-Entropy Coordinates (color=S_entropy)')
        plt.colorbar(scatter, ax=axes[0, 2])
        axes[0, 2].grid(True, alpha=0.3)

        # Droplet parameters: velocity vs radius
        axes[1, 0].scatter(df['velocity_mean'], df['radius_mean'],
                          c=df['phase_coherence_mean'], cmap='plasma', alpha=0.6)
        axes[1, 0].set_xlabel('Mean Velocity (m/s)')
        axes[1, 0].set_ylabel('Mean Radius (mm)')
        axes[1, 0].set_title('Droplet Parameters (color=phase coherence)')
        axes[1, 0].grid(True, alpha=0.3)

        # Physics subscores
        subscores = ['ion_validation_score', 'droplet_validation_score', 'energy_conservation_score']
        mean_scores = [df[col].mean() for col in subscores]
        axes[1, 1].bar(['Ion', 'Droplet', 'Energy'], mean_scores, alpha=0.7, color=['blue', 'green', 'orange'])
        axes[1, 1].set_xlabel('Validation Category')
        axes[1, 1].set_ylabel('Mean Score')
        axes[1, 1].set_title('Physics Validation Breakdown')
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].grid(True, alpha=0.3)

        # Bijective vs non-bijective
        n_bijective = df['is_bijective'].sum()
        n_non_bijective = len(df) - n_bijective
        bijective_data = [n_bijective, n_non_bijective]
        bijective_labels = ['Bijective', 'Non-bijective']
        bijective_colors = ['green', 'red']

        # Filter out zero values for pie chart
        non_zero_data = [(d, l, c) for d, l, c in zip(bijective_data, bijective_labels, bijective_colors) if d > 0]
        if non_zero_data:
            data, labels, colors = zip(*non_zero_data)
            axes[1, 2].pie(data, labels=labels, autopct='%1.1f%%', colors=colors)
        else:
            axes[1, 2].text(0.5, 0.5, 'No data', ha='center', va='center')
        axes[1, 2].set_title('Bijective Property Verification')

        plt.tight_layout()
        plt.savefig(figures_dir / 'bijective_validation.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Saved: bijective_validation.png")

    def _plot_charge_validation(self, figures_dir: Path):
        """Plot charge localization and redistribution validation."""
        csv_path = self.output_dir / 'charge_validation.csv'
        if not csv_path.exists():
            return

        df = pd.read_csv(csv_path)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Charge balance distribution
        axes[0, 0].hist(df['charge_balance'], bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(1.0, color='r', linestyle='--', label='Conservation (1.0)')
        axes[0, 0].set_xlabel('Charge Balance (Fragment/Parent)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Charge Balance Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Redistribution factor distribution
        axes[0, 1].hist(df['mean_redistribution_factor'], bins=20, alpha=0.7, edgecolor='black', color='green')
        axes[0, 1].axvline(1.0, color='r', linestyle='--', label='No redistribution (1.0)')
        axes[0, 1].set_xlabel('Mean Redistribution Factor (C_i)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Charge Redistribution Factor')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # S-entropy error distribution
        axes[0, 2].hist(df['mean_s_entropy_error'], bins=20, alpha=0.7, edgecolor='black', color='orange')
        axes[0, 2].axvline(0.2, color='r', linestyle='--', label='Valid threshold (0.2)')
        axes[0, 2].set_xlabel('Mean S-Entropy Error')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('S-Entropy Prediction Error')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # Charge density vs redistribution factor
        axes[1, 0].scatter(df['parent_charge_density'], df['mean_redistribution_factor'], alpha=0.6)
        axes[1, 0].set_xlabel('Parent Charge Density')
        axes[1, 0].set_ylabel('Mean Redistribution Factor')
        axes[1, 0].set_title('Charge Density vs Redistribution')
        axes[1, 0].grid(True, alpha=0.3)

        # Conservation pie chart
        n_conserved = df['charge_conserved'].sum()
        n_not_conserved = len(df) - n_conserved
        if n_conserved > 0 or n_not_conserved > 0:
            data = []
            labels = []
            colors = []
            if n_conserved > 0:
                data.append(n_conserved)
                labels.append('Conserved')
                colors.append('green')
            if n_not_conserved > 0:
                data.append(n_not_conserved)
                labels.append('Not conserved')
                colors.append('red')
            axes[1, 1].pie(data, labels=labels, autopct='%1.1f%%', colors=colors)
        axes[1, 1].set_title('Charge Conservation')

        # Charge balance vs S-entropy error
        axes[1, 2].scatter(df['charge_balance'], df['mean_s_entropy_error'], alpha=0.6, c='purple')
        axes[1, 2].set_xlabel('Charge Balance')
        axes[1, 2].set_ylabel('Mean S-Entropy Error')
        axes[1, 2].set_title('Charge Balance vs S-Entropy Error')
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(figures_dir / 'charge_validation.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Saved: charge_validation.png")

    # ========================================================================
    # MAIN PIPELINE
    # ========================================================================

    def run_full_pipeline(
        self,
        mgf_file: str = "PXD000001.mgf",
        mztab_file: str = "PXD000001.mztab",
        max_spectra: int = 2000
    ) -> Dict:
        """
        Run complete analysis pipeline.

        Args:
            mgf_file: MGF filename
            mztab_file: mzTab filename
            max_spectra: Maximum spectra per analysis

        Returns:
            Complete statistics dictionary
        """
        print("\n" + "=" * 80)
        print("COMPREHENSIVE PROTEOMICS ANALYSIS PIPELINE")
        print("=" * 80)
        print(f"Data directory: {self.data_dir}")
        print(f"Output directory: {self.output_dir}")

        start_time = time.time()

        # Load data
        self.load_data(mgf_file, mztab_file, max_spectra)

        # Run analyses
        self.run_stellas_transformation()
        self.run_fragment_analysis(max_spectra)
        self.run_sequence_reconstruction(min(100, max_spectra))
        self.run_sentropy_encoding(max_spectra)

        # CRITICAL: Bijective validation (grounds theory in physical reality)
        self.run_bijective_validation(min(200, max_spectra))

        # Charge localization validation (mobile proton model)
        self.run_charge_validation(min(200, max_spectra))

        # Generate figures
        self.generate_figures()

        elapsed = time.time() - start_time

        # Final summary
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETE")
        print("=" * 80)
        print(f"\nTotal time: {elapsed:.2f} seconds")

        # Save summary
        summary = {
            'data_dir': str(self.data_dir),
            'output_dir': str(self.output_dir),
            'n_spectra': len(self.spectra),
            'n_psms': len(self.psms),
            'total_time_seconds': elapsed,
            'statistics': self.statistics
        }

        with open(self.output_dir / 'pipeline_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\nSaved: pipeline_summary.json")

        return summary


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run the proteomics analysis pipeline on downloaded data."""
    # Paths
    data_dir = Path(__file__).parent.parent.parent.parent / 'public' / 'proteomics'
    output_dir = Path(__file__).parent.parent.parent.parent / 'results' / 'proteomics_analysis'

    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")

    # Create data directory if needed
    data_dir.mkdir(parents=True, exist_ok=True)

    # Run pipeline
    pipeline = ProteomicsPipelineRunner(str(data_dir), str(output_dir))

    # Priority 1: Try to load mzML files using YOUR SpectraReader
    mzml_files = list(data_dir.glob('*.mzML')) + list(data_dir.glob('*.mzml'))
    if mzml_files and SPECTRA_READER_AVAILABLE:
        print(f"Found mzML files: {[f.name for f in mzml_files]}")
        print("\n*** Using YOUR SpectraReader.extract_mzml() ***\n")

        mzml_file = mzml_files[0].name
        mztab_file = mzml_file.replace('.mzML', '.mztab').replace('.mzml', '.mztab')

        # Load using YOUR SpectraReader
        pipeline.load_data_mzml(
            mzml_file=mzml_file,
            mztab_file=mztab_file if (data_dir / mztab_file).exists() else None,
            rt_range=[0.0, 100.0],
            max_spectra=2000,
            vendor="thermo"
        )

    # Priority 2: Fall back to MGF files
    else:
        mgf_files = list(data_dir.glob('*.mgf'))
        if mgf_files:
            print(f"Found MGF files: {[f.name for f in mgf_files]}")
            mgf_file = mgf_files[0].name
            mztab_file = mgf_file.replace('.mgf', '.mztab')

            pipeline.load_data(
                mgf_file=mgf_file,
                mztab_file=mztab_file,
                max_spectra=2000
            )
        else:
            # Priority 3: Download from PRIDE
            print("No local data found. Downloading from PRIDE...")
            pipeline.download_and_load_from_pride(
                project_accession="PXD000001",
                max_files=1,
                rt_range=[0.0, 100.0],
                max_spectra=2000
            )

    # Run analyses
    pipeline.run_stellas_transformation()
    pipeline.run_fragment_analysis(500)
    pipeline.run_sequence_reconstruction(100, use_circular_validation=True)
    pipeline.run_sentropy_encoding(500)
    pipeline.run_bijective_validation(200)
    pipeline.run_charge_validation(200)

    # Generate figures
    pipeline.generate_figures()

    # Save summary
    summary = {
        'data_dir': str(data_dir),
        'output_dir': str(output_dir),
        'n_spectra': len(pipeline.spectra),
        'n_psms': len(pipeline.psms),
        'statistics': pipeline.statistics
    }

    with open(output_dir / 'pipeline_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(json.dumps(summary, indent=2, default=str))


def main_pride_download():
    """
    Example: Download from PRIDE and run pipeline.

    This demonstrates how to download mzML from PRIDE
    and use YOUR existing SpectraReader for parsing.
    """
    data_dir = Path(__file__).parent.parent.parent.parent / 'public' / 'proteomics'
    output_dir = Path(__file__).parent.parent.parent.parent / 'results' / 'proteomics_pride'

    print("\n" + "=" * 80)
    print("PRIDE DOWNLOAD + YOUR SPECTRA READER DEMO")
    print("=" * 80)

    pipeline = ProteomicsPipelineRunner(str(data_dir), str(output_dir))

    # Download from PRIDE and load using YOUR SpectraReader
    pipeline.download_and_load_from_pride(
        project_accession="PXD000001",  # Classic TMT dataset
        max_files=1,
        rt_range=[0.0, 60.0],  # First 60 minutes
        max_spectra=1000,
        vendor="thermo"
    )

    print(f"\nLoaded {len(pipeline.spectra)} spectra using YOUR SpectraReader")

    # Run validation
    pipeline.run_sequence_reconstruction(50, use_circular_validation=True)

    print("\nDone! Check results in:", output_dir)


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--pride':
        main_pride_download()
    else:
        main()
