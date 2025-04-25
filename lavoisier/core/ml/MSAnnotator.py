# annotation.py
from dataclasses import dataclass
from io import StringIO
from typing import List, Dict, Optional, Union, Tuple, Any
import pandas as pd
import numpy as np
import ray
import logging
import pubchempy as pcp
from spec2vec import Spec2Vec, SpectrumDocument
from gensim.models.basemodel import BaseTopicModel
import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder
import dask.dataframe as dd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from matchms.similarity import CosineGreedy, ModifiedCosine
import tensorflow as tf
from scipy import signal
import json
import os


@dataclass
class AnnotationParameters:
    """Parameters for annotation pipeline"""
    ms1_ppm_tolerance: float = 5.0
    ms2_ppm_tolerance: float = 10.0
    rt_tolerance: float = 0.5
    min_intensity: float = 500.0
    batch_size: int = 500
    n_workers: int = -1  # Use all available CPUs
    lipidmaps_url: str = 'http://lipidmaps-dev.babraham.ac.uk/tools/ms/py_bulk_search.php'
    mslipids_url: str = 'http://mslipids.org/api/search'
    pubchem_batch_size: int = 100
    spec2vec_intensity_power: float = 0.5
    spec2vec_allowed_missing: float = 0.0
    adducts_positive: List[str] = None
    adducts_negative: List[str] = None
    
    # Additional database URLs
    hmdb_url: str = 'https://hmdb.ca/api'
    metlin_url: str = 'https://metlin.scripps.edu/rest/api'
    massbank_url: str = 'https://massbank.eu/rest/spectra'
    mzcloud_url: str = 'https://mzcloud.org/api'
    kegg_url: str = 'https://rest.kegg.jp'
    humancyc_url: str = 'https://humancyc.org/api'
    
    # API keys (should be provided by the user)
    metlin_api_key: str = ''
    mzcloud_api_key: str = ''
    
    # Score weights for annotation
    mass_error_weight: float = 0.3
    rt_match_weight: float = 0.2
    spectral_similarity_weight: float = 0.3
    isotope_pattern_weight: float = 0.1
    database_score_weight: float = 0.1
    fragmentation_tree_weight: float = 0.2
    
    # Similarity thresholds
    cosine_similarity_threshold: float = 0.7
    modified_cosine_threshold: float = 0.6
    
    # Deep learning model paths
    fragmentation_model_path: str = ''
    structure_prediction_model_path: str = ''
    
    # Spectral library paths
    massbank_library_path: str = ''
    metlin_library_path: str = ''
    mzcloud_library_path: str = ''
    hmdb_library_path: str = ''
    in_house_library_path: str = ''
    
    # Pathway database cache paths
    kegg_cache_path: str = ''
    humancyc_cache_path: str = ''
    
    # Feature flags to enable/disable different search types
    enable_spectral_matching: bool = True
    enable_accurate_mass: bool = True
    enable_pathway_search: bool = True
    enable_fragmentation_prediction: bool = True
    enable_deep_learning: bool = True

    def __post_init__(self):
        if self.adducts_positive is None:
            self.adducts_positive = ['[M+H]+', '[M+Na]+', '[M+K]+', '[M+NH4]+']
        if self.adducts_negative is None:
            self.adducts_negative = ['[M-H]-', '[M+Cl]-', '[M+HCOO]-']


class MSAnnotator:
    """Unified MS annotation pipeline combining multiple databases and approaches for comprehensive annotation"""

    def __init__(self, params: AnnotationParameters,
                 model: Optional[BaseTopicModel] = None,
                 rt_model_path: Optional[str] = None,
                 library_path: Optional[str] = None,
                 deep_learning_model_path: Optional[str] = None):
        self.params = params
        self.spec2vec_model = model
        ray.init(ignore_reinit_error=True)
        self.logger = logging.getLogger(__name__)

        # Initialize similarity calculators
        self.cosine_similarity = CosineGreedy(
            tolerance=self.params.ms2_ppm_tolerance,
            mz_power=0.0,
            intensity_power=self.params.spec2vec_intensity_power
        )

        self.modcos_similarity = ModifiedCosine(
            tolerance=self.params.ms2_ppm_tolerance,
            mz_power=0.0,
            intensity_power=self.params.spec2vec_intensity_power
        )

        # Load RT prediction model and spectral library
        self.rt_model = self._load_rt_model(rt_model_path)
        self.spectral_library = self._load_spectral_library(library_path)
        
        # Load additional spectral libraries
        self.massbank_library = self._load_spectral_library(self.params.massbank_library_path)
        self.metlin_library = self._load_spectral_library(self.params.metlin_library_path)
        self.mzcloud_library = self._load_spectral_library(self.params.mzcloud_library_path)
        self.hmdb_library = self._load_spectral_library(self.params.hmdb_library_path)
        self.in_house_library = self._load_spectral_library(self.params.in_house_library_path)
        
        # Load pathway databases
        self.kegg_database = self._load_pathway_database(self.params.kegg_cache_path)
        self.humancyc_database = self._load_pathway_database(self.params.humancyc_cache_path)
        
        # Load deep learning models
        self.fragmentation_model = self._load_fragmentation_model(self.params.fragmentation_model_path)
        self.structure_prediction_model = self._load_structure_prediction_model(self.params.structure_prediction_model_path)
        self.deep_learning_model = self._load_deep_learning_model(deep_learning_model_path)
        
        # Initialize session keys/tokens for APIs
        self._initialize_api_keys()
        
        # Cache for database queries
        self.query_cache = {}

    def _initialize_api_keys(self):
        """Initialize API connections and validate keys"""
        self.api_keys_valid = {}
        
        # Validate and store status of each API connection
        if self.params.metlin_api_key:
            self.api_keys_valid["metlin"] = self._validate_metlin_api()
        
        if self.params.mzcloud_api_key:
            self.api_keys_valid["mzcloud"] = self._validate_mzcloud_api()
            
        # Test HMDB connection
        try:
            response = requests.get(f"{self.params.hmdb_url}/metabolites", timeout=5)
            self.api_keys_valid["hmdb"] = response.status_code == 200
        except:
            self.api_keys_valid["hmdb"] = False
            self.logger.warning("Could not connect to HMDB")
            
        # Test KEGG connection
        try:
            response = requests.get(f"{self.params.kegg_url}/list/compound", timeout=5)
            self.api_keys_valid["kegg"] = response.status_code == 200
        except:
            self.api_keys_valid["kegg"] = False
            self.logger.warning("Could not connect to KEGG")
            
        # Test HumanCyc connection
        try:
            response = requests.get(f"{self.params.humancyc_url}/status", timeout=5)
            self.api_keys_valid["humancyc"] = response.status_code == 200
        except:
            self.api_keys_valid["humancyc"] = False
            self.logger.warning("Could not connect to HumanCyc")
            
        # Log status of API connections
        self.logger.info(f"API connections status: {self.api_keys_valid}")

    def _validate_metlin_api(self):
        """Validate Metlin API key"""
        try:
            headers = {"x-api-key": self.params.metlin_api_key}
            response = requests.get(f"{self.params.metlin_url}/status", headers=headers, timeout=5)
            if response.status_code == 200:
                return True
            self.logger.warning(f"Invalid Metlin API key: {response.status_code}")
            return False
        except Exception as e:
            self.logger.warning(f"Error validating Metlin API: {str(e)}")
            return False

    def _validate_mzcloud_api(self):
        """Validate MzCloud API key"""
        try:
            headers = {"x-api-key": self.params.mzcloud_api_key}
            response = requests.get(f"{self.params.mzcloud_url}/status", headers=headers, timeout=5)
            if response.status_code == 200:
                return True
            self.logger.warning(f"Invalid MzCloud API key: {response.status_code}")
            return False
        except Exception as e:
            self.logger.warning(f"Error validating MzCloud API: {str(e)}")
            return False

    def _load_deep_learning_model(self, model_path: Optional[str]) -> Optional[tf.keras.Model]:
        """Load pre-trained deep learning model for MS/MS prediction"""
        if model_path and os.path.exists(model_path):
            try:
                return tf.keras.models.load_model(model_path)
            except Exception as e:
                self.logger.error(f"Error loading deep learning model: {e}")
        return None

    def _load_fragmentation_model(self, model_path: Optional[str]) -> Optional[object]:
        """Load fragmentation prediction model"""
        if model_path and os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    import pickle
                    return pickle.load(f)
            except Exception as e:
                self.logger.error(f"Error loading fragmentation model: {e}")
        return None

    def _load_structure_prediction_model(self, model_path: Optional[str]) -> Optional[object]:
        """Load structure prediction model"""
        if model_path and os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    import pickle
                    return pickle.load(f)
            except Exception as e:
                self.logger.error(f"Error loading structure prediction model: {e}")
        return None
        
    def _load_pathway_database(self, database_path: Optional[str]) -> Dict:
        """Load pathway database from file"""
        if database_path and os.path.exists(database_path):
            try:
                with open(database_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading pathway database: {e}")
        return {}

    def _load_rt_model(self, model_path: Optional[str]) -> Optional[tf.keras.Model]:
        """Load pre-trained RT prediction model"""
        if model_path and os.path.exists(model_path):
            try:
                return tf.keras.models.load_model(model_path)
            except Exception as e:
                self.logger.error(f"Error loading RT model: {e}")
        return None

    def _load_spectral_library(self, library_path: Optional[str]) -> Dict:
        """Load MS/MS spectral library"""
        if library_path and os.path.exists(library_path):
            try:
                with open(library_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading spectral library: {e}")
        return {}

    @ray.remote
    def _process_lipidmaps_batch(self, mz_batch: List[float], adducts: str) -> pd.DataFrame:
        """Process a batch of m/z values against LipidMaps database"""
        mz_str = '\n'.join(map(str, mz_batch))
        tolerance = mz_batch[-1] * (self.params.ms1_ppm_tolerance / 1e6)

        categories = ["Fatty Acyls [FA]", "Glycerolipids [GL]",
                      "Glycerophospholipids [GP]", "Sphingolipids [SP]",
                      "Sterol Lipids [ST]", "Prenol Lipids [PR]",
                      "Saccharolipids [SL]", "Polyketides [PK]"]

        mp_data = MultipartEncoder(
            fields={
                'CHOICE': 'COMP_DB',
                'sort': 'DELTA',
                'file': ('file', StringIO(mz_str), 'text/plain'),
                'tol': str(tolerance),
                'ion': adducts,
                'category': ','.join(categories)
            }
        )

        try:
            response = requests.post(
                self.params.lipidmaps_url,
                data=mp_data,
                headers={'Content-Type': mp_data.content_type}
            )
            return pd.read_csv(StringIO(response.text), sep='\t', engine='python')
        except Exception as e:
            self.logger.error(f"Error in LipidMaps batch processing: {e}")
            return pd.DataFrame()

    @ray.remote
    def _process_mslipids_batch(self, mz_batch: List[float], ms2_data: List[Dict]) -> pd.DataFrame:
        """Process a batch of m/z values against MSLipids database"""
        try:
            payload = {
                'mz_values': mz_batch,
                'ms2_data': ms2_data,
                'tolerance_ppm': self.params.ms1_ppm_tolerance
            }

            response = requests.post(
                self.params.mslipids_url,
                json=payload,
                headers={'Content-Type': 'application/json'}
            )

            if response.status_code == 200:
                return pd.DataFrame(response.json()['results'])
            else:
                self.logger.error(f"MSLipids API error: {response.status_code}")
                return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Error in MSLipids batch processing: {e}")
            return pd.DataFrame()

    @ray.remote
    def _process_pubchem_batch(self,
                               specs: List[SpectrumDocument],
                               compound_names: List[str]) -> Dict:
        """Process a batch of spectra against PubChem"""
        results = {}
        for spec, name in zip(specs, compound_names):
            try:
                pubchem_results = pcp.get_compounds(name, 'name',
                                                    listkey_count=self.params.pubchem_batch_size)
                if pubchem_results:
                    results[spec] = pubchem_results[0]
            except Exception as e:
                self.logger.error(f"Error in PubChem processing for {name}: {e}")
        return results

    def _parallel_spec2vec_similarity(self,
                                      query_docs: List[SpectrumDocument],
                                      library_docs: List[SpectrumDocument]) -> np.ndarray:
        """Calculate Spec2Vec similarities in parallel"""
        spec2vec = Spec2Vec(
            model=self.spec2vec_model,
            intensity_weighting_power=self.params.spec2vec_intensity_power,
            allowed_missing_percentage=self.params.spec2vec_allowed_missing
        )

        # Split into chunks for parallel processing
        chunks = np.array_split(query_docs, self.params.n_workers)

        @ray.remote
        def process_chunk(chunk):
            return spec2vec.matrix(library_docs, chunk)

        # Process chunks in parallel
        futures = [process_chunk.remote(chunk) for chunk in chunks]
        results = ray.get(futures)

        # Combine results
        return np.vstack(results)

    def predict_rt(self, smiles: str) -> Dict[str, float]:
        """Predict retention time for a compound using loaded model"""
        if not self.rt_model:
            return {'predicted_rt': None, 'confidence': 0.0}

        try:
            # Generate molecular descriptors using RDKit
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {'predicted_rt': None, 'confidence': 0.0}

            descriptors = []
            for desc_name in ['MolLogP', 'MolWt', 'TPSA', 'NumRotatableBonds']:
                desc_value = getattr(Descriptors, desc_name)(mol)
                descriptors.append(desc_value)

            # Make prediction
            features = np.array(descriptors).reshape(1, -1)
            prediction = self.rt_model.predict(features)

            # Calculate confidence based on prediction uncertainty
            confidence = 1.0 - np.std(prediction) / np.mean(prediction)

            return {
                'predicted_rt': float(np.mean(prediction)),
                'confidence': float(confidence)
            }
        except Exception as e:
            self.logger.error(f"Error in RT prediction: {e}")
            return {'predicted_rt': None, 'confidence': 0.0}

    def search_spectral_library(self, spectrum: Dict) -> List[Dict]:
        """Search spectrum against loaded spectral library"""
        if not self.spectral_library:
            return []

        matches = []
        query_peaks = np.array([(p['mz'], p['intensity'])
                                for p in spectrum['peaks']])

        for lib_id, lib_spectrum in self.spectral_library.items():
            lib_peaks = np.array([(p['mz'], p['intensity'])
                                  for p in lib_spectrum['peaks']])

            # Calculate different similarity scores
            cosine_score = self.cosine_similarity.pair(query_peaks, lib_peaks)
            modcos_score = self.modcos_similarity.pair(query_peaks, lib_peaks)

            if cosine_score > 0.7 or modcos_score > 0.6:  # Adjustable thresholds
                matches.append({
                    'library_id': lib_id,
                    'compound_name': lib_spectrum.get('name', ''),
                    'cosine_similarity': float(cosine_score),
                    'modcos_similarity': float(modcos_score),
                    'smiles': lib_spectrum.get('smiles', ''),
                    'inchikey': lib_spectrum.get('inchikey', ''),
                    'source': 'spectral_library'
                })

        return sorted(matches,
                      key=lambda x: max(x['cosine_similarity'], x['modcos_similarity']),
                      reverse=True)

    def _process_ms2_spectrum(self, spectrum: Dict) -> np.ndarray:
        """Process MS2 spectrum for improved matching"""
        peaks = np.array([(p['mz'], p['intensity']) for p in spectrum['peaks']])

        # Sort by m/z
        peaks = peaks[peaks[:, 0].argsort()]

        # Noise filtering
        intensity_threshold = np.percentile(peaks[:, 1], 1)
        peaks = peaks[peaks[:, 1] > intensity_threshold]

        # Peak deconvolution using continuous wavelet transform
        widths = np.arange(1, 10)
        peaks_idx = signal.find_peaks_cwt(peaks[:, 1], widths)

        return peaks[peaks_idx]

    def _calculate_isotope_pattern(self, formula: str) -> np.ndarray:
        """Calculate theoretical isotope pattern"""
        try:
            mol = Chem.MolFromSmiles(formula)
            if mol is None:
                return np.array([])

            isotope_pattern = []
            for i in range(3):  # Calculate up to M+2
                pattern = AllChem.GetIsotopicDistribution(mol, i)
                if pattern:
                    isotope_pattern.extend([(p.mass, p.abundance) for p in pattern])

            return np.array(isotope_pattern)
        except Exception as e:
            self.logger.error(f"Error calculating isotope pattern: {e}")
            return np.array([])

    def annotate(self, spectra: List[Union[SpectrumDocument, Dict]],
                 polarity: str = 'positive') -> pd.DataFrame:
        """Main annotation method combining all approaches"""
        # Convert input to dask dataframe for better memory handling
        spec_df = dd.from_pandas(pd.DataFrame(spectra), npartitions=self.params.n_workers)

        # Process MS2 data
        ms2_data = []
        for spectrum in spectra:
            if isinstance(spectrum, dict) and 'peaks' in spectrum:
                processed_peaks = self._process_ms2_spectrum(spectrum)
                ms2_data.append({
                    'mz': spectrum.get('precursor_mz', 0),
                    'peaks': processed_peaks.tolist(),
                    'rt': spectrum.get('rt', 0)
                })

        # Determine which adducts to use based on polarity
        adducts = (self.params.adducts_positive if polarity == 'positive'
                   else self.params.adducts_negative)

        # Get unique m/z values for database searches
        mz_values = spec_df['mz'].unique().compute() if 'mz' in spec_df.columns else []
        
        # If no m/z values, try to get them from ms2_data
        if len(mz_values) == 0 and ms2_data:
            mz_values = [data['mz'] for data in ms2_data if 'mz' in data]

        # Split into batches for parallel processing
        mz_batches = [mz_values[i:i + self.params.batch_size]
                      for i in range(0, len(mz_values), self.params.batch_size)]

        # STEP 1: Launch all database searches in parallel
        search_futures = {}
        
        # Mass-based database searches
        if self.params.enable_accurate_mass:
            # LipidMaps search
            search_futures['lipidmaps'] = [
                self._process_lipidmaps_batch.remote(batch, adducts)
                for batch in mz_batches
            ]
            
            # MSLipids search with MS2 data
            search_futures['mslipids'] = [
                self._process_mslipids_batch.remote(batch, ms2_data)
                for batch in mz_batches
            ]
            
            # HMDB search
            search_futures['hmdb'] = [
                self._search_hmdb.remote(mz, self.params.ms1_ppm_tolerance)
                for mz in mz_values
            ]
            
            # KEGG search
            search_futures['kegg'] = [
                self._search_kegg.remote(mz, self.params.ms1_ppm_tolerance)
                for mz in mz_values
            ]
        
        # STEP 2: Process spectral matching in parallel
        library_search_results = []
        if self.params.enable_spectral_matching:
            # Search in spectral libraries
            for idx, spectrum in enumerate(spectra):
                if isinstance(spectrum, dict) and 'peaks' in spectrum:
                    # Combined search across all libraries
                    matches = self._search_multiple_spectral_libraries(spectrum)
                    for match in matches:
                        match['query_index'] = idx
                    library_search_results.extend(matches)
        
        # STEP 3: Process PubChem search in parallel
        pubchem_futures = []
        if self.params.enable_accurate_mass:
            # Split into batches
            spec_batches = [spectra[i:i + self.params.pubchem_batch_size]
                            for i in range(0, len(spectra), self.params.pubchem_batch_size)]
            
            # Launch PubChem searches for compound names
            compound_names = []
            for spectrum in spectra:
                name = spectrum.get('compound_name', '')
                if not name and isinstance(spectrum, SpectrumDocument):
                    name = getattr(spectrum, 'compound_name', '')
                compound_names.append(name if name else '')
            
            pubchem_futures = [
                self._process_pubchem_batch.remote(
                    batch,
                    [compound_names[i:i + self.params.pubchem_batch_size]
                     for i in range(0, len(compound_names), self.params.pubchem_batch_size)][batch_idx]
                ) for batch_idx, batch in enumerate(spec_batches)
            ]
        
        # STEP 4: Calculate spectral similarities if using spectrum embeddings
        similarities = None
        if hasattr(self, 'library_spectra') and self.spec2vec_model:
            similarities = self._parallel_spec2vec_similarity(
                spectra, self.library_spectra
            )
        
        # STEP 5: Run deep learning predictions if model is available
        dl_predictions = []
        if self.params.enable_deep_learning and self.deep_learning_model:
            for spectrum in spectra:
                if isinstance(spectrum, dict) and 'peaks' in spectrum:
                    prediction = self._get_deep_learning_prediction(spectrum)
                    dl_predictions.append(prediction)
                else:
                    dl_predictions.append({})
        
        # STEP 6: Generate fragmentation trees for candidate structures
        fragmentation_results = {}
        if self.params.enable_fragmentation_prediction and (self.fragmentation_model or True):
            # We'll collect compounds with SMILES to generate fragmentation trees
            # This will be done during result integration
            pass

        # STEP 7: Gather all search results
        search_results = {}
        
        for source, futures in search_futures.items():
            try:
                results = [ray.get(future) for future in futures]
                # Combine results
                if results:
                    if all(isinstance(r, pd.DataFrame) for r in results):
                        search_results[source] = pd.concat(results, ignore_index=True)
                    else:
                        self.logger.warning(f"Unexpected result type from {source} search")
            except Exception as e:
                self.logger.error(f"Error gathering results from {source}: {e}")
                search_results[source] = pd.DataFrame()
        
        # Get PubChem results
        pubchem_results = [ray.get(future) for future in pubchem_futures] if pubchem_futures else []
        
        # STEP 8: Combine and score all results
        combined_results = []
        
        # Process each spectrum
        for idx, spectrum in enumerate(spectra):
            mz = spectrum.get('precursor_mz', 0) if isinstance(spectrum, dict) else getattr(spectrum, 'mz', 0)
            rt = spectrum.get('rt', 0) if isinstance(spectrum, dict) else getattr(spectrum, 'rt', 0)
            
            # Collect all matching results from each source
            matches_by_source = {}
            
            # Get mass-based search results for this spectrum
            for source, results_df in search_results.items():
                if results_df.empty:
                    continue
                
                # Filter by m/z range
                if 'Query_mass' in results_df.columns:
                    # For LipidMaps format
                    matches = results_df[
                        results_df['Query_mass'].between(
                            mz - mz * self.params.ms1_ppm_tolerance / 1e6,
                            mz + mz * self.params.ms1_ppm_tolerance / 1e6
                        )
                    ]
                elif 'precursor_mz' in results_df.columns:
                    # For MSLipids format
                    matches = results_df[
                        results_df['precursor_mz'].between(
                            mz - mz * self.params.ms1_ppm_tolerance / 1e6,
                            mz + mz * self.params.ms1_ppm_tolerance / 1e6
                        )
                    ]
                elif 'monoisotopic_mass' in results_df.columns:
                    # For HMDB/KEGG format
                    matches = results_df[
                        results_df['monoisotopic_mass'].between(
                            mz - mz * self.params.ms1_ppm_tolerance / 1e6,
                            mz + mz * self.params.ms1_ppm_tolerance / 1e6
                        )
                    ]
                else:
                    # Unknown format, skip
                    matches = pd.DataFrame()
                
                matches_by_source[source] = matches
            
            # Get spectral matching results for this spectrum
            library_matches = [match for match in library_search_results if match.get('query_index') == idx]
            
            # Get deep learning predictions for this spectrum
            dl_prediction = dl_predictions[idx] if idx < len(dl_predictions) else {}
            
            # Get PubChem results if available
            pubchem_match = {}
            if idx < len(pubchem_results) and pubchem_results[idx]:
                # Extract pubchem info for this spectrum
                if isinstance(spectrum, SpectrumDocument) and spectrum in pubchem_results[idx]:
                    pubchem_info = pubchem_results[idx][spectrum]
                    pubchem_match = {
                        'compound_name': pubchem_info.name if hasattr(pubchem_info, 'name') else '',
                        'formula': pubchem_info.molecular_formula if hasattr(pubchem_info, 'molecular_formula') else '',
                        'pubchem_cid': pubchem_info.cid if hasattr(pubchem_info, 'cid') else '',
                        'source': 'pubchem'
                    }
            
            # STEP 9: Collect compound IDs for pathway search
            compound_ids = {
                "kegg": [],
                "hmdb": []
            }
            
            # Integrate all results and calculate confidence scores
            all_compounds = []
            
            # Process each source
            for source, matches in matches_by_source.items():
                for _, match in matches.iterrows():
                    # Extract relevant compound info based on source
                    if source == 'lipidmaps':
                        compound_info = {
                            'compound_name': match.get('Name', ''),
                            'formula': match.get('Formula', ''),
                            'adduct': match.get('Ion_type', ''),
                            'source': 'lipidmaps',
                            'smiles': match.get('SMILES', ''),
                            'inchikey': match.get('InChIKey', ''),
                            'mass_error_ppm': match.get('PPM_Error', 0.0)
                        }
                    elif source == 'mslipids':
                        compound_info = {
                            'compound_name': match.get('lipid_name', ''),
                            'formula': match.get('formula', ''),
                            'adduct': match.get('adduct', ''),
                            'source': 'mslipids',
                            'category': match.get('category', ''),
                            'mass_error_ppm': match.get('mass_error', 0.0)
                        }
                    elif source == 'hmdb':
                        compound_info = {
                            'compound_name': match.get('compound_name', ''),
                            'formula': match.get('formula', ''),
                            'source': 'hmdb',
                            'hmdb_id': match.get('hmdb_id', ''),
                            'smiles': match.get('smiles', ''),
                            'inchikey': match.get('inchikey', '')
                        }
                        # Add to compounds for pathway search
                        if match.get('hmdb_id'):
                            compound_ids['hmdb'].append(match.get('hmdb_id'))
                    elif source == 'kegg':
                        compound_info = {
                            'compound_name': match.get('compound_name', ''),
                            'formula': match.get('formula', ''),
                            'source': 'kegg',
                            'kegg_id': match.get('kegg_id', '')
                        }
                        # Add to compounds for pathway search
                        if match.get('kegg_id'):
                            compound_ids['kegg'].append(match.get('kegg_id'))
                    else:
                        # Unknown source
                        continue
                    
                    # Calculate confidence score
                    ms2_data_for_spectrum = ms2_data[idx] if idx < len(ms2_data) else None
                    similarity_score = similarities[idx] if similarities is not None else None
                    
                    confidence_score = self._calculate_confidence(
                        match=match,
                        source=source,
                        ms2_data=ms2_data_for_spectrum,
                        similarity_score=similarity_score
                    )
                    
                    compound_info['confidence_score'] = confidence_score
                    
                    # Generate fragmentation tree if SMILES is available
                    if self.params.enable_fragmentation_prediction and 'smiles' in compound_info and compound_info['smiles']:
                        fragmentation_result = self._generate_fragmentation_tree(
                            compound_info['smiles'], mz
                        )
                        compound_info['fragmentation'] = fragmentation_result
                    
                    # Get RT prediction if available
                    if hasattr(self, 'rt_model') and 'smiles' in compound_info and compound_info['smiles']:
                        rt_prediction = self.predict_rt(compound_info['smiles'])
                        compound_info['predicted_rt'] = rt_prediction.get('predicted_rt')
                        compound_info['rt_confidence'] = rt_prediction.get('confidence', 0.0)
                    
                    # Add common fields
                    compound_info['query_mz'] = mz
                    compound_info['query_rt'] = rt
                    
                    all_compounds.append(compound_info)
            
            # Add library search results
            for match in library_matches:
                library_info = {
                    'compound_name': match.get('compound_name', ''),
                    'source': match.get('source', 'unknown_library'),
                    'library_id': match.get('library_id', ''),
                    'smiles': match.get('smiles', ''),
                    'inchikey': match.get('inchikey', ''),
                    'spectral_similarity': max(
                        match.get('cosine_similarity', 0.0),
                        match.get('modcos_similarity', 0.0)
                    ),
                    'query_mz': mz,
                    'query_rt': rt
                }
                
                # Calculate confidence based mainly on spectral similarity
                library_info['confidence_score'] = library_info['spectral_similarity'] * 0.9  # Weight spectral matches highly
                
                # Generate fragmentation tree if SMILES is available
                if self.params.enable_fragmentation_prediction and library_info['smiles']:
                    fragmentation_result = self._generate_fragmentation_tree(
                        library_info['smiles'], mz
                    )
                    library_info['fragmentation'] = fragmentation_result
                
                # Get RT prediction if available
                if hasattr(self, 'rt_model') and library_info['smiles']:
                    rt_prediction = self.predict_rt(library_info['smiles'])
                    library_info['predicted_rt'] = rt_prediction.get('predicted_rt')
                    library_info['rt_confidence'] = rt_prediction.get('confidence', 0.0)
                
                all_compounds.append(library_info)
            
            # Add PubChem result if available
            if pubchem_match:
                pubchem_match['query_mz'] = mz
                pubchem_match['query_rt'] = rt
                pubchem_match['confidence_score'] = 0.5  # Default mid-level confidence for PubChem
                all_compounds.append(pubchem_match)
            
            # Add deep learning predictions if available
            if dl_prediction and 'predicted_compounds' in dl_prediction:
                for pred_compound in dl_prediction.get('predicted_compounds', []):
                    dl_info = {
                        'compound_name': pred_compound.get('name', ''),
                        'formula': pred_compound.get('formula', ''),
                        'source': 'deep_learning',
                        'confidence_score': pred_compound.get('probability', 0.0),
                        'query_mz': mz,
                        'query_rt': rt
                    }
                    all_compounds.append(dl_info)
            
            # STEP 10: Search pathways for identified compounds
            if self.params.enable_pathway_search and any(compound_ids.values()):
                pathway_results = self._search_pathways(compound_ids)
                
                # Add pathway information to relevant compounds
                for compound in all_compounds:
                    if 'kegg_id' in compound and compound['kegg_id'] and 'kegg' in pathway_results:
                        # Find pathways containing this compound
                        compound_pathways = []
                        for pathway_id, pathway_info in pathway_results['kegg'].items():
                            if f"cpd:{compound['kegg_id']}" in pathway_info.get('compounds', []):
                                pathway_data = {
                                    'pathway_id': pathway_id,
                                    'pathway_name': pathway_info.get('info', {}).get('name', ''),
                                    'pathway_url': pathway_info.get('info', {}).get('url', '')
                                }
                                compound_pathways.append(pathway_data)
                        
                        if compound_pathways:
                            compound['pathways'] = compound_pathways
                    
                    # Similar for other pathway databases
            
            # Add all compounds to result
            combined_results.extend(all_compounds)
        
        # Convert to DataFrame and sort by confidence score
        results_df = pd.DataFrame(combined_results)
        if not results_df.empty and 'confidence_score' in results_df.columns:
            results_df = results_df.sort_values('confidence_score', ascending=False)
        
        return results_df

    def _calculate_confidence(self, match: Union[pd.Series, Dict],
                              source: str,
                              ms2_data: Optional[Dict] = None,
                              similarity_score: Optional[float] = None) -> float:
        """Calculate confidence score for a match"""
        confidence = 0.0
        score_components = {}
        
        # Use weights from parameters
        weights = {
            'mass_error': self.params.mass_error_weight,
            'rt_match': self.params.rt_match_weight,
            'spectral_similarity': self.params.spectral_similarity_weight,
            'isotope_pattern': self.params.isotope_pattern_weight,
            'database_score': self.params.database_score_weight,
            'fragmentation_tree': self.params.fragmentation_tree_weight
        }

        # Mass error score
        mass_error = abs(float(match.get('PPM_Error', match.get('mass_error', 0))))
        mass_score = max(0, 1 - (mass_error / self.params.ms1_ppm_tolerance))
        confidence += weights['mass_error'] * mass_score
        score_components['mass_error'] = mass_score

        # RT match score (if available)
        rt_score = 0.0
        if hasattr(self, 'rt_model') and 'smiles' in match:
            smiles = match.get('SMILES', match.get('smiles', ''))
            if smiles:
                rt_prediction = self.predict_rt(smiles)
                predicted_rt = rt_prediction.get('predicted_rt')
                if predicted_rt is not None and 'query_rt' in match:
                    rt_diff = abs(match['query_rt'] - predicted_rt)
                    rt_score = max(0, 1 - (rt_diff / self.params.rt_tolerance))
                    confidence += weights['rt_match'] * rt_score
                    score_components['rt_match'] = rt_score

        # Spectral similarity score
        spec_score = 0.0
        if source in ['massbank', 'metlin', 'mzcloud', 'spectral_library', 'massbank_online', 'metlin_online']:
            # For spectral library matches
            if isinstance(match, dict):
                spec_score = match.get('cosine_similarity', match.get('similarity_score', 0))
            else:
                spec_score = match.get('similarity_score', 0)
        elif similarity_score is not None:
            # For embedding similarity
            spec_score = similarity_score
            
        confidence += weights['spectral_similarity'] * spec_score
        score_components['spectral_similarity'] = spec_score

        # Isotope pattern score
        isotope_score = 0.0
        if ms2_data and ('formula' in match or 'Formula' in match):
            formula = match.get('formula', match.get('Formula', ''))
            if formula:
                theoretical_pattern = self._calculate_isotope_pattern(formula)
                if len(theoretical_pattern) > 0:
                    observed_pattern = np.array(ms2_data['peaks'])
                    isotope_score = self._compare_isotope_patterns(
                        theoretical_pattern, observed_pattern
                    )
                    confidence += weights['isotope_pattern'] * isotope_score
                    score_components['isotope_pattern'] = isotope_score

        # Database source score
        source_scores = {
            'spectral_library': 1.0,
            'massbank': 0.9,
            'metlin': 0.9,
            'mzcloud': 0.9,
            'hmdb': 0.8,
            'lipidmaps': 0.8,
            'mslipids': 0.7,
            'kegg': 0.7,
            'pubchem': 0.6,
            'massbank_online': 0.8,
            'metlin_online': 0.8,
            'deep_learning': 0.7
        }
        
        source_score = source_scores.get(source, 0.5)
        confidence += weights['database_score'] * source_score
        score_components['database_score'] = source_score
        
        # Fragmentation tree score
        frag_score = 0.0
        if (isinstance(match, dict) and 'fragmentation' in match and 
            isinstance(match['fragmentation'], dict) and 'score' in match['fragmentation']):
            frag_score = match['fragmentation']['score']
            confidence += weights['fragmentation_tree'] * frag_score
            score_components['fragmentation_tree'] = frag_score
        
        # Store score components for debugging/auditing
        if isinstance(match, dict):
            match['score_components'] = score_components
        
        # Adjust confidence by scaling to [0, 1] range if needed
        total_weight = sum(weights.values())
        if total_weight > 0:
            confidence = min(confidence / total_weight, 1.0)

        return confidence

    def _compare_isotope_patterns(self, theoretical: np.ndarray,
                                  observed: np.ndarray) -> float:
        """Compare theoretical and observed isotope patterns"""
        if len(theoretical) == 0 or len(observed) == 0:
            return 0.0

        score = 0.0
        for theo_peak in theoretical:
            # Find closest observed peak
            mass_diffs = np.abs(observed[:, 0] - theo_peak[0])
            if min(mass_diffs) <= self.params.ms1_ppm_tolerance * theo_peak[0] / 1e6:
                closest_idx = np.argmin(mass_diffs)
                intensity_diff = abs(observed[closest_idx, 1] - theo_peak[1])
                score += 1 - min(intensity_diff, 1.0)

        return score / len(theoretical)

    @ray.remote
    def _search_hmdb(self, mz: float, ppm_tolerance: float) -> pd.DataFrame:
        """Search Human Metabolome Database by exact mass"""
        if not self.api_keys_valid.get("hmdb", False):
            return pd.DataFrame()
            
        try:
            # Calculate mass range
            mass_delta = mz * ppm_tolerance / 1e6
            min_mass = mz - mass_delta
            max_mass = mz + mass_delta
            
            # Build HMDB API request
            url = f"{self.params.hmdb_url}/metabolites.json"
            params = {
                "query_type": "monisotopic_molecular_weight",
                "min": str(min_mass),
                "max": str(max_mass)
            }
            
            response = requests.get(url, params=params)
            if response.status_code != 200:
                self.logger.error(f"HMDB API error: {response.status_code}")
                return pd.DataFrame()
                
            results = response.json()
            
            # Convert to DataFrame
            compounds = []
            for result in results:
                compound = {
                    'compound_name': result.get('name', ''),
                    'formula': result.get('chemical_formula', ''),
                    'hmdb_id': result.get('accession', ''),
                    'monoisotopic_mass': result.get('monisotopic_molecular_weight', 0),
                    'inchikey': result.get('inchikey', ''),
                    'smiles': result.get('smiles', ''),
                    'database': 'HMDB'
                }
                compounds.append(compound)
                
            return pd.DataFrame(compounds)
            
        except Exception as e:
            self.logger.error(f"Error in HMDB search: {e}")
            return pd.DataFrame()
    
    @ray.remote
    def _search_kegg(self, mz: float, ppm_tolerance: float) -> pd.DataFrame:
        """Search KEGG database by exact mass"""
        if not self.api_keys_valid.get("kegg", False):
            return pd.DataFrame()
            
        try:
            # Calculate mass range
            mass_delta = mz * ppm_tolerance / 1e6
            min_mass = mz - mass_delta
            max_mass = mz + mass_delta
            
            # First get list of all compounds
            if "kegg_compounds" not in self.query_cache:
                response = requests.get(f"{self.params.kegg_url}/list/compound")
                if response.status_code != 200:
                    self.logger.error(f"KEGG API error: {response.status_code}")
                    return pd.DataFrame()
                    
                # Parse compound list
                compounds = {}
                for line in response.text.strip().split("\n"):
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        compound_id = parts[0].replace("cpd:", "")
                        name = parts[1]
                        compounds[compound_id] = name
                
                self.query_cache["kegg_compounds"] = compounds
            else:
                compounds = self.query_cache["kegg_compounds"]
            
            # Get compound details for each compound
            results = []
            
            # To avoid overwhelming the API, we'll limit to 20 random compounds for testing
            # In production, you would process all compounds or implement a better filtering strategy
            import random
            sample_ids = random.sample(list(compounds.keys()), min(20, len(compounds)))
            
            for compound_id in sample_ids:
                # Get compound details
                response = requests.get(f"{self.params.kegg_url}/get/cpd:{compound_id}")
                if response.status_code != 200:
                    continue
                    
                # Parse compound details
                formula = ""
                exact_mass = 0
                for line in response.text.strip().split("\n"):
                    if line.startswith("FORMULA"):
                        formula = line.replace("FORMULA", "").strip()
                    elif line.startswith("EXACT_MASS"):
                        try:
                            exact_mass = float(line.replace("EXACT_MASS", "").strip())
                        except:
                            continue
                
                # Check if mass is within range
                if min_mass <= exact_mass <= max_mass:
                    results.append({
                        'compound_name': compounds[compound_id],
                        'formula': formula,
                        'kegg_id': compound_id,
                        'monoisotopic_mass': exact_mass,
                        'database': 'KEGG'
                    })
            
            return pd.DataFrame(results)
            
        except Exception as e:
            self.logger.error(f"Error in KEGG search: {e}")
            return pd.DataFrame()
    
    @ray.remote
    def _search_massbank(self, spectrum: Dict, ppm_tolerance: float) -> pd.DataFrame:
        """Search MassBank using spectrum"""
        try:
            # Extract peaks from spectrum
            peaks = []
            for peak in spectrum.get('peaks', []):
                peaks.append(f"{peak.get('mz', 0)}:{peak.get('intensity', 0)}")
            
            # Build query
            params = {
                'mz': spectrum.get('precursor_mz', 0),
                'ppm': ppm_tolerance,
                'peaks': ';'.join(peaks)
            }
            
            response = requests.get(f"{self.params.massbank_url}/similarity", params=params)
            if response.status_code != 200:
                self.logger.error(f"MassBank API error: {response.status_code}")
                return pd.DataFrame()
                
            results = response.json()
            
            # Convert to DataFrame
            compounds = []
            for result in results:
                compound = {
                    'compound_name': result.get('name', ''),
                    'formula': result.get('formula', ''),
                    'massbank_id': result.get('id', ''),
                    'similarity_score': result.get('score', 0),
                    'smiles': result.get('smiles', ''),
                    'database': 'MassBank'
                }
                compounds.append(compound)
                
            return pd.DataFrame(compounds)
            
        except Exception as e:
            self.logger.error(f"Error in MassBank search: {e}")
            return pd.DataFrame()
    
    @ray.remote
    def _search_metlin(self, mz: float, ms2_spectrum: Optional[Dict] = None) -> pd.DataFrame:
        """Search Metlin database by exact mass and optionally MS2 spectrum"""
        if not self.api_keys_valid.get("metlin", False):
            return pd.DataFrame()
            
        try:
            # Build headers with API key
            headers = {"x-api-key": self.params.metlin_api_key}
            
            # Search by exact mass
            params = {
                'mz': mz,
                'ppm': self.params.ms1_ppm_tolerance
            }
            
            if ms2_spectrum:
                # If MS2 spectrum is provided, use MS2 search
                ms2_peaks = []
                for peak in ms2_spectrum.get('peaks', []):
                    ms2_peaks.append(f"{peak.get('mz', 0)}:{peak.get('intensity', 0)}")
                
                params['ms2'] = ';'.join(ms2_peaks)
                url = f"{self.params.metlin_url}/ms2_search"
            else:
                # Otherwise use exact mass search
                url = f"{self.params.metlin_url}/exact_mass_search"
            
            response = requests.get(url, headers=headers, params=params)
            if response.status_code != 200:
                self.logger.error(f"Metlin API error: {response.status_code}")
                return pd.DataFrame()
                
            results = response.json()
            
            # Convert to DataFrame
            compounds = []
            for result in results:
                compound = {
                    'compound_name': result.get('name', ''),
                    'formula': result.get('formula', ''),
                    'metlin_id': result.get('id', ''),
                    'monoisotopic_mass': result.get('monoisotopic_mass', 0),
                    'similarity_score': result.get('similarity', 0) if ms2_spectrum else None,
                    'database': 'Metlin'
                }
                compounds.append(compound)
                
            return pd.DataFrame(compounds)
            
        except Exception as e:
            self.logger.error(f"Error in Metlin search: {e}")
            return pd.DataFrame()
            
    def _search_multiple_spectral_libraries(self, spectrum: Dict) -> List[Dict]:
        """Search across multiple spectral libraries"""
        all_matches = []
        
        # Process peaks
        processed_peaks = self._process_ms2_spectrum(spectrum)
        
        # Search in loaded libraries
        libraries = [
            ('spectral_library', self.spectral_library),
            ('massbank', self.massbank_library),
            ('metlin', self.metlin_library),
            ('mzcloud', self.mzcloud_library),
            ('hmdb', self.hmdb_library),
            ('in_house', self.in_house_library)
        ]
        
        for lib_name, library in libraries:
            if library:
                matches = self._search_in_library(processed_peaks, library, lib_name)
                all_matches.extend(matches)
        
        # Search in online databases if needed and enabled
        if self.params.enable_spectral_matching:
            # MassBank online search
            massbank_results = self._search_massbank.remote(spectrum, self.params.ms1_ppm_tolerance)
            
            # Metlin search if API key is available
            if self.api_keys_valid.get("metlin", False):
                metlin_results = self._search_metlin.remote(
                    spectrum.get('precursor_mz', 0), 
                    spectrum
                )
            else:
                metlin_results = pd.DataFrame()
            
            # Get results
            massbank_df = ray.get(massbank_results)
            metlin_df = ray.get(metlin_results) if isinstance(metlin_results, ray._raylet.ObjectRef) else metlin_results
            
            # Convert to list of dicts
            for _, row in massbank_df.iterrows():
                match = {
                    'library_id': row.get('massbank_id', ''),
                    'compound_name': row.get('compound_name', ''),
                    'cosine_similarity': row.get('similarity_score', 0),
                    'modcos_similarity': 0,
                    'smiles': row.get('smiles', ''),
                    'inchikey': '',
                    'source': 'massbank_online'
                }
                all_matches.append(match)
                
            for _, row in metlin_df.iterrows():
                match = {
                    'library_id': row.get('metlin_id', ''),
                    'compound_name': row.get('compound_name', ''),
                    'cosine_similarity': row.get('similarity_score', 0) if row.get('similarity_score') else 0,
                    'modcos_similarity': 0,
                    'smiles': '',
                    'inchikey': '',
                    'source': 'metlin_online'
                }
                all_matches.append(match)
        
        return all_matches
    
    def _search_in_library(self, query_peaks: np.ndarray, library: Dict, library_name: str) -> List[Dict]:
        """Search spectrum against a spectral library"""
        matches = []
        
        for lib_id, lib_spectrum in library.items():
            lib_peaks = np.array([(p['mz'], p['intensity'])
                                for p in lib_spectrum['peaks']])

            # Calculate different similarity scores
            cosine_score = self.cosine_similarity.pair(query_peaks, lib_peaks)
            modcos_score = self.modcos_similarity.pair(query_peaks, lib_peaks)

            if cosine_score > self.params.cosine_similarity_threshold or modcos_score > self.params.modified_cosine_threshold:
                matches.append({
                    'library_id': lib_id,
                    'compound_name': lib_spectrum.get('name', ''),
                    'cosine_similarity': float(cosine_score),
                    'modcos_similarity': float(modcos_score),
                    'smiles': lib_spectrum.get('smiles', ''),
                    'inchikey': lib_spectrum.get('inchikey', ''),
                    'source': library_name
                })

        return sorted(matches,
                    key=lambda x: max(x['cosine_similarity'], x['modcos_similarity']),
                    reverse=True)

    def _search_pathways(self, compound_ids: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Search for pathways containing the identified compounds
        
        Args:
            compound_ids: Dictionary mapping database names to lists of compound IDs
            
        Returns:
            Dictionary of pathway information
        """
        pathways = {}
        
        # Search in KEGG pathways
        if "kegg" in compound_ids and self.api_keys_valid.get("kegg", False):
            kegg_ids = compound_ids["kegg"]
            kegg_pathways = self._search_kegg_pathways(kegg_ids)
            pathways["kegg"] = kegg_pathways
            
        # Search in HumanCyc
        if "humancyc" in compound_ids and self.api_keys_valid.get("humancyc", False):
            humancyc_ids = compound_ids["humancyc"]
            humancyc_pathways = self._search_humancyc_pathways(humancyc_ids)
            pathways["humancyc"] = humancyc_pathways
            
        return pathways
        
    def _search_kegg_pathways(self, compound_ids: List[str]) -> Dict[str, Any]:
        """Search KEGG for pathways containing the given compounds"""
        pathway_hits = {}
        
        try:
            # Query each compound
            for compound_id in compound_ids:
                # Ensure proper format
                if not compound_id.startswith("cpd:"):
                    compound_id = f"cpd:{compound_id}"
                
                # Get pathways for this compound
                response = requests.get(f"{self.params.kegg_url}/link/pathway/{compound_id}")
                if response.status_code != 200:
                    continue
                    
                # Parse pathway links
                for line in response.text.strip().split("\n"):
                    if not line:
                        continue
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        pathway_id = parts[1].replace("path:", "")
                        
                        # Count hits for each pathway
                        if pathway_id in pathway_hits:
                            pathway_hits[pathway_id]["compounds"].append(compound_id)
                        else:
                            pathway_hits[pathway_id] = {
                                "compounds": [compound_id],
                                "info": {}
                            }
            
            # Get pathway details
            for pathway_id in list(pathway_hits.keys()):
                response = requests.get(f"{self.params.kegg_url}/get/path:{pathway_id}")
                if response.status_code != 200:
                    continue
                
                # Parse pathway information
                pathway_name = ""
                for line in response.text.strip().split("\n"):
                    if line.startswith("NAME"):
                        pathway_name = line.replace("NAME", "").strip()
                        break
                
                pathway_hits[pathway_id]["info"] = {
                    "name": pathway_name,
                    "id": pathway_id,
                    "url": f"https://www.kegg.jp/pathway/map{pathway_id}"
                }
                
            return pathway_hits
            
        except Exception as e:
            self.logger.error(f"Error in KEGG pathway search: {e}")
            return {}
    
    def _search_humancyc_pathways(self, compound_ids: List[str]) -> Dict[str, Any]:
        """Search HumanCyc for pathways containing the given compounds"""
        pathway_hits = {}
        
        try:
            # This is a simplified implementation - actual HumanCyc API would differ
            # In a real implementation, you would use the actual HumanCyc API
            
            # For demonstration, we'll return empty results
            self.logger.info("HumanCyc pathway search not fully implemented")
            return {}
            
        except Exception as e:
            self.logger.error(f"Error in HumanCyc pathway search: {e}")
            return {}
            
    def _generate_fragmentation_tree(self, smiles: str, precursor_mz: float) -> Dict[str, Any]:
        """
        Generate a fragmentation tree for a compound
        
        Args:
            smiles: SMILES string of the compound
            precursor_mz: Precursor m/z value
            
        Returns:
            Dictionary containing the fragmentation tree
        """
        # This would use an actual fragmentation tree algorithm like MetFrag or CFM-ID
        # Here we're implementing a simplified version
        
        if not smiles:
            return {"fragments": [], "score": 0}
            
        try:
            # Create molecule from SMILES
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                return {"fragments": [], "score": 0}
                
            # Generate fragmentation if we have a model
            if self.fragmentation_model:
                # Call the model to predict fragments
                # This is a placeholder - actual implementation depends on the model
                return self._predict_fragments_with_model(mol, precursor_mz)
            else:
                # Fallback to rule-based fragmentation
                return self._generate_rule_based_fragments(mol, precursor_mz)
                
        except Exception as e:
            self.logger.error(f"Error generating fragmentation tree: {e}")
            return {"fragments": [], "score": 0}
            
    def _predict_fragments_with_model(self, mol, precursor_mz: float) -> Dict[str, Any]:
        """Use loaded model to predict fragments"""
        # This is a placeholder for the actual model prediction
        # The implementation would depend on the specific model
        
        if not hasattr(self, 'fragmentation_model') or not self.fragmentation_model:
            return {"fragments": [], "score": 0}
            
        try:
            # Convert molecule to features expected by the model
            # This would be model-specific
            
            # Generate Morgan fingerprints as example features
            from rdkit.Chem import AllChem
            features = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            features = np.array(features)
            
            # Make prediction with model
            fragments_pred = self.fragmentation_model.predict(features.reshape(1, -1))
            
            # Process predictions - this is model-specific
            fragments = []
            for i, prob in enumerate(fragments_pred[0]):
                if prob > 0.5:  # Probability threshold
                    # In a real model, you would decode fragment ID to structure
                    fragment_mass = precursor_mz * (i / len(fragments_pred[0]))  # Dummy calculation
                    fragments.append({
                        "fragment_mz": fragment_mass,
                        "probability": float(prob),
                        "formula": f"Fragment_{i}"  # Placeholder
                    })
            
            return {
                "fragments": fragments,
                "score": float(np.mean([f["probability"] for f in fragments]) if fragments else 0)
            }
            
        except Exception as e:
            self.logger.error(f"Error predicting fragments with model: {e}")
            return {"fragments": [], "score": 0}
    
    def _generate_rule_based_fragments(self, mol, precursor_mz: float) -> Dict[str, Any]:
        """Generate fragments using rule-based approach"""
        fragments = []
        
        try:
            # This is a simplified rule-based fragmentation
            # A real implementation would use more sophisticated rules
            
            # Break bonds and create fragments
            bonds = mol.GetBonds()
            for bond_idx in range(min(5, mol.GetNumBonds())):  # Limit to 5 fragments for demo
                bond = bonds[bond_idx]
                
                # Skip if this is a ring bond
                if bond.IsInRing():
                    continue
                    
                # Break bond
                atom1 = bond.GetBeginAtomIdx()
                atom2 = bond.GetEndAtomIdx()
                
                # Calculate approximate mass of fragments
                # This is very simplified - real implementation would calculate actual fragment masses
                atoms = mol.GetAtoms()
                fragment1_size = len(atoms) // 2
                fragment2_size = len(atoms) - fragment1_size
                
                # Estimate fragment masses (very approximate)
                fragment1_mass = precursor_mz * (fragment1_size / len(atoms))
                fragment2_mass = precursor_mz * (fragment2_size / len(atoms))
                
                fragments.append({
                    "fragment_mz": fragment1_mass,
                    "probability": 0.8,  # Dummy probability
                    "formula": f"Fragment1_{bond_idx}"
                })
                
                fragments.append({
                    "fragment_mz": fragment2_mass,
                    "probability": 0.7,  # Dummy probability
                    "formula": f"Fragment2_{bond_idx}"
                })
            
            return {
                "fragments": fragments,
                "score": 0.75  # Dummy score
            }
            
        except Exception as e:
            self.logger.error(f"Error in rule-based fragmentation: {e}")
            return {"fragments": [], "score": 0}
            
    def _get_deep_learning_prediction(self, spectrum: Dict) -> Dict[str, Any]:
        """
        Get predictions from the deep learning model
        
        Args:
            spectrum: Spectrum data
            
        Returns:
            Dictionary with model predictions
        """
        if not self.deep_learning_model:
            return {}
            
        try:
            # Process spectrum to model input format
            processed_spectrum = self._process_spectrum_for_model(spectrum)
            
            # Make prediction
            predictions = self.deep_learning_model.predict(processed_spectrum)
            
            # Process predictions
            results = self._process_model_predictions(predictions)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in deep learning prediction: {e}")
            return {}
    
    def _process_spectrum_for_model(self, spectrum: Dict) -> np.ndarray:
        """Process spectrum into format expected by deep learning model"""
        try:
            # Extract peaks
            peaks = np.array([(p['mz'], p['intensity']) for p in spectrum.get('peaks', [])])
            
            # Sort by m/z
            peaks = peaks[peaks[:, 0].argsort()]
            
            # Normalize intensities
            if len(peaks) > 0:
                max_intensity = np.max(peaks[:, 1])
                if max_intensity > 0:
                    peaks[:, 1] = peaks[:, 1] / max_intensity
            
            # Create fixed-length vector
            # This would be model-specific
            mz_range = (0, 2000)  # Example range
            bin_size = 1.0  # Example bin size
            num_bins = int((mz_range[1] - mz_range[0]) / bin_size)
            
            binned_spectrum = np.zeros(num_bins)
            
            for mz, intensity in peaks:
                if mz_range[0] <= mz < mz_range[1]:
                    bin_idx = int((mz - mz_range[0]) / bin_size)
                    binned_spectrum[bin_idx] = max(binned_spectrum[bin_idx], intensity)
            
            # Add batch dimension for model
            return binned_spectrum.reshape(1, -1)
            
        except Exception as e:
            self.logger.error(f"Error processing spectrum for model: {e}")
            return np.zeros((1, 2000))  # Default empty spectrum
    
    def _process_model_predictions(self, predictions) -> Dict[str, Any]:
        """Process model predictions into usable results"""
        # This would be specific to the model architecture and output format
        # Here's a simplified example
        
        try:
            # Assuming the model outputs a vector of compound probabilities
            # and we have a mapping of indices to compounds
            
            # For demonstration, create dummy results
            results = {
                "predicted_compounds": [],
                "confidence": 0.0
            }
            
            # Get top predictions
            if len(predictions.shape) > 1 and predictions.shape[1] > 0:
                top_indices = np.argsort(predictions[0])[::-1][:5]  # Top 5 predictions
                
                for idx in top_indices:
                    if predictions[0][idx] > 0.1:  # Confidence threshold
                        # In a real model, you would map index to actual compound
                        compound = {
                            "name": f"Compound_{idx}",
                            "probability": float(predictions[0][idx]),
                            "formula": f"C{idx}H{2*idx}O{idx//2}"  # Dummy formula
                        }
                        results["predicted_compounds"].append(compound)
                
                results["confidence"] = float(np.max(predictions[0]))
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing model predictions: {e}")
            return {"predicted_compounds": [], "confidence": 0.0}



