# annotation.py
from dataclasses import dataclass
from io import StringIO
from typing import List, Dict, Optional, Union, Tuple
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

    def __post_init__(self):
        if self.adducts_positive is None:
            self.adducts_positive = ['[M+H]+', '[M+Na]+', '[M+K]+', '[M+NH4]+']
        if self.adducts_negative is None:
            self.adducts_negative = ['[M-H]-', '[M+Cl]-', '[M+HCOO]-']


class MSAnnotator:
    """Unified MS annotation pipeline combining LipidMaps, MSLipids, and Spec2Vec/PubChem approaches"""

    def __init__(self, params: AnnotationParameters,
                 model: Optional[BaseTopicModel] = None,
                 rt_model_path: Optional[str] = None,
                 library_path: Optional[str] = None):
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
                    'peaks': processed_peaks.tolist()
                })

        # Process LipidMaps search
        mz_values = spec_df['mz'].unique().compute()
        adducts = (self.params.adducts_positive if polarity == 'positive'
                   else self.params.adducts_negative)

        # Split into batches and process in parallel
        mz_batches = [mz_values[i:i + self.params.batch_size]
                      for i in range(0, len(mz_values), self.params.batch_size)]

        # Launch parallel searches
        lipidmaps_futures = [
            self._process_lipidmaps_batch.remote(batch, adducts)
            for batch in mz_batches
        ]

        # Process MSLipids search
        mslipids_futures = [
            self._process_mslipids_batch.remote(batch, ms2_data)
            for batch in mz_batches
        ]

        # Process Spec2Vec/PubChem in parallel if model is available
        pubchem_futures = []
        similarities = None
        if self.spec2vec_model:
            spec_batches = [spectra[i:i + self.params.pubchem_batch_size]
                            for i in range(0, len(spectra), self.params.pubchem_batch_size)]

            pubchem_futures = [
                self._process_pubchem_batch.remote(
                    batch,
                    [s.get('compound_name', '') for s in batch]
                ) for batch in spec_batches
            ]

            # Calculate similarities if library spectra are available
            if hasattr(self, 'library_spectra'):
                similarities = self._parallel_spec2vec_similarity(
                    spectra, self.library_spectra
                )

        # Process spectral library search
        library_results = []
        for spectrum in spectra:
            if isinstance(spectrum, dict) and 'peaks' in spectrum:
                matches = self.search_spectral_library(spectrum)
                library_results.extend(matches)

        # Gather all results
        lipidmaps_results = pd.concat(ray.get(lipidmaps_futures))
        mslipids_results = pd.concat(ray.get(mslipids_futures))
        pubchem_results = ray.get(pubchem_futures) if pubchem_futures else None

        # Combine and score results
        combined_results = []

        for idx, spectrum in enumerate(spectra):
            mz = spectrum.get('precursor_mz', 0) if isinstance(spectrum, dict) else spectrum.mz
            rt = spectrum.get('rt', 0) if isinstance(spectrum, dict) else getattr(spectrum, 'rt', 0)

            # Get matching results from each source
            lipidmaps_matches = lipidmaps_results[
                (lipidmaps_results['Query_mass'].between(
                    mz - mz * self.params.ms1_ppm_tolerance / 1e6,
                    mz + mz * self.params.ms1_ppm_tolerance / 1e6
                ))
            ]

            mslipids_matches = mslipids_results[
                mslipids_results['precursor_mz'].between(
                    mz - mz * self.params.ms1_ppm_tolerance / 1e6,
                    mz + mz * self.params.ms1_ppm_tolerance / 1e6
                )
            ]

            # Get spectral library matches for this spectrum
            lib_matches = [m for m in library_results if m.get('query_index') == idx]

            # Calculate confidence scores and combine results
            for source, matches in [
                ('lipidmaps', lipidmaps_matches),
                ('mslipids', mslipids_matches),
                ('library', lib_matches)
            ]:
                for _, match in (matches.iterrows() if isinstance(matches, pd.DataFrame)
                else enumerate(matches)):

                    confidence_score = self._calculate_confidence(
                        match=match,
                        source=source,
                        ms2_data=ms2_data[idx] if ms2_data else None,
                        similarity_score=similarities[idx] if similarities is not None else None
                    )

                    # Get RT prediction if SMILES is available
                    rt_prediction = {}
                    if hasattr(self, 'rt_model'):
                        smiles = match.get('SMILES', '') if source == 'lipidmaps' else \
                            match.get('smiles', '')
                        if smiles:
                            rt_prediction = self.predict_rt(smiles)

                    result = {
                        'query_mz': mz,
                        'query_rt': rt,
                        'compound_name': match.get('Name', match.get('compound_name', '')),
                        'formula': match.get('Formula', match.get('formula', '')),
                        'adduct': match.get('Ion_type', match.get('adduct', '')),
                        'source': source,
                        'confidence_score': confidence_score,
                        'smiles': match.get('SMILES', match.get('smiles', '')),
                        'inchikey': match.get('InChIKey', match.get('inchikey', '')),
                        'predicted_rt': rt_prediction.get('predicted_rt'),
                        'rt_confidence': rt_prediction.get('confidence', 0.0),
                        'mass_error_ppm': match.get('PPM_Error', match.get('mass_error', 0.0)),
                        'spectral_similarity': match.get('cosine_similarity',
                                                         match.get('similarity_score', 0.0))
                    }

                    combined_results.append(result)

        # Convert to DataFrame and sort by confidence score
        results_df = pd.DataFrame(combined_results)
        results_df = results_df.sort_values('confidence_score', ascending=False)

        return results_df

    def _calculate_confidence(self, match: Union[pd.Series, Dict],
                              source: str,
                              ms2_data: Optional[Dict] = None,
                              similarity_score: Optional[float] = None) -> float:
        """Calculate confidence score for a match"""
        confidence = 0.0
        weights = {
            'mass_error': 0.3,
            'rt_match': 0.2,
            'spectral_similarity': 0.3,
            'isotope_pattern': 0.1,
            'database_score': 0.1
        }

        # Mass error score
        mass_error = abs(float(match.get('PPM_Error', match.get('mass_error', 0))))
        mass_score = max(0, 1 - (mass_error / self.params.ms1_ppm_tolerance))
        confidence += weights['mass_error'] * mass_score

        # RT match score (if available)
        if hasattr(self, 'rt_model') and 'predicted_rt' in match:
            rt_diff = abs(match['query_rt'] - match['predicted_rt'])
            rt_score = max(0, 1 - (rt_diff / self.params.rt_tolerance))
            confidence += weights['rt_match'] * rt_score

        # Spectral similarity score
        spec_score = 0.0
        if source == 'library':
            spec_score = float(match.get('cosine_similarity', 0))
        elif similarity_score is not None:
            spec_score = similarity_score
        confidence += weights['spectral_similarity'] * spec_score

        # Isotope pattern score
        if ms2_data and 'formula' in match:
            theoretical_pattern = self._calculate_isotope_pattern(match['formula'])
            if len(theoretical_pattern) > 0:
                observed_pattern = np.array(ms2_data['peaks'])
                isotope_score = self._compare_isotope_patterns(
                    theoretical_pattern, observed_pattern
                )
                confidence += weights['isotope_pattern'] * isotope_score

        # Database source score
        source_scores = {
            'library': 1.0,
            'lipidmaps': 0.8,
            'mslipids': 0.7
        }
        confidence += weights['database_score'] * source_scores.get(source, 0.5)

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



