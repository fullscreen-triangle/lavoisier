from matchms.filtering import default_filters, select_by_intensity
from matchms.filtering import repair_inchi_inchikey_smiles
from matchms.filtering import derive_inchikey_from_inchi
from matchms.filtering import derive_smiles_from_inchi
from matchms.filtering import derive_inchi_from_smiles
from matchms.filtering import harmonize_undefined_inchi
from matchms.filtering import harmonize_undefined_inchikey
from matchms.filtering import harmonize_undefined_smiles
import numpy as np
from matchms.filtering import add_losses
from matchms.filtering import add_parent_mass
from matchms.filtering import default_filters
from matchms.filtering import normalize_intensities
from matchms.filtering import reduce_to_number_of_peaks
from matchms.filtering import require_minimum_number_of_peaks
from matchms.filtering import select_by_mz
from matchms.importing import load_from_mgf
from matchms.similarity import CosineGreedy
from spec2vec import SpectrumDocument
from spec2vec.model_building import train_new_word2vec_model
import gensim
from matchms import calculate_scores
from rdkit import Chem
from rdkit.Chem import MCS


def spectrum_processing(s):
    """This is how one would typically design a desired pre- and post-
    processing pipeline."""
    s = default_filters(s)
    s = add_parent_mass(s)
    s = normalize_intensities(s)
    s = reduce_to_number_of_peaks(s, n_required=10, ratio_desired=0.5, n_max=500)
    s = select_by_mz(s, mz_from=0, mz_to=1000)
    s = add_losses(s, loss_mz_from=10.0, loss_mz_to=200.0)
    s = require_minimum_number_of_peaks(s, n_required=10)
    return s


spectrums = [spectrum_processing(s) for s in load_from_mgf("reference_spectrums.mgf")]

# Omit spectrums that didn't qualify for analysis
spectrums = [s for s in spectrums if s is not None]

# Inspect the spectra data
inchikeys = [s.get("inchikey") for s in spectrums]
found_inchikeys = np.sum([1 for x in inchikeys if x is not None])


def metadata_processing(spectrum):
    spectrum = default_filters(spectrum)
    spectrum = repair_inchi_inchikey_smiles(spectrum)
    spectrum = derive_inchi_from_smiles(spectrum)
    spectrum = derive_smiles_from_inchi(spectrum)
    spectrum = derive_inchikey_from_inchi(spectrum)
    spectrum = harmonize_undefined_smiles(spectrum)
    spectrum = harmonize_undefined_inchi(spectrum)
    spectrum = harmonize_undefined_inchikey(spectrum)
    return spectrum


def peak_processing(spectrum):
    spectrum = default_filters(spectrum)
    spectrum = normalize_intensities(spectrum)
    spectrum = select_by_intensity(spectrum, intensity_from=0.01)
    spectrum = select_by_mz(spectrum, mz_from=10, mz_to=1000)
    return spectrum

def spectra_smiles(spectrums):
    spectrums = [metadata_processing(s) for s in spectrums]
    spectrums = [peak_processing(s) for s in spectrums]
    inchikeys = [s.get("inchikey") for s in spectrums]
    similarity_measure = CosineGreedy(tolerance=0.005)
    scores = calculate_scores(spectrums, spectrums, similarity_measure, is_symmetric=True)
    best_matches = scores.scores_by_query(spectrums[5], sort=True)[:10]
    for i, smiles in enumerate([x[0].get("smiles") for x in best_matches]):
        m = Chem.MolFromSmiles(smiles)


