from pathlib import Path

from matchms.importing import load_from_mgf
from tensorflow import keras
import pandas as pd

from ms2deepscore import SpectrumBinner
from ms2deepscore.data_generators import DataGeneratorAllSpectrums
from ms2deepscore.models import SiameseModel
from ms2deepscore import MS2DeepScore

from matchms.filtering import default_filters, add_parent_mass, reduce_to_number_of_peaks, select_by_mz, add_losses, \
    require_minimum_number_of_peaks
from matchms.filtering import normalize_intensities
from matchms import calculate_scores
from matchms.similarity import CosineGreedy
from spec2vec import Spec2Vec
from spec2vec import SpectrumDocument


def predict_structures(path: str):

    def get_mgf_spectrum(file):
        spectrums = []
        for spectrum in file:
            spectrum = default_filters(spectrum)
            spectrum = normalize_intensities(spectrum)
            spectrums.append(spectrum)

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

    def spectra_similarity(spectrums):
        scores = calculate_scores(references=spectrums, queries=spectrums, similarity_function=CosineGreedy())
        return scores

    def data_preprocessing(spectrums):
        spectrum_binner = SpectrumBinner(1000, mz_min=10.0, mz_max=1000.0, peak_scaling=0.5)
        binned_spectrums = spectrum_binner.fit_transform(spectrums)
        dimension = len(spectrum_binner.known_bins)
        data_generator = DataGeneratorAllSpectrums(binned_spectrums, binned_spectrums, dim=dimension)
        model = SiameseModel(spectrum_binner, base_dims=(200, 200, 200), embedding_dim=200,
                             dropout_rate=0.2)
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=0.001))
        model.fit(data_generator, validation_data=data_generator, epochs=2)
        similarity_measure = MS2DeepScore(model)
        score = similarity_measure.pair(spectrums[0], spectrums[1])


## train own ms2 deep score model

from ms2deepscore import SpectrumBinner
spectrum_binner = SpectrumBinner(1000, mz_min=10.0, mz_max=1000.0, peak_scaling=0.5)
binned_spectrums = spectrum_binner.fit_transform(spectrums)

from ms2deepscore.data_generators import DataGeneratorAllSpectrums
dimension = len(spectrum_binner.known_bins)
data_generator = DataGeneratorAllSpectrums(binned_spectrums, tanimoto_scores_df,
                                           dim=dimension)

from tensorflow import keras
from ms2deepscore.models import SiameseModel
model = SiameseModel(spectrum_binner, base_dims=(200, 200, 200), embedding_dim=200,
                     dropout_rate=0.2)
model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=0.001))
model.fit(data_generator,
          validation_data=data_generator,
          epochs=2)


from ms2deepscore import MS2DeepScore
similarity_measure = MS2DeepScore(model)
score = similarity_measure.pair(spectrums[0], spectrums[1])
