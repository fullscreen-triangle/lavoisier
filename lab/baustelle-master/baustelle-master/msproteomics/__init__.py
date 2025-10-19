import logging
import os
import warnings
import pandas as pd

from lipidomics import print_progress_bar
from msproteomics.MSFilters import remove_rows_matching, remove_reverse, remove_only_identified_by_site, \
    filter_localization_probability
from msproteomics.MSImputation import pls
from msproteomics.MSVizualise import plot_point_cov, pca, plsda, plsr, box
from msproteomics.proteomics_module import FolderWideSearch, GroupSearch

warnings.simplefilter(action='ignore', category=FutureWarning)
# Progress bar increment per step
INCREMENT = 100.0 / 18


def _update_status(data, stepDst, verbose, stepNum):
    print_progress_bar(INCREMENT * stepNum, 100, prefix='PeakFilter progress:')
    if verbose:
        # Create a CSV file with the whole processed dataframe
        outFileName = 'peakfilter_step_{:02d}.csv'.format(stepNum)
        data.to_csv(os.path.join(stepDst, outFileName), index=False)
    stepNum += 1
    return stepNum


def convert_files(data, src, fasta, verbose=False):
    print_progress_bar(0, 100, prefix='PeakFilter progress:')
    # Set the log file where the information about the steps performed
    # is saved
    logFilePath = 'peakfilter.log'
    logFilePath = os.path.join(src, logFilePath)
    # Create logger and its file handler
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(logFilePath)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # Write initial information in log file
    logger.info('Starting file conversion. Input dataframe ("%s") has %d rows.',
                data.src, len(data.index))
    # Prepare the folder structure to store the intermediate files
    stepDst = os.path.join(src, 'step_by_step')
    if verbose and not os.path.isdir(stepDst):
        os.makedirs(stepDst)
    stepNum = 1
    zip_files = [os.path.join(src, file) for file in os.listdir(src) if file.endswith(".zip")]
    if len(zip_files) > 0:
        extract_zip_files(src)
        logger.info('Unzipping files. Input folder  has %d files.', len(zip_files))
    stepNum = _update_status(zip_files, stepDst, verbose, stepNum)
    spec_files = []
    raw_files = [os.path.join(src, file) for file in os.listdir(src) if
                 file.endswith(".mzML")]
    logger.info('Input folder  has %d raw files.', len(raw_files))
    if len(raw_files) > 0:
        for mzml in raw_files:
            spec_files.append(mzml)
    stepNum = _update_status(zip_files, stepDst, verbose, stepNum)
    # add fasta file here
    if fasta == 'HUMAN':
        generate_decoy_database()


def group_search(data, src, fasta, verbose=False):
    results = GroupSearch.group_search(src)


def folder_wide_search(data, src, fasta, verbose=False):
    results = FolderWideSearch.folder_wide_search(src)


def process_results(data, src, verbose=False):
    print_progress_bar(0, 100, prefix='PeakFilter progress:')
    logFilePath = 'peakfilter.log'
    logFilePath = os.path.join(src, logFilePath)
    # Create logger and its file handler
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(logFilePath)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # Write initial information in log file
    logger.info('Starting Database Search. Input dataframe ("%s") has %d rows.',
                data.src, len(data.index))
    stepDst = os.path.join(src, 'step_by_step')
    if verbose and not os.path.isdir(stepDst):
        os.makedirs(stepDst)
    stepNum = 1
    filteredData = remove_reverse(data)
    logger.info("Reverse sequences removed")
    stepNum = _update_status(filteredData, stepDst, verbose, stepNum)
    filteredData = remove_only_identified_by_site(filteredData)
    logger.info("Removed by identified site")
    stepNum = _update_status(filteredData, stepDst, verbose, stepNum)
    filteredData = filter_localization_probability(filteredData)
    logger.info("Filtering by localization probability")
    stepNum = _update_status(filteredData, stepDst, verbose, stepNum)
    imputedData = pls(filteredData)
    logger.info("PLS missing value imputation")
    stepNum = _update_status(imputedData, stepDst, verbose, stepNum)
    outFileName = 'peakfilter_{0}.csv'.format(parameters['polarity'].lower())
    data.to_csv(os.path.join(src, outFileName), index=False)
    # Update progress bar
    print_progress_bar(100, 100, prefix='PeakFilter progress:')
    logger.info('PeakFilter completed. Output dataframe has %d rows.\n',
                len(data.index))
    handler.close()
    logger.removeHandler(handler)


def plot_ms_results(data, src, verbose=False):
    cov = plot_point_cov()
    scores, weights = pca(data)
    scores, weights = plsda(data)
    scores, weights = plsr(data)
    figures = box(data)

