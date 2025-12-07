import logging
import os
import warnings
import pandas as pd

from lipidomics import print_progress_bar

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

def convert_files(data, src, fasta,  verbose=False):
    print_progress_bar(0, 100, prefix='PeakFilter progress:')
    logFilePath = 'peakfilter.log'
    logFilePath = os.path.join(src, logFilePath)
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