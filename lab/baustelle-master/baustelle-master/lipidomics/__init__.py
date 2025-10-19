import logging
import os
import warnings
import pandas


from lipidomics.Conversion import Conversion
from lipidomics.MSSearch import bulk_structure_search
from lipidomics.PeakFilter import qc_rsd_ratio, correct_retention_time, remove_isotopes
from lipidomics.utils import print_progress_bar, extract_zip_files
from metabolomics.SVMclassify import calFDR

warnings.simplefilter(action='ignore', category=FutureWarning)
INCREMENT = 100.0 / 18


def _update_status(data, stepDst, verbose, stepNum):
    print_progress_bar(INCREMENT * stepNum, 100, prefix='Lipidomics progress:')
    if verbose:
        # Create a CSV file with the whole processed dataframe
        outFileName = 'lipidomics_step_{:02d}.csv'.format(stepNum)
        data.to_csv(os.path.join(stepDst, outFileName), index=False)
    stepNum += 1
    return stepNum


def file_converter(src, verbose=False):
    print_progress_bar(0, 100, prefix='Conversion progress:')
    logFilePath = 'file_conversion.log'
    logFilePath = os.path.join(src, logFilePath)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(logFilePath)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.info('Starting conversion')
    stepDst = os.path.join(src, 'step_by_step')
    if verbose and not os.path.isdir(stepDst):
        os.makedirs(stepDst)
    stepNum = 1
    zip_files = [os.path.join(src, file) for file in os.listdir(src) if file.endswith(".zip")]
    if len(zip_files) > 0:
        extract_zip_files(src)
        logger.info('Unzipping files. Input folder  has %d files.', len(zip_files))
    stepNum = _update_status(zip_files, stepDst, verbose, stepNum)
    raw_files = [os.path.join(src, file) for file in os.listdir(src) if file.endswith(".mzML")]
    dataContainer = Conversion.converter(src)
    logger.info('Starting extraction. Input folder  has %d files.', len(raw_files))
    print_progress_bar(100, 100, prefix='Conversion progress:')
    logger.info('Conversion completed. Output dataframe has %d rows.\n', len(dataContainer.index))
    handler.close()
    logger.removeHandler(handler)
    return dataContainer


def peak_filter(data, src, fdrtype, verbose=False):
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
    logger.info('Starting PeakFilter. Input dataframe ("%s") has %d rows.',
                data.src, len(data.index))
    # Prepare the folder structure to store the intermediate files
    stepDst = os.path.join(dst, 'step_by_step')
    if verbose and not os.path.isdir(stepDst):
        os.makedirs(stepDst)
    stepNum = 1
    ##  make function that grabs all the qc file names
    if len(qc_files) > 0:
        qc_metrics = qc_rsd_ratio(data)
        logger.info(("QC Sample Calculations completed. %.1f%% samples between"
                     " %d%% and %d%% QC-RSD"))
    stepNum = _update_status(data, stepDst, verbose, stepNum)
    filtData = remove_low_intensities(data)
    stepNum = _update_status(data, stepDst, verbose, stepNum)
    filtData = process_features(data)
    stepNum = _update_status(data, stepDst, verbose, stepNum)
    filtData = correct_retention_time(data)
    stepNum = _update_status(data, stepDst, verbose, stepNum)
    iltData = remove_isotopes(data)
    stepNum = _update_status(data, stepDst, verbose, stepNum)
    if fdrtype == 'database':
        try:
            fdrValue = FalseDiscoveryRate.get_fdr(data, parameters)
            message = ("False Discovery Rate for selected data and parameters: "
                       "{0:.2%}").format(fdrValue)
        except ValueError as e:
            message = 'ValueError: ' + e.args[0]
        except Exception as oe:
            message = 'OtherError: ' + oe.args[0]
        logger.info(message)
        stepNum = _update_status(data, stepDst, verbose, stepNum)
    else:
        try: fdrValue = calFDR(data)

    stepNum = _update_status(data, stepDst, verbose, stepNum)
    summary = create_summary(data)
    stepNum = _update_status(data, stepDst, verbose, stepNum)
    data['Polarity'] = parameters['polarity']
    outFileName = 'peakfilter_{0}.csv'.format(parameters['polarity'].lower())
    data.to_csv(os.path.join(dst, outFileName), index=False)
    # Update progress bar
    print_progress_bar(100, 100, prefix='PeakFilter progress:')
    logger.info('PeakFilter completed. Output dataframe has %d rows.\n',
                len(data.index))
    handler.close()
    logger.removeHandler(handler)


def msdatabase_search(data, src,  verbose=False):
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
    bulk_structure_search(data)

def msndata_annotation(data, src, verbose=False):
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
    logger.info('Starting PeakFilter. Input dataframe ("%s") has %d rows.',
                 data.src, len(data.index))
    # Prepare the folder structure to store the intermediate files
    stepDst = os.path.join(src, 'step_by_step')
    if verbose and not os.path.isdir(stepDst):
        os.makedirs(stepDst)
    stepNum = 1
    # extract xic
    # compose lipids







