import logging
import os
import threading
from typing import Optional

from lipidomics.PeakFilter import qc_rsd_ratio
from lipidomics.utils import extract_zip_files, extraction, print_progress_bar


class MetaboConversion:

    def __int__(self, srcpath: str, dstpath: str, vendor: Optional[str] = None,
                experiment_mode: Optional[str] = None, dda_top: Optional[int] = None,
                instrument: Optional[str] = None):
        self.srcpath = srcpath
        self.dstpath = dstpath

        if vendor is None:
            vendor = "thermo"
        self.vendor = vendor

        if experiment_mode is None:
            experiment_mode = "LC-MS"
        self.experiment_mode = experiment_mode

        if instrument is None:
            instrument = "orbitrap"
        self.instrument = instrument

        if dda_top is None:
            dda_top = 5
        self.dda_top = dda_top

    def converter(self):
        extract_zip_files(self.srcpath)
        raw_files = [os.path.join(self.srcpath, file) for file in os.listdir(self.srcpath) if file.endswith(".mzML")]
        # add function to update status
        threads = []
        for raw_file in raw_files:
            thread = threading.Thread(target=self.conversion_process,
                                      args=(raw_file, self.dstpath),
                                      daemon=True)
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

    def conversion_process(self, raw_file):
        extraction(raw_file, self.dstpath)


def _update_status(data, stepDst, verbose, stepNum):
    print_progress_bar(INCREMENT * stepNum, 100, prefix='PeakFilter progress:')
    if verbose:
        outFileName = 'peakfilter_step_{:02d}.csv'.format(stepNum)
        data.to_csv(os.path.join(stepDst, outFileName), index=False)
    stepNum += 1
    return stepNum


class PeakFiltering:
    def __int__(self, mzFixedError: float, mzPPMError: float, numIsotopes: int, dst: str):
        self.mzFixedError = 0.005
        self.mzPPMError = 4.0
        self.numIsotopes = 4
        self.dst = dst

    def peak_filter(self, data, verbose=False):
        print_progress_bar(0, 100, prefix='PeakFilter progress:')
        logFilePath = 'peakfilter.log'
        logFilePath = os.path.join(self.dst, logFilePath)
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
        stepDst = os.path.join(self.dst, 'step_by_step')
        if verbose and not os.path.isdir(stepDst):
            os.makedirs(stepDst)
        stepNum = 1

        # qc
        qcRatio = qc_rsd_ratio(data)

        # same as lipidomics pipeline minus the solvent, salt and whatnot removal

        # includes feature correspondence with matches features across different samples