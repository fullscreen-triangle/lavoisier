import logging
import os
import threading
from typing import Optional

from lipidomics.PeakFilter import qc_rsd_ratio
from lipidomics.utils import extract_zip_files, extraction, print_progress_bar


class Conversion:

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


INCREMENT = 100.0 / 18


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


class MSDataBaseSearch:
    def __int__(self, dstsrc: str, targetAdducts: Optional[list] = None, lipidCategories: Optional[list] = None):
        self.dstsrc = dstsrc

        if targetAdducts is None:
            targetAdducts = []
        self.targetAdducts = targetAdducts

        if lipidCategories is None:
            lipidCategories = []
        self.lipidCategories = lipidCategories

    # def ms_database_search(self ):


class Chromatograms:
    def __int__(self, dstsrc: str, lctype: Optional[str] = None, usr_xic_ppm: Optional[int] = None,
                usr_core_num: Optional[int] = None, os_type: Optional[str] = None):
        self.dstsrc = dstsrc

        if lctype is None:
            lctype = 'uplc'
        self.lctype = lctype

    # def extract_chromatogram(self,):


class RuleBasedMSn:

    def __init__(self, srcdir: str, outdir: str):
        self.srcdir = srcdir
        self.outdir = outdir
        self.param_dict = {'fawhitelist_path_str': r'D:\lipidhunter\ConfigurationFiles\FA_Whitelist.xlsx',
                           'mzml_path_str': r'D:\lipidhunter\test\mzML\PL_Neg_Waters_qTOF.mzML',
                           'img_output_folder_str': r'D:\lipidhunter\Temp\Test2',
                           'xlsx_output_path_str': r'D:\lipidhunter\Temp\Test2\t2.xlsx',
                           'lipid_specific_cfg': r'D:\lipidhunter\ConfigurationFiles\PL_specific_ion_cfg.xlsx',
                           'hunter_start_time': '2017-12-21_15-27-49',
                           'vendor': 'waters', 'experiment_mode': 'LC-MS', 'lipid_class': 'PC',
                           'charge_mode': '[M+HCOO]-',
                           'rt_start': 20.0, 'rt_end': 25.0, 'mz_start': 700.0, 'mz_end': 800.0,
                           'rank_score': True, 'rank_score_filter': 27.5, 'score_filter': 27.5,
                           'isotope_score_filter': 75.0, 'fast_isotope': False,
                           'ms_th': 1000, 'ms_ppm': 20, 'ms_max': 0, 'pr_window': 0.75, 'dda_top': 6,
                           'ms2_th': 10, 'ms2_ppm': 50, 'ms2_infopeak_threshold': 0.001,
                           'score_cfg': r'D:\lipidhunter\ConfigurationFiles\Score_cfg.xlsx',
                           'hunter_folder': r'D:\lipidhunter',
                           'core_number': 4, 'max_ram': 10, 'img_type': u'png', 'img_dpi': 300}

        # then hunt lipids


class QSAR:
    def __init__(self, srcdir: str, outdir: str):
        self.srcdir = srcdir
        self.outdir = outdir

        # def smilesFromSpectra(self):

        """This  is where the default function then metadata filter then peakprocessing
           The matchMS spectrum module has a metadata parameter containing peak information for ms1,
           include hierarchical qsar modelling notebook utils.py """

class SimilarityNetworks:
    def __init__(self, srcdir: str, outdir: str):
        self.srcdir = srcdir
        self.outdir = outdir

        """ Code is in Spectra file"""

