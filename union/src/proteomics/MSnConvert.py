import os
import sys
import gzip
import time
import glob
import shutil
import logging
import numpy as np
import pandas as pd
from scipy.integrate import trapz

COMPUTE_CENTROIDS = False

# flag for computing peak areas (trapezoidal integrations are slow)
COMPUTE_AREAS = False


class Spectra:
    """Container for Spectrum objects."""

    def __init__(self, count=15, intensity=500, msn_level=2, header_block=None, parent=None):
        self.count = count  # minimum peak count
        self.intensity = intensity  # minimum total absolute intensity
        self.parent = parent  # parent object
        self.version = 1.04  # this program version
        self.xcalibur_version = None  # Xcalibur version
        self.msconvert_version = None  # Proteowizard version
        self.instrument = None  # Instrument type
        self.instrument_SN = None  # Instrument serial number
        self.msn = msn_level  # msn level being extracted
        self.spectra = []  # list of Spectrum objects
        self.mz_zero = 0  # how many scans had no m/z value
        self.z_zero = 0  # how many scans had no charge state
        self.freq = {}  # QC check on decimal places in m/z values
        self.get_versions(header_block)

    def get_versions(self, header_block):
        """Get Proteowizard version numbers."""
        if header_block:
            for i, line in enumerate(header_block):
                if line.startswith('cvParam: Xcalibur'):
                    self.xcalibur_version = header_block[i - 1].split('version:')[1].strip()
                if line.startswith('cvParam: ProteoWizard software'):
                    self.msconvert_version = header_block[i - 1].split('version:')[1].strip()
                if line.startswith('cvParam: instrument serial number'):
                    try:
                        self.instrument_SN = line.split('number,')[1].strip()
                    except IndexError:
                        self.instrument_SN = '0'
                    try:
                        self.instrument = header_block[i - 1].split('cvParam:')[1].strip()
                    except IndexError:
                        self.instrument = 'unknown'
        return  # nothing happens if there is no header_block

    def add(self, spectrum):
        """Add a spectrum if it meets the criteria."""
        if (spectrum.intensity >= self.intensity and
                len(spectrum.mz_array) >= self.count and
                spectrum.msn == self.msn):
            if spectrum.m_over_z == 0.0:
                spectrum.m_over_z = spectrum.cv_moverz
            self.check_spectrum(spectrum)
            if spectrum.m_over_z == 0.0:
                self.substitute_mz(spectrum)
            if spectrum.charge == [] or spectrum.charge[0] == 0:
                self.determine_charge(spectrum)
            self.spectra.append(spectrum)

    def determine_charge(self, spectrum):
        """Tests unknown charge state scans for 1+ or 2+/3+."""
        total_int = sum(spectrum.int_array)
        test_int = 0.0
        for i, mz in enumerate(spectrum.mz_array):
            if mz < spectrum.m_over_z:
                test_int += spectrum.int_array[i]
        if test_int >= 0.90 * total_int:
            spectrum.charge = [1]
        else:
            spectrum.charge = [2, 3]

    def substitute_mz(self, spectrum):
        """Sometimes the m/z value in userParam: line is zero so get value from scan header string."""
        m_over_z = spectrum.filter_string.split('string,')[-1]
        spectrum.m_over_z = float(m_over_z.split('@')[0].split()[-1])

    def check_spectrum(self, spectrum):
        """Checks spectrum for missing charge state, missing m/z, and checks accuracy of m/z."""
        if spectrum.m_over_z == 0.0:
            self.mz_zero += 1
        if spectrum.charge == [] or spectrum.charge[0] == 0:
            self.z_zero += 1
        places = len(spectrum.mzstr) - (spectrum.mzstr.index('.') + 1)
        self.freq[places] = self.freq.setdefault(places, 0) + 1

    def report(self):
        """Prints some summary stats."""
        print('......%s m/z values were zero and replaced by m/z in scan label' % self.mz_zero)
        print('......%s charge states were zero or missing' % self.z_zero)
        print('......m_over_z decimal places frequency counts:')
        for k in sorted(self.freq.keys()):
            print('.........%d places occurred %d times' % (k, self.freq[k]))

    def write_msn(self, msn_name, low=0.0, high=2000.0):
        """Write spectra in MSn format (MS2 and MS3 are same format)."""
        msn = open(msn_name, 'w')

        # write header lines
        msn.write('H\tCreationDate\t' + time.ctime() + '\n')
        msn.write('H\tInstrument\t' + self.instrument + '\n')
        msn.write('H\tInstrumentSerialNumber\t' + self.instrument_SN + '\n')
        msn.write('H\tExtractor\t' + 'Proteowizard + PAW\n')
        msn.write('H\tExtractorVersion\t1.03\n')
        msn.write('H\tExcaliburVersion\t' + self.xcalibur_version + '\n')
        msn.write('H\tMSConvertVersion\t' + self.msconvert_version + '\n')
        msn.write('H\tSourceFile\t' + os.path.basename(msn_name) + '.RAW\n')

        for spectrum in self.spectra:
            spectrum.write_msn_block(msn, low, high)

        msn.close()

    # end Spectra class


class Spectrum:
    def __init__(self, parent, block):
        """Parses info out of spectrum blocks."""
        # initializations
        self.parent = parent  # pointer to Spectra (parent) object
        get_mz = True  # flag
        self.mz_array = []  # holds m/z values
        self.int_array = []  # holds intensity values
        self.charge = []  # can have multiple charge states for low res data
        self.scan = None  # scan number
        self.cv_moverz = 0.0  # also m/z of precursor?
        self.m_over_z = 0.0  # m/z value of precursor
        self.mzstr = ''  # scan header string (has useful data)

        # parse out the desired items
        for line in block:
            if line.startswith('id:') and not self.scan:
                self.scan = int(line.split('scan=')[-1])
            if line.startswith('defaultArrayLength:'):
                self.count = int(line.split()[1])  # length of data arrays
            if line.startswith('cvParam:'):
                if 'ms level,' in line:
                    self.msn = int(line.split('ms level,')[-1])
                if 'total ion current,' in line:
                    self.intensity = float(line.split('total ion current,')[-1])
                if 'spectrum title,' in line:
                    self.title = line.split('spectrum title,')[-1].split()[0]
                if 'scan start time,' in line:
                    self.rt_min = float(line.split('time,')[-1].split(',')[0])
                if 'filter string,' in line:
                    self.filter_string = line.split('filter string, ')[-1]
                if 'ion injection time,' in line:
                    self.inj_time_ms = float(line.split('ion injection time,')[-1].split(',')[0])
                if 'selected ion m/z,' in line:  # this seems to be rounded to 2 decimal places
                    self.cv_moverz = float(line.split('selected ion m/z,')[-1].split(',')[0])
                if 'charge state,' in line:
                    self.charge.append(int(line.split('charge state,')[-1]))
            if line.startswith('userParam:'):
                if 'Monoisotopic M/Z:,' in line:
                    self.m_over_z = float(line.split(',')[1])
                    self.mzstr = line.split(',')[1].strip()
                    if '.' not in self.mzstr:
                        self.mzstr += '.'
            if line.startswith('binary:'):
                if get_mz:
                    self.mz_array = [float(x) for x in line.split(']')[1].split()]
                    get_mz = False
                else:
                    self.int_array = [float(x) for x in line.split(']')[1].split()]

        # check that array lengths match
        if len(self.mz_array) != self.count or len(self.int_array) != self.count \
                or len(self.mz_array) != len(self.int_array):
            print('...WARNING: non-equal ion and intensity counts!')

        return

    def console_dump(self):
        """Prints parsed values to console."""
        print('\nScan: %d, MSn: %d, Int: %0.0f, Title: %s, RT: %0.2f, Inj: %0.2f, M/Z: %0.2f, Charge: %s' %
              (self.scan, self.msn, self.intensity, self.title, self.rt_min,
               self.inj_time_ms, self.m_over_z, self.charge))
        print('number of data points:', len(self.mz_array), len(self.int_array))

    def write_msn_block(self, fout, low, high):
        """Writes self as one spectrum in an MSn file."""
        fout.write('S\t%s\t%s\t%0.5f\n' % (self.scan, self.scan, self.m_over_z))
        fout.write('I\tText\t%s\n' % self.filter_string)
        fout.write('I\tMSn\t%d\n' % self.msn)
        fout.write('I\tRTime\t%0.2f\n' % self.rt_min)
        fout.write('I\tTIC\t%0.1f\n' % self.intensity)
        for z in self.charge:
            MHplus = (self.m_over_z * float(z)) - (z - 1) * (1.007825)
            fout.write('Z\t%d\t%0.5f\n' % (z, MHplus))
        for i in range(self.count):
            if low <= self.mz_array[i] <= high:
                fout.write('%0.4f %0.2f\n' % (self.mz_array[i], self.int_array[i]))
        return

    # end Spectrum class


class MSnConvert:
    def __init__(self, path: str, msn_total: int, msn_level: int):
        self.msn_total = msn_total  # number of MSn scans written to MSn files
        self.path = path
        self.msn_level = msn_level

    def start_processing(self):
        raw_name_list = [os.path.join(self.path, file) for file in os.listdir(self.path) if file.endswith(".raw")]
        logFilePath = 'file_conversion.log'
        logFilePath = os.path.join(self.path, logFilePath)
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(logFilePath)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(asctime)s] %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.info('Conversions starting at....', time.ctime())
        # build the MSConvert command options
        for raw_name in raw_name_list:
            logger.info('Converting RAW to Text%s' % (os.path.basename(raw_name)))
            lc_name = os.path.splitext(os.path.basename(raw_name))[0]
            if raw_name.lower().endswith('.raw'):
                msconvert_name = raw_name[:-4] + '.txt.gz'

            command_line = raw_name + '2' + '2' + ' --text --gzip'
            if not os.path.exists(raw_name):
                logger.info('MSConvert ' + command_line)
                os.system('START "MSConvert" /WAIT /MIN /LOW CMD /C MSConvert ' + command_line)
            else:
                logger.info('Skipping conversion of:', lc_name, )

            txt_name_short = os.path.splitext(os.path.basename(raw_name))[0] + '.txt.gz'
            txt_name = os.path.join(os.path.dirname(raw_name), txt_name_short)
            if os.path.exists(txt_name):
                self.process_msn_level(lc_name, 2)
        self.move_files()

        def process_ms3_reporter_ions(self, lc_name):
            """Parses ms2 and ms3 spectrum blocks; gets scan numbers, and reporter ion data.
            """
            ms2_count = 0  # MS2 scan counter
            ms3_count = 0  # MS3 scan counter

            # parse files to get ms2 scan numbers, ms3 scan numbers, and data
            ms_dict = {}  # used to link MS3 scan numbers to ms2 scan numbers
            spectrum_flag = False  # limits parsing to spectrum blocks
            msn_level = None  # the MSn level of the scan
            #        mz_arr_flag = False     # True if data array is m/z values
            ms1_prev = 0  # previous (maybe current?) MS1 scan number
            #        moverz_key = ''         # relevant string from scan header line to link MS2 and MS3
            dict_list = []  # keeps track of two scan cycles to support RTS data
            buff = []

            # read lines in MSConvert file
            for k, line in enumerate(gzip.open(self.txt_name, 'rt')):
                line = line.strip()  # remove leading, trailing white space

                # look for each spectrum block
                if line.startswith('spectrum:') or line.startswith('chromatogramList'):
                    spectrum_flag = True
                    if buff:  # parse the previous spectrum block
                        for buff_line in buff:
                            if buff_line.startswith('cvParam: ms level,'):  # get MSn level (2 or 3)
                                msn_level = int(buff_line.split()[-1])

                                # MS2 scan parsing
                                if msn_level == 2:
                                    ms2_count += 1
                                    if (ms2_count % 1000) == 0:
                                        for obj in self.log_obj:
                                            print('......%d MS2 scans processed...' % ms2_count, file=obj)
                                    ms1_scan = self.parse_ms2_scan(buff, ms_dict)

                                    # look for next instrument cycle (a new MS1 scan)
                                    if ms1_scan != ms1_prev:
                                        dict_list.append(ms_dict)
                                        ms_dict = {}
                                        ms1_prev = ms1_scan

                                # MS3 scan parsing
                                elif msn_level == 3:
                                    ms3_count += 1
                                    self.parse_ms3_scan(buff, ms_dict, dict_list, lc_name)

                                break

                    # reset buffer
                    buff = []

                # spectrum_flag skips header lines until first spectrum block. Stops at chromato info.
                if line.startswith('chromatogramList'):
                    spectrum_flag = False

                # save lines in buffer if inside a spectrum block
                if spectrum_flag:
                    buff.append(line)

            # update total MS3 counter
            self.reporter_total += ms3_count

    def parse_ms2_scan(self, buffer, ms_dict):
        """Parses an MS2 scan block.
        """
        # read lines
        for line in buffer:

            # get the scan number
            if line.startswith('id:') and 'scan=' in line:
                scan_num = int(line.split()[-1].split('=')[-1])

            # get MS1 scan number
            if line.startswith('spectrumRef: '):
                ms1_scan = int(line.split()[-1].split('=')[-1])

            # get MS2 dissociation key (m/z value) and link to MS2 scan number
            if line.startswith('cvParam: filter string'):
                if '@cid3' in line:
                    moverz_key = line.split('@cid3')[0].split()[-1]
                elif '@hcd3' in line:
                    moverz_key = line.split('@hcd3')[0].split()[-1]
                else:
                    for obj in self.log_obj:
                        print('WARNING: dissociation key (@cid or @hcd) not found', file=obj)
                    return
                ms_dict[moverz_key] = scan_num

        return ms1_scan

    def process_msn_level(self, lc_name, msn_level=2, ion_count=15, min_intensity=100.0, reporter_ions=False):
        """Converts one Proteowizard TEXT formatted file to MSn format.
        Includes RT, etc. in MSn Information (I) lines. -PW June 2014
        Added summary statistics: missing charge states and m/z values -PW 6/15/2014
        Added extraction of reporter ions (if applicable) -PW 20180711
        """
        # get stuff for filenames
        if msn_level == 2:
            msn_extension = '.ms2'
        else:
            msn_extension = '.ms3'

        # get the header lines for Spectra setup
        header_block = []
        block = []
        in_spec = False  # this skips lines until first spectrum block
        spectra = None  # initial value
        for obj in self.log_obj:
            print('...Starting %s.txt.gz file scan' % (lc_name,), file=obj)
        for line in gzip.open(self.txt_name, 'rt'):
            line = line.strip()
            if line.startswith('chromatogramList'):
                break  # towards end of file after spectrum info
            if spectra is None:
                header_block.append(line)
            if line.startswith('spectrum:'):
                # create container for all spectra when at first "spectrum" line
                if spectra is None:
                    spectra = Spectra(ion_count, min_intensity, msn_level, header_block, self)
                # regular spectrum block processing
                in_spec = True
                if block:  # process previous spectrum block
                    spectrum = Spectrum(spectra, block)
                    spectra.add(spectrum)
                    if reporter_ions:
                        centroids, areas, heights = self.process_tmt_data(spectrum.mz_array, spectrum.int_array,
                                                                          spectrum.scan)
                        self.tmt_data.append(
                            Reporter_ion(lc_name, spectrum.scan, spectrum.scan, centroids, areas, heights))
                    block = []  # reset block
            if in_spec:
                block.append(line)

        # need to parse last block
        if block:
            spectrum = Spectrum(spectra, block)
            spectra.add(spectrum)
            if reporter_ions:
                centroids, areas, heights = self.process_tmt_data(spectrum.mz_array, spectrum.int_array, spectrum.scan)
                self.tmt_data.append(Reporter_ion(lc_name, spectrum.scan, spectrum.scan, centroids, areas, heights))

        # write diagnostic stats from conversion
        for obj in self.log_obj:
            print('...Diagnostics for:', lc_name, file=obj)
        if spectra:
            spectra.report()

            # write data in desired formats
            total_scans = len(spectra.spectra)
            self.msn_total += total_scans
            for obj in self.log_obj:
                print('...writing MS%s file: %d scans passed cutoffs' % (msn_level, total_scans), file=obj)
            msn_name = os.path.join(self.raw_path, lc_name + msn_extension)
            spectra.write_msn(msn_name)
        spectra = None
        return

    def move_files(self):
        """Moves .ms2 and PAW_tmt.txt file into msn_files folder"""
        # get paths for source and destination folders
        source_folder = self.raw_path
        destination_folder = os.path.join(os.path.dirname(source_folder), 'msn_files')
        if not os.path.exists(destination_folder):
            os.mkdir(destination_folder)

        # get a list of all files to move
        files_to_move = []
        for pattern in ['*.ms2', '*.ms3', '*.PAW_tmt.txt']:
            files_to_move += glob.glob(pattern)

        # move the files (checks and deletes any files with the same names)
        for file in files_to_move:
            source = os.path.join(source_folder, file)
            dest = os.path.join(destination_folder, file)
            if os.path.exists(dest):
                os.remove(dest)
            shutil.move(source, dest)
        return
