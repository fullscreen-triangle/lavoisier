import os
import sys
import gzip
import time
import glob
import shutil

import numpy as np
import pandas as pd
from scipy.integrate import trapz

COMPUTE_CENTROIDS = False
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
        self.freq = {}
        self.get_versions(header_block)

    def get_versions(self, header_block):
        """Get Proteowizard version numbers."""
        if header_block:
            for i, line in enumerate(header_block):
                if line.startswith('cvParam: Xcalibur'):
                    self.xcalibur_version = header_block[i-1].split('version:')[1].strip()
                if line.startswith('cvParam: ProteoWizard software'):
                    self.msconvert_version = header_block[i-1].split('version:')[1].strip()
                if line.startswith('cvParam: instrument serial number'):
                    try:
                        self.instrument_SN = line.split('number,')[1].strip()
                    except IndexError:
                        self.instrument_SN = '0'
                    try:
                        self.instrument = header_block[i-1].split('cvParam:')[1].strip()
                    except IndexError:
                        self.instrument = 'unknown'
        return

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


class Spectrum:
    def __init__(self, parent, block):
        """Parses info out of spectrum blocks."""
        # initializations
        self.parent = parent    # pointer to Spectra (parent) object
        get_mz = True           # flag
        self.mz_array = []      # holds m/z values
        self.int_array = []     # holds intensity values
        self.charge = []        # can have multiple charge states for low res data
        self.scan = None        # scan number
        self.cv_moverz = 0.0    # also m/z of precursor?
        self.m_over_z = 0.0     # m/z value of precursor
        self.mzstr = ''         # scan header string (has useful data)

        # parse out the desired items
        for line in block:
            if line.startswith('id:') and not self.scan:
                self.scan = int(line.split('scan=')[-1])
            if line.startswith('defaultArrayLength:'):
                self.count = int(line.split()[1]) # length of data arrays
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
                if 'selected ion m/z,' in line: # this seems to be rounded to 2 decimal places
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
            MHplus = (self.m_over_z * float(z)) - (z-1)*(1.007825)
            fout.write('Z\t%d\t%0.5f\n' % (z, MHplus))
        for i in range(self.count):
            if low <= self.mz_array[i] <= high:
                fout.write('%0.4f %0.2f\n' % (self.mz_array[i], self.int_array[i]))
        return


