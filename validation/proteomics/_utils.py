import os
import sys
import glob
import fnmatch
import gzip
import time
import re
import copy
from pprint import pprint
from collections import OrderedDict

import pandas as pd
import numpy as np


class FastaReader:
    """Reads FASTA entries from a file-like object.
    methods:
    __init__: basic constructor, no parameters.
    readProtein: reads one FASTA entry from a file object (text or zipped)
        arguments are "next_protein" and "file_obj"
        returns True (next protein) or False (EOF or not FASTA).
    written by Phil Wilmarth, OHSU, 2009.
    """

    def __init__(self, fasta_file):
        """Basic constructor function.  No parameters
        self._last_line retains the previous '>' line and
        self._valid is a dictionary of valid protein FASTA chars.
        """
        # attribute to save last line from previous read
        self._last_line = 'start value'
        self._file_obj = None
        self._fasta_file = fasta_file

        # list of valid amino acid characters
        self._valid = {'X': True, 'G': True, 'A': True, 'S': True, 'P': True, \
                       'V': True, 'T': True, 'C': True, 'L': True, 'I': True, \
                       'J': True, 'N': True, 'O': True, 'B': True, 'D': True, \
                       'Q': True, 'K': True, 'Z': True, 'E': True, 'M': True, \
                       'H': True, 'F': True, 'R': True, 'Y': True, 'W': True, \
                       'U': True, '*': True, '-': True}

        # get file object and save as attribute
        if not os.path.exists(fasta_file):
            ext_list = [('FASTA files', '*.fasta'), ('Zipped FASTA files', '*.gz'), ('All files', '*.*')]
            fasta_file = get_file(default_location, extension_list, title_string="Select FASTA file")
        try:
            if fasta_file.endswith('.gz'):
                self._file_obj = gzip.open(fasta_file, 'rt')
            else:
                self._file_obj = open(fasta_file, 'rt')
        except IOError:
            print('   WARNING: Fasta database could not be opened!')
            raise
        return

    def readNextProtein(self, next_protein, check_for_errs=False):
        """Loads one FASTA protein text entry into a Protein object.
        Returns True (protein entry found) or False (end of file).
        If "check_for_errs" flag is set, amino acid chars are checked.
        Written by Phil Wilmarth, OHSU, 2009.
        """
        # at first call, start reading lines
        if self._last_line == 'start value':
            self._last_line = self._file_obj.readline()
            if not self._last_line:
                self._file_obj.close()
                return (False)
            self._last_line = self._last_line.strip()

        # get next protein's info from _last_line
        if self._last_line.startswith('>'):
            next_protein.accession = self._last_line.split()[0][1:]
            start = len(next_protein.accession) + 2
            next_protein.description = self._last_line[start:]

        # return if empty line (EOF) or non-description line
        else:
            self._file_obj.close()
            return (False)

        # reset variables and read in next entry
        next_protein.sequence = ""
        line = self._last_line
        self._last_line = ""
        bad_char = {}
        while line:
            line = self._file_obj.readline()
            if not line:
                break
            else:
                testline = line.strip()
            if testline == '':
                continue

            # stop reading at next descriptor line (and save line)
            if line.startswith('>'):
                self._last_line = line.strip()

                # report bad characters if conditions were met
                bad_char = sorted(bad_char.keys())
                if len(bad_char) > 0 and check_for_errs:
                    print('   WARNING: unknown symbol(s) (%s) in %s' %
                          (''.join(bad_char), next_protein.accession))
                break

            # add next sequence line to protein's sequence
            else:
                line = line.rstrip()
                line = line.upper()
                if check_for_errs:  # checking chars slows down the program
                    for char in line:
                        if self._valid.get(char, False):
                            next_protein.sequence += char
                        else:
                            bad_char[char] = True
                else:  # blindly adding the line is faster...
                    next_protein.sequence += line

        # return (protein info retained in next_protein)
        return True

    # end class


class FileLoader:
    def __init__(self, file_list):
        self.peptideMassTol = 1.25  # search parent ion tolerance default (plus/minus in Da)
        self.modStrings = []  # list of variable modification symbols specified in search
        self.enzyme = True  # False if a no enzyme search was used (default=True)

        # get the list of TXT files and figure out instrument type
        folder = os.path.dirname(file_list[0])

        # parse the params file and get relevant settings
        self.params = CometParams()
        self.params.load_from_folder(folder)
        if self.params.peptide_mass_units == 'Da':
            self.peptideMassTol = self.params.peptide_mass_tolerance

        # Construct list of differential mods
        self.modStrings = self.generateModStrings()

        self.enzyme = self.parseEnzymeInfo()

        self.fileList = file_list

        self.txt_info_list = []
        self.frame = pd.DataFrame()
        print('\nProcessing TXT files in:', os.path.basename(folder), time.asctime())
        for i, file_name in enumerate(file_list):
            file_obj = open(file_name, 'rU')
            name = os.path.basename(file_name)[:-4]
            frame = pd.read_csv(file_obj, sep='\t')
            info = TxtInfo(name)
            self.txt_info_list.append(info)
            frame['TxtIdx'] = i

            # save the frame's contents
            self.frame = pd.concat([self.frame, frame])
            print('...%s had %s lines' % (info.basename, len(frame)))

        print('...%s total lines read in' % len(self.frame))

    def getFrame(self):
        return self.frame

    def getParams(self):
        return self.params

    def getPeptideMassTol(self):
        return self.peptideMassTol

    def generateModStrings(self):
        mod_list = []
        last_deltamass_seen = 0
        for k, v in self.params.variable_mods.items():
            if v[1] == 'X':  # skip invalid residues
                continue
            elif float(v[0]) == 0.0:
                continue
            last_deltamass_seen = '%+0.4f' % float(v[0])
            mod_list.append(v[1] + last_deltamass_seen)
        return mod_list

    def parseEnzymeInfo(self):
        if self.params.search_enzyme_number == 0:
            return False
        else:
            return True


class TxtInfo:
    """Container for information and stats about main data.
    """

    def __init__(self, full_txt_name):
        self.path = os.path.dirname(full_txt_name)  # path name of folder containing the TXT files
        self.basename = os.path.basename(full_txt_name)  # basename of the TXT file (no extension)
        self.target_top_hits = 0  # total number of target top hits
        self.decoy_top_hits = 0  # total number of decoy top hits
        self.target_scans = 0  # total number of target scans
        self.decoy_scans = 0  # total number of decoy scans
        self.target_matches_top = None  # multi-dimensional collection of counters for target matches
        self.decoy_matches_top = None  # multi-dimensional collection of counters for decoys
        self.target_matches_scan = None  # multi-dimensional collection of counters for target matches
        self.decoy_matches_scan = None  # multi-dimensional collection of counters for decoys


class BRTxtInfo:
    """Container for information and stats about each TXT file.
    """

    def __init__(self, full_txt_name):
        """full_txt_name should have file extension removed
        """
        self.path = os.path.dirname(full_txt_name)  # path name of folder containing the TXT files
        self.basename = os.path.basename(full_txt_name)  # basename of the TXT file (no extension)
        self.target_top_hits = 0  # total number of target top hits
        self.decoy_top_hits = 0  # total number of decoy top hits
        self.target_scans = 0  # total number of target scans
        self.decoy_scans = 0  # total number of decoy scans

        self.target_matches_top = None  # multi-dimensional collection of counters for target matches
        self.decoy_matches_top = None  # multi-dimensional collection of counters for decoys
        self.target_matches_scan = None  # multi-dimensional collection of counters for target matches
        self.decoy_matches_scan = None  # multi-dimensional collection of counters for decoys

        self.target_subclass = None  # multi-dimensional collection of counters for target matches
        self.decoy_subclass = None  # multi-dimensional collection of counters for decoys
        self.target_filtered = 0
        self.decoy_filtered = 0
        self.min_length = MIN_PEP_LEN  # minimum peptide length (should be passed in)
        self.maxMods = 3  # maximum number of mods per peptide (should be passed in)

    def getStats(self, frame, dm_list, z_list, ntt_list, mod_list, masses, scores):
        """Computes several stats on numbers of target and decoy matches
        masses are the deltamass windows
        scores are the score thresholds
        """
        # restrict by minimunm peptide length first
        len_frame = frame[(frame.Length >= self.min_length) & (frame.NumMods <= self.maxMods)]

        # get the global stats first
        self.target_top_hits = len(len_frame[len_frame.ForR == 'F'])
        self.decoy_top_hits = len(len_frame[len_frame.ForR == 'R'])
        #   print(self.basename, self.target_top_hits, self.decoy_top_hits, self.target_top_hits - self.decoy_top_hits)
        self.target_scans = len(len_frame[len_frame.ForR == 'F'].drop_duplicates(['start', 'end', 'Z']))
        self.decoy_scans = len(len_frame[len_frame.ForR == 'R'].drop_duplicates(['start', 'end', 'Z']))
        for dm in range(len(dm_list)):
            mass_frame = len_frame[(len_frame.dmassDa >= masses[dm]['low']) &
                                   (len_frame.dmassDa <= masses[dm]['high'])]
            for z in range(len(z_list)):  # z is one less than the charge state
                for ntt in range(len(ntt_list)):
                    for mod in range(len(mod_list)):
                        subclass_frame = mass_frame[(mass_frame.Z == z + 1) &
                                                    (mass_frame.ntt == ntt) &
                                                    (mass_frame.ModsStr == mod_list[mod])]

                        s = scores[dm][z][ntt][mod]
                        threshold = s.histo.DiscScore[s.threshold]

                        self.target_subclass[dm][z][ntt][mod] = len(subclass_frame[(subclass_frame.ForR == 'F') &
                                                                                   (
                                                                                               subclass_frame.NewDisc >= threshold)])
                        self.target_filtered += len(subclass_frame[(subclass_frame.ForR == 'F') &
                                                                   (subclass_frame.NewDisc >= threshold)])
                        self.decoy_subclass[dm][z][ntt][mod] = len(subclass_frame[(subclass_frame.ForR == 'R') &
                                                                                  (
                                                                                              subclass_frame.NewDisc >= threshold)])
                        self.decoy_filtered += len(subclass_frame[(subclass_frame.ForR == 'R') &
                                                                  (subclass_frame.NewDisc >= threshold)])


class DataInfoAndFilter:
    """Container for global stats on all loaded data.
    Uses TxtInfo objects to keep stats for each TXT file.
    Has aggregate counters and totaling method. Counters
    track both before filtering and post filtering counts.
    """

    def __init__(self, folder, frame, txt_info_list, dm_list, z_list, ntt_list, mod_list,
                 min_length=MIN_PEP_LEN, max_mods=2, parent_tol=2.5):
        import copy

        # main data passed in
        self.folder = folder  # full path to TXT files
        self.frame = frame  # pandas dataframe of TXT file contents and some extras
        self.pre_filter = txt_info_list  # list of TxtInfo objects for pre-filter stats
        self.post_filter = [copy.deepcopy(x) for x in txt_info_list]  # list of TxtInfo objects for post-filter stats
        self.dm_list = dm_list  # list of delta mass window names
        self.z_list = z_list  # list of allowed charge states (contiguous range)
        self.ntt_list = ntt_list  # list of number of tryptic termini (o, 1, 2)
        self.mod_list = mod_list  # full, ordered list of variable modification symbols starting with a space for unmodified residues

        # some restrictions and limits
        self.min_length = min_length  # minimum peptide length (should be passed in)
        self.max_mods = max_mods  # maximum number of mods per peptide (should be passed in)
        self.z_offset = min([int(z) for z in z_list])  # to map peptide charge to z-axis index
        self.z_max = max([int(z) for z in z_list])  # maximum peptide charge
        self.ntt_offset = min([int(ntt) for ntt in ntt_list])  # to map ntt range to index range
        self.ntt_max = max([int(ntt) for ntt in ntt_list])  # maximum ntt value
        self.parent_tol = parent_tol  # parent ion tolerance (should be passed in)

        # data structures for counting statistics
        self.short_target_top_hits = 0  # total number of target top hits below min length
        self.short_decoy_top_hits = 0  # total number of decoy top hits below min length
        self.short_target_scans = 0  # total number of target scans below min length
        self.short_decoy_scans = 0  # total number of decoy scans below min length
        self.pre_target_top_hits = 0  # total number of target top hits
        self.pre_decoy_top_hits = 0  # total number of decoy top hits
        self.pre_target_scans = 0  # total number of target scans
        self.pre_decoy_scans = 0  # total number of decoy scans
        self.pre_target_matches_top = None  # multi-dimensional collection of counters for target matches
        self.pre_decoy_matches_top = None  # multi-dimensional collection of counters for decoys
        self.pre_target_matches_scan = None  # multi-dimensional collection of counters for target matches
        self.pre_decoy_matches_scan = None  # multi-dimensional collection of counters for decoys
        self.post_target_top_hits = 0  # total number of target top hits
        self.post_decoy_top_hits = 0  # total number of decoy top hits
        self.post_target_scans = 0  # total number of target scans
        self.post_decoy_scans = 0  # total number of decoy scans
        self.post_target_matches_top = None  # multi-dimensional collection of counters for target matches
        self.post_decoy_matches_top = None  # multi-dimensional collection of counters for decoys
        self.post_target_matches_scan = None  # multi-dimensional collection of counters for target matches
        self.post_decoy_matches_scan = None  # multi-dimensional collection of counters for decoys

        # set up for log file
        ##        self.sqt_container = os.path.dirname(self.folder)
        ##        self.filtered_folder = os.path.join(self.sqt_container, 'filtered_files')
        self.filtered_folder = os.path.join(os.path.dirname(self.folder), 'filtered_files')
        self.log_file = open(os.path.join(self.folder, os.path.basename(self.folder) + '_PAW_log.txt'), 'a')
        self.write = [None, self.log_file]

        return

    def aggregate_pre_global_stats(self):
        """Sums up the per TXT stats for pre-filtered data
        """
        dm_list = ['All']  # we don't care about mass windows for aggregate stats

        # initialize counters and counter containers
        self.pre_target_top_hits = 0
        self.pre_decoy_top_hits = 0
        self.pre_target_scans = 0
        self.pre_decoy_scans = 0
        self.pre_target_matches_top = np.array(
            [[[[0 for mod in self.mod_list] for ntt in self.ntt_list] for z in self.z_list] for dm in dm_list])
        self.pre_decoy_matches_top = np.array(
            [[[[0 for mod in self.mod_list] for ntt in self.ntt_list] for z in self.z_list] for dm in dm_list])
        self.pre_target_matches_scan = np.array(
            [[[[0 for mod in self.mod_list] for ntt in self.ntt_list] for z in self.z_list] for dm in dm_list])
        self.pre_decoy_matches_scan = np.array(
            [[[[0 for mod in self.mod_list] for ntt in self.ntt_list] for z in self.z_list] for dm in dm_list])

        # sum over the individual TXT file data
        for txt_info in self.pre_filter:
            self.pre_target_top_hits += txt_info.target_top_hits
            self.pre_decoy_top_hits += txt_info.decoy_top_hits
            self.pre_target_scans += txt_info.target_scans
            self.pre_decoy_scans += txt_info.decoy_scans
            self.pre_target_matches_top += txt_info.target_matches_top
            self.pre_decoy_matches_top += txt_info.decoy_matches_top
            self.pre_target_matches_scan += txt_info.target_matches_scan
            self.pre_decoy_matches_scan += txt_info.decoy_matches_scan
        return

    def aggregate_post_global_stats(self):
        """Sums up the per TXT stats after filtering
        """
        dm_list = ['All']  # we don't care about mass windows for aggregate stats

        # initialize counters and counter containers
        self.post_target_top_hits = 0
        self.post_decoy_top_hits = 0
        self.post_target_scans = 0
        self.post_decoy_scans = 0
        self.post_target_matches_top = np.array(
            [[[[0 for mod in self.mod_list] for ntt in self.ntt_list] for z in self.z_list] for dm in self.dm_list])
        self.post_decoy_matches_top = np.array(
            [[[[0 for mod in self.mod_list] for ntt in self.ntt_list] for z in self.z_list] for dm in self.dm_list])
        self.post_target_matches_scan = np.array(
            [[[[0 for mod in self.mod_list] for ntt in self.ntt_list] for z in self.z_list] for dm in self.dm_list])
        self.post_decoy_matches_scan = np.array(
            [[[[0 for mod in self.mod_list] for ntt in self.ntt_list] for z in self.z_list] for dm in self.dm_list])

        # sum across the filtered TXT file data
        for txt_info in self.post_filter:
            self.post_target_top_hits += txt_info.target_top_hits
            self.post_decoy_top_hits += txt_info.decoy_top_hits
            self.post_target_scans += txt_info.target_scans
            self.post_decoy_scans += txt_info.decoy_scans
            self.post_target_matches_top += txt_info.target_matches_top
            self.post_decoy_matches_top += txt_info.decoy_matches_top
            self.post_target_matches_scan += txt_info.target_matches_scan
            self.post_decoy_matches_scan += txt_info.decoy_matches_scan
        return

    def get_pre_stats(self):
        """Computes numbers of target and decoy matches in various sliced and diced ways
        """
        import pprint

        for obj in self.write:
            print('\nCompiling pre-filter stats', time.asctime(), file=obj)
        # get stats on short peptides first
        self.short_target_top_hits = len(self.frame[(self.frame.Length < self.min_length) & (self.frame.ForR == 'F')])
        self.short_decoy_top_hits = len(self.frame[(self.frame.Length < self.min_length) & (self.frame.ForR == 'R')])
        self.short_target_scans = len(
            self.frame[(self.frame.Length < self.min_length) & (self.frame.ForR == 'F')].drop_duplicates(
                ['start', 'end', 'Z', 'TxtIdx']))
        self.short_decoy_scans = len(
            self.frame[(self.frame.Length < self.min_length) & (self.frame.ForR == 'R')].drop_duplicates(
                ['start', 'end', 'Z', 'TxtIdx']))
        for obj in self.write:
            print('...short peptide top hits: %s (%s)' % (self.short_target_top_hits, self.short_decoy_top_hits),
                  file=obj)
            print('...short peptide scans: %s (%s)\n' % (self.short_target_scans, self.short_decoy_scans), file=obj)

        # restrict by minimum peptide length, maximum number of mods, charge state range and ntt range first
        frame = self.frame[(self.frame.Length >= self.min_length) & (self.frame.NumMods <= self.max_mods) &
                           ((self.frame.Z >= self.z_offset) & (self.frame.Z <= self.z_max)) &
                           ((self.frame.ntt >= self.ntt_offset) & (self.frame.ntt <= self.ntt_max))]

        print('*** pre-filter ***:', len(self.frame), len(frame))

        dm_list = ['all']
        for i, txt_info_obj in enumerate(self.pre_filter):
            # create multidimensional counters for each TXT file
            self.pre_filter[i].target_matches_top = np.array(
                [[[[0 for mod in self.mod_list] for ntt in self.ntt_list] for z in self.z_list] for dm in dm_list])
            self.pre_filter[i].decoy_matches_top = np.array(
                [[[[0 for mod in self.mod_list] for ntt in self.ntt_list] for z in self.z_list] for dm in dm_list])
            self.pre_filter[i].target_matches_scan = np.array(
                [[[[0 for mod in self.mod_list] for ntt in self.ntt_list] for z in self.z_list] for dm in dm_list])
            self.pre_filter[i].decoy_matches_scan = np.array(
                [[[[0 for mod in self.mod_list] for ntt in self.ntt_list] for z in self.z_list] for dm in dm_list])

            # get the global stats first (this prevents scans from being overcounted; top hit ties can have different ntt or mods)
            self.pre_filter[i].target_top_hits = len(frame[(frame.TxtIdx == i) & (frame.ForR == 'F')])
            self.pre_filter[i].decoy_top_hits = len(frame[(frame.TxtIdx == i) & (frame.ForR == 'R')])
            self.pre_filter[i].target_scans = len(
                frame[(frame.TxtIdx == i) & (frame.ForR == 'F')].drop_duplicates(['start', 'end', 'Z']))
            self.pre_filter[i].decoy_scans = len(
                frame[(frame.TxtIdx == i) & (frame.ForR == 'R')].drop_duplicates(['start', 'end', 'Z']))
            y = self.pre_filter[i]
            for obj in self.write:
                print('...%s: all top hits %s (%s), all scans %s (%s), fdr %0.2f' %
                      (y.basename, y.target_top_hits, y.decoy_top_hits, y.target_scans, y.decoy_scans,
                       100.0 * y.decoy_scans / y.target_scans), file=obj)

            # get the peptide subclass stats
            for dm in range(len(dm_list)):  # different thresholds for each mass window
                for charge in self.z_list:  # different thresholds for each charge, z is one less than the charge state
                    z = int(charge) - self.z_offset
                    mass_frame = frame[(frame.dmassDa >= (-1) * self.parent_tol) & (frame.dmassDa <= self.parent_tol)]
                    for ntt in self.ntt_list:  # different thresholds for each number of tryptic termini
                        ntt_idx = int(ntt) - self.ntt_offset
                        for mod in range(
                                len(self.mod_list)):  # and different thresholds for each homogeneous modification state
                            subclass_frame = mass_frame[(mass_frame.TxtIdx == i) & (mass_frame.Z == int(charge)) & (
                                        mass_frame.ntt == int(ntt)) &
                                                        (mass_frame.ModsStr.map(lambda s: s[0]) == self.mod_list[
                                                            mod])]  # first get basic data subclass (need first char of ModsStr to get all)
                            self.pre_filter[i].target_matches_top[dm][z][ntt_idx][mod] = len(
                                subclass_frame[(subclass_frame.ForR == 'F')])
                            self.pre_filter[i].decoy_matches_top[dm][z][ntt_idx][mod] = len(
                                subclass_frame[(subclass_frame.ForR == 'R')])
                            self.pre_filter[i].target_matches_scan[dm][z][ntt_idx][mod] += len(
                                subclass_frame[(subclass_frame.ForR == 'F')].drop_duplicates(['start', 'end', 'Z']))
                            self.pre_filter[i].decoy_matches_scan[dm][z][ntt_idx][mod] += len(
                                subclass_frame[(subclass_frame.ForR == 'R')].drop_duplicates(['start', 'end', 'Z']))

        # get aggregate stats
        self.aggregate_pre_global_stats()
        for obj in self.write:
            try:
                print('\n...Aggregate top hits: %s (%s) %0.2f' %
                      (self.pre_target_top_hits, self.pre_decoy_top_hits,
                       100.0 * self.pre_decoy_top_hits / self.pre_target_top_hits), file=obj)
            except ZeroDivisionError:
                print(
                    '\n...Aggregate top hits: %s (%s) %0.2f' % (self.pre_target_top_hits, self.pre_decoy_top_hits, 0.0),
                    file=obj)
                # raise ZeroDivisionError # will this cause the program to terminate?
            try:
                print('...Aggregate scans: %s (%s) %0.2f\n' %
                      (self.pre_target_scans, self.pre_decoy_scans,
                       100.0 * self.pre_decoy_scans / self.pre_target_scans), file=obj)
            except ZeroDivisonError:
                print('...Aggregate scans: %s (%s) %0.2f\n' % (self.pre_target_scans, self.pre_decoy_scans, 0.0),
                      file=obj)
                # raise ZeroDivisionError
            print('All target subclass matches (all top hits):', file=obj)
            pprint.pprint(self.pre_target_matches_top, stream=obj)
            print('All decoy subclass matches (all top hits):', file=obj)
            pprint.pprint(self.pre_decoy_matches_top, stream=obj)
            print('All net subclass matches (all top hits):', file=obj)
            pprint.pprint(self.pre_target_matches_top - self.pre_decoy_matches_top, stream=obj)
            print('All target subclass matches (scans):', file=obj)
            pprint.pprint(self.pre_target_matches_scan, stream=obj)
            print('All decoy subclass matches (scans):', file=obj)
            pprint.pprint(self.pre_decoy_matches_scan, stream=obj)
            print('All net subclass matches (scans):', file=obj)
            pprint.pprint(self.pre_target_matches_scan - self.pre_decoy_matches_scan, stream=obj)

    def return_threshold(self, index_tuple, z_offset, ntt_offset, mod_list, dm_scores):
        """Returns the largest threshold associated with any modifications.
        (index_tuple) => (Z, ntt, mods_str);
            Z -> charge state; nnt -> number of tryptic termini; mods_str -> list of modification symbols in peptide
        z_offset => maps charge state to z-index,
        mod_list => full, ordered list of variable modifications specified in search
        dm_scores => subset of thresholds for respective deltamass window.
        """
        thresholds = [dm_scores[index_tuple[0] - z_offset][index_tuple[1] - ntt_offset][mod_list.index(mod)] for mod in
                      index_tuple[2]]
        return max(thresholds)

    def copy_params_files(self):
        """Copies any params files to filtered folder."""
        import shutil
        params_list = [p for p in os.listdir(self.folder) if p.endswith('.params')]
        print('params_list:', params_list)
        for param in params_list:
            try:
                shutil.copy2(os.path.join(self.folder, param), os.path.join(self.filtered_folder, param))
            except PermissionError:
                pass

    def filter_with_stats(self, mass_thresholds, score_thresholds):
        """Filters and computes numbers of target and decoy matches passing thresholds over the peptide subclasses
        """
        import pprint

        for obj in self.write:
            print('\nFiltering data and compiling stats', time.asctime(), file=obj)
        scores = np.array(score_thresholds)
        masses = mass_thresholds
        all_filter_frame = pd.DataFrame()

        # print out global limits, various lists, and threshold values
        for obj in self.write:
            print('...Minimum peptide length:', self.min_length, file=obj)
            print('...Maximum number of mods per peptide:', self.max_mods, file=obj)
            print('...DeltaMass list:', self.dm_list, file=obj)
            print('...Charge state list:', self.z_list, file=obj)
            print('...NTT list:', self.ntt_list, file=obj)
            print('...Modifications list:', self.mod_list, file=obj)
            print('...Delta mass windows:', file=obj)
            for i, z in enumerate(self.z_list):
                print('......Z = %s' % z, file=obj)
                for j, dm in enumerate(self.dm_list[:-1]):
                    print('.........DeltaMass %s: %0.4f to %0.4f' % (dm, masses[i][j].low, masses[i][j].high), file=obj)
            print('...Conditional score thresholds:', file=obj)
            pprint.pprint(scores, stream=obj)
            print(file=obj)

        # lets see what columns get loaded from TXT files
        print('Frame columns at start of filtering:')
        for col in self.frame.columns:
            print(col, self.frame[col].dtype)

        # restrict by minimum peptide length first
        frame = self.frame[(self.frame.Length >= self.min_length) & (self.frame.NumMods <= self.max_mods) &
                           ((self.frame.Z >= self.z_offset) & (self.frame.Z <= self.z_max)) &
                           ((self.frame.ntt >= self.ntt_offset) & (self.frame.ntt <= self.ntt_max))].copy()

        for i, txt_info_obj in enumerate(self.pre_filter):  # loop over all TXT file data
            filter_frame = pd.DataFrame()

            # create multidimensional counters for each TXT file
            self.post_filter[i].target_matches_top = np.array(
                [[[[0 for mod in self.mod_list] for ntt in self.ntt_list] for z in self.z_list] for dm in self.dm_list])
            self.post_filter[i].decoy_matches_top = np.array(
                [[[[0 for mod in self.mod_list] for ntt in self.ntt_list] for z in self.z_list] for dm in self.dm_list])
            self.post_filter[i].target_matches_scan = np.array(
                [[[[0 for mod in self.mod_list] for ntt in self.ntt_list] for z in self.z_list] for dm in self.dm_list])
            self.post_filter[i].decoy_matches_scan = np.array(
                [[[[0 for mod in self.mod_list] for ntt in self.ntt_list] for z in self.z_list] for dm in self.dm_list])

            # get the peptide subclass stats
            for dm in range(len(self.dm_list)):  # different thresholds for each mass window
                for charge in self.z_list:  # different thresholds for each charge, z is z_offset less than the charge state
                    z = int(charge) - self.z_offset
                    if dm < 2:  # these are simple inclusive windows
                        mass_frame = frame[(frame.dmassDa >= masses[z][dm].low) &
                                           (frame.dmassDa <= masses[z][dm].high) &
                                           (frame.TxtIdx == i) & (frame.Z == int(charge))].copy()
                    else:  # this is everything outside of the previous two windows
                        mass_frame = frame[
                            (((frame.dmassDa >= masses[z][dm].low) & (frame.dmassDa < masses[z][0].low)) |
                             ((frame.dmassDa > masses[z][0].high) & (frame.dmassDa < masses[z][1].low)) |
                             ((frame.dmassDa > masses[z][1].high) & (frame.dmassDa <= masses[z][dm].high))) &
                            (frame.TxtIdx == i) & (frame.Z == int(charge))].copy()

                    # this finds the most stringent threshold to test heterogeneously modified peptides against
                    mass_frame['IndexTuple'] = list(zip(mass_frame.Z, mass_frame.ntt, mass_frame.ModsStr))
                    mass_frame['TestValue'] = mass_frame.IndexTuple.map(
                        lambda index_tuple: self.return_threshold(index_tuple, self.z_offset, self.ntt_offset,
                                                                  self.mod_list, scores[dm]))

                    for ntt in self.ntt_list:  # different thresholds for each number of tryptic termini
                        ntt_idx = int(ntt) - self.ntt_offset
                        for mod in range(len(self.mod_list)):  # and different thresholds for each modification state
                            subclass_frame = mass_frame[
                                (mass_frame.ntt == int(ntt)) & (mass_frame.ModsStr == self.mod_list[mod]) &
                                (
                                            mass_frame.NewDisc >= mass_frame.TestValue)]  # get any data subclass hits above threshold
                            filter_frame = pd.concat(
                                [filter_frame, subclass_frame])  # save the filtered peptide subclass frame

                            # save some stats on the filtered subclass hits
                            self.post_filter[i].target_matches_top[dm][z][ntt_idx][mod] = len(
                                subclass_frame[subclass_frame.ForR == 'F'])
                            self.post_filter[i].decoy_matches_top[dm][z][ntt_idx][mod] = len(
                                subclass_frame[subclass_frame.ForR == 'R'])
                            self.post_filter[i].target_matches_scan[dm][z][ntt_idx][mod] += len(
                                subclass_frame[subclass_frame.ForR == 'F'
                                               ].drop_duplicates(['start', 'end', 'Z']))
                            self.post_filter[i].decoy_matches_scan[dm][z][ntt_idx][mod] += len(
                                subclass_frame[subclass_frame.ForR == 'R'
                                               ].drop_duplicates(['start', 'end', 'Z']))

            # get the global stats last from the filtered frame
            self.post_filter[i].target_top_hits = self.post_filter[i].target_matches_top.sum()
            self.post_filter[i].decoy_top_hits = self.post_filter[i].decoy_matches_top.sum()
            self.post_filter[i].target_scans = self.post_filter[i].target_matches_scan.sum()
            self.post_filter[i].decoy_scans = self.post_filter[i].decoy_matches_scan.sum()
            y = self.post_filter[i]
            for obj in self.write:
                print('...%s: filtered top hits %s (%s), filtered scans %s (%s), fdr %0.2f' %
                      (y.basename, y.target_top_hits, y.decoy_top_hits, y.target_scans,
                       y.decoy_scans, 100.0 * y.decoy_scans / y.target_scans), file=obj)
            # write the filtered scans to TXT file
            self.write_data(filter_frame, txt_info_obj.basename)
            all_filter_frame = pd.concat(
                [all_filter_frame, filter_frame])  # we may not need to keep this dataframe in memory

        # get aggregate stats
        self.aggregate_post_global_stats()
        for obj in self.write:
            try:
                print('\n...Aggregate filtered top hits: %s (%s) %0.2f' %
                      (self.post_target_top_hits, self.post_decoy_top_hits,
                       100.0 * self.post_decoy_top_hits / self.post_target_top_hits), file=obj)
            except ZeroDivisionError:
                print('\n...Aggregate filtered top hits: %s (%s) %0.2f' % (
                self.post_target_top_hits, self.post_decoy_top_hits, 0.0), file=obj)
            try:
                print('...Aggregate filtered scans: %s (%s) %0.2f\n' %
                      (self.post_target_scans, self.post_decoy_scans,
                       100.0 * self.post_decoy_scans / self.post_target_scans), file=obj)
            except ZeroDivisionError:
                print('...Aggregate filtered scans: %s (%s) %0.2f\n' % (
                self.post_target_scans, self.post_decoy_scans, 0.0), file=obj)
            print('Filtered target subclass matches (all top hits):', file=obj)
            pprint.pprint(self.post_target_matches_top, stream=obj)
            print('Filtered decoy subclass matches (all top hits):', file=obj)
            pprint.pprint(self.post_decoy_matches_top, stream=obj)
            print('Filtered net subclass matches (all top hits):', file=obj)
            pprint.pprint(self.post_target_matches_top - self.post_decoy_matches_top, stream=obj)
            print('Filtered target subclass matches (scans):', file=obj)
            pprint.pprint(self.post_target_matches_scan, stream=obj)
            print('Filtered decoy subclass matches (scans):', file=obj)
            pprint.pprint(self.post_decoy_matches_scan, stream=obj)
            print('Filtered net subclass matches (scans):', file=obj)
            pprint.pprint(self.post_target_matches_scan - self.post_decoy_matches_scan, stream=obj)

        return all_filter_frame

    def write_data(self, frame, basename):
        """Writes a pandas dataframe to a TXT file.
        """
        if not os.path.exists(self.filtered_folder):
            os.mkdir(os.path.join(self.filtered_folder))
        file_name = os.path.join(self.filtered_folder, basename + '_filtered.txt')
        ##        cols = ['start', 'end', 'Z', 'expM', 'SpRank', 'theoM', 'deltaCN', 'Xcorr', 'Sequence', 'Loci',
        ##                'NewDeltaCN', 'NewDisc', 'ntt', 'ForR', 'dmassDa', 'dmassPPM', 'Length',
        ##                'NumMods', 'ModsStr', 'IndexTuple', 'TestValue']
        ##        frame.to_csv(file_name, sep='\t', index=False, columns=cols)
        frame.to_csv(file_name, sep='\t', index=False)
        for i, obj in enumerate(self.write):
            print('......filtered TXT file written', file=obj)

        # get the list of passing scans and extract the SQT and MS2 information
        scan_list = list(zip(frame.start, frame.end, frame.Z))
        self.write_sqt(basename, scan_list)
        self.write_ms2(basename, scan_list)

    def write_sqt(self, basename, scan_list):
        """Writes a filtered SQT file to filtered_files folder
        """
        # see if there is anything to process
        sqt_out = os.path.join(self.filtered_folder, basename + '_filtered.sqt')
        if os.path.exists(os.path.join(self.folder, basename + '.sqt')):
            gz_flag = False
        elif os.path.exists(os.path.join(self.folder, basename + '.sqt.gz')):
            gz_flag = True
        else:
            print('in "write_sqt" fall through:')
            print('sqt_folder:', self.folder)
            print('basename:', basename)
            print(os.path.exists(os.path.join(self.folder, basename + '.sqt')))
            return
        if not scan_list:
            return

        # open the original SQT file and the output file
        if gz_flag:
            sqt_in = gzip.open(os.path.join(self.folder, basename + '.sqt.gz'))
        else:
            sqt_in = open(os.path.join(self.folder, basename + '.sqt'), 'rU')
        sqt_out = open(os.path.join(self.filtered_folder, basename + '_filtered.sqt'), 'w')

        # pass header lines and scan blocks that match scans in scan_list
        scan_set = set(scan_list)
        sqt_out.write('H\tComment\tFiltered SQT file created ' + time.asctime() + '\n')
        buff = []
        passes = False
        for line in sqt_in:
            line = line.strip()
            items = line.split('\t')
            if items[0] == 'H':
                sqt_out.write(line + '\n')
            if items[0] == 'S' and buff:
                for new in buff:
                    sqt_out.write(new + '\n')
                buff = []
                passes = False
            if items[0] == 'S' and (int(items[1]), int(items[2]), int(items[3])) in scan_set:
                passes = True
            if passes:
                buff.append(line)

        # need to process possible passing last scan
        if buff:
            for new in buff:
                sqt_out.write(new + '\n')
        for obj in self.write:
            print('......filtered SQT file written', file=obj)

        # close files and return
        sqt_in.close()
        sqt_out.close()
        return

    def write_ms2(self, basename, scan_list):
        """Writes a filtered MS2 file to 'path' folder
        """
        # see if there is anything to process
        ms2_out = os.path.join(self.filtered_folder, basename + '_filtered.ms2')
        if os.path.exists(os.path.join(self.folder, basename + '.ms2')):
            gz_flag = False
        elif os.path.exists(os.path.join(self.folder, basename + '.ms2.gz')):
            gz_flag = True
        else:
            return
        if not scan_list:
            return
        if gz_flag:
            ms2_in = gzip.open(os.path.join(self.folder, basename + '.ms2.gz'))
        else:
            ms2_in = open(os.path.join(self.folder, basename + '.ms2'), 'r')
        ms2_out = open(os.path.join(self.filtered_folder, basename + '_filtered.ms2'), 'w')
        scan_set = set([(x[0], x[1]) for x in scan_list])
        ms2_out.write('H\tComment\tFiltered MS2 file created ' + time.asctime() + '\n')
        buff = []
        passes = False
        for line in ms2_in:
            line = line.strip()
            items = line.split('\t')
            if items[0] == 'H':
                ms2_out.write(line + '\n')
            if items[0] == 'S' and buff:
                for new in buff:
                    ms2_out.write(new + '\n')
                buff = []
                passes = False
            if items[0] == 'S' and (int(items[1]), int(items[2])) in scan_set:
                passes = True
            if passes:
                buff.append(line)
        if buff:
            for new in buff:
                ms2_out.write(new + '\n')
        for obj in self.write:
            print('......filtered MS2 file written', file=obj)
        ms2_in.close()
        ms2_out.close()
        return



class Threshold:
    """Hold low and high deltamass thresholds (Da).
    """
    def __init__(self):
        self.low = -5.0
        self.high = +5.0