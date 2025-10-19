import os
import time
import gzip
import sys
import re
import copy
import glob
import fnmatch
from .ProteinGrouper import *
from .Protein import *

minimum_peptide_per_protein = 2  # min. number distinct peptides/protein/sample2
# set peptide requirements here
minimum_ntt_per_peptide = 2  # how many ntt for distinct peptides?

# seldom changed parameters:
# turns on or off calculating sum of MS2 fragment ion intensities
calc_ms2_int = False
max_num_peaks = 50
# turn on reporting of all peptide copies (True) or just best scoring spectrum (False)
full_peptide_list = True  # needs to be True for TMT data processing
# True for parsimony filtering, Note: subset proteins will not have any unique peptides
remove_subsets = True
# others:
minimum_unique_per_protein = 0  # min. number UNIQUE peptides/protein/sample (this should usually be zero!)
multiple_charge_states_ok = False  # are different charge states distinct?
modifications_ok = True  # are modified peptides distinct?
allow_prot_nterm_acytl = False  # +42 on M1 or Res2 considered as NTT=2

decoy_string = 'REV'
default_location = 'F:\PSR_Core_Analysis'

if not os.path.exists(default_location):
    default_location = os.getcwd()


def is_peptide_valid(peptide):
    """Tests if peptides meet sufficient criteria.
    Usage boolean = is_peptide_valid(peptide)
        where "peptide" is a peptide sequence (with bounding amino acids).
    Returns True if peptide criteria met and False otherwise.
    """
    # see if N-term acetylation candidate
    pep = peptide
    nterm = False
    if ']' in pep and allow_prot_nterm_acytl == True and (pep.startswith('M') or pep.startswith('-')):
        nterm = True
        pep = pep.replace(']', '')  # remove nt mod symbol

    # get ntt value
    """Should see if ntt needs to be replaced"""
    NTT = ntt(pep.split('_')[0])
    if nterm and pep.startswith('M'):
        NTT = NTT + 1

    # see if modifications are OK
    if (len(peptide_mods(pep.split('_')[0])) > 0) and not modifications_ok:
        print('failed mod test')
        return False

    # see if ntt criteria is satisfied (THIS IS ONLY CORRECT FOR TRYPSIN)
    if NTT >= minimum_ntt_per_peptide:
        return True
    else:
        return False


def is_protein_valid(prot, distinct_dict, unique_dict):
    """Test proteins for sufficient per sample evidence.
    Usage boolean = is_protein_valid(prot, distinct_dict, unique_dict)
        where "prot" is a PAW_Protein instance, "distinct_dict" is a
        list of full peptide sequences (with bounding amino acids) plus
        appended charge state, and "unique_dict" is ...

    Returns True if protein criteria met and False otherwise.
    """
    # remember that "distinct" keys have charge appended to sequence
    valid_peptides = {}
    unique_peptides = {}
    for peptide in distinct_dict.keys():
        if not is_peptide_valid(peptide):
            continue
        if not multiple_charge_states_ok:  # are two different charge states OK
            peptide = peptide[:-2]
        valid_peptides[peptide] = 1
    for hit in prot.tophit_list:
        unique_peptides[hit.Filename] = unique_dict.get(hit.Filename, 0)

    if ((sum(valid_peptides.values()) >= minimum_peptide_per_protein) and
            (sum(unique_peptides.values()) >= minimum_unique_per_protein)):
        prot.valid.append(True)
    else:
        prot.valid.append(False)
    return


class DistProtein:
    """Holds information about distinct protein groups, no fields per se.
    """

    def __init__(self):
        self.locus_list = []  # list of PAW_Loci classes for protein group (may be one or more)
        self.tophit_list = []  # list of PAW_TopHit classes for each MS/MS spectra
        self.supersets = []  # list of protein group supersets (if any)
        self.subsets = []  # list of proteins with peptide subsets contained in this set
        self.spc_list = []  # list of total spectral count and total MS2 intensity per sample
        self.valid = []  # list of valid protein flags per sample

    def peptides(self):
        """Returns a set of peptides mapped to protein.
        """
        peptides = {}

        # loop over DTA files mapped to protein
        for hit in self.tophit_list:

            # see if different charge states of same peptide are OK
            if multiple_charge_states_ok:
                peptide = hit.distinct
            else:
                peptide = hit.distinct[:-2]

            # peptide count is the dictionary value
            if peptides.get(peptide, False):
                peptides[peptide] = peptides[peptide] + 1
            else:
                peptides[peptide] = 1

        # dictionary keys are the peptide set elements
        return set(peptides.keys())

    def peptides_strict(self):
        """Returns a set of strictly defined peptides mapped to protein.
        """
        peptides = {}

        # loop over DTA files mapped to protein
        for hit in self.tophit_list:
            if not is_peptide_valid(hit.Sequence):
                continue

            # see if different charge states of same peptide are OK
            if multiple_charge_states_ok:
                peptide = hit.distinct.split('.')[1] + hit.distinct[-2:]
            else:
                peptide = hit.distinct.split('.')[1]

                # take care of I/L ambiguity
            peptide = re.sub(r'[IL]', r'j', peptide)

            # peptide count is the dictionary value
            if peptide in peptides:
                peptides[peptide] = peptides[peptide] + 1
            else:
                peptides[peptide] = 1

        # dictionary keys are the peptide set elements
        return set(peptides.keys())


class Locus:
    """Holds protein (locus) information.
    """

    def __init__(self):
        self.ID = 'Acession_Number'
        self.DTACount = '2'
        self.SpectrumCount = '2'
        self.Coverage = '100.0'
        self.SeqLength = '250'
        self.MW = '25000'
        self.pI = '5.7'
        self.Description = 'A long text string'

    def txt_parse_locus(self, line_items, col_map):
        """Loads locus data structure from PAW txt file line.
        """
        self.ID = line_items[col_map['Loci']]
        self.DTACount = 1
        self.SpectrumCount = 1
        self.Coverage = 0.0
        self.SeqLength = 0
        self.MW = 0.0
        self.pI = 0.0
        self.Description = ''
        return


class TopHit:
    """Holds top hit peptide information.
    """

    def __init__(self):
        self.Filename = 'my_dta.100.100.2'
        self.XCorr = '2.500'
        self.DeltCN = '0.200'
        self.NewDisc = '2.500'
        self.PrecursorMass = '1500.0'
        self.CalculatedMass = '1500.0'
        self.TotalIntensity = '5000.0'
        self.SpRank = '1'
        self.FragmentIonPercentage = '50.0'
        self.CopyCount = '1'
        self.Sequence = 'K.VTDDFGHR.A'
        self.Unique = 'FALSE'
        # calculated fields:
        self.DBUnique = 'FALSE'
        self.distinct = 'K.VTDDFGHR.A_2'
        self.ntt = 2
        self.sample = 'sample'
        self.beg = 0
        self.end = 0

    def txt_parse_tophit(self, line_items, dta_name, col_map):
        """Loads a tophit data structure from PAW txt line.
        """
        self.Filename = dta_name
        self.XCorr = line_items[col_map['Xcorr']]
        self.DeltCN = line_items[col_map['deltaCN']]
        self.NewDisc = line_items[col_map['NewDisc']]
        self.PrecursorMass = line_items[col_map['expM']]
        self.CalculatedMass = line_items[col_map['theoM']]
        self.TotalIntensity = '0.0'
        self.SpRank = line_items[col_map['SpRank']]
        self.FragmentIonPercentage = '0.0'
        self.CopyCount = 1
        self.Sequence = line_items[col_map['Sequence']]
        self.Unique = 'FALSE'
        self.distinct = self.Sequence + '_' + self.Filename[-1]
        self.ntt = line_items[col_map['ntt']]
        return


class FDR_Counter:
    """Data structure of counters for various target and decoy matches.
    """

    def __init__(self):
        self.FR = ['F', 'R']  # target or decoy
        self.NTT = ['0', '1', '2']  # number of tryptic termini
        self.Z = ['1', '2', '3', '4']  # peptide charge states
        """need to handle new terminla mod definitions"""
        self.mods = ['*', '#', '@', '^', '~', '$', '[', ']']
        self.unmod = [[[0 for z in self.Z] for ntt in self.NTT] for fr in self.FR]  # unmodified peptide counters
        self.mod = [[[0 for z in self.Z] for ntt in self.NTT] for fr in self.FR]  # modified peptide counters
        self.distinct = {}  # dictionary for distinct decoy sequences (conditioned on valid peptide criteria)
        self.tot_target = 0
        self.tot_valid_target = 0
        self.tot_decoy = 0
        self.tot_valid_decoy = 0

    def increment(self, seq, z, ntt, fr):
        """Increments counters for z, ntt, fr class.
        """
        # test for valid z, ntt, and fr values
        if z not in self.Z:
            print('...FDR_counter WARNING: %s out of charge state range' % (z,))
            return
        if ntt not in self.NTT:
            print('...FDR_counter WARNING: %s out of NTT range' % (ntt,))
            return
        if fr not in self.FR:
            print('...FDR_counter WARNING: %s is not "F" or "R"' % (fr,))
            return

        # check seq for mods and if it meets valid peptide criteria
        unmod = False
        mod = False
        num_mods = 0
        valid = is_peptide_valid(seq)
        temp = seq.split('.')  # remove bounding residues if present
        if len(temp) > 0:
            seq = temp[1]
        num_mods = len(peptide_mods(seq))
        if num_mods == 0:  # unmodified peptides
            unmod = True
        else:
            mod = True

        # increment the appropriate counters
        Z = int(z) - 1
        NTT = int(ntt)
        if fr == 'F':
            FR = 0
            self.tot_target += 1
            if valid:
                self.tot_valid_target += 1
        else:
            FR = 1
            self.tot_decoy += 1
            if valid:
                self.tot_valid_decoy += 1
                self.distinct[seq] = True
        if unmod:
            self.unmod[FR][NTT][Z] += 1
        else:
            self.mod[FR][NTT][Z] += 1
        return

    def report(self, write):
        """Prints out a report with the FDR information.
        """
        for obj in write:
            print('\n########### FDR REPORT ############', file=obj)
            print('\nunmodified peptides:', file=obj)
            for ntt in [2, 1, 0]:
                string = 'ntt=%s: ' % (ntt,)
                for z in [0, 1, 2, 3]:
                    try:
                        rate = 100 * float(self.unmod[1][ntt][z]) / float(self.unmod[0][ntt][z])
                    except ZeroDivisionError:
                        rate = 0.0
                    string += '%s(%s)%0.2f, ' % (self.unmod[0][ntt][z], self.unmod[1][ntt][z], rate)
                print(string, file=obj)
            print('\nmodified peptides:', file=obj)
            for ntt in [2, 1, 0]:
                string = 'ntt=%s: ' % (ntt,)
                for z in [0, 1, 2, 3]:
                    try:
                        rate = 100 * float(self.mod[1][ntt][z]) / float(self.mod[0][ntt][z])
                    except ZeroDivisionError:
                        rate = 0.0
                    string += '%s(%s)%0.2f, ' % (self.mod[0][ntt][z], self.mod[1][ntt][z], rate)
                print(string, file=obj)
            print('\ndistinct decoy sequences: %s' % (len(self.distinct),), file=obj)
            try:
                rate = 100 * float(self.tot_decoy) / float(self.tot_target)
            except ZeroDivisionError:
                rate = 0.0
            print('total matches: %s (%s) %0.2f' % (self.tot_target, self.tot_decoy, rate), file=obj)
            try:
                rate = 100 * float(self.tot_valid_decoy) / float(self.tot_valid_target)
            except ZeroDivisionError:
                rate = 0.0
            print('total valid matches: %s (%s) %0.2f' % (self.tot_valid_target, self.tot_valid_decoy, rate), file=obj)

    # end class FDR_Counter


"""Only need database name! Do not need a data container."""


def get_database(folder):
    """Get FASTA database name from a Comet parameters file or an SQT file header.
    """
    database = ''
    if os.path.exists(os.path.join(folder, 'comet.params')):  # get DB from params file
        for line in open(os.path.join(folder, 'comet.params')):
            line = line.strip()
            if line.startswith('database_name'):
                database = line.split('= ')[1]
    elif os.path.exists(os.path.join(folder, 'sequest.params')):  # get DB from params file
        for line in open(os.path.join(folder, 'sequest.params')):
            line = line.strip()
            if line.startswith('first_database_name'):
                database = line.split('= ')[1]

    # try getting database from SQT files
    else:
        hbuff = []
        try:
            # get header lines
            if glob.glob('*.sqt')[0]:
                for line in open(glob.glob('*.sqt')[0]):
                    if line.startswith('H'):
                        hbuff.append(line.strip())
                    if line.startswith('S'):
                        break
            elif glob.glob('*.sqt.gz')[0]:
                for line in gzip.open(glob.glob('*.sqt.gz')[0]):
                    if line.startswith('H'):
                        hbuff.append(line.strip())
                    if line.startswith('S'):
                        break

            # extract DB from header
            for line in hbuff:
                if line.startswith('H\tDatabase'):
                    database = line.split('\t')[2]

        except IndexError:
            print('...WARNING: no params file or SQT files')

        if not os.path.exists(database):  # browse to database if path not found
            ext_list = [('FASTA files', '*.fasta'), ('Zipped files', '*.gz'), ('All files', '*.*')]
            title = 'Please locate the FASTA file'
            database = PAW_lib.get_file(default_location, ext_list, title)

    return database


def load_results_from_txt_files(txt_file_list, txt_to_sample, write):
    """Loads results from PAW "txt" files.
    """
    # build a (partial) PAW_Protein list from the filtered txt files
    matches = {}
    proteins = []
    fdr = FDR_Counter()
    for txt_file in txt_file_list:

        # get the header line and make the column map
        col_map = {}
        txt_base_name = os.path.basename(txt_file)
        base_name = txt_base_name.replace('.txt.gz', '')  # do the longer possible extension first
        base_name = base_name.replace('.txt', '')  # in case the extension was not '.txt.gz'
        try:
            if txt_file.endswith('.gz'):
                contents = [x.strip() for x in gzip.open(txt_file).readlines()]
            else:
                contents = [x.strip() for x in open(txt_file, 'r').readlines()]
            for obj in write:
                print('...processing', os.path.split(txt_file)[1], file=obj)
        except:
            for obj in write:
                print('...WARNING: TXT and SQT file mis-match', file=obj)  # TXT list built from SQT list
            continue
        try:
            if contents[0].startswith('start\tend'):  # test for PAW header line
                for i, item in enumerate(contents[0].split('\t')):
                    col_map[item] = i
            else:
                print('...WARNING: non-PAW TXT file:', base_name)
                continue
        except IndexError:
            print('...WARNING: empty TXT file:', base_name)
            continue

        for line in contents[1:]:
            temp = line.split('\t')
            fdr.increment(temp[col_map['Sequence']], temp[col_map['Z']], temp[col_map['ntt']],
                          temp[col_map['ForR']])  # count target, decoy matches
            name_list = [base_name] + [str(int(x)) for x in temp[:3]]
            dta_name = '.'.join(name_list)
            prot = temp[col_map['Loci']]

            # if protein already in "matches" dictionary, add the DTA information
            if matches.get(prot, False):
                old_prot = matches[prot]
                new_tophit = PAW_TopHit()
                new_tophit.txt_parse_tophit(temp, dta_name, col_map)
                new_tophit.sample = txt_to_sample[txt_base_name]
                old_prot.tophit_list.append(copy.deepcopy(new_tophit))
                old_prot.locus_list[0].DTACount += 1
                old_prot.locus_list[0].SpectrumCount += 1
                matches[prot] = old_prot

            # if not, add a new protein match to "matches" dictionary
            else:
                new_prot = PAW_Protein()
                new_locus = PAW_Locus()
                new_locus.txt_parse_locus(temp, col_map)
                new_prot.locus_list.append(copy.deepcopy(new_locus))
                new_tophit = PAW_TopHit()
                new_tophit.txt_parse_tophit(temp, dta_name, col_map)
                new_tophit.sample = txt_to_sample[txt_base_name]
                new_prot.tophit_list.append(copy.deepcopy(new_tophit))
                matches[prot] = copy.deepcopy(new_prot)

    # print out the FDR information
    fdr.report(write)

    # need strict list of peptides for each protein: minimum ntt, allow modifications or not, I/L, etc.
    prot_pep_list = {}
    for prot in matches.keys():
        prot_pep_list[prot] = matches[prot].peptides_strict()

    #############################################################
    """this should be a function. we do this more than once"""
    # make list of proteins sorted by (strict) number of peptides
    prot_list = []
    for prot in matches.keys():
        prot_list.append((len(matches[prot].peptides_strict()), prot))
    prot_list.sort()
    ##############################################################

    # skip proteins with too few peptides to speed things up
    try:
        skip = [x[0] for x in prot_list].index(minimum_peptide_per_protein)
    except ValueError:
        skip = 0
    for obj in write:
        print('\n################ PARSIMONY REPORT ###############', file=obj)
        print('\n   there were', len(matches), 'protein matches', file=obj)
        print('   there were', skip, 'proteins with too few potential peptide(s)', file=obj)

    # clean up "matches" by removing skipped proteins
    for x, acc in prot_list[:skip]:
        del matches[acc]
    for obj in write:
        print('   new matches length is:', len(matches), file=obj)

    ##############################################################
    """this should be a function
    should have a protein sorting function with option for ascending or descending"""
    # find redundant proteins to group together
    redundant_group = 1
    redundants = {}
    for i in range(skip, len(prot_list)):
        for j in range(i + 1, len(prot_list)):
            s1 = prot_pep_list[prot_list[i][1]]
            s2 = prot_pep_list[prot_list[j][1]]
            if s1.issubset(s2) and len(s1) == len(s2):
                if redundants.get(prot_list[i][1], False):
                    redundants[prot_list[j][1]] = redundants[prot_list[i][1]]
                else:
                    redundants[prot_list[i][1]] = redundant_group
                    redundants[prot_list[j][1]] = redundant_group
                    redundant_group += 1

    # collaspe the redundant groups
    to_group = {}
    for (prot, group) in redundants.items():
        if to_group.get(group, False):
            temp = to_group[group]
            temp.append(prot)
            temp.sort()
            temp.insert(0, temp.pop())
            to_group[group] = temp
        else:
            to_group[group] = [prot]
    summary = list(to_group.items())
    summary.sort()
    for group_number, group_list in summary:
        group_list.sort()  # this determines protein group order
        keeper = group_list[0]
        for redundant in group_list[1:]:
            temp = matches[redundant].locus_list[0]
            matches[keeper].locus_list.append(copy.deepcopy(temp))

            # print out any redundant sets that are not exactly identical
            keep_set = matches[keeper].peptides()
            redun_set = matches[redundant].peptides()
            if keep_set != redun_set:
                keep_keys = sorted(list(keep_set))
                redun_keys = sorted(list(redun_set))
                for obj in write:
                    print('\n   redundant "mis-match": %s and %s' %
                          (matches[keeper].locus_list[0].ID, matches[redundant].locus_list[0].ID), file=obj)
                for i, first in enumerate(keep_keys):
                    try:
                        for obj in write:
                            if first != redun_keys[i]:
                                print('      %s %s %s' % (i, first, redun_keys[i]), file=obj)
                    except IndexError:
                        for obj in write:
                            print('      %s %s' % (i, first), file=obj)
            #
            del matches[redundant]
        for obj in write:
            print('   (%s) redundant group: %s' % (group_number, group_list), file=obj)
    for obj in write:
        print('\n   now matches is this long:', len(matches), file=obj)

    # build the list of protein objects sorted by accession
    proteins_list = []
    for i in range(skip, len(prot_list)):
        try:
            proteins_list.append((prot_list[i][1], matches[prot_list[i][1]]))
        except KeyError:  # redundant proteins have been deleted from "matches"
            pass
    proteins_list.sort()
    for prot in proteins_list:
        proteins.append(prot[1])

    ################################################
    ################################################
    """This should be a function"""
    # identify any protein/groups with peptide sets that are subsets
    # need to rebuild sets keyed to index in "proteins"
    prot_pep_list = {}
    for i in range(len(proteins)):
        prot_pep_list[i] = proteins[i].peptides_strict()

    # make list of proteins sorted by number of peptides
    prot_list = []
    keys = list(prot_pep_list.keys())
    keys.sort()
    for prot in keys:
        prot_list.append((len(prot_pep_list[prot]), prot))
    prot_list.sort()

    # test for subsets
    for i in range(len(prot_list)):
        for j in range(i + 1, len(prot_list)):
            s1 = prot_pep_list[prot_list[i][1]]
            s2 = prot_pep_list[prot_list[j][1]]
            if s1.issubset(s2):
                proteins[prot_list[i][1]].supersets.append(prot_list[j][1])
                proteins[prot_list[j][1]].subsets.append(prot_list[i][1])

    # replace list of indexes with list of accessions for each protein
    for prot in proteins:
        superset_temp = []
        for index in prot.supersets:
            temp = []
            for locus in proteins[index].locus_list:
                temp.append(locus.ID)
            if len(temp) != 0:
                superset_temp.append(temp)
        if len(superset_temp) != 0:
            prot.supersets = superset_temp
        #
        subset_temp = []
        for index in prot.subsets:
            temp = []
            for locus in proteins[index].locus_list:
                temp.append(locus.ID)
            if len(temp) != 0:
                subset_temp.append(temp)
        if len(subset_temp) != 0:
            prot.subsets = subset_temp

    # print information about subsets before removing them from "proteins"
    subset_list = []
    for obj in write:
        print(file=obj)
    for i, prot in enumerate(proteins):
        if len(prot.supersets) != 0:
            subset_list.append(i)
            subset = [x.ID for x in prot.locus_list]
            for obj in write:
                print('   %s- %s subset of %s' % (i, subset, prot.supersets), file=obj)
    for obj in write:
        print(file=obj)
    subset_list.reverse()  # have to delete from top to bottom
    if remove_subsets:
        for i in subset_list:
            del proteins[i]

    ##    # print information from the supersets data
    ##    for i, prot in enumerate(proteins):
    ##        if len(prot.subsets) != 0:
    ##            superset = [x.ID for x in prot.locus_list]
    ##            print('   %s- %s superset of %s' % (i, superset, prot.subsets))
    ##    print()
    #
    for obj in write:
        print('   length of proteins is: %s\n' % (len(proteins),), file=obj)
    return proteins


######################################################################

def reversed_hit(locus_list, decoy_string):
    """Checks if any proteins are reversed (decoy) entries.
    """
    rev = False
    for loci in locus_list:
        if decoy_string in loci.ID:
            rev = True
    return rev


def ntt(sequence):
    """Counts number of tryptic termini for peptides with bounding residues.
    """
    """This has already been computed in the TXT files. Need to use that instead! (otherwise need to know enzyme)"""
    ntt = 0
    parts = sequence.split('.')
    if len(parts) < 3:
        print('   ### WARNING: need bounding residues to determine ntt! ###')
    else:
        length, ntt = PAW_lib.amino_acid_count(sequence)  # this is going to be trypsin!
    return ntt


def get_database_proteins(fasta_full_name, proteins, write):
    """Get FASTA database entries for the identified proteins.
    Usage: (DBProteins, DB_map, fasta_file_name, DB_total) = get_database_proteins(database, proteins, write),
        where "DBProteins" is the list of FASTA objects, "DB_map" is a dictionary of accessions to indices,
        "fasta_file_name" is the basename for the database, "DB_total" is the toal number
        of proteins in the FASTA database, "database" is the path to the FASTA database,
        "proteins" is the list of identified proteins, and "write" is for console and log file use.
    """
    # if database path can't be found, then browse to database
    if not os.path.exists(fasta_full_name):
        fasta_name = os.path.basename(fasta_full_name)
        # bugger in an alternative database location:
        if os.path.exists(os.path.join('E:\Carr_plasma\databases', fasta_name)):
            fasta_full_name = os.path.join('E:\Carr_plasma\databases', fasta_name)
        else:
            print('...select the FASTA file')
            extensions = [('FASTA files', '*.fasta')]
            title = 'Select the %s database' % (os.path.basename(database),)
            fasta_full_name = PAW_lib.get_file(default, extensions, title)
            if not fasta_full_name: sys.exit()  # cancel button repsonse
    fasta_name = os.path.basename(fasta_full_name)

    # make a dictionary of identified loci so we don't have to load all proteins
    prot_id_map = {}
    for prot in proteins:
        for loci in prot.locus_list:
            prot_id_map[loci.ID] = True

    # open FASTA reader and create a Protein instance.
    for obj in write:
        print('...reading:', fasta_name, file=obj)
    f = PAW_lib.FastaReader(fasta_full_name)
    p = PAW_lib.Protein()

    # start looping over FASTA entries and keep the ones we need
    DBProteins = []
    DB_map = {}
    DB_len = 0
    DB_total = 0
    while f.readNextProtein(p):  # lets get everything copied into the data structures
        DB_total += 1
        if p.accession in prot_id_map:
            DBProteins.append(copy.deepcopy(p))
            DB_map[p.accession] = DB_len
            DB_len += 1
    for obj in write:
        print('...closing database (%s entries)...' % (DB_total,), file=obj)

    # compute coverage, length, and MW (needed when reading PAW TXT files)
    for prot in proteins:
        for loci in prot.locus_list:
            seq_dict = {}
            for hit in prot.tophit_list:
                seq_dict[hit.Sequence] = True
            try:
                loci.Coverage = '%0.01f' % (DBProteins[DB_map[loci.ID]].calcCoverage(seq_dict.keys())[0],)
                loci.SeqLength = DBProteins[DB_map[loci.ID]].seqlenProtein()
                loci.MW = '%0.0f' % (DBProteins[DB_map[loci.ID]].molwtProtein(),)
            except KeyError:  # phrog DB
                loci.Coverage = '%0.01f' % (DBProteins[DB_map[loci.ID.split()[0]]].calcCoverage(seq_dict.keys())[0],)
                loci.SeqLength = DBProteins[DB_map[loci.ID.split()[0]]].seqlenProtein()
                loci.MW = '%0.0f' % (DBProteins[DB_map[loci.ID.split()[0]]].molwtProtein(),)

    # loop over proteins and get description strings
    for prot in proteins:
        for loci in prot.locus_list:
            try:
                loci.Description = DBProteins[DB_map[loci.ID]].description
            except KeyError:  # phrog DB
                loci.Description = DBProteins[DB_map[loci.ID.split()[0]]].description
    #
    for obj in write:
        print('...done updating accessions and descriptions...', file=obj)
    return (DBProteins, DB_map, fasta_name, DB_total)


def compute_ms2_intensity(proteins, all_ms2, folder, write):
    """Computes total MS2 intensity from DTA files.
    Usage: compute_ms2_intensity(proteins, all_ms2, folder)
        where "proteins" is list of PAW_Protein objects, "all_ms2"
        is a dictionary of all protein matches of each MS2 file (by protein
        list index and tophit_list index), and "folder" is the
        folder containing the filtered MS2 format files.
    """
    # Assumes that MS2 files are in the same location as the TXT files,
    # open each MS2 file, build the filename for each scan, and check
    # it against the 'all_ms2' list.  If so, calculate DTA fragment ion sum
    # and replace the field(s) in the corresponding (protein, tophit) data structure.
    # max_num_peaks is a global set near top of module.
    os.chdir(folder)
    ms2_list = glob.glob('*.ms2')
    if not ms2_list:
        ms2_list = glob.glob('*.ms2.gz')
    if not ms2_list:
        print('...WARNING: MS2 file(s) were not found!')
        return
    for ms2 in ms2_list:
        if ms2.endswith('.gz'):
            ms2_file = gzip.open(ms2)
        else:
            ms2_file = open(ms2, 'r')
        buff = []  # temporary buffer to hold data for one MS2 scan
        for line in ms2_file:
            if line.startswith('H'):  # skip header lines
                continue
            line = line.rstrip()
            buff.append(line)  # save lines in scan buffer
            if line.startswith('S\t') and len(buff) > 1:  # process previous scan block
                file_name = []
                start, end = buff[0].split('\t')[1], buff[0].split('\t')[2]  # start, stop scan numbers
                start = str(int(start))
                end = str(int(end))
                for top in buff[:15]:  # parse enough top lines to get charge states
                    if top.startswith('Z'):  # get the charge state from the Z line(s)
                        z = top.split('\t')[1]
                        dta_name = ms2[:-3] + start + '.' + end + '.' + z  # build original DTA filename
                        if dta_name in all_ms2:
                            file_name.append(dta_name)
                for f in file_name:  # file_name list might be empty most of the time
                    frag_int = []
                    for i in range(len(buff)):  # get fragment ion intensities
                        if buff[i].startswith('S\t') or buff[i].startswith('Z\t') or buff[i].startswith('I\t'):
                            continue
                        frag_int.append(float(buff[i].split()[1]))
                    frag_int.sort()
                    frag_int.reverse()
                    int_sum = sum(frag_int[:max_num_peaks])  # sum up desired number of peaks
                    for i in range(len(all_ms2[f])):  # DTA may match to more than one protein
                        prot, hit = all_ms2[f][i]  # get protein, tophit indices for new TotalIntensity
                        proteins[prot].tophit_list[hit].TotalIntensity = int_sum
                # reset scan buffer
                buff = []
                buff.append(line)

        # process the last buffer
        if len(buff) > 1:
            file_name = []
            start, end = buff[0].split('\t')[1], buff[0].split('\t')[2]  # start, stop scan numbers
            for j in buff[:15]:  # parse enough top lines to get charge states
                if j.startswith('Z'):
                    z = j.split('\t')[1]  # get the charge state from the Z line(s)
                    dta_name = ms2[:-3] + start + '.' + end + '.' + z  # build original DTA filename
                    if dta_name in all_ms2:
                        file_name.append(dta_name)
            for f in file_name:
                frag_int = []
                for i in range(len(buff)):  # get fragment ion intensities
                    if buff[i].startswith('S\t') or buff[i].startswith('Z\t') or buff[i].startswith('I\t'):
                        continue
                    frag_int.append(float(buff[i].split()[1]))
                frag_int.sort()
                frag_int.reverse()
                int_sum = sum(frag_int[:max_num_peaks])  # sum up desired number of peaks
                for i in range(len(all_ms2[f])):  # DTA may match to more than one protein
                    prot, hit = all_ms2[f][i]  # get protein, tophit indices for new TotalIntensity
                    proteins[prot].tophit_list[hit].TotalIntensity = int_sum
    #
    for obj in write:
        print('...done calculating fragment ion intensity sums...', file=obj)
    #
    return


def peptide_mods(peptide):
    """Looks for modification symbols in peptides.  Returns list of mod symbols.
    THIS NEEDS TO BE CHANGED TO HANDLE NEW COMET MODS
    """

    # see if there are bounding residues
    temp = peptide.split('.')
    if len(temp) > 1:
        peptide = temp[1]

    # check for Comet modification symbols (also has older style n-term and c-term symbols)
    mod_list = []
    for char in peptide:
        if char in ['*', '#', '@', '^', '~', '$', '%', '!', '+', 'n', 'c', '[', ']']:
            mod_list.append(char)
    #
    return mod_list


def get_filter_tag(accession, decoy_string, count):
    """Checks if accession number corresponds to a decoy or contaminant.
    """
    tag = ''
    if 'CONT|' in accession or 'CONT_' in accession:
        tag = 'contaminant'
    if decoy_string in accession:
        tag = 'reversed'
    if count > 0:
        tag = 'redundant'
    return tag


def calc_split_count(prot, proteins, all_ms2_sample, s):
    """Calculates total SpC after spliting shared peptide counts.
    Also normalizes counts (adds 0.15 [or 100.0] to all zero values).
    """
    other_loci = {}
    split_count_total = 0.0

    # get correct values if using intensities
    if calc_ms2_int:
        self_unique = float(prot.spc_list[s][3])
    else:
        self_unique = float(prot.spc_list[s][1])

    for hit in prot.tophit_list:  # loop over all MS2 for protein "prot"
        if calc_ms2_int:
            value = float(hit.TotalIntensity)
        else:
            value = 1.0  # what we are splitting (a count or MS2 intensity)

        other_unique = 0.0
        if hit.Filename in all_ms2_sample:
            if len(all_ms2_sample[hit.Filename]) > 1:

                # split counts based on relative unique counts per sample
                for (p, x) in all_ms2_sample[hit.Filename]:
                    if calc_ms2_int:
                        other_unique += float(proteins[p].spc_list[s][3])
                    else:
                        other_unique += proteins[p].spc_list[s][1]
                try:
                    split_count_total += value * (self_unique / other_unique)
                except ZeroDivisionError:  # do equal splitting if zero unique total
                    split_count_total += value / float(len(all_ms2_sample[hit.Filename]))
            else:
                split_count_total += value

    if calc_ms2_int:
        try:
            if split_count_total < .0001:
                split_count_total = 0.0
            value = split_count_total
            if value < 0.0:
                value = 0.0
        except:
            print('   calc_split_count WARNING: calc_ms2_int try except failed')
            value = 0.0
    else:
        value = split_count_total

    return '%0.03f' % (value,)