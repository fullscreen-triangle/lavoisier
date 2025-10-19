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

MIN_PEP_LEN = 7


class PeptideInfo:
    """Data structure for some basic peptide information."""

    def __init__(self, sequence='', begin=0, end=0, mass=0, missed=0):
        self.seq = sequence
        self.beg = begin
        self.end = end
        self.mass = mass
        self.missed = missed
        return


class Peptide:
    """An object for Comet peptide strings.
    """

    def __init__(self, sequence, delim='.', enzyme='Trypsin'):
        self.full_seq = sequence  # original string
        self.enzyme = enzyme
        self.prefix = None  # preceeding residue string
        self.seq = None  # actual peptide sequence
        self.suffix = None  # following residue string
        self.base_seq = None  # actual peptide sequence without any mods
        self.net = None  # number of enzymatic termini (given enzyme)
        self.length = None  # number of amino acids in base sequence

        # compile a couple of regex
        self.new_mods = re.compile(r'\[[-+]?([0-9]+(\.[0-9]*)?|\.[0-9]+)\]')
        self.old_mods = re.compile(r'[*#@^~$%!+nc\[\]\{\}\(\)]')

        # load attributes
        self.split_peptide(delim)
        self.compute_net(enzyme)

    def split_peptide(self, delim):
        """This splits SEQUEST/Comet peptide strings into prefix, sequence, and suffix.
        Computes some things and sets some attributes; supports the new bracketed
        floating point modification format (Comet 2017 and newer).
        """
        # this removes new Comet modification notation (bracketed floating points)
        base_seq = self.new_mods.sub('', self.full_seq)

        # probably have bounding residues delimited by periods
        items = base_seq.split(delim)
        if len(items) == 3:
            self.prefix, middle, self.suffix = items
            self.seq = self.full_seq[len(self.prefix) + 1: -(len(self.suffix) + 1)]
        elif len(items) == 1:
            self.prefix, self.suffix = 'X', 'X'
            middle = items[0]
            self.seq = self.full_seq
        else:
            print('WARNING: malformed peptide string:', self.full_seq)

        # remove older style modification symbols: *#@^~$%!+[](){} and 'n', 'c'
        self.base_seq = self.old_mods.sub('', middle)
        self.length = len(self.base_seq)
        return

    def _N_side_cleavage(self, prefix, prefix_pattern, nterm, nterm_pattern, suffix, suffix_pattern):
        """Computes number of termini constent with protease cleavage for N-terminal side cutters."""
        self.net = 0
        if (prefix in prefix_pattern) or (nterm in nterm_pattern):
            self.net += 1
        if suffix in suffix_pattern:
            self.net += 1

    def _C_side_cleavage(self, prefix, prefix_pattern, cterm, cterm_pattern, suffix, suffix_pattern, noP=True):
        """Computes number of termini constent with protease cleavage for C-terminal side cutters."""
        self.net = 0
        ct_okay = False
        if prefix in prefix_pattern:
            self.net += 1
        if (cterm in cterm_pattern) or (suffix in suffix_pattern):
            self.net += 1
            ct_okay = True
        if noP and (suffix == 'P') and (self.net > 0) and ct_okay:  # trypsin strict
            self.net -= 1

    def compute_net(self, enzyme):
        """Figures out the number of peptide termini consistent with the enzyme cleavage.
        Written by Phil Wilmarth, OHSU, 2008, rewritten 2017.
        """
        # valid amino acid characters
        amino_acids = set(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'Y', 'Z', 'X'])

        # get the prefix amino acid residue
        i = len(self.prefix) - 1
        while self.prefix[i] not in amino_acids:
            i = i - 1
            if i < 0:
                break
        if i >= 0:
            prefix = self.prefix[i]
        else:
            prefix = 'X'
        # get suffix amino acid residue
        i = 0
        while self.suffix[i] not in amino_acids:
            i = i + 1
            if i >= len(self.suffix):
                break
        if i < len(self.suffix):
            suffix = self.suffix[i]
        else:
            suffix = 'X'

        cterm = self.base_seq[-1]  # last amino acid in sequence
        nterm = self.base_seq[0]  # first amino acid in sequence
        print(prefix, nterm, cterm, suffix)

        # determine number of enzymatic termini, net
        """need to support different enzymes and deal with proline.
        Seems Comet deals with premature stop codons as sequence breaks (* might be in prefix or suffix)."""
        if enzyme == 'Trypsin':  # cleaves at C-side of K, R (except if P)
            self._C_side_cleavage(prefix, 'KR-*', cterm, 'KR', suffix, '-*', noP=True)
        elif enzyme == 'Trypsin/P':  # cleaves at C-side of K, R
            self._C_side_cleavage(prefix, 'KR-*', cterm, 'KR', suffix, '-*', noP=False)
        elif enzyme == 'Lys_C':  # cleaves at C-side of K (except if P)
            self._C_side_cleavage(prefix, 'K-*', cterm, 'K', suffix, '-*', noP=True)
        elif enzyme == 'Lys_N':  # cleaves at N-side of K
            self._N_side_cleavage(prefix, '-*', nterm, 'K', suffix, 'K-*')
        elif enzyme == 'Arg_C':  # cleaves at C-side of R (except if P)
            self._C_side_cleavage(prefix, 'R-*', cterm, 'R', suffix, '-*', noP=True)
        elif enzyme == 'Asp_N':  # cleaves at N-side of D
            self._N_side_cleavage(prefix, '-*', nterm, 'D', suffix, 'D-*')
        elif enzyme == 'CNBr':  # cleaves at C-side of M
            self._C_side_cleavage(prefix, 'M-*', cterm, 'M', suffix, '-*', noP=False)
        elif enzyme == 'Glu_C':  # cleaves at C-side of D, E (except if P)
            self._C_side_cleavage(prefix, 'DE-*', cterm, 'DE', suffix, '-*', noP=True)
        elif enzyme == 'PepsinA':  # cleaves at C-side of F, L (except if P)
            self._C_side_cleavage(prefix, 'FL-*', cterm, 'FL', suffix, '-*', noP=True)
        elif enzyme == 'Chymotrypsin':  # cleaves at C-side of FWYL (except if P)
            self._C_side_cleavage(prefix, 'FWYL-*', cterm, 'FWYL', suffix, '-*', noP=True)
        elif enzyme == 'No_enzyme':
            self.net = 2
        else:
            print('WARNING: unknown enzyme specified', enzyme)
            self.net = 0

    def mask_base(self):
        """Masks I and L to j in base_seq."""
        return re.sub(r'[IL]', 'j', self.base_seq)


def find_peptide(peptide, proteins, mask=True, verbose=True):
    """Finds peptides in protein sequences.  Returns list of match tuples.
    This version requires additional attributes for protein objects.
    Usage: List = find_peptide(peptide, proteins, [limit=999]),
        where "peptide" is an uppercase peptide sequence string,
        "proteins" is a list of FASTAProtein objects,
        optional "limit" is the maximum number of protein matches (def=something large)
        and "List" is the returned list of match tuples.
            tuples: (accession, beginning res. #, ending res. #, full sequence)
    Written by Phil Wilmarth, OHSU, 9/4/08, 10/28/2015.
    """
    import types

    # make sure "proteins" is iterable
    protein_list = []
    if isinstance(proteins, list):
        protein_list = proteins
    else:
        print('FIND_PEPTIDE WARNING: "proteins" was not a list or Protein object.')

    matches = []
    for p in protein_list:
        matches += p.findPeptide(peptide, mask, pad_count=1)

    # if no matches, print warning (wrong database?)
    if verbose and len(matches) == 0:
        print('FIND_PEPTIDE WARNING: "%s" not found in protein sequences.' % (peptide,))
        if len(protein_list) <= 20:
            for p in protein_list:
                print('...', p.accession)

    # return the match list (empty list if no matches)
    return matches
    # end


class Protein:
    """Object to hold protein accession numbers, descriptions, and sequences.
    Methods:
        __init_:standard constructor, no parameters.
        readProtein: returns next protein from "fasta_reader"
        printProtein: prints sequence in FASTA format
        reverseProtein: reverses sequences and modifies accession/descriptions
        molwtProtein: computes average MW of sequence
        frequencyProtein: returns aa composition dictionary
        seqlenProtein: returns aa sequence length
        findPeptide: finds location of peptide in protein sequence
        coverage: calculates coverage and aa counts from peptide list
        enzymaticDigest: theroetical enzymatic digest of protein sequence
    Written by Phil Wilmarth, OHSU, 2009, 2016.
    Updated for new Comet mod formats -PW 10/27/2017
    Removed any parsing of accessions and descriptions methods and attributes -PW 20180711
    """

    def __init__(self):
        """Basic constructor, no parameters.
        """
        # bare bones __init__ function
        self.accession = 'blank'
        self.description = 'blank'
        self.sequence = ''
        self.sequence_padded = None
        self.sequence_masked = None
        self.pad_count = None
        self.length = 0
        self.peptides = []
        return

    def readProtein(self, fasta_reader):
        """Gets the next FASTA protein entry from FastaReader object.
        Usage: Boolean = object.readProtein(fasta_reader),
            where "object" is an instance of a Protein object and
            "fasta_reader" is an instance of a FastaReader object.
            Return value is "False" when EOF encountered.
        """
        status = fasta_reader.readNextProtein(self)
        return status

    def printProtein(self, file_obj=None, length=80):
        """Prints FASTA protein entry to file (stdout is default).
        Usage: object.printProtein([file_obj=None, length=80]),
            where "object" is an instance of a Protein object, and
            "file_obj" is a file object (a value of None will print
            to standard out stream.  Optional "length" is number of
            characters per line for the protein sequence.
        """
        if file_obj == None:
            file_obj = sys.stdout

        # print new accession and new descriptor on first line
        if self.description == '':
            print('>' + self.accession, file=file_obj)
        else:
            print('>' + self.accession, self.description, file=file_obj)

        # initialize some things
        char_count = 0
        char_line = ''

        # build up sequence line with "length" characters per line
        for char in self.sequence:
            if char_count < length:  # do not have "width" chars yet
                char_line += char
                char_count += 1
            else:  # line is "width" long so print and reset
                print(char_line, file=file_obj)
                char_line = char
                char_count = 1

        # print last sequence line (often less than "width" long) and return
        if len(char_line):
            print(char_line, file=file_obj)
        return

    def reverseProtein(self, decoy_string):
        """Reverses protein sequence and returns new Protein object.
        Usage: rev_prot = object.reverseProtein(decoy_string),
            where "object" is a Protein object, "decoy_string" is the
            unique identifier text to add to the beginning of the
            protein accesion number, and "rev_prot" is new Protein object.
        """
        # make sure decoy_string ends with an undescore
        if not decoy_string.endswith('_'):
            decoy_string = decoy_string + '_'

        # create a new Protein instance
        rev_prot = Protein()

        # prefix the decoy_string to desired parts of accession
        if self.accession.startswith('CONT_'):
            new_acc = decoy_string + self.accession.split('|')[0]
        else:
            new_acc = decoy_string + self.accession.replace('|', '&')  # best to remove "|"
        rev_prot.accession = new_acc

        # change the desciptions, too.
        rev_prot.description = 'REVERSED'

        # reversed the protein sequence and return new protein object
        rev_prot.sequence = self.sequence[::-1]
        return rev_prot

    def molwtProtein(self, show_errs=True):
        """Returns protein molecular weight as the sum of average aa masses.
        If "show_errs" flag set, invalid amino acid characters are reported.
        Does not add any modification delta masses (fixed or variable).
        """
        # start with water then add aa masses
        self.setMasses()
        bad_char = {}
        molwt = self.ave_masses['water']
        for aa in self.sequence:
            try:
                molwt += self.ave_masses[aa]
            except:  # keep track of bad characters
                bad_char[aa] = True

        bad_char = sorted(bad_char.keys())
        if len(bad_char) > 0 and show_errs:  # report bad chars if desired
            print('   WARNING: unknown symbol(s) (%s) in %s:\n%s' %
                  (''.join(bad_char), self.accession, self.sequence))
        return molwt

    def frequencyProtein(self, show_errs=True):
        """Returns aa frequency distrubution as a dictionary.
        If "show_errs" flag set, invalid amino acid characters are reported.
        """
        freq = {'X': 0, 'G': 0, 'A': 0, 'S': 0, 'P': 0, 'V': 0, 'T': 0,
                'C': 0, 'L': 0, 'I': 0, 'J': 0, 'N': 0, 'O': 0, 'B': 0,
                'D': 0, 'Q': 0, 'K': 0, 'Z': 0, 'E': 0, 'M': 0, 'H': 0,
                'F': 0, 'R': 0, 'Y': 0, 'W': 0, 'U': 0, '*': 0, '-': 0}

        # count the amino acids for all residues in sequence
        bad_char = {}
        for aa in self.sequence:
            try:
                freq[aa] += 1
            except:  # keep track of bad characters
                bad_char[aa] = True

        bad_char = sorted(bad_char.keys())
        if len(bad_char) > 0 and show_errs:  # report any bad chars, if desired
            print('   WARNING: unknown symbol(s) (%s) in %s:\n%s' %
                  (''.join(bad_char), self.accession, self.sequence))
        return freq

    def seqlenProtein(self):
        """Calculates protein sequence length.
        """
        self.length = len(self.sequence)
        return self.length

    def split_peptide(self, sequence):
        """Splits peptide assuming that there might be single preceeding and following residues with periods."""
        if re.match(r'[-A-Z]\..+\.[-A-Z]', sequence):
            return sequence[0], sequence[2:-2], sequence[-1]
        else:
            if min(sequence.count('['), sequence.count(']')) != sequence.count('.'):
                print('   WARNING: possible malformed peptide string:', sequence)
            return '', sequence, ''

    def peptide_decorations(self, sequence):
        """Separate modifications from amino acid residues so that mods can be put back later."""
        residues = []
        decorations = []
        char_count = 0
        decoration = ''
        for char in sequence:
            if re.match('[A-Z]', char):
                residues.append(char)
                decorations.append(decoration)
                decoration = ''
            else:
                decoration += char
        # might have C-terminal mod
        residues.append('')
        decorations.append(decoration)

        return residues, decorations

    def redecorate_peptide(self, peptide, decorations):
        """Redecorates a peptide sequence with mods."""
        residues = list(peptide + '')
        return ''.join(['' + x + y for (x, y) in zip(decorations, residues)])

    def base_peptide_sequence(self, sequence, mask=True):
        """Returns the peptide amino acid residues from SEQUEST peptide strings
        """
        # remove bounding residues (SEQUEST/Comet format: A.BCD.E)
        prefix, peptide, suffix = self.split_peptide(sequence)

        # remove the 2017 Comet style mod strings
        peptide = re.sub(r'\[[-+]?[0-9]*(.)?[0-9]*\]', '', peptide)
        # remove modification symbols: '*', '#', '@', '^', '~', '$', '%', '!', '+', 'n', 'c', '[', ']', "(', ')', '{', '}'
        peptide = re.sub(r'[*#@^~$%!+nc\[\]\{\}\(\)]', '', peptide)

        # mask I/L if needed:
        if mask:
            return re.sub(r'[IL]', 'j', peptide)
        else:
            return peptide

    def findPeptide(self, peptide, mask=True, pad_count=1):
        """Calculates location of all 'peptide' matches in 'self.sequence.'
        Returns a match tuple list.
        Match tuples: (accession, beginning res. #, ending res. #, peptide sequence with context)
        """
        matches = []

        # remove bounding residues (SEQUEST/Comet format: A.BCD.E)
        prefix, middle, suffix = self.split_peptide(peptide)

        # get a clean peptide sequence to lookup (retain mods)
        residues, decorations = self.peptide_decorations(middle)
        base_pep_masked = ''.join(residues)
        if mask:
            base_pep_masked = re.sub(r'[IL]', 'j', base_pep_masked)

        # fix the protein sequence for peptide lookups (pad and mask I/L). Save the results to save time
        if (not self.sequence_masked) or (pad_count != self.pad_count):
            self.sequence_padded = ('-' * pad_count) + self.sequence + ('-' * pad_count)  # add bounding symbols
            if mask:
                self.sequence_masked = re.sub(r'[IL]', 'j', self.sequence_padded)
            else:
                self.sequence_masked = self.sequence_padded
            self.pad_count = pad_count

        # find all matches of base_pep_masked to protein sequence (padded and masked)
        search_string = '(?=(%s))' % base_pep_masked  # need to do this to get overlapping matches (new python regex not yet ready: overlapped=True flag)
        for match in re.finditer(search_string, self.sequence_masked):
            start = match.span()[0]  # NOTE: look ahead matching does not give ending position of string (beg=end)
            end = start + len(base_pep_masked)
            start_prot, end_prot = start - self.pad_count + 1, end - self.pad_count

            # add bounding AAs, periods, and put back modification special chars
            pre = self.sequence_padded[start - self.pad_count:start]
            post = self.sequence_padded[end:end + self.pad_count]
            middle = self.redecorate_peptide(self.sequence_padded[start:end], decorations)
            full_seq = pre + '.' + middle + '.' + post
            """might want to create a match object instead of tuple."""
            matches.append((self.accession, start_prot, end_prot, full_seq))

        # return the match list (empty list if no matches)
        return matches

    def calcCoverage(self, peptide_list):
        """Calculates % coverage and aa frequency map of matched peptides.
        "peptide_list" is list of sequences with optional counts (as tuples).
        """
        freq_dict = {}
        try:  # see if peptide_list is a list of tuples or not
            for peptide, count in peptide_list:
                for (acc, beg, end, seq) in self.findPeptide(peptide):
                    for key in [str(i) for i in range(beg, end + 1)]:
                        if freq_dict.get(key, False):
                            freq_dict[key] = freq_dict[key] + count
                        else:
                            freq_dict[key] = count
        except ValueError:
            for peptide in peptide_list:
                for (acc, beg, end, seq) in self.findPeptide(peptide):
                    for key in [str(i) for i in range(beg, end + 1)]:
                        if freq_dict.get(key, False):
                            freq_dict[key] = freq_dict[key] + 1
                        else:
                            freq_dict[key] = 1

        coverage = 100.0 * float(len(freq_dict)) / float(len(self.sequence))
        coverage_map = []
        for i, aa in enumerate(self.sequence):
            coverage_map.append((str(i + 1), aa, freq_dict.get(str(i + 1), 0)))
        return (coverage, coverage_map)

    def enzymaticDigest(self, enzyme_regex=None, low=400.0, high=10000.0, length=6, missed=3, mass='mono'):
        """Performs a tryptic digest of a protein sequence. This does not
        do any modifications to residues except for reduction/alkylation of
        cys residues (C+57). Mass filters should be relaxed.
        Returns a list of digested peptides.
        enzyme_regex is a compiled re object for the enzyme cleavage
            (if enzyme_regex not defined, do tryptic digest by default)
        low, high - mass limits for peptides.
        length - minimum amino acid length
        missed - maximum number of missed cleavages.
        mass - 'ave' average or 'mono' monoisotopic masses.
        """

        """Regular expression digestion table:
        trypsin from: http://stackoverflow.com/questions/18364380/python-3-cut-peptide-regular-expression
        regex = re.compile(r".")                        # no enzyme
        regex = re.compile(r".(?:(?<![KR](?!P)).)*")    # trypsin strict
        regex = re.compile(r".(?:(?<![KR]).)*")         # trypsin with cleavage at P
        regex = re.compile(r".(?:(?<![K](?!P)).)*")     # Lys-C strict
        regex = re.compile(r".(?:(?<![K]).)*")          # Lys-C with cleavage at P
        regex = re.compile(r".(?:(?![K]).)*")           # Lys-N
        regex = re.compile(r".(?:(?<![R](?!P)).)*")     # Arg-C strict
        regex = re.compile(r".(?:(?![D]).)*")           # Asp-N
        regex = re.compile(r".(?:(?<![M]).)*")          # CnBr
        regex = re.compile(r".(?:(?<![DE](?!P)).)*")    # Glu-C
        regex = re.compile(r".(?:(?<![FL](?!P)).)*")    # PepsinA
        regex = re.compile(r".(?:(?<![FWYL](?!P)).)*")  # chymotrypsin
        """
        # skip if there is no sequence to digest
        if len(self.sequence) == 0:
            return []

        # tryptic digestion is the default
        if not enzyme_regex:
            enzyme_regex = re.compile(r".(?:(?<![KR](?!P)).)*")

        # set up masses, default is alkylated cysteine. No mechanism for other modifications yet.
        self.setMasses()
        if mass == 'ave':
            masses = copy.deepcopy(self.ave_masses)
            masses['C'] = 160.197
        elif mass == 'mono':
            masses = copy.deepcopy(self.mono_masses)
            masses['C'] = 160.03065
        else:
            print('...WARNING: masses must be "ave" or "mono"')

        # digest the sequence
        digest_matches = [x for x in enzyme_regex.finditer(self.sequence)]  # list of re match objects

        # get info from match objects into PeptideInfo object attributes
        digest = [PeptideInfo(mass=masses['water']) for x in digest_matches]
        for i, match in enumerate(digest_matches):
            digest[i].seq = match.group()
            digest[i].beg, digest[i].end = match.span()
            digest[i].beg += 1
            for aa in match.group():
                try:
                    digest[i].mass += masses[aa]
                except KeyError:
                    print('...WARNING: bad amino acid character!')
                    print('...bad character:', aa)
                    print('...in protein:', self.accession, self.description)

        # test peptides and missed cleavage peptides for mass ranges and min length
        valid_digest = []
        for i in range(len(digest)):

            # check if peptide is within the mass range and meets min length
            if (low <= digest[i].mass <= high) and (len(digest[i].seq) >= length):
                valid_digest.append(digest[i])

            # create and check missed cleavages
            for j in range(1, missed + 1):
                if (i + j) > len(digest) - 1:
                    continue
                temp = PeptideInfo(begin=100000)  # a peptide object for missed cleavages

                # calculate running sums for each number of missed cleavages
                for k in range(j + 1):
                    if (i + k) > len(digest) - 1:
                        continue
                    temp.seq += digest[i + k].seq
                    temp.beg = min(temp.beg, digest[i + k].beg)
                    temp.end = max(temp.end, digest[i + k].end)
                    temp.mass += (digest[i + k].mass - masses['water'])
                temp.mass += masses['water']
                temp.missed = k

                # check missed cleavage peptide for valid mass range and length
                if (low <= temp.mass <= high) and (len(temp.seq) >= length):
                    valid_digest.append(temp)

        return valid_digest

    def setMasses(self):
        """Set average and monoisotopic mass dictionaries."""
        self.ave_masses = {'X': 0.0000, 'G': 57.0513, 'A': 71.0779, 'S': 87.0773, 'P': 97.1152,
                           'V': 99.1311, 'T': 101.1039, 'C': 103.1429, 'L': 113.1576, 'I': 113.1576,
                           'J': 113.1576, 'N': 114.1026, 'O': 114.1472, 'B': 114.5950, 'D': 115.0874,
                           'Q': 128.1292, 'K': 128.1723, 'Z': 128.6216, 'E': 129.1140, 'M': 131.1961,
                           'H': 137.1393, 'F': 147.1739, 'R': 156.1857, 'Y': 163.1733, 'W': 186.2099,
                           'U': 150.0379, '*': 0.00000, '-': 0.00000, 'water': 18.02}
        self.mono_masses = {'X': 0.000000, 'G': 57.021464, 'A': 71.037114, 'S': 87.032028, 'P': 97.052764,
                            'V': 99.068414, 'T': 101.047679, 'C': 103.009185, 'L': 113.084064, 'I': 113.084064,
                            'J': 113.084064, 'N': 114.042927, 'O': 114.147200, 'B': 114.595000, 'D': 115.026943,
                            'Q': 128.058578, 'K': 128.094963, 'Z': 128.621600, 'E': 129.042593, 'M': 131.040485,
                            'H': 137.058912, 'F': 147.068414, 'R': 156.101111, 'Y': 163.063320, 'W': 186.079313,
                            'U': 150.953630, '*': 0.000000, '-': 0.000000, 'water': 18.01057}
        return


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


def amino_acid_count(sequence_string, enzyme='Tryp', return_base_pep=False):
    """Counts amino acids in peptides.  Returns (length, ntt) tuple.
    Usage: (length, ntt) = amino_acid_count(sequence_string),
        where "sequence_string" is a peptide sequence with bounding residues,
        "enzyme" is a string for the specific protease used,
        "length" is the returned number of amino acids, and
        "ntt" is the number of tryptic termini.
    Written by Phil Wilmarth, OHSU, 2008.
    THIS NEEDS TO BE REWRITTEN!!!
    """
    import re
    # supported enzymes are: 'Tryp', 'GluC', 'AspN', and 'LysC'
    #
    # This routine removes bounding amino acids, removes special characters
    # (mods), is now case sensitive, and computes number of enzymatic termini.
    # Assumes periods are used to separate bounding AAs from peptide.
    # Bounding AAs can be more than one character ("-" for N-term or C-term).
    # Modifications are Comet/SEQUEST format: special characters, "n", and "c";
    #   and are within the bounding periods (if present).
    #
    # Fixed bug in ntt caclulation, 4/30/07 -PW
    # Added support for different enzymes, 7/6/2010 -PW
    # Supports Comet PTM format ("n" and "c" for termini), 6/9/2015 -PW
    # Simplified ntt calculations

    # find the string between the bounding periods '.'
    parts = len(sequence_string.split('.'))
    if parts == 3:  # we have bounding residues
        start = sequence_string.index('.') + 1  # start is after first period
        temp = sequence_string[::-1]  # reverse string
        end = temp.index('.') + 1  # find first period in reversed string
        end = len(sequence_string) - end  # end is past the last period
    elif parts == 1:
        start = 0
        end = len(sequence_string)
    else:
        print('...amino_acid_count WARNING: number of "periods" was not 2 or 0', sequence_string)
        if return_base_pep:
            return (0, 0, "")
        else:
            return (0, 0)
    sequence = sequence_string[start:end]

    # remove any modification symbols:
    # '*', '#', '@', '^', '~', '$', '%', '!', '+', 'n', 'c', '[', ']' (Current Comet along with old style nt, ct)
    splitter = re.compile(r'[*#@^~$%!+nc\[\]]')
    base_seq = ''.join(splitter.split(sequence))
    prefix = sequence_string[start - 2:start - 1]
    if (prefix == "") or (start == 0):
        prefix = "X"  # no bounding residue info so unknown AA
    cterm = base_seq[-1]  # last amino acid in sequence
    nterm = base_seq[0]  # first amino acid in sequence
    suffix = sequence_string[end + 1:end + 2]
    if suffix == "":
        suffix = "X"  # no bounding residue info so unknown AA

    # determine number of enzymatic termini, ntt
    ntt = 0
    cterm_flag = False
    if enzyme.upper() == 'TRYP':  # cleaves at c-side of K, R
        if prefix in 'KR-*':
            ntt += 1
        if (cterm in 'KR') or (suffix in '-*'):
            ntt += 1
            cterm_flag = True
        if suffix == 'P' and cterm_flag and ntt > 0:  # trypsin strict???
            ntt -= 1
    elif enzyme.upper() == 'GLUC':  # cleaves at c-side of D, E
        if prefix in 'DE-*':
            ntt += 1
        if (cterm in 'DE') or (suffix in '-*'):
            ntt += 1
            cterm_flag = True
        if suffix == 'P' and cterm_flag and ntt > 0:  # trypsin strict???
            ntt -= 1
    elif enzyme.upper() == 'ASPN':  # cleaves at n-side of D
        if (prefix in '-*') or (nterm == 'D'):
            ntt += 1
        if suffix in 'D-*':
            ntt += 1
    elif enzyme.upper() == 'LYSC':
        if prefix in 'K-*':
            ntt += 1
        if (cterm in 'K') or (suffix in '-*'):
            ntt += 1
            cterm_flag = True
        if suffix == 'P' and cterm_flag and ntt > 0:  # trypsin strict???
            ntt -= 1
    else:
        print('   amino_acid_count WARNING: unknown enzyme specified', enzyme)

    if return_base_pep:
        return len(base_seq), ntt, base_seq
    else:
        return len(base_seq), ntt


def get_base_peptide_sequence(sequence, mask=True):
    """Returns the amino acid sequence from SEQUEST peptide sequences
    """
    # get rid of bounding residues, if any
    try:
        peptide = sequence.split('.')[1]
    except IndexError:
        peptide = sequence

    splitter = re.compile(r'[*#@^~$%!+nc\[\]]')
    base_pep = ''.join(splitter.split(peptide))
    if mask:
        base_pep = re.sub(r'[IL]', 'j', base_pep)
    return base_pep
