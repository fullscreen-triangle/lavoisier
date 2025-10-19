import csv

import ursgal


class RawFile:

    def __int__(self, srcpath: str,  filetype: Optional[str] = None):
        self.srcpath = srcpath
        if filetype is None:
            filetype = '.mzML'
        self.filetype = filetype

    def get_files(self):
        spec_files = []
        if filetype is None:
            raw_files = [os.path.join(self.srcpath, file) for file in os.listdir(self.srcpath) if
                         file.endswith(".mzML")]
            for mzML_file in [os.path.join(input_folder, file) for file in os.listdir(input_folder) if
                              file.endswith(".mzML")]:
                spec_files.append(mzML_file)
            return spec_files

        else:
            cun = ursgal.UController()
            input_file_list = []
            if input_path.lower().endswith(".raw"):
                input_file_list.append(self.srcpath)
            else:
                for raw in glob.glob(os.path.join("{0}".format(self.srcpath), "*.raw")):
                    input_file_list.append(raw)
                    for raw_file in input_file_list:
                        mzml_file = cun.convert(input_file=raw_file, engine="thermo_raw_file_parser_1_1_2", )
                        spec_files.append(mzml_file)
                    return spec_files


def generate_decoy_database(self):
    params = {
        "enzyme": "trypsin",
        "decoy_generation_mode": "reverse_protein",
    }
    fasta_database_list = [os.path.join(self.srcpath, file) for file in os.listdir(self.srcpath) if
                           file.endswith(".fasta")]
    uc = ursgal.UController(params=params)
    new_target_decoy_db_name = uc.execute_misc_engine(
        input_file=fasta_database_list,
        engine="generate_target_decoy_1_0_0",
        output_file_name="trypsin_target_decoy.fasta",
    )


class FolderWideSearch:

    def __init__(self, srcdir: str, outdir: str, massspec: str):
        self.srcdir = srcdir
        self.outdir = outdir
        self.massspec = massspec
        self.search_engines = ["omssa", "xtandem_piledriver", "msgfplus_v9979", "msamanda_1_0_0_5243", ]
        self.validation_engines = ["percolator_2_08", "qvality", ]
        self.groups = {"0": "", "1": "Oxidation", "2": "Deamidated", "3": "Methyl", "4": "Acetyl", "5": "Phospho", }

    def folder_wide_search(self):
        spec_files = get_files(self.srcdir)
        target_decoy_database = [os.path.join(self.srcdir, file) for file in os.listdir(self.srcdir) if
                                 file.endswith(".target_decoy.fasta")]
        all_mods = ["C,fix,any,Carbamidomethyl", "M,opt,any,Oxidation", "*,opt,Prot-N-term,Acetyl", ]
        params = {"database": target_decoy_database, "modifications": all_mods,
                  "csv_filter_rules": [["Is decoy", "equals", "false"], ["PEP", "lte", 0.01], ], }

        uc = ursgal.UController(profile=self.massspec, params=params)

        for validation_engine in validation_engines:
            result_files = []
            for spec_file in spec_files:
                validated_results = []
                for search_engine in search_engines:
                    unified_search_results = uc.search(
                        input_file=spec_file,
                        engine=search_engine,
                    )
                    validated_csv = uc.validate(
                        input_file=unified_search_results,
                        engine=validation_engine,
                    )
                    validated_results.append(validated_csv)

                validated_results_from_all_engines = uc.execute_misc_engine(
                    input_file=validated_results,
                    engine="merge_csvs_1_0_0",
                )
                filtered_validated_results = uc.execute_misc_engine(
                    input_file=validated_results_from_all_engines,
                    engine="filter_csv_1_0_0",
                )
                result_files.append(filtered_validated_results)

            results_all_files = uc.execute_misc_engine(
                input_file=result_files,
                engine="merge_csvs_1_0_0", )
            return results_all_files


class GroupSearch:

    def __init__(self, srcdir: str, outdir: str, massspec: str):
        self.srcdir = srcdir
        self.outdir = outdir
        self.massspec = massspec
        self.search_engines = ["omssa", "xtandem_piledriver", "msgfplus_v9979", "msamanda_1_0_0_5243", ]
        self.validation_engines = ["percolator_2_08", "qvality", ]
        self.groups = {"0": "", "1": "Oxidation", "2": "Deamidated", "3": "Methyl", "4": "Acetyl", "5": "Phospho", }

    def group_search(self):
        target_decoy_database = [os.path.join(self.srcdir, file) for file in os.listdir(self.srcdir) if
                                 file.endswith(".target_decoy.fasta")]
        params = {"database": target_decoy_database,
                  "csv_filter_rules": [["Is decoy", "equals", "false"], ["PEP", "lte", 0.01], ],
                  "modifications": ["C,fix,any,Carbamidomethyl", "M,opt,any,Oxidation", "N,opt,any,Deamidated",
                                    "Q,opt,any,Deamidated", "E,opt,any,Methyl", "K,opt,any,Methyl", "R,opt,any,Methyl",
                                    "*,opt,Prot-N-term,Acetyl", "S,opt,any,Phospho",
                                    "T,opt,any,Phospho", ]}
        mass_spectrometer = self.massspec
        spec_files = get_files(self.srcdir)
        uc = ursgal.UController(profile=mass_spectrometer, params=params)
        result_files = []
        for n, spec_file in enumerate(spec_files):
            validated_results = []
            for search_engine in search_engines:
                unified_search_results = uc.search(input_file=spec_file, engine=search_engine, )
                group_list = sorted(groups.keys())
                for p, group in enumerate(group_list):
                    if group == "0":
                        uc.params["csv_filter_rules"] = [
                            ["Modifications", "contains_not", "{0}".format(groups["1"])],
                            ["Modifications", "contains_not", "{0}".format(groups["2"])],
                            ["Modifications", "contains_not", "{0}".format(groups["3"])],
                            ["Modifications", "contains_not", "{0}".format(groups["4"])],
                            ["Modifications", "contains_not", "{0}".format(groups["5"])],
                        ]
                    else:
                        uc.params["csv_filter_rules"] = [
                            ["Modifications", "contains", "{0}".format(groups[group])]
                        ]
                        for other_group in group_list:
                            if other_group == "0" or other_group == group:
                                continue
                            uc.params["csv_filter_rules"].append(
                                [
                                    "Modifications",
                                    "contains_not",
                                    "{0}".format(groups[other_group]),
                                ],
                            )
                    uc.params["prefix"] = "grouped-{0}".format(group)
                    filtered_results = uc.execute_misc_engine(
                        input_file=unified_search_results, engine="filter_csv"
                    )
                    uc.params["prefix"] = ""
                    validated_search_results = uc.validate(
                        input_file=filtered_results,
                        engine=validation_engine,
                    )
                    validated_results.append(validated_search_results)

            uc.params["prefix"] = "file{0}".format(n)
            validated_results_from_all_engines = uc.execute_misc_engine(
                input_file=sorted(validated_results),
                engine="merge_csvs",
            )
            uc.params["prefix"] = ""
            uc.params["csv_filter_rules"] = [
                ["Is decoy", "equals", "false"],
                ["PEP", "lte", 0.01],
            ]
            filtered_validated_results = uc.execute_misc_engine(
                input_file=validated_results_from_all_engines, engine="filter_csv"
            )
            result_files.append(filtered_validated_results)

        results_all_files = uc.execute_misc_engine(input_file=sorted(result_files), engine="merge_csvs", )
        return results_all_files

    def analyze(collector):
        mod_list = ["Oxidation", "Deamidated", "Methyl", "Acetyl", "Phospho"]
        fieldnames = (
                ["approach", "count_type", "validation_engine", "unmodified", "multimodified"]
                + mod_list
                + ["total"]
        )

        csv_writer = csv.DictWriter(open("grouped_results.csv", "w"), fieldnames)
        csv_writer.writeheader()
        uc = ursgal.UController()
        uc.params["validation_score_field"] = "PEP"
        uc.params["bigger_scores_better"] = False

        for validation_engine, result_file in collector.items():
            counter_dict = {"psm": ddict(set), "pep": ddict(set)}
            grouped_psms = uc._group_psms(
                result_file, validation_score_field="PEP", bigger_scores_better=False
            )
            for spec_title, grouped_psm_list in grouped_psms.items():
                best_score, best_line_dict = grouped_psm_list[0]
                if len(grouped_psm_list) > 1:
                    second_best_score, second_best_line_dict = grouped_psm_list[1]
                    best_peptide_and_mod = (
                            best_line_dict["Sequence"] + best_line_dict["Modifications"]
                    )
                    second_best_peptide_and_mod = (
                            second_best_line_dict["Sequence"]
                            + second_best_line_dict["Modifications"]
                    )

                    if best_peptide_and_mod == second_best_peptide_and_mod:
                        line_dict = best_line_dict
                    elif best_line_dict["Sequence"] == second_best_line_dict["Sequence"]:
                        if best_score == second_best_score:
                            line_dict = best_line_dict
                        else:
                            if (-1 * math.log10(best_score)) - (
                                    -1 * math.log10(second_best_score)
                            ) >= 2:
                                line_dict = best_line_dict
                            else:
                                continue
                    else:
                        if (-1 * math.log10(best_score)) - (
                                -1 * math.log10(second_best_score)
                        ) >= 2:
                            line_dict = best_line_dict
                        else:
                            continue
                else:
                    line_dict = best_line_dict

                count = 0
                for mod in mod_list:
                    if mod in line_dict["Modifications"]:
                        count += 1
                key_2_add = ""
                if count == 0:
                    key_2_add = "unmodified"
                elif count >= 2:
                    key_2_add = "multimodified"
                elif count == 1:
                    for mod in mod_list:
                        if mod in line_dict["Modifications"]:
                            key_2_add = mod
                            break
                # for peptide identification comparison
                counter_dict["pep"][key_2_add].add(
                    line_dict["Sequence"] + line_dict["Modifications"]
                )
                # for PSM comparison
                counter_dict["psm"][key_2_add].add(
                    line_dict["Spectrum Title"]
                    + line_dict["Sequence"]
                    + line_dict["Modifications"]
                )
            for counter_key, count_dict in counter_dict.items():
                dict_2_write = {
                    "approach": "grouped",
                    "count_type": counter_key,
                    "validation_engine": validation_engine,
                }
                total_number = 0
                for key, obj_set in count_dict.items():
                    dict_2_write[key] = len(obj_set)
                    total_number += len(obj_set)
                dict_2_write["total"] = total_number
                csv_writer.writerow(dict_2_write)
        return


def group_filter(input_folder):
    collector = {}
    for validation_engine in validation_engines:
        results_all_files = group_search(validation_engine, input_folder, mass_spec='LTQ XL low res')
        collector[validation_engine] = results_all_files
    analyze(collector)


class MSPeakProcessing:

    def __init__(self, srcdir: str):
        self.srcdir = srcdir

    def peakfilter(self):
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

        # Step 1 : remove rows

        clean_df = remove_rows_matching(data)
        logger.info()
        stepNum = _update_status(data, stepDst, verbose, stepNum)

        # Step 2 remove reverse

        # Step 3
