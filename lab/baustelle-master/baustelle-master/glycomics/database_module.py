import glypy
from glypy import plot
from glypy.io import glycoct
from glypy.structure import (glycan, monosaccharide, substituent)
from glypy.algorithms import subtree_search, database
from glypy.composition.composition_transform import derivatize


glycomedb = database.dbopen("./glycomedb.db")


def construct_species_database(species_id):
    human_n_glycans = []
    for row in glycomedb.execute("SELECT * FROM GlycanRecord WHERE is_n_glycan=1;"):
        record = glycomedb.record_type.from_sql(row, glycomedb)  # Convert each raw row into GlycanRecord instance
        for taxon in record.taxa:
            if taxon.tax_id == species_id:
                human_n_glycans.append(record)
                break

    for record in human_n_glycans:
        record.structure.set_reducing_end(True)
        derivatize(record.structure, "methyl")

    def is_high_mannose(record):
        return int(record.monosaccharides['Hex'] > 4)

    @database.column_data("is_high_mannose", "BOOLEAN NOT NULL", is_high_mannose)
    class IsHighMannoseGlycanRecord(database.GlycanRecord):
        pass

    experiment_db = database.dbopen("experiment.db", record_type=IsHighMannoseGlycanRecord, flag='w')
    experiment_db.load_data(human_n_glycans, set_id=False)
    experiment_db.apply_indices()

    # mass searching
    for match in (experiment_db.ppm_match_tolerance_search(2063.0773, 1e-5)):
        plot.plot(match, label=True, scale=0.135)



