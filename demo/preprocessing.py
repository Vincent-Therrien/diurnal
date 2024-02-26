"""
    Data pre-processing (installation and formatting) demonstration.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: June 2023
    - License: MIT
"""


from diurnal import align, database, structure


SIZE = 256
DATABASE = "archiveII"

SRC = f"./data/{DATABASE}"
DST = f"./data/families_{SIZE}/"


# Preprocessing
database.download("./data/", DATABASE)
names = database.format_filenames(SRC, DST + "names.txt", SIZE)

database.format_primary_structure(
    names, f"{DST}optimal_fold_alignments.npy",
    SIZE, align.optimal_fold_contact_matrix
)
database.format_primary_structure(
    names, f"{DST}fold_alignments_3.npy",
    SIZE, align.fold_contact_matrix
)
alignment_4 = lambda x, y : align.fold_contact_matrix(x, y, 4)
database.format_primary_structure(
    names, f"{DST}fold_alignments_4.npy",
    SIZE, alignment_4
)
database.format_primary_structure(
    names, f"{DST}potential_pairings.npy",
    SIZE, structure.Primary.to_matrix
)
database.format_primary_structure(
    names, f"{DST}masks.npy",
    SIZE, structure.Primary.to_mask
)
database.format_primary_structure(
    names, f"{DST}onehot.npy",
    SIZE, structure.Primary.to_onehot
)
database.format_secondary_structure(
    names, f"{DST}contact.npy", SIZE, structure.Secondary.to_matrix
)
database.format_secondary_structure(
    names, f"{DST}bracket.npy", SIZE, structure.Secondary.to_onehot
)
