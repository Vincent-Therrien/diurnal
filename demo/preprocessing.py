"""
    Data pre-processing (installation and formatting) demonstration.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: June 2023
    - License: MIT
"""


from diurnal import align, database, structure, train
from diurnal.utils import log


SIZE = 128
DATABASE = "RNASTRalign"
DOWNLOAD_DIR = "../data/"
FRACTIONS = (0.8, 0.1, 0.1)

log.title(f"{DATABASE} molecule structures formatting.")

SRC = f"{DOWNLOAD_DIR}{DATABASE}"
DST = f"./data/{DATABASE}_formatted_{SIZE}/"

database.download(DOWNLOAD_DIR, DATABASE)
names = database.format_filenames(SRC, DST + "names.txt", SIZE)

train_names, validation_names, test_names = train.split(names, FRACTIONS)

def format(dst: str, names: list[str]):
    # Primary structures
    database.format_primary_structure(
        names, f"{dst}fold_alignments_optimal.npy",
        SIZE, align.optimal_fold_contact_matrix
    )
    database.format_primary_structure(
        names, f"{dst}fold_alignments_3.npy",
        SIZE, align.fold_contact_matrix
    )
    alignment_4 = lambda x, y : align.fold_contact_matrix(x, y, 4)
    database.format_primary_structure(
        names, f"{dst}fold_alignments_4.npy", SIZE, alignment_4
    )
    # database.format_primary_structure(
    #     names, f"{dst}potential_pairings_onehot.npy",
    #     SIZE, structure.Primary.to_matrix
    # )
    pp_scalar = lambda x, y: structure.Primary.to_matrix(
        x, y, structure.Schemes.IUPAC_PAIRINGS_SCALARS
    )
    database.format_primary_structure(
        names, f"{dst}potential_pairings_scalar.npy", SIZE, pp_scalar
    )
    database.format_primary_structure(
        names, f"{dst}masks.npy", SIZE, structure.Primary.to_mask
    )
    database.format_primary_structure(
        names, f"{dst}primary_onehot.npy",
        SIZE, structure.Primary.to_onehot
    )
    # Secondary structures.
    database.format_secondary_structure(
        names, f"{dst}bracket.npy", SIZE, structure.Secondary.to_onehot
    )
    database.format_secondary_structure(
        names, f"{dst}contact.npy", SIZE, structure.Secondary.to_matrix
    )
    database.format_secondary_structure(
        names, f"{dst}distance.npy", SIZE,
        structure.Secondary.to_distance_matrix
    )


format(f"{DST}validation/", validation_names)
format(f"{DST}test/", test_names)
format(f"{DST}train/", train_names)
