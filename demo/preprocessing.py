"""
    Data pre-processing (installation and formatting) demonstration.
"""

from diurnal import database
import diurnal.structure as structure
import diurnal.family as family

database.download("./data/", "archiveII")

database.format(
    "./data/archiveII",         # Directory of the raw data to format.
    "./data/formatted_matrix",  # Formatted data output directory.
    256,  # Normalized size.
    structure.Primary.to_matrix,    # RNA encoding scheme.
    structure.Secondary.to_onehot,  # RNA encoding scheme.
    family.to_onehot
)
