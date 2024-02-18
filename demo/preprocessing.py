"""
    Data pre-processing (installation and formatting) demonstration.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: June 2023
    - License: MIT
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
