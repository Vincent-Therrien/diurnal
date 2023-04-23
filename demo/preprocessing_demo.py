"""
    Data pre-processing (installation and formatting) demonstration.
"""

import diurnal.database as db
from diurnal.transform import PrimaryStructure as s1
from diurnal.transform import SecondaryStructure as s2
from diurnal.transform import Family as f

# Download the dataset.
db.download("./data/", "archiveII")


# Format the dataset into numpy `.npy` files.
db.format(
    "./data/archiveII", # Directory of the raw data to format.
    "./data/formatted", # Formatted data output directory.
    512, # Normalized size.
    s1.iupac_to_onehot, # RNA encoding scheme.
    s2.pairings_to_onehot, # RNA encoding scheme.
    f.onehot
)
