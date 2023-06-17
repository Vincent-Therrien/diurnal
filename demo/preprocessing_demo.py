"""
    Data pre-processing (installation and formatting) demonstration.
"""

import diurnal.database as db
import diurnal.structure as structure
import diurnal.family as family

# Download the dataset.
db.download("./data/", "archiveII")

# Format the dataset into numpy `.npy` files.
db.format(
    "./data/archiveII", # Directory of the raw data to format.
    "./data/formatted", # Formatted data output directory.
    512, # Normalized size.
    structure.Primary.to_vector, # RNA encoding scheme.
    structure.Secondary.to_vector, # RNA encoding scheme.
    family.to_vector
)
