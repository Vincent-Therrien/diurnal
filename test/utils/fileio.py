"""
    Test suite utility module for file input/output operations.

    Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    Affiliation: Département d'informatique, UQÀM
    File creation date: June 2023
    License: MIT
"""

import os


MODULE_PATH = os.path.dirname(os.path.realpath(__file__)) + "/.."
STRUCTURE_PATH = MODULE_PATH + "/data/ct"
TMP_PATH = MODULE_PATH + "/tmp"
FILENAMES = ("families.npy", "info.rst", "names.txt",
    "primary_structures.npy", "secondary_structures.npy")
