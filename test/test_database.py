"""
    Test the diurnal.database module.

    Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    Affiliation: Département d'informatique, UQÀM
    File creation date: April 2023
    License: MIT
"""

import requests
import os
import numpy as np

import diurnal.database as database
import utils.fileio as fileio


def test_repository_availability():
    """Test the availability of the database repository."""
    response = requests.get(database.URL_PREFIX)
    assert response.status_code == 200, \
        f"The address {database.URL_PREFIX} is unaccessible."


def test_ct_file_format(tmp_rna_structure_files):
    """Ensure that CT files are correctly converted into a vector
    representation."""
    DIM = 512
    database.format(fileio.STRUCTURE_PATH, fileio.TMP_PATH, DIM)
    for filename in fileio.FILENAMES:
        path = f"{fileio.TMP_PATH}/{filename}"
        assert os.path.isfile(path), f"The file `{filename}` cannot be found."
        if (path.endswith("primary_structures.npy")
                or path.endswith("secondary_structures.npy")):
            values = np.load(path, allow_pickle=True)
            length = len(values[0])
            assert length == DIM, \
                f"Incorrect dimension for {path}; expected {DIM}, got {length}."
        if path.endswith("families.npy"):
            values = np.load(path, allow_pickle=True)
