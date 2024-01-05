"""
    Test the diurnal.database module.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: April 2023
    - License: MIT
"""

import requests
import os
import numpy as np

import diurnal.database as database
import diurnal.structure as structure
import utils.fileio as fileio


def test_repository_availability():
    """Test the availability of the database repository."""
    file_URL = database.URL_PREFIX + database._DB_IDs["archiveII.rst"]
    response = requests.get(file_URL)
    assert response.status_code == 200, \
        f"The address {database.URL_PREFIX} is unaccessible."


def test_ct_file_format():
    """Ensure that CT files are correctly converted into a vector
    representation."""
    DIM = 512
    database.format(fileio.STRUCTURE_PATH, fileio.TMP_PATH, DIM)
    for filename in fileio.FILENAMES:
        path = f"{fileio.TMP_PATH}/{filename}"
        assert os.path.isfile(path), f"The file `{filename}` cannot be found."
        if (path.endswith("primary_structures.npy")
                or path.endswith("secondary_structures.npy")):
            values = np.load(path)
            assert values.shape[0] == 3, "Unexpected number of samples."
            lengths = [len(v) for v in values]
            assert len(set(lengths)) == 1 and lengths[0] == DIM, \
                (f"Incorrect dimension for `{path}`; "
                 + f"expected {DIM}, got {lengths[0]}.")
            n_onehot = [len(v[0]) for v in values]
            assert len(set(n_onehot)) == 1 and n_onehot[0] in (3, 4), \
                f"Incorrect dimension for one-hot encoding."
        # Test the primary encoding of `5s_Acanthamoeba-castellanii-1.ct`
        if path.endswith("primary_structures.npy"):
            values = np.load(path)
            acanthamoeba = "GGAUACGGCCAUACUGCGC"
            sequence = "".join(structure.Primary.to_sequence(values[0]))
            assert sequence.startswith(acanthamoeba), "Incorrect sequence."
        # Test the secondary encoding of `5s_Acanthamoeba-castellanii-1.ct`
        if path.endswith("secondary_structures.npy"):
            values = np.load(path)
            acanthamoeba = "(((((((((....((((((((....."
            sequence = "".join(structure.Secondary.to_bracket(values[0]))
            assert sequence.startswith(acanthamoeba), "Incorrect sequence."
