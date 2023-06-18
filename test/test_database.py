"""
    Test the diurnal.database module.

    Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    Affiliation: Département d'informatique, UQÀM
    File creation date: April 2023
    License: MIT
"""

import pytest
import requests
import os
import numpy as np
import shutil

import diurnal.database as database


STRUCTURE_PATH = os.path.dirname(os.path.realpath(__file__)) + "/data/ct"
TMP_PATH = os.path.dirname(os.path.realpath(__file__)) + "/tmp"
FILENAMES = ["families.npy", "info.rst", "names.txt",
    "primary_structures.npy", "secondary_structures.npy"]


@pytest.fixture()
def tmp_rna_structure_files(request):
    if os.path.isdir(TMP_PATH):
        shutil.rmtree(TMP_PATH)
    os.makedirs(TMP_PATH)
    def teardown():
        shutil.rmtree(TMP_PATH)
    request.addfinalizer(teardown)


def test_repository_availability():
    """Test the availability of the database repository."""
    response = requests.get(database.URL_PREFIX)
    assert response.status_code == 200, \
        f"The address {database.URL_PREFIX} is unaccessible."


def test_ct_file_format(tmp_rna_structure_files):
    """Ensure that CT files are correctly converted into a vector
    representation."""
    database.format(STRUCTURE_PATH, TMP_PATH, 512)
    for filename in FILENAMES:
        path = f"{TMP_PATH}/{filename}"
        assert os.path.isfile(path), f"The file `{filename}` cannot be found."
        if path.endswith(".npy"):
            np.load(path, allow_pickle=True)
