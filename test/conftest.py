"""
    Test suite configuration module.

    Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    Affiliation: Département d'informatique, UQÀM
    File creation date: June 2023
    License: MIT
"""


import pytest
import sys
import os
import shutil


sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))


import utils


@pytest.fixture()
def tmp_rna_structure_files(request):
    if os.path.isdir(utils.fileio.TMP_PATH):
        shutil.rmtree(utils.fileio.TMP_PATH)
    os.makedirs(utils.fileio.TMP_PATH)
    def teardown():
        shutil.rmtree(utils.fileio.TMP_PATH)
    request.addfinalizer(teardown)
