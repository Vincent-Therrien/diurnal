"""
    Test the diurnal.transform module.
"""

from diurnal.transform import PrimaryStructure, SecondaryStructure

def test_encode_primary_structure():
    """
    Validate the encoding of a text-based primary structure to a one-hot
    ecoding determined by the IUPAC symbol list.
    """
    structure = "AAACCCUUU"
    expected_encoding = [
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 1]
    ]
    encoding = PrimaryStructure.iupac_to_onehot(structure, 0)
    assert expected_encoding == encoding

def test_encode_secondary_structure():
    """
    Validate the encoding of a pairings-based primary structure to a
    one-hot ecoding.
    """
    structure = [8, 7, 6, -1, -1, -1, 2, 1, 0]
    expected_encoding = [
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
    ]
    encoding = SecondaryStructure.pairings_to_onehot(structure)
    assert expected_encoding == encoding
