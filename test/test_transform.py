"""
    Test the diurnal.transform module.
"""

from diurnal.transform import PrimaryStructure, SecondaryStructure, Structure

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
    encoding = SecondaryStructure.pairings_to_onehot(structure, 0)
    assert expected_encoding == encoding

def test_encode_structure():
    """
    Validate the encoding of primary and secondary structures in a
    single matrix.
    """
    primary = "AAACCCUUU"
    secondary = [8, 7, 6, -1, -1, -1, 2, 1, 0]
    size = 9
    # Encode the structure.
    encoding = Structure.structure_to_2D_matrix(primary, secondary, size)
    # COmpare against the expected format.
    AU = Structure.IUPAC_ONEHOT_PAIRINGS["AU"]
    UA = Structure.IUPAC_ONEHOT_PAIRINGS["UA"]
    empty = Structure.IUPAC_ONEHOT_PAIRINGS["-"]
    pattern = [[empty for _ in range(size)] for _ in range(size)]
    pattern[8][0] = AU
    pattern[7][1] = AU
    pattern[6][2] = AU
    pattern[0][8] = UA
    pattern[1][7] = UA
    pattern[2][6] = UA
    assert pattern == encoding
