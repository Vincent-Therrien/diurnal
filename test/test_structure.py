"""
    Test the diurnal.structure module.
"""

import diurnal.structure as Structure


def test_encode_primary_structure():
    """
    Validate the encoding of a text-based primary structure to a one-hot
    encoding determined by the IUPAC symbol list.
    """
    structure = list("AAACCCUUU")
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
    encoding = Structure.Primary.to_vector(structure)
    assert expected_encoding == encoding, \
        "Primary structure is incorrectly encoded."


def test_primary_structure_padding():
    """Ensure that primary structures are well padded."""
    structure = list("AAACCC")
    total_size = 9
    expected_encoding = [
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ]
    encoding = Structure.Primary.to_vector(structure, total_size)
    assert expected_encoding == encoding, \
        "Primary structure is incorrectly padded."


def test_decode_primary_structure():
    """Ensure than a vectorized primary structure can be converted back
    to a sequence of characters."""
    encoding = [
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ]
    decoding_unpadded = Structure.Primary.to_bases(encoding)
    assert decoding_unpadded == list("AAACCC"), \
        "Primary structure unpadded decoding produced an unexpected result."
    decoding_padded = Structure.Primary.to_bases(encoding, False)
    assert decoding_padded == list("AAACCC..."), \
        "Primary structure padded decoding produced an unexpected result."


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
    encoding = Structure.Secondary.to_vector(structure)
    assert expected_encoding == encoding, \
        "Secondary structure is incorrectly encoded."


def test_secondary_structure_padding():
    """Ensure that primary structures are well padded."""
    structure = [8, 7, 6, -1, -1, -1, 2, 1, 0]
    total_size = 11
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
        [0, 0, 0],
        [0, 0, 0]
    ]
    encoding = Structure.Secondary.to_vector(structure, total_size)
    assert expected_encoding == encoding, \
        "Secondary structure is incorrectly padded."


def test_decode_secondary_structure():
    """Ensure than a vectorized secondary structure can be converted
    back to a sequence of characters."""
    encoding = [
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]
    decoding = Structure.Secondary.to_bracket(encoding)
    assert decoding == list("(((..."), \
        "Primary structure unpadded decoding produced an unexpected result."
    decoding = Structure.Secondary.to_bracket(encoding, False)
    assert decoding == list("(((...   "), \
        "Primary structure padded decoding produced an unexpected result."


def test_encode_structure():
    """
    Validate the encoding of primary and secondary structures in a
    single matrix.
    """
    primary = list("AAACCCUUU")
    secondary = [8, 7, 6, -1, -1, -1, 2, 1, 0]
    size = 9
    # Encode the structure.
    encoding = Structure.to_matrix(primary, secondary)
    # COmpare against the expected format.
    AU = Structure.Schemes.IUPAC_ONEHOT_PAIRINGS["AU"]
    UA = Structure.Schemes.IUPAC_ONEHOT_PAIRINGS["UA"]
    empty = Structure.Schemes.IUPAC_ONEHOT_PAIRINGS["-"]
    pattern = [[empty for _ in range(size)] for _ in range(size)]
    pattern[8][0] = AU
    pattern[7][1] = AU
    pattern[6][2] = AU
    pattern[0][8] = UA
    pattern[1][7] = UA
    pattern[2][6] = UA
    assert pattern == encoding, \
        "The matrix encoding the structure is incorrectly formatted."
