"""
    Test the diurnal.structure module.

    Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    Affiliation: Département d'informatique, UQÀM
    File creation date: May 2023
    License: MIT
"""

import diurnal.structure as structure


def test_primary_structure_to_vector():
    """
    Validate the encoding of a text-based primary structure to a one-hot
    encoding determined by the IUPAC symbol list.
    """
    sequence = list("AAACCCUUU")
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
    encoding = structure.Primary.to_vector(sequence)
    assert (expected_encoding == encoding).all(), \
        "Primary structure is incorrectly encoded."


def test_primary_structure_to_vector_padding():
    """Ensure that primary structures are well padded."""
    sequence = list("AAACCC")
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
    encoding = structure.Primary.to_vector(sequence, total_size)
    assert (expected_encoding == encoding).all(), \
        "Primary structure is incorrectly padded."


def test_primary_structure_vector_to_bases():
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
    decoding_unpadded = structure.Primary.to_bases(encoding)
    assert decoding_unpadded == list("AAACCC"), \
        "Primary structure unpadded decoding produced an unexpected result."
    decoding_padded = structure.Primary.to_bases(encoding, False)
    assert decoding_padded == list("AAACCC..."), \
        "Primary structure padded decoding produced an unexpected result."


def test_primary_structure_bases_to_matrix():
    """Ensure that a sequence of bases can be encoded in a matrix of
    potential pairings."""
    bases = list("AAACCUUUU")
    i = structure.Schemes.IUPAC_ONEHOT_PAIRINGS["invalid"]
    u = structure.Schemes.IUPAC_ONEHOT_PAIRINGS["unpaired"]
    z = structure.Schemes.IUPAC_ONEHOT_PAIRINGS["-"]
    a = structure.Schemes.IUPAC_ONEHOT_PAIRINGS["UA"]
    b = structure.Schemes.IUPAC_ONEHOT_PAIRINGS["AU"]
    expected_matrix = [
        [u, i, i, i, i, b, b, b, b, z],
        [i, u, i, i, i, b, b, b, b, z],
        [i, i, u, i, i, i, b, b, b, z],
        [i, i, i, u, i, i, i, i, i, z],
        [i, i, i, i, u, i, i, i, i, z],
        [a, a, i, i, i, u, i, i, i, z],
        [a, a, a, i, i, i, u, i, i, z],
        [a, a, a, i, i, i, i, u, i, z],
        [a, a, a, i, i, i, i, i, u, z],
        [z, z, z, z, z, z, z, z, z, z]
    ]
    encoding = structure.Primary.to_matrix(bases, 10)
    n_errors = 0
    for row in range(10):
        for col in range(10):
            if list(expected_matrix[row][col]) != list(encoding[row][col]):
                n_errors += 1
    assert (expected_matrix == encoding).all(), \
        f"Incorrectly encoded primary structure matrix. N errors: {n_errors}"


def test_secondary_structure_to_vector():
    """
    Validate the encoding of a pairings-based primary structure to a
    one-hot ecoding.
    """
    pairings = [8, 7, 6, -1, -1, -1, 2, 1, 0]
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
    encoding = structure.Secondary.to_vector(pairings)
    assert (expected_encoding == encoding).all(), \
        "Secondary structure is incorrectly encoded."


def test_secondary_structure_to_vector_padding():
    """Ensure that primary structures are well padded."""
    pairings = [8, 7, 6, -1, -1, -1, 2, 1, 0]
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
    encoding = structure.Secondary.to_vector(pairings, total_size)
    assert (expected_encoding == encoding).all(), \
        "Secondary structure is incorrectly padded."


def test_secondary_structure_vector_to_brackets():
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
    decoding = structure.Secondary.to_bracket(encoding)
    assert decoding == list("(((..."), \
        "Primary structure unpadded decoding produced an unexpected result."
    decoding = structure.Secondary.to_bracket(encoding, False)
    assert decoding == list("(((...   "), \
        "Primary structure padded decoding produced an unexpected result."


def test_secondary_structure_to_matrix():
    """Ensure that the secondary structure can be well formatted into
    a vector of binary element."""
    pairings = [8, 7, 6, -1, -1, -1, 2, 1, 0]
    expected_matrix = [
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
    encoding = structure.Secondary.to_matrix(pairings)
    assert (expected_matrix == encoding).all(), \
        "Secondary structure is incorrectly formatted into a matrix."
    expected_matrix = [
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
    encoding = structure.Secondary.to_matrix(pairings, len(pairings) + 1)
    assert (expected_matrix == encoding).all(), \
        "Secondary structure is incorrectly padded in a matrix."
