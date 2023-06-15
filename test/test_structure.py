"""
    Test the diurnal.structure module.

    Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    Affiliation: Département d'informatique, UQÀM
    File creation date: May 2023
    License: MIT
"""

import diurnal.structure as Structure


def test_primary_structure_to_vector():
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


def test_primary_structure_to_vector_padding():
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
    decoding_unpadded = Structure.Primary.to_bases(encoding)
    assert decoding_unpadded == list("AAACCC"), \
        "Primary structure unpadded decoding produced an unexpected result."
    decoding_padded = Structure.Primary.to_bases(encoding, False)
    assert decoding_padded == list("AAACCC..."), \
        "Primary structure padded decoding produced an unexpected result."


def test_secondary_structure_to_vector():
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


def test_secondary_structure_to_vector_padding():
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
    decoding = Structure.Secondary.to_bracket(encoding)
    assert decoding == list("(((..."), \
        "Primary structure unpadded decoding produced an unexpected result."
    decoding = Structure.Secondary.to_bracket(encoding, False)
    assert decoding == list("(((...   "), \
        "Primary structure padded decoding produced an unexpected result."


def test_secondary_structure_to_matrix():
    """Ensure that the secondary structure can be well formatted into
    a vector of binary element."""
    structure = [8, 7, 6, -1, -1, -1, 2, 1, 0]
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
    encoding = Structure.Secondary.to_matrix(structure)
    assert expected_matrix == encoding, \
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
    encoding = Structure.Secondary.to_matrix(structure, len(structure) + 1)
    assert expected_matrix == encoding, \
        "Secondary structure is incorrectly padded in a matrix."
