"""
    Test the diurnal.structure module.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: May 2023
    - License: MIT
"""

import pytest
import numpy as np

import diurnal.structure as structure


def test_primary_structure_to_onehot():
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
    encoding = structure.Primary.to_onehot(sequence)
    assert (expected_encoding == encoding).all(), \
        "Primary structure is incorrectly encoded."


def test_primary_structure_to_onehot_padding():
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
    encoding = structure.Primary.to_onehot(sequence, total_size)
    assert (expected_encoding == encoding).all(), \
        "Primary structure is incorrectly padded."


def test_primary_structure_vector_to_sequence():
    """Ensure than a vectorized primary structure can be converted back
    to a sequence of characters."""
    encoding = np.array([
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])
    decoding_unpadded = structure.Primary.to_sequence(encoding)
    assert decoding_unpadded == list("AAACCC"), \
        "Primary structure unpadded decoding produced an unexpected result."
    decoding_padded = structure.Primary.to_sequence(encoding, False)
    assert decoding_padded == list("AAACCC..."), \
        "Primary structure padded decoding produced an unexpected result."


def test_primary_structure_bases_to_matrix():
    """Ensure that a sequence of bases can be encoded in a matrix of
    potential pairings."""
    bases = list("AAACCUUUU")
    i = structure.Schemes.IUPAC_ONEHOT_PAIRINGS_VECTOR["invalid"]
    u = structure.Schemes.IUPAC_ONEHOT_PAIRINGS_VECTOR["unpaired"]
    z = structure.Schemes.IUPAC_ONEHOT_PAIRINGS_VECTOR["-"]
    a = structure.Schemes.IUPAC_ONEHOT_PAIRINGS_VECTOR["UA"]
    b = structure.Schemes.IUPAC_ONEHOT_PAIRINGS_VECTOR["AU"]
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


def test_secondary_structure_to_onehot():
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
    encoding = structure.Secondary.to_onehot(pairings)
    assert (expected_encoding == encoding).all(), \
        "Secondary structure is incorrectly encoded."


def test_secondary_structure_to_onehot_padding():
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
    encoding = structure.Secondary.to_onehot(pairings, total_size)
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
    assert decoding == list("(((...---"), \
        "Primary structure decoding produced an unexpected result."


@pytest.mark.parametrize(
    "secondary_structure, shadow, size",
    [
        ("(((...)))", [1, 1, 1, 0, 0, 0, 1, 1, 1], 0),
        ("((..))", [1, 1, 0, 0, 1, 1, -1, -1], 8)
    ]
)
def test_secondary_structure_shadow(secondary_structure, shadow, size):
    """Test shadow encoding."""
    encoding = structure.Secondary.to_shadow(secondary_structure, size)
    assert encoding.tolist() == shadow, \
        f"Incorrectly encoded shadows: Expected {shadow}, got {encoding}."


@pytest.mark.parametrize(
    "brackets, pairings",
    [
        ("(((...)))", [8, 7, 6, -1, -1, -1, 2, 1, 0]),
        ("(((((((((....((((((((.....((((((............))))..))....))))))"
         + ".)).(((((......(((((.(((....)))))))).....))))).))))))))).",
         [117, 116, 115, 114, 113, 112, 111, 110, 109, -1, -1, -1, -1, 64,
          63, 61, 60, 59, 58, 57, 56, -1, -1, -1, -1, -1, 51, 50, 47, 46,
          45, 44, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 31, 30,
          29, 28, -1, -1, 27, 26, -1, -1, -1, -1, 20, 19, 18, 17, 16, 15,
          -1, 14, 13, -1, 107, 106, 105, 104, 103, -1, -1, -1, -1, -1, -1,
          97, 96, 95, 94, 93, -1, 92, 91, 90, -1, -1, -1, -1, 85, 84, 83,
          81, 80, 79, 78, 77, -1, -1, -1, -1, -1, 70, 69, 68, 67, 66, -1,
          8, 7, 6, 5, 4, 3, 2, 1, 0, -1]
         )
    ]
)
def test_bracket_to_pairings(brackets, pairings):
    """Ensure that the function `Secondary.to_pairing` works well.

    The second structure is the molecule `5s_Acanthamoeba-castellanii-1`
    from the archiveII dataset.
    """
    encoding = structure.Secondary.to_pairings(brackets)
    assert pairings == encoding, \
        f"Pairings `{encoding}` do not match `{brackets}`."


@pytest.mark.parametrize(
    "bracket, elements",
    [
        ("(((((((((...((((((.........))))))........((((((.......)))))).."
            + ")))))))))",
         "sssssssssmmmsssssshhhhhhhhhssssssmmmmmmmmsssssshhhhhhhssssssmms"
            + "ssssssss"),
        ("..((...((.....))...)).....", "eessiiisshhhhhssiiisseeeee"),
        ("((...((.....))))......", "ssbbbsshhhhhsssseeeeee")
    ]
)
def test_structure_elements(bracket, elements):
    """Ensure that secondary structure elements can be accurately
    identified.
    """
    assert elements == structure.Secondary.to_elements(bracket)


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


def test_secondary_structure_folding():
    """Ensure that errors can be found in the pairing matrix."""
    VALID_PAIRINGS = ((0, 19), (19, 0), (3, 17), (17, 3), (5, 14), (14, 5))
    INVALID_PAIRINGS = ((1, 18), (18, 2), (4, 18), (15, 4))
    matrix = np.zeros((20, 20))
    for a, b in (VALID_PAIRINGS + INVALID_PAIRINGS):
        matrix[a][b] = 1
    fold = structure.Secondary.fold_matrix(matrix)
    for i in range(fold.shape[0]):
        for j in range(fold.shape[0]):
            if (i, j) in INVALID_PAIRINGS or (j, i) in INVALID_PAIRINGS:
                assert fold[i][j] == 0
            else:
                assert fold[i][j] == 1
