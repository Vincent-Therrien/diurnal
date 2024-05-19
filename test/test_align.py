"""
    Test the diurnal.align module.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: February 2024
    - License: MIT
"""


from diurnal import align, structure, visualize


def test_needleman_wunsch():
    a = "AAACCCCTA"
    b = "CCCTA"
    alignment = align.needleman_wunsch(a, b)
    expected_alignment = [(0, None), (1, None), (2, None), (3, 0), (4, 1), (5, 2), (6, None), (7, 3), (8, 4)]  # nopep8
    assert expected_alignment == alignment


def test_optimal_fold():
    #  GGAAGGUU
    #    ||  ||
    a = "AACCUUCC"
    alignment = align.optimal_fold(a)
    pairings = align.to_pairings(alignment, a)
    brackets = "".join(structure.Secondary.to_bracket(pairings))
    assert brackets == "((..)).."


def test_continuous():
    a = "AAACCCUUUGGG"
    b = "ACAGGCUAAAGC"
    alignment = align.continuous(a, b)
    expected_alignment = [(0, 0), (1, None), (2, 2), (3, None), (4, None), (5, 5), (6, 6), (7, None), (8, None), (9, None), (10, 10), (11, None)]  # nopep8
    assert expected_alignment == alignment


def test_longest():
    a = "AAACCCUUUGGG"
    b = "AAACUUUGGGGG"
    expected = [(0, 0), (1, 1), (2, 2), (3, 3), (6, 6), (9, 9), (10, 10), (11, 11)]  # nopep8
    alignments = align.longest(a, b)
    assert expected == alignments


def test_longest_fold():
    a = "AAACCCUUUGGGUUU"
    fold = align.longest(a, align.inverse(a))
    assert fold == [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14)]  # nopep8


def test_longest_fold_sliding():
    a = "GGUGACCCUAGG"
    longest = align.fold(a, 3)
    expected_longest = [(0, 6), (1, 5), (2, 4), (4, 2), (5, 1), (6, 0), (6, 11), (7, 10), (8, 9), (9, 8), (10, 7), (11, 6)]
    assert longest == expected_longest
