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

    # a = "GGUGACCCUAGGGUCCCCCCGCAAUGAUAACUUGUGAACUCGGCCAGGCCCGGAAGGGAGCAACCGCAGCAAGCGACUCGUGUGCCGGGGUGUCGGCUCUAGGGGAACCUCCAA"
    # longest = align.fold(a, 3)
    # matrix = align.fold_contact_matrix(a, 4)
    # potential = structure.Primary.to_matrix(a)
    # mask = structure.Primary.to_mask(potential)
    # matrix *= mask
    # visualize.secondary_structures_heatmap(matrix)
    # p = [107, -1, 105, -1, -1, 102, 101, -1, -1, -1, 97, 96, -1, 94, -1, -1, 89, 88, 87, 86, 85, 81, -1, 80, 79, 78, 77, -1, -1, -1, 72, 71, 70, 69, 68, -1, -1, -1, -1, -1, 65, 64, 63, -1, -1, -1, -1, -1, 57, 56, 55, -1, -1, -1, -1, 50, 49, 48, -1, -1, -1, -1, -1, 42, 41, 40, -1, -1, 34, 33, 32, 31, 30, -1, -1, -1, -1, 26, 25, 24, 23, 21, -1, -1, -1, 20, 19, 18, 17, 16, -1, -1, -1, -1, 13, -1, 11, 10, -1, -1, -1, 6, 5, -1, -1, 2, -1, 0, -1, -1, -1, -1, -1, -1]
    # matrix = structure.Secondary.to_matrix(p)
    # visualize.secondary_structures_heatmap(matrix)


# data\archiveII\srp_Shew.onei._AE014299.ct
# original: GGUGACCCUAGGGUCCCCCCGCAAUGAUAACUUGUGAACUCGGCCAGGCCCGGAAGGGAGCAACCGCAGCAAGCGACUCGUGUGCCGGGGUGUCGGCUCUAGGGGAACCUCCAA
# reverse:  AACCUCCAAGGGGAUCUCGGCUGUGGGGCCGUGUGCUCAGCGAACGACGCCAACGAGGGAAGGCCCGGACCGGCUCAAGUGUUCAAUAGUAACGCCCCCCUGGGAUCCCAGUGG
# [107, -1, 105, -1, -1, 102, 101, -1, -1, -1, 97, 96, -1, 94, -1, -1, 89, 88, 87, 86, 85, 81, -1, 80, 79, 78, 77, -1, -1, -1, 72, 71, 70, 69, 68, -1, -1, -1, -1, -1, 65, 64, 63, -1, -1, -1, -1, -1, 57, 56, 55, -1, -1, -1, -1, 50, 49, 48, -1, -1, -1, -1, -1, 42, 41, 40, -1, -1, 34, 33, 32, 31, 30, -1, -1, -1, -1, 26, 25, 24, 23, 21, -1, -1, -1, 20, 19, 18, 17, 16, -1, -1, -1, -1, 13, -1, 11, 10, -1, -1, -1, 6, 5, -1, -1, 2, -1, 0, -1, -1, -1, -1, -1, -1]
# Best ali. ((.(.((((((.(.((((((((((.(((..((((((...(.(.((...((.))...))..).)...)).))))..).)).).))).)))).).)..).))))))..))).....
# True      (.(..((...((.(..((((((.((((...(((((.....(((.....(((....))).....)))..)))))....)))))...)))))....).))...))..).)......
