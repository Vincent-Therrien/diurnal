"""
    Test the diurnal.align module.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: February 2024
    - License: MIT
"""


from diurnal import align, structure


def test_needleman_wunsch():
    a = "AAACCCCTA"
    b = "CCCTA"
    alignment = align.needleman_wunsch(a, b)
    expected_alignment = [(0, None), (1, None), (2, None), (3, 0), (4, 1), (5, 2), (6, None), (7, 3), (8, 4)]  # nopep8
    assert expected_alignment == alignment


def test_fold():
    #  GGAAGGUU
    #    ||  ||
    a = "AACCUUCC"
    pairings = align.fold(a)
    brackets = "".join(structure.Secondary.to_bracket(pairings))
    assert brackets == "((..)).."

    # print()
    # a = "GGUGACCCUAGGGUCCCCCCGCAAUGAUAACUUGUGAACUCGGCCAGGCCCGGAAGGGAGCAACCGCAGCAAGCGACUCGUGUGCCGGGGUGUCGGCUCUAGGGGAACCUCCAA"
    # pairings = align.fold(a)
    # brackets = "".join(structure.Secondary.to_bracket(pairings))
    # print(brackets)
    # pairings = [107, -1, 105, -1, -1, 102, 101, -1, -1, -1, 97, 96, -1, 94, -1, -1, 89, 88, 87, 86, 85, 81, -1, 80, 79, 78, 77, -1, -1, -1, 72, 71, 70, 69, 68, -1, -1, -1, -1, -1, 65, 64, 63, -1, -1, -1, -1, -1, 57, 56, 55, -1, -1, -1, -1, 50, 49, 48, -1, -1, -1, -1, -1, 42, 41, 40, -1, -1, 34, 33, 32, 31, 30, -1, -1, -1, -1, 26, 25, 24, 23, 21, -1, -1, -1, 20, 19, 18, 17, 16, -1, -1, -1, -1, 13, -1, 11, 10, -1, -1, -1, 6, 5, -1, -1, 2, -1, 0, -1, -1, -1, -1, -1, -1]
    # brackets = "".join(structure.Secondary.to_bracket(pairings))
    # print(brackets)


# data\archiveII\srp_Shew.onei._AE014299.ct
# original: GGUGACCCUAGGGUCCCCCCGCAAUGAUAACUUGUGAACUCGGCCAGGCCCGGAAGGGAGCAACCGCAGCAAGCGACUCGUGUGCCGGGGUGUCGGCUCUAGGGGAACCUCCAA
# reverse:  AACCUCCAAGGGGAUCUCGGCUGUGGGGCCGUGUGCUCAGCGAACGACGCCAACGAGGGAAGGCCCGGACCGGCUCAAGUGUUCAAUAGUAACGCCCCCCUGGGAUCCCAGUGG
# [107, -1, 105, -1, -1, 102, 101, -1, -1, -1, 97, 96, -1, 94, -1, -1, 89, 88, 87, 86, 85, 81, -1, 80, 79, 78, 77, -1, -1, -1, 72, 71, 70, 69, 68, -1, -1, -1, -1, -1, 65, 64, 63, -1, -1, -1, -1, -1, 57, 56, 55, -1, -1, -1, -1, 50, 49, 48, -1, -1, -1, -1, -1, 42, 41, 40, -1, -1, 34, 33, 32, 31, 30, -1, -1, -1, -1, 26, 25, 24, 23, 21, -1, -1, -1, 20, 19, 18, 17, 16, -1, -1, -1, -1, 13, -1, 11, 10, -1, -1, -1, 6, 5, -1, -1, 2, -1, 0, -1, -1, -1, -1, -1, -1]
# Best ali. ((.(.((((((.(.((((((((((.(((..((((((...(.(.((...((.))...))..).)...)).))))..).)).).))).)))).).)..).))))))..))).....
# True      (.(..((...((.(..((((((.((((...(((((.....(((.....(((....))).....)))..)))))....)))))...)))))....).))...))..).)......
