"""
    RNA sequence alignment module.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: February 2024
    - License: MIT
"""


import numpy as np
from collections import deque
from itertools import product

from diurnal import structure


PAIRS = {
    "A": "U",
    "C": "G",
    "G": "C",
    "U": "A",
    "-": "-"
}


def needleman_wunsch(x: str, y: str) -> list[tuple[int]]:
    """Run the Needleman-Wunsch algorithm on two sequences.

    From https://johnlekberg.com/blog/2020-10-25-seq-align.html

    Args:
        x (str): First sequence
        y (str): Second sequence

    Returns (list[tuple[int]]): List of alignments.
    """
    N, M = len(x), len(y)
    s = lambda a, b: int(a == b)

    DIAG = -1, -1
    LEFT = -1, 0
    UP = 0, -1

    # Create tables F and Ptr
    F = {}
    Ptr = {}

    F[-1, -1] = 0
    for i in range(N):
        F[i, -1] = -i
    for j in range(M):
        F[-1, j] = -j

    option_Ptr = DIAG, LEFT, UP
    for i, j in product(range(N), range(M)):
        option_F = (
            F[i - 1, j - 1] + s(x[i], y[j]),
            F[i - 1, j] - 1,
            F[i, j - 1] - 1,
        )
        F[i, j], Ptr[i, j] = max(zip(option_F, option_Ptr))

    # Work backwards from (N - 1, M - 1) to (0, 0)
    # to find the best alignment.
    alignment = deque()
    i, j = N - 1, M - 1
    while i >= 0 and j >= 0:
        direction = Ptr[i, j]
        if direction == DIAG:
            element = i, j
        elif direction == LEFT:
            element = i, None
        elif direction == UP:
            element = None, j
        alignment.appendleft(element)
        di, dj = direction
        i, j = i + di, j + dj
    while i >= 0:
        alignment.appendleft((i, None))
        i -= 1
    while j >= 0:
        alignment.appendleft((None, j))
        j -= 1

    return list(alignment)


def complementary(x: str) -> str:
    """Return a reversed and complementary sequence.

    Example:
        >>> reverse("AACGT")
        ACGTT

    Args:
        x (str): RNA sequence.

    Returns (str): Reverse complementary sequence.
    """
    reverse = [i for i in x[::-1]]
    return "".join([PAIRS[i] for i in reverse])


def optimal_fold(x: str) -> list[int]:
    """Align a sequence with its complementary sequence with the
    Needleman-Wunsch algorithm.

    Args:
        x: RNA sequence.

    Returns (list[int]): List of pairings.
    """
    inverse = complementary(x)
    alignment = needleman_wunsch(x, inverse)
    alignment = [a for a in alignment if a[0] != None]
    pairings = []
    for i in alignment:
        if i[1] != None and i[1] > -1:
            pairings.append(len(x) - i[1] - 1)
        else:
            pairings.append(-1)
    for i, p in enumerate(pairings):
        if p > -1 and x[i] != PAIRS[x[p]]:
            pairings[i] = -1
    return pairings


def optimal_contact_matrix(x: str, size: int) -> np.array:
    """Return the contact matrix of the optimal alignment of a folded
    sequence.

    Args:
        x (str): RNA sequence.
        size (int): Contact matrix dimension. `0` for no padding.
    """
    pairings = optimal_fold(x)
    return structure.Secondary.to_matrix(pairings, size)


def continuous(x: str, y: str) -> list[tuple[int]]:
    """Find uninterrupted alignments between two sequences.

    Args:
        x (str): RNA sequence.
        y (str): RNA sequence.

    Returns (list[tuple[int]]): Alignment.
    """
    assert len(x) == len(y)
    alignment = []
    for i in range(len(x)):
        if x[i] == y[i]:
            alignment.append((i, i))
        else:
            alignment.append((i, None))
    return alignment


def display(x: str, y: str, alignment: list[str]) -> None:
    """Display an alignment of two sequences.

    Args:
        x (str): RNA sequence.
        y (str): RNA sequence.
        alignment (list[tuple[int]]): Alignment.
    """
    print("".join(
        "-" if i is None else x[i] for i, _ in alignment
    ))
    print("".join(
        "-" if j is None else y[j] for _, j in alignment
    ))


def longest(x: str, y: str, minimum: int = 0) -> list[tuple[int]]:
    """Find the longest continuous alignments between two sequences.

    Args:
        x (str): RNA sequence.
        y (str): RNA sequence.
        minimum (int): Sequence minimum length. Shorter sequences are
            not included in the returned list. `0` for not minimum.

    Returns (list[tuple[int]]): List aligned indices.
    """
    assert len(x) == len(y)
    alignments = [[]]
    for i in range(len(x)):
        if x[i] == y[i]:
            alignments[-1].append(i)
        elif alignments[-1]:
            if minimum and len(alignments[-1]) < minimum:
                alignments[-1] = []
            else:
                alignments.append([])
    alignments.sort(key=lambda b: len(b), reverse=True)
    return [a for a in alignments if len(a) >= minimum]


def display_longest(x: str, y: str, alignments: list[tuple[int]]) -> None:
    """Display the continuous alignments beginning by the longest.

    Args:
        x (str): RNA sequence.
        y (str): RNA sequence.
        alignment (list[tuple[int]]): Alignment list.
    """
    prefix_len = 4
    print(f"{' ' * prefix_len}{x}")
    print(f"{' ' * prefix_len}{y}")
    for alignment in alignments:
        prefix = str(len(alignment))
        sequence = ["." for _ in range(len(x))]
        for a in alignment:
            sequence[a] = "|"
        print(f"{prefix}{' ' * (prefix_len - len(prefix))}{''.join(sequence)}")


def longest_fold(x: str, minimum: int = 0) -> list[tuple[int]]:
    """Find the longest alignment of an RNA sequence with its
    complementary sequence.

    Args:
        x (str): RNA sequence.
        minimum (int): Minimum length of an alignment.

    Returns (list[tuple[int]]): Alignment list.
    """
    y = complementary(x)
    N = len(x)
    alignments = [[]]
    for i in range(N):
        if x[i] == y[i]:
            alignments[-1].append(i)
        elif alignments[-1]:
            if minimum and len(alignments[-1]) < minimum:
                alignments[-1] = []
            else:
                alignments.append([])
    alignments.sort(key=lambda b: len(b), reverse=True)
    return [a for a in alignments if len(a) >= minimum]


def longest_sliding_fold(x: str, minimum: int = 0) -> list[tuple[int]]:
    """Find the longest alignment of an RNA sequence with its
    complementary sequence slid at all positions.

    Args:
        x (str): RNA sequence.
        minimum (int): Minimum length of an alignment.

    Returns (list[tuple[int]]): Pairing list.
    """
    N = len(x)
    pairings = []
    for i in range(-len(x) + minimum, len(x) - minimum):
        padding = ["-"] * N
        y = padding + list(complementary(x)) + padding
        y = y[N - i:2 * N - i]
        alignment = sum(longest(x, y, minimum), [])
        window_pairings = []
        for a in alignment:
            corresponding = N - 1 + i - a
            if corresponding in alignment:
                window_pairings.append((a, corresponding))
        if window_pairings:
            pairings.append(tuple(window_pairings))
    return pairings


def display_longest_fold(x: str, alignments: list[tuple[int]]) -> None:
    """Display the continuous alignments beginning by the longest.

    Args:
        x (str): RNA sequence.
        alignment (list[tuple[int]]): Alignment list.
    """
    prefix_len = 4
    print(f"{' ' * prefix_len}{x}")
    print(f"{' ' * prefix_len}{complementary(x)}")
    for alignment in alignments:
        prefix = str(len(alignment))
        sequence = ["." for _ in range(len(x))]
        for a in alignment:
            sequence[a[0]] = "^"
            sequence[a[1]] = "v"
        print(f"{prefix}{' ' * (prefix_len - len(prefix))}{''.join(sequence)}")


def sliding_contact_matrix(x: str, size: int, minimum: int = 0) -> np.array:
    """Return the contact matrix of all possible alignments of a folded
    sequence.

    Args:
        x (str): RNA sequence.
        size (int): Contact matrix dimension. `0` for no padding.

    """
    pairings = longest_sliding_fold(x, minimum)
    if size == 0:
        size = len(x)
    matrix = np.zeros((size, size))
    for pairs in pairings:
        for p in pairs:
            matrix[p[1], p[0]] = 1
    return matrix
