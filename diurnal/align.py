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
    "-": "-",
    "N": "-"
}


def needleman_wunsch(x: str, y: str) -> list[tuple[int]]:
    """Run the Needleman-Wunsch algorithm on two sequences to find
    their optimal alignment.

    From https://johnlekberg.com/blog/2020-10-25-seq-align.html

    Args:
        x (str): First sequence
        y (str): Second sequence

    Returns (list[tuple[int]]): List of alignments as index pairs.
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


def inverse(x: str) -> str:
    """Return a reversed complementary sequence.

    Example:
        >>> inverse("AACGU")
        ACGUU

    Args:
        x (str): RNA sequence.

    Returns (str): Inverse sequence.
    """
    reverse = [i for i in x[::-1]]
    return "".join([PAIRS[i] for i in reverse])


def optimal_fold(x: str) -> list[tuple[int]]:
    """Align a sequence with its inverse sequence with the
    Needleman-Wunsch algorithm.

    Args:
        x: RNA sequence.

    Returns (list[tuple[int]]): List of alignments as index pairs.
    """
    alignment = needleman_wunsch(x, inverse(x))
    alignment = [a for a in alignment if a[0] != None]
    return alignment


def to_pairings(alignment: list[tuple[int]], x: str) -> list[int]:
    """Convert an alignment represented as index pairs to a list of
    pairings, as represented in CT files.

    Args:
        alignment (list[tuple[int]]): Alignment as index pairs.
        x (str): RNA sequence.

    Returns (list[int]): List of pairings.
    """
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


def to_matrix(alignments: list[tuple[int]], size: int) -> np.array:
    """Return the contact matrix corresponding to a list of alignments.

    Args:
        alignment (list[tuple[int]]): Alignment list.
        size (int): Contact matrix dimension.
    """
    matrix = np.zeros((size, size))
    for pair in alignments:
        matrix[pair[1], pair[0]] = 1
    return matrix


def continuous(x: str, y: str) -> list[tuple[int]]:
    """Find uninterrupted alignments between two sequences.

    Args:
        x (str): RNA sequence.
        y (str): RNA sequence.

    Returns (list[tuple[int]]): List of alignments as index pairs.
    """
    assert len(x) == len(y)
    alignment = []
    for i in range(len(x)):
        if x[i] == y[i]:
            alignment.append((i, i))
        else:
            alignment.append((i, None))
    return alignment


def display(x: str, y: str, alignment: list[tuple[int]]) -> None:
    """Display an alignment of two sequences.

    Args:
        x (str): RNA sequence.
        y (str): RNA sequence.
        alignment (list[tuple[int]]): List of alignments as index pairs.
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

    Returns (list[tuple[int]]): List pairing indices.
    """
    assert len(x) == len(y)
    N = len(x)
    alignments = [[]]
    for i in range(N):
        if x[i] == y[i]:
            alignments[-1].append((i, i))
        elif alignments[-1]:
            if minimum and len(alignments[-1]) < minimum:
                alignments[-1] = []
            else:
                alignments.append([])
    alignments = [a for a in alignments if len(a) >= minimum]
    return sum(alignments, [])


def fold(x: str, minimum: int = 0) -> list[tuple[int]]:
    """Find the all alignments of an RNA sequence with its
    inverse sequence.

    Args:
        x (str): RNA sequence.
        minimum (int): Minimum length of an alignment.

    Returns (list[tuple[int]]): List of alignment index pairs.
    """
    N = len(x)
    pairings = []
    for i in range(-len(x) + minimum, len(x) - minimum):
        padding = ["-"] * N
        y = padding + list(inverse(x)) + padding
        y = y[N - i:2 * N - i]
        alignment = longest(x, y, minimum)
        for a in alignment:
            pairings.append((a[0], N - 1 + i - a[0]))
    return pairings


def optimal_fold_contact_matrix(x: str, size: int = 0) -> np.array:
    """Return the contact matrix of the best folded alignment.

    Args:
        x (str): RNA sequence.
        size (int): Matrix dimension. Use `0` for no padding.

    Returns (np.array): Optimal contact matrix.
    """
    if not size:
        size = len(x)
    matrix = np.zeros((size, size))
    pairings = optimal_fold(x)
    for pair in pairings:
        matrix[pair[1], pair[0]] = 1
    return matrix


def fold_contact_matrix(x: str, size: int = 0, minimum: int = 3) -> np.array:
    """Return the contact matrix of all possible alignments of a folded
    sequence.

    Args:
        x (str): RNA sequence.
        size (int): Matrix dimension. Use `0` for no padding.
        minimum (int): Minimum alignment length.

    Returns (np.array): Contact matrix.
    """
    if not size:
        size = len(x)
    matrix = np.zeros((size, size))
    pairings = fold(x, minimum)
    for pair in pairings:
        matrix[pair[1], pair[0]] = 1
    return matrix
