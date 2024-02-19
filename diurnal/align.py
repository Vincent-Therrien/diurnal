"""
    RNA structure alignment module.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: February 2024
    - License: MIT
"""


from collections import deque
from itertools import product


def needleman_wunsch(x: str, y: str) -> list[str]:
    """Run the Needleman-Wunsch algorithm on two sequences.

    From https://johnlekberg.com/blog/2020-10-25-seq-align.html

    Args:
        x (str): First sequence
        y (str): Second sequence

    Returns (list[str]): List of alignments.
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


def fold(x: str) -> list[str]:
    """Align a sequence with its inverse."""
    reverse = [i for i in x[::-1]]
    pairs = {
        "A": "U",
        "C": "G",
        "G": "C",
        "U": "A"
    }
    complementary = [pairs[i] for i in reverse]
    alignment = needleman_wunsch(x, complementary)
    alignment = [a for a in alignment if a[0] != None]
    pairings = [len(x) - i[1] - 1 if i[1] != None and i[1] > -1 else -1 for i in alignment]
    for i, p in enumerate(pairings):
        if p > -1 and x[i] != pairs[x[p]]:
            pairings[i] = -1
    return pairings


def display(x, y, alignment):
    print("".join(
        "-" if i is None else x[i] for i, _ in alignment
    ))
    print("".join(
        "-" if j is None else y[j] for _, j in alignment
    ))
