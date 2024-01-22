"""
    RNA structure mockup module intended for prototyping.

    This module can generate simple datasets that imitate the data
    format of RNA structures. These data can be used to ensure that
    a predictive model accepts the values, use them for training, and
    produce an output, but they **do not reflect the structures of real
    RNA molecules**.
"""


from random import randint, choice, uniform
import numpy as np

from diurnal import structure


BASES = ["A", "T", "C", "G"]
PAIRINGS = {
    "A": "T",
    "T": "A",
    "C": "G",
    "G": "C"
}
LOOP_MIN_DISTANCE = 3  # Minimum number of bases between `(` and `)`.
OPPOSED_MIN_DISTANCE = 1  # Minimum number of bases between `)` and `(`.


def make_primary(pairings: list[int]) -> str:
    """Make a primary structure from a pairing list.

    Args:
        pairings (list[int]): Pairing list. `-1` indicates an unpaired
            base.

    Returns (str): Primary structure.
    """
    primary = [" "] * len(pairings)
    for i, pair in enumerate(pairings):
        if primary[i] != " ":
            continue
        primary[i] = choice(BASES)
        if pair >= 0:
            primary[pair] = PAIRINGS[primary[i]]
    return "".join(primary)


def make_structures(n: int) -> tuple:
    """Make synthetic primary and secondary structures.

    The secondary structure is determined as follows:
    - A third of the bases a unpaired.
    - A third are paired to 3' (downstream) bases.
    - The rest are paired to 5' bases.
    - There is at least 3 bases between paired bases.

    Args:
        n (int): Length of the synthetic structure.

    Returns (tuple): Two-value tuple:
    - The first element is the primary structure as a `str` object.
    - The second element is the secondary structure as a list of
      pairing indices.
    """
    if n < LOOP_MIN_DISTANCE + 2:
        return f"{choice(BASES)}" * n, [-1 for _ in range(n)]
    else:
        pairings = [-1] * n
        n_paired = int(n / 3) * 2
        TO_PAIR = -2
        for _ in range(n_paired):
            while True:
                i = randint(0, n - 1)
                if pairings[i] == -1:
                    pairings[i] = TO_PAIR
                    break
        # Assign 5' tp 3' pairings.
        for i in range(n):
            if pairings[i] == -2:
                for j in range(n - 1, -1, -1):
                    if pairings[j] == -2:
                        pairings[j] = i
                        pairings[i] = j
                        break
        # Fix the inconsistencies entailed by the 5' to 3' pairings.
        for i in range(n - 1):
            if pairings[i] == i + 1:
                pairings.insert(i + 1, -1)
                while True:
                    i = randint(0, n - 1)
                    if pairings[i] == -1:
                        pairings.pop(i)
                        break
        return make_primary(pairings), pairings


class PairingMatrix:
    """Set of functions designed to generate pairing matrices for
    prototyping.
    """

    def single_pairing(dim: int, n: int) -> tuple(list[np.array]):
        """Create pairing matrices that each have a single, random
        pairing.

        Args:
            dim (int): Dimension of the 2D pairing matrices.
            n (int): Number of samples.

        Returns (tuple): Packed primary and secondary structures.
        """
        if dim < 4:
            print(f"Incorrect dimension ({dim}), minimum is 4.")
            return None
        P = list()
        S = list()
        for _ in range(n):
            while True:
                i = randint(4, dim - 1)
                j = randint(0, dim - 5)
                if abs(i - j) > 3:
                    break
            secondary = np.zeros((dim, dim))
            secondary[i][j] = 1
            secondary[j][i] = 1
            codes = structure.Schemes.IUPAC_ONEHOT_PAIRINGS_VECTOR
            primary = np.zeros((dim, dim, len(codes["-"])))
            for x in range(dim):
                for y in range(dim):
                    if x == i and y == j:
                        primary[x][y] = codes["UA"]
                    elif x == j and y == i:
                        primary[x][y] = codes["AU"]
                    else:
                        primary[x][y] = codes["unpaired"]
            P.append(primary)
            S.append(secondary)
        return P, S
