"""
    RNA structure mockup module intended for prototyping.

    This module can generate simple datasets that imitate the data
    format of RNA structures. These data can be used to ensure that
    a predictive model accepts the values, use them for training, and
    produce an output, but they **do not reflect the structures of real
    RNA molecules**.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: January 2024
    - License: MIT
"""


from random import uniform, choice
import numpy as np

from diurnal import structure


BASES = ["A", "U", "C", "G"]
PAIRINGS = {
    "A": "U",
    "U": "A",
    "C": "G",
    "G": "C"
}
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
    if n < structure.Constants.LOOP_MIN_DISTANCE + 2:
        return f"{choice(BASES)}" * n, [-1 for _ in range(n)]
    else:
        pairings = '.' * structure.Constants.LOOP_MIN_DISTANCE
        while len(pairings) < n:
            if len(pairings) == n - 1:
                pairings = pairings + '.'
            else:
                if uniform(0, 1) < 0.5:
                    pairings = '(' + pairings + ')'
                else:
                    if uniform(0, 1) < 0.5:
                        pairings = pairings + '.'
                    else:
                        pairings = '.' + pairings
        pairings = structure.Secondary.to_pairings(pairings)
        return make_primary(pairings), pairings


class PairingMatrix:
    """Set of functions designed to generate pairing matrices for
    prototyping.
    """

    def single_pairing(dim: int, n: int) -> tuple:
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
                        primary[x][y] = codes["invalid"]
            P.append(primary)
            S.append(secondary)
        return P, S
