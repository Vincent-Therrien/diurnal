"""
    RNA structure mockup module intended for prototyping.

    This module can generate simple datasets that imitate the data
    format of RNA structures. These data can be used to ensure that
    a predictive model accepts the values, use them for training, and
    produce an output.
"""


from random import randint
import numpy as np

from diurnal import structure


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