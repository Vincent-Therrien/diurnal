"""
    Transform RNA structures into useful representations.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: June 2023
    - License: MIT
"""


import inspect
import numpy as np


class Schemes:
    """RNA structural codes to transform data into other representations.

    Attributes:
        IUPAC_TO_ONEHOT (dict): One-hot encoding dictionary for IUPAC
            symbols. See: https://www.bioinformatics.org/sms/iupac.html
        IUPAC_ONEHOT_PAIRINGS (dict): One-hot encoded nucleotide
            pairings, including normal ones (AU, UA, CG, and GC) and
            wobble pairs (GU and UG).
        BRACKET_TO_ONEHOT (dict): One-hot encoding dictionary for a
            secondary structure that relies on the bracket notation. `.`
            is an unpaired base. `(` is a base paired to a downstream
            base. `)` is a base paired to an upstream base. ` ` is a
            padding (i.e. empty) base.
        SHADOW_ENCODING (dict): One-hot encoding dictionary to encode
            the shadow of the secondary structure (i.e. the symbols `(`
            and `)` of the bracket notation are considered identical).
    """
    IUPAC_TO_ONEHOT = {
        #     A  C  G  U
        "A": [1, 0, 0, 0],
        "C": [0, 1, 0, 0],
        "G": [0, 0, 1, 0],
        "U": [0, 0, 0, 1],
        "T": [0, 0, 0, 1],
        ".": [0, 0, 0, 0],
        "-": [0, 0, 0, 0],
        "R": [1, 0, 1, 0],
        "Y": [0, 1, 0, 1],
        "S": [0, 1, 1, 0],
        "W": [1, 0, 0, 1],
        "K": [0, 0, 1, 1],
        "M": [1, 1, 0, 0],
        "B": [0, 1, 1, 1],
        "D": [1, 0, 1, 1],
        "H": [1, 1, 0, 1],
        "V": [1, 1, 1, 0],
        "N": [1, 1, 1, 1],
    }

    IUPAC_ONEHOT_PAIRINGS = {
        "AU":       [1, 0, 0, 0, 0, 0, 0, 0],
        "UA":       [0, 1, 0, 0, 0, 0, 0, 0],
        "CG":       [0, 0, 1, 0, 0, 0, 0, 0],
        "GC":       [0, 0, 0, 1, 0, 0, 0, 0],
        "GU":       [0, 0, 0, 0, 1, 0, 0, 0],
        "UG":       [0, 0, 0, 0, 0, 1, 0, 0],
        "unpaired": [0, 0, 0, 0, 0, 0, 1, 0],  # Unpaired base.
        "invalid":  [0, 0, 0, 0, 0, 0, 0, 1],  # Impossible pairing (e.g. AA).
        "-":        [0, 0, 0, 0, 0, 0, 0, 0]   # Padding element (i.e. empty).
    }

    BRACKET_TO_ONEHOT = {
        "(": [1, 0, 0],
        ".": [0, 1, 0],
        ")": [0, 0, 1],
        " ": [0, 0, 0],
    }

    SHADOW_ENCODING = {
        "(": 1,
        ".": 0,
        ")": 1,
        " ": 0
    }


class Primary:
    """Transform RNA primary structures into useful formats."""

    def to_vector(bases: list, size: int = 0) -> np.array:
        """Transform a sequence of bases into a vector.

        Args:
            bases (list(str)): A sequence of bases. E.g.: ``['A', 'U']``.
            size (int): Size of a normalized vector. `0` for no padding.

        Returns (np.array): One-hot encoded primary structure.
        """
        vector = [Schemes.IUPAC_TO_ONEHOT[base] for base in bases]
        if size:
            element = Schemes.IUPAC_TO_ONEHOT['.']
            return Primary._pad_vector(vector, size, element)
        return np.array(vector)

    def to_matrix(bases: list, size: int = 0) -> np.array:
        """Encode a primary structure in a matrix of potential pairings.

        Create an `n` by `n` matrix, where `n` is the number of bases,
        whose element each represent a potential RNA base pairing. For
        instance, the pairing `AA` is not possible and will be assigned
        the `invalid` value of the map. `AU` is a valid pairing and the
        corresponding element will be assigned to its value in the map.

        Args:
            bases (list(str)): Sequence of bases.
            size (int): Matrix dimension. `0` for no padding.

        Returns (np.array): Encoded matrix.
        """
        N_MINIMUM_DISTANCE = 4
        if size == 0:
            size = len(bases)
        map = Schemes.IUPAC_ONEHOT_PAIRINGS
        empty = map['-']
        matrix = [[empty for _ in range(size)] for _ in range(size)]
        for row in range(len(bases)):
            for col in range(len(bases)):
                pairing = bases[row] + bases[col]
                if row == col:
                    matrix[row][col] = map["unpaired"]
                elif (abs(row - col) < N_MINIMUM_DISTANCE
                      or len(set(pairing)) == 1):
                    matrix[row][col] = map["invalid"]
                elif pairing in map:
                    matrix[row][col] = map[pairing]
                else:
                    matrix[row][col] = map["invalid"]
        return np.array(matrix)

    def to_bases(
            vector, strip: bool = True, map: dict = Schemes.IUPAC_TO_ONEHOT
            ) -> list:
        """Transform a vector into a sequence of bases.

        Args:
            vector (list-like): One-hot encoded primary structure.
            strip (bool): Remove empty elements at the vector's right end.
            map: A dictionary or function that maps bases to vectors.

        Returns (list): A sequence of bases. E.g.: ``['A', 'U']``.
        """
        if strip:
            element = map['.'] if type(map) == dict else map('.')
            vector = Primary._unpad_vector(vector, element)
        if inspect.isfunction(map):
            return map(vector)
        bases = []
        for base in vector:
            base = list(base)
            for key, code in map.items():
                if code == base:
                    bases.append(key)
                    break
        return bases

    def _pad_vector(vector: np.array, size: int, element: list) -> np.array:
        """Append elements at the right extremity of a vector.

        Args:
            vector (list): A vector of elements.
            size (int): The final size of the vector.
            element (list): The element to add to the vector.

        Returns (np.array): The padded list of size "size".
        """
        difference = size - len(vector)
        if difference > 0:
            return np.concatenate((vector, difference * [element]))
        return np.array(vector)

    def _unpad_vector(vector: np.array, element: list) -> np.array:
        """Remove the empty elements appended to the right side of a
        vector.

        Args:
            vector (np.array): Vector-encoded primary structure.
            element (list): Empty element of the set (e.g. `[0, 0, 0, 0]`).

        Returns (np.array): Unpadded vector.
        """
        i = len(vector) - 1
        while i > 0:
            nonzero = False
            for j, e in enumerate(vector[i]):
                if e != element[j]:
                    nonzero = True
            if nonzero:
                return vector[0:i+1]
            i -= 1
        return vector

    def unpad_matrix(matrix: np.array) -> np.array:
        """Strip a matrix of its padding elements.

        Args:
            matrix: Input matrix (Numpy array of Python lists).

        Returns (list): Unpadded matrix.
        """
        for i, row in enumerate(matrix):
            element = list(row[0])
            if element == Schemes.IUPAC_ONEHOT_PAIRINGS["-"]:
                return matrix[:i, :i]
        return matrix


class Secondary:
    """Transform RNA secondary structures into useful formats."""

    def to_vector(pairings: list, size: int = 0) -> np.array:
        """Encode pairings in a one-hot bracket-based secondary
        structure.

        Args:
            pairings (List[int|str]): A list of nucleotide pairings.
                The pairing `(((...)))` can be represented as
                `[8, 7, 6, -1, -1, -1, 2, 1, 0]` or
                `['(', '(', '(', '.', '.', '.', ')', ')', ')']`.

        Returns (np.array): One-hot encoded secondary structure.
        """
        if type(pairings[0]) is int:
            bracket = Secondary.to_bracket(pairings)
        elif type(pairings[0]) is str:
            bracket = pairings
        else:
            raise RuntimeError(f"Unrecognized type: {type(pairings[0])}")
        vector = [Schemes.BRACKET_TO_ONEHOT[symbol] for symbol in bracket]
        vector = np.array(vector)
        if size:
            element = Schemes.BRACKET_TO_ONEHOT[' ']
            vector = Secondary._pad(vector, size, element)
        return vector

    def to_matrix(pairings: list, size: int = 0) -> np.array:
        """Encode a secondary structure in a matrix.

        Transform the sequence of pairings into an `n` by `n` matrix,
        where `n` is the number of pairings, whose elements can be `0`
        for an unpaired base and `1` for a paired base.

        Args:
            pairings (list(int): List of base pairings.
            size (int): Dimension of the matrix. `0` for default.

        Returns (np.array): Matrix encoding of the secondary structure.
        """
        if size == 0:
            size = len(pairings)
        matrix = np.array([[0 for _ in range(size)] for _ in range(size)])
        for i in range(len(pairings)):
            if pairings[i] >= 0:
                matrix[i][pairings[i]] = 1
        return matrix

    def to_bracket(pairings: list) -> list:
        """Convert a list of nucleotide pairings into a secondary
        structure bracket notation, e.g. `'(((...)))`'.

        Args:
            pairings (list(int)): A list of nucleotide pairings, e.g.
                the pairing `(((...)))` is represented as
                `[8, 7, 6, -1, -1, -1, 2, 1, 0]`.

        Returns (list): Secondary structure bracket notation.
        """
        if type(pairings[0]) == int:
            encoding = []
            for i, p in enumerate(pairings):
                if p < 0:
                    encoding.append('.')
                elif i < p:
                    encoding.append('(')
                else:
                    encoding.append(')')
            return encoding
        else:
            encoding = []
            characters = list(Schemes.BRACKET_TO_ONEHOT.keys())
            for p in pairings:
                p = p.tolist() if type(p) == np.ndarray else p
                if sum(p) == 0:
                    encoding.append(" ")
                else:
                    encoding.append(characters[p.index(max(p))])
            return encoding

    def _pad(vector: np.array, size: int, element: list) -> np.array:
        """Append elements at the right extremity of a vector.

        Args:
            vector (np.array): A vector of elements.
            size (int): The final size of the vector.
            element (list): The element to add to the vector.

        Returns (np.array): The padded list of size "size:.
        """
        difference = size - len(vector)
        if difference > 0:
            return np.concatenate((vector, difference * [element]))
        return vector
