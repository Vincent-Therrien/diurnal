"""
    Transform RNA structures into useful representations.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: June 2023
    - License: MIT
"""


import inspect
import numpy as np

from diurnal.utils import log


class Schemes:
    """RNA structural codes to transform data into other representations.

    Attributes:
        IUPAC_TO_ONEHOT (dict): One-hot encoding dictionary for IUPAC
            symbols. See: https://www.bioinformatics.org/sms/iupac.html
        IUPAC_ONEHOT_PAIRINGS_VECTOR (dict): One-hot encoded nucleotide
            pairings, including normal ones (AU, UA, CG, and GC) and
            wobble pairs (GU and UG). Taken from CNNFold by Booy et al.
        BRACKET_TO_ONEHOT (dict): One-hot encoding dictionary for a
            secondary structure that relies on the bracket notation. `.`
            is an unpaired base. `(` is a base paired to a downstream
            base. `)` is a base paired to an upstream base. `-` is a
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

    IUPAC_ONEHOT_PAIRINGS_VECTOR = {
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

    IUPAC_ONEHOT_PAIRINGS_SCALARS = {
        "AU":       2,
        "UA":       2,
        "CG":       3,
        "GC":       3,
        "GU":       1,
        "UG":       1,
        "unpaired": 0,  # Unpaired base.
        "invalid":  -1,  # Impossible pairing (e.g. AA).
        "-":        -1   # Padding element (i.e. empty).
    }

    BRACKET_TO_ONEHOT = {
        "(": [1, 0, 0],
        ".": [0, 1, 0],
        ")": [0, 0, 1],
        "-": [0, 0, 0],
    }

    SHADOW_ENCODING = {
        "(": 1,
        ".": 0,
        ")": 1,
        "-": 0
    }


class Primary:
    """Transform RNA primary structures into useful formats."""

    def to_onehot(
            bases: list, size: int = 0,
            map: dict = Schemes.IUPAC_TO_ONEHOT) -> np.array:
        """Transform a sequence of bases into a one-hot encoded vector.

        Args:
            bases (List[str] | str): A sequence of bases.
                E.g.: ``['A', 'U']`` or ``AU``.
            size (int): Size of a normalized vector. `0` for no padding.
            map (dict): Assign an input to a vector.

        Returns (np.array): One-hot encoded primary structure.
            E.g.: ``[[1, 0, 0, 0], [0, 1, 0, 0]]``
        """
        vector = [map[base] for base in bases]
        if size:
            element = map['.']
            return Primary._pad_vector(vector, size, element)
        return np.array(vector)

    def to_matrix(
            bases: list, size: int = 0,
            map: dict = Schemes.IUPAC_ONEHOT_PAIRINGS_VECTOR) -> np.array:
        """Encode a primary structure in a matrix of potential pairings.

        Create an `n` by `n` matrix, where `n` is the number of bases,
        whose element each represent a potential RNA base pairing. For
        instance, the pairing `AA` is not possible and will be assigned
        the `invalid` value of the map. `AU` is a valid pairing and the
        corresponding element will be assigned to its value in the map.

        Args:
            bases (list(str)): Sequence of bases.
            size (int): Matrix dimension. `0` for no padding.
            map (dict): Assign a pairing to a matrix element.

        Returns (np.array): Encoded matrix.
        """
        N_MINIMUM_DISTANCE = 4
        if size == 0:
            size = len(bases)
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

    def to_sequence(
            vector, strip: bool = True, map: dict = Schemes.IUPAC_TO_ONEHOT
            ) -> list:
        """Transform a one-hot encoded vector into a sequence of bases.

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
                if e.all() != element[j]:
                    nonzero = True
            if nonzero:
                return vector[0:i+1]
            i -= 1
        return vector

    def unpad_matrix(
            matrix: np.array, map: dict = Schemes.IUPAC_ONEHOT_PAIRINGS_VECTOR
            ) -> np.array:
        """Strip a matrix of its padding elements.

        Args:
            matrix: Input matrix (Numpy array of Python lists).
            map (dict): Assign a pairing to a matrix element.

        Returns (list): Unpadded matrix.
        """
        for i, row in enumerate(matrix):
            element = list(row[0])
            if element == map["-"]:
                return matrix[:i, :i]
        return matrix


class Secondary:
    """Transform RNA secondary structures into useful formats."""

    def to_onehot(
            pairings: list, size: int = 0,
            map: dict = Schemes.BRACKET_TO_ONEHOT) -> np.array:
        """Encode pairings in a one-hot encoded dot-bracket secondary
        structure.

        Args:
            pairings (List[int|str]): A list of nucleotide pairings.
                The pairing `(((...)))` can be represented as
                `[8, 7, 6, -1, -1, -1, 2, 1, 0]` or
                `['(', '(', '(', '.', '.', '.', ')', ')', ')']`.
            size (int): Size of the output. `0` for no padding.
            map (dict): Assign an input to a vector.

        Returns (np.array): One-hot encoded secondary structure.
        """
        if type(pairings[0]) is int:
            bracket = Secondary.to_bracket(pairings)
        elif type(pairings[0]) is str:
            bracket = pairings
        else:
            raise RuntimeError(f"Unrecognized type: {type(pairings[0])}")
        vector = [map[symbol] for symbol in bracket]
        vector = np.array(vector)
        if size:
            element = map['-']
            vector = Secondary._pad(vector, size, element)
        return vector

    def to_matrix(pairings: list, size: int = 0) -> np.array:
        """Encode a secondary structure in a matrix.

        Transform the sequence of pairings into an `n` by `n` matrix,
        where `n` is the number of pairings, whose elements can be `0`
        for an unpaired base and `1` for a paired base.

        Args:
            pairings (list(int): List of base pairings.
            size (int): Dimension of the matrix. `0` for no padding.

        Returns (np.array): Encoded matrix of the secondary structure.
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
        if type(pairings[0]) in (int, float, np.float16, np.float32):
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
                    encoding.append("-")
                else:
                    encoding.append(characters[p.index(max(p))])
            return encoding

    def to_shadow(pairings: list, size: int = 0) -> list:
        """Return the shadow of a secondary structure.

        Args:
            Pairings (List[str]): Secondary structure.
            size (int): Final sequence length.
        """
        if type(pairings[0]) == int:
            shadow = [0 if p == -1 else 1 for p in pairings]
        else:
            shadow = [0 if p == '.' else 1 for p in pairings]
        if size:
            shadow += (size - len(shadow)) * [-1]
        return np.array(shadow)

    def to_pairings(bracket: list) -> list:
        """Convert the bracket notation to a list of pairings.

        Args:
            bracket (List[str] | str): Secondary structure.

        Returns (List[int]): List of pairings.
        """
        pairings = []
        for i, b in enumerate(bracket):
            if b == ".":
                pairings.append(-1)
            elif b == '(':
                count = 0
                for j in range(i + 1, len(bracket)):
                    if bracket[j] == '(':
                        count += 1
                    if bracket[j] == ')':
                        count -= 1
                        if count < 0:
                            pairings.append(j)
                            break
            elif b == ')':
                count = 0
                for j in range(i - 1, -1, -1):
                    if bracket[j] == ')':
                        count += 1
                    if bracket[j] == '(':
                        count -= 1
                        if count < 0:
                            pairings.append(j)
                            break
        return pairings

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

    def _find_external_loops(bracket: list) -> str:
        """Find external loops, i.e. unpaired endings."""
        elements = list("-" * len(bracket))
        for i in range(len(bracket)):
            if bracket[i] == ".":
                elements[i] = "e"
            else:
                break
        for i in range(len(bracket) - 1, 0, -1):
            if bracket[i] == ".":
                elements[i] = "e"
            else:
                break
        return "".join(elements)

    def _find_hairpin_loops(bracket: list) -> str:
        """Find hairpin loops in a secondary structure in bracket
        notation.

        There is a hairpin loop if and only if:

        - bases `i` and `j` are paired and
        - all bases between `i` and `j` are unpaired.

        Reference: https://math.mit.edu/classes/18.417/Slides/rna-prediction-zuker.pdf
        """
        elements = list("-" * len(bracket))
        for i in range(len(bracket)):
            if bracket[i] == ")":
                for j in range(i - 1, 0, -1):
                    if bracket[j] == "(":
                        for e in range(j + 1, i):
                            elements[e] = "h"
                        break
                    if bracket[j] == ")":
                        break
        return "".join(elements)

    def _find_internal_loops(pairings: list) -> str:
        """Find internal loops in the secondary structure.

        Let (i, j) be a pairing and (i', j') be another pairing with
        i < i' < j' < j. (i, j) and (i', j') form an internal loop if
        bases i + 1 to i' - 1 and j + 1 to j' - 1 are unpaired.
        """
        elements = list("-" * len(pairings))
        for i in range(len(pairings) - 1):
            j = pairings[i]
            if i < j and pairings[i + 1] == -1:
                for i_prime in range(i + 1, j):
                    j_prime = pairings[i_prime]
                    if j_prime > -1:
                        break
                i_bulge = set(pairings[i + 1:i_prime - 1]) == {-1}
                j_bulge = set(pairings[j_prime + 1:j - 1]) == {-1}
                len_i = i_prime - i - 1
                len_j = j - j_prime - 1
                if i_bulge and j_bulge:
                    elements[i + 1:i_prime] = ["i"] * len_i
                    elements[j_prime + 1:j] = ["i"] * len_j
                elif i_bulge and len_j == 0:
                    elements[i + 1:i_prime] = ["b"] * len_i
                elif j_bulge and len_i == 0:
                    elements[j_prime + 1:j] = ["b"] * len_j
        return "".join(elements)

    def _find_stems(bracket: list) -> str:
        """Find stems in a secondary structure in bracket notation.

        There is a stem if and only if:

        - bases `i` and `j` are paired and
        - bases `i + 1` and `j - 1` are paired.

        Reference: https://math.mit.edu/classes/18.417/Slides/rna-prediction-zuker.pdf
        """
        elements = list("-" * len(bracket))
        for i in range(len(bracket)):
            if bracket[i] in ['(', ')']:
                elements[i] = "s"
        return "".join(elements)

    def to_elements(pairings: list) -> str:
        """Convert pairings into secondary structure elements.

        The possible *elements* or *loops* are:

        | element         | character |
        +=================+===========+
        | bulge           | `b`       |
        | external loop   | `e`       |
        | hairpin loop    | `h`       |
        | internal loop   | `i`       |
        | multiloop       | `m`       |
        | stem / stacking | `s`       |
        """
        if type(pairings[0]) is not str:
            bracket = Secondary.to_bracket(pairings)
        elif type(pairings[0]) is str:
            bracket = pairings
            pairings = Secondary.to_pairings(bracket)
        # Find structural elements.
        e = Secondary._find_external_loops(bracket)
        h = Secondary._find_hairpin_loops(bracket)
        i = Secondary._find_internal_loops(pairings)
        s = Secondary._find_stems(bracket)
        # Assemble elements.
        possibilities = [e, h, i, s]
        elements = list("-" * len(pairings))
        for i in range(len(pairings)):
            for possibility in possibilities:
                if possibility[i] != "-":
                    if elements[i] == "-":
                        elements[i] = possibility[i]
                    else:
                        log.error(f"Conflict at position {i}.")
                        raise RuntimeError
            if elements[i] == "-":
                elements[i] = 'm'
        return "".join(elements)
