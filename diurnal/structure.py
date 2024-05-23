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
    """RNA structure schemes

    The attributes of this class are used to transform raw RNA sequence
    data into other representations that can be used for prediction
    problems.

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
        "A": (1, 0, 0, 0),
        "C": (0, 1, 0, 0),
        "G": (0, 0, 1, 0),
        "U": (0, 0, 0, 1),
        "T": (0, 0, 0, 1),
        ".": (0, 0, 0, 0),
        "-": (0, 0, 0, 0),
        "R": (1, 0, 1, 0),
        "Y": (0, 1, 0, 1),
        "S": (0, 1, 1, 0),
        "W": (1, 0, 0, 1),
        "K": (0, 0, 1, 1),
        "M": (1, 1, 0, 0),
        "B": (0, 1, 1, 1),
        "D": (1, 0, 1, 1),
        "H": (1, 1, 0, 1),
        "V": (1, 1, 1, 0),
        "N": (1, 1, 1, 1),
    }

    IUPAC_ONEHOT_PAIRINGS_VECTOR = {
        "AU":       (1, 0, 0, 0, 0, 0, 0, 0),
        "UA":       (0, 1, 0, 0, 0, 0, 0, 0),
        "CG":       (0, 0, 1, 0, 0, 0, 0, 0),
        "GC":       (0, 0, 0, 1, 0, 0, 0, 0),
        "GU":       (0, 0, 0, 0, 1, 0, 0, 0),
        "UG":       (0, 0, 0, 0, 0, 1, 0, 0),
        "invalid":  (0, 0, 0, 0, 0, 0, 1, 0),  # Impossible pairing (e.g. AA).
        "-":        (0, 0, 0, 0, 0, 0, 0, 0),  # Padding element (i.e. empty).
    }

    IUPAC_PAIRINGS_SCALARS = {
        "AU":       2,
        "UA":       2,
        "CG":       3,
        "GC":       3,
        "GU":       1,
        "UG":       1,
        "invalid":  0,  # Impossible pairing (e.g. AA).
        "-":        0  # Padding element (i.e. empty).
    }

    IUPAC_PAIRINGS_SCALARS_NEGATIVE_PADDING = {
        "AU":       2,
        "UA":       2,
        "CG":       3,
        "GC":       3,
        "GU":       1,
        "UG":       1,
        "invalid":  0,  # Impossible pairing (e.g. AA).
        "-":        -1   # Padding element (i.e. empty).
    }

    BRACKET_TO_ONEHOT = {
        "(": (1, 0, 0),
        ".": (0, 1, 0),
        ")": (0, 0, 1),
        "-": (0, 0, 0),
    }

    SHADOW_ENCODING = {
        "(": 1,
        ".": 0,
        ")": 1,
        "-": 0
    }


class Constants:
    """Set of physical values that contraint RNA structures.

    Attributes:
        LOOP_MIN_DISTANCE (int): Minimum number of nucleotides between
            two bases paired to each other. For instance, in the
            sequence `ACCCU`, the bases `A` and `U` can be paired
            because they are separated by three bases. However, in the
            sequence `ACU`, the bases `A` and `U` cannot be paired
            because they are too close.
    """
    LOOP_MIN_DISTANCE = 3


class Primary:
    """Transform RNA primary structures into useful formats."""

    def to_onehot(
            bases: list, size: int = 0,
            map: dict = Schemes.IUPAC_TO_ONEHOT) -> np.ndarray:
        """Transform a sequence of bases into a one-hot encoded vector.

        Args:
            bases (List[str] | str): A sequence of bases.
                E.g.: ``['A', 'U']`` or ``AU``.
            size (int): Size of a normalized vector. `0` for no padding.
            map (dict): Assign an input to a vector.

        Returns (np.ndarray): One-hot encoded primary structure.
            E.g.: ``[[1, 0, 0, 0], [0, 1, 0, 0]]``
        """
        vector = [map[base] for base in bases]
        if size:
            element = map['.']
            return Primary._pad_vector(vector, size, element)
        return np.array(vector)

    def to_matrix(
            bases: list[str],
            size: int = 0,
            map: dict = Schemes.IUPAC_ONEHOT_PAIRINGS_VECTOR
        ) -> np.ndarray:
        """Encode a primary structure in a matrix of potential pairings.

        Create an `n` by `n` matrix, where `n` is the number of bases,
        in which element each represent a potential RNA base pairing.
        For instance, the pairing `AA` is not possible and will be
        assigned the `invalid` value of the `map` parameter. `AU` is a
        valid pairing and the corresponding element will be assigned to
        its value in the `map`.

        Args:
            bases (list(str)): Primary structure (sequence of bases).
            size (int): Matrix dimension. `0` for no padding.
            map (dict): Assign a pairing to a matrix element. The
                elements of the map must be (1) convertible to a Numpy
                array and (2) of the same dimension.

        Returns (np.ndarray): Encoded matrix.
        """
        if type(bases[0]) != type(list(map.keys())[0]):
            a = str(type(bases[0]))
            b = str(type(list(map.keys())[0]))
            log.warning(f"Primary.to_matrix type mismatch: {a} and {b}")
        N_MINIMUM_DISTANCE = 4
        if size == 0:
            size = len(bases)
        empty = map['-']
        matrix = [[empty for _ in range(size)] for _ in range(size)]
        for row in range(len(bases)):
            for col in range(len(bases)):
                pairing = bases[row] + bases[col]
                if (abs(row - col) < N_MINIMUM_DISTANCE
                      or len(set(pairing)) == 1):
                    matrix[row][col] = map["invalid"]
                elif pairing in map:
                    matrix[row][col] = map[pairing]
                else:
                    matrix[row][col] = map["invalid"]
        return np.array(matrix)

    def to_mask(
            pairings: np.ndarray,
            size: int = 0,
            map: dict = Schemes.IUPAC_ONEHOT_PAIRINGS_VECTOR
        ) -> np.ndarray:
        """Make a primary structure pairing mask.

        Return the a copy of the input matrix in which impossible
        pairings are set to 0 and possible pairings are set to 1.

        Args:
            pairings (np.ndarray): Primary structure potential pairing
                matrix.
            size (int): Matrix dimension. `0` for no padding.
            map (dict): Dictionary that assigns a type of pairing to an
                encoding.

        Returns (np.ndarray): Pairing matrix mask.
        """
        # Convert the input into an array, if required.
        if type(pairings) == str:
            pairings = Primary.to_matrix(pairings, size)
        if type(pairings) == list and type(pairings[0]) == str:
            pairings = Primary.to_matrix(pairings, size)
        # Create the mask.
        inv_constraints = {v: k for k, v in map.items()}
        output = np.zeros((len(pairings), len(pairings)))
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                pairing = pairings[i][j]
                if type(pairing) in (float, int):
                    pairing = int(pairing)
                else:
                    pairing = tuple(pairing)
                constraint = inv_constraints[pairing]
                if constraint in ("invalid", "-"):
                    output[i][j] = 0
                else:
                    output[i][j] = 1
        return output

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
            base = tuple(base)
            for key, code in map.items():
                if code == base:
                    bases.append(key)
                    break
        return bases

    def _pad_vector(vector: np.ndarray, size: int, element: list) -> np.ndarray:
        """Append elements at the right extremity of a vector.

        Args:
            vector (list): A vector of elements.
            size (int): The final size of the vector.
            element (list): The element to add to the vector.

        Returns (np.ndarray): The padded list of size "size".
        """
        difference = size - len(vector)
        if difference > 0:
            return np.concatenate((vector, difference * [element]))
        return np.array(vector)

    def _unpad_vector(vector: np.ndarray, element: list) -> np.ndarray:
        """Remove the empty elements appended to the right side of a
        vector.

        Args:
            vector (np.ndarray): Vector-encoded primary structure.
            element (list): Empty element of the set (e.g. `[0, 0, 0, 0]`).

        Returns (np.ndarray): Unpadded vector.
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
            matrix: np.ndarray, map: dict = Schemes.IUPAC_ONEHOT_PAIRINGS_VECTOR
            ) -> np.ndarray:
        """Strip a matrix of its padding elements.

        Args:
            matrix: Input matrix (Numpy array of Python lists).
            map (dict): Assign a pairing to a matrix element.

        Returns (list): Unpadded matrix.
        """
        for i in range(len(matrix)):
            element = tuple(matrix[i])
            if element[0] == map["-"]:
                return matrix[:i, :i]
        return matrix


class Secondary:
    """Transform RNA secondary structures into useful formats."""

    def to_onehot(
            pairings: list,
            size: int = 0,
            map: dict = Schemes.BRACKET_TO_ONEHOT
        ) -> np.ndarray:
        """Encode pairings in a one-hot encoded dot-bracket secondary
        structure.

        Args:
            pairings (List[int|str]): A list of nucleotide pairings.
                The pairing `(((...)))` can be represented as
                `[8, 7, 6, -1, -1, -1, 2, 1, 0]` or
                `['(', '(', '(', '.', '.', '.', ')', ')', ')']`.
            size (int): Size of the output. `0` for no padding.
            map (dict): Assign an input to a vector.

        Returns (np.ndarray): One-hot encoded secondary structure.
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

    def to_matrix(pairings: list, size: int = 0) -> np.ndarray:
        """Encode a secondary structure into a contact matrix.

        Transform the sequence of pairings into an `n` by `n` matrix,
        where `n` is the number of pairings, whose elements can be `0`
        for an unpaired base and `1` for a paired base.

        Args:
            pairings (list(int): List of base pairings.
            size (int): Dimension of the matrix. `0` for no padding.

        Returns (np.ndarray): Encoded matrix of the secondary structure.
        """
        if size == 0:
            size = len(pairings)
        matrix = np.zeros((size, size))
        for i in range(len(pairings)):
            if pairings[i] >= 0:
                matrix[i][pairings[i]] = 1
        return matrix

    def to_distance_matrix(
            pairings: list,
            size: int = 0,
            normalize: bool = True,
            power: float = 1
        ) -> np.ndarray:
        """Encode a secondary structure into a score contact matrix.

        Transform the sequence of pairings into an `n` by `n` matrix,
        where `n` is the number of pairings, whose elements can be `1`
        for a paired base and `x` for unpaired bases, where `x` is
        given by: `x = 1 - (d / n)`, in which `d` is the Manhattan
        distance with the closest paired base.

        Args:
            pairings (list(int): List of base pairings.
            size (int): Dimension of the matrix. `0` for no padding.
            normalize (bool): If True, scale distances so that
                paired elements are 1 and the farthest elements are 0.
            power (float): Power to apply to normalized distances.

        Returns (np.ndarray): Encoded matrix of the secondary structure.
        """
        n = len(pairings)
        contact = Secondary.to_matrix(pairings, size)
        contact -= 1
        contact *= -1
        contact *= n
        directions = (
            (1, 0), (1, -1), (0, -1), (-1, -1),
            (-1, 0), (-1, 1), (0, 1), (1, 1)
        )
        for distance in range(n):
            for i in range(n):
                for j in range(n):
                    if contact[i, j] == distance:
                        for d in directions:
                            I = i + d[0]
                            J = j + d[1]
                            if I < 0 or I >= n or J < 0 or J >= n:
                                continue
                            value = contact[I, J]
                            if value > distance + 1:
                                contact[I, J] = distance + 1
        contact /= n
        contact *= -1
        contact += 1
        if normalize:
            contact = Secondary.normalize_distance_matrix(contact)
            return contact ** power
        else:
            return contact

    def normalize_distance_matrix(distance_matrix) -> np.ndarray:
        """Normalize the distance matrix.

        This function returns a new distance matrix whose elements are
        normalized within the range 0.0 (farthest from a paired base)
        to 1.0 (paired base).

        Args:
            distance_matrix (np.ndarray): Result of the function
                `to_distance_matrix`.

        Returns (np.ndarray): Normalized distance matrix.
        """
        normalized = distance_matrix.copy()
        normalized -= 1
        normalized *= -1
        normalized /= np.amax(normalized)
        normalized -= 1
        normalized *= -1
        return normalized

    def quantize_distance_matrix(distance_matrix: np.ndarray) -> np.ndarray:
        """Create a contact matrix from a distance matrix.

        Args:
            distance_matrix (np.ndarray): Result of the function
                `to_distance_matrix`.

        Returns (np.ndarray): Contact matrix.
        """
        contact = distance_matrix.copy()
        contact[contact != 1] = 0
        return contact

    def quantize_vector(vector: np.ndarray) -> np.ndarray:
        """Quantize a secondary structure vector.

        Convert a vector of predicted brackets into a one-hot vector. For
        instance, `[[0.9, 0.5, 0.1], [0.0, 0.5, 0.1]]` is converted to
        `[[1, 0, 0], [0, 1, 0]]`.

        Args:
            prediction (list-like): Secondary structure prediction.

        Returns: Reformatted secondary structure.
        """
        if (type(vector[0]) in (int, float, np.float16, np.float32)
                or len(vector[0]) == 1):
            return np.ndarray([round(p) for p in vector])
        indices = [n.index(max(n)) if sum(n) else -1 for n in vector]
        sub_indices = list(range(len(vector[0])))
        return np.ndarray(
            [[1 if j == i else 0 for j in sub_indices] for i in indices]
        )

    def quantize(
            matrix: np.ndarray, mask: np.ndarray, threshold: float = None
        ) -> np.ndarray:
        """Eliminate invalid pairings in a secondary structure matrix.

        Let the following represent a secondary structure matrix:

        ```
            [[_, _, _, _, c, b],
             [_, _, _, _, _, a],
             [_, _, _, _, _, _],
             [_, _, _, _, _, _],
             [x, _, _, _, _, _],
             [y, z, _, _, _, _]]
        ```

        It follows that (x, a), (y, b), and (z, c) must all be pairs
        of identical elements because they represent either paired or
        unpaired bases. Differing elements would indicate that a base is
        both paired and unpaired, which is impossible. This function
        assigns the value `0` to all impossible pairings and `1` to all
        other values.

        Steps:
        - Symmetrize the matrix by multiplying it by its transpose.
        - Determine a threshold value from the average of non-paired
          elements.
        - Assign `0` to all the elements below the threshold.
        - Quantize the matrix along both axes and multiply the result
          with each other.

        Args:
            matrix (np.ndarray): Contact matrix.
            mask (np.ndarray): Valid pairing mask.
            threshold (float): Value below which elements are discarded.
                Determined at runtime if not provided.

        Returns (np.ndarray): Folded pairing matrix.
        """
        folded = matrix * matrix.T
        if not threshold:
            inverse_mask = np.ones_like(folded) - mask
            threshold = np.sum(folded * inverse_mask) / np.sum(inverse_mask)
        print("Threshold: ", threshold)
        print("Average: ", np.sum(folded) / np.sum(mask + inverse_mask))
        print("High average: ", np.sum(folded * mask) / np.sum(mask))
        print("Low average: ", np.sum(folded * inverse_mask) / np.sum(inverse_mask))
        folded[folded < threshold] = 0
        rows = np.zeros_like(folded)
        rows[np.arange(folded.shape[0]), np.argmax(folded, axis=0)] = 1
        columns = np.zeros_like(folded)
        columns[np.argmax(folded, axis=0), np.arange(folded.shape[1])] = 1
        return rows * columns

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

    def _pad(vector: np.ndarray, size: int, element: list) -> np.ndarray:
        """Append elements at the right extremity of a vector.

        Args:
            vector (np.ndarray): A vector of elements.
            size (int): The final size of the vector.
            element (list): The element to add to the vector.

        Returns (np.ndarray): The padded list of size "size:.
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

        Args:
            pairings: List of pairings as indices or bracket notations.

        Returns (str): List of elements.
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
