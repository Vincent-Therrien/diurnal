"""
    Transform RNA structures into useful representations.

    Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    Affiliation: Département d'informatique, UQÀM
    File creation date: June 2023
    License: MIT
"""


import inspect

from .utils import file_io


class Schemes:
    """RNA structural codes to transform data into other representations.

    Attributes:
        IUPAC_TO_ONEHOT (dict): One-hot encoding dictionary for IUPAC
            symbols. Reference: https://www.bioinformatics.org/sms/iupac.html
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
        ".": [0, 0, 0, 0],
        "-": [0, 0, 0, 0],
    }

    IUPAC_ONEHOT_PAIRINGS = {
        "AU": [1, 0, 0, 0, 0, 0],
        "UA": [0, 1, 0, 0, 0, 0],
        "CG": [0, 0, 1, 0, 0, 0],
        "GC": [0, 0, 0, 1, 0, 0],
        "GU": [0, 0, 0, 0, 1, 0],
        "UG": [0, 0, 0, 0, 0, 1],
        "-":  [0, 0, 0, 0, 0, 0]
    }

    BRACKET_TO_ONEHOT = {
        "(": [1, 0, 0],
        ".": [0, 1, 0],
        ")": [0, 0, 1],
        " ": [0, 0, 0],
    }

    SHADOW_ENCODING = {
        "(" : 1,
        "." : 0,
        ")" : 1,
        " " : 0
    }


class Primary:
    """Transform RNA primary structures into useful formats."""

    def to_vector(bases, map=Schemes.IUPAC_TO_ONEHOT) -> list:
        """Transform a sequence of bases into a vector.

        Args:
            bases (list): A sequence of bases. E.g.: ``['A', 'U']``.
            map: A dictionary or function that maps bases to vectors.

        Returns (list): One-hot encoded primary structure.
        """
        if inspect.isfunction(map):
            return map(bases)
        if type(map) == dict:
            return [map[base] for base in bases]
        message = (f"Type `{type(map)}` is not allowed for primary structure "
            + "encoding. Use a mapping function or dictionary.")
        file_io.log(message, -1)

    def from_vector(vector, map=Schemes.IUPAC_TO_ONEHOT) -> list:
        """Transform a vector into a sequence of bases.

        Args:
            vector (list-like): One-hot encoded primary structure.
            map: A dictionary or function that maps bases to vectors.

        Returns (list): A sequence of bases. E.g.: ``['A', 'U']``.
        """
        if inspect.isfunction(map):
            return map(vector)
        bases = []
        for base in vector:
            base = list(base)
            for key, code in map:
                if code == base:
                    bases.append(key)
                    break
        return bases

    def pad(vector: list, size: int,
            element: list = Schemes.IUPAC_TO_ONEHOT['.']) -> list:
        """Append elements at the right extremity of a vector.

        Args:
            vector (list): A vector of elements.
            size (int): The final size of the vector.
            element (list): The element to add to the vector.

        Returns (list): The padded list of size "size".
        """
        difference = size - len(vector)
        if difference > 0:
            return vector + difference * [element]
        return vector

    def to_padded_vector(bases, size: int, map=Schemes.IUPAC_TO_ONEHOT) -> list:
        """Transform a sequence of bases into a vector of a specific size.

        Args:
            vector (list): A vector of elements.
            size (int): The final size of the vector.
            map: A dictionary or function that maps bases to vectors.

        Returns (list): The padded list of size "size".
        """
        return Primary.pad(Primary.to_vector(bases, map), size)


class Secondary:
    """Transform RNA secondary structures into useful formats."""

    def to_bracket(pairings: list) -> list:
        """Convert a list of nucleotide pairings into a secondary
        structure bracket notation, e.g. `'(((...)))`.'

        Args:
            pairings (list(int)): A list of nucleotide pairings, e.g.
                the pairing `(((...)))` is represented as
                `[8, 7, 6, -1, -1, -1, 2, 1, 0]`.

        Returns (list): Secondary structure bracket notation.
        """
        encoding = []
        for i, p in enumerate(pairings):
            if p < 0:
                encoding.append('.')
            elif i < p:
                encoding.append('(')
            else:
                encoding.append(')')
        return encoding

    def to_vector(pairings: list, map=Schemes.BRACKET_TO_ONEHOT) -> list:
        """Encode pairings in a one-hot bracket-based secondary structure.

        Args:
            pairings (list(int)): A list of nucleotide pairings, e.g.
                the pairing `(((...)))` is represented as
                `[8, 7, 6, -1, -1, -1, 2, 1, 0]`.

        Returns (list): One-hot encoded secondary structure.
        """
        bracket = Secondary.to_bracket(pairings)
        if inspect.isfunction(map):
            return map(bracket)
        elif type(map) == dict:
            return [map[symbol] for symbol in bracket]
        message = (f"Type `{type(map)}` is not allowed for secondary structure "
            + "encoding. Use a mapping function or dictionary.")
        file_io.log(message, -1)

    def from_vector_bracket(vector: list) -> list:
        """Convert a one-hot-encoded pairing sequence into bracket
        notation, e.g. `(((...)))`.

        Args:
            onehot (list-like): A one-hot encoded secondary structure,
                e.g. `[[1,0,0], [0,1,0], [0,0,1]]`

        Returns (list): A bracket notation secondary structure,
            e.g. `['(', '(', '(', '.', '.', '.', ')', ')', ')']`
        """
        # TODO: Generalize to other mappings
        values = [n.index(max(n)) for n in vector]
        encoding = []
        characters = list(Schemes.BRACKET_TO_ONEHOT.keys())
        for value in values:
            encoding.append(characters[value])
        return encoding

    def pad(vector: list, size: int,
            element: str = Schemes.BRACKET_TO_ONEHOT[' ']) -> list:
        """Append elements at the right extremity of a vector.

        Args:
            vector (list): A vector of elements.
            size (int): The final size of the vector.
            element (list): The element to add to the vector.

        Returns (list): The padded list of size "size:.
        """
        difference = size - len(vector)
        if difference > 0:
            if type(vector) == str:
                return vector + difference * element
            else:
                return vector + difference * [element]
        return vector

    def to_padded_vector(pairings, size: int,
            map = Schemes.BRACKET_TO_ONEHOT) -> list:
        """Encode pairings in a one-hot bracket-based secondary structure.

        Args:
            pairings (list(int)): A list of nucleotide pairings
            size (int): The final size of the vector.
            map: A dictionary or function that maps bases to vectors.

        Returns (list): The padded list of size "size".
        """
        return Secondary.pad(Secondary.to_vector(pairings, map), size)


def to_matrix(bases: list, pairings: list,
        map: dict = Schemes.IUPAC_ONEHOT_PAIRINGS) -> list:
    """Convert a list of nucleotide pairings into a 2D anti-diagonal
    matrix of one-hot-encoded pairing.

    For instance, the molecule `AAACCUUU` with secondary structure
    `(((...)))` will be represented as:

        [0 0 0 0 0 0 0 0 y]
        [0 0 0 0 0 0 0 y 0]
        [0 0 0 0 0 0 y 0 0]
        [0 0 0 0 0 0 0 0 0]
        [0 0 0 0 0 0 0 0 0]
        [0 0 0 0 0 0 0 0 0]
        [0 0 x 0 0 0 0 0 0]
        [0 x 0 0 0 0 0 0 0]
        [x 0 0 0 0 0 0 0 0]

    where `x` is the vector `[1 0 0 0 0 0]`, a one-hot encoded
    representation of the `AU`, and `y` is the vector
    `[0 1 0 0 0 0]`, a one-hot encoded representation of `UA`.

    Args:
        bases (list(str)): A list of nucleotides, i.e. primary structure
        pairings (list(int)): A list of nucleotide pairings, i.e. the
            secondary structure.
        map (dict): Assign pairs of symbols to vectors.

    Returns (str): RNA structure.
    """
    # Obtain the list of bonds (e.g.: AU, AU, ...)
    encoding = []
    empty = map['-']
    for i, p in enumerate(pairings):
        if p < 0:
            encoding.append(empty)
        else:
            bonds = bases[i] + bases[p]
            encoding.append(map[bonds])
    # Convert the list of bonds into a 2D anti-diagonal matrix.
    size = len(encoding)
    matrix = [[empty for _ in range(size)] for _ in range(size)]
    for i in range(size):
        matrix[size - i - 1][i] = encoding[i]
    return matrix


def pad_matrix(matrix: list, size: int,
        map: dict = Schemes.IUPAC_ONEHOT_PAIRINGS) -> list:
    """Add neutral elements to structure matrix to fit ``size``.

    Args:
        matrix (list): Matrix obtained from ``to_matrix()``.
        size (int): Dimension of the matrix to obtain.
        map (dict): Assign pairs of symbols to vectors.

    Returns (list): Padded matrix.
    """
    original_size = len(matrix)
    if original_size >= size:
        file_io.log(f"Matrix of size {original_size} cannot be padded "
            + f"to size {size}")
        return matrix
    # Append elements on the right side.
    append_size = size - original_size
    for i in range(original_size):
        matrix[i] = matrix[i] + append_size * map['-']
    # Append elements at the bottom of the matrix.
    for i in range(append_size):
        matrix.append(size * map['-'])


def from_matrix(matrix: list, map: dict = Schemes.IUPAC_ONEHOT_PAIRINGS
        ) -> tuple:
    """Convert a structural matrix to primary and secondary structures.

    Args:
        matrix (list): RNA structure represented as a matrix.
        map (dict): Assign a vector to a type of bond.

    Returns (tuple): Result arranged as
        (primary structure (list), secondary structure (list))
    """
    pass
