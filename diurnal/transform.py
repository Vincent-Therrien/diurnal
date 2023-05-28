"""
    RNA secondary structure encoding formats.

    Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    Affiliation: Département d'informatique, UQÀM
    File creation date: April 2023
    License: MIT
"""

class PrimaryStructure:
    """A utility class designed to transform RNA primary structures into
    useful formats.

    Attributes:
        IUPAC_ONEHOT (dict): One-hot encoding dictionary for IUPAC
            symbols. Reference: https://www.bioinformatics.org/sms/iupac.html
    """
    IUPAC_ONEHOT = {
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

    def iupac_to_onehot(bases, size: int = 0) -> list:
        """Convert a list of IUPAC symbols into a one-hot encoded list
        to represent the primary structure of an RNA molecule.

        Args:
            bases: A sequence of nucleotides, e.g. `['A', 'U']` or
                `'AU'`.
            size (int): Output size. `0` for no padding.

        Returns (list(list(int))): One-hot encoded primary structure,
            e.g. `[[1,0,0,0], [0,1,0,0]]`. Null elements are appended at
            the right end if the size is greater than the length of the
            `bases` argument.
        """
        encoding = [PrimaryStructure.IUPAC_ONEHOT[base] for base in bases]
        if len(bases) < size:
            for _ in range(size - len(bases)):
                encoding.append(PrimaryStructure.IUPAC_ONEHOT['.'])
        return encoding

    def onehot_to_iupac(onehot: list) -> list:
        """Convert one-hot primary structure encoding into a sequence of
        nucleotides.

        Args:
            onehot (list(list(int))): One-hot encoded primary structure.

        Returns (list(str)): A list of nucleotides encoded with IUPAC
            symbols.
        """
        nt = []
        for base in onehot:
            base = list(base)
            for key, code in PrimaryStructure.IUPAC_ONEHOT.items():
                if code == base:
                    nt.append(key)
                    break
        i = len(nt) - 1
        while i > 0:
            if nt[i] != ".":
                break
            i -= 1
        return nt[:i]

class SecondaryStructure:
    """A utility class designed to transform RNA secondary structures
    into useful formats.

    Attributes:
        BRACKET_ONEHOT (dict): One-hot encoding dictionary for a
            secondary structure that relies on the bracket notation.
        SHADOW_ENCODING (dict): One-hot encoding dictionary to encode
            the shadow of the secondary structure (i.e. the symbols `(`
            and `)` of the bracket notation are considered identical).
    """
    BRACKET_ONEHOT = {
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

    def pairings_to_bracket(pairings: list, size: int) -> str:
        """Convert a list of nucleotide pairings into a secondary
        structure bracket notation, e.g. `'(((...)))`.'
        
        Args:
            pairings (list(int)): A list of nucleotide pairings, e.g.
                the pairing `(((...)))` is represented as
                `[8, 7, 6, -1, -1, -1, 2, 1, 0]`.
            size (int): Output size. `0` for no padding.
        
        Returns (str): Secondary structure bracket notation. Null
            elements are appended at the right end of the vector if the
            provided size is greater than the length of the pairings.
        """
        encoding = ""
        for i, p in enumerate(pairings):
            if p < 0:
                encoding += '.'
            elif i < p:
                encoding += '('
            else:
                encoding += ')'
        if len(pairings) < size:
            encoding += ' ' * (size - len(pairings))
        return encoding

    def pairings_to_shadow(pairings: list, size: int) -> list:
        """Convert a list of nucleotide pairings into a secondary
        structure shadow, e.g. `'111000111'`.

        Args:
            pairings (list(int)): A list of nucleotide pairings, e.g.
                the pairing `(((...)))` is represented as
                `[8, 7, 6, -1, -1, -1, 2, 1, 0]`.
            size (int): Output size. `0` for no padding.
        
        Returns (list(int)): Secondary structure shadow. `0` elements
            are appended at the right end of the vector if the provided
            size is greater than the length of the pairings.
        """
        encoding = []
        for p in pairings:
            if p < 0:
                encoding.append(SecondaryStructure.SHADOW_ENCODING['.'])
            else:
                encoding.append(SecondaryStructure.SHADOW_ENCODING['('])
        if len(pairings) < size:
            for _ in range(size - len(pairings)):
                encoding.append(SecondaryStructure.SHADOW_ENCODING[' '])
        return encoding
    
    def pairings_to_onehot(pairings: list, size: int) -> list:
        """Convert a list of nucleotide pairings into a one-hot encoded
        secondary structure, e.g. `[[1,0,0], [0,1,0], [0,0,1]]`.

        Args:
            pairings (list(int)): A list of nucleotide pairings, e.g.
                the pairing `(((...)))` is represented as
                `[8, 7, 6, -1, -1, -1, 2, 1, 0]`.
            size (int): Output size. `0` for no padding.

        Returns (list): One-hot encoded secondary structure. Null
            elements are appended at the right end of the vector if the
            provided size is greater than the length of the pairings.
        """
        bracket = SecondaryStructure.pairings_to_bracket(pairings, size)
        encoding = []
        for b in bracket:
            encoding.append(SecondaryStructure.BRACKET_ONEHOT[b])
        return encoding

    def onehot_to_bracket(onehot: list) -> str:
        """Convert a one-hot-encoded pairing sequence into bracket
        notation, e.g. `(((...)))`.

        Args:
            onehot (list-like): A one-hot encoded secondary structure,
                e.g. `[[1,0,0], [0,1,0], [0,0,1]]`

        Returns (str): A bracket notation secondary structure,
            e.g. `(((...)))`
        """
        values = [n.index(max(n)) for n in onehot]
        encoding = ""
        characters = list(SecondaryStructure.BRACKET_ONEHOT.keys())
        for value in values:
            encoding += characters[value]
        return encoding
    
    def remove_onehot_padding(sequence: list) -> list:
        """Remove zero-valued elements at the end of a one-hot encoded
        secondary structure.

        Args:
            sequence (list-like): One-hot encoded secondary structure.
        """
        i = len(sequence) - 1
        while i > 0:
            nonzero = False
            for e in sequence[i]:
                if e != 0:
                    nonzero = True
            if nonzero:
                return sequence[0:i+1]
            i -= 1
        return None
    
    def remove_bracket_padding(sequence: str) -> str:
        """Remove zero-valued elements at the end of a bracket-notation
        encoded secondary structure.

        Args:
            sequence (str): Bracket-encoded secondary structure.
        """
        i = len(sequence) - 1
        while i > 0:
            if sequence[i] != " ":
                return sequence[0:i+1]
            i -= 1
        return None

class Structure:
    """Representations that combine the primary and secondary structures
    into a single data structure.

    Attributes:
        IUPAC_ONEHOT_PAIRINGS (dict): One-hot encoded nucleotide
            pairings, including normal ones (AU, UA, CG, and GC) and
            wobble pairs (GU and UG).
    """
    IUPAC_ONEHOT_PAIRINGS = {
        "AU": [1, 0, 0, 0, 0, 0],
        "UA": [0, 1, 0, 0, 0, 0],
        "CG": [0, 0, 1, 0, 0, 0],
        "GC": [0, 0, 0, 1, 0, 0],
        "GU": [0, 0, 0, 0, 1, 0],
        "UG": [0, 0, 0, 0, 0, 1],
        "-":  [0, 0, 0, 0, 0, 0]
    }

    def structure_to_2D_matrix(bases: list, pairings: list, size: int) -> list:
        """Convert a list of nucleotide pairings into a 2D anti-diagonal
        matrix of one-hot-encoded pairing. For instance, the molecule
        `AAACCUUU` with secondary structure `(((...)))` will be
        represented as:

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
            bases (list(str)): A list of nucleotides.
            pairings (list(int)): A list of nucleotide pairings.
            size (int): Output size. `0` for no padding.

        Returns (str): RNA structure.
        """
        # Obtain the list of bonds (e.g.: AU, AU, ...)
        encoding = []
        empty = Structure.IUPAC_ONEHOT_PAIRINGS['-']
        for i, p in enumerate(pairings):
            if p < 0:
                encoding.append(empty)
            else:
                bonds = bases[i] + bases[p]
                encoding.append(Structure.IUPAC_ONEHOT_PAIRINGS[bonds])
        if len(pairings) < size:
            encoding += empty * (size - len(pairings))
        # Convert the list of bonds into a 2D anti-diagonal matrix.
        matrix = [[empty for _ in range(size)] for _ in range(size)]
        for i in range(size):
            matrix[size - i - 1][i] = encoding[i]
        return matrix

class Family:
    # One-hot encoding for RNA families.
    """RNA family encoding utility class.

    Attributes:
        ONEHOT (dict): Map an RNA family to a one-hot vector.
    """
    ONEHOT = {
        "5s"         : [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "16s"        : [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        "23s"        : [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        "grp1"       : [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        "grp2"       : [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        "RNaseP"     : [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        "SRP"        : [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        "telomerase" : [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        "tmRNA"      : [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        "tRNA"       : [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    }

    def onehot(family: str) -> list:
        """Encode a family into a vector.

        Args:
            family (str): RNA family.

            Returns (list(int)): One-hot encoded family.
        """
        return Family.ONEHOT[family]

    def onehot_to_family(vector: list) -> str:
        """Convert a one-hot-encoded family back into its name.

        Args:
            vector (list): A one-hot encoded family.

        Returns (str): Family name.
        """
        v = list(vector)
        index = v.index(max(v))
        for family, onehot in Family.ONEHOT.items():
            if onehot[index]:
                return family
        return ""
