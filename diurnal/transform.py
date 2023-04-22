"""
    RNA secondary structure encoding formats.

    Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    Affiliation: Département d'informatique, UQÀM
    File creation date: April 2023
    License: MIT
"""

class PrimaryStructure:
    # One-hot encoding dictionary for IUPAC symbols
    # Reference: https://www.bioinformatics.org/sms/iupac.html
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

    def iupac_to_onehot(bases: list, size: int = 0):
        """
        Convert pairings returned by `diurnal.utils.read_ct_file` into
        a primary structure, e.g. `[[1,0,0,0], [0,1,0,0]]`.
        
        Args:
            bases (list(str)): A list of nucleotides.
            size (int): Output size. `0` for no padding.
        
        Returns (list(list(int))): One-hot encoded primary structure.
        """
        encoding = [PrimaryStructure.IUPAC_ONEHOT[base] for base in bases]
        if len(bases) < size:
            for _ in range(size - len(bases)):
                encoding.append(PrimaryStructure.IUPAC_ONEHOT['.'])
        return encoding

    def onehot_to_iupac(onehot: list):
        """
        Convert one-hot encoding into a sequence of nucleotides.
        
        Args:
            (list(list(int))): One-hot encoded primary structure.
        
        Returns (list(str)): A list of nucleotide.
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
    # One-hot encoding dictionary for secondary structure bracket notation.
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

    def pairings_to_bracket(pairings: list, size: int):
        """
        Convert pairings returned by `diurnal.utils.read_ct_file` into
        a bracket notation, e.g. `(((...)))`.
        
        Args:
            pairings (list(int)): A list of nucleotide pairings.
            size (int): Output size. `0` for no padding.
        
        Returns (str): Secondary structure bracket notation.
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
    
    def pairings_to_shadow(pairings: list, size: int):
        """
        Convert pairings returned by `diurnal.utils.read_ct_file` into
        a secondary structure, e.g. `111000111`.
        
        Args:
            pairings (list(int)): A list of nucleotide pairings.
            size (int): Output size. `0` for no padding.
        
        Returns (str): Secondary structure shadow.
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
    
    def pairings_to_onehot(pairings: list, size: int = 0):
        """
        Convert pairings returned by `diurnal.utils.read_ct_file` into
        a secondary structure, e.g. `[[1,0,0], [0,1,0], [0,0,1]]`.
        
        Args:
            pairings (list(int)): A list of nucleotide pairings.
            size (int): Output size. `0` for no padding.
        
        Returns (list(list(int))): One-hot encoded secondary structure.
        """
        encoding = []
        for i, p in enumerate(pairings):
            if p < 0:
                encoding.append(SecondaryStructure.BRACKET_ONEHOT['.'])
            elif i < p:
                encoding.append(SecondaryStructure.BRACKET_ONEHOT['('])
            else:
                encoding.append(SecondaryStructure.BRACKET_ONEHOT[')'])
        # Add padding
        if len(pairings) < size:
            for _ in range(size - len(pairings)):
                encoding.append(SecondaryStructure.BRACKET_ONEHOT[' '])
        return encoding
    
    def onehot_to_bracket(onehot):
        """
        Convert a one-hot-encoded pairing sequence into bracket
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
    
    def remove_onehot_padding(sequence):
        """
        Remove zero-valued elements at the end of a one-hot encoded
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
    
    def remove_bracket_padding(sequence):
        """
        Remove zero-valued elements at the end of a bracket-notation
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

class Family:
    # One-hot encoding for RNA families.
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

    def onehot(family: str):
        """
        Encode a family into a vector.
        
        Args:
            family (str): RNA family.
        
            Returns (list(int)): One-hot encoded family.
        """
        return Family.ONEHOT[family]
