import numpy as np

# One-hot encoding dictionary for IUPAC symbols
# See: https://www.bioinformatics.org/sms/iupac.html
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
IUPAC_MAPPING = {b: np.where(np.array(s) == 1) for b, s in IUPAC_ONEHOT.items()}


def read_ct(path: str, number: int = 0) -> tuple:
    bases = []
    pairings = []
    with open(path) as f:
        # deal w/ header
        header = f.readline()
        if header[0] == ">":
            length = int(header.split()[2])
        else:
            length = int(header.split()[0])

        title = " ".join(header.split()[1:])
        f.seek(0)

        # skip to structure corresponding to number
        start_index = number * length + number
        for _ in range(start_index):
            next(f)

        for i, line in enumerate(f):
            # deal w/ header for nth structure
            if i == 0:
                if header[0] == ">":
                    length = int(line.split()[2])
                else:
                    length = int(header.split()[0])

                title = " ".join(line.split()[1:])
                continue

            bn, b, _, _, p, _ = line.split()

            if int(bn) != i:
                raise NotImplementedError(
                    "Skipping indices in CT files is not supported."
                )

            bases.append(b)
            pairings.append(int(p) - 1)

            if i == length:
                break

    # this shouldn't really ever happen, probs unnecessary check
    if length != len(bases) and length != len(pairings):
        raise RuntimeError("Length of parsed RNA does not match expected length.")

    return title, bases, pairings

def read_seq(path):
    title = None
    sequence = ""
    with open(path) as f:
        for line in f:
            if line[0] == ";":
                continue
            elif title is None:
                title = line.rstrip()
            else:
                sequence += "".join(line.split())

    return title, sequence[:-1]


def get_dot_bracket_from_ct(pairings: list) -> str:
    sequence = ""
    for i, p in enumerate(pairings):
        if p < 0:
            sequence += "."
        else:
            sequence += "(" if i < p else ")"
    return sequence

def get_dot_bracket_from_matrix(pairings: list) -> str:
    sequence = ""
    for p in pairings:
        if p == 0:
            sequence += "."
        else:
            sequence += "(" if p > 0 else ")"
    return sequence

def sequence_to_one_hot(sequence: list) -> list:
    return [IUPAC_ONEHOT[nt] for nt in sequence]

def one_hot_to_sequence(onehot: list) -> list:
    nt = []
    for base in onehot:
        base = list(base)
        for key, code in IUPAC_ONEHOT.items():
            if code == base:
                nt.append(key)
                break
    return nt

def pairings_to_one_hot(pairings: list) -> list:
    sequence = []
    for i, p in enumerate(pairings):
        if p < 0:
            sequence.append(0)
        else:
            sequence.append(1 if i < p else -1)
    return sequence

def one_hot_to_pairing(onehot: list) -> list:
    nt = ""
    for pairing in onehot:
        if pairing == 0:
            nt += "."
        elif pairing == 1:
            nt += "("
        else:
            nt += ")"
    return nt

def pad_one_hot_sequence(sequence: list, total_size: int) -> list:
    for _ in range(total_size - len(sequence)):
        sequence.append([0, 0, 0, 0])
    return sequence

def pad_one_hot_pairing(pairings: list, total_size: int) -> list:
    for _ in range(total_size - len(pairings)):
        pairings.append(0)
    return pairings

def remove_pairing_padding(pairings: list) -> list:
    i = len(pairings) - 1
    while i > 0:
        if pairings[i] not in [0, "."]:
            return pairings[0:i+2]
        i -= 1
    return pairings

def remove_sequence_padding(sequence: list) -> list:
    i = len(sequence) - 1
    while i > 0:
        if type(sequence[i]) == str and sequence[i] == ".":
            return sequence[0:i+1]
        else:
            nonzero = False
            for e in sequence[i]:
                if e != 0:
                    nonzero = True
            if nonzero:
                return sequence[0:i+1]
        i -= 1
    return None

def get_rna_x_y(filename: str, max_size: int) -> tuple:
    _, bases, pairings = read_ct(filename)
    if len(pairings) > max_size:
        return None, None
    x = pad_one_hot_sequence(sequence_to_one_hot(bases), max_size)
    y = pad_one_hot_pairing(pairings_to_one_hot(pairings), max_size)
    return x, y
