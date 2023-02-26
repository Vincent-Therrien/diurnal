import numpy as np
import pathlib

# One-hot encoding dictionary for IUPAC symbols
# See: https://www.bioinformatics.org/sms/iupac.html
IUPAC_ONEHOT = {
    #     A  C  G  U
    "A": [1, 0, 0, 0],
    "C": [0, 1, 0, 0],
    "G": [0, 0, 1, 0],
    "T": [0, 0, 0, 1],
    "U": [0, 0, 0, 1],
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


def get_dot_bracket(pairings: list) -> str:
    sequence = ""
    for i, p in enumerate(pairings):
        if p < 0:
            sequence += "."
        else:
            sequence += "(" if i < p else ")"
    return sequence

def pairings_to_one_hot(pairings: list) -> list:
    return [0 if p < 0 else 1 for p in pairings]

def sequence_to_one_hot(sequence: list) -> list:
    return [IUPAC_ONEHOT[nt] for nt in sequence]
