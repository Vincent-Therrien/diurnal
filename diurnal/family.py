"""
    RNA family utility module.

    This module simplifies operations related to the encoding of RNA
    families into other representations.
"""

import inspect

from diurnal.utils import file_io

NAMES = [
    ["5s"],  # 5s ribosomal RNA (rRNA)
    ["16s"], # 16s ribosomal RNA (rRNA)
    ["23s"], # 23s ribosomal RNA (rRNA)
    ["grp1", "group_I_introns"],
    ["grp2", "group_II_introns"],
    ["RNaseP"],
    ["SRP"],
    ["telomerase"],
    ["tmRNA"],
    ["tRNA"]
]

ONEHOT = dict()
for i, name in enumerate(NAMES):
    ONEHOT[name[0]] = [1 if x == i else 0 for x in range(len(NAMES))]


def to_vector(family: str, map: dict = ONEHOT) -> list:
    """Encode a family into a vector.

    Args:
        family (str): RNA family.

    Returns (list(int)): One-hot encoded family.
    """
    if inspect.isfunction(map):
        return map(family)
    if type(map) == dict:
        return map[family]
    message = (f"Type `{type(map)}` is not allowed for family "
        + "encoding. Use a mapping function or dictionary.")
    file_io.log(message, -1)


def from_vector(vector: list) -> str:
    """Convert a one-hot-encoded family back into its name.

    Args:
        vector (list): A one-hot encoded family.

    Returns (str): Family name.
    """
    v = list(vector)
    index = v.index(max(v))
    for family, onehot in ONEHOT.items():
        if onehot[index]:
            return family
    return ""


def get_name(filename: str) -> str:
    """Attempt to determine the family of an RNA molecule based on its
    filename.

    Args:
        filename (str): Name of the file containing the representation
            of the RNA molecule.

    Returns (str): RNA family if found, empty string otherwise.
    """
    candidates = []
    for family in NAMES:
        for alias in family:
            if alias.upper() in filename.upper():
                candidates.append(family[0])
                break
    if not candidates:
        file_io.log(f"Unknown family for `{filename}`.", -1)
        return ""
    if len(candidates) == 1:
        return candidates[0]
    else:
        c = {}
        for candidate in candidates:
            c[filename.upper().find(candidate.upper())] = candidate
        return c[max(c.keys())]
