"""
    RNA family utility module.

    This module simplifies operations related to the encoding of RNA
    families into other representations.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: June 2023
    - License: MIT
"""

from diurnal.utils import log


NAMES = [
    ["5s"],   # 5s ribosomal RNA (rRNA)
    ["16s"],  # 16s ribosomal RNA (rRNA)
    ["23s"],  # 23s ribosomal RNA (rRNA)
    ["grp1", "group_I_introns"],
    ["grp2", "group_II_introns"],
    ["RNaseP"],
    ["SRP"],
    ["telomerase"],
    ["tmRNA"],
    ["tRNA"]
]

NAME_SET = []
for name in NAMES:
    for alias in name:
        NAME_SET.append(alias)

ONEHOT = dict()
for i, name in enumerate(NAMES):
    ONEHOT[name[0]] = [1 if x == i else 0 for x in range(len(NAMES))]


def to_vector(family: str) -> list:
    """Encode a family into a one-hot vector.

    Args:
        family (str): RNA family.

    Returns (list(int)): One-hot encoded family.
    """
    for name in NAMES:
        if family in name:
            return ONEHOT[name[0]]
    return None


def to_name(vector: list) -> str:
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
        log.error(f"Unknown family for `{filename}`.")
        return ""
    if len(candidates) == 1:
        return candidates[0]
    else:
        c = {}
        for candidate in candidates:
            c[filename.upper().find(candidate.upper())] = candidate
        return c[max(c.keys())]


def is_known(family: str) -> bool:
    """Check if a family is recognized.

    Args:
        family (str): Family test name.

    Returns (bool): True if the family is recognized, False otherwise.
    """
    return family in NAME_SET
