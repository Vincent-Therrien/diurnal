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


_NAMES = [
    ["5s"],   # 5s ribosomal RNA (rRNA)
    ["16s"],  # 16s ribosomal RNA (rRNA)
    ["23s"],  # 23s ribosomal RNA (rRNA)
    ["grp1", "group_I_introns", "group_I_intron"],
    ["grp2", "group_II_introns", "group_II_intron"],
    ["RNaseP"],  # Ribonuclease P
    ["SRP"],  # Signal Recognition Particle RNA
    ["telomerase"],
    ["tmRNA"],  # Transfer-messenger RNA
    ["tRNA"]
]

NAMES = [name[0] for name in _NAMES]

_NAME_SET = []
for name in _NAMES:
    for alias in name:
        _NAME_SET.append(alias)

ONEHOT = dict()
for i, name in enumerate(_NAMES):
    ONEHOT[name[0]] = [1 if x == i else 0 for x in range(len(_NAMES))]


def is_known(family: str) -> bool:
    """Check if an RNA family is recognized.

    Args:
        family (str): Family test name.

    Returns (bool): True if the family is recognized, False otherwise.
    """
    return family in _NAME_SET


def all_but(families: list[str]) -> bool:
    """Return all RNA family names except those provided as arguments.

    Args:
        families (List(str) | str): RNA families to exclude.

    Returns (List(str)): The list of selected RNA families.
    """
    if type(families) is str:
        families = [families]
    excluded_families = []
    for family in families:
        for name in _NAMES:
            if family in name:
                excluded_families.append(name[0])
    excluded_families = set(excluded_families)
    selected_families = []
    for name in _NAMES:
        if name[0] not in excluded_families:
            selected_families.append(name[0])
    return selected_families


def to_onehot(family: str, map: dict = ONEHOT) -> list:
    """Encode a family into a one-hot vector.

    Args:
        family (str): RNA family.
        map (dict): A dictionary that assigns a family to a vector.

    Returns (list(int)): One-hot encoded family.
    """
    for name in _NAMES:
        if family in name:
            return map[name[0]]
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
    for family in _NAMES:
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


def select(names: list[str], families: str | list[str]) -> list[str]:
    """Return a list of molecule names that belong to a provided family.

    Args:
        names (list[str]): List of names to filter.
        families (str | list[str]): Family or families to preserve.

    Returns (list[str]) List of names.
    """
    if type(families) == str:
        families = [families]
    return [n for n in names if get_name(n) in families]
