"""
    Handle RNA structure data files.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: June 2023
    - License: MIT
"""

from diurnal.utils import log


def read_ct_file(path: str) -> tuple:
    """
    Read a CT (Connect table) file and return its information.

    Args:
        path (str): File path of the CT file.

    Returns (tuple):
        The returned tuple contains the following data:
        - RNA molecule title.
        - Primary structure (i.e. a list of 'A', 'C', 'G', and 'U').
        - Pairings (i.e. a list of integers indicating the index of the
            paired based, with `-1` indicating unpaired bases).
    """
    bases = []
    pairings = []
    with open(path) as f:
        header = f.readline()
        if header[0] == ">":
            length = int(header.split()[2])
        else:
            length = int(header.split()[0])

        title = " ".join(header.split()[1:])
        f.seek(0)

        i = 0
        for _, line in enumerate(f):
            # deal w/ header for nth structure
            if i == 0:
                if header[0] == ">":
                    length = int(line.split()[2])
                else:
                    length = int(header.split()[0])

                title = " ".join(line.split()[1:])
                i += 1
                continue

            bn, b, _, _, p, _ = line.split()

            if int(bn) != i:
                #  log.info(f"Skipping CT index {i} in the file `{path}`.")
                bases.append("N")
                pairings.append(-1)
                i += 1
            else:
                bases.append(b)
                pairings.append(int(p) - 1)

            if i == length:
                break

            i += 1

    return title, bases, pairings


def read_ct_file_length(path: str) -> int:
    """Get the size of the sequence written in a CT file.

    Args:
        path (str): File path of the CT file.

    Returns (int): Number of bases in the sequence.
    """
    length = 0
    with open(path) as f:
        header = f.readline()
        if header[0] == ">":
            length = int(header.split()[2])
        else:
            length = int(header.split()[0])
    return length
