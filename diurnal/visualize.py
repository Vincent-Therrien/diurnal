"""
    Data visualization module.

    Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    Affiliation: Département d'informatique, UQÀM
    File creation date: April 2023
    License: MIT
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import diurnal.family
import diurnal.structure


PAIRING_COLORS = {
    "AU":       (0.4, 0.0, 0.0),
    "UA":       (0.7, 0.0, 0.0),
    "CG":       (0.0, 0.4, 0.0),
    "GC":       (0.0, 0.7, 0.0),
    "GU":       (0.0, 0.0, 0.4),
    "UG":       (0.0, 0.0, 0.7),
    "unpaired": (0.5, 0.5, 0.5),     # Unpaired base.
    "invalid":  (0.85, 0.85, 0.85),  # Impossible pairing (e.g. AA).
    "padding":  (0.95, 0.95, 0.95)   # Padding elements.
}
PAIRING_CMAP = [(i, v) for i, v in enumerate(PAIRING_COLORS.values())]


def structure_length_per_family(path: str) -> None:
    """Display a histogram of RNA lengths.

    Args:
        path (str): Directory name of the folder that contains the data
            files.
    """
    # Read data.
    if path[-1] != '/':
        path += '/'
    P = np.load(path + "primary_structures.npy")
    F = np.load(path + "families.npy")
    # Obtain lengths.
    families = {}
    for p, f in zip(P, F):
        family = diurnal.family.to_name(f)
        if family not in families:
            families[family] = []
        bases = diurnal.structure.Primary.to_bases(p)
        families[family].append(len(bases))
    # Plot data
    n_bins = 50
    for family, lengths in families.items():
        plt.hist(
            lengths, n_bins, density=True, histtype='bar',
            label=f"{family}, N = {len(lengths)}")
    plt.legend()
    plt.title("Number of Bases in RNA Molecules")
    plt.xlabel('Number of bases')
    plt.ylabel('Count')
    plt.show()


def potential_pairings(
        matrix: list,
        title: str = "RNA Molecule Potential Pairings",
        map: dict = diurnal.structure.Schemes.IUPAC_ONEHOT_PAIRINGS
        ) -> None:
    """Display a heatmap of potential pairings."""
    # Obtain data.
    matrix = diurnal.structure.Primary.unpad_matrix(matrix)
    C = []
    for row in list(range(len(matrix))):
        line = []
        for col in list(range(len(matrix[0]))):
            element = list(matrix[row][col])
            color = PAIRING_CMAP[list(map.values()).index(element)][1]
            line.append(color)
        C.append(line)
    # Plot the structure.
    plt.imshow(C, interpolation='none')
    patches = [
        mpl.patches.Patch(color=v, label=k) for k, v in PAIRING_COLORS.items()]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2)
    plt.xticks(np.arange(-.5, len(C), 1), np.arange(1, len(C) + 2, 1))
    plt.yticks(np.arange(-.5, len(C), 1), np.arange(1, len(C) + 2, 1))
    plt.grid()
    plt.title(title)
    plt.show()
