"""
    Data visualization module.

    Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    Affiliation: Département d'informatique, UQÀM
    File creation date: April 2023
    License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt

import diurnal.family
import diurnal.structure


def count_structures_per_family(path: str) -> None:
    """Display a histogram of RNA lengths.

    Args:
        path (str): Directory name of the folder that contains the data
            files.
    """
    # Read data.
    if path[-1] != '/': path += '/'
    P = np.load(path + "primary_structures.npy")
    F = np.load(path + "families.npy")
    # Obtain lengths.
    families = {}
    for p, f in zip(P, F):
        family = diurnal.family.from_vector(f)
        if family not in families:
            families[family] = []
        bases = diurnal.structure.Primary.to_bases(p)
        families[family].append(len(bases))
    # Plot data
    n_bins = 50
    for family, lengths in families.items():
        plt.hist(lengths, n_bins, density=True, histtype='bar',
            label=f"{family}, N = {len(lengths)}")
    plt.legend()
    plt.title("Number of Bases in RNA Molecules")
    plt.xlabel('Number of bases')
    plt.ylabel('Count')
    plt.show()


def visualize_potential_pairings(matrix: list,
        map: dict=diurnal.structure.Schemes.IUPAC_ONEHOT_PAIRINGS) -> None:
    """Display a heatmap of potential pairings."""
    x = list(range(len(matrix[0])))
    y = list(range(len(matrix)))
    C = []
    for row in y[::-1]:
        line = []
        for col in x:
            element = matrix[row][col]
            if sum(element) == 0:
                line.append(0)
            else:
                line.append(element.index(max(element)) + 1)
        C.append(line)
    plt.pcolormesh(x, y, C)
    plt.show()
