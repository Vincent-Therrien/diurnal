"""
    Data visualization module.

    Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    Affiliation: Département d'informatique, UQÀM
    File creation date: April 2023
    License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt

from diurnal import transform

def count_structures_per_family(path: str) -> None:
    """
    Display a histogram of RNA lengths.

    Args:
        path (str): Directory name of the folder that contains the data
            files.
    """
    # Read data.
    if path[-1] != '/': path += '/'
    S = np.load(path + "secondary_structures.npy")
    F = np.load(path + "families.npy")
    # Obtain lengths.
    families = {}
    for s, f in zip(S, F):
        family = transform.Family.onehot_to_family(f)
        if family not in families:
            families[family] = []
        L = len(transform.SecondaryStructure.remove_onehot_padding(s))
        families[family].append(L)
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
