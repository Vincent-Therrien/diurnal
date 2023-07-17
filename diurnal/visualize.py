"""
    Data visualization module.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: April 2023
    - License: MIT
"""

import shutil
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx

import diurnal.family
import diurnal.structure
import diurnal.train


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
        bases = diurnal.structure.Primary.to_sequence(p)
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
        map: dict = diurnal.structure.Schemes.IUPAC_ONEHOT_PAIRINGS_VECTOR
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


def prediction(primary, true, pred) -> None:
    """Compare true and predicted secondary structures."""
    columns, rows = shutil.get_terminal_size()
    primary, true, pred = diurnal.train.clean_vectors(primary, true, pred)
    true = diurnal.structure.Secondary.to_bracket(true)
    pred = diurnal.structure.Secondary.to_bracket(pred)
    differences = ""
    correct = 0
    for i in range(len(primary)):
        if true[i] == pred[i]:
            differences += "_"
            correct += 1
        else:
            differences += "^"
    print(f"       Primary structure: {''.join(primary)}")
    print(f"True secondary structure: {''.join(true[:len(primary)])}")
    print(f"Predicted sec. structure: {''.join(pred[:len(primary)])}")
    ratio = str(correct) + "/" + str(len(primary))
    prefix = " " * (14 - len(ratio))
    print(f"{prefix}Matches ({ratio}): {differences}")


def secondary_structure(pairings) -> None:
    """Plot the secondary structure."""
    G = nx.Graph()
    for i in range(len(pairings) - 1):
        G.add_edge(i, i + 1)
    for i, p in enumerate(pairings):
        if p >= 0:
            G.add_edge(i, p)
    pos = nx.spiral_layout(G)
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, edge_color='r', arrows=True)
    nx.draw_networkx_edges(G, pos, arrows=False)
    plt.show()
