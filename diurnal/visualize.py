"""
    Data visualization module.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: April 2023
    - License: MIT
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.ticker import AutoMinorLocator

import diurnal.family
import diurnal.structure
import diurnal.train
import diurnal.database
from diurnal.utils import rna_data


PAIRING_COLORS = {
    "AU":       (0.4, 0.0, 0.0),
    "UA":       (0.7, 0.0, 0.0),
    "CG":       (0.0, 0.4, 0.0),
    "GC":       (0.0, 0.7, 0.0),
    "GU":       (0.0, 0.0, 0.4),
    "UG":       (0.0, 0.0, 0.7),
    "invalid":  (0.5, 0.5, 0.5),  # Impossible pairing (e.g. AA).
    "padding":  (0.85, 0.85, 0.85),  # Padding elements.
    "paired":   (1.0, 1.0, 1.0)  # Paired bases, used for sec. struct.
}
PAIRING_CMAP = [(i, v) for i, v in enumerate(PAIRING_COLORS.values())]


def structure_length_per_family(path: str, max_size: int = None) -> None:
    """Display a histogram of RNA lengths.

    Args:
        path (str): Database file path.
        max_size (int): If provided, reject larger molecules.
    """
    # Read data.
    names = diurnal.database.format_filenames(
        path, size=max_size, randomize=False
    )
    # Obtain lengths.
    families = {}
    for name in names:
        family = diurnal.family.get_name(name)
        if family not in families:
            families[family] = []
        families[family].append(rna_data.read_ct_file_length(name))
    # Plot data
    n_bins = 50
    for family, lengths in families.items():
        plt.hist(
            lengths, n_bins, histtype='bar',
            label=f"{family}, N = {len(lengths)}")
    plt.legend()
    plt.title("Number of Bases in RNA Molecules")
    plt.xlabel('Number of bases')
    plt.ylabel('Count')
    plt.show()


def lengths(data) -> None:
    """Display a histogram of the length of the data."""
    lengths = [len(x) for x in data]
    plt.hist(
        lengths,
        100,
        histtype='bar',
    )
    plt.title(f"Length of RNA Molecules (n = {len(data)})")
    plt.xlabel('Number of bases')
    plt.ylabel('Count')
    plt.show()


def potential_pairings(
        primary: str,
        secondary: list | tuple[list] = None,
        title: str = "RNA Molecule Potential Pairings",
        map: dict = diurnal.structure.Schemes.IUPAC_ONEHOT_PAIRINGS_VECTOR
    ) -> None:
    """Display a heatmap of potential pairings.

    Args:
        primary (str): List of bases.
        secondary (list): Secondary structure or tuple of secondary
            structures represented as contact matrices or lists of
            pairings.
        title (str): Name of the graph.
        map: Potential pairing to string map.
    """
    # Obtain data.
    matrix = diurnal.structure.Primary.to_matrix(primary)
    C = []
    for row in tuple(range(len(matrix))):
        line = []
        for col in tuple(range(len(matrix[0]))):
            element = tuple(matrix[row][col])
            if sum(element) == 0:
                color = PAIRING_COLORS["padding"]
            else:
                color = PAIRING_CMAP[tuple(map.values()).index(element)][1]
            line.append(color)
        C.append(line)
    # Plot the potential pairings.
    plt.imshow(C, interpolation='none')
    patches = [
        mpl.patches.Patch(color=v, label=k) for k, v in PAIRING_COLORS.items()]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2)
    # Plot the primary structure.
    index_base_vertical = [f"{i}: {b}" for i, b in enumerate(primary)]
    plt.yticks(np.arange(0, len(C), 1), index_base_vertical)
    index_base_horizontal = [f"{i}\n{b}" for i, b in enumerate(primary)]
    plt.xticks(np.arange(0, len(C), 1), index_base_horizontal)
    # Plot the secondary structure.
    if not secondary is None:
        if type(secondary) != tuple:
            secondary = (secondary, )
        markers = ["." , "+" , "o" , "v" , "^" , "<", ">"]
        color = PAIRING_COLORS["paired"]
        for sec, marker in zip(secondary, markers):
            if type(sec) == list and type(sec[0]) == int:
                sec = diurnal.structure.Secondary.to_matrix(sec)
            x = []
            y = []
            for i, row in enumerate(sec):
                for j, col in enumerate(row):
                    if col:
                        x.append(j)
                        y.append(i)
            plt.scatter(x, y, color=color, marker=marker, s=40)
            color = tuple([c - 0.1 for c in color])
    minor_locator_x = AutoMinorLocator(2)
    plt.gca().xaxis.set_minor_locator(minor_locator_x)
    minor_locator_y = AutoMinorLocator(2)
    plt.gca().yaxis.set_minor_locator(minor_locator_y)
    plt.grid(which='minor')
    plt.title(title)
    plt.show()


def _add_pairing_element(value) -> int:
    if type(value) is int:
        if value:
            return (0, 0, 0)
        else:
            return (255, 255, 255)
    elif 0.0 <= value <= 1.0:
        v = int((1 - value) * 255)
        return (v, v, v)
    return (0, 0, 0)


def print_contact_matrix(matrix: np.array):
    """Print a contact matrix in the terminal."""
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            print(f"{int(matrix[i][j])}", end="")
        print()


def secondary_structure(
        matrix,
        primary: list = None,
        title: str = "RNA Molecule Pairings"
    ) -> None:
    """Display a heatmap of the secondary structure.

    Args:
        primary (List[str]): Primary structure.
        matrix (List[List[bool]]): Secondary structure.
        title (str)
    """
    C = []
    for row in matrix:
        line = []
        for element in row:
            line.append(_add_pairing_element(element))
        C.append(line)
    plt.imshow(C, interpolation='none')
    if primary:
        index_base = [f"{i}: {b}" for i, b in enumerate(primary)]
        plt.xticks(np.arange(0, len(C), 1), index_base)
        plt.yticks(np.arange(0, len(C), 1), index_base)
    minor_locator_x = AutoMinorLocator(2)
    plt.gca().xaxis.set_minor_locator(minor_locator_x)
    minor_locator_y = AutoMinorLocator(2)
    plt.gca().yaxis.set_minor_locator(minor_locator_y)
    plt.grid(which='minor')
    plt.title(title)
    plt.show()


def heatmap(
        matrices: np.array,
        title: str = "Aggregated heatmaps",
        label: bool = False
    ) -> None:
    """Visualize heatmaps.

    The function opens a plot that visualizes the `matrices` argument.
    If the `matrices` is a 3D array, the heatmap is the sum of all
    arrays along the 0 axis. If `matrices` is a 2D array, it is used as
    the heatmap.

    Args:
        matrices: Set 2D matrices or one 2D matrix.
        title (str): Graph title.
        label (bool): If True, label each axis.
    """
    if len(matrices.shape) == 2:
        total = matrices
        N = 1
    else:
        N = len(matrices)
        total = np.sum(matrices, axis=0)
        total /= np.max(total)
    L = total.shape[0]
    plt.imshow(total, cmap='viridis', interpolation='none')
    plt.colorbar()
    if label:
        index_base = [f"{i}" for i in range(L)]
        plt.xticks(np.arange(0, L, 1), index_base)
        plt.yticks(np.arange(0, L, 1), index_base)
        minor_locator_x = AutoMinorLocator(2)
        plt.gca().xaxis.set_minor_locator(minor_locator_x)
        minor_locator_y = AutoMinorLocator(2)
        plt.gca().yaxis.set_minor_locator(minor_locator_y)
        plt.grid(which='minor')
    if N > 1:
        plt.title(title + f" (n = {N})")
    else:
        plt.title(title)
    plt.show()


def prediction(primary, true, pred) -> None:
    """Compare true and predicted secondary structures."""
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


def primary_structure(primary) -> None:
    """Print the sequence of nucleotides from a one-hot encoded
    primary structure.

    Args:
        primary: Primary structure.
    """
    bases = diurnal.structure.Primary.to_sequence(primary)
    print(f"Primary structure: {''.join(bases)}")
    N = len(bases)
    if N > 10:
        RULER = 10
        ruler = ""
        for i in range(0, N, RULER):
            item = str(i)
            ruler += item
            ruler += " " * (RULER - len(item))
        print(f"      Base number: {ruler}")


def shadow(primary, true, pred) -> None:
    """Compare shadows."""
    primary = diurnal.structure.Primary.to_sequence(primary)
    true = ['0' if x < 0.5 else '1' for x in true]
    pred = ['0' if x < 0.5 else '1' for x in pred]
    differences = ""
    correct = 0
    for i in range(len(primary)):
        if true[i] == pred[i]:
            differences += "_"
            correct += 1
        else:
            differences += "^"
    print(f"Primary structure: {''.join(primary)}")
    print(f"      True shadow: {''.join(true[:len(primary)])}")
    print(f" Predicted shadow: {''.join(pred[:len(primary)])}")
    ratio = str(correct) + "/" + str(len(primary))
    prefix = " " * (7 - len(ratio))
    print(f"{prefix}Matches ({ratio}): {differences}")
