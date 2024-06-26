"""
    RNA secondary structure training utility module.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: April 2023
    - License: MIT
"""

from random import shuffle
import numpy as np
import torch
import os.path

from diurnal.utils import log
import diurnal.family
import diurnal.structure


def split(data, fractions: tuple[float], offset: int = 0) -> list:
    """Split an array of data.

    Args:
        data (any): Array-like data to split.
        fractions (tuple[float]): Fraction of data in each resulting
            set. Elements must sum to 1.
        offset (int): Index offset.

    Returns (list[any]): List of split sets.

    Example:

    >>> data = [0, 1, 2, 3, 4, 5, 6, 8, 9]
    >>> split(data, (0.2, 0.8), 1)
    [[1, 2], [3, 4, 5, 6, 7, 8, 9, 0]]
    """
    assert abs(sum(fractions) - 1) < 0.01, f"`{fractions}` does not sum to 1."
    subarrays = []
    index = int(offset)
    n = len(data)
    for f in fractions:
        # Determine the indices to use.
        index2 = (index + int(n*f))
        if index2 > n:
            index2 %= n
        # Make the subarray according to the indices.
        if index < index2:
            subarrays.append(data[index:index2])
        else:
            tmp = data[index:]
            tmp += data[:index2]
            subarrays.append(tmp)
        index = index2
    return subarrays


def _check_homogeneity(data) -> None:
    """Ensure that the data are of the same length."""
    try:
        lengths = [len(d) for d in data]
    except:
        return
    if len(set(lengths)) != 1:
        log.error(f"Inhomogeneous data: {lengths}")
        raise RuntimeError


def split_data(data, fractions: list, offset: int = 0) -> list:
    """Split data in subsets according to the specified fractions.

    Args:
        data: Array-like object containing the data to split.
        fractions: Proportion of each subset. For instance, to use 80%
            of the data for training and 20% for testing, use [0.8, 0.2].
        offset: Number of indices to offset to assemble the subsets.
            Used for K-fold data splits.

    Returns:
        A list containing the split data object.
    """
    if sum(fractions) != 1.0:
        log.error("Invalid data split proportions.")
        raise RuntimeError
    if type(data) == dict:
        keys = list(data.keys())
        values = []
        length = max([len(data[k]) for k in keys])
        for i in range(length):
            element = []
            for k in keys:
                if len(data[k]) == 1:
                    element.append(data[k][0][i])
                else:
                    element.append(data[k][i])
            values.append(element)
        split_values = split(values, fractions, offset)
        new_data = []
        for split_value in split_values:
            d = {}
            for i, k in enumerate(keys):
                d[k] = [element[i] for element in split_value]
                if k == "input":
                    d[k] = (d[k], )
            new_data.append(d)
        return new_data
    else:
        _check_homogeneity(data)
        return split(data, fractions, offset)


def split_indices(fractions: list, n: int) -> list:
    """Split a range of indices in subsets according to the specified
    fractions.

    Args:
        fractions: Proportion of each subset. For instance, to use 80% of the
            data for training and 20% for testing, use [0.8, 0.2].
        n: Number of indices.

    Returns (list): A list containing the split data object.
    """
    return split_data(list(range(n)), fractions)


def k_fold_split(data, fractions: list, k: int, i: int) -> list:
    """Split the data to make a K-fold split.

    Args:
        data: Array-like object containing the data to split.
        fractions: Proportion of each subset. For instance, to use 80% of the
            data for training and 20% for testing, use [0.8, 0.2].
        k: Number of folds.
        i: Zero-based index of the fold.

    Returns:
        A tuple containing the split data object.
    """
    if k <= 0 or i >= k:
        raise RuntimeError("Invalid K-fold parameters.")
    offset = (len(data) / k) * i
    return split_data(data, fractions, offset)


def k_fold_indices(fractions: list, k: int, n: int) -> list:
    """Return tuples of indices for K-fold splits.

    Args:
        fractions: Proportion of each subset. For instance, to use 80% of the
            data for training and 20% for testing, use [0.8, 0.2].
        k: Number of folds.
        n: Number of indices.

    Returns (list): `k` tuples containing `len(fractions)` of index lists.
    """
    if k <= 0 or k > n:
        raise RuntimeError("Invalid K-fold parameters.")
    indices = list(range(n))
    shuffle(indices)
    k_folds = []
    for i in range(k):
        offset = (n / k) * i
        k_folds.append(offset)
    return k_folds


def shuffle_data(*args) -> tuple:
    """Shuffle vectors to preserve one-to-one original pairings.

    For instance, consider
    - a = [ 0,   1,   2 ]
    - b = ['a', 'b', 'c']
    Shuffling lists `a` and `b` may result in:
    - a = [ 2,   1,   0 ]
    - b = ['c', 'b', 'a']

    Args:
        args: List-like elements to be shuffled. They need to be of the
            same dimensions.

    Returns (tuple): Shuffled data. The vector are returned in the same
        order as they were provided.
    """
    # Parameter validation
    lengths = []
    for a in args:
        lengths.append(len(a))
    if lengths.count(lengths[0]) != len(lengths):
        log.error("Shuffled data are not homogeneous.")
        raise RuntimeError
    # Shuffling
    tmp = list(zip(*args))
    shuffle(tmp)
    args = zip(*tmp)
    return args


# Training functions
def _read_formatted_data(path: str) -> tuple:
    """Read an RNA dataset encoded into .npy Numpy files.

    Args:
        path (str): Directory path of the folder containing the dataset
            files.

    Returns (tuple(list)):  List-converted file content in the following
        order: (primary structure, secondary structure, families, names)
    """
    if path[-1] != '/':
        path += '/'
    # Load data.
    relative_paths = [
        "primary_structures.npy",
        "secondary_structures.npy",
        "names.txt",
        "families.txt"
    ]
    data = []
    for p in relative_paths:
        if p.endswith(".npy"):
            array = np.load(path + p) if os.path.isfile(path + p) else None
            if array.any():
                data.append(array)
        elif p.endswith(".txt"):
            f = open(path + p, "r")
            names = f.read().split('\n')
            data.append(names)
    # Validate the loaded data.
    if data:
        lengths = [len(d) for d in data]
        if lengths.count(lengths[0]) != len(lengths):
            log.error(f"load_data: Inhomogeneous sequences: {lengths}")
            raise RuntimeError
    return data


def _convert_data_to_dict(data: list) -> dict:
    """Format loaded data into a dictionary.

    Args:
        data (list): Input data.

    Return (dict): Input data represented in a labelled dictionary.
    """
    if len(data) < 4 or len(data[0]) < 1:
        return {
            "input": tuple(),
            "output": [],
            "names": [],
            "families": []
        }
    return {
        "input": tuple(data[:-3]),
        "output": data[-3],
        "names": data[-2],
        "families": data[-1]
    }


def load_data(path: str, randomize: bool = True) -> tuple:
    """Read formatted data into tensors.

    Args:
        path (str): Name of the directory that contains the Numpy files
            written by the function `diurnal.database.format`.
        randomize (bool): Randomize data if set to True.

    Returns:
        list: Loaded data represented as
            [primary structure, secondary structure, family].
    """
    data = _read_formatted_data(path)
    # Shuffle data.
    if randomize:
        data = list(shuffle_data(*data))
    return _convert_data_to_dict(data)


def load_families(
        path: str, families: list, randomize = True,
        verbose: bool = True) -> list:
    """Read formatted molecules of the specified RNA family.

    Args:
        path (str): Name of the directory that contains the Numpy files
            written by the function `diurnal.database.format`.
        families (List(str) | str): Families to read.
        randomize (bool): Randomize data if set to True.
        verbose (bool): Print informative messages.

    Returns (dict): Loaded data represented as
        `{
            "input": tuple[list],
            "secondary": list,
            "names": list(str),
            "family": list
        }`
    """
    if type(families) is str:
        families = [families]
    if verbose:
        if len(families) > 1:
            log.info(f"Loading the families {families} from `{path}`.")
        else:
            log.info(f"Loading the family {families[0]} from `{path}`.")
    data = _read_formatted_data(path)
    selected_data = [[], [], [], []]
    for family in families:
        if not diurnal.family.is_known(family):
            log.error(f"Family `{family}` not recognized.")
            raise ValueError
    for i in range(len(data[0])):
        if data[-1][i] in families:
            for j in range(4):
                selected_data[j].append(data[j][i])
    if randomize:
        selected_data = list(shuffle_data(*selected_data))
    return _convert_data_to_dict(selected_data)


# Output data manipulation
def categorize_vector(prediction: list) -> list:
    """Convert a vector of predicted pairings into a one-hot vector. For
    instance, `[[0.9, 0.5, 0.1], [0.0, 0.5, 0.1]]` is converted to
    `[[1, 0, 0], [0, 1, 0]]`.

    Args:
        prediction (list-like): Secondary structure prediction.

    Returns: Reformatted secondary structure.
    """
    if type(prediction) in (np.ndarray, torch.Tensor):
        pred_vector = prediction.tolist()
    else:
        pred_vector = list(prediction)
    if (type(prediction[0]) in (int, float, np.float16, np.float32)
            or len(prediction[0]) == 1):
        return [round(p) for p in prediction]
    indices = [n.index(max(n)) if sum(n) else -1 for n in pred_vector]
    element_size = len(pred_vector[0])
    return [[1 if j == i else 0 for j in range(element_size)] for i in indices]


def categorize_matrix(prediction: np.ndarray) -> np.ndarray:
    pred = np.array(prediction)
    for i in range(len(prediction)):
        for j in range(len(prediction[i])):
            pred[i][j] = round(prediction[i][j])
    return pred


def clean_vectors(primary: list, true: list, pred: list) -> tuple:
    """Prepare a secondary structure prediction for evaluation.

    Args:
        primary (list): Vector-encoded primary structure.
        true (list): True vector-encoded secondary structure.
        pred (list): Predicted vector-encoded secondary structure.

    Returns (tuple): A tuple of elements organized as:
      - sequence of bases
      - stripped true secondary structure
      - stripped predicted secondary structure
    """
    bases = diurnal.structure.Primary.to_sequence(primary)
    return bases, true[:len(bases)], categorize_vector(pred[:len(bases)])


def clean_matrices(primary: list, true: list, pred: list) -> tuple:
    """Prepare a secondary structure prediction for evaluation.

    Args:
        primary (list): Vector-encoded primary structure.
        true (list): True vector-encoded secondary structure.
        pred (list): Predicted vector-encoded secondary structure.

    Returns (tuple): A tuple of elements organized as:
      - sequence of bases
      - stripped true secondary structure matrix
      - stripped predicted secondary structure matrix
    """
    bases = diurnal.structure.Primary.unpad_matrix(primary)
    secondary = true[:len(bases), :len(bases)]
    return (
        bases,
        secondary,
        categorize_matrix(secondary)
    )


def quantize_matrix(matrix: list[list[float]], dim: int = 0) -> None:
    """Quantize a matrix.

    All the rows of the matrix are formatted as follows:
    - The maximum element is set to 1.
    - The other elements are set to 0.

    Args:
        matrix: Input matrix
        dim: Dimension along which to quantize the matrix.
    """
    for i in range(len(matrix)):
        if dim == 0:
            maximum = max(matrix[i,:])
            total = sum(matrix[i,:])
        elif dim == 1:
            maximum = max(matrix[:,i])
            total = sum(matrix[:,i])
        for j in range(len(matrix[i])):
            if total <= 0:
                matrix[i][j] = 0.0
            elif matrix[i][j] == 0 or matrix[i][j] != maximum:
                matrix[i][j] = 0.0
            else:
                matrix[i][j] = 1.0
