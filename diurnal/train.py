"""
    RNA secondary structure training utility module.

    Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    Affiliation: Département d'informatique, UQÀM
    File creation date: April 2023
    License: MIT
"""

from random import shuffle
import numpy as np
import torch
import os.path

from diurnal.utils import log
import diurnal.family
import diurnal.structure


# Input data transformation
def _split_arrays(data, fractions: list, offset: int = 0) -> list:
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


def split_data(data, fractions: list, offset: int = 0) -> list:
    """Split data in subsets according to the specified fractions.

    Args:
        data: Array-like object containing the data to split.
        fractions: Proportion of each subset. For instance, to use 80% of the
            data for training and 20% for testing, use [0.8, 0.2].
        offset: Number of indices to offset to assemble the subsets. Used for
            K-fold data splits.

    Returns:
        A list containing the split data object.
    """
    if sum(fractions) != 1.0:
        log.error("Invalid data split proportions.")
        raise RuntimeError
    if type(data) == dict:
        keys = list(data.keys())
        values = []
        for i in range(len(data[keys[0]])):
            element = []
            for k in keys:
                element.append(data[k][i])
            values.append(element)
        split_values = _split_arrays(values, fractions, offset)
        new_data = []
        for split in split_values:
            d = {}
            for i, k in enumerate(keys):
                d[k] = [element[i] for element in split]
            new_data.append(d)
        return new_data
    else:
        return _split_arrays(data, fractions, offset)


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
        raise "Invalid K-fold parameters."
    offset = (len(data) / k) * i
    return split_data(data, fractions, offset)


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
def _read_npy_files(path: str) -> tuple:
    """Read an RNA dataset encoded into .npy Numpy files.

    Args:
        path (str): Directory path of the folder containing the dataset
            files.

    Returns (tuple(list)):  List-converted file content in the following
        order: (primary structure, secondary structure, families, names)
    """
    if path[-1] != '/': path += '/'
    # Load data.
    relative_paths = [
        "primary_structures.npy",
        "secondary_structures.npy",
        "names.txt",
        "families.npy"
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
    return {
        "primary_structures": data[0],
        "secondary_structures": data[1],
        "names": data[2],
        "families": data[3] if len(data) > 3 else None
    }


def _convert_to_tensor(data: list) -> torch.Tensor:
    """Convert matrix-like objects into pyTorch tensors.

    Args:
        data (list-like): Array of number to convert into a tensor.

    Returns (torch.Tensor): Tensor-converted data.
    """
    tensors = []
    if len(data) < 1:
        return None
    for i in range(len(data[0])):
        tensor = [
            torch.tensor(data[0][i].T, dtype=torch.float32),
            torch.tensor(data[1][i],   dtype=torch.float32),
            torch.tensor(data[2][i],   dtype=torch.float32)
        ]
        tensors.append(tensor)
    return tensors


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
    data = _read_npy_files(path)
    # Shuffle data.
    if randomize:
        data = list(shuffle_data(*data))
    return _convert_data_to_dict(data)


def load_inter_family(path: str, family: str, randomize: bool = True) -> list:
    """Read formatted data into a tensor that contains the specified
    family and another tensor that contains all the other families.

    Args:
        path (str): Name of the directory that contains the Numpy files
            written by the function `diurnal.database.format`.
        family (str): Family to place in a different tensor.
        randomize (bool): Randomize data if set to True.

    Returns:
        tuple: Loaded data represented as
            [primary structure, secondary structure, family].
            The first element is the test family. The second element
            comprises all other families.
    """
    if not diurnal.family.is_known(family):
        log.error(f"Family `{family}` not recognized.")
        raise ValueError
    data = _read_npy_files(path)
    family_vector = diurnal.family.to_vector(family)
    k = [[], [], [], []] # Test family
    n = [[], [], [], []] # Other families
    for i in range(len(data[0])):
        if np.array_equal(data[3][i], family_vector):
            for j in range(4):
                k[j].append(data[j][i])
        else:
            for j in range(4):
                n[j].append(data[j][i])
    # Shuffle data.
    if randomize:
        k = list(shuffle_data(*k))
        n = list(shuffle_data(*n))
    return _convert_data_to_dict(k), _convert_data_to_dict(n)


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
    indices = [n.index(max(n)) if sum(n) else -1 for n in pred_vector]
    element_size = len(pred_vector[0])
    return [[1 if j == i else 0 for j in range(element_size)] for i in indices]


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
    bases = diurnal.structure.Primary.to_bases(primary)
    return bases, true[:len(bases)], categorize_vector(pred[:len(bases)])
