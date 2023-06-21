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

from .utils import file_io
import diurnal.family
import diurnal.structure


# Input data transformation
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
        file_io.log("Invalid data split proportions.", -1)
        raise RuntimeError
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
        file_io.log("Shuffled data are not homogeneous.", -1)
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
        "families.npy",
        "names.txt"]
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
            file_io.log(f"load_data: Inhomogeneous sequences: {lengths}", -1)
            raise RuntimeError
    else:
        file_io.log(f"load_data: path `{path}` is empty.", -1)
        raise RuntimeError
    return data


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
    return _convert_to_tensor(data[:3]), data[-1]


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
    data = _read_npy_files(path)
    if not diurnal.family.is_known(family):
        file_io.log(f"Family `{family}` not recognized.", -1)
        raise ValueError
    family_vector = diurnal.family.to_vector(family)
    k = [[], [], [], []] # Test family
    n = [[], [], [], []] # Other families
    for i in range(len(data[0])):
        if np.array_equal(data[2][i], family_vector):
            for j in range(4):
                k[j].append(data[j][i])
        else:
            for j in range(4):
                n[j].append(data[j][i])
    # Shuffle data.
    if randomize:
        k = list(shuffle_data(*k))
        n = list(shuffle_data(*n))
    K = _convert_to_tensor(k[:3]) if k else None
    N = _convert_to_tensor(n[:3]) if n else None
    k_names = k[-1] if k else None
    n_names = n[-1] if n else None
    return K, N, k_names, n_names


# Output data manipulation
def categorize_vector(prediction: list) -> list:
    """Convert a vector of predicted pairings into a one-hot vector. For
    instance, `[[0.9, 0.5, 0.1], [0.0, 0.5, 0.1]]` is converted to
    `[[1, 0, 0], [0, 1, 0]]`.

    Args:
        prediction (list-like): Secondary structure prediction.

    Returns: Reformatted secondary structure.
    """
    if type(prediction) == np.ndarray:
        pred_vector = prediction.tolist()
    else:
        pred_vector = list(prediction)
    indices = [n.index(max(n)) for n in pred_vector]
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
