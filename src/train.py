"""
    RNA secondary structure training utility module.
"""

from random import shuffle

from .utils import file_io

# Data transformation
def split_data(data, fractions: list, offset: int = 0) -> list:
    """
    Split data in subsets according to the specified fractions.
    
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
        raise "Invalid data split proportions."
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
    """
    Split the data to make a K-fold split.
    
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
    """
    """
    # Validation
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
