"""
    Transform matrices into other formats.

    This module is intended to prepare matrices for training.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: June 2023
    - License: MIT
"""

import numpy as np


def halve_matrix(
        matrix: np.ndarray,
        empty_element: any = 0,
        padding_element: any = -1
    ) -> None:
    """Convert a matrix to a lower triangular matrix by replacing all
    elements in the upper triangle by `empty_element`.

    Args:
        matrix: Matrix to halve. Must have at least 2 dimensions.
        empty_element: Element that represents an unpaired base.
        padding_element: Element that represents an out-of-bound base.
    """
    for i in range(matrix.shape[0]):
        for j in range(i, matrix.shape[0]):
            if matrix[i, j] == padding_element:
                pass
            else:
                matrix[i, j] = empty_element


def unhalve_matrix(matrix: np.ndarray) -> np.ndarray:
    """Convert a lower triangular matrix to a full matrix by assigning
    to each [i, j] element the sum of the [i, j] and [j, i] elements.

    Args:
        matrix: Lower triangular matrix to convert to a full matrix.
            Must have at least 2 dimensions.
    """
    for i in range(matrix.shape[0]):
        for j in range(i, matrix.shape[0]):
            matrix[i, j] = matrix[i, j] + matrix[j, i]


def linearize_half_matrix(
        matrix: np.ndarray,
        n_rows: int = None,
        N: int = None,
        offset: int = 4,
        padding_element: int | np.ndarray = 0
    ) -> np.ndarray:
    """Unfold the lower half of a 2D matrix into a 1D vector according
    to the direction of the anti-diagonal.

    Args:
        matrix: Square matrix to linearize. Not modified.
        n_rows: Number of rows from the original matrix to include. If
            `None`, set to the length of the matrix.
        N: Length of the resulting vector. If longer than the linearized
            vector, the right side is padded. Set to the minimum length
            if `None`.
        offset: Number of elements away from the diagonal to exclude
            from the resulting vector. `1` excludes the diagonal. Use
            `4` to exclude impossible RNA pairings.
        padding_element: Element added at the end of the resulting
            vector.

    Returns: Linearized vector.

    Example:

    >>> a = np.array(
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [2, 3, 0, 0, 0],
            [4, 5, 6, 0, 0],
            [7, 8, 9, 1, 0],
        )
    >>> linearize_half_matrix(a, 4)
    [1, 2, 4, 3, 5, 6]
    """
    # Argument validation.
    if n_rows is None:
        n_rows = len(matrix)
    n_central = n_rows
    for i in range(1, offset):
        n_central += (n_rows - i) * 2
    minimum_length = int((n_rows ** 2 - n_central) / 2)
    if N is None:
        N = minimum_length
    elif N < minimum_length:
        raise ValueError(f"{N} is too small. Minimum value: {minimum_length}")
    # Matrix linearization.
    linear = np.full(N, padding_element)
    linear_index = 0
    vertical = n_rows - offset - 1
    for i in range(offset, vertical * 2 + offset + 1):
        for j in range(0, i):
            column = j
            row = i - column
            if column > row - offset:
                break
            if row < n_rows:
                linear[linear_index] = matrix[row, column]
                linear_index += 1
    return linear


def delinearize_half_matrix(
        vector: np.ndarray,
        n_rows: int,
        N: int = None,
        offset: int = 4,
        padding_element: int | np.ndarray = 0
    ) -> np.ndarray:
    """Fold a 1D array into a lower triangular 2D matrix.

    This is the reverse operation of the function
    `transform.linearize_half_matrix`.

    Args:
        vector: Vector to linearize. Not modified.
        n_rows: Number of rows from the original matrix to include.
        N: Dimension of the resulting matrix. If `None`, set to
            `n_rows`.
        offset: Number of elements away from the diagonal to exclude
            from the resulting vector. `1` excludes the diagonal. Use
            `2` to exclude impossible RNA pairings.
        padding_element: Element added at the end of the resulting
            vector.

    Returns: Delinearized vector.

    Example:

    >>> a = np.array([1, 2, 4, 3, 5, 6])
    >>> delinearize_half_matrix(a, 4, 1)

    [[0, 0, 0, 0],
     [1, 0, 0, 0],
     [2, 3, 0, 0],
     [4, 5, 6, 0]]
    """
    if not N:
        N = n_rows
    matrix = np.full((N, N), padding_element)
    linear_index = 0
    vertical = n_rows - offset - 1
    for i in range(offset, vertical * 2 + offset + 1):
        for j in range(0, i):
            column = j
            row = i - column
            if column > row - offset:
                break
            if row < n_rows:
                matrix[row, column] = vector[linear_index]
                linear_index += 1
    return matrix


def collapse_linearized_matrix(
        vector: np.ndarray,
        collapsed_element: int | np.ndarray = 0,
        replacement: int | np.ndarray = -1,
        N: int = None
    ) -> np.ndarray:
    """Collapse a linearized half matrix by regrouping collapsed
    elements into sums.

    Args:
        vector: Linearized lower half matrix.
        collapsed_element: Elements to collapse.
        replacement: The new elements that replace `collapsed_element`.
        N: Size of the resulting vector. If `None`, `N` is set to the
            minimum length.

    Returns: Collapsed vector.

    Example:
    >>> a = np.array([2, 0, 0, 0, 1, 0, 0])
    >>> collapse_linearized_matrix(a)
    array([2, -3, 1, -2, 0, 0, 0])
    """
    crop = False
    if N is None:
        N = len(vector)
        crop = True
    if len(vector.shape) == 1:
        new_vector = np.zeros(N)
    else:
        new_vector = np.zeros((N, vector.shape[1]))
    current_sum = collapsed_element
    j = 0
    for i in range(len(vector)):
        if vector[i] == collapsed_element:
            current_sum += replacement
        elif current_sum:
            new_vector[j] = current_sum
            current_sum = collapsed_element
            j += 1
            new_vector[j] = vector[i]
            j += 1
        else:
            new_vector[j] = vector[i]
            j += 1
        # Last element.
        if i == len(vector) - 1:
            if current_sum:
                new_vector[j] = current_sum
                j += 1
    if crop:
        return new_vector[:j]
    return new_vector


def _coincident(one, two):
    return np.dot(one,two)*np.dot(one,two) == np.dot(one,one)*np.dot(two,two)


def decollapse_linearized_matrix(
        vector: np.ndarray,
        collapsed_element: int | np.ndarray = 0,
        replacement: int | np.ndarray = -1,
        N_input: int = None,
        N_output: int = None
    ) -> np.ndarray:
    """Unfold a collapsed linearized half matrix.

    Args:
        vector: Collapsed linearized lower half matrix.
        collapsed_element: Original elements before collapsing.
        replacement: The new elements that replaced `collapsed_element`.
        N_input: Size of the original vector.
        N_output: Size of the resulting vector. If `None`, `N` is set
            to the length of the input vector.

    Returns: Unfolded vector.

    Example:
    >>> a = np.array([2, -3, 1, -2, 0, 0, 0])
    >>> decollapse_linearized_matrix(a)
    array([2, 0, 0, 0, 1, 0, 0])
    """
    if N_output is None:
        N_output = len(vector)
    if len(vector.shape) == 1:
        new_vector = np.zeros(N_output)
    else:
        new_vector = np.zeros((N_output, vector.shape[1]))
    j = 0
    if N_input is None:
        N_input = len(vector)
    for i in range(N_input):
        if type(replacement) == int:
            if vector[i] < 0:
                for _ in range(int(-1 * vector[i])):
                    new_vector[j] = collapsed_element
                    j += 1
            else:
                new_vector[j] = vector[i]
                j += 1
        else:
            if _coincident(vector[i], replacement):
                partial_sum = vector[i].copy()
                while partial_sum.sum() > 0:
                    new_vector[j] = collapsed_element
                    partial_sum -= replacement
                    j += 1
            else:
                new_vector[j] = vector[i]
                j += 1
    return new_vector