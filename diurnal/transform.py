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
        offset: int = 1
    ) -> np.ndarray:
    """Unfold the lower half of a 2D matrix into a 1D vector according
    to the direction of the anti-diagonal. The diagonal is excluded.

    Args:
        matrix: Matrix to linearize. Not modified.
        n_rows: Index of the first matrix row that only contains zero
            elements. If `None`, set to the length of the matrix.
        N: Length of the resulting array. If longer than the linearized
            vector, the right side is zero-padded.
        offset: Number of elements away from the diagonal to exclude
            from the resulting vector.

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
    linear = np.zeros(N)
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
