"""
    Transform matrices into other formats.

    This module is intended to prepare matrices for training.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: May 2024
    - License: MIT
"""

import numpy as np

from diurnal import structure


# Approximate minimum compression rate when collapsing a structure
COMPRESSION_RATE = 1


def halve_matrix(
        matrix: np.ndarray,
        empty_element: any = 0,
        padding_element: any = -1
    ) -> np.ndarray:
    """Convert a matrix to a lower triangular matrix by replacing all
    elements in the upper triangle by `empty_element`.

    Args:
        matrix: Matrix to halve. Must have at least 2 dimensions.
        empty_element: Element that represents an unpaired base.
        padding_element: Element that represents an out-of-bound base.

    Returns: Half matrix.
    """
    half = matrix.copy()
    for i in range(half.shape[0]):
        for j in range(i, half.shape[0]):
            if half[i, j] == padding_element:
                pass
            else:
                half[i, j] = empty_element
    return half


def unhalve_matrix(matrix: np.ndarray) -> np.ndarray:
    """Convert a lower triangular matrix to a full matrix by assigning
    to each [i, j] element the sum of the [i, j] and [j, i] elements.

    Args:
        matrix: Lower triangular matrix to convert to a full matrix.
            Must have at least 2 dimensions.
    """
    full = matrix.copy()
    for i in range(full.shape[0]):
        for j in range(i, full.shape[0]):
            full[i, j] = full[i, j] + full[j, i]
    return full


def rotate_and_offset(matrix: np.ndarray, offset: int = 3) -> np.ndarray:
    """Rotate a matrix90 degrees clockwise and shift it up.

    Args:
        matrix: Input value.
        offset: Upper shift.

    Returns: Rotated and shifted matrix.

    Example:

    >>> a = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 1, 0, 0],
    ])
    >>> rotate_and_offset(a, 2)
    [[1, 1, 0, 0],
     [1, 0, 0, 0],
     [0, 0, 0, 0],
     [0, 0, 0, 0]]
    """
    raise RuntimeError("Unsupported")


def linearize_half_matrix(
        matrix: np.ndarray,
        n_rows: int = None,
        N: int = None,
        offset: int = 3,
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
        offset: int = 3,
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
    array([2, -3, 1, -2])
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


def collapse_like(
        collapsed: np.ndarray,
        vector: np.ndarray,
        collapsed_replacement: int | np.ndarray = -1,
        vector_replacement: int | np.ndarray = 0
    ) -> np.ndarray:
    """Collapse a matrix in the same way as another.

    Args:
        collapsed: An already collapsed array.
        vector: A non-collapsed array.
        collapsed_replacement: The value used in the collapsed array to
            replace sequences of empty elements.
        vector_replacement: The value used in the non collapsed array
            to replace sequences of empty elements.

    Returns: Collapsed vector.

    Example:
    >>> a = np.array([2, 0, 0, 0, 1, 0, 0])
    >>> b = collapse_linearized_matrix(a)
    >>> c = np.array([0, 0, 0, 0, 1, 0, 0])
    >>> d = collapse_like(a, c)
    >>> a
    array([2, -3, 1, -2])
    >>> d
    array([0, 0, 1, 0])
    """
    if type(vector_replacement) == int:
        new_vector = np.zeros(len(collapsed))
    else:
        new_vector = np.zeros_like(collapsed)
    j = 0
    for i in range(len(collapsed)):
        if type(collapsed_replacement) == int:
            if collapsed[i] < 0:
                n = int(-1 * collapsed[i])
                new_vector[i] = vector_replacement * n
                j += n
                if j >= new_vector.shape[0]:
                    break
            else:
                new_vector[i] = vector[j]
                j += 1
        else:
            if _coincident(collapsed[i], collapsed_replacement):
                partial_sum = vector[j].copy()
                coefficient = 0
                while partial_sum.sum() > 0:
                    partial_sum -= collapsed_replacement
                    coefficient += 1
                new_vector[i] = coefficient * vector_replacement
                j += coefficient
                if j >= new_vector.shape[0]:
                    break
            else:
                new_vector[i] = vector[j]
                j += 1
    return new_vector


def decollapse_like(
        collapsed: np.ndarray,
        vector: np.ndarray,
        N: int,
        collapsed_replacement: int | np.ndarray = -1,
        vector_replacement: int | np.ndarray = 0
    ) -> np.ndarray:
    """Decollapse a matrix in the same way as another.

    Args:
        collapsed: An already collapsed array.
        vector: A collapsed array to decollapse.
        N: Size of the decollapsed array.
        collapsed_replacement: The value used in the collapsed array to
            replace sequences of empty elements.
        vector_replacement: The value used in the non collapsed array
            to replace sequences of empty elements.

    Returns: Decollapsed vector.

    Example:
    >>> a = np.array([2, 0, 0, 0, 1, 0, 0])
    >>> b = collapse_linearized_matrix(a)
    >>> c = np.array([0, 0, 0, 0, 1, 0, 0])
    >>> d = collapse_like(a, c)
    >>> a
    array([2, -3, 1, -2])
    >>> d
    array([0, 0, 1, 0])
    """
    if type(vector_replacement) == int:
        new_vector = np.zeros(N)
    else:
        new_vector = np.zeros((N, vector.shape[1]))
    j = 0
    for i in range(len(collapsed)):
        if type(collapsed_replacement) == int:
            if collapsed[i] < 0:
                for _ in range(int(-1 * collapsed[i])):
                    new_vector[j] = vector_replacement
                    j += 1
            else:
                new_vector[j] = vector[i]
                j += 1
        else:
            if _coincident(collapsed[i], collapsed_replacement):
                partial_sum = collapsed[i].copy()
                while partial_sum.sum() > 0:
                    new_vector[j] = vector_replacement
                    partial_sum -= collapsed_replacement
                    j += 1
            else:
                new_vector[j] = vector[i]
                j += 1
    return new_vector


def to_monomial_matrix(matrix: np.ndarray) -> np.ndarray:
    """Convert a matrix to a monomial matrix by successively selecting
    the maximum element of the array and setting the rest of the values
    in its row and column to 0.

    Args:
        matrix: 2D array. Cannot contain negative values.

    Returns: A monomial (i.e. generalized permutation) matrix.

    Example:
    >>> a = np.array([
        [0, 2, 3, 2],
        [2, 7, 2, 2],
        [2, 2, 2, 2],
        [2, 2, 2, 9],
    ])
    >>> to_monomial_matrix(a)
    array([
        [0, 0, 3, 0],
        [0, 7, 0, 0],
        [2, 0, 0, 0],
        [0, 0, 0, 9],
    ])
    """
    mask = np.ones_like(matrix)
    monomial = np.zeros_like(matrix)
    N = len(matrix)
    for _ in range(N):
        probe = matrix * mask
        index = probe.argmax(axis=None)
        row = index // N
        column = index % N
        monomial[row, column] = matrix[row, column]
        mask[row, :] = 0
        mask[:, column] = 0
    return monomial


def to_binary_matrix(matrix: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """Transform the argument into a binary matrix.

    Args:
        matrix: 2D array.

    Returns: Binary matrix.
    """
    binary = matrix.copy()
    binary[binary > threshold] = 1
    binary[binary <= threshold] = 0
    return binary


def quantize(matrix: np.ndarray, is_half: bool = False) -> np.ndarray:
    """Quantize a matrix to a monomial binary matrix.

    Intended to be used to convert a pseudo energy term matrix to a
    contact matrix.

    Args:
        matrix: Input matrix.
        is_half: If True, the matrix is first converted into a full
            matrix.

    Result: Quantized matrix.
    """
    if is_half:
        matrix = matrix + matrix.T
    else:
        matrix = matrix * matrix.T
    matrix = to_monomial_matrix(matrix)
    matrix = to_binary_matrix(matrix)
    return matrix


def primary_linear_formatter(x: str | list[str], y: int) -> np.ndarray:
    """Format a primary structure into a linear representation.

    Args:
        x: Primary structure as a sequence of characters.
        y: Normalized size.

    Returns: Linear array.
    """
    potential_pairings = structure.Primary.to_matrix(
        x, y, structure.Schemes.IUPAC_PAIRINGS_SCALARS
    )
    linear_size = int(y**2 / 2 * COMPRESSION_RATE)
    potential_pairings = linearize_half_matrix(
        potential_pairings, len(x), N=linear_size
    )
    return potential_pairings


def to_mask(x: str | list[str], y: int) -> np.ndarray:
    """Format a primary structure into a binary matrix whose zero
    element indicate impossible pairings.

    Args:
        x: Primary structure as a sequence of characters.
        y: Normalized size.

    Returns: Mask.
    """
    return structure.Primary.to_mask(x, y)


def primary_linear_collapse_formatter(
        x: str | list[str], y: int
    ) -> np.ndarray:
    """Format a primary structure into a collapsed linear
    representation.

    Args:
        x: Primary structure as a sequence of characters.
        y: Normalized size.

    Returns: Collapsed array.
    """
    potential_pairings = structure.Primary.to_matrix(
        x, y, structure.Schemes.IUPAC_PAIRINGS_SCALARS
    )
    linear_size = int(y**2 / 2 * COMPRESSION_RATE)
    potential_pairings = linearize_half_matrix(
        potential_pairings, len(x), N=linear_size
    )
    potential_pairings = collapse_linearized_matrix(
        potential_pairings, N=linear_size
    )
    return potential_pairings


def secondary_linear_formatter(
        x: str | list[str], size: int, power: int = 0
    ) -> np.ndarray:
    """Format a secondary structure into a linear representation.

    Args:
        x: Secondary structure as a sequence of pairing indices.
        size: Normalized size.

    Returns: Linear array.
    """
    linear_size = int(size**2 / 2 * COMPRESSION_RATE)
    if power:
        contact = structure.Secondary.to_matrix(y, size)
    else:
        contact = structure.Secondary.to_distance_matrix(x, size, power=power)
    contact = linearize_half_matrix(contact, len(x), N=linear_size)
    return contact


def secondary_linear_collapse_formatter(
        x: str | list[str], y: list[int], size: int, power: int = 0
    ) -> np.ndarray:
    """Format a secondary structure into a linear collapsed
    representation.

    Args:
        x: Primary structure as a sequence of characters.
        y: Secondary structure as a sequence of pairing indices.
        size: Normalized size.

    Returns: Collapsed array.
    """
    potential_pairings = structure.Primary.to_matrix(
        x, size, structure.Schemes.IUPAC_PAIRINGS_SCALARS
    )
    linear_size = int(size**2 / 2 * COMPRESSION_RATE)
    potential_pairings = linearize_half_matrix(
        potential_pairings, len(x), N=linear_size
    )
    potential_pairings = collapse_linearized_matrix(
        potential_pairings, N=linear_size
    )
    if power:
        contact = structure.Secondary.to_matrix(y, size)
    else:
        contact = structure.Secondary.to_distance_matrix(y, size, power=power)
    contact = linearize_half_matrix(contact, len(x), N=linear_size)
    contact = collapse_like(potential_pairings, contact)
    return contact


def to_distance(
        matrix: np.ndarray, power: float = 1, normalize: bool = True
    ) -> np.ndarray:
    """Encode a secondary structure into a score contact matrix.

    Transform the sequence of pairings into an `n` by `n` matrix,
    where `n` is the number of pairings, whose elements can be `1`
    for a paired base and `x` for unpaired bases, where `x` is
    given by: `x = 1 - (d / n)`, in which `d` is the Manhattan
    distance with the closest paired base.

    Args:
        matrix: Input matrix.
        power (float): Power to apply to normalized distances.
        normalize (bool): If True, scale distances so that
            paired elements are 1 and the farthest elements are 0.

    Returns (np.ndarray): Encoded matrix of the secondary structure.
    """
    n = len(matrix)
    contact = matrix.copy()
    contact -= 1
    contact *= -1
    contact *= n
    directions = (
        (1, 0), (1, -1), (0, -1), (-1, -1),
        (-1, 0), (-1, 1), (0, 1), (1, 1)
    )
    for distance in range(n):
        for i in range(n):
            for j in range(n):
                if contact[i, j] == distance:
                    for d in directions:
                        I = i + d[0]
                        J = j + d[1]
                        if I < 0 or I >= n or J < 0 or J >= n:
                            continue
                        value = contact[I, J]
                        if value > distance + 1:
                            contact[I, J] = distance + 1
    if not normalize:
        return contact
    contact /= n
    contact *= -1
    contact += 1
    return normalize_distance_matrix(contact) ** power


def normalize_distance_matrix(distance_matrix: np.ndarray) -> np.ndarray:
    """Normalize the distance matrix.

    This function returns a new distance matrix whose elements are
    normalized within the range 0.0 (farthest from a paired base)
    to 1.0 (paired base).

    Args:
        distance_matrix (np.ndarray): Result of the function
            `to_distance_matrix`.

    Returns (np.ndarray): Normalized distance matrix.
    """
    normalized = distance_matrix.copy()
    normalized -= 1
    normalized *= -1
    normalized /= np.amax(normalized)
    normalized -= 1
    normalized *= -1
    return normalized
