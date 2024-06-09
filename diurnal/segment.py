"""
    Perform segmentation-related operations on matrices.

    This module is intended to post-process structure predictions.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: June 2024
    - License: MIT
"""

import numpy as np
import scipy.ndimage

from diurnal import transform, structure


def convolutional_denoise(
        matrix: np.ndarray, kernel: int = 3, threshold: int = 2
    ) -> np.ndarray:
    """Remove noise with a convolutional low-pass filter.

    Args:
        matrix: Matrix to denoise.
        kernel: Size of the convolutional kernel.
        threshold: Sum of the elements in the kernel below which the
            element is set to 0.

    Returns: Modified matrix. Each element
    """
    tmp = scipy.ndimage.convolve(
        matrix, np.ones((kernel, kernel)), mode='constant'
    )
    return np.logical_and(tmp >= threshold, matrix).astype(np.float32)


def expand_regions(matrix: np.ndarray, distance: float | int = 2) -> np.ndarray:
    """Expand non zero elements in a binary matrix.

    Args:
        matrix: Sparse binary matrix.
        distance: Normalized distance away from a non-zero element to
            be included in the region.

    Return: Region-expanded matrix.
    """
    result = np.zeros_like(matrix)
    normalize = type(distance) == float
    distances = transform.to_distance(matrix, normalize = normalize)
    for row in range(len(result)):
        for column in range(len(result[0])):
            if distances[row, column] <= distance:
                result[row, column] = 1
    return result


def expansion_formatter(
        pairings: list[int], size: int, distance: float | int = 2
    ) -> np.ndarray:
    """Format a secondary structure into an expanded contact matrix.

    Args:
        pairings: Raw secondary structure.
        size: Normalized matrix size.
        distance: Expansion distance.

    Returns: Formatted secondary structure.
    """
    matrix = structure.Secondary.to_matrix(pairings, size)
    return expand_regions(matrix, distance)


def sample_areas(
        matrix: np.ndarray,
        kernel: int,
        threshold: float = 0.0,
        n_minimum: int = None,
        stride: int = None
    ) -> list[tuple[int]]:
    """Obtain indices of areas in the matrix that sum to a threshold.

    Args:
        matrix: Input 2D matrix.
        kernel: Dimension of the sampled areas.
        threshold: Areas with a larger sum are sampled.
        n_minimum: Minimum number of areas to fetch.
        stride: Distance between sampling areas.

    Returns: Tuple of int pairs that indicate sampling areas.
    """
    areas = []
    if not stride:
        stride = kernel
    v_stride = 0
    while v_stride + kernel < len(matrix) + 1:
        h_stride = 0
        while h_stride + kernel < len(matrix) + 1:
            total = matrix[
                v_stride:v_stride + kernel,
                h_stride:h_stride + kernel
            ].sum()
            if total > threshold:
                areas.append(((v_stride, h_stride), total))
            h_stride += stride
        v_stride += stride
    areas.sort(key = lambda x: x[1], reverse = True)
    if n_minimum:
        n_minimum = min(len(areas), n_minimum)
        return areas[:n_minimum]
    return areas
