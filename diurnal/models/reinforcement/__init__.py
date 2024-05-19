"""
    Reinforcement learning (RL) module.

    File information:

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: May 2024
    - License: MIT
"""

import numpy as np


class ContactMatrix:
    """Operations to interact with a RL environment based on contact
    matrices.

    Note: The methods of this class are **unsafe** in the sense that
    they never validate the input data. The arguments have to be
    provided as described by the doscstrings to ensure proper behavior.
    """
    def get_free_rows(matrix: np.ndarray)-> np.ndarray:
        """Find the rows that contain no pairing.

        Args:
            matrix: Contact matrix to analyze.

        Returns: A vector whose non-zero elements indicate free rows.

        Example:

        >>> a = np.array([[0, 1], [0, 0]])
        >>> ContactMatrix.get_free_rows(a)
        array([0, 1])
        """
        indices = np.clip(np.sum(matrix, axis=1), 0, 1)
        return np.ones_like(indices) - indices

    def get_free_columns(matrix: np.ndarray)-> np.ndarray:
        """Find the columns that contain no pairing.

        Args:
            matrix: Contact matrix to analyze.

        Returns: A vector whose non-zero elements indicate free
            columns.

        Example:

        >>> a = np.array([[0, 1], [0, 0]])
        >>> ContactMatrix.get_free_columns(a)
        array([1, 0])
        """
        indices = np.clip(np.sum(matrix, axis=0), 0, 1)
        return np.ones_like(indices) - indices

    def insert(
            matrix: np.ndarray, rows: np.ndarray, columns: np.ndarray
        ) -> None:
        """Insert a 1 at the specified index.

        Args:
            matrix: Contact matrix. Modified in place.
            rows: Vector of scalars normalized between 0 and 1 whose
                maximum corresponds to the row of the inserted element.
            columns: Vector of scalars normalized between 0 and 1 whose
                maximum corresponds to the column of the inserted
                element.

        Example:

        >>> a = np.array([[0, 0], [0, 0]])
        >>> rows = np.array([0.1, 0.5])
        >>> columns = np.array([0.0, 0.9])
        >>> ContactMatrix.insert(a, rows, columns)
        >>> a
        array([[0, 0],
               [0, 1]])
        """
        matrix[rows.argmax(), columns.argmax()] = 1

    def clear_row(matrix: np.ndarray, rows: np.ndarray) -> None:
        """Remove all pairings in a row.

        Args:
            matrix: Contact matrix. Modified in place.
            rows: Vector of scalars normalized between 0 and 1 whose
                maximum corresponds to the row to clear.

        Example:

        >>> a = np.array([[0, 1], [1, 0]])
        >>> rows = np.array([0.1, 0.5])
        >>> ContactMatrix.clear_row(a, rows)
        >>> a
        array([[0, 1],
               [0, 0]])
        """
        matrix[rows.argmax(), :] = 0

    def clear_column(matrix: np.ndarray, columns: np.ndarray) -> None:
        """Remove all pairings in a column.

        Args:
            matrix: Contact matrix. Modified in place.
            rows: Vector of scalars normalized between 0 and 1 whose
                maximum corresponds to the column to clear.

        Example:

        >>> a = np.array([[0, 1], [1, 0]])
        >>> rows = np.array([0.1, 0.5])
        >>> ContactMatrix.clear_row(a, rows)
        >>> a
        array([[0, 0],
               [1, 0]])
        """
        matrix[:, columns.argmax()] = 0
