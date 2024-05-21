"""
    Reinforcement learning (RL) models.

    Note: The methods of this file are **unsafe** in the sense that
    they never validate the input data. This design decision accelerate
    execution, but also makes it less safe. The arguments have to be
    provided as described by the doscstrings to ensure proper behavior.

    File information:

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: May 2024
    - License: MIT
"""

from torch import Tensor, roll
import numpy as np


class BasicContactMatrixOperations:
    """Operations to interact with a RL environment based on contact
    matrices.
    """
    def get_free_rows(matrix: np.ndarray | Tensor)-> np.ndarray:
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
            matrix: np.ndarray | Tensor,
            rows: np.ndarray | Tensor,
            columns: np.ndarray | Tensor
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

    def clear_row(
            matrix: np.ndarray | Tensor, rows: np.ndarray | Tensor
        ) -> None:
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

    def clear_column(
            matrix: np.ndarray | Tensor, columns: np.ndarray | Tensor
        ) -> None:
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


class SRLA1:
    """Supervised reinforcement learning agent number 1 to predict RNA
    secondary structures in a contact matrix.

    Environment:

    - Scalar matrix of potential pairings.
    - Tentative contact matrix.

    Reward:

    - Loss between the tentative contact matrix and the real contact
      matrix.

    States:

    - A **cursor**, that is, a two-component index that points at a
      potential pairing in the contact matrix.
    - A Q table represented with a neural network.

    Actions:

    - Move the cursor down.
    - Move the cursor up.
    - Move the cursor left.
    - Move the cursor right.
    - Assign 1 to the cursor.
    - Assign 0 to the cursor.
    """

    def act(
            tentative: np.ndarray | Tensor,
            potential: np.ndarray | Tensor,
            cursor: np.ndarray | Tensor,
            actions: np.ndarray | Tensor
        ) -> None:
        """Modify the tentative contact matrix and cursor based on
        actions.

        Args:
            tentative: Tentative contact matrix (i.e. the result that
                the model is currently producing). Modified in place.
            potential: Scalar matrix of potential pairings.
            cursor: A single-entry matrix that indicates a selected
                element. Modified in place.
            actions: A probability vector containing exactly 6 elements
                that each indicate the probability of performing the
                following actions:
                0. Move the cursor down.
                1. Move the cursor up.
                2. Move the cursor left.
                3. Move the cursor right.
                4. Assign 1 to the cursor in the tentative matrix.
                    Performed only if the potential matrix allows it.
                5. Assign 0 to the cursor in the tentative matrix.
                    Performed only if the potential matrix allows it.
            int: Sequence length
        """
        match actions.argmax():
            case 0:
                cursor[:, :] = roll(cursor, 1, 0)
            case 1:
                cursor[:, :] = roll(cursor, -1, 0)
            case 2:
                cursor[:, :] = roll(cursor, -1, 1)
            case 3:
                cursor[:, :] = roll(cursor, 1, 1)
            case 4:
                index = cursor.argmax(axis=None)
                row = index // len(tentative)
                column = index % len(tentative)
                if potential[row, column] == 0:
                    return
                tentative[row, :] = 0
                tentative[:, column] = 0
                tentative[row, column] = 1
            case 5:
                index = cursor.argmax(axis=None)
                row = index // len(tentative)
                column = index % len(tentative)
                if potential[row, column] == 0:
                    return
                index = cursor.argmax(axis=None)
                tentative[row, column] = 0

    def reward(
            tentative: np.ndarray | Tensor,
            contact: np.ndarray | Tensor,
            n: int
        ) -> float:
        """Compute the reward after an action.

        Args:
            tentative: Approximated contact matrix.
            contact: Real contact matrix (blurred or not).
            n: Sequence length

        Returns: Reward score comprised within the range (-inf, 0).
        """
        difference = float((contact - tentative).sum)
        return difference / n
