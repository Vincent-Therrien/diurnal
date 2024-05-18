"""
    RNA secondary prediction evaluation module.

    This module contains functions to evaluate RNA secondary predictions
    by comparing a predicted structure to a reference structure.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: April 2023
    - License: MIT
"""

import statistics
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix

from diurnal.utils import log
from diurnal import train


def to_shadow(bracket: list[str] | str) -> list[int]:
    """Convert a bracket notation to a secondary structure shadow.

    Args:
        bracket: Secondary structure represented in bracket notation
            with the characters `(`, `.`, and `)`.

    Returns: Secondary structure shadow in which `0` stands for `(` or
        `)` and `1` stands for `.`.
    """
    return [0 if c == '.' else 1 for c in bracket]


class Shadow:
    """Evaluate predictions made with secondary structure shadows,
    i.e. a sequence of paired / unpaired bases.
    """

    def TP(true: list[int], pred: list[int]) -> float:
        """Compute the true positive value (predicted paired bases that
        are actually paired).
        """
        tp = 0
        for i in range(len(true)):
            if pred[i] and true[i]:
                tp += 1
        return tp

    def TN(true: list[int], pred: list[int]) -> float:
        """Compute the true negative value (predicted unpaired bases
        that are actually unpaired).
        """
        tn = 0
        for i in range(len(true)):
            if pred[i] == 0 and true[i] == 0:
                tn += 1
        return tn

    def FP(true: list[int], pred: list[int]) -> float:
        """Compute the false positive value (predicted paired bases that
        are actually unpaired).
        """
        fp = 0
        for i in range(len(true)):
            if pred[i] and true[i] == 0:
                fp += 1
        return fp

    def FN(true: list[int], pred: list[int]) -> float:
        """Compute the false negative value (predicted unpaired bases
        that are actually unpaired).
        """
        fn = 0
        for i in range(len(true)):
            if pred[i] == 0 and true[i]:
                fn += 1
        return fn

    def recall(true, pred) -> float:
        """Compute the recall value obtained by comparing two
        secondary structures. Recall is defined as:

        .. math::

            TP / (TP + FN).
        """
        tp = Shadow.TP(true, pred)
        fn = Shadow.FN(true, pred)
        if tp + fn:
            return tp / (tp + fn)
        else:
            return 0.0

    def precision(true, pred) -> float:
        """Compute the precision obtained by comparing two
        secondary structures. Precision is defined as:

        .. math::

            TP / (TP + FP).
        """
        tp = Shadow.TP(true, pred)
        fp = Shadow.FP(true, pred)
        if tp + fp:
            return tp / (tp + fp)
        else:
            return 0.0

    def recall_precision_f1(true, pred):
        """Compute the F1-score obtained by comparing two secondary
        structures. The f1-score is defined as:

        .. math::

            F1 = 2 \times \frac{recall \times precision}{recall + precision}
        """
        r = Shadow.recall(true, pred)
        p = Shadow.precision(true, pred)
        if r + p:
            f1 = 2 * ((r * p) / (r + p))
            return r, p, f1
        else:
            return r, p, 0.0

    def crop(shadow: list[int], length: int) -> list[int]:
        """Return a cropped shadow to exclude padding.

        Args:
            shadow: Shadow of the secondary structure.
            length: Number of bases in the primary structure.

        Returns: The `shadow` argument from element `0` to `length`.
        """
        return shadow[:length]


class Bracket:
    """Evaluate predictions made with the bracket notation."""

    def convert_to_scalars(
            true: list[str], pred: list[str], symbols: tuple[str]
        ) -> tuple:
        """Convert a vector of vectors into a vector of scalars.
        For instance, `[[0, 1], [0, 1], [1, 0]]` and `['.', '.', '(']`
        are converted to `[0, 0, 1]`.

        Args:
            true (list-like): Vector of the true structure.
            pred (list-like): Vector of the predicted structure.
            symbols: Set of possible elements.

        Returns (list): Tuple containing the scalar vectors.
        """
        if type(true[0]) == str:
            digits = {}
            for i, s in enumerate(symbols):
                digits[s] = i
            pred = [digits[e] for e in pred]
            true = [digits[e] for e in true]
            return true, pred, symbols
        return train.categorize_vector(true), train.categorize_vector(pred), None

    def micro_f1(
            true: list[str], pred: list[str], symbols: str = "(.)"
        ) -> float:
        """ Compute the micro F1-score by considering the secondary
        structure symbols '(', '.', and ')' as three different classes.

        Args:
            true (list-like): Vector of the true structure.
            pred (list-like): Vector of the predicted structure.
            symbols: Set of possible elements.

        Returns (float): F1-score of the prediction, i.e. a value
            between 0 and 1.
        """
        scalar_true, scalar_pred, _ = Bracket.convert_to_scalars(
            true, pred, symbols
        )
        return f1_score(scalar_pred, scalar_true, average='micro')

    def confusion_matrix(
            true: list[str], pred: list[str], symbols: str = "(.)"
        ) -> float:
        """Get the confusion matrix of the prediction.

        Args:
            true (list-like): Vector of the true structure.
            pred (list-like): Vector of the predicted structure.
            symbols: Set of possible elements.

        Returns (tuple): A tuple containing the confusion matrix and a
            list of symbols that correspond to each row of the matrix.
        """
        scalar_true, scalar_pred, _ = Bracket.convert_to_scalars(
            true, pred, symbols
        )
        return confusion_matrix(scalar_true, scalar_pred), symbols

    def crop(bracket: list[str | int], length: int) -> list[str | int]:
        """Return a cropped secondary structure to exclude padding.

        Args:
            bracket: Bracket notation of the secondary structure.
            length: Number of bases in the primary structure.

        Returns: The `bracket` argument from element `0` to `length`.
        """
        return bracket[:length]


class ContactMatrix:
    """Evaluate predictions made with contact matrices."""

    def TP(true: np.ndarray, pred: np.ndarray) -> int:
        """Get the number of true positives."""
        return np.sum(true * pred)

    def FP(true: np.ndarray, pred: np.ndarray) -> int:
        """Get the number of false positives."""
        return np.sum((np.ones_like(true) - true) * pred)

    def TN(true: np.ndarray, pred: np.ndarray) -> int:
        """Get the number of true negatives."""
        return np.sum((np.ones_like(true) - true) * (np.ones_like(pred) - pred))

    def FN(true: np.ndarray, pred: np.ndarray) -> int:
        """Get the number of false negatives."""
        return np.sum(true * (np.ones_like(pred) - pred))

    def precision(true: np.ndarray, pred: np.ndarray) -> float:
        """Compute the precision obtained by comparing two secondary
        structures. Precision is defined as:

        .. math::

            TP / (TP + FP).
        """
        tp = ContactMatrix.TP(true, pred)
        fp = ContactMatrix.FP(true, pred)
        if tp + fp:
            return tp / (tp + fp)
        else:
            return 0.0

    def recall(true: np.ndarray, pred: np.ndarray) -> float:
        """Compute the recall obtained by comparing two secondary
        structures. Precision is defined as:

        .. math::

            TP / (TP + FN).
        """
        tp = ContactMatrix.TP(true, pred)
        fn = ContactMatrix.FN(true, pred)
        if tp + fn:
            return tp / (tp + fn)
        else:
            return 0.0

    def f1(true, pred) -> float:
        """Compute the F1 score, a harmonic mean of precision and
        recall.
        """
        r = ContactMatrix.recall(true, pred)
        p = ContactMatrix.precision(true, pred)
        if r + p:
            f1 = 2 * ((r * p) / (r + p))
            return f1
        else:
            return 0.0

    def crop(contact: np.ndarray, length: int) -> list[int]:
        """Return a cropped contact matrix to exclude padding.

        Args:
            contact: Contact matrix of the secondary structure.
            length: Number of bases in the primary structure.

        Returns: The `length` by `length` upper left square of the
            contact matrix.
        """
        return contact[:length, :length]


# Result presentation
def summarize_results(f1_scores: list, name: str) -> None:
    """Summarize the f1-scores.

    Args:
        f1_scores (list(float)): List of f1-scores.
        name (str): Name of the results printed along with the summary.
    """
    log.info(f"Results for `{name}`:")
    log.trace(f"Number of elements: {len(f1_scores)}")
    log.trace(f"Mean: {np.mean(f1_scores)}")
    log.trace(f"Harmonic mean: {statistics.harmonic_mean(f1_scores)}")
    log.trace(f"Maximum: {max(f1_scores)}")
    log.trace(f"Median:  {np.median(f1_scores)}")
    log.trace(f"Minimum: {min(f1_scores)}")
