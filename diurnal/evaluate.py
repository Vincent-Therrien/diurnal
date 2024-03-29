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
from diurnal.structure import Schemes
from diurnal import train


def _convert_to_scalars(true, pred) -> tuple:
    """Convert a vector of vectors into a vector of scalars.
    For instance, `[[0, 1], [0, 1], [1, 0]]` and `['.', '.', '(']`
    are converted to `[0, 0, 1]`.

    Args:
        true (list-like): Vector of the true structure.
        pred (list-like): Vector of the predicted structure.

    Returns (list): Tuple containing the scalar vectors.
    """
    if type(true[0]) == str:
        symbols = set(pred + true)
        digits = {}
        for i, s in enumerate(symbols):
            digits[s] = i
        pred = [digits[e] for e in pred]
        true = [digits[e] for e in true]
        return true, pred, symbols
    return train.categorize_vector(true), train.categorize_vector(pred), None


def micro_f1(true, pred) -> float:
    """ Compute the micro F1-score by considering the secondary
    structure symbols '(', '.', and ')' as three different classes.

    Args:
        true (list-like): Vector of the true structure.
        pred (list-like): Vector of the predicted structure.

    Returns (float): F1-score of the prediction, i.e. a value
        between 0 and 1.
    """
    scalar_true, scalar_pred, _ = _convert_to_scalars(true, pred)
    return f1_score(scalar_pred, scalar_true, average='micro')


def get_confusion_matrix(true, pred) -> tuple:
    """Get the confusion matrix of the prediction.

    Args:
        true (list-like): Vector of the true structure.
        pred (list-like): Vector of the predicted structure.

    Returns (tuple): A tuple containing the confusion matrix and a
        list of symbols that correspond to each row of the matrix.
    """
    t, p, symbols = _convert_to_scalars(true, pred)
    return confusion_matrix(t, p), symbols


class ContactMatrix:
    """Evaluate predictions made with matrices."""
    def TP(true, pred) -> int:
        return np.sum(true * pred)

    def FP(true, pred) -> int:
        return np.sum((np.ones_like(true) - true) * pred)

    def TN(true, pred) -> int:
        return np.sum((np.ones_like(true) - true) * (np.ones_like(pred) - pred))

    def FN(true, pred) -> int:
        return np.sum(true * (np.ones_like(pred) - pred))

    def precision(true, pred) -> float:
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

    def recall(true, pred) -> float:
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


def get_TP(
        true, pred, unpaired_symbol: any = Schemes.BRACKET_TO_ONEHOT['.']
        ) -> float:
    """Compute the true positive value (predicted paired bases that
    are actually paired).
    """
    tp = 0
    for i in range(len(true)):
        if pred[i] != unpaired_symbol and true[i] != unpaired_symbol:
            tp += 1
    return tp

def get_TN(
        true, pred, unpaired_symbol: any = Schemes.BRACKET_TO_ONEHOT['.']
        ) -> float:
    """Compute the true negative value (predicted unpaired bases
    that are actually unpaired).
    """
    tn = 0
    for i in range(len(true)):
        if pred[i] == unpaired_symbol and true[i] == unpaired_symbol:
            tn += 1
    return tn

def get_FP(
        true, pred, unpaired_symbol: any = Schemes.BRACKET_TO_ONEHOT['.']
        ) -> float:
    """Compute the false positive value (predicted paired bases that
    are actually unpaired).
    """
    fp = 0
    for i in range(len(true)):
        if pred[i] != unpaired_symbol and true[i] == unpaired_symbol:
            fp += 1
    return fp

def get_FN(
        true, pred, unpaired_symbol: any = Schemes.BRACKET_TO_ONEHOT['.']
        ) -> float:
    """Compute the false negative value (predicted unpaired bases
    that are actually unpaired).
    """
    fn = 0
    for i in range(len(true)):
        if pred[i] == unpaired_symbol and pred[i] != true[i]:
            fn += 1
    return fn

def recall(
        true, pred, unpaired_symbol: any = Schemes.BRACKET_TO_ONEHOT['.']
        ) -> float:
    """Compute the recall value obtained by comparing two
    secondary structures. Recall is defined as:

    .. math::

        TP / (TP + FN).
    """
    tp = get_TP(true, pred, unpaired_symbol)
    fn = get_FN(true, pred, unpaired_symbol)
    if tp + fn:
        return tp / (tp + fn)
    else:
        return 0.0

def precision(
        true, pred, unpaired_symbol: any = Schemes.BRACKET_TO_ONEHOT['.']
        ) -> float:
    """Compute the precision obtained by comparing two
    secondary structures. Precision is defined as:

    .. math::

        TP / (TP + FP).
    """
    tp = get_TP(true, pred, unpaired_symbol)
    fp = get_FP(true, pred, unpaired_symbol)
    if tp + fp:
        return tp / (tp + fp)
    else:
        return 0.0

def recall_precision_f1(true, pred, unpaired_symbol="."):
    """Compute the F1-score obtained by comparing two secondary
    structures. The f1-score is defined as:

    .. math::

        F1 = 2 \times \frac{recall \times precision}{recall + precision}
    """
    r = recall(true, pred, unpaired_symbol)
    p = precision(true, pred, unpaired_symbol)
    if r + p:
        f1 = 2 * ((r * p) / (r + p))
        return r, p, f1
    else:
        return r, p, 0.0


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
