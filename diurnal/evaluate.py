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


class Vector:
    """Evaluate secondary structure predictions for vectors."""
    def _convert_to_scalars(true, pred) -> tuple:
        """Convert a vector of vectors into a vector of scalars.

        For instance, `[[0, 1], [0, 1], [1, 0]]` and `['.', '.', '(']`
        are converted to `[0, 0, 1]`.

        Args:
            true (list-like): Vector of the true structure.
            pred (list-like): Vector of the predicted structure.

        Returns (list): Tuple containing the scalar vectors.
        """
        symbols = set(pred + true)
        digits = {}
        for i, s in enumerate(symbols):
            digits[s] = i
        pred = [digits[e] for e in pred]
        true = [digits[e] for e in true]
        return true, pred, symbols

    def get_f1(true, pred) -> float:
        """ Compute the micro F1-score by considering the secondary
        structure symbols '(', '.', and ')' as three different classes.

        Args:
            true (list-like): Vector of the true structure.
            pred (list-like): Vector of the predicted structure.

        Returns (float): F1-score of the prediction, i.e. a value
            between 0 and 1.
        """
        scalar_true, scalar_pred, _ = Vector._convert_to_scalars(true, pred)
        return f1_score(scalar_pred, scalar_true, average='micro')

    def get_confusion_matrix(true, pred) -> tuple:
        """Get the confusion matrix of the prediction.

        Args:
            true (list-like): Vector of the true structure.
            pred (list-like): Vector of the predicted structure.

        Returns (tuple): A tuple containing the confusion matrix and a
            list of symbols that correspond to each row of the matrix.
        """
        t, p, symbols = Vector._convert_to_scalars(true, pred)
        return confusion_matrix(t, p), symbols


class Matrix:
    """Evaluate predictions made with matrices."""
    pass


class TwoClassVector:
    """Evaluate predictions by considering the paired and unpaired bases
    of an RNA secondary structures.

    The methods in this class take as an input secondary structures
    formatted in bracket notation and evaluate the predictions by making
    no difference between thw `(` and `)` symbols.

    These metrics are derived from the work of Wang et al., ATTFold.
    """
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

    def get_recall(
            true, pred, unpaired_symbol: any = Schemes.BRACKET_TO_ONEHOT['.']
            ) -> float:
        """Compute the recall value obtained by comparing two
        secondary structures. Recall is defined as:

        .. math::

            TP / (TP + FN).
        """
        tp = TwoClassVector.get_TP(true, pred, unpaired_symbol)
        fn = TwoClassVector.get_FN(true, pred, unpaired_symbol)
        if tp + fn:
            return tp / (tp + fn)
        else:
            return 0.0

    def get_precision(
            true, pred, unpaired_symbol: any = Schemes.BRACKET_TO_ONEHOT['.']
            ) -> float:
        """Compute the precision obtained by comparing two
        secondary structures. Precision is defined as:

        .. math::

            TP / (TP + FP).
        """
        tp = TwoClassVector.get_TP(true, pred, unpaired_symbol)
        fp = TwoClassVector.get_FP(true, pred, unpaired_symbol)
        if tp + fp:
            return tp / (tp + fp)
        else:
            return 0.0

    def get_recall_precision_f1(true, pred, unpaired_symbol="."):
        """Compute the F1-score obtained by comparing two secondary
        structures. The f1-score is defined as:

        .. math::

            F1 = 2 \times \frac{recall \times precision}{recall + precision}
        """
        recall = TwoClassVector.get_recall(true, pred, unpaired_symbol)
        precision = TwoClassVector.get_precision(true, pred, unpaired_symbol)
        if recall + precision:
            f1 = 2 * ((recall*precision) / (recall+precision))
            return recall, precision, f1
        else:
            return recall, precision, 0.0


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
