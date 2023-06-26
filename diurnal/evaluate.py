"""
    RNA secondary prediction evaluation module.

    This module contains functions to evaluate RNA secondary predictions
    by comparing a predicted structure to a reference structure.

    Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    Affiliation: Département d'informatique, UQÀM
    File creation date: April 2023
    License: MIT
"""

import statistics
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix

from diurnal.utils import file_io, log
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
        scalar_true, scalar_pred, symbols=Vector._convert_to_scalars(true, pred)
        return confusion_matrix(scalar_true, scalar_pred), symbols


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
    def get_TP(true, pred, unpaired_symbol = Schemes.BRACKET_TO_ONEHOT['.']):
        """ Compute the true positive value (predicted paired bases that
        are actually paired).
        """
        tp = 0
        for i in range(len(true)):
            if pred[i] != unpaired_symbol and true[i] != unpaired_symbol:
                tp += 1
        return tp

    def get_TN(true, pred, unpaired_symbol = Schemes.BRACKET_TO_ONEHOT['.']):
        """Compute the true negative value (predicted unpaired bases
        that are actually unpaired).
        """
        tn = 0
        for i in range(len(true)):
            if pred[i] == unpaired_symbol and true[i] == unpaired_symbol:
                tn += 1
        return tn

    def get_FP(true, pred, unpaired_symbol = Schemes.BRACKET_TO_ONEHOT['.']):
        """Compute the false positive value (predicted paired bases that
        are actually unpaired).
        """
        fp = 0
        for i in range(len(true)):
            if pred[i] != unpaired_symbol and true[i] == unpaired_symbol:
                fp += 1
        return fp

    def get_FN(true, pred, unpaired_symbol = Schemes.BRACKET_TO_ONEHOT['.']):
        """Compute the false negative value (predicted unpaired bases
        that are actually unpaired).
        """
        fn = 0
        for i in range(len(true)):
            if pred[i] == unpaired_symbol and pred[i] != true[i]:
                fn += 1
        return fn

    def get_sensitivity(true, pred,
            unpaired_symbol=Schemes.BRACKET_TO_ONEHOT['.']):
        """Compute the sensitivity value (SEN) obtained by comparing two
        secondary structures. The sensitivity is defined as:

        .. :math:

            TP / (TP + FN).
        """
        tp = TwoClassVector.get_TP(true, pred, unpaired_symbol)
        fn = TwoClassVector.get_FN(true, pred, unpaired_symbol)
        if tp + fn:
            return tp / (tp + fn)
        else:
            return 0.0

    def get_PPV(true, pred, unpaired_symbol = Schemes.BRACKET_TO_ONEHOT['.']):
        """Compute the positive predictive value (PPV) obtained by
        comparing two secondary structures. The PPV is defined as:

        .. :math:

            TP / (TP + FP).
        """
        tp = TwoClassVector.get_TP(true, pred, unpaired_symbol)
        fp = TwoClassVector.get_FP(true, pred, unpaired_symbol)
        if tp + fp:
            return tp / (tp + fp)
        else:
            return 0.0

    def get_sen_PPV_f1(true, pred, unpaired_symbol="."):
        """Compute the F1-score obtained by comparing two secondary
        structures. The f1-score is defined as:

        .. :math:

            2 * ((SEN*PPV) / (SEN+PPV)).

        Also return the sensitivity and precision.
        """
        sen = TwoClassVector.get_sensitivity(true, pred, unpaired_symbol)
        ppv = TwoClassVector.get_PPV(true, pred, unpaired_symbol)
        if sen + ppv:
            f1 = 2 * ((sen*ppv) / (sen+ppv))
            return sen, ppv, f1
        else:
            return sen, ppv, 0.0


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
