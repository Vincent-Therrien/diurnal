"""
    Test te diurnal.evaluate module.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: April 2023
    - License: MIT
"""

import pytest
import numpy as np

from diurnal import evaluate


@pytest.mark.parametrize(
    "true, pred, R, P, F",
    [
        (list("(((...)))"), list("(((...)))"), 1.0, 1.0, 1.0),
        (list("(((...)))"), list("(((...((("), 1.0, 1.0, 1.0),
        (list("(((...)))"), list("(((......"), 0.5, 1.0, 2/3),
    ]
)
def test_two_class_metrics(true, pred, R, P, F):
    """
    Test the validity of recall and precision evaluation criteria.

    Args:
        true: True secondary structure.
        pred: Predicted secondary structure.
        R: Expected recall.
        P: Expected precision.
        F: Expected F-score.
    """
    r, p, f = evaluate.recall_precision_f1(true, pred)
    assert r == R, "Incorrect sensitivity."
    assert p == P, "Incorrect positive predictive value."
    assert f == F, "Incorrect F1-score."


@pytest.mark.parametrize(
    "true, pred, F",
    [
        (list("(((...)))"), list("(((...)))"), 1.0),
        (list("(((...)))"), list("(((...((("), 2/3),
        (list("(((...)))"), list("(((......"), 2/3),
    ]
)
def test_micro_f1(pred, true, F):
    """
    Test the validity of sensitivity and positive predictive value
    evaluation criteria.

    Args:
        true: True secondary structure.
        pred: Predicted secondary structure.
        F: Expected F-score.
    """
    f = evaluate.micro_f1(true, pred)
    assert f == F, "Incorrect F1-score."


@pytest.mark.parametrize(
    "true, pred, is_diagonal",
    [
        ("(((...)))", "(((...)))", True),
        ("(((...)))", "(((...(((", False),
        ("(((...)))", "(((......", False),
    ]
)
def test_confusion_matrix(pred, true, is_diagonal):
    """
    Test the confusion matrix of the vector-based evaluations.

    Args:
        true: True secondary structure.
        pred: Predicted secondary structure.
        is_diagonal (bool): True if the expected confusion matrix is
            diagonal, False otherwise.
    """
    cm, _ = evaluate.get_confusion_matrix(true, pred)
    cm_is_diagonal = np.count_nonzero(cm - np.diag(np.diagonal(cm))) == 0
    assert cm_is_diagonal == is_diagonal, \
        "Confusion matrix does not match the expected result."
