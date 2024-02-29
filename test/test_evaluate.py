"""
    Test te diurnal.evaluate module.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: April 2023
    - License: MIT
"""

import pytest
import numpy as np
from random import uniform

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


def test_contact_matrix():
    true = np.array(
        [
            [1, 1, 0, 0, 0, 1],
            [1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]
    )
    pred = np.array(
        [
            [1, 1, 1, 0, 0, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]
    )
    assert evaluate.ContactMatrix.TP(true, pred) == 3
    assert evaluate.ContactMatrix.TN(true, pred) == 30
    assert evaluate.ContactMatrix.FP(true, pred) == 1
    assert evaluate.ContactMatrix.FN(true, pred) == 2
    assert evaluate.ContactMatrix.precision(true, pred) == 0.75
    assert evaluate.ContactMatrix.recall(true, pred) == 0.6
    assert abs(evaluate.ContactMatrix.f1(true, pred) - 2/3) < 0.001


def test_contact_matrix_cleanup():
    """Ensure that a contact score matrix can be quantized."""
    # Create the test matrix
    VALID_PAIRINGS = ((0, 19), (19, 0), (3, 17), (17, 3), (5, 14), (14, 5))
    INVALID_PAIRINGS = ((1, 18), (18, 2), (4, 18), (15, 4))
    matrix = np.zeros((20, 20))
    for i in range(20):
        for j in range(20):
            if (i, j) in (VALID_PAIRINGS + INVALID_PAIRINGS):
                matrix[i][j] = 1
            matrix[i][j] += uniform(0, 0.9)
    matrix = matrix.clip(0, 1)
    mask = np.zeros((20, 20))
    for i in range(20):
        for j in range(20):
            if (i, j) in (VALID_PAIRINGS + INVALID_PAIRINGS):
                mask[i][j] = 1
    inverse_mask = np.ones((20, 20)) - mask
