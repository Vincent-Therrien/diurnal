"""
    Test te diurnal.evaluate module.

    Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    Affiliation: Département d'informatique, UQÀM
    File creation date: April 2023
    License: MIT
"""

import pytest
import numpy as np

import diurnal.evaluate as evaluate


@pytest.mark.parametrize(
    "true, pred, S, P, F",
    [
        ("(((...)))", "(((...)))", 1.0, 1.0, 1.0),
        ("(((...)))", "(((...(((", 1.0, 1.0, 1.0),
        ("(((...)))", "(((......", 0.5, 1.0, 2/3),
    ]
)
def test_two_class_metrics(true, pred, S, P, F):
    """
    Test the validity of sensitivity and positive predictive value
    evaluation criteria.

    Args:
        true: True secondary structure.
        pred: Predicted secondary structure.
        S: Expected sensitivity.
        P: Expected positive predictive value.
        F: Expected F-score.
    """
    s, p, f = evaluate.TwoClassVector.get_sen_PPV_f1(true, pred)
    assert s == S, "Incorrect sensitivity."
    assert p == P, "Incorrect positive predictive value."
    assert f == F, "Incorrect F1-score."


@pytest.mark.parametrize(
    "true, pred, F",
    [
        ("(((...)))", "(((...)))", 1.0),
        ("(((...)))", "(((...(((", 2/3),
        ("(((...)))", "(((......", 2/3),
    ]
)
def test_vector_f1(pred, true, F):
    """
    Test the validity of sensitivity and positive predictive value
    evaluation criteria.

    Args:
        true: True secondary structure.
        pred: Predicted secondary structure.
        F: Expected F-score.
    """
    f = evaluate.Vector.get_f1(true, pred)
    assert f == F, "Incorrect F1-score."


@pytest.mark.parametrize(
    "true, pred, is_diagonal",
    [
        ("(((...)))", "(((...)))", True),
        ("(((...)))", "(((...(((", False),
        ("(((...)))", "(((......", False),
    ]
)
def test_vector_confusion_matrix(pred, true, is_diagonal):
    """
    Test the confusion matrix of the vector-based evaluations.

    Args:
        true: True secondary structure.
        pred: Predicted secondary structure.
        is_diagonal (bool): True if the expected confusion matrix is
            diagonal, False otherwise.
    """
    cm, _ = evaluate.Vector.get_confusion_matrix(true, pred)
    cm_is_diagonal = np.count_nonzero(cm - np.diag(np.diagonal(cm))) == 0
    assert cm_is_diagonal == is_diagonal, \
        "Confusion matrix does not match the expected result."
