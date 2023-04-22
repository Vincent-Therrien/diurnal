"""
    Test te diurnal.evaluate module.
"""

import pytest

import diurnal.evaluate as evaluate

@pytest.mark.parametrize(
    "pred, true, S, P, F",
    [
        ("(((...)))", "(((...)))", 1.0, 1.0, 1.0),
        ("(((...(((", "(((...)))", 1.0, 1.0, 1.0),
        ("(((......", "(((...)))", 0.5, 1.0, 2/3),
    ]
)
def test_sen_PPV_f1(pred, true, S, P, F):
    """
    Test the validity of sensitivity and positive predictive value
    evaluation criteria.

    Args:
        pred: Predicted secondary structure.
        true: True secondary structure.
        S: Expected sensitivity.
        P: Expected positive predictive value.
        F: Expected F-score.
    """
    s, p, f = evaluate.get_sen_PPV_f1(pred, true)
    assert s == S
    assert p == P
    assert f == F

@pytest.mark.parametrize(
    "pred, true, F",
    [
        ("(((...)))", "(((...)))", 1.0),
        ("(((...(((", "(((...)))", 7/9),
        ("(((......", "(((...)))", 7/9),
    ]
)
def test_three_class_f1_score(pred, true, F):
    """
    Test the validity of sensitivity and positive predictive value
    evaluation criteria.

    Args:
        pred: Predicted secondary structure.
        true: True secondary structure.
        F: Expected F-score.
    """
    f = evaluate.three_class_f1_score(pred, true)
    assert f == F
