"""
    Test the diurnal.train module.

    Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    Affiliation: Département d'informatique, UQÀM
    File creation date: May 2023
    License: MIT
"""

import pytest

from diurnal.structure import Primary, Secondary
import diurnal.train as train


def test_split_data():
    """Validate that data can be split in subarrays."""
    N = 100
    data = list(range(N))
    fractions = [0.8, 0.1, 0.1]
    offset = 5
    subarrays = train.split_data(data, fractions, offset)
    beginning = offset
    for s, f in zip(subarrays, fractions):
        expected_array = []
        for e in range(beginning, beginning + int(f*N)):
            if e >= N:
                expected_array.append(e - N)
            else:
                expected_array.append(e)
        beginning += int(f*N)
        assert s == expected_array


def test_categorize_vectors():
    """Ensure that predictions are correctly converted into one-hot
    vectors."""
    expected = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ]
    test_vector = [
        [0.9, 0.5, 0.1],
        [0.0, 0.1, 0.05],
        [0.0, 0.0, 0.8]
    ]
    assert train.categorize_vector(test_vector) == expected, \
        "Prediction vector is incorrectly categorized."

@pytest.mark.parametrize(
    "bases, true, pred",
    [
        (
            list("AAACCCUUUCC"),
            [8, 7, 6, -1, -1, -1, 2, 1, 0, -1, -1], # (((...)))..
            [8, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1] # (((........
        ),
        (
            list("AAACCCUUUCC-----"),
            [8, 7, 6, -1, -1, -1, 2, 1, 0, -1, -1], # (((...)))..
            [8, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1] # (((........
        )
    ]
)
def test_prediction_post_processing(bases, true, pred):
    """Ensure that the ending of predictions can be removed."""
    bases_v = Primary.to_vector(bases)
    true_v  = Secondary.to_vector(true, len(bases))
    pred_v  = Secondary.to_vector(pred, len(bases))
    b, t, p = train.clean_vectors(bases_v, true_v, pred_v)
    assert len(b) == len(t) == len(p), "Non-homogeneous vector dimensions."

