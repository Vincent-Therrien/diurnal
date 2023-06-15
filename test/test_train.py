"""
    Test the diurnal.train module.

    Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    Affiliation: Département d'informatique, UQÀM
    File creation date: May 2023
    License: MIT
"""

from diurnal.structure import Secondary
import diurnal.train as train

def test_split_data():
    """
    Validate that data can be split in subarrays.
    """
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

def test_prediction_post_processing_str():
    """
    Ensure that the ending of predictions can be removed.
    """
    true_pairings = [8, 7, 6, -1, -1, -1, 2, 1, 0, -1, -1] # (((...)))..
    size_A = len(true_pairings)
    true = Secondary.to_vector(true_pairings, size_A + 1)
    size_B = len(true)
    pred_pairings = [8, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1] # (((........
    pred = Secondary.to_vector(pred_pairings, size_B)
    assert len(true) == len(pred)
    true, pred = train.clean_true_pred(true, pred)
    assert len(true) == size_A and len(pred) == size_A

def test_prediction_post_processing_vector():
    """
    Ensure that the ending of predictions can be removed.
    """
    true_pairings = [8, 7, 6, -1, -1, -1, 2, 1, 0, -1, -1] # (((...)))..
    size_A = len(true_pairings)
    true = Secondary.to_vector(true_pairings, size_A + 1)
    size_B = len(true)
    pred_pairings = [8, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1] # (((........
    pred = Secondary.to_vector(pred_pairings, size_B)
    assert len(true) == len(pred)
    true, pred = train.clean_true_pred(true, pred)
    assert len(true) == size_A and len(pred) == size_A
