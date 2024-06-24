"""
    Test the diurnal.transform module.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: May 2024
    - License: MIT
"""

import numpy as np

from diurnal import transform


def test_linearize():
    a = np.array([
        [0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [2, 3, 0, 0, 0, 0],
        [4, 5, 6, 0, 0, 0],
        [7, 8, 9, 1, 0, 0],
        [2, 3, 4, 5, 6, 0],
    ])
    # Test linearization.
    result = transform.linearize_half_matrix(a, 2, offset=1)
    assert np.array_equal(result, np.array([1]))
    result = transform.linearize_half_matrix(a, 3, offset=1)
    assert np.array_equal(result, np.array([1, 2, 3]))
    result = transform.linearize_half_matrix(a, 4, offset=1)
    assert np.array_equal(result, np.array([1, 2, 4, 3, 5, 6]))
    result = transform.linearize_half_matrix(a, 5, offset=1)
    assert np.array_equal(result, np.array([1, 2, 4, 3, 7, 5, 8, 6, 9, 1]))
    result = transform.linearize_half_matrix(a, 6, offset=1)
    assert np.array_equal(
        result,
        np.array([1, 2, 4, 3, 7, 5, 2, 8, 6, 3, 9, 4, 1, 5, 6])
    )
    # Test padding.
    result = transform.linearize_half_matrix(a, 4, N=10, offset=1)
    assert np.array_equal(result, np.array([1, 2, 4, 3, 5, 6, 0, 0, 0, 0]))
    # Test offsets.
    result = transform.linearize_half_matrix(a, 6, offset=2)
    assert np.array_equal(
        result,
        np.array([2, 4, 7, 5, 2, 8, 3, 9, 4, 5])
    )
    result = transform.linearize_half_matrix(a, 6, offset=3)
    assert np.array_equal(result, np.array([4, 7, 2, 8, 3, 4]))
    result = transform.linearize_half_matrix(a, 6, offset=4)
    assert np.array_equal(result, np.array([7, 2, 3]))
    result = transform.linearize_half_matrix(a, 6, offset=5)
    assert np.array_equal(result, np.array([2]))
    # Integration
    result = transform.linearize_half_matrix(a, 6, N=8, offset=4)
    assert np.array_equal(result, np.array([7, 2, 3, 0, 0, 0, 0, 0]))


def test_delinearize():
    a = np.array([1, 2, 4, 3, 5, 6])
    matrix = transform.delinearize_half_matrix(a, 4, offset=1)
    expected = np.array([
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [2, 3, 0, 0],
        [4, 5, 6, 0]]
    )
    assert np.array_equal(matrix, expected)
    matrix = transform.delinearize_half_matrix(a, 4, offset=2)
    expected = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [2, 4, 0, 0]]
    )
    assert np.array_equal(matrix, expected)
    matrix = transform.delinearize_half_matrix(a, 4, N=5, offset=2)
    expected = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [2, 4, 0, 0, 0],
        [0, 0, 0, 0, 0]]
    )
    assert np.array_equal(matrix, expected)


def test_collapse():
    a = np.array([2, 0, 0, 0, 1, 0, 0])
    collapsed = transform.collapse_linearized_matrix(a)
    expected = np.array([ 2, -3,  1, -2])
    assert np.array_equal(collapsed, expected)
    collapsed = transform.collapse_linearized_matrix(a, N=6)
    expected = np.array([ 2, -3,  1, -2,  0,  0])
    assert np.array_equal(collapsed, expected)


def test_decollapse():
    a = np.array([ 2, -3,  1, -2])
    collapsed = transform.decollapse_linearized_matrix(a, N_output=7)
    expected = np.array([2, 0, 0, 0, 1, 0, 0])
    assert np.array_equal(collapsed, expected)
    a = np.array([ 2, -3,  1, -2, 0, 0, 0])
    collapsed = transform.decollapse_linearized_matrix(a, N_input=4, N_output=7)
    expected = np.array([2, 0, 0, 0, 1, 0, 0])
    assert np.array_equal(collapsed, expected)


def test_monomial():
    a = np.array([
        [0, 2, 3, 2],
        [2, 7, 2, 2],
        [2, 2, 2, 2],
        [2, 2, 2, 9],
    ])
    monomial = transform.to_monomial_matrix(a)
    expected = np.array([
        [0, 0, 3, 0],
        [0, 7, 0, 0],
        [2, 0, 0, 0],
        [0, 0, 0, 9],
    ])
    assert np.array_equal(monomial, expected)
    binary = transform.to_binary_matrix(monomial)
    expected = np.array([
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
    ])
    assert np.array_equal(binary, expected)
