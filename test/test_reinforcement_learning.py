"""
    Test the diurnal.models.reinforcement module.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: May 2024
    - License: MIT
"""

import numpy as np

from diurnal.models import reinforcement


def test_free_pairings():
    """
    Ensure that free rows and columns can be correctly identified.
    """
    CONTACT = np.array([
        [0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ])
    FREE_ROWS = np.array([1, 0, 0, 1, 1, 1])
    FREE_COLUMNS = np.array([0, 1, 0, 0, 1, 1])
    assert np.array_equal(
        FREE_ROWS, reinforcement.ContactMatrix.get_free_rows(CONTACT)
    )
    assert np.array_equal(
        FREE_COLUMNS, reinforcement.ContactMatrix.get_free_columns(CONTACT)
    )


def test_insert():
    """Ensure that pairings are well inserted in a contact matrix."""
    CONTACT = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 1, 0]
    ])
    ROWS = np.array([0.0, 0.1, 0.8])
    COLUMNS = np.array([0.9, 0.1, 0.8])
    reinforcement.ContactMatrix.insert(CONTACT, ROWS, COLUMNS)
    assert np.array_equal(
        CONTACT,
        np.array([
            [0, 0, 0],
            [0, 0, 0],
            [1, 1, 0]
        ])
    )


def test_clear():
    """Ensure that pairings can be cleared properly."""
    CONTACT = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0]
    ])
    # Clear a row.
    ROWS = np.array([0.0, 0.1, 0.8])
    reinforcement.ContactMatrix.clear_row(CONTACT, ROWS)
    assert np.array_equal(
        CONTACT,
        np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 0, 0]
        ])
    )
    # Clear a column.
    COLUMNS = np.array([0.9, 0.99, 0.8])
    reinforcement.ContactMatrix.clear_column(CONTACT, COLUMNS)
    assert np.array_equal(
        CONTACT,
        np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 0, 0]
        ])
    )
    COLUMNS = np.array([0.9, 0.5, 0.8])
    reinforcement.ContactMatrix.clear_column(CONTACT, COLUMNS)
    assert np.array_equal(
        CONTACT,
        np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ])
    )
