"""
    Test segmentation.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: June 2024
    - License: MIT
"""

import numpy as np

from diurnal import segment


def test_sampling():
    a = np.array([
        [1, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 3]
    ])
    areas = segment.sample_areas(a, 2)
    assert areas == [((6, 6), 3), ((0, 0), 1), ((0, 2), 1)]
    areas = segment.sample_areas(a, 2, 2)
    assert areas == [((6, 6), 3)]
    areas = segment.sample_areas(a, 4, stride = 2)
    assert areas == [((4, 4), 3), ((0, 0), 2), ((0, 2), 1)]
