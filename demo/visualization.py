"""
    Demonstration for RNA databases visualization.

    Assumes that the directory "data/formatted" contains formatted RNA
    structure data. This dataset can be generated by executing the
    script `demo/preprocessing.py`.
"""

import numpy as np

from diurnal import visualize, structure


visualize.structure_length_per_family("data/formatted")

matrices = np.load(
    "data/formatted_matrix/primary_structures.npy", mmap_mode='r')
example = structure.Primary.unpad_matrix(matrices[0])
visualize.potential_pairings(example)
