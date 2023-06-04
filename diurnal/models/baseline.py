"""
    Baseline models to ensure the validity of predictions.

    This module contains functions that predict the secondary structure
    of RNA molecules without any knowledge of the primary structure. For
    example, it predicts all nucleotides as unpaired or randomly assigns
    a result to each nucleotide. The purpose of this module is to
    provide baseline results to demonstrate that predictive models offer
    higher performances than chance.

    Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    Affiliation: Département d'informatique, UQÀM
    File creation date: June 2023
    License: MIT
"""

from random import choice

def random_prediction(size: int, symbols: list):
    """Predict a random symbol for each base.

    Args:
        size (int): Number of bases.
        symbols (list): Set of symbols of the prediction.

    Returns (list):
        Random prediction.
    """
    return [choice(symbols) for _ in range(size)]
