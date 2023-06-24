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
import numpy as np


class Random():
    """Baseline model that makes random predictions."""
    def __init__(self) -> None:
        self.symbols = []

    def train(self, primary, secondary) -> None:
        """Simulate a training of the model."""
        for symbol in secondary[0]:
            symbol = symbol.tolist()
            if symbol not in self.symbols:
                self.symbols.append(symbol)

    def predict(self, primary) -> np.array:
        """Predict a random secondary structure. Argument is ignored."""
        return [choice(self.symbols) for _ in range(len(primary))]


class Uniform():
    """Baseline model that predicts a uniform vector."""
    def __init__(self, symbol) -> None:
        self.symbol = symbol

    def train(self, primary, secondary) -> None:
        """Simulate a training of the model."""
        pass

    def predict(self, primary) -> np.array:
        """Predict a random secondary structure. Argument is ignored."""
        return [self.symbol for _ in range(len(primary))]
