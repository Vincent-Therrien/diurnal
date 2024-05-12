"""
    Baseline models to ensure the validity of predictions.

    This module contains models that predict the secondary structure
    of RNA molecules without any knowledge of the primary structure. For
    example, they may predict all nucleotides as unpaired or randomly
    assign a prediction to each nucleotide. The purpose of this module
    is to provide baseline results to demonstrate that predictive models
    offer higher performances than chance.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: June 2023
    - License: MIT
"""

from random import choice
import numpy as np

from diurnal.models import Basic


class Random(Basic):
    """Baseline model that makes random predictions."""
    def __init__(self) -> None:
        self.symbols = []

    def _train(self) -> None:
        """Simulate a training of the model."""
        for symbol in self.output[0]:
            symbol = symbol.tolist()
            if symbol not in self.symbols and sum(symbol):
                self.symbols.append(symbol)

    def _predict(self, primary) -> np.ndarray:
        """Predict a random secondary structure."""
        return [choice(self.symbols) for _ in range(len(primary[0]))]

    def _save(self, directory) -> None:
        """Since the model is entirely random, no data is saved."""
        pass

    def _load(self, directory) -> None:
        """Since the model is entirely random, no data is loaded."""
        pass


class Uniform(Basic):
    """Baseline model that predicts a uniform vector."""
    def __init__(self, symbol) -> None:
        self.symbol = symbol

    def _train(self) -> None:
        """Simulate a training of the model."""
        pass

    def _predict(self, primary) -> np.ndarray:
        """Predict a random secondary structure."""
        return [self.symbol for _ in range(len(primary[0]))]

    def _save(self, directory) -> None:
        """Save the model."""
        np.save(directory + "model.npy", np.array(self.symbol))

    def _load(self, directory) -> None:
        """Write the model."""
        self.symbol = np.load(directory + "model.npy", mmap_mode='r')


class Majority(Basic):
    """Baseline model that predicts a uniform vector whose elements are
    the most common element in the training data.
    """
    def __init__(self) -> None:
        self.symbol = None

    def _train(self) -> None:
        """Simulate a training of the model."""
        symbols = {}
        for structure in self.output:
            for element in structure:
                if sum(element) == 0:
                    continue
                # Since lists are unhashable, use their string representation
                # to count their occurrences.
                symbol = repr(element.tolist())
                if symbol in symbols:
                    symbols[symbol] += 1
                else:
                    symbols[symbol] = 1
        k = list(symbols.keys())
        v = list(symbols.values())
        exec(f"self.symbol = {k[v.index(max(v))]}")

    def _predict(self, primary) -> np.ndarray:
        """Predict a random secondary structure."""
        return [self.symbol for _ in range(len(primary[0]))]

    def _save(self, directory) -> None:
        """Save the model."""
        np.save(directory + "model.npy", np.ndarray(self.symbol))

    def _load(self, directory) -> None:
        """Write the model."""
        self.symbol = np.load(directory + "model.npy", mmap_mode='r')
