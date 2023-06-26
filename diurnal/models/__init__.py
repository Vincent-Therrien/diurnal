"""
    Diurnal model package.

    The package contains models designed to predict RNA secondary
    structures from primary structures. The models of the package
    inherit the class `diurnal.models.Basic`, which defines the
    following interface:

    - `train(data) -> None`
      Train a model with primary and secondary structures.
    - `predict(primary_structure) -> secondary_structure`
      Predict a secondary structure from a primary structure.
    - `test(primary_structures, evaluation) -> list`
      Evaluate the performance of the model.
    - `save(directory) -> None`
      Write the model into a directory.
    - `load(directory) -> None`
      Read a model from a directory.

    Subclasses must implement the following methods:

    - `_train(primary, secondary) -> None` (train the model)
    - `_predict(primary) -> np.array` (make a prediction)
    - `_save(directory) -> None` (save the model)
    - `_load(directory) -> None` (load a model from files)

    Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    Affiliation: Département d'informatique, UQÀM
    File creation date: June 2023
    License: MIT
"""

from typing import Callable
import numpy as np
from datetime import datetime

from diurnal import evaluate, train, structure
import diurnal.utils.file_io as file_io

__all__ = ["baseline"]


class Basic():
    """Diurnal basic RNA secondary structure prediction model."""
    def train(self, data: dict) -> None:
        """Train the model. Abstract method.

        Args:
            data (dict): Training data organized as:

                .. block::

                    {
                        "primary_structures": <vector>,
                        "secondary_structures": <vector>,
                        "names": List[str]
                        # The `families` field is not mandatory.
                        "families": <vector>
                    }
        """
        self.names = data["names"]
        self.primary = data["primary_structures"]
        self.secondary = data["secondary_structures"]
        args = [self.primary, self.secondary]
        if "families" in data:
            self.families = data["families"]
            args.append(self.families)
        self._train(*args)

    def predict(self, primary) -> np.array:
        """Predict a random secondary structure.

        Args:
            primary: RNA primary structure.

        Returns (np.array): Predicted structure.
        """
        return self._predict(primary)

    def save(self, directory: str) -> None:
        """Write a model into the filesystem.

        Args:
            directory (str): Directory into which the model is written.
        """
        directory = file_io.clean_dir_path(directory)
        with open(directory + "training_molecule_list.txt", "w") as f:
            f.writelines(self.names)
        with open(directory + "ìnfo.rst", "w") as f:
            f.writelines(
                [
                    f"RNA Secondary Structure Prediction Model",
                    f"========================================",
                    f"",
                    f"Generation timestamp: {datetime.utcnow()} UTC",
                    f"",
                    f"Training data listed in ``training_molecule_list.txt``."
                ]
            )
        self._save(directory)

    def load(self, directory: str) -> None:
        """Read a model from a directory.

        Args:
            directory (str): Directory into which the model is written.
        """
        directory = file_io.clean_dir_path(directory)
        with open(directory + "training_molecule_list.txt", "r") as f:
            self.names - f.read().split('\n')
        self._load(directory)

    def test(self, data: list,
            evaluation: Callable = evaluate.Vector.get_f1) -> list:
        """Evaluate the performance of the model.

        Args:
            data (list): List of primary structures for predictions.
            evaluation (Callable): Evaluation function.

        Returns (list): The evaluation obtained for each structure.
        """
        results = []
        n = len(data["primary_structures"])
        for i in range(n):
            primary = data["primary_structures"][i]
            true = data["secondary_structures"][i]
            pred = self.predict(primary)
            _, true, pred = train.clean_vectors(primary, true, pred)
            true = structure.Secondary.to_bracket(true)
            pred = structure.Secondary.to_bracket(pred)
            results.append(evaluation(true, pred))
        return results
