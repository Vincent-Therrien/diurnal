"""
    RNA secondary structure prediction model package.

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

    - `_train(data) -> None` (train the model)
    - `_predict(primary) -> np.ndarray` (make a prediction)
    - `_save(directory) -> None` (save the model)
    - `_load(directory) -> None` (load a model from files)

    File information:

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: June 2023
    - License: MIT
"""

from typing import Callable
from datetime import datetime

import numpy as np

from diurnal import evaluate, train
from diurnal.utils import file_io, log

__all__ = ["deep", "reinforcement", "baseline"]


class Basic():
    """Diurnal basic RNA secondary structure prediction model."""
    def train(
            self, training_data: dict, validation_data: dict = None,
            verbose: bool = True) -> None:
        """Train the model. Abstract method.

        Args:
            training_data (dict): Training data organized as:

                .. block::

                    {
                        "input": tuple(<vector>),
                        "output": <vector>,
                        "names": List[str]
                    }

            validation_data (dict)
            verbose (bool): Print informative messages.
        """
        if verbose:
            N = len(training_data["output"])
            log.info(f"Training the model with {N} data points.")
            if validation_data:
                n = len(validation_data["output"])
                log.trace(f"Using {n} data points for validation.")
        self.names = training_data["names"]
        self.input = training_data["input"]
        self.output = training_data["output"]
        self.N = len(self.output)
        if type(self.input) != tuple:
            log.error(f"The `input` must be a tuple, got {type(self.input)}")
            raise RuntimeError
        self.input_shape = self.input[0][0].shape
        self.length = len(self.output[0])
        self.output_shape = self.output[0].shape
        self.validate = False
        self.validation_names = None
        self.validation_input = None
        self.validation_output = None
        self.validation_N = None
        if validation_data:
            self.validate = True
            self.validation_names = validation_data["names"]
            self.validation_input = validation_data["input"]
            self.validation_output = validation_data["output"]
            self.validation_N = len(self.validation_output)
            if len(self.validation_input) == self.validation_N:
                self.validation_input = [self.validation_input]
        self._train()

    def predict(self, input) -> np.ndarray:
        """Predict a random secondary structure.

        Args:
            input: RNA primary structure data.

        Returns (np.ndarray): Predicted structure.
        """
        return self._predict(input)

    def save(self, directory: str, verbose: bool = True) -> None:
        """Write a model into the filesystem.

        Args:
            directory (str): Directory into which the model is written.
            verbose (bool): Print informative messages.
        """
        directory = file_io.clean_dir_path(directory)
        if verbose:
            log.info(f"Saving the model at `{directory}`.")
        with open(directory + "training_molecule_list.txt", "w") as f:
            for name in self.names:
                f.write(name + "\n")
        with open(directory + "ìnfo.rst", "w") as f:
            f.writelines([
                "RNA Secondary Structure Prediction Model\n",
                "========================================\n",
                "\n",
                f"Generation timestamp: {datetime.utcnow()} UTC\n",
                "\n",
                "Training data listed in ``training_molecule_list.txt``.\n",
                f"{len(self.output)} molecules were used for training.\n\n",
                f"Input shape: {self.input_shape}\n",
                f"Output shape: {self.output_shape}\n"])
        self._save(directory)

    def load(self, directory: str, verbose: bool = True) -> None:
        """Read a model from a directory.

        Args:
            directory (str): Directory into which the model is written.
            verbose (bool): Print informative messages.
        """
        directory = file_io.clean_dir_path(directory)
        if verbose:
            log.info(f"Loading the model from the files at `{directory}`.")
        with open(directory + "training_molecule_list.txt", "r") as f:
            self.names = f.read().split('\n')
        self._load(directory)

    def test(
            self, data: list,
            evaluation: Callable = evaluate.Bracket.micro_f1,
            verbose: bool = True) -> list:
        """Evaluate the performance of the model.

        Args:
            data (list): List of primary structures for predictions.
            map (Callable): A function that converts the output of the
                model into a secondary structure. `None` for no
                transformation.
            evaluation (Callable): Evaluation function.
            verbose (bool): Print informative messages.

        Returns (list): The evaluation obtained for each structure.
        """
        results = []
        n = len(data["output"])
        if verbose:
            log.info(f"Testing the model with {n} data points.")
        for i in range(n):
            n_args = len(data["input"])
            input = tuple([data["input"][x][i] for x in range(n_args)])
            true = data["output"][i]
            pred = self.predict(input)
            if (len(input[0].shape)) == 2:
                _, true, pred = train.clean_vectors(input[0], true, pred)
            elif (len(input[0].shape)) == 3:
                _, true, pred = train.clean_matrices(input[0], true, pred)
            results.append(evaluation(true, pred))
        return results
