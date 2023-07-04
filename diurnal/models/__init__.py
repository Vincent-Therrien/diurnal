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
import torch.optim as optim
from torch import cuda, nn
from torch import load as torch_load
from torch import save as torch_save
from torch import from_numpy
from torch.utils.data import DataLoader

from diurnal import evaluate, train, structure
from diurnal.utils import file_io, log


__all__ = ["baseline", "networks"]


class Basic():
    """Diurnal basic RNA secondary structure prediction model."""
    def train(self, training_data: dict, validation_data: dict = None) -> None:
        """Train the model. Abstract method.

        Args:
            training_data (dict): Training data organized as:

                .. block::

                    {
                        "primary_structures": <vector>,
                        "secondary_structures": <vector>,
                        "names": List[str]
                        "families": <vector>
                    }

            validation_data (dict)
        """
        self.names = training_data["names"]
        self.primary = np.array(training_data["primary_structures"])
        self.secondary = np.array(training_data["secondary_structures"])
        self.families = np.array(training_data["families"])
        if validation_data:
            self.validation_names = validation_data["names"]
            self.validation_primary = validation_data["primary_structures"]
            self.validation_secondary = validation_data["secondary_structures"]
            self.validation_families = validation_data["families"]
        else:
            self.validation_names = None
            self.validation_primary = None
            self.validation_secondary = None
            self.validation_families = None
        self._train()

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

    def test(
            self, data: list,
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


class NN(Basic):
    """A model that relies on a neural network to make predictions."""
    def __init__(
            self, model: nn,
            N: int,
            n_epochs: int,
            optimizer: optim,
            loss_fn: nn.functional,
            optimizer_args: dict = None,
            loss_fn_args: dict = None,
            use_half: bool = True,
            verbosity: int = 0) -> None:
        self.device = "cuda" if cuda.is_available() else "cpu"
        self.use_half = use_half
        if self.use_half:
            self.nn = model(N).to(self.device).half()
        else:
            self.nn = model(N).to(self.device)
        # Optimizer
        if optimizer_args:
            args = ""
            for arg, value in optimizer_args.items():
                args += f"{arg}={value}, "
            exec(f"self.optimizer = optimizer(self.nn.parameters(), {args})")
        else:
            self.optimizer = optimizer(self.nn.parameters())
        # Loss function
        if loss_fn_args:
            args = ""
            for arg, value in loss_fn_args.items():
                args += f"{arg}={value}, "
            exec(f"self.loss_fn = loss_fn({args})")
        else:
            self.loss_fn = loss_fn()
        # Other parameters
        self.n_epochs = n_epochs
        self.verbosity = verbosity

    def _train(self) -> tuple:
        """Train the neural network."""
        self.nn.train()
        losses = []
        if self.verbosity:
            threshold = int(len(self.primary) * 0.05)
            threshold = 1 if threshold < 1 else threshold
            log.trace("Beginning training.")
        # TMP
        data = []
        for i in range(self.primary.shape[0]):
            d = [self.primary[i].T, self.secondary[i], self.families[i]]
            data.append(d)
        # TMP
        training_set = DataLoader(data, batch_size=32)
        if self.validation_primary:
            validation_set = DataLoader(
                [self.validation_primary,
                 self.validation_secondary,
                 self.validation_families], batch_size=32)
        for epoch in range(self.n_epochs):
            for batch, (x, y, f) in enumerate(training_set):
                if self.use_half:
                    x = x.to(self.device).half()
                    y = y.to(self.device).half()
                    f = f.to(self.device).half()
                else:
                    x = x.to(self.device)
                    y = y.to(self.device)
                    f = f.to(self.device)
                self.optimizer.zero_grad()
                pred = self.nn(x)
                loss = self.loss_fn(pred, y)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
                if self.verbosity > 1 and batch % threshold == 0:
                    progress = f"{batch} / {len(self.verbosity)}"
                    log.trace(f"Loss: {loss:.4f}    Batch {progress}", 1)
            if self.verbosity:
                prefix = f"{epoch} / {self.n_epochs} "
                log.progress_bar(self.n_epochs, epoch, prefix)
        if self.verbosity:
            print()

    def _predict(self, primary: np.ndarray) -> np.ndarray:
        self.nn.eval()
        primary = from_numpy(np.array([primary.T]))
        if self.use_half:
            primary = primary.to(self.device).half()
        else:
            primary = primary.to(self.device)
        return self.nn(primary)[0]

    def _save(self, path: str) -> None:
        torch_save(self.nn.state_dict(), path)

    def _load(self, path: str) -> None:
        self.nn.load_state_dict(torch_load(path))
