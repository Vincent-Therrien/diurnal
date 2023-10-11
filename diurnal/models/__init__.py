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
    - `_predict(primary) -> np.array` (make a prediction)
    - `_save(directory) -> None` (save the model)
    - `_load(directory) -> None` (load a model from files)

    File information:

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: June 2023
    - License: MIT
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

from diurnal import evaluate, train
from diurnal.utils import file_io, log


__all__ = ["baseline", "networks"]


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
                        "primary_structures": <vector>,
                        "secondary_structures": <vector>,
                        "names": List[str],
                        "families": List[str]
                    }

            validation_data (dict)
            verbose (bool): Print informative messages.
        """
        if verbose:
            N = len(training_data["primary_structures"])
            log.info(f"Training the model with {N} data points.")
            if validation_data:
                n = len(validation_data["primary_structures"])
                log.trace(f"Using {n} data points for validation.")
        self.names = training_data["names"]
        self.primary = training_data["primary_structures"]
        self.secondary = training_data["secondary_structures"]
        self.families = training_data["families"]
        if validation_data:
            self.validate = True
            self.validation_names = validation_data["names"]
            self.validation_primary = validation_data["primary_structures"]
            self.validation_secondary = validation_data["secondary_structures"]
            self.validation_families = validation_data["families"]
        else:
            self.validate = False
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
                f"{len(self.primary)} molecules were used for training.\n"])
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
            evaluation: Callable = evaluate.micro_f1,
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
        n = len(data["primary_structures"])
        if verbose:
            log.info(f"Testing the model with {n} data points.")
        for i in range(n):
            primary = data["primary_structures"][i]
            true = data["secondary_structures"][i]
            pred = self.predict(primary)
            if (len(primary.shape)) == 2:
                _, true, pred = train.clean_vectors(primary, true, pred)
            elif (len(primary.shape)) == 3:
                _, true, pred = train.clean_matrices(primary, true, pred)
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
        self.use_half = use_half and self.device == "cuda"
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
        print(self.n_epochs)
        self.verbosity = verbosity
        self.batch = 16

    def _train(self) -> tuple:
        """Train the neural network."""
        self.nn.train()
        if self.verbosity:
            threshold = int(len(self.primary) * 0.05)
            threshold = 1 if threshold < 1 else threshold
            log.trace("Beginning the training.")
        # TMP
        data = []
        self.primary = np.array(self.primary)
        self.secondary = np.array(self.secondary)
        N = len(self.primary)
        N_PRINTS = int(N / self.batch)
        threshold = int((N / self.batch) / 10)
        for i in range(self.primary.shape[0]):
            d = [self.primary[i].T, self.secondary[i]]
            data.append(d)
        training_set = DataLoader(data, batch_size=self.batch)
        if self.validate:
            data = []
            self.validation_primary = np.array(self.validation_primary)
            self.validation_secondary = np.array(self.validation_secondary)
            for i in range(self.validation_primary.shape[0]):
                d = [self.validation_primary[i].T, self.validation_secondary[i]]
                data.append(d)
            validation_set = DataLoader(data, batch_size=self.batch)
        # TMP
        patience = 5
        average_losses = []
        for epoch in range(self.n_epochs):
            losses = []
            for batch, (x, y) in enumerate(training_set):
                if self.use_half:
                    x = x.to(self.device).half()
                    y = y.to(self.device).half()
                else:
                    x = x.to(self.device)
                    y = y.to(self.device)
                self.optimizer.zero_grad()
                pred = self.nn(x)
                loss = self.loss_fn(pred, y)
                loss.backward()
                self.optimizer.step()
                if self.verbosity > 2 and batch % threshold == 0:
                    log.trace(f"Loss: {loss:.4f} | Batch {batch} / {N_PRINTS}")
            if self.validate:
                losses = []
                for batch, (x, y) in enumerate(validation_set):
                    if self.use_half:
                        x = x.to(self.device).half()
                        y = y.to(self.device).half()
                    else:
                        x = x.to(self.device)
                        y = y.to(self.device)
                    pred = self.nn(x)
                    losses.append(self.loss_fn(pred, y).item())
                average_loss = sum(losses) / len(losses)
                average_losses.append(average_loss)
                if (len(average_losses) > 2
                        and average_losses[-1] >= min(average_losses[:-1])):
                    patience -= 1
                    if patience <= 0:
                        break
            if self.verbosity:
                prefix = f"{epoch} / {self.n_epochs} "
                if self.validate:
                    loss_value = f" Validation loss: {average_losses[-1]:.10f}     Min: {min(average_losses):.10f}"
                    suffix = loss_value + f" | Patience: {patience}"
                    log.progress_bar(self.n_epochs, epoch, prefix, suffix)
                    if self.verbosity > 1:
                        print()
                else:
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
        pred = self.nn(primary)[0]
        if self.device == "cuda":
            return pred.detach().cpu().numpy()
        return pred

    def _save(self, path: str) -> None:
        torch_save(self.nn.state_dict(), path + "model.pt")

    def _load(self, path: str) -> None:
        self.nn.load_state_dict(torch_load(path + "model.pt"))
