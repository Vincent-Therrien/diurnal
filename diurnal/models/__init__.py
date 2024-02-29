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
                        "input": tuple(<vector>),
                        "output": <vector>,
                        "names": List[str],
                        "families": List[str]
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
        self.families = training_data["families"]
        self.validate = False
        self.validation_names = None
        self.validation_input = None
        self.validation_output = None
        self.validation_N = None
        self.validation_families = None
        if validation_data:
            self.validate = True
            self.validation_names = validation_data["names"]
            self.validation_input = validation_data["input"]
            self.validation_output = validation_data["output"]
            self.validation_N = len(self.validation_output)
            if len(self.validation_input) == self.validation_N:
                self.validation_input = [self.validation_input]
            self.validation_families = validation_data["families"]
        self._train()

    def predict(self, input) -> np.array:
        """Predict a random secondary structure.

        Args:
            input: RNA primary structure data.

        Returns (np.array): Predicted structure.
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
        self.verbosity = verbosity
        self.batch = 16

    def _train(self) -> tuple:
        """Train the neural network."""
        self.nn.train()
        if self.verbosity:
            threshold = int(len(self.output) * 0.05)
            threshold = 1 if threshold < 1 else threshold
            log.trace("Beginning the training.")
        # TMP
        data = []
        N_PRINTS = int(self.N / self.batch)
        threshold = int((self.N / self.batch) / 10)
        for i in range(self.N):
            input = []
            for j in range(len(self.input)):
                input.append(np.array(self.input[j][i].T))
            data.append([input, np.array(self.output[i])])
        training_set = DataLoader(data, batch_size=self.batch)
        if self.validate:
            data = []
            for i in range(self.validation_N):
                input = []
                for j in range(len(self.validation_input)):
                    input.append(np.array(self.validation_input[j][i].T))
                data.append([input, self.validation_output[i]])
            validation_set = DataLoader(data, batch_size=self.batch)
        # TMP
        patience = 5
        average_losses = []
        for epoch in range(self.n_epochs):
            losses = []
            for batch, (x, y) in enumerate(training_set):
                if self.use_half:
                    x = [x.to(self.device).half() for x in x]
                    y = y.to(self.device).half()
                else:
                    x = [x.to(self.device) for x in x]
                    y = y.to(self.device)
                self.optimizer.zero_grad()
                pred = self.nn(*x)
                loss = self.loss_fn(pred, y)
                loss.backward()
                self.optimizer.step()
                if self.verbosity > 2 and batch % threshold == 0:
                    log.trace(f"Loss: {loss:.5f} | Batch {batch} / {N_PRINTS}")
            if self.validate:
                losses = []
                for batch, (x, y) in enumerate(validation_set):
                    if self.use_half:
                        x = [x.to(self.device).half() for x in x]
                        y = y.to(self.device).half()
                    else:
                        x = [x.to(self.device) for x in x]
                        y = y.to(self.device)
                    pred = self.nn(*x)
                    losses.append(self.loss_fn(pred, y).item())
                average_loss = sum(losses) / len(losses)
                average_losses.append(average_loss)
                if (len(average_losses) > 2
                        and average_losses[-1] >= min(average_losses[:-1])):
                    patience -= 1
                    if patience <= 0:
                        break
            if self.verbosity:
                if self.validate:
                    loss_value = f" Loss: {average_losses[-1]:.5f}"
                    suffix = f"{loss_value}  Patience: {patience}"
                    log.progress_bar(self.n_epochs, epoch, suffix)
                    if self.verbosity > 1:
                        print()
                else:
                    log.progress_bar(self.n_epochs, epoch)
        if self.verbosity:
            print()

    def _predict(self, input: any) -> np.ndarray:
        self.nn.eval()
        if self.use_half:
            input_values = []
            for i in input:
                value = from_numpy(np.array(i.T))
                input_values.append(value.to(self.device).half())
        else:
            input_values = []
            for i in input:
                value = from_numpy(np.array(i.T))
                input_values.append(value.to(self.device))
        if len(input) == 1:
            input_values[0] = input_values[0][None, :, :]
        pred = self.nn(*input_values)[0]
        if self.device == "cuda":
            return pred.detach().cpu().numpy()
        return pred

    def _save(self, path: str) -> None:
        torch_save(self.nn.state_dict(), path + "model.pt")

    def _load(self, path: str) -> None:
        self.nn.load_state_dict(torch_load(path + "model.pt"))
