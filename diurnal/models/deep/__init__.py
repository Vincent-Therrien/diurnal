"""
    RNA secondary structure prediction deep learning model package.

    File information:

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: June 2023
    - License: MIT
"""

__all__ = ["cnn", "mlp"]

import numpy as np
import torch.optim as optim
from torch import cuda, nn
from torch import load as torch_load
from torch import save as torch_save
from torch import from_numpy
from torch.utils.data import DataLoader

from diurnal.utils import log
from diurnal.models import Basic

__all__ = ["cnn", "mlp"]


class NN(Basic):
    """A model that relies on a neural network to make predictions."""
    def __init__(
        self,
        model: nn,
        n_epochs: int,
        optimizer: optim,
        loss_fn: nn.functional,
        optimizer_args: dict = None,
        loss_fn_args: dict = None,
        use_half: bool = True,
        patience: int = 5,
        verbosity: int = 0,
        batch: int = 16
    ) -> None:
        self.device = "cuda" if cuda.is_available() else "cpu"
        self.use_half = use_half and self.device == "cuda"
        if self.use_half:
            self.nn = model.to(self.device).half()
        else:
            self.nn = model.to(self.device)
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
        self.batch = batch
        self.PATIENCE = patience

    def _train(self) -> None:
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
        patience = self.PATIENCE
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
                if len(input) > 1:
                    value = from_numpy(np.array([i.T]))
                    input_values.append(value.to(self.device))
                else:
                    value = from_numpy(np.array(i.T))
                    input_values.append(value.to(self.device))
        if len(input) == 1:
            if len(input_values[0].shape) > 2:
                input_values[0] = input_values[0][None, :, :]
            else:
                input_values[0] = input_values[0][None, :]
        pred = self.nn(*input_values)
        if self.device == "cuda":
            return pred.detach().cpu().numpy()
        return pred

    def _save(self, path: str) -> None:
        torch_save(self.nn.state_dict(), path + "model.pt")

    def _load(self, path: str) -> None:
        self.nn.load_state_dict(torch_load(path + "model.pt"))
