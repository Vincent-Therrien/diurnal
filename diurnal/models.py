"""
    Basic model used by the library/
"""

import torch
from torch import nn, cuda, no_grad
import torch.optim as optim
from torch.utils.data import DataLoader

from .utils import file_io
from .train import prediction_to_onehot, clean_true_pred
from .transform import SecondaryStructure as s2
from sklearn.metrics import f1_score

class DiurnalBasicModel():
    """
    Generic RNA secondary prediction model.

    This class defines training and testing functions common to
    different neural networks.
    """
    def __init__(self, nn, nn_args, optimizer, optim_args, loss_fn) -> None:
        self.device = "cuda" if cuda.is_available() else "cpu"
        self.nn = nn(512).to(self.device).half()
        self.optimizer = optimizer(self.nn.parameters(), eps=1e-4)
        self.loss_fn = loss_fn
    
    def train_with_families(self,
            dataloader: DataLoader,
            n_epochs: int,
            validation: DataLoader = None,
            verbosity: int = 1) -> tuple:
        """
        Train a model with primary structure and family.

        Args:
            dataloader (DataLoader): Input data.
            n_epochs (int): Number of training epochs.
            verbosity (int): Verbosity level.
        """
        self.nn.train()
        losses = []

        if verbosity:
            threshold = int(len(dataloader) * 0.05)
            threshold = 1 if threshold < 1 else threshold
            file_io.log("Beginning training.")
        for epoch in range(n_epochs):
            for batch, (x, y, f) in enumerate(dataloader):
                x, y = x.to(self.device).half(), y.to(self.device).half()
                f = f.to(self.device).half()
                self.optimizer.zero_grad()
                pred = self.nn(x, f)
                loss = self.loss_fn(pred, y)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
                if verbosity > 1 and batch % threshold == 0:
                    progress = f"{batch} / {len(dataloader)}"
                    file_io.log(f"Loss: {loss:.4f}    Batch {progress}", 1)
            if verbosity:
                prefix = f"{epoch} / {n_epochs} "
                if validation is not None:
                    suffix = f"{self.get_f1(validation)}"
                else:
                    suffix = ""
                file_io.progress_bar(n_epochs, epoch, prefix, suffix)
        if verbosity:
            print()
        
        return losses
    
    def test_with_family(self, dataloader: DataLoader, evaluate) -> tuple:
        """
        Test a model with a dataset.

        Args:
            dataloader (Dataloader): Test data.

        Returns (list): F1-score of each prediction-true value pairs.
        """
        self.nn.eval()
        f1 = []
        with no_grad():
            for batch, (x, y, f) in enumerate(dataloader):
                x, y = x.to(self.device).half(), y.to(self.device).half()
                f = f.to(self.device).half()
                output = self.nn(x, f)
                for i, j in zip(output, y):
                    pred = prediction_to_onehot(i.tolist())
                    true = j.tolist()
                    true, pred = clean_true_pred(true, pred)
                    y_pred = [n.index(max(n)) for n in pred]
                    y_true = [n.index(max(n)) for n in true]
                    f1.append(evaluate(y_pred, y_true))
        return f1

    def predict(self, input) -> list:
        """
        Predict values from an input array.
        """
        return self.nn(input)
    
    def save(self, path: str) -> None:
        """
        Save the model in a .pt file.

        Args:
            path (str): File path of the file to save. Must end in `pt`.
        """
        torch.save(self.nn.state_dict(), path)
    
    def load(self, path: str) -> None:
        """
        Load a model from a .pt file.

        Args:
            path (str): File path of the file to load. Must end in `pt`.
        """
        self.nn.load_state_dict(torch.load(path))
