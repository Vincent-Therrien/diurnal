"""
    Basic model used by the library/
"""

from torch import nn, cuda, no_grad
import torch.optim as optim
from torch.utils.data import DataLoader

from .utils import file_io
from .evaluate import get_sen_PPV_f1
from sklearn.metrics import f1_score

class DiurnalBasicModel():
    """
    Generic RNA secondary prediction model.

    This class defines training and testing functions common to
    different neural networks.
    """
    def __init__(self, nn, optimizer, optim_args, loss_fn) -> None:
        self.nn = nn
        self.optimiser: optim = optimizer(self.nn.parameters(), *optim_args)
        self.loss_fn = loss_fn
        self.device = "cuda" if cuda.is_available() else "cpu"

    def train(self,
              dataloader: DataLoader,
              n_epochs: int,
              validation: DataLoader = None,
              verbosity: int = 1) -> tuple:
        """
        Train a model.

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
            for batch, (x, y) in enumerate(dataloader):
                x, y = x.to(self.device).half(), y.to(self.device).half()
                self.optimizer.zero_grad()
                pred = self.nn(x)
                loss = self.loss_fn(pred, y)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
                if verbosity > 1 and batch % threshold == 0:
                    progress = f"{batch} / {len(dataloader)}"
                    file_io.log(f"Loss: {loss:.4f}    Batch {progress}", 1)
            if verbosity and validation:
                prefix = f"{epoch} / {n_epochs} "
                suffix = f"{self.get_f1(validation)}" if validation else ""
                file_io.progress_bar(n_epochs, epoch, prefix, suffix)
        
        return losses
    
    def train_with_families(self,
            dataloader: DataLoader,
            n_epochs: int,
            validation: DataLoader = None,
            verbosity: int = 1) -> tuple:
        """
        Train a model.

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
            if verbosity and validation:
                prefix = f"{epoch} / {n_epochs} "
                suffix = f"{self.get_f1(validation)}" if validation else ""
                file_io.progress_bar(n_epochs, epoch, prefix, suffix)
        
        return losses

    def test(self, dataloader: DataLoader) -> tuple:
        """
        """
        self.nn.eval()
        y_pred = []
        y_true = []
        with no_grad():
            for x, y in dataloader:
                x, y = x.to(self.device).half(), y.to(self.device).half()
                output = self.nn(x)
                for i, j in zip(output, y):
                    pred = i.tolist()
                    y_pred.append(pred.index(max(pred)))
                    true = j.tolist()
                    y_true.append(true.index(max(true)))
        return f1_score(y_true, y_pred, average='weighted')
    
    def test_with_family(self, dataloader: DataLoader) -> tuple:
        """
        """
        self.nn.eval()
        y_pred = []
        y_true = []
        with no_grad():
            for x, y, f in dataloader:
                x, y = x.to(self.device).half(), y.to(self.device).half()
                f = f.to(self.device).half()
                output = self.nn(x, f)
                for i, j in zip(output, y):
                    pred = i.tolist()
                    y_pred.append(pred.index(max(pred)))
                    true = j.tolist()
                    y_true.append(true.index(max(true)))
        return f1_score(y_true, y_pred, average='weighted')

    def predict(self, input) -> list:
        """
        Predict values from an input array.
        """
        return self.nn(input)
