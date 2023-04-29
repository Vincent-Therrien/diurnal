"""
    Basic model used by the library.

    Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    Affiliation: Département d'informatique, UQÀM
    File creation date: April 2023
    License: MIT
"""

import torch
from torch import cuda, no_grad
import torch.optim as optim
from torch.utils.data import DataLoader

from .utils import file_io
from .train import prediction_to_onehot, clean_true_pred

class DiurnalBasicModel():
    """
    Generic RNA secondary prediction model.

    This class defines training and testing functions common to
    different neural networks.
    """
    def __init__(self, nn, nn_args, optimizer, optim_args, loss_fn,
                 use_half: bool = True) -> None:
        self.device = "cuda" if cuda.is_available() else "cpu"
        self.use_half = use_half
        if self.use_half:
            self.nn = nn(512).to(self.device).half()
        else:
            self.nn = nn(512).to(self.device).half()
        self.optimizer = optimizer(self.nn.parameters(), eps=1e-4)
        self.loss_fn = loss_fn
    
    def train(self,
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
                if self.use_half:
                    x = x.to(self.device).half()
                    y = y.to(self.device).half()
                    f = f.to(self.device).half()
                else:
                    x = x.to(self.device)
                    y = y.to(self.device)
                    f = f.to(self.device)
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
    
    def test(self, dataloader: DataLoader, evaluate) -> tuple:
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
                if self.use_half:
                    x = x.to(self.device).half()
                    y = y.to(self.device).half()
                    f = f.to(self.device).half()
                else:
                    x = x.to(self.device).half()
                    y = y.to(self.device).half()
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


class SimilarityModel():
    """
    Experimental secondary prediction model that relies of families.
    """
    def __init__(self, n: int, n_families: int,
                 classifier, predictor):
        self.device = "cuda" if cuda.is_available() else "cpu"
        self.classifier = classifier(n, n_families).to(self.device).half()
        self.classifier_optimizer = optim.Adam(
            self.classifier.parameters(), eps=1e-04)
        self.classifier_loss_fn = torch.nn.MSELoss()
        self.predictor  = predictor(n, n_families).to(self.device).half()
        self.predictor_optimizer = optim.Adam(
            self.predictor.parameters(), eps=1e-04)
        self.predictor_loss_fn = torch.nn.MSELoss()

    def train_classifier(self, data, epochs) -> None:
        self.classifier.train()
        for epoch in range(epochs):
            for batch, (x, _, f) in enumerate(data):
                x, f = x.to(self.device).half(), f.to(self.device).half()
                self.classifier_optimizer.zero_grad()
                pred = self.classifier(x)
                loss = self.classifier_loss_fn(pred, f)
                loss.backward()
                self.classifier_optimizer.step()
            #print(f"ec{epoch}")
    
    def test_classifier(self, data, evaluate) -> None:
        self.classifier.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for x, _, f in data:
                x, f = x.to(self.device).half(), f.to(self.device).half()
                output = self.classifier(x)
                for i, j in zip(output, f):
                    pred = i.tolist()
                    y_pred.append(pred.index(max(pred)))
                    true = j.tolist()
                    y_true.append(true.index(max(true)))
                    #if len(y_pred) == 1:
                    #    print(f"P: {pred}: {pred.index(max(pred))}")
                    #    print(f"R: {true}: {true.index(max(true))}")
        #print(confusion_matrix(y_true, y_pred))
        #print(f1_score(y_true, y_pred, average='weighted'))
        return [evaluate(a, b) for (a, b) in zip(y_pred, y_true)]

    def train_predictor(self, data, epochs) -> None:
        self.predictor.train()
        for epoch in range(epochs):
            for batch, (x, y, _) in enumerate(data):
                x, y = x.to(self.device).half(), y.to(self.device).half()
                self.predictor_optimizer.zero_grad()
                pred = self.predictor(x, self.classifier(x))
                loss = self.predictor_loss_fn(pred, y)
                loss.backward()
                self.predictor_optimizer.step()
            #print(f"ep{epoch}")

    def train(self, data, epochs: int) -> None:
        self.train_classifier(data, 1)
        #self.test_classifier(data)
        self.train_predictor(data, epochs)

    def predict(self, x) -> tuple:
        return self.predictor(x, self.classifier(x))

    def test(self, data, evaluate) -> float:
        self.predictor.eval()
        f1 = []
        with torch.no_grad():
            for x, y, _ in data:
                x, y = x.to(self.device).half(), y.to(self.device).half()
                output = self.predict(x)
                for i, j in zip(output, y):
                    pred = prediction_to_onehot(i.tolist())
                    true = j.tolist()
                    true, pred = clean_true_pred(true, pred)
                    y_pred = [n.index(max(n)) for n in pred]
                    y_true = [n.index(max(n)) for n in true]
                    f1.append(evaluate(y_pred, y_true))
                    # debug
                    #p = datahandler.prediction_to_secondary_structure(prediction)
                    #r = datahandler.prediction_to_secondary_structure(real_seq)
                    #print(f"{len(f1)} P: {p}")
                    #print(f"{len(f1)} R: {r}")
                    #print(f"F1: {f1[-1]}")
                    #print()
        return f1
