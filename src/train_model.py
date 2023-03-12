import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
from torch import nn
import torch
from tqdm import tqdm

import utils.datahandler as utils
import networks.cnn as cnn

# Set working directory to the location of the script to retrieve files.
import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
formatted_path = "../data/archiveII-arrays/"

# Parameters
batch_size = 32
device = "cuda" if torch.cuda.is_available() else "cpu"
family = "5s"

# Functions
def train(model: nn, dataloader: DataLoader, optimizer: optim, loss_fn,
        n_epochs: int,
        validation: DataLoader = None,
        verbose = True) -> None:
    """
    Train a model with the specified parameters.
    
    Args:
        model: Model (i.e. neural network) to train.
        dataloader: Torch dataloader containing sequences and labels.
        optimizer: Model optimizer.
        loss_fn: Loss function.
        n_epochs: Number of epochs.
        validation: Validation set. If not none, used to evaluate the model
            after each epoch.
        verbose: Verbosity of the function.
    """
    model.train()
    for epoch in tqdm(range(n_epochs)) if verbose else range(n_epochs):
        for batch, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            loss = loss.item()
            if verbose and batch % int(len(dataloader) * 0.2) == 0:
                print(f"Loss: {loss:.4f}    Batch {batch} / {len(dataloader)}")
        if verbose and validation:
            _, _, f1 = test(model, validation)
            print(f"Epoch {epoch} validation score: {f1:.4}")

def test(model: nn, dataloader: DataLoader) -> tuple:
    """
    Evaluate the performance of the model.

    Args:
        model: Model to evaluate.
        dataloader: DataLoader object containing the test data and labels.
    
    Returns:
        tuple: Performances metric organized as: sensitivity, ppv, f1-score.
    """
    model.eval()
    sensitivity, ppv, f1 = [], [], []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            for i, j in zip(output[10:], y[10:]):
                print("Prediction:")
                print(utils.prediction_to_secondary_structure(i.tolist()))
                print("Label:")
                print(utils.prediction_to_secondary_structure(j.tolist()))
                exit()
                s, p, f = utils.get_evaluation_metrics(
                    utils.prediction_to_secondary_structure(i.tolist()),
                    utils.prediction_to_secondary_structure(j.tolist()))
                sensitivity.append(s)
                ppv.append(p)
                f1.append(f)
    return np.mean(sensitivity), np.mean(ppv), np.mean(f1)

def k_fold_benchmark(model: nn, data: list, K: int, loss_fn,
        optimizer: optim,
        n_epochs: int,
        use_validation: bool = True,
        verbose: bool = True) -> tuple:
    """
    Use K-fold to successively train and test the model on different parts
    of the dataset.

    Args:
        model: Model to evaluate.
        data: Training and testing data organized as [sequences, labels].
        K: Number of folds.
        loss_fn: Loss function.
        optimizer: Optimizer to use.
        n_epochs: Number of epochs.
        use_validation: Reserve data to evaluate the model during training.
        verbose: Verbosity of the function.
    
    Returns:
        tuple: Performances metric organized as the average of k-fold attempts
            for sensitivity, ppv, and f1-score.
    """
    sensitivity, ppv, f1 = [], [], []
    for k in range(K):
        # Split the data for the current fold.
        train_f = 0.8
        valid_f = 0.1 if use_validation else 0.0
        test_f = 1 - train_f - valid_f
        training_data, valid_data, test_data = (
            utils.k_fold_split(data, [train_f, valid_f, test_f], K, k))
        train_dataloader = DataLoader(training_data, batch_size=batch_size)
        if valid_f:
            valid_dataloader = DataLoader(valid_data, batch_size=batch_size)
        else:
            valid_dataloader = None
        test_dataloader = DataLoader(test_data, batch_size=batch_size)
        # Train the model with the current set of data.
        train(model, train_dataloader, optimizer, loss_fn, n_epochs,
            valid_dataloader, verbose)
        # Collect performance metric for the current fold.
        s, p, f = test(model, test_dataloader)
        sensitivity.append(s)
        ppv.append(p)
        f1.append(f)
    return np.mean(sensitivity), np.mean(ppv), np.mean(f1)

def load_data(family: str) -> list:
    """
    Read data files formatted by the script `prepare_data.py`.

    Args:
        family: Name of the family to load.
    
    Returns:
        list: Loaded data represented as [x, y].
    """
    x = np.load(formatted_path + family + "_x.npy")
    y = np.load(formatted_path + family + "_y.npy")
    data = []
    for i in range(len(x)):
        data.append([
            torch.tensor(x[i].T, dtype=torch.float32).to(device),
            torch.tensor(y[i], dtype=torch.float32).to(device)])
    return data

# Usage
data = load_data(family)
rna_length = len(data[0][0].T)

model = cnn.RNA_CNN(rna_length).to(device)
optimizer = optim.Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss()

print(k_fold_benchmark(model, data, 5, loss_fn, optimizer, 5, False, True))
