import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
from torch import nn
import torch
from tqdm import tqdm
from sklearn.metrics import f1_score
import statistics

import utils.datahandler as utils
import networks.cnn as cnn

# Set working directory to the location of the script to retrieve files.
import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# Parameters
#formatted_path = "../data/archiveII-shadows/"
#model = cnn.RNA_CNN_shadow
formatted_path = "../data/archiveII-classes/"
families = utils.getDatasetFilesnames(formatted_path)
model = cnn.RNA_CNN_classes
batch_size = 8
optimizer = optim.Adam
loss_fn = nn.MSELoss()

device = "cuda" if torch.cuda.is_available() else "cpu"

# Functions
def train(model: nn, dataloader: DataLoader, optimizer: optim, loss_fn,
        n_epochs: int,
        validation: DataLoader = None,
        verbose = 1) -> None:
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
    threshold = int(len(dataloader) * 0.05)
    threshold = 1 if threshold < 1 else threshold
    for epoch in tqdm(range(n_epochs)) if verbose else range(n_epochs):
        for batch, (x, y) in enumerate(dataloader):
            x, y = x.to(device).half(), y.to(device).half()
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            if verbose > 1 and batch % threshold == 0:
                loss = loss.item()
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
    f1 = []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device).half(), y.to(device).half()
            output = model(x)
            for i, j in zip(output, y):
                y_pred = utils.prediction_to_classes(i.tolist(), j.tolist())
                y_true = utils.prediction_to_classes(j.tolist(), j.tolist())
                f1.append(f1_score(y_true, y_pred, average='weighted'))
                # Debugging
                if len(f1) == 1:
                    ref = utils.prediction_to_secondary_structure(j.tolist())
                    print(f"R: {ref}")
                    pred = utils.prediction_to_secondary_structure(i.tolist())
                    print(f"P: {pred[:len(ref)]}")
                    print(f"F1-score for the predicted structure: {f1[-1]}")
                    print()
    return f1

def load_data(families: str) -> list:
    """
    Read data files formatted by the script `prepare_data.py`.

    Args:
        family: Name of the family to load.
    
    Returns:
        list: Loaded data represented as [x, y].
    """
    # Load data.
    x, y = None, None
    for family in families:
        if x is not None:
            new = np.load(families[family]['x'])
            if new.shape[0]:
                x = np.concatenate((x, new))
        else:
            x = np.load(families[family]['x'])
        if y is not None:
            new = np.load(families[family]['y'])
            if new.shape[0]:
                y = np.concatenate((y, new))
        else:
            y = np.load(families[family]['y'])
    # Format data into tensors.
    if len(x) < 1 or len(y) < 1:
        return []
    x, y = utils.shuffle_x_y(x, y)
    data = []
    for i in range(len(x)):
        data.append([
            torch.tensor(x[i].T, dtype=torch.float32),
            torch.tensor(y[i], dtype=torch.float32)])
    return data

def model_benchmark(model: nn,
        loss_fn,
        optimizer: optim,
        n_epochs: int,
        train_data,
        test_data,
        validation_data = None,
        verbose: int = 1) -> tuple:
    """
    Train and test a model.

    Args:
        model: Model to benchmark.
        sequence_len: Length of each sequence.
        data: Training and testing data organized as [sequences, labels].
        loss_fn: Loss function.
        optimizer: Optimizer to use.
        n_epochs: Number of epochs.
        train_data: Training dataloader.
        test_data: Test dataloader.
        validation_data: Validation dataloader.
        verbose: Verbosity of the function.
    
    Returns:
        tuple: f1-score
    """
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    train(model, train_dataloader, optimizer, loss_fn, n_epochs,
        validation_data, verbose)
    return test(model, test_dataloader)

def k_fold_benchmark(model_type: nn,
        sequence_len: int,
        data: list,
        K: int,
        loss_fn,
        optimizer_type: optim,
        n_epochs: int,
        use_validation: bool = True,
        verbose: int = 1) -> float:
    """
    Use K-fold to successively train and test the model on different parts
    of the dataset.

    Args:
        model_type: Model to create and benchmark.
        sequence_len: Length of each sequence.
        data: Training and testing data organized as [sequences, labels].
        K: Number of folds.
        loss_fn: Loss function.
        optimizer_type: Optimizer to use.
        n_epochs: Number of epochs.
        use_validation: Reserve data to evaluate the model during training.
        verbose: Verbosity of the function.
    
    Returns:
        tuple: Performances metric organized as the average of k-fold attempts
            for sensitivity, ppv, and f1-score.
    """
    f1 = []
    for k in range(K):
        model = model_type(sequence_len).to(device).half()
        optimizer = optimizer_type(model.parameters(), eps=1e-04)
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
        f1_tmp = test(model, test_dataloader)
        if verbose:
            print(f"K={k}: {np.mean(f1_tmp)}")
        f1 += f1_tmp
    return statistics.harmonic_mean(f1)

def inter_family_benchmark(model_type: nn,
        families: list,
        loss_fn,
        optimizer_type: optim,
        n_epochs: int,
        verbose: int = 1):
    """
    """
    for family in families:
        # Obtain training and test data.
        training_families = dict(families)
        training_families.pop(family)
        test_family = {family: families[family]}
        training_data = load_data(training_families)
        test_data = load_data(test_family)
        n_samples = len(test_data)
        if n_samples == 0:
            print(f"Family {family} is empty. Cannot test.")
            continue
        rna_length = len(training_data[0][0].T)
        # Test the model.
        model = model_type(rna_length).to(device).half()
        optimizer = optimizer_type(model.parameters(), eps=1e-04)
        f1 = statistics.harmonic_mean(model_benchmark(model, loss_fn,
            optimizer, n_epochs, training_data, test_data, None, verbose))
        print(f"F1-score with family {family} ({n_samples} samples): {f1}")
        del model

# Usage
data = load_data(families)
rna_length = len(data[0][0].T)

print(k_fold_benchmark(model, rna_length, data, 5,
    loss_fn, optimizer, n_epochs=5, use_validation=False, verbose=1))

"""
Inter-family testing
F1-score with family 16s (67 samples): 0.4274904819443681
F1-score with family 23s (15 samples): 0.42571868133414276
F1-score with family 5s (1283 samples): 0.46542208054459
F1-score with family grp1 (74 samples): 0.4046830391814072
Family grp2 is empty. Cannot test.
F1-score with family RNaseP (454 samples): 0.41030305339320533
F1-score with family srp (918 samples): 0.4373253793615782
F1-score with family telomerase (35 samples): 0.42221828099033387
F1-score with family tmRNA (462 samples): 0.41678871678540724
F1-score with family tRNA (557 samples): 0.4857507636848714

Omni-family testing
K=0: 0.8933596157291852
K=1: 0.8909089869833738
K=2: 0.8769196932433203
K=3: 0.8825623680353792
K=4: 0.9083502525299767
Harmonic mean: 0.8436663994854746
"""
