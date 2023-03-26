import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
from torch import nn
import torch
import statistics

import utils.datahandler as utils
from networks.similarity_model import SimilarityModel

# Set working directory to the location of the script to retrieve files.
import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# Parameters
formatted_path = "../data/archiveII-classes/"
families = utils.getDatasetFilesnames(formatted_path)
batch_size = 16

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_family_one_hot(families, family):
    one_hot = []
    for f in families:
        if f == family:
            one_hot.append(1)
        else:
            one_hot.append(0)
    return one_hot

def load_data(families: str) -> list:
    """
    Read data files formatted by the script `prepare_data.py`.

    Args:
        family: Name of the family to load.
    
    Returns:
        list: Loaded data represented as [x, y].
    """
    # Load data.
    x, y, f = None, None, None
    for family in families:
        input = np.load(families[family]['x'])
        size = len(input)
        if size < 1:
            continue
        # x (primary structure)
        if x is not None:
            if input.shape[0]:
                x = np.concatenate((x, input))
        else:
            x = np.load(families[family]['x'])
        # y (secondary structure)
        if y is not None:
            new = np.load(families[family]['y'])
            if new.shape[0]:
                y = np.concatenate((y, new))
        else:
            y = np.load(families[family]['y'])
        # f (family)
        label_value = get_family_one_hot(list(families.keys()), family)
        label = np.array([label_value for _ in range(size)])
        if f is not None:
            f = np.concatenate((f, label))
        else:
            f = label
    # Format data into tensors.
    if len(x) < 1 or len(y) < 1:
        return []
    x, y, f = utils.shuffle_x_y_f(x, y, f)
    data = []
    for i in range(len(x)):
        data.append([
            torch.tensor(x[i].T, dtype=torch.float32),
            torch.tensor(y[i], dtype=torch.float32),
            torch.tensor(f[i], dtype=torch.float32)])
    return data

def model_benchmark(
        n_epochs: int,
        n_families: int,
        train_data,
        test_data,
        validation_data = None,
        verbose: int = 1) -> tuple:
    """
    Train and test a model.

    Args:
        n_epochs: Number of epochs.
        train_data: Training dataloader.
        test_data: Test dataloader.
        validation_data: Validation dataloader.
        verbose: Verbosity of the function.
    
    Returns:
        tuple: f1-score
    """
    rna_length = len(train_data[0][0].T)
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    model = SimilarityModel(rna_length, n_families, device)
    model.train(train_dataloader, n_epochs)
    return model.test(test_dataloader)

def k_fold_benchmark(
        data: list,
        K: int,
        n_epochs: int,
        n_families: int,
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
        # Training and testing
        model = SimilarityModel(rna_length, n_families, device)
        model.train(train_dataloader, test_dataloader, n_epochs)
        f1 += model.test(test_dataloader)
    return statistics.harmonic_mean(f1)

def inter_family_benchmark(
        families: list,
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
        # Test the model.
        f1 = np.mean(model_benchmark(
            n_epochs, len(training_families), training_data, test_data, verbose))
        print(f"F1-score with family {family} ({n_samples} samples): {f1}")

# Usage
inter_family_benchmark(families, 5)
exit()

data = load_data(families)
rna_length = len(data[0][0].T)

