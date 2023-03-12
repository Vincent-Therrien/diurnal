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
def train(model, dataloader, optimizer, loss_fn, epochs, validation=None,
        verbose=True):
    model.train()
    for epoch in tqdm(range(epochs)) if verbose else range(epochs):
        for batch, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            loss = loss.item(), (batch + 1) * len(x)
            if verbose and batch % 10 == 0:
                print(f"Loss: {loss}")
        if verbose and validation:
            _, _, f1 = test(model, validation)
            print(f"Epoch {epoch} validation score: {f1:.4}")

def test(model, dataloader):
    model.eval()
    sensitivity, ppv, f1 = [], [], []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            for i, j in zip(output, y):
                s, p, f = utils.get_evaluation_metrics(
                    utils.prediction_to_secondary_structure(i.tolist()),
                    utils.prediction_to_secondary_structure(j.tolist()))
                sensitivity.append(s)
                ppv.append(p)
                f1.append(f)
    return np.mean(sensitivity), np.mean(ppv), np.mean(f1)

def k_fold_benchmark(model, data, K, loss_fn, optimizer, n_epochs,
        verbose=True) -> tuple:
    sensitivity, ppv, f1 = [], [], []
    for k in range(K):
        # Split the data for the current fold.
        training_data, validation_data, test_data = (
            utils.k_fold_split(data, [0.8, 0.1, 0.1], K, k))
        train_dataloader = DataLoader(training_data, batch_size=batch_size)
        valid_dataloader = DataLoader(validation_data, batch_size=batch_size)
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

def load_data(family):
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

model = cnn.convolutionalNN(rna_length).to(device)
optimizer = optim.Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss()

print(k_fold_benchmark(model, data, 5, loss_fn, optimizer, 5, False))
