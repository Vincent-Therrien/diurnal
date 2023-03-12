import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
from torch import nn
import torch
from tqdm import tqdm

import cnn.network as cnn
import utils.datahandler as utils

# Set working directory to the location of the script.
import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

formatted_path = "../data/archiveII-arrays/"

# Parameters
batch_size = 32
device = "cuda" if torch.cuda.is_available() else "cpu"

# Functions
def train(model, dataloader, optimizer, loss_fn, epochs, validation = None):
    model.train()
    for _ in tqdm(range(epochs)):
        for batch, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            loss = loss.item(), (batch + 1) * len(x)
        if validation:
            test(model, validation)

def test(model, dataloader):
    model.eval()
    metrics = [[], [], []]
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            for i, j in zip(output, y):
                s, p, f = utils.get_evaluation_metrics(
                    utils.prediction_to_secondary_structure(i.tolist()),
                    utils.prediction_to_secondary_structure(j.tolist()))
                metrics[0].append(s)
                metrics[1].append(p)
                metrics[2].append(f)
    print(f"Sensitivity: {np.mean(metrics[0]):.4f}    ", end="")
    print(f"PPV: {np.mean(metrics[1]):.4f}    ", end="")
    print(f"F1-score: {np.mean(metrics[2]):.4f}")

# Usage
print("Loading data")
x = np.load(formatted_path + "5s_x.npy")
y = np.load(formatted_path + "5s_y.npy")

print("Splitting data")
data = []
for i in range(len(x)):
    data.append([
        torch.tensor(x[i].T, dtype=torch.float32).to(device),
        torch.tensor(y[i], dtype=torch.float32).to(device)])

training_data, validation_data, test_data = utils.k_fold_split(data, [0.8, 0.1, 0.1], 4, 0)

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
validation_dataloader = DataLoader(validation_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

print("Beginning the training")
model = cnn.convolutionalNN(len(x[0])).to(device)
optimizer = optim.Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss()

train(model, train_dataloader, optimizer, loss_fn, 5, validation_dataloader)
test(model, test_dataloader)
