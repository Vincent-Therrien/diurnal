"""
    Demonstration script for a simple CNN that predicts synthetic RNA
    secondary structures.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: February 2024
    - License: MIT
"""

import numpy as np
from torch import nn, optim
import torch.nn.functional as F

from diurnal import train, structure
import diurnal.models
from diurnal.utils import synthetic, log


# Dataset
primary = []
masks = []
secondary = []
N = 2000  # Number of different structures
n = 10  # Structure length
EPOCHS = 1000

log.title("Synthetic pipeline script.")
log.info("Create a synthetic dataset.")
log.trace(f"Number of samples: {N}")
log.trace(f"Sequence length: {n}")

for _ in range(N):
    p, s = synthetic.make_structures(n)
    primary.append(structure.Primary.to_matrix(p))
    masks.append(structure.Primary.to_mask(primary[-1]))
    secondary.append(structure.Secondary.to_matrix(s))

primary = np.array(primary)
masks = np.array(masks)
secondary = np.array(secondary)


x_train = primary[:int(N * 0.8)]
x_valid = primary[int(N * 0.8):int(N * 0.9)]
x_test = primary[int(N * 0.9):]

y_train = secondary[:int(N * 0.8)]
y_valid = secondary[int(N * 0.8):int(N * 0.9)]
y_test = secondary[int(N * 0.9):]

z_train = masks[:int(N * 0.8)]
z_valid = masks[int(N * 0.8):int(N * 0.9)]
z_test = masks[int(N * 0.9):]

train_set = {
    "input": (x_train, z_train),
    "output": y_train,
    "names": [],
    "families": []
}
valid_set = {
    "input": (x_valid, z_valid),
    "output": y_valid,
    "names": [],
    "families": []
}
test_set = {
    "input": (x_test, z_test),
    "output": y_test,
    "names": [],
    "families": []
}

# Training
class RNA_CNN_raw(nn.Module):
    def __init__(self, n):
        super().__init__()
        kernel = 3
        one_hot_dim = 8
        self.conv1 = nn.Conv2d(one_hot_dim, 1, kernel, padding="same")
        self.fc1 = nn.Linear(n, n)
        self.output = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = x.squeeze(1)
        x = F.relu(x)
        x = self.fc1(x)
        x = self.output(x)
        return x

class RNA_CNN_masked(nn.Module):
    def __init__(self, n):
        super().__init__()
        kernel = 3
        one_hot_dim = 8
        self.conv1 = nn.Conv2d(one_hot_dim, 1, kernel, padding="same")
        self.fc1 = nn.Linear(n, n)
        self.output = nn.Sigmoid()

    def forward(self, input, mask):
        input = self.conv1(input)
        input = input.squeeze(1)
        input = F.relu(input)
        input = self.fc1(input)
        input = self.output(input)
        input = input * mask
        return input


model = diurnal.models.NN(
    model=RNA_CNN_masked,
    N=n,
    n_epochs=EPOCHS,
    optimizer=optim.Adam,
    loss_fn=nn.MSELoss,
    optimizer_args={"eps": 1e-4},
    loss_fn_args=None,
    verbosity=2,
    use_half=True,
)
model.train(train_set, valid_set)


# Evaluation
for i in range(5):
    print("Raw NN prediction")
    pred = model.predict((x_test[i], z_test[i]))
    print(pred)

    print("Masked NN prediction")
    pred = np.multiply(pred, z_test[i])
    print(pred)

    print("Quantized NN prediction")
    train.quantize_matrix(pred)
    print(pred)

    print("Real structure")
    print(y_test[i])
