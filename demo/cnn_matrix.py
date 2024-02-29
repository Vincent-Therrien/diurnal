"""
    Demonstration for a simple CNN that predicts RNA secondary
    structures with pseudo-knots.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: September 2023
    - License: MIT
"""

import numpy as np
from torch import nn, optim
import torch.nn.functional as F
import torch

from diurnal import align, database, evaluate, models, train, visualize, structure
import diurnal.utils.rna_data as rna_data
from diurnal.utils import log


log.title("RNA secondary structure prediction with a CNN")

SIZE = 128
DATABASE = "archiveII"

SRC = f"./data/{DATABASE}"
DST = f"./data/cnn_matrix_data/"

# Preprocessing

database.download("./data/", DATABASE)
names = database.format_filenames(SRC, DST + "names.txt", SIZE)
train_names, validation_names, test_names = train.split(
    names, (0.8, 0.1, 0.1))

def format(dst: str, names):
    database.format_primary_structure(
        names, f"{dst}optimal_fold_alignments.npy",
        SIZE, align.optimal_fold_contact_matrix
    )
    database.format_primary_structure(
        names, f"{dst}fold_alignments_3.npy",
        SIZE, align.fold_contact_matrix
    )
    alignment_4 = lambda x, y : align.fold_contact_matrix(x, y, 4)
    database.format_primary_structure(
        names, f"{dst}fold_alignments_4.npy",
        SIZE, alignment_4
    )
    database.format_primary_structure(
        names, f"{dst}potential_pairings.npy",
        SIZE, structure.Primary.to_matrix
    )
    database.format_primary_structure(
        names, f"{dst}masks.npy",
        SIZE, structure.Primary.to_mask
    )
    database.format_primary_structure(
        names, f"{dst}onehot.npy",
        SIZE, structure.Primary.to_onehot
    )
    database.format_secondary_structure(
        names, f"{dst}contact.npy", SIZE, structure.Secondary.to_matrix
    )
    database.format_secondary_structure(
        names, f"{dst}bracket.npy", SIZE, structure.Secondary.to_onehot
    )

format(f"{DST}train/", train_names)
format(f"{DST}validation/", validation_names)
format(f"{DST}test/", test_names)

class CNN(nn.Module):
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
        #for i in range(len(input)):
        #    input[i] = input[i] * input[i].T
        return input

# Training
model = models.NN(
    model=CNN,
    N=SIZE,
    n_epochs=500,
    optimizer=torch.optim.Adam,
    loss_fn=torch.nn.MSELoss,
    optimizer_args={"eps": 1e-4},
    loss_fn_args=None,
    verbosity=2,
    use_half=True)

train_set = {
    "input": (
        np.load(f"{DST}train/potential_pairings.npy"),
        np.load(f"{DST}train/masks.npy")),
    "output": np.load(f"{DST}train/contact.npy"),
    "names": [],
    "families": []
}
validation_set = {
    "input": (
        np.load(f"{DST}validation/potential_pairings.npy"),
        np.load(f"{DST}validation/masks.npy")),
    "output": np.load(f"{DST}validation/contact.npy"),
    "names": [],
    "families": []
}
test_set = {
    "input": (
        np.load(f"{DST}test/potential_pairings.npy"),
        np.load(f"{DST}test/masks.npy")),
    "output": np.load(f"{DST}test/contact.npy"),
    "names": [],
    "families": []
}
model.train(train_set, validation_set)

i = 100
n = len(structure.Primary.unpad_matrix(test_set["input"][0][i]))
p = model.predict((test_set["input"][0][i], test_set["input"][1][i]))
p = p[:n, :n]
mask = test_set["input"][1][i][:n, :n]
visualize.heatmap(p * mask)
p = structure.Secondary.quantize(p, mask)
onehot = np.load("./data/cnn_matrix_data/test/onehot.npy")[i]
sequence = structure.Primary.to_sequence(onehot)
contact = np.load("./data/cnn_matrix_data/test/contact.npy")[i]
potential = test_set["input"][0][i][:n, :n]
t = test_set["output"][0]
t = t[:n, :n]
visualize.potential_pairings(potential, sequence, (p, t))
print(evaluate.ContactMatrix.f1(t, p))
exit()

f = model.test(test_set, evaluation=evaluate.ContactMatrix.f1)
print(f"Average F1-score: {sum(f)/len(f):.4}")

model.save("saved_model")

del model

loaded_model = models.NN(
    model=CNN,
    N=SIZE,
    n_epochs=10,
    optimizer=torch.optim.Adam,
    loss_fn=torch.nn.MSELoss,
    optimizer_args={"eps": 1e-4},
    loss_fn_args=None,
    verbosity=2,
    use_half=True)
loaded_model.load("saved_model")

print(test_set["input"][0])
visualize.potential_pairings(test_set["input"][0])
print(test_set["secondary_structures"][0])
visualize.secondary_structure(test_set["output"][0])
pred = loaded_model.predict(test_set["input"][0])
print(pred.shape)
print(pred)
np.save("test.npy", pred)
visualize.secondary_structure(pred[0])

f = loaded_model.test(test_set)
print(f"Average F1-score of the saved model: {sum(f)/len(f):.4}")

print(f"\nSample prediction from the test set (`{test_set['names'][0]}`).")
p = test_set["input"][0]
s = test_set["output"][0]
visualize.prediction(p, s, loaded_model.predict(p))
