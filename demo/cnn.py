"""
    Demonstration for a simple CNN that predicts RNA secondary
    structures.
"""

import torch

from diurnal import database, train
import diurnal.models
from diurnal.models.networks import cnn


SIZE = 512

database.download("./data/", "archiveII")
database.format(
    "./data/archiveII",  # Directory of the raw data to format.
    "./data/formatted",  # Formatted data output directory.
    SIZE,  # Normalized size.
)

test_set, other_data = train.load_inter_family("./data/formatted", "5s")
train_set, validate_set = train.split_data(other_data, [0.8, 0.2])

model = diurnal.models.NN(
    model=cnn.Pairings_1,
    N=SIZE,
    n_epochs=3,
    optimizer=torch.optim.Adam,
    loss_fn=torch.nn.MSELoss,
    optimizer_args={"eps": 1e-4},
    loss_fn_args=None,
    verbosity=1)
model.train(train_set)

f = model.test(test_set)
print(f"Average F1-score: {sum(f)/len(f):.4}")

model.save("saved_model")

del model

model2 = diurnal.models.NN(
    cnn.Pairings_1,
    SIZE,
    3,
    torch.optim.Adam,
    torch.nn.MSELoss,
    {"eps": 1e-4},
    None,
    verbosity=1)
model2.load("saved_model")

f = model2.test(test_set)
print(f"Average F1-score of the saved model: {sum(f)/len(f):.4}")
