"""
    Demonstration for a simple CNN that predicts RNA secondary
    structures.
"""

import torch

from diurnal import database, train
import diurnal.models
from diurnal.models.networks import cnn


SIZE = 512

print("1. Obtaining raw data.")
database.download("./data/", "archiveII")
database.format(
    "./data/archiveII", # Directory of the raw data to format.
    "./data/formatted", # Formatted data output directory.
    SIZE, # Normalized size.
)

print("2. Obtaining formatted data.")
test_set, other_data = train.load_inter_family("./data/formatted", "5s")
train_set, validate_set = train.split_data(other_data, [0.8, 0.2])

print("3. Training the model.")
model = diurnal.models.NN(cnn.Pairings_1,
    SIZE,
    3,
    torch.optim.Adam,
    torch.nn.MSELoss,
    {"eps": 1e-4},
    None,
    verbosity=1)
model.train(train_set)

print("4. Testing the model.")
f = model.test(test_set)
print(f"Average F1-score: {sum(f)/len(f):.4}")
