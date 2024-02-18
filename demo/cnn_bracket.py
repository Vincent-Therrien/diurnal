"""
    Demonstration for a simple CNN that predicts RNA secondary
    structures.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: September 2023
    - License: MIT
"""

import torch

from diurnal import database, train, visualize, family
import diurnal.models
from diurnal.models.networks import cnn


SIZE = 512

database.download("./data/", "archiveII")
database.format(
    "./data/archiveII",  # Directory of the raw data to format.
    "./data/formatted",  # Formatted data output directory.
    SIZE,  # Normalized size.
)

test_family = "5s"
train_families = family.all_but(test_family)

test_set = train.load_families("./data/formatted", test_family)
train_set = train.load_families("./data/formatted", train_families)
train_set, validation_set = train.split_data(train_set, (0.9, 0.1))

model = diurnal.models.NN(
    model=cnn.Dot_Bracket,
    N=SIZE,
    n_epochs=2,
    optimizer=torch.optim.Adam,
    loss_fn=torch.nn.MSELoss,
    optimizer_args={"eps": 1e-4},
    loss_fn_args=None,
    verbosity=2)
model.train(train_set, validation_set)

f = model.test(test_set)
print(f"Average F1-score: {sum(f)/len(f):.4}")

model.save("saved_model")

del model

loaded_model = diurnal.models.NN(
    cnn.Dot_Bracket,
    SIZE,
    2,
    torch.optim.Adam,
    torch.nn.MSELoss,
    {"eps": 1e-4},
    None,
    verbosity=2)
loaded_model.load("saved_model")

f = loaded_model.test(test_set)
print(f"Average F1-score of the saved model: {sum(f)/len(f):.4}")

print(f"\nSample prediction from the test set (`{test_set['names'][0]}`).")
input = (test_set["input"][0][0], )
p = input[0]
s = test_set["output"][0]
visualize.prediction(p, s, loaded_model.predict(input))
