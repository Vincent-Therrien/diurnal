"""
    Demonstration for a simple CNN that predicts RNA secondary
    structures.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: September 2023
    - License: MIT
"""

import torch

from diurnal import database, train, visualize, family, structure
import diurnal.models
from diurnal.models.networks import cnn


SIZE = 512

database.download("./data/", "archiveII")
database.format(
    "./data/archiveII",  # Directory of the raw data to format.
    "./data/formatted_shadow",  # Formatted data output directory.
    SIZE,  # Normalized size.
    structure.Primary.to_onehot,
    structure.Secondary.to_shadow
)

test_family = "5s"
train_families = family.all_but(test_family)

test_set = train.load_families("./data/formatted_shadow", test_family)
train_set = train.load_families("./data/formatted_shadow", train_families)

model = diurnal.models.NN(
    model=cnn.Shadow,
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

print(f"\nSample prediction from the test set (`{test_set['names'][0]}`).")
input = (test_set["input"][0][0], )
p = input[0]
s = test_set["output"][0]
visualize.shadow(p, s, model.predict(input))
