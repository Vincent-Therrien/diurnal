"""
    Demonstration for a simple CNN that predicts RNA secondary
    structures with pseudo-knots.
"""

import torch
from torchsummary import summary

from diurnal import database, train, visualize, family, structure
import diurnal.models
from diurnal.models.networks import cnn


SIZE = 256
PATH = "./data/formatted_matrix"

database.download("./data/", "archiveII")
database.format(
    "./data/archiveII",  # Directory of the raw data to format.
    PATH,  # Formatted data output directory.
    SIZE,  # Normalized size.
    structure.Primary.to_matrix,
    structure.Secondary.to_matrix
)

test_family = "5s"
train_families = family.all_but(test_family)

test_set = train.load_families(PATH, test_family, randomize=False)
train_set = train.load_families(PATH, train_families, randomize=False)

model = diurnal.models.NN(
    model=cnn.RNA_CNN,
    N=SIZE,
    n_epochs=3,
    optimizer=torch.optim.Adam,
    loss_fn=torch.nn.MSELoss,
    optimizer_args={"eps": 1e-4},
    loss_fn_args=None,
    verbosity=1,
    use_half=False)
model.train(train_set)

f = model.test(test_set)
print(f"Average F1-score: {sum(f)/len(f):.4}")

model.save("saved_model")

del model

loaded_model = diurnal.models.NN(
    model=cnn.RNA_CNN,
    N=SIZE,
    n_epochs=3,
    optimizer=torch.optim.Adam,
    loss_fn=torch.nn.MSELoss,
    optimizer_args={"eps": 1e-4},
    loss_fn_args=None,
    verbosity=2,
    use_half=False)
loaded_model.load("saved_model")

f = loaded_model.test(test_set)
print(f"Average F1-score of the saved model: {sum(f)/len(f):.4}")

print(f"\nSample prediction from the test set (`{test_set['names'][0]}`).")
p = test_set["primary_structures"][0]
s = test_set["secondary_structures"][0]
visualize.prediction(p, s, loaded_model.predict(p))
