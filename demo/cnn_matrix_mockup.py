"""
    Demonstration for a simple CNN that predicts RNA secondary
    structures with mockup input data.
"""

import torch
import numpy as np

from diurnal import train, visualize, family, structure
import diurnal.models
from diurnal.models.networks import cnn
from diurnal.utils import synthetic


def to_dict(t) -> dict:
    return {
        "names": None,
        "families": None,
        "primary_structures": t[0],
        "secondary_structures": t[1]
    }

SIZE = 32

test_set       = to_dict(synthetic.PairingMatrix.single_pairing(SIZE, 100))
train_set      = to_dict(synthetic.PairingMatrix.single_pairing(SIZE, 800))
validation_set = to_dict(synthetic.PairingMatrix.single_pairing(SIZE, 100))

model = diurnal.models.NN(
    model=cnn.RNA_CNN,
    N=SIZE,
    n_epochs=10,
    optimizer=torch.optim.Adam,
    loss_fn=torch.nn.MSELoss,
    optimizer_args={"eps": 1e-4},
    loss_fn_args=None,
    verbosity=2,
    use_half=True)
model.train(train_set, validation_set)

print(test_set["primary_structures"][0])
visualize.primary_structure(test_set["primary_structures"][0])
visualize.potential_pairings(test_set["primary_structures"][0])
visualize.pairing_matrix(test_set["secondary_structures"][0])
pred = model.predict(test_set["primary_structures"][0])
visualize.pairing_matrix(pred[0])

f = model.test(test_set)
print(f"Average F1-score: {sum(f)/len(f):.4}")

f = model.test(test_set)
print(f"Average F1-score of the saved model: {sum(f)/len(f):.4}")

print(f"\nSample prediction from the test set (`{test_set['primary_structures'][0]}`).")
p = test_set["primary_structures"][0]
s = test_set["secondary_structures"][0]
visualize.prediction(p, s, model.predict(p))
