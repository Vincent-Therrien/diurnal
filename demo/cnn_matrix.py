"""
    Demonstration for a simple CNN that predicts RNA secondary
    structures with pseudo-knots.
"""

#import torch
#import numpy as np

from diurnal import database, train, visualize, family, structure
#import diurnal.models
#from diurnal.models.networks import cnn
from diurnal.utils import synthetic

p, s = synthetic.make_structures(10)
print(p)
print("".join(structure.Secondary.to_bracket(s)))
exit()


SIZE = 128
PATH = f"./data/formatted_matrix_{SIZE}"

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
train_set, validation_set = train.split_data(train_set, (0.9, 0.1))

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

f = model.test(test_set)
print(f"Average F1-score: {sum(f)/len(f):.4}")

model.save("saved_model")

del model

loaded_model = diurnal.models.NN(
    model=cnn.RNA_CNN,
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
