"""
    Demonstration for a simple CNN that predicts RNA secondary
    structures.
"""

import torch
from torch.utils.data import DataLoader

from diurnal import train, evaluate
from diurnal.models import DiurnalBasicModel
from diurnal.networks import cnn as diurnalCNN

# Load data from `.npy` files.
data, names = train.load_data("./data/formatted/")

# Split the data in training and test sets.
train_set, test_set = train.split_data(data, [0.8, 0.2])

# Create the RNA secondary structure prediction model.
model = DiurnalBasicModel(
        diurnalCNN.RNA_CNN_classes, [512],
        torch.optim.Adam, [1e-04],
        torch.nn.MSELoss()
    )

model.train(DataLoader(train_set, batch_size=32), 5)
f1 = model.test(DataLoader(test_set, batch_size=32),
    evaluate.three_class_f1)

# Display performances.
evaluate.summarize_results(f1, "CNN, Three-class evaluation")
