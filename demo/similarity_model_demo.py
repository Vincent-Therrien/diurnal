"""
    Demonstration for a family-aware secondary structure predictor.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from diurnal import train, evaluate
from diurnal.networks.mlp import RNA_MLP_classifier
from diurnal.networks.cnn import RNA_CNN_family_aware
from diurnal.models import SimilarityModel
from diurnal.utils import file_io
from diurnal import transform

families = {}
for family in list(transform.Family.ONEHOT.keys()):
    families[family] = 0

for family in families:
    # Load data from `.npy` files.
    test_set, train_set, _, _ = train.load_inter_family(
        "./data/formatted/", family)
    if not test_set:
        file_io.log(f"Cannot test family {family}: no data point.")
        continue

    # Build the model.
    model = SimilarityModel(
        512,
        len(families),
        RNA_MLP_classifier,
        RNA_CNN_family_aware
    )

    # Train the model.
    model.train(DataLoader(train_set, batch_size=32), 5)
    
    # Evaluate the model.
    f1 = model.test(DataLoader(test_set, batch_size=32),evaluate.three_class_f1)
    families[family] = np.mean(f1)
    evaluate.summarize_results(f1, f"CNN, inter-family test {family}")

    # Erase the model before the next iteration to avoid wasting memory.
    del model

print("Resulting F1-scores for each family:")
print(families)