"""
    Diurnal baseline demonstration script.

    This script uses a baseline model to demonstrate the usage of the
    library. More precisely, it:

    1. downloads data,
    2. formats data,
    2. creates and trains a baseline model, and
    4. evaluates the model.

    The creation of the model (step 2) can be replaced by the creation
    of another model to test other prediction methods.

    Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    Affiliation: Département d'informatique, UQÀM
    File creation date: June 2023
    License: MIT
"""

from diurnal import database, train, visualize
from diurnal.models import baseline


database.download("./data/", "archiveII")
database.format(
    "./data/archiveII",  # Directory of the raw data to format.
    "./data/formatted",  # Formatted data output directory.
    512,  # Normalized size.
)

test_set, other_data = train.load_inter_family("./data/formatted", "5s")
train_set, validate_set = train.split_data(other_data, [0.8, 0.2])

model = baseline.Random()
model.train(train_set)

f = model.test(test_set)
print(f"Average F1-score: {sum(f)/len(f):.4}")

p = test_set["primary_structures"][0]
s = test_set["secondary_structures"][0]
visualize.prediction(p, s, model.predict(p))
