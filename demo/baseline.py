"""
    Diurnal baseline demonstration script.

    This script uses a baseline model to demonstrate the usage of the
    library. More precisely, it:

    1. downloads and formats data,
    2. creates a baseline model that makes random predictions,
    3. trains the model, and
    4. evaluate the model.

    The creation of the model (step 2) can be replaced by the creation
    of another model to test other prediction methods.

    Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    Affiliation: Département d'informatique, UQÀM
    File creation date: June 2023
    License: MIT
"""

from diurnal import database, train
from diurnal.models import baseline


# Step 1: Obtain and format data
database.download("./data/", "archiveII")

database.format(
    "./data/archiveII", # Directory of the raw data to format.
    "./data/formatted", # Formatted data output directory.
    512, # Normalized size.
)

# Step 2: Prediction model creation
data = train.load_data("./data/formatted/")
train_set, test_set, validate_set = train.split_data(data, [0.8, 0.02, 0.18])

# Step 3: Model training
model = baseline.Random()
model.train(train_set)

# Step 4: Model evaluation
f = model.test(test_set)
print(f"Average F1-score: {sum(f)/len(f):.4}")
