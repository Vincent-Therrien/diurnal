import numpy as np
import torch
from torch.utils.data import DataLoader

import src.database as db
from src.transform import PrimaryStructure as s1
from src.transform import SecondaryStructure as s2
from src.transform import Family as f

from src import train
from src.networks import cnn
from src.models import DiurnalBasicModel

#db.download("./data/", "archiveII")
#exit()

# db.format(
#    "./data/archiveII", # Directory of the raw data to format.
#    "./data/formatted", # Formatted data output directory.
#    512,
#    s1.iupac_to_onehot, # RNA encoding scheme.
#    s2.pairings_to_onehot, # RNA encoding scheme.
#    f.onehot
# )

#db.visualize("./data/formatted")

data = train.load_data("./data/formatted/")
t1, t2 = train.split_data(data, [0.8, 0.2])

m = DiurnalBasicModel(cnn.RNA_CNN_classes, [512],
                      torch.optim.Adam, [1e-04],
                      torch.nn.MSELoss())

m.train_with_families(DataLoader(t1, batch_size=32), 3)
f1 = m.test_with_family(DataLoader(t2, batch_size=32))
print(f1[:25])
print(np.mean(f1))

m.save("test/model.pt")

del m

M = DiurnalBasicModel(cnn.RNA_CNN_classes, [512],
                      torch.optim.Adam, [1e-04],
                      torch.nn.MSELoss())
M.load("test/model.pt")
f1 = M.test_with_family(DataLoader(t2, batch_size=32))
print(f1[:25])
print(np.mean(f1))
