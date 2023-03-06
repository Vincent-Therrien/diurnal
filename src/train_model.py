import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
import cnn.network as cnn

# Set working directory to the location of the script.
import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

formatted_path = "../data/archiveII-arrays/"

batch_size = 64
test_fraction = 0.1
validation_fraction = 0.1
train_fraction = 0.8

print("Loading data")
x = np.load(formatted_path + "5s_x.npy")
y = np.load(formatted_path + "5s_y.npy")

print("Splitting data")
data = []
for i in range(len(x)):
    data.append([x[i].T, y[i]])

training_data   = data[0:int(len(data)*0.8)]
validation_data = data[len(training_data):int(len(data)*0.9)]
test_data       = data[len(validation_data):]

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
validation_dataloader = DataLoader(validation_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

print("Beginning the training")
model = cnn.convolutionalNN()
optimizer = optim.Adadelta(model.parameters())

cnn.train(model, train_dataloader, optimizer, 5)
cnn.test(model, test_dataloader)
