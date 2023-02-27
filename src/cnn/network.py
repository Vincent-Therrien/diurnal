import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

class convolutionalNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(4, 256, 3, padding="same")
        self.conv2 = nn.Conv1d(256, 256, 3, padding="same")

        self.fc1 = nn.Linear(512 * 256, 512)
        self.fc2 = nn.Linear(512, 512)

        self.flatten = nn.Flatten()

        self.spatial_dropout = nn.Dropout2d(0.25)
        self.dropout = nn.Dropout(0.25)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.spatial_dropout(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.spatial_dropout(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

def train_one_epoch(model, dataloader, size):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    for batch, (x, y) in enumerate(dataloader):
        # Compute prediction error
        pred = model(x)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(batch)
        if batch % 20 == 0:
            loss, current = loss.item(), (batch + 1) * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(model, dataloader):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            print(pred)
            print(len(pred))
            if len(pred) < 512:
                pred += [0 for _ in range(512 - len(pred))]
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def train(model, train_data, test_data, size, epochs):
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_one_epoch(model, train_data, size)
        test(model, test_data)
    print("Completed")
