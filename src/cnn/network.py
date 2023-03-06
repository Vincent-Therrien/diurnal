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

def train_one_epoch(model, dataloader, optimizer):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    for batch, (x, y) in enumerate(dataloader):
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        if batch % 20 == 0:
            loss = loss.item(), (batch + 1) * len(x)
            print(f"loss: {loss}  [{batch * len(x)}/{len(dataloader)}]")

def test(model, dataloader):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            output = model(X)
            test_loss += loss_fn(output, y).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(y.view_as(pred)).sum().item()
    size = len(dataloader.dataset)
    test_loss /= size
    print(f"Accuracy: {(100*correct / size):.4f}")

def train(model, train_data, optimizer, epochs):
    for t in range(epochs):
        print(f"Epoch {t}\n-------------------------------")
        train_one_epoch(model, train_data, optimizer)
    print("Completed training.")
