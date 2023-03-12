from torch import nn

class RNA_CNN(nn.Module):
    def __init__(self, n: int):
        super().__init__()
        width = 256
        self.conv1 = nn.Conv1d(4, width, 3, padding="same")
        self.conv2 = nn.Conv1d(width, width, 3, padding="same")
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(n *width, n)
        self.flatten = nn.Flatten()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.sigmoid(x)
        return x
