from torch import nn, reshape
import torch.nn.functional as F

class RNA_MLP_classifier(nn.Module):
    """
    Neural network used to classify RNA families from their primary structure.

    Input: RNA sequence one-hot encoding represented as a 2D array.
        Example: [[0, 0, 0, 1], [1, 0, 0, 0], ...]
    
    Output: 1D vector representing a one-hot encoding of the family.
    """
    def __init__(self, rna_length: int, n_families: int):
        super().__init__()
        one_hot_dim = 4
        kernel = 3
        width = 512
        self.conv = nn.Conv1d(one_hot_dim, width, kernel, padding="same")
        self.conv2 = nn.Conv1d(width, width, kernel, padding="same")
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(rna_length * width, 100)
        self.fc2 = nn.Linear(100, n_families)
        self.output = nn.Softmax(1)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.output(x)
        return x