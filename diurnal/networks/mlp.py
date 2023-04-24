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
        width = rna_length
        self.conv = nn.Conv1d(one_hot_dim, width, kernel, padding="same")
        self.conv2 = nn.Conv1d(width, int(width/2), 5, padding="same")
        self.conv3 = nn.Conv1d(int(width/2), int(width/4), kernel, padding="same")
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(rna_length * int(width/4), 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 25)
        self.fc4 = nn.Linear(25, n_families)
        self.output = nn.Softmax(1)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.leaky_relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.output(x)
        return x

class RNA_MLP_classes(nn.Module):
    """
    Neural network used to determine the secondary structure of a sequence.

    Input: RNA sequence one-hot encoding represented as a 2D array.
        Example: [[0, 0, 0, 1], [1, 0, 0, 0], ...]
    
    Output: RNA secondary structure represented as a matrix whose element
        are vectors of 3 terms that correspond to the probabiliy of each class.
        Example: [[0, 0, 1], [0, 1, 0], [1, 0, 0]] in which `[0, 0, 1]`
        represents a nucleotide paired to a downstream nucleotide, `[0, 1, 0]`,
        an unpaired nucleotide, and `[1, 0, 0]`, a nucleotide paired with an
        upstream nucleotide.
    """
    def __init__(self, n: int):
        super().__init__()
        width = n
        one_hot_dim = 4
        kernel = 3
        self.n = n
        self.conv1 = nn.Conv1d(one_hot_dim, width, kernel, padding="same")
        self.conv2 = nn.Conv1d(width, n, kernel, padding="same")
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(n * n, n * 3)
        self.output = nn.Softmax(2)

    def forward(self, x, f):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = reshape(x, (x.shape[0], self.n, 3))
        x = self.output(x)
        return x