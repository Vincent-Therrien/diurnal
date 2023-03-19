from torch import nn, reshape
import torch.nn.functional as F

class RNA_CNN_shadow(nn.Module):
    """
    Neural network used to determine the shadow of a sequence.

    Input: RNA sequence one-hot encoding represented as a 2D array.
        Example: [[0, 0, 0, 1], [1, 0, 0, 0], ...]
    
    Output: RNA secondary structure shadow represented as a vector.
        Example: [1, 1, 1, 0 , 0, 0, 1, 1, 1] in which `1` represents
        a paired nucleotide and `0`, and unpaired nucleotide. 
    """
    def __init__(self, n: int):
        super().__init__()
        width = 256
        one_hot_dim = 4
        kernel = 3
        self.conv1 = nn.Conv1d(one_hot_dim, width, kernel, padding="same")
        self.conv2 = nn.Conv1d(width, width, kernel, padding="same")
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(n * width, n)
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

class RNA_CNN(nn.Module):
    """
    Neural network used to determine the secondary structure of a sequence.

    Input: RNA sequence one-hot encoding represented as a 2D array.
        Example: [[0, 0, 0, 1], [1, 0, 0, 0], ...]
    
    Output: RNA secondary structure represented as a vector.
        Example: [1, 1, 1, 0 , 0, 0, -1, -1, -1] in which `1` represents
        a nucleotide paired to a downstream nucleotide, `0`, an unpaired
        nucleotide, and `-1`, a nucleotide paired with an upstream nucleotide.
    """
    def __init__(self, n: int):
        super().__init__()
        width = n
        one_hot_dim = 4
        kernel = 3
        self.conv1 = nn.Conv1d(one_hot_dim, width, kernel, padding="same")
        self.conv2 = nn.Conv1d(width, width, kernel, padding="same")
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(n * width, n)
        self.output = nn.Tanh()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        # Normalize before producing the output.
        x -= x.min(1, keepdim=True)[0]
        x /= x.max(1, keepdim=True)[0]
        x *= 2
        x -= 1
        x = self.output(x)
        return x

class RNA_CNN_classes(nn.Module):
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
        self.fc1 = nn.Linear(width * n, n * 3)
        self.output = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = reshape(x, (x.shape[0], self.n, 3))
        x = self.output(x)
        return x
