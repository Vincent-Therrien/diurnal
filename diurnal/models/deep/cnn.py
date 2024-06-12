"""
    RNA secondary prediction models based on convolutional neural
    networks (CNN).

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: July 2023
    - License: MIT
"""


from typing import Callable

from torch import nn, reshape, cat, stack, squeeze, Tensor
import torch.nn.functional as F


class UNet1D(nn.Module):
    """U-net neural network.

    Input: 1D array.

    Output: 1D array with elements comprised within the range (0, 1).
    """
    def __init__(
            self,
            n: int,
            depth: int,
            kernel: int = 3,
            downsizer: Callable = nn.AdaptiveAvgPool1d,
            upsizer: Callable = nn.Upsample
        ):
        super().__init__()
        # Parameters.
        self.n = n
        self.depth = depth
        self.kernel = kernel
        # Network components.
        self.activation = F.relu
        self.conv1 = nn.Conv2d(1, 1, self.kernel, padding="same")
        self.downsizers = []
        self.upsizers = []
        for _ in range(depth):
            n /= 2
            self.downsizers.append(downsizer(n))
            self.upsizers.append(upsizer(scale_factor=2))

    def forward(self, x: Tensor) -> Tensor:
        # Input processing.
        x = stack((x, ), dim=1)
        x = self.activation(x)
        # Downsizing.

        # Upsizing.
        x = self.conv1(x)
        x = self.downsize(x)
        x = self.activation(x)
        x = self.linear1(x)
        x = self.upsample(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)
        x = self.output(x)
        x = squeeze(x)
        return x


class Autoencoder1D(nn.Module):
    """Neural network used to predict a contact matrix.

    Input: 1D matrix.

    Output: 1D matrix with elements comprised within the range (0, 1).
    """
    def __init__(self, n: int):
        super().__init__()
        kernel = 3
        n_half = int(n / 2)
        self.conv1 = nn.Conv2d(1, 1, kernel, padding="same")
        self.downsize = nn.AdaptiveAvgPool1d(n_half)
        self.activation = F.relu
        self.linear1 = nn.Linear(n_half, n_half)
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True
        )
        self.linear2 = nn.Linear(n, n)
        self.linear3 = nn.Linear(n, n)
        self.output = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        """Forward propagation.

        Args:
            x: Potential pairing matrix.

        Returns: Blurry distance matrix.
        """
        x = stack((x, ), dim=1)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.downsize(x)
        x = self.activation(x)
        x = self.linear1(x)
        x = self.upsample(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)
        x = self.output(x)
        x = squeeze(x)
        return x


class UNet2D(nn.Module):
    """U-net neural network with linear layers.

    Input: 2D array.

    Output: 2D array with elements comprised within the range (0, 1).
    """
    def __init__(
            self,
            n: int,
            depth: int,
            kernel: int = 3,
            downsizer=nn.AdaptiveAvgPool2d,
            upsizer = nn.Upsample
        ):
        super().__init__()
        # Parameters.
        self.n = n
        self.depth = depth
        self.kernel = kernel
        # Network components.
        self.activation = F.relu
        self.conv1 = nn.Conv2d(1, 1, self.kernel, padding="same")
        self.downsizers = []
        self.upsizers = []
        self.down_full_layers = []
        self.up_full_layers = []
        for _ in range(depth):
            self.down_full_layers.append(nn.Linear(n, n))
            self.up_full_layers.append(nn.Linear(n, n))
            n = int(n / 2)
            self.downsizers.append(downsizer(n))
            self.upsizers.append(upsizer(
                scale_factor=2, mode='bilinear'
            ))
        self.bottom = nn.Linear(n, n)
        self.down_full_layers = nn.Sequential(*self.down_full_layers)
        self.up_full_layers = nn.Sequential(*self.up_full_layers)
        self.output = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        # Input processing.
        x = stack((x, ), dim=1)
        x = self.conv1(x)
        reserve = []
        # Downsizing.
        for i in range(self.depth):
            x = self.down_full_layers[i](x)
            reserve.append(x.clone())
            x = self.downsizers[i](x)
            x = self.activation(x)
        # Lowest layer.
        x = self.bottom(x)
        x = self.activation(x)
        # Upsizing.
        for i in range(self.depth - 1, -1, -1):
            x = self.upsizers[i](x)
            x = self.up_full_layers[i](x)
            x += reserve[i]
            x = self.activation(x)
        x = self.output(x)
        x = squeeze(x)
        return x


class UNet2DRefiner(nn.Module):
    def __init__(self, n: int, depth: int):
        super().__init__()
        self.convInput = nn.Conv2d(2, 1, 1, padding="same")
        # Parameters.
        self.n = n
        self.depth = depth
        self.kernel = 3
        # Network components.
        self.activation = F.relu
        self.conv1 = nn.Conv2d(1, 1, self.kernel, padding="same")
        self.downsizers = []
        self.upsizers = []
        self.down_full_layers = []
        self.up_full_layers = []
        for _ in range(depth):
            self.down_full_layers.append(nn.Linear(n, n))
            self.up_full_layers.append(nn.Linear(n, n))
            n = int(n / 2)
            self.downsizers.append(nn.AdaptiveAvgPool2d(n))
            self.upsizers.append(nn.Upsample(
                scale_factor=2, mode='bilinear'
            ))
        self.bottom = nn.Linear(n, n)
        self.down_full_layers = nn.Sequential(*self.down_full_layers)
        self.up_full_layers = nn.Sequential(*self.up_full_layers)
        self.output = nn.Sigmoid()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        # Input processing.
        x = stack((x, ), dim=1)
        y = stack((y, ), dim=1)
        z = cat((x, y), 1)
        z = self.convInput(z)
        x = self.conv1(z)
        reserve = []
        # Downsizing.
        for i in range(self.depth):
            x = self.down_full_layers[i](x)
            reserve.append(x.clone())
            x = self.downsizers[i](x)
            x = self.activation(x)
        # Lowest layer.
        x = self.bottom(x)
        x = self.activation(x)
        # Upsizing.
        for i in range(self.depth - 1, -1, -1):
            x = self.upsizers[i](x)
            x = self.up_full_layers[i](x)
            x += reserve[i]
            x = self.activation(x)
        x = self.output(x)
        x = squeeze(x)
        return x


class Autoencoder2D(nn.Module):
    """Neural network used to predict a contact matrix.

    Input: 2D matrix.

    Output: 2D matrix with elements comprised within the range (0, 1).
    """
    def __init__(self, n: int):
        super().__init__()
        kernel = 3
        n_half = int(n / 2)
        self.conv1 = nn.Conv2d(1, 1, kernel, padding="same")
        self.downsize = nn.AdaptiveAvgPool2d(n_half)
        self.activation = F.relu
        self.linear1 = nn.Linear(n_half, n_half)
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True
        )
        self.linear2 = nn.Linear(n, n)
        self.linear3 = nn.Linear(n, n)
        self.output = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        """Forward propagation.

        Args:
            x: Potential pairing matrix.

        Returns: Blurry distance matrix.
        """
        x = stack((x, ), dim=1)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.downsize(x)
        x = self.activation(x)
        x = self.linear1(x)
        x = self.upsample(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)
        x = self.output(x)
        x = squeeze(x)
        return x


class RNA_CNN(nn.Module):
    """Neural network used to determine the secondary structure of a
    sequence.

    Input: RNA sequence one-hot encoding represented as a 3D array.

    Output: RNA secondary structure represented as a vector.
        Example: [1, 1, 1, 0 , 0, 0, -1, -1, -1] in which `1` represents
        a nucleotide paired to a downstream nucleotide, `0`, an unpaired
        nucleotide, and `-1`, a nucleotide paired with an upstream nucleotide.
    """
    def __init__(self, n: int):
        super().__init__()
        one_hot_dim = 8
        kernel = 3
        self.conv1 = nn.Conv2d(one_hot_dim, 1, kernel, padding="same")
        self.fc1 = nn.Linear(n, n)
        self.output = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.fc1(x)
        x = self.output(x)
        return x


class Dot_Bracket(nn.Module):
    """Simple CNN network to predict RNA secondary structures.

    Input: RNA sequence one-hot encoding represented as a 2D array.
        Example: [[0, 0, 0, 1], [1, 0, 0, 0], ...]

    Output: RNA secondary structure represented as a matrix whose
        element are vectors of 3 terms that correspond to the
        probability of each class.
        Example: [[0, 0, 1], [0, 1, 0], [1, 0, 0]] in which `[0, 0, 1]`
        represents a nucleotide paired to a downstream nucleotide,
        `[0, 1, 0]`, an unpaired nucleotide, and `[1, 0, 0]`, a
        nucleotide paired with an upstream nucleotide.
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


class Shadow(nn.Module):
    """Neural network used to determine the shadow of a sequence.

    Input: RNA sequence one-hot encoding represented as a 2D array.
        Example: [[0, 0, 0, 1], [1, 0, 0, 0], ...]

    Output: RNA secondary structure shadow represented as a vector.
        Example: [1, 1, 1, 0 , 0, 0, 1, 1, 1] in which `1` represents
        a paired nucleotide and `0`, and unpaired nucleotide.
    """
    def __init__(self, n: int):
        super().__init__()
        self.width = n
        one_hot_dim = 4
        kernel = 3
        self.conv1 = nn.Conv1d(one_hot_dim, self.width, kernel, padding="same")
        self.conv2 = nn.Conv1d(self.width, self.width, kernel, padding="same")
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(n * self.width, n)
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


class RNA_CNN_family_aware(nn.Module):
    """
    Neural network used to determine the secondary structure of a sequence by
    considering the

    Input: RNA sequence one-hot encoding represented as a 2D array.
        Example: [[0, 0, 0, 1], [1, 0, 0, 0], ...]

    Output: RNA secondary structure represented as a matrix whose element
        are vectors of 3 terms that correspond to the probabiliy of each class.
        Example: [[0, 0, 1], [0, 1, 0], [1, 0, 0]] in which `[0, 0, 1]`
        represents a nucleotide paired to a downstream nucleotide, `[0, 1, 0]`,
        an unpaired nucleotide, and `[1, 0, 0]`, a nucleotide paired with an
        upstream nucleotide.
    """
    def __init__(self, n: int, n_families: int):
        super().__init__()
        self.n_families = n_families
        width = n
        one_hot_dim = 4
        kernel = 3
        self.n = n
        self.m = 1
        self.conv1 = nn.Conv1d(one_hot_dim, width, kernel, padding="same")
        self.conv2 = nn.Conv1d(width, n, kernel, padding="same")
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(width * n, n * self.m)
        self.conv3 = nn.Conv1d(n * self.m, n * (n_families + 1) * self.m, kernel, padding="same")
        self.fc2 = nn.Linear(n * (n_families + 1) * self.m * (n_families + 1), n * 3)
        self.output = nn.Softmax(2)

    def forward(self, x, family):
        """
        x: [[0, 1, 0, 0], ...]      2 X 512
        family: [0, 0, 0, 1, 0, 0]  1 X 512
        """
        # Use convolution to transform the input.
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        # Inject the family into the transformed input.
        f = family.repeat(1, self.n * self.m)
        f = reshape(f, (f.shape[0], self.n * self.m, self.n_families))
        x = reshape(x, (x.shape[0], self.n * self.m, 1))
        x = cat((x, f), 2)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.flatten(x)
        x = self.fc2(x)
        # Format output
        x = reshape(x, (x.shape[0], self.n, 3))
        x = self.output(x)
        return x
