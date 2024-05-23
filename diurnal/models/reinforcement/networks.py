"""
    Collection of neural networks adapted for reinforcement learning.

    File information:

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: May 2024
    - License: MIT
"""

from torch import stack, squeeze, Tensor, flatten
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class DQN1(nn.Module):

    def __init__(self, n: int, n_actions: int):
        super().__init__()
        kernel = 3
        n2 = int(n / 2)
        n4 = int(n / 4)
        self.conv = nn.Conv2d(1, 1, kernel, padding="same")
        self.downsize = nn.AdaptiveAvgPool2d(n2)
        self.activation = F.relu
        self.linear1 = nn.Linear(n2, n2)
        self.downsize2 = nn.AdaptiveAvgPool2d(n4)
        self.linear2 = nn.Linear(n4, n4)
        self.linear3 = nn.Linear(n4**2, n_actions)
        self.output = nn.Softmax(dim=0)

    def forward(self, primary: Tensor, cursor: Tensor) -> Tensor:
        x = stack((primary, ), dim=0)
        #print(x)
        y = stack((cursor, ), dim=0)
        x = self.conv(x) * self.conv(y)
        #print(x)
        x = self.activation(x)
        x = self.downsize(x)
        #print(x)
        x = self.activation(x)
        x = self.linear1(x)
        #print(x)
        x = self.activation(x)
        x = self.downsize2(x)
        #print(x)
        x = self.activation(x)
        x = self.linear2(x)
        #print(x)
        x = self.activation(x)
        x = squeeze(x)
        x = flatten(x)
        x = flatten(x)
        #print(x)
        x = self.linear3(x)
        x = self.output(x)
        #print(x)
        return x
