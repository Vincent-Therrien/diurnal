"""
    Collection of neural networks adapted for reinforcement learning.

    File information:

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: May 2024
    - License: MIT
"""

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class DQN1(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.activation = F.relu
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.activation(x)
        x = F.relu(self.layer2(x))
        x = self.activation(x)
        return self.layer3(x)
