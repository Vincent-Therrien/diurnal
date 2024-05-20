"""
    Reinforcement learning (RL) package.



    File information:

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: May 2024
    - License: MIT
"""

from collections import namedtuple, deque
import random
import numpy as np
from torch import cuda, nn, optim

from diurnal.models import Basic


Transition = namedtuple('Transition',
    ('state', 'action', 'next_state', 'reward')
)


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class RL(Basic):
    """A model that relies on RL to make predictions."""
    def __init__(
        self,
        q_table: nn,
        N: int,
        n_max_epochs: int,
        n_max_episodes: int,
        optimizer: optim,
        loss_fn: nn.functional,
        optimizer_args: dict = None,
        loss_fn_args: dict = None,
        use_half: bool = True,
        patience: int = 5,
        verbosity: int = 0
    ) -> None:
        self.device = "cuda" if cuda.is_available() else "cpu"
        self.use_half = use_half and self.device == "cuda"
        if self.use_half:
            self.nn = q_table(N).to(self.device).half()
        else:
            self.nn = q_table(N).to(self.device)
        # Optimizer
        if optimizer_args:
            args = ""
            for arg, value in optimizer_args.items():
                args += f"{arg}={value}, "
            exec(f"self.optimizer = optimizer(self.nn.parameters(), {args})")
        else:
            self.optimizer = optimizer(self.nn.parameters())
        # Loss function
        if loss_fn_args:
            args = ""
            for arg, value in loss_fn_args.items():
                args += f"{arg}={value}, "
            exec(f"self.loss_fn = loss_fn({args})")
        else:
            self.loss_fn = loss_fn()
        # Other parameters
        self.n_max_epochs = n_max_epochs
        self.n_max_episodes = n_max_episodes
        self.verbosity = verbosity
        self.batch = 16
        self.PATIENCE = patience

    def _train(self) -> None:
        pass

    def _predict(self, input) -> np.ndarray:
        pass

    def _save(self, path: str) -> None:
        pass

    def _load(self, path: str) -> None:
        pass
