"""
    Reinforcement learning (RL) package.

    File information:

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: May 2024
    - License: MIT
"""

__all__ = ["networks", "agents"]

from collections import namedtuple, deque
import random
import numpy as np
from torch import cuda, nn, optim, tensor

from diurnal.models import Basic
from diurnal.utils import log


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
        agent: any,
        N: int,
        n_actions: int,
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
        self.length = N
        self.device = "cuda" if cuda.is_available() else "cpu"
        self.use_half = use_half and self.device == "cuda"
        if self.use_half:
            self.nn = q_table(N, n_actions).to(self.device).half()
        else:
            self.nn = q_table(N, n_actions).to(self.device)
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
        self.agent = agent
        # Other parameters
        self.n_max_epochs = n_max_epochs
        self.n_max_episodes = n_max_episodes
        self.verbosity = verbosity
        self.batch = 16
        self.PATIENCE = patience

    def _train(self) -> None:
        """
        Input: (Potential pairing matrix, sequence length)
        Output: Contact matrix
        """
        self.nn.train()
        if self.verbosity:
            threshold = int(len(self.output) * 0.05)
            threshold = 1 if threshold < 1 else threshold
            log.trace("Beginning the training.")
        N_EPISODES = len(self.input[0])
        for episode in range(N_EPISODES):
            log.trace(f"Episode {episode}")
            sequence_length = self.input[1][episode]
            N_ACTIONS = sequence_length * 10
            x = tensor(self.input[0][episode]).to(self.device).half()
            y = tensor(self.output[episode]).to(self.device).half()
            cursor = tensor(np.zeros((self.length, self.length))).to(self.device).half()
            initial = int(sequence_length * 0.75)
            cursor[initial, initial] = 1
            tentative = tensor(np.zeros((self.length, self.length))).to(self.device).half()
            for a in range(N_ACTIONS):
                self.optimizer.zero_grad()
                actions = self.nn(x, cursor).detach().cpu().numpy()
                actions += (np.random.rand(len(actions)) * 0.2 - 0.1)
                self.agent.act(tentative, x, cursor, actions)
                loss = self.loss_fn(tentative, y)
                loss.requires_grad = True
                loss.backward()
                self.optimizer.step()

    def _predict(self, x, sequence_length) -> np.ndarray:
        self.nn.eval()
        x = tensor(x).to(self.device).half()
        N_ACTIONS = self.length * 10
        initial = int(sequence_length * 0.75)
        cursor = tensor(np.zeros((self.length, self.length))).to(self.device).half()
        cursor[initial, initial] = 1
        tentative = tensor(np.zeros((self.length, self.length))).to(self.device).half()
        for _ in range(N_ACTIONS):
            actions = self.nn(x, cursor)
            self.agent.act(tentative, x, cursor, actions)
        return tentative

    def _save(self, path: str) -> None:
        pass

    def _load(self, path: str) -> None:
        pass
