from abc import ABC, abstractmethod
from typing import Any

import torch
from pixyz.models import Model

from pixyzrl.memory import BaseBuffer


class RLModel(Model, ABC):
    """Base class for reinforcement learning models."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the model.

        Args:
            *args (Any): Positional arguments.
            **kwargs (Any): Keyword arguments.
        """
        super().__init__(*args, **kwargs)

        self._is_on_policy = False
        self._action_var = "a"

    @abstractmethod
    def select_action(self, state: Any) -> Any: ...

    @abstractmethod
    def train_step(self, memory: BaseBuffer, batch_size: int = 128, num_epochs: int = 4) -> float: ...

    @abstractmethod
    def transfer_state_dict(self) -> None: ...

    @property
    def is_on_policy(self) -> bool:
        """Return whether the model is on-policy.

        Returns:
            bool: Whether the model is on-policy.
        """
        return self._is_on_policy

    @property
    def action_var(self) -> str:
        """Return the action variable.

        Returns:
            str: Action variable.
        """
        return self._action_var

    def save(self, path: str) -> None:
        """Save the trained model.

        Args:
            path (str): Path to save
        """
        dists = [dist.state_dict() for dist in self.distributions]
        torch.save({"distributions": dists, "optimizer": self.optimizer.state_dict()}, path)

    def load(self, path: str) -> None:
        """Load a trained model.

        Args:
            path (str): Path to load
        """
        checkpoint = torch.load(path)
        dists = list(self.distributions)
        for dist, state_dict in zip(dists, checkpoint["distributions"], strict=False):
            dist.load_state_dict(state_dict)
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.transfer_state_dict()
