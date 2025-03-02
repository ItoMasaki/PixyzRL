from abc import ABC, abstractmethod
from typing import Any

from pixyz.models import Model

from pixyzrl.memory import BaseBuffer


class RLModel(Model, ABC):
    """Base class for reinforcement learning models."""

    @abstractmethod
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
    def select_action(self, state: Any) -> Any:
        """Select an action."""
        ...

    @abstractmethod
    def train_step(self, memory: BaseBuffer, batch_size: int = 128, num_epochs: int = 4) -> float:
        """Perform a single training step."""
        ...

    @property
    def is_on_policy(self) -> bool:
        """Return whether the model is on-policy."""
        return self._is_on_policy

    @property
    def action_var(self) -> str:
        """Return the action variable."""
        return self._action_var
