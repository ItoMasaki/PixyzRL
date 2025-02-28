from abc import ABC, abstractmethod
from typing import Any

from pixyz.models import Model


class RLModel(Model, ABC):
    """Base class for reinforcement learning models."""

    @abstractmethod
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the model."""
        super().__init__(*args, **kwargs)

        self._is_on_policy = False
        self._action_var = "a"

    @abstractmethod
    def select_action(self, state: Any) -> Any:
        """Select an action."""
        ...

    @property
    def is_on_policy(self) -> bool:
        """Return whether the model is on-policy."""
        return self._is_on_policy

    @property
    def action_var(self) -> str:
        """Return the action variable."""
        return self._action_var
