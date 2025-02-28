from abc import ABC, abstractmethod
from typing import Any

from pixyz.models import Model


class RLModel(Model, ABC):
    """Base class for reinforcement learning models."""

    @abstractmethod
    def __init__(self, **kwargs: dict[str, Any]) -> None:
        """Initialize the model."""
        super().__init__(**kwargs)

    @abstractmethod
    def select_action(self, state: Any) -> Any:
        """Select an action."""
        ...

    @abstractmethod
    def is_on_policy(self) -> bool:
        """Check if the model is on-policy."""
        ...
