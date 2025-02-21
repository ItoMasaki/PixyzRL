"""Base class for RL environments."""

from abc import ABC, abstractmethod
from typing import Any, SupportsFloat

from gymnasium.spaces import Space
from numpy.typing import NDArray


class BaseEnv(ABC):
    """Base class for RL environments."""

    def __init__(self, env_name: str, num_envs: int = 1, seed: int = 42) -> None:
        """
        Initialize the environment.
        :param env_name: Name of the gym environment.
        :param num_envs: Number of environments (1 for single, >1 for vectorized).
        :param seed: Random seed for reproducibility.
        """
        self.env_name = env_name
        self.num_envs = num_envs
        self.seed = seed

    @abstractmethod
    def reset(self) -> tuple[Any, dict[str, Any]]:
        """Reset the environment."""
        ...

    @abstractmethod
    def step(self, action: NDArray[Any]) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        """Step through the environment."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Close the environment."""
        ...

    @abstractmethod
    def render(self) -> None:
        """Render the environment."""
        ...

    @property
    @abstractmethod
    def observation_space(self) -> Space[Any]:
        """Return observation space."""
        ...

    @property
    @abstractmethod
    def action_space(self) -> Space[Any]:
        """Return action space."""
        ...
