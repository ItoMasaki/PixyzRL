"""Single Gym environment wrapper."""

from typing import Any, SupportsFloat

import gymnasium as gym
from numpy.typing import NDArray

from .base_env import BaseEnv


class Env(BaseEnv):
    """Standard single Gym environment wrapper."""

    def __init__(self, env_name: str, seed: int = 42) -> None:
        """
        Initialize the environment.

        :param env_name: Name of the gym environment.
        :param seed: Random seed for reproducibility.
        """
        super().__init__(env_name, num_envs=1, seed=seed)
        self.env = gym.make(env_name)
        self.env.reset(seed=seed)

    def reset(self) -> tuple[Any, dict[str, Any]]:
        """Reset the environment."""
        return self.env.reset()

    def step(self, action: NDArray[Any]) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        """Take a step in the environment."""
        return self.env.step(action)

    def close(self) -> None:
        """Close the environment."""
        self.env.close()

    def render(self) -> None:
        """Render the environment."""
        self.env.render()

    @property
    def observation_space(self) -> gym.Space[Any]:
        """Return observation space."""
        return self.env.observation_space

    @property
    def action_space(self) -> gym.Space[Any]:
        """Return action space."""
        return self.env.action_space
