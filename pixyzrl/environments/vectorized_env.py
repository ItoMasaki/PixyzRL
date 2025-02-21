"""Vectorized environment wrapper using Gymnasium's SyncVectorEnv."""

from collections.abc import Callable
from typing import Any, SupportsFloat

import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
from numpy.typing import NDArray

from .base_env import BaseEnv


class VectorizedEnv(BaseEnv):
    """Vectorized environment wrapper using Gymnasium's SyncVectorEnv."""

    def __init__(self, env_name: str, num_envs: int, seed: int = 42) -> None:
        """
        Initialize the environment.

        :param env_name: Name of the gym environment.
        :param num_envs: Number of environments.
        :param seed: Random seed for reproducibility.
        """
        super().__init__(env_name, num_envs, seed)
        self.envs = SyncVectorEnv([self.make_env(env_name, seed + i) for i in range(num_envs)])

    def make_env(self, env_name: str, seed: int) -> Callable[[], gym.Env[Any, Any]]:
        """Create a single environment instance."""

        def _init() -> gym.Env[Any, Any]:
            """
            Initialize the environment with the given name and seed.

            Returns:
            gym.Env: An instance of the initialized environment.
            """
            env = gym.make(env_name)
            env.reset(seed=seed)
            return env

        return _init

    def reset(self) -> tuple[Any, dict[str, Any]]:
        """Reset all environments."""
        return self.envs.reset()

    def step(self, action: NDArray[Any]) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        """Take a step in all environments."""
        return self.envs.step(action)

    def close(self) -> None:
        """Close all environments."""
        self.envs.close()

    def render(self) -> None:
        """Render one of the environments (only the first one)."""
        self.envs.envs[0].render()

    @property
    def observation_space(self) -> gym.Space[Any]:
        """Return observation space (assumed to be the same across environments)."""
        return self.envs.single_observation_space

    @property
    def action_space(self) -> gym.Space[Any]:
        """Return action space (assumed to be the same across environments)."""
        return self.envs.single_action_space
