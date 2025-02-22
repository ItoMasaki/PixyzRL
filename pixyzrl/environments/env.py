"""Single Gym environment wrapper."""

from abc import ABC, abstractmethod
from typing import Any, SupportsFloat

import gymnasium as gym
import torch
from gymnasium.spaces import Space


class BaseEnv(ABC):
    """Base class for RL environments.

    This class defines the interface for RL environments.
    """

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
    def step(self, action: dict[str, torch.Tensor]) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
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


class Env(BaseEnv):
    """Standard single Gym environment wrapper."""

    def __init__(self, env_name: str, action_var: str = "a", seed: int = 42) -> None:
        """
        Initialize the environment.

        Args:
            env_name (str): Name of the gym environment.
            action_var (str, optional): Name of the action variable. Defaults to "a".
            seed (int, optional): Random seed for reproducibility. Defaults to 42.
        Returns:

        Example:
            env = Env("CartPole-v1", action_var="a", seed=42)
        """
        super().__init__(env_name, num_envs=1, seed=seed)
        self.env = gym.make(env_name)
        self.action_var = action_var
        self.env.reset(seed=seed)

    def reset(self) -> tuple[Any, dict[str, Any]]:
        """Reset the environment.

        Args:
        Returns:
            tuple[Any, dict[str, Any]]: Observation and additional information.

        Example:
            env = Env("CartPole-v1", action_var="a", seed=42)
            obs, info = env.reset()
        """
        return self.env.reset()

    def step(self, action: dict[str, torch.Tensor] | torch.Tensor) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        """Take a step in the environment.

        Args:
            action (dict[str, torch.Tensor] | torch.Tensor): Action to take.
        Returns:
            tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]: Observation, reward, done, info.

        Example:
            env = Env("CartPole-v1", action_var="a", seed=42)
            obs, info = env.reset()
            obs, reward, done, info = env.step({"a": torch.tensor([0])})
        """

        if isinstance(action, torch.Tensor):
            return self.env.step(action.to("cpu").numpy())

        if self.action_var not in action:
            msg = f"Action variable '{self.action_var}' not found in action dict."
            raise ValueError(msg)

        return self.env.step(action[self.action_var].squeeze().to("cpu").numpy())

    def close(self) -> None:
        """Close the environment.

        Args:
        Returns:

        Example:
            env = Env("CartPole-v1", action_var="a", seed=42)
            obs, info = env.reset()
            env.close()
        """
        self.env.close()

    def render(self) -> None:
        """Render the environment.

        Args:
        Returns:

        Example:
            env = Env("CartPole-v1", action_var="a", seed=42)
            obs, info = env.reset()
            env.render()
        """
        self.env.render()

    @property
    def observation_space(self) -> gym.Space[Any]:
        """Return observation space.

        Args:
        Returns:
            gym.Space[Any]: Observation space.

        Example:
            env = Env("CartPole-v1", action_var="a", seed=42)
            obs_space = env.observation_space
        """
        return self.env.observation_space

    @property
    def action_space(self) -> gym.Space[Any]:
        """Return action space.

        Args:
        Returns:
            gym.Space[Any]: Action space.

        Example:
            env = Env("CartPole-v1", action_var="a", seed=42)
            action_space = env.action_space
        """
        return self.env.action_space
