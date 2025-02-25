"""Single Gym environment wrapper."""

from abc import ABC, abstractmethod
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box, Discrete, Space
from numpy.typing import NDArray


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
    def reset(self) -> tuple[torch.Tensor, dict[str, Any]]:
        """Reset the environment."""
        ...

    @abstractmethod
    def step(self, action: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
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

    def get_num_envs(self) -> int:
        """Return the number of environments."""
        return self.num_envs

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

    def __init__(self, env_name: str, action_var: str = "a", seed: int = 42, render_mode: str = "human") -> None:
        """
        Initialize the environment.

        Args:
            env_name (str): Name of the gym environment.
            action_var (str): Name of the action variable.
            seed (int): Random seed for reproducibility.
            render_mode (str): Rendering mode (e.g., "human", "rgb_array", "ansi").

        Examples:
            >>> env = Env("CartPole-v1")
        """
        super().__init__(env_name, num_envs=1, seed=seed)
        self.env = gym.make(env_name, render_mode=render_mode)
        self.action_var = action_var
        self.env.reset(seed=seed)

    def reset(self) -> tuple[torch.Tensor, dict[str, Any]]:
        """Reset the environment.

        Returns:
            tuple[NDArray[Any], dict[str, Any]]: Observation

        Examples:
            >>> obs, info = env.reset()
        """
        obs, info = self.env.reset(seed=self.seed)
        return torch.Tensor(obs), info

    def step(self, action: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Take a step in the environment with support for both discrete and continuous actions.

        Args:
            action (Any): Action to take in the environment.

        Returns:
            tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any], dict[str, Any]]: Observation, reward, truncated, terminated, info

        Examples:
            >>> obs, reward, truncated, terminated, info = env.step(action)
        """
        if isinstance(action, dict):
            action = action[self.action_var]

        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()

        elif isinstance(self.env.action_space, Box):
            action = np.clip(action, self.env.action_space.low, self.env.action_space.high)  # 連続値を制限

        obs, reward, truncated, terminated, info = self.env.step(action)
        return torch.Tensor(obs), torch.Tensor([reward]), torch.Tensor([truncated]), torch.Tensor([terminated]), info

    def close(self) -> None:
        """Close the environment.

        Examples:
            >>> env.close()
        """
        self.env.close()

    def render(self) -> None:
        """Render the environment.

        Examples:
            >>> env.render()
        """
        self.env.render()

    @property
    def observation_space(self) -> Space[Any]:
        """Return observation space.

        Examples:
            >>> obs_space = env.observation_space
        """
        return self.env.observation_space

    @property
    def action_space(self) -> Space[Any]:
        """Return action space.

        Examples:
            >>> action_space = env.action_space
        """
        return self.env.action_space
