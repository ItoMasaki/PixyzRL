"""Single Gym environment wrapper."""

import re
from abc import ABC, abstractmethod
from re import S
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box, Discrete, Space


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
        self.seed = seed

        self._observation_space = Space()
        self._action_space = Space()
        self._is_discrete = False
        self._num_envs = num_envs

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

    @property
    def num_envs(self) -> int:
        """Return the number of environments."""
        return self._num_envs

    @property
    def observation_space(self) -> Space[Any]:
        """Return observation space."""
        return self._observation_space

    @property
    def action_space(self) -> Space[Any]:
        """Return action space."""
        return self._action_space

    @property
    def is_discrete(self) -> bool:
        """Return whether the action space is discrete."""
        return self._is_discrete


class Env(BaseEnv):
    """Standard single Gym environment wrapper."""

    def __init__(self, env_name: str, env_num: int = 1, action_var: str = "a", seed: int = 42, render_mode: str = "human") -> None:
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
        self._env = gym.make(env_name, render_mode=render_mode)
        self.action_var = action_var
        self._env.reset(seed=seed)

        self._env_num = env_num
        self._observation_space = self._env.observation_space
        self._action_space = self._env.action_space
        self._is_discrete = isinstance(self._env.action_space, Discrete)

    def reset(self) -> tuple[torch.Tensor, dict[str, Any]]:
        """Reset the environment.

        Returns:
            tuple[NDArray[Any], dict[str, Any]]: Observation

        Examples:
            >>> obs, info = env.reset()
        """
        obs, info = self._env.reset(seed=self.seed)
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

        elif isinstance(self._env.action_space, Box):
            action = np.clip(action, self._env.action_space.low, self._env.action_space.high)  # 連続値を制限

        obs, reward, truncated, terminated, info = self._env.step(action)
        return torch.Tensor(obs), torch.Tensor([reward]), torch.Tensor([truncated]), torch.Tensor([terminated]), info

    def close(self) -> None:
        """Close the environment.

        Examples:
            >>> env.close()
        """
        self._env.close()

    def render(self) -> None:
        """Render the environment.

        Examples:
            >>> env.render()
        """
        self._env.render()
