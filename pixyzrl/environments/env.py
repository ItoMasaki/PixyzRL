"""Single Gym environment wrapper."""

from abc import ABC, abstractmethod
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box, Discrete, MultiDiscrete, Space
from matplotlib import pyplot as plt


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
        self._env = None
        self._render_mode = "rgb_array"

    @abstractmethod
    def reset(self, **kwargs: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
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

    @property
    def env(self) -> gym.Env[Any, Any] | None:
        """Return the gym environment."""
        return self._env

    @property
    def render_mode(self) -> str:
        """Return the rendering mode."""
        return self._render_mode


class Env(BaseEnv):
    """Standard single Gym environment wrapper."""

    def __init__(self, env_name: str, num_envs: int = 1, action_var: str = "a", seed: int = 42, render_mode: str = "human") -> None:
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
        super().__init__(env_name, num_envs=num_envs, seed=seed)

        if num_envs > 1:
            self._env = gym.make_vec(env_name, num_envs=num_envs, render_mode=render_mode, vectorization_mode="sync")
        else:
            self._env = gym.make(env_name, render_mode=render_mode)

        self.action_var = action_var
        self._render_mode = render_mode
        self._env.reset(seed=seed)

        self._num_envs = num_envs
        self._observation_space = self._env.observation_space
        self._action_space = self._env.action_space
        self._is_discrete = isinstance(self._env.action_space, Discrete)

    def reset(self, **kwargs: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        """Reset the environment.

        Returns:
            tuple[NDArray[Any], dict[str, Any]]: Observation

        Examples:
            >>> env = Env("CartPole-v1")
            >>> obs, info = env.reset()
        """
        obs, info = self._env.reset(seed=self.seed, options=kwargs)
        return torch.Tensor(obs), info

    def step(self, action: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Take a step in the environment with support for both discrete and continuous actions.

        Args:
            action (Any): Action to take in the environment.

        Returns:
            tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any], dict[str, Any]]: Observation, reward, truncated, terminated, info

        Examples:
            >>> import torch
            >>> env = Env("CartPole-v1")
            >>> obs, info = env.reset()
            >>> action = torch.Tensor(1)
            >>> obs, reward, truncated, terminated, info = env.step({"a": torch.argmax(action).item()})
        """

        if isinstance(action, dict):
            action = action[self.action_var]

        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()

        if isinstance(self._env.action_space, Discrete | MultiDiscrete):
            action = np.argmax(action, axis=-1)

        elif isinstance(self._env.action_space, Box):
            action = np.clip(action, self._env.action_space.low, self._env.action_space.high)  # 連続値を制限

        obs, reward, terminated, truncated, info = self._env.step(action)
        return torch.Tensor(obs), torch.Tensor([reward] if isinstance(reward, float) else reward).reshape(-1, 1), torch.Tensor([terminated] if isinstance(terminated, bool) else terminated).reshape(-1, 1), torch.Tensor([truncated] if isinstance(truncated, bool) else truncated).reshape(-1, 1), info

    def close(self) -> None:
        """Close the environment.

        Examples:
            >>> env = Env("CartPole-v1")
            >>> env.close()
        """
        self._env.close()

    def render(self) -> None:
        """Render the environment.

        Examples:
            >>> env = Env("CartPole-v1")
            >>> env.render()
        """
        frames = self._env.render()

        if frames is None:
            return

        if self._render_mode == "rgb_array":
            plt.cla()
            plt.clf()
            plt.imshow(np.array(frames).mean(axis=0).astype(np.uint8))
            plt.axis("off")
            plt.pause(0.01)
