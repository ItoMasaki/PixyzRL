"""Single Gym environment wrapper."""

from abc import ABC, abstractmethod
from typing import Any, cast

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Discrete, MultiDiscrete, Space
from gymnasium.vector import VectorEnv


class BaseEnv(ABC):
    """Base class for RL environments."""

    def __init__(self, env_name: str, num_envs: int = 1, seed: int = 42) -> None:
        """Base class for RL environments.

        Args:
            env_name (str): Name of the gym environment.
            num_envs (int): Number of environments.
            seed (int): Random seed for reproducibility.

        Examples:
            >>> env = Env("CartPole-v1")
        """

        self.env_name = env_name
        self.seed = seed

        self._observation_space: Space[Any] = Space()
        self._action_space: Space[Any] = Space()
        self._is_discrete = False
        self._num_envs = num_envs
        self._env: VectorEnv | None = None
        self._render_mode = "rgb_array"

    @abstractmethod
    def reset(self, **kwargs: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        """Reset the environment."""
        ...

    @abstractmethod
    def step(
        self, action: Any
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Step through the environment."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Close the environment."""
        ...

    @abstractmethod
    def render(self, return_frame: bool = False) -> Any:
        """Render the environment."""
        ...

    @property
    def num_envs(self) -> int:
        """Return the number of environments."""
        return self._num_envs

    @property
    def observation_space(self) -> tuple[int, ...]:
        """Return observation space.

        Returns:
            tuple[int, ...]: Observation space shape.

        Examples:
            >>> env = Env("CartPole-v1")
            >>> obs_shape = env.observation_space
        """
        if (
            hasattr(self._observation_space, "shape")
            and self._observation_space.shape is not None
        ):
            return self._observation_space.shape[1:]
        msg = "Unsupported observation space type"
        raise ValueError(msg)

    @property
    def action_space(self) -> int:
        """Return the size of the action space."""
        if isinstance(self._action_space, Discrete):
            return int(self._action_space.n)
        if isinstance(self._action_space, MultiDiscrete):
            return int(self._action_space.nvec[-1])
        if hasattr(self._action_space, "shape") and (
            self._action_space.shape is not None
        ):
            return self._action_space.shape[-1]
        msg = "Unsupported action space type"
        raise ValueError(msg)

    @property
    def is_discrete(self) -> bool:
        """Return whether the action space is discrete."""
        return self._is_discrete

    @property
    def env(self) -> VectorEnv | None:
        """Return the gym environment."""
        return self._env

    @property
    def render_mode(self) -> str:
        """Return the rendering mode."""
        return self._render_mode


class Env(BaseEnv):
    """Standard single Gym environment wrapper."""

    def __init__(
        self,
        env_name: str,
        num_envs: int = 1,
        action_var: str = "a",
        seed: int = 42,
        render_mode: str = "human",
        **kwargs: dict[str, Any],
    ) -> None:
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

        wrappers = kwargs.pop("wrappers", None)
        self._env = gym.make_vec(
            env_name,
            num_envs=num_envs,
            render_mode=render_mode,
            vectorization_mode="sync",
            wrappers=cast(Any, wrappers),
            **kwargs,
        )

        self.action_var = action_var
        self._render_mode = render_mode
        if self._env is None:
            msg = "Failed to create gym vector environment"
            raise RuntimeError(msg)

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
        if self._env is None:
            msg = "Environment is not initialized"
            raise RuntimeError(msg)
        obs, info = self._env.reset(seed=self.seed, options=kwargs)
        return torch.Tensor(obs), info

    def step(
        self, action: Any
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Take a step in the environment with support for both discrete and continuous actions.

        Args:
            action (Any): Action to take in the environment.

        Returns:
            tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any], dict[str, Any]]: Observation, reward, truncated, terminated, info

        Examples:
            >>> import torch
            >>> env = Env("CartPole-v1")
            >>> obs, info = env.reset()
            >>> action = torch.zeros((1, 2))
            >>> obs, reward, terminated, truncated, info = env.step({"a": action})
            >>> env.close()
        """
        if self._env is None:
            msg = "Environment is not initialized"
            raise RuntimeError(msg)

        if self._env.action_space.shape is None:
            msg = "Unsupported action space type"
            raise ValueError(msg)

        if isinstance(action, dict):
            action = action[self.action_var]

        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()

        if isinstance(self._env.action_space, Discrete | MultiDiscrete):
            action = np.argmax(action, axis=-1)
        elif action.shape != self._env.action_space.shape:
            action = action.reshape(*self._env.action_space.shape)

        obs, reward, terminated, truncated, info = self._env.step(action)
        return (
            torch.Tensor(obs),
            torch.Tensor(
                [reward] if isinstance(reward, float | int) else reward
            ).reshape(-1, 1),
            torch.tensor(
                [terminated] if isinstance(terminated, bool) else terminated,
                dtype=torch.bool,
            ).reshape(-1, 1),
            torch.tensor(
                [truncated] if isinstance(truncated, bool) else truncated,
                dtype=torch.bool,
            ).reshape(-1, 1),
            info,
        )

    def close(self) -> None:
        """Close the environment.

        Examples:
            >>> env = Env("CartPole-v1")
            >>> env.close()
        """
        if self._env is not None:
            self._env.close()

    def render(self, return_frame: bool = False) -> Any:
        """Render the environment.

        Examples:
            >>> env = Env("CartPole-v1")
            >>> env.render()
        """
        if return_frame:
            if self._env is None:
                msg = "Environment is not initialized"
                raise RuntimeError(msg)
            return self._env.render()

        return None
