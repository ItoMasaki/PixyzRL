"""Single Gym environment wrapper."""

import random
from abc import ABC, abstractmethod
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data as pd
import torch
from gymnasium import spaces
from gymnasium.spaces import Box, Discrete, MultiDiscrete, Space
from matplotlib import pyplot as plt
from numpy.typing import NDArray


class BipedalRobotEnv(gym.Env[Any, Any]):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self, render_mode: str = "human"):
        super().__init__()

        self.render_mode = render_mode

        if self.render_mode == "human":
            self.client_id = p.connect(p.GUI)
        else:
            self.client_id = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pd.getDataPath())
        self._setup_simulation()

        self.lower_upper = [
            {"idx": 1, "lower": -0.26017, "upper": 0.26017},
            {"idx": 2, "lower": -0.26017, "upper": 0.26017},
            {"idx": 5, "lower": -0.13, "upper": 0.13},
            {"idx": 6, "lower": -0.13, "upper": 0.13},
            {"idx": 7, "lower": -0.165, "upper": 0.365},
            {"idx": 8, "lower": -0.165, "upper": 0.365},
            {"idx": 11, "lower": -0.065, "upper": 0.565},
            {"idx": 12, "lower": -0.065, "upper": 0.565},
            {"idx": 15, "lower": -0.465, "upper": 0.165},
            {"idx": 16, "lower": -0.465, "upper": 0.165},
        ]

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(len(self.lower_upper),), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(p.getNumJoints(self.botId) + 1,), dtype=np.float32)
        self.step_counter = 0
        self.max_steps = 200

    def _setup_simulation(self) -> None:
        p.setGravity(0, 0, -9.8, physicsClientId=self.client_id)
        p.setTimeStep(1 / 50, physicsClientId=self.client_id)

        # Create Terrain
        numHeightfieldRows = 256
        numHeightfieldColumns = 256
        heightfieldData = [random.uniform(0, 0.0) for _ in range(numHeightfieldRows * numHeightfieldColumns)]
        terrainShape = p.createCollisionShape(shapeType=p.GEOM_HEIGHTFIELD, meshScale=[0.1, 0.1, 1], heightfieldTextureScaling=numHeightfieldRows, heightfieldData=heightfieldData, numHeightfieldRows=numHeightfieldRows, numHeightfieldColumns=numHeightfieldColumns, physicsClientId=self.client_id)
        self.terrain = p.createMultiBody(0, terrainShape, physicsClientId=self.client_id)
        p.resetBasePositionAndOrientation(self.terrain, [0, 0, 0], [0, 0, 0, 1], physicsClientId=self.client_id)

        # Load Robot
        cubeStartPos = [0, 0, 0.57]
        cubeStartOrientation = p.getQuaternionFromEuler([0.0, 0, 0])
        self.botId = p.loadURDF("/Users/itomasaki/Desktop/PixyzRL/examples/mybot.urdf", cubeStartPos, cubeStartOrientation, physicsClientId=self.client_id)

        for joint in range(p.getNumJoints(self.botId)):
            p.setJointMotorControl2(self.botId, joint, p.POSITION_CONTROL, force=15.0, physicsClientId=self.client_id)

    def step(self, action: NDArray[np.float32]) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        # Action [-1, 1] -> [-0.165, 0.365]

        for joint_idx, joint_action in enumerate(action):
            p.setJointMotorControl2(self.botId, self.lower_upper[joint_idx]["idx"], p.VELOCITY_CONTROL, targetVelocity=joint_action, force=100.0)

        p.stepSimulation(physicsClientId=self.client_id)

        body_height = p.getLinkState(self.botId, 1, physicsClientId=self.client_id)[0][2]
        obs = self._get_observation()

        reward = 0.1 if (body_height > 0.455) and (body_height < 0.475) else -0.1  # Reward for staying high (simplified)
        reward = -p.getLinkState(self.botId, 0, physicsClientId=self.client_id)[0][1]

        collision = p.getContactPoints(self.botId, self.terrain, -1, -1, physicsClientId=self.client_id)
        if len(collision) > 0:
            reward = -3
        done = len(collision) > 0

        if self.step_counter >= self.max_steps:
            done = True
        self.step_counter += 1

        info = {}

        reward = np.array([reward], dtype=np.float32)
        done = np.array([done], dtype=bool)

        return obs, reward, done, done, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed, options=options)
        p.resetBasePositionAndOrientation(self.botId, [0, 0, 0.57], [0, 0, 0, 1], physicsClientId=self.client_id)
        for joint_idx in range(p.getNumJoints(self.botId)):
            p.resetJointState(self.botId, joint_idx, 0, physicsClientId=self.client_id)
        self.step_counter = 0

        return self._get_observation(), {}

    def _get_observation(self) -> NDArray[Any]:
        joint_positions = [p.getJointState(self.botId, self.lower_upper[joint_idx]["idx"], physicsClientId=self.client_id)[0] for joint_idx in range(len(self.lower_upper))]
        base_link_quat = p.getLinkState(self.botId, 0, physicsClientId=self.client_id)[1]
        return np.array([*joint_positions, *base_link_quat], dtype=np.float32)

    def render(self, mode: str = "human") -> Any:
        if mode == "rgb_array":
            width, height, _, _, _, _ = p.getCameraImage(320, 240, physicsClientId=self.client_id)
            return np.array(width).reshape((240, 320, 3))
        return None

    def close(self) -> None:
        p.disconnect(self.client_id)


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
        # self._env = BipedalRobotEnv()

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
