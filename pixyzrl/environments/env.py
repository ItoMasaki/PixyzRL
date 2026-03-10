"""Single Gym environment wrapper."""

from abc import ABC, abstractmethod
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Discrete, MultiDiscrete, Space
from typing import Callable, List
import cv2
from gymnasium.spaces import Box
import math
import time


class BaseEnv(ABC):
    def __init__(self, env_name: str, num_envs: int = 1, seed: int = 42) -> None:
        self.env_name = env_name
        self.seed = seed

        self._observation_space = Space()
        self._action_space = Space()
        self._is_discrete = False
        self._num_envs = num_envs
        self._env = None
        self._render_mode = "rgb_array"

    @abstractmethod
    def reset(self, **kwargs: dict[str, Any]):
        ...

    @abstractmethod
    def step(self, action: Any):
        ...

    @abstractmethod
    def close(self) -> None:
        ...

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def observation_space(self):
        return self._observation_space.shape

    @property
    def action_space(self):
        if isinstance(self._action_space, Discrete):
            return int(self._action_space.n)
        if isinstance(self._action_space, MultiDiscrete):
            return int(self._action_space.nvec[-1])
        return self._action_space.shape[-1]

    @property
    def is_discrete(self) -> bool:
        return self._is_discrete


class Env(BaseEnv):
    def __init__(
        self,
        env_name: str,
        num_envs: int = 1,
        seed: int = 42,
        transforms: List[Callable] | None = None,
        enable_render: bool = False,
        render_scale: float = 1.0,
        render_fps: int = 30,
        render_interval: int = 10,   # ← 追加
        **kwargs,
    ):
        super().__init__(env_name, num_envs, seed)

        self.enable_render = enable_render
        self.render_scale = render_scale
        self.render_fps = render_fps
        self._last_render_time = 0.0
        self._window_name = f"{env_name}-VecRender"

        def make_single_env():
            env = gym.make(env_name, render_mode="rgb_array", **kwargs)

            if transforms:
                for transform in transforms:
                    # 🔥 transformがWrapperクラスならインスタンス化
                    if isinstance(transform, type):
                        env = transform(env)
                    # 🔥 transformが既にcallableならそのまま適用
                    elif callable(transform):
                        env = transform(env)
                    else:
                        raise TypeError(
                            f"Unsupported transform type: {type(transform)}"
                        )

            return env

        self._env = gym.vector.SyncVectorEnv(
            [make_single_env for _ in range(num_envs)]
        )

        self._env.reset(seed=seed)

        self._observation_space = self._env.single_observation_space
        self._action_space = self._env.single_action_space
        self._is_discrete = isinstance(self._action_space, Discrete)
        self.render_interval = render_interval
        self._step_count = 0

    # --------------------------------------------------
    # Core API
    # --------------------------------------------------

    def reset(self, **kwargs):
        obs, info = self._env.reset(seed=self.seed, options=kwargs)
        return torch.as_tensor(obs, dtype=torch.float32), info

    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()

        if self._is_discrete:
            action = np.argmax(action, axis=-1)

        obs, reward, terminated, truncated, info = self._env.step(action)

        # 🔥 描画間引き
        if self.enable_render:
            self._step_count += 1
            if self._step_count % self.render_interval == 0:
                self.render()

        return (
            torch.as_tensor(obs, dtype=torch.float32),
            torch.as_tensor(reward, dtype=torch.float32).reshape(-1, 1),
            torch.as_tensor(terminated, dtype=torch.bool).reshape(-1, 1),
            torch.as_tensor(truncated, dtype=torch.bool).reshape(-1, 1),
            info,
        )

    def close(self):
        self._env.close()
        if self.enable_render:
            cv2.destroyAllWindows()

    # --------------------------------------------------
    # Custom Renderer
    # --------------------------------------------------

    def _make_grid(self, frames: np.ndarray):
        n, h, w, c = frames.shape

        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)

        grid = np.zeros((rows * h, cols * w, c), dtype=frames.dtype)

        for i in range(n):
            r = i // cols
            c_ = i % cols
            grid[r*h:(r+1)*h, c_*w:(c_+1)*w] = frames[i]

        return grid

    def render(self):
        now = time.time()
        if now - self._last_render_time < 1.0 / self.render_fps:
            return

        self._last_render_time = now

        frames = self._env.render()

        # 🔥 ここが重要
        if isinstance(frames, tuple):
            frames = np.stack(frames, axis=0)

        if isinstance(frames, list):
            frames = np.stack(frames, axis=0)

        # もし単一環境なら (H,W,C) → (1,H,W,C)
        if frames.ndim == 3:
            frames = np.expand_dims(frames, 0)

        grid = self._make_grid(frames)

        if self.render_scale != 1.0:
            grid = cv2.resize(
                grid,
                None,
                fx=self.render_scale,
                fy=self.render_scale,
                interpolation=cv2.INTER_NEAREST,
            )

        grid = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)

        cv2.imshow(self._window_name, grid)
        cv2.waitKey(1)



class NormalizeObservation(gym.ObservationWrapper):
    def __init__(self, normalize_factor: float = 255):

        self.normalize_factor = normalize_factor

    def observation(self, obs):
        return obs / self.normalize_factor


class ToTensorObservation(gym.ObservationWrapper):
    def observation(self, obs):
        return torch.from_numpy(obs).float()


class ScaleAction:
    def __init__(
        self,
        from_low,
        from_high,
        to_low,
        to_high,
        clip: bool = True,
    ):
        self.from_low = np.asarray(from_low, dtype=np.float32)
        self.from_high = np.asarray(from_high, dtype=np.float32)
        self.to_low = np.asarray(to_low, dtype=np.float32)
        self.to_high = np.asarray(to_high, dtype=np.float32)
        self.clip = clip

    def __call__(self, env: gym.Env):
        return _ScaleActionWrapper(
            env,
            self.from_low,
            self.from_high,
            self.to_low,
            self.to_high,
            self.clip,
        )


class ResizeObservation:
    def __init__(
        self,
        width: int,
        height: int,
        grayscale: bool = False,
        channel_first: bool = False,
    ):
        self.width = width
        self.height = height
        self.grayscale = grayscale
        self.channel_first = channel_first

    def __call__(self, env: gym.Env):
        return _ResizeObservationWrapper(
            env,
            self.width,
            self.height,
            self.grayscale,
            self.channel_first,
        )

class BitDepthQuantize:
    def __init__(self, bit_depth=5):
        self.bit_depth = bit_depth

    def __call__(self, env: gym.Env):
        return _BitDepthQuantizeWrapper(env, self.bit_depth)

class _ScaleActionWrapper(gym.ActionWrapper):
    def __init__(
        self,
        env: gym.Env,
        from_low,
        from_high,
        to_low,
        to_high,
        clip: bool,
    ):
        super().__init__(env)

        self.from_low = np.broadcast_to(
            from_low, env.action_space.shape
        )
        self.from_high = np.broadcast_to(
            from_high, env.action_space.shape
        )
        self.to_low = np.broadcast_to(
            to_low, env.action_space.shape
        )
        self.to_high = np.broadcast_to(
            to_high, env.action_space.shape
        )
        self.clip = clip

        # policy側が出力するレンジ
        self.action_space = Box(
            low=self.from_low,
            high=self.from_high,
            dtype=np.float32,
        )

    def action(self, action):
        action = np.asarray(action, dtype=np.float32)

        if self.clip:
            action = np.clip(action, self.from_low, self.from_high)

        action = (action - self.from_low) / (
            self.from_high - self.from_low + 1e-8
        )

        action = action * (self.to_high - self.to_low) + self.to_low

        return action
    
class _BitDepthQuantizeWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, bit_depth: int):
        super().__init__(env)
        self.bit_depth = bit_depth

        self.observation_space = Box(
            low=-0.5,
            high=0.5,
            shape=env.observation_space.shape,
            dtype=np.float32,
        )

    def observation(self, obs):
        # Quantise to given bit depth and centre
        obs = np.floor(obs / (2 ** (8 - self.bit_depth))) / (2 ** self.bit_depth) - 0.5
        # Dequantise (to approx. match likelihood of PDF of continuous images vs. PMF of discrete images)
        return obs + np.random.uniform(0, 1 / (2 ** self.bit_depth), size=obs.shape).astype(np.float32)
    
class _ResizeObservationWrapper(gym.ObservationWrapper):
    def __init__(
        self,
        env: gym.Env,
        width: int,
        height: int,
        grayscale: bool,
        channel_first: bool,
    ):
        super().__init__(env)

        self.width = width
        self.height = height
        self.grayscale = grayscale
        self.channel_first = channel_first

        obs_space = env.observation_space

        if not isinstance(obs_space, Box):
            raise TypeError("ResizeObservation only works with Box spaces.")

        if len(obs_space.shape) != 3:
            raise ValueError("ResizeObservation requires 3D image input.")

        if obs_space.shape[0] in [1, 3]:
            self.input_channel_first = True
            c, h, w = obs_space.shape
        else:
            self.input_channel_first = False
            h, w, c = obs_space.shape

        out_c = 1 if grayscale else c

        if channel_first:
            new_shape = (out_c, height, width)
        else:
            new_shape = (height, width, out_c)

        self.observation_space = Box(
            low=0,
            high=255,
            shape=new_shape,
            dtype=np.uint8,
        )

    def observation(self, obs):

        if self.input_channel_first:
            obs = np.transpose(obs, (1, 2, 0))

        if self.grayscale:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)

        obs = cv2.resize(
            obs,
            (self.width, self.height),
            interpolation=cv2.INTER_AREA,
        )

        if self.grayscale:
            obs = np.expand_dims(obs, axis=-1)

        if self.channel_first:
            obs = np.transpose(obs, (2, 0, 1))

        return obs.astype(np.uint8)