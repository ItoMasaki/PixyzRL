"""Refactored implementation of experience replay and rollout buffer with BaseStorage."""

from collections.abc import Sequence
from typing import Any

import torch
from tensordict import TensorDict
from torchrl.data import LazyTensorStorage
from torchrl.data import ReplayBuffer as TorchRLReplayBuffer


class BaseStorage:
    """Base class for managing storage of experience data."""

    def __init__(self, buffer_size: int, batch_size: int, device: str = "cpu") -> None:
        """Initialize the base storage."""
        self.device: str = device
        self.batch_size: int = batch_size
        self.buffer: TorchRLReplayBuffer = TorchRLReplayBuffer(storage=LazyTensorStorage(buffer_size), batch_size=batch_size)

    def add(self, data: TensorDict) -> None:
        """Add a new experience to the buffer."""
        self.buffer.add(data)

    def sample(self) -> TensorDict:
        """Sample a batch of experiences."""
        return self.buffer.sample().to(self.device)

    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer = TorchRLReplayBuffer(storage=LazyTensorStorage(self.buffer.storage.max_size), batch_size=self.batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class ExperienceReplay(BaseStorage):
    """Experience replay buffer using torchrl's ReplayBuffer."""

    def __init__(self, obs_shape: tuple[int, ...], action_shape: tuple[int, ...], buffer_size: int, batch_size: int, device: str = "cpu") -> None:
        super().__init__(buffer_size, batch_size, device)
        self.obs_shape: tuple[int, ...] = obs_shape
        self.action_shape: tuple[int, ...] = action_shape

    def add_experience(self, obs: torch.Tensor | list[Any], action: torch.Tensor | list[Any], reward: torch.Tensor | list[Any], done: torch.Tensor | list[Any]) -> None:
        """Add new experiences to the buffer."""
        data = TensorDict(
            {
                "obs": torch.tensor(obs, dtype=torch.float32, device=self.device),
                "action": torch.tensor(action, dtype=torch.float32, device=self.device),
                "reward": torch.tensor(reward, dtype=torch.float32, device=self.device).unsqueeze(-1),
                "done": torch.tensor(done, dtype=torch.float32, device=self.device).unsqueeze(-1),
            },
            batch_size=len(obs),
        )
        super().add(data)


class RolloutBuffer(BaseStorage):
    """Rollout buffer for PPO-style training."""

    def __init__(self, obs_shape: tuple[int, ...], action_shape: tuple[int, ...], buffer_size: int, batch_size: int, device: str = "cpu") -> None:
        super().__init__(buffer_size, batch_size, device)
        self.obs_shape = obs_shape
        self.action_shape = action_shape

    def add_rollout(
        self,
        obs: torch.Tensor | list[Any],
        action: torch.Tensor | list[Any],
        logprob: torch.Tensor | list[Any],
        reward: torch.Tensor | list[Any],
        state_value: torch.Tensor | list[Any],
        done: torch.Tensor | list[Any],
    ) -> None:
        """Add new experiences to the buffer."""
        data = TensorDict(
            {
                "obs": torch.tensor(obs, dtype=torch.float32, device=self.device),
                "action": torch.tensor(action, dtype=torch.float32, device=self.device),
                "logprob": torch.tensor(logprob, dtype=torch.float32, device=self.device),
                "reward": torch.tensor(reward, dtype=torch.float32, device=self.device).unsqueeze(-1),
                "state_value": torch.tensor(state_value, dtype=torch.float32, device=self.device).unsqueeze(-1),
                "done": torch.tensor(done, dtype=torch.float32, device=self.device).unsqueeze(-1),
            },
            batch_size=len(obs),
        )
        super().add(data)
