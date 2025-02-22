from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from tensordict import TensorDict
from torchrl.data import LazyTensorStorage, ReplayBuffer


class Memory:
    """Base class for memory buffer."""

    def __init__(self, obs_shape: tuple[int, ...], action_shape: tuple[int, ...], buffer_size: int, num_envs: int = 1, device: str | torch.device = "cpu", key_mapping: dict[str, str] | None = None, **kwargs: dict[str, Any]) -> None:
        """Initialize the memory buffer.

        Args:
            obs_shape (tuple): Shape of the observation space.
            action_shape (tuple): Shape of the action space.
            buffer_size (int): Size of the buffer.
            num_envs (int, optional): Number of environments. Defaults to 1.
            device (str | torch.device, optional): Device to run the memory on. Defaults to "cpu".
            key_mapping (dict[str, str], optional): Key mapping for the data. Defaults to None.
            **kwargs (dict[str, Any]): Additional keyword arguments.
        Returns:

        Examples:
            >>> memory = Memory(obs_shape=(4,), action_shape=(2,), buffer_size=100)
        """

        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.buffer_size = buffer_size
        self.num_envs = num_envs
        self.device = device
        self.key_mapping = key_mapping if key_mapping else {"obs": "s", "action": "a", "reward": "r", "done": "d"}
        self.storage = LazyTensorStorage(buffer_size * num_envs)
        self.buffer = ReplayBuffer(storage=self.storage, batch_size=buffer_size * num_envs)

        self.kwargs = kwargs

    def add(self, obs: NDArray[Any], action: NDArray[Any], reward: NDArray[Any], done: NDArray[Any]) -> None:
        """Add a new experience to the buffer.

        Args:
            obs (NDArray): Observation.
            action (NDArray): Action.
            reward (NDArray): Reward.
            done (NDArray): Done flag.

        Examples:
            >>> memory.add(obs, action, reward, done)
        """

        data = TensorDict(
            {
                "obs": torch.tensor(obs, dtype=torch.float32, device=self.device),
                "action": torch.tensor(action, dtype=torch.float32, device=self.device),
                "reward": torch.tensor(reward, dtype=torch.float32, device=self.device),
                "done": torch.tensor(done, dtype=torch.float32, device=self.device),
            },
            batch_size=self.num_envs,
        )

        self.buffer.add(data)

    def sample(self) -> dict[str, torch.Tensor]:
        """Sample a batch of experiences.

        Returns:
            dict[str, torch.Tensor]: Sampled batch of experiences.

        Examples:
            >>> sample = memory.sample()
        """
        sample = self.buffer.sample()
        return {self.key_mapping[key]: value for key, value in sample.items()}

    def clear(self) -> None:
        """Clear the buffer.

        Examples:
            >>> memory.clear()
        """
        self.buffer = ReplayBuffer(storage=LazyTensorStorage(self.buffer_size * self.num_envs), batch_size=self.buffer_size * self.num_envs)

    def __len__(self) -> int:
        """Return the length of the buffer.

        Returns:
            int: Length of the buffer.

        Examples:
            >>> len(memory)
        """
        return len(self.buffer)


class ExperienceMemory(Memory):
    """Experience replay buffer for off-policy learning."""

    pass  # 現在のMemoryがそのまま利用可能


class RolloutBuffer(Memory):
    """Rollout buffer for on-policy algorithms like PPO."""

    def __init__(self, obs_shape, action_shape, buffer_size, num_envs=1, device="cpu", key_mapping=None):
        super().__init__(obs_shape, action_shape, buffer_size, num_envs, device, key_mapping)

    def compute_returns(self, last_value, gamma=0.99, gae_lambda=0.95):
        """Compute advantage and returns using GAE (Generalized Advantage Estimation)."""
        advantages = torch.zeros_like(self.buffer.storage[:][self.key_mapping["reward"]], device=self.device)
        last_advantage = 0

        for t in reversed(range(len(self.buffer.storage[:][self.key_mapping["reward"]]))):
            mask = 1.0 - self.buffer.storage[:][self.key_mapping["done"]][t]
            delta = self.buffer.storage[:][self.key_mapping["reward"]][t] + gamma * last_value * mask - self.buffer.storage[:][self.key_mapping["value"]][t]
            advantages[t] = last_advantage = delta + gamma * gae_lambda * mask * last_advantage
            last_value = self.buffer.storage[:][self.key_mapping["value"]][t]

        returns = advantages + self.buffer.storage[:][self.key_mapping["value"]]
        self.buffer.storage[:]["advantage"] = advantages
        self.buffer.storage[:]["return"] = returns


class VectorizedMemory(RolloutBuffer):
    """Vectorized version of RolloutBuffer for parallel environments."""

    def __init__(self, obs_shape, action_shape, buffer_size, num_envs, device="cpu", key_mapping=None):
        super().__init__(obs_shape, action_shape, buffer_size, num_envs, device, key_mapping)

    def add(self, data: TensorDict):
        """Override add method to handle vectorized environments."""
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                data[key] = torch.tensor(value, dtype=torch.float32, device=self.device)
        self.buffer.add(data)
